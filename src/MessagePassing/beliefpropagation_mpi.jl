using Dictionaries: Dictionary, set!
using Graphs: AbstractGraph, is_tree, nv
using ITensors.NDTensors: NDTensors, scalartype
using ITensors: Algorithm, ITensor, itensor, dim, id, plev, permute
using MPI
using NamedGraphs.GraphsExtensions: boundary_edges

# ---------------------------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------------------------
#
# Tensors are moved as a small serialised header describing the index structure followed by the
# raw storage as one contiguous buffer, one pair of messages per peer per exchange. The point of
# splitting them is the payload: handing MPI a bare array (rather than `MPI.isend`ing the
# ITensor, which runs it through Julia's serialiser) means a CUDA-aware MPI moves device memory
# straight from one GPU to another, and a host-side run reuses the same buffer every sweep
# instead of allocating a serialisation blob per message.
#
# The header cannot be dropped: `apply_gate!` truncates bonds, which mints new `Index` objects,
# so neither the dimensions nor the index identities of a boundary tensor are known ahead of
# time by the receiver.

const TAG_MESSAGE_HEADER = Cint(11)
const TAG_MESSAGE_PAYLOAD = Cint(12)
const TAG_FACTOR_HEADER = Cint(13)
const TAG_FACTOR_PAYLOAD = Cint(14)

# One per peer per exchange. `counts` records how many tensors each logical item (an edge for
# messages, a vertex for factors) contributed, so a message stored as a `Vector{ITensor}`
# survives the round trip.
struct TensorBatch
    inds::Vector{Vector{Index}}
    counts::Vector{Int}
    eltype::DataType
end

_batch_lengths(b::TensorBatch) = [prod(dim.(is); init = 1) for is in b.inds]

# Sender and receiver have to agree on the memory layout of a tensor without having exchanged
# it, so indices are always permuted into ascending `(id, plev)` order before the storage is
# read, and reinterpreted in that same order on arrival. Index ids are globally unique and
# byte-identical across ranks (they are drawn once and broadcast), so both sides sort alike.
_canonical_inds(t::ITensor) = sort(collect(inds(t)); by = i -> (id(i), plev(i)))

function _canonical_permute(t::ITensor)
    cinds = _canonical_inds(t)
    # `permute` has no zero-index method, and a scalar tensor needs no permuting anyway.
    isempty(cinds) && return t
    return permute(t, cinds...; allow_alias = true)
end

# Only dense storage has a flat buffer whose length is the product of the dimensions. Anything
# else (block-sparse, diagonal) falls back to serialising the tensors themselves.
function _is_raw_transferable(t::ITensor)
    ITensors.storage(t) isa NDTensors.Dense || return false
    return length(ITensors.data(t)) == prod(dim.(inds(t)); init = 1)
end

# Allocate on whichever device the network lives on: `datatype` is e.g. `Vector{Float64}` on the
# host and `CuArray{Float64,1,…}` after `adapt`ing the cache to a GPU.
function _alloc_buffer(bp_cache, T::Type, n::Integer)
    return similar(datatype(bp_cache)(undef, 0), T, n)
end

# Payload buffers are pure scratch, so they are grown rather than reallocated. Keyed by element
# type as well as peer because messages and factors need not share a scalar type.
function _staging!(store::Dict, bp_cache, peer::Integer, T::Type, n::Integer)
    key = (Int32(peer), T)
    buf = get(store, key, nothing)
    if buf === nothing || length(buf) < n
        buf = _alloc_buffer(bp_cache, T, max(n, 1))
        store[key] = buf
    end
    return buf
end

# Host mirrors of the payload buffers, used only when the MPI build cannot read device memory.
_alloc_host_buffer(T::Type, n::Integer) = Vector{T}(undef, n)

function _host_staging!(store::Dict, peer::Integer, T::Type, n::Integer)
    key = (Int32(peer), T)
    buf = get(store, key, nothing)
    if buf === nothing || length(buf) < n
        buf = _alloc_host_buffer(T, max(n, 1))
        store[key] = buf
    end
    return buf
end

# Scratch space for the raw payload exchange. Shared by reference across `copy`, like `comm`:
# the contents are only meaningful for the duration of one exchange, and exchanges are
# collective so they never interleave. Sharing is what keeps a BP sweep from reallocating a
# device buffer per boundary edge.
struct ExchangeBuffers
    send::Dict{Tuple{Int32, DataType}, Any}
    recv::Dict{Tuple{Int32, DataType}, Any}
    send_host::Dict{Tuple{Int32, DataType}, Any}
    recv_host::Dict{Tuple{Int32, DataType}, Any}
end
function ExchangeBuffers()
    D() = Dict{Tuple{Int32, DataType}, Any}()
    return ExchangeBuffers(D(), D(), D(), D())
end

# `true` once the network's tensors live on an accelerator: `datatype` is `Vector{T}` on the host
# and e.g. `CuArray{T,1,…}` after `adapt`ing the cache to a GPU. Tested structurally so that
# nothing here has to depend on CUDA.jl.
_is_device_backed(bp_cache) = !(unspecify_type_parameters(datatype(bp_cache)) <: Array)

# Device tensors are copied to host memory before MPI sees them, and copied back on arrival.
# This is the default, and deliberately so, even though it costs a device->host->device round
# trip per message. Passing device pointers to MPI instead fails in two ways that are both worse
# than the copy:
#
#  * An MPI not built against CUDA/ROCm does not reject a device pointer. Its transport layer
#    (UCX, say) assumes host memory and issues a plain CPU memcpy, segfaulting inside the MPI
#    library with a stack trace pointing at `MPI_Isend` rather than at anything here.
#    `MPI.has_cuda()` is not a reliable guard: it reports a capability flag, and builds that
#    report `true` while UCX lacks `cuda_copy`/`cuda_ipc` still crash.
#  * Even with a genuinely GPU-aware MPI, the buffer handed over has just been written by a
#    device-to-device `copyto!`, which is queued on a stream and does not synchronise with the
#    host. MPI can read it before the copy lands, which corrupts messages silently and
#    intermittently rather than failing. Staging through the host avoids this for free, because
#    `copyto!(::Array, ::CuArray)` blocks and is ordered behind previously queued stream work.
#
# `gpu_direct_mpi!(true)` opts into the direct path for a verified setup; see its docstring for
# what "verified" has to mean.
const _GPU_DIRECT = Ref(false)

# Test hook: forces the staging path so its offset and copy bookkeeping is exercised on the host,
# where both mirrors happen to be plain arrays, without needing a GPU.
const _FORCE_HOST_STAGING = Ref(false)

"""
    gpu_direct_mpi() -> Bool
    gpu_direct_mpi!(enabled::Bool)

Whether boundary tensors held on a GPU are handed to MPI as device pointers (`true`) or copied
through host memory first (`false`, the default).

The default is the safe one. Enabling this is only correct when **both** hold:

  * MPI is genuinely GPU-aware -- built against CUDA/ROCm, with the transport layer configured
    for it (for UCX, `ucx_info -d` showing `cuda_copy`/`cuda_ipc`, and `UCX_TLS` either unset or
    including them). `MPI.has_cuda()` returning `true` is necessary but not sufficient.
  * The device queue is synchronised before MPI reads a buffer. Define a method for
    [`mpi_device_synchronize`](@ref) to do that, e.g. `mpi_device_synchronize() =
    CUDA.synchronize()`; without it, sends can race the device-to-device copy that fills them.

Getting either wrong produces a segfault inside MPI or silent message corruption, not an error
from this package.
"""
gpu_direct_mpi() = _GPU_DIRECT[]

function gpu_direct_mpi!(enabled::Bool)
    if enabled && !(MPI.has_cuda() || MPI.has_rocm())
        @warn """
        gpu_direct_mpi!(true) with an MPI that reports no CUDA/ROCm support. Sending a device
        pointer to it will segfault inside the MPI library. Point MPI.jl at a GPU-aware system
        MPI (`MPIPreferences.use_system_binary()`) -- the JLL binaries installed by default are
        not GPU-aware -- or leave this disabled to stage through host memory.
        """
    end
    _GPU_DIRECT[] = enabled
    return enabled
end

"""
    mpi_device_synchronize()

Hook called before a device-resident payload is handed to MPI and after one is received, to wait
for queued device work. The default does nothing, which is correct while payloads are staged
through host memory. Override it when enabling [`gpu_direct_mpi!`](@ref):

    TensorNetworkQuantumSimulator.mpi_device_synchronize() = CUDA.synchronize()
"""
mpi_device_synchronize() = nothing

function _needs_host_staging(bp_cache)
    _FORCE_HOST_STAGING[] && return true
    return _is_device_backed(bp_cache) && !_GPU_DIRECT[]
end

function _flatten_values(values)
    tensors, counts = ITensor[], Int[]
    for m in values
        if m isa ITensor
            push!(tensors, m)
            push!(counts, 1)
        else
            append!(tensors, m)
            push!(counts, length(m))
        end
    end
    return tensors, counts
end

# Returns `(header, payload)`. A `nothing` payload means the header carries the tensors itself,
# in which case it holds the per-item values rather than the flattened list, so that a message
# stored as a `Vector{ITensor}` is reassembled correctly on the far side.
function _prepare_send(bp_cache, peer, values)
    tensors, counts = _flatten_values(values)
    (isempty(tensors) || !all(_is_raw_transferable, tensors)) && return values, nothing

    T = promote_type(map(eltype, tensors)...)
    canonical = ITensor[
        _canonical_permute(eltype(t) === T ? t : adapt(T, t)) for t in tensors
    ]
    header = TensorBatch([collect(inds(t)) for t in canonical], counts, T)

    lengths = [length(ITensors.data(t)) for t in canonical]
    host_staging = _needs_host_staging(bp_cache)

    # One tensor is the common case (a partition boundary is usually one edge per peer): hand its
    # storage straight to MPI with no staging copy at all. Not available when the payload has to
    # be mirrored to the host first.
    if isone(length(canonical)) && !host_staging
        return header, ITensors.data(only(canonical))
    end

    n = sum(lengths)
    buf = _staging!(bp_cache.buffers.send, bp_cache, peer, T, n)
    offset = 0
    for (t, len) in zip(canonical, lengths)
        copyto!(view(buf, (offset + 1):(offset + len)), ITensors.data(t))
        offset += len
    end
    if !host_staging
        # The packing copies above are queued on a device stream; MPI must not read the buffer
        # until they land.
        _is_device_backed(bp_cache) && mpi_device_synchronize()
        return header, view(buf, 1:n)
    end

    hbuf = view(_host_staging!(bp_cache.buffers.send_host, peer, T, n), 1:n)
    copyto!(hbuf, view(buf, 1:n))
    return header, hbuf
end

function _scatter!(setter, items, header::TensorBatch, payload)
    lengths = _batch_lengths(header)
    offset, k = 0, 0
    for (item, count) in zip(items, header.counts)
        ts = ITensor[]
        for _ in 1:count
            k += 1
            len = lengths[k]
            data = similar(payload, header.eltype, len)
            copyto!(data, view(payload, (offset + 1):(offset + len)))
            push!(ts, itensor(NDTensors.Dense(data), Tuple(header.inds[k])))
            offset += len
        end
        setter(item, isone(count) ? only(ts) : ts)
    end
    return nothing
end

# `send_items`/`recv_items` map a peer rank to the items going to / coming from it, in an order
# both ranks derive from `super_graph`. Distinct tags per phase plus that shared ordering are
# what make matching explicit: it no longer depends on the two ranks happening to walk
# `shared_vertices` in step.
function _exchange!(
        bp_cache, send_items, recv_items, getter, setter, header_tag, payload_tag
    )
    comm = communicator(bp_cache)
    requests = MPI.Request[]

    # Headers. Non-blocking sends are posted before the blocking receives so this cannot
    # deadlock however the peers are ordered.
    send_payloads = Pair{Int32, Any}[]
    for (peer, items) in pairs(send_items)
        isempty(items) && continue
        header, payload = _prepare_send(bp_cache, peer, [getter(item) for item in items])
        push!(requests, MPI.isend(header, comm; dest = peer, tag = header_tag))
        isnothing(payload) || push!(send_payloads, Int32(peer) => payload)
    end

    recv_headers = Pair{Int32, Any}[]
    for (peer, items) in pairs(recv_items)
        isempty(items) && continue
        push!(
            recv_headers,
            Int32(peer) => MPI.recv(comm; source = peer, tag = header_tag)
        )
    end

    # Payloads. Receives are posted first so an eagerly-sent message always has somewhere to
    # land. `recv_payloads` and `send_payloads` also keep every buffer reachable until
    # `Waitall`, so none of them can be collected while MPI is reading or writing it.
    # Keyed rather than positional: a peer whose header took the serialised fallback has no
    # payload, so a parallel array here would slip out of step with `recv_headers`.
    # `_scatter!` must read a payload that lives wherever the network's tensors do, so when MPI
    # cannot write into device memory the data lands in a host mirror and is copied across after
    # `Waitall`. `recv_payloads` always holds the device-resident view.
    host_staging = _needs_host_staging(bp_cache)
    recv_payloads = Dict{Int32, Any}()
    recv_mirrors = Dict{Int32, Any}()
    for (peer, header) in recv_headers
        header isa TensorBatch || continue
        n = sum(_batch_lengths(header); init = 0)
        payload = view(_staging!(bp_cache.buffers.recv, bp_cache, peer, header.eltype, n), 1:n)
        recv_payloads[peer] = payload
        target = payload
        if host_staging
            target = view(
                _host_staging!(bp_cache.buffers.recv_host, peer, header.eltype, n), 1:n
            )
            recv_mirrors[peer] = target
        end
        push!(requests, MPI.Irecv!(target, comm; source = peer, tag = payload_tag))
    end
    for (peer, payload) in send_payloads
        push!(requests, MPI.Isend(payload, comm; dest = peer, tag = payload_tag))
    end
    MPI.Waitall(requests)
    for (peer, mirror) in recv_mirrors
        copyto!(recv_payloads[peer], mirror)
    end
    # On the direct path MPI wrote into device memory; wait for that before reading it back out
    # into tensors. (Staged receives need no barrier: the host->device copies just above are
    # ordered ahead of the reads in `_scatter!` on the same stream.)
    if !host_staging && _is_device_backed(bp_cache) && !isempty(recv_payloads)
        mpi_device_synchronize()
    end

    for (peer, header) in recv_headers
        items = recv_items[peer]
        if header isa TensorBatch
            _scatter!(setter, items, header, recv_payloads[peer])
        else
            for (item, t) in zip(items, header)
                setter(item, t)
            end
        end
    end
    return bp_cache
end

# ---------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------

"""
    BeliefPropagationCacheMPI(cache::BeliefPropagationCache, super_graph, shared_vertices; comm, validate = true)

A `BeliefPropagationCache` holding one partition of a tensor network distributed across the
ranks of `comm`, so that no rank ever stores the whole network.

`super_graph` is the graph of the *global* network (the same on every rank -- only its tensors
are distributed, not its connectivity). `shared_vertices` maps each vertex that sits on a
partition boundary to the pair of ranks holding it; a boundary vertex's tensor is duplicated on
both, and the neighbours it has on the far side become ghost vertices carrying the incoming
boundary messages. Local BP then runs unchanged, with one message exchange per sweep.

`cache` is adopted, not copied: its message dictionary gains an entry per incoming boundary
edge.

The partition must satisfy:

  * every shared vertex is shared by exactly two ranks, and is held by exactly those two;
  * shared vertices are pairwise non-adjacent in `super_graph`, so that every edge lies inside
    exactly one partition;
  * each rank's local network is the subgraph `super_graph` induces on the rank's vertices;
  * every rank passes an identical, identically-ordered `shared_vertices`.

These are checked unless `validate = false` (two `Allreduce`s over the global vertex and edge
lists, worth skipping only for very large graphs). Violating them otherwise tends to hang
rather than fail, because the exchange schedule stops matching up.

This constructor is collective: every rank of `comm` must call it.

See also [`apply_gates_mpi`](@ref), [`inner_mpi`](@ref).
"""
struct BeliefPropagationCacheMPI{
        V,
        BPC <: BeliefPropagationCache{V},
        G <: AbstractGraph,
        SG <: AbstractGraph,
    } <: AbstractBeliefPropagationCache{V}
    local_cache::BPC
    messages_graph::G # local graph plus a ghost vertex per remote neighbour
    super_graph::SG # connectivity of the global network; needed for the exchange order
    vertex_order::Dictionary{V, Int} # deterministic vertex numbering, identical on every rank
    shared_vertices::Dictionary{V, Int32} # duplicated vertex -> the peer rank
    edges_to_send::Dictionary{NamedEdge{V}, Int32} # edge -> mpi rank
    edges_to_recv::Dictionary{NamedEdge{V}, Int32} # edge -> mpi rank
    send_order::Dictionary{Int32, Vector{NamedEdge{V}}} # peer -> edges, canonically ordered
    recv_order::Dictionary{Int32, Vector{NamedEdge{V}}} # peer -> edges, canonically ordered
    buffers::ExchangeBuffers
    comm::MPI.Comm
end

local_cache(bp_cache::BeliefPropagationCacheMPI) = bp_cache.local_cache
messages_graph(bp_cache::BeliefPropagationCacheMPI) = bp_cache.messages_graph
super_graph(bp_cache::BeliefPropagationCacheMPI) = bp_cache.super_graph
communicator(bp_cache::BeliefPropagationCacheMPI) = bp_cache.comm

# The wrapped cache's network and messages are shared by reference, so mutating through
# either view is visible to both.
for f in [
        :(messages),
        :(network),
        :(graph),
        :(contraction_sequences),
        :(edge_sequence),
        :(default_update_alg),
        :(default_message_update_alg),
    ]
    @eval begin
        $f(bp_cache::BeliefPropagationCacheMPI) = $f(local_cache(bp_cache))
    end
end

function edge_scalar(bp_cache::BeliefPropagationCacheMPI, edge::AbstractEdge)
    return edge_scalar(local_cache(bp_cache), edge)
end

# Without this, copy() falls through to NamedGraphs.copy(::AbstractNamedGraph).
function Base.copy(bp_cache::BeliefPropagationCacheMPI)
    return BeliefPropagationCacheMPI(
        copy(local_cache(bp_cache)),
        copy(messages_graph(bp_cache)),
        # Shared, not duplicated: the graph and the derived exchange schedule are never
        # mutated, the buffers are scratch, and a copy stays on the same communicator.
        super_graph(bp_cache),
        bp_cache.vertex_order,
        copy(bp_cache.shared_vertices),
        copy(bp_cache.edges_to_send),
        copy(bp_cache.edges_to_recv),
        bp_cache.send_order,
        bp_cache.recv_order,
        bp_cache.buffers,
        communicator(bp_cache)
    )
end

# Both orientations of every edge, and an empty vector (rather than a `reduce` error) for a
# partition that holds a single vertex.
function _directed_edges(x)
    es = collect(edges(x))
    return isempty(es) ? es : reduce(vcat, [[e, reverse(e)] for e in es])
end

# Moving the cache to a GPU is how the raw-payload exchange becomes GPU-direct: once the
# tensors are `CuArray`-backed, `_alloc_buffer` allocates device staging buffers and MPI is
# handed device pointers. The buffers have to be dropped rather than carried over, though --
# they are keyed by element type only, so host scratch would otherwise be reused for a device
# cache and quietly stage every message through the wrong memory space.
function Adapt.adapt_structure(to, bpc::BeliefPropagationCacheMPI)
    adapted = adapt_factors(to, adapt_messages(to, bpc))
    return BeliefPropagationCacheMPI(
        local_cache(adapted),
        messages_graph(adapted),
        super_graph(adapted),
        adapted.vertex_order,
        adapted.shared_vertices,
        adapted.edges_to_send,
        adapted.edges_to_recv,
        adapted.send_order,
        adapted.recv_order,
        ExchangeBuffers(),
        communicator(adapted)
    )
end

# Index ids are random, so ranks cannot agree on them independently: they are drawn on `root`
# and broadcast. Keyed in both orientations, so lookup works from either endpoint.
function shared_bond_inds(
        super_graph::AbstractGraph;
        bond_dimension::Integer = 1,
        comm::MPI.Comm = MPI.COMM_WORLD,
        root::Integer = 0
    )
    es = collect(edges(super_graph))
    ls = MPI.Comm_rank(comm) == root ? [Index(bond_dimension) for _ in es] : nothing
    ls = MPI.bcast(ls, comm; root)
    return Dictionary([es; reverse.(es)], [ls; ls])
end

# Unlike insert_virtualinds!, this works w.r.t `super_graph`, so an edge leaving `tn`
# (connecting to a vertex on a different rank) needs to get a dangling leg. These legs
# are generated once on root and broadcast as to be consistent across all ranks, and
# passed in as `bond_inds`.
function insert_partition_virtualinds!(
        tn::AbstractTensorNetwork, super_graph::AbstractGraph, bond_inds
    )
    dtype = datatype(tn)
    for v in vertices(tn)
        t = tn[v]
        for vn in neighbors(super_graph, v)
            l = bond_inds[NamedEdge(v => vn)]
            l ∈ inds(t) && continue
            t *= adapt(dtype)(onehot(l => 1))
        end
        setindex_preserve!(tn, t, v)
    end
    return tn
end

# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------

# Every check below *records* what is wrong instead of throwing, and `_validate_partition` runs
# all of them on every rank before anyone throws. Bailing out early on a rank-local condition
# would leave the other ranks waiting in a collective their peer has already abandoned -- which
# is the hang this validation exists to replace.

# True on every rank, or false on every rank: safe to branch on.
function _ranks_agree(value, comm)
    h = hash(repr(value))
    return MPI.Allreduce(h, MPI.MIN, comm) == MPI.Allreduce(h, MPI.MAX, comm)
end

function _check_local!(problems, cache, sgraph, shared_vertices, comm, is_local)
    nranks = MPI.Comm_size(comm)
    me = MPI.Comm_rank(comm)
    local_graph = graph(cache)

    for (v, ranks) in pairs(shared_vertices)
        if !has_vertex(sgraph, v)
            push!(problems, "shared vertex $v is not a vertex of `super_graph`.")
            continue
        end
        if length(ranks) != 2
            push!(
                problems,
                "shared vertex $v is listed as shared between $(length(ranks)) ranks ($ranks). " *
                    "A shared vertex is duplicated on exactly two ranks; separating three or " *
                    "more partitions at one vertex needs an edge-cut partition instead."
            )
            continue
        end
        first(ranks) == last(ranks) &&
            push!(problems, "shared vertex $v lists rank $(first(ranks)) twice.")
        all(r -> 0 <= r < nranks, ranks) || push!(
            problems,
            "shared vertex $v refers to ranks $ranks, outside 0:$(nranks - 1) for this " *
                "communicator."
        )
        if me in ranks
            is_local(v) || push!(
                problems,
                "`shared_vertices` says this rank holds $v, but it is not in the local network."
            )
        else
            is_local(v) && push!(
                problems,
                "this rank holds $v, but `shared_vertices` lists it as shared between $ranks."
            )
        end
    end

    shared = Set(keys(shared_vertices))
    for v in shared, w in neighbors(sgraph, v)
        w in shared && push!(
            problems,
            "shared vertices $v and $w are adjacent in `super_graph`. Both holders would own " *
                "the edge between them, so its message would be sent by both and received by " *
                "neither, and `freenergy` would count it twice. Pick a separator whose " *
                "vertices are pairwise non-adjacent."
        )
    end

    for v in vertices(local_graph)
        has_vertex(sgraph, v) ||
            push!(problems, "local vertex $v is not a vertex of `super_graph`.")
    end
    for e in edges(sgraph)
        (is_local(src(e)) && is_local(dst(e))) || continue
        has_edge(local_graph, e) || push!(
            problems,
            "`super_graph` has the edge $e between two vertices this rank holds, but the local " *
                "network does not. Pass the subgraph induced on the rank's vertices."
        )
    end
    return problems
end

# Every edge must lie inside exactly one partition. An edge in none is never updated; an edge in
# two has its message updated twice per sweep and is double-counted by `freenergy`.
function _check_edge_cover!(problems, sgraph, is_local, comm)
    es = collect(edges(sgraph))
    held = Int32[(is_local(src(e)) && is_local(dst(e))) ? Int32(1) : Int32(0) for e in es]
    total = MPI.Allreduce(held, MPI.SUM, comm)
    bad = findfirst(!isequal(Int32(1)), total)
    isnothing(bad) && return problems
    n = total[bad]
    push!(
        problems,
        iszero(n) ?
            "edge $(es[bad]) of `super_graph` is inside no rank's partition, so belief " *
            "propagation would never update its messages. Partitions must cover every edge." :
            "edge $(es[bad]) of `super_graph` lies inside $n partitions. Every edge must be " *
            "held by exactly one rank; adjacent shared vertices are the usual cause."
    )
    return problems
end

function _check_vertex_cover!(problems, sgraph, is_local, shared_vertices, comm)
    vs = collect(vertices(sgraph))
    held = Int32[is_local(v) ? Int32(1) : Int32(0) for v in vs]
    total = MPI.Allreduce(held, MPI.SUM, comm)
    for (v, n) in zip(vs, total)
        expected = haskey(shared_vertices, v) ? 2 : 1
        n == expected && continue
        push!(
            problems,
            iszero(n) ?
                "vertex $v of `super_graph` is held by no rank." :
                "vertex $v of `super_graph` is held by $n ranks but `shared_vertices` implies " *
                "$expected. A vertex is either local to one rank or shared by exactly two."
        )
    end
    return problems
end

function _validate_partition(cache, sgraph, shared_vertices, comm)
    local_vertices = Set(vertices(graph(cache)))
    is_local = v -> v in local_vertices
    problems = String[]

    _check_local!(problems, cache, sgraph, shared_vertices, comm, is_local)
    _ranks_agree(collect(pairs(shared_vertices)), comm) || push!(
        problems,
        "ranks were given different `shared_vertices`. Every rank must pass an identical, " *
            "identically-ordered dictionary: the message exchange schedule is derived from it."
    )

    # The cover checks reduce one entry per global vertex/edge, so their buffer sizes have to
    # match. Guard them with a fixed-size agreement test first -- and branch on its (globally
    # identical) result, never on anything rank-local.
    if _ranks_agree((nv(sgraph), length(collect(edges(sgraph)))), comm)
        _check_edge_cover!(problems, sgraph, is_local, comm)
        _check_vertex_cover!(problems, sgraph, is_local, shared_vertices, comm)
    else
        push!(problems, "ranks were given `super_graph`s of different sizes.")
    end

    nbad = MPI.Allreduce(isempty(problems) ? 0 : 1, MPI.SUM, comm)
    iszero(nbad) && return nothing
    throw(
        ArgumentError(
            isempty(problems) ?
                "the partition was rejected on $nbad other rank(s); see their errors." :
                "invalid partition on rank $(MPI.Comm_rank(comm)):\n  - " *
                join(problems, "\n  - ")
        )
    )
end

# ---------------------------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------------------------

function BeliefPropagationCacheMPI(
        cache::BeliefPropagationCache,
        sgraph::AbstractGraph,
        shared_vertices::Dictionary; # vertex -> (rank, rank)
        comm::MPI.Comm = MPI.COMM_WORLD,
        validate::Bool = true
    )
    MPI.Initialized() || error(
        "MPI has not been initialised. Call `MPI.Init()` before building a " *
            "BeliefPropagationCacheMPI."
    )
    validate && _validate_partition(cache, sgraph, shared_vertices, comm)

    V = vertextype(sgraph)
    me = MPI.Comm_rank(comm)
    local_graph = graph(cache)
    local_vertices = Set(vertices(local_graph))

    edges_to_send = Dictionary{NamedEdge{V}, Int32}()
    edges_to_recv = Dictionary{NamedEdge{V}, Int32}()

    # Only this rank's shared vertices, so callers can trust keys() when deciding which
    # factors to exchange.
    shared_vertices_other = Dictionary{V, Int32}()

    for (shared_vertex, involved_ranks) in pairs(shared_vertices)
        # Checked even when `validate` is off: destructuring a longer tuple would silently drop
        # the extra ranks and build a partition that is quietly wrong rather than rejected.
        length(involved_ranks) == 2 || throw(
            ArgumentError(
                "shared vertex $shared_vertex is shared between $(length(involved_ranks)) " *
                    "ranks ($involved_ranks); exactly two are supported."
            )
        )
        rank1, rank2 = involved_ranks
        if me == rank1
            other_rank = Int32(rank2)
        elseif me == rank2
            other_rank = Int32(rank1)
        else
            continue
        end
        insert!(shared_vertices_other, shared_vertex, other_rank)

        for neighbor in neighbors(sgraph, shared_vertex)
            edge = NamedEdge{V}(neighbor => shared_vertex)
            if neighbor in local_vertices
                # This rank owns the neighbour, so it owns the message flowing out of it.
                insert!(edges_to_send, edge, other_rank)
            else
                insert!(edges_to_recv, edge, other_rank)
            end
        end
    end

    # A deterministic global vertex numbering. Both ends of every exchange sort their batch by
    # it, so the sender's k-th tensor is the receiver's k-th tensor no matter what order the
    # two ranks discovered their boundary in.
    vertex_order = Dictionary(collect(vertices(sgraph)), 1:nv(sgraph))
    edge_key = e -> (vertex_order[src(e)], vertex_order[dst(e)])
    send_order = _order_by_peer(edges_to_send, edge_key)
    recv_order = _order_by_peer(edges_to_recv, edge_key)

    # Ghost vertices hold the incoming boundary messages. They live only here, so
    # graph(network) stays ghost-free and vertex iteration is unaffected.
    _messages_graph = copy(local_graph)
    for edge in keys(edges_to_recv)
        ghost = src(edge)
        has_vertex(_messages_graph, ghost) || add_vertex!(_messages_graph, ghost)
        add_edge!(_messages_graph, edge)
    end

    bp_cache = BeliefPropagationCacheMPI(
        cache,
        _messages_graph,
        sgraph,
        vertex_order,
        shared_vertices_other,
        edges_to_send,
        edges_to_recv,
        send_order,
        recv_order,
        ExchangeBuffers(),
        comm
    )

    # Populate the ghost messages, which have no entry in `cache` yet.
    return communicate_messages!(bp_cache)
end

function _order_by_peer(items::Dictionary, key)
    K = keytype(items)
    out = Dictionary{Int32, Vector{K}}()
    for (item, peer) in pairs(items)
        haskey(out, peer) || insert!(out, peer, K[])
        push!(out[peer], item)
    end
    for peer in keys(out)
        sort!(out[peer]; by = key)
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------------------------

#TODO: use MPI graph communication primatives.
function communicate_messages!(bp_cache::BeliefPropagationCacheMPI)
    return _exchange!(
        bp_cache,
        bp_cache.send_order,
        bp_cache.recv_order,
        e -> message(bp_cache, e),
        (e, m) -> setmessage!(bp_cache, e, m),
        TAG_MESSAGE_HEADER,
        TAG_MESSAGE_PAYLOAD
    )
end

# Assumes each vertex is on one side of the exchange only. should_apply_gate keeps that true by
# never electing a single rank for a one-site gate.
function communicate_factors!(
        bp_cache::BeliefPropagationCacheMPI,
        vertices_to_send,
        vertices_to_recv
    )
    tn = network(bp_cache)
    key = v -> bp_cache.vertex_order[v]
    _exchange!(
        bp_cache,
        _order_by_peer(_peer_map(bp_cache, vertices_to_send), key),
        _order_by_peer(_peer_map(bp_cache, vertices_to_recv), key),
        v -> tn[v],
        # Not tn[v] = ...: that goes through add_tensor!, which rewires the graph from index
        # overlap.
        (v, t) -> setindex_preserve!(bp_cache, t, v),
        TAG_FACTOR_HEADER,
        TAG_FACTOR_PAYLOAD
    )

    # A received factor carries the sender's index on every bond it truncated, so refresh the
    # boundary messages to match. Otherwise the message and the factor share no index on that
    # bond and BP contracts them into an outer product, adding two free indices per sweep.
    return communicate_messages!(bp_cache)
end

function _peer_map(bp_cache::BeliefPropagationCacheMPI, vs)
    shared = bp_cache.shared_vertices
    out = Dictionary{keytype(shared), Int32}()
    for v in vs
        haskey(out, v) || insert!(out, v, shared[v])
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Belief propagation
# ---------------------------------------------------------------------------------------------

# Boundary messages arrive on ghost edges, which only messages_graph knows about.
function incoming_messages(
        bp_cache::BeliefPropagationCacheMPI, vertices::Vector{<:Any}; ignore_edges = []
    )
    b_edges = boundary_edges(messages_graph(bp_cache), vertices; dir = :in)
    b_edges = !isempty(ignore_edges) ? setdiff(b_edges, ignore_edges) : b_edges
    return messages(bp_cache, b_edges)
end

# Cannot forward to the local cache: updated_message() must dispatch on this type for
# incoming_messages() to pick up the ghost edges.
function update_message!(
        message_update_alg::Algorithm, bp_cache::BeliefPropagationCacheMPI, edge::AbstractEdge
    )
    m, (cache_key, sequence, seq_changed) =
        updated_message(message_update_alg, bp_cache, edge)
    seq_changed && set!(contraction_sequences(bp_cache), cache_key, sequence)
    return setmessage!(bp_cache, edge, m)
end

# Boundary messages advance one partition per sweep, so a rank's own subgraph does not say how
# many sweeps are needed: `is_tree` on a partition of a tree is true, but a chain of P
# partitions still needs P sweeps for information to cross it. Worse, if two ranks disagree
# about their local tree-ness they pick different sweep counts, and the one that finishes first
# stops calling `communicate_messages!` and leaves the others blocked in `MPI.Recv`.
function default_bp_maxiter(bp_cache::BeliefPropagationCacheMPI)
    nparts = MPI.Comm_size(communicator(bp_cache))
    serial_maxiter = is_tree(super_graph(bp_cache)) ? 1 : _default_bp_update_maxiter
    return serial_maxiter + nparts - 1
end

function default_bp_update_kwargs(bp_cache::BeliefPropagationCacheMPI)
    return (;
        maxiter = default_bp_maxiter(bp_cache),
        tolerance = default_tolerance(scalartype(bp_cache)),
        verbose = false,
    )
end

# Collective: harmonises the loop bounds so that ranks handed different kwargs (or relying on
# partition-dependent defaults) still run the same number of sweeps instead of deadlocking.
function set_default_kwargs(alg::Algorithm"bp", bp_cache::BeliefPropagationCacheMPI)
    cache = local_cache(bp_cache)
    comm = communicator(bp_cache)

    verbose = get(alg.kwargs, :verbose, default_verbose(alg))
    maxiter = get(alg.kwargs, :maxiter, default_bp_maxiter(bp_cache))
    _edge_sequence = get(alg.kwargs, :edge_sequence, edge_sequence(cache))
    tolerance = get(alg.kwargs, :tolerance, default_tolerance(alg))
    message_update_alg = set_default_kwargs(
        get(alg.kwargs, :message_update_alg, Algorithm(default_message_update_alg(cache))),
        cache
    )

    maxiter = Int(MPI.Allreduce(Int(maxiter), MPI.MAX, comm))
    # Whether the error is computed at all has to agree too: a rank with no tolerance never
    # breaks out early, so mixing the two hangs just as surely as mismatched `maxiter`.
    any_tolerance = !iszero(
        MPI.Allreduce(isnothing(tolerance) ? Int32(0) : Int32(1), MPI.MAX, comm)
    )
    tolerance = if any_tolerance
        MPI.Allreduce(isnothing(tolerance) ? Inf : Float64(tolerance), MPI.MIN, comm)
    else
        nothing
    end

    return Algorithm(
        "bp"; verbose, maxiter, edge_sequence = _edge_sequence, tolerance, message_update_alg
    )
end

function update_iteration!(
        alg::Algorithm"bp",
        bpc::BeliefPropagationCacheMPI,
        edges::Vector;
        (update_diff!) = nothing
    )
    for e in edges
        prev_message = !isnothing(update_diff!) ? message(bpc, e) : nothing
        update_message!(alg.kwargs.message_update_alg, bpc, e)
        if !isnothing(update_diff!)
            update_diff![] += message_diff(message(bpc, e), prev_message)
        end
    end
    return communicate_messages!(bpc)
end

# Collective, and a near-copy of the generic `update` -- the difference is that the convergence
# test is taken on the global average so that every rank leaves the loop on the same sweep.
function update(alg::Algorithm"bp", bpc::BeliefPropagationCacheMPI)
    isnothing(alg.kwargs.maxiter) && error("You need to specify a number of iterations for BP!")
    comm = communicator(bpc)
    isroot = iszero(MPI.Comm_rank(comm))
    compute_error = !isnothing(alg.kwargs.tolerance)

    bpc = copy(bpc)
    invalidate_contraction_sequences!(bpc)

    # `update_iteration!` only accumulates this rank's edges. Every graph edge lies inside
    # exactly one partition, so the local counts sum to the global one.
    nglobal = MPI.Allreduce(length(alg.kwargs.edge_sequence), MPI.SUM, comm)

    converged, avg_diff, niter = false, nothing, alg.kwargs.maxiter
    for i in 1:alg.kwargs.maxiter
        diff = compute_error ? Ref(0.0) : nothing
        update_iteration!(alg, bpc, alg.kwargs.edge_sequence; (update_diff!) = diff)
        compute_error || continue
        # Allreduce yields the same value on every rank, so the test below breaks the loop at
        # the same sweep everywhere. Reducing the sum and dividing once also keeps a rank that
        # holds no edges from computing 0/0 and never converging.
        total = MPI.Allreduce(diff[], MPI.SUM, comm)
        avg_diff = iszero(nglobal) ? 0.0 : total / nglobal
        if avg_diff <= alg.kwargs.tolerance
            converged, niter = true, i
            break
        end
    end

    # Reported once for the run rather than once per rank.
    if compute_error && isroot
        if converged
            alg.kwargs.verbose &&
                println("BP converged to desired precision after $niter iterations.")
        else
            msg = "BP did not converge to tolerance $(alg.kwargs.tolerance) after $niter iterations (final average message change: $avg_diff)."
            alg.kwargs.verbose ? println(msg) : @warn(msg)
        end
    end

    invalidate_contraction_sequences!(bpc)
    return bpc
end

# Local edges only -- every edge lies inside exactly one partition. Rescaling one changes a
# message the peer holds a copy of, hence the refresh.
function rescale_messages!(
        bp_cache::BeliefPropagationCacheMPI, edges::Vector{<:AbstractEdge}
    )
    rescale_messages!(local_cache(bp_cache), edges)
    return communicate_messages!(bp_cache)
end

# A shared vertex is rescaled on both holders. They divide by the same `vertex_scalar` -- same
# factor, same incoming messages -- so the duplicated tensors stay identical without any
# exchange. This cannot delegate to the local cache, whose `vertex_scalar` would miss the ghost
# half of a shared vertex's environment.
function rescale_vertices!(bp_cache::BeliefPropagationCacheMPI, vertices::Vector)
    tn = network(bp_cache)
    for v in vertices
        vn = vertex_scalar(bp_cache, v)
        s = isreal(vn) ? sign(vn) : one(vn)
        if tn isa TensorNetworkState
            setindex_preserve!(tn, tn[v] * s * inv(sqrt(vn)), v)
        elseif tn isa TensorNetwork
            setindex_preserve!(tn, tn[v] * s * inv(vn), v)
        else
            error("Don't know how to rescale the vertices of this type")
        end
    end
    return communicate_messages!(bp_cache)
end

# Edge terms partition cleanly, since every edge lies inside one region. Vertex terms do not: a
# shared vertex is held by two ranks, so it is counted on the lower-ranked one.
function freenergy(bp_cache::BeliefPropagationCacheMPI)
    comm = communicator(bp_cache)
    me = MPI.Comm_rank(comm)
    shared = bp_cache.shared_vertices
    owned = filter(v -> !haskey(shared, v) || me < shared[v], collect(vertices(bp_cache)))

    numerator_terms = vertex_scalars(bp_cache, owned)
    denominator_terms = edge_scalars(bp_cache, collect(edges(bp_cache)))

    # The reduction needs one scalar type across all ranks, so the choice is made collectively.
    # It can only ever promote: the terms are already complex for a complex network, and a
    # negative term in a real one forces a complex logarithm. Demoting would throw an
    # InexactError on the small imaginary parts a complex network leaves behind.
    S = float(real(scalartype(bp_cache)))
    negative =
        any(t -> real(t) < 0, numerator_terms) || any(t -> real(t) < 0, denominator_terms)
    complex_here = scalartype(bp_cache) <: Complex || negative
    if iszero(MPI.Allreduce(complex_here ? Int32(1) : Int32(0), MPI.MAX, comm))
        return MPI.Allreduce(_local_freenergy(S, numerator_terms, denominator_terms), MPI.SUM, comm)
    end
    return MPI.Allreduce(
        _local_freenergy(Complex{S}, numerator_terms, denominator_terms), MPI.SUM, comm
    )
end

# -Inf rather than an early return: every rank must reach the Allreduce.
function _local_freenergy(::Type{T}, numerator_terms, denominator_terms) where {T}
    any(iszero, denominator_terms) && return T(-Inf)
    return sum(log.(T.(numerator_terms)); init = zero(T)) -
        sum(log.(T.(denominator_terms)); init = zero(T))
end

# ---------------------------------------------------------------------------------------------
# Gate application
# ---------------------------------------------------------------------------------------------

"""
    apply_gates_mpi(circuit, ψ, super_graph, shared_vertices; comm, kwargs...)

Apply `circuit` to a tensor network state distributed across the ranks of `comm`, where `ψ` is
this rank's partition and `super_graph` the graph of the global state.

`circuit` and `shared_vertices` must be identical on every rank; gate supports are resolved
against `super_graph`, so a gate may straddle a partition boundary. Returns this rank's updated
partition and the truncation errors, which are reduced across ranks so that every rank sees the
error of every gate.

Collective: every rank of `comm` must call it.

See also [`BeliefPropagationCacheMPI`](@ref).
"""
function apply_gates_mpi(
        circuit::Vector,
        ψ::TensorNetworkState,
        super_graph::AbstractGraph,
        shared_vertices::Dictionary;
        comm::MPI.Comm = MPI.COMM_WORLD,
        bp_update_kwargs = nothing,
        validate::Bool = true,
        kwargs...
    )
    ψ_bpc = BeliefPropagationCache(ψ)
    # Seed deltas, not a serial update: that update's incoming_messages sees only local edges,
    # so it never contracts the cut bonds' dangling indices and they survive as free indices in
    # the boundary messages, which then have ndims > 2.
    es = _directed_edges(ψ)
    setmessages!(ψ_bpc, es, [default_message(ψ_bpc, e) for e in es])
    ψ_bpc = BeliefPropagationCacheMPI(ψ_bpc, super_graph, shared_vertices; comm, validate)
    # Resolved here rather than as a keyword default: the defaults have to be read off the
    # distributed cache, since the local partition's tree-ness says nothing about the global
    # network's.
    bp_update_kwargs =
        isnothing(bp_update_kwargs) ? default_bp_update_kwargs(ψ_bpc) : bp_update_kwargs
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    ψ_bpc, truncation_errors = apply_gates(circuit, ψ_bpc; bp_update_kwargs, kwargs...)
    return network(ψ_bpc), truncation_errors
end

function apply_gates(
        circuit::Vector,
        ψ_bpc::BeliefPropagationCacheMPI;
        kwargs...
    )
    # Resolved against `super_graph`, so every rank agrees on the support of every gate --
    # including the gates it does not hold. Reading supports off the local network instead would
    # silently shrink a boundary-straddling two-site gate to one vertex, and desynchronise the
    # cache-update points in `_apply_gates`.
    sgraph = super_graph(ψ_bpc)
    gate_vertices = [collect_vertices(gate[2], sgraph) for gate in circuit]
    return _apply_gates(circuit, ψ_bpc; gate_vertices, kwargs...)
end

function apply_gates(
        circuit::Vector{<:ITensor},
        ψ_bpc::BeliefPropagationCacheMPI;
        gate_vertices::Vector = _gate_vertices_required(),
        kwargs...
    )
    return _apply_gates(circuit, ψ_bpc; gate_vertices, kwargs...)
end

function _gate_vertices_required()
    return error(
        "apply_gates on a BeliefPropagationCacheMPI needs an explicit `gate_vertices`, resolved " *
            "against the super graph. A gate's ITensor only names the site indices of the " *
            "partition it came from, so the support of a gate that crosses a boundary cannot be " *
            "recovered from it. Either pass the circuit as tuples, which are resolved against " *
            "the super graph automatically, or pass `gate_vertices` yourself."
    )
end

# Tuples are converted to ITensors lazily, per rank, because `toitensor` needs the site index of
# every vertex the gate acts on and only a rank holding all of them has them. A rank that skips
# a gate never needs its ITensor.
_gate_itensor(gate::ITensor, ψ_bpc) = gate
function _gate_itensor(gate, ψ_bpc)
    return first(toitensor(gate, graph(ψ_bpc), siteinds(network(ψ_bpc))))
end

function _apply_gates(
        circuit::Vector,
        ψ_bpc::BeliefPropagationCacheMPI;
        gate_vertices::Vector,
        apply_kwargs = (;),
        bp_update_kwargs = default_bp_update_kwargs(ψ_bpc),
        update_cache = true,
        verbose = false
    )
    comm = communicator(ψ_bpc)
    isroot = iszero(MPI.Comm_rank(comm))
    ψ_bpc = copy(ψ_bpc)

    V = eltype(vertices(network(ψ_bpc)))
    # Hoisted out of the loop: `vertices(...)` is a lazy view whose `in` is linear in the
    # partition size, and `should_apply_gate` tests it once per gate vertex.
    my_vertices = Set{V}(vertices(network(ψ_bpc)))
    shared_vertices_dict = ψ_bpc.shared_vertices

    # we keep track of the vertices that have been acted on by 2-qubit gates
    # only they increase the counter
    # this is the set that keeps track.
    affected_vertices = Set{V}()
    truncation_errors = zeros((length(circuit)))

    vertices_to_send = V[]
    vertices_to_recv = V[]

    # If the circuit is applied in the Heisenberg picture, the circuit needs to already be reversed
    for (ii, gate) in enumerate(circuit)
        v⃗ = gate_vertices[ii]

        # check if the gate is a 2-qubit gate and whether it affects the counter
        # we currently only increment the counter if the gate affects vertices that have already been affected
        # This depends only on the circuit and the gate index, both identical on every rank, so
        # all ranks reach the exchange below on the same gate. That is why `affected_vertices`
        # is updated for every gate, not just the ones this rank applies.
        cache_update_required =
            length(v⃗) >= 2 &&
            any(vert in affected_vertices for vert in v⃗)

        # update the BP cache
        if update_cache && cache_update_required
            if verbose && isroot
                println("Updating BP cache")
            end

            communicate_factors!(ψ_bpc, vertices_to_send, vertices_to_recv)

            t = @timed ψ_bpc = update(ψ_bpc; bp_update_kwargs...)

            empty!(affected_vertices)
            empty!(vertices_to_send)
            empty!(vertices_to_recv)

            if verbose && isroot
                println("Done in $(t.time) secs")
            end
        end

        # Only apply the gate if *all* gate vertices are local/shared.
        iapply, shared_vertex = should_apply_gate(v⃗, my_vertices, shared_vertices_dict)

        if iapply
            gate = adapt_gate(_gate_itensor(gate, ψ_bpc), ψ_bpc)

            ψ_bpc, truncation_errors[ii] = apply_gate!(
                gate,
                ψ_bpc;
                v⃗,
                apply_kwargs
            )

            # We did this, so we must send the shared vertices (if any)
            isnothing(shared_vertex) || push!(vertices_to_send, shared_vertex)
        elseif !isnothing(shared_vertex) && shared_vertex ∈ my_vertices
            # Someone else did this, so make sure to later get the shared vertex
            push!(vertices_to_recv, shared_vertex)
        end

        for v in v⃗
            push!(affected_vertices, v)
        end
    end

    if update_cache
        communicate_factors!(ψ_bpc, vertices_to_send, vertices_to_recv)
        ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    end

    # Each gate is applied by one rank -- or by both holders of a shared vertex for a one-site
    # gate, which truncate identically -- so the ranks that skipped it contribute 0. Without
    # this, `maximum(truncation_errors)` would be a partition-local answer.
    truncation_errors = MPI.Allreduce(truncation_errors, MPI.MAX, comm)

    return ψ_bpc, truncation_errors
end

function adapt_gate(gate::ITensor, ψ_bpc::BeliefPropagationCacheMPI)
    gate = if scalartype(gate) <: Complex
        adapt(complex(scalartype(ψ_bpc)), gate)
    else
        adapt(scalartype(ψ_bpc), gate)
    end
    return adapt(unspecify_type_parameters(datatype(ψ_bpc)), gate)
end

function should_apply_gate(gate_vertices, local_vertices, shared_vertices)
    touched = filter(in(keys(shared_vertices)), gate_vertices)

    isempty(touched) && return all(in(local_vertices), gate_vertices), nothing

    # Two shared vertices would have to be adjacent, which the constructor rejects.
    shared_vertex = only(touched)

    if shared_vertex ∈ local_vertices
        if length(gate_vertices) == 1
            # Both holders apply it and nothing is exchanged: a one-site gate preserves indices,
            # so the copies stay identical. Electing one rank instead puts the vertex in both
            # holders' send and recv lists, because a one-site gate never sets
            # cache_update_required and so shares a batch with a two-site gate on the same
            # vertex; communicate_factors! then swaps the two tensors rather than propagating
            # one, leaving the shared vertex's bond index disagreeing with its neighbour's.
            return true, nothing
        elseif length(gate_vertices) == 2
            all(in(local_vertices), gate_vertices) && return (true, shared_vertex)
        else
            throw(ArgumentError("got gate on more than 2 vertices: $gate_vertices"))
        end
    end

    return false, shared_vertex
end

# ---------------------------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------------------------

"""
    inner_mpi(ψ, ϕ, super_graph, shared_vertices; comm, bp_update_kwargs)

⟨ψ|ϕ⟩ over a network distributed across the ranks of `comm`, with `ψ` and `ϕ` this rank's
partitions. Returns the same global scalar on every rank.

Collective: every rank of `comm` must call it.
"""
function inner_mpi(
        ψ::TensorNetworkState,
        ϕ::TensorNetworkState,
        super_graph::AbstractGraph,
        shared_vertices::Dictionary;
        comm::MPI.Comm = MPI.COMM_WORLD,
        bp_update_kwargs = nothing,
        validate::Bool = true
    )
    bpc = BeliefPropagationCache(BilinearForm(ψ, ϕ))
    es = _directed_edges(bpc)
    setmessages!(bpc, es, [default_message(bpc, e) for e in es])
    bpc = BeliefPropagationCacheMPI(bpc, super_graph, shared_vertices; comm, validate)
    bp_update_kwargs =
        isnothing(bp_update_kwargs) ? default_bp_update_kwargs(bpc) : bp_update_kwargs
    bpc = update(bpc; bp_update_kwargs...)
    return inner(Algorithm("bp"), bpc)
end

default_alg(bp_cache::BeliefPropagationCacheMPI) = "bp"

"""
    expect(cache::BeliefPropagationCacheMPI, observable; alg = "bp")

Expectation value of `observable` on a distributed cache. Unlike [`inner_mpi`](@ref) this is a
purely local quantity: the observable's support -- and the region BP contracts for it -- must
lie inside this rank's partition, and only the ranks holding it get an answer. Ranks can
therefore measure different observables, and no communication happens.
"""
function expect(
        alg::Algorithm"bp",
        cache::BeliefPropagationCacheMPI,
        obs::Tuple
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(cache))
    iszero(coeff) && return zero(coeff)
    for v in obs_vs
        has_vertex(graph(cache), v) || error(
            "observable $obs is supported on vertex $v, which rank " *
                "$(MPI.Comm_rank(communicator(cache))) does not hold. An observable measured on a " *
                "distributed cache must lie inside one partition."
        )
    end
    return _expect_bp(cache, op_strings, obs_vs, coeff)
end

function expect(
        alg::Algorithm"bp",
        cache::BeliefPropagationCacheMPI,
        observables::Vector{<:Tuple};
        kwargs...
    )
    return map(obs -> expect(alg, cache, obs; kwargs...), observables)
end

function expect(
        cache::BeliefPropagationCacheMPI, observable;
        alg::Union{String, Nothing} = default_alg(cache), kwargs...
    )
    alg == "bp" || error(
        "Only the 'bp' algorithm is supported on a BeliefPropagationCacheMPI, got '$alg'."
    )
    return expect(Algorithm(alg), cache, observable; kwargs...)
end
