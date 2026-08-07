using ITensors: Algorithm, array, dim
using TensorOperations: TensorOperations, TensorOperations as TO, BufferAllocator

# A memory-bounded message update for a double-layer network, reached as
# `update(bpc; message_update_alg = Algorithm("blocked"; b = 64))` and falling back to "contract"
# on anything else. One block is an `ncon` network -- one label per bond -- so the kernel is generic
# in the number of legs, and blocking the outgoing leg keeps peak at ~1 factor plus a self-sizing
# `BufferAllocator` arena, against the ~6 factors the generic path holds.

# The arena, reused across edges and sweeps when the cache carries one (`BeliefPropagationCacheMPI`
# has a `scratch` field); any other cache gets a throwaway per call, same peak but more churn.
message_scratch(::AbstractBeliefPropagationCache) = Base.RefValue{Any}(nothing)

# Byte storage in the prototype's own array family, so a device ket gets a device arena.
_byte_storage_type(proto::AbstractArray) = typeof(similar(proto, UInt8, 0))

function message_allocator!(ref::Base.RefValue{Any}, proto::AbstractArray)
    S = _byte_storage_type(proto)
    alloc = ref[]
    if !(alloc isa BufferAllocator{S})
        @debug "building message arena" storage = S replacing = typeof(alloc) discarded_bytes =
            _arena_bytes(alloc)
        # Drop the old buffer before asking for the replacement, or both are live across the build.
        ref[] = nothing
        alloc isa BufferAllocator && free_scratch_buffer!(alloc.buffer)
        ref[] = BufferAllocator{S}()
    end
    return ref[]::BufferAllocator{S}
end

_arena_bytes(alloc::BufferAllocator) = length(alloc)
_arena_bytes(::Any) = 0

"""
    message_arena_stats(bp_cache)

Capacity, live offset and high-water mark (bytes) of the arena backing the "blocked" message
update, or `nothing` when this cache has no arena yet. `offset` is nonzero only mid-contraction, so
anything else after a solve means a block leaked its temporaries.

`ENV["JULIA_DEBUG"] = "TensorNetworkQuantumSimulator"` reports the same per message.
"""
function message_arena_stats(bp_cache::AbstractBeliefPropagationCache)
    alloc = message_scratch(bp_cache)[]
    alloc isa BufferAllocator || return nothing
    return (;
        capacity = length(alloc), offset = Int(alloc.offset),
        high_water = Int(alloc.max_offset), storage = typeof(alloc.buffer),
    )
end

# Hook for handing a device buffer back eagerly; a GPU caller adds
# `TensorNetworkQuantumSimulator.free_scratch_buffer!(x::CuArray) = CUDA.unsafe_free!(x)`.
# Deliberately not `CUDA.reclaim()`, which returns the block to the driver instead of the pool.
free_scratch_buffer!(x) = nothing

# Called once a solve is done, so the arena does not squat through gate application.
release_message_scratch!(bpc::AbstractBeliefPropagationCache) = bpc

# Strided/Base backends contract by permuting into temporaries, so the closing layer must be
# pre-permuted or `makeblascontractable` copies it once per block. cuTENSOR takes arbitrary index
# modes, so there the copy is a wasted factor.
_needs_aligned_closer(::TO.AbstractBackend) = true
_needs_aligned_closer(::TO.cuTENSORBackend) = false

# "blocked agrees with contract" holds trivially whenever blocked fell back, so a kernel that has
# gone inert is indistinguishable from one that works without this.
const _BLOCKED_MESSAGE_HITS = Ref(0)

# Compiled once per message, not per block: `ncon` re-derives its tree and index permutations on
# every call, over runtime-length tuples, which measured 2-3x the whole kernel at chi=16. The
# network is a chain, so this is a fold. Steps read slots `ia`/`ib` and write `io`; slots
# `1:length(network)` are inputs and `io == 0` is the caller's destination.
function _contraction_plan(network, conjs, output)
    n = length(network)
    steps = Any[]
    ia, IA, conjA = 1, tuple(network[1]...), conjs[1]
    for k in 2:n
        final = k == n
        IB, conjB = tuple(network[k]...), conjs[k]
        IC = final ? tuple(output...) : tuple(symdiff(IA, IB)...)
        pA, pB, pAB = TO.contract_indices(IA, IB, IC)
        io = final ? 0 : n + k - 1
        # Inputs own slots `1:n` and must survive; anything above is an intermediate this step
        # consumes. Decided here, where the slots are handed out, rather than inferred downstream.
        push!(steps, (; ia, ib = k, io, conjA, conjB, pA, pB, pAB, free_a = ia > n, free_b = k > n))
        ia, IA, conjA = io, IC, false
    end
    return steps, 2n - 2
end

# Both the checkpoint/reset and the per-step `tensorfree!` are needed: a `BufferAllocator` reclaims
# on the reset and no-ops the free, a `ManualAllocator` mallocs and no-ops the reset.
function _run_plan!(dest, slots, steps, backend, alloc)
    final = length(steps)
    cp = TO.allocator_checkpoint!(alloc)
    try
        for (k, s) in enumerate(steps)
            A, B = slots[s.ia], slots[s.ib]
            C = if k == final
                dest
            else
                TC = TO.promote_contract(TO.scalartype(A), TO.scalartype(B))
                TO.tensoralloc_contract(
                    TC, A, s.pA, s.conjA, B, s.pB, s.conjB, s.pAB, Val(true), alloc
                )
            end
            TO.tensorcontract!(
                C, A, s.pA, s.conjA, B, s.pB, s.conjB, s.pAB,
                TO.One(), TO.Zero(), backend, alloc
            )
            s.free_a && TO.tensorfree!(A, alloc)
            s.free_b && TO.tensorfree!(B, alloc)
            k == final || (slots[s.io] = C)
        end
    finally
        TO.allocator_reset!(alloc, cp)
    end
    return dest
end

# out[l_e', l_e] = Σ conj(T)[s, l_a', l_b', l_e'] ma[l_a, l_a'] mb[l_b, l_b'] T[s, l_a, l_b, l_e],
# generalised to any number of legs. `tensors[1]`/`network[1]` is the layer being sliced; `-1` is
# the bra's outgoing leg and `-2` the ket's.
function blocked_message!(
        out, tensors, network, conjs, sliced, slicedim, b, backend, alloc
    )
    _BLOCKED_MESSAGE_HITS[] += 1
    steps, nslots = _contraction_plan(network, conjs, (-1, -2))
    slots = Vector{Any}(undef, nslots)
    slots[eachindex(tensors)] .= tensors

    ke = size(sliced, slicedim)
    @debug "blocked message" backend blocksize = b nblocks = cld(ke, b) outgoing = ke steps =
        length(steps) arena_bytes = _arena_bytes(alloc)
    for lo in 1:b:ke
        cols = lo:min(lo + b - 1, ke)
        slots[1] = selectdim(sliced, slicedim, cols)
        # Stride-1 rows, so the closing gemm lands straight in it with no temporary.
        message_slice = view(out, :, cols)
        _run_plan!(message_slice, slots, steps, backend, alloc)
    end
    return out
end

# Block scratch is O(b/chi) of a factor, so `b` has to track chi to hold a bound: chi/16 keeps it at
# 12.5%, and the cap is where the closing gemm is already compute-bound in fp32.
default_blocked_blocksize(chi::Integer) = clamp(chi ÷ 16, 1, 64)
default_normalize(::Algorithm"blocked") = true

# Per-block overhead is fixed while `b` tracks chi, so on a small vertex it is the whole runtime
# (1.8x at chi=16). Below this much scratch there is no pressure to relieve, so take the leg whole.
const _BLOCK_MIN_BYTES = 4 * 2^20

# Only ever raises `default_blocked_blocksize`, so its bound still binds wherever it matters.
# `slabbytes` is one outgoing index's worth of the ket.
function _blocked_blocksize(ke::Integer, slabbytes::Integer)
    floor_b = cld(_BLOCK_MIN_BYTES, max(slabbytes, 1))
    return clamp(max(default_blocked_blocksize(ke), floor_b), 1, ke)
end

# Which two layers close at a vertex, or `nothing` to fall back. `bra === nothing` is the derived
# bra (`conj(ket)` on the ket's own array), `sites === nothing` means the layers share their site
# indices, `op === nothing` means they are joined directly. Dispatch is on the network type because
# a single-layer `TensorNetwork` shows one virtual index per edge just like a norm network.
_blocked_layers(::AbstractTensorNetwork, v, edge) = nothing

function _blocked_layers(tns::TensorNetworkState, v, edge)
    les = virtualinds(tns, edge)
    length(les) == 1 || return nothing
    le = only(les)
    return (; ket = tns[v], bra = nothing, le_ket = le, le_bra = prime(dag(le)),
        sites = nothing, op = nothing)
end

# A `QuadraticForm`'s bra is `dag(prime(ket))`, so it takes the derived route and needs no copy --
# which is what makes `⟨O|V|O⟩` cost the same as `‖O‖²`.
_blocked_layers(qf::QuadraticForm, v, edge) = _blocked_form_layers(qf, v, edge, nothing)

_blocked_layers(form::AbstractForm, v, edge) =
    _blocked_form_layers(form, v, edge, bra_tensor(form, v))

function _blocked_form_layers(form, v, edge, bra)
    lek, leb = virtualinds(ket(form), edge), bra_virtualinds(form, edge)
    (length(lek) == 1 && length(leb) == 1) || return nothing
    # An operator with its own bond on this edge is a third layer, and the message is not rank 2.
    isempty(virtualinds(operator(form), edge)) || return nothing

    K, op = ket(form)[v], operator(form)[v]
    sites = collect(commoninds(K, op))
    pairing = _operator_site_pairing(op, sites)
    isnothing(pairing) && return nothing
    partners, isidentity = pairing
    return (; ket = K, bra, le_ket = only(lek), le_bra = only(leb),
        sites = (sites, partners), op = isidentity ? nothing : op)
end

# The bra-side partner of each ket site index, plus whether `op` is the identity pairing them -- in
# which case the caller drops it from the network. Anything else is carried as one more tensor.
function _operator_site_pairing(op::ITensor, sites)
    isempty(sites) && return nothing
    partners = [prime(dag(s)) for s in sites]
    ndims(op) == 2 * length(sites) || return nothing
    all(i -> i ∈ inds(op), sites) && all(i -> i ∈ inds(op), partners) || return nothing
    S = prod(dim, sites; init = 1)
    # On the host: `isapprox(A, I)` reaches for the diagonal by scalar index. S x S, so free.
    m = Array(reshape(array(op, sites..., partners...), S, S))
    isid = isapprox(m, LinearAlgebra.I; atol = sqrt(eps(real(float(eltype(m))))))
    return (partners, isid)
end

function set_default_kwargs(alg::Algorithm"blocked", bp_cache::AbstractBeliefPropagationCache)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    # `b` defers to the outgoing dimension, `backend` to TensorOperations' own selection.
    b = get(alg.kwargs, :b, nothing)
    backend = get(alg.kwargs, :backend, nothing)
    return Algorithm("blocked"; normalize, b, backend)
end

# Axes in order with `last` moved to the end, or `nothing` when it already is.
function _closer_align_perm(n::Int, last::Int)
    last == n && return nothing
    return (ntuple(i -> i < last ? i : i + 1, n - 1)..., last)
end

# The bra-side partner of each ket site index, in the ket's stored order.
function _bra_sites(sites, ksites)
    isnothing(sites) && return ksites
    formket, formbra = sites
    pos = [findfirst(==(s), formket) for s in ksites]
    any(isnothing, pos) && return nothing
    return formbra[pos]
end

# Each axis carries whichever bond its index names. `0` is never a valid `ncon` label, so it marks
# an index belonging to no bond -- a shape this kernel does not handle.
function _bond_labels(indices, lab::AbstractDict)
    labels = [get(lab, x, 0) for x in indices]
    return all(!=(0), labels) ? labels : nothing
end

# The *unnormalised* message along `edge`, or `nothing` when the kernel does not apply. Split out
# because `vertex_scalar` wants the same contraction and has to know whether it ran.
function _blocked_message(alg::Algorithm"blocked", bp_cache::AbstractBeliefPropagationCache, edge)
    v = src(edge)
    layers = _blocked_layers(network(bp_cache), v, edge)
    isnothing(layers) && return nothing
    K, B, le, le_bra = layers.ket, layers.bra, layers.le_ket, layers.le_bra

    # Messages are read as (ket leg, bra leg); a boundary-MPS cache stores something else.
    ms = incoming_messages(bp_cache, v; ignore_edges = (reverse(edge),))
    all(m -> m isa ITensor && length(inds(m)) == 2, ms) || return nothing
    all(m -> length(commoninds(m, K)) == 1, ms) || return nothing

    is = collect(inds(K))
    epos = findfirst(==(le), is)
    isnothing(epos) && return nothing
    legs = [only(commoninds(m, K)) for m in ms]
    bralegs = [only(setdiff(collect(inds(m)), [legs[i]])) for (i, m) in enumerate(ms)]
    # The layers must stay distinguishable, or the labelling aliases a ket bond onto a bra one.
    (allunique(legs) && le ∉ legs && isdisjoint(legs, bralegs)) || return nothing

    # Whatever is left on the ket after the message legs and the outgoing leg is a site index.
    ksites = [i for i in is if i != le && i ∉ legs]
    isempty(ksites) && return nothing
    bsites = _bra_sites(layers.sites, ksites)
    isnothing(bsites) && return nothing

    # Ket legs take 1..nm and everything else sits above, matching the order the tensors are listed
    # in: messages fold into the block one at a time, the bra closes last.
    nm, ns = length(ms), length(ksites)
    ketlab, bralab = Dict{Index, Int}(le => -2), Dict{Index, Int}(le_bra => -1)
    for (i, l) in enumerate(legs)
        ketlab[l], bralab[bralegs[i]] = i, nm + i
    end
    for (j, s) in enumerate(ksites)
        # Without an operator the layers are joined directly and share one site bond.
        ketlab[s] = 2nm + j
        bralab[bsites[j]] = isnothing(layers.op) ? 2nm + j : 2nm + ns + j
    end
    bothlab = merge(ketlab, bralab)   # the messages and the operator straddle both layers

    # A derived bra *is* the ket's array, so it is labelled through the ket's axis order: axis `p`
    # carries the bra-side partner of `is[p]`. That makes the two bra routes one code path.
    partner = Dict{Index, Index}(le => le_bra)
    for (i, l) in enumerate(legs)
        partner[l] = bralegs[i]
    end
    for (j, s) in enumerate(ksites)
        partner[s] = bsites[j]
    end
    bis = isnothing(B) ? [partner[x] for x in is] : collect(inds(B))
    length(bis) == nm + ns + 1 || return nothing
    bepos = findfirst(==(le_bra), bis)
    isnothing(bepos) && return nothing

    ketlabels = _bond_labels(is, ketlab)
    bralabels = _bond_labels(bis, bralab)
    mlabels = [_bond_labels(collect(inds(m)), bothlab) for m in ms]
    oplabels = isnothing(layers.op) ? Int[] :
        _bond_labels(collect(inds(layers.op)), bothlab)
    (
        isnothing(ketlabels) || isnothing(bralabels) || isnothing(oplabels) ||
            any(isnothing, mlabels)
    ) && return nothing

    A = array(K)                                  # dims follow inds(K); a view when dense
    Barr = isnothing(B) ? nothing : array(B)
    marrays = [array(m) for m in ms]
    oparr = isnothing(layers.op) ? nothing : array(layers.op)

    # Once per message: it picks the backend *and* whether the closing layer needs aligning.
    backend = something(
        get(alg.kwargs, :backend, nothing),
        TO.select_backend(TO.tensorcontract!, A, A, A)
    )
    alloc = message_allocator!(message_scratch(bp_cache), A)

    # `clamp`, not `min`: b < 1 gives the block loop a zero step.
    ke = dim(le)
    bkw = get(alg.kwargs, :b, nothing)
    b = if isnothing(bkw)
        _blocked_blocksize(ke, (length(A) ÷ ke) * sizeof(eltype(A)))
    else
        clamp(bkw, 1, ke)
    end

    # A non-identity operator and a separate bra are each just one more entry in the network.
    op_entry = isnothing(oparr) ? () : (oparr,)
    T = TO.promote_contract(
        map(eltype, (A, marrays..., op_entry..., (isnothing(Barr) ? () : (Barr,))...))...
    )
    out = similar(A, T, (dim(le_bra), ke))

    cp = TO.allocator_checkpoint!(alloc)          # the aligned copy is arena-backed too
    try
        closer = isnothing(B) ? A : Barr
        closerlabels = bralabels
        sliced, slicedim, ketblocklabels = A, epos, ketlabels

        perm = _needs_aligned_closer(backend) ?
            _closer_align_perm(length(closerlabels), bepos) : nothing

        if !isnothing(perm)
            pv = collect(perm)
            closer = _aligned_copy(closer, perm, backend, alloc)
            closerlabels = bralabels[pv]
            # A derived bra makes the aligned copy do double duty, so the blocks come out of it. A
            # separate bra closes instead, leaving the ket to be read one raw slab at a time.
            if isnothing(B)
                sliced, slicedim = closer, length(is)
                ketblocklabels = ketlabels[pv]
            end
        end

        tensors = Any[sliced, marrays..., op_entry..., closer]
        conjs = Bool[false, falses(nm)..., falses(length(op_entry))..., isnothing(B)]
        net = Vector{Int}[
            ketblocklabels, mlabels...,
            (isempty(op_entry) ? () : (oplabels,))..., closerlabels,
        ]

        blocked_message!(out, tensors, net, conjs, sliced, slicedim, b, backend, alloc)
    finally
        TO.allocator_reset!(alloc, cp)
    end

    # `out` is [bra, ket] -- label it rather than transpose. No copy: it is not arena memory.
    return itensor(out, le_bra, le)
end

# `tensoradd!` rather than `permutedims!` so the permute goes through the chosen backend.
function _aligned_copy(A, perm, backend, alloc)
    C = TO.tensoralloc_add(eltype(A), A, (perm, ()), false, Val(true), alloc)
    TO.tensoradd!(C, A, (perm, ()), false, TO.One(), TO.Zero(), backend, alloc)
    return C
end

# Z(v) = Σ_{lₑ,lₑ'} out[lₑ', lₑ] · m_{w→v}[lₑ, lₑ'] -- the same contraction, closed against the one
# message the kernel leaves free. Worth routing here because `contract` holds 5.0-5.4 factors on a
# degree-3 vertex against ~1.2 here, and this is the engine under `freenergy`/`norm_sqr`/`inner`.
function blocked_vertex_scalar(bp_cache::AbstractBeliefPropagationCache, v)
    # Same accessor `incoming_messages` uses, so ghost and shared vertices resolve identically.
    in_edges = NamedGraphs.GraphsExtensions.boundary_edges(
        messages_graph(bp_cache), [v]; dir = :in
    )
    isempty(in_edges) && return nothing
    back = first(in_edges)
    out = _blocked_message(
        Algorithm("blocked"; normalize = false, b = nothing, backend = nothing),
        bp_cache, reverse(back)
    )
    isnothing(out) && return nothing
    closing = message(bp_cache, back)
    closing isa ITensor || return nothing
    # Both of `out`'s indices must be contracted, or this is not the scalar it claims to be.
    isempty(uniqueinds(out, closing)) && isempty(uniqueinds(closing, out)) || return nothing
    return scalar(out * closing)
end

function updated_message(
        alg::Algorithm"blocked", bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge
    )
    m = _blocked_message(alg, bp_cache, edge)
    isnothing(m) && return updated_message(
        set_default_kwargs(
            Algorithm("contract"; normalize = get(alg.kwargs, :normalize, true)), bp_cache
        ),
        bp_cache, edge
    )
    if get(alg.kwargs, :normalize, true)
        message_norm = sum(m)
        if !iszero(message_norm)
            m = m / message_norm
        end
    end
    # No contraction sequence is used, so `seq_changed = false` leaves that cache untouched.
    return m, (src(edge) => edge, nothing, false)
end
