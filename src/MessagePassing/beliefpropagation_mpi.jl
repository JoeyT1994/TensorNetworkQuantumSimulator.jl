using Dictionaries: Dictionary, delete!, set!
using Graphs: AbstractGraph, connected_components, is_tree
using ITensors.NDTensors: scalartype
using ITensors: Algorithm, ITensor, delta
using MPI
using NamedGraphs.GraphsExtensions: a_star, boundary_edges, default_root_vertex,
    forest_cover, forest_cover_edge_sequence, leaf_vertices, post_order_dfs_edges

# An EDGE CUT partitioning: every vertex is owned by one rank and no tensor is duplicated. For a cut
# edge, this rank computes the outgoing message and receives the incoming one, hence `edge_sequence`.
struct BeliefPropagationCacheMPI{
        V,
        BPC <: BeliefPropagationCache{V},
        G <: AbstractGraph,
    } <: AbstractBeliefPropagationCache{V}
    local_cache::BPC
    messages_graph::G # local graph plus a ghost vertex per remote neighbour
    ghost_ranks::Dictionary{V, Int32} # remote neighbour of a local vertex -> its owning rank
    edges_to_send::Dictionary{NamedEdge{V}, Int32} # local -> ghost: computed here, sent there
    edges_to_recv::Dictionary{NamedEdge{V}, Int32} # ghost -> local: computed there, received here
    comm::MPI.Comm
    # Work buffer for the blocked message update (blockedmessage.jl), grown on first use. A `Ref`
    # so an immutable struct can replace it; holds no semantic state, so `copy` shares it.
    scratch::Base.RefValue{Any}
end

local_cache(bp_cache::BeliefPropagationCacheMPI) = bp_cache.local_cache
messages_graph(bp_cache::BeliefPropagationCacheMPI) = bp_cache.messages_graph
communicator(bp_cache::BeliefPropagationCacheMPI) = bp_cache.comm
message_scratch(bp_cache::BeliefPropagationCacheMPI) = bp_cache.scratch

function BeliefPropagationCacheMPI(
        local_cache, messages_graph, ghost_ranks, edges_to_send, edges_to_recv, comm
    )
    return BeliefPropagationCacheMPI(
        local_cache, messages_graph, ghost_ranks, edges_to_send, edges_to_recv, comm,
        Base.RefValue{Any}(Bool[])
    )
end

# The wrapped cache's network and messages are shared by reference, so mutating through
# either view is visible to both.
for f in [
        :(messages),
        :(network),
        :(graph),
        :(contraction_sequences),
        :(default_update_alg),
        :(default_message_update_alg),
        :(default_bp_update_kwargs),
    ]
    @eval begin
        $f(bp_cache::BeliefPropagationCacheMPI) = $f(local_cache(bp_cache))
    end
end

# The local forest cover omits the outgoing cut edges, which no other rank can compute.
function edge_sequence(bp_cache::BeliefPropagationCacheMPI)
    return [edge_sequence(local_cache(bp_cache)); collect(keys(bp_cache.edges_to_send))]
end

function set_default_kwargs(alg::Algorithm"bp", bp_cache::BeliefPropagationCacheMPI)
    # Not left to the local cache, whose edge set is incomplete and whose `maxiter` varies by rank.
    maxiter = haskey(alg.kwargs, :maxiter) ? alg.kwargs.maxiter : default_bp_maxiter(bp_cache)
    kwargs = merge((; edge_sequence = edge_sequence(bp_cache)), alg.kwargs, (; maxiter))
    return set_default_kwargs(Algorithm("bp"; kwargs...), local_cache(bp_cache))
end

# Sweep counts must agree: `update_iteration!` ends in `communicate_messages!`, so a rank stopping
# early leaves its peers blocked in `MPI.recv`. The local default varies with `is_tree`.
function default_bp_maxiter(bp_cache::BeliefPropagationCacheMPI)
    local_maxiter = default_bp_maxiter(local_cache(bp_cache))
    return MPI.Allreduce(local_maxiter, MPI.MAX, communicator(bp_cache))
end

function edge_scalar(bp_cache::BeliefPropagationCacheMPI, edge::AbstractEdge)
    return edge_scalar(local_cache(bp_cache), edge)
end

# Without this, copy() falls through to NamedGraphs.copy(::AbstractNamedGraph).
function Base.copy(bp_cache::BeliefPropagationCacheMPI)
    return BeliefPropagationCacheMPI(
        copy(local_cache(bp_cache)),
        copy(messages_graph(bp_cache)),
        copy(bp_cache.ghost_ranks),
        copy(bp_cache.edges_to_send),
        copy(bp_cache.edges_to_recv),
        communicator(bp_cache), # shared, not duplicated: a copy stays on the same comm
        message_scratch(bp_cache)
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

# Unlike insert_virtualinds!, this works w.r.t `super_graph`, so an edge leaving `tn` for a vertex
# on another rank gets a dangling leg. `bond_inds` must agree across ranks -- see `shared_bond_inds`.
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

# Cut edges in an order every rank agrees on, so that within a rank pair the k-th send meets the
# k-th receive without tagging. Sorted, so agreement does not rest on how the graph was built.
function cut_edges(super_graph::AbstractGraph, vertex_ranks::Dictionary)
    es = filter(e -> vertex_ranks[src(e)] != vertex_ranks[dst(e)], collect(edges(super_graph)))
    return sort(es; by = repr)
end

# `cache` is adopted, not copied: it gains a message entry per cut edge, in both directions.
function BeliefPropagationCacheMPI(
        cache::BeliefPropagationCache,
        super_graph::AbstractGraph,
        vertex_ranks::Dictionary; # vertex -> the mpi rank owning it
        comm::MPI.Comm = MPI.COMM_WORLD
    )
    V = vertextype(super_graph)
    me = Int32(MPI.Comm_rank(comm))
    local_graph = graph(cache)
    local_vertices = collect(vertices(local_graph))
    tn = network(cache)
    ms = messages(cache)

    wrong = filter(v -> vertex_ranks[v] != me, local_vertices)
    isempty(wrong) || error(
        "BeliefPropagationCacheMPI: rank $me holds $(wrong), which `vertex_ranks` assigns to " *
            "$(map(v -> vertex_ranks[v], wrong)). Each vertex must be held by its owner and no " *
            "one else.",
    )

    edges_to_send = Dictionary{NamedEdge{V}, Int32}()
    edges_to_recv = Dictionary{NamedEdge{V}, Int32}()
    ghost_ranks = Dictionary{V, Int32}()

    # Oriented away from this rank, one per cut edge it touches, in the agreed order.
    my_cut_edges = NamedEdge{V}[]
    for e in cut_edges(super_graph, vertex_ranks)
        vertex_ranks[src(e)] == me && push!(my_cut_edges, NamedEdge{V}(src(e) => dst(e)))
        vertex_ranks[dst(e)] == me && push!(my_cut_edges, NamedEdge{V}(dst(e) => src(e)))
    end

    # Both sides send and both receive, so the sends have to be non-blocking.
    requests = MPI.Request[]
    for e in my_cut_edges
        peer = vertex_ranks[dst(e)]
        push!(requests, MPI.isend(factor_inds(tn, src(e)), comm; dest = peer))
    end
    for e in my_cut_edges
        peer = vertex_ranks[dst(e)]
        remote_inds = MPI.recv(comm; source = peer)
        linds = intersect(factor_inds(tn, src(e)), remote_inds)
        isempty(linds) && error(
            "BeliefPropagationCacheMPI: the endpoints of the cut edge $e share no index, so there " *
                "is no bond to pass a message along. $(src(e)) needs a dangling leg for it -- see " *
                "`insert_partition_virtualinds!`.",
        )
        # The two-argument `default_message` would need both endpoints of the edge.
        set!(ms, e, default_message(tn, e, linds))
        set!(ms, reverse(e), default_message(tn, reverse(e), linds))
        insert!(edges_to_send, e, peer)
        insert!(edges_to_recv, reverse(e), peer)
        haskey(ghost_ranks, dst(e)) || insert!(ghost_ranks, dst(e), peer)
    end
    MPI.Waitall(requests)

    # Ghosts live only here, so `graph(network)` stays ghost-free. `add_edge!` is undirected, which
    # also gives `incoming_messages` the direction it needs when updating an outgoing cut edge.
    _messages_graph = copy(local_graph)
    for edge in keys(edges_to_recv)
        ghost = src(edge)
        has_vertex(_messages_graph, ghost) || add_vertex!(_messages_graph, ghost)
        add_edge!(_messages_graph, edge)
    end

    return BeliefPropagationCacheMPI(
        cache,
        _messages_graph,
        ghost_ranks,
        edges_to_send,
        edges_to_recv,
        comm
    )
end

# A cut edge is keyed by the same directed edge on both sides, so the dictionaries line up. Untagged:
# matching relies on per-peer ordering, which holds because both lists came from `cut_edges`.
#TODO: use MPI graph communication primatives.
function communicate_messages!(bp_cache::BeliefPropagationCacheMPI)
    comm = communicator(bp_cache)
    ms = messages(bp_cache)
    requests = MPI.Request[]

    for (edge, rank) in pairs(bp_cache.edges_to_send)
        push!(requests, MPI.isend(ms[edge], comm; dest = rank))
    end

    for (edge, rank) in pairs(bp_cache.edges_to_recv)
        ms[edge] = MPI.recv(comm; source = rank)
    end

    MPI.Waitall(requests)
    return bp_cache
end

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
    communicate_messages!(bpc)
    if !isnothing(update_diff!)
        # `update` divides this by this rank's edge count, so scaling the global sum by
        # nlocal/nglobal gives every rank the serial all-edge average, hence a common exit sweep.
        comm = communicator(bpc)
        total = MPI.Allreduce(update_diff![], MPI.SUM, comm)
        nglobal = MPI.Allreduce(length(edges), MPI.SUM, comm)
        update_diff![] = total * length(edges) / nglobal
    end
    return bpc
end

# The scratch buffer is read only within a sweep, so releasing it here frees the memory for
# `apply_gate!`'s SVD. Every copy of the cache shares it, so the release reaches the caller's too.
function update(alg::Algorithm"bp", bp_cache::BeliefPropagationCacheMPI)
    bp_cache = @invoke update(alg::Algorithm"bp", bp_cache::AbstractBeliefPropagationCache)
    return release_message_scratch!(bp_cache)
end

# A gate on a cut edge carries the site index of both its vertices, and a rank knows only its own.
# One broadcast per rank, carrying `Index` objects only.
function allgather_siteinds(ψ::TensorNetworkState, comm::MPI.Comm)
    me = MPI.Comm_rank(comm)
    all_sinds = Dictionary{vertextype(ψ), Vector{<:Index}}()
    for rank in 0:(MPI.Comm_size(comm) - 1)
        part = MPI.bcast(me == rank ? siteinds(ψ) : nothing, comm; root = rank)
        for (v, is) in pairs(part)
            set!(all_sinds, v, is)
        end
    end
    return all_sinds
end

# Converted here rather than in `apply_gates`, whose conversion uses the rank-local graph and so
# cannot reach the far endpoint of a gate on a cut edge.
function apply_gates_mpi(
        circuit::Vector,
        ψ::TensorNetworkState,
        super_graph::AbstractGraph,
        vertex_ranks::Dictionary;
        comm::MPI.Comm = MPI.COMM_WORLD,
        kwargs...
    )
    gates = toitensor(circuit, super_graph, allgather_siteinds(ψ, comm))
    return apply_gates_mpi(
        ITensor[gate[1] for gate in gates], ψ, super_graph, vertex_ranks;
        comm, gate_vertices = [gate[2] for gate in gates], kwargs...
    )
end

# `gate_vertices` must be resolved against `super_graph`: the default in `apply_gates` reads the local
# network, silently reducing a gate on a cut edge to one vertex.
function apply_gates_mpi(
        circuit::Vector{<:ITensor},
        ψ::TensorNetworkState,
        super_graph::AbstractGraph,
        vertex_ranks::Dictionary;
        comm::MPI.Comm = MPI.COMM_WORLD,
        bp_update_kwargs = default_bp_update_kwargs(ψ; istree = is_tree(super_graph)),
        kwargs...
    )
    ψ_bpc = BeliefPropagationCache(ψ)
    # Seed deltas, not a serial update: that update's incoming_messages sees only local edges, so the
    # cut bonds' dangling indices stay uncontracted and the boundary messages end up with ndims > 2.
    es = reduce(vcat, [[e, reverse(e)] for e in edges(ψ)])
    setmessages!(ψ_bpc, es, [default_message(ψ_bpc, e) for e in es])
    ψ_bpc = BeliefPropagationCacheMPI(ψ_bpc, super_graph, vertex_ranks; comm)
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    ψ_bpc, truncation_errors = apply_gates(circuit, ψ_bpc; bp_update_kwargs, kwargs...)
    return network(ψ_bpc), truncation_errors
end

# Requires every gate to act on vertices this rank holds, since toitensor() gets the rank-local graph
# and site indices. A circuit spanning partitions must go through `apply_gates_mpi` instead.
function apply_gates(
        circuit::Vector,
        ψ_bpc::BeliefPropagationCacheMPI;
        kwargs...
    )
    g = graph(ψ_bpc)
    circuit = toitensor(circuit, g, siteinds(network(ψ_bpc)))
    gate_vertices = [gate[2] for gate in circuit]
    itensors = [gate[1] for gate in circuit]
    return apply_gates(itensors, ψ_bpc; gate_vertices, kwargs...)
end

function apply_gates(
        circuit::Vector{<:ITensor},
        ψ_bpc::BeliefPropagationCacheMPI;
        gate_vertices::Vector = vertices.(circuit, (network(ψ_bpc),)),
        apply_kwargs = (;),
        bp_update_kwargs = default_bp_update_kwargs(ψ_bpc),
        update_cache = true,
        verbose = false
    )
    # This `copy` is SHALLOW: it shares the ITensor objects and hence their buffers, so it does not
    # protect the caller's network from the consuming apply paths overwriting each vertex tensor.
    ψ_bpc = copy(ψ_bpc)

    affected_vertices = Set{eltype(vertices(network(ψ_bpc)))}()
    truncation_errors = zeros((length(circuit)))

    # EVERY DECISION BELOW MUST COME OUT THE SAME ON EVERY RANK: `update` and the boundary factor
    # exchange both block, so a rank skipping one its peers reach deadlocks rather than erroring.

    # If the circuit is applied in the Heisenberg picture, the circuit needs to already be reversed
    for (ii, gate) in enumerate(circuit)
        v⃗ = gate_vertices[ii]

        # A two-site gate hitting an already-affected vertex is acting on stale messages.
        cache_update_required =
            length(v⃗) >= 2 &&
            any(vert in affected_vertices for vert in v⃗)

        if update_cache && cache_update_required
            if verbose
                println("Updating BP cache")
            end

            t = @timed ψ_bpc = update(ψ_bpc; bp_update_kwargs...)

            empty!(affected_vertices)

            if verbose
                println("Done in $(t.time) secs")
            end
        end

        role = gate_role(v⃗, vertices(network(ψ_bpc)), ψ_bpc.ghost_ranks)

        if role.kind !== :skip
            gate = adapt_gate(gate, ψ_bpc)

            @timed ψ_bpc, truncation_errors[ii] = apply_gate!(
                gate,
                ψ_bpc;
                v⃗,
                role,
                apply_kwargs
            )
        end

        for v in v⃗
            push!(affected_vertices, v)
        end
    end

    if update_cache
        ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    end

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

# `:local`, `:boundary` or `:skip`, decided from the circuit and the partitioning alone so that the
# two ranks of a cut edge agree without exchanging anything. The gate's vertex order picks `compute`.
function gate_role(gate_vertices, local_vertices, ghost_ranks)
    nlocal = count(in(local_vertices), gate_vertices)

    nlocal == length(gate_vertices) && return (; kind = :local, peer = nothing, compute = true)
    iszero(nlocal) && return (; kind = :skip, peer = nothing, compute = false)

    length(gate_vertices) == 2 || throw(
        ArgumentError(
            "gate on $(length(gate_vertices)) vertices $gate_vertices straddles a partition " *
                "boundary; only one- and two-site gates are supported."
        )
    )

    remote = only(filter(!in(local_vertices), gate_vertices))
    haskey(ghost_ranks, remote) || error(
        "gate_role: $remote is neither local nor a neighbour of this partition, so this rank " *
            "cannot reach it. A two-site gate must act on an edge of the super graph.",
    )

    return (;
        kind = :boundary,
        peer = ghost_ranks[remote],
        compute = first(gate_vertices) in local_vertices,
    )
end

# Vertex terms partition cleanly; a cut edge's message pair is held by both ranks, so it is counted on
# the lower-ranked one. Terms are complexified so a negative one contributes its phase through log.
function freenergy(bp_cache::BeliefPropagationCacheMPI)
    comm = communicator(bp_cache)
    me = MPI.Comm_rank(comm)
    cut = [e for (e, rank) in pairs(bp_cache.edges_to_send) if me < rank]

    numer = complex.(vertex_scalars(bp_cache, collect(vertices(bp_cache))))
    denom = complex.(edge_scalars(bp_cache, [collect(edges(bp_cache)); cut]))
    # -Inf rather than an early return: every rank must reach the Allreduce.
    local_f = any(iszero, denom) ? complex(-Inf) : sum(log.(numer)) - sum(log.(denom))
    return MPI.Allreduce(local_f, MPI.SUM, comm)
end

# ψ and ϕ are the rank-local partitions; the same global scalar comes back on every rank.
function inner_mpi(
        ψ::TensorNetworkState,
        ϕ::TensorNetworkState,
        super_graph::AbstractGraph,
        vertex_ranks::Dictionary;
        comm::MPI.Comm = MPI.COMM_WORLD,
        bp_update_kwargs = default_bp_update_kwargs(ψ)
    )
    bpc = BeliefPropagationCache(BilinearForm(ψ, ϕ))
    es = reduce(vcat, [[e, reverse(e)] for e in edges(bpc)])
    setmessages!(bpc, es, [default_message(bpc, e) for e in es])
    bpc = BeliefPropagationCacheMPI(bpc, super_graph, vertex_ranks; comm)
    bpc = update(bpc; bp_update_kwargs...)
    return inner(Algorithm("bp"), bpc)
end

function apply_gate!(
        gate::ITensor,
        ψ_bpc::BeliefPropagationCacheMPI;
        v⃗ = vertices(gate, network(ψ_bpc)),
        role = gate_role(v⃗, vertices(network(ψ_bpc)), ψ_bpc.ghost_ranks),
        apply_kwargs
    )
    nv = length(v⃗)

    1 <= nv <= 2 || error(
        "apply_gate!: only one- and two-site gates are supported; " *
            "received a gate acting on $nv vertices: $v⃗.",
    )

    role.kind === :boundary &&
        return apply_boundary_gate!(gate, ψ_bpc; v⃗, role, apply_kwargs)

    if nv == 2
        has_edge(graph(ψ_bpc), NamedEdge(first(v⃗) => last(v⃗))) || error(
            "apply_gate!: cannot apply a two-site gate on the non-adjacent vertices " *
                "$(first(v⃗)) and $(last(v⃗)). Simple update requires the two sites to share an " *
                "edge of the tensor-network graph.",
        )
    end

    envs = nv == 1 ? nothing : incoming_messages(ψ_bpc, v⃗)

    ψ⃗ = ITensor[network(ψ_bpc)[v] for v in v⃗]

    foreach(v⃗) do v
        # Allow deallocation; `simple_update_dense` consumes these tensors' storage.
        setindex_preserve!(ψ_bpc, ITensor(), v)
    end

    updated_tensors, s_values, err = simple_update_dense(gate, ψ⃗; envs, apply_kwargs...)
    if nv == 2
        v1, v2 = v⃗
        setbondmessages!(ψ_bpc, NamedEdge(v1 => v2), s_values, first(updated_tensors))
    end

    for (i, v) in enumerate(v⃗)
        setindex_preserve!(ψ_bpc, updated_tensors[i], v)
    end

    return ψ_bpc, err
end

# The new bond's gauge as its pair of messages; `t` only identifies which index of `s_values` it is.
# Both ranks of a cut edge pass the same `s_values`, so their messages agree without an exchange.
function setbondmessages!(
        ψ_bpc::AbstractBeliefPropagationCache, e::AbstractEdge, s_values::ITensor, t::ITensor
    )
    ind2 = commonind(s_values, t)
    δuv = dag(copy(s_values))
    δuv = replaceind(δuv, ind2, ind2')
    map_diag!(sign, δuv, δuv)
    s_values = denseblocks(s_values) * denseblocks(δuv)
    setmessage!(ψ_bpc, e, dag(s_values))
    setmessage!(ψ_bpc, reverse(e), s_values)
    return ψ_bpc
end

# A two-site gate whose other vertex lives on `role.peer`; both ranks run it, exchanging QR factors.
function apply_boundary_gate!(
        gate::ITensor,
        ψ_bpc::BeliefPropagationCacheMPI;
        v⃗,
        role,
        apply_kwargs
    )
    local_vertices = vertices(network(ψ_bpc))
    v = only(filter(in(local_vertices), v⃗))
    remote = only(filter(!in(local_vertices), v⃗))
    e_in = NamedEdge(remote => v)

    has_edge(messages_graph(ψ_bpc), e_in) || error(
        "apply_boundary_gate!: $v and $remote are not joined by a cut edge of this partition, so " *
            "there is no bond for a two-site gate to act on.",
    )

    # The partner direction is the bond being updated, not an environment.
    envs = incoming_messages(ψ_bpc, [v]; ignore_edges = [e_in])
    ψᵥ = network(ψ_bpc)[v]
    # The messages are the only record of the cut bond's current index, which truncation replaces.
    lb = only(commoninds(message(ψ_bpc, e_in), ψᵥ))

    # Allow deallocation; `simple_update_dense_boundary` consumes this tensor's storage.
    setindex_preserve!(ψ_bpc, ITensor(), v)

    u, s_values, err = simple_update_dense_boundary(
        gate, ψᵥ;
        envs, lb, compute = role.compute, other_rank = role.peer,
        comm = communicator(ψ_bpc), apply_kwargs...,
    )

    setindex_preserve!(ψ_bpc, u, v)
    setbondmessages!(ψ_bpc, NamedEdge(v => remote), s_values, u)

    return ψ_bpc, err
end
