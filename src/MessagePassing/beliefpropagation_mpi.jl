using Dictionaries: Dictionary, delete!, set!
using Graphs: AbstractGraph, connected_components, is_tree
using ITensors.NDTensors: scalartype
using ITensors: Algorithm, ITensor, delta, dim
using LinearAlgebra: normalize
using MPI
using NamedGraphs.GraphsExtensions: a_star, boundary_edges, default_root_vertex,
    forest_cover, forest_cover_edge_sequence, leaf_vertices, post_order_dfs_edges

struct BeliefPropagationCacheMPI{
        V,
        BPC <: BeliefPropagationCache{V},
        G <: AbstractGraph,
    } <: AbstractBeliefPropagationCache{V}
    local_cache::BPC
    messages_graph::G # local graph plus a ghost vertex per remote neighbour
    shared_vertices::Dictionary{V, Int32} # duplicated vertex -> the peer rank
    edges_to_send::Dictionary{NamedEdge{V}, Int32} # edge -> mpi rank
    edges_to_recv::Dictionary{NamedEdge{V}, Int32} # edge -> mpi rank
    comm::MPI.Comm
    # Flat work buffer for the "blocked" message update (see blockedmessage.jl). Starts
    # empty and is grown to fit on first use; a `Ref` so it can be replaced from an
    # immutable struct. It holds no semantic state, so `copy` shares it rather than
    # reallocating -- `update` copies the cache on every call.
    scratch::Base.RefValue{Any}
end

local_cache(bp_cache::BeliefPropagationCacheMPI) = bp_cache.local_cache
messages_graph(bp_cache::BeliefPropagationCacheMPI) = bp_cache.messages_graph
communicator(bp_cache::BeliefPropagationCacheMPI) = bp_cache.comm
message_scratch(bp_cache::BeliefPropagationCacheMPI) = bp_cache.scratch

# Five-field form for callers predating the scratch buffer: starts empty, grown on demand.
function BeliefPropagationCacheMPI(
        local_cache, messages_graph, shared_vertices, edges_to_send, edges_to_recv, comm
    )
    return BeliefPropagationCacheMPI(
        local_cache, messages_graph, shared_vertices, edges_to_send, edges_to_recv, comm,
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
        :(edge_sequence),
        :(default_update_alg),
        :(default_message_update_alg),
        :(default_bp_update_kwargs),
    ]
    @eval begin
        $f(bp_cache::BeliefPropagationCacheMPI) = $f(local_cache(bp_cache))
    end
end

function set_default_kwargs(alg::Algorithm"bp", bp_cache::BeliefPropagationCacheMPI)
    return set_default_kwargs(alg, local_cache(bp_cache))
end

function edge_scalar(bp_cache::BeliefPropagationCacheMPI, edge::AbstractEdge)
    return edge_scalar(local_cache(bp_cache), edge)
end

# Without this, copy() falls through to NamedGraphs.copy(::AbstractNamedGraph).
function Base.copy(bp_cache::BeliefPropagationCacheMPI)
    return BeliefPropagationCacheMPI(
        copy(local_cache(bp_cache)),
        copy(messages_graph(bp_cache)),
        copy(bp_cache.shared_vertices),
        copy(bp_cache.edges_to_send),
        copy(bp_cache.edges_to_recv),
        communicator(bp_cache), # shared, not duplicated: a copy stays on the same comm
        message_scratch(bp_cache) # also shared: pure scratch, and `update` copies per call
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

# Creates an "MPI-aware" cache. `cache` is adopted, not copied: its message dictionary gains
# an entry per incoming boundary edge.
function BeliefPropagationCacheMPI(
        cache::BeliefPropagationCache,
        super_graph::AbstractGraph,
        shared_vertices::Dictionary; # vertex -> mpi rank
        comm::MPI.Comm = MPI.COMM_WORLD
    )
    V = vertextype(super_graph)
    local_graph = graph(cache)
    ms = messages(cache)

    edges_to_send = Dictionary{NamedEdge{V}, Int32}()
    edges_to_recv = Dictionary{NamedEdge{V}, Int32}()

    # Only this rank's shared vertices, so callers can trust keys() when deciding which
    # factors to exchange.
    shared_vertices_other = Dictionary{V, Int32}()

    requests = MPI.Request[]

    for (shared_vertex, involved_ranks) in pairs(shared_vertices)
        rank1, rank2 = involved_ranks

        if MPI.Comm_rank(comm) == rank1
            other_rank = rank2
        elseif MPI.Comm_rank(comm) == rank2
            other_rank = rank1
        else
            continue
        end

        insert!(shared_vertices_other, shared_vertex, other_rank)

        # There should be exactly 2 neighbors
        for neighbor in neighbors(super_graph, shared_vertex)
            edge = NamedEdge{V}(neighbor => shared_vertex)

            if neighbor in vertices(local_graph)
                # If this neighbor is in my rank then I need to send the message
                push!(requests, MPI.isend(ms[edge], comm; dest = other_rank))
                insert!(edges_to_send, edge, other_rank)
            else
                message = MPI.recv(comm; source = other_rank)
                insert!(ms, edge, message)
                insert!(edges_to_recv, edge, other_rank)
            end
        end
    end

    MPI.Waitall(requests)

    # Ghost vertices hold the incoming boundary messages. They live only here, so
    # graph(network) stays ghost-free and vertex iteration is unaffected.
    _messages_graph = copy(local_graph)
    for edge in keys(edges_to_recv)
        ghost = src(edge)
        has_vertex(_messages_graph, ghost) || add_vertex!(_messages_graph, ghost)
        add_edge!(_messages_graph, edge)
    end

    return BeliefPropagationCacheMPI(
        cache,
        _messages_graph,
        shared_vertices_other,
        edges_to_send,
        edges_to_recv,
        comm
    )
end

# Untagged, so matching relies on MPI's per-peer ordering: sender and receiver both build their
# lists from the same walk over shared_vertices, so the k-th send meets the k-th receive.
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
        # update() divides this by `length(edges)`, this rank's edge count. Rescaling the
        # global sum by nlocal/nglobal makes that quotient both identical on every rank -- so
        # a tolerance exit breaks the loop at the same sweep everywhere -- and equal to the
        # serial average over all edges. Edge counts sum to the global count because every
        # graph edge lies inside exactly one region.
        comm = communicator(bpc)
        total = MPI.Allreduce(update_diff![], MPI.SUM, comm)
        nglobal = MPI.Allreduce(length(edges), MPI.SUM, comm)
        update_diff![] = total * length(edges) / nglobal
    end
    return bpc
end

# The blocked message update's scratch buffer is only read inside a sweep, so it is released
# on the way out and reallocated by the next sweep's first `updated_message`. The buffer is
# shared with every copy of this cache, including the one `update` was handed, so the release
# frees the memory for whatever runs next (`apply_gate!`'s SVD) rather than only for the
# returned cache.
function update(alg::Algorithm"bp", bp_cache::BeliefPropagationCacheMPI)
    bp_cache = @invoke update(alg::Algorithm"bp", bp_cache::AbstractBeliefPropagationCache)
    return release_message_scratch!(bp_cache)
end

# Pass `gate_vertices` resolved against `super_graph`. The default reads them off the local
# network, silently reducing a remote gate to zero vertices and a boundary-straddling two-site
# gate to one.
function apply_gates_mpi(
        circuit::Vector,
        ψ::TensorNetworkState,
        super_graph::AbstractGraph,
        shared_vertices::Dictionary;
        comm::MPI.Comm = MPI.COMM_WORLD,
        bp_update_kwargs = default_bp_update_kwargs(ψ; istree = is_tree(super_graph)),
        kwargs...
    )
    ψ_bpc = BeliefPropagationCache(ψ)
    # Seed deltas, not a serial update: that update's incoming_messages sees only local edges,
    # so it never contracts the cut bonds' dangling indices and they survive as free indices in
    # the boundary messages, which then have ndims > 2.
    es = reduce(vcat, [[e, reverse(e)] for e in edges(ψ)])
    setmessages!(ψ_bpc, es, [default_message(ψ_bpc, e) for e in es])
    ψ_bpc = BeliefPropagationCacheMPI(ψ_bpc, super_graph, shared_vertices; comm)
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    ψ_bpc, truncation_errors = apply_gates(circuit, ψ_bpc; bp_update_kwargs, kwargs...)
    return network(ψ_bpc), truncation_errors
end

# Requires every gate to act on vertices this rank holds: toitensor() gets the rank-local graph
# and site indices. For a circuit spanning partitions, convert against the super graph and call
# the Vector{<:ITensor} method with an explicit `gate_vertices`.
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
    ψ_bpc = copy(ψ_bpc)

    # we keep track of the vertices that have been acted on by 2-qubit gates
    # only they increase the counter
    # this is the set that keeps track.
    affected_vertices = Set{eltype(vertices(network(ψ_bpc)))}()
    truncation_errors = zeros((length(circuit)))

    vertices_to_send = Vector{eltype(vertices(network(ψ_bpc)))}()
    vertices_to_recv = Vector{eltype(vertices(network(ψ_bpc)))}()

    # If the circuit is applied in the Heisenberg picture, the circuit needs to already be reversed
    for (ii, gate) in enumerate(circuit)
        v⃗ = gate_vertices[ii]

        # check if the gate is a 2-qubit gate and whether it affects the counter
        # we currently only increment the counter if the gate affects vertices that have already been affected
        cache_update_required =
            length(v⃗) >= 2 &&
            any(vert in affected_vertices for vert in v⃗)

        # update the BP cache
        if update_cache && cache_update_required
            if verbose
                println("Updating BP cache")
            end

            communicate_factors!(ψ_bpc, vertices_to_send, vertices_to_recv)

            t = @timed ψ_bpc = update(ψ_bpc; bp_update_kwargs...)

            empty!(affected_vertices)
            empty!(vertices_to_send)
            empty!(vertices_to_recv)

            if verbose
                println("Done in $(t.time) secs")
            end
        end

        my_vertices = vertices(network(ψ_bpc))
        shared_vertices_dict = ψ_bpc.shared_vertices

        # Only apply the gate if *all* gate vertices are local/shared.
        iapply, shared_vertex = should_apply_gate(v⃗, my_vertices, shared_vertices_dict)

        if iapply
            gate = adapt_gate(gate, ψ_bpc)

            @timed ψ_bpc, truncation_errors[ii] = apply_gate!(
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

    # Two shared vertices would have to be adjacent, which the constructor does not support.
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

# Assumes each vertex is on one side of the exchange only. should_apply_gate keeps that true by
# never electing a single rank for a one-site gate.
function communicate_factors!(
        bp_cache::BeliefPropagationCacheMPI,
        vertices_to_send,
        vertices_to_recv
    )
    comm = communicator(bp_cache)
    tn = network(bp_cache)
    requests = MPI.Request[]
    # unique() keeps first-occurrence order, so sender and receiver stay in step.
    for vertex in unique(vertices_to_send)
        other_rank = bp_cache.shared_vertices[vertex]
        push!(requests, MPI.isend(tn[vertex], comm; dest = other_rank))
    end
    for vertex in unique(vertices_to_recv)
        other_rank = bp_cache.shared_vertices[vertex]
        # Not tn[v] = ...: that goes through add_tensor!, which rewires the graph from index
        # overlap.
        setindex_preserve!(bp_cache, MPI.recv(comm; source = other_rank), vertex)
    end
    MPI.Waitall(requests)

    # A received factor carries the sender's index on every bond it truncated, so refresh the
    # boundary messages to match. Otherwise the message and the factor share no index on that
    # bond and BP contracts them into an outer product, adding two free indices per sweep.
    return communicate_messages!(bp_cache)
end

# Edge terms partition cleanly, since every edge lies inside one region. Vertex terms do not: a
# shared vertex is held by two ranks, so it is counted on the lower-ranked one. Terms are
# complexified so a negative one contributes its phase through log.
function freenergy(bp_cache::BeliefPropagationCacheMPI)
    comm = communicator(bp_cache)
    me = MPI.Comm_rank(comm)
    shared = bp_cache.shared_vertices
    owned = filter(v -> !haskey(shared, v) || me < shared[v], collect(vertices(bp_cache)))

    numer = complex.(vertex_scalars(bp_cache, owned))
    denom = complex.(edge_scalars(bp_cache, collect(edges(bp_cache))))
    # -Inf rather than an early return: every rank must reach the Allreduce.
    local_f = any(iszero, denom) ? complex(-Inf) : sum(log.(numer)) - sum(log.(denom))
    return MPI.Allreduce(local_f, MPI.SUM, comm)
end

# ⟨ψ|ϕ⟩ over a partitioned network, with ψ and ϕ the rank-local partitions. Returns the same
# global scalar on every rank.
function inner_mpi(
        ψ::TensorNetworkState,
        ϕ::TensorNetworkState,
        super_graph::AbstractGraph,
        shared_vertices::Dictionary;
        comm::MPI.Comm = MPI.COMM_WORLD,
        bp_update_kwargs = default_bp_update_kwargs(ψ)
    )
    bpc = BeliefPropagationCache(BilinearForm(ψ, ϕ))
    es = reduce(vcat, [[e, reverse(e)] for e in edges(bpc)])
    setmessages!(bpc, es, [default_message(bpc, e) for e in es])
    bpc = BeliefPropagationCacheMPI(bpc, super_graph, shared_vertices; comm)
    bpc = update(bpc; bp_update_kwargs...)
    return inner(Algorithm("bp"), bpc)
end
