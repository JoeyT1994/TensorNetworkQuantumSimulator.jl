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
        N <: AbstractTensorNetwork{V},
        M <: Union{ITensor, Vector{ITensor}},
        G <: AbstractGraph,
    } <: AbstractBeliefPropagationCache{V}
    network::N
    messages::Dictionary{NamedEdge, M}
    contraction_sequences::Dictionary{Pair, Vector}
    edge_sequence::Vector
    messages_graph::G # local graph plus a ghost vertex per remote neighbour
    shared_vertices::Dictionary{V, Int32}
    edges_to_send::Dictionary{NamedEdge, Int32} # edge -> mpi rank
    edges_to_recv::Dictionary{NamedEdge, Int32} # edge -> mpi rank
    function BeliefPropagationCacheMPI(
        network::N,
        messages::Dictionary{NamedEdge, M},
        contraction_sequences::Dictionary{Pair, Vector},
        edge_sequence::Vector,
        messages_graph::G,
        shared_vertices::Dictionary{V, Int32},
        edges_to_send::Dictionary{NamedEdge, Int32},
        edges_to_recv::Dictionary{NamedEdge, Int32}
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}, G <: AbstractGraph}
        new{V, N, M, G}(
            network,
            messages,
            contraction_sequences,
            edge_sequence,
            messages_graph,
            shared_vertices,
            edges_to_send,
            edges_to_recv
        )
    end
end

messages(bp_cache::BeliefPropagationCacheMPI) = bp_cache.messages
network(bp_cache::BeliefPropagationCacheMPI) = bp_cache.network
graph(bp_cache::BeliefPropagationCacheMPI) = graph(network(bp_cache))
messages_graph(bp_cache::BeliefPropagationCacheMPI) = bp_cache.messages_graph

# Without this, copy() falls through to NamedGraphs.copy(::AbstractNamedGraph).
function Base.copy(bp_cache::BeliefPropagationCacheMPI)
    return BeliefPropagationCacheMPI(
        copy(network(bp_cache)),
        copy(messages(bp_cache)),
        copy(contraction_sequences(bp_cache)),
        copy(edge_sequence(bp_cache)),
        copy(messages_graph(bp_cache)),
        copy(bp_cache.shared_vertices),
        copy(bp_cache.edges_to_send),
        copy(bp_cache.edges_to_recv)
    )
end

# Creates an "MPI-aware" cache.
function BeliefPropagationCacheMPI(
        cache::BeliefPropagationCache,
        super_graph::AbstractGraph,
        shared_vertices::Dictionary; # vertex -> mpi rank
        comm = MPI.COMM_WORLD
    )
    V = vertextype(super_graph)
    network = cache.network
    messages = cache.messages

    local_graph = graph(network)

    edges_to_send = Dictionary{NamedEdge, Int32}()
    edges_to_recv = Dictionary{NamedEdge, Int32}()

    shared_vertices_other = similar(shared_vertices, Int32)

    for (shared_vertex, involved_ranks) in pairs(shared_vertices)
        rank1, rank2 = involved_ranks

        if MPI.Comm_rank(comm) == rank1
            other_rank = rank2
        elseif MPI.Comm_rank(comm) == rank2
            other_rank = rank1
        else
            continue
        end

        shared_vertices_other[shared_vertex] = other_rank

        # There should be exactly 2 neighbors
        for neighbor in neighbors(super_graph, shared_vertex)
            edge = NamedEdge{V}(neighbor => shared_vertex)

            if neighbor in vertices(local_graph)
                # If this neighbor is in my rank then I need to send the message
                MPI.send(messages[edge], comm; dest = other_rank)
                insert!(edges_to_send, edge, other_rank)
            else
                message = MPI.recv(comm; source = other_rank)
                insert!(messages, edge, message)
                insert!(edges_to_recv, edge, other_rank)
            end
        end
    end

    # Ghost vertices hold the incoming boundary messages. They live only here, so
    # graph(network) stays ghost-free and vertex iteration is unaffected.
    _messages_graph = copy(local_graph)
    for edge in keys(edges_to_recv)
        ghost = src(edge)
        has_vertex(_messages_graph, ghost) || add_vertex!(_messages_graph, ghost)
        add_edge!(_messages_graph, edge)
    end

    return BeliefPropagationCacheMPI(
        network,
        messages,
        cache.contraction_sequences,
        cache.edge_sequence,
        _messages_graph,
        shared_vertices_other,
        edges_to_send,
        edges_to_recv
    )
end

function communicate_messages!(bp_cache::BeliefPropagationCacheMPI; comm = MPI.COMM_WORLD)
    requests = [
        MPI.isend(bp_cache.messages[edge], comm; dest = rank)
            for (edge, rank) in pairs(bp_cache.edges_to_send)
    ]
    for (edge, rank) in pairs(bp_cache.edges_to_recv)
        bp_cache.messages[edge] = MPI.recv(comm; source = rank)
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

contraction_sequences(bp_cache::BeliefPropagationCacheMPI) = bp_cache.contraction_sequences
edge_sequence(bp_cache::BeliefPropagationCacheMPI) = bp_cache.edge_sequence

#Algorithmic defaults
default_update_alg(::BeliefPropagationCacheMPI) = "bp"
default_message_update_alg(::BeliefPropagationCacheMPI) = "contract"

function set_default_kwargs(alg::Algorithm"bp", bp_cache::BeliefPropagationCacheMPI)
    verbose = get(alg.kwargs, :verbose, default_verbose(alg))
    maxiter = get(alg.kwargs, :maxiter, default_bp_maxiter(bp_cache))
    _edge_sequence = get(alg.kwargs, :edge_sequence, edge_sequence(bp_cache))
    tolerance = get(alg.kwargs, :tolerance, default_tolerance(alg))
    message_update_alg = set_default_kwargs(
        get(
            alg.kwargs,
            :message_update_alg,
            Algorithm(default_message_update_alg(bp_cache))
        ), bp_cache
    )
    return Algorithm(
        "bp";
        verbose,
        maxiter,
        edge_sequence = _edge_sequence,
        tolerance,
        message_update_alg
    )
end

function update_message!(
        message_update_alg::Algorithm, bp_cache::BeliefPropagationCacheMPI, edge::AbstractEdge
    )
    m, (cache_key, sequence, seq_changed) =
        updated_message(message_update_alg, bp_cache, edge)
    seq_changed && set!(contraction_sequences(bp_cache), cache_key, sequence)
    return setmessage!(bp_cache, edge, m)
end

function default_bp_update_kwargs(bp_cache::BeliefPropagationCacheMPI)
    return default_bp_update_kwargs(network(bp_cache))
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
    return bpc
end

function apply_gates_mpi(
        circuit::Vector,
        ψ::TensorNetworkState,
        super_graph::AbstractGraph,
        shared_vertices::Dictionary;
        comm = MPI.COMM_WORLD,
        bp_update_kwargs = default_bp_update_kwargs(ψ),
        kwargs...
    )
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    ψ_bpc = BeliefPropagationCacheMPI(ψ_bpc, super_graph, shared_vertices; comm)
    ψ_bpc, truncation_errors = apply_gates(circuit, ψ_bpc; bp_update_kwargs, kwargs...)
    return network(ψ_bpc), truncation_errors
end

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

        # check if the gate is a 2-qubit gate and whether it affects the counter
        # we currently only increment the counter if the gate affects vertices that have already been affected
        cache_update_required =
            length(gate_vertices[ii]) >= 2 &&
            any(vert in affected_vertices for vert in gate_vertices[ii])

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

        # actually apply the gate

        shared_vertices = filter(in(keys(ψ_bpc.shared_vertices)), gate_vertices[ii])
        local_vertices = vertices(network(ψ_bpc))

        # Only apply the gate if all vertices are local/shared.
        if should_apply_gate(gate_vertices[ii], local_vertices, ψ_bpc.shared_vertices)
            gate = adapt_gate(gate, ψ_bpc)
            t = @timed ψ_bpc, truncation_errors[ii] =
                apply_gate!(gate, ψ_bpc; v⃗ = gate_vertices[ii], apply_kwargs)

            # We did this, so we must send the shared vertices
            append!(vertices_to_send, shared_vertices)
        else
            # Someone else did this, so make sure to later get the shared vertex
            append!(vertices_to_recv, shared_vertices)
        end

        for v in gate_vertices[ii]
            push!(affected_vertices, v)
        end
    end

    if update_cache
        ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    end

    return ψ_bpc, truncation_errors
end

function should_apply_gate(gate_vertices, local_vertices, shared_vertices)
    overlapping_vertices = filter(in(local_vertices), gate_vertices)

    if overlapping_vertices == gate_vertices
        # Gate vertices are all local, so we should apply the gate, unless...
        if length(gate_vertices) == 1 && only(gate_vertices) ∈ keys(shared_vertices)
            # this is a single qubit gate applied to a shared vertex, so make sure
            # we only choose one rank to apply it.
            v = only(gate_vertices)
            other_rank = shared_vertices[v]
            my_rank = MPI.Comm_rank(comm)
            if my_rank > other_rank
                return true
            end
        else
            return true
        end
    end
    return false
end

function communicate_factors!(
        bp_cache::BeliefPropagationCacheMPI,
        vertices_to_send,
        vertices_to_recv
    )
    for vertex in vertices_to_send
        other_rank = bp_cache.shared_vertices[vertex]
        MPI.isend(factor(bp_cache, edge), comm, other_rank)
    end
    for vertex in vertices_to_recv
        other_rank = bp_cache.shared_vertices[vertex]
        setfactor!(bp_cache, vertex, MPI.recv(comm, other_rank))
    end
    return bp_cache
end

