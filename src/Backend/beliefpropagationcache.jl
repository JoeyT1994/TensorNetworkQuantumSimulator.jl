using Dictionaries: Dictionary, set!, delete!
using DataGraphs: AbstractDataGraph
using Graphs: AbstractGraph, is_tree, connected_components
using NamedGraphs.GraphsExtensions: default_root_vertex, forest_cover, post_order_dfs_edges
using ITensors: dim, ITensor, delta, Algorithm
using ITensors.NDTensors: scalartype
using LinearAlgebra: normalize

struct BeliefPropagationCache{V, N <: AbstractDataGraph{V}} <:
    AbstractBeliefPropagationCache{V}
    network::N
    messages::Dictionary
end

#TODO: Take `dot` without precontracting the messages to allow scaling to more complex messages
function message_diff(message_a::ITensor, message_b::ITensor)
    f = abs2(dot((message_a / norm(message_a)), (message_b / norm(message_b))))
    return 1 - f
end

messages(bp_cache::BeliefPropagationCache) = bp_cache.messages
network(bp_cache::BeliefPropagationCache) = bp_cache.network
default_messages() = Dictionary()

BeliefPropagationCache(network) = BeliefPropagationCache(network, default_messages())

function Base.copy(bp_cache::BeliefPropagationCache)
    return BeliefPropagationCache(copy(network(bp_cache)), copy(messages(bp_cache)))
end

function deletemessage!(bp_cache::BeliefPropagationCache, e::AbstractEdge)
    ms = messages(bp_cache)
    delete!(ms, e)
    return bp_cache
end

function setmessage!(bp_cache::BeliefPropagationCache, e::AbstractEdge, message::Union{ITensor, Vector{<:ITensor}})
    ms = messages(bp_cache)
    set!(ms, e, message)
    return bp_cache
end

function message(bp_cache::BeliefPropagationCache, edge::AbstractEdge; kwargs...)
    ms = messages(bp_cache)
    return get(() -> default_message(bp_cache, edge; kwargs...), ms, edge)
end

function messages(bp_cache::BeliefPropagationCache, edges::Vector{<:AbstractEdge})
    isempty(edges) && return ITensor[]
    return reduce(vcat, [message(bp_cache, e) for e in edges])
end

default_bp_maxiter(g::AbstractGraph) = is_tree(g) ? 1 : _default_bp_update_maxiter
#Forward onto the network
for f in [
        :(Graphs.vertices),
        :(Graphs.edges),
        :(Graphs.is_tree),
        :(NamedGraphs.GraphsExtensions.boundary_edges),
        :(bp_factors),
        :(default_bp_maxiter),
        :(ITensorNetworks.linkinds),
        :(ITensorNetworks.underlying_graph),
        :(ITensors.datatype),
        :(ITensors.scalartype),
        :(ITensorNetworks.setindex_preserve_graph!),
        :(ITensorNetworks.maxlinkdim)
    ]
    @eval begin
        function $f(bp_cache::BeliefPropagationCache, args...; kwargs...)
            return $f(network(bp_cache), args...; kwargs...)
        end
    end
end

#TODO: Get subgraph working on an ITensorNetwork to overload this directly
function default_bp_edge_sequence(bp_cache::BeliefPropagationCache)
    return forest_cover_edge_sequence(ITensorNetworks.underlying_graph(bp_cache))
end

function bp_factors(tn::ITensorNetwork, vertex)
    return [tn[vertex]]
end

function edge_scalar(bp_cache::BeliefPropagationCache, edge::AbstractEdge)
    return (message(bp_cache, edge)* message(bp_cache, reverse(edge)))[]
end

function vertex_scalar(bp_cache::BeliefPropagationCache, vertex)
    incoming_ms = incoming_messages(bp_cache, vertex)
    state = bp_factors(bp_cache, vertex)
    contract_list = [state; incoming_ms]
    sequence = contraction_sequence(contract_list; alg="optimal")
    return contract(contract_list; sequence)[]
end

function default_message(bp_cache::BeliefPropagationCache, edge::AbstractEdge)
    return default_message(network(bp_cache), edge::AbstractEdge)
end

function default_message(tn::ITensorNetwork, edge::AbstractEdge)
    return adapt(datatype(tn))(denseblocks(delta(linkinds(tn, edge))))
end

#Algorithmic defaults
default_update_alg(bp_cache::BeliefPropagationCache) = "bp"
default_message_update_alg(bp_cache::BeliefPropagationCache) = "contract"
default_normalize(::Algorithm"contract") = true
default_sequence_alg(::Algorithm"contract") = "optimal"
default_enforce_hermicity(::Algorithm"contract", bp_cache) = isa(network(bp_cache), TensorNetworkState)  ? true : false
function set_default_kwargs(alg::Algorithm"contract", bp_cache::BeliefPropagationCache)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    sequence_alg = get(alg.kwargs, :sequence_alg, default_sequence_alg(alg))
    enforce_hermiticity = get(alg.kwargs, :enforce_hermiticity, default_enforce_hermicity(alg, bp_cache))
    return Algorithm("contract"; normalize, sequence_alg, enforce_hermiticity)
end
default_verbose(::Algorithm"bp") = false
default_tolerance(::Algorithm"bp") = nothing
function set_default_kwargs(alg::Algorithm"bp", bp_cache::BeliefPropagationCache)
    verbose = get(alg.kwargs, :verbose, default_verbose(alg))
    maxiter = get(alg.kwargs, :maxiter, default_bp_maxiter(bp_cache))
    edge_sequence = get(alg.kwargs, :edge_sequence, default_bp_edge_sequence(bp_cache))
    tolerance = get(alg.kwargs, :tolerance, default_tolerance(alg))
    message_update_alg = set_default_kwargs(
        get(alg.kwargs, :message_update_alg, Algorithm(default_message_update_alg(bp_cache))), bp_cache
    )
    return Algorithm("bp"; verbose, maxiter, edge_sequence, tolerance, message_update_alg)
end

#TODO: Update message etc should go here...
function updated_message(
        alg::Algorithm"contract", bp_cache::BeliefPropagationCache, edge::AbstractEdge
    )
    vertex = src(edge)
    incoming_ms = incoming_messages(
        bp_cache, vertex; ignore_edges = typeof(edge)[reverse(edge)]
    )
    state = bp_factors(bp_cache, vertex)
    contract_list = ITensor[incoming_ms; state]
    sequence = contraction_sequence(contract_list; alg=alg.kwargs.sequence_alg)
    updated_message = contract(contract_list; sequence)

    if alg.kwargs.enforce_hermiticity
        updated_message = make_hermitian(updated_message)
    end

    if alg.kwargs.normalize
        message_norm = LinearAlgebra.norm(updated_message)
        if !iszero(message_norm)
            updated_message /= message_norm
        end
    end

    return updated_message
end

function updated_message(
        bp_cache::BeliefPropagationCache,
        edge::AbstractEdge;
        alg = default_message_update_alg(bp_cache),
        kwargs...,
    )
    return updated_message(set_default_kwargs(Algorithm(alg; kwargs...)), bp_cache, edge)
end

function update_message!(
        message_update_alg::Algorithm, bp_cache::BeliefPropagationCache, edge::AbstractEdge
    )
    return setmessage!(bp_cache, edge, updated_message(message_update_alg, bp_cache, edge))
end

"""
Do a sequential update of the message tensors on `edges`
"""
function update_iteration(
        alg::Algorithm"bp",
        bpc::AbstractBeliefPropagationCache,
        edges::Vector;
        (update_diff!) = nothing,
    )
    bpc = copy(bpc)
    for e in edges
        prev_message = !isnothing(update_diff!) ? message(bpc, e) : nothing
        update_message!(alg.kwargs.message_update_alg, bpc, e)
        if !isnothing(update_diff!)
            update_diff![] += message_diff(message(bpc, e), prev_message)
        end
    end
    return bpc
end

"""
Do parallel updates between groups of edges of all message tensors
Currently we send the full message tensor data struct to update for each edge_group. But really we only need the
mts relevant to that group.
"""
function update_iteration(
        alg::Algorithm"bp",
        bpc::AbstractBeliefPropagationCache,
        edge_groups::Vector{<:Vector{<:AbstractEdge}};
        (update_diff!) = nothing,
    )
    new_mts = empty(messages(bpc))
    for edges in edge_groups
        bpc_t = update_iteration(alg.kwargs.message_update_alg, bpc, edges; (update_diff!))
        for e in edges
            set!(new_mts, e, message(bpc_t, e))
        end
    end
    return set_messages(bpc, new_mts)
end

"""
More generic interface for update, with default params
"""
function update(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache)
    compute_error = !isnothing(alg.kwargs.tolerance)
    if isnothing(alg.kwargs.maxiter)
        error("You need to specify a number of iterations for BP!")
    end
    for i in 1:alg.kwargs.maxiter
        diff = compute_error ? Ref(0.0) : nothing
        bpc = update_iteration(alg, bpc, alg.kwargs.edge_sequence; (update_diff!) = diff)
        if compute_error && (diff.x / length(alg.kwargs.edge_sequence)) <= alg.kwargs.tolerance
            if alg.kwargs.verbose
                println("BP converged to desired precision after $i iterations.")
            end
            break
        end
    end
    return bpc
end

function update(bpc::AbstractBeliefPropagationCache; alg = default_update_alg(bpc), kwargs...)
    return update(set_default_kwargs(Algorithm(alg; kwargs...), bpc), bpc)
end

#Edge sequence stuff
function forest_cover_edge_sequence(g::AbstractGraph; root_vertex = default_root_vertex)
    forests = forest_cover(g)
    edges = edgetype(g)[]
    for forest in forests
        trees = [forest[vs] for vs in connected_components(forest)]
        for tree in trees
            tree_edges = post_order_dfs_edges(tree, root_vertex(tree))
            push!(edges, vcat(tree_edges, reverse(reverse.(tree_edges)))...)
        end
    end
    return edges
end

function rescale_vertices!(
  bpc::BeliefPropagationCache,
  vertices::Vector;
)
  tn = network(bpc)

  for v in vertices
    vn = vertex_scalar(bpc, v)
    s = isreal(vn) ? sign(vn) : one(vn)
    if tn isa TensorNetworkState
        setindex_preserve_graph!(tn, tn[v]*s*inv(sqrt(vn)), v)
    elseif tn isa ITensorNetwork
        setindex_preserve_graph!(tn, tn[v]*s*inv(vn), v)
    else
        error("Don't know how to rescale the vertices of this type")
    end  
  end

  return bpc
end

const _default_bp_update_maxiter = 25
function default_tolerance(type)
    (type == Float32 || type == ComplexF32) && return 1e-5
    (type == Float64 || type == ComplexF64) && return 1e-8
end

function default_bp_update_kwargs(tns::TensorNetworkState)
    maxiter = is_tree(tns) ? 1 : _default_bp_update_maxiter
    tolerance = default_tolerance(ITensors.NDTensors.scalartype(tns))
    verbose =false
    return (; maxiter, tolerance, verbose)
end

default_bp_update_kwargs(bp_cache::BeliefPropagationCache) = default_bp_update_kwargs(network(bp_cache))

function make_hermitian(A::ITensor)
    A_inds = ITensors.inds(A)
    @assert length(A_inds) == 2
    return (A + ITensors.swapind(dag(A), first(A_inds), last(A_inds))) / 2
end

function rescale_messages!(bp_cache::BeliefPropagationCache, edges::Vector{<:AbstractEdge})
    ms = messages(bp_cache)
    for e in edges
        me, mer = normalize(message(bp_cache, e)), normalize(message(bp_cache, reverse(e)))
        n = (me*mer)[]
        if isreal(n)
            me *= sign(n)
            n *= sign(n)
        end
        set!(ms, e, me *inv(sqrt(n)))
        set!(ms, reverse(e), mer * inv(sqrt(n)))
    end
    return bp_cache
end

#Calculate the correlation flowing around single loop of the bp cache via an eigendecomposition
function loop_correlation(bpc::BeliefPropagationCache, loop::Vector{<:NamedEdge}, target_e::NamedEdge)

    is_tree(bpc) && return 0

    es = vcat(loop, [target_e])
    incoming_es = boundary_edges(bpc, es)
    incoming_messages = ITensor[message(bpc, e) for e in incoming_es]
    vs = unique(vcat(src.(loop), dst.(loop)))

    src_vertex = src(target_e)
    e_linkinds = inds(message(bpc, target_e))
    e_linkinds_sim = sim.(e_linkinds)

    local_tensors = ITensor[]
    ts = bp_factors(bpc, src_vertex)

    for t in ts
        t_inds = filter(i -> i ∈ e_linkinds, inds(t))
        if !isempty(t_inds)
            t_ind = only(t_inds)
            t_ind_pos = findfirst(x -> x == t_ind, e_linkinds)
            t = replaceind(t, t_ind, e_linkinds_sim[t_ind_pos])
        end
        push!(local_tensors, t)
    end

    tensors = ITensor[local_tensors; reduce(vcat, [bp_factors(bpc, v) for v in setdiff(vs, [src_vertex])]); incoming_messages]
    seq = ITensorNetworks.contraction_sequence(tensors; alg = "einexpr", optimizer = Greedy())
    t = contract(tensors; sequence = seq)

    row_combiner, col_combiner = ITensors.combiner(e_linkinds), ITensors.combiner(e_linkinds_sim)
    t = t * row_combiner * col_combiner
    t = ITensors.NDTensors.array(t)
    λs = reverse(sort(LinearAlgebra.eigvals(t); by = abs))
    err = 1.0 - abs(λs[1]) / sum(abs.(λs))
    return err
end

#Calculate the correlations flowing around each of the primitive loops of the BP cache
function loop_correlations(bpc::BeliefPropagationCache, smallest_loop_size::Int; kwargs...)
    g = underlying_graph(bpc)
    cycles = NamedGraphs.cycle_to_path.(NamedGraphs.unique_simplecycles_limited_length(g, smallest_loop_size))
    corrs = []
    for loop in cycles
        corrs = append!(corrs, loop_correlation(bpc, loop[1:(length(loop)-1)], reverse(last(loop)); kwargs...))
    end
    return corrs
end