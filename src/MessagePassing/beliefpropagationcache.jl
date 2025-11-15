using Dictionaries: Dictionary, set!, delete!
using Graphs: AbstractGraph, is_tree, connected_components
using NamedGraphs.GraphsExtensions: default_root_vertex, forest_cover, post_order_dfs_edges
using ITensors: dim, ITensor, delta, Algorithm
using ITensors.NDTensors: scalartype
using LinearAlgebra: normalize

#TODO: Make this show() nicely.
struct BeliefPropagationCache{V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}} <:
    AbstractBeliefPropagationCache{V}
    network::N
    messages::Dictionary{NamedEdge, M}
end

#TODO: Take `dot` without precontracting the messages to allow scaling to more complex messages
function message_diff(message_a::ITensor, message_b::ITensor)
    n_a, n_b = norm(message_a), norm(message_b)
    f = abs2(dot(message_a, message_b) / (n_a * n_b))
    return 1 - f
end

messages(bp_cache::BeliefPropagationCache) = bp_cache.messages
network(bp_cache::BeliefPropagationCache) = bp_cache.network

BeliefPropagationCache(network) = BeliefPropagationCache(network, default_messages())

function Base.copy(bp_cache::BeliefPropagationCache)
    return BeliefPropagationCache(copy(network(bp_cache)), copy(messages(bp_cache)))
end

default_bp_maxiter(g::AbstractGraph) = is_tree(g) ? 1 : _default_bp_update_maxiter

#TODO: Get subgraph working on an TensorNetwork to overload this directly
function default_bp_edge_sequence(bp_cache::BeliefPropagationCache)
    return forest_cover_edge_sequence(graph(bp_cache))
end

function edge_scalar(bp_cache::BeliefPropagationCache, edge::AbstractEdge)
    return (message(bp_cache, edge) * message(bp_cache, reverse(edge)))[]
end

#Algorithmic defaults
default_update_alg(bp_cache::BeliefPropagationCache) = "bp"
default_message_update_alg(bp_cache::BeliefPropagationCache) = "contract"
default_normalize(::Algorithm"contract") = true
default_sequence_alg(::Algorithm"contract") = "optimal"
default_enforce_hermicity(::Algorithm"contract", bp_cache::AbstractBeliefPropagationCache) = isa(network(bp_cache), TensorNetworkState) ? true : false
function set_default_kwargs(alg::Algorithm"contract", bp_cache::AbstractBeliefPropagationCache)
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

function update_message!(
        message_update_alg::Algorithm, bp_cache::BeliefPropagationCache, edge::AbstractEdge
    )
    return setmessage!(bp_cache, edge, updated_message(message_update_alg, bp_cache, edge))
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
        vertices::Vector
    )
    tn = network(bpc)

    for v in vertices
        vn = vertex_scalar(bpc, v)
        s = isreal(vn) ? sign(vn) : one(vn)
        if tn isa TensorNetworkState
            setindex_preserve!(tn, tn[v] * s * inv(sqrt(vn)), v)
        elseif tn isa TensorNetwork
            setindex_preserve!(tn, tn[v] * s * inv(vn), v)
        else
            error("Don't know how to rescale the vertices of this type")
        end
    end

    return bpc
end

const _default_bp_update_maxiter = 25
function default_tolerance(type)
    (type == Float32 || type == ComplexF32) && return 1.0e-5
    return (type == Float64 || type == ComplexF64) && return 1.0e-8
end

function default_bp_update_kwargs(tn::AbstractTensorNetwork)
    maxiter = is_tree(tn) ? 1 : _default_bp_update_maxiter
    tolerance = default_tolerance(ITensors.NDTensors.scalartype(tn))
    verbose = false
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
        n = (me * mer)[]
        if isreal(n)
            me *= sign(n)
            n *= sign(n)
        end
        set!(ms, e, me * inv(sqrt(n)))
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
    e_virtualinds = inds(message(bpc, target_e))
    e_virtualinds_sim = sim.(e_virtualinds)

    local_tensors = ITensor[]
    ts = bp_factors(bpc, src_vertex)

    for t in ts
        t_inds = filter(i -> i ∈ e_virtualinds, inds(t))
        if !isempty(t_inds)
            t_ind = only(t_inds)
            t_ind_pos = findfirst(x -> x == t_ind, e_virtualinds)
            t = replaceind(t, t_ind, e_virtualinds_sim[t_ind_pos])
        end
        push!(local_tensors, t)
    end

    tensors = ITensor[local_tensors; reduce(vcat, [bp_factors(bpc, v) for v in setdiff(vs, [src_vertex])]); incoming_messages]
    seq = contraction_sequence(tensors; alg = "einexpr", optimizer = Greedy())
    t = contract(tensors; sequence = seq)

    row_combiner, col_combiner = ITensors.combiner(e_virtualinds), ITensors.combiner(e_virtualinds_sim)
    t = t * row_combiner * col_combiner
    t = adapt(Vector{ComplexF64})(t)
    t = ITensors.NDTensors.array(t)
    λs = reverse(sort(LinearAlgebra.eigvals(t); by = abs))
    err = 1 - abs(λs[1]) / sum(abs.(λs))
    return err
end

#Calculate the correlations flowing around each of the primitive loops of the BP cache
function loop_correlations(bpc::BeliefPropagationCache, smallest_loop_size::Integer; kwargs...)
    g = graph(bpc)
    cycles = NamedGraphs.cycle_to_path.(NamedGraphs.unique_simplecycles_limited_length(g, smallest_loop_size))
    corrs = []
    for loop in cycles
        corrs = append!(corrs, loop_correlation(bpc, loop[1:(length(loop) - 1)], reverse(last(loop)); kwargs...))
    end
    return corrs
end

function loop_correlations(tn::AbstractTensorNetwork, smallest_loop_size::Integer; bp_update_kwargs = default_bp_update_kwargs(tn), kwargs...)
    return loop_correlations(update(BeliefPropagationCache(tn); bp_update_kwargs...), smallest_loop_size; kwargs...)
end
