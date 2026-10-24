using Dictionaries: Dictionary, set!, delete!
using Graphs: AbstractGraph, is_tree, connected_components
using NamedGraphs: post_order_dfs_edges, forest_cover_edge_sequence, boundary_edges, leaf_vertices, a_star
using LinearAlgebra: normalize

#TODO: Make this show() nicely.
struct BeliefPropagationCache{V, N <: AbstractTensorNetwork{V}, M <: Union{Tensor, Vector{Tensor}}} <:
    AbstractBeliefPropagationCache{V}
    network::N
    messages::Dictionary{NamedEdge, M}
    contraction_sequences::Dictionary{Pair, Vector}
    edge_sequence::Vector
end

function message_diff(message_a::Tensor, message_b::Tensor)
    n_a, n_b = norm(message_a), norm(message_b)
    f = abs2(dot(message_a, message_b) / (n_a * n_b))
    return 1 - f
end

messages(bp_cache::BeliefPropagationCache) = bp_cache.messages
network(bp_cache::BeliefPropagationCache) = bp_cache.network
graph(bp_cache::BeliefPropagationCache) = graph(network(bp_cache))

function BeliefPropagationCache(network, messages, contraction_sequences)
    return BeliefPropagationCache(network, messages, contraction_sequences, forest_cover_edge_sequence(graph(network)))
end
BeliefPropagationCache(network, messages) = BeliefPropagationCache(network, messages, Dictionary{Pair, Vector}())
BeliefPropagationCache(network) = BeliefPropagationCache(network, default_messages())

contraction_sequences(bp_cache::BeliefPropagationCache) = bp_cache.contraction_sequences

function Base.copy(bp_cache::BeliefPropagationCache)
    return BeliefPropagationCache(copy(network(bp_cache)), copy(messages(bp_cache)), copy(contraction_sequences(bp_cache)), copy(edge_sequence(bp_cache)))
end

default_bp_maxiter(g::AbstractGraph) = is_tree(g) ? 1 : _default_bp_update_maxiter

edge_sequence(bp_cache::BeliefPropagationCache) = bp_cache.edge_sequence

function edge_scalar(bp_cache::BeliefPropagationCache, edge::AbstractEdge)
    return scalar(message(bp_cache, edge) * message(bp_cache, reverse(edge)))
end

#Algorithmic defaults
default_update_alg(bp_cache::BeliefPropagationCache) = "bp"
default_message_update_alg(bp_cache::BeliefPropagationCache) = "contract"
function set_default_kwargs(alg::Algorithm"contract", bp_cache::AbstractBeliefPropagationCache)
    return Algorithm("contract"; merge((; normalize = true, sequence_alg = "optimal"), alg.kwargs)...)
end
function set_default_kwargs(alg::Algorithm"bp", bp_cache::BeliefPropagationCache)
    defaults = (; verbose = false, maxiter = default_bp_maxiter(bp_cache), edge_sequence = edge_sequence(bp_cache), tolerance = nothing)
    kwargs = merge(defaults, alg.kwargs)
    message_update_alg = set_default_kwargs(get(kwargs, :message_update_alg, Algorithm(default_message_update_alg(bp_cache))), bp_cache)
    return Algorithm("bp"; merge(kwargs, (; message_update_alg))...)
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
    (type == Float64 || type == ComplexF64) && return 1.0e-8
    return nothing
end

# One sweep is exact on a tree; otherwise sweep to a tolerance set by the precision.
function default_bp_update_kwargs(tn::AbstractTensorNetwork)
    tree = is_tree(tn)
    return (; maxiter = default_bp_maxiter(graph(tn)), tolerance = tree ? nothing : default_tolerance(scalartype(tn)), verbose = false)
end

default_bp_update_kwargs(bp_cache::BeliefPropagationCache) = default_bp_update_kwargs(network(bp_cache))

function rescale_messages!(bp_cache::BeliefPropagationCache, edges::Vector{<:AbstractEdge})
    for e in edges
        me, mer = normalize(message(bp_cache, e)), normalize(message(bp_cache, reverse(e)))
        n = scalar(me * mer)
        if isreal(n)
            me *= sign(n)
            n *= sign(n)
        end
        setmessage!(bp_cache, e, me * inv(sqrt(n)))
        setmessage!(bp_cache, reverse(e), mer * inv(sqrt(n)))
    end
    return bp_cache
end

#Calculate the correlation flowing around single loop of the bp cache via an eigendecomposition
function loop_correlation(bpc::BeliefPropagationCache, loop::Vector{<:NamedEdge}, target_e::NamedEdge)

    is_tree(bpc) && return 0

    es = vcat(loop, [target_e])
    incoming_messages = Tensor[message(bpc, e) for e in incoming_edges(bpc, es)]
    vs = unique(vcat(src.(loop), dst.(loop)))

    src_vertex = src(target_e)
    e_virtualinds = inds(message(bpc, target_e))
    e_virtualinds_sim = sim.(e_virtualinds)

    local_tensors = Tensor[replaceinds(t, e_virtualinds, e_virtualinds_sim) for t in bp_factors(bpc, src_vertex)]

    tensors = Tensor[local_tensors; reduce(vcat, [bp_factors(bpc, v) for v in setdiff(vs, [src_vertex])]); incoming_messages]
    seq = contraction_sequence(tensors; alg = "omeinsum", optimizer = GreedyMethod())
    t = contract(tensors; sequence = seq)

    row_combiner, col_combiner = combiner(e_virtualinds), combiner(e_virtualinds_sim)
    t = t * row_combiner * col_combiner
    t = adapt(Vector{ComplexF64})(t)
    t = array(t)
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
