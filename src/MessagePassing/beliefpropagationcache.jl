using Dictionaries: Dictionary, set!, delete!
using Graphs: AbstractGraph, is_tree, connected_components
using NamedGraphs: post_order_dfs_edges, forest_cover_edge_sequence, boundary_edges, leaf_vertices, a_star
using LinearAlgebra: normalize

#TODO: Make this show() nicely.
struct BeliefPropagationCache{V, F, M <: ITNN.MessageCache} <: AbstractBeliefPropagationCache{V}
    factors::F
    messages::M
end

function message_diff(message_a, message_b)
    n_a, n_b = norm(message_a), norm(message_b)
    f = abs2(dot(message_a, message_b) / (n_a * n_b))
    return 1 - f
end

function BeliefPropagationCache(factors, messages::ITNN.MessageCache)
    return BeliefPropagationCache{vertextype(factors), typeof(factors), typeof(messages)}(factors, messages)
end
function BeliefPropagationCache(form::ITNN.AbstractBilinearFormNetwork)
    return BeliefPropagationCache(form, ITNN.message_environment(one, form))
end
# A network without site indices is contracted as it is; a state through its norm network.
function BeliefPropagationCache(ψ::ITensorNetwork)
    all(isempty, siteinds(ψ)) && return BeliefPropagationCache(ψ, default_messages(ψ))
    return BeliefPropagationCache(ITNN.normnetwork(ITensorBase.prime, ψ))
end
BeliefPropagationCache(blf::BilinearForm) = BeliefPropagationCache(blf, default_messages(blf))

# Starting messages for a network whose factor at each vertex is contracted as plain tensors:
# a `delta` over the link indices of each edge, in both directions.
function default_messages(tn::AbstractTensorNetwork)
    es = collect(edges(tn))
    return ITNN.messagecache(e -> adapt_like(tn, dense(delta(virtualinds(tn, e)))), [es; reverse.(es)])
end

messages(bp_cache::BeliefPropagationCache) = bp_cache.messages
message(bp_cache::BeliefPropagationCache, edge::AbstractEdge) = ITensorBase.state(bp_cache.messages[edge])
function network(bp_cache::AbstractBeliefPropagationCache)
    return bp_cache.factors isa ITNN.NormNetwork ? ITNN.ketnetwork(bp_cache.factors) : bp_cache.factors
end
graph(bp_cache::BeliefPropagationCache) = graph(network(bp_cache))

copy_factors(factors) = copy(factors)
copy_factors(nn::ITNN.NormNetwork) = ITNN.normnetwork(copy(ITNN.ketnetwork(nn)), ITNN.branamemap(nn))
function copy_factors(qf::ITNN.QuadraticFormNetwork)
    return ITNN.quadraticformnetwork(copy(ITNN.ketnetwork(qf)), ITNN.operatornetwork(qf), ITNN.branamemap(qf))
end

function Base.copy(bp_cache::BeliefPropagationCache)
    return BeliefPropagationCache(copy_factors(bp_cache.factors), copy(bp_cache.messages))
end

default_bp_maxiter(g::AbstractGraph) = is_tree(g) ? 1 : DEFAULT_BP_MAXITER

#Algorithmic defaults
function set_default_kwargs(alg::Algorithm"contract", bp_cache::AbstractBeliefPropagationCache)
    return Algorithm("contract"; merge((; normalize = true, sequence_alg = "optimal"), alg.kwargs)...)
end

function update(
        bp_cache::BeliefPropagationCache;
        maxiter = default_bp_maxiter(bp_cache),
        tolerance = nothing,
        verbose = false,
        edge_sequence = forest_cover_edge_sequence(graph(bp_cache)),
    )
    stopping_criterion = isnothing(tolerance) ? (; maxiter) : (; maxiter, tol = tolerance)
    factors = copy_factors(bp_cache.factors)
    ms = ITNN.beliefpropagation(factors, bp_cache.messages; edges = edge_sequence, stopping_criterion)
    return BeliefPropagationCache(factors, ms)
end

freenergy(bp_cache::BeliefPropagationCache) = ITNN.bethe_free_energy(bp_cache.factors, bp_cache.messages)

function rescale_vertices!(
        bpc::BeliefPropagationCache,
        vertices::Vector
    )
    tn = network(bpc)

    for v in vertices
        vn = ITNN.vertex_scalar(bpc.factors, bpc.messages, v)
        s = isreal(vn) ? sign(vn) : one(vn)
        if bpc.factors isa ITNN.NormNetwork
            tn[v] = tn[v] * s * inv(sqrt(vn))
        elseif bpc.factors isa ITensorNetwork
            tn[v] = tn[v] * s * inv(vn)
        else
            error("Don't know how to rescale the vertices of this type")
        end
    end

    return bpc
end

const DEFAULT_BP_MAXITER = 25
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
    ms = messages(bp_cache)
    for e in edges
        me, mer = normalize(ms[e]), normalize(ms[reverse(e)])
        n = (me * mer)[]
        if isreal(n)
            me *= sign(n)
            n *= sign(n)
        end
        ms[e] = me * inv(sqrt(n))
        ms[reverse(e)] = mer * inv(sqrt(n))
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

    local_tensors = Tensor[replace_matching_ind(t, e_virtualinds, e_virtualinds_sim) for t in ITNN.factor_tensors(bpc.factors, src_vertex)]

    tensors = Tensor[local_tensors; reduce(vcat, [collect(ITNN.factor_tensors(bpc.factors, v)) for v in setdiff(vs, [src_vertex])]); incoming_messages]
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
function loop_correlations(bpc::BeliefPropagationCache, smallest_loop_size::Integer)
    g = graph(bpc)
    cycles = NamedGraphs.cycle_to_path.(NamedGraphs.unique_simplecycles_limited_length(g, smallest_loop_size))
    corrs = []
    for loop in cycles
        corrs = append!(corrs, loop_correlation(bpc, loop[1:(length(loop) - 1)], reverse(last(loop))))
    end
    return corrs
end

function loop_correlations(tn::AbstractTensorNetwork, smallest_loop_size::Integer; cache_update_kwargs = default_bp_update_kwargs(tn), kwargs...)
    return loop_correlations(update(BeliefPropagationCache(tn); cache_update_kwargs...), smallest_loop_size; kwargs...)
end
