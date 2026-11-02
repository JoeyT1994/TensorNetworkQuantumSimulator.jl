using SimpleGraphAlgorithms: SimpleGraphAlgorithms

default_truncate_alg(tns::TensorNetworkState) = nothing

function truncatable_edge(cache::AbstractBeliefPropagationCache, e::NamedEdge)
    vinds = virtualinds(cache, e)
    isempty(vinds) && return false
    all([dim(vind) == 1 for vind in vinds]) && return false
    return true
end

# The identity on the sites of both ends of e, in the storage of `ref`.
identity_gate(ref, s, e) = adapt(datatype(ref))(prod(op("I", sv) for sv in [s[src(e)]; s[dst(e)]]))

# The edges of g as the colours of an edge colouring (disjoint edges), or one edge per group.
function truncation_edge_groups(g; edge_color = true)
    edge_color || return [[e] for e in edges(g)]
    z = maximum(v -> degree(g, v), vertices(g))
    #Vizing's theorem only guarantees Δ + 1 colours suffice. Class 2 graphs (odd cycles, and any
    #lattice containing a triangle) are not Δ-edge-colourable, so fall back to Δ + 1 for those.
    try
        return SimpleGraphAlgorithms.edge_color(g, z)
    catch err
        err isa InterruptException && rethrow()
        return SimpleGraphAlgorithms.edge_color(g, z + 1)
    end
end

function truncate(bpc::BeliefPropagationCache; cache_update_kwargs = default_bp_update_kwargs(bpc), maxdim::Integer, cutoff = nothing, edge_color = true, normalize_tensors = true)
    bpc = copy(bpc)
    s = siteinds(network(bpc))
    apply_kwargs = (; maxdim, cutoff, normalize_tensors)
    # Truncate a group of disjoint edges at a time (the colours of an edge colouring), or one edge
    # at a time, updating the cache after each group.
    edge_groups = truncation_edge_groups(graph(network(bpc)); edge_color)
    for eg in edge_groups
        for e in eg
            truncatable_edge(bpc, e) && apply_gate!(identity_gate(bpc, s, e), bpc; v⃗ = [src(e), dst(e)], apply_kwargs)
        end
        bpc = update(bpc; cache_update_kwargs...)
    end
    return bpc
end

function truncate(alg::Algorithm"bp", tns::TensorNetworkState; cache_update_kwargs = default_bp_update_kwargs(tns), kwargs...)
    bp_cache = converged_cache(alg, tns; cache_update_kwargs)
    return network(truncate(bp_cache; cache_update_kwargs, kwargs...))
end

# Truncate every bond by a full update of the identity in its CTMRG environment, a colour of an edge
# colouring at a time, updating the environment between colours.
function truncate(alg::Algorithm"ctmrg", tns::TensorNetworkState; env_maxdim::Integer, edge_color = true, kwargs...)
    s = siteinds(tns)
    gates = [
        (identity_gate(tns, s, e), [src(e), dst(e)])
            for eg in truncation_edge_groups(graph(tns); edge_color) for e in eg if any(i -> dim(i) > 1, virtualinds(tns, e))
    ]
    return first(full_update_sequence(gates, tns; env_maxdim, kwargs...))
end

"""
    truncate(tns::TensorNetworkState; alg = default_truncate_alg(tns), kwargs...)

Truncate the virtual indices of tensors in the given `TensorNetworkState` using the specified algorithm.

# Arguments
- `tns::TensorNetworkState`: The tensor network state to be truncated.

# Keyword Arguments
- `alg`: The algorithm to use for truncation. Supported algorithms:
    - `"bp"`: Belief propagation-based truncation (works on any network, cheap but can be less accurate when loop correlations are present).
    - `"ctmrg"`: Full update of each bond in its CTMRG environment (requires `env_maxdim`; works only on planar networks, more expensive but more accurate when loop correlations are present).
- `maxdim::Integer`: The maximum bond dimension to retain after truncation.
- `cutoff::Number`: The singular value cutoff for truncation (optional).

# Returns
- The truncated `TensorNetworkState`.
"""
function truncate(tns::TensorNetworkState; alg = default_truncate_alg(tns), kwargs...)
    algorithm_check(tns, "truncate", alg)
    return truncate(Algorithm(alg), tns; kwargs...)
end
