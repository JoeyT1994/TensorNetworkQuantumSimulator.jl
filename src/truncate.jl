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

function truncate(bpc::BeliefPropagationCache; cache_update_kwargs = default_bp_update_kwargs(bpc), maxdim::Integer, cutoff = nothing, edge_color = true, normalize_tensors = true)
    bpc = copy(bpc)
    s = siteinds(network(bpc))
    apply_kwargs = (; maxdim, cutoff, normalize_tensors)
    # Truncate a group of disjoint edges at a time (the colours of an edge colouring), or one edge
    # at a time, updating the cache after each group.
    edge_groups = if edge_color
        g = graph(network(bpc))
        z = maximum(v -> degree(g, v), vertices(g))
        #Vizing's theorem only guarantees Δ + 1 colours suffice. Class 2 graphs (odd cycles, and any
        #lattice containing a triangle) are not Δ-edge-colourable, so fall back to Δ + 1 for those.
        try
            SimpleGraphAlgorithms.edge_color(g, z)
        catch err
            err isa InterruptException && rethrow()
            SimpleGraphAlgorithms.edge_color(g, z + 1)
        end
    else
        [[e] for e in edges(bpc)]
    end
    for eg in edge_groups
        for e in eg
            truncatable_edge(bpc, e) && apply_gate!(identity_gate(bpc, s, e), bpc; v⃗ = [src(e), dst(e)], apply_kwargs)
        end
        bpc = update(bpc; cache_update_kwargs...)
    end
    return bpc
end

function truncate(bmps_cache::BoundaryMPSCache; maxdim::Integer, cutoff = nothing, normalize_tensors = true)
    bmps_cache = copy(bmps_cache)
    s = siteinds(network(bmps_cache))
    apply_kwargs = (; maxdim, cutoff)
    ps = sort(parent.(quotientvertices(supergraph(bmps_cache))))
    for (i, p) in enumerate(ps)
        g = partition_graph(bmps_cache, QuotientVertex(p))
        leaves = leaf_vertices(g)
        seq = a_star(g, last(leaves), first(leaves))
        !isempty(seq) && update_partition!(bmps_cache, seq)
        for e in reverse.(reverse(seq))
            if truncatable_edge(bmps_cache, e)
                envs = incoming_messages(bmps_cache, [src(e), dst(e)])
                ρv1, ρv2 = full_update(identity_gate(bmps_cache, s, e), network(bmps_cache), [src(e), dst(e)]; envs, apply_kwargs...)
                if normalize_tensors
                    ρv1 = normalize(ρv1)
                    ρv2 = normalize(ρv2)
                end
                setindex_preserve!(bmps_cache, ρv1, src(e))
                setindex_preserve!(bmps_cache, ρv2, dst(e))
            end
            update_partition!(bmps_cache, [e])
        end

        if i != length(ps)
            bmps_cache = update(bmps_cache; alg = "bp", edge_sequence = [QuotientEdge(ps[i] => ps[i + 1])], maxiter = 1)
        end
    end

    return bmps_cache
end

function truncate(alg::Algorithm"bp", tns::TensorNetworkState; cache_update_kwargs = default_bp_update_kwargs(tns), kwargs...)
    bp_cache = converged_cache(alg, tns; cache_update_kwargs)
    return network(truncate(bp_cache; cache_update_kwargs, kwargs...))
end

# Truncate the bonds within the rows, then within the columns.
function truncate(alg::Algorithm"boundarymps", tns::TensorNetworkState; mps_bond_dimension::Integer, gauge_state = true, kwargs...)
    tns = copy(tns)
    for partition_by in ("row", "col")
        bmps_cache = BoundaryMPSCache(tns, mps_bond_dimension; partition_by, gauge_state)
        quotient = quotient_graph(supergraph(bmps_cache))
        leaves = leaf_vertices(quotient)
        seq = QuotientEdge.(a_star(quotient, last(leaves), first(leaves)))
        bmps_cache = update(bmps_cache; alg = "bp", edge_sequence = seq, maxiter = 1)
        tns = network(truncate(bmps_cache; kwargs...))
    end
    return tns
end

"""
    truncate(tns::TensorNetworkState; alg = default_truncate_alg(tns), kwargs...)

Truncate the virtual indices of tensors in the given `TensorNetworkState` using the specified algorithm.

# Arguments
- `tns::TensorNetworkState`: The tensor network state to be truncated.

# Keyword Arguments
- `alg`: The algorithm to use for truncation. Supported algorithms:
    - `"bp"`: Belief propagation-based truncation (works on any network, cheap but can be less accurate when loop correlations are present).
    - `"boundarymps"`: Boundary MPS-based truncation (requires `mps_bond_dimension`; works only on planar networks, more expensive but more accurate with large MPS bond dimension).
- `maxdim::Integer`: The maximum bond dimension to retain after truncation.
- `cutoff::Number`: The singular value cutoff for truncation (optional).

# Returns
- The truncated `TensorNetworkState`.
"""
function truncate(tns::TensorNetworkState; alg = default_truncate_alg(tns), kwargs...)
    algorithm_check(tns, "truncate", alg)
    return truncate(Algorithm(alg), tns; kwargs...)
end
