using SimpleGraphAlgorithms: SimpleGraphAlgorithms

default_truncate_alg(tns::TensorNetworkState) = nothing

function truncatable_edge(cache::AbstractBeliefPropagationCache, e::NamedEdge)
    vinds = virtualinds(cache, e)
    isempty(vinds) && return false
    all([dim(vind) == 1 for vind in vinds]) && return false
    return true
end

function ITensors.truncate(bpc::BeliefPropagationCache; bp_update_kwargs = default_bp_update_kwargs(bpc), maxdim::Integer, cutoff = nothing, edge_color = true, normalize_tensors = true)
    bpc = copy(bpc)
    s = siteinds(network(bpc))
    apply_kwargs = (; maxdim, cutoff, normalize_tensors)
    dtype = datatype(bpc)
    if edge_color
        g = graph(network(bpc))
        z = maximum([degree(g, v) for v in vertices(g)])
        edge_groups = SimpleGraphAlgorithms.edge_color(g, z)
        for eg in edge_groups
            for e in eg
                if truncatable_edge(bpc, e)
                    g1, g2 = reduce(*, [ITensors.op("I", sv) for sv in s[src(e)] ]), reduce(*, [ITensors.op("I", sv) for sv in s[dst(e)] ])
                    apply_gate!(adapt(dtype)(g1 * g2), bpc; v⃗ = [src(e), dst(e)], apply_kwargs)
                end
            end
            bpc = update(bpc; bp_update_kwargs...)
        end
    else
        for e in edges(bpc)
            g1, g2 = reduce(*, [ITensors.op("I", sv) for sv in s[src(e)]]), reduce(*, [ITensors.op("I", sv) for sv in s[dst(e)]])
            apply_gate!(adapt(dtype)(g1 * g2), bpc; v⃗ = [src(e), dst(e)], apply_kwargs)
            bpc = update(bpc; bp_update_kwargs...)
        end
    end
    return bpc
end

function ITensors.truncate(bmps_cache::BoundaryMPSCache; maxdim::Integer, cutoff = nothing, normalize_tensors = true)
    bmps_cache = copy(bmps_cache)
    s = siteinds(network(bmps_cache))
    apply_kwargs = (; maxdim, cutoff)
    dtype = datatype(bmps_cache)
    ps = sort(partitionvertices(supergraph(bmps_cache)); by = v -> parent(v))
    for (i, p) in enumerate(ps)
        g = partition_graph(bmps_cache, p)
        leaves = leaf_vertices(g)
        seq = a_star(g, last(leaves), first(leaves))
        !isempty(seq) && update_partition!(bmps_cache, seq)
        for e in reverse.(reverse(seq))
            if truncatable_edge(bmps_cache, e)
                g1, g2 = reduce(*, [ITensors.op("I", sv) for sv in s[src(e)]]), reduce(*, [ITensors.op("I", sv) for sv in s[dst(e)]])
                envs = incoming_messages(bmps_cache, [src(e), dst(e)])
                ρv1, ρv2 = full_update(adapt(dtype)(g1 * g2), network(bmps_cache), [src(e), dst(e)]; envs, apply_kwargs...)
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
            bmps_cache = update(bmps_cache; alg = "bp", edge_sequence = [PartitionEdge(parent(ps[i]) => parent(ps[i + 1]))], maxiter = 1)
        end
    end

    return bmps_cache
end

function ITensors.truncate(alg::Algorithm"bp", tns::TensorNetworkState; kwargs...)
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = update(bp_cache)
    bp_cache = truncate(bp_cache; kwargs...)
    return network(bp_cache)
end

function ITensors.truncate(alg::Algorithm"boundarymps", tns::TensorNetworkState; mps_bond_dimension::Integer, gauge_state = true, kwargs...)
    tns = copy(tns)
    bmps_cache = BoundaryMPSCache(tns, mps_bond_dimension; partition_by = "row", gauge_state)
    leaves = leaf_vertices(partitions_graph(supergraph(bmps_cache)))
    seq = PartitionEdge.(a_star(partitions_graph(supergraph(bmps_cache)), last(leaves), first(leaves)))
    bmps_cache = update(bmps_cache; alg = "bp", edge_sequence = seq, maxiter = 1)
    bmps_cache = truncate(bmps_cache; kwargs...)

    bmps_cache = BoundaryMPSCache(network(bmps_cache), mps_bond_dimension; partition_by = "col", gauge_state)
    leaves = leaf_vertices(partitions_graph(supergraph(bmps_cache)))
    seq = PartitionEdge.(a_star(partitions_graph(supergraph(bmps_cache)), last(leaves), first(leaves)))
    bmps_cache = update(bmps_cache; alg = "bp", edge_sequence = seq, maxiter = 1)
    bmps_cache = truncate(bmps_cache; kwargs...)

    return network(bmps_cache)
end

"""
    truncate(tns::TensorNetworkState; alg = default_truncate_alg(tns), kwargs...)
    Truncate the virtual indexes of tensors in the given `TensorNetworkState` `tns` using the specified algorithm `alg`.

    Arguments
    - `tns::TensorNetworkState`: The tensor network state to be truncated.
    Keyword Arguments
    - `alg`: The contraction algorithm to use for truncation. 
        Supported contraction algorithms include:
        - `"bp"`: Belief propagation-based truncation (works on any network, cheap but can be less accurate when loop correlations are present).
        - `"boundarymps"`: Boundary MPS-based truncation. Requires `mps_bond_dimension` Kwarg. Works only on a planar network and is more expensive, but more accurate if a large mps bond dim is used).
    - `maxdim::Integer`: The maximum bond dimension to retain after truncation.
    - `cutoff::Number`: The singular value cutoff for truncation (optional).
    Returns
    - The truncated `TensorNetworkState`.
"""
function ITensors.truncate(tns::TensorNetworkState; alg = default_truncate_alg(tns), kwargs...)
    algorithm_check(tns, "truncate", alg)
    return truncate(Algorithm(alg), tns; kwargs...)
end
