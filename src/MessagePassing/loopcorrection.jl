using NamedGraphs: boundary_edges

function loopcorrected_partitionfunction(
        bp_cache::BeliefPropagationCache,
        max_configuration_size::Integer,
    )
    zbp = partitionfunction(bp_cache)
    bp_cache = rescale(bp_cache)
    egs =
        leafless_edge_induced_subgraphs(graph(bp_cache), max_configuration_size)
    isempty(egs) && return zbp
    ws = weights(bp_cache, egs)
    return zbp * (1 + sum(ws))
end

#Transform the indices in the given subgraph of the tensornetwork so that antiprojectors can be inserted without duplicate indices appearing
function sim_edgeinduced_subgraph(bpc::BeliefPropagationCache, eg)
    bpc = copy(bpc)
    vs = collect(vertices(eg))
    es =
        unique(collect(Iterators.flatten(boundary_edges(bpc, [v]; dir = :out) for v in vs)))
    updated_es = NamedEdge[]
    antiprojectors = Tensor[]
    for e in es
        if reverse(e) ∉ updated_es
            mer = message(bpc, reverse(e))
            linds = collect(inds(mer; plev = 0))
            linds_sim = sim.(linds)
            # the message's legs: the virtual legs, and for a state their bra copies too
            row_inds, col_inds = message_legs(network(bpc), linds), message_legs(network(bpc), linds_sim)
            mer = replaceinds(mer, row_inds, col_inds)
            setmessage!(bpc, reverse(e), mer)
            t = network(bpc)[src(e)]
            hascommoninds(t, linds) && setindex_preserve!(bpc, replaceinds(t, linds, linds_sim), src(e))
            push!(updated_es, e)

            if e ∈ edges(eg) || reverse(e) ∈ edges(eg)
                row_combiner, col_combiner = combiner(row_inds), combiner(col_inds)
                ap =
                    adapt_like(message(bpc, e), dense(delta(combinedind(col_combiner), dag(combinedind(row_combiner)))))
                ap = ap * row_combiner * dag(col_combiner)
                ap = ap - message(bpc, e) * mer
                push!(antiprojectors, ap)
            end
        end
    end
    return bpc, antiprojectors
end

# The edges into the region spanned by the edges `es`, from outside it.
function incoming_edges(bpc::BeliefPropagationCache, es::Vector{<:NamedEdge})
    vs = unique(vcat(src.(es), dst.(es)))
    return NamedEdge[e for v in vs for e in boundary_edges(bpc, [v]; dir = :in) if e ∉ es && reverse(e) ∉ es]
end

#Compute the contraction of the bp configuration specified by the edge induced subgraph eg
function weight(bpc::BeliefPropagationCache, eg)
    vs = collect(vertices(eg))
    es = collect(edges(eg))
    bpc, antiprojectors = sim_edgeinduced_subgraph(bpc, eg)
    incoming_ms = Tensor[message(bpc, e) for e in incoming_edges(bpc, es)]
    local_tensors = collect(Iterators.flatten(bp_factors(bpc, v) for v in vs))
    ts = [incoming_ms; local_tensors; antiprojectors]
    seq = any(isgraded, ts) ? contraction_sequence(ts; alg = "optimal") : contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod())
    return scalar(contract(ts; sequence = seq))
end

#Vectorized version of weight
function weights(bpc::BeliefPropagationCache, egs)
    return [weight(bpc, eg) for eg in egs]
end
