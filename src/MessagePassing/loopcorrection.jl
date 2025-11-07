using NamedGraphs.GraphsExtensions: boundary_edges

function loopcorrected_partitionfunction(
        bp_cache::BeliefPropagationCache,
        max_configuration_size::Integer,
    )
    zbp = partitionfunction(bp_cache)
    bp_cache = rescale(bp_cache)
    #TODO: Fix edgeinduced_subgraphs_no_leaves for PartitionedGraphView type
    #Count the cycles using NamedGraphs
    egs =
        edgeinduced_subgraphs_no_leaves(graph(bp_cache), max_configuration_size)
    isempty(egs) && return zbp
    ws = weights(bp_cache, egs)
    return zbp * (1 + sum(ws))
end

#Transform the indices in the given subgraph of the tensornetwork so that antiprojectors can be inserted without duplicate indices appearing
function sim_edgeinduced_subgraph(bpc::BeliefPropagationCache, eg)
    bpc = copy(bpc)
    vs = collect(vertices(eg))
    es =
        unique(reduce(vcat, [boundary_edges(bpc, [v]; dir = :out) for v in vs]))
    updated_es = NamedEdge[]
    antiprojectors = ITensor[]
    for e in es
        if reverse(e) ∉ updated_es
            mer = message(bpc, reverse(e))
            linds = filter(i -> plev(i) == 0, inds(mer))
            linds_sim = sim.(linds)
            mer = replaceinds(mer, linds, linds_sim)
            if network(bpc) isa TensorNetworkState
                mer = replaceinds(mer, dag.(prime.(linds)), dag.(prime.(linds_sim)))
            end
            ms = messages(bpc)
            set!(ms, reverse(e), mer)
            t = network(bpc)[src(e)]
            t_inds = filter(i -> i ∈ linds, inds(t))
            if !isempty(t_inds)
                t_ind = only(t_inds)
                t_ind_pos = findfirst(x -> x == t_ind, linds)
                t = replaceind(t, t_ind, linds_sim[t_ind_pos])
                setindex_preserve!(bpc, t, src(e))
            end
            push!(updated_es, e)

            if e ∈ edges(eg) || reverse(e) ∈ edges(eg)
                row_inds, col_inds = linds, linds_sim
                if network(bpc) isa TensorNetworkState
                    row_inds = vcat(row_inds, dag.(prime.(row_inds)))
                    col_inds = vcat(col_inds, dag.(prime.(col_inds)))
                end
                row_combiner, col_combiner = combiner(row_inds), combiner(col_inds)
                ap =
                    adapt(datatype(message(bpc, e)))(denseblocks(delta(combinedind(col_combiner), dag(combinedind(row_combiner)))))
                ap = ap * row_combiner * dag(col_combiner)
                ap = ap - message(bpc, e) * mer
                push!(antiprojectors, ap)
            end
        end
    end
    return bpc, antiprojectors
end

#Get the all edges incident to the region specified by the vector of edges passed
function NamedGraphs.GraphsExtensions.boundary_edges(
        bpc::BeliefPropagationCache,
        es::Vector{<:NamedEdge},
    )
    vs = unique(vcat(src.(es), dst.(es)))
    bpes = NamedEdge[]
    for v in vs
        incoming_es = NamedGraphs.GraphsExtensions.boundary_edges(bpc, [v]; dir = :in)
        incoming_es = filter(e -> e ∉ es && reverse(e) ∉ es, incoming_es)
        append!(bpes, incoming_es)
    end
    return bpes
end

#Compute the contraction of the bp configuration specified by the edge induced subgraph eg
function weight(bpc::BeliefPropagationCache, eg)
    vs = collect(vertices(eg))
    es = collect(edges(eg))
    bpc, antiprojectors = sim_edgeinduced_subgraph(bpc, eg)
    incoming_ms =
        ITensor[message(bpc, e) for e in boundary_edges(bpc, es)]
    local_tensors = reduce(vcat, [bp_factors(bpc, v) for v in vs])
    ts = [incoming_ms; local_tensors; antiprojectors]
    seq = any(hasqns.(ts)) ? contraction_sequence(ts; alg = "optimal") : contraction_sequence(ts; alg = "einexpr", optimizer = Greedy())
    return contract(ts; sequence = seq)[]
end

#Vectorized version of weight
function weights(bpc::BeliefPropagationCache, egs)
    return [weight(bpc, eg) for eg in egs]
end
