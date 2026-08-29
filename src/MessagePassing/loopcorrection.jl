using NamedGraphs.GraphsExtensions: boundary_edges

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
    antiprojectors, projectors = tensortype(bpc)[], tensortype(bpc)[]
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
            t2 = replace_matching_ind(t, linds, linds_sim)
            t2 === t || setindex_preserve!(bpc, t2, src(e))
            push!(updated_es, e)

            if e ∈ edges(eg) || reverse(e) ∈ edges(eg)
                #the identity part of the antiprojector: one identity per rail (ket and,
                #for norm networks, bra), inserted as per-rail deltas. NEVER build this
                #as a fused-combiner sandwich C† δ C: for fermionic sectors that detour
                #picks up a parity twist on the odd fused sector and is not the identity
                #morphism (verified numerically; the two coincide for bosonic/dense data).
                parts = [delta(dag(l), ls) for (l, ls) in zip(linds, linds_sim)]
                if network(bpc) isa TensorNetworkState
                    append!(parts, [delta(prime(l), dag(prime(ls))) for (l, ls) in zip(linds, linds_sim)])
                end
                p = message(bpc, e) * mer
                ap = adapt_like(message(bpc, e), reduce(*, parts)) - p
                push!(antiprojectors, ap)
                push!(projectors, p)
            end
        end
    end
    return bpc, antiprojectors, projectors
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
    bpc, antiprojectors, projectors = sim_edgeinduced_subgraph(bpc, eg)
    incoming_ms =
        [message(bpc, e) for e in boundary_edges(bpc, es)]
    local_tensors = collect(Iterators.flatten(bp_factors(bpc, v) for v in vs))
    ts = [incoming_ms; local_tensors; antiprojectors]
    seq = contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod())
    w = scalar(contract(ts; sequence = seq))
    if any(has_closure_gauge, local_tensors)
        #fermionic multi-vertex closures pick up a parity-gauge sign when a fermion-odd
        #charge line threads the region (odd total fermion number only — all local
        #tensors are always parity-even). Expectation values are immune (the gauge
        #cancels in their closure ratios) but the loop weights are additive, so
        #normalize by the same region's bare all-projector closure, whose true value is 1.
        ts0 = [incoming_ms; local_tensors; projectors]
        seq0 = contraction_sequence(ts0; alg = "omeinsum", optimizer = GreedyMethod())
        w /= scalar(contract(ts0; sequence = seq0))
    end
    return w
end

#Vectorized version of weight
function weights(bpc::BeliefPropagationCache, egs)
    return [weight(bpc, eg) for eg in egs]
end
