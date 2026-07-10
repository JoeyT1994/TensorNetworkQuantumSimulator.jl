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
    # Aux (dangling non-physical) legs recorded before the bond relabeling below: the
    # relabeled bond dangles in the modified network, and the live classification in
    # `norm_factors` would misread it as a charge leg and unprime it on the bra.
    aux = if network(bpc) isa TensorNetworkState
        Dictionary(vs, [auxinds(network(bpc), v) for v in vs])
    else
        nothing
    end
    es =
        unique(collect(Iterators.flatten(boundary_edges(bpc, [v]; dir = :out) for v in vs)))
    updated_es = NamedEdge[]
    antiprojectors = ITensor[]
    for e in es
        if reverse(e) ∉ updated_es
            mer = message(bpc, reverse(e))
            linds = filter(i -> plev(i) == 0, inds(mer))
            linds_sim = sim.(linds)
            mer = replaceinds(mer, (linds .=> linds_sim)...)
            if network(bpc) isa TensorNetworkState
                mer = replaceinds(mer, (conj.(prime.(linds)) .=> conj.(prime.(linds_sim)))...)
            end
            ms = messages(bpc)
            set!(ms, reverse(e), mer)
            t = network(bpc)[src(e)]
            # On a graded backend the network tensor carries the dual of the message's axis.
            # Index equality is dual-insensitive, so `intersect` matches the two.
            t_inds = intersect(inds(t), linds)
            if !isempty(t_inds)
                t_ind = only(t_inds)
                t_ind_pos = findfirst(==(t_ind), linds)
                t = replaceinds(t, t_ind => linds_sim[t_ind_pos])
                setindex_preserve!(bpc, t, src(e))
            end
            push!(updated_es, e)

            if e ∈ edges(eg) || reverse(e) ∈ edges(eg)
                me = message(bpc, e)
                # The identity legs take their axes from the actual message legs (`me`
                # for the rows, the relabeled `mer` for the columns) so `ap - me * mer`
                # lines up on every backend: the two messages carry mutually dual copies
                # of the bond axes. The domain of the fused identity comes out dualized
                # relative to the passed indices, so the columns go in `conj`ed.
                row_inds = Index[only(intersect(inds(me), [l])) for l in linds]
                col_inds = Index[only(intersect(inds(mer), [l])) for l in linds_sim]
                if network(bpc) isa TensorNetworkState
                    append!(row_inds, Index[only(intersect(inds(me), [prime(l)])) for l in linds])
                    append!(col_inds, Index[only(intersect(inds(mer), [prime(l)])) for l in linds_sim])
                end
                ap = adapt_like(me, identity_tensor(row_inds, conj.(col_inds)))
                ap = ap - me * mer
                push!(antiprojectors, ap)
            end
        end
    end
    return bpc, antiprojectors, aux
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
    bpc, antiprojectors, aux = sim_edgeinduced_subgraph(bpc, eg)
    incoming_ms =
        ITensor[message(bpc, e) for e in boundary_edges(bpc, es)]
    local_tensors = if isnothing(aux)
        collect(Iterators.flatten(bp_factors(bpc, v) for v in vs))
    else
        collect(Iterators.flatten(norm_factors(network(bpc), [v]; auxinds_f = u -> aux[u]) for v in vs))
    end
    ts = [incoming_ms; local_tensors; antiprojectors]
    seq = contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod())
    return scalar(contract_network(ts; sequence = seq))
end

#Vectorized version of weight
function weights(bpc::BeliefPropagationCache, egs)
    return [weight(bpc, eg) for eg in egs]
end
