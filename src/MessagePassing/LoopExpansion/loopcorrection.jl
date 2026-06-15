using NamedGraphs.GraphsExtensions: boundary_edges

function loopcorrected_partitionfunction(
        bp_cache::BeliefPropagationCache,
        max_configuration_size::Integer,
    )
    zbp = partitionfunction(bp_cache)
    bp_cache = rescale(bp_cache)
    egs =
        connected_edgeinduced_subgraphs_no_leaves(graph(bp_cache), max_configuration_size)
    isempty(egs) && return zbp
    ws = weights(bp_cache, egs)
    return zbp * (1 + sum(ws))
end

# Linked-cluster (free-energy) form of the loop correction.
#
#   F = ln Z = ln Z_BP + ln(Z / Z_BP),   Z / Z_BP = Σ_{polymer configs} ∏ w_C ,
#
# where the polymers are the *connected* no-leaf edge subgraphs (generalized loops) and the
# hard-core exclusion is vertex-sharing. The linked-cluster theorem makes F extensive: its
# expansion is a sum over CONNECTED clusters only — the vertex-disjoint products that the
# partition-function series `loopcorrected_partitionfunction` would have to enumerate beyond
# total size `2·girth − 1` are resummed by the exponential and never appear here. To leading
# cumulant order this is `ln Z_BP + Σ_C w_C` over the connected no-leaf clusters (overlapping-
# cluster corrections, `−½ Σ_{C∼C'} w_C w_{C'} + …`, are higher order). Both this and
# `loopcorrected_partitionfunction` enumerate the same generalized loops via
# `connected_edgeinduced_subgraphs_no_leaves` (so both capture the bridge "dumbbell"
# diagrams); they differ only in how the cluster weights are resummed.
#
# Note `loopcorrected_free_energy` and `log(loopcorrected_partitionfunction)` differ at
# O(w²): they agree exactly only when no loops fit (Σw = 0), where both reduce to `ln Z_BP`.
function loopcorrected_free_energy(
        bp_cache::BeliefPropagationCache,
        max_configuration_size::Integer,
    )
    zbp = partitionfunction(bp_cache)
    F = log(complex(zbp))
    bp_cache = rescale(bp_cache)
    g = graph(bp_cache)
    egs = connected_edgeinduced_subgraphs_no_leaves(g, max_configuration_size)
    isempty(egs) && return F
    return F + sum(weights(bp_cache, egs))
end

# Free-energy (generating-function) estimate of a single-site observable, exposed through
# `expect(...; alg = "loopcorrections")`.
#
#   ⟨Ô⟩ = ∂_ε ln⟨ψ|e^{ε Ô}|ψ⟩|_{ε=0}  ≈  [F(ε) − F(−ε)] / (2ε),
#   F(t) = ln⟨ψ|e^{t Ô}|ψ⟩ = ln‖e^{t Ô/2}|ψ⟩‖²        (Hermitian Ô),
#
# Each F is the loop-corrected free energy of a *genuine norm network*: the gate
# e^{±ε Ô/2} is absorbed (un-normalized) into the ket at the observable site and BP is
# re-solved, so the loop expansion prunes ALL leaves — there is no protected operator
# vertex, hence no "anomalous" leaf-containing cluster, which is what tends to make this
# estimator converge smoothly. BP is re-solved on each shifted network so the loop
# corrections also pick up the linear response of the messages.

# F(α-shifted) = loop-corrected free energy of e^{α Ô}|ψ⟩, built by absorbing the
# (un-normalized) one-site gate into the ket and re-solving BP for the perturbed network.
# `F = ln Z_BP + Σ_C w_C` is the additive linked-cluster free energy (`loopcorrected_free_energy`),
# the genuine extensive log-partition-function whose smooth ε-dependence makes this estimator
# converge well — NOT `log(loopcorrected_partitionfunction) = ln Z_BP + ln(1 + Σ_C w_C)`, which
# only agrees with it to O(w) and resums the same clusters multiplicatively instead.
function _gated_loop_free_energy(
        ψ::TensorNetworkState, op_string::String, v, α, max_configuration_size::Integer;
        cache_update_kwargs,
    )
    s = only(siteinds(ψ)[v])
    G = ITensors.exp(α * ITensors.op(op_string, s); ishermitian = true)
    bpc = update(BeliefPropagationCache(ψ); cache_update_kwargs...)
    # `normalize_tensors = false`: the un-normalized gated tensor is exactly what makes the
    # squared norm equal the partition function ⟨ψ|e^{2α Ô}|ψ⟩ we want to differentiate.
    bpc, _ = apply_gate!(G, bpc; v⃗ = [v], apply_kwargs = (; normalize_tensors = false))
    bpc = update(bpc; cache_update_kwargs...)
    return loopcorrected_free_energy(bpc, max_configuration_size)
end

#Transform the indices in the given subgraph of the tensornetwork so that antiprojectors can be inserted without duplicate indices appearing
function sim_edgeinduced_subgraph(bpc::BeliefPropagationCache, eg)
    bpc = copy(bpc)
    vs = collect(vertices(eg))
    es =
        unique(collect(Iterators.flatten(boundary_edges(bpc, [v]; dir = :out) for v in vs)))
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
                    adapt_like(message(bpc, e), denseblocks(delta(combinedind(col_combiner), dag(combinedind(row_combiner)))))
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
    local_tensors = collect(Iterators.flatten(bp_factors(bpc, v) for v in vs))
    ts = [incoming_ms; local_tensors; antiprojectors]
    seq = any(hasqns.(ts)) ? contraction_sequence(ts; alg = "optimal") : contraction_sequence(ts; alg = "einexpr", optimizer = Greedy())
    return scalar(contract(ts; sequence = seq))
end

#Vectorized version of weight
function weights(bpc::BeliefPropagationCache, egs)
    return [weight(bpc, eg) for eg in egs]
end
