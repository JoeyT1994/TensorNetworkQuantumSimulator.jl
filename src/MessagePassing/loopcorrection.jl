using NamedGraphs.GraphsExtensions: boundary_edges

function loopcorrected_partitionfunction(
        bp_cache::BeliefPropagationCache,
        max_configuration_size::Integer,
    )
    zbp = partitionfunction(bp_cache)
    bp_cache = rescale(bp_cache)
    # Enumerate the generalized loops (connected, leaf-free edge subgraphs) directly. This is
    # connected-by-construction and bridge-complete, so it also captures the "dumbbell"
    # diagrams (no-leaf subgraphs joined by a bridge edge) that the cycle-union enumerator
    # silently drops.
    g = graph(bp_cache)
    egs = connected_edgeinduced_subgraphs_no_leaves(g, max_configuration_size)
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

# Relabel one bond index of a FermionicITensor (no-op if the index is absent), keeping
# the leg order, arrows and grading consistent.
function _ft_sim(ft::FermionicITensor, old::Index, new::Index)
    old in ft.order || return ft
    return replaceind(ft, old, new)
end
_ft_sim2(ft::FermionicITensor, a, as, b, bs) = _ft_sim(_ft_sim(ft, a, as), b, bs)

# A two-leg fermionic bond "identity" δ(s_loop, s_sim) that, when contracted between the
# two factors sharing a doubled-network bond, reproduces the direct bond exactly. `dirs`
# are chosen opposite to the two neighbouring factors' arrows, and `ng` (= 0 or 1) counts
# the supertrace-metric insertions on `s_loop`. Empirically (validated to machine precision
# on grid/hex lattices) the faithful single-leg replacement needs `ng = Int(a_dst)`, i.e.
# a metric exactly when the destination factor holds the bond IN.
function _fermionic_bond_identity(grade::Vector{Bool}, s_loop::Index, s_sim::Index, dir_loop::Bool, dir_sim::Bool, ng::Integer)
    T = denseblocks(delta(s_loop, s_sim))
    gr = Dictionary{Index, Vector{Bool}}([s_loop, s_sim], [grade, grade])
    for _ in 1:ng
        T = _apply_parity(gr, T, s_loop)
    end
    return FermionicITensor(T, Index[s_loop, s_sim], Bool[dir_loop, dir_sim], gr)
end

# Fermionic loop weight for an arbitrary no-leaf edge-induced subgraph `eg`.
#
# This is the fermionic analogue of `sim_edgeinduced_subgraph`: on every loop edge we
# insert the antiprojector Q_e = 𝟙_e − P_e, where 𝟙_e is the faithful doubled-bond
# identity (`_fermionic_bond_identity` on the ket and bra layers) and P_e = |m_ē⟩⟨m_e| is
# the rank-1 BP projector (the message outer product `m_ē ⊗ m_e`, matching the bosonic
# `ap - message(e)*mer`). Each loop edge has its bond legs `sim`'d on the destination side
# so the identity/antiprojector can bridge them back. Edges incident to the loop region
# that are NOT loop edges are capped by their BP component instead: a chord (both endpoints
# in the loop) gets the rank-1 cap `m_ē ⊗ m_e` (its bond `sim`'d on one side), while a true
# boundary edge contributes its single external cavity message.
#
# Expanding ∏_e(𝟙_e − P_e) over the subgraph, every term that puts a projector on a subset
# of edges propagates the BP cap and contributes ∏_{v∈eg} z_v, so the antiprojected
# contraction equals C_full − ∏z; the gauge-invariant weight returned is that divided by
# the mean-field value ∏z (z_v = vertex_scalar). For bosons ∏z ≡ 1; here the per-vertex
# z_v can be 4th-roots of unity, so dividing (rather than subtracting) keeps the result
# real. Validated to machine precision against the exact partition function on 2×2, 2×3 and
# 3×3 grids (the full no-leaf-subgraph series reconstructs Z exactly).
function _fermionic_loop_weight(bpc::BeliefPropagationCache, eg)
    vs = collect(vertices(eg))
    vset = Set(vs)
    es = collect(edges(eg))
    facs = Dict(v => collect(bp_factors(bpc, v)) for v in vs)
    Qs = FermionicITensor[]
    caps = FermionicITensor[]
    _arrow(flist, s) = _dir(only(filter(f -> s in f.order, flist)), s)
    _ketbra(me) = (only(filter(i -> plev(i) == 0, me.order)), only(filter(i -> plev(i) == 1, me.order)))

    # Loop edges: antiprojector Q_e = 𝟙_e − P_e, bond `sim`'d on the destination side.
    for e in es
        me = message(bpc, e)
        sket, sbra = _ketbra(me)
        sket_s, sbra_s = sim(sket), sim(sbra)
        ask = _arrow(facs[src(e)], sket); adk = _arrow(facs[dst(e)], sket)
        asb = _arrow(facs[src(e)], sbra); adb = _arrow(facs[dst(e)], sbra)
        facs[dst(e)] = [_ft_sim2(f, sket, sket_s, sbra, sbra_s) for f in facs[dst(e)]]
        Ik = _fermionic_bond_identity(me.grading[sket], sket, sket_s, !ask, !adk, Int(adk))
        Ib = _fermionic_bond_identity(me.grading[sbra], sbra, sbra_s, !asb, !adb, Int(adb))
        one_e = Ik * Ib
        mer = message(bpc, reverse(e))                       # original legs, caps the source
        me_s = _ft_sim2(me, sket, sket_s, sbra, sbra_s)       # sim'd legs, caps the destination
        push!(Qs, one_e - mer * me_s)
    end

    # Incident non-loop edges: chords get the rank-1 BP cap, true boundary the cavity message.
    done = Set{NamedEdge}()
    for ein in boundary_edges(bpc, es)
        (ein in done || reverse(ein) in done) && continue
        push!(done, ein)
        u, v = src(ein), dst(ein)                            # v ∈ vs by construction
        if u in vset
            me = message(bpc, ein)
            sket, sbra = _ketbra(me)
            sket_s, sbra_s = sim(sket), sim(sbra)
            facs[v] = [_ft_sim2(f, sket, sket_s, sbra, sbra_s) for f in facs[v]]
            push!(caps, message(bpc, reverse(ein)))           # caps u (original legs)
            push!(caps, _ft_sim2(me, sket, sket_s, sbra, sbra_s))  # caps v (sim'd legs)
        else
            push!(caps, message(bpc, ein))
        end
    end

    ts = FermionicITensor[]
    for v in vs
        append!(ts, facs[v])
    end
    append!(ts, Qs)
    append!(ts, caps)
    seq = contraction_sequence(ts; alg = "optimal")
    cfull_anti = scalar(contract(ts; sequence = seq))
    zprod = prod(vertex_scalar(bpc, v) for v in vs)
    return cfull_anti / zprod
end

#Transform the indices in the given subgraph of the tensornetwork so that antiprojectors can be inserted without duplicate indices appearing.
# Bosonic only: the fermionic loop weight (`weight`) uses `_fermionic_loop_weight` and
# never calls this routine.
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
    if is_fermionic(network(bpc))
        return _fermionic_loop_weight(bpc, eg)
    end
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
