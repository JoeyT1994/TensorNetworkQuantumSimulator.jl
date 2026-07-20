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
# `leafless_edge_induced_subgraphs` (so both capture the bridge "dumbbell"
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
    egs = leafless_edge_induced_subgraphs(g, max_configuration_size)
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
# Differentiable `exp(scale·M)` for a FIXED Hermitian operator matrix `M`: diagonalize once
# in floats (LAPACK acts only on the constant `M`) and exponentiate the eigenvalues with the
# possibly-`Dual` scalar `scale`. This sidesteps the `eigen`/Padé matrix-`exp` paths that
# cannot accept Dual-valued matrices, while staying exact for an ordinary float `scale`.
function _hermitian_exp(M::AbstractMatrix, scale::Number)
    E = eigen(Hermitian(Matrix{ComplexF64}(M)))
    return E.vectors * Diagonal(exp.(scale .* E.values)) * E.vectors'
end

# Dual-aware `exp(α·Ô)` single-site gate on legs `[s', s]` (bosonic), reproducing
# `ITensors.exp(α·op(op_string,s); ishermitian=true)` for ordinary α but letting α carry a
# `Dual` seed for forward-mode AD of the observable.
function _onsite_exp_gate_ad(op_string::String, s::Index, α::Number)
    o = ITensors.op(op_string, s)
    return ITensor(_hermitian_exp(ITensors.array(o, prime(s), s), α), prime(s), s)
end

# Fermionic analogue of `_onsite_exp_gate_ad`: same construction as `fermionic_onsite_exp_gate`
# but the exponential goes through `_hermitian_exp` so α may be a `Dual`.
function _onsite_exp_gate_ad_fermionic(op_string::String, s::Index, α::Number)
    sgr = _fermionic_site_grading(s)
    M = fermion_op_matrix(op_string, s)
    LinearAlgebra.ishermitian(M) || throw(ArgumentError(
        "loopcorrections autodiff requires a Hermitian on-site observable; \"$op_string\" is not Hermitian."))
    H = _onsite_even_ft(s, M, sgr)
    outs, ins = Index[prime(s)], Index[s]
    U = _hermitian_exp(_operator_matrix(H, outs, ins), α)
    dims = (Int[dim(i) for i in outs]..., Int[dim(i) for i in ins]...)
    T = ITensor(reshape(Array(U), dims...), Index[outs; ins]...)
    return FermionicITensor(T, copy(H.order), copy(H.dirs), H.grading)
end

# Float/complex-float α keeps the EXACT existing builders (finite-difference path unchanged);
# any other Number (`Dual`, `Complex{Dual}`) routes to the analytic eigen-based builders so a
# Dual seed propagates. `Dual <: Real` but not `<: AbstractFloat`, so this dispatch is clean.
_gf_onsite_gate(op_string, s, α::Union{AbstractFloat, Complex{<:AbstractFloat}}, isfermionic::Bool) =
    isfermionic ? fermionic_onsite_exp_gate(op_string, s, α) :
                  ITensors.exp(α * ITensors.op(op_string, s); ishermitian = true)
_gf_onsite_gate(op_string, s, α::Number, isfermionic::Bool) =
    isfermionic ? _onsite_exp_gate_ad_fermionic(op_string, s, α) :
                  _onsite_exp_gate_ad(op_string, s, α)

# Vector form: apply the generating-function gate `e^{α Ôᵢ}` to each `(op_stringsᵢ, vsᵢ)` pair
# (one un-normalized single-site gate per vertex) and return the loop-corrected free energy of the
# resulting squared-norm network. The gates commute (distinct vertices), so the bra/ket sandwich is
# `⟨ψ| ∏ᵢ e^{2α Ôᵢ} |ψ⟩ = ⟨ψ| e^{2α Σᵢ Ôᵢ} |ψ⟩`: with a shared scalar α this is the generating
# function of the EXTENSIVE sum `Σᵢ Ôᵢ`, whose ½ ∂_α F|₀ is `Σᵢ ⟨Ôᵢ⟩` in a single BP re-solve.
function gated_lc_free_energy(
        ψ_bpc::BeliefPropagationCache, op_strings::AbstractVector, vs::AbstractVector, α,
        max_configuration_size::Integer; cache_update_kwargs = default_bp_update_kwargs(ψ_bpc),
    )
    length(op_strings) == length(vs) || throw(ArgumentError(
        "gated_lc_free_energy: need one operator string per vertex (got $(length(op_strings)) ops, $(length(vs)) vertices)."))
    allunique(vs) || throw(ArgumentError(
        "gated_lc_free_energy: vertices must be distinct so the single-site gates commute; got repeated vertices in $vs."))
    ψ = network(ψ_bpc)
    for (op_string, v) in zip(op_strings, vs)
        s = only(siteinds(ψ)[v])
        # Map the single-site Hermitian observable to its generating-function gate `e^{α Ô}`.
        # Fermionic sites need the locally-ordered `FermionicITensor` exponential (built from the
        # on-site Fock matrix) rather than the spin `ITensors.op`/`ITensors.exp` path. The builder
        # is dispatched on `α`'s type so a `Dual`-seeded α takes a matrix-`exp`-free analytic path.
        G = _gf_onsite_gate(op_string, s, α, has_fermionic_tag(s))
        # `normalize_tensors = false`: the un-normalized gated tensor is exactly what makes the
        # squared norm equal the partition function ⟨ψ|e^{2α Ô}|ψ⟩ we want to differentiate.
        ψ_bpc, _ = apply_gate(G, ψ_bpc; v⃗ = [v], apply_kwargs = (; normalize_tensors = false))
    end
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)
    return loopcorrected_free_energy(ψ_bpc, max_configuration_size)
end

# Single-site convenience wrapper (unchanged call sites in `expect`).
function gated_lc_free_energy(
        ψ_bpc::BeliefPropagationCache, op_string::String, v, α, max_configuration_size::Integer;
        cache_update_kwargs,
    )
    return gated_lc_free_energy(ψ_bpc, [op_string], [v], α, max_configuration_size; cache_update_kwargs)
end

#Transform the indices in the given subgraph of the tensornetwork so that antiprojectors can be inserted without duplicate indices appearing
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
    seq = any(hasqns.(ts)) ? contraction_sequence(ts; alg = "optimal") : contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod())
    return scalar(contract(ts; sequence = seq))
end

#Vectorized version of weight
function weights(bpc::BeliefPropagationCache, egs)
    return [weight(bpc, eg) for eg in egs]
end
