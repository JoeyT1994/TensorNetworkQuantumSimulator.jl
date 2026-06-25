# ===========================================================================
# Triangle-exact cluster-update helpers for the chiral Kagome model, in the
# Kagome -> honeycomb incidence-graph representation.
#
#   * centre  = a NO-MODE virtual hub  (honeycomb vertex / Kagome triangle)
#   * leaf    = a Kagome site          (honeycomb edge), one fermion mode + the
#               virtual bonds to the (one or two) triangle centres it belongs to.
#
# These live OUTSIDE the library on purpose (the library's gate factory only builds
# 1- and 2-site fermionic gates). `include` this file at the top of a driver script.
# The primitives below are exactly the ones validated against ED in the graded
# single-triangle and bowtie prototypes (/tmp/tnqs_graded_triangle.jl,
# /tmp/tnqs_graded_bowtie.jl): clean O(dt^2) inter-triangle Trotter scaling,
# Richardson -> ED to ~1e-9, and lossless (machine-eps) graded star re-splits.
#
# The one fermionic subtlety, handled in `apply_cluster_gate`: a 3-site gate is an
# even operator with its sites ADJACENT in a fixed mode order, so it must be applied
# by first bringing the cluster's three site legs adjacent via a fermionic `permute`
# (which threads the Jordan-Wigner sign through each leaf's EXTERNAL bond to the other
# triangle) and THEN contracting ordinarily. A fermionic `contract` of gate-onto-blob,
# or an ordinary contract without the permute, corrupts the odd-odd (hopping) channel.
# ===========================================================================

using TensorNetworkQuantumSimulator
const TNS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: FermionicITensor, fermionic_exp_gate,
    _fermionic_site_grading, pseudo_sqrt_inv_sqrt, symmetric_svd
using ITensors: ITensors, Index, ITensor, prime, noprime, inds, commoninds, dag
using LinearAlgebra: norm
using Dictionaries: Dictionary, set!

# ---------------------------------------------------------------------------
# 3-site chiral-triangle Hamiltonian / gate
# ---------------------------------------------------------------------------

"""
    triangle_fock_matrix(w_ab, w_bc, w_ca; μ = 0.0) -> 8x8 ComplexF64

Jordan-Wigner Fock matrix (mode order a<b<c) of the 3-mode hopping Hamiltonian
`H = w_ab c†_a c_b + w_bc c†_b c_c + w_ca c†_c c_a + h.c.` along the directed bonds
a->b->c->a. Pass the (generally complex) Peierls amplitude *along each arrow*; the
h.c. partner supplies the reverse bond, so `H` is Hermitian for any complex `w`s.
For the chiral Kagome an up- or down-triangle has all three along-arrow amplitudes
equal to `t·(t1 - i·λ1) = t·e^{-iϕ}` (Tang et al., arXiv:1111.1172).

`μ` adds an on-site chemical potential `-μ(n_a + n_b + n_c)` to `H`. Because every Kagome
site sits in exactly one up-triangle, folding `-μ N` into the up-triangle gates (and only
those) realises the grand-canonical generator `H_hop - μN` with no double counting; with `μ`
inside the single-particle gap this pins the imaginary-time evolution at the target filling
(the tensors track only parity, so truncation would otherwise leak particle number).
"""
function triangle_fock_matrix(w_ab::Number, w_bc::Number, w_ca::Number; μ::Real = 0.0)
    a  = ComplexF64[0 1; 0 0]; Z = ComplexF64[1 0; 0 -1]; I2 = ComplexF64[1 0; 0 1]
    k3(X, Y, W) = kron(kron(X, Y), W)            # mode a = MSB of the kron
    ca = k3(a, I2, I2); cb = k3(Z, a, I2); cc = k3(Z, Z, a)
    H = w_ab * (ca' * cb) + w_bc * (cb' * cc) + w_ca * (cc' * ca)
    H = H + H'
    iszero(μ) || (H -= μ * (ca' * ca + cb' * cb + cc' * cc))   # -μ Σ n_i
    return H
end

# linear index of Fock state (na,nb,nc) in the kron (mode-a MSB) convention
_triL(na, nb, nc) = 1 + 4na + 2nb + nc

"""
    triangle_hamiltonian_ft(sites::NTuple{3,Index}, ws::NTuple{3,Number}; μ = 0.0) -> FermionicITensor

The 3-site triangle Hamiltonian (see [`triangle_fock_matrix`](@ref)) as a parity-even
operator `FermionicITensor` on `sites = (s_a, s_b, s_c)` (legs `[u_a,s_a,u_b,s_b,u_c,s_c]`,
`u = prime(s)`, OUT/IN arrows). `ws = (w_ab, w_bc, w_ca)` are the along-arrow amplitudes; `μ`
adds the on-site chemical potential `-μ Σ n` (see [`triangle_fock_matrix`](@ref)).
"""
function triangle_hamiltonian_ft(sites::NTuple{3, Index}, ws::NTuple{3, <:Number}; μ::Real = 0.0)
    sa, sb, sc = sites
    ua, ub, uc = prime(sa), prime(sb), prime(sc)
    sgr = _fermionic_site_grading(sa)
    grd = Dictionary{Index, Vector{Bool}}()
    for (sx, ux) in ((sa, ua), (sb, ub), (sc, uc))
        set!(grd, sx, sgr); set!(grd, ux, sgr)
    end
    H8 = triangle_fock_matrix(ws...; μ = μ)
    A = Array{ComplexF64}(undef, 2, 2, 2, 2, 2, 2)
    for oa in 1:2, ob in 1:2, oc in 1:2, ia in 1:2, ib in 1:2, ic in 1:2
        A[oa, ob, oc, ia, ib, ic] = H8[_triL(oa - 1, ob - 1, oc - 1), _triL(ia - 1, ib - 1, ic - 1)]
    end
    Ht  = ITensor(A, ua, ub, uc, sa, sb, sc)
    ord = Index[ua, sa, ub, sb, uc, sc]
    return FermionicITensor(Ht, ord, Bool[false, true, false, true, false, true], grd)
end

"""
    triangle_gate(dt, sites::NTuple{3,Index}, ws::NTuple{3,Number}; coeff = -1.0, μ = 0.0) -> FermionicITensor

The EXACT 3-site triangle propagator `exp(coeff·dt·H_△)` built by exponentiating the
full 8×8 Fock matrix via the library's `fermionic_exp_gate`. `coeff = -1` (default) gives
the imaginary-time gate `exp(-dt·H_△)`; pass `coeff = -im` for real time. Applying this
removes the INTRA-triangle Trotter error entirely (only the inter-triangle error remains,
and that is O(dt²) because the BCH leading term is anti-Hermitian). `μ` folds the on-site
chemical potential `-μ Σ n` into `H_△` (see [`triangle_fock_matrix`](@ref)).
"""
function triangle_gate(dt::Number, sites::NTuple{3, Index}, ws::NTuple{3, <:Number}; coeff::Number = -1.0, μ::Real = 0.0)
    H = triangle_hamiltonian_ft(sites, ws; μ = μ)
    sa, sb, sc = sites
    return fermionic_exp_gate(H; outs = Index[prime(sa), prime(sb), prime(sc)],
        ins = Index[sa, sb, sc], dt = dt, coeff = coeff)
end

# ---------------------------------------------------------------------------
# cluster gather / gate-apply / star re-split  (the "three-site apply + leaves")
# ---------------------------------------------------------------------------

"""
    gather_cluster(centre, leaves) -> FermionicITensor

Contract a triangle cluster `centre * leaf_1 * leaf_2 * leaf_3` (fermionic `contract`).
The open legs are the leaves' site indices plus each leaf's EXTERNAL bond(s) to the other
triangle centre(s); the centre-leaf bonds are summed over.
"""
gather_cluster(centre::FermionicITensor, leaves::AbstractVector{<:FermionicITensor}) =
    reduce(*, leaves; init = centre)

"""
    external_legs(leaf, site, centre) -> Vector{Index}

The legs of `leaf` that must be PRESERVED through the re-split: everything except its own
physical `site` index and its bond(s) to `centre` (the triangle being updated). For a bulk
Kagome leaf this is the single bond to its other triangle's centre; for a boundary leaf with
a dangling edge it is that dangling leg (or empty).
"""
function external_legs(leaf::FermionicITensor, site::Index, centre::FermionicITensor)
    to_centre = commoninds(leaf, centre)
    return Index[i for i in inds(leaf) if i != site && !(i in to_centre)]
end

"""
    apply_cluster_gate(blob, gate, sites) -> FermionicITensor

Apply an exact 3-site `gate` to a gathered cluster `blob` on the three physical `sites`.
Brings the site legs adjacent (front) with a fermionic `permute` — threading the JW sign
through any spectator/external bond — then contracts the gate ordinarily and `noprime`s.
Mirrors the 2-site `simple_update` pattern, generalised to three sites.
"""
function apply_cluster_gate(blob::FermionicITensor, gate::FermionicITensor, sites::AbstractVector{<:Index})
    spect = filter(i -> !(i in sites), blob.order)
    P = ITensors.permute(blob, Index[collect(Index, sites); spect])
    T = noprime(gate.tensor * P.tensor)
    return FermionicITensor(T, copy(P.order), copy(P.dirs), P.grading)
end

"""
    star_resplit(blob, sites, exts; cutoff = 0.0, maxdim = nothing) -> (leaves::Vector, centre, Ss::Vector)

Graded STAR re-split of a post-gate cluster `blob` back into `length(sites)` leaf tensors plus a
no-mode centre hub. `sites[k]` is leaf k's physical index; `exts[k]` lists the external legs leaf
k must keep (from [`external_legs`](@ref)). Each leaf is peeled with a fermionic **symmetric**
`svd` on `[sites[k]; exts[k]...]`: `blob ≈ (U√S) ∘ (√S V)`, so the leaf `X = U√S` and the running
remainder `Y = √S V` each carry √S on their shared bond — the Vidal/BP-symmetric gauge, in which
the bond's belief-propagation message is exactly the singular-value matrix `S`. The final
remainder is the centre (carrying the new leaf bonds). `cutoff`/`maxdim` are forwarded to the
fermionic `symmetric_svd` (both parity sectors share one truncation); with `cutoff = 0.0,
maxdim = nothing` the re-split is lossless. Works for any leaf count (e.g. a 2-leaf boundary
cluster, not just a 3-leaf triangle).

Returns `(leaves, centre, Ss)` where `Ss[k]` is the diagonal bond spectrum (a 2-leg
`FermionicITensor` on `[b_k, prime(b_k)]`, `b_k = commonind(leaves[k], centre)`) of leaf k's bond
to the centre. The caller installs `Ss[k]` as the BP message on that bond so the cache stays
consistent when the bond index/dimension changes under truncation (cf. `apply_gate!`).
"""
function star_resplit(blob::FermionicITensor, sites::AbstractVector{<:Index},
        exts::AbstractVector; cutoff = 0.0, maxdim = nothing)
    length(sites) == length(exts) || error("sites and exts must align (one external-leg list per leaf).")
    leaves = FermionicITensor[]
    Ss = FermionicITensor[]
    R = blob
    for k in eachindex(sites)
        X, Y, S, _ = symmetric_svd(R, Index[sites[k]; exts[k]]; cutoff = cutoff, maxdim = maxdim)
        push!(leaves, X)                          # leaf X = U√S carries √S on its centre bond
        push!(Ss, S)                              # bond spectrum (-> BP message on that bond)
        R = Y                                     # remainder Y = √S V; final R = centre hub
    end
    return leaves, R, Ss
end

"""
    cluster_update_triangle(centre, leaves, sites, gate; cutoff = 0.0, maxdim = nothing, envs = nothing) -> (newleaves, newcentre, Ss)

One exact cluster update: gather `centre * leaves`, apply the exact `gate` on `sites`,
normalise, and graded-star-re-split — keeping each leaf's external bond to its OTHER cluster.
`leaves[k]` carries physical index `sites[k]`. The external legs are detected automatically
via [`external_legs`](@ref). `cutoff`/`maxdim` are forwarded to the re-split SVDs.

`envs` (default `nothing`) is the cluster's incoming BP messages — e.g.
`incoming_messages(ψ_bpc, [centre_vert; leaf_verts...])`, one 2-leg `FermionicITensor` per
external bond. When supplied, each external leg is GAUGED by the matrix square root of its
incoming message before the gate is applied and UN-GAUGED (by `dag` of the inverse root)
afterwards — exactly the Vidal-gauge round-trip `simple_update` uses, generalised to a
multi-leaf star. The round-trip `X·dag(X⁻¹) = I` is lossless, so with no truncation the
result is identical to the bare update; with truncation the re-split SVDs now act in the
BP/Vidal gauge and the truncation is (approximately) variational rather than bare-norm. An
external leg with no matching message (a dangling boundary leg) is left ungauged.

Works for any leaf count: a 3-leaf triangle (`gate = triangle_gate`) or a 2-leaf boundary
cluster (`gate = fermionic_hopping_gate`). Returns `(newleaves, newcentre, Ss)`; `Ss[k]` is the
bond spectrum of leaf k's bond to the centre (see [`star_resplit`](@ref)). The caller writes the
tensors back into the network AND installs each `Ss[k]` as the BP message on that bond — without
that, a bond whose index/dimension changed under truncation leaves a stale message in the cache
and the next `update` reads it on the wrong (old) bond, corrupting BP.
"""
function cluster_update_triangle(centre::FermionicITensor, leaves::AbstractVector{<:FermionicITensor},
        sites::AbstractVector{<:Index}, gate::FermionicITensor; cutoff = 0.0, maxdim = nothing,
        envs = nothing)
    exts = [external_legs(leaves[k], sites[k], centre) for k in eachindex(leaves)]
    blob = gather_cluster(centre, leaves)
    blob = apply_cluster_gate(blob, gate, sites)
    blob = blob / norm(blob)

    envs === nothing && return star_resplit(blob, sites, exts; cutoff = cutoff, maxdim = maxdim)

    # Gauge each external leg of the (already-gated) blob by the matrix square root of its
    # incoming BP message, then re-split, then un-gauge. Gauging happens AFTER the gate — its
    # `noprime` would otherwise collapse the gauged leg `prime(b)` back onto `b` — and is
    # equivalent to gauging the leaves first, since the gate touches only the site legs.
    gexts   = Vector{Vector{Index}}(undef, length(exts))
    ungauge = Dictionary{Index, FermionicITensor}()         # gauged ext index -> dag(X⁻¹)
    for k in eachindex(exts)
        newext = Index[]
        for b in exts[k]
            env_b = nothing
            for e in envs
                (b in e.order) && (env_b = e; break)
            end
            if env_b === nothing
                push!(newext, b)                            # dangling leg: no message, keep as-is
                continue
            end
            X, Xinv = pseudo_sqrt_inv_sqrt(env_b)
            blob = X * blob                                 # external index b -> prime(b) (= bo)
            bo = only(Index[i for i in env_b.order if i != b])
            push!(newext, bo)
            set!(ungauge, bo, dag(Xinv))                    # contract on bo to restore b
        end
        gexts[k] = newext
    end

    newleaves, R, Ss = star_resplit(blob, sites, gexts; cutoff = cutoff, maxdim = maxdim)
    for k in eachindex(newleaves), b in gexts[k]
        haskey(ungauge, b) && (newleaves[k] = ungauge[b] * newleaves[k])  # prime(b) -> b
    end
    return newleaves, R, Ss
end

# ---------------------------------------------------------------------------
# coarse-grain: absorb the bond fermions onto one triangle sublattice
# ---------------------------------------------------------------------------

"""
    absorb_bond_fermions(ψ::TensorNetworkState; onto = :up) -> TensorNetworkState
    absorb_bond_fermions(ψ_bpc::BeliefPropagationCache; onto = :up) -> TensorNetworkState

Coarse-grain the Kagome→honeycomb incidence network: contract every bond-fermion leaf `(:s,…)`
into ONE of its two incident triangle centres, then drop the leaf vertices. `onto = :up` absorbs
each leaf into its up-triangle centre `(:up,…)`; `onto = :dn` absorbs into the down layer
(`(:dn,…)` triangles and `(:dh,…)` dangling hubs). Each leaf's surviving bond to its *other*
centre becomes a direct centre↔centre bond, so the result is the honeycomb lattice of triangle
centres with the absorbed fermion modes now carried by the chosen sublattice (the other sublattice
stays no-mode). The fermionic `*` is the validated locally-ordered contraction, so this is a pure
regrouping — the represented state is unchanged.

`onto = :up` is always well defined (every Kagome site lies in exactly one up-triangle). For
`onto = :dn`, a boundary leaf can touch two dangling hubs (and no down-triangle); that leaf is
ambiguous and raises an error. Returns a fresh `TensorNetworkState` (its graph is re-inferred from
the surviving shared bonds); wrap it in a new `BeliefPropagationCache` if you need BP.
"""
function absorb_bond_fermions(ψ::TNS.TensorNetworkState; onto::Symbol = :up)
    onto in (:up, :dn) || throw(ArgumentError("`onto` must be :up or :dn (got :$onto)."))
    target_tags = onto === :up ? (:up,) : (:dn, :dh)        # the down layer includes dangling hubs
    g     = TNS.graph(ψ)
    sinds = TNS.siteinds(ψ)
    isleaf(v)   = first(v) === :s
    istarget(v) = first(v) in target_tags

    verts   = collect(TNS.vertices(g))
    V       = eltype(verts)
    centres = [v for v in verts if !isleaf(v)]
    leaves  = [v for v in verts if  isleaf(v)]

    # Assign each leaf to its unique incident target-layer centre.
    absorb = Dictionary(centres, [V[] for _ in centres])
    for lf in leaves
        tcs = [c for c in TNS.neighbors(g, lf) if istarget(c)]
        length(tcs) == 1 || throw(ArgumentError(
            "leaf $lf has $(length(tcs)) incident $(onto)-layer centre(s); cannot absorb it " *
            "unambiguously (boundary dangling case). `onto = :up` is always unique."))
        push!(absorb[only(tcs)], lf)
    end

    # Contract leaves into their target centres and move the physical site indices onto the centre.
    # Non-target centres pass through untouched (their leaf bonds are now shared with the centres
    # that absorbed those leaves, so the re-inferred graph wires up the honeycomb automatically).
    new_tensors = Dictionary{V, FermionicITensor}()
    new_sinds   = Dictionary{V, Vector{<:Index}}()
    for c in centres
        T = ψ[c]
        for lf in absorb[c]
            T = T * ψ[lf]                                   # fermionic contract kills the centre↔leaf bond
        end
        set!(new_tensors, c, T)
        set!(new_sinds, c, Index[i for lf in absorb[c] for i in sinds[lf]])
    end

    return TNS.TensorNetworkState(TNS.TensorNetwork(new_tensors), new_sinds)
end

absorb_bond_fermions(ψ_bpc::TNS.BeliefPropagationCache; kwargs...) =
    absorb_bond_fermions(TNS.network(ψ_bpc); kwargs...)
