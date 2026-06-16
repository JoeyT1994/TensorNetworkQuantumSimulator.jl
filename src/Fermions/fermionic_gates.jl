# Pre-built fermionic gate constructors for Trotterised dynamics.
#
# A gate is a parity-even operator tensor in the locally-ordered formalism
# (arXiv:2410.02215). Its legs are the usual operator legs: for each site a primed
# OUT leg `u = prime(s)` and an unprimed IN leg `s`, with arrows `out` (false) on the
# `u`s and `in` (true) on the `s`s — exactly the layout of `even_op_tensor`/`odd_op_tensor`.
#
# Construction principle (derived, not fitted):
#   * The correct locally-ordered single-tensor array of ANY physical operator equals
#     that operator's matrix elements ⟨out|O|in⟩ in a fixed mode ordering. This is true
#     by construction here because every Hamiltonian below is assembled out of the
#     already-validated single-site primitives (`even_op_tensor`, and `odd_op_tensor`
#     pairs joined over the operator-string dummy bond by `contract`), whose arrays were
#     checked to reproduce the Fock matrix elements.
#   * Physical operator multiplication is ordinary matrix multiplication in that basis
#     (e.g. for hopping, H² = n_i + n_j − 2 n_i n_j, whose matrix is the plain matmul
#     H·H). Therefore exp(coeff·dt·H) is the ordinary dense matrix exponential of the
#     Hamiltonian's Fock matrix, re-wrapped with the Hamiltonian's own leg metadata.
#
# NOTE: fermionic `contract` of two multi-site OPERATOR tensors is *network*
# contraction, not operator composition — closing several operator legs at once injects
# extra supertrace signs. So gates must be exponentiated as matrices here (not by
# composing tensors).
#
# APPLYING a gate to a state: the gate's dense array is an even operator with its sites
# ADJACENT in a fixed mode order, so it acts as `o ⊗ I` on every other leg. Apply it by
# (1) bringing the state's physical legs for the gate's sites adjacent via a fermionic
# `permute` (this threads the correct Koszul sign through any leg — e.g. a QR/virtual
# bond — sitting between them), then (2) contracting the gate onto those adjacent legs
# with an ORDINARY (non-fermionic) contraction. See `simple_update`. A fermionic
# `contract` blob (`o * ψ`) instead injects spurious supertrace signs and is wrong;
# ordinary contraction WITHOUT the adjacency permute misses the reorder sign. Either
# error corrupts the gate's odd-odd (e.g. hopping) channel whenever a fermionic bond
# lies between the two sites.

using ITensors: ITensors, ITensor, Index, prime, array, dim
using Dictionaries: Dictionary
using LinearAlgebra: LinearAlgebra

# --- low-level builders -----------------------------------------------------

# Single-site even operator tensor from a raw matrix `M[out, in]` (legs [u, s]).
function _onsite_even_ft(s::Index, M::AbstractMatrix, sgr::Vector{Bool})
    u = prime(s)
    gr = Dictionary{Index, Vector{Bool}}(Index[u, s], Vector{Bool}[sgr, sgr])
    return FermionicITensor(ITensor(M, u, s), Index[u, s], Bool[false, true], gr)
end

# Plain (NON-fermionic) operator matrix of `H`: rows = OUT legs `outs`, columns = IN
# legs `ins`, each combined in the given order. This just *reads off* the tensor
# components ⟨out|H|in⟩ — no Koszul sign — which is exactly the basis in which physical
# operator multiplication is ordinary matmul.
function _operator_matrix(H::FermionicITensor, outs::Vector{<:Index}, ins::Vector{<:Index})
    A = ITensors.array(H.tensor, Index[outs; ins]...)
    nout = prod(Int[dim(i) for i in outs]; init = 1)
    nin = prod(Int[dim(i) for i in ins]; init = 1)
    return reshape(Array(A), nout, nin)
end

"""
    fermionic_exp_gate(H::FermionicITensor; outs, ins, dt, coeff = -im)

Exponentiate a parity-even Hamiltonian operator tensor `H` into the gate
`exp(coeff · dt · H)` as a `FermionicITensor` carrying the same leg order, arrows and
grading as `H`. `outs`/`ins` list `H`'s OUT (primed) and IN (unprimed) legs; they must
be equal-dimensional and pair up so the operator matrix is square.

`coeff` defaults to `-im`, giving the real-time propagator `exp(-i dt H)`; pass
`coeff = -1` for imaginary-time `exp(-dt H)`.
"""
function fermionic_exp_gate(H::FermionicITensor; outs::Vector{<:Index}, ins::Vector{<:Index}, dt::Number, coeff::Number = -im)
    M = _operator_matrix(H, outs, ins)
    U = exp((coeff * dt) * M)
    dims = (Int[dim(i) for i in outs]..., Int[dim(i) for i in ins]...)
    T = ITensor(reshape(Array(U), dims...), Index[outs; ins]...)
    return FermionicITensor(T, copy(H.order), copy(H.dirs), H.grading)
end

# --- Hamiltonian-term assembly ----------------------------------------------

# Two-site fermion-bilinear `c^{name_i}_i c^{name_j}_j` (each `name` a single odd
# creation/annihilation operator) as its Jordan-Wigner Fock matrix on the ADJACENT pair,
# with site `i` ordered before site `j`. Returned in canonical leg order
# [prime(s_i), s_i, prime(s_j), s_j] (i.e. components A[out_i, in_i, out_j, in_j]).
#
# Derivation (no ED fitting): in the mode order (site-i modes, then site-j modes) the
# product is `(M_i ⊗ 𝟙_j)(P_i ⊗ M_j) = (M_i P_i) ⊗ M_j`, where `M_i`/`M_j` are the
# single-site Fock matrices (`fermion_op_matrix`, which already bake in each site's INTRA-site
# spin string) and `P_i = diag((−1)^{p})` is site i's parity. The factor `P_i` is the
# Jordan-Wigner string that the operator on site j drags across all of site i's modes — for a
# spinful site this is the FULL on-site parity `(−1)^{n↑+n↓}`, so a hop onto/off a site that
# already holds the opposite spin (a double-occupancy transition) correctly picks up the
# spectator-spin sign. (Closing the two odd legs through the operator-string dummy bond instead
# threads only a single-mode string and drops exactly those doubly-occupied signs.)
#
# This is the matrix the gate's `simple_update` application expects: it carries the i→j string
# for the two sites held ADJACENT, while the fermionic `permute` in `simple_update` supplies the
# remaining string for any virtual/spectator legs physically lying between them.
function _two_site_hop_array(s_i::Index, name_i::String, s_j::Index, name_j::String, sgr_i)
    Mi = fermion_op_matrix(name_i, s_i)            # [out_i, in_i]
    Mj = fermion_op_matrix(name_j, s_j)            # [out_j, in_j]
    MiP = Mi * LinearAlgebra.Diagonal(ComplexF64[b ? -1 : 1 for b in sgr_i])   # M_i · P_i
    di, dj = dim(s_i), dim(s_j)
    A = Array{ComplexF64}(undef, di, di, dj, dj)
    for oi in 1:di, ii in 1:di, oj in 1:dj, ij in 1:dj
        A[oi, ii, oj, ij] = MiP[oi, ii] * Mj[oj, ij]
    end
    return A
end

"""
    fermionic_hopping_hamiltonian(s_i::Index, s_j::Index) -> FermionicITensor

The nearest-neighbour hopping Hamiltonian `H = Σ_σ (c†_{iσ} c_{jσ} + c†_{jσ} c_{iσ})`
as a parity-even operator `FermionicITensor` on the two site indices (legs
`[prime(s_i), s_i, prime(s_j), s_j]`). Spinless (dimension-2) sites carry a single
mode; spinful (dimension-4) sites sum over the up and down modes. Each bilinear is built as
its adjacent-pair Jordan-Wigner Fock matrix (see `_two_site_hop_array`), so the on-site spin
string is carried correctly through double-occupancy transitions.
"""
function fermionic_hopping_hamiltonian(s_i::Index, s_j::Index)
    dim(s_i) == dim(s_j) || error("Hopping requires two sites of equal local dimension.")
    sgr_i = _fermionic_site_grading(s_i)
    sgr_j = _fermionic_site_grading(s_j)
    # Each entry is (coeff, name_i, name_j) for the i-first physical product
    # c^{name_i}_i c^{name_j}_j. The h.c. partner c†_{jσ} c_{iσ} is built as the
    # i-first product c_{iσ} c†_{jσ} = −c†_{jσ} c_{iσ}, hence coefficient −1.
    terms = if dim(s_i) == 2
        ((1, "Cdag", "C"), (-1, "C", "Cdag"))
    elseif dim(s_i) == 4
        ((1, "Cupdag", "Cup"), (-1, "Cup", "Cupdag"),
            (1, "Cdndag", "Cdn"), (-1, "Cdn", "Cdndag"))
    else
        error("Hopping supports spinless (dim 2) or spinful (dim 4) sites only.")
    end
    A = zeros(ComplexF64, dim(s_i), dim(s_i), dim(s_j), dim(s_j))
    for (c, ni, nj) in terms
        A .+= c .* _two_site_hop_array(s_i, ni, s_j, nj, sgr_i)
    end
    order = Index[prime(s_i), s_i, prime(s_j), s_j]
    gr = Dictionary{Index, Vector{Bool}}(order, Vector{Bool}[sgr_i, sgr_i, sgr_j, sgr_j])
    return FermionicITensor(ITensor(A, order...), order, Bool[false, true, false, true], gr)
end

"""
    fermionic_number_hamiltonian(s::Index) -> FermionicITensor

The on-site total number operator `N` (spinless `n`; spinful `n↑ + n↓`) as a
parity-even operator `FermionicITensor` on site `s` (legs `[prime(s), s]`).
"""
function fermionic_number_hamiltonian(s::Index)
    sgr = _fermionic_site_grading(s)
    M = if dim(s) == 2
        fermion_op_matrix("N", s)
    elseif dim(s) == 4
        fermion_op_matrix("Nup", s) + fermion_op_matrix("Ndn", s)
    else
        error("Number operator supports spinless (dim 2) or spinful (dim 4) sites only.")
    end
    return _onsite_even_ft(s, M, sgr)
end

"""
    fermionic_sz(s::Index) -> FermionicITensor

The on-site magnetisation as a
parity-even operator `FermionicITensor` on site `s` (legs `[prime(s), s]`).
"""
function fermionic_sz(s::Index)
    sgr = _fermionic_site_grading(s)
    M = if dim(s) == 4
        fermion_op_matrix("Sz", s)
    else
        error("Number operator supports spinful (dim 4) sites only.")
    end
    return _onsite_even_ft(s, M, sgr)
end

"""
    fermionic_interaction_hamiltonian(s::Index) -> FermionicITensor

The on-site Hubbard interaction operator `n↑ n↓` as a parity-even operator
`FermionicITensor` on a spinful (dimension-4) site `s` (legs `[prime(s), s]`); it is the
projector onto the doubly-occupied state. Defined for spinful sites only — `n↑ n↓` has no
meaning on a spinless (dimension-2) site.
"""
function fermionic_interaction_hamiltonian(s::Index)
    dim(s) == 4 || error("Interaction n↑n↓ requires a spinful (dim 4) site.")
    sgr = _fermionic_site_grading(s)
    M = fermion_op_matrix("NupNdn", s)
    return _onsite_even_ft(s, M, sgr)
end

# --- public gate constructors -----------------------------------------------

"""
    fermionic_hopping_gate(dt, s_i::Index, s_j::Index; coeff = -im) -> FermionicITensor

The two-site hopping propagator `exp(coeff · dt · H_hop)` where
`H_hop = Σ_σ (c†_{iσ} c_{jσ} + h.c.)`. With the default `coeff = -im` this is the
real-time Trotter gate `exp(-i dt H_hop)`.
"""
function fermionic_hopping_gate(dt::Number, s_i::Index, s_j::Index; coeff::Number = -im)
    H = fermionic_hopping_hamiltonian(s_i, s_j)
    return fermionic_exp_gate(H; outs = Index[prime(s_i), prime(s_j)], ins = Index[s_i, s_j], dt, coeff)
end

"""
    fermionic_number_gate(dt, s::Index; coeff = -im) -> FermionicITensor

The on-site number propagator `exp(coeff · dt · N)` (`N = n` spinless, `n↑ + n↓`
spinful). With the default `coeff = -im` this is `exp(-i dt N)`.
"""
function fermionic_number_gate(dt::Number, s::Index; coeff::Number = -im)
    H = fermionic_number_hamiltonian(s)
    return fermionic_exp_gate(H; outs = Index[prime(s)], ins = Index[s], dt, coeff)
end

"""
    fermionic_sz_gate(dt, s::Index; coeff = -im) -> FermionicITensor

The on-site magnetisation propagator `exp(coeff · dt · Sz)` .
With the default `coeff = -im` this is `exp(-i dt N)`.
"""
function fermionic_sz_gate(dt::Number, s::Index; coeff::Number = -im)
    H = fermionic_sz(s)
    return fermionic_exp_gate(H; outs = Index[prime(s)], ins = Index[s], dt, coeff)
end


"""
    fermionic_interaction_gate(dt, s::Index; coeff = -im) -> FermionicITensor

The on-site Hubbard interaction propagator `exp(coeff · dt · n↑ n↓)` on a spinful
(dimension-4) site. With the default `coeff = -im` this is the real-time Trotter gate
`exp(-i dt n↑ n↓)`; pass `coeff = -1` for the imaginary-time `exp(-dt n↑ n↓)`. Because
`n↑ n↓` is the double-occupancy projector, the gate is diagonal and only the doubly-occupied
state picks up the phase/weight `exp(coeff · dt)`.
"""
function fermionic_interaction_gate(dt::Number, s::Index; coeff::Number = -im)
    H = fermionic_interaction_hamiltonian(s)
    return fermionic_exp_gate(H; outs = Index[prime(s)], ins = Index[s], dt, coeff)
end

# --- single-site generating-function gate -----------------------------------

"""
    fermionic_onsite_exp_gate(op_string::String, s::Index, α::Number) -> FermionicITensor

Single-site generating-function gate `exp(α · Ô)` for a **Hermitian** on-site fermionic
observable `Ô` named by `op_string` (e.g. `"N"`, `"Nup"`, `"Ndn"`, `"NupNdn"`), as a
parity-even `FermionicITensor` on legs `[prime(s), s]`. This is the fermionic analogue of
`ITensors.exp(α * ITensors.op(op_string, s); ishermitian = true)` and is what the
loop-corrected free-energy observable estimator (`expect(...; alg = "loopcorrections")`)
absorbs into the ket.

Unlike `fermionic_number_gate` (which bakes in `coeff = -im` for a real-time *rotation*),
this uses `coeff = 1`, i.e. it builds `exp(α·Ô)` literally — `α` is the (generally real)
generating-function shift, not a rotation angle. `op_string` must name a Hermitian on-site
operator; odd operators such as `"C"`/`"Cdag"` are rejected (they are neither Hermitian nor
valid single-site observables).
"""
function fermionic_onsite_exp_gate(op_string::String, s::Index, α::Number)
    sgr = _fermionic_site_grading(s)
    M = fermion_op_matrix(op_string, s)
    LinearAlgebra.ishermitian(M) || throw(ArgumentError(
        "fermionic_onsite_exp_gate requires a Hermitian on-site observable; " *
        "\"$op_string\" is not Hermitian. Supported Hermitian on-site observables: " *
        "\"N\" (spinless), \"Nup\"/\"Ndn\"/\"NupNdn\" (spinful)."))
    H = _onsite_even_ft(s, M, sgr)
    return fermionic_exp_gate(H; outs = Index[prime(s)], ins = Index[s], dt = α, coeff = 1)
end

# --- circuit-tuple → fermionic gate -----------------------------------------

"""
    tofermionicitensor(name::String, θ, s_inds::Vector{<:Index}) -> FermionicITensor

Build the rotated fermionic Trotter gate `exp(-i · θ · H)` named by `name`, acting on the
fermionic site indices `s_inds`. This is the fermionic analogue of the spin path in
`totensor`: it turns a circuit tuple `(name, vertices, θ)` into the operator
`FermionicITensor` that `apply_gates` / `simple_update` consume.

The `R` prefix denotes a rotation (cf. the spin rotations `Rxx(θ) = exp(-i θ XX)`): `θ` is
the angle and the gate bakes in the `-i`, so the generator's default coefficient is `-im`.
For imaginary-time evolution `exp(-τ H)` pass an imaginary angle `θ = -im · τ`.

Supported gate names (with the required site count):
- `"RHop"` (2 sites): rotated hopping, `exp(-i·θ·H)`, `H = Σ_σ (c†_{iσ} c_{jσ} + h.c.)`
- `"RInt"` (1 spinful site): rotated interaction, `exp(-0.5i·θ·H)`, `H = n↑ n↓`. The angle
  `θ` multiplies the `-0.5·im` half-step exponent (so `θ = U·dt` for a Hubbard half-step).
- `"RN"` (1 site): rotated total number, `exp(-i·θ·H)`, `H = N` (spinless `n`; spinful `n↑ + n↓`)
"""
function tofermionicitensor(name::String, θ, s_inds::Vector{<:Index})
    if name == "RHop"
        length(s_inds) == 2 || throw(ArgumentError(
            "Fermionic gate \"$name\" acts on 2 sites, got $(length(s_inds))."))
        return fermionic_hopping_gate(θ, s_inds[1], s_inds[2])
    elseif name == "RInt"
        length(s_inds) == 1 || throw(ArgumentError(
            "Fermionic gate \"$name\" acts on 1 site, got $(length(s_inds))."))
        return fermionic_interaction_gate(θ, only(s_inds))
    elseif name == "RN"
        length(s_inds) == 1 || throw(ArgumentError(
            "Fermionic gate \"$name\" acts on 1 site, got $(length(s_inds))."))
        return fermionic_number_gate(θ, only(s_inds))
    elseif name == "RSz"
        length(s_inds) == 1 || throw(ArgumentError(
            "Fermionic gate \"$name\" acts on 1 site, got $(length(s_inds))."))
        return fermionic_sz_gate(θ, only(s_inds))
    end
    throw(ArgumentError(
        "Unknown fermionic gate \"$name\". Supported: \"RHop\", \"RInt\", \"RN\", \"RSz\"."))
end
