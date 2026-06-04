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
# composing tensors), and applied to a state by closing one site bond at a time (the
# `fermionic_norm_factors` pattern), never as a single double-bond blob.

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

# Locally-ordered two-site product c^{name_i}_i c^{name_j}_j (each `name` a single odd
# creation/annihilation operator) assembled from the validated `odd_op_tensor` pair
# joined over the operator-string dummy bond. Returns the array in canonical leg order
# [prime(s_i), s_i, prime(s_j), s_j].
function _two_site_odd_pair_array(s_i::Index, name_i::String, s_j::Index, name_j::String, sgr_i, sgr_j)
    d = Index(1, "Fermion,OpString")
    ti = odd_op_tensor(s_i, name_i, d, sgr_i)
    tj = odd_op_tensor(s_j, name_j, d, sgr_j)
    H = contract(ti, tj)
    return ITensors.array(H.tensor, prime(s_i), s_i, prime(s_j), s_j)
end

"""
    fermionic_hopping_hamiltonian(s_i::Index, s_j::Index) -> FermionicITensor

The nearest-neighbour hopping Hamiltonian `H = Σ_σ (c†_{iσ} c_{jσ} + c†_{jσ} c_{iσ})`
as a parity-even operator `FermionicITensor` on the two site indices (legs
`[prime(s_i), s_i, prime(s_j), s_j]`). Spinless (dimension-2) sites carry a single
mode; spinful (dimension-4) sites sum over the up and down modes. Assembled from the
validated odd-operator primitives.
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
        A .+= c .* _two_site_odd_pair_array(s_i, ni, s_j, nj, sgr_i, sgr_j)
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
