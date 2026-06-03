# Locally-ordered fermionic tensor algebra.
#
# This implements the "locally ordered" formalism of Gao, Zhai, Gray et al.,
# "Fermionic tensor network contraction for arbitrary geometries", arXiv:2410.02215
# (the quimb default). A fermionic tensor is a dense ITensor plus:
#   * an explicit fermionic leg ORDER (the order the legs appear in `A_{ijkl}`),
#   * an ARROW per leg (`true` = in/−, `false` = out/+); a shared bond's arrow
#     points from its out endpoint to its in endpoint, and
#   * a Z2 GRADING per leg (the per-component parity bits).
#
# Reordering two legs costs `(−1)^{p_i p_j}` (Eq. 6). A pairwise contraction is a
# plain BLAS contraction once the contracted legs are made adjacent (Eq. 8); the
# arrow tells us whether the chosen operand order agrees with the bond, and if not
# we insert the diagonal bond-parity tensor `g = diag((−1)^{p})` (Eq. 9/13). With
# arrows handled per-bond, the binary contraction is order-independent for
# parity-even tensors, so arbitrary contraction sequences are safe.
#
# NOTE: global phases from contracting ODD-parity tensors past each other (the
# `(−1)^{p_A p_B}` factor in Eq. 13, handled in the paper via a dummy odd index)
# are NOT yet implemented here — all state tensors are parity-even. This is the
# next piece for the doubled-network/observable wiring.

using Dictionaries: Dictionary
using ITensors: ITensors, ITensor, Index, dim, inds, dag, commoninds

"""
    FermionicTensor

A dense ITensor together with the metadata for locally-ordered fermionic
operations: the fermionic leg `order`, the per-leg arrow `dirs` (`true` = in/−,
`false` = out/+), and a reference to the global Z2 `grading` (per-component
parity bits of each `Index`). `order` and `dirs` are parallel vectors.
"""
struct FermionicTensor
    tensor::ITensor
    order::Vector{Index}
    dirs::Vector{Bool}
    grading::Dictionary{Index, Vector{Bool}}
end

# Direction (arrow) of leg `i` in `ft`: true = in/−, false = out/+.
_dir(ft::FermionicTensor, i::Index) = ft.dirs[findfirst(==(i), ft.order)]

# Diagonal Koszul-sign array (in `from` layout) for reordering legs `from -> to`.
# For component I the sign is (−1)^{Σ p_a p_b} over leg pairs (a,b) whose relative
# order is inverted between `from` and `to`. Depends only on parity bits, never on
# the numerical positions, so it is a pure broadcast over components.
function _reorder_sign(gr::Dictionary, from::Vector{<:Index}, to::Vector{<:Index})
    n = length(from)
    topos = Dict(ind => p for (p, ind) in enumerate(to))
    invpairs = Tuple{Int, Int}[]
    for a in 1:n, b in (a + 1):n
        topos[from[a]] > topos[from[b]] && push!(invpairs, (a, b))
    end
    bits = [gr[from[k]] for k in 1:n]
    dims = ntuple(k -> length(bits[k]), n)
    sgn = ones(Int8, dims...)
    isempty(invpairs) && return sgn
    for I in CartesianIndices(dims)
        s = false
        for (a, b) in invpairs
            s ⊻= (bits[a][I[a]] & bits[b][I[b]])
        end
        s && (sgn[I] = -one(Int8))
    end
    return sgn
end

# Apply the fermionic transposition sign for `from -> to` to a raw ITensor. The
# returned ITensor carries the SAME index objects (ITensors are storage-order
# agnostic); only the component signs change. The caller updates its `order`.
function _apply_reorder_sign(gr::Dictionary, T::ITensor, from::Vector{<:Index}, to::Vector{<:Index})
    from == to && return T
    sgn = _reorder_sign(gr, from, to)
    arr = ITensors.array(T, from...) .* sgn
    return ITensor(arr, from...)
end

# Multiply a tensor by the diagonal bond-parity operator g = diag((−1)^{p}) on
# leg `k` (the even-parity-tensor form of Eq. 13).
function _apply_parity(gr::Dictionary, T::ITensor, k::Index)
    is = collect(inds(T))
    pos = findfirst(==(k), is)
    arr = ITensors.array(T, is...)
    s = Int8[b ? -1 : 1 for b in gr[k]]
    shape = ntuple(d -> d == pos ? length(s) : 1, ndims(arr))
    return ITensor(arr .* reshape(s, shape), is...)
end

"""
    fermionic_transpose(ft::FermionicTensor, neworder::Vector{<:Index})

Reorder the legs of `ft` to `neworder` (a permutation of `ft.order`), applying the
fermionic transposition sign `(−1)^{p_i p_j}` for each swapped odd pair.
"""
function fermionic_transpose(ft::FermionicTensor, neworder::Vector{<:Index})
    T = _apply_reorder_sign(ft.grading, ft.tensor, ft.order, neworder)
    perm = [findfirst(==(i), ft.order) for i in neworder]
    return FermionicTensor(T, copy(neworder), ft.dirs[perm], ft.grading)
end

"""
    fermionic_dag(ft::FermionicTensor)

Hermitian conjugate in the fermionic sense: conjugate the data, reverse the leg
order (with the accompanying reversal sign), and flip every arrow (in ↔ out). This
is the per-tensor operation behind taking ⟨Ψ| from |Ψ⟩ (Fig. 4: all bra arrows are
reversed relative to the ket).
"""
function fermionic_dag(ft::FermionicTensor)
    rev = reverse(ft.order)
    # Eq.33/108: conjugation maps each space to its dual and REVERSES the leg
    # order. Reversing the labels relabels which dual sits where; it is NOT a
    # permutation of stored components, so there is NO extra Koszul sign here.
    return FermionicTensor(dag(ft.tensor), rev, .!reverse(ft.dirs), ft.grading)
end

"""
    fermionic_contract(A::FermionicTensor, B::FermionicTensor)

Contract `A` and `B` over their shared legs in the locally-ordered formalism. The
result's leg order is `[open legs of A; open legs of B]` (A's legs first). The bond
parity tensor `g` is inserted on every shared bond whose arrow disagrees with this
operand order, so for parity-even tensors the result is independent of which
operand is passed first (the swap rule, Eq. 9).

When more than one bond is contracted at once, the contracted ("fused") modes are
placed in order `Kc` on A but REVERSED on B (arXiv:2410.02215 §IV A, point 3:
`Σ A_{m(ijk)} B_{(ijk)n} = Σ A_{m(ijk)} B̃_{(kji)n}`). Reversing the fused block on
one side reproduces the result of contracting the bonds one at a time, and is what
makes the contraction associative on loops. The reversal sign is accumulated
automatically by `_apply_reorder_sign`; for a single shared bond it is a no-op.
"""
function fermionic_contract(A::FermionicTensor, B::FermionicTensor)
    gr = A.grading
    K = commoninds(A.tensor, B.tensor)
    Kset = Set(K)
    A_open = filter(!in(Kset), A.order)
    B_open = filter(!in(Kset), B.order)
    Kc = filter(in(Kset), A.order)                 # canonical contracted order = A's order

    A_to = Index[A_open; Kc]
    B_to = Index[reverse(Kc); B_open]               # B's contracted block REVERSED (nesting sign)
    TA = _apply_reorder_sign(gr, A.tensor, A.order, A_to)
    TB = _apply_reorder_sign(gr, B.tensor, B.order, B_to)

    # Supertrace (bond-parity g) per shared bond. After the reorder above the
    # contracted pair is linearized as (A's leg)(B's leg). The contraction map C adds
    # a supertrace (−1)^{p} exactly for a KET-BRA pair (SciPost "Fermionic tensor
    # networks" Eq. 107): A's leg first as a ket, B's leg as a bra. A holds the leg as
    # a ket iff it is OUT, so insert g iff A holds k as out (arrow points A → B).
    for k in Kc
        !_dir(A, k) && (TA = _apply_parity(gr, TA, k))
    end

    C = TA * TB
    order_C = Index[A_open; B_open]
    dirs_C = Bool[[_dir(A, i) for i in A_open]; [_dir(B, i) for i in B_open]]
    return FermionicTensor(C, order_C, dirs_C, gr)
end
