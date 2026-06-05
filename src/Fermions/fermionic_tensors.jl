using ITensors.NDTensors: NDTensors

"""
    random_even_itensor(eltype, is::Vector{<:Index}, grading)

Build a random ITensor over indices `is` whose components with *odd* total Z2 parity are
zeroed, so the result is parity even with respect to `grading`. Index directions and order are defined later.
"""
function random_even_itensor(eltype, is::Vector{<:Index}, grading::Dictionary{Index, Vector{Bool}})
    bits = [grading[i] for i in is]
    dims = ntuple(k -> dim(is[k]), length(is))
    arr = zeros(eltype, dims...)
    for I in CartesianIndices(dims)
        odd = false
        for k in 1:length(is)
            odd ⊻= bits[k][I[k]]
        end
        odd || (arr[I] = randn(eltype))
    end
    return ITensor(arr, is...)
end

random_even_itensor(is::Vector{<:Index}, grading::Dictionary{Index, Vector{Bool}}) = random_even_itensor(Float64, is, grading)

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
using Adapt: Adapt, adapt

"""
    FermionicITensor

A dense ITensor together with the metadata for locally-ordered fermionic
operations: the fermionic leg `order`, the per-leg arrow `dirs` (`true` = in/−,
`false` = out/+), and a reference to the global Z2 `grading` (per-component
parity bits of each `Index`). `order` and `dirs` are parallel vectors.
"""
struct FermionicITensor
    tensor::ITensor
    order::Vector{Index}
    dirs::Vector{Bool}
    grading::Dictionary{Index, Vector{Bool}}
end

Base.copy(ft::FermionicITensor) = FermionicITensor(copy(ft.tensor), copy(ft.order), copy(ft.dirs), copy(ft.grading))

# Direction (arrow) of leg `i` in `ft`: true = in/−, false = out/+.
_dir(ft::FermionicITensor, i::Index) = ft.dirs[findfirst(==(i), ft.order)]
ITensors.inds(ft::FermionicITensor) = ft.order
ITensors.ndims(ft::FermionicITensor) = ndims(ft.tensor)
ITensors.noncommoninds(ft1::FermionicITensor, ft2::FermionicITensor) = noncommoninds(ft1.tensor, ft2.tensor)
ITensors.commoninds(ft1::FermionicITensor, ft2::FermionicITensor) = commoninds(ft1.tensor, ft2.tensor)
ITensors.uniqueinds(ft1::FermionicITensor, ft2::FermionicITensor) = uniqueinds(ft1.tensor, ft2.tensor)
ITensors.commonind(ft1::FermionicITensor, ft2::FermionicITensor) = commonind(ft1.tensor, ft2.tensor)
ITensors.scalar(ft::FermionicITensor) = ITensors.scalar(ft.tensor)
ITensors.ITensor(ft::FermionicITensor) = ft.tensor

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
    permute(ft::FermionicITensor, neworder::Vector{<:Index})

Reorder the legs of `ft` to `neworder` (a permutation of `ft.order`), applying the
fermionic transposition sign `(−1)^{p_i p_j}` for each swapped odd pair.
"""
function ITensors.permute(ft::FermionicITensor, neworder::Vector{<:Index})
    T = _apply_reorder_sign(ft.grading, ft.tensor, ft.order, neworder)
    perm = [findfirst(==(i), ft.order) for i in neworder]
    return FermionicITensor(T, copy(neworder), ft.dirs[perm], ft.grading)
end

"""
    dag(ft::FermionicITensor)

Hermitian conjugate in the fermionic sense: conjugate the data, reverse the leg
order (with the accompanying reversal sign), and flip every arrow (in ↔ out). This
is the per-tensor operation behind taking ⟨Ψ| from |Ψ⟩ (Fig. 4: all bra arrows are
reversed relative to the ket).
"""
function ITensors.dag(ft::FermionicITensor)
    rev = reverse(ft.order)
    # Eq.33/108: conjugation maps each space to its dual and REVERSES the leg
    # order. Reversing the labels relabels which dual sits where; it is NOT a
    # permutation of stored components, so there is NO extra Koszul sign here.
    return FermionicITensor(dag(ft.tensor), rev, .!reverse(ft.dirs), ft.grading)
end

# Bra factor for the doubled network: `dag(ket)` (σ-free conjugation: conjugate data,
# reversed leg order, flipped arrows) with every leg primed EXCEPT those in `keep` (the
# un-operated site legs, which stay unprimed so the bra contracts straight onto the ket).
# Priming a leg that carries an operator lets it join the operator's primed output `u =
# prime(s)`. The result carries a LOCAL grading over its own (primed) legs only.
function ITensors.dag(ft::FermionicITensor, keep)
    bt = dag(ft)
    oldis = bt.order
    newis = Index[i in keep ? i : prime(i) for i in oldis]
    gr = Dictionary{Index, Vector{Bool}}(newis, [bt.grading[i] for i in oldis])
    T = replaceinds(bt.tensor, oldis, newis)
    return FermionicITensor(T, newis, bt.dirs, gr)
end

"""
    fit_adjoint(ft)
    fit_adjoint(ft, metric_legs)

Metric-corrected fermionic adjoint used by the boundary-MPS variational fit. Equal to
`g · dag(ft)`: on top of the ordinary `dag`, apply the supertrace metric `g = diag((−1)^p)`
on every leg in `metric_legs` that `ft` holds OUT (the legs on which `contract` would
otherwise insert `g`). The one-argument form uses `metric_legs = ft.order`, i.e. all legs.

The point: closing the metricised legs of `ft` against `fit_adjoint(ft)` gives a genuine
positive (Euclidean) closure `Σ_I |ft_I|²` rather than the *signed* supertrace that plain
`dag` produces, because the metric `g` cancels the `g` that `contract` re-inserts on those
legs. This makes the Euclidean (LAPACK) QR used to move the orthogonality centre the correct
orthogonalisation in the fermionic inner product.

The boundary-MPS fit does NOT want this on every leg: its bra-rail tensors close their
*crossing* legs against the double-layer bulk (a genuine ket/bra supertrace, which needs the
metric) but carry *virtual MPS bonds* that are pure fitting indices orthogonalised by the
Euclidean QR (which must NOT be metricised). `fit_adjoint_message` therefore calls the
two-argument form with `metric_legs` = just the crossing legs; doing so is what makes the
orthogonal sweep reproduce the exact boundary environment for both update directions.

On parity-EVEN tensors the all-legs form is an involution: the first application puts `g` on
the OUT legs; after the arrow flip the second application puts `g` on the (now-OUT) IN legs,
so together they apply `g` on *all* legs, whose net sign `(−1)^{total parity}` is `+1`.

Bosonic tensors need no metric, so the fallback is just `dag`.
"""
fit_adjoint(t::ITensor, args...) = dag(t)
fit_adjoint(ft::FermionicITensor) = fit_adjoint(ft, ft.order)
function fit_adjoint(ft::FermionicITensor, metric_legs)
    b = dag(ft)
    T = b.tensor
    for k in ft.order
        (k in metric_legs && !_dir(ft, k)) && (T = _apply_parity(b.grading, T, k))
    end
    return FermionicITensor(T, b.order, b.dirs, b.grading)
end

"""
    contract(A::FermionicITensor, B::FermionicITensor)

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
function ITensors.contract(A::FermionicITensor, B::FermionicITensor)
    gr = merge(A.grading, B.grading)
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
    return FermionicITensor(C, order_C, dirs_C, gr)
end

Base.:*(ft1::FermionicITensor, ft2::FermionicITensor) = ITensors.contract(ft1, ft2)
function Base.:/(ft::FermionicITensor, λ::Number)
    t = ft.tensor / λ
    return FermionicITensor(t, ft.order, ft.dirs, ft.grading)
end
function Base.:*(ft::FermionicITensor, λ::Number)
    t = ft.tensor * λ
    return FermionicITensor(t, ft.order, ft.dirs, ft.grading)
end

# Walk the nested binary contraction tree `seq` (integer indices into `fts`),
# folding pairs together with `contract`. Because `contract` is contraction-order
# independent for parity-even tensors, the result equals the naive fold; the
# ordering only changes intermediate cost.
function follow_sequence(seq, fts::Vector{<:FermionicITensor})
    seq isa Integer && return fts[seq]
    acc = follow_sequence(seq[1], fts)
    for k in 2:length(seq)
        acc = acc * follow_sequence(seq[k], fts)
    end
    return acc
end

# Contract a list of FermionicITensors using the bosonic optimal contraction order
# (`contraction_sequence(..., alg="optimal")`) on the underlying ITensors, then
# follow that tree through `contract`.
function ITensors.contract(fts::Vector{<:FermionicITensor}; sequence = contraction_sequence(fts; alg = "optimal"))
    length(fts) == 1 && return only(fts)
    return follow_sequence(sequence, fts)
end

function Adapt.adapt_structure(to, ft::FermionicITensor)
    t = adapt(to)(ft.tensor)
    return FermionicITensor(t, ft.order, ft.dirs, ft.grading)
end

function ITensors.noprime(ft::FermionicITensor)
    t = noprime(ft.tensor)
    return FermionicITensor(t, noprime.(ft.order), ft.dirs, ft.grading)
end

function NDTensors.scalartype(ft::FermionicITensor)
    return scalartype(ft.tensor)
end

function ITensors.datatype(ft::FermionicITensor)
    return ITensors.datatype(ft.tensor)
end

function ITensors.sum(ft::FermionicITensor)
    return ITensors.sum(ft.tensor)
end

function ITensors.norm(ft::FermionicITensor)
    return ITensors.norm(ft.tensor)
end

LinearAlgebra.normalize(ft::FermionicITensor) = ft / norm(ft)


const Tensor = Union{ITensor, FermionicITensor}

function message_diff(message_a::FermionicITensor, message_b::FermionicITensor)
    message_b_inds = inds(message_b)
    p_message_a = ITensors.permute(message_a, message_b_inds)
    return message_diff(p_message_a.tensor, message_b.tensor)
end