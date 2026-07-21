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

# Sign masks can be reused while a contraction tree keeps the same structural leg
# ordering and grading. These plans deliberately contain no tensor data.
struct FermionicReorderPlan
    signature::UInt
    mask::Union{Nothing, Array{Bool}}
end

struct FermionicBinaryContractionPlan
    plan_a::Union{Nothing, FermionicReorderPlan}
    plan_b::Union{Nothing, FermionicReorderPlan}
end

struct FermionicContractionSequence
    sequence::Vector
    sign_plans::Vector{FermionicBinaryContractionPlan}
end

const ContractionCacheEntry = Union{Vector, FermionicContractionSequence}

Base.copy(ft::FermionicITensor) = FermionicITensor(copy(ft.tensor), copy(ft.order), copy(ft.dirs), copy(ft.grading))

# Direction (arrow) of leg `i` in `ft`: true = in/−, false = out/+.
_dir(ft::FermionicITensor, i::Index) = ft.dirs[findfirst(==(i), ft.order)]

# Total element count of an ITensor (product of its index dimensions), computed from the
# index metadata so no array is materialised.
_nelements(T::ITensor) = prod(i -> dim(i), inds(T); init = 1)
ITensors.inds(ft::FermionicITensor) = ft.order
ITensors.ndims(ft::FermionicITensor) = ndims(ft.tensor)
ITensors.noncommoninds(ft1::FermionicITensor, ft2::FermionicITensor) = noncommoninds(ft1.tensor, ft2.tensor)
ITensors.commoninds(ft1::FermionicITensor, ft2::FermionicITensor) = commoninds(ft1.tensor, ft2.tensor)
ITensors.uniqueinds(ft1::FermionicITensor, ft2::FermionicITensor) = uniqueinds(ft1.tensor, ft2.tensor)
ITensors.commonind(ft1::FermionicITensor, ft2::FermionicITensor) = commonind(ft1.tensor, ft2.tensor)
ITensors.scalar(ft::FermionicITensor) = ITensors.scalar(ft.tensor)
ITensors.ITensor(ft::FermionicITensor) = ft.tensor

# Complete the square of the GF(2) quadratic form whose symmetric coupling is `M` (`M[a,b]=1`
# for an inverted leg pair) and linear part is `lin`. Returns a list of products `(uset,vset)`
# meaning `U = ⊕_{a∈uset} x_a`, `V = ⊕_{b∈vset} x_b`, such that the pair sum
# `⊕_{a<b} M[a,b] x_a x_b  =  ⊕_j U_j V_j`; diagonal remainders generated along the way are
# folded back into `lin`. Each elimination step consumes two legs, so there are at most
# `⌊n/2⌋` products — turning an O(n²)-term mask into an O(n)-term one. Mutates `M` and `lin`.
#
# Step (standard symplectic/Arf reduction): pick active legs i<j with `M[i,j]=1`. With
# `A_j = ⊕_{k} M[j,k] x_k` and `B_i = ⊕_{k} M[i,k] x_k` over the OTHER active legs, the GF(2)
# identity `x_i x_j ⊕ x_i B_i ⊕ x_j A_j = (x_i ⊕ A_j)(x_j ⊕ B_i) ⊕ A_j B_i` peels off one
# product `U=x_i⊕A_j`, `V=x_j⊕B_i` and leaves `A_j B_i` to fold into the form on the rest.
function _gf2_complete_square!(M::AbstractMatrix{Bool}, lin::AbstractVector{Bool}, n::Int)
    active = trues(n)
    products = Tuple{Vector{Int},Vector{Int}}[]
    while true
        i = 0; j = 0; found = false
        @inbounds for a in 1:n
            active[a] || continue
            for b in (a + 1):n
                (active[b] && M[a, b]) && (i = a; j = b; found = true; break)
            end
            found && break
        end
        found || break
        Vp = Int[k for k in 1:n if active[k] && k != i && k != j]
        uset = Int[i]; @inbounds for k in Vp; M[j, k] && push!(uset, k); end   # x_i ⊕ A_j
        vset = Int[j]; @inbounds for k in Vp; M[i, k] && push!(vset, k); end   # x_j ⊕ B_i
        push!(products, (uset, vset))
        # fold A_j·B_i into the remaining form: diagonal (k==m) -> linear, k<m -> pair coeff.
        @inbounds for k in Vp
            (M[j, k] & M[i, k]) && (lin[k] ⊻= true)
        end
        @inbounds for ki in 1:length(Vp), mi in (ki + 1):length(Vp)
            k = Vp[ki]; m = Vp[mi]
            if (M[j, k] & M[i, m]) ⊻ (M[j, m] & M[i, k])
                M[k, m] ⊻= true; M[m, k] ⊻= true
            end
        end
        active[i] = false; active[j] = false
    end
    return products
end

# Koszul-reorder sign mask for `from -> to`, PLUS the diagonal bond-parity `g =
# diag((−1)^p)` on each leg in `parity_legs`, returned as an `Array{Bool}` in `T`'s NATIVE
# storage layout `is` (`true` marks the components that flip sign).
#
# The sign of a component is (−1) raised to Σ over (i) leg pairs whose relative order is
# inverted by the permutation and (ii) the parity legs — a function of the *index pairs*
# and parity bits only, never of memory layout. Building the mask directly in the native
# layout `is` (rather than in `from`) is what lets the caller multiply against the cheap
# native `array(T)` and skip the ~100× more expensive permuting `array(T, from...)`.
#
# The exponent is a GF(2) quadratic form `⊕_{inverted (a,b)} x_a x_b ⊕ ⊕_{parity k} x_k`
# (`x_a = bits[a][i_a]`). Rather than materialise one rank-2 broadcast per inverted pair
# (up to n(n-1)/2 terms), we complete the square over GF(2) into ≤ ⌊n/2⌋ products `U_j V_j`
# of linear (XOR) forms plus a linear remainder, then XOR them into `E` in ONE fused pass.
# Each `U`/`V`/linear form is a lazy XOR of reshaped per-leg parity vectors; an `Array{Bool}`
# (not a `BitArray`) keeps the singleton-reshaped broadcast contiguous.
function _sign_mask(gr::Dictionary, is::Vector{<:Index}, from::Vector{<:Index}, to::Vector{<:Index}, parity_legs)
    n = length(is)
    dims = ntuple(k -> length(gr[is[k]]), n)
    bits = ntuple(k -> gr[is[k]], n)
    rshp(k) = reshape(bits[k], ntuple(d -> d == k ? dims[k] : 1, n))
    # Precompute each native leg's position in `from`/`to` ONCE (O(n) instead of an
    # O(n^3) sweep of `findfirst` over Index vectors inside the pair loop below).
    pf = ntuple(k -> findfirst(==(is[k]), from), n)
    pt = ntuple(k -> findfirst(==(is[k]), to), n)
    # Symmetric GF(2) coupling `M` (inverted pairs) and linear part `lin` (parity legs).
    M = falses(n, n)
    @inbounds for a in 1:n, b in (a + 1):n
        if (pf[a] < pf[b]) != (pt[a] < pt[b])      # inverted iff order differs `from` vs `to`
            M[a, b] = true; M[b, a] = true
        end
    end
    lin = falses(n)
    for k in parity_legs
        lin[findfirst(==(k), is)] ⊻= true          # duplicates cancel (mod 2)
    end
    # Reduce the O(n²)-pair quadratic form to ≤ ⌊n/2⌋ products of linear XOR forms.
    products = _gf2_complete_square!(M, lin, n)
    # Each linear XOR form as a lazy broadcast of reshaped per-leg parity vectors.
    linform(set) = length(set) == 1 ? rshp(set[1]) :
                   reduce((x, y) -> Base.broadcasted(⊻, x, y), (rshp(k) for k in set))
    terms = Any[]
    for (uset, vset) in products
        push!(terms, Base.broadcasted(&, linform(uset), linform(vset)))
    end
    Lset = Int[c for c in 1:n if lin[c]]
    isempty(Lset) || push!(terms, linform(Lset))
    E = Array{Bool}(undef, dims...)
    if isempty(terms)
        fill!(E, false)                                  # no inverted pairs and no parity
    else
        Base.materialize!(E, reduce((x, y) -> Base.broadcasted(⊻, x, y), terms))
    end
    return E
end

# Multiply `T` by the `from -> to` reorder sign and the bond-parity `g` on `parity_legs`,
# all in one native-layout pass. Returns an ITensor with the SAME index objects (only the
# component signs change); the caller updates its fermionic `order`.
function _apply_reorder_sign(gr::Dictionary, T::ITensor, from::Vector{<:Index}, to::Vector{<:Index}, parity_legs = Index[])
    (from == to && isempty(parity_legs)) && return T
    is = collect(inds(T))                       # native storage order: no permute
    E = _sign_mask(gr, is, from, to, parity_legs)
    any(E) || return T
    arr = ITensors.array(T)                      # native layout; skips index-matching/permute
    # Apply the ±1 sign mask with a hand-written `@simd` loop rather than a broadcast: the
    # `arr .* (1 .- 2 .* E)` broadcast fails to vectorise (the `Bool` operand breaks SIMD)
    # and ran ~4× the memcpy floor at large χ. `ifelse` is branchless and generic (negates
    # both parts for Complex). The fresh array is aliased into the ITensor (AllowAlias).
    return ITensors.itensor(_flip_signs(arr, E), is...)
end

# Hash the structural information that determines a sign mask. Index identities are
# intentionally excluded: an untruncated fermionic QR creates fresh bond indices while
# preserving dimensions, relative orders, and parity grading.
function _reorder_signature(
        gr::Dictionary,
        is::Vector{<:Index},
        from::Vector{<:Index},
        to::Vector{<:Index},
        parity_legs,
    )
    h = hash(length(is), UInt(0))
    for i in is
        h = hash(dim(i), h)
        h = hash(findfirst(==(i), from), h)
        h = hash(findfirst(==(i), to), h)
        h = hash(i in parity_legs, h)
        for bit in gr[i]
            h = hash(bit, h)
        end
    end
    return h
end

# Apply a cached mask, rebuilding it if the structural signature changed. This is kept
# separate from `_apply_reorder_sign` so one-off contractions pay no cache bookkeeping.
function _apply_reorder_sign(
        gr::Dictionary,
        T::ITensor,
        from::Vector{<:Index},
        to::Vector{<:Index},
        parity_legs,
        plan::Union{Nothing, FermionicReorderPlan},
    )
    is = collect(inds(T))
    signature = _reorder_signature(gr, is, from, to, parity_legs)
    if isnothing(plan) || plan.signature != signature
        E = _sign_mask(gr, is, from, to, parity_legs)
        plan = FermionicReorderPlan(signature, any(E) ? E : nothing)
    end
    isnothing(plan.mask) && return T, plan
    arr = ITensors.array(T)
    return ITensors.itensor(_flip_signs(arr, plan.mask), is...), plan
end

# Return a copy of `arr` with the sign of every entry flagged in `E` flipped, via a
# vectorisable `@simd` loop (≈ memcpy speed; the equivalent broadcast does not vectorise).
function _flip_signs(arr::Array, E::Array{Bool})
    out = similar(arr)
    @inbounds @simd for i in eachindex(arr, E)
        out[i] = ifelse(E[i], -arr[i], arr[i])
    end
    return out
end

# GPU / generic: move the mask onto arr's device, apply as a fused broadcast kernel
function _flip_signs(arr::AbstractArray, E::AbstractArray{Bool})
    signs = similar(arr, Bool, size(E))
    copyto!(signs, E)                 # one H2D transfer
    return ifelse.(signs, -arr, arr)  # fuses into a single GPU kernel, no scalar indexing
end

# Multiply a tensor by the diagonal bond-parity operator g = diag((−1)^{p}) on
# leg `k` (the even-parity-tensor form of Eq. 13).
function _apply_parity(gr::Dictionary, T::ITensor, k::Index)
    is = collect(inds(T))
    pos = findfirst(==(k), is)
    arr = ITensors.array(T)                      # native layout; skips index-matching/permute
    s = Int8[b ? -1 : 1 for b in gr[k]]
    shape = ntuple(d -> d == pos ? length(s) : 1, ndims(arr))
    return ITensors.itensor(_scale_axis(arr, s, shape), is...)
end

# Scale `arr` by the reshaped ±1 vector `s` (broadcast along one axis).
_scale_axis(arr::Array, s::Vector, shape) = arr .* reshape(s, shape)

# GPU / generic: move the sign vector onto arr's device so the broadcast fuses into a
# single kernel rather than scalar-indexing a host operand.
function _scale_axis(arr::AbstractArray, s::Vector, shape)
    sdev = similar(arr, eltype(s), length(s))
    copyto!(sdev, s)                  # one H2D transfer
    return arr .* reshape(sdev, shape)
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
function _reorder_with_plan(
        ft::FermionicITensor,
        to::Vector{<:Index},
        parity_legs,
        plan::Union{Nothing, FermionicReorderPlan},
        cache_signs::Bool,
    )
    (ft.order == to && isempty(parity_legs)) && return ft.tensor, nothing
    if cache_signs
        return _apply_reorder_sign(
            ft.grading, ft.tensor, ft.order, to, parity_legs, plan
        )
    end
    return _apply_reorder_sign(ft.grading, ft.tensor, ft.order, to, parity_legs), nothing
end

function _contract_fermionic(
        A::FermionicITensor,
        B::FermionicITensor,
        plan::Union{Nothing, FermionicBinaryContractionPlan} = nothing;
        cache_signs::Bool = false,
    )
    # The grading is a GLOBAL, immutable property (an `Index`'s parity bits never
    # change), so we never need to merge A's and B's dictionaries: A's grading covers
    # all of A's legs, B's covers all of B's, and a shared bond carries identical bits
    # in both. Each sign computation below uses whichever operand owns the legs it
    # touches, and the result carries a TIGHT grading over only its own open legs.
    # Split each operand's legs into shared (contracted) and open. Membership is tested
    # directly against the other operand's own `order` vector — a linear `in` over the
    # handful of legs a tensor carries — rather than building a `Set` and calling
    # `commoninds` on the underlying ITensors. Both were pure per-call fixed overhead
    # (a Set is a hash table; `commoninds` re-derives the shared indices ITensor already
    # knows about) and dominate the cost at the small bond dimensions of fermionic BP.
    # `filter` preserves order, so `Kc` is still A's contracted order.
    Aord, Bord = A.order, B.order
    A_open = filter(!in(Bord), Aord)
    B_open = filter(!in(Aord), Bord)
    Kc = filter(in(Bord), Aord)                    # canonical contracted order = A's order

    A_to = Index[A_open; Kc]
    B_to = Index[reverse(Kc); B_open]               # B's contracted block REVERSED (nesting sign)

    # Supertrace (bond-parity g) per shared bond. The contraction map C adds a supertrace
    # (−1)^{p} exactly for a KET-BRA pair (SciPost "Fermionic tensor networks" Eq. 107):
    # A's leg first as a ket, B's leg as a bra. A holds the leg as a ket iff it is OUT, so
    # insert g iff A holds k as out (arrow points A → B). This is diagonal in component space
    # like the reorder sign, so it folds into a single reorder pass.
    #
    # Crucially, g lives on a CONTRACTED leg, which is summed over: `Σ_k A[..,k] g_k B[k,..]`
    # is identical whether g weights A's or B's k-axis. We therefore attach g to whichever
    # operand is ALREADY being copied for its reorder, so it costs no extra allocation. When
    # an operand does not reorder, `_apply_reorder_sign` returns it untouched (no copy), so
    # pinning g to a non-reordering operand (the old behaviour: always A) would force a needless
    # full-size copy — exactly the multi-bond blow-up. Only when NEITHER operand reorders does g
    # require a copy, and then we flip the smaller operand.
    parity = Index[k for k in Kc if !_dir(A, k)]
    A_reorder = A.order != A_to
    B_reorder = B.order != B_to

    parity_a, parity_b = Index[], Index[]
    if B_reorder
        parity_b = parity
    elseif A_reorder
        parity_a = parity
    elseif _nelements(A.tensor) <= _nelements(B.tensor)
        parity_a = parity
    else
        parity_b = parity
    end

    plan_a = isnothing(plan) ? nothing : plan.plan_a
    plan_b = isnothing(plan) ? nothing : plan.plan_b
    TA, plan_a = _reorder_with_plan(A, A_to, parity_a, plan_a, cache_signs)
    TB, plan_b = _reorder_with_plan(B, B_to, parity_b, plan_b, cache_signs)

    C = TA * TB
    order_C = Index[A_open; B_open]
    dirs_C = Bool[[_dir(A, i) for i in A_open]; [_dir(B, i) for i in B_open]]
    gr_C = Dictionary{Index, Vector{Bool}}(
        order_C,
        Vector{Bool}[[A.grading[i] for i in A_open]; [B.grading[i] for i in B_open]],
    )
    result = FermionicITensor(C, order_C, dirs_C, gr_C)
    new_plan = cache_signs ? FermionicBinaryContractionPlan(plan_a, plan_b) : nothing
    return result, new_plan
end

function ITensors.contract(A::FermionicITensor, B::FermionicITensor)
    result, _ = _contract_fermionic(A, B)
    return result
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

# Linear combination of two fermionic tensors over the SAME leg set. `b` is first
# permuted to `a`'s fermionic leg order (which applies the Koszul reordering sign), so
# the two dense arrays are aligned component-for-component before the bosonic ITensor `±`
# (which itself index-matches). The result inherits `a`'s order/dirs/grading; `b` must
# carry the same arrows/grading on those legs (true when `a` and `b` are two pieces of
# the same bond object, e.g. an identity and a message outer product).
function Base.:-(a::FermionicITensor, b::FermionicITensor)
    bp = ITensors.permute(b, a.order)
    return FermionicITensor(a.tensor - bp.tensor, a.order, a.dirs, a.grading)
end
function Base.:+(a::FermionicITensor, b::FermionicITensor)
    bp = ITensors.permute(b, a.order)
    return FermionicITensor(a.tensor + bp.tensor, a.order, a.dirs, a.grading)
end

# Single-index relabel (`replaceinds` for FermionicITensor lives in
# factorize_fermionic_tensors.jl): parity-neutral renaming, arrows unchanged.
ITensors.replaceind(ft::FermionicITensor, i::Index, j::Index) = ITensors.replaceinds(ft, (i,), (j,))

# Walk the nested binary contraction tree `seq` (integer indices into `fts`).
function follow_sequence(seq, fts::Vector{<:FermionicITensor})
    seq isa Integer && return fts[seq]
    acc = follow_sequence(seq[1], fts)
    for k in 2:length(seq)
        acc = acc * follow_sequence(seq[k], fts)
    end
    return acc
end

# Cached variant of `follow_sequence`. Binary contractions are encountered in a stable
# depth-first order, so each node can reuse its mask plans directly without a dictionary
# lookup or rebuilding a structural cache key.
function follow_sequence(
        seq,
        fts::Vector{<:FermionicITensor},
        sign_plans::Vector{FermionicBinaryContractionPlan},
        cursor::Base.RefValue{Int},
    )
    seq isa Integer && return fts[seq]
    acc = follow_sequence(seq[1], fts, sign_plans, cursor)
    for k in 2:length(seq)
        rhs = follow_sequence(seq[k], fts, sign_plans, cursor)
        slot = cursor[]
        plan = slot <= length(sign_plans) ? sign_plans[slot] : nothing
        acc, plan = _contract_fermionic(acc, rhs, plan; cache_signs = true)
        if slot <= length(sign_plans)
            sign_plans[slot] = plan
        else
            push!(sign_plans, plan)
        end
        cursor[] += 1
    end
    return acc
end

# Contract a list of FermionicITensors using the bosonic optimal contraction order
# (`contraction_sequence(..., alg="optimal")`) on the underlying ITensors, then
# follow that tree through `contract`.
function ITensors.contract(
        fts::Vector{<:FermionicITensor};
        sequence = contraction_sequence(fts; alg = "optimal"),
        sign_plans::Union{Nothing, Vector{FermionicBinaryContractionPlan}} = nothing,
    )
    length(fts) == 1 && return only(fts)
    isnothing(sign_plans) && return follow_sequence(sequence, fts)
    cursor = Ref(1)
    result = follow_sequence(sequence, fts, sign_plans, cursor)
    @assert cursor[] == length(sign_plans) + 1
    return result
end

function Adapt.adapt_structure(to, ft::FermionicITensor)
    t = adapt(to)(ft.tensor)
    return FermionicITensor(t, ft.order, ft.dirs, ft.grading)
end

function ITensors.noprime(ft::FermionicITensor)
    t = noprime(ft.tensor)
    neworder = noprime.(ft.order)
    # Priming is parity-neutral, so re-key the grading onto the noprimed legs (the
    # bits are unchanged). `contract` relies on each tensor's grading covering exactly
    # its own legs, so the keys must track the relabel.
    gr = Dictionary{Index, Vector{Bool}}(neworder, Vector{Bool}[ft.grading[i] for i in ft.order])
    return FermionicITensor(t, neworder, ft.dirs, gr)
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
