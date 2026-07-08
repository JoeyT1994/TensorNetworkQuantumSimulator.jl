# Compatibility shims bridging the legacy `ITensors.jl` API that TNQS was written
# against to the next-gen `ITensorBase.jl` backend.
#
# Strategy (see ITensorDevelopmentPlans api_migration_map.md):
#   - Names below are thin wrappers over ITensorBase / TensorAlgebra / MatrixAlgebraKit.
#   - The factorization return shapes, the operator/SiteType system, and
#     the boundary-MPS (ITensorMPS) paths are NOT wrapped here; they need callsite
#     translation or upstream stack work and are tracked separately. The legacy
#     `combiner` is retired rather than wrapped: call sites use the next-gen fusion
#     primitives (`matricize` below, the fused identity `Base.one`) directly.
#
# ITensorBase keeps most of this API internal (unexported), so we reach for the
# qualified names and re-publish the legacy spellings into the TNQS namespace.

import MatrixAlgebraKit as MAK
using Adapt: Adapt
using ITensorBase: ITensorBase, AbstractITensor, ITensor, Index, NamedUnitRange, name,
    nameddims, plev, tags, unnamed
using LinearAlgebra: LinearAlgebra
using TensorAlgebra.MatrixAlgebra: MatrixAlgebra
using TensorAlgebra: TensorAlgebra

# Legacy `inds(t; plev, tags)` took index-filtering keywords; `ITensorBase.inds` takes none. Compat
# `inds` forwards to it and applies the legacy filters (a compat-owned function, not a pirated method).
function inds(t::AbstractITensor; plev = nothing, tags = nothing)
    is = ITensorBase.inds(t)
    isnothing(plev) || (is = filter(i -> ITensorBase.plev(i) == plev, is))
    isnothing(tags) || (is = filter(i -> hastags(i, tags), is))
    return is
end

# Module-owned functions TNQS extends for its own types: `scalartype` falls back to ITensorBase for
# tensors/arrays; `contract`/`inner` are TNQS operations whose base methods live in the library.
scalartype(x) = ITensorBase.scalartype(x)
function inner end

# Base contraction of a list of ITensors along a (possibly nested) pairwise `sequence` (legacy
# `ITensors.contract(tensors; sequence)`); leaves are integer indices into `tensors`. Typed
# `AbstractVector` (not `{<:AbstractITensor}`) because callers concatenate in ways that widen to
# `Vector{Any}` (e.g. splicing in an empty environment list).
function contract end
function contract(tensors::AbstractVector; sequence = nothing)
    return isnothing(sequence) ? reduce(*, tensors) : _contract_sequence(tensors, sequence)
end
_contract_sequence(tensors, s::Integer) = tensors[s]
_contract_sequence(tensors, s) = reduce(*, (_contract_sequence(tensors, x) for x in s))

# Get the index collection of an `ITensor`, or pass an index collection through
# unchanged. TNQS nests these calls (e.g. `uniqueinds(uniqueinds(...), ...)`), so the
# index-set helpers below accept both tensors and bare index collections.
_compat_inds(t::AbstractITensor) = inds(t)
_compat_inds(is) = is

#
# Small-collection set operations keyed by a transform `by` (elements compare equal when
# `by(x) == by(y)`). These scan linearly rather than building `Set`s: Base's `Set`-based ops hash,
# and hashing an `Index` over a TensorKit `GradedSpace` can fall back to iterating the space, which
# throws. The intersect/setdiff/union/symdiff forms return elements of the first argument as
# `Vector`s.
#
smallintersect(a, b; by = identity) = (kb = Iterators.map(by, b); [x for x in a if by(x) ∈ kb])
smallsetdiff(a, b; by = identity) = (kb = Iterators.map(by, b); [x for x in a if by(x) ∉ kb])
smallunion(a, b; by = identity) = vcat(collect(a), smallsetdiff(b, a; by))
smallsymdiff(a, b; by = identity) = vcat(smallsetdiff(a, b; by), smallsetdiff(b, a; by))
smallisdisjoint(a, b; by = identity) = (kb = Iterators.map(by, b); !any(x -> by(x) ∈ kb, a))
smallissubset(a, b; by = identity) = (kb = Iterators.map(by, b); all(x -> by(x) ∈ kb, a))
smallissetequal(a, b; by = identity) = smallissubset(a, b; by) && smallissubset(b, a; by)

#
# Name-based index-set algebra: the small-collection ops keyed by `IndexName` (`by = name`) rather
# than full `Index` equality. On a graded axis a shared bond appears as an index on one tensor and
# its dual (`conj`) on the other — same name, opposite arrow — so full-`Index` `==` misses it while
# the names match; on the dense backend the two coincide. The `Index`-returning ops give back full
# `Index`es (callers need the space to feed `Index`/`qr`/`replaceinds`).
#
nameisequal(i, j) = name(i) == name(j)
nameintersect(a, b) = smallintersect(a, b; by = name)
namesetdiff(a, b) = smallsetdiff(a, b; by = name)
nameunion(a, b) = smallunion(a, b; by = name)
namesymdiff(a, b) = smallsymdiff(a, b; by = name)
nameisdisjoint(a, b) = smallisdisjoint(a, b; by = name)
nameissubset(a, b) = smallissubset(a, b; by = name)
nameissetequal(a, b) = smallissetequal(a, b; by = name)

#
# Named-tensor index-set algebra (the `commoninds`/etc. surface). `_compat_inds` coerces a tensor to
# its indices and passes an index collection through unchanged, so these accept either; the `name*`
# ops take it from there (always returning `Vector`s, so a result can be fed back in, as
# `reduce(noncommoninds, tensors)` does).
#
commoninds(a, b) = nameintersect(_compat_inds(a), _compat_inds(b))
uniqueinds(a, b) = namesetdiff(_compat_inds(a), _compat_inds(b))
unioninds(a, b) = nameunion(_compat_inds(a), _compat_inds(b))
hascommoninds(a, b) = !nameisdisjoint(_compat_inds(a), _compat_inds(b))

# Singular forms: the one common / one unique index. `noncommonind` is used in TNQS
# as "the index of `a` not shared with `b`" (e.g. the non-contracted leg of an
# eigenvalue tensor); revisit if a symmetric-difference callsite turns up.
commonind(a, b) = (cs = commoninds(a, b); isempty(cs) ? nothing : first(cs))
noncommonind(a, b) = (us = uniqueinds(a, b); isempty(us) ? nothing : first(us))
# Plural: indices not shared by both (symmetric difference).
noncommoninds(a, b) = namesymdiff(_compat_inds(a), _compat_inds(b))

# Index dimension (legacy `dim`). `dim(i)` is the length; `dim(is)` the product.
dim(i::Index) = length(i)
dim(is::Union{Tuple, AbstractVector}) = isempty(is) ? 1 : prod(length, is)

#
# Index operations.
#
# A fresh index with the same length, tags, and prime level (legacy `sim`).
sim(i::Index) = ITensorBase.uniquename(i)
sim(is::Union{Tuple, AbstractVector{<:Index}}) = map(sim, is)

# Conjugate (legacy `dag`): `conj` the tensor, and on bare indices flip the sector arrows
# on a graded axis. `conj(::Index)` is id-preserving on the dense backend, so `dag` there
# is effectively the identity on indices, matching legacy behavior.
dag(t::AbstractITensor) = conj(t)
dag(i::Index) = conj(i)
dag(is::Union{Tuple, AbstractVector}) = map(conj, is)

# `prime` / `noprime`: ITensorBase primes an `Index`; TNQS also primes whole ITensors
# (all dimnames). Owned here with a fallback to ITensorBase for the index case.
prime(x) = ITensorBase.prime(x)
function prime(t::AbstractITensor)
    return nameddims(unnamed(t), map(ITensorBase.prime, ITensorBase.dimnames(t)))
end
prime(is::Union{Tuple, AbstractVector{<:Index}}) = map(prime, is)
noprime(x) = ITensorBase.noprime(x)
function noprime(t::AbstractITensor)
    return nameddims(unnamed(t), map(ITensorBase.noprime, ITensorBase.dimnames(t)))
end
noprime(is::Union{Tuple, AbstractVector{<:Index}}) = map(noprime, is)

# `replaceind` (singular) maps to a single-pair replacement, forwarding to the
# compat-owned `replaceinds` (below).
replaceind(t, p::Pair) = replaceinds(t, p)
replaceind(t, from::Index, to::Index) = replaceinds(t, from => to)

# Concatenate indices and/or index collections into a single `Vector{Index}`. Legacy
# ITensors code spells this `vcat(i, j)` / `vcat(is, i)`, but next-gen `Index` is a
# `NamedUnitRange` (`<: AbstractVector`), so `vcat` of bare indices concatenates their
# integer ranges instead of collecting the indices. Used where the legacy idiom builds
# an index list from a mix of single indices and collections.
_as_index_vec(x::Index) = [x]
_as_index_vec(xs) = collect(xs)
cat_inds(xs...) = reduce(vcat, map(_as_index_vec, xs))
# Legacy `replaceinds` took collection arguments (`replaceinds(t, [i,k], [j,l])`, `... => ...`);
# ITensorBase provides only the pair-splat form, so this compat handles the collection forms.
#
# The base case relabels *names* via `replacedimnames`, not `ITensorBase.replaceinds` (which replaces
# the index *space*, scalar-indexing a graded axis and erroring). Keys are stripped to `IndexName`
# because `replacedimnames` matches on `dimnames` and silently no-ops on a raw `Index` key.
#
# NB: an `Index` is itself an `AbstractVector` (`NamedUnitRange`), so the collection types are
# constrained to `AbstractVector{<:Index}` to avoid capturing a bare `Index` and iterating its range.
const _IndexColl = Union{Tuple{Vararg{Index}}, AbstractVector{<:Index}}
function replaceinds(t, pairs::Pair...)
    return ITensorBase.replacedimnames(
        t,
        map(p -> name(first(p)) => name(last(p)), pairs)...
    )
end
function replaceinds(t, from::_IndexColl, to::_IndexColl)
    return replaceinds(t, map(=>, from, to)...)
end
function replaceinds(t::AbstractITensor, p::Pair{<:_IndexColl, <:_IndexColl})
    return replaceinds(t, first(p), last(p))
end

#
# ITensor construction.
#
# Legacy `itensor(array, inds)`: inherit the index spaces. NB: `ITensor(array, inds)`
# with raw `Index` objects is intentionally NOT supported by ITensorBase (the space
# is underdefined); use the indexing form, which inherits the indices' spaces, or
# `ITensor(array, name.(inds))` to take the space from the array. Like the legacy
# constructor, a matching total length is accepted and reshaped to the index
# dimensions (e.g. a `d^2 × d^2` two-site gate matrix over four site legs).
function itensor(array, is...)
    length(array) == prod(length, is) ||
        throw(
        DimensionMismatch(
            "array with $(length(array)) elements cannot fill indices of dimensions $(length.(is))"
        )
    )
    return reshape(array, map(length, is))[is...]
end
itensor(array, is::Union{Tuple, AbstractVector}) = itensor(array, is...)

# TYPE PIRACY (temporary, compat-owned — NOT upstream): rank-0 `ITensor(x::Number)`, which
# ITensorBase deliberately omits. Legacy ITensors uses it to seed a product accumulator
# (`out = ITensor(1); out *= t`). Retired once those call sites are rewritten.
ITensorBase.ITensor(x::Number) = nameddims(fill(x), ())

# Random ITensor over the given indices (legacy `random_itensor`).
random_itensor(eltype::Type, is::Index...) = randn(eltype, is...)
random_itensor(eltype::Type, is::Union{Tuple, AbstractVector}) = randn(eltype, is...)
random_itensor(is::Index...) = random_itensor(Float64, is...)
random_itensor(is::Union{Tuple, AbstractVector}) = random_itensor(Float64, is)

# Rank-0 scalar extraction (legacy `scalar`).
scalar(t::AbstractITensor) = t[]

# Dense Kronecker delta tensor (legacy `delta`), vendored from ITensorNetworksNext's
# `ITensorNetworkGenerators/delta_network.jl` (TNQS doesn't depend on it for this migration). A
# graded/sector-aware `delta` is a stack gap (tracked separately).
diaglength(a::AbstractArray) = minimum(size(a))
function diagstride(a::AbstractArray)
    s = 1
    p = 1
    for i in 1:(ndims(a) - 1)
        p *= size(a, i)
        s += p
    end
    return s
end
function diagindices(a::AbstractArray)
    maxdiag = LinearIndices(a)[CartesianIndex(ntuple(Returns(diaglength(a)), ndims(a)))]
    return 1:diagstride(a):maxdiag
end
diagview(a::AbstractArray) = @view a[diagindices(a)]

function diagonaltensor(diag::AbstractVector, ax::Tuple{Vararg{AbstractUnitRange}})
    a = similar(diag, ax)
    fill!(a, zero(eltype(a)))
    diagview(a) .= diag
    return a
end
function diagonaltensor(
        diag::AbstractVector,
        is::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return nameddims(diagonaltensor(diag, unnamed.(is)), name.(is))
end

# Allocate an identity-map-shaped tensor over the `codomain`/`domain` indices, following
# `prototype`'s backend/eltype (domain axes dualized in storage). Unlike `Base.one(a, codomain,
# domain)` — which needs `a` to already carry the map's legs — this lets a caller whose prototype
# lacks a leg still build a device-following identity via `one(similar_map(...), codomain, domain)`.
function similar_map(prototype::AbstractITensor, eltype::Type, codomain, domain)
    raw = TensorAlgebra.similar_map(
        unnamed(prototype), eltype, unnamed.(codomain), unnamed.(domain)
    )
    return nameddims(raw, (name.(codomain)..., name.(domain)...))
end
function similar_map(prototype::AbstractITensor, codomain, domain)
    return similar_map(prototype, scalartype(prototype), codomain, domain)
end

# From-scratch identity map: a dense identity embedded onto the `codomain`/`domain` index
# partition via checked `project`, so the index axes select the backend (dense, graded,
# `TensorMap`). Unlike `one(a, codomain, domain)` it needs no prototype tensor, so it is the
# right primitive when only the indices and an element type are in hand (e.g. `op("I")`).
function id(eltype::Type, codomain, domain)
    m = Matrix{eltype}(LinearAlgebra.I, prod(length, codomain), prod(length, domain))
    return TensorAlgebra.project(m, Tuple(codomain), Tuple(domain))
end

# Dense Kronecker copy (`delta`) tensor over the index axes. Dense-only: a super-diagonal
# generally cannot be embedded while preserving a nontrivial symmetry, so graded/`TensorMap`
# callers that want an order-2 identity build it via `id`/`one` at the callsite instead.
delta(eltype::Type, is::Tuple) = diagonaltensor(ones(eltype, minimum(length, is)), is)
delta(eltype::Type, is::Index...) = delta(eltype, is)
delta(eltype::Type, is::AbstractVector{<:Index}) = delta(eltype, Tuple(is))
delta(is::Tuple) = delta(Float64, is)
delta(is::Index...) = delta(Float64, is)
delta(is::AbstractVector{<:Index}) = delta(Float64, Tuple(is))

# Trace over prime pairs (legacy `tr`): contract with the identity map pairing each unprimed index
# with its prime. The domain is built as `dag.(prime.(codomain))`, not `inds(t; plev=1)` — `plev`
# filtering does not preserve the pairing order between the plev-0 and plev-1 groups. Accessed
# qualified (`ITensors.tr`) so it doesn't shadow `LinearAlgebra.tr`, which TNQS calls on matrices.
function tr(t::AbstractITensor)
    unprimed = inds(t; plev = 0)
    codomain, domain = dag.(unprimed), dag.(prime.(unprimed))
    return scalar(t * one(similar_map(t, codomain, domain), codomain, domain))
end

# One-hot vector along `i` at position `p` (legacy `onehot(i => p)`), through `project_aux`
# (in `ops.jl`), which follows the index's backend and carries the basis vector's charge on
# a derived auxiliary leg.
function onehot(eltype::Type, (i, p)::Pair{<:Index})
    v = zeros(eltype, length(i))
    v[p] = one(eltype)
    return project_aux(v, i)
end
onehot(p::Pair{<:Index}) = onehot(Float64, p)

#
# Factorizations. These map onto MatrixAlgebraKit's named-tensor methods
# (`f(a, codomain, domain)`). The legacy return shapes differ from MAK's, so the
# heavy factorization callsites (simple_update / full_update / symmetric_gauge) are
# translated directly rather than fully wrapped here; these aliases cover the simple
# uses. See api_migration_map.md.
#
const svd_trunc = MAK.svd_trunc

# Legacy `qr(a, linds)`: `Q` over `(linds..., q)` (isometric), `R` over `(q, rest...)`.
# `linds` may be splatted, a single index, or one collection.
function qr(a::AbstractITensor, linds...)
    left = cat_inds(linds...)
    right = namesetdiff(inds(a), left)
    return MAK.qr_compact(a, Tuple(left), Tuple(right))
end

# Legacy `apply(o, ψ)` (gate application): contract `o`'s unprimed legs with `ψ`'s
# matching indices, then drop the prime so `o`'s output legs become `ψ`'s site legs.
# Covers the one- and two-site gates TNQS applies to states without pre-existing primes.
apply(o::AbstractITensor, ψ::AbstractITensor) = noprime(o * ψ)

# `svd` / `eigen` accept a single `Index` or a collection as the codomain, and
# `eigen` auto-partitions a 2-index tensor when no split is given (legacy behavior).
_astuple(x::Index) = (x,)
_astuple(x) = Tuple(x)

function svd(a::AbstractITensor, codomain; kwargs...)
    return MAK.svd_compact(a, _astuple(codomain); kwargs...)
end
function svd(a::AbstractITensor, codomain, domain; kwargs...)
    return MAK.svd_compact(a, _astuple(codomain), _astuple(domain); kwargs...)
end

# Legacy `eigen` here is a hermitian eigendecomposition returning `(D, U)`. It routes
# to `MAK.eigh_full` and then renames indices into the legacy ITensors convention the
# TNQS callsites assume:
#   - `U` over `(domain..., u)` where `u` is the eigenvalue index shared with `D`.
#   - `D` diagonal over `(prime(u), u)` — a prime pair, like legacy `eigen`.
# `eigh_full` returns `D` over two independent fresh indices; we rename its `U`-disjoint
# index to `prime(u)` so `D` becomes the legacy prime-paired diagonal. The legacy
# truncation kwargs (`cutoff`/`maxdim`/`mindim`) are not yet translated to MAK's
# `trunc=(; ...)` spec — see api_migration_map.md; the current callsites pass
# `cutoff = nothing` (full decomposition).
function _eigh(
        m::AbstractITensor,
        codomain,
        domain;
        ishermitian = false,
        cutoff = nothing
    )
    ishermitian ||
        error("the compat `eigen` only supports the hermitian case (ishermitian = true)")
    isnothing(cutoff) || error(
        "the compat `eigen` does not yet translate the `cutoff` truncation kwarg to MatrixAlgebraKit's `trunc` spec"
    )
    cod, dom = _astuple(codomain), _astuple(domain)
    D, U = MAK.eigh_full(m, cod, dom)
    u = only(commoninds(D, U))         # eigenvalue index shared with U
    t = only(uniqueinds(D, U))         # D's other (independent) index
    D = replaceinds(D, t => ITensorBase.prime(u))
    return D, U
end

# Partitioned form `eigen(m, Linds, Rinds)` reproduces legacy ITensors' reconstruction
# `m = Vt * D * dag(V)` (with `Vt` the relabeling of `V` from `Rinds` to `Linds`) —
# no conjugation of `U`.
function eigen(m::AbstractITensor, codomain, domain; kwargs...)
    return _eigh(m, codomain, domain; kwargs...)
end

# No-partition form `eigen(m)` matches legacy ITensors' `eigen(A)`, which auto-partitions
# `Ris = filterinds(plev = 0)`, `Lis = Ris'`. That orientation is the adjoint view of the
# partitioned call, so `eigh_full` returns the conjugate eigenvectors; `conj(U)` recovers
# the convention `symmetric_gauge` uses (`U * D * prime(dag(U)) == m`, a true matrix sqrt).
function eigen(m::AbstractITensor; kwargs...)
    is = collect(inds(m))
    p0 = filter(i -> plev(i) == 0, is)
    p1 = filter(i -> plev(i) != 0, is)
    length(p0) == length(p1) || error(
        "`eigen` without an index partition expects each plev-0 index to be paired with its prime"
    )
    D, U = _eigh(m, Tuple(p1), Tuple(p0); kwargs...)
    return D, conj(U)
end

# ITensors-style truncation as a MatrixAlgebraKit strategy. `maxdim` caps the kept rank
# (`truncrank`); `cutoff` discards the smallest singular values whose summed squares are a fraction
# ≤ cutoff of the total, i.e. a relative 2-norm error of `sqrt(cutoff)` on the singular-value vector
# (`truncerror(; rtol = sqrt(cutoff), p = 2)`). Either kwarg may be `nothing` (or trivial) to drop
# that constraint; with neither, the result is `notrunc()`.
function itensor_trunc(; maxdim = nothing, cutoff = nothing)
    trunc = MAK.notrunc()
    isnothing(maxdim) || (trunc &= MAK.truncrank(maxdim))
    isnothing(cutoff) || iszero(cutoff) || (trunc &= MAK.truncerror(; rtol = sqrt(cutoff), p = 2))
    return trunc
end

# Split `a`'s indices into (left, right) by the requested `linds`, matching by name:
# the caller's `Index` objects may carry the other endpoint's (dual) space on a graded
# backend, so `Index`-equality filtering silently drops them (see the name-matching
# note on the index-set algebra above). Both groups are `a`'s own indices, so the
# spaces handed to the factorization are `a`'s.
function _bipartition_inds(a::AbstractITensor, linds)
    lnames = name.(cat_inds(linds...))
    allinds = collect(inds(a))
    left = filter(i -> name(i) ∈ lnames, allinds)
    right = filter(i -> name(i) ∉ lnames, allinds)
    return left, right
end

# Legacy `factorize(a, linds...; ortho, cutoff, maxdim, tags)` splits `a` into `L * R`
# with the new bond between them. `linds` selects `L`'s indices (besides the bond) and
# may be passed splatted, as a single index, or as one collection. `ortho = "left"`
# makes `L` isometric, `"right"` makes `R` isometric. With no truncation this is a
# plain QR/LQ; with `cutoff`/`maxdim` it is a truncated SVD whose singular values are
# absorbed into the non-isometric factor. `tags`, if given, names the new bond.
function factorize(
        a::AbstractITensor,
        linds...;
        ortho = "left",
        cutoff = nothing,
        maxdim = nothing,
        tags = nothing
    )
    left, right = _bipartition_inds(a, linds)
    notruncation = (isnothing(cutoff) || iszero(cutoff)) && isnothing(maxdim)
    if notruncation
        if ortho == "left"
            L, R = MAK.qr_compact(a, Tuple(left), Tuple(right))
        elseif ortho == "right"
            L, R = MAK.lq_compact(a, Tuple(left), Tuple(right))
        else
            error(
                "compat `factorize` supports ortho = \"left\" / \"right\" (got $(repr(ortho)))"
            )
        end
    else
        U, S, Vt = MAK.svd_trunc(a, Tuple(left), Tuple(right); trunc = itensor_trunc(; cutoff, maxdim))
        if ortho == "left"
            L, R = U, S * Vt
        elseif ortho == "right"
            L, R = U * S, Vt
        else
            error(
                "compat `factorize` supports ortho = \"left\" / \"right\" (got $(repr(ortho)))"
            )
        end
    end
    if !isnothing(tags)
        b = only(commoninds(L, R))
        bnew = settags(b, tags)
        L, R = replaceind(L, b, bnew), replaceind(R, b, bnew)
    end
    return L, R
end



#
# Index fusion. The legacy `combiner` is retired (no compat shim); call sites fuse
# index groups with the next-gen `matricize(t, row_inds => row_name, col_inds =>
# col_name)` (minting each fused name via `uniquename(IndexName)`) or build a fused
# identity with `Base.one`, both of which are graded-capable. `matricize` is the
# `TensorAlgebra` generic, extended by ITensorBase for named tensors.
using TensorAlgebra: matricize

#
# Storage / element type accessors.
#
# `scalartype` is re-exported above. `datatype` is the underlying storage array type
# (used by `adapt`); `data` exposes the plain unnamed array. `array` densifies (legacy
# `array` materialized a dense array from any storage): a no-op on a dense backend,
# while graded / `TensorMap` storage converts through its canonical flat basis, so
# positions agree with `onehot` / `project` on the same axes.
datatype(T::AbstractITensor) = typeof(unnamed(T))
array(T::AbstractITensor) = convert(Array, unnamed(T))
data(T::AbstractITensor) = unnamed(T)

# TYPE PIRACY (temporary, compat-owned — NOT upstream): extends `Adapt.adapt_structure` for
# `AbstractITensor` with a `Number` eltype target, reproducing legacy `adapt(eltype)(t)` scalar-type
# conversion (ITensorBase's Adapt integration adapts storage/device but leaves the eltype alone).
# Using Adapt for eltype conversion is an abuse, so it does not belong upstream; retired once the
# call sites move to a different pattern.
function Adapt.adapt_structure(::Type{elt}, T::AbstractITensor) where {elt <: Number}
    # Short-circuit the no-op case: a non-`AbstractArray` backend (e.g. a `TensorMap`)
    # has no `convert(AbstractArray{elt}, ...)` method, but needs none when the element
    # type already matches.
    eltype(T) === elt && return T
    return nameddims(convert(AbstractArray{elt}, unnamed(T)), ITensorBase.dimnames(T))
end

# `swapind`: swap two indices (legacy convenience over `replaceinds`).
swapind(T::AbstractITensor, i::Index, j::Index) = replaceinds(T, i => j, j => i)

# Dense no-ops. Legacy QN-storage helpers; on the dense next-gen backend the tensor
# is already dense, so these are identities. (Graded/QN path is a stack gap.)
denseblocks(T::AbstractITensor) = T
dense(T::AbstractITensor) = T
# Whether a tensor carries quantum-number (graded) block structure: true when any of its
# indices is graded. `loopcorrection` branches on this to pick a contraction-order
# algorithm. A graded axis differs from its conjugate (conjugation flips the sector
# arrows) while a dense axis is self-conjugate — a dependency-free discriminator that needs
# no GradedArrays import and hardcodes no dense range type.
hasqns(i::Index) = conj(unnamed(i)) != unnamed(i)
hasqns(t::AbstractITensor) = any(hasqns, inds(t))
hasqns(::Any) = false

# The operator / named-state system (`op` / `state`) is vendored separately in
# `ops.jl`, included right after this file by the module file.

# Direct sum (legacy `directsum`): block-diagonal placement of tensors along the summed axes, shared
# axes preserved. Vendored densely — the next-gen stack has no `directsum` yet (tracked upstream).
#   directsum(out_inds, (t1 => summed_inds1), (t2 => summed_inds2), ...)
function directsum(out_inds, pairs::Pair...)
    out_inds = Tuple(out_inds)
    t1, s1 = first(pairs[1]), Tuple(last(pairs[1]))
    shared = Tuple(filter(i -> !(i in s1), collect(inds(t1))))
    target = (shared..., out_inds...)
    out = zeros(scalartype(t1), length.(target))
    offsets = zeros(Int, length(out_inds))
    for p in pairs
        t, sinds = first(p), Tuple(last(p))
        order = (shared..., sinds...)
        cur = collect(ITensorBase.dimnames(t))
        perm = [findfirst(==(name(o)), cur) for o in order]
        a = permutedims(unnamed(t), perm)
        ranges = (
            Base.OneTo.(length.(shared))...,
            ntuple(k -> (offsets[k] + 1):(offsets[k] + length(sinds[k])), length(sinds))...,
        )
        out[ranges...] .= a
        offsets .+= length.(sinds)
    end
    return out[target...]
end

# No index-order warnings in the next-gen stack (legacy `disable_warn_order`).
disable_warn_order(args...) = nothing

#
# Algorithm dispatch tag (legacy `Algorithm` / `@Algorithm_str`). `Algorithm"exact"`
# is the type `Algorithm{:exact}` (usable in `::Algorithm"exact"` signatures);
# `Algorithm("exact"; kwargs...)` constructs an instance carrying keyword options.
# Faithful minimal copy of the ITensors/NDTensors helper.
struct Algorithm{Alg, Kwargs <: NamedTuple}
    kwargs::Kwargs
end
Algorithm{Alg}(kwargs::NamedTuple) where {Alg} = Algorithm{Alg, typeof(kwargs)}(kwargs)
Algorithm{Alg}(; kwargs...) where {Alg} = Algorithm{Alg}((; kwargs...))
Algorithm(alg::Symbol; kwargs...) = Algorithm{alg}(; kwargs...)
Algorithm(alg::AbstractString; kwargs...) = Algorithm(Symbol(alg); kwargs...)
Algorithm(alg::Algorithm) = alg
function Base.getproperty(alg::Algorithm, name::Symbol)
    return if name === :kwargs
        getfield(alg, :kwargs)
    else
        getfield(getfield(alg, :kwargs), name)
    end
end
algorithm_name(::Algorithm{Alg}) where {Alg} = Alg
macro Algorithm_str(s)
    return :(Algorithm{$(Expr(:quote, Symbol(s)))})
end

#
# Tags. Legacy ITensors uses a flat tag set (`Index(dim, "S=1/2,Site")`); ITensorBase
# stores tags as a `Dict{String, String}`. For the legacy flat-tag usage TNQS has
# (site-type labels, link names), we store each token as a keyed tag with empty
# value, and `hastags` checks membership. (A fuller tag-compat story is a follow-up.)
function settags(i::Index, tagstr::AbstractString)
    for t in split(tagstr, ",")
        s = String(strip(t))
        isempty(s) || (i = ITensorBase.settag(i, s, ""))
    end
    return i
end
# Copy a whole tag dictionary onto an index (legacy `settags(i, tags(j))`, used when a
# factorization is asked to give its new bond the same tags as an existing bond).
function settags(i::Index, d::AbstractDict)
    for (k, v) in d
        i = ITensorBase.settag(i, k, v)
    end
    return i
end
# TYPE PIRACY (temporary, compat-owned — NOT upstream): the two methods below add `Index`
# constructors taking a tag string / tag dict (legacy positional-tag form; next-gen passes tags via
# the `tags` kwarg). Retired once the call sites move to the kwarg form. Only the `dim` first-argument
# form is pirated: a range/space first argument collides with ITensorBase's own two-argument `Index`
# constructors, so tagging an axis-minted index spells the two steps (`settags(Index(r), "tag")`).
ITensorBase.Index(dim::Integer, tagstr::AbstractString) = settags(Index(dim), tagstr)
# Build a fresh index carrying a tag dictionary (legacy `Index(dim, tags(i))`, where
# the next-gen `tags` returns a `Dict{String, String}`).
function ITensorBase.Index(dim::Integer, tags::AbstractDict)
    i = Index(dim)
    for (k, v) in tags
        i = ITensorBase.settag(i, k, v)
    end
    return i
end
function hastags(i::Index, tagstr::AbstractString)
    return all(
        haskey(tags(i), String(strip(t))) for t in split(tagstr, ",") if !isempty(strip(t))
    )
end

#
# Owned `exp` (avoids type piracy: `AbstractITensor` is not ours, so we do not add a
# `Base.exp` method for it). Legacy `ITensors.exp(::ITensor)` exponentiates an operator
# ITensor over its `(i, prime(i))` index pairs by forwarding to ITensorBase's
# graded-capable matricization `Base.exp(a, codomain, domain)`. Other arguments forward
# to `Base.exp`.
exp(x) = Base.exp(x)
function exp(t::AbstractITensor)
    p0 = filter(i -> ITensorBase.plev(i) == 0, collect(inds(t)))
    isempty(p0) && error("exp(::ITensor) expects indices paired as (i, prime(i))")
    p1 = map(ITensorBase.prime, p0)
    return Base.exp(t, Tuple(p1), Tuple(p0))
end

# TODO (small inline residue — can't be a drop-in shim):
#   - `contract` / `inner` / `truncate`: TNQS *extends* these (method definitions),
#     so the call sites drop the `ITensors.` qualifier to extend the generics this
#     module owns rather than ITensors'. (`truncate` clashes with `Base.truncate`.)
#
# TODO (stack gaps — own roadmap items, not solvable in this file):
#   - op / @OpName_str / @SiteType_str  (operator/site system)
#   - MPS / MPO / boundary-MPS          (no next-gen MPS layer yet)
#   - hasqns / QN-aware storage         (next-gen symmetry via GradedArrays)
