#=
KTensors: the TensorKit-stack tensor backend, behind the TensorInterface seam.

A `KTensor` is a named-index tensor: a dense N-d array plus a vector of `KIndex` labels.
Index identity (id, plev) — not position — drives contraction, exactly like the ITensor
semantics the library is written against. Contraction is executed by TensorOperations
(the engine underneath TensorKit) with dynamically generated labels; factorizations are
LAPACK via LinearAlgebra, matching the ITensors conventions pinned down empirically
(see examples/benchmark_bp_square.jl's agreement digest and test/test_ktensors.jl).

Versioning plan (deliberate, incremental):
  v1 (this file): dense `Array` data. Proves the seam, matches ITensors numerics, and
      gives TensorOperations' allocator hooks for the later allocation work.
  v2: `data` becomes a TensorKit `TensorMap` over graded spaces — `KIndex.dual` is already
      maintained by `dag` so bond orientations are in place; symmetric/fermionic sectors
      enter through the space type without touching the seam or call sites.
  v3: workspace-reusing kernels for the BP message update (target ≤ 2F + out) and
      MatrixAlgebraKit in-place factorizations.

Not yet implemented (error clearly if hit): combiner, directsum, boundary-MPS specific
paths. The gate/operator library covers S=1/2 with the gate registry's conventions.
=#
module KTensors

using LinearAlgebra: LinearAlgebra, Hermitian, Diagonal, norm, mul!, diag, rmul!, BlasFloat
using MatrixAlgebraKit: MatrixAlgebraKit, qr_compact, qr_compact!, svd_compact, svd_trunc,
    eigh_full, truncrank, truncerror
using TensorOperations: TensorOperations, ncon
using Adapt: Adapt, adapt
using ..TensorInterface: TensorInterface

export KIndex, KTensor

# ── Index ───────────────────────────────────────────────────────────────────────────────

struct KIndex
    id::UInt64
    d::Int
    plev::Int
    tags::String
    dual::Bool   # bond orientation; bookkeeping for the graded v2, inert for dense data
end

KIndex(d::Integer, tags::AbstractString = "") = KIndex(rand(UInt64), Int(d), 0, String(tags), false)

# Identity is (id, plev): `dag` (dual flip) and tag changes never change which index this is.
Base.:(==)(a::KIndex, b::KIndex) = a.id == b.id && a.plev == b.plev
Base.hash(i::KIndex, h::UInt) = hash((i.id, i.plev), h)
Base.adjoint(i::KIndex) = TensorInterface.prime(i)
Base.copy(i::KIndex) = i

function Base.show(io::IO, i::KIndex)
    print(io, "(d=", i.d, "|id=", repr(i.id % UInt16), "|\"", i.tags, "\")", "'"^i.plev, i.dual ? "†" : "")
    return nothing
end

TensorInterface.dim(i::KIndex) = i.d
TensorInterface.plev(i::KIndex) = i.plev
TensorInterface.tags(i::KIndex) = i.tags
TensorInterface.dag(i::KIndex) = KIndex(i.id, i.d, i.plev, i.tags, !i.dual)
TensorInterface.prime(i::KIndex, n::Integer = 1) = KIndex(i.id, i.d, i.plev + n, i.tags, i.dual)
TensorInterface.noprime(i::KIndex) = KIndex(i.id, i.d, 0, i.tags, i.dual)
TensorInterface.sim(i::KIndex) = KIndex(rand(UInt64), i.d, i.plev, i.tags, i.dual)

for f in [:dag, :prime, :noprime, :sim]
    @eval TensorInterface.$f(is::AbstractVector{KIndex}, args...) = map(i -> TensorInterface.$f(i, args...), is)
end

TensorInterface.new_index(::Union{KIndex, AbstractVector{KIndex}}, d::Integer; tags = "") = KIndex(d, tags)

# ── Tensor ──────────────────────────────────────────────────────────────────────────────

struct KTensor{T, N, A <: AbstractArray{T, N}}
    inds::Vector{KIndex}
    data::A
    function KTensor(inds::Vector{KIndex}, data::A) where {T, N, A <: AbstractArray{T, N}}
        length(inds) == N || error("KTensor: $(length(inds)) indices for a rank-$N array")
        all(i -> i.d == size(data, findfirst(==(i), inds)), unique(inds)) ||
            error("KTensor: index dimensions $(TensorInterface.dim.(inds)) don't match array size $(size(data))")
        return new{T, N, A}(inds, data)
    end
end

KTensor(x::Number) = KTensor(KIndex[], fill(x))

Base.eltype(::KTensor{T}) where {T} = T
Base.copy(t::KTensor) = KTensor(copy(t.inds), copy(t.data))
Base.sum(t::KTensor) = sum(t.data)
Base.ndims(::KTensor{T, N}) where {T, N} = N

function Base.show(io::IO, t::KTensor{T, N}) where {T, N}
    print(io, "KTensor{", T, "} inds: ", t.inds)
    return nothing
end

TensorInterface.inds(t::KTensor) = t.inds
TensorInterface.scalartype(::KTensor{T}) where {T} = T
TensorInterface.datatype(::KTensor{T, N, A}) where {T, N, A <: Array} = Vector{T}
TensorInterface.hasqns(::KTensor) = false
TensorInterface.hasqns(::KIndex) = false
TensorInterface.dense(t::KTensor) = t
TensorInterface.denseblocks(t::KTensor) = t
TensorInterface.array(t::KTensor) = t.data
TensorInterface.data(t::KTensor) = vec(t.data)
TensorInterface.new_index(t::KTensor, d::Integer; tags = "") = KIndex(d, tags)

function TensorInterface.scalar(t::KTensor)
    length(t.data) == 1 || error("scalar: KTensor with inds $(t.inds) is not a scalar")
    return t.data[Base.firstindex(t.data)]
end

# ── Index-set queries ───────────────────────────────────────────────────────────────────

_indvec(t::KTensor) = t.inds
_indvec(i::KIndex) = KIndex[i]
_indvec(is::AbstractVector) = collect(KIndex, is)
_indvec(is::Tuple) = collect(KIndex, is)

const KIndsLike = Union{KTensor, KIndex, AbstractVector{KIndex}, Tuple{KIndex, Vararg{KIndex}}}

TensorInterface.commoninds(a::KIndsLike, b::KIndsLike) = filter(i -> i ∈ _indvec(b), _indvec(a))
# ITensors convention: the singular forms return the FIRST match (or nothing), not `only`.
function TensorInterface.commonind(a::KIndsLike, b::KIndsLike)
    cs = TensorInterface.commoninds(a, b)
    return isempty(cs) ? nothing : first(cs)
end
TensorInterface.uniqueinds(a::KIndsLike, b::KIndsLike) = filter(i -> i ∉ _indvec(b), _indvec(a))
TensorInterface.unioninds(a::KIndsLike, b::KIndsLike) = unique(vcat(_indvec(a), _indvec(b)))
function TensorInterface.noncommoninds(a::KIndsLike, b::KIndsLike)
    return vcat(TensorInterface.uniqueinds(a, b), TensorInterface.uniqueinds(b, a))
end
function TensorInterface.noncommonind(a::KIndsLike, b::KIndsLike)
    ns = TensorInterface.noncommoninds(a, b)
    return isempty(ns) ? nothing : first(ns)
end
TensorInterface.hascommoninds(a::KIndsLike, b::KIndsLike) = !isempty(TensorInterface.commoninds(a, b))

# ── Index transforms on tensors (relabeling only, no data movement) ─────────────────────

_mapinds(f, t::KTensor) = KTensor(map(f, t.inds), t.data)

TensorInterface.prime(t::KTensor, n::Integer = 1) = _mapinds(i -> TensorInterface.prime(i, n), t)
TensorInterface.noprime(t::KTensor) = _mapinds(TensorInterface.noprime, t)
TensorInterface.sim(t::KTensor) = _mapinds(TensorInterface.sim, t)
TensorInterface.dag(t::KTensor) = KTensor(map(TensorInterface.dag, t.inds), conj(t.data))

function TensorInterface.replaceinds(t::KTensor, old, new)
    oldv, newv = _indvec(old), _indvec(new)
    length(oldv) == length(newv) || error("replaceinds: length mismatch")
    newinds = map(t.inds) do i
        k = findfirst(==(i), oldv)
        if k === nothing
            i
        else
            n = newv[k]
            n.d == i.d || error("replaceinds: dimension mismatch $(i) → $(n)")
            n
        end
    end
    return KTensor(newinds, t.data)
end
TensorInterface.replaceind(t::KTensor, old::KIndex, new::KIndex) = TensorInterface.replaceinds(t, [old], [new])

# ── Construction ────────────────────────────────────────────────────────────────────────

TensorInterface.from_array(A::AbstractArray, is::KIndex...) = KTensor(collect(KIndex, is), reshape(complex_or_real_copy(A), TensorInterface.dim.(is)...))
TensorInterface.from_array(A::AbstractVector, i::KIndex) = KTensor(KIndex[i], copy(A))
complex_or_real_copy(A) = copy(A)

TensorInterface.random_itensor(elt::Type, is::AbstractVector{KIndex}) = KTensor(collect(is), randn(elt, TensorInterface.dim.(is)...))
TensorInterface.random_itensor(elt::Type, is::KIndex...) = TensorInterface.random_itensor(elt, collect(is))

function TensorInterface.onehot(elt::Type, p::Pair{KIndex, <:Integer})
    i, v = p
    data = zeros(elt, i.d)
    data[v] = one(elt)
    return KTensor(KIndex[i], data)
end
TensorInterface.onehot(p::Pair{KIndex, <:Integer}) = TensorInterface.onehot(Float64, p)

function TensorInterface.delta(elt::Type, is::AbstractVector{KIndex})
    isempty(is) && return KTensor(one(elt))
    data = zeros(elt, TensorInterface.dim.(is)...)
    for k in 1:minimum(TensorInterface.dim.(is))
        data[ntuple(_ -> k, length(is))...] = one(elt)
    end
    return KTensor(collect(is), data)
end
TensorInterface.delta(is::AbstractVector{KIndex}) = TensorInterface.delta(Float64, is)
TensorInterface.delta(elt::Type, is::KIndex...) = TensorInterface.delta(elt, collect(is))
TensorInterface.delta(is::KIndex...) = TensorInterface.delta(Float64, collect(is))

TensorInterface.combiner(is::KIndex...; kwargs...) = error("combiner: not yet implemented for the KTensors backend")
TensorInterface.combiner(is::AbstractVector{KIndex}; kwargs...) = error("combiner: not yet implemented for the KTensors backend")
TensorInterface.directsum(args::Pair{<:KTensor}...; kwargs...) = error("directsum: not yet implemented for the KTensors backend")

# ── Arithmetic, contraction ─────────────────────────────────────────────────────────────

Base.:*(t::KTensor, x::Number) = KTensor(copy(t.inds), t.data * x)
Base.:*(x::Number, t::KTensor) = t * x
Base.:/(t::KTensor, x::Number) = t * inv(x)

# Permute `b`'s data into `a`'s index order.
function _align(a::KTensor, b::KTensor)
    a.inds == b.inds && return b.data
    perm = map(i -> findfirst(==(i), b.inds), a.inds)
    any(isnothing, perm) && error("tensors have different index sets: $(a.inds) vs $(b.inds)")
    return permutedims(b.data, perm)
end

Base.:+(a::KTensor, b::KTensor) = KTensor(copy(a.inds), a.data + _align(a, b))
Base.:-(a::KTensor, b::KTensor) = KTensor(copy(a.inds), a.data - _align(a, b))

LinearAlgebra.norm(t::KTensor) = norm(t.data)
LinearAlgebra.normalize(t::KTensor) = t * inv(norm(t))
LinearAlgebra.dot(a::KTensor, b::KTensor) = LinearAlgebra.dot(vec(a.data), vec(_align(a, b)))
LinearAlgebra.tr(t::KTensor) = _trace_all(t)

# Pairwise contraction over all common (id, plev) indices; repeated indices on one tensor
# (traces) are supported through the same ncon labeling.
function Base.:*(a::KTensor, b::KTensor)
    # scalar fast paths
    ndims(a) == 0 && return KTensor(copy(b.inds), TensorInterface.scalar(a) * b.data)
    ndims(b) == 0 && return KTensor(copy(a.inds), TensorInterface.scalar(b) * a.data)

    common = TensorInterface.commoninds(a, b)
    aopen = TensorInterface.uniqueinds(a, b)
    bopen = TensorInterface.uniqueinds(b, a)

    nextlabel = Ref(0)
    labels = Dict{KIndex, Int}()
    for i in common
        labels[i] = (nextlabel[] += 1)
    end
    openlabel = Ref(0)
    outinds = vcat(aopen, bopen)
    for i in outinds
        labels[i] = -(openlabel[] += 1)
    end

    la = Int[labels[i] for i in a.inds]
    lb = Int[labels[i] for i in b.inds]

    out = ncon([a.data, b.data], [la, lb])
    data = out isa Number ? fill(out) : out
    return KTensor(outinds, data)
end

function _trace_all(t::KTensor)
    # contract each plev-1 index with its plev-0 partner (same id), matching ITensors tr
    lo = filter(i -> i.plev == 0, t.inds)
    hi = filter(i -> i.plev == 1, t.inds)
    labels = Dict{KIndex, Int}()
    n = 0
    for i in lo
        j = findfirst(x -> x.id == i.id, hi)
        j === nothing && error("tr: no primed partner for index $(i)")
        labels[i] = (n += 1)
        labels[hi[j]] = n
    end
    l = Int[labels[i] for i in t.inds]
    out = ncon([t.data], [l])
    return out isa Number ? out : sum(out)
end

function TensorInterface.contract(ts::Vector{<:KTensor}; sequence = nothing, kwargs...)
    isnothing(sequence) && return reduce(*, ts)
    return _contract_seq(ts, sequence)
end
_contract_seq(ts::Vector{<:KTensor}, s::Integer) = ts[s]
function _contract_seq(ts::Vector{<:KTensor}, s::Union{Vector, Tuple})
    return mapreduce(x -> _contract_seq(ts, x), *, s)
end

TensorInterface.apply(o::KTensor, t::KTensor) = TensorInterface.noprime(o * t)

# ── Diagonal operations ─────────────────────────────────────────────────────────────────

function TensorInterface.map_diag(f::Function, t::KTensor)
    out = copy(t)
    TensorInterface.map_diag!(f, out, out)
    return out
end

function TensorInterface.map_diag!(f::Function, out::KTensor, t::KTensor)
    ndims(t) == 2 || error("map_diag: expected a 2-index KTensor")
    d = _align(out, t)
    d === out.data || copyto!(out.data, d)
    for k in 1:minimum(size(out.data))
        out.data[k, k] = f(out.data[k, k])
    end
    return out
end

# ── Adapt / storage ─────────────────────────────────────────────────────────────────────

Adapt.adapt_structure(elt::Type{<:Number}, t::KTensor) = KTensor(copy(t.inds), convert(Array{elt}, t.data))
function Adapt.adapt_structure(to::Type{<:AbstractVector}, t::KTensor)
    return KTensor(copy(t.inds), reshape(adapt(to, vec(copy(t.data))), size(t.data)))
end
Adapt.adapt_structure(to, t::KTensor) = KTensor(copy(t.inds), adapt(to, t.data))

# ── Factorizations (MatrixAlgebraKit; in-place/preallocated variants are the v3 target) ─

struct KSpectrum
    eigs::Vector{Float64}
    truncerr::Float64
end

# Permute+reshape to a (linds...) × (rest...) matrix. Returns matrix, left inds, right inds.
function _matricize(t::KTensor, linds::Vector{KIndex})
    rinds = TensorInterface.uniqueinds(t, linds)
    perm = map(i -> findfirst(==(i), t.inds), vcat(linds, rinds))
    any(isnothing, perm) && error("factorize: indices $(linds) not all found on tensor $(t.inds)")
    A = permutedims(t.data, perm)
    dl = prod(Int[i.d for i in linds]; init = 1)
    dr = prod(Int[i.d for i in rinds]; init = 1)
    return reshape(A, dl, dr), linds, rinds
end

# ITensors-convention truncation as a MatrixAlgebraKit strategy: keep the smallest set of
# singular values whose discarded Σs² fraction is ≤ cutoff (⇔ 2-norm rtol = √cutoff),
# capped at maxdim. `nothing` means no constraint.
function _mak_trunc(; maxdim = nothing, cutoff = nothing)
    strategies = Any[]
    isnothing(maxdim) || push!(strategies, truncrank(Int(maxdim)))
    isnothing(cutoff) || push!(strategies, truncerror(; rtol = sqrt(cutoff), p = 2))
    return isempty(strategies) ? nothing : reduce(&, strategies)
end

function LinearAlgebra.qr(t::KTensor, linds; kwargs...)
    A, li, ri = _matricize(t, _indvec(linds))
    Q, R = qr_compact(A)
    b = KIndex(size(Q, 2), "Link,qr")
    Qt = KTensor(vcat(li, [b]), reshape(Q, (Int[i.d for i in li]..., size(Q, 2))))
    Rt = KTensor(vcat([b], ri), reshape(R, (size(R, 1), Int[i.d for i in ri]...)))
    return Qt, Rt
end

# Matrix-level truncated SVD with the ITensors truncerr convention (discarded Σs²/total).
function _svd_matrix(A::AbstractMatrix; maxdim = nothing, cutoff = nothing)
    trunc = _mak_trunc(; maxdim, cutoff)
    if trunc === nothing
        U, S, Vt = svd_compact(A)
        truncerr = 0.0
    else
        U, S, Vt, err = svd_trunc(A; trunc)
        # MAK reports ‖discarded‖₂; the ITensors convention is discarded Σs² / total Σs².
        kept2 = sum(abs2, diag(S))
        total = kept2 + err^2
        truncerr = total > 0 ? err^2 / total : 0.0
    end
    return U, diag(S), Vt, truncerr
end

function _svd_split(t::KTensor, linds::Vector{KIndex}; maxdim = nothing, cutoff = nothing, mindim = 1)
    mindim > 1 && error("_svd_split: mindim > 1 is not supported by the KTensors backend yet")
    A, li, ri = _matricize(t, linds)
    U, S, Vt, truncerr = _svd_matrix(A; maxdim, cutoff)
    return U, S, Vt, li, ri, truncerr
end

"""
factorize_svd matching the ITensors convention used by `simple_update`:
`ortho = "none"` returns F1 = U√S and F2 = √S·V sharing the primed bond `u′`, with the
singular values reported on unprimed `(u, v)`; `spec.eigs` are the kept s².
"""
function TensorInterface.factorize_svd(
        t::KTensor, linds;
        ortho = "none", singular_values! = nothing,
        maxdim = nothing, cutoff = nothing, mindim = 1, tags = nothing, kwargs...,
    )
    U, S, Vt, li, ri, truncerr = _svd_split(t, _indvec(linds); maxdim, cutoff, mindim)
    k = length(S)
    T = eltype(U)
    u = KIndex(k, "Link,u")
    v = KIndex(k, "Link,v")
    up = TensorInterface.prime(u)

    if ortho == "none"
        sq = sqrt.(S)
        F1m = U .* reshape(T.(sq), 1, k)
        F2m = reshape(T.(sq), k, 1) .* Vt
        F1 = KTensor(vcat(li, [up]), reshape(F1m, (Int[i.d for i in li]..., k)))
        F2 = KTensor(vcat([up], ri), reshape(F2m, (k, Int[i.d for i in ri]...)))
    elseif ortho == "left"
        F1 = KTensor(vcat(li, [up]), reshape(U, (Int[i.d for i in li]..., k)))
        F2 = KTensor(vcat([up], ri), reshape(Diagonal(T.(S)) * Vt, (k, Int[i.d for i in ri]...)))
    elseif ortho == "right"
        F1 = KTensor(vcat(li, [up]), reshape(U * Diagonal(T.(S)), (Int[i.d for i in li]..., k)))
        F2 = KTensor(vcat([up], ri), reshape(Vt, (k, Int[i.d for i in ri]...)))
    else
        error("factorize_svd: unknown ortho = $ortho")
    end

    if singular_values! !== nothing
        singular_values![] = KTensor(KIndex[u, v], Matrix(Diagonal(S)))
    end
    return F1, F2, KSpectrum(abs2.(S), truncerr)
end

# ITensors-style factorize: L isometric for ortho="left" (L=U, R=SV), mirrored for "right".
# The bond is unprimed and carries `tags`.
function LinearAlgebra.factorize(
        t::KTensor, linds...;
        ortho = "left", maxdim = nothing, cutoff = nothing, mindim = 1, tags = "Link,fact", kwargs...,
    )
    lv = length(linds) == 1 ? _indvec(only(linds)) : collect(KIndex, linds)
    U, S, Vt, li, ri, _ = _svd_split(t, lv; maxdim, cutoff, mindim)
    k = length(S)
    T = eltype(U)
    b = KIndex(k, String(tags))
    if ortho == "left"
        L = KTensor(vcat(li, [b]), reshape(U, (Int[i.d for i in li]..., k)))
        R = KTensor(vcat([b], ri), reshape(Diagonal(T.(S)) * Vt, (k, Int[i.d for i in ri]...)))
    elseif ortho == "right"
        L = KTensor(vcat(li, [b]), reshape(U * Diagonal(T.(S)), (Int[i.d for i in li]..., k)))
        R = KTensor(vcat([b], ri), reshape(Vt, (k, Int[i.d for i in ri]...)))
    else
        error("factorize: unknown ortho = $ortho")
    end
    return L, R
end

# ITensors-style svd: U(linds, u), S(u, v), V(rinds, v).
function LinearAlgebra.svd(t::KTensor, linds; maxdim = nothing, cutoff = nothing, mindim = 1, kwargs...)
    U, S, Vt, li, ri, _ = _svd_split(t, _indvec(linds); maxdim, cutoff, mindim)
    k = length(S)
    u = KIndex(k, "Link,u")
    v = KIndex(k, "Link,v")
    Ut = KTensor(vcat(li, [u]), reshape(U, (Int[i.d for i in li]..., k)))
    St = KTensor(KIndex[u, v], Matrix(Diagonal(S)))
    Vt_ = KTensor(vcat(ri, [v]), reshape(copy(transpose(Vt)), (Int[i.d for i in ri]..., k)))
    return Ut, St, Vt_
end

# eigen matching the ITensors hermitian convention probed empirically:
# D on (link′, link) with the eigenvalues, U on (rinds..., link); Ul·D·dag(U) reconstructs.
function LinearAlgebra.eigen(t::KTensor, linds, rinds; ishermitian::Bool = false, kwargs...)
    ishermitian || error("eigen: only ishermitian = true is implemented for KTensors")
    lv, rv = _indvec(linds), _indvec(rinds)
    A, li, ri = _matricize(t, lv)
    ri == rv || error("eigen: rinds don't match the remaining indices")
    D, vecs = eigh_full((A + A') / 2)
    vals = diag(D)
    k = length(vals)
    lk = KIndex(k, "Link,eigen")
    D = KTensor(KIndex[TensorInterface.prime(lk), lk], Matrix(Diagonal(vals)))
    U = KTensor(vcat(rv, [lk]), reshape(vecs, (Int[i.d for i in rv]..., k)))
    return D, U
end

# ── Fused BP norm-message kernel ────────────────────────────────────────────────────────
#
# The double-layer message update m_out = (∏ m_in) · ψ · conj(ψ̃) computed with:
#   * sequential absorption of each incoming message into the ket (F-sized GEMMs),
#   * a closing GEMM against ψ itself with the conjugation fused into the BLAS call —
#     `dag(prime(ψ))` is never materialized,
#   * all chain intermediates (and TensorOperations' internal permute scratch) living in a
#     task-local, reusable BufferAllocator; only the small outgoing message touches the heap.
#
# Steady-state heap cost per call: the output message + O(1) bookkeeping — the "2F + out"
# target, with the 2F living in the reused buffer.

function _bp_buffer()
    return get!(task_local_storage(), :ktensors_bp_buffer) do
        TensorOperations.BufferAllocator()
    end::TensorOperations.BufferAllocator
end

# One pairwise contraction at the data level. `pairfn` maps an index of `b` to the index of
# `a` it should contract with (identity for plain absorption; the site/prime partner map for
# the closing bra contraction). Returns (data, out_inds).
function _tc_pair(
        da, ia::Vector{KIndex}, db, ib::Vector{KIndex}, conjB::Bool, pairfn,
        istemp::Val, allocator
    )
    ca, cb = Int[], Int[]
    for (j, bj) in enumerate(ib)
        p = pairfn(bj)
        i = findfirst(==(p), ia)
        i === nothing && continue
        push!(ca, i)
        push!(cb, j)
    end
    oa = setdiff(1:length(ia), ca)
    ob = setdiff(1:length(ib), cb)
    pA = (Tuple(oa), Tuple(ca))
    pB = (Tuple(cb), Tuple(ob))
    pAB = (Tuple(1:(length(oa) + length(ob))), ())
    TC = promote_type(eltype(da), eltype(db))
    C = TensorOperations.tensoralloc_contract(TC, da, pA, false, db, pB, conjB, pAB, istemp, allocator)
    backend = TC <: BlasFloat ? TensorOperations.StridedBLAS() : TensorOperations.StridedNative()
    TensorOperations.tensorcontract!(C, da, pA, false, db, pB, conjB, pAB, one(TC), zero(TC), backend, allocator)
    return C, ia[oa], ib[ob]
end

# Pattern check: `m` must be a standard doubled norm-network message for `ψ` — every index
# either a plev-0 index of ψ (ket side) or the prime of one (bra side).
function _is_norm_message(m::KTensor, ψinds::Vector{KIndex})
    return all(m.inds) do i
        (i.plev == 0 && i ∈ ψinds) || (i.plev == 1 && TensorInterface.noprime(i) ∈ ψinds)
    end
end

"""
Fused ket-side closure of vertex tensor `ψ` (site indices `sinds`) against its own
conjugate, given standard doubled `incoming` messages and an optional single-site operator
`op` inserted between the layers. ψ-legs without a partner (e.g. the target edge of a
message update) come out as an unprimed/primed pair; a fully surrounded vertex closes to a
scalar (0-index) tensor. Returns `nothing` when the structure doesn't match.
"""
function fused_norm_closure(
        ψ::KTensor, sinds::Vector{KIndex}, incoming::Vector{<:KTensor};
        op::Union{Nothing, KTensor} = nothing
    )
    ψ.data isa Array || return nothing
    all(m -> m.data isa Array && _is_norm_message(m, ψ.inds), incoming) || return nothing
    op === nothing || op.data isa Array || return nothing

    buf = _bp_buffer()
    cp = TensorOperations.allocator_checkpoint!(buf)
    X, ix = ψ.data, ψ.inds
    for m in incoming
        X, oa, ob = _tc_pair(X, ix, m.data, m.inds, false, identity, Val(true), buf)
        ix = vcat(oa, ob)
    end
    covered = op === nothing ? KIndex[] : KIndex[i for i in op.inds if i.plev == 0]
    if op !== nothing
        X, oa, ob = _tc_pair(X, ix, op.data, op.inds, false, identity, Val(true), buf)
        ix = vcat(oa, ob)
    end
    # Closing: ψ-leg i pairs with X-leg i (uncovered sites, ket↔bra direct) or prime(i)
    # (message-bridged virtuals and operator-covered sites); unpaired ψ-legs come out as an
    # (i, i′) pair. istemp = Val(false) puts the output on the heap while the internal
    # permute scratch of the contraction still lives in the buffer.
    partner = i -> (i ∈ sinds && i ∉ covered) ? i : TensorInterface.prime(i)
    out, o_ket, o_bra = _tc_pair(X, ix, ψ.data, ψ.inds, true, partner, Val(false), buf)
    oinds = vcat(o_ket, TensorInterface.prime.(o_bra))
    TensorOperations.allocator_reset!(buf, cp)

    return KTensor(oinds, out)
end

"""
Fused computation of the outgoing BP message from vertex tensor `ψ` (site indices `sinds`)
given standard doubled `incoming` messages. Returns `nothing` when the structure doesn't
match (caller falls back to the generic contraction path).
"""
function fused_norm_message(
        ψ::KTensor, sinds::Vector{KIndex}, incoming::Vector{<:KTensor};
        normalize::Bool = true
    )
    m = fused_norm_closure(ψ, sinds, incoming)
    m === nothing && return nothing
    if normalize
        s = sum(m.data)
        iszero(s) || rmul!(vec(m.data), inv(s))
    end
    return m
end

# ── Fused two-site simple-update gate kernel ────────────────────────────────────────────
#
# The BP-gauged simple update with the same buffer discipline as the message kernel:
# √env absorption, the matricize copies, Q factors and the de-gauging chain all live in the
# task-local buffer; `dag` of the inverse-√ environments is fused as a conj flag. Only the
# two updated site tensors and the singular-value tensor escape to the heap.

# Buffered permute+reshape to a (linds × rest) matrix. Returns (matrix, linds, rinds).
function _matricize_temp(da, ia::Vector{KIndex}, linds::Vector{KIndex}, buf)
    rinds = filter(i -> i ∉ linds, ia)
    perm = map(i -> findfirst(==(i), ia), vcat(linds, rinds))
    T = eltype(da)
    dims = ntuple(k -> size(da, perm[k]), ndims(da))
    C = TensorOperations.tensoralloc(Array{T, ndims(da)}, dims, Val(true), buf)
    TensorOperations.tensoradd!(C, da, (Tuple(perm), ()), false, one(T), zero(T))
    dl = prod(Int[i.d for i in linds]; init = 1)
    return reshape(C, dl, :), linds, rinds
end

# Buffered thin QR; destroys Amat (which is itself a buffer temp).
function _qr_temp(Amat::AbstractMatrix, buf)
    m, n = size(Amat)
    k = min(m, n)
    T = eltype(Amat)
    Q = TensorOperations.tensoralloc(Matrix{T}, (m, k), Val(true), buf)
    R = TensorOperations.tensoralloc(Matrix{T}, (k, n), Val(true), buf)
    qr_compact!(Amat, (Q, R))
    return Q, R
end

# Absorb a chain of 2-index gauge tensors into (X, ix) as buffer temps; conjB fuses dag.
function _absorb_chain(X, ix::Vector{KIndex}, ms::Vector{<:KTensor}, conjB::Bool, buf)
    for m in ms
        X, oa, ob = _tc_pair(X, ix, m.data, m.inds, conjB, identity, Val(true), buf)
        ix = vcat(oa, ob)
    end
    return X, ix
end

function fused_two_site_gate(
        o::KTensor, ψ1::KTensor, ψ2::KTensor,
        sqrt1::Vector{<:KTensor}, inv1::Vector{<:KTensor},
        sqrt2::Vector{<:KTensor}, inv2::Vector{<:KTensor},
        s1::Vector{KIndex}, s2::Vector{KIndex};
        maxdim = nothing, cutoff = nothing,
    )
    buf = _bp_buffer()
    cp = TensorOperations.allocator_checkpoint!(buf)

    # Gauge: X_k = ψ_k · ∏ √env (buffer temps)
    X1, ix1 = _absorb_chain(ψ1.data, ψ1.inds, sqrt1, false, buf)
    X2, ix2 = _absorb_chain(ψ2.data, ψ2.inds, sqrt2, false, buf)

    # QR split: Q side = gauged environment legs (not shared with partner, not gate sites)
    ql1 = filter(i -> i ∉ ψ2.inds && i ∉ s1, ix1)
    ql2 = filter(i -> i ∉ ψ1.inds && i ∉ s2, ix2)
    A1, l1, r1 = _matricize_temp(X1, ix1, ql1, buf)
    A2, l2, r2 = _matricize_temp(X2, ix2, ql2, buf)
    Q1m, R1m = _qr_temp(A1, buf)
    Q2m, R2m = _qr_temp(A2, buf)
    b1 = KIndex(size(Q1m, 2), "Link,qr")
    b2 = KIndex(size(Q2m, 2), "Link,qr")
    iR1 = vcat([b1], r1)
    iR2 = vcat([b2], r2)
    R1 = reshape(R1m, (b1.d, Int[i.d for i in r1]...))
    R2 = reshape(R2m, (b2.d, Int[i.d for i in r2]...))

    # oR = noprime(o · (R1 · R2)) — small, buffer temps
    RR, oa, ob = _tc_pair(R1, iR1, R2, iR2, false, identity, Val(true), buf)
    iRR = vcat(oa, ob)
    oR, oa, ob = _tc_pair(RR, iRR, o.data, o.inds, false, identity, Val(true), buf)
    ioR = map(vcat(oa, ob)) do i
        i.plev == 1 && (TensorInterface.noprime(i) ∈ s1 || TensorInterface.noprime(i) ∈ s2) ?
            TensorInterface.noprime(i) : i
    end

    # SVD across the (b1, s1) | (b2, s2) cut
    AoR, lo, ro = _matricize_temp(oR, ioR, vcat([b1], s1), buf)
    U, S, Vt, truncerr = _svd_matrix(AoR; maxdim, cutoff)
    k = length(S)
    u = KIndex(k, "Link,u")
    v = KIndex(k, "Link,v")
    up = TensorInterface.prime(u)
    TS = promote_type(eltype(U), eltype(S))
    sq = sqrt.(S)
    F1 = reshape(U .* reshape(TS.(sq), 1, k), (Int[i.d for i in lo]..., k))
    F2 = reshape(reshape(TS.(sq), k, 1) .* Vt, (k, Int[i.d for i in ro]...))
    iF1 = vcat(lo, [up])
    iF2 = vcat([up], ro)

    # De-gauge: Y_k = Q_k · ∏ conj(inv-√env), the dag fused as a conj flag
    iQ1 = vcat(l1, [b1])
    iQ2 = vcat(l2, [b2])
    Y1, iy1 = _absorb_chain(reshape(Q1m, (Int[i.d for i in l1]..., b1.d)), iQ1, inv1, true, buf)
    Y2, iy2 = _absorb_chain(reshape(Q2m, (Int[i.d for i in l2]..., b2.d)), iQ2, inv2, true, buf)

    # Reassemble; these two escape to the heap
    T1, oa, ob = _tc_pair(Y1, iy1, F1, iF1, false, identity, Val(false), buf)
    iT1 = vcat(oa, ob)
    T2, oa, ob = _tc_pair(Y2, iy2, F2, iF2, false, identity, Val(false), buf)
    iT2 = vcat(oa, ob)

    s_values = KTensor(KIndex[u, v], Matrix(Diagonal(S)))
    TensorOperations.allocator_reset!(buf, cp)

    return KTensor(iT1, T1), KTensor(iT2, T2), s_values, truncerr
end

# ── S=1/2 operator & state library (conventions pinned against ITensors) ────────────────

const _σI = ComplexF64[1 0; 0 1]
const _σx = ComplexF64[0 1; 1 0]
const _σy = ComplexF64[0 -im; im 0]
const _σz = ComplexF64[1 0; 0 -1]

_is_spinhalf(i::KIndex) = i.d == 2

function _op1(name::String; kwargs...)
    name == "I" && return _σI
    name == "X" && return _σx
    name == "Y" && return _σy
    name == "Z" && return _σz
    name == "H" && return ComplexF64[1 1; 1 -1] / sqrt(2)
    name == "S+" && return ComplexF64[0 1; 0 0]
    name == "S-" && return ComplexF64[0 0; 1 0]
    name == "Sz" && return _σz / 2
    name == "Sx" && return _σx / 2
    name == "Sy" && return _σy / 2
    if name == "Rx"
        θ = kwargs[:θ]
        return exp(-im * θ / 2 * _σx)
    elseif name == "Ry"
        θ = kwargs[:θ]
        return exp(-im * θ / 2 * _σy)
    elseif name == "Rz"
        θ = kwargs[:θ]
        return exp(-im * θ / 2 * _σz)
    elseif name == "P"
        ϕ = kwargs[:ϕ]
        return ComplexF64[1 0; 0 exp(im * ϕ)]
    end
    return nothing
end

# Two-site 4×4 matrices in the (first index fastest) convention that maps onto
# data[s1', s2', s1, s2] via column-major reshape — matching the ITensors op layouts.
function _op2(name::String; kwargs...)
    kr(A, B) = kron(B, A)   # s1 fastest
    if name == "Rzz"
        ϕ = kwargs[:ϕ]
        return exp(-im * ϕ * kr(_σz, _σz))
    elseif name == "Rxx"
        ϕ = kwargs[:ϕ]
        return exp(-im * ϕ * kr(_σx, _σx))
    elseif name == "Ryy"
        ϕ = kwargs[:ϕ]
        return exp(-im * ϕ * kr(_σy, _σy))
    elseif name == "Rxxyy"
        θ = kwargs[:θ]
        h = 0.5 * (kr(_σx, _σx) + kr(_σy, _σy))
        return exp(-0.5 * im * θ * h)
    elseif name == "Rxxyyzz"
        θ = kwargs[:θ]
        h = 0.5 * (kr(_σx, _σx) + kr(_σy, _σy) + kr(_σz, _σz))
        return exp(-0.5 * im * θ * h)
    elseif name == "xx_plus_yy"
        θ, β = kwargs[:θ], kwargs[:β]
        # matches gate_definitions: exp(-i θ/2 (cos β (XX+YY)/2 + sin β (YX-XY)/2)) convention
        h = cos(β) * 0.5 * (kr(_σx, _σx) + kr(_σy, _σy)) + sin(β) * 0.5 * (kr(_σy, _σx) - kr(_σx, _σy))
        return exp(-0.5 * im * θ * h)
    elseif name == "CZ"
        return ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
    elseif name == "CNOT" || name == "CX"
        # control = first index (fastest); |s1=1⟩ flips s2
        m = zeros(ComplexF64, 4, 4)
        m[1, 1] = 1; m[3, 3] = 1   # s1=0 (slots 1,3): identity on s2
        m[4, 2] = 1; m[2, 4] = 1   # s1=1 (slots 2,4): flip s2
        return m
    elseif name == "CPHASE"
        ϕ = kwargs[:ϕ]
        return ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(im * ϕ)]
    elseif name == "SWAP"
        return ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    end
    return nothing
end

function TensorInterface.op(name::String, i::KIndex; kwargs...)
    _is_spinhalf(i) || error("op: KTensors operator library currently covers d=2 (S=1/2) sites only")
    m = _op1(name; kwargs...)
    m === nothing && error("op: unknown single-site operator \"$name\" for the KTensors backend")
    return KTensor(KIndex[TensorInterface.prime(i), i], copy(m))
end

function TensorInterface.op(name::String, i1::KIndex, i2::KIndex; kwargs...)
    (_is_spinhalf(i1) && _is_spinhalf(i2)) || error("op: KTensors operator library currently covers d=2 (S=1/2) sites only")
    m = _op2(name; kwargs...)
    m === nothing && error("op: unknown two-site operator \"$name\" for the KTensors backend")
    data = reshape(copy(m), 2, 2, 2, 2)
    return KTensor(KIndex[TensorInterface.prime(i1), TensorInterface.prime(i2), i1, i2], data)
end

function TensorInterface.state(name::String, i::KIndex)
    _is_spinhalf(i) || error("state: KTensors state library currently covers d=2 (S=1/2) sites only")
    vecmap = Dict(
        "↑" => [1.0, 0.0], "0" => [1.0, 0.0], "Up" => [1.0, 0.0], "Z+" => [1.0, 0.0],
        "↓" => [0.0, 1.0], "1" => [0.0, 1.0], "Dn" => [0.0, 1.0], "Z-" => [0.0, 1.0],
        "+" => [1.0, 1.0] / sqrt(2), "X+" => [1.0, 1.0] / sqrt(2),
        "-" => [1.0, -1.0] / sqrt(2), "X-" => [1.0, -1.0] / sqrt(2),
    )
    haskey(vecmap, name) || error("state: unknown state \"$name\" for the KTensors backend")
    return KTensor(KIndex[i], copy(vecmap[name]))
end

end
