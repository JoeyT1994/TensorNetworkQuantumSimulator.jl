#=
KTensors: the tensor engine — named-index tensors over dense arrays.

A `Tensor` is a dense N-d array plus a vector of `KIndex` labels. Index identity
(id, plev) — not position — drives contraction. Contraction is executed by
TensorOperations with dynamically generated labels; factorizations run through
MatrixAlgebraKit (which also supplies the in-place variants and the CUSOLVER/ROCSOLVER
algorithm selection for a future GPU story). The hot paths — the double-layer BP message
update, the two-site simple-update gate, and BP expectation-value closures — run as fused
kernels: conjugation is folded into the BLAS calls (no materialized `dag`), and all chain
intermediates live in a task-local, reusable BufferAllocator so that only results touch
the heap. They attach to the generic algorithms through the hooks in
src/kernel_hooks.jl and fall back to the plain seam-verb path whenever a network's
structure doesn't match.

The operator/state library is a runtime registry (`register_op!`) covering S=1/2 gates in
the first-index-fastest matrix convention; conventions were validated against the
historical ITensors implementation, which survives as a test-only cross-check
(test/test_ktensors.jl).

The second data layer is `TKTensor` (tktensor.jl): the same `KIndex` labels over a
TensorKit `TensorMap`, serving every graded backend — bosonic Z2/U(1), fermionic parity,
fU(1) and dual fU(1)×U(1) product sectors — through one code path. Nothing algebraic is
implemented there (hard rule): contraction, permutation signs (Jordan-Wigner strings
emerge from the braiding), and blockwise factorizations all delegate to TensorKit and
MatrixAlgebraKit; the file is label↔slot bookkeeping plus the operator/state quack layer.
The per-copy `KIndex.dual` flag carries bond orientation (live for TKTensor, inert for
dense Tensor).
=#
module KTensors

using LinearAlgebra: LinearAlgebra, Hermitian, Diagonal, norm, mul!, diag, rmul!, BlasFloat
using MatrixAlgebraKit: MatrixAlgebraKit, qr_compact, qr_compact!, svd_compact, svd_trunc,
    eigh_full, truncrank, truncerror
using TensorOperations: TensorOperations, ncon
using VectorInterface: VectorInterface
using Adapt: Adapt, adapt
import TensorKit as TK
using ..TensorInterface: TensorInterface

export KIndex, Tensor, TKTensor, register_op!, graded_space, new_fermion_index

# ── Index ───────────────────────────────────────────────────────────────────────────────

space_dim(s::Integer) = Int(s)

"""
    KIndex(d::Integer, tags = "")
    KIndex(space, tags = "")

A named tensor index: identified by `(id, plev)`, carrying a space (a plain dimension for
dense tensors, a TensorKit `GradedSpace` for symmetric ones), cosmetic tags, and a
dual/arrow flag.
"""
struct KIndex{S}
    id::UInt64
    space::S
    plev::Int
    tags::String
    dual::Bool   # bond orientation (arrow); inert for dense data
end

KIndex(d::Integer, tags::AbstractString = "") = KIndex(rand(UInt64), Int(d), 0, String(tags), false)
dimof(i::KIndex) = space_dim(i.space)
space(i::KIndex) = i.space

# Identity is (id, plev): `dag` (dual flip) and tag changes never change which index this is.
Base.:(==)(a::KIndex, b::KIndex) = a.id == b.id && a.plev == b.plev
Base.hash(i::KIndex, h::UInt) = hash((i.id, i.plev), h)
Base.adjoint(i::KIndex) = TensorInterface.prime(i)
Base.copy(i::KIndex) = i

function Base.show(io::IO, i::KIndex)
    print(io, "(d=", dimof(i), "|id=", repr(i.id % UInt16), "|\"", i.tags, "\")", "'"^i.plev, i.dual ? "†" : "")
    return nothing
end

TensorInterface.dim(i::KIndex) = dimof(i)
TensorInterface.dim(is::AbstractVector{<:KIndex}) = prod(TensorInterface.dim.(is); init = 1)
TensorInterface.plev(i::KIndex) = i.plev
TensorInterface.tags(i::KIndex) = i.tags
TensorInterface.dag(i::KIndex) = KIndex(i.id, i.space, i.plev, i.tags, !i.dual)
TensorInterface.prime(i::KIndex, n::Integer = 1) = KIndex(i.id, i.space, i.plev + n, i.tags, i.dual)
TensorInterface.noprime(i::KIndex) = KIndex(i.id, i.space, 0, i.tags, i.dual)
TensorInterface.sim(i::KIndex) = KIndex(rand(UInt64), i.space, i.plev, i.tags, i.dual)

for f in [:dag, :prime, :noprime, :sim]
    @eval TensorInterface.$f(is::AbstractVector{<:KIndex}, args...) = map(i -> TensorInterface.$f(i, args...), is)
end

TensorInterface.new_index(::Union{KIndex, AbstractVector{<:KIndex}}, d::Integer; tags = "") = KIndex(d, tags)

# ── AbstractTensor ──────────────────────────────────────────────────────────────────────
# Common supertype of the backends' tensor types (dense `Tensor`, graded `TKTensor`).
# Every subtype carries `inds::Vector{<:KIndex}` + `data` and implements the structural
# primitive `_like(t, inds, data)` (same backend, new labels/data). Everything here is
# label bookkeeping — the data never moves.

abstract type AbstractTensor end

_indvec(t::AbstractTensor) = t.inds
_mapinds(f, t::AbstractTensor) = _like(t, map(f, t.inds), t.data)

Base.ndims(t::AbstractTensor) = length(t.inds)
Base.copy(t::AbstractTensor) = _like(t, copy(t.inds), copy(t.data))
Base.show(io::IO, t::AbstractTensor) = print(io, nameof(typeof(t)), "{", eltype(t), "} inds: ", t.inds)

Base.:*(t::AbstractTensor, x::Number) = _like(t, copy(t.inds), t.data * x)
Base.:*(x::Number, t::AbstractTensor) = t * x
Base.:/(t::AbstractTensor, x::Number) = t * inv(x)
LinearAlgebra.norm(t::AbstractTensor) = norm(t.data)

function TensorInterface.inds(t::AbstractTensor; plev = nothing)
    plev === nothing && return t.inds
    return filter(i -> i.plev == plev, t.inds)
end
TensorInterface.scalartype(t::AbstractTensor) = eltype(t)
TensorInterface.prime(t::AbstractTensor, n::Integer = 1) = _mapinds(i -> TensorInterface.prime(i, n), t)
TensorInterface.noprime(t::AbstractTensor) = _mapinds(TensorInterface.noprime, t)
TensorInterface.sim(t::AbstractTensor) = _mapinds(TensorInterface.sim, t)
TensorInterface.replaceind(t::AbstractTensor, old::KIndex, new::KIndex) = TensorInterface.replaceinds(t, [old], [new])
TensorInterface.replaceinds(t::AbstractTensor, p::Pair) = TensorInterface.replaceinds(t, first(p), last(p))
TensorInterface.apply(o::AbstractTensor, t::AbstractTensor) = TensorInterface.noprime(o * t)

# ── Tensor ──────────────────────────────────────────────────────────────────────────────

struct Tensor{T, N, A <: AbstractArray{T, N}} <: AbstractTensor
    inds::Vector{<:KIndex}
    data::A
    function Tensor(inds::AbstractVector, data::A) where {T, N, A <: AbstractArray{T, N}}
        inds = collect(KIndex, inds)
        length(inds) == N || error("Tensor: $(length(inds)) indices for a rank-$N array")
        all(i -> dimof(i) == size(data, findfirst(==(i), inds)), unique(inds)) ||
            error("Tensor: index dimensions $(TensorInterface.dim.(inds)) don't match array size $(size(data))")
        return new{T, N, A}(inds, data)
    end
end

Tensor(x::Number) = Tensor(KIndex[], fill(x))

_like(t::Tensor, inds, data) = Tensor(inds, data)

Base.eltype(::Tensor{T}) where {T} = T
Base.sum(t::Tensor) = sum(t.data)

TensorInterface.datatype(::Tensor{T, N, A}) where {T, N, A <: Array} = Vector{T}
TensorInterface.array(t::Tensor) = t.data
TensorInterface.data(t::Tensor) = vec(t.data)
TensorInterface.new_index(t::Tensor, d::Integer; tags = "") = KIndex(d, tags)

function TensorInterface.scalar(t::Tensor)
    length(t.data) == 1 || error("scalar: Tensor with inds $(t.inds) is not a scalar")
    return t.data[Base.firstindex(t.data)]
end

# ── Index-set queries ───────────────────────────────────────────────────────────────────

_indvec(i::KIndex) = KIndex[i]
_indvec(is::AbstractVector) = collect(KIndex, is)
_indvec(is::Tuple) = collect(KIndex, is)

const KIndsLike = Union{AbstractTensor, KIndex, AbstractVector{<:KIndex}, Tuple{KIndex, Vararg{KIndex}}}

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

TensorInterface.dag(t::Tensor) = Tensor(map(TensorInterface.dag, t.inds), conj(t.data))

function TensorInterface.replaceinds(t::Tensor, old, new)
    oldv, newv = _indvec(old), _indvec(new)
    length(oldv) == length(newv) || error("replaceinds: length mismatch")
    newinds = map(t.inds) do i
        k = findfirst(==(i), oldv)
        if k === nothing
            i
        else
            n = newv[k]
            dimof(n) == dimof(i) || error("replaceinds: dimension mismatch $(i) → $(n)")
            n
        end
    end
    return Tensor(newinds, t.data)
end

# ── Construction ────────────────────────────────────────────────────────────────────────

TensorInterface.from_array(A::AbstractArray, is::KIndex{<:Integer}...) = Tensor(collect(KIndex, is), reshape(copy(A), TensorInterface.dim.(is)...))
TensorInterface.from_array(A::AbstractVector, i::KIndex{<:Integer}) = Tensor(KIndex[i], copy(A))

function TensorInterface.random_itensor(elt::Type, is::AbstractVector{<:KIndex})
    return Tensor(collect(is), randn(elt, TensorInterface.dim.(is)...))
end
TensorInterface.random_itensor(elt::Type, is::KIndex...) = TensorInterface.random_itensor(elt, collect(is))
TensorInterface.random_itensor(is::AbstractVector{<:KIndex}) = TensorInterface.random_itensor(Float64, is)
TensorInterface.random_itensor(is::KIndex...) = TensorInterface.random_itensor(Float64, collect(is))

function TensorInterface.onehot(elt::Type, p::Pair{<:KIndex, <:Integer})
    i, v = p
    data = zeros(elt, dimof(i))
    data[v] = one(elt)
    return Tensor(KIndex[i], data)
end
TensorInterface.onehot(p::Pair{<:KIndex, <:Integer}) = TensorInterface.onehot(Float64, p)

#Index vectors are often abstractly typed, so the dense/graded split is decided by content
function TensorInterface.delta(elt::Type, is::AbstractVector{<:KIndex})
    isempty(is) && return Tensor(one(elt))
    all(i -> space(i) isa TK.GradedSpace, is) && return _delta_tk(elt, is)
    data = zeros(elt, TensorInterface.dim.(is)...)
    for k in 1:minimum(TensorInterface.dim.(is))
        data[ntuple(_ -> k, length(is))...] = one(elt)
    end
    return Tensor(collect(is), data)
end
TensorInterface.delta(is::AbstractVector{<:KIndex}) = TensorInterface.delta(Float64, is)
TensorInterface.delta(elt::Type, is::KIndex...) = TensorInterface.delta(elt, collect(is))
TensorInterface.delta(is::KIndex...) = TensorInterface.delta(Float64, collect(is))

# A combiner is an explicit reshape isometry: identity data between the combined index
# (first) and the product of the combined indices. `t * C` combines; multiplying by `C`
# again splits back.
function TensorInterface.combiner(is::AbstractVector{<:KIndex}; tags = "CMB,Link")
    isempty(is) && error("combiner: no indices to combine")
    D = prod(TensorInterface.dim.(is))
    c = KIndex(D, String(tags))
    data = reshape(Matrix{Float64}(LinearAlgebra.I, D, D), (D, TensorInterface.dim.(is)...))
    return Tensor(vcat([c], collect(KIndex, is)), data)
end
TensorInterface.combiner(is::KIndex...; kwargs...) = TensorInterface.combiner(collect(KIndex, is); kwargs...)
TensorInterface.combinedind(C::Tensor) = first(C.inds)

# Direct sum of two tensors along the paired index axes (`olds1[k]`/`olds2[k]` → `news[k]`,
# with dim(news[k]) = dim(olds1[k]) + dim(olds2[k])); all other indices must coincide.
function TensorInterface.directsum(
        news::AbstractVector{<:KIndex}, p1::Pair{<:Tensor}, p2::Pair{<:Tensor}
    )
    t1, olds1 = first(p1), collect(KIndex, last(p1))
    t2, olds2 = first(p2), collect(KIndex, last(p2))
    length(news) == length(olds1) == length(olds2) || error("directsum: length mismatch")
    oinds = map(t1.inds) do i
        k = findfirst(==(i), olds1)
        k === nothing ? i : news[k]
    end
    T = promote_type(eltype(t1), eltype(t2))
    out = zeros(T, TensorInterface.dim.(oinds)...)
    r1 = map(i -> begin
        k = findfirst(==(i), olds1)
        k === nothing ? Colon() : (1:dimof(i))
    end, t1.inds)
    out[r1...] .= t1.data
    perm2 = map(t1.inds) do i
        k = findfirst(==(i), olds1)
        j = k === nothing ? findfirst(==(i), t2.inds) : findfirst(==(olds2[k]), t2.inds)
        j === nothing && error("directsum: tensors do not share the unsummed index $(i)")
        j
    end
    d2 = permutedims(t2.data, perm2)
    r2 = map(enumerate(t1.inds)) do (ax, i)
        k = findfirst(==(i), olds1)
        k === nothing ? Colon() : ((dimof(i) + 1):(dimof(i) + size(d2, ax)))
    end
    out[r2...] .= d2
    return Tensor(oinds, out)
end

#Default-backend index constructor (no reference index/tensor in scope, e.g. fresh networks)
TensorInterface.new_index(d::Integer; tags = "") = KIndex(d, String(tags))

# ── Arithmetic, contraction ─────────────────────────────────────────────────────────────


# Permute `b`'s data into `a`'s index order.
function _align(a::Tensor, b::Tensor)
    a.inds == b.inds && return b.data
    perm = map(i -> findfirst(==(i), b.inds), a.inds)
    any(isnothing, perm) && error("tensors have different index sets: $(a.inds) vs $(b.inds)")
    return permutedims(b.data, perm)
end

Base.:+(a::Tensor, b::Tensor) = Tensor(copy(a.inds), a.data + _align(a, b))
Base.:-(a::Tensor, b::Tensor) = Tensor(copy(a.inds), a.data - _align(a, b))
Base.isapprox(a::Tensor, b::Tensor; kwargs...) = isapprox(a.data, _align(a, b); kwargs...)


# VectorInterface (KrylovKit's vector-space contract, used by full_update's linsolve).
# Copying implementations are always legal for the !!-variants.
VectorInterface.scalartype(::Type{<:Tensor{T}}) where {T} = T
VectorInterface.zerovector(t::Tensor, S::Type{<:Number}) = Tensor(copy(t.inds), zeros(S, size(t.data)))
VectorInterface.scale(t::Tensor, α::Number) = t * α
VectorInterface.scale!!(t::Tensor, α::Number) = t * α
VectorInterface.scale!!(y::Tensor, x::Tensor, α::Number) = x * α
function VectorInterface.add(y::Tensor, x::Tensor, α::Number, β::Number)
    return Tensor(copy(y.inds), β * y.data + α * _align(y, x))
end
VectorInterface.add!!(y::Tensor, x::Tensor, α::Number, β::Number) = VectorInterface.add(y, x, α, β)
VectorInterface.inner(x::Tensor, y::Tensor) = LinearAlgebra.dot(x, y)
LinearAlgebra.normalize(t::Tensor) = t * inv(norm(t))
LinearAlgebra.dot(a::Tensor, b::Tensor) = LinearAlgebra.dot(vec(a.data), vec(_align(a, b)))
LinearAlgebra.tr(t::Tensor) = _trace_all(t)

# Pairwise contraction over all common (id, plev) indices; repeated indices on one tensor
# (traces) are supported through the same ncon labeling.
function Base.:*(a::Tensor, b::Tensor)
    # scalar fast paths
    ndims(a) == 0 && return Tensor(copy(b.inds), TensorInterface.scalar(a) * b.data)
    ndims(b) == 0 && return Tensor(copy(a.inds), TensorInterface.scalar(b) * a.data)

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
    return Tensor(outinds, data)
end

function _trace_all(t::Tensor)
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

#Buffered sequence execution: contraction trees run pairwise with every intermediate in
#the task-local buffer (checkpoint/reset per call); only the root result touches the heap.
#This gives EVERY cached-sequence contraction (boundary MPS, exact, loop corrections, ...)
#the allocation discipline of the fused kernels without structure-specific code. Falls back
#to the plain heap path for non-CPU storage or when the tree's total intermediate footprint
#exceeds `SEQ_BUFFER_MAXBYTES` (the task-local buffer only ever grows, so unboundedly large
#trees — e.g. exact contraction of big networks — should not pin their peak memory forever).
const SEQ_BUFFER_MAXBYTES = Ref(2^30)

function TensorInterface.contract(ts::Vector{<:Tensor}; sequence = nothing, kwargs...)
    isnothing(sequence) && return reduce(*, ts)
    sequence isa Integer && return ts[sequence]
    if all(t -> t.data isa Array, ts)
        elsize = sizeof(promote_type(map(eltype, ts)...))
        _, total = _seq_temp_bytes(ts, sequence, elsize)
        if total <= SEQ_BUFFER_MAXBYTES[]
            buf = _kernel_buffer()
            cp = TensorOperations.allocator_checkpoint!(buf)
            data, is = _exec_seq(ts, sequence, buf, true)
            TensorOperations.allocator_reset!(buf, cp)
            return Tensor(is, data)
        end
    end
    return _contract_seq(ts, sequence)
end

_contract_seq(ts::Vector{<:Tensor}, s::Integer) = ts[s]
function _contract_seq(ts::Vector{<:Tensor}, s::Union{Vector, Tuple})
    return mapreduce(x -> _contract_seq(ts, x), *, s)
end

#Symbolic walk of the sequence tree: per-node open indices and the summed byte size of all
#intermediates (the buffer is only reset at the root, so the requirement is the total, not
#the live peak). Returns (open_inds, total_intermediate_bytes).
function _seq_temp_bytes(ts, s::Integer, elsize)
    return ts[s].inds, 0
end
function _seq_temp_bytes(ts, s::Union{Vector, Tuple}, elsize)
    ix, total = _seq_temp_bytes(ts, s[1], elsize)
    for k in 2:length(s)
        iy, sub = _seq_temp_bytes(ts, s[k], elsize)
        total += sub
        ix = vcat(filter(i -> i ∉ iy, ix), filter(i -> i ∉ ix, iy))
        total += prod(Int[dimof(i) for i in ix]; init = 1) * elsize
    end
    return ix, total
end

#Execute the tree: leaves are the input arrays, intermediates are buffer temps, and the
#root-level final pairwise contraction writes its result on the heap.
function _exec_seq(ts, s::Integer, buf, isroot)
    t = ts[s]
    return t.data, t.inds
end
function _exec_seq(ts, s::Union{Vector, Tuple}, buf, isroot)
    X, ix = _exec_seq(ts, s[1], buf, false)
    n = length(s)
    for k in 2:n
        Y, iy = _exec_seq(ts, s[k], buf, false)
        if ndims(X) == 0 || ndims(Y) == 0
            #scalar × tensor: cheap, sidestep the pairwise machinery
            data = ndims(X) == 0 ? X[] .* Y : Y[] .* X
            X, ix = data, (ndims(X) == 0 ? iy : ix)
            continue
        end
        istemp = (isroot && k == n) ? Val(false) : Val(true)
        X, oa, ob = _tc_pair(X, ix, Y, iy, false, identity, istemp, buf)
        ix = vcat(oa, ob)
    end
    return X, ix
end


# ── Diagonal operations ─────────────────────────────────────────────────────────────────

function TensorInterface.map_diag(f::Function, t::Tensor)
    out = copy(t)
    TensorInterface.map_diag!(f, out, out)
    return out
end

function TensorInterface.map_diag!(f::Function, out::Tensor, t::Tensor)
    ndims(t) == 2 || error("map_diag: expected a 2-index Tensor")
    d = _align(out, t)
    d === out.data || copyto!(out.data, d)
    for k in 1:minimum(size(out.data))
        out.data[k, k] = f(out.data[k, k])
    end
    return out
end

# ── Adapt / storage ─────────────────────────────────────────────────────────────────────

Adapt.adapt_structure(elt::Type{<:Number}, t::Tensor) = Tensor(copy(t.inds), convert(Array{elt}, t.data))
function Adapt.adapt_structure(to::Type{<:AbstractVector}, t::Tensor)
    return Tensor(copy(t.inds), reshape(adapt(to, vec(copy(t.data))), size(t.data)))
end
Adapt.adapt_structure(to, t::Tensor) = Tensor(copy(t.inds), adapt(to, t.data))

# ── Factorizations (MatrixAlgebraKit; in-place/preallocated variants are the v3 target) ─

struct KSpectrum
    eigs::Vector{Float64}
    truncerr::Float64
end

# Permute+reshape to a (linds...) × (rest...) matrix. Returns matrix, left inds, right inds.
function _matricize(t::Tensor, linds::Vector{<:KIndex})
    rinds = TensorInterface.uniqueinds(t, linds)
    perm = map(i -> findfirst(==(i), t.inds), vcat(linds, rinds))
    any(isnothing, perm) && error("factorize: indices $(linds) not all found on tensor $(t.inds)")
    A = permutedims(t.data, perm)
    dl = prod(Int[dimof(i) for i in linds]; init = 1)
    dr = prod(Int[dimof(i) for i in rinds]; init = 1)
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

function LinearAlgebra.qr(t::Tensor, linds; kwargs...)
    A, li, ri = _matricize(t, _indvec(linds))
    Q, R = qr_compact(A)
    b = KIndex(size(Q, 2), "Link,qr")
    Qt = Tensor(vcat(li, [b]), reshape(Q, (Int[dimof(i) for i in li]..., size(Q, 2))))
    Rt = Tensor(vcat([b], ri), reshape(R, (size(R, 1), Int[dimof(i) for i in ri]...)))
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

function _svd_split(t::Tensor, linds::Vector{<:KIndex}; maxdim = nothing, cutoff = nothing, mindim = 1)
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
        t::Tensor, linds;
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
        F1 = Tensor(vcat(li, [up]), reshape(F1m, (Int[dimof(i) for i in li]..., k)))
        F2 = Tensor(vcat([up], ri), reshape(F2m, (k, Int[dimof(i) for i in ri]...)))
    elseif ortho == "left"
        F1 = Tensor(vcat(li, [up]), reshape(U, (Int[dimof(i) for i in li]..., k)))
        F2 = Tensor(vcat([up], ri), reshape(Diagonal(T.(S)) * Vt, (k, Int[dimof(i) for i in ri]...)))
    elseif ortho == "right"
        F1 = Tensor(vcat(li, [up]), reshape(U * Diagonal(T.(S)), (Int[dimof(i) for i in li]..., k)))
        F2 = Tensor(vcat([up], ri), reshape(Vt, (k, Int[dimof(i) for i in ri]...)))
    else
        error("factorize_svd: unknown ortho = $ortho")
    end

    if singular_values! !== nothing
        singular_values![] = Tensor(KIndex[u, v], Matrix(Diagonal(S)))
    end
    return F1, F2, KSpectrum(abs2.(S), truncerr)
end

# ITensors-style factorize: L isometric for ortho="left" (L=U, R=SV), mirrored for "right".
# The bond is unprimed and carries `tags`.
function LinearAlgebra.factorize(
        t::Tensor, linds...;
        ortho = "left", maxdim = nothing, cutoff = nothing, mindim = 1, tags = "Link,fact", kwargs...,
    )
    lv = length(linds) == 1 ? _indvec(only(linds)) : collect(KIndex, linds)
    # ITensors convention: entries of linds not present on the tensor are ignored
    lv = filter(i -> i ∈ t.inds, lv)
    U, S, Vt, li, ri, _ = _svd_split(t, lv; maxdim, cutoff, mindim)
    k = length(S)
    T = eltype(U)
    b = KIndex(k, String(tags))
    if ortho == "left"
        L = Tensor(vcat(li, [b]), reshape(U, (Int[dimof(i) for i in li]..., k)))
        R = Tensor(vcat([b], ri), reshape(Diagonal(T.(S)) * Vt, (k, Int[dimof(i) for i in ri]...)))
    elseif ortho == "right"
        L = Tensor(vcat(li, [b]), reshape(U * Diagonal(T.(S)), (Int[dimof(i) for i in li]..., k)))
        R = Tensor(vcat([b], ri), reshape(Vt, (k, Int[dimof(i) for i in ri]...)))
    else
        error("factorize: unknown ortho = $ortho")
    end
    return L, R
end

# ITensors-style svd: U(linds, u), S(u, v), V(rinds, v).
function LinearAlgebra.svd(t::Tensor, linds; maxdim = nothing, cutoff = nothing, mindim = 1, kwargs...)
    U, S, Vt, li, ri, _ = _svd_split(t, filter(i -> i ∈ t.inds, _indvec(linds)); maxdim, cutoff, mindim)
    k = length(S)
    u = KIndex(k, "Link,u")
    v = KIndex(k, "Link,v")
    Ut = Tensor(vcat(li, [u]), reshape(U, (Int[dimof(i) for i in li]..., k)))
    St = Tensor(KIndex[u, v], Matrix(Diagonal(S)))
    Vt_ = Tensor(vcat(ri, [v]), reshape(copy(transpose(Vt)), (Int[dimof(i) for i in ri]..., k)))
    return Ut, St, Vt_
end

# Index-free hermitian eigen on a (l, l′)-paired tensor: linds = the plev-0 indices,
# rinds = their primes, and U comes back labeled by the UNPRIMED side, so that
# U · D · prime(dag(U)) reconstructs the tensor (the convention symmetric_gauge relies on).
function LinearAlgebra.eigen(t::Tensor; ishermitian::Bool = false, kwargs...)
    lv = filter(i -> i.plev == 0, t.inds)
    rv = collect(KIndex, TensorInterface.prime.(lv))
    D, U = LinearAlgebra.eigen(t, lv, rv; ishermitian, kwargs...)
    return D, TensorInterface.replaceinds(U, rv, lv)
end

# eigen matching the ITensors hermitian convention probed empirically:
# D on (link′, link) with the eigenvalues, U on (rinds..., link); Ul·D·dag(U) reconstructs.
function LinearAlgebra.eigen(t::Tensor, linds, rinds; ishermitian::Bool = false, kwargs...)
    ishermitian || error("eigen: only ishermitian = true is implemented for KTensors")
    lv, rv = _indvec(linds), _indvec(rinds)
    A, li, ri = _matricize(t, lv)
    ri == rv || error("eigen: rinds don't match the remaining indices")
    D, vecs = eigh_full((A + A') / 2)
    vals = diag(D)
    k = length(vals)
    lk = KIndex(k, "Link,eigen")
    D = Tensor(KIndex[TensorInterface.prime(lk), lk], Matrix(Diagonal(vals)))
    U = Tensor(vcat(rv, [lk]), reshape(vecs, (Int[dimof(i) for i in rv]..., k)))
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

function _kernel_buffer()
    return get!(task_local_storage(), :ktensors_kernel_buffer) do
        TensorOperations.BufferAllocator()
    end::TensorOperations.BufferAllocator
end

# One pairwise contraction at the data level. `pairfn` maps an index of `b` to the index of
# `a` it should contract with (identity for plain absorption; the site/prime partner map for
# the closing bra contraction). Returns (data, out_inds).
function _tc_pair(
        da, ia::Vector{<:KIndex}, db, ib::Vector{<:KIndex}, conjB::Bool, pairfn,
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
function _is_norm_message(m::Tensor, ψinds::Vector{<:KIndex})
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
        ψ::Tensor, sinds::Vector{<:KIndex}, incoming::Vector{<:Tensor};
        op::Union{Nothing, Tensor} = nothing
    )
    ψ.data isa Array || return nothing
    all(m -> m.data isa Array && _is_norm_message(m, ψ.inds), incoming) || return nothing
    op === nothing || op.data isa Array || return nothing

    buf = _kernel_buffer()
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

    return Tensor(oinds, out)
end

"""
Fused computation of the outgoing BP message from vertex tensor `ψ` (site indices `sinds`)
given standard doubled `incoming` messages. Returns `nothing` when the structure doesn't
match (caller falls back to the generic contraction path).
"""
function fused_norm_message(
        ψ::Tensor, sinds::Vector{<:KIndex}, incoming::Vector{<:Tensor};
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
function _matricize_temp(da, ia::Vector{<:KIndex}, linds::Vector{<:KIndex}, buf)
    rinds = filter(i -> i ∉ linds, ia)
    perm = map(i -> findfirst(==(i), ia), vcat(linds, rinds))
    T = eltype(da)
    dims = ntuple(k -> size(da, perm[k]), ndims(da))
    C = TensorOperations.tensoralloc(Array{T, ndims(da)}, dims, Val(true), buf)
    TensorOperations.tensoradd!(C, da, (Tuple(perm), ()), false, one(T), zero(T))
    dl = prod(Int[dimof(i) for i in linds]; init = 1)
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
function _absorb_chain(X, ix::Vector{<:KIndex}, ms::Vector{<:Tensor}, conjB::Bool, buf)
    for m in ms
        X, oa, ob = _tc_pair(X, ix, m.data, m.inds, conjB, identity, Val(true), buf)
        ix = vcat(oa, ob)
    end
    return X, ix
end

function fused_two_site_gate(
        o::Tensor, ψ1::Tensor, ψ2::Tensor,
        sqrt1::Vector{<:Tensor}, inv1::Vector{<:Tensor},
        sqrt2::Vector{<:Tensor}, inv2::Vector{<:Tensor},
        s1::Vector{<:KIndex}, s2::Vector{<:KIndex};
        maxdim = nothing, cutoff = nothing,
    )
    buf = _kernel_buffer()
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
    R1 = reshape(R1m, (dimof(b1), Int[dimof(i) for i in r1]...))
    R2 = reshape(R2m, (dimof(b2), Int[dimof(i) for i in r2]...))

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
    F1 = reshape(U .* reshape(TS.(sq), 1, k), (Int[dimof(i) for i in lo]..., k))
    F2 = reshape(reshape(TS.(sq), k, 1) .* Vt, (k, Int[dimof(i) for i in ro]...))
    iF1 = vcat(lo, [up])
    iF2 = vcat([up], ro)

    # De-gauge: Y_k = Q_k · ∏ conj(inv-√env), the dag fused as a conj flag
    iQ1 = vcat(l1, [b1])
    iQ2 = vcat(l2, [b2])
    Y1, iy1 = _absorb_chain(reshape(Q1m, (Int[dimof(i) for i in l1]..., dimof(b1))), iQ1, inv1, true, buf)
    Y2, iy2 = _absorb_chain(reshape(Q2m, (Int[dimof(i) for i in l2]..., dimof(b2))), iQ2, inv2, true, buf)

    # Reassemble; these two escape to the heap
    T1, oa, ob = _tc_pair(Y1, iy1, F1, iF1, false, identity, Val(false), buf)
    iT1 = vcat(oa, ob)
    T2, oa, ob = _tc_pair(Y2, iy2, F2, iF2, false, identity, Val(false), buf)
    iT2 = vcat(oa, ob)

    s_values = Tensor(KIndex[u, v], Matrix(Diagonal(S)))
    TensorOperations.allocator_reset!(buf, cp)

    return Tensor(iT1, T1), Tensor(iT2, T2), s_values, truncerr
end

# ── S=1/2 operator & state library (conventions pinned against ITensors) ────────────────

const _σI = ComplexF64[1 0; 0 1]
const _σx = ComplexF64[0 1; 1 0]
const _σy = ComplexF64[0 -im; im 0]
const _σz = ComplexF64[1 0; 0 -1]

_is_spinhalf(i::KIndex) = dimof(i) == 2

"""
    register_op!(name::String, f::Function; nsites::Int = 1)

Register a custom operator matrix for the KTensors backend. `f(; kwargs...)` must return
the operator matrix: 2×2 for `nsites = 1`, 4×4 for `nsites = 2` in the first-index-fastest
(column-major) basis convention. Overwrites any existing entry with the same name.
"""
function register_op!(name::String, f::Function; nsites::Int = 1)
    nsites == 1 && (OP1_REGISTRY[name] = f; return nothing)
    nsites == 2 && (OP2_REGISTRY[name] = f; return nothing)
    return error("register_op!: only 1- and 2-site operators are supported")
end

const OP1_REGISTRY = Dict{String, Function}(
    "I" => (; kwargs...) -> _σI,
    "X" => (; kwargs...) -> _σx,
    "Y" => (; kwargs...) -> _σy,
    "Z" => (; kwargs...) -> _σz,
    "H" => (; kwargs...) -> ComplexF64[1 1; 1 -1] / sqrt(2),
    "S+" => (; kwargs...) -> ComplexF64[0 1; 0 0],
    "S-" => (; kwargs...) -> ComplexF64[0 0; 1 0],
    "Sz" => (; kwargs...) -> _σz / 2,
    "Sx" => (; kwargs...) -> _σx / 2,
    "Sy" => (; kwargs...) -> _σy / 2,
    "Rx" => (; θ) -> exp(-im * θ / 2 * _σx),
    "Ry" => (; θ) -> exp(-im * θ / 2 * _σy),
    "Rz" => (; θ) -> exp(-im * θ / 2 * _σz),
    "P" => (; ϕ) -> ComplexF64[1 0; 0 exp(im * ϕ)],
)

_op1(name::String; kwargs...) = haskey(OP1_REGISTRY, name) ? OP1_REGISTRY[name](; kwargs...) : nothing

# Two-site 4×4 matrices in the (first index fastest) convention that maps onto
# data[s1', s2', s1, s2] via column-major reshape. Conventions validated against the
# historical ITensors library (test/test_ktensors.jl).
_kr(A, B) = kron(B, A)   # s1 fastest

const OP2_REGISTRY = Dict{String, Function}(
    "Rzz" => (; ϕ) -> exp(-im * ϕ * _kr(_σz, _σz)),
    "Rxx" => (; ϕ) -> exp(-im * ϕ * _kr(_σx, _σx)),
    "Ryy" => (; ϕ) -> exp(-im * ϕ * _kr(_σy, _σy)),
    "Rxxyy" => (; θ) -> exp(-im * θ * 0.5 * (_kr(_σx, _σx) + _kr(_σy, _σy))),
    "Rxxyyzz" => (; θ) -> exp(-im * θ * 0.5 * (_kr(_σx, _σx) + _kr(_σy, _σy) + _kr(_σz, _σz))),
    "xx_plus_yy" => (; θ, β) -> exp(
        -0.5 * im * θ * (
            cos(β) * 0.5 * (_kr(_σx, _σx) + _kr(_σy, _σy)) +
                sin(β) * 0.5 * (_kr(_σy, _σx) - _kr(_σx, _σy))
        )
    ),
    "CZ" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1],
    "CNOT" => (; kwargs...) -> _controlled(_σx),
    "CX" => (; kwargs...) -> _controlled(_σx),
    "CY" => (; kwargs...) -> _controlled(_σy),
    "CRx" => (; θ) -> _controlled(exp(-im * θ / 2 * _σx)),
    "CRy" => (; θ) -> _controlled(exp(-im * θ / 2 * _σy)),
    "CRz" => (; θ) -> _controlled(exp(-im * θ / 2 * _σz)),
    "CPHASE" => (; ϕ) -> ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(im * ϕ)],
    "SWAP" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1],
    "iSWAP" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1],
    "√SWAP" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 (1 + im)/2 (1 - im)/2 0; 0 (1 - im)/2 (1 + im)/2 0; 0 0 0 1],
    "√iSWAP" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 1/sqrt(2) im/sqrt(2) 0; 0 im/sqrt(2) 1/sqrt(2) 0; 0 0 0 1],
)

_op2(name::String; kwargs...) = haskey(OP2_REGISTRY, name) ? OP2_REGISTRY[name](; kwargs...) : nothing

# Controlled single-qubit gate, control = FIRST index (which is the fastest in our
# column-major fastest-first layout): slots (1,3) are control=0 (identity on the target),
# slots (2,4) are control=1 (apply `u`).
function _controlled(u::AbstractMatrix)
    m = Matrix{ComplexF64}(LinearAlgebra.I, 4, 4)
    m[2, 2] = u[1, 1]; m[2, 4] = u[1, 2]
    m[4, 2] = u[2, 1]; m[4, 4] = u[2, 2]
    return m
end

function TensorInterface.op(name::String, i::KIndex; kwargs...)
    _is_spinhalf(i) || error("op: KTensors operator library currently covers d=2 (S=1/2) sites only")
    m = _op1(name; kwargs...)
    m === nothing && error("op: unknown single-site operator \"$name\" for the KTensors backend")
    return Tensor(KIndex[TensorInterface.prime(i), i], copy(m))
end

function TensorInterface.op(name::String, i1::KIndex, i2::KIndex; kwargs...)
    (_is_spinhalf(i1) && _is_spinhalf(i2)) || error("op: KTensors operator library currently covers d=2 (S=1/2) sites only")
    m = _op2(name; kwargs...)
    m === nothing && error("op: unknown two-site operator \"$name\" for the KTensors backend")
    data = reshape(copy(m), 2, 2, 2, 2)
    return Tensor(KIndex[TensorInterface.prime(i1), TensorInterface.prime(i2), i1, i2], data)
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
    return Tensor(KIndex[i], copy(vecmap[name]))
end

include("tktensor.jl")

end
