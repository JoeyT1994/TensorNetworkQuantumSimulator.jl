#=
Tensors: the tensor engine — named-index tensors over dense arrays.

A `Tensor` is a dense N-d array plus a vector of `Index` labels. Index identity
(id, plev) — not position — drives contraction. Contraction is executed by
TensorOperations with dynamically generated labels; factorizations run through
MatrixAlgebraKit (which also supplies the in-place variants and the CUSOLVER/ROCSOLVER
algorithm selection on GPU). The hot paths — the double-layer BP message
update, the two-site simple-update gate, and BP expectation-value closures — run as fused
kernels: conjugation is folded into the BLAS calls (no materialized `dag`), and all chain
intermediates live in a task-local, reusable BufferAllocator so that only results touch
the heap. They attach to the generic algorithms through the hooks in
src/kernel_hooks.jl and fall back to the plain seam-verb path whenever a network's
structure doesn't match.

The operator/state library is a runtime registry (`register_op!`) covering S=1/2 gates in
the first-index-fastest matrix convention; gate matrices and factorization conventions
were validated against the original ITensors-backed implementation before its removal and
are pinned by the frozen digests in test/test_tensors.jl.

The second data layer is `GradedTensor` (gradedtensor.jl): the same `Index` labels over a
TensorKit `TensorMap`, serving every graded backend — bosonic Z2/U(1), fermionic parity,
fU(1) and dual fU(1)×U(1) product sectors — through one code path. Nothing algebraic is
implemented there (hard rule): contraction, permutation signs (Jordan-Wigner strings
emerge from the braiding), and blockwise factorizations all delegate to TensorKit and
MatrixAlgebraKit; the file is label↔slot bookkeeping plus the operator/state quack layer.
The per-copy `Index.dual` flag carries bond orientation (live for GradedTensor, inert for
dense Tensor).
=#
module Tensors

using LinearAlgebra: LinearAlgebra, Diagonal, norm, diag, rmul!
using MatrixAlgebraKit: MatrixAlgebraKit, qr_compact, qr_compact!, svd_compact, svd_compact!,
    svd_trunc, svd_trunc!, eigh_full, truncrank, truncerror
using TensorOperations: TensorOperations, ncon
using VectorInterface: VectorInterface
using Adapt: Adapt, adapt
using GPUArraysCore: AbstractGPUArray
using StridedViews: StridedViews
import TensorKit as TK
using ..TensorInterface: TensorInterface

export Index, Tensor, GradedTensor, register_op!, graded_space, new_fermion_index

# ── Index ───────────────────────────────────────────────────────────────────────────────

space_dim(s::Integer) = Int(s)

"""
    Index(d::Integer, tags = "")
    Index(space, tags = "")

A named tensor index: identified by `(id, plev)`, carrying a space (a plain dimension for
dense tensors, a TensorKit `GradedSpace` for symmetric ones), cosmetic tags, and a
dual/arrow flag.
"""
struct Index{S}
    id::UInt64
    space::S
    plev::Int
    tags::String
    dual::Bool   # bond orientation (arrow); inert for dense data
end

Index(d::Integer, tags::AbstractString = "") = Index(rand(UInt64), Int(d), 0, String(tags), false)
dimof(i::Index) = space_dim(i.space)
space(i::Index) = i.space

# Identity is (id, plev): `dag` (dual flip) and tag changes never change which index this is.
Base.:(==)(a::Index, b::Index) = a.id == b.id && a.plev == b.plev
Base.hash(i::Index, h::UInt) = hash((i.id, i.plev), h)
Base.adjoint(i::Index) = TensorInterface.prime(i)
Base.copy(i::Index) = i

function Base.show(io::IO, i::Index)
    print(io, "(d=", dimof(i), "|id=", repr(i.id % UInt16), "|\"", i.tags, "\")", "'"^i.plev, i.dual ? "†" : "")
    return nothing
end

TensorInterface.dim(i::Index) = dimof(i)
TensorInterface.dim(is::AbstractVector{<:Index}) = prod(TensorInterface.dim.(is); init = 1)
TensorInterface.plev(i::Index) = i.plev
TensorInterface.tags(i::Index) = i.tags
TensorInterface.dag(i::Index) = Index(i.id, i.space, i.plev, i.tags, !i.dual)
TensorInterface.prime(i::Index, n::Integer = 1) = Index(i.id, i.space, i.plev + n, i.tags, i.dual)
TensorInterface.noprime(i::Index) = Index(i.id, i.space, 0, i.tags, i.dual)
TensorInterface.sim(i::Index) = Index(rand(UInt64), i.space, i.plev, i.tags, i.dual)

for f in [:dag, :prime, :noprime, :sim]
    @eval TensorInterface.$f(is::AbstractVector{<:Index}, args...) = map(i -> TensorInterface.$f(i, args...), is)
end

TensorInterface.new_index(::Union{Index, AbstractVector{<:Index}}, d::Integer; tags = "") = Index(d, tags)

# ── AbstractTensor ──────────────────────────────────────────────────────────────────────
# Common supertype of the backends' tensor types (dense `Tensor`, graded `GradedTensor`).
# Every subtype carries `inds::Vector{<:Index}` + `data` and implements the structural
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
TensorInterface.replaceind(t::AbstractTensor, old::Index, new::Index) = TensorInterface.replaceinds(t, [old], [new])
TensorInterface.replaceinds(t::AbstractTensor, p::Pair) = TensorInterface.replaceinds(t, first(p), last(p))
TensorInterface.apply(o::AbstractTensor, t::AbstractTensor) = TensorInterface.noprime(o * t)

function TensorInterface.map_diag(f::Function, t::AbstractTensor)
    out = copy(t)
    TensorInterface.map_diag!(f, out, out)
    return out
end
TensorInterface.combinedind(C::AbstractTensor) = first(C.inds)

#Index-free hermitian eigen on a (l, l′)-paired tensor: linds = the plev-0 indices,
#rinds = their primes, and U comes back labeled by the UNPRIMED side, so that
#U · D · prime(dag(U)) reconstructs the tensor (the convention symmetric_gauge relies on).
function LinearAlgebra.eigen(t::AbstractTensor; ishermitian::Bool = false, kwargs...)
    lv = filter(i -> i.plev == 0, t.inds)
    rv = collect(Index, TensorInterface.prime.(lv))
    D, U = LinearAlgebra.eigen(t, lv, rv; ishermitian, kwargs...)
    return D, TensorInterface.replaceinds(U, rv, lv)
end

#Cached-sequence tree walk shared by both backends' contract methods
#Abstractly-typed tensor lists (e.g. from Any-valued network dictionaries) route to
#the concrete backend method (bodies resolve at call time, so GradedTensor is fine here)
function TensorInterface.contract(ts::Vector; kwargs...)
    all(t -> t isa Tensor, ts) && return TensorInterface.contract(collect(Tensor, ts); kwargs...)
    all(t -> t isa GradedTensor, ts) && return TensorInterface.contract(collect(GradedTensor, ts); kwargs...)
    return error("contract: expected a homogeneous tensor list, got $(unique(typeof.(ts)))")
end

_contract_seq(ts::Vector, x::Integer) = ts[x]
_contract_seq(ts::Vector, x::Union{Vector, Tuple}) = mapreduce(y -> _contract_seq(ts, y), *, x)

#Indices of `linds` actually present on `t` (the seam convention: absent entries are
#ignored), as a concrete vector
_present_inds(t::AbstractTensor, linds) = filter(i -> i ∈ t.inds, _indvec(linds))

# ── Tensor ──────────────────────────────────────────────────────────────────────────────

struct Tensor{T, N, A <: AbstractArray{T, N}} <: AbstractTensor
    inds::Vector{Index{Int}}
    data::A
    function Tensor(inds::AbstractVector, data::A) where {T, N, A <: AbstractArray{T, N}}
        inds = collect(Index{Int}, inds)
        length(inds) == N || error("Tensor: $(length(inds)) indices for a rank-$N array")
        all(i -> dimof(i) == size(data, findfirst(==(i), inds)), unique(inds)) ||
            error("Tensor: index dimensions $(TensorInterface.dim.(inds)) don't match array size $(size(data))")
        return new{T, N, A}(inds, data)
    end
end

Tensor(x::Number) = Tensor(Index{Int}[], fill(x))

_like(t::Tensor, inds, data) = Tensor(inds, data)

Base.eltype(::Tensor{T}) where {T} = T
Base.sum(t::Tensor) = sum(t.data)

TensorInterface.datatype(t::Tensor) = typeof(vec(t.data))
TensorInterface.array(t::Tensor) = t.data
TensorInterface.data(t::Tensor) = vec(t.data)
TensorInterface.new_index(t::Tensor, d::Integer; tags = "") = Index(d, tags)

function TensorInterface.scalar(t::Tensor)
    length(t.data) == 1 || error("scalar: Tensor with inds $(t.inds) is not a scalar")
    #single-element reduce: reads the value without scalar indexing on GPU arrays
    return sum(t.data)
end

# ── Index-set queries ───────────────────────────────────────────────────────────────────

_indvec(i::Index) = Index[i]
_indvec(is::AbstractVector) = collect(Index, is)
_indvec(is::Tuple) = collect(Index, is)

const IndsLike = Union{AbstractTensor, Index, AbstractVector{<:Index}, Tuple{Index, Vararg{Index}}}

TensorInterface.commoninds(a::IndsLike, b::IndsLike) = filter(i -> i ∈ _indvec(b), _indvec(a))
# Convention: the singular forms return the FIRST match (or nothing), not `only`.
function TensorInterface.commonind(a::IndsLike, b::IndsLike)
    cs = TensorInterface.commoninds(a, b)
    return isempty(cs) ? nothing : first(cs)
end
TensorInterface.uniqueinds(a::IndsLike, b::IndsLike) = filter(i -> i ∉ _indvec(b), _indvec(a))
TensorInterface.unioninds(a::IndsLike, b::IndsLike) = unique(vcat(_indvec(a), _indvec(b)))
function TensorInterface.noncommoninds(a::IndsLike, b::IndsLike)
    return vcat(TensorInterface.uniqueinds(a, b), TensorInterface.uniqueinds(b, a))
end
function TensorInterface.noncommonind(a::IndsLike, b::IndsLike)
    ns = TensorInterface.noncommoninds(a, b)
    return isempty(ns) ? nothing : first(ns)
end
TensorInterface.hascommoninds(a::IndsLike, b::IndsLike) = !isempty(TensorInterface.commoninds(a, b))

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

TensorInterface.from_array(A::AbstractArray, is::Index{<:Integer}...) = Tensor(collect(Index, is), reshape(copy(A), TensorInterface.dim.(is)...))
TensorInterface.from_array(A::AbstractVector, i::Index{<:Integer}) = Tensor(Index[i], copy(A))

function TensorInterface.random_tensor(elt::Type, is::AbstractVector{<:Index})
    return Tensor(collect(is), randn(elt, TensorInterface.dim.(is)...))
end
TensorInterface.random_tensor(elt::Type, is::Index...) = TensorInterface.random_tensor(elt, collect(is))
TensorInterface.random_tensor(is::AbstractVector{<:Index}) = TensorInterface.random_tensor(Float64, is)
TensorInterface.random_tensor(is::Index...) = TensorInterface.random_tensor(Float64, collect(is))

function TensorInterface.onehot(elt::Type, p::Pair{<:Index, <:Integer})
    i, v = p
    data = zeros(elt, dimof(i))
    data[v] = one(elt)
    return Tensor(Index[i], data)
end
TensorInterface.onehot(p::Pair{<:Index, <:Integer}) = TensorInterface.onehot(Float64, p)

#Index vectors are often abstractly typed, so the dense/graded split is decided by content
function TensorInterface.delta(elt::Type, is::AbstractVector{<:Index})
    isempty(is) && return Tensor(one(elt))
    all(i -> space(i) isa TK.GradedSpace, is) && return _delta_tk(elt, is)
    data = zeros(elt, TensorInterface.dim.(is)...)
    for k in 1:minimum(TensorInterface.dim.(is))
        data[ntuple(_ -> k, length(is))...] = one(elt)
    end
    return Tensor(collect(is), data)
end
TensorInterface.delta(is::AbstractVector{<:Index}) = TensorInterface.delta(Float64, is)
TensorInterface.delta(elt::Type, is::Index...) = TensorInterface.delta(elt, collect(is))
TensorInterface.delta(is::Index...) = TensorInterface.delta(Float64, collect(is))

# A combiner is an explicit reshape isometry: identity data between the combined index
# (first) and the product of the combined indices. `t * C` combines; multiplying by `C`
# again splits back.
function TensorInterface.combiner(is::AbstractVector{<:Index}; tags = "CMB,Link")
    isempty(is) && error("combiner: no indices to combine")
    D = prod(TensorInterface.dim.(is))
    c = Index(D, String(tags))
    data = reshape(Matrix{Float64}(LinearAlgebra.I, D, D), (D, TensorInterface.dim.(is)...))
    return Tensor(vcat([c], collect(Index, is)), data)
end
TensorInterface.combiner(is::Index...; kwargs...) = TensorInterface.combiner(collect(Index, is); kwargs...)

# Direct sum of two tensors along the paired index axes (`olds1[k]`/`olds2[k]` → `news[k]`,
# with dim(news[k]) = dim(olds1[k]) + dim(olds2[k])); all other indices must coincide.
function TensorInterface.directsum(
        news::AbstractVector{<:Index}, p1::Pair{<:Tensor}, p2::Pair{<:Tensor}
    )
    t1, olds1 = first(p1), collect(Index, last(p1))
    t2, olds2 = first(p2), collect(Index, last(p2))
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
TensorInterface.new_index(d::Integer; tags = "") = Index(d, String(tags))

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
LinearAlgebra.dot(a::Tensor, b::Tensor) = LinearAlgebra.dot(vec(a.data), vec(_align(a, b)))
LinearAlgebra.tr(t::Tensor) = _trace_all(t)

# Pairwise contraction over all common (id, plev) indices, through the same low-level
# TensorOperations path as the graded backend: the output comes out directly in the
# GEMM-natural (openA | openB) partition with no staging copy. Self-traces (a repeated
# index on ONE operand) are not supported here — use `tr`.
function Base.:*(a::Tensor, b::Tensor)
    # scalar fast paths
    ndims(a) == 0 && return Tensor(copy(b.inds), TensorInterface.scalar(a) * b.data)
    ndims(b) == 0 && return Tensor(copy(a.inds), TensorInterface.scalar(b) * a.data)

    oa, ca, cb = Int[], Int[], Int[]
    for (k, i) in enumerate(a.inds)
        j = findfirst(==(i), b.inds)
        j === nothing ? push!(oa, k) : (push!(ca, k); push!(cb, j))
    end
    ob = Int[k for k in 1:length(b.inds) if k ∉ cb]

    pA = (Tuple(oa), Tuple(ca))
    pB = (Tuple(cb), Tuple(ob))
    pAB = (Tuple(1:length(oa)), Tuple(length(oa) .+ (1:length(ob))))
    TC = promote_type(eltype(a), eltype(b))
    C = TensorOperations.tensoralloc_contract(TC, a.data, pA, false, b.data, pB, false, pAB, Val(false))
    TensorOperations.tensorcontract!(C, a.data, pA, false, b.data, pB, false, pAB)
    return Tensor(vcat(a.inds[oa], b.inds[ob]), C)
end

function _trace_all(t::Tensor)
    # contract each plev-1 index with its plev-0 partner (same id): the partial trace
    lo = filter(i -> i.plev == 0, t.inds)
    hi = filter(i -> i.plev == 1, t.inds)
    labels = Dict{Index, Int}()
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

function TensorInterface.contract(ts::Vector{<:Tensor}; sequence = nothing, dest = nothing, kwargs...)
    isnothing(sequence) && return reduce(*, ts)
    sequence isa Integer && return ts[sequence]
    #`dest` is a tensor the caller has finished with, offered as storage for the result.
    #Only safe once it has been consumed by an earlier step of the tree, which needs at
    #least three factors; with two, the root still reads it.
    length(ts) >= 3 || (dest = nothing)
    dest = dest isa Tensor ? dest.data : dest
    if _uniform_kernel_storage([t.data for t in ts]) !== nothing
        elsize = sizeof(mapreduce(eltype, promote_type, ts))
        _, total = _seq_temp_bytes(ts, sequence, elsize)
        if total <= SEQ_BUFFER_MAXBYTES[]
            buf = _kernel_buffer(first(ts).data)
            cp = TensorOperations.allocator_checkpoint!(buf)
            data, is = _exec_seq(ts, sequence, buf, true; dest)
            TensorOperations.allocator_reset!(buf, cp)
            return Tensor(is, data)
        end
    end
    return _contract_seq(ts, sequence)
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
function _exec_seq(ts, s::Integer, buf, isroot; dest = nothing)
    t = ts[s]
    return t.data, t.inds
end
function _exec_seq(ts, s::Union{Vector, Tuple}, buf, isroot; dest = nothing)
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
        if isroot && k == n
            X, oa, ob = _tc_pair_into(dest, X, ix, Y, iy, buf)
        else
            X, oa, ob = _tc_pair(X, ix, Y, iy, false, identity, Val(true), buf)
        end
        ix = vcat(oa, ob)
    end
    return X, ix
end


# ── Diagonal operations ─────────────────────────────────────────────────────────────────


function TensorInterface.map_diag!(f::Function, out::Tensor, t::Tensor)
    ndims(t) == 2 || error("map_diag: expected a 2-index Tensor")
    d = _align(out, t)
    d === out.data || copyto!(out.data, d)
    #broadcast over a diagonal view (device-friendly: no scalar indexing on GPU arrays)
    dv = view(out.data, LinearAlgebra.diagind(out.data)[1:minimum(size(out.data))])
    dv .= f.(dv)
    return out
end

# ── Adapt / storage ─────────────────────────────────────────────────────────────────────

#eltype conversion preserves the storage container (GPU arrays stay on device)
Adapt.adapt_structure(elt::Type{<:Number}, t::Tensor) = Tensor(copy(t.inds), elt.(t.data))
function Adapt.adapt_structure(to::Type{<:AbstractVector}, t::Tensor)
    return Tensor(copy(t.inds), reshape(adapt(to, vec(copy(t.data))), size(t.data)))
end
Adapt.adapt_structure(to, t::Tensor) = Tensor(copy(t.inds), adapt(to, t.data))

# ── Factorizations (MatrixAlgebraKit; in-place/preallocated variants are the v3 target) ─

#Truncation report for factorize_svd (the kept spectrum itself is available through
#`singular_values!`)
struct Spectrum
    truncerr::Float64
end

# Permute+reshape to a (linds...) × (rest...) matrix. Returns matrix, left inds, right inds.
function _matricize(t::Tensor, linds::Vector{<:Index})
    rinds = TensorInterface.uniqueinds(t, linds)
    perm = map(i -> findfirst(==(i), t.inds), vcat(linds, rinds))
    any(isnothing, perm) && error("factorize: indices $(linds) not all found on tensor $(t.inds)")
    A = permutedims(t.data, perm)
    dl = prod(Int[dimof(i) for i in linds]; init = 1)
    dr = prod(Int[dimof(i) for i in rinds]; init = 1)
    return reshape(A, dl, dr), linds, rinds
end

# Seam-convention truncation as a MatrixAlgebraKit strategy: keep the smallest set of
# singular values whose discarded Σs² fraction is ≤ cutoff (⇔ 2-norm rtol = √cutoff),
# capped at maxdim. `nothing` means no constraint.
function _mak_trunc(; maxdim = nothing, cutoff = nothing)
    strategies = Any[]
    isnothing(maxdim) || push!(strategies, truncrank(Int(maxdim)))
    isnothing(cutoff) || push!(strategies, truncerror(; rtol = sqrt(cutoff), p = 2))
    return isempty(strategies) ? nothing : reduce(&, strategies)
end

function LinearAlgebra.qr(t::Tensor, linds; kwargs...)
    #`_matricize` hands back a freshly permuted matrix nobody else references, so the
    #in-place factorization (which destroys its input) is safe here and skips the copy
    #`qr_compact` would otherwise make — measured 2.8|A| of allocation down to 0.13|A|.
    A, li, ri = _matricize(t, _indvec(linds))
    m, n = size(A)
    k = min(m, n)
    Q, R = similar(A, m, k), similar(A, k, n)
    qr_compact!(A, (Q, R))
    b = Index(size(Q, 2), "Link,qr")
    Qt = Tensor(vcat(li, [b]), reshape(Q, (Int[dimof(i) for i in li]..., size(Q, 2))))
    Rt = Tensor(vcat([b], ri), reshape(R, (size(R, 1), Int[dimof(i) for i in ri]...)))
    return Qt, Rt
end

#Dense diagonal matrix in the same storage family as `S` (device-friendly: broadcast
#into a diagind view, no scalar indexing).
function _diag_matrix(S::AbstractVector)
    M = similar(S, length(S), length(S))
    fill!(M, zero(eltype(S)))
    view(M, LinearAlgebra.diagind(M)) .= S
    return M
end

# Matrix-level truncated SVD with the seam truncerr convention (discarded Σs²/total).
#`A` is always a freshly matricized copy (see `_svd_split`), so the in-place
#factorizations — which destroy their input — are safe and skip the copy the
#non-mutating forms make (measured 6.5|A| -> 2.8|A| compact, 6.7|A| -> 4.7|A| truncated).
function _svd_matrix(A::AbstractMatrix; maxdim = nothing, cutoff = nothing)
    trunc = _mak_trunc(; maxdim, cutoff)
    if trunc === nothing
        m, n = size(A)
        k = min(m, n)
        U = similar(A, m, k)
        S = Diagonal(similar(A, real(eltype(A)), k))
        Vt = similar(A, k, n)
        svd_compact!(A, (U, S, Vt))
        truncerr = 0.0
    else
        U, S, Vt, err = svd_trunc!(A; trunc)
        # MAK reports ‖discarded‖₂; the seam convention is discarded Σs² / total Σs².
        kept2 = sum(abs2, diag(S))
        total = kept2 + err^2
        truncerr = total > 0 ? err^2 / total : 0.0
    end
    return U, diag(S), Vt, truncerr
end

function _svd_split(t::Tensor, linds::Vector{<:Index}; maxdim = nothing, cutoff = nothing)
    A, li, ri = _matricize(t, linds)
    U, S, Vt, truncerr = _svd_matrix(A; maxdim, cutoff)
    return U, S, Vt, li, ri, truncerr
end

"""
factorize_svd matching the seam convention used by `simple_update`:
`ortho = "none"` returns F1 = U√S and F2 = √S·V sharing the primed bond `u′`, with the
singular values reported on unprimed `(u, v)`; `spec.truncerr` is the discarded Σs² fraction.
"""
function TensorInterface.factorize_svd(
        t::Tensor, linds;
        ortho = "none", singular_values! = nothing,
        maxdim = nothing, cutoff = nothing, tags = nothing, kwargs...,
    )
    U, S, Vt, li, ri, truncerr = _svd_split(t, _present_inds(t, linds); maxdim, cutoff)
    k = length(S)
    T = eltype(U)
    u = Index(k, "Link,u")
    v = Index(k, "Link,v")
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
        singular_values![] = Tensor(Index[u, v], _diag_matrix(S))
    end
    return F1, F2, Spectrum(truncerr)
end

# factorize: L isometric for ortho="left" (L=U, R=SV), mirrored for "right".
# The bond is unprimed and carries `tags`.
function LinearAlgebra.factorize(
        t::Tensor, linds...;
        ortho = "left", maxdim = nothing, cutoff = nothing, tags = "Link,fact", kwargs...,
    )
    lv = length(linds) == 1 ? _indvec(only(linds)) : collect(Index, linds)
    # Convention: entries of linds not present on the tensor are ignored
    lv = filter(i -> i ∈ t.inds, lv)
    U, S, Vt, li, ri, _ = _svd_split(t, lv; maxdim, cutoff)
    k = length(S)
    T = eltype(U)
    b = Index(k, String(tags))
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

# svd: U(linds, u), S(u, v), V(rinds, v).
function LinearAlgebra.svd(t::Tensor, linds; maxdim = nothing, cutoff = nothing, kwargs...)
    U, S, Vt, li, ri, _ = _svd_split(t, _present_inds(t, linds); maxdim, cutoff)
    k = length(S)
    u = Index(k, "Link,u")
    v = Index(k, "Link,v")
    Ut = Tensor(vcat(li, [u]), reshape(U, (Int[dimof(i) for i in li]..., k)))
    St = Tensor(Index[u, v], _diag_matrix(S))
    Vt_ = Tensor(vcat(ri, [v]), reshape(copy(transpose(Vt)), (Int[dimof(i) for i in ri]..., k)))
    return Ut, St, Vt_
end


# eigen, hermitian convention:
# D on (link′, link) with the eigenvalues, U on (rinds..., link); Ul·D·dag(U) reconstructs.
function LinearAlgebra.eigen(t::Tensor, linds, rinds; ishermitian::Bool = false, kwargs...)
    ishermitian || error("eigen: only ishermitian = true is implemented for Tensors")
    lv, rv = _indvec(linds), _indvec(rinds)
    A, li, ri = _matricize(t, lv)
    ri == rv || error("eigen: rinds don't match the remaining indices")
    D, vecs = eigh_full((A + A') / 2)
    vals = diag(D)
    k = length(vals)
    lk = Index(k, "Link,eigen")
    D = Tensor(Index[TensorInterface.prime(lk), lk], _diag_matrix(vals))
    U = Tensor(vcat(rv, [lk]), reshape(vecs, (Int[dimof(i) for i in rv]..., k)))
    return D, U
end

# ── Contraction workspace ───────────────────────────────────────────────────────────────
# Task-local reusable arenas backing `contract(ts; sequence)`: chain temporaries live
# here instead of the heap, and a GPU tensor gets an arena carved from its own device
# memory (TensorOperations >= 5.8 hands out device temporaries from device-backed
# buffers), so the discipline holds on device too.

#Task-local reusable buffers, one per storage family: host tensors get the default
#(Memory-backed) buffer; GPU tensors get a buffer carved from their own device memory
#(TensorOperations ≥ 5.8 hands out device temporaries from device-backed buffers), so the
#same discipline holds on device.
_buffer_storage(::Array) = TensorOperations.DefaultStorageType
_buffer_storage(A::Base.ReshapedArray) = _buffer_storage(parent(A))
_buffer_storage(A::SubArray) = _buffer_storage(parent(A))
_buffer_storage(A::AbstractArray) = typeof(similar(A, UInt8, 0))
function _kernel_buffer(A::AbstractArray)
    S = _buffer_storage(A)
    d = get!(task_local_storage(), :tensors_kernel_buffers) do
        Dict{DataType, TensorOperations.BufferAllocator}()
    end::Dict{DataType, TensorOperations.BufferAllocator}
    return get!(d, S) do
        TensorOperations.BufferAllocator{S}()
    end::TensorOperations.BufferAllocator{S}
end

#Storage families the buffered path supports: host arrays and GPU arrays (via
#TensorOperations' GPUArrays-backed buffer and contraction extensions). All participating
#tensors must share one family — mixed-device lists take the unbuffered path.
_kernel_storage_ok(A::Array) = true
_kernel_storage_ok(A::AbstractGPUArray) = true
_kernel_storage_ok(A::Base.ReshapedArray) = _kernel_storage_ok(parent(A))
_kernel_storage_ok(A::SubArray) = _kernel_storage_ok(parent(A))
_kernel_storage_ok(A) = false

#underlying storage of (possibly view/reshape-wrapped) tensor data
_root_storage(A::Base.ReshapedArray) = _root_storage(parent(A))
_root_storage(A::SubArray) = _root_storage(parent(A))
_root_storage(A) = A
function _uniform_kernel_storage(arrays)
    all(_kernel_storage_ok, arrays) || return nothing
    W = typeof(_root_storage(first(arrays))).name.wrapper
    all(a -> typeof(_root_storage(a)).name.wrapper === W, arrays) || return nothing
    return first(arrays)
end


# Pairwise contraction writing into `dest`'s storage (a reshaped prefix view) when it fits
# and the eltypes match; a fresh allocation otherwise. Used to let a caller hand ownership
# of an input it no longer needs — the storage, not the contraction, is what is recycled.
function _tc_pair_into(dest, da, ia::Vector{<:Index}, db, ib::Vector{<:Index}, allocator)
    ca, cb = Int[], Int[]
    for (j, bj) in enumerate(ib)
        i = findfirst(==(bj), ia)
        i === nothing && continue
        push!(ca, i)
        push!(cb, j)
    end
    oa = setdiff(1:length(ia), ca)
    ob = setdiff(1:length(ib), cb)
    pA, pB = (Tuple(oa), Tuple(ca)), (Tuple(cb), Tuple(ob))
    pAB = (Tuple(1:(length(oa) + length(ob))), ())
    TC = promote_type(eltype(da), eltype(db))
    dims = (size(da)[oa]..., size(db)[ob]...)
    n = prod(dims; init = 1)
    root = dest === nothing ? nothing : _root_storage(dest)
    C = if root !== nothing && eltype(root) === TC && n <= length(root)
        reshape(view(vec(root), 1:n), dims)
    else
        TensorOperations.tensoralloc_contract(TC, da, pA, false, db, pB, false, pAB, Val(false))
    end
    backend = TensorOperations.select_backend(TensorOperations.tensorcontract!, C, da, db)
    TensorOperations.tensorcontract!(C, da, pA, false, db, pB, false, pAB, one(TC), zero(TC), backend, allocator)
    return C, ia[oa], ib[ob]
end

# One pairwise contraction at the data level. `pairfn` maps an index of `b` to the index of
# `a` it should contract with (identity for plain absorption; the site/prime partner map for
# the closing bra contraction). Returns (data, out_inds).
function _tc_pair(
        da, ia::Vector{<:Index}, db, ib::Vector{<:Index}, conjB::Bool, pairfn,
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
    backend = TensorOperations.select_backend(TensorOperations.tensorcontract!, C, da, db)
    TensorOperations.tensorcontract!(C, da, pA, false, db, pB, conjB, pAB, one(TC), zero(TC), backend, allocator)
    return C, ia[oa], ib[ob]
end



# ── Fused double-layer BP kernel ────────────────────────────────────────────────────────
# The ONE specialised path in the package: the double-layer closure
#     out = (∏ incoming) · ψ · conj(ψ)      (open outgoing bond, or scalar, or with an op)
# which is the inner loop of every BP sweep and every BP expectation value. Measured on a
# comb tree, χ=60, ComplexF32: 5.1F against the generic path's 6.1F, at parity on walltime.
# Two savings, both free:
#
#   * never materialises `dag(prime(ψ))` — the conjugation rides the closing gemm     (-1F)
#   * ping-pongs two slots instead of accumulating one intermediate per factor        (-1F)
#
# It does NOT reach 3F. Getting there means also owning the layout permutes so that TO
# never allocates its own scratch (2F of the 5.1F is exactly that scratch) — measured, the
# hand-placed version hits 3.1F but runs 2× slower, so it is not what is implemented here.
# None of these numbers are known to transfer to GPU: cuTENSOR plans its own workspace.
#
# Layout is NOT ours: every step is a plain `TensorOperations.tensorcontract!` in the
# operands' natural index order, so TO keeps its transpose-flag and strided-view tricks and
# permutes only when it must. Hand-placing the permutes ourselves — two earlier versions of
# this kernel did, unconditionally and then conditionally — costs 8× and 2× respectively,
# because a materialised permute is strictly more data movement than the gemm TO can do
# in place. The kernel's whole contribution is *where the outputs land*, not how they are
# laid out.
#
# Everything else in the package goes through `contract(ts; sequence)`.

#Absorb 2-index factors (messages, then optionally an operator) into ψ, ping-ponging two
#slots so the chain never holds more than one intermediate. Returns the result, its indices,
#and the slot the result does NOT occupy, which the caller's closing gemm can then reuse.
function _absorb_chain_2slot(X, ix::Vector{<:Index}, ms::Vector{<:Tensor}, buf)
    TC = isempty(ms) ? eltype(X) : promote_type(eltype(X), mapreduce(eltype, promote_type, ms))
    n = length(X)
    at = _temp_arraytype(X, TC, 1)
    slots = (TensorOperations.tensoralloc(at, (n,), Val(true), buf),
             TensorOperations.tensoralloc(at, (n,), Val(true), buf))
    isempty(ms) && return X, ix, slots[1]
    cur = 1
    for m in ms
        j = findfirst(i -> i ∈ ix, m.inds)
        j === nothing && error("_absorb_chain_2slot: factor shares no index with the chain")
        c = findfirst(==(m.inds[j]), ix)
        nk = length(ix)
        oa = Int[a for a in 1:nk if a != c]
        odims = (ntuple(a -> size(X, oa[a]), nk - 1)..., dimof(m.inds[3 - j]))
        C = reshape(view(slots[cur], 1:prod(odims)), odims)
        backend = TensorOperations.select_backend(TensorOperations.tensorcontract!, C, X, m.data)
        TensorOperations.tensorcontract!(
            C, X, (Tuple(oa), (c,)), false, m.data, ((j,), (3 - j,)), false,
            (Tuple(1:nk), ()), one(TC), zero(TC), backend, buf
        )
        X, ix, cur = C, vcat(ix[oa], [m.inds[3 - j]]), 3 - cur
    end
    return X, ix, slots[cur]
end

function fused_norm_closure(
        ψ::Tensor, sinds::Vector{<:Index}, incoming::Vector{<:Tensor};
        op::Union{Nothing, Tensor} = nothing
    )
    arrays = Any[ψ.data]
    append!(arrays, (m.data for m in incoming))
    op === nothing || push!(arrays, op.data)
    _uniform_kernel_storage(arrays) === nothing && return nothing
    all(m -> _is_norm_message(m, ψ.inds), incoming) || return nothing

    buf = _kernel_buffer(ψ.data)
    cp = TensorOperations.allocator_checkpoint!(buf)
    chain = op === nothing ? incoming : vcat(incoming, [op])
    covered = op === nothing ? Index[] : Index[i for i in op.inds if i.plev == 0]
    #ψ-leg i closes against chain-leg `partner(i)`: itself for uncovered sites (ket meets
    #bra directly), its prime for message-bridged virtuals and operator-covered sites
    partner = i -> (i ∈ sinds && i ∉ covered) ? i : TensorInterface.prime(i)
    X, ix, _ = _absorb_chain_2slot(ψ.data, ψ.inds, chain, buf)

    cX, cP, oP = Int[], Int[], Int[]
    for (b, i) in enumerate(ψ.inds)
        a = findfirst(==(partner(i)), ix)
        a === nothing ? push!(oP, b) : (push!(cX, a); push!(cP, b))
    end
    oX = Int[a for a in 1:length(ix) if a ∉ cX]
    TC = promote_type(eltype(X), eltype(ψ.data))
    #`conjB = true` closes against the bra without ever materialising `dag(prime(ψ))`
    odims = (ntuple(a -> size(X, oX[a]), length(oX))..., ntuple(a -> size(ψ.data, oP[a]), length(oP))...)
    out = TensorOperations.tensoralloc(_temp_arraytype(ψ.data, TC, length(odims)), odims, Val(false))
    bk = TensorOperations.select_backend(TensorOperations.tensorcontract!, out, X, ψ.data)
    TensorOperations.tensorcontract!(out, X, (Tuple(oX), Tuple(cX)), false,
        ψ.data, (Tuple(cP), Tuple(oP)), true, (Tuple(1:length(odims)), ()),
        one(TC), zero(TC), bk, buf)
    oinds = vcat(ix[oX], TensorInterface.prime.(ψ.inds[oP]))
    TensorOperations.allocator_reset!(buf, cp)
    return Tensor(oinds, out)
end

#Outgoing BP message from ψ given standard doubled incoming messages; `nothing` when the
#structure does not match, so the caller falls back to the generic path.
function fused_norm_message(
        ψ::Tensor, sinds::Vector{<:Index}, incoming::Vector{<:Tensor}; normalize::Bool = true
    )
    m = fused_norm_closure(ψ, sinds, incoming)
    m === nothing && return nothing
    if normalize
        s = sum(m.data)
        iszero(s) || rmul!(vec(m.data), inv(s))
    end
    return m
end

#`m` must be a standard doubled norm-network message for `ψ`: every index either a plev-0
#index of ψ or the prime of one.
function _is_norm_message(m::Tensor, ψinds::Vector{<:Index})
    return all(m.inds) do i
        (i.plev == 0 && i ∈ ψinds) || (i.plev == 1 && TensorInterface.noprime(i) ∈ ψinds)
    end
end

#Buffer-temp array type in the same storage family as `ref`.
_temp_arraytype(ref::AbstractArray, T::Type, N::Integer) =
    Base.typename(typeof(_root_storage(ref))).wrapper{T, N}


# ── S=1/2 operator & state library ──────────────────────────────────────────────────────

const _σI = ComplexF64[1 0; 0 1]
const _σx = ComplexF64[0 1; 1 0]
const _σy = ComplexF64[0 -im; im 0]
const _σz = ComplexF64[1 0; 0 -1]

_is_spinhalf(i::Index) = dimof(i) == 2

"""
    register_op!(name::String, f::Function; nsites::Int = 1)

Register a custom operator matrix for the Tensors backend. `f(; kwargs...)` must return
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
# data[s1', s2', s1, s2] via column-major reshape (pinned by the digests in
# test/test_tensors.jl).
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

function TensorInterface.op(name::String, i::Index; kwargs...)
    _is_spinhalf(i) || error("op: Tensors operator library currently covers d=2 (S=1/2) sites only")
    m = _op1(name; kwargs...)
    m === nothing && error("op: unknown single-site operator \"$name\" for the Tensors backend")
    return Tensor(Index[TensorInterface.prime(i), i], copy(m))
end

function TensorInterface.op(name::String, i1::Index, i2::Index; kwargs...)
    (_is_spinhalf(i1) && _is_spinhalf(i2)) || error("op: Tensors operator library currently covers d=2 (S=1/2) sites only")
    m = _op2(name; kwargs...)
    m === nothing && error("op: unknown two-site operator \"$name\" for the Tensors backend")
    data = reshape(copy(m), 2, 2, 2, 2)
    return Tensor(Index[TensorInterface.prime(i1), TensorInterface.prime(i2), i1, i2], data)
end

function TensorInterface.state(name::String, i::Index)
    _is_spinhalf(i) || error("state: Tensors state library currently covers d=2 (S=1/2) sites only")
    vecmap = Dict(
        "↑" => [1.0, 0.0], "0" => [1.0, 0.0], "Up" => [1.0, 0.0], "Z+" => [1.0, 0.0],
        "↓" => [0.0, 1.0], "1" => [0.0, 1.0], "Dn" => [0.0, 1.0], "Z-" => [0.0, 1.0],
        "+" => [1.0, 1.0] / sqrt(2), "X+" => [1.0, 1.0] / sqrt(2),
        "-" => [1.0, -1.0] / sqrt(2), "X-" => [1.0, -1.0] / sqrt(2),
    )
    haskey(vecmap, name) || error("state: unknown state \"$name\" for the Tensors backend")
    return Tensor(Index[i], copy(vecmap[name]))
end

include("gradedtensor.jl")

end
