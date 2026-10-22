# The TensorInterface seam on ITensors (dense storage only). Most verbs forward to their
# ITensors namesake; graded queries answer for dense data.
module ITensorsBackend

using ITensors: ITensors, ITensor, Index
using ITensors.NDTensors: NDTensors
using LinearAlgebra: LinearAlgebra, I, diag, rmul!
using Random: Random
using Adapt: adapt

import ..TensorInterface as TI

const Tensor = ITensor

export Tensor, Index

# Index-carrying arguments the forwarded verbs accept: a tensor, an index, or a collection.
const IndexCollection = Union{AbstractVector{<:Index}, Tuple{Vararg{Index}}}
const IndexLike = Union{Index, IndexCollection}
const TensorOrIndex = Union{ITensor, IndexLike}

# ── Index queries, transforms, replacement ──────────────────────────────────────────────
TI.inds(t::ITensor; kwargs...) = ITensors.inds(t; kwargs...)
for f in (:commonind, :commoninds, :uniqueinds, :unioninds, :noncommonind, :noncommoninds,
        :hascommoninds)
    @eval TI.$f(a::TensorOrIndex, b...; kwargs...) = ITensors.$f(a, b...; kwargs...)
end
TI.dim(x::Union{ITensor, IndexLike}) = ITensors.dim(x)
TI.plev(i::Index) = ITensors.plev(i)
TI.tags(x::Union{ITensor, Index}) = ITensors.tags(x)
TI.isgraded(x::Union{ITensor, Index}) = ITensors.hasqns(x)
TI.isdual(i::Index) = false
TI.space(i::Index) = ITensors.dim(i)

TI.dag(x::TensorOrIndex) = ITensors.dag(x)
TI.dag(t::ITensor, cod) = ITensors.dag(t)          # dense: the map adjoint is the bra (rule 5)
TI.prime(x::TensorOrIndex, args...; kwargs...) = ITensors.prime(x, args...; kwargs...)
TI.noprime(x::TensorOrIndex, args...; kwargs...) = ITensors.noprime(x, args...; kwargs...)
TI.sim(i::Index; kwargs...) = ITensors.sim(i; kwargs...)
TI.sim(is::IndexCollection; kwargs...) = map(i -> ITensors.sim(i; kwargs...), is)
TI.replaceind(t::ITensor, args...) = ITensors.replaceind(t, args...)
TI.replaceinds(t::ITensor, args...) = ITensors.replaceinds(t, args...)

# ── Construction ────────────────────────────────────────────────────────────────────────
TI.new_index(d::Integer; tags = "") = Index(Int(d), tags)
TI.new_index(ref, d::Integer; tags = "") = Index(Int(d), tags)
function TI.pad_index(w::Index, ins, kt::Integer; tags = "Link,pad")
    k = ITensors.dim(w)
    return k >= kt ? nothing : Index(Int(kt) - k, tags)
end

TI.from_array(A::AbstractArray, is::IndexCollection) = ITensor(copy(A), is...)
TI.from_array(A::AbstractArray, i::Index, is::Index...) = ITensor(copy(A), i, is...)
TI.from_array(A::AbstractArray, outs::IndexCollection, ins::IndexCollection) =
    ITensor(copy(A), outs..., ins...)

function TI.random_tensor(rng::Random.AbstractRNG, elt::Type, is::IndexCollection)
    isempty(is) && return ITensor(fill(randn(rng, elt)))
    return ITensors.random_itensor(rng, elt, is...)
end
TI.random_tensor(rng::Random.AbstractRNG, elt::Type, i::Index, is::Index...) =
    TI.random_tensor(rng, elt, [i, is...])
TI.random_tensor(elt::Type, is::IndexCollection) = TI.random_tensor(Random.default_rng(), elt, is)
TI.random_tensor(elt::Type, i::Index, is::Index...) = TI.random_tensor(elt, [i, is...])
TI.random_tensor(is::IndexCollection) = TI.random_tensor(Float64, is)
TI.random_tensor(i::Index, is::Index...) = TI.random_tensor(Float64, [i, is...])

TI.onehot(p::Pair{<:Index}...) = ITensors.onehot(p...)
TI.onehot(elt::Type, p::Pair{<:Index}...) = ITensors.onehot(elt, p...)
TI.delta(x::IndexLike, xs::IndexLike...) = ITensors.delta(x, xs...)
TI.delta(elt::Type{<:Number}, xs::IndexLike...) = ITensors.delta(elt, xs...)
# Diagonal storage densifies directly. A combiner is a structured object that can only fuse or
# unfuse its legs whole, so it is made explicit by fusing an identity through it.
function TI.dense(t::ITensor)
    ITensors.storage(t) isa NDTensors.Combiner || return ITensors.denseblocks(t)
    is = collect(ITensors.inds(t))[2:end]   # the combined index comes first
    n = ITensors.dim(is)
    js = ITensors.sim.(is)                  # not prime: b and b′ legs could collide
    Id = ITensor(Matrix{Float64}(I, n, n), is..., js...)
    return ITensors.replaceinds(Id * t, js, is)
end
# collections may nest (a vector of index tuples); ITensors flattens them
TI.combiner(is::Union{Index, AbstractVector, Tuple}...; kwargs...) = ITensors.combiner(is...; kwargs...)
TI.combinedind(C::ITensor) = ITensors.combinedind(C)
TI.directsum(p1::Pair{ITensor}, p2::Pair{ITensor}; tags = "sum") =
    first(ITensors.directsum(p1, p2; tags))
TI.directsum(news, p1::Pair{ITensor}, p2::Pair{ITensor}; kwargs...) =
    ITensors.directsum(news, p1, p2; kwargs...)
TI.charge_sectors(legs) = [Index(1, "Charge")]

# ── Contraction ─────────────────────────────────────────────────────────────────────────
# The package chooses its own sequences, so ITensors' high-order warning is noise.
function TI.contract(ts::AbstractVector{<:ITensor}; kwargs...)
    return ITensors.@disable_warn_order ITensors.contract(ts; kwargs...)
end
function TI.contract(ts::AbstractVector; kwargs...)
    all(t -> t isa ITensor, ts) || throw(ArgumentError("contract: expected a vector of tensors"))
    return TI.contract(convert(Vector{ITensor}, ts); kwargs...)
end
TI.scalar(t::ITensor) = ITensors.scalar(t)
TI.apply(o::ITensor, t::ITensor; kwargs...) = ITensors.apply(o, t; kwargs...)

# ── Diagonal ops, factorizations ────────────────────────────────────────────────────────
TI.map_diag(f::Function, t::ITensor) = ITensors.map_diag(f, t)
TI.map_diag!(f::Function, out::ITensor, t::ITensor) = ITensors.map_diag!(f, out, t)
TI.factorize_svd(t::ITensor, args...; kwargs...) = ITensors.factorize_svd(t, args...; kwargs...)
TI.gram(a::ITensor, b::ITensor, legs) = ITensors.dag(a) * b

# Full SVD, then restrict U, S, V to the kept singular triplets on fresh bonds.
function TI.truncated_svd(t::ITensor, linds; trunc::TI.TruncationStrategy)
    U, S, V = LinearAlgebra.svd(t, linds)
    u = ITensors.commonind(U, S)
    v = ITensors.commonind(S, V)
    s = Array(abs.(diag(ITensors.array(S, u, v))))       # the rule runs on the host
    k = TI.kept_count(s, trunc)
    k == length(s) && return U, S, V
    uk, vk = Index(k, ITensors.tags(u)), Index(k, ITensors.tags(v))
    on_device(t) = adapt(ITensors.datatype(S))(t)
    Pu = on_device(ITensor(Matrix{real(eltype(S))}(I, length(s), k), u, uk))
    Pv = on_device(ITensor(Matrix{real(eltype(S))}(I, length(s), k), v, vk))
    return U * Pu, Pu * ITensors.denseblocks(S) * Pv, V * Pv
end

# ── Storage / type queries ──────────────────────────────────────────────────────────────
TI.datatype(t::ITensor) = ITensors.datatype(t)
TI.scalartype(t::ITensor) = NDTensors.scalartype(t)
TI.array(t::ITensor, is::Index...) = ITensors.array(t, is...)
TI.data(t::ITensor) = ITensors.data(t)
function TI.scale!(t::ITensor, x::Number)
    rmul!(ITensors.data(t), x)
    return t
end

include("ITensorsKernels.jl")

end
