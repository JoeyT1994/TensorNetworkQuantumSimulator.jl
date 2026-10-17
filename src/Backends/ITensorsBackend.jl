#=
ITensorsBackend: the TensorInterface seam on ITensors 0.9 (General registry), dense storage only.

Every verb forwards to its ITensors namesake, whose conventions the seam follows (see the header
of src/tensor_interface.jl); the few the seam adds (`dense`, `scale!`, ...) are written here in
ITensors terms. This is the only module that references ITensors, together with the
compatibility shim ITensorsCompat.jl.
=#
module ITensorsBackend

using ITensors: ITensors, ITensor, Index
using ITensors.NDTensors: NDTensors
using LinearAlgebra: LinearAlgebra, rmul!
using Random: Random

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
TI.dense(t::ITensor) = ITensors.denseblocks(t)
# collections may nest (a vector of index tuples); ITensors flattens them
TI.combiner(is::Union{Index, AbstractVector, Tuple}...; kwargs...) = ITensors.combiner(is...; kwargs...)
TI.combinedind(C::ITensor) = ITensors.combinedind(C)
TI.directsum(p1::Pair{ITensor}, p2::Pair{ITensor}; tags = "sum") =
    first(ITensors.directsum(p1, p2; tags))
TI.directsum(news, p1::Pair{ITensor}, p2::Pair{ITensor}; kwargs...) =
    ITensors.directsum(news, p1, p2; kwargs...)

# ── Contraction ─────────────────────────────────────────────────────────────────────────
# ITensors warns about high-order intermediates; the package chooses its own sequences, so the
# warning is noise here.
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

# ── Storage / type queries ──────────────────────────────────────────────────────────────
TI.datatype(t::ITensor) = ITensors.datatype(t)
TI.scalartype(t::ITensor) = NDTensors.scalartype(t)
TI.array(t::ITensor, is::Index...) = ITensors.array(t, is...)
function TI.scale!(t::ITensor, x::Number)
    rmul!(ITensors.data(t), x)
    return t
end

# ── Hooks ───────────────────────────────────────────────────────────────────────────────
# ITensors' own operator library, for names the package registry does not define; ITensors
# signals an unknown name with an ArgumentError.
function TI.backend_op(name::String, sites::Index...; kwargs...)
    try
        return ITensors.op(name, sites...; kwargs...)
    catch err
        err isa ArgumentError && return nothing
        rethrow()
    end
end

include("ITensorsKernels.jl")

end
