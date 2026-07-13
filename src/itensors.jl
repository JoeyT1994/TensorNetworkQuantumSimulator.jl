# TNQS-owned tensor utilities and the legacy `ITensors.jl`-style API, implemented over the
# next-gen `ITensorBase` / `TensorAlgebra` / `MatrixAlgebraKit` stack.

import MatrixAlgebraKit as MAK
using Adapt: Adapt
using ITensorBase: ITensorBase, ITensor, Index, inds, noprime, plev, prime, space, unnamed
using TensorAlgebra: TensorAlgebra, project, tryproject

function project_aux(v::AbstractVector{<:Number}, i::Index)
    length(v) == length(i) ||
        error(
        "state vector has dimension $(length(v)) but the site index has dimension $(length(i))"
    )
    ψ = tryproject(v, (i,))
    isnothing(ψ) || return ψ
    raw = project(reshape(v, (length(v), 1)), (unnamed(i),), ())
    aux = Index(TensorAlgebra.axes(raw, 2))
    return ITensor(raw, (i, aux))
end

function onehot(eltype::Type, (i, p)::Pair{<:Index})
    v = zeros(eltype, length(i))
    v[p] = one(eltype)
    return project_aux(v, i)
end
onehot(p::Pair{<:Index}) = onehot(Float64, p)

function contract_network end
function contract_network(tensors::AbstractVector; sequence = nothing)
    return isnothing(sequence) ? reduce(*, tensors) : _contract_sequence(tensors, sequence)
end
_contract_sequence(tensors, s::Integer) = tensors[s]
_contract_sequence(tensors, s) = reduce(*, (_contract_sequence(tensors, x) for x in s))

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
function diagonaltensor(diag::AbstractVector, is::Tuple{Index, Vararg{Index}})
    return ITensor(diagonaltensor(diag, space.(is)), is)
end

delta(eltype::Type, is::Tuple) = diagonaltensor(ones(eltype, minimum(length, is)), is)
delta(eltype::Type, is::Index...) = delta(eltype, is)
delta(eltype::Type, is::AbstractVector{<:Index}) = delta(eltype, Tuple(is))
delta(is::Tuple) = delta(Float64, is)
delta(is::Index...) = delta(Float64, is)
delta(is::AbstractVector{<:Index}) = delta(Float64, Tuple(is))

# The codomain/domain bipartition of an operator tensor: each plev-0 index paired with its
# prime. Viewing the operator as this square map is what `tr` factors through.
function operator_inds(a::ITensor)
    domain = filter(i -> plev(i) == 0, inds(a))
    return prime.(domain), domain
end

apply(o::ITensor, ψ::ITensor) = noprime(o * ψ)

function itensor_trunc(; maxdim = nothing, cutoff = nothing)
    trunc = MAK.notrunc()
    isnothing(maxdim) || (trunc &= MAK.truncrank(maxdim))
    isnothing(cutoff) || iszero(cutoff) || (trunc &= MAK.truncerror(; rtol = sqrt(cutoff), p = 2))
    return trunc
end

struct ScalarTypeAdaptor{T} end
ScalarTypeAdaptor(T::Type) = ScalarTypeAdaptor{T}()
adapt_scalartype(T::Type) = Adapt.adapt(ScalarTypeAdaptor(T))
adapt_scalartype(T::Type, x) = Adapt.adapt(ScalarTypeAdaptor(T), x)
function Adapt.adapt_structure(::ScalarTypeAdaptor{elt}, T::ITensor) where {elt}
    eltype(T) === elt && return T
    return ITensor(convert(AbstractArray{elt}, unnamed(T)), ITensorBase.dimnames(T))
end

function directsum(out_inds, pairs::Pair...)
    t1, s1 = first(pairs[1]), last(pairs[1])
    shared = setdiff(inds(t1), s1)
    target = (shared..., out_inds...)
    out = zeros(eltype(t1), length.(target))
    offsets = zeros(Int, length(out_inds))
    for p in pairs
        t, sinds = first(p), last(p)
        a = ITensorBase.unname(t, (shared..., sinds...))
        ranges = (
            Base.OneTo.(length.(shared))...,
            ntuple(k -> (offsets[k] + 1):(offsets[k] + length(sinds[k])), length(sinds))...,
        )
        out[ranges...] .= a
        offsets .+= length.(sinds)
    end
    return out[target...]
end

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
macro Algorithm_str(s)
    return :(Algorithm{$(Expr(:quote, Symbol(s)))})
end
