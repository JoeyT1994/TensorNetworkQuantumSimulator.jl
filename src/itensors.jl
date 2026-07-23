# TNQS-owned tensor utilities and the legacy `ITensors.jl`-style API, implemented over the
# next-gen `ITensorBase` / `TensorAlgebra` / `MatrixAlgebraKit` stack.

import MatrixAlgebraKit as MAK
using Adapt: Adapt
using ITensorBase: ITensorBase, ITensor, Index, inds, noprime, plev, prime, space, unnamed
using TensorAlgebra: TensorAlgebra, project, tryproject

# Project `a` onto the symmetry-restricted space given by `codomain`/`domain`, mirroring
# `TensorAlgebra.project`/`tryproject` (three-argument operator form; `a` is indexed positionally
# as `(codomain..., domain...)`). When the plain projection is charge-forbidden — a parity-odd
# state or operator — the residual charge is absorbed into a trailing auxiliary index instead of
# throwing, so the returned ITensor carries that extra aux leg last.
function project_aux(a::AbstractArray, codomain, domain)
    return @something tryproject(a, codomain, domain) begin
        projected_a = project(reshape(a, (size(a)..., 1)), space.(codomain), space.(domain))
        aux = Index(TensorAlgebra.axes(projected_a, ndims(a) + 1))
        ITensor(projected_a, (codomain..., domain..., aux))
    end
end
project_aux(a::AbstractArray, codomain) = project_aux(a, codomain, ())
project_aux(v::AbstractVector{<:Number}, i::Index) = project_aux(v, (i,))

# Build two tensors that share one contractible auxiliary leg, so `t1 * t2` gives the fermion
# string of e.g. `c†ᵢcⱼ` (a `c†` and a `c`) with no `flip`. The aux is minted by projecting `a1`
# with a trailing dummy axis, which absorbs `a1`'s residual charge, then attaching it to `a2`.
# This bare-axis core takes spaces (no names), so it could live beside `TensorAlgebra.project`.
function project_pair(a1::AbstractArray, codomain1, domain1, a2::AbstractArray, codomain2, domain2)
    p1 = project(reshape(a1, (size(a1)..., 1)), codomain1, domain1)
    aux = TensorAlgebra.axes(p1, ndims(a1) + 1)
    p2 = project(reshape(a2, (size(a2)..., 1)), codomain2, (domain2..., aux))
    return p1, p2
end

# Named worker: strip to bare axes, call the core, and reattach names — the one extra step vs
# `project`'s wrapper is minting a single fresh aux `Index` shared by both outputs, so `t1 * t2`
# contracts them by name.
function project_pair_itensor(a1, codomain1, domain1, a2, codomain2, domain2)
    p1, p2 = project_pair(
        a1, space.(codomain1), space.(domain1),
        a2, space.(codomain2), space.(domain2),
    )
    aux = Index(TensorAlgebra.axes(p1, ndims(a1) + 1))
    t1 = ITensor(p1, (codomain1..., domain1..., aux))
    t2 = ITensor(p2, (codomain2..., domain2..., aux))
    return t1, t2
end

# Named layer (mirrors ITensorBase's named `project`): two entries read operand 1's flavor from
# whichever of its sides is non-empty (an empty `codomain1` is the all-domain/state mirror of the
# empty-domain form), so a named operand never falls through to the bare-axis core.
function project_pair(
        a1::AbstractArray, codomain1::Tuple{Index, Vararg{Index}}, domain1::Tuple{Vararg{Index}},
        a2::AbstractArray, codomain2::Tuple{Vararg{Index}}, domain2::Tuple{Vararg{Index}}
    )
    return project_pair_itensor(a1, codomain1, domain1, a2, codomain2, domain2)
end
function project_pair(
        a1::AbstractArray, codomain1::Tuple{}, domain1::Tuple{Index, Vararg{Index}},
        a2::AbstractArray, codomain2::Tuple{Vararg{Index}}, domain2::Tuple{Vararg{Index}}
    )
    return project_pair_itensor(a1, codomain1, domain1, a2, codomain2, domain2)
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

# Whether `a` is an operator tensor: its indices are exactly the plev-0 indices paired with
# their primes. A plain state/vector has the plev-0 indices but not their primed partners.
function is_operator(a::ITensor)
    domain = filter(i -> plev(i) == 0, inds(a))
    codomain = prime.(domain)
    return issetequal(inds(a), [codomain; domain])
end

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
    return ITensor(convert(AbstractArray{elt}, unnamed(T)), inds(T))
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
