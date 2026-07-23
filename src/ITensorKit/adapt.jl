# Element-type / device genericity (`datatype`, `adapt`) and diagonal maps.
#
# TNQS threads storage/eltype through `datatype(t)` and `adapt(to)(t)`, and maps
# functions over the diagonal of the singular-value / eigenvalue factors with
# `map_diag`/`map_diag!`. TensorKit's `TensorMap` already integrates with Adapt
# for array-type targets (it converts both storage container and scalartype);
# we forward to it and add a `Number`-target case that converts scalartype.

export datatype, map_diag, map_diag!

"""
    datatype(t::ITensorMap)

The underlying storage container type of `t` (e.g. `Vector{Float64}`,
`CuVector{ComplexF64}`). The ITensor name for TensorKit's `storagetype`; used as an
`adapt` target.
"""
datatype(t::ITensorMap) = storagetype(t.data)

# Generic adapt: forward to TensorKit's TensorMap adapt (handles array-type and
# `Vector{T}` targets, converting storage and scalartype); the labels carry no
# device data, so they pass through unchanged.
Adapt.adapt_structure(to, t::ITensorMap) = unsafe_itensormap(Adapt.adapt(to, t.data), t.inds)

# `adapt(T::Type{<:Number}, t)` converts the scalartype to `T`. Adapt otherwise
# treats a bare number type as a no-op, so this is handled explicitly. `similar`
# preserves the storage container kind (also correct for GPU arrays).
function Adapt.adapt_structure(::Type{T}, t::ITensorMap) where {T <: Number}
    scalartype(t) === T && return t
    data = copy!(similar(t.data, T), t.data)
    return unsafe_itensormap(data, t.inds)
end

# --- diagonal maps ----------------------------------------------------------
_diagdata(t::ITensorMap) = t.data isa DiagonalTensorMap ? t.data :
    throw(ArgumentError("map_diag expects a diagonal tensor (got $(typeof(t.data)))"))

"""
    map_diag(f, t::ITensorMap) -> ITensorMap

Apply `f` elementwise to the diagonal entries of a diagonal tensor (e.g. the `S`/`D`
factor from [`svd`](@ref)/[`eigen`](@ref)), returning a new diagonal tensor with the
same legs.
"""
function map_diag(f, t::ITensorMap)
    d = _diagdata(t)
    return unsafe_itensormap(DiagonalTensorMap(map(f, d.data), d.domain), t.inds)
end

"""
    map_diag!(f, dst::ITensorMap, src::ITensorMap) -> dst

In-place [`map_diag`](@ref): write `f` of `src`'s diagonal into `dst` (which may
alias `src`). Both must be diagonal tensors with matching legs.
"""
function map_diag!(f, dst::ITensorMap, src::ITensorMap)
    _diagdata(dst).data .= f.(_diagdata(src).data)
    return dst
end
