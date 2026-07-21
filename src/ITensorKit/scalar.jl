# Scalar extraction and dense-array conversion for `ITensorMap`.

export scalar, array

"""
    scalar(t::ITensorMap)
    t[]

The single number held by a 0-leg tensor (e.g. the result of a full contraction).
Errors if `t` has any legs.
"""
function TensorKit.scalar(t::ITensorMap)
    numind(t) == 0 || throw(ArgumentError("scalar: tensor has $(numind(t)) legs, expected 0"))
    return scalar(t.data)
end
Base.getindex(t::ITensorMap) = scalar(t)

"""
    array(t::ITensorMap) -> Array

A dense Julia `Array` of `t`'s entries with axes in `inds(t)` order. Inverse of
[`itensor`](@ref): `itensor(array(t), inds(t)) == t`.
"""
function array(t::ITensorMap)
    numind(t) == 0 && return fill(scalar(t))
    A = convert(Array, t.data)
    return reshape(A, map(dim, inds(t)))
end

"""
    diag(t::ITensorMap)

The diagonal of a 2-leg tensor as a vector (e.g. sampling probabilities from a
density-matrix tensor).
"""
LinearAlgebra.diag(t::ITensorMap) = LinearAlgebra.diag(array(t))

"""
    tr(t::ITensorMap)

The trace of a 2-leg (square) tensor.
"""
LinearAlgebra.tr(t::ITensorMap) = LinearAlgebra.tr(array(t))
