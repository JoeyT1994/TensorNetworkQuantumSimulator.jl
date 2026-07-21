# VectorInterface for `ITensorMap`.
# =================================
# TensorKit already implements the full VectorInterface for `AbstractTensorMap`, so these
# methods are mostly delegation to `t.data`, re-wrapping the result with the same `inds`.
#
# The one subtlety: VectorInterface's binary ops on `TensorMap`s are *positional* (they
# require `space(ty) == space(tx)`), whereas `ITensorMap` legs are matched by *identity*
# (the `(id, plev)` key) and may sit in different positions / codomain–domain partitions on
# the two operands. We therefore compute the identity permutation and fold the relabel
# *into* the operation via TensorKit's permuting `permute!` (which computes
# `tdst = β·tdst + α·permute(tsrc, p)` in one pass, no temporary tensor).

# --- identity permutation ---------------------------------------------------
# The `(p₁, p₂)::Index2Tuple` (codomain, domain positions into `tx`'s legs) that reorders
# `tx`'s legs to line up with `ty`, matched by `Index` identity. Feeds TensorKit's permuting
# `permute!`/`permute`. A genuine space mismatch surfaces later as a `SpaceMismatch`.
function _permutation(tx::ITensorMap, ty::ITensorMap)
    p = map(ty.inds) do iy
        px = findfirst(==(iy), tx.inds)
        px === nothing && throw(
            ArgumentError(
                "VectorInterface op requires matching index sets; $iy not present in both operands"
            )
        )
        px
    end
    No = numout(ty)
    return (p[1:No], p[(No + 1):end])
end

# --- zerovector -------------------------------------------------------------
function VectorInterface.zerovector(t::ITensorMap, ::Type{S}) where {S <: Number}
    return unsafe_itensormap(zerovector(t.data, S), t.inds)
end
VectorInterface.zerovector!(t::ITensorMap) = (zerovector!(t.data); t)
VectorInterface.zerovector!!(t::ITensorMap) = unsafe_itensormap(zerovector!!(t.data), t.inds)

# --- scale ------------------------------------------------------------------
VectorInterface.scale(t::ITensorMap, α::Number) = unsafe_itensormap(scale(t.data, α), t.inds)
VectorInterface.scale!(t::ITensorMap, α::Number) = (scale!(t.data, α); t)
VectorInterface.scale!!(t::ITensorMap, α::Number) = unsafe_itensormap(scale!!(t.data, α), t.inds)
function VectorInterface.scale!(ty::ITensorMap, tx::ITensorMap, α::Number)
    permute!(ty.data, tx.data, _permutation(tx, ty), α, Zero())
    return ty
end
function VectorInterface.scale!!(ty::ITensorMap, tx::ITensorMap, α::Number)
    T = VectorInterface.promote_scale(tx.data, α)
    return T <: scalartype(ty) ? scale!(ty, tx, α) : scale(tx, α)
end

# --- add --------------------------------------------------------------------
function VectorInterface.add(ty::ITensorMap, tx::ITensorMap, α::Number, β::Number)
    T = VectorInterface.promote_add(ty.data, tx.data, α, β)
    tdst = scale!(zerovector(ty.data, T), ty.data, β)        # β·ty in fresh storage of type T
    permute!(tdst, tx.data, _permutation(tx, ty), α, One())
    return unsafe_itensormap(tdst, ty.inds)
end
function VectorInterface.add!(ty::ITensorMap, tx::ITensorMap, α::Number, β::Number)
    permute!(ty.data, tx.data, _permutation(tx, ty), α, β)
    return ty
end
function VectorInterface.add!!(ty::ITensorMap, tx::ITensorMap, α::Number, β::Number)
    T = VectorInterface.promote_add(ty.data, tx.data, α, β)
    return T <: scalartype(ty) ? add!(ty, tx, α, β) : add(ty, tx, α, β)
end

# --- inner ------------------------------------------------------------------
# No permuting `inner` exists, so align `ty` to `tx` out-of-place. Conjugates `tx`
# (VectorInterface convention); a scalar result.
VectorInterface.inner(tx::ITensorMap, ty::ITensorMap) =
    inner(tx.data, permute(ty.data, _permutation(ty, tx)))

# `LinearAlgebra.dot` (used e.g. in BP message comparison) is the identity-aligned inner product.
LinearAlgebra.dot(tx::ITensorMap, ty::ITensorMap) = VectorInterface.inner(tx, ty)

# --- Base arithmetic (in terms of the above, so identity alignment is inherited) ---
Base.:+(x::ITensorMap, y::ITensorMap) = add(x, y)
Base.:-(x::ITensorMap, y::ITensorMap) = add(x, y, -one(scalartype(y)), one(scalartype(x)))
Base.:-(x::ITensorMap) = scale(x, -one(scalartype(x)))
Base.:/(x::ITensorMap, α::Number) = scale(x, one(scalartype(x)) / α)
