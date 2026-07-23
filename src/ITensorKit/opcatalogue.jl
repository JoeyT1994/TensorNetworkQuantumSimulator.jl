# Operator / state catalogue bridged through ITensors.
#
# We keep ITensors purely as a *matrix-data source*: given an operator/state name
# and ITensorKit site `Index`es, build temporary ITensors indices carrying the
# matching SiteType (chosen from the leg dimension), call `ITensors.op`/`state`,
# extract the dense array in a chosen leg order, and wrap it as an `ITensorMap`.
# Custom `ITensors.op(::OpName…)` methods defined downstream resolve automatically
# through the `ITensors.op` call.
#
# Leg layout of the returned operator: outputs first (primed, plev 1) then inputs
# (unprimed, plev 0), in the site order the caller passed:
#   op on (s1,…,sn)  ->  legs (s1', …, sn', s1, …, sn)
# so contracting it with a ket sharing `sk` leaves `sk'` (then `noprime`).

export op, state, sitetype_from_dim, apply

"""
    apply(A::ITensorMap, B::ITensorMap) -> ITensorMap

Apply operator `A` to `B`: contract over their shared legs and reset prime levels
(the ITensors `apply`/`product` semantics). For an operator with legs `(s', s)` and
a tensor carrying `s`, this contracts `s` and returns a tensor carrying `s`.
"""
apply(A::ITensorMap, B::ITensorMap) = noprime(contract(A, B))

"""
    sitetype_from_dim(d::Integer) -> String

The ITensors `SiteType` string for a physical leg of dimension `d`
(`2 => "S=1/2"`, `3 => "S=1"`). Errors otherwise (e.g. the dim-4 Pauli/PTM space,
which never goes through [`op`](@ref)).
"""
function sitetype_from_dim(d::Integer)
    d == 2 && return "S=1/2"
    d == 3 && return "S=1"
    throw(ArgumentError("no operator SiteType for physical dimension $d (only 2 => S=1/2, 3 => S=1)"))
end

# Temporary ITensors index carrying the SiteType tag matched to `i`'s dimension.
_itensors_index(i::Index) = ITensors.Index(dim(i), sitetype_from_dim(dim(i)))

"""
    op(name::AbstractString, is::Index...; kwargs...) -> ITensorMap

The named operator on the site legs `is`, as an `ITensorMap` with legs
`(prime.(is)..., is...)` (outputs then inputs). The matrix is sourced from
`ITensors.op`; `kwargs` (gate parameters) are forwarded.
"""
function op(name::AbstractString, is::Index...; kwargs...)
    ti = map(_itensors_index, is)
    T = ITensors.op(name, ti...; kwargs...)
    # array indexed [out1,…,outn, in1,…,inn] in the caller's site order
    A = ITensors.array(T, ITensors.prime.(ti)..., ti...)
    legs = (map(prime, is)..., is...)
    return itensor(A, legs)
end

"""
    state(name::AbstractString, i::Index; kwargs...) -> ITensorMap

The named single-site state on `i`, as a 1-leg ket `ITensorMap`. Sourced from
`ITensors.state`.
"""
function state(name::AbstractString, i::Index; kwargs...)
    ti = _itensors_index(i)
    T = ITensors.state(name, ti; kwargs...)
    v = ITensors.array(T, ti)
    return itensor(v, (i,))
end
