# Operator / named-state system, scoped to the `S=1/2` single- and two-qubit gate set and the
# computational/Pauli-basis product states, using the `OpName` / `SiteType` dispatch that gate
# definitions (`Apply/gate_definitions.jl`) and user-registered gates extend.
#
# `state` is called qualified (`Ops.state`) at call sites because ITensorBase also exports an
# unrelated `state`, and the two do not extend each other.
module Ops

using ITensorBase: ITensorBase, Index, id
using TensorAlgebra: project
using ..TensorNetworkQuantumSimulator: project_aux

export state, op, OpName, SiteType, @OpName_str, @SiteType_str

#
# Named single-site states. `state(name, i)` returns the named state vector as an
# `ITensor` over `i`.
#
const _STATE_VECTORS = Dict{String, Vector{ComplexF64}}(
    "Up" => [1, 0], "↑" => [1, 0], "Z+" => [1, 0], "0" => [1, 0],
    "Dn" => [0, 1], "Down" => [0, 1], "↓" => [0, 1], "Z-" => [0, 1], "1" => [0, 1],
    "X+" => [1, 1] / sqrt(2), "+" => [1, 1] / sqrt(2),
    "X-" => [1, -1] / sqrt(2), "-" => [1, -1] / sqrt(2),
    "Y+" => [1, im] / sqrt(2),
    "Y-" => [1, -im] / sqrt(2)
)
function state(name::AbstractString, i::Index)
    v = get(_STATE_VECTORS, name) do
        return error(
            "unknown single-site state \"$name\" (states: $(sort(collect(keys(_STATE_VECTORS)))))"
        )
    end
    return state(v, i)
end
# Vector form (legacy `ITensor(v, i)` for a state vector): the state vector as an `ITensor`
# over `i`, through the `project_aux` utility.
state(v::AbstractVector{<:Number}, i::Index) = project_aux(v, i)

#
# Operators. Legacy ITensors exposes operators through the `OpName` / `SiteType`
# dispatch system, where each gate defines a method `op(::OpName"name",
# ::SiteType"...")` returning its matrix (or an index-form method returning an
# `ITensor` directly). We reproduce that machinery here so gate definitions and
# user-registered gates (`Apply/gate_definitions.jl`) work unchanged: a gate either
# defines a matrix method or an index method, and the generic fallback embeds a
# matrix into an `ITensor` over `(sites'..., sites...)` (primed legs are outputs).
#
struct OpName{Name} end
OpName(name::AbstractString) = OpName{Symbol(name)}()
macro OpName_str(name)
    return :(OpName{$(QuoteNode(Symbol(name)))})
end

struct SiteType{Name} end
SiteType(name::AbstractString) = SiteType{Symbol(name)}()
macro SiteType_str(name)
    return :(SiteType{$(QuoteNode(Symbol(name)))})
end

# Embed a `d^n × d^n` operator matrix (computational basis, first site most
# significant) into an `ITensor` with codomain `prime.(sites)` (outputs) and
# domain `sites` (inputs). Routed through `project_aux` (like `state`): a graded
# operator that is odd under the site grading (e.g. `X`/`Y`, bare `c`/`c†`) has no
# parity-block-diagonal `(prime.(sites), sites)` map, so it gets a trailing auxiliary
# charge leg instead of throwing an `InexactError`. Even operators and dense (ungraded)
# sites project cleanly, so they are unaffected (no aux leg).
function _op_matrix_to_itensor(M::AbstractMatrix, sites::Tuple)
    ds = length.(sites)
    n = length(sites)
    A = reshape(Matrix{ComplexF64}(M), (reverse(ds)..., reverse(ds)...))
    A = permutedims(A, (reverse(1:n)..., reverse((n + 1):(2n))...))
    return project_aux(A, ITensorBase.prime.(sites), sites)
end

# Top-level `op(name, sites...; kwargs...)`. The identity is dimension-general (used
# by `rdm` / `norm_factors` / `truncate` on arbitrary site dimensions, e.g. S=1);
# everything else dispatches through the `OpName` / `SiteType` system on qubit sites.
function op(name::AbstractString, sites::Index...; kwargs...)
    if name in ("I", "Id") && length(sites) == 1
        s = only(sites)
        return id(ComplexF64, (ITensorBase.prime(s),), (s,))
    end
    return op(OpName(name), SiteType("S=1/2"), sites...; kwargs...)
end

# Generic index-form fallback: build the operator's matrix, then embed it. Gates that
# need an index-form definition (e.g. ones built from other `op`s) override this.
function op(on::OpName, st::SiteType, s1::Index, srest::Index...; kwargs...)
    return _op_matrix_to_itensor(ComplexF64.(op(on, st; kwargs...)), (s1, srest...))
end

# Single-qubit matrices (fixed + parametric), in the qiskit/ITensors convention.
const _GATE_MATRICES = Dict{String, Matrix{ComplexF64}}(
    "I" => [1 0; 0 1], "Id" => [1 0; 0 1],
    "X" => [0 1; 1 0],
    "Y" => [0 -im; im 0],
    "Z" => [1 0; 0 -1],
    "H" => [1 1; 1 -1] / sqrt(2),
    "S" => [1 0; 0 im],
    "T" => [1 0; 0 cis(π / 4)]
)
function _gate_matrix(name::AbstractString; kwargs...)
    name in ("Rx", "Ry", "Rz", "P") || return get(_GATE_MATRICES, name) do
        return error(
            "unknown single-site operator \"$name\" (operators: $(sort(collect(keys(_GATE_MATRICES)))) plus Rx/Ry/Rz/P)"
        )
    end
    if name == "Rx"
        θ = kwargs[:θ]
        return ComplexF64[cos(θ / 2) (-im * sin(θ / 2)); (-im * sin(θ / 2)) cos(θ / 2)]
    elseif name == "Ry"
        θ = kwargs[:θ]
        return ComplexF64[cos(θ / 2) (-sin(θ / 2)); sin(θ / 2) cos(θ / 2)]
    elseif name == "Rz"
        θ = kwargs[:θ]
        return ComplexF64[cis(-θ / 2) 0; 0 cis(θ / 2)]
    else # "P"
        ϕ = kwargs[:ϕ]
        return ComplexF64[1 0; 0 cis(ϕ)]
    end
end
# Matrix method for any single-qubit name handled by `_gate_matrix`.
function op(::OpName{Name}, ::SiteType"S=1/2"; kwargs...) where {Name}
    return _gate_matrix(String(Name); kwargs...)
end

# Two-qubit gate matrices (computational basis, first site most significant).
const _σx = ComplexF64[0 1; 1 0]
const _σy = ComplexF64[0 -im; im 0]
const _σz = ComplexF64[1 0; 0 -1]
op(::OpName"Rxx", ::SiteType"S=1/2"; ϕ) = exp(-im * ϕ * kron(_σx, _σx))
op(::OpName"Ryy", ::SiteType"S=1/2"; ϕ) = exp(-im * ϕ * kron(_σy, _σy))
op(::OpName"Rzz", ::SiteType"S=1/2"; ϕ) = exp(-im * ϕ * kron(_σz, _σz))
function op(::OpName"CPHASE", ::SiteType"S=1/2"; ϕ)
    return ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 cis(ϕ)]
end
function op(::OpName"SWAP", ::SiteType"S=1/2")
    return Float64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
end

end
