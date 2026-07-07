# Vendored minimal operator / named-state system (legacy `ITensors.op` / `ITensors.state`).
#
# Round-1 scope is the single-qubit (`S=1/2`) gate set and the computational/Pauli-basis
# product states — enough for `siteinds` / `tensornetworkstate` state construction and
# single-site `expect`. The full two-qubit gate set, the parametric and in-house gates,
# and the Heisenberg/PTM (PauliPropagation) path belong to the gate-application round
# (`Apply/`), which is still excluded; they are not vendored here.
#
# Legacy ITensors exposes these through the `OpName` / `SiteType` dispatch system. TNQS
# only ever calls `op(name, sites...; kwargs...)` and `state(name, site)`, so the vendor
# is a plain name-keyed lookup rather than a reimplementation of that dispatch machinery.

using TensorAlgebra: TensorAlgebra, project, tryproject

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
            "unknown single-site state \"$name\" (vendored states: $(sort(collect(keys(_STATE_VECTORS)))))"
        )
    end
    return state(v, i)
end
# Project a raw vector as a state `ITensor` over `i`, adding an auxiliary leg only when the
# vector cannot live in the flux-zero space over `i` alone. The index axis selects the
# backend (dense, graded, `TensorMap`). Shared by the state constructors and `onehot`.
function project_aux(v::AbstractVector{<:Number}, i::Index)
    length(v) == length(i) ||
        error(
        "state vector has dimension $(length(v)) but the site index has dimension $(length(i))"
    )
    ψ = tryproject(v, (i,))
    isnothing(ψ) || return ψ
    # The vector carries a charge under `i`'s grading (e.g. "Dn" on a U(1) site, or a
    # one-hot on a graded link), so it can't live in the flux-zero space over `i` alone.
    # Carry the charge on an explicit length-1 auxiliary leg: project with a trailing axis
    # so the backend derives the leg's sector from the vector, then wrap the derived axis
    # in a freshly named `Index`.
    raw = project(reshape(v, (length(v), 1)), (unnamed(i),), ())
    aux = Index(TensorAlgebra.axes(raw, 2))
    return nameddims(raw, (ITensorBase.name(i), ITensorBase.name(aux)))
end
# Vector form (legacy `ITensor(v, i)` for a state vector): the state vector as an `ITensor`
# over `i`.
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
# domain `sites` (inputs). `project` is checked, so on graded sites an operator
# that is not symmetric under the site index's grading throws an `InexactError`.
function _op_matrix_to_itensor(M::AbstractMatrix, sites::Tuple)
    ds = length.(sites)
    n = length(sites)
    A = reshape(Matrix{ComplexF64}(M), (reverse(ds)..., reverse(ds)...))
    A = permutedims(A, (reverse(1:n)..., reverse((n + 1):(2n))...))
    return project(A, ITensorBase.prime.(sites), sites)
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
            "unknown single-site operator \"$name\" (vendored operators: $(sort(collect(keys(_GATE_MATRICES)))) plus Rx/Ry/Rz/P)"
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

# TYPE PIRACY (temporary): extends `Base.exp` for an operator `ITensor`, inferring the
# prime-pair codomain/domain and forwarding to ITensorBase's matricization
# `exp(a, dimnames_codomain, dimnames_domain)` (graded-capable). Legacy ITensors provided
# `exp(::ITensor)`; gates defined as `exp` of a Hamiltonian operator rely on it (e.g. a user
# `op(::OpName"MyZRot", ...) = exp(-im θ/2 * op("Z", s))`). To de-pirate: make this compat-owned
# (an `exp` in this module, not a `Base.exp` method). Not an upstream candidate (the upstream
# matricization `exp` is the target, not a `Base.exp(::ITensor)`).
function Base.exp(t::AbstractITensor)
    p0 = filter(i -> ITensorBase.plev(i) == 0, collect(inds(t)))
    isempty(p0) && error("exp(::ITensor) expects indices paired as (i, prime(i))")
    p1 = map(ITensorBase.prime, p0)
    return exp(t, Tuple(p1), Tuple(p0))
end
