# Operator and state library as plain matrices, keyed by number of sites, site dimension (2 for
# S=1/2, 3 for S=1) and name. Conventions follow ITensors: |↑⟩ = |0⟩ is the first basis state and
# the first site is the most significant (A on site 1, B on site 2 is `kron(A, B)`). Unknown names
# fall through to the backend's library (`backend_op`).

const _σ0 = [1.0 0.0; 0.0 1.0]
const _σx = [0.0 1.0; 1.0 0.0]
const _σy = [0.0 -1.0im; 1.0im 0.0]
const _σz = [1.0 0.0; 0.0 -1.0]

# Controlled single-qubit `u`, control on the first site.
_controlled(u::AbstractMatrix) = [1 0 0 0; 0 1 0 0; 0 0 u[1, 1] u[1, 2]; 0 0 u[2, 1] u[2, 2]]

_rx(θ) = [cos(θ / 2) -im * sin(θ / 2); -im * sin(θ / 2) cos(θ / 2)]
_ry(θ) = [cos(θ / 2) -sin(θ / 2); sin(θ / 2) cos(θ / 2)]
_rz(θ) = [exp(-im * θ / 2) 0; 0 exp(im * θ / 2)]

# Rz and CRz accept the deprecated ϕ in place of θ, as ITensors does.
function _rz_angle(θ, ϕ)
    isone(count(isnothing, (θ, ϕ))) ||
        throw(ArgumentError("Rz: specify the keyword argument `θ` (or the deprecated `ϕ`), not both."))
    return isnothing(θ) ? ϕ : θ
end

# (nsites, site dimension) => name => (; kwargs...) -> matrix
const OPERATORS = Dict{Tuple{Int, Int}, Dict{String, Function}}(
    (1, 2) => Dict{String, Function}(
        "I" => (; kw...) -> _σ0,
        "Id" => (; kw...) -> _σ0,
        "X" => (; kw...) -> _σx,
        "Y" => (; kw...) -> _σy,
        "Z" => (; kw...) -> _σz,
        "iY" => (; kw...) -> [0.0 1.0; -1.0 0.0],
        "H" => (; kw...) -> [1 1; 1 -1] / sqrt(2),
        "√NOT" => (; kw...) -> [(1 + im)/2 (1 - im)/2; (1 - im)/2 (1 + im)/2],
        "Phase" => (; ϕ::Number = π / 2, kw...) -> [1 0; 0 exp(im * ϕ)],
        "π/8" => (; kw...) -> [1 0; 0 1 / sqrt(2) + im / sqrt(2)],
        "Rx" => (; θ::Number, kw...) -> _rx(θ),
        "Ry" => (; θ::Number, kw...) -> _ry(θ),
        "Rz" => (; θ = nothing, ϕ = nothing, kw...) -> _rz(_rz_angle(θ, ϕ)),
        # diag(1, e^{-iθ/2}): a Z rotation acting on |1⟩ only, at a halved angle
        "Rz+" => (; θ::Number, kw...) -> [1 0; 0 exp(-im * θ / 2)],
        "Sx" => (; kw...) -> _σx / 2,
        "Sy" => (; kw...) -> _σy / 2,
        "iSy" => (; kw...) -> [0.0 0.5; -0.5 0.0],
        "Sz" => (; kw...) -> _σz / 2,
        "S+" => (; kw...) -> [0.0 1.0; 0.0 0.0],
        "S-" => (; kw...) -> [0.0 0.0; 1.0 0.0],
        "S2" => (; kw...) -> 0.75 * _σ0,
        "Proj0" => (; kw...) -> [1.0 0.0; 0.0 0.0],
        "Proj1" => (; kw...) -> [0.0 0.0; 0.0 1.0],
    ),
    (2, 2) => Dict{String, Function}(
        "CNOT" => (; kw...) -> _controlled(_σx),
        "CY" => (; kw...) -> _controlled(_σy),
        "CZ" => (; kw...) -> _controlled(_σz),
        "CRx" => (; θ::Number, kw...) -> _controlled(_rx(θ)),
        "CRy" => (; θ::Number, kw...) -> _controlled(_ry(θ)),
        "CRz" => (; θ = nothing, ϕ = nothing, kw...) -> _controlled(_rz(_rz_angle(θ, ϕ))),
        "CPHASE" => (; ϕ::Number, kw...) -> _controlled([1 0; 0 exp(im * ϕ)]),
        "SWAP" => (; kw...) -> [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1],
        "√SWAP" => (; kw...) -> [1 0 0 0; 0 (1 + im)/2 (1 - im)/2 0; 0 (1 - im)/2 (1 + im)/2 0; 0 0 0 1],
        "iSWAP" => (; kw...) -> [1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1],
        "√iSWAP" => (; kw...) -> [1 0 0 0; 0 1/√2 im/√2 0; 0 im/√2 1/√2 0; 0 0 0 1],
        # Ising couplings exp(-iϕ σσ)
        "Rxx" => (; ϕ::Number, kw...) -> cos(ϕ) * kron(_σ0, _σ0) - im * sin(ϕ) * kron(_σx, _σx),
        "Ryy" => (; ϕ::Number, kw...) -> cos(ϕ) * kron(_σ0, _σ0) - im * sin(ϕ) * kron(_σy, _σy),
        "Rzz" => (; ϕ::Number, kw...) -> cos(ϕ) * kron(_σ0, _σ0) - im * sin(ϕ) * kron(_σz, _σz),
        "Rxy" => (; ϕ::Number, kw...) -> [1 0 0 0; 0 cos(ϕ) -im*sin(ϕ) 0; 0 -im*sin(ϕ) cos(ϕ) 0; 0 0 0 1],
        # diag(1, 1, 1, e^{-iθ/2}): a ZZ-type rotation acting on |11⟩ only, at a halved angle
        "Rz+z+" => (; θ::Number, kw...) -> [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(-im * θ / 2)],
        # exp(-iθ (XX + YY) / 2)
        "Rxxyy" => (; θ::Number = 1, kw...) -> exp(-im * θ * 0.5 * (kron(_σx, _σx) + kron(_σy, _σy))),
        # exp(-iθ (XX + YY + ZZ) / 2)
        "Rxxyyzz" => (; θ::Number = 1, kw...) ->
            exp(-im * θ * 0.5 * (kron(_σx, _σx) + kron(_σy, _σy) + kron(_σz, _σz))),
        # Rotation by XX+YY at angle θ with Rz(β) rotations either side; consistent with qiskit.
        "xx_plus_yy" => (; θ::Number, β::Number, kw...) -> [
            1 0 0 0
            0 cos(θ / 2) -im*sin(θ / 2)*exp(-im * β) 0
            0 -im*sin(θ / 2)*exp(im * β) cos(θ / 2) 0
            0 0 0 1
        ],
    ),
    (1, 3) => Dict{String, Function}(
        "I" => (; kw...) -> [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
        "Id" => (; kw...) -> [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
        "Sz" => (; kw...) -> [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -1.0],
        "S+" => (; kw...) -> [0.0 √2 0.0; 0.0 0.0 √2; 0.0 0.0 0.0],
        "S-" => (; kw...) -> [0.0 0.0 0.0; √2 0.0 0.0; 0.0 √2 0.0],
        "Sx" => (; kw...) -> [0.0 1/√2 0.0; 1/√2 0.0 1/√2; 0.0 1/√2 0.0],
        "Sy" => (; kw...) -> [0.0 -im/√2 0.0; im/√2 0.0 -im/√2; 0.0 im/√2 0.0],
        "iSy" => (; kw...) -> [0.0 1/√2 0.0; -1/√2 0.0 1/√2; 0.0 -1/√2 0.0],
        "Sz2" => (; kw...) -> [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0],
        "Sx2" => (; kw...) -> [0.5 0.0 0.5; 0.0 1.0 0.0; 0.5 0.0 0.5],
        "Sy2" => (; kw...) -> [0.5 0.0 -0.5; 0.0 1.0 0.0; -0.5 0.0 0.5],
        "S2" => (; kw...) -> [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0],
    ),
)

# Alternative names, per (nsites, site dimension) table.
const _OPERATOR_ALIASES = Dict(
    (1, 2) => [
        "σx" => "X", "σ1" => "X", "σy" => "Y", "σ2" => "Y", "σz" => "Z", "σ3" => "Z",
        "iσy" => "iY", "iσ2" => "iY", "√X" => "√NOT", "P" => "Phase", "T" => "π/8",
        "Sˣ" => "Sx", "Sʸ" => "Sy", "iSʸ" => "iSy", "Sᶻ" => "Sz", "S⁺" => "S+", "Splus" => "S+",
        "S⁻" => "S-", "Sminus" => "S-", "S²" => "S2",
        "ProjUp" => "Proj0", "projUp" => "Proj0", "ProjDn" => "Proj1", "projDn" => "Proj1",
    ],
    (2, 2) => [
        "CX" => "CNOT", "CRX" => "CRx", "CRY" => "CRy", "CRZ" => "CRz", "Cphase" => "CPHASE",
        "Swap" => "SWAP", "√Swap" => "√SWAP", "iSwap" => "iSWAP", "√iSwap" => "√iSWAP",
        "RXX" => "Rxx", "RYY" => "Ryy", "RZZ" => "Rzz", "RXY" => "Rxy",
    ],
    (1, 3) => [
        "Sˣ" => "Sx", "Sʸ" => "Sy", "iSʸ" => "iSy", "Sᶻ" => "Sz", "S⁺" => "S+", "Splus" => "S+",
        "Sp" => "S+", "S⁻" => "S-", "Sminus" => "S-", "Sm" => "S-", "S²" => "S2",
    ],
)
for (key, pairs) in _OPERATOR_ALIASES, (alias, canon) in pairs
    OPERATORS[key][alias] = OPERATORS[key][canon]
end
# "S" is the phase gate at ϕ = π/2
OPERATORS[(1, 2)]["S"] = (; kw...) -> OPERATORS[(1, 2)]["Phase"](; ϕ = π / 2)

"""
    register_op!(name::String, f; nsites = 1, sitedim = 2)

Register the operator `name` acting on `nsites` sites of dimension `sitedim`, so that
`op(name, sites...; kwargs...)` builds it. Registrations last for the current Julia session.

# Arguments
- `name::String`: The operator name. A gate registered with [`register_gate!`](@ref) under the
  same name can then be used in circuits.
- `f`: A `sitedim^nsites` square matrix, or a function `f(; kwargs...)` returning one. The first
  site is the most significant (`kron(A, B)` is A on the first site, B on the second).

# Keyword Arguments
- `nsites::Integer`: Number of sites the operator acts on (default 1).
- `sitedim::Integer`: Dimension of each site (default 2).

# Returns
- `name`.
"""
function register_op!(name::String, f; nsites::Integer = 1, sitedim::Integer = 2)
    g = f isa AbstractMatrix ? (; kw...) -> f : f
    get!(() -> Dict{String, Function}(), OPERATORS, (Int(nsites), Int(sitedim)))[name] = g
    return name
end

# The registered matrix of `name` on `nsites` sites of dimension `d`, or nothing.
function operator_matrix(name::AbstractString, nsites::Integer, d::Integer; kwargs...)
    f = get(get(OPERATORS, (Int(nsites), Int(d)), Dict{String, Function}()), name, nothing)
    return isnothing(f) ? nothing : f(; kwargs...)
end

# A matrix on (sites′, sites) as a tensor. Legs are reversed so that, read column-major, the first
# site is the most significant. Integer matrices are promoted to floating point.
function operator_tensor(m::AbstractMatrix, sites::Vector)
    return from_array(float.(m), reverse(prime.(sites)), reverse(sites))
end

"""
    op(name::String, sites...; kwargs...)

Build the operator `name` on `sites` from the package's registry, falling back to the backend's
own library.

# Arguments
- `name::String`: The operator name, e.g. `"X"`, `"CNOT"`, `"Rx"`.
- `sites...`: The site indices it acts on.

# Keyword Arguments
- Parameters of the operator, e.g. `θ` for `"Rx"`.

# Returns
- A tensor with legs (sites′..., sites...).
"""
function op(name::AbstractString, sites...; kwargs...)
    d = dim(first(sites))
    m = all(s -> dim(s) == d, sites) ? operator_matrix(name, length(sites), d; kwargs...) : nothing
    isnothing(m) || return operator_tensor(m, collect(sites))
    t = backend_op(String(name), sites...; kwargs...)
    isnothing(t) || return t
    throw(ArgumentError(
        "op: no operator \"$name\" on $(length(sites)) site(s) of dimension $d. " *
            "Register one with `register_op!`."
    ))
end

# (site dimension) => state name => vector
const STATES = Dict{Int, Dict{String, Vector}}(
    2 => Dict{String, Vector}(
        "0" => [1.0, 0.0], "1" => [0.0, 1.0],
        "+" => [1.0, 1.0] / √2, "-" => [1.0, -1.0] / √2,
        "i" => [1.0, im] / √2, "-i" => [1.0, -im] / √2,
    ),
    3 => Dict{String, Vector}(
        "Up" => [1.0, 0.0, 0.0], "Z0" => [0.0, 1.0, 0.0], "Dn" => [0.0, 0.0, 1.0],
        "X+" => [1 / 2, 1 / √2, 1 / 2], "X0" => [-1 / √2, 0, 1 / √2], "X-" => [1 / 2, -1 / √2, 1 / 2],
        "Y+" => [-1 / 2, -im / √2, 1 / 2], "Y0" => [1 / √2, 0, 1 / √2], "Y-" => [-1 / 2, im / √2, 1 / 2],
    ),
)
for (alias, canon) in ["Up" => "0", "Dn" => "1", "↑" => "0", "↓" => "1", "X+" => "+", "Xp" => "+",
        "X-" => "-", "Xm" => "-", "Y+" => "i", "Yp" => "i", "Y-" => "-i", "Ym" => "-i",
        "Z+" => "0", "Zp" => "0", "Z-" => "1", "Zm" => "1"]
    STATES[2][alias] = STATES[2][canon]
end
for (alias, canon) in ["↑" => "Up", "0" => "Z0", "↓" => "Dn", "Z+" => "Up", "Z-" => "Dn"]
    STATES[3][alias] = STATES[3][canon]
end

"""
    state(name::String, site)

Build the single-site state `name` (e.g. `"0"`, `"+"`, `"Up"`) on the index `site`.

# Returns
- A one-leg tensor on `site`.
"""
function state(name::AbstractString, site)
    v = get(get(STATES, dim(site), Dict{String, Vector}()), name, nothing)
    isnothing(v) && throw(ArgumentError("state: no state \"$name\" on a site of dimension $(dim(site))."))
    return from_array(v, [site])
end
