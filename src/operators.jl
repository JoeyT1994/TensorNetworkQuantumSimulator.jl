# Operator and state library as plain matrices, keyed by number of sites, site dimension (2 for
# S=1/2, 3 for S=1) and name. |↑⟩ = |0⟩ is the first basis state and the first site is the most
# significant (A on site 1, B on site 2 is `kron(A, B)`). Parameters follow qiskit.

const σ0 = [1.0 0.0; 0.0 1.0]
const σx = [0.0 1.0; 1.0 0.0]
const σy = [0.0 -1.0im; 1.0im 0.0]
const σz = [1.0 0.0; 0.0 -1.0]
const HADAMARD = [1 1; 1 -1] / sqrt(2)
const SX_MATRIX = [(1 + im)/2 (1 - im)/2; (1 - im)/2 (1 + im)/2]
const SWAP_MATRIX = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
const S_MATRIX = [1 0; 0 im]

# diag(1, e^{iθ})
phase_matrix(θ) = [1 0; 0 cis(θ)]

# `u` controlled on all preceding sites, i.e. applied when they are all |1⟩.
function controlled(u::AbstractMatrix, ncontrols::Integer = 1)
    n = 2^ncontrols * size(u, 1)
    m = Matrix{eltype(u)}(LinearAlgebra.I, n, n)
    m[(end - size(u, 1) + 1):end, (end - size(u, 1) + 1):end] = u
    return m
end

const CNOT_MATRIX = controlled(σx)

rn_matrix(θ, ϕ, λ) = [cos(θ / 2) -exp(im * λ) * sin(θ / 2); exp(im * ϕ) * sin(θ / 2) exp(im * (ϕ + λ)) * cos(θ / 2)]

rx_matrix(θ) = [cos(θ / 2) -im * sin(θ / 2); -im * sin(θ / 2) cos(θ / 2)]
ry_matrix(θ) = [cos(θ / 2) -sin(θ / 2); sin(θ / 2) cos(θ / 2)]
rz_matrix(θ) = [exp(-im * θ / 2) 0; 0 exp(im * θ / 2)]

# exp(-i θ/2 P⊗Q)
pauli_pair_rotation(θ, P, Q = P) = cos(θ / 2) * kron(σ0, σ0) - im * sin(θ / 2) * kron(P, Q)

# (nsites, site dimension) => name => (; kwargs...) -> matrix
const OPERATORS = Dict{Tuple{Int, Int}, Dict{String, Function}}(
    (1, 2) => Dict{String, Function}(
        "I" => (; kw...) -> σ0,
        "Id" => (; kw...) -> σ0,
        "X" => (; kw...) -> σx,
        "Y" => (; kw...) -> σy,
        "Z" => (; kw...) -> σz,
        "iY" => (; kw...) -> [0.0 1.0; -1.0 0.0],
        "H" => (; kw...) -> HADAMARD,
        "√NOT" => (; kw...) -> SX_MATRIX,
        "SXdg" => (; kw...) -> SX_MATRIX',
        "S" => (; kw...) -> S_MATRIX,
        "Sdg" => (; kw...) -> S_MATRIX',
        "Phase" => (; θ::Number, kw...) -> phase_matrix(θ),
        "π/8" => (; kw...) -> [1 0; 0 1 / sqrt(2) + im / sqrt(2)],
        "Tdg" => (; kw...) -> [1 0; 0 1 / sqrt(2) - im / sqrt(2)],
        "Rx" => (; θ::Number, kw...) -> rx_matrix(θ),
        "Ry" => (; θ::Number, kw...) -> ry_matrix(θ),
        "Rz" => (; θ::Number, kw...) -> rz_matrix(θ),
        # diag(1, e^{-iθ/2}): a Z rotation acting on |1⟩ only, at a halved angle
        "Rz+" => (; θ::Number, kw...) -> phase_matrix(-θ / 2),
        # general rotation, U(θ, ϕ, λ) in qiskit's notation
        "Rn" => (; θ::Number, ϕ::Number, λ::Number, kw...) -> rn_matrix(θ, ϕ, λ),
        "U2" => (; ϕ::Number, λ::Number, kw...) -> rn_matrix(π / 2, ϕ, λ),
        # rotation by θ about cos(ϕ) X + sin(ϕ) Y
        "R" => (; θ::Number, ϕ::Number, kw...) -> [cos(θ / 2) -im*exp(-im * ϕ)*sin(θ / 2); -im*exp(im * ϕ)*sin(θ / 2) cos(θ / 2)],
        "Sx" => (; kw...) -> σx / 2,
        "Sy" => (; kw...) -> σy / 2,
        "iSy" => (; kw...) -> [0.0 0.5; -0.5 0.0],
        "Sz" => (; kw...) -> σz / 2,
        "S+" => (; kw...) -> [0.0 1.0; 0.0 0.0],
        "S-" => (; kw...) -> [0.0 0.0; 1.0 0.0],
        "S2" => (; kw...) -> 0.75 * σ0,
        "Proj0" => (; kw...) -> [1.0 0.0; 0.0 0.0],
        "Proj1" => (; kw...) -> [0.0 0.0; 0.0 1.0],
    ),
    (2, 2) => Dict{String, Function}(
        "CNOT" => (; kw...) -> CNOT_MATRIX,
        "CH" => (; kw...) -> controlled(HADAMARD),
        "CS" => (; kw...) -> controlled(S_MATRIX),
        "CSdg" => (; kw...) -> controlled(S_MATRIX'),
        "CSX" => (; kw...) -> controlled(SX_MATRIX),
        # e^{iγ} U(θ, ϕ, λ), controlled
        "CU" => (; θ::Number, ϕ::Number, λ::Number, γ::Number, kw...) -> controlled(exp(im * γ) * rn_matrix(θ, ϕ, λ)),
        # CNOT on (1, 2) then on (2, 1)
        "DCX" => (; kw...) -> SWAP_MATRIX * CNOT_MATRIX * SWAP_MATRIX * CNOT_MATRIX,
        # echoed cross-resonance, (X⊗1 - Y⊗X)/√2
        "ECR" => (; kw...) -> (kron(σx, σ0) - kron(σy, σx)) / sqrt(2),
        "CY" => (; kw...) -> controlled(σy),
        "CZ" => (; kw...) -> controlled(σz),
        "CRx" => (; θ::Number, kw...) -> controlled(rx_matrix(θ)),
        "CRy" => (; θ::Number, kw...) -> controlled(ry_matrix(θ)),
        "CRz" => (; θ::Number, kw...) -> controlled(rz_matrix(θ)),
        "CRn" => (; θ::Number, ϕ::Number, λ::Number, kw...) -> controlled(rn_matrix(θ, ϕ, λ)),
        "CPHASE" => (; θ::Number, kw...) -> controlled(phase_matrix(θ)),
        "SWAP" => (; kw...) -> SWAP_MATRIX,
        "√SWAP" => (; kw...) -> [1 0 0 0; 0 (1 + im)/2 (1 - im)/2 0; 0 (1 - im)/2 (1 + im)/2 0; 0 0 0 1],
        "iSWAP" => (; kw...) -> [1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1],
        "√iSWAP" => (; kw...) -> [1 0 0 0; 0 1/√2 im/√2 0; 0 im/√2 1/√2 0; 0 0 0 1],
        "Rxx" => (; θ::Number, kw...) -> pauli_pair_rotation(θ, σx),
        "Ryy" => (; θ::Number, kw...) -> pauli_pair_rotation(θ, σy),
        "Rzz" => (; θ::Number, kw...) -> pauli_pair_rotation(θ, σz),
        "Rzx" => (; θ::Number, kw...) -> pauli_pair_rotation(θ, σz, σx),
        # diag(1, 1, 1, e^{-iθ/2}): a ZZ-type rotation acting on |11⟩ only, at a halved angle
        "Rz+z+" => (; θ::Number, kw...) -> controlled(phase_matrix(-θ / 2)),
        # exp(-iθ (XX + YY) / 2)
        "Rxxyy" => (; θ::Number, kw...) -> exp(-im * θ * 0.5 * (kron(σx, σx) + kron(σy, σy))),
        # exp(-iθ (XX + YY + ZZ) / 2)
        "Rxxyyzz" => (; θ::Number, kw...) ->
            exp(-im * θ * 0.5 * (kron(σx, σx) + kron(σy, σy) + kron(σz, σz))),
        # qiskit's XXPlusYYGate and XXMinusYYGate: Rz(∓β)·exp(-iθ (XX ± YY)/4)·Rz(±β), Rz on the first site
        "xx_plus_yy" => (; θ::Number, β::Number, kw...) -> [
            1 0 0 0
            0 cos(θ / 2) -im*sin(θ / 2)*exp(im * β) 0
            0 -im*sin(θ / 2)*exp(-im * β) cos(θ / 2) 0
            0 0 0 1
        ],
        "xx_minus_yy" => (; θ::Number, β::Number, kw...) -> [
            cos(θ / 2) 0 0 -im*sin(θ / 2)*exp(-im * β)
            0 1 0 0
            0 0 1 0
            -im*sin(θ / 2)*exp(im * β) 0 0 cos(θ / 2)
        ],
    ),
    (3, 2) => Dict{String, Function}(
        "Toffoli" => (; kw...) -> controlled(σx, 2),
        "Fredkin" => (; kw...) -> controlled(SWAP_MATRIX),
    ),
    (4, 2) => Dict{String, Function}(
        "CCCNOT" => (; kw...) -> controlled(σx, 3),
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

# Alternative names, per (nsites, site dimension) table. The spin operators share theirs.
const SPIN_ALIASES = [
    "Sˣ" => "Sx", "Sʸ" => "Sy", "iSʸ" => "iSy", "Sᶻ" => "Sz", "S⁺" => "S+", "Splus" => "S+",
    "Sp" => "S+", "S⁻" => "S-", "Sminus" => "S-", "Sm" => "S-", "S²" => "S2",
]
const OPERATOR_ALIASES = Dict(
    (1, 2) => [
        "σx" => "X", "σ1" => "X", "σy" => "Y", "σ2" => "Y", "σz" => "Z", "σ3" => "Z",
        "iσy" => "iY", "iσ2" => "iY", "√X" => "√NOT", "SX" => "√NOT", "P" => "Phase", "T" => "π/8", "Rn̂" => "Rn",
        "U" => "Rn", "U3" => "Rn", "U1" => "Phase",
        SPIN_ALIASES...,
        "ProjUp" => "Proj0", "projUp" => "Proj0", "ProjDn" => "Proj1", "projDn" => "Proj1",
    ],
    (2, 2) => [
        "CX" => "CNOT", "CRX" => "CRx", "CRY" => "CRy", "CRZ" => "CRz", "Cphase" => "CPHASE",
        "Swap" => "SWAP", "√Swap" => "√SWAP", "iSwap" => "iSWAP", "√iSwap" => "√iSWAP",
        "RXX" => "Rxx", "RYY" => "Ryy", "RZZ" => "Rzz", "RZX" => "Rzx", "Rxy" => "Rxxyy", "RXY" => "Rxxyy",
        "CRn̂" => "CRn", "CU3" => "CRn", "CU1" => "CPHASE",
    ],
    (3, 2) => ["CCNOT" => "Toffoli", "CCX" => "Toffoli", "TOFF" => "Toffoli",
        "CSWAP" => "Fredkin", "CSwap" => "Fredkin"],
    (1, 3) => SPIN_ALIASES,
)
for (key, pairs) in OPERATOR_ALIASES, (alias, canon) in pairs
    OPERATORS[key][alias] = OPERATORS[key][canon]
end

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

# The registered matrix of `name` on `nsites` sites of dimension `d`, or nothing. The identity and
# Haar-random unitaries are defined for any number of sites of any dimension.
function operator_matrix(name::AbstractString, nsites::Integer, d::Integer; kwargs...)
    f = get(get(OPERATORS, (Int(nsites), Int(d)), Dict{String, Function}()), name, nothing)
    isnothing(f) || return f(; kwargs...)
    n = d^nsites
    name in ("Id", "I") && return identity_matrix(n; kwargs...)
    name in ("RandomUnitary", "randU") && return random_unitary_matrix(n; kwargs...)
    return nothing
end

identity_matrix(n; eltype = Float64, kw...) = Matrix{eltype}(LinearAlgebra.I, n, n)

# Haar-random unitary: the Q of a QR decomposition of a Gaussian matrix, with R's diagonal made
# positive.
function random_unitary_matrix(n; eltype = ComplexF64, random_matrix = randn(eltype, n, n), kw...)
    Q, R = LinearAlgebra.qr(random_matrix)
    return Matrix(Q) * LinearAlgebra.Diagonal(sign.(LinearAlgebra.diag(R)))
end

# A matrix on (sites′, sites) as a tensor. Legs are reversed so that, read column-major, the first
# site is the most significant. Integer matrices are promoted to floating point.
function operator_tensor(m::AbstractMatrix, sites::Vector)
    return from_array(float.(m), reverse(prime.(sites)), reverse(sites))
end

"""
    op(name::String, sites...; kwargs...)

Build the operator `name` on `sites` from the package's operator registry.

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
