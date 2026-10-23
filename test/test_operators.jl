@eval module $(gensym())
using LinearAlgebra: I, Diagonal, kron, norm
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: GATES, OPERATORS, STATES, toitensor, operator_matrix, state, register_op!
using TensorNetworkQuantumSimulator.TensorInterface: apply, array, inds, prime
using Test: @testset, @test, @test_throws

# The operator library is checked against its defining identities (qiskit conventions, first site
# most significant), not against another library.

const θ, ϕ, λ, β = 0.37, -0.81, 0.59, 0.23
const X, Y, Z, Id2 = [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1], [1 0; 0 1]

m(name, nsites = 1, d = 2; kw...) = operator_matrix(name, nsites, d; kw...)
cexp(A) = exp(Matrix{ComplexF64}(A))
controlled(u, ncontrols = 1) = (n = 2^ncontrols * size(u, 1); M = Matrix{ComplexF64}(I, n, n); M[(end - size(u, 1) + 1):end, (end - size(u, 1) + 1):end] = u; M)
isunitary(U) = U' * U ≈ I

@testset "Qubit operators" begin
    @test m("X") == X && m("Y") == Y && m("Z") == Z && m("I") == Id2 && m("Id") == Id2
    @test m("iY") ≈ im * Y
    @test m("H") ≈ (X + Z) / √2
    @test m("Rx"; θ) ≈ cexp(-im * θ / 2 * X)
    @test m("Ry"; θ) ≈ cexp(-im * θ / 2 * Y)
    @test m("Rz"; θ) ≈ cexp(-im * θ / 2 * Z)
    @test m("Rz"; ϕ = θ) ≈ m("Rz"; θ)
    @test_throws ArgumentError m("Rz"; θ, ϕ)
    @test m("Phase"; θ) ≈ Diagonal([1, exp(im * θ)])
    @test m("P"; θ) ≈ m("Phase"; ϕ = θ)
    @test m("S") ≈ Diagonal([1, im]) && m("T") ≈ Diagonal([1, exp(im * π / 4)])
    @test m("T")^2 ≈ m("S") && m("S")^2 ≈ Z && m("√NOT")^2 ≈ X
    @test m("Sdg") ≈ m("S")' && m("Tdg") ≈ m("T")'
    @test m("SX") == m("√NOT") && m("SXdg") ≈ m("SX")'
    @test m("R"; θ, ϕ) ≈ cexp(-im * θ / 2 * (cos(ϕ) * X + sin(ϕ) * Y))
    # qiskit's U(θ, ϕ, λ)
    @test m("Rn"; θ, ϕ = -π / 2, λ = π / 2) ≈ m("Rx"; θ)
    @test m("Rn"; θ, ϕ = 0, λ = 0) ≈ m("Ry"; θ)
    @test m("Rn̂"; θ, ϕ, λ) == m("Rn"; θ, ϕ, λ)
    @test m("U"; θ, ϕ, λ) == m("U3"; θ, ϕ, λ) == m("Rn"; θ, ϕ, λ)
    @test m("U1"; θ) ≈ m("Phase"; θ) && m("U2"; ϕ, λ) ≈ m("U"; θ = π / 2, ϕ, λ)
    @test m("Rz+"; θ) ≈ Diagonal([1, exp(-im * θ / 2)])
    @test m("Sx") ≈ X / 2 && m("Sy") ≈ Y / 2 && m("Sz") ≈ Z / 2
    @test m("S+") ≈ m("Sx") + im * m("Sy") && m("S-") ≈ m("Sx") - im * m("Sy")
    @test m("iSy") ≈ im * m("Sy")
    @test m("S2") ≈ m("Sx")^2 + m("Sy")^2 + m("Sz")^2
    @test m("Proj0") + m("Proj1") == Id2 && m("Proj0") * m("Proj1") == zeros(2, 2)
    @test m("F") == Id2
end

@testset "Multi-qubit operators" begin
    XX, YY, ZZ = kron(X, X), kron(Y, Y), kron(Z, Z)
    @test m("CNOT", 2) ≈ controlled(X) && m("CY", 2) ≈ controlled(Y) && m("CZ", 2) ≈ controlled(Z)
    @test m("CRx", 2; θ) ≈ controlled(m("Rx"; θ)) && m("CRy", 2; θ) ≈ controlled(m("Ry"; θ))
    @test m("CRz", 2; θ) ≈ controlled(m("Rz"; θ))
    @test m("CRn", 2; θ, ϕ, λ) ≈ controlled(m("Rn"; θ, ϕ, λ))
    @test m("CPHASE", 2; θ) ≈ controlled(m("Phase"; θ)) && m("CU1", 2; θ) ≈ m("CPHASE", 2; θ)
    @test m("CH", 2) ≈ controlled(m("H")) && m("CSX", 2) ≈ controlled(m("SX"))
    @test m("CS", 2) ≈ controlled(m("S")) && m("CSdg", 2) ≈ controlled(m("Sdg"))
    @test m("CU", 2; θ, ϕ, λ, γ = β) ≈ controlled(exp(im * β) * m("U"; θ, ϕ, λ))
    @test m("CU3", 2; θ, ϕ, λ) ≈ controlled(m("U"; θ, ϕ, λ))
    # qiskit: R_PP(θ) = exp(-iθ/2 P⊗P)
    @test m("Rxx", 2; θ) ≈ cexp(-im * θ / 2 * XX)
    @test m("Ryy", 2; θ) ≈ cexp(-im * θ / 2 * YY)
    @test m("Rzz", 2; θ) ≈ cexp(-im * θ / 2 * ZZ)
    @test m("Rzx", 2; θ) ≈ cexp(-im * θ / 2 * kron(Z, X))
    @test m("Rxxyy", 2; θ) ≈ cexp(-im * θ / 2 * (XX + YY)) && m("Rxy", 2; θ) == m("Rxxyy", 2; θ)
    @test m("Rxxyyzz", 2; θ) ≈ cexp(-im * θ / 2 * (XX + YY + ZZ))
    # qiskit's XXPlusYYGate(θ, β) and XXMinusYYGate(θ, β), first site = qiskit's qubit 0
    Rz0, Rz1 = kron(m("Rz"; θ = β), Id2), kron(Id2, m("Rz"; θ = β))
    @test m("xx_plus_yy", 2; θ, β) ≈ Rz0' * cexp(-im * θ / 4 * (XX + YY)) * Rz0
    @test m("xx_minus_yy", 2; θ, β) ≈ Rz1 * cexp(-im * θ / 4 * (XX - YY)) * Rz1'
    @test m("Rz+z+", 2; θ) ≈ Diagonal([1, 1, 1, exp(-im * θ / 2)])
    SWAP = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    # DCX: CX from the first site, then from the second; ECR as qiskit decomposes it
    @test m("DCX", 2) ≈ SWAP * controlled(X) * SWAP * controlled(X)
    @test m("ECR", 2) ≈ m("Rzx", 2; θ = -π / 4) * kron(X, Id2) * m("Rzx", 2; θ = π / 4)
    @test m("SWAP", 2) == SWAP && m("√SWAP", 2)^2 ≈ SWAP
    @test m("iSWAP", 2) ≈ [1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1] && m("√iSWAP", 2)^2 ≈ m("iSWAP", 2)
    @test m("Toffoli", 3) ≈ controlled(X, 2) && m("CCNOT", 3) == m("Toffoli", 3)
    @test m("Fredkin", 3) ≈ controlled(SWAP) && m("CSWAP", 3) == m("Fredkin", 3)
    @test m("CCCNOT", 4) ≈ controlled(X, 3)
    @test m("Id", 3) == Matrix(I, 8, 8)
    @test isunitary(m("RandomUnitary", 2)) && size(m("randU", 3)) == (8, 8)
end

@testset "Spin-1 operators" begin
    Sx, Sy, Sz = m("Sx", 1, 3), m("Sy", 1, 3), m("Sz", 1, 3)
    @test Sx * Sy - Sy * Sx ≈ im * Sz
    @test m("S+", 1, 3) ≈ Sx + im * Sy && m("S-", 1, 3) ≈ Sx - im * Sy
    @test m("S2", 1, 3) ≈ Sx^2 + Sy^2 + Sz^2 ≈ 2I
    @test m("Sx2", 1, 3) ≈ Sx^2 && m("Sy2", 1, 3) ≈ Sy^2 && m("Sz2", 1, 3) ≈ Sz^2
    @test m("iSy", 1, 3) ≈ im * Sy
end

@testset "States" begin
    v(name, d = 2) = STATES[d][name]
    for (d, table) in STATES, (name, s) in table
        @test norm(s) ≈ 1
    end
    @test v("0") == v("↑") == v("Up") && v("1") == v("↓") == v("Dn")
    @test X * v("+") ≈ v("+") && X * v("-") ≈ -v("-")
    @test Y * v("i") ≈ v("i") && Y * v("-i") ≈ -v("-i")
    # the tetrahedral states are a SIC-POVM: pairwise overlaps of 1/3
    tetra = [v("Tetra$k") for k in 1:4]
    @test all(abs2(tetra[j]' * tetra[k]) ≈ 1 / 3 for j in 1:4 for k in 1:4 if j != k)
    Sx, Sy, Sz = m("Sx", 1, 3), m("Sy", 1, 3), m("Sz", 1, 3)
    @test Sz * v("Up", 3) ≈ v("Up", 3) && Sz * v("Z0", 3) ≈ zeros(3) && Sz * v("Dn", 3) ≈ -v("Dn", 3)
    @test Sx * v("X+", 3) ≈ v("X+", 3) && Sy * v("Y+", 3) ≈ v("Y+", 3)
end

@testset "Tensors from op and state" begin
    g = named_path_graph(2)
    s = siteinds("S=1/2", g)
    s1, s2 = only(s[1]), only(s[2])
    # the first site is the most significant: CNOT, controlled on s1, flips s2
    ψ = apply(op("CNOT", s1, s2), state("1", s1) * state("0", s2))
    @test array(ψ, s1, s2) ≈ array(state("1", s1) * state("1", s2), s1, s2)
    @test issetequal(inds(op("Rxx", s1, s2; θ)), [s1, s2, prime(s1), prime(s2)])
    @test_throws ArgumentError op("NotAnOperator", s1)
    @test_throws ArgumentError state("NotAState", s1)

    # every circuit gate is its registry operator, with the parameters forwarded unchanged
    for (name, spec) in GATES
        nsites = haskey(OPERATORS[(1, 2)], spec.opname) ? 1 : 2
        params = (θ, ϕ, λ, β)[1:length(spec.paramkeys)]
        gate = isempty(params) ? (name, collect(1:nsites)) : (name, collect(1:nsites), length(params) == 1 ? only(params) : params)
        t, vs = toitensor(gate, g, s)
        @test t ≈ op(spec.opname, [s1, s2][1:nsites]...; NamedTuple{spec.paramkeys}(params)...)
    end
    # qiskit's gate names (lowercase) resolve to the same gates
    for name in ("id", "s", "sdg", "t", "tdg", "sx", "sxdg", "ch", "cs", "csdg", "csx", "dcx", "ecr")
        nsites = name in ("ch", "cs", "csdg", "csx", "dcx", "ecr") ? 2 : 1
        @test first(toitensor((name, collect(1:nsites)), g, s)) isa TensorNetworkQuantumSimulator.Tensor
    end

    register_op!("MyHalfZ", [0.5 0.0; 0.0 -0.5])
    @test op("MyHalfZ", s1) ≈ op("Sz", s1)
    register_op!("MySWAP", [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]; nsites = 2)
    @test op("MySWAP", s1, s2) ≈ op("SWAP", s1, s2)
end

end
