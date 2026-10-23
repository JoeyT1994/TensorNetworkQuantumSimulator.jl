@eval module $(gensym())
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws

const HEADER = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
run(circuit, ψ0) = first(apply_gates(circuit, ψ0; apply_kwargs = (; maxdim = 64, cutoff = 1.0e-14), verbose = false))
initial(g) = tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2")
overlap(ψ, ϕ) = abs(inner(ψ, ϕ; alg = "exact")) / sqrt(abs(inner(ψ, ψ; alg = "exact")) * abs(inner(ϕ, ϕ; alg = "exact")))

@testset "OpenQASM: a GHZ state, with registers broadcast and the final measurements dropped" begin
    circuit = circuit_from_qasm(HEADER * """
        qreg q[3];
        creg c[3];
        h q[0];
        cx q[0],q[1];  // entangle
        cx q[1],q[2];
        barrier q;
        measure q -> c;
    """)
    @test circuit == [("H", [1]), ("CX", [1, 2]), ("CX", [2, 3])]
    ψ = run(circuit, initial(build_graph_from_circuit(circuit)))
    @test expect(ψ, [("ZZ", [1, 3]), ("XXX", [1, 2, 3]), ("Z", [2])]; alg = "exact") ≈ [1, 1, 0] atol = 1.0e-12
    # broadcasting: one gate per qubit, and pairwise over registers
    @test length(circuit_from_qasm(HEADER * "qreg q[4];\nh q;")) == 4
    @test circuit_from_qasm(HEADER * "qreg a[2];\nqreg b[2];\ncx a, b;") == [("CX", [1, 3]), ("CX", [2, 4])]
end

@testset "OpenQASM: qubit order, parameters and custom vertices" begin
    # the first operand of cx is the control; registers are numbered in declaration order
    circuit = circuit_from_qasm(HEADER * "qreg a[1];\nqreg b[1];\nx a[0];\ncx a[0], b[0];\n")
    ψ = run(circuit, initial(build_graph_from_circuit(circuit)))
    @test expect(ψ, [("Z", [1]), ("Z", [2])]; alg = "exact") ≈ [-1, -1]
    # parameter expressions, onto vertices of our choosing
    circuit = circuit_from_qasm(HEADER * "qreg q[2];\nrz(-pi/2) q[0];\nu(pi/2, 0.25*pi, -2.5e-1*pi) q[1];\nrzz(2^-1*sin(pi/2)) q[0], q[1];"; qubits = [(1, 1), (1, 2)])
    @test circuit == [("Rz", [(1, 1)], -π / 2), ("U", [(1, 2)], (π / 2, π / 4, -π / 4)), ("Rzz", [(1, 1), (1, 2)], 0.5)]
end

@testset "OpenQASM: gate definitions" begin
    g = named_path_graph(2)
    ψ0 = initial(g)
    reference = run([("H", [1]), ("Ry", [2], 0.7), ("CX", [1, 2])], ψ0)
    custom = circuit_from_qasm(HEADER * """
        qreg q[2];
        gate rot(theta) a { ry(-theta*2^-1 + 3*theta/2) a; }
        gate mycx c, t { h t; cz c, t; h t; }
        gate prep(theta) a, b { h a; rot(theta) b; }
        prep(0.7) q[0], q[1];
        mycx q[0], q[1];
    """)
    @test overlap(run(custom, ψ0), reference) ≈ 1
    # gates the registry knows are taken from it; their definitions (as qiskit writes them) agree
    definitions = """
        gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
        gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
        gate myrzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
        gate myecr q0,q1 { myrzx(pi/4) q0,q1; x q0; myrzx(-pi/4) q0,q1; }
        qreg q[2];
        h q[0];
        ry(0.3) q[1];
    """
    registry = circuit_from_qasm(HEADER * definitions * "ecr q[0], q[1];\nrzx(0.4) q[1], q[0];")
    @test registry[3:4] == [("ECR", [1, 2]), ("Rzx", [2, 1], 0.4)]
    expanded = circuit_from_qasm(HEADER * definitions * "myecr q[0], q[1];\nmyrzx(0.4) q[1], q[0];")
    @test overlap(run(registry, ψ0), run(expanded, ψ0)) ≈ 1
end

@testset "OpenQASM: every one- and two-qubit qiskit gate agrees with qiskit" begin
    # a circuit written by qiskit, and the Pauli expectation values of its statevector
    circuit = circuit_from_qasm(read(joinpath(@__DIR__, "qasm", "qiskit_standard_gates.qasm"), String))
    reference = [split(l) for l in readlines(joinpath(@__DIR__, "qasm", "qiskit_standard_gates.txt")) if !startswith(l, "#")]
    ψ = run(circuit, initial(named_path_graph(4)))
    observables = [(String(r[1]), parse.(Int, r[2:(end - 1)])) for r in reference]
    @test expect(ψ, observables; alg = "exact") ≈ parse.(Float64, last.(reference)) atol = 1.0e-12
end

@testset "OpenQASM: what is not supported" begin
    @test_throws ArgumentError circuit_from_qasm(HEADER * "qreg q[3];\nccx q[0], q[1], q[2];")
    @test_throws ArgumentError circuit_from_qasm(HEADER * "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nx q[0];")
    @test_throws ArgumentError circuit_from_qasm(HEADER * "qreg q[1];\nreset q[0];")
    @test_throws ArgumentError circuit_from_qasm(HEADER * "qreg q[1];\nnotagate q[0];")
    @test_throws ArgumentError circuit_from_qasm(HEADER * "qreg q[1];\nrx q[0];")
    @test_throws ArgumentError circuit_from_qasm(HEADER * "qreg q[2];\nh q[0];"; qubits = [1])
end

end
