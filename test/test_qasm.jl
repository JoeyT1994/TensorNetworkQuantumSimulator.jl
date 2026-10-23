@eval module $(gensym())
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws

const HEADER = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"

@testset "OpenQASM: every one- and two-qubit qiskit gate agrees with qiskit" begin
    # a circuit written by qiskit, and the Pauli expectation values of its statevector
    circuit = circuit_from_qasm(read(joinpath(@__DIR__, "qasm", "qiskit_standard_gates.qasm"), String))
    reference = [split(l) for l in readlines(joinpath(@__DIR__, "qasm", "qiskit_standard_gates.txt")) if !startswith(l, "#")]
    ψ0 = tensornetworkstate(ComplexF64, v -> "↑", named_path_graph(4), "S=1/2")
    ψ, _ = apply_gates(circuit, ψ0; apply_kwargs = (; maxdim = 64, cutoff = 1.0e-14))
    observables = [(String(r[1]), parse.(Int, r[2:(end - 1)])) for r in reference]
    @test expect(ψ, observables; alg = "exact") ≈ parse.(Float64, last.(reference)) atol = 1.0e-12
end

@testset "OpenQASM: registers, measurements, definitions and parameters" begin
    # registers are numbered in declaration order and broadcast; final measurements are dropped
    @test circuit_from_qasm(HEADER * "qreg a[2];\nqreg b[2];\ncreg c[2];\ncx a, b;\nbarrier a;\nmeasure b -> c;") ==
        [("CX", [1, 3]), ("CX", [2, 4])]
    # gates the registry does not know are expanded from their definitions
    @test circuit_from_qasm(HEADER * "qreg q[2];\ngate mycx c, t { h t; cz c, t; h t; }\nmycx q[1], q[0];") ==
        [("H", [1]), ("CZ", [2, 1]), ("H", [1])]
    # parameter expressions, onto vertices of our choosing
    circuit = circuit_from_qasm(HEADER * "qreg q[2];\nu(pi/2, 0.25*pi, -2.5e-1*pi) q[1];\nrzz(2^-1*sin(pi/2)) q[0], q[1];"; qubits = [(1, 1), (1, 2)])
    @test circuit == [("U", [(1, 2)], (π / 2, π / 4, -π / 4)), ("Rzz", [(1, 1), (1, 2)], 0.5)]
end

@testset "OpenQASM: what is not supported" begin
    for program in [
            "qreg q[3];\nccx q[0], q[1], q[2];",                     # three-qubit gates
            "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\nx q[0];", # mid-circuit measurements
            "qreg q[1];\nreset q[0];",
            "qreg q[1];\nnotagate q[0];",
            "qreg q[1];\nrx q[0];",                                   # a missing parameter
            "qreg q[2];\nh q[0], q[1];",                              # the wrong number of qubits
            "qreg q[1];\nrx(run(`true`)) q[0];",                      # parameters are never run as code
        ]
        @test_throws ArgumentError circuit_from_qasm(HEADER * program)
    end
    @test_throws ArgumentError circuit_from_qasm("OPENQASM 3.0;\nqubit[2] q;\nh q[0];")
    @test_throws ArgumentError circuit_from_qasm(HEADER * "qreg q[2];\nh q[0];"; qubits = [1])
end

end
