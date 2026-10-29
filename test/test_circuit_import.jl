@eval module $(gensym())
using Qiskit: QuantumCircuit
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws

const HEADER = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nu3(0.3,0.5,0.7) q[0];\nu3(1.1,0.2,0.4) q[1];\n"

const ψ0 = tensornetworkstate(ComplexF64, v -> "↑", named_path_graph(2), "S=1/2")

# The state from a two-qubit QASM program, after the fixed non-trivial preparation in `HEADER`.
function run_qasm(body)
    circuit = circuit_from_qasm(HEADER * body)
    return first(apply_gates(circuit, ψ0; apply_kwargs = (; maxdim = 4, cutoff = 1.0e-14)))
end

fidelity(ψ, ϕ) = abs2(inner(ψ, ϕ; alg = "exact")) / (norm_sqr(ψ; alg = "exact") * norm_sqr(ϕ; alg = "exact"))

@testset "Imported gates follow qiskit's conventions" begin
    # Each registry gate against qiskit's own definition of it, written as a custom QASM gate
    definitions = [
        "rzx(0.7) q[0],q[1];" => "gate d(t) a,b { h b; cx a,b; rz(t) b; cx a,b; h b; }\nd(0.7) q[0],q[1];",
        "r(0.7,-0.4) q[0];" => "gate d(t,p) a { u3(t,p-pi/2,-p+pi/2) a; }\nd(0.7,-0.4) q[0];",
        "csx q[0],q[1];" => "gate d a,b { h b; cu1(pi/2) a,b; h b; }\nd q[0],q[1];",
        "cu(0.7,-0.4,0.9,0.3) q[0],q[1];" => """
        gate d(t,p,l,g) a,b { u1(g) a; u1((l+p)/2) a; u1((l-p)/2) b; cx a,b; u3(-t/2,0,-(p+l)/2) b; cx a,b; u3(t/2,p,0) b; }
        d(0.7,-0.4,0.9,0.3) q[0],q[1];""",
    ]
    for (gate, definition) in definitions
        @test fidelity(run_qasm(gate), run_qasm(definition)) ≈ 1
    end
end

@testset "Qiskit.jl and QASM imports agree" begin
    qc = QuantumCircuit(2, 2)
    qc.h(1)
    qc.cx(1, 2)
    qc.rzx(0.7, 1, 2)
    qc.r(0.7, -0.4, 1)
    qc.csx(1, 2)
    qc.cu(0.7, -0.4, 0.9, 0.3, 1, 2)
    qc.measure(1, 1)
    qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n" *
        "rzx(0.7) q[0],q[1];\nr(0.7,-0.4) q[0];\ncsx q[0],q[1];\ncu(0.7,-0.4,0.9,0.3) q[0],q[1];\nmeasure q[0] -> c[0];\n"
    @test circuit_from_qiskit(qc).gates == circuit_from_qasm(qasm).gates
    @test_throws ArgumentError circuit_from_qiskit(qc; qubits = [(1, 1), (1, 1)])
end
end
