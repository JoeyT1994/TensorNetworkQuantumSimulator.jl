@eval module $(gensym())
using Qiskit: QuantumCircuit
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws

# Qiskit.jl's one- and two-qubit gates, with their number of parameters
const GATES_1Q = [:h => 0, :id => 0, :x => 0, :y => 0, :z => 0, :p => 1, :r => 2, :rx => 1, :ry => 1, :rz => 1,
    :s => 0, :sdg => 0, :sx => 0, :sxdg => 0, :t => 0, :tdg => 0, :u => 3]
const GATES_2Q = [:ch => 0, :cx => 0, :cy => 0, :cz => 0, :dcx => 0, :ecr => 0, :swap => 0, :iswap => 0, :cp => 1,
    :crx => 1, :cry => 1, :crz => 1, :cs => 0, :csdg => 0, :csx => 0, :cu => 4, :rxx => 1, :ryy => 1, :rzz => 1, :rzx => 1]

# The same circuit as OpenQASM, from its instructions
function qasm(qc)
    lines = ["OPENQASM 2.0;", "include \"qelib1.inc\";", "qreg q[$(qc.num_qubits)];", "creg c[$(qc.num_clbits)];"]
    for inst in qc.data
        inst.name == "barrier" && continue
        qs = join(("q[$(k - 1)]" for k in inst.qubits), ",")
        inst.name == "measure" && (push!(lines, "measure $qs -> c[$(only(inst.clbits) - 1)];"); continue)
        params = isempty(inst.params) ? "" : "(" * join(repr.(inst.params), ",") * ")"
        push!(lines, "$(inst.name)$params $qs;")
    end
    return join(lines, "\n")
end

@testset "Qiskit.jl circuits" begin
    Random.seed!(2)
    qc = QuantumCircuit(4, 4)
    for (gate, np) in [GATES_1Q; GATES_2Q]
        pair = randperm(4)[1:2]
        qubits = (gate => np) in GATES_1Q ? pair[1:1] : pair
        getproperty(qc, gate)((π * (2rand() - 1) for _ in 1:np)..., qubits...)
    end
    qc.barrier()
    foreach(k -> qc.measure(k, k), 1:4)

    circuit = circuit_from_qiskit(qc)
    @test length(circuit) == length(GATES_1Q) + length(GATES_2Q)
    @test circuit == circuit_from_qasm(qasm(qc))
    @test circuit_from_qiskit(qc; qubits = [(1, k) for k in 1:4]) == circuit_from_qasm(qasm(qc); qubits = [(1, k) for k in 1:4])

    # it runs, unitarily
    ψ0 = tensornetworkstate(ComplexF64, v -> "↑", build_graph_from_circuit(circuit), "S=1/2")
    ψ, _ = apply_gates(circuit, ψ0; apply_kwargs = (; maxdim = 64, cutoff = 1.0e-14, normalize_tensors = false), verbose = false)
    @test norm_sqr(ψ; alg = "exact") ≈ 1

    three = QuantumCircuit(3)
    three.ccx(1, 2, 3)
    @test_throws ArgumentError circuit_from_qiskit(three)
    after = QuantumCircuit(1, 1)
    after.measure(1, 1)
    after.x(1)
    @test_throws ArgumentError circuit_from_qiskit(after)
    reset = QuantumCircuit(1)
    reset.reset(1)
    @test_throws ArgumentError circuit_from_qiskit(reset)
end

end
