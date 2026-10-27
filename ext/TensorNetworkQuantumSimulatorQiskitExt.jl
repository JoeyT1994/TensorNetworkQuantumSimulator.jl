module TensorNetworkQuantumSimulatorQiskitExt

using Qiskit: QuantumCircuit
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator, ImportedCircuit, add_gate!, finish_circuit,
    measure!, unsupported_gate

function TensorNetworkQuantumSimulator.circuit_from_qiskit(qc::QuantumCircuit; qubits = nothing)
    circuit = ImportedCircuit(qc.num_qubits)
    for instruction in qc.data
        name = instruction.name
        qs = instruction.qubits .- (qc.offset - 1)
        if name in ("barrier", "delay", "global_phase")
            continue
        elseif name == "measure"
            measure!(circuit, qs)
        elseif name in ("reset", "unitary")
            throw(ArgumentError("$name instructions are not supported"))
        else
            add_gate!(circuit, name, instruction.params, qs) || unsupported_gate(name, qs)
        end
    end
    return finish_circuit(circuit; qubits)
end

end
