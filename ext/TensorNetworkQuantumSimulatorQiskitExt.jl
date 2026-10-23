module TensorNetworkQuantumSimulatorQiskitExt

using Qiskit: QuantumCircuit
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator, _ImportedCircuit, _add_gate!, _gate_list,
    _measure!, _unsupported_gate

function TensorNetworkQuantumSimulator.circuit_from_qiskit(qc::QuantumCircuit; qubits = nothing)
    circuit = _ImportedCircuit(qc.num_qubits)
    for instruction in qc.data
        name = instruction.name
        qs = instruction.qubits .- (qc.offset - 1)
        if name in ("barrier", "delay", "global_phase")
            continue
        elseif name == "measure"
            _measure!(circuit, qs)
        elseif name in ("reset", "unitary")
            throw(ArgumentError("$name instructions are not supported"))
        else
            _add_gate!(circuit, name, instruction.params, qs) || _unsupported_gate(name, qs)
        end
    end
    return _gate_list(circuit; qubits)
end

end
