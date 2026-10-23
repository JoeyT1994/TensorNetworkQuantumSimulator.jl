# --- Circuits from other frameworks ---------------------------------------------
# Shared by `circuit_from_qasm` and `circuit_from_qiskit`: instructions on numbered qubits
# (1, 2, ...) become a gate list.

mutable struct _ImportedCircuit
    nqubits::Int
    measured::Set{Int}
    gates::Vector{Tuple{String, Vector{Int}, Vector{Float64}}}
end
_ImportedCircuit(nqubits::Integer = 0) = _ImportedCircuit(nqubits, Set{Int}(), [])

_measure!(circuit::_ImportedCircuit, qubits) = (union!(circuit.measured, qubits); circuit)

# Add the registry's gate `name` on `qubits`; false if the registry does not have it.
function _add_gate!(circuit::_ImportedCircuit, name, params, qubits)
    for k in qubits
        k in circuit.measured && throw(ArgumentError(
            "gate \"$name\" acts on a qubit after it is measured; mid-circuit measurements are not supported"
        ))
    end
    spec = _resolve_gate(name)
    (isnothing(spec) || length(qubits) > 2) && return false
    length(params) == length(spec.paramkeys) || throw(ArgumentError(
        "gate \"$name\" takes $(length(spec.paramkeys)) parameter(s), got $(length(params))"
    ))
    push!(circuit.gates, (name, collect(qubits), collect(Float64, params)))
    return true
end

function _unsupported_gate(name, qubits)
    length(qubits) > 2 && throw(ArgumentError(
        "gate \"$name\" acts on $(length(qubits)) qubits; only one- and two-qubit gates are supported, so decompose it first"
    ))
    throw(ArgumentError("unknown gate \"$name\""))
end

# The gate list, on the vertices `qubits` (the qubit numbers by default), under the registry's
# canonical gate names.
function _gate_list(circuit::_ImportedCircuit; qubits = nothing)
    if !isnothing(qubits) && length(qubits) != circuit.nqubits
        throw(ArgumentError("the circuit has $(circuit.nqubits) qubits but `qubits` has $(length(qubits)) vertices"))
    end
    vertex = isnothing(qubits) ? identity : (k -> qubits[k])
    return map(circuit.gates) do (name, qs, params)
        canon = haskey(GATES, name) ? name : ALIASES[name]
        isempty(params) && return (canon, vertex.(qs))
        return (canon, vertex.(qs), length(params) == 1 ? only(params) : Tuple(params))
    end
end

"""
    circuit_from_qiskit(qc::Qiskit.QuantumCircuit; qubits = nothing)

The gate list (see `apply_gates`) of a Qiskit.jl circuit. Available once Qiskit.jl is loaded
(`using Qiskit`); for circuits built in Python, export them with `qasm2.dumps` and use
`circuit_from_qasm`.

Every one- and two-qubit qiskit standard gate is supported. Barriers, delays, the global phase
and measurements at the end of the circuit are dropped. Build the graph from the circuit with
`build_graph_from_circuit`, or pass the vertex of each qubit as `qubits`; two-qubit gates must
then act on neighbouring vertices.

# Keyword Arguments
- `qubits = nothing`: The vertex of each qubit, in order. Defaults to the qubit numbers `1, 2, ...`.

Gates on three or more qubits, `unitary` instructions, resets and gates after a measurement are
not supported; transpile such circuits to one- and two-qubit basis gates first.
"""
function circuit_from_qiskit end
