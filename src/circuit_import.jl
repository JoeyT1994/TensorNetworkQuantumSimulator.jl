# --- Circuits from other frameworks ---------------------------------------------
# Shared by `circuit_from_qasm` and `circuit_from_qiskit`: instructions on numbered qubits
# (1, 2, ...) become a circuit with declared qubits and a gate list.

"""
    Circuit(qubits, gates)

A circuit's qubits, in declaration order, and its gates as `(name, vertices)` or
`(name, vertices, parameters)` tuples. Idle qubits remain in `qubits` without adding gates.
Pass the circuit directly to `build_graph_from_circuit` or `apply_gates`; access its gate
list as `circuit.gates`.
"""
struct Circuit{V, G}
    qubits::Vector{V}
    gates::Vector{G}
end

build_graph_from_circuit(circuit::Circuit) = build_graph_from_circuit(circuit.gates; qubits = circuit.qubits)

function apply_gates(circuit::Circuit, ψ::Union{TensorNetworkState, BeliefPropagationCache}; kwargs...)
    all(v -> has_vertex(graph(ψ), v), circuit.qubits) ||
        throw(ArgumentError("the state must contain every declared qubit of the circuit"))
    isempty(circuit.gates) && return copy(ψ), Float64[]
    return apply_gates(circuit.gates, ψ; kwargs...)
end

mutable struct ImportedCircuit
    nqubits::Int
    measured::Set{Int}
    gates::Vector{Tuple{String, Vector{Int}, Vector{Float64}}}
end
ImportedCircuit(nqubits::Integer = 0) = ImportedCircuit(nqubits, Set{Int}(), [])

measure!(circuit::ImportedCircuit, qubits) = (union!(circuit.measured, qubits); circuit)

# Add the registry's gate `name` on `qubits`, under its canonical name; false if the registry does
# not have the gate.
function add_gate!(circuit::ImportedCircuit, name, params, qubits)
    any(in(circuit.measured), qubits) && throw(ArgumentError(
        "gate \"$name\" acts on a qubit after it is measured; mid-circuit measurements are not supported"
    ))
    canon = canonical_gate_name(name)
    isnothing(canon) && return false
    spec = GATES[canon]
    haskey(get(OPERATORS, (length(qubits), 2), Dict()), spec.opname) ||
        throw(ArgumentError("gate \"$name\" does not act on $(length(qubits)) qubit(s)"))
    length(params) == length(spec.paramkeys) || throw(ArgumentError(
        "gate \"$name\" takes $(length(spec.paramkeys)) parameter(s), got $(length(params))"
    ))
    push!(circuit.gates, (canon, collect(qubits), collect(Float64, params)))
    return true
end

function unsupported_gate(name, qubits)
    length(qubits) > 2 && throw(ArgumentError(
        "gate \"$name\" acts on $(length(qubits)) qubits; only one- and two-qubit gates are supported, so decompose it first"
    ))
    throw(ArgumentError("unknown gate \"$name\""))
end

# The circuit, on the vertices `qubits` (the qubit numbers by default).
function finish_circuit(circuit::ImportedCircuit; qubits = nothing)
    if !isnothing(qubits) && length(qubits) != circuit.nqubits
        throw(ArgumentError("the circuit has $(circuit.nqubits) qubits but `qubits` has $(length(qubits)) vertices"))
    end
    !isnothing(qubits) && !allunique(qubits) && throw(ArgumentError("`qubits` must not repeat a vertex"))
    qubits = isnothing(qubits) ? collect(1:circuit.nqubits) : collect(qubits)
    gates = map(circuit.gates) do (name, qs, params)
        isempty(params) && return (name, qubits[qs])
        return (name, qubits[qs], length(params) == 1 ? only(params) : Tuple(params))
    end
    return Circuit(qubits, gates)
end

"""
    circuit_from_qiskit(qc::Qiskit.QuantumCircuit; qubits = nothing)

Import a Qiskit.jl circuit for `apply_gates`. Available once Qiskit.jl is loaded
(`using Qiskit`); for circuits built in Python, export them with `qasm2.dumps` and use
`circuit_from_qasm`.

Every one- and two-qubit qiskit standard gate is supported. Barriers, delays, the global phase
and measurements at the end of the circuit are dropped. Build the graph from the circuit with
`build_graph_from_circuit`, or pass the vertex of each qubit as `qubits`; two-qubit gates must
then act on neighbouring vertices.

Gates on three or more qubits, `unitary` instructions, resets and gates after a measurement are
not supported; transpile such circuits to one- and two-qubit basis gates first.

# Keyword Arguments
- `qubits = nothing`: The vertex of each qubit, in order. Defaults to the qubit numbers `1, 2, ...`.

# Returns
- A [`Circuit`](@ref) with all declared `qubits` and the imported `gates`.
"""
function circuit_from_qiskit end
