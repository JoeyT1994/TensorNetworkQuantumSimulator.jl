# --- Circuits from OpenQASM 2 ---------------------------------------------------

"""
    circuit_from_qasm(qasm; qubits = nothing)

The gate list (see `apply_gates`) of an OpenQASM 2 program, e.g. one written by qiskit's
`qasm2.dumps(circuit)`. Gates are looked up by their qiskit names in the gate registry; gates
defined in the program (`gate name(params) args { ... }`) that the registry does not know are
expanded from their definitions. Barriers and measurements at the end of the circuit are
dropped.

Build the graph from the circuit with `build_graph_from_circuit`, or simulate on a graph of
your own by passing the vertex of each qubit as `qubits`; two-qubit gates must then act on
neighbouring vertices (transpile the circuit onto the graph's connectivity first).

# Arguments
- `qasm::AbstractString`: The program text (use `read(path, String)` for a file).

# Keyword Arguments
- `qubits = nothing`: The vertex of each qubit, in the order the qubits are declared (across
  `qreg`s). Defaults to the qubit numbers `1, 2, ...`.

# Returns
- A `Vector` of gates, `(name, vertices)` or `(name, vertices, parameters)`.

Gates on three or more qubits, resets, classically controlled gates and gates after a
measurement are not supported; decompose such circuits first (e.g. qiskit's `transpile` with
one- and two-qubit `basis_gates`).

# Example
```julia
circuit = circuit_from_qasm(\"\"\"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    h q[0];
    cx q[0], q[1];
    rz(pi/4) q[2];
\"\"\")
g = build_graph_from_circuit(circuit)
ψ, errors = apply_gates(circuit, tensornetworkstate(v -> "↑", g))
```
"""
function circuit_from_qasm(qasm::AbstractString; qubits = nothing)
    program = _QASMProgram(_qasm_tokens(qasm))
    _parse_program!(program)
    return _gate_list(program.circuit; qubits)
end

# --- Tokens ---

function _qasm_tokens(text::AbstractString)
    text = replace(text, r"//[^\n]*" => "")
    pattern = r"\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_]*|(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|->|\S"
    return [m.match for m in eachmatch(pattern, text)]
end

mutable struct _QASMProgram
    tokens::Vector{SubString{String}}
    position::Int
    registers::Dict{String, UnitRange{Int}}             # qreg name → its qubits
    definitions::Dict{String, Any}                       # gate name → (params, args, body tokens)
    circuit::_ImportedCircuit
end
_QASMProgram(tokens) = _QASMProgram(tokens, 1, Dict(), Dict(), _ImportedCircuit())

_peek(p::_QASMProgram) = p.position <= length(p.tokens) ? p.tokens[p.position] : nothing
_next!(p::_QASMProgram) = (t = _peek(p); isnothing(t) && _qasm_error("unexpected end of program"); p.position += 1; t)
function _expect!(p::_QASMProgram, token)
    t = _next!(p)
    t == token || _qasm_error("expected \"$token\", got \"$t\"")
    return t
end
_skip_statement!(p::_QASMProgram) = (while _next!(p) != ";" end)
_qasm_error(msg) = throw(ArgumentError("OpenQASM: $msg"))

# Tokens up to the closing `close` (exclusive), consuming it; nesting of `open`/`close` is respected.
function _until!(p::_QASMProgram, open, close)
    depth, out = 0, SubString{String}[]
    while true
        t = _next!(p)
        t == close && depth == 0 && return out
        t == open && (depth += 1)
        t == close && (depth -= 1)
        push!(out, t)
    end
end

# Comma-separated groups of tokens.
function _split_commas(tokens)
    groups, depth, current = Vector{SubString{String}}[], 0, SubString{String}[]
    for t in tokens
        if t == "," && depth == 0
            push!(groups, current); current = SubString{String}[]
            continue
        end
        t in ("(", "[") && (depth += 1)
        t in (")", "]") && (depth -= 1)
        push!(current, t)
    end
    isempty(current) || push!(groups, current)
    return groups
end

# --- Statements ---

function _parse_program!(p::_QASMProgram)
    while !isnothing(_peek(p))
        t = _next!(p)
        if t == "OPENQASM"
            startswith(_next!(p), "2") || _qasm_error("only OpenQASM 2 is supported")
            _skip_statement!(p)
        elseif t in ("include", "creg", "opaque", "barrier")
            _skip_statement!(p)
        elseif t == "qreg"
            name = String(_next!(p)); _expect!(p, "[")
            n = parse(Int, _next!(p)); _expect!(p, "]"); _expect!(p, ";")
            nq = p.circuit.nqubits
            p.registers[name] = (nq + 1):(nq + n)
            p.circuit.nqubits += n
        elseif t == "gate"
            name = String(_next!(p))
            params = _peek(p) == "(" ? (_next!(p); String.(first.(_split_commas(_until!(p, "(", ")"))))) : String[]
            args = String[]
            while (a = _next!(p)) != "{"
                a == "," || push!(args, String(a))
            end
            p.definitions[name] = (params, args, _until!(p, "{", "}"))
        elseif t == "measure"
            _measure!(p.circuit, _qubit_list(p, _tokens_until!(p, "->")))
            _skip_statement!(p)
        elseif t == "reset"
            _qasm_error("reset is not supported")
        elseif t == "if"
            _qasm_error("classically controlled gates are not supported")
        else
            _gate_statement!(p, String(t))
        end
    end
    return p
end

function _tokens_until!(p::_QASMProgram, stop = ";")
    out = SubString{String}[]
    while (t = _next!(p)) != stop
        push!(out, t)
    end
    return out
end

# A gate call `name(params) args;`, with registers broadcast over their qubits. In a gate
# definition's body, `env` binds its parameters and `bound` its qubit arguments.
function _gate_statement!(p::_QASMProgram, name; env = Dict{String, Float64}(), bound = nothing)
    exprs = _peek(p) == "(" ? (_next!(p); _split_commas(_until!(p, "(", ")"))) : Vector{SubString{String}}[]
    params = [_evaluate_expression(e, env) for e in exprs]
    operands = [_operand(p, a, bound) for a in _split_commas(_tokens_until!(p))]
    n = maximum(length, operands; init = 1)
    all(o -> length(o) in (1, n), operands) || _qasm_error("registers of different sizes in \"$name\"")
    for i in 1:n
        _call!(p, name, params, [length(o) == 1 ? only(o) : o[i] for o in operands])
    end
    return p
end

# The qubits of an operand: `q[i]`, the whole register `q`, or a bound argument of a definition.
function _operand(p::_QASMProgram, tokens, bound = nothing)
    name = String(first(tokens))
    isnothing(bound) || return [get(() -> _qasm_error("unknown argument \"$name\" in a gate definition"), bound, name)]
    haskey(p.registers, name) || _qasm_error("unknown register \"$name\"")
    reg = p.registers[name]
    length(tokens) == 1 && return collect(reg)
    (length(tokens) == 4 && tokens[2] == "[" && tokens[4] == "]") || _qasm_error("malformed operand $(join(tokens))")
    i = parse(Int, tokens[3])
    0 <= i < length(reg) || _qasm_error("qubit $name[$i] is out of range")
    return [reg[i + 1]]
end

_qubit_list(p::_QASMProgram, tokens) = reduce(vcat, [_operand(p, a) for a in _split_commas(tokens)]; init = Int[])

# Apply `name` to `qubits`: from the registry if it has the gate, otherwise from its definition.
function _call!(p::_QASMProgram, name, params, qubits)
    _add_gate!(p.circuit, name, params, qubits) && return p
    haskey(p.definitions, name) || _unsupported_gate(name, qubits)
    pnames, args, body = p.definitions[name]
    (length(pnames) == length(params) && length(args) == length(qubits)) ||
        _qasm_error("gate \"$name\" called with the wrong number of parameters or qubits")
    return _expand!(p, body, Dict(zip(pnames, params)), Dict(zip(args, qubits)))
end

# The body of a gate definition, with its parameters and qubit arguments bound.
function _expand!(p::_QASMProgram, body, env, bound)
    q = _QASMProgram(body, 1, p.registers, p.definitions, p.circuit)
    while !isnothing(_peek(q))
        t = _next!(q)
        t == "barrier" ? _skip_statement!(q) : _gate_statement!(q, String(t); env, bound)
    end
    return p
end

# --- Parameter expressions ---
# Parsed as Julia, whose precedence matches OpenQASM's, and evaluated by walking the expression
# over a whitelist (numbers, pi, the definition's parameters, + - * / ^ and a few functions),
# never with `eval`.

const _QASM_OPERATORS = Dict(:+ => +, :- => -, :* => *, :/ => /, :^ => ^)
const _QASM_FUNCTIONS = Dict(:sin => sin, :cos => cos, :tan => tan, :exp => exp, :ln => log, :sqrt => sqrt)

function _evaluate_expression(tokens, env)
    text = join(tokens)
    return _evaluate(Meta.parse(text; raise = false), env, text)
end

_evaluate(x::Real, env, text) = Float64(x)
function _evaluate(x::Symbol, env, text)
    x === :pi && return Float64(π)
    return get(() -> _qasm_error("unknown symbol \"$x\" in expression $text"), env, String(x))
end
function _evaluate(x::Expr, env, text)
    if x.head === :call
        f, args = x.args[1], [_evaluate(a, env, text) for a in x.args[2:end]]
        haskey(_QASM_OPERATORS, f) && return _QASM_OPERATORS[f](args...)
        haskey(_QASM_FUNCTIONS, f) && length(args) == 1 && return _QASM_FUNCTIONS[f](only(args))
    end
    return _qasm_error("unsupported expression $text")
end
_evaluate(x, env, text) = _qasm_error("unsupported expression $text")
