# --- Circuits from OpenQASM 2 ---------------------------------------------------

"""
    circuit_from_qasm(qasm; qubits = nothing)

The gate list (see `apply_gates`) of an OpenQASM 2 program, e.g. one written by qiskit's
`qasm2.dumps(circuit)`. Gates defined in the program (`gate name(params) args { ... }`) are
expanded from their definitions; other gates are looked up by their qiskit names in the gate
registry. Barriers and measurements at the end of the circuit are dropped.

Build the graph from the circuit with `build_graph_from_circuit`, or simulate on a graph of
your own by passing the vertex of each qubit as `qubits`; two-qubit gates must then act on
neighbouring vertices (transpile the circuit onto the graph's connectivity first).

# Arguments
- `qasm::AbstractString`: The program text (use `read(path, String)` for a file).

# Keyword Arguments
- `qubits = nothing`: The vertex of each qubit, in the order the qubits are declared (across
  `qreg`s). Defaults to the qubit numbers `1, 2, ...`.

# Returns
- A `Vector` of gates, `(name, vertices)` or `(name, vertices, parameters)`, with identity gates
  on idle qubits to preserve them.

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
    program = QASMProgram(qasm_tokens(qasm))
    parse_program!(program)
    return gate_list(program.circuit; qubits)
end

# --- Tokens ---

function qasm_tokens(text::AbstractString)
    text = replace(text, r"//[^\n]*" => "")
    pattern = r"\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_]*|(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?|->|\S"
    return [m.match for m in eachmatch(pattern, text)]
end

mutable struct QASMProgram
    tokens::Vector{SubString{String}}
    position::Int
    registers::Dict{String, UnitRange{Int}}             # qreg name → its qubits
    # gate name → (params, args, body tokens)
    definitions::Dict{String, Tuple{Vector{String}, Vector{String}, Vector{SubString{String}}}}
    circuit::ImportedCircuit
end
QASMProgram(tokens) = QASMProgram(tokens, 1, Dict(), Dict(), ImportedCircuit())

peek_token(p::QASMProgram) = p.position <= length(p.tokens) ? p.tokens[p.position] : nothing
next_token!(p::QASMProgram) = (t = peek_token(p); isnothing(t) && qasm_error("unexpected end of program"); p.position += 1; t)
function expect_token!(p::QASMProgram, token)
    t = next_token!(p)
    t == token || qasm_error("expected \"$token\", got \"$t\"")
    return t
end
qasm_error(msg) = throw(ArgumentError("OpenQASM: $msg"))

# Tokens up to `stop` (exclusive), consuming it.
function tokens_until!(p::QASMProgram, stop = ";")
    out = SubString{String}[]
    while (t = next_token!(p)) != stop
        push!(out, t)
    end
    return out
end
skip_statement!(p::QASMProgram) = (tokens_until!(p); nothing)

# Tokens up to the closing `close` (exclusive), consuming it; nesting of `open`/`close` is respected.
function bracketed!(p::QASMProgram, open, close)
    depth, out = 0, SubString{String}[]
    while true
        t = next_token!(p)
        t == close && depth == 0 && return out
        t == open && (depth += 1)
        t == close && (depth -= 1)
        push!(out, t)
    end
end

# Comma-separated groups of tokens.
function split_commas(tokens)
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

# A parenthesised, comma-separated list `( ... )` if one comes next, else no groups.
parenthesised!(p::QASMProgram) = peek_token(p) == "(" ? (next_token!(p); split_commas(bracketed!(p, "(", ")"))) : Vector{SubString{String}}[]

# --- Statements ---

function parse_program!(p::QASMProgram)
    while !isnothing(peek_token(p))
        t = next_token!(p)
        if t == "OPENQASM"
            startswith(next_token!(p), "2") || qasm_error("only OpenQASM 2 is supported")
            skip_statement!(p)
        elseif t in ("include", "creg", "opaque", "barrier")
            skip_statement!(p)
        elseif t == "qreg"
            name = String(next_token!(p)); expect_token!(p, "[")
            n = parse(Int, next_token!(p)); expect_token!(p, "]"); expect_token!(p, ";")
            nq = p.circuit.nqubits
            p.registers[name] = (nq + 1):(nq + n)
            p.circuit.nqubits += n
        elseif t == "gate"
            name = String(next_token!(p))
            params = String.(first.(parenthesised!(p)))
            args = String.(filter(!=(","), tokens_until!(p, "{")))
            p.definitions[name] = (params, args, bracketed!(p, "{", "}"))
        elseif t == "measure"
            measure!(p.circuit, qubit_list(p, tokens_until!(p, "->")))
            skip_statement!(p)
        elseif t == "reset"
            qasm_error("reset is not supported")
        elseif t == "if"
            qasm_error("classically controlled gates are not supported")
        else
            gate_statement!(p, String(t))
        end
    end
    return p
end

# A gate call `name(params) args;`, with registers broadcast over their qubits. In a gate
# definition's body, `env` binds its parameters and `bound` its qubit arguments.
function gate_statement!(p::QASMProgram, name; env = Dict{String, Float64}(), bound = nothing)
    params = [evaluate_expression(e, env) for e in parenthesised!(p)]
    operands = [operand(p, a, bound) for a in split_commas(tokens_until!(p))]
    n = maximum(length, operands; init = 1)
    all(o -> length(o) in (1, n), operands) || qasm_error("registers of different sizes in \"$name\"")
    for i in 1:n
        call_gate!(p, name, params, [length(o) == 1 ? only(o) : o[i] for o in operands])
    end
    return p
end

# The qubits of an operand: `q[i]`, the whole register `q`, or a bound argument of a definition.
function operand(p::QASMProgram, tokens, bound = nothing)
    name = String(first(tokens))
    isnothing(bound) || return [get(() -> qasm_error("unknown argument \"$name\" in a gate definition"), bound, name)]
    haskey(p.registers, name) || qasm_error("unknown register \"$name\"")
    reg = p.registers[name]
    length(tokens) == 1 && return collect(reg)
    (length(tokens) == 4 && tokens[2] == "[" && tokens[4] == "]") || qasm_error("malformed operand $(join(tokens))")
    i = parse(Int, tokens[3])
    0 <= i < length(reg) || qasm_error("qubit $name[$i] is out of range")
    return [reg[i + 1]]
end

qubit_list(p::QASMProgram, tokens) = reduce(vcat, [operand(p, a) for a in split_commas(tokens)]; init = Int[])

# Explicit definitions take precedence over the registry, even for familiar gate names.
function call_gate!(p::QASMProgram, name, params, qubits)
    if !haskey(p.definitions, name)
        add_gate!(p.circuit, name, params, qubits) || unsupported_gate(name, qubits)
        return p
    end
    pnames, args, body = p.definitions[name]
    (length(pnames) == length(params) && length(args) == length(qubits)) ||
        qasm_error("gate \"$name\" called with the wrong number of parameters or qubits")
    return expand_definition!(p, body, Dict(zip(pnames, params)), Dict(zip(args, qubits)))
end

# The body of a gate definition, with its parameters and qubit arguments bound.
function expand_definition!(p::QASMProgram, body, env, bound)
    q = QASMProgram(body, 1, p.registers, p.definitions, p.circuit)
    while !isnothing(peek_token(q))
        t = next_token!(q)
        t == "barrier" ? skip_statement!(q) : gate_statement!(q, String(t); env, bound)
    end
    return p
end

# --- Parameter expressions ---
# Parsed as Julia, whose precedence matches OpenQASM's, and evaluated by walking the expression
# over a whitelist (numbers, pi, the definition's parameters, + - * / ^ and a few functions),
# never with `eval`.

const QASM_OPERATORS = Dict(:+ => +, :- => -, :* => *, :/ => /, :^ => ^)
const QASM_FUNCTIONS = Dict(:sin => sin, :cos => cos, :tan => tan, :exp => exp, :ln => log, :sqrt => sqrt)

function evaluate_expression(tokens, env)
    text = join(tokens)
    return evaluate_parameter(Meta.parse(text; raise = false), env, text)
end

evaluate_parameter(x::Real, env, text) = Float64(x)
function evaluate_parameter(x::Symbol, env, text)
    x === :pi && return Float64(π)
    return get(() -> qasm_error("unknown symbol \"$x\" in expression $text"), env, String(x))
end
function evaluate_parameter(x::Expr, env, text)
    if x.head === :call
        f, args = x.args[1], [evaluate_parameter(a, env, text) for a in x.args[2:end]]
        haskey(QASM_OPERATORS, f) && return QASM_OPERATORS[f](args...)
        haskey(QASM_FUNCTIONS, f) && length(args) == 1 && return QASM_FUNCTIONS[f](only(args))
    end
    return qasm_error("unsupported expression $text")
end
evaluate_parameter(x, env, text) = qasm_error("unsupported expression $text")
