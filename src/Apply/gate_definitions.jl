# --- Gate registry -----------------------------------------------------------

# Internal dispatch record for a circuit-tuple gate name.
#
# - `opname`: the operator name looked up by `op` (see operators.jl). Usually equal to
#   the user-facing key, but kept separate so a registry entry can rename if needed.
# - `paramkeys`: keyword names accepted by the underlying operator definition, e.g.
#   `(:θ,)` or `(:θ, :β)`. Empty for fixed gates. Parameters are forwarded unchanged.
struct GateSpec
    opname::String
    paramkeys::Tuple{Vararg{Symbol}}
end
GateSpec(opname; paramkeys = ()) = GateSpec(opname, paramkeys)

# Registry of circuit-tuple gates. Adding a new gate is one entry here (plus its matrix in
# `OPERATORS`, operators.jl, if it is not there already).
const GATES = Dict{String, GateSpec}(
    # Single-qubit fixed
    "I" => GateSpec("I"),
    "X" => GateSpec("X"),
    "Y" => GateSpec("Y"),
    "Z" => GateSpec("Z"),
    "H" => GateSpec("H"),
    "S" => GateSpec("S"),
    "Sdg" => GateSpec("Sdg"),
    "T" => GateSpec("T"),
    "Tdg" => GateSpec("Tdg"),
    "SX" => GateSpec("SX"),
    "SXdg" => GateSpec("SXdg"),

    # Single-qubit parametric
    "Rx"  => GateSpec("Rx";  paramkeys = (:θ,)),
    "Ry"  => GateSpec("Ry";  paramkeys = (:θ,)),
    "Rz"  => GateSpec("Rz";  paramkeys = (:θ,)),
    "P"   => GateSpec("P";   paramkeys = (:θ,)),
    "Rz+" => GateSpec("Rz+"; paramkeys = (:θ,)),
    "R"   => GateSpec("R";   paramkeys = (:θ, :ϕ)),
    "U"   => GateSpec("U";   paramkeys = (:θ, :ϕ, :λ)),
    "U2"  => GateSpec("U2";  paramkeys = (:ϕ, :λ)),

    # Two-qubit fixed
    "CNOT"   => GateSpec("CNOT"),
    "CX"     => GateSpec("CX"),
    "CY"     => GateSpec("CY"),
    "CZ"     => GateSpec("CZ"),
    "CH"     => GateSpec("CH"),
    "CS"     => GateSpec("CS"),
    "CSdg"   => GateSpec("CSdg"),
    "CSX"    => GateSpec("CSX"),
    "DCX"    => GateSpec("DCX"),
    "ECR"    => GateSpec("ECR"),
    "SWAP"   => GateSpec("SWAP"),
    "iSWAP"  => GateSpec("iSWAP"),
    "√SWAP"  => GateSpec("√SWAP"),
    "√iSWAP" => GateSpec("√iSWAP"),

    # Two-qubit parametric
    "Rxx" => GateSpec("Rxx"; paramkeys = (:θ,)),
    "Ryy" => GateSpec("Ryy"; paramkeys = (:θ,)),
    "Rzz" => GateSpec("Rzz"; paramkeys = (:θ,)),
    "Rzx" => GateSpec("Rzx"; paramkeys = (:θ,)),

    "CRx"    => GateSpec("CRx";    paramkeys = (:θ,)),
    "CRy"    => GateSpec("CRy";    paramkeys = (:θ,)),
    "CRz"    => GateSpec("CRz";    paramkeys = (:θ,)),
    "CPHASE" => GateSpec("CPHASE"; paramkeys = (:θ,)),
    "CU"     => GateSpec("CU";     paramkeys = (:θ, :ϕ, :λ, :γ)),
    "CU3"    => GateSpec("CU3";    paramkeys = (:θ, :ϕ, :λ)),
    "xx_plus_yy"  => GateSpec("xx_plus_yy";  paramkeys = (:θ, :β)),
    "xx_minus_yy" => GateSpec("xx_minus_yy"; paramkeys = (:θ, :β)),

    # No qiskit equivalent
    "Rz+z+"   => GateSpec("Rz+z+";   paramkeys = (:θ,)),
    "Rxxyy"   => GateSpec("Rxxyy";   paramkeys = (:θ,)),
    "Rxxyyzz" => GateSpec("Rxxyyzz"; paramkeys = (:θ,)),
)

# Snapshot of built-in canonical names taken at module load. Used to prevent
# `register_gate!` / `unregister_gate!` from mutating the library's own gates;
# user-registered gates remain freely overwritable.
const BUILTIN_GATES = Set(keys(GATES))

# Aliases mapping qiskit-style names to our canonical `GATES` keys. Most of the
# difference is casing (qiskit uses lowercase), so lowercase aliases are derived
# automatically. Only genuine name differences are listed explicitly.
const ALIASES = let
    m = Dict{String, String}()
    for canon in keys(GATES)
        l = lowercase(canon)
        l != canon && (m[l] = canon)
    end
    # Genuine name differences (qiskit name => our canonical name), in either case
    for (alias, canon) in ["cp" => "CPHASE", "id" => "I", "U1" => "P", "U3" => "U", "CU1" => "CPHASE"]
        m[alias] = m[lowercase(alias)] = canon
    end
    m
end

# The canonical name of a gate, or `nothing` if it is not registered.
canonical_gate_name(name::AbstractString) = haskey(GATES, name) ? name : get(ALIASES, name, nothing)

function resolve_gate(name::AbstractString)
    canon = canonical_gate_name(name)
    return isnothing(canon) ? nothing : GATES[canon]
end

# Suggest canonical gate names close to `name` (case-insensitive edit distance).
# Returns up to `topk` keys ranked by distance, only those within `maxdist`.
function gate_suggestions(name::AbstractString; topk::Int = 3, maxdist::Int = 2)
    lname = lowercase(name)
    scored = [(g, levenshtein(lname, lowercase(g))) for g in keys(GATES)]
    filter!(p -> last(p) <= maxdist, scored)
    sort!(scored; by = p -> (p[2], p[1]))
    return [first(p) for p in Iterators.take(scored, topk)]
end

# --- Circuit-tuple → Tensor -------------------------------------------------

# Vector of gates → vector of (Tensor, vertices)
function toitensor(circuit::Vector, g::NamedGraph, sinds::Dictionary)
    return [toitensor(gate, g, sinds) for gate in circuit]
end

# Single circuit tuple → (Tensor, vertices)
function toitensor(gate::Tuple, g::NamedGraph, siteinds::Dictionary)
    name = gate[1]
    verts = collect_vertices(gate[2], g)
    s_inds = [only(siteinds[v]) for v in verts]

    # Multi-letter Pauli-string sugar: "XYZ" → X⊗Y⊗Z applied componentwise.
    # Single-letter "X"/"Y"/"Z" goes through the registry below.
    if length(name) > 1 && all(in("XYZ"), name)
        length(name) == length(verts) || throw(ArgumentError(
            "Pauli string \"$name\" has $(length(name)) letters but acts on $(length(verts)) vertices."
        ))
        return prod(op(string(c), sind) for (c, sind) in zip(name, s_inds)), verts
    end

    spec = resolve_gate(name)
    if spec === nothing
        suggestions = gate_suggestions(name)
        msg = "Unknown gate \"$name\"."
        if !isempty(suggestions)
            msg *= " Did you mean: " * join(("\"$s\"" for s in suggestions), ", ") * "?"
        else
            msg *= " Registered gates: $(sort(collect(keys(GATES))))."
        end
        throw(ArgumentError(msg))
    end

    if isempty(spec.paramkeys)
        return op(spec.opname, s_inds...), verts
    end

    length(gate) >= 3 || throw(ArgumentError(
        "Gate \"$name\" expects $(length(spec.paramkeys)) parameter(s) as the third entry of its tuple, got none."
    ))
    raw = gate[3]
    pvals = raw isa Union{Tuple, AbstractVector} ? Tuple(raw) : (raw,)
    length(pvals) == length(spec.paramkeys) || throw(ArgumentError(
        "Gate \"$name\" expects $(length(spec.paramkeys)) parameter(s), got $(length(pvals))."
    ))
    kwargs = NamedTuple{spec.paramkeys}(pvals)
    return op(spec.opname, s_inds...; kwargs...), verts
end

# --- Public registration API ------------------------------------------------

"""
    register_gate!(name::String; opname = name, paramkeys = ())

Register a custom gate `name` so it can be used in circuit-tuple form
`(name, vertices, parameter)` with `apply_gates`, for the current Julia session.

The matrix itself must be registered separately with [`register_op!`](@ref)
under `opname` (defaults to `name`). See "Custom Gates" in the gate docs for a
worked example. Built-in gates cannot be overwritten (`ArgumentError`); gates
registered earlier can.

# Arguments
- `name`: name used in circuit tuples.

# Keyword Arguments
- `opname`: the operator name looked up by `op`. Defaults to `name`.
- `paramkeys`: tuple of keyword names accepted by the underlying operator, e.g.
  `(:θ,)` for a single rotation angle, `(:θ, :β)` for a two-parameter gate.
  Empty (`()`) for non-parametric gates.
"""
function register_gate!(name::String; opname::String = name, paramkeys::Tuple = ())
    name in BUILTIN_GATES && throw(ArgumentError(
        "\"$name\" is a built-in gate and cannot be overwritten. " *
        "Choose a different name for your custom gate, or register a new " *
        "operator with `register_op!` if you need a different matrix."
    ))
    GATES[name] = GateSpec(opname, paramkeys)
    return name
end

"""
    register_alias!(alias::String, canonical::String)

Register `alias` as an alternative name resolving to the gate `canonical`,
which must already be registered (built-in or registered via [`register_gate!`](@ref)).

Like [`register_gate!`](@ref), the alias lives only in the current Julia session.
"""
function register_alias!(alias::String, canonical::String)
    haskey(GATES, canonical) || throw(ArgumentError(
        "Cannot register alias \"$alias\" → \"$canonical\": " *
        "canonical gate is not registered. " *
        "Call `register_gate!(\"$canonical\"; ...)` first."
    ))
    ALIASES[alias] = canonical
    return alias
end

"""
    unregister_gate!(name::String)

Remove `name` from the gate registry. Also removes any aliases pointing to it.
Returns `name`. No-op if `name` is not registered.

Built-in gates are locked: attempting to unregister one throws `ArgumentError`.
"""
function unregister_gate!(name::String)
    name in BUILTIN_GATES && throw(ArgumentError(
        "\"$name\" is a built-in gate and cannot be unregistered."
    ))
    delete!(GATES, name)
    for (alias, canonical) in collect(ALIASES)
        canonical == name && delete!(ALIASES, alias)
    end
    return name
end
