# --- Gate registry -----------------------------------------------------------

# Internal dispatch record for a circuit-tuple gate name.
#
# - `opname`: the operator name looked up in the backend op registry. Usually equal to the
#   user-facing key, but kept separate so a registry entry can rename if needed.
# - `paramkeys`: keyword names accepted by the underlying `op` definition, e.g.
#   `(:θ,)`, `(:ϕ,)`, or `(:θ, :β)`. Empty for fixed gates.
# - `rescale`: applied to the user-supplied parameter(s) before forwarding. Used
#   when our (qiskit) convention differs from the registry convention. For
#   multi-parameter gates, `rescale` receives and returns a tuple/vector.
struct GateSpec
    opname::String
    paramkeys::Tuple{Vararg{Symbol}}
    rescale::Function
end
GateSpec(opname; paramkeys = (), rescale = identity) = GateSpec(opname, paramkeys, rescale)

# Registry of circuit-tuple gates. Adding a new gate is one entry here (plus an
# `register_op!` entry if the library doesn't already provide one).
const GATES = Dict{String, GateSpec}(
    # Single-qubit fixed
    "X" => GateSpec("X"),
    "Y" => GateSpec("Y"),
    "Z" => GateSpec("Z"),
    "H" => GateSpec("H"),

    # Single-qubit parametric (qiskit and the registry agree on convention)
    "Rx"  => GateSpec("Rx";  paramkeys = (:θ,)),
    "Ry"  => GateSpec("Ry";  paramkeys = (:θ,)),
    "Rz"  => GateSpec("Rz";  paramkeys = (:θ,)),
    "P"   => GateSpec("P";   paramkeys = (:ϕ,)),
    "Rz+" => GateSpec("Rz+"; paramkeys = (:θ,)),

    # Two-qubit fixed
    "CNOT"   => GateSpec("CNOT"),
    "CX"     => GateSpec("CX"),
    "CY"     => GateSpec("CY"),
    "CZ"     => GateSpec("CZ"),
    "SWAP"   => GateSpec("SWAP"),
    "iSWAP"  => GateSpec("iSWAP"),
    "√SWAP"  => GateSpec("√SWAP"),
    "√iSWAP" => GateSpec("√iSWAP"),

    # Two-qubit parametric.
    # qiskit:   Rxx(θ) = exp(-i θ XX / 2)
    # registry: op("Rxx"; ϕ) = exp(-i ϕ XX)
    # We expose qiskit's θ and forward ϕ = θ/2 to the registry.
    "Rxx" => GateSpec("Rxx"; paramkeys = (:ϕ,), rescale = θ -> θ / 2),
    "Ryy" => GateSpec("Ryy"; paramkeys = (:ϕ,), rescale = θ -> θ / 2),
    "Rzz" => GateSpec("Rzz"; paramkeys = (:ϕ,), rescale = θ -> θ / 2),

    "CRx"    => GateSpec("CRx";    paramkeys = (:θ,)),
    "CRy"    => GateSpec("CRy";    paramkeys = (:θ,)),
    "CRz"    => GateSpec("CRz";    paramkeys = (:θ,)),
    "CPHASE" => GateSpec("CPHASE"; paramkeys = (:ϕ,)),

    "Rz+z+" => GateSpec("Rz+z+"; paramkeys = (:θ,)),

    # In-house parametric gates (definitions below)
    "Rxxyy"      => GateSpec("Rxxyy";      paramkeys = (:θ,)),
    "Rxxyyzz"    => GateSpec("Rxxyyzz";    paramkeys = (:θ,)),
    "xx_plus_yy" => GateSpec("xx_plus_yy"; paramkeys = (:θ, :β)),

    # Fermionic gates (spinless, Z2-parity sites; resolved by the fermionic op methods).
    # F_hop(θ)    = exp(-iθ (c†ᵢcⱼ + c†ⱼcᵢ))
    # F_nn(θ)     = exp(-iθ nᵢnⱼ)
    # F_hop_nn    = exp(-i (θ (c†ᵢcⱼ + c†ⱼcᵢ) + ϕ nᵢnⱼ))
    # F_phase(θ)  = exp(-iθ nᵢ)
    # F_pair(θ)   = exp(-iθ (c†ᵢc†ⱼ + cⱼcᵢ))
    # F_hop_up/dn(θ) = per-spin hopping on spinful (d = 4) sites; F_int(θ) = exp(-iθ n↑n↓)
    "F_hop"    => GateSpec("F_hop";    paramkeys = (:θ,)),
    "F_nn"     => GateSpec("F_nn";     paramkeys = (:θ,)),
    "F_hop_nn" => GateSpec("F_hop_nn"; paramkeys = (:θ, :ϕ)),
    "F_pair"   => GateSpec("F_pair";   paramkeys = (:θ,)),
    "F_phase"  => GateSpec("F_phase";  paramkeys = (:θ,)),
    "F_hop_up" => GateSpec("F_hop_up"; paramkeys = (:θ,)),
    "F_hop_dn" => GateSpec("F_hop_dn"; paramkeys = (:θ,)),
    "F_int"    => GateSpec("F_int";    paramkeys = (:θ,)),
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
    # Genuine name differences (qiskit name => our canonical name)
    m["cp"] = "CPHASE"
    m
end

# Resolve a gate name to its `GateSpec`, consulting `ALIASES` on miss. Returns
# `nothing` if the name is not registered under either.
function _resolve_gate(name::AbstractString)
    spec = get(GATES, name, nothing)
    spec !== nothing && return spec
    canon = get(ALIASES, name, nothing)
    canon === nothing ? nothing : GATES[canon]
end

# True if `s` is a string of Pauli letters (X/Y/Z, case-insensitive)
_ispaulistring(s::String) = all(c ∈ ('X', 'Y', 'Z', 'x', 'y', 'z') for c in s)

# Suggest canonical gate names close to `name` (case-insensitive edit distance).
# Returns up to `topk` keys ranked by distance, only those within `maxdist`.
function _gate_suggestions(name::AbstractString; topk::Int = 3, maxdist::Int = 2)
    lname = lowercase(name)
    scored = [(g, levenshtein(lname, lowercase(g))) for g in keys(GATES)]
    filter!(p -> last(p) <= maxdist, scored)
    sort!(scored; by = p -> (p[2], p[1]))
    return [first(p) for p in Iterators.take(scored, topk)]
end

# --- Circuit-tuple → ITensor -------------------------------------------------

# Vector of gates → vector of (ITensor, vertices)
function toitensor(circuit::Vector, g::NamedGraph, sinds::Dictionary)
    return [toitensor(gate, g, sinds) for gate in circuit]
end

# Single circuit tuple → (ITensor, vertices)
function toitensor(gate::Tuple, g::NamedGraph, siteinds::Dictionary)
    name = gate[1]
    verts = collect_vertices(gate[2], g)
    s_inds = [only(siteinds[v]) for v in verts]

    # Multi-letter Pauli-string sugar: "XYZ" → X⊗Y⊗Z applied componentwise.
    # Single-letter "X"/"Y"/"Z" goes through the registry below.
    if _ispaulistring(name) && length(name) > 1
        t = prod(op(string(c), sind) for (c, sind) in zip(name, s_inds))
        return t, verts
    end

    spec = _resolve_gate(name)
    if spec === nothing
        suggestions = _gate_suggestions(name)
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

    raw = spec.rescale(gate[3])
    pvals = raw isa Union{Tuple, AbstractVector} ? Tuple(raw) : (raw,)
    length(pvals) == length(spec.paramkeys) || throw(ArgumentError(
        "Gate \"$name\" expects $(length(spec.paramkeys)) parameter(s), got $(length(pvals))."
    ))
    kwargs = NamedTuple{spec.paramkeys}(pvals)
    return op(spec.opname, s_inds...; kwargs...), verts
end

# --- Public registration API ------------------------------------------------

"""
    register_gate!(name::String; opname = name, paramkeys = (), rescale = identity)

Register a custom gate `name` so it can be used in circuit-tuple form
`(name, vertices, parameter)` with `apply_gates`.

The matrix itself must be registered separately via `register_op!` under a name
matching `opname` (defaults to `name`). See "Custom Gates" in the gate
docs for a worked example.

Modifies the runtime gate registry. The registration lives only in the current
Julia session — to persist it across sessions, place the `register_gate!` call
in your script's startup, or in a downstream package's `__init__()`.

Built-in gates are locked: passing a built-in name throws `ArgumentError`.
Choose a different name for your custom gate, or — if you really need a new
matrix under an existing name — register your own matrix via `register_op!` directly.
Previously user-registered names may be overwritten freely.

# Arguments
- `name`: name used in circuit tuples.

# Keyword Arguments
- `opname`: the operator name looked up in the backend op registry. Defaults to `name`.
- `paramkeys`: tuple of keyword names accepted by the underlying `op`, e.g.
  `(:θ,)` for a single rotation angle, `(:θ, :β)` for a two-parameter gate.
  Empty (`()`) for non-parametric gates.
- `rescale`: applied to the user-supplied parameter(s) before forwarding. Use
  this if your `op` definition expects a different convention from your
  circuit-level parameter (e.g. half-angle conventions). For multi-parameter
  gates, `rescale` receives and returns a tuple/vector.
"""
function register_gate!(
        name::String;
        opname::String = name,
        paramkeys::Tuple = (),
        rescale = identity,
    )
    name in BUILTIN_GATES && throw(ArgumentError(
        "\"$name\" is a built-in gate and cannot be overwritten. " *
        "Choose a different name for your custom gate, or define your own " *
        "`register_op!` entry directly if you need to override the matrix."
    ))
    GATES[name] = GateSpec(opname, paramkeys, rescale)
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

# --- In-house gate definitions ----------------------------------------------
# The operator matrices themselves live in the KTensors registry (see
# `KTensors.register_op!` and OP1_REGISTRY/OP2_REGISTRY): Rxxyy, Rxxyyzz and the
# qiskit-convention xx_plus_yy are registered there alongside the standard gates.
