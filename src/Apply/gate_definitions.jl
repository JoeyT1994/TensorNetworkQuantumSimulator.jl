
# --- Gate registry -----------------------------------------------------------

# Internal dispatch record for a circuit-tuple gate name.
#
# - `opname`: the `OpName` string forwarded to `Ops.op`. Usually equal to the
#   user-facing key, but kept separate so a registry entry can rename if needed.
# - `paramkeys`: keyword names accepted by the underlying `op` definition, e.g.
#   `(:θ,)`, `(:ϕ,)`, or `(:θ, :β)`. Empty for fixed gates.
# - `rescale`: applied to the user-supplied parameter(s) before forwarding. Used
#   when our (qiskit) convention differs from the `Ops.op` convention. For
#   multi-parameter gates, `rescale` receives and returns a tuple/vector.
struct GateSpec
    opname::String
    paramkeys::Tuple{Vararg{Symbol}}
    rescale::Function
end
GateSpec(opname; paramkeys = (), rescale = identity) = GateSpec(opname, paramkeys, rescale)

# Registry of circuit-tuple gates. Adding a new gate is one entry here (plus an
# `Ops.op` method if upstream doesn't already provide one).
const GATES = Dict{String, GateSpec}(
    # Single-qubit fixed
    "X" => GateSpec("X"),
    "Y" => GateSpec("Y"),
    "Z" => GateSpec("Z"),
    "H" => GateSpec("H"),

    # Single-qubit parametric (qiskit and ITensors agree on convention)
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
    # ITensors: op("Rxx"; ϕ) = exp(-i ϕ XX)
    # We expose qiskit's θ and forward ϕ = θ/2 to ITensors.
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

# Already an ITensor: pass through
toitensor(gate::ITensor, sinds::Dictionary) = gate

# Single circuit tuple → (ITensor, vertices)
function toitensor(gate::Tuple, g::NamedGraph, siteinds::Dictionary)
    name = gate[1]
    verts = collect_vertices(gate[2], g)
    s_inds = [only(siteinds[v]) for v in verts]

    # Multi-letter Pauli-string sugar: "XYZ" → X⊗Y⊗Z applied componentwise.
    # Single-letter "X"/"Y"/"Z" goes through the registry below.
    if _ispaulistring(name) && length(name) > 1
        t = prod(Ops.op(string(c), sind) for (c, sind) in zip(name, s_inds))
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
        return Ops.op(spec.opname, s_inds...), verts
    end

    raw = spec.rescale(gate[3])
    pvals = raw isa Union{Tuple, AbstractVector} ? Tuple(raw) : (raw,)
    length(pvals) == length(spec.paramkeys) || throw(ArgumentError(
        "Gate \"$name\" expects $(length(spec.paramkeys)) parameter(s), got $(length(pvals))."
    ))
    kwargs = NamedTuple{spec.paramkeys}(pvals)
    return Ops.op(spec.opname, s_inds...; kwargs...), verts
end

# --- Public registration API ------------------------------------------------

"""
    register_gate!(name::String; opname = name, paramkeys = (), rescale = identity)

Register a custom gate `name` so it can be used in circuit-tuple form
`(name, vertices, parameter)` with `apply_gates`.

The matrix itself must be defined separately as an `Ops.op` method whose
`OpName` matches `opname` (defaults to `name`). See "Custom Gates" in the gate
docs for a worked example.

Modifies the runtime gate registry. The registration lives only in the current
Julia session — to persist it across sessions, place the `register_gate!` call
in your script's startup, or in a downstream package's `__init__()`.

Built-in gates are locked: passing a built-in name throws `ArgumentError`.
Choose a different name for your custom gate, or — if you really need a new
matrix under an existing name — define your own `Ops.op` method directly.
Previously user-registered names may be overwritten freely.

# Arguments
- `name`: name used in circuit tuples.

# Keyword Arguments
- `opname`: the `OpName` string forwarded to `Ops.op`. Defaults to `name`.
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
        "`Ops.op` method directly if you need to override the matrix."
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

"""
    Ops.op(::OpName"xx_plus_yy", ::SiteType"S=1/2"; θ::Number, β::Number)

Gate for rotation by XX+YY at a given angle with Rz rotations either side. Consistent with qiskit.
"""
function Ops.op(::OpName"xx_plus_yy", ::SiteType"S=1/2"; θ::Number, β::Number)
    return [
        [1 0 0 0];
        [0 cos(θ / 2) -im * sin(θ / 2) * exp(-im * β) 0]
        [0 -im * sin(θ / 2) * exp(im * β) cos(θ / 2) 0]
        [0 0 0 1]
    ]
end
Ops.op(o::OpName"xx_plus_yy", ::SiteType"Qubit"; θ::Number, β::Number) =
    Ops.op(o, SiteType("S=1/2"); θ, β)

"""
    Ops.op(::OpName"Rxxyy", ::SiteType"S=1/2"; θ::Number)

Gate for rotation by XXYY at a given angle.
"""
function Ops.op(::OpName"Rxxyy", ::SiteType"S=1/2"; θ = 1)
    # Built as one two-site matrix in the manifestly charge-conserving σ± form,
    # ½(XX + YY) = σ⁺σ⁻ + σ⁻σ⁺, rather than from single-site `op("X", s)` factors:
    # the gate conserves U(1) charge, but a standalone `X` does not, so the factored
    # construction has no symmetric representation even though the sum does.
    σp, σm = [0.0 1.0; 0.0 0.0], [0.0 0.0; 1.0 0.0]
    h = kron(σp, σm) + kron(σm, σp)
    return exp(-im * θ * h)
end
Ops.op(o::OpName"Rxxyy", ::SiteType"Qubit"; θ::Number) =
    Ops.op(o, SiteType("S=1/2"); θ)

"""
    Ops.op(::OpName"Rxxyyzz", ::SiteType"S=1/2"; θ::Number)

Gate for rotation by XXYYZZ at a given angle.
"""
function Ops.op(::OpName"Rxxyyzz", ::SiteType"S=1/2"; θ = 1)
    # One two-site matrix in the σ± form, for the same reason as `Rxxyy` above:
    # ½(XX + YY + ZZ) = σ⁺σ⁻ + σ⁻σ⁺ + ½ ZZ.
    σp, σm, σz = [0.0 1.0; 0.0 0.0], [0.0 0.0; 1.0 0.0], [1.0 0.0; 0.0 -1.0]
    h = kron(σp, σm) + kron(σm, σp) + 0.5 * kron(σz, σz)
    return exp(-im * θ * h)
end
Ops.op(o::OpName"Rxxyyzz", ::SiteType"Qubit"; θ::Number) =
    Ops.op(o, SiteType("S=1/2"); θ)
