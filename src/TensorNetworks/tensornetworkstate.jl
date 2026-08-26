
"""
    TensorNetworkState{V} <: AbstractTensorNetwork{V}

A tensor network state defined on a graph with vertices of type `V`. Wraps a `TensorNetwork` together with a dictionary of site indices (physical degrees of freedom) at each vertex.

# Fields
- `tensornetwork::TensorNetwork{V}`: The underlying tensor network.
- `siteinds::Dictionary{V, Vector{<:Index}}`: A dictionary mapping each vertex to its physical (site) indices.
"""
struct TensorNetworkState{V, TN <: TensorNetwork{V}, SI <: Dictionary} <: AbstractTensorNetwork{V}
    tensornetwork::TN
    siteinds::SI
end

tensornetwork(tns::TensorNetworkState) = tns.tensornetwork
siteinds(tns::TensorNetworkState) = tns.siteinds
graph(tns::TensorNetworkState) = graph(tensornetwork(tns))
tensors(tns::TensorNetworkState) = tensors(tensornetwork(tns))

Base.copy(tns::TensorNetworkState) = TensorNetworkState(copy(tensornetwork(tns)), copy(siteinds(tns)))

TensorNetworkState(tn::TensorNetwork) = TensorNetworkState(tn, siteinds(tn))
TensorNetworkState(tensors::Dictionary, g::NamedGraph) = TensorNetworkState(TensorNetwork(tensors, g))
TensorNetworkState(tensors::Union{Dictionary, Vector}) = TensorNetworkState(TensorNetwork(tensors))

#Forward onto the tn
for f in [
        :(Base.getindex),
    ]
    @eval begin
        function $f(tns::TensorNetworkState, args...; kwargs...)
            return $f(tensornetwork(tns), args...; kwargs...)
        end
    end
end

siteinds(tns::TensorNetworkState, v) = siteinds(tns)[v]

function Base.setindex!(tns::TensorNetworkState, value, v)
    setindex!(tensornetwork(tns), value, v)
    sinds = siteinds(tns)
    for vn in vcat(neighbors(tns, v), [v])
        set!(sinds, vn, uniqueinds(tns, vn))
    end
    return tns
end

function norm_factors(tns::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I")
    factors = tensortype(tns)[]
    for v in verts
        sinds = siteinds(tns, v)
        tnv = tns[v]
        tnv_dag = dag(prime(tnv))
        if op_strings(v) == "ρ" || isempty(sinds)
            append!(factors, [tnv, tnv_dag])
        elseif op_strings(v) == "I"
            tnv_dag = replaceinds(tnv_dag, prime.(sinds), sinds)
            append!(factors, [tnv, tnv_dag])
        else
            op_tensor = adapt_like(tnv, op(op_strings(v), only(sinds)))
            append!(factors, [tnv, tnv_dag, op_tensor])
        end
    end
    return factors
end

norm_factors(tns::TensorNetworkState, v; kwargs...) = norm_factors(tns, [v]; kwargs...)
bp_factors(tns::TensorNetworkState, v) = norm_factors(tns, v)

#Fused KTensors fast path for the double-layer BP message update (see KTensors module).
#Falls through to the generic contraction path when the message structure doesn't match
#(e.g. boundary-MPS messages with link indices) or the site indices aren't KIndex.
function norm_message_kernel(tns::TensorNetworkState, v, incoming_ms::Vector{<:KTensor}; normalize)
    ψ = tns[v]
    ψ isa KTensor || return nothing
    sinds = siteinds(tns, v)
    all(i -> i isa KIndex, sinds) || return nothing
    return KTensors.fused_norm_message(ψ, collect(KIndex, sinds), incoming_ms; normalize)
end

#Fused KTensors fast path for BP region scalars (expectation-value numerators/denominators
#and vertex scalars): one- and two-vertex regions close through the same fused kernel — a
#two-vertex region is "message from v1 with its operator inserted" followed by a full
#closure at v2. Larger Steiner regions and non-standard structures fall back.
function norm_scalar_kernel(tns::TensorNetworkState, vs::Vector, incoming_ms::Vector{<:KTensor}; op_strings::Function)
    1 <= length(vs) <= 2 || return nothing
    ψs, sindss, ops = KTensor[], Vector{KIndex}[], Union{Nothing, KTensor}[]
    for v in vs
        ψ = tns[v]
        ψ isa KTensor || return nothing
        sinds = siteinds(tns, v)
        all(i -> i isa KIndex, sinds) || return nothing
        str = op_strings(v)
        if str == "I"
            push!(ops, nothing)
        elseif str == "ρ" || length(sinds) != 1
            return nothing
        else
            push!(ops, adapt_like(ψ, op(str, only(sinds))))
        end
        push!(ψs, ψ)
        push!(sindss, collect(KIndex, sinds))
    end

    if length(vs) == 1
        c = KTensors.fused_norm_closure(ψs[1], sindss[1], incoming_ms; op = ops[1])
        (c === nothing || !isempty(inds(c))) && return nothing
        return scalar(c)
    end

    #Partition the region's incoming messages by which vertex tensor they attach to
    ms1, ms2 = KTensor[], KTensor[]
    for m in incoming_ms
        ket_legs = filter(i -> plev(i) == 0, inds(m))
        if all(i -> i ∈ inds(ψs[1]), ket_legs)
            push!(ms1, m)
        elseif all(i -> i ∈ inds(ψs[2]), ket_legs)
            push!(ms2, m)
        else
            return nothing
        end
    end
    T1 = KTensors.fused_norm_closure(ψs[1], sindss[1], ms1; op = ops[1])
    T1 === nothing && return nothing
    c = KTensors.fused_norm_closure(ψs[2], sindss[2], vcat(ms2, [T1]); op = ops[2])
    (c === nothing || !isempty(inds(c))) && return nothing
    return scalar(c)
end

function default_message(tns::TensorNetworkState, edge::AbstractEdge)
    linds = virtualinds(tns, edge)
    return adapt_like(tns, denseblocks(delta(vcat(linds, prime(dag(linds))))))
end

"""
    random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary; bond_dimension::Integer = 1)

Generate a random `TensorNetworkState` on graph `g` with local state indices given by `siteinds`.

# Arguments
- `eltype`: The number type of the tensor elements (e.g. `Float64`, `ComplexF32`). Default is `Float64`.
- `g::AbstractGraph`: The underlying graph of the tensor network.
- `siteinds::Dictionary`: A dictionary mapping vertices to ITensor indices representing the local states. Defaults to spin-1/2.

# Keyword Arguments
- `bond_dimension::Integer`: The bond dimension of the virtual indices connecting neighbouring tensors (default is `1`).

# Returns
- A `TensorNetworkState` representing the random tensor network state.
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g); bond_dimension::Integer = 1)
    vs = collect(vertices(g))
    l = Dict(e => new_index(only(siteinds[src(e)]), bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tensors = Dictionary{vertextype(g), Any}()
    for v in vs
        is = vcat(siteinds[v], [l[NamedEdge(v => vn)] for vn in neighbors(g, v)])
        set!(tensors, v, random_itensor(eltype, is))
    end
    tensors = Dictionary(vs, identity.(collect(tensors)))
    return TensorNetworkState(TensorNetwork(tensors, g), siteinds)
end

"""
    random_tensornetworkstate(eltype, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype); bond_dimension::Integer = 1)

Generate a random `TensorNetworkState` on graph `g` with local state indices generated from the `sitetype` string (e.g. `"S=1/2"`, `"S=1"`) and the local dimension `d`.

# Arguments
- `eltype`: The number type of the tensor elements (e.g. `Float64`, `ComplexF32`). Default is `Float64`.
- `g::AbstractGraph`: The underlying graph of the tensor network.
- `sitetype::String`: A string representing the type of local site (e.g. `"S=1/2"`, `"S=1"`).
- `d::Integer`: The local dimension of the site (default is determined by `sitetype`).

# Keyword Arguments
- `bond_dimension::Integer`: The bond dimension of the virtual indices connecting neighboring tensors (default is `1`).

# Returns
- A `TensorNetworkState` representing the random tensor network state.
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype); bond_dimension::Integer = 1)
    return random_tensornetworkstate(eltype, g, siteinds(sitetype, g, d); bond_dimension)
end

"""
    tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary)

Construct a `TensorNetworkState` on graph `g` where the function `f` maps vertices to local states.
The local states can be given as strings (e.g. `"↑"`, `"↓"`, `"0"`, `"1"`) or as vectors of numbers (e.g. `[1,0]`, `[0,1]`).

# Arguments
- `eltype`: The number type of the tensor elements (e.g. `Float64`, `ComplexF32`). Default is `Float64`.
- `f::Function`: A function mapping vertices of the graph to local states.
- `g::AbstractGraph`: The underlying graph of the tensor network.
- `siteinds::Dictionary`: A dictionary mapping vertices to ITensor indices representing the local states. Defaults to spin-1/2.

# Returns
- A `TensorNetworkState` representing the constructed tensor network state.
"""
function tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g))
    vs = collect(vertices(g))
    tensors = Dictionary{vertextype(g), Any}()
    for v in vs
        tnv = f(v)
        if tnv isa String
            set!(tensors, v, adapt(eltype)(state(f(v), only(siteinds[v]))))
        elseif tnv isa Vector{<:Number}
            set!(tensors, v, adapt(eltype)(from_array(f(v), only(siteinds[v]))))
        else
            error("Unrecognized local state constructor. Currently supported: Strings and Vectors.")
        end
    end

    l = Dict(e => new_index(only(siteinds[src(e)]), 1) for e in edges(g))
    for e in edges(g)
        tensors[src(e)] *= onehot(eltype, l[e] => 1)
        tensors[dst(e)] *= onehot(eltype, l[e] => 1)
    end
    tensors = Dictionary(vs, identity.(collect(tensors)))
    return TensorNetworkState(tensors, g)
end

"""
    tensornetworkstate(eltype, f::Function, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype))

Construct a `TensorNetworkState` on graph `g` where the function `f` maps vertices to local states.
The local states can be given as strings (e.g. `"↑"`, `"↓"`, `"0"`, `"1"`) or as vectors of numbers (e.g. `[1,0]`, `[0,1]`).

# Arguments
- `eltype`: The number type of the tensor elements (e.g. `Float64`, `ComplexF32`). Default is `Float64`.
- `f::Function`: A function mapping vertices of the graph to local states.
- `g::AbstractGraph`: The underlying graph of the tensor network.
- `sitetype::String`: A string representing the type of local site (e.g. `"S=1/2"`, `"S=1"`).
- `d::Integer`: The local dimension of the site (default is determined by `sitetype`).

# Returns
- A `TensorNetworkState` representing the constructed tensor network state.
"""
function tensornetworkstate(eltype, f::Function, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype))
    return tensornetworkstate(eltype, f, g, siteinds(sitetype, g, d))
end

function random_tensornetworkstate(g::AbstractGraph, args...; kwargs...)
    return random_tensornetworkstate(Float64, g, args...; kwargs...)
end

function tensornetworkstate(f::Function, args...)
    return tensornetworkstate(Float64, f, args...)
end

function NamedGraphs.vertices(t, tns::TensorNetworkState)
    t_inds = inds(t)
    return filter(v -> !isempty(intersect(t_inds, siteinds(tns, v))), collect(vertices(tns)))
end
