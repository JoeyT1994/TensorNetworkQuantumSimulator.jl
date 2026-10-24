"""
    TensorNetworkState{V} <: AbstractTensorNetwork{V}

A tensor network state defined on a graph with vertices of type `V`. Wraps a `TensorNetwork` together with a dictionary of site indices (physical degrees of freedom) at each vertex.

# Fields
- `tensornetwork::TensorNetwork{V}`: The underlying tensor network.
- `siteinds::Dictionary{V, Vector{Index}}`: A dictionary mapping each vertex to its physical (site) indices.
"""
struct TensorNetworkState{V} <: AbstractTensorNetwork{V}
    tensornetwork::TensorNetwork{V}
    siteinds::Dictionary{V, Vector{Index}}
end

tensornetwork(tns::TensorNetworkState) = tns.tensornetwork
siteinds(tns::TensorNetworkState) = tns.siteinds
graph(tns::TensorNetworkState) = graph(tensornetwork(tns))
tensors(tns::TensorNetworkState) = tensors(tensornetwork(tns))

Base.copy(tns::TensorNetworkState) = TensorNetworkState(copy(tensornetwork(tns)), copy(siteinds(tns)))

TensorNetworkState(tn::TensorNetwork) = TensorNetworkState(tn, siteinds(tn))
TensorNetworkState(tensors::Dictionary, g::NamedGraph) = TensorNetworkState(TensorNetwork(tensors, g))
TensorNetworkState(tensors::Union{Dictionary, Vector{<:Tensor}}) = TensorNetworkState(TensorNetwork(tensors))

Base.getindex(tns::TensorNetworkState, args...; kwargs...) = getindex(tensornetwork(tns), args...; kwargs...)

siteinds(tns::TensorNetworkState, v) = siteinds(tns)[v]

function Base.setindex!(tns::TensorNetworkState, value::Tensor, v)
    setindex!(tensornetwork(tns), value, v)
    sinds = siteinds(tns)
    for vn in vcat(neighbors(tns, v), [v])
        set!(sinds, vn, uniqueinds(tns, vn))
    end
    return tns
end

function norm_factors(tns::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I")
    factors = Tensor[]
    for v in verts
        sinds = siteinds(tns, v)
        tnv = tns[v]
        tnv_dag = dag(prime(tnv))
        if op_strings(v) == "ρ" || isempty(sinds)
            append!(factors, Tensor[tnv, tnv_dag])
        elseif op_strings(v) == "I"
            tnv_dag = replaceinds(tnv_dag, prime.(sinds), sinds)
            append!(factors, Tensor[tnv, tnv_dag])
        else
            o = adapt_like(tnv, op(op_strings(v), only(sinds)))
            append!(factors, Tensor[tnv, tnv_dag, o])
        end
    end
    return factors
end

norm_factors(tns::TensorNetworkState, v; kwargs...) = norm_factors(tns, [v]; kwargs...)
bp_factors(tns::TensorNetworkState, v) = norm_factors(tns, v)

message_legs(::TensorNetworkState, l) = [collect(l); dag.(prime.(l))]

"""
    random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary; bond_dimension::Integer = 1)

Generate a random `TensorNetworkState` on graph `g` with local state indices given by `siteinds`.

# Arguments
- `eltype`: The number type of the tensor elements (e.g. `Float64`, `ComplexF32`). Default is `Float64`.
- `g::AbstractGraph`: The underlying graph of the tensor network.
- `siteinds::Dictionary`: A dictionary mapping vertices to their site indices. Defaults to spin-1/2.

# Keyword Arguments
- `bond_dimension::Integer`: The bond dimension of the virtual indices connecting neighbouring tensors (default is `1`).

# Returns
- A `TensorNetworkState` representing the random tensor network state.
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g); bond_dimension::Integer = 1)
    vs = collect(vertices(g))
    l = edge_indices(g, bond_dimension)
    tensors = Dictionary{vertextype(g), Tensor}()
    for v in vs
        is = vcat(siteinds[v], [l[NamedEdge(v => vn)] for vn in neighbors(g, v)])
        set!(tensors, v, random_tensor(eltype, is))
    end
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
- `siteinds::Dictionary`: A dictionary mapping vertices to their site indices. Defaults to spin-1/2.

# Returns
- A `TensorNetworkState` representing the constructed tensor network state.
"""
function tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g))
    vs = collect(vertices(g))
    tensors = Dictionary{vertextype(g), Tensor}()
    for v in vs
        tnv = f(v)
        if tnv isa String
            set!(tensors, v, adapt(eltype)(state(tnv, only(siteinds[v]))))
        elseif tnv isa Vector{<:Number}
            set!(tensors, v, adapt(eltype)(from_array(tnv, [only(siteinds[v])])))
        else
            error("Unrecognized local state constructor. Currently supported: Strings and Vectors.")
        end
    end

    # product state: a trivial bond on every edge
    for e in edges(g)
        l = new_index(1)
        tensors[src(e)] *= onehot(eltype, l => 1)
        tensors[dst(e)] *= onehot(eltype, l => 1)
    end
    return TensorNetworkState(TensorNetwork(tensors, g), siteinds)
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

function NamedGraphs.vertices(t::Tensor, tns::TensorNetworkState)
    t_inds = inds(t)
    return filter(v -> !isempty(intersect(t_inds, siteinds(tns, v))), collect(vertices(tns)))
end
