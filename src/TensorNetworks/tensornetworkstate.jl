using ITensors: random_itensor

#TODO: Make this show() nicely.
struct TensorNetworkState{V} <: AbstractTensorNetwork{V}
    tensornetwork::TensorNetwork{V}
    siteinds::Dictionary{V, Vector{<:Index}}
end

tensornetwork(tns::TensorNetworkState) = tns.tensornetwork
siteinds(tns::TensorNetworkState) = tns.siteinds
graph(tns::TensorNetworkState) = graph(tensornetwork(tns))
tensors(tns::TensorNetworkState) = tensors(tensornetwork(tns))

Base.copy(tns::TensorNetworkState) = TensorNetworkState(copy(tensornetwork(tns)), copy(siteinds(tns)))

TensorNetworkState(tn::TensorNetwork) = TensorNetworkState(tn, siteinds(tn))
TensorNetworkState(tensors::Dictionary, g::NamedGraph) = TensorNetworkState(TensorNetwork(tensors, g))
TensorNetworkState(tensors::Union{Dictionary, Vector{<:ITensor}}) = TensorNetworkState(TensorNetwork(tensors))

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

function Base.setindex!(tns::TensorNetworkState, value::ITensor, v)
    setindex!(tensornetwork(tns), value, v)
    sinds = siteinds(tns)
    for vn in vcat(neighbors(tns, v), [v])
        set!(sinds, vn, uniqueinds(tns, vn))
    end
    return tns
end

function norm_factors(tns::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I")
    factors = ITensor[]
    for v in verts
        sinds = siteinds(tns, v)
        tnv = tns[v]
        tnv_dag = dag(prime(tnv))
        if op_strings(v) == "I"
            tnv_dag = replaceinds(tnv_dag, prime.(sinds), sinds)
            append!(factors, ITensor[tnv, tnv_dag])
        else
            op = adapt(datatype(tnv))(ITensors.op(op_strings(v), only(sinds)))
            append!(factors, ITensor[tnv, tnv_dag, op])
        end
    end
    return factors
end

norm_factors(tns::TensorNetworkState, v) = norm_factors(tns, [v])
bp_factors(tns::TensorNetworkState, v) = norm_factors(tns, v)

function default_message(tns::TensorNetworkState, edge::AbstractEdge)
    linds = virtualinds(tns, edge)
    return adapt(datatype(tns))(denseblocks(delta(vcat(linds, prime(dag(linds))))))
end

#TODO: Default to spin 1/2
"""
    random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary; bond_dimension::Integer = 1)
    Generate a random TensorNetworkState on graph `g` with local state indices given by the dictionary `siteinds`.

    Arguments:
    - `eltype`: (Optional) The number type of the tensor elements (e.g. Float64, ComplexF32). Default is Float64.
    - `g::AbstractGraph`: The underlying graph of the tensor network.
    - `siteinds::Dictionary`: A dictionary mapping vertices to ITensor indices representing the local states. Defaults to spin 1/2.
    - `bond_dimension::Integer`: The bond dimension of the virtual indices connecting neighboring tensors (default is 1).

    Returns:
    - A `TensorNetworkState` object representing the random tensor network state.
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g); bond_dimension::Integer = 1)
    vs = collect(vertices(g))
    l = Dict(e => Index(bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vs
        is = vcat(siteinds[v], [l[NamedEdge(v => vn)] for vn in neighbors(g, v)])
        set!(tensors, v, random_itensor(eltype, is))
    end
    return TensorNetworkState(TensorNetwork(tensors, g), siteinds)
end

"""
    random_tensornetworkstate(eltype, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype); bond_dimension::Integer = 1)
    Generate a random TensorNetworkState on graph `g` with local state indices generated from the `sitetype` string (e.g. "S=1/2", "Pauli") and the local dimension `d` (default is 2 for "S=1/2", 4 for Pauli etc).

    Arguments:
    - `eltype`: (Optional) The number type of the tensor elements (e.g. Float64, ComplexF32). Default is Float64.
    - `g::AbstractGraph`: The underlying graph of the tensor network.
    - `sitetype::String`: A string representing the type of local site (e.g. "S=1/2", "Pauli").
    - `d::Integer`: The local dimension of the site (default is determined by `sitetype`).
    - `bond_dimension::Integer`: The bond dimension of the virtual indices connecting neighboring tensors (default is 1).
    Returns:
    - A `TensorNetworkState` object representing the random tensor network state.
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype); bond_dimension::Integer = 1)
    return random_tensornetworkstate(eltype, g, siteinds(sitetype, g, d); bond_dimension)
end

"""
    tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary)
    Construct a TensorNetworkState on graph `g` where the function `f` maps vertices to local states.
    The local states can be given as strings (e.g. "↑", "↓", "0", "1", "I", "X", "Y", "Z") or as vectors of numbers (e.g. [1,0], [0,1], [1/sqrt(2), 1/sqrt(2)]).

    Arguments:
    - `eltype`: (Optional) The number type of the tensor elements (e.g. Float64, ComplexF32). Default is Float64.
    - `f::Function`: A function mapping vertices of the graph to local states.
    - `g::AbstractGraph`: The underlying graph of the tensor network.
    - `siteinds::Dictionary`: A dictionary mapping vertices to ITensor indices representing the local states. Defaults to spin 1/2.
    Returns:    
    - A `TensorNetworkState` object representing the constructed tensor network state.
"""
function tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g))
    vs = collect(vertices(g))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vs
        tnv = f(v)
        if tnv isa String
            set!(tensors, v, adapt(eltype)(ITensors.state(f(v), only(siteinds[v]))))
        elseif tnv isa Vector{<:Number}
            set!(tensors, v, adapt(eltype)(ITensors.ITensor(f(v), only(siteinds[v]))))
        else
            error("Unrecognized local state constructor. Currently supported: Strings and Vectors.")
        end
    end

    l = Dict(e => Index(1) for e in edges(g))
    for e in edges(g)
        tensors[src(e)] *= onehot(eltype, l[e] => 1)
        tensors[dst(e)] *= onehot(eltype, l[e] => 1)
    end
    return TensorNetworkState(tensors, g)
end

"""
    tensornetworkstate(eltype, f::Function, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype))
    Construct a TensorNetworkState on graph `g` where the function `f` maps vertices to local states.
    The local states can be given as strings (e.g. "↑", "↓", "0", "1", "I", "X", "Y", "Z") or as vectors of numbers (e.g. [1,0], [0,1], [1/sqrt(2), 1/sqrt(2)]).

    Arguments:
    - `eltype`: (Optional) The number type of the tensor elements (e.g. Float64, ComplexF32). Default is Float64.
    - `f::Function`: A function mapping vertices of the graph to local states.
    - `g::AbstractGraph`: The underlying graph of the tensor network.
    - `sitetype::String`: A string representing the type of local site (e.g. "S=1/2", "Pauli").
    - `d::Integer`: The local dimension of the site (default is determined by `sitetype`).

    Returns:
    - A `TensorNetworkState` object representing the constructed tensor network state.
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

function NamedGraphs.vertices(t::ITensor, tns::TensorNetworkState)
    t_inds = inds(t)
    return filter(v -> !isempty(intersect(t_inds, siteinds(tns, v))), collect(vertices(tns)))
end
