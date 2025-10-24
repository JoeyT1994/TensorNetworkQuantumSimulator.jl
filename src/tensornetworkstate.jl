using ITensors: random_itensor

struct TensorNetworkState{V} <: AbstractITensorNetwork{V}
    tensornetwork::ITensorNetwork{V}
    siteinds::Dictionary
end

tensornetwork(tns::TensorNetworkState) = tns.tensornetwork
siteinds(tns::TensorNetworkState) = tns.siteinds

Base.copy(tns::TensorNetworkState) = TensorNetworkState(copy(tensornetwork(tns)), copy(siteinds(tns)))

siteinds(tn::ITensorNetwork) = Dictionary(collect(vertices(tn)), [uniqueinds(tn, v) for v in collect(vertices(tn))])
TensorNetworkState(tn::ITensorNetwork) = TensorNetworkState(tn, siteinds(tn))
TensorNetworkState(vertices::Vector, tensors::Vector{<:ITensor}) = TensorNetworkState(ITensorNetwork(vertices, tensors))

#Forward onto the itn
for f in [
    :(ITensorNetworks.underlying_graph),
    :(ITensorNetworks.data_graph_type),
    :(ITensorNetworks.data_graph),
    :(ITensors.datatype),
    :(ITensors.NDTensors.scalartype),
    :(ITensorNetworks.setindex_preserve_graph!)
]
@eval begin
    function $f(tns::TensorNetworkState, args...; kwargs...)
        return $f(tensornetwork(tns), args...; kwargs...)
    end
end
end

#Forward onto the underlying_graph
for f in [
    :(NamedGraphs.edgeinduced_subgraphs_no_leaves)
]
@eval begin
    function $f(tns::TensorNetworkState, args...; kwargs...)
        return $f(ITensorNetworks.underlying_graph(tensornetwork(tns)), args...; kwargs...)
    end
end
end

siteinds(tns::TensorNetworkState, v) = siteinds(tns)[v]

function ITensorNetworks.data_graph_type(TNS::Type{<:TensorNetworkState})
    return ITensorNetworks.data_graph_type(fieldtype(TNS, :tensornetwork))
end

function ITensorNetworks.uniqueinds(tns::TensorNetworkState, v)
    is = ITensorNetworks.uniqueinds(tensornetwork(tns), v)
    is isa Vector{<:Index} && return is
    return Index[i for i in is]
end

ITensorNetworks.uniqueinds(tns::TensorNetworkState, edge::AbstractEdge) = ITensorNetworks.uniqueinds(tensornetwork(tns), edge)

function Base.setindex!(tns::TensorNetworkState, value, v)
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
    linds = linkinds(tns, edge)
    return adapt(datatype(tns))(denseblocks(delta(vcat(linds, prime(dag(linds))))))
end

"""
Generate a random TensorNetworkState on a `NamedGraph`` `g` with local state indices given by the dictionary `siteinds` which maps vertices to ITensor indices.
The bond dimension of the virtual indices connecting neighboring tensors can be set with the `bond_dimension` keyword argument (default is 1).
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary; bond_dimension::Int = 1)
    vs = collect(vertices(g))
    l = Dict(e => Index(bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tn = ITensorNetwork(g)
    for v in vs
       is = vcat(siteinds[v], [l[NamedEdge(v => vn)] for vn in neighbors(g,v)])
       tn[v] = random_itensor(eltype, is)
    end
    return TensorNetworkState(tn, siteinds)
end

"""
    random_tensornetworkstate(eltype, g::AbstractGraph, sitetype::String, d::Int = site_dimension(sitetype); bond_dimension::Int = 1)
    Generate a random TensorNetworkState on graph `g` with local state indices generated from the `sitetype` string (e.g. "S=1/2", "Pauli") and the local dimension `d` (default is 2 for "S=1/2", 4 for Pauli etc).
    The bond dimension of the virtual indices connecting neighboring tensors can be set with the `bond_dimension` keyword argument (default is 1).
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, sitetype::String, d::Int = site_dimension(sitetype); bond_dimension::Int = 1)
    return random_tensornetworkstate(eltype, g, siteinds(g, sitetype, d); bond_dimension)
end

"""
    tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary)
    Construct a TensorNetworkState on graph `g` where the function `f` maps vertices to local states. 
    The local states can be given as strings (e.g. "↑", "↓", "0", "1", "I", "X", "Y", "Z") or as vectors of numbers (e.g. [1,0], [0,1], [1/sqrt(2), 1/sqrt(2)]).
    The local state indices are given by the dictionary `siteinds` which maps vertices to ITensor indices.
    The number type of the tensor elements can be set with the `eltype` argument (default is Float64).
"""
function tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary)
    vs = collect(vertices(g))
    tn = ITensorNetwork(g)
    for v in vs
        tnv = f(v)
        if tnv isa String
            tn[v] = adapt(eltype)(ITensors.state(f(v), only(siteinds[v])))
        elseif tnv isa Vector{<:Number}
            tn[v] = adapt(eltype)(ITensors.ITensor(f(v), only(siteinds[v])))
        else
            error("Unrecognized local state constructor. Currently supported: Strings and Vectors.")
        end
    end

    l = Dict(e => Index(1) for e in edges(g))
    for e in edges(g)
        tn[src(e)] *= onehot(eltype, l[e] => 1)
        tn[dst(e)] *= onehot(eltype, l[e] => 1)
    end
    return TensorNetworkState(tn, siteinds)
end

"""
    tensornetworkstate(eltype, f::Function, g::AbstractGraph, sitetype::String, d::Int = site_dimension(sitetype))
    Construct a TensorNetworkState on graph `g` where the function `f` maps vertices to local states.
    The local states can be given as strings (e.g. "↑", "↓", "0", "1", "I", "X", "Y", "Z") or as vectors of numbers (e.g. [1,0], [0,1], [1/sqrt(2), 1/sqrt(2)]).
    The local state indices are generated from the `sitetype` string (e.g. "S=1/2", "Pauli") and the local dimension `d` (default is 2 for "S=1/2", 4 for Pauli etc).
    The number type of the tensor elements can be set with the `eltype` argument (default is Float64).
"""
function tensornetworkstate(eltype, f::Function, g::AbstractGraph, sitetype::String, d::Int = site_dimension(sitetype))
    return tensornetworkstate(eltype, f, g, siteinds(g, sitetype, d))
end

function random_tensornetworkstate(g::AbstractGraph, args...; kwargs...)
    return random_tensornetworkstate(Float64, g, args...; kwargs...)
end

function tensornetworkstate(f::Function, args...)
    return tensornetworkstate(Float64, f, args...)
end
    