using Dictionaries: Dictionary
using Graphs: Graphs
using ITensors: ITensors, ITensor
using NamedGraphs: NamedGraphs, add_edge!, incident_edges
using NamedGraphs.GraphsExtensions: rem_edges!
using Adapt

#TODO: Make this show() nicely.
struct TensorNetwork{V} <: AbstractTensorNetwork{V}
    tensors::Dictionary{V, ITensor}
    graph::NamedGraph{V}
end

graph(tn::TensorNetwork) = tn.graph
tensors(tn::TensorNetwork) = tn.tensors

Base.getindex(tn::TensorNetwork, v) = getindex(tensors(tn), v)

function TensorNetwork(tensors::Dictionary)
    g = NamedGraph(keys(tensors))
    vs = collect(vertices(g))
    for (i, v) in enumerate(vs)
        for vp in vs[(i + 1):length(vs)]
            if !isempty(commoninds(tensors[v], tensors[vp]))
                add_edge!(g, NamedEdge(v => vp))
            end
        end
    end
    return TensorNetwork(tensors, g)
end

Base.copy(tn::TensorNetwork) = TensorNetwork(copy(tensors(tn)), copy(graph(tn)))

function TensorNetwork(tensors::Vector{<:ITensor})
    return TensorNetwork(Dictionary([i for i in 1:length(tensors)], tensors))
end

function NamedGraphs.rem_vertex!(tn::TensorNetwork, v)
    NamedGraphs.rem_vertex!(graph(tn), v)
    delete!(tensors(tn), v)
    return tn
end

function add_tensor!(tn::TensorNetwork, tensor::ITensor, v)
    vs = collect(vertices(tn))
    g = graph(tn)
    if !has_vertex(g, v)
        add_vertex!(g, v)
    else
        rem_edges!(g, incident_edges(g, v; dir = :in))
    end
    ts = tensors(tn)
    set!(ts, v, tensor)
    for vp in vs
        if !isempty(commoninds(ts[v], ts[vp])) && v != vp
            add_edge!(g, NamedEdge(v => vp))
        end
    end
    return tn
end

function default_message(tn::TensorNetwork, edge::NamedEdge)
    return adapt(datatype(tn))(denseblocks(delta(virtualinds(tn, edge))))
end

function bp_factors(tn::TensorNetwork, vertex)
    return ITensor[tn[vertex]]
end

function random_tensornetwork(eltype, g::AbstractGraph; bond_dimension::Integer = 1)
    vs = collect(vertices(g))
    l = Dict(e => Index(bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vs
        is = [l[NamedEdge(v => vn)] for vn in neighbors(g, v)]
        set!(tensors, v, random_itensor(eltype, is))
    end
    return TensorNetwork(tensors, g)
end

random_tensornetwork(g::AbstractGraph; kwargs...) = random_tensornetwork(Float64, g; kwargs...)

function siteinds(tn::TensorNetwork)
    s = Dictionary{vertextype(tn), Vector{<:Index}}()
    for v in vertices(tn)
        is = uniqueinds(tn, v)
        isempty(is) ? set!(s, v, Index[]) : set!(s, v, is)
    end
    return s
end
