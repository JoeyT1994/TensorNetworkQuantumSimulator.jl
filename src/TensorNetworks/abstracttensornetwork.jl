using Graphs: Graphs, has_vertex
using ITensors: ITensors
using ITensors.NDTensors: NDTensors
using NamedGraphs: NamedGraphs
using Adapt

abstract type AbstractTensorNetwork{V} <: AbstractNamedGraph{V} end

graph(tn::AbstractTensorNetwork) = not_implemented()
tensors(tn::AbstractTensorNetwork) = not_implemented()
NamedGraphs.rem_vertex!(tn::AbstractTensorNetwork, v) = not_implemented()
add_tensor!(tn::AbstractTensorNetwork, tensor::ITensor, v) = not_implemented()

Graphs.is_directed(::Type{<:AbstractTensorNetwork}) = false

NamedGraphs.vertex_positions(tn::AbstractTensorNetwork) = NamedGraphs.vertex_positions(graph(tn))
NamedGraphs.ordered_vertices(tn::AbstractTensorNetwork) = NamedGraphs.ordered_vertices(graph(tn))
NamedGraphs.position_graph(tn::AbstractTensorNetwork) = NamedGraphs.position_graph(graph(tn))
NamedGraphs.vertices(tn::AbstractTensorNetwork) = NamedGraphs.vertices(graph(tn))
NamedGraphs.edges(tn::AbstractTensorNetwork) = NamedGraphs.edges(graph(tn))
NamedGraphs.edgetype(tn::AbstractTensorNetwork) = NamedGraphs.edgetype(graph(tn))
NamedGraphs.vertextype(tn::AbstractTensorNetwork) = NamedGraphs.vertextype(graph(tn))
NamedGraphs.steiner_tree(tn::AbstractTensorNetwork, vs) = NamedGraphs.steiner_tree(graph(tn), vs)

virtualinds(tn::AbstractTensorNetwork, e::NamedEdge) = ITensors.commoninds(tn[src(e)], tn[dst(e)])

function maxvirtualdim(tn::AbstractTensorNetwork)
    return maximum(maximum.([dim.(virtualinds(tn, e)) for e in edges(tn)]))
end

function ITensors.uniqueinds(tn::AbstractTensorNetwork, v)
    tv_inds = Index[i for i in inds(tn[v])]
    vns = neighbors(tn, v)
    isempty(vns) && return tv_inds
    neighbor_inds = reduce(vcat, [Index[i for i in inds(tn[vn])] for vn in vns])
    is = setdiff(tv_inds, neighbor_inds)
    return is
end

function setindex_preserve!(tn::AbstractTensorNetwork, value::ITensor, vertex)
    tensors(tn)[vertex] = value
    return tn
end

function Base.setindex!(tn::AbstractTensorNetwork, value::ITensor, vertex)
    !has_vertex(graph(tn), vertex) && error("Vertex not in tensor network")
    add_tensor!(tn, value, vertex)
    return tn
end

function NDTensors.scalartype(tn::AbstractTensorNetwork)
    return mapreduce(v -> scalartype(tn[v]), promote_type, vertices(tn))
end

function ITensors.datatype(tn::AbstractTensorNetwork)
    return mapreduce(v -> ITensors.datatype(tn[v]), promote_type, vertices(tn))
end

function map_tensors(f::Function, tn::AbstractTensorNetwork)
    tn = copy(tn)
    for v in vertices(tn)
        tn[v] = f(tn[v])
    end
    return tn
end

function Adapt.adapt_structure(to, tn::AbstractTensorNetwork)
    return map_tensors(x -> adapt(to)(x), tn)
end
