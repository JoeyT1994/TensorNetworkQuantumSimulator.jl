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

function map_tensors!(f::Function, tn::AbstractTensorNetwork)
    for v in vertices(tn)
        setindex_preserve!(tn, f(tn[v]), v)
    end
    return tn
end

function map_tensors(f::Function, tn::AbstractTensorNetwork)
    tn = copy(tn)
    return map_tensors!(f, tn)
end

function Adapt.adapt_structure(to, tn::AbstractTensorNetwork)
    return map_tensors(x -> adapt(to)(x), tn)
end

function insert_virtualinds!(tn::AbstractTensorNetwork)
    dtype = datatype(tn)
    for e in edges(tn)
        if isempty(ITensors.commoninds(tn[src(e)], tn[dst(e)]))
            l = Index(1)
            p = adapt(dtype)(onehot(l => 1))
            TN.setindex_preserve!(tn, tn[src(e)] * p, src(e))
            TN.setindex_preserve!(tn, tn[dst(e)] * p, dst(e))
        end
    end
    return tn
end

function insert_virtualinds(tn::AbstractTensorNetwork)
    tn = copy(tn)
    return insert_virtualinds!(tn)
end

function map_virtualinds!(f::Function, tn::AbstractTensorNetwork)
    for e in edges(tn)
        vinds = ITensors.commoninds(tn[src(e)], tn[dst(e)])
        vinds_sim = f(vinds)
        setindex_preserve!(tn, ITensors.replaceinds(tn[src(e)], vinds, vinds_sim), src(e))
        setindex_preserve!(tn, ITensors.replaceinds(tn[dst(e)], vinds, vinds_sim), dst(e))
    end
    return tn
end

function map_virtualinds(f::Function, tn::AbstractTensorNetwork)
    tn = copy(tn)
    return map_virtualinds!(f, tn)
end

function combine_virtualinds!(tn::AbstractTensorNetwork)
    dtype = datatype(tn)
    for e in edges(tn)
        vinds = ITensors.commoninds(tn[src(e)], tn[dst(e)])
        if length(vinds) > 1
            C = adapt(dtype)(ITensors.combiner(vinds))
            setindex_preserve!(tn, tn[src(e)] * C, src(e))
            setindex_preserve!(tn, tn[dst(e)] * C, dst(e))
        end
    end
    return tn
end

function combine_virtualinds(tn::AbstractTensorNetwork)
    tn = copy(tn)
    return combine_virtualinds!(tn)
end
