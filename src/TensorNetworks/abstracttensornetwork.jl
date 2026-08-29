using Graphs: Graphs, has_vertex
using NamedGraphs: NamedGraphs
using Adapt

abstract type AbstractTensorNetwork{V} <: AbstractNamedGraph{V} end

graph(tn::AbstractTensorNetwork) = not_implemented()
tensors(tn::AbstractTensorNetwork) = not_implemented()
NamedGraphs.rem_vertex!(tn::AbstractTensorNetwork, v) = not_implemented()
add_tensor!(tn::AbstractTensorNetwork, tensor, v) = not_implemented()

#The tensor type this network holds (e.g. Tensor), stripped of type parameters
#since derived tensors (e.g. real singular-value messages in a complex network) may differ in
#eltype or rank.
tensortype(tn::AbstractTensorNetwork) = unspecify_type_parameters(eltype(tensors(tn)))

Graphs.is_directed(::Type{<:AbstractTensorNetwork}) = false

NamedGraphs.vertex_positions(tn::AbstractTensorNetwork) = NamedGraphs.vertex_positions(graph(tn))
NamedGraphs.ordered_vertices(tn::AbstractTensorNetwork) = NamedGraphs.ordered_vertices(graph(tn))
NamedGraphs.position_graph(tn::AbstractTensorNetwork) = NamedGraphs.position_graph(graph(tn))
NamedGraphs.vertices(tn::AbstractTensorNetwork) = NamedGraphs.vertices(graph(tn))
NamedGraphs.edges(tn::AbstractTensorNetwork) = NamedGraphs.edges(graph(tn))
NamedGraphs.edgetype(tn::AbstractTensorNetwork) = NamedGraphs.edgetype(graph(tn))
NamedGraphs.vertextype(tn::AbstractTensorNetwork) = NamedGraphs.vertextype(graph(tn))

virtualinds(tn::AbstractTensorNetwork, e::NamedEdge) = commoninds(tn[src(e)], tn[dst(e)])
virtualind(tn::AbstractTensorNetwork, e::NamedEdge) = only(virtualinds(tn, e))

function maxvirtualdim(tn::AbstractTensorNetwork)
    return maximum(maximum.([dim.(virtualinds(tn, e)) for e in edges(tn)]))
end

function uniqueinds(tn::AbstractTensorNetwork, v)
    tv_inds = collect(inds(tn[v]))
    vns = neighbors(tn, v)
    isempty(vns) && return tv_inds
    neighbor_inds = reduce(vcat, [collect(inds(tn[vn])) for vn in vns])
    is = setdiff(tv_inds, neighbor_inds)
    return is
end

function setindex_preserve!(tn::AbstractTensorNetwork, value, vertex)
    tensors(tn)[vertex] = value
    return tn
end

function Base.setindex!(tn::AbstractTensorNetwork, value, vertex)
    !has_vertex(graph(tn), vertex) && error("Vertex not in tensor network")
    add_tensor!(tn, value, vertex)
    return tn
end

function scalartype(tn::AbstractTensorNetwork)
    return mapreduce(v -> scalartype(tn[v]), promote_type, vertices(tn))
end

function datatype(tn::AbstractTensorNetwork)
    return mapreduce(v -> datatype(tn[v]), promote_type, vertices(tn))
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

function map_virtualinds!(f::Function, tn::AbstractTensorNetwork)
    for e in edges(tn)
        vinds = commoninds(tn[src(e)], tn[dst(e)])
        vinds_sim = f(vinds)
        setindex_preserve!(tn, replaceinds(tn[src(e)], vinds, vinds_sim), src(e))
        setindex_preserve!(tn, replaceinds(tn[dst(e)], vinds, vinds_sim), dst(e))
    end
    return tn
end

function map_virtualinds(f::Function, tn::AbstractTensorNetwork)
    tn = copy(tn)
    return map_virtualinds!(f, tn)
end

"""Add two tensornetworks together. The network structures need to be have the same graph structure"""
function add(tn1::AbstractTensorNetwork, tn2::AbstractTensorNetwork)
    graph(tn1) == graph(tn2) || error("add: the two networks must share the same graph")
    if tn1 isa TensorNetworkState && tn2 isa TensorNetworkState
        siteinds(tn1) == siteinds(tn2) || error("add: the two states must share the same site indices")
    else
        tn1 isa TensorNetwork && tn2 isa TensorNetwork ||
            error("add: expected two TensorNetworks or two TensorNetworkStates")
    end

    es = edges(tn1)
    tn12 = copy(tn1)
    new_edge_indices = Dict(
        zip(
            es,
            [
                new_index(
                        dim(only(virtualinds(tn1, e))) + dim(only(virtualinds(tn2, e))),
                    ) for e in es
            ],
        ),
    )

    #Create vertices of tn12 as direct sum of tn1[v] and tn2[v]. Work out the matching indices by matching edges. Make index tags those of tn1[v]
    for v in vertices(tn1)
        es_v = filter(x -> src(x) == v || dst(x) == v, es)

        tn1v_linkinds = [only(virtualinds(tn1, e)) for e in es_v]
        tn2v_linkinds = [only(virtualinds(tn2, e)) for e in es_v]
        tn12v_linkinds = [new_edge_indices[e] for e in es_v]

        setindex_preserve!(
            tn12, directsum(
                tn12v_linkinds,
                tn1[v] => Tuple(tn1v_linkinds),
                tn2[v] => Tuple(tn2v_linkinds)
            ), v
        )
    end

    return tn12
end

Base.:+(tn1::AbstractTensorNetwork, tn2::AbstractTensorNetwork) = add(tn1, tn2)
