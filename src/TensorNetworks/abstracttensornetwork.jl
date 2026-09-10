using Adapt
using VectorInterface: VectorInterface, scalartype

const AbstractTensorNetwork = AbstractITensorNetwork

virtualinds(tn::AbstractTensorNetwork, e::NamedEdge) = linkinds(tn, e)
virtualind(tn::AbstractTensorNetwork, e::NamedEdge) = only(virtualinds(tn, e))

function maxvirtualdim(tn::AbstractTensorNetwork)
    return maximum(maximum.([length.(virtualinds(tn, e)) for e in edges(tn)]))
end

uniqueinds(tn::AbstractTensorNetwork, v) = ITensorNetworksNext.siteinds(tn, v)

siteinds(tn::AbstractTensorNetwork, v) = Index[i for i in uniqueinds(tn, v)]
function siteinds(tn::AbstractTensorNetwork)
    return Dictionary{vertextype(tn), Vector{<:Index}}(
        collect(vertices(tn)), [siteinds(tn, v) for v in vertices(tn)]
    )
end

function VectorInterface.scalartype(tn::AbstractTensorNetwork)
    return mapreduce(v -> scalartype(tn[v]), promote_type, vertices(tn))
end

function datatype(tn::AbstractTensorNetwork)
    return mapreduce(v -> datatype(tn[v]), promote_type, vertices(tn))
end

function map_tensors!(f::Function, tn::AbstractTensorNetwork)
    for v in vertices(tn)
        tn[v] = f(tn[v])
    end
    return tn
end

map_tensors(f::Function, tn::AbstractTensorNetwork) = map_vertex_data(f, tn)

function Adapt.adapt_structure(to, tn::AbstractTensorNetwork)
    return map_tensors(x -> adapt(to)(x), tn)
end

function map_virtualinds!(f::Function, tn::AbstractTensorNetwork)
    for e in edges(tn)
        vinds = virtualinds(tn, e)
        vinds_sim = f.(vinds)
        tn[src(e)] = replaceinds(tn[src(e)], (vinds .=> vinds_sim)...)
        tn[dst(e)] = replaceinds(tn[dst(e)], (vinds .=> vinds_sim)...)
    end
    return tn
end

function map_virtualinds(f::Function, tn::AbstractTensorNetwork)
    tn = copy(tn)
    return map_virtualinds!(f, tn)
end

"""Add two tensornetworks together. The network structures need to be have the same graph structure"""
function add(tn1::AbstractTensorNetwork, tn2::AbstractTensorNetwork)
    @assert graph(tn1) == graph(tn2)

    if tn1 isa TensorNetworkState && tn2 isa TensorNetworkState
        @assert siteinds(tn1) == siteinds(tn2)
    else
        @assert tn1 isa TensorNetwork && tn2 isa TensorNetwork
    end

    es = collect(edges(tn1))
    tn12 = copy(tn1)
    new_edge_indices = Dict(
        zip(
            es,
            [
                Index(
                        length(only(virtualinds(tn1, e))) + length(only(virtualinds(tn2, e))),
                    ) for e in es
            ],
        ),
    )

    #Create vertices of tn12 as direct sum of tn1[v] and tn2[v]. Work out the matching indices by matching edges. Make index tags those of tn1[v]
    for v in vertices(tn1)
        es_v = filter(x -> src(x) == v || dst(x) == v, es)

        tn1v_linkinds = Index[only(virtualinds(tn1, e)) for e in es_v]
        tn2v_linkinds = Index[only(virtualinds(tn2, e)) for e in es_v]
        tn12v_linkinds = Index[new_edge_indices[e] for e in es_v]

        tn12[v] = directsum(
            tn12v_linkinds,
            tn1[v] => Tuple(tn1v_linkinds),
            tn2[v] => Tuple(tn2v_linkinds)
        )
    end

    return tn12
end

Base.:+(tn1::AbstractTensorNetwork, tn2::AbstractTensorNetwork) = add(tn1, tn2)
