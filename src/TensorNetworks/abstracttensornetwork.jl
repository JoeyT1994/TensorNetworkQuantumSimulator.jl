using Graphs: Graphs, has_vertex
using NamedGraphs: NamedGraphs
using Adapt

const AbstractTensorNetwork = ITNN.AbstractITensorNetwork

graph(tn::AbstractTensorNetwork) = DataGraphs.underlying_graph(tn)

#The tensor type this network holds (e.g. Tensor), stripped of type parameters
#since derived tensors (e.g. real singular-value messages in a complex network) may differ in
#eltype or rank.
tensortype(::AbstractTensorNetwork{T}) where {T} = unspecify_type_parameters(T)

# A new index of dimension `d` for every edge of `g` (tagged `tags(e)`), under both orientations.
function edge_indices(g::AbstractGraph, d::Integer; tags = e -> "")
    l = Dictionary(collect(edges(g)), [new_index(d; tags = tags(e)) for e in edges(g)])
    return merge(l, Dictionary(reverse.(collect(edges(g))), collect(l)))
end

virtualinds(tn::AbstractTensorNetwork, e::NamedEdge) = ITNN.linkinds(tn, e)
virtualind(tn::AbstractTensorNetwork, e::NamedEdge) = only(virtualinds(tn, e))

# The legs a BP message carries for the virtual legs `l` of an edge: `l` itself (a form's virtual
# legs already include every layer), and for a state's norm network also the bra copies `l′†`.
message_legs(tn::AbstractTensorNetwork, l) = l
message_inds(tn::AbstractTensorNetwork, e) = message_legs(tn, virtualinds(tn, e))

maxvirtualdim(tn::AbstractTensorNetwork) = maximum(e -> maximum(dim, virtualinds(tn, e)), edges(tn))

function scalartype(tn::AbstractTensorNetwork)
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

function map_tensors(f::Function, tn::AbstractTensorNetwork)
    tn = copy(tn)
    return map_tensors!(f, tn)
end

function Adapt.adapt_structure(to, tn::AbstractTensorNetwork)
    return map_tensors(x -> adapt(to)(x), tn)
end

function map_virtualinds!(f::Function, tn::AbstractTensorNetwork)
    for e in collect(edges(tn))
        vinds = commoninds(tn[src(e)], tn[dst(e)])
        vinds_sim = f(vinds)
        tn[src(e)] = replaceinds(tn[src(e)], vinds, vinds_sim)
        tn[dst(e)] = replaceinds(tn[dst(e)], vinds, vinds_sim)
    end
    return tn
end

function map_virtualinds(f::Function, tn::AbstractTensorNetwork)
    tn = copy(tn)
    return map_virtualinds!(f, tn)
end

function combine_virtualinds!(tn::AbstractTensorNetwork)
    dtype = datatype(tn)
    for e in collect(edges(tn))
        vinds = commoninds(tn[src(e)], tn[dst(e)])
        if length(vinds) > 1
            C = adapt(dtype)(combiner(vinds))
            tn[src(e)] = tn[src(e)] * C
            # the other end contracts the combined leg with the opposite orientation
            tn[dst(e)] = tn[dst(e)] * dag(C)
        end
    end
    return tn
end

function combine_virtualinds(tn::AbstractTensorNetwork)
    tn = copy(tn)
    return combine_virtualinds!(tn)
end

"""Add two tensornetworks together. The network structures need to be have the same graph structure"""
function add(tn1::AbstractTensorNetwork, tn2::AbstractTensorNetwork)
    graph(tn1) == graph(tn2) || error("add: the two networks must share the same graph")
    siteinds(tn1) == siteinds(tn2) || error("add: the two networks must share the same site indices")

    es = collect(edges(tn1))
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

        tn12[v] = directsum(
            tn12v_linkinds,
            tn1[v] => Tuple(tn1v_linkinds),
            tn2[v] => Tuple(tn2v_linkinds)
        )
    end

    return tn12
end

Base.:+(tn1::AbstractTensorNetwork, tn2::AbstractTensorNetwork) = add(tn1, tn2)
