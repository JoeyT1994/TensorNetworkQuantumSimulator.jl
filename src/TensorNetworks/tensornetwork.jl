using Dictionaries: Dictionary
using Graphs: Graphs
using NamedGraphs: NamedGraphs, add_edge!, incident_edges
using Adapt

const TensorNetwork = ITensorNetwork

graph(tn::TensorNetwork) = tn.underlying_graph
tensors(tn::TensorNetwork) = tn.tensors

function TensorNetwork(tensors::Vector{<:ITensor})
    return TensorNetwork(Dictionary(eachindex(tensors), tensors))
end

function default_message(tn::TensorNetwork, edge::NamedEdge)
    return adapt_like(tn, delta(virtualinds(tn, edge)))
end

function bp_factors(tn::TensorNetwork, vertex)
    return ITensor[tn[vertex]]
end

function bp_factors(tn::TensorNetwork, vertices::Vector)
    return ITensor[tn[v] for v in vertices]
end

function random_tensornetwork(eltype, g::AbstractGraph; bond_dimension::Integer = 1)
    l = Dict(e => Index(bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vertices(g)
        is = [l[NamedEdge(v => vn)] for vn in neighbors(g, v)]
        set!(tensors, v, randn(eltype, is...))
    end
    return TensorNetwork(tensors)
end

random_tensornetwork(g::AbstractGraph; kwargs...) = random_tensornetwork(Float64, g; kwargs...)
