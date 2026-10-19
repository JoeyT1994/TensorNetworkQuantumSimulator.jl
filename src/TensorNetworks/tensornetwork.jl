using Dictionaries: Dictionary
using NamedGraphs: incident_edges

const TensorNetwork = ITensorNetwork

function random_tensornetwork(eltype, g::AbstractGraph; bond_dimension::Integer = 1)
    vs = collect(vertices(g))
    l = Dict(e => Index(bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vs
        is = [l[NamedEdge(v => vn)] for vn in neighbors(g, v)]
        set!(tensors, v, randn(eltype, is...))
    end
    return TensorNetwork(tensors)
end

random_tensornetwork(g::AbstractGraph; kwargs...) = random_tensornetwork(Float64, g; kwargs...)

siteinds(tn::AbstractTensorNetwork, v) = ITNN.siteinds(tn, v)
siteinds(tn::AbstractTensorNetwork) = Dictionary(vertices(tn), [siteinds(tn, v) for v in vertices(tn)])
