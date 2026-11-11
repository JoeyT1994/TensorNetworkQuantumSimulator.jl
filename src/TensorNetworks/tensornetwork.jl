using Dictionaries: Dictionary
using NamedGraphs: incident_edges

const TensorNetwork = ITensorNetwork

function random_tensornetwork(eltype, g::AbstractGraph; bond_dimension::Integer = 1)
    vs = collect(vertices(g))
    l = edge_indices(g, bond_dimension)
    tensors = Dictionary{vertextype(g), Any}()
    for v in vs
        is = [l[NamedEdge(v => vn)] for vn in neighbors(g, v)]
        set!(tensors, v, random_tensor(eltype, is))
    end
    tensors = Dictionary(vs, narrow_tensors(collect(tensors)))
    return TensorNetwork(tensors)
end

random_tensornetwork(g::AbstractGraph; kwargs...) = random_tensornetwork(Float64, g; kwargs...)

# A charged graded state's dangling "Charge" leg is not a site index (see `unprime_charge_legs`).
siteinds(tn::AbstractTensorNetwork, v) = filter(i -> !occursin("Charge", tags(i)), collect(ITNN.siteinds(tn, v)))
siteinds(tn::AbstractTensorNetwork) = Dictionary(vertices(tn), [siteinds(tn, v) for v in vertices(tn)])
