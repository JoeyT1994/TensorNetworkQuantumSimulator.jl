using TensorNetworkQuantumSimulator

using TensorNetworkQuantumSimulator: apply_gates, update_message!, network, norm_factors, QuadraticForm, partitionfunction,
    norm_sqr
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using Graphs
using Statistics
using ITensors: ITensor, Index, Algorithm, ITensors, inds, dag, prime
using ITensorNetworks: @preserve_graph

using NamedGraphs: NamedGraphs, neighbors, NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.PartitionedGraphs: PartitionEdge, PartitionVertex

using LinearAlgebra
using Dictionaries: Dictionary

function main()
    #Define the lattice
    ITensors.disable_warn_order()
    g = named_grid((4, 1))
    #g = TN.heavy_hexagonal_lattice(3,3)

    siteinds = TN.siteinds(g, "S=1/2")
    ψ = random_tensornetworkstate(ComplexF64, g, siteinds; bond_dimension = 4)
    ψ = normalize(ψ; alg = "bp")

    qf = QuadraticForm(ψ, v -> v == (1, 1) ? "Z" : "I")
    bpc = BeliefPropagationCache(qf)
    bpc = update(bpc)
    @show partitionfunction(bpc)

    return @show expect(ψ, ("Z", (1, 1)); alg = "bp")

end
main()
