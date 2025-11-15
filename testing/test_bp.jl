using TensorNetworkQuantumSimulator

using TensorNetworkQuantumSimulator: apply_gates, update_message!, network, norm_factors, AbstractBeliefPropagationCache
const TN = TensorNetworkQuantumSimulator

using Graphs
using Statistics
using ITensors: ITensor, Index, Algorithm, ITensors, inds, dag, prime

using NamedGraphs: NamedGraphs, neighbors, NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.PartitionedGraphs: PartitionEdge, PartitionVertex

using LinearAlgebra
using Dictionaries: Dictionary
using Random

using ITensors: @Algorithm_str, contract
using Adapt: adapt

function main()
    Random.seed!(1234)
    ITensors.disable_warn_order()
    nx, ny = 10, 10
    g = named_grid((nx, ny))

    siteinds = TN.siteinds("S=1/2", g)
    ψ = random_tensornetworkstate(ComplexF32, g, siteinds; bond_dimension = 50)

    return @show Base.summarysize(ψ) / (1.0e9)

    # ψ_bmps = TN.update(TN.BoundaryMPSCache(ψ, 2; gauge_state = false))

    # ψ_bmps = TN.update_partition(ψ_bmps, PartitionVertex(2))

    # @show TN.vertex_scalar(ψ_bmps, (2,2))
    # @show TN.vertex_scalar(ψ_bmps, (2,3))
    # @show TN.vertex_scalar(ψ_bmps, (2,4))


end
main()
