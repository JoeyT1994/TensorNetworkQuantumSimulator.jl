using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator
using ITensors: ITensors
Random.seed!(1234)

function main()
    ITensors.disable_warn_order()
    χ = 64
    g = named_hexagonal_lattice_graph(4,4)

    s = siteinds("fermion", g)
    ψf = random_fermionic_tensornetworkstate(g, s; bond_dimension = χ)
    ψf_bpc = BeliefPropagationCache(ψf)
    @time ψf_bpc = update(ψf_bpc; tolerance = nothing, maxiter = 10)

    @show TensorNetworkQuantumSimulator.freenergy(ψf_bpc)

    s = siteinds("S=1/2", g)
    ψf = random_tensornetworkstate(g, s; bond_dimension = χ)
    ψf_bpc = BeliefPropagationCache(ψf)
    @time ψf_bpc = update(ψf_bpc; tolerance = nothing, maxiter = 10)
end

main()


