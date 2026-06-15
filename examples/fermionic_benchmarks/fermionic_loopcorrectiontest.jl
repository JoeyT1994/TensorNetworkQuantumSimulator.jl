using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator, loopcorrected_partitionfunction
using ITensors: ITensors
Random.seed!(1234)

function main()
    ITensors.disable_warn_order()
    χ = 2
    g = named_hexagonal_lattice_graph(4,4)

    println("Random hexagonal fermionic tensor network state of BD $(χ)")
    s = siteinds("fermion", g)
    ψf = random_fermionic_tensornetworkstate(g, s; bond_dimension = χ)
    ψf_bpc = BeliefPropagationCache(ψf)
    ψf_bpc = update(ψf_bpc)
    rescale!(ψf_bpc)

    ψf = network(ψf_bpc)

    norm_bp = norm_sqr(ψf; alg = "bp")
    norm_lc = norm_sqr(ψf; alg = "loopcorrections", max_configuration_size = 6)
    norm_bmps = norm_sqr(ψf; alg = "boundarymps", mps_bond_dimension = 32)
    norm_exact= norm_sqr(ψf; alg = "exact")

    println("BP norm is $norm_bp")
    println("First order LC norm is $norm_lc")
    println("BoundaryMPS norm with MPS dim 32 is $norm_bmps")
    println("Exact norm is $norm_exact")
end


main()


