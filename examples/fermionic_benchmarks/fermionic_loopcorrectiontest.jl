using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator, loopcorrected_partitionfunction
using ITensors: ITensors
Random.seed!(1234)

function main()
    ITensors.disable_warn_order()
    χ = 4
    g = named_hexagonal_lattice_graph(4,4)

    println("Random hexagonal fermionic tensor network state of BD $(χ)")
    s = siteinds("spinful_fermion", g)
    ψf = random_fermionic_tensornetworkstate(g, s; bond_dimension = χ)
    ψf_bpc = BeliefPropagationCache(ψf)
    ψf_bpc = update(ψf_bpc)
    rescale!(ψf_bpc)

    ψf = network(ψf_bpc)

    v = first(center(g))
    ndn_bp = expect(ψf, (["NupNdn"], [v]); alg = "bp")
    #ndn_exact = expect(ψf, (["NupNdn"], [v]); alg = "exact")
    t1 = time()
    ndn_lc = expect(ψf, (["NupNdn"], [v]); alg = "loopcorrections", max_configuration_size =6, ε = 1e-3)
    t2 = time()
    ndn_lc_ad = expect(ψf, (["NupNdn"], [v]); alg = "loopcorrections", max_configuration_size =6, autodiff = true)
    t3 = time()

    #@show ndn_exact
    @show ndn_bp
    @show ndn_lc
    @show ndn_lc_ad

    println("Non AD took $(t2 - t1) secs")
    println("AD took $(t3 - t2) secs")
end


main()


