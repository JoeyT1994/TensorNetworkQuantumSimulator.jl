using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient
using ITensors: ITensors
Random.seed!(1234)

function main()
    ITensors.disable_warn_order()
    χ = 3
    println("Building Random Complex Spinful Fermion Tensor Network State |psi> of Bond Dim $χ on 3x4 grid of hexagons")
    g = named_hexagonal_lattice_graph(3,4)
    s = siteinds("spinful_fermion", g)
    ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = χ)
    ψ = normalize(ψ; alg = "bp")

    println("Testing the norm <psi|psi>")
    println("Exact norm is $(norm_sqr(ψ; alg = "exact"))")
    println("BP norm is $(norm_sqr(ψ; alg = "bp"))")

    println("Running through BMPS Ranks")
    Rs =  [2,4,8, 16]
    for R in Rs
        ψ_bmps = BoundaryMPSCache(ψ, R)
        ψ_bmps = update(ψ_bmps)
        println("BMPS Norm at Rank $R is $(partitionfunction(ψ_bmps))")
    end
end

main()


