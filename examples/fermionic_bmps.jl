using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient
using ITensors: ITensors
Random.seed!(1234)

function main()
    ITensors.disable_warn_order()
    χ = 3
    println("Building Random Complex Spinful Fermion Tensor Network State |psi> of Bond Dim $χ on 5x4 grid")
    #g = named_hexagonal_lattice_graph(3,3)
    g = named_grid((5,4))
    v1, v2= (2,2), (2,3)
    s = siteinds("spinful_fermion", g)
    ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = χ)
    ψ = normalize(ψ; alg = "bp")

    println("Testing the norm <psi|psi>")
    println("Exact norm is $(norm_sqr(ψ; alg = "exact"))")
    println("BP norm is $(norm_sqr(ψ; alg = "bp"))")

    println("Running through BMPS Ranks")
    Rs =  [2,4,8,16, 32, 64]
    for R in Rs
        ψ_bmps = BoundaryMPSCache(ψ, R)
        ψ_bmps = update(ψ_bmps)
        println("BMPS Norm at Rank $R is $(partitionfunction(ψ_bmps))")
    end

    println("Testing hopping CiupCjupdag + CjupCupidag on nearest neighbors")
    e_exact = expect(ψ, (["Cupdag", "Cup"], [v1, v2]); alg = "exact") + expect(ψ, (["Cupdag", "Cup"], [v2, v1]); alg = "exact")
    e_bp = expect(ψ, (["Cupdag", "Cup"], [v1, v2]); alg = "bp") + expect(ψ, (["Cupdag", "Cup"], [v2, v1]); alg = "bp")
    println("Exact hopping is $e_exact")
    println("BP hopping is $e_bp")

    println("Running through BMPS Ranks")
    Rs =  [2,4,8,16, 32, 64]
    for R in Rs
        e_bmps = expect(ψ, (["Cupdag", "Cup"], [v1, v2]); alg = "boundarymps", mps_bond_dimension = R) + expect(ψ, (["Cupdag", "Cup"], [v2, v1]); alg = "boundarymps", mps_bond_dimension = R)
        println("BMPS hopping at Rank $R is $e_bmps")
    end

    χ = 3
    println("Building Random Complex Spinful Fermion Tensor Network State |psi> of Bond Dim $χ on 3x3 hexagonal grid")
    g = named_hexagonal_lattice_graph(3,3)
    v1, v2= (2,2), (2,4)
    s = siteinds("spinful_fermion", g)
    ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = χ)
    ψ = normalize(ψ; alg = "bp")

    println("Testing the norm <psi|psi>")
    println("Exact norm is $(norm_sqr(ψ; alg = "exact"))")
    println("BP norm is $(norm_sqr(ψ; alg = "bp"))")

    println("Running through BMPS Ranks")
    Rs =  [2,4,8,16, 32, 64]
    for R in Rs
        ψ_bmps = BoundaryMPSCache(ψ, R)
        ψ_bmps = update(ψ_bmps)
        println("BMPS Norm at Rank $R is $(partitionfunction(ψ_bmps))")
    end

    println("Testing hopping CiupCjupdag + CjupCupidag on next nearest neighbors")
    e_exact = expect(ψ, (["Cupdag", "Cup"], [v1, v2]); alg = "exact") + expect(ψ, (["Cupdag", "Cup"], [v2, v1]); alg = "exact")
    e_bp = expect(ψ, (["Cupdag", "Cup"], [v1, v2]); alg = "bp") + expect(ψ, (["Cupdag", "Cup"], [v2, v1]); alg = "bp")
    println("Exact hopping is $e_exact")
    println("BP hopping is $e_bp")

    println("Running through BMPS Ranks")
    Rs =  [2,4,8,16, 32, 64]
    for R in Rs
        e_bmps = expect(ψ, (["Cupdag", "Cup"], [v1, v2]); alg = "boundarymps", mps_bond_dimension = R) + expect(ψ, (["Cupdag", "Cup"], [v2, v1]); alg = "boundarymps", mps_bond_dimension = R)
        println("BMPS hopping at Rank $R is $e_bmps")
    end


end

main()


