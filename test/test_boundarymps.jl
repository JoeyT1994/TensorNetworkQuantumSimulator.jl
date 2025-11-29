@eval module $(gensym())
using ITensors: datatype, norm
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test BoundaryMPS" begin
    Random.seed!(123)
    g = named_grid((3, 3))

    #BMPS Cache
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetwork(eltype, g; bond_dimension = 2)
        ψ_BMPS = BoundaryMPSCache(ψ, 4)
        @test network(ψ_BMPS) isa TensorNetwork
        @test ψ_BMPS isa BoundaryMPSCache
        @test graph(ψ_BMPS) == g
        @test datatype(ψ_BMPS) == datatype(ψ)
        @test scalartype(ψ_BMPS) == scalartype(ψ)

        ψ_BMPS = update(ψ_BMPS)
        z_bmps = partitionfunction(ψ_BMPS)
        @test z_bmps ≈ contract(ψ; alg = "exact")
        @test z_bmps ≈ contract(ψ; alg = "boundarymps", mps_bond_dimension = 4)
    end

    #BMPS Cache
    s = siteinds("S=1", g)
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetworkstate(eltype, g; bond_dimension = 2)
        ψ_BMPS = BoundaryMPSCache(ψ, 4)
        @test network(ψ_BMPS) isa TensorNetworkState
        @test ψ_BMPS isa BoundaryMPSCache
        @test graph(ψ_BMPS) == g
        @test datatype(ψ_BMPS) == datatype(ψ)
        @test scalartype(ψ_BMPS) == scalartype(ψ)

        ψ_BMPS = update(ψ_BMPS)
        z_bmps = partitionfunction(ψ_BMPS)
        @test z_bmps ≈ norm_sqr(ψ; alg = "exact")
        @test z_bmps ≈ norm_sqr(ψ; alg = "boundarymps", mps_bond_dimension = 4)

        vs = [(2, 1), (2, 3)]
        ρ_bmps_1 = rdm(ψ_BMPS, vs)
        ρ_bmps_2 = reduced_density_matrix(ψ, vs; alg = "boundarymps", mps_bond_dimension = 4)
        ρ_exact = reduced_density_matrix(ψ, vs; alg = "exact")
        @test norm(ρ_bmps_1 - ρ_bmps_2) <= 10 * eps(real(eltype))
        @test norm(ρ_bmps_1 - ρ_exact) <= 10 * eps(real(eltype))
    end
end

end
