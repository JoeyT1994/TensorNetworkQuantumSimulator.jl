@eval module $(gensym())
using ITensors: datatype, norm
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test BP" begin
    Random.seed!(123)
    g = named_comb_tree((3, 3))

    #BP Cache
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetwork(eltype, g; bond_dimension = 2)
        ψ_BPC = BeliefPropagationCache(ψ)
        @test network(ψ_BPC) isa TensorNetwork
        @test ψ_BPC isa BeliefPropagationCache
        @test graph(ψ_BPC) == g
        @test isempty(messages(ψ_BPC))
        @test datatype(ψ_BPC) == datatype(ψ)
        @test scalartype(ψ_BPC) == scalartype(ψ)

        ψ_BPC = update(ψ_BPC)
        @test !isempty(messages(ψ_BPC))
        @test length(keys(messages(ψ_BPC))) == 2 * length(edges(g))
        z_bp = partitionfunction(ψ_BPC)
        @test z_bp ≈ contract(ψ; alg = "exact")
        @test z_bp ≈ contract(ψ; alg = "bp")
    end

    #BP Cache
    s = siteinds("S=1", g)
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetworkstate(eltype, g; bond_dimension = 2)
        ψ_BPC = BeliefPropagationCache(ψ)
        @test ψ_BPC isa BeliefPropagationCache
        @test network(ψ_BPC) isa TensorNetworkState
        @test graph(ψ_BPC) == g
        @test isempty(messages(ψ_BPC))
        @test datatype(ψ_BPC) == datatype(ψ)
        @test scalartype(ψ_BPC) == scalartype(ψ)

        ψ_BPC = update(ψ_BPC)
        @test !isempty(messages(ψ_BPC))
        @test length(keys(messages(ψ_BPC))) == 2 * length(edges(g))
        z_bp = partitionfunction(ψ_BPC)
        @test z_bp ≈ norm_sqr(ψ; alg = "exact")
        @test z_bp ≈ norm_sqr(ψ; alg = "bp")

        vc = first(center(g))
        ρ_bp = reduced_density_matrix(ψ, vc; alg = "bp")
        ρ_exact = reduced_density_matrix(ψ, vc; alg = "exact")
        @test norm(ρ_bp - ρ_exact) <= 10 * eps(real(eltype))
    end
end

end
