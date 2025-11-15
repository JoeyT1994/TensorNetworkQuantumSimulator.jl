@eval module $(gensym())
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test forms" begin
    Random.seed!(123)
    g = named_grid((3, 3))
    s = siteinds("S=1/2", g)

    #Quadratic Form
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetworkstate(eltype, g, s; bond_dimension = 2)
        ψ = normalize(ψ; alg = "bp")
        ψψ = QuadraticForm(ψ)
        @test ψψ isa QuadraticForm
        @test scalartype(ψψ) == eltype
        @test graph(ψψ) == g

        ψψ_bpc = update(BeliefPropagationCache(ψψ))
        @test partitionfunction(ψψ_bpc) ≈ norm_sqr(ψ; alg = "bp")

        mps_bond_dimension = 16
        ψψ_bmps = update(BoundaryMPSCache(ψψ, mps_bond_dimension))
        @test partitionfunction(ψψ_bmps) ≈ norm_sqr(ψ; alg = "exact")
    end

    #BiLinear Form
    g = named_comb_tree((3, 3))
    s = siteinds("S=1/2", g)
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetworkstate(eltype, g, s; bond_dimension = 3)
        ϕ = random_tensornetworkstate(eltype, g, s; bond_dimension = 4)
        ψ, ϕ = normalize(ψ; alg = "bp"), normalize(ϕ; alg = "bp")
        ψϕ = BilinearForm(ψ, ϕ)
        @test ψϕ isa BilinearForm
        @test scalartype(ψϕ) == eltype
        @test graph(ψϕ) == g

        ψϕ_bpc = update(BeliefPropagationCache(ψϕ))

        @test partitionfunction(ψϕ_bpc) ≈ inner(ψ, ϕ; alg = "bp")
    end

end

end
