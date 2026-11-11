@eval module $(gensym())
using LinearAlgebra: norm
using ITensorBase: ITensorBase
import ITensorNetworksNext as ITNN
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test
const TNQS = TensorNetworkQuantumSimulator


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
        @test length(keys(messages(ψ_BPC))) == 2 * length(edges(g))
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
        @test length(keys(messages(ψ_BPC))) == 2 * length(edges(g))
        @test datatype(ψ_BPC) == datatype(ψ)
        @test scalartype(ψ_BPC) == scalartype(ψ)

        ψ_BPC = update(ψ_BPC)
        @test !isempty(messages(ψ_BPC))
        @test length(keys(messages(ψ_BPC))) == 2 * length(edges(g))
        z_bp = partitionfunction(ψ_BPC)
        @test z_bp ≈ norm_sqr(ψ; alg = "exact")
        @test z_bp ≈ norm_sqr(ψ; alg = "bp")

        # converged messages are fixed points of the message update
        e = first(edges(g))
        ms = copy(messages(ψ_BPC))
        ITNN.message_update!(ms, ψ_BPC.factors, e)
        @test ITensorBase.state(ms[e]) ≈ message(ψ_BPC, e)

        vc = first(center(g))
        ρ_bp = reduced_density_matrix(ψ, vc; alg = "bp")
        ρ_exact = reduced_density_matrix(ψ, vc; alg = "exact")
        @test norm(ρ_bp - ρ_exact) <= 10 * eps(real(eltype))
    end
end

@testset "Test setting multiple messages" begin
    g = named_path_graph(2)
    tn = random_tensornetwork(Float64, g; bond_dimension = 2)
    bpc = BeliefPropagationCache(tn)
    e = first(edges(g))
    directed_edges = [e, reverse(e)]
    new_messages = [2 * message(bpc, edge) for edge in directed_edges]

    @test TNQS.setmessages!(bpc, directed_edges, new_messages) === bpc
    @test all(message(bpc, edge) == new_message for (edge, new_message) in zip(directed_edges, new_messages))
end

@testset "Loop corrections are exact on a single loop" begin
    # On one cycle the loop series ends after the loop itself: BP times (1 + its weight) is exact.
    Random.seed!(5)
    ψ = random_tensornetworkstate(ComplexF64, named_grid((2, 2)); bond_dimension = 2)
    @test norm_sqr(ψ; alg = "loopcorrections", max_configuration_size = 4) ≈ norm_sqr(ψ; alg = "exact")
end

end
