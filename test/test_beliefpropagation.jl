@eval module $(gensym())
using ITensors: datatype, norm
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

        # converged messages are fixed points of the message update
        e = first(edges(g))
        m, _ = TNQS.updated_message(ψ_BPC, e)
        @test m ≈ message(ψ_BPC, e)

        vc = first(center(g))
        ρ_bp = reduced_density_matrix(ψ, vc; alg = "bp")
        ρ_exact = reduced_density_matrix(ψ, vc; alg = "exact")
        @test norm(ρ_bp - ρ_exact) <= 10 * eps(real(eltype))
    end
end

@testset "Test contraction sequence cache clearing" begin
    Random.seed!(456)
    g = named_comb_tree((3, 3))

    # Test that sequences are empty before update
    ψ = random_tensornetworkstate(Float64, g; bond_dimension = 2)
    bpc = BeliefPropagationCache(ψ)
    @test isempty(TNQS.contraction_sequences(bpc))

    # Test that sequences are cleared after update returns (only live during update)
    bpc = update(bpc)
    @test isempty(TNQS.contraction_sequences(bpc))
end

@testset "Test setting multiple messages" begin
    g = named_path_graph(2)
    tn = random_tensornetwork(Float64, g; bond_dimension = 2)
    bpc = BeliefPropagationCache(tn)
    e = first(edges(g))
    directed_edges = [e, reverse(e)]
    new_messages = [TNQS.default_message(bpc, edge) for edge in directed_edges]

    @test TNQS.setmessages!(bpc, directed_edges, new_messages) === bpc
    @test all(message(bpc, edge) == new_message for (edge, new_message) in zip(directed_edges, new_messages))
end

@testset "BP uses the fused kernels, which match the generic contraction" begin
    Random.seed!(1)
    ψ = random_tensornetworkstate(ComplexF64, named_grid((3, 3)); bond_dimension = 2)
    bpc = update(BeliefPropagationCache(ψ))
    v = (2, 2)
    incoming = TNQS.incoming_messages(bpc, v; ignore_edges = (NamedEdge((2, 3) => v),))
    fast = TNQS.norm_message_kernel(ψ, v, incoming)
    @test !isnothing(fast)
    ts = [incoming; TNQS.bp_factors(ψ, v)]
    @test TNQS.normalize_message(fast) ≈ TNQS.normalize_message(contract(ts; sequence = TNQS.contraction_sequence(ts; alg = "optimal")))
    @test !isnothing(TNQS.norm_scalar_kernel(ψ, [v], TNQS.incoming_messages(bpc, v); op_strings = _ -> "Z"))
end

end
