@eval module $(gensym())
using ITensors: ITensors
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test

@testset "Contraction sequences (omeinsum backend)" begin
    Random.seed!(1234)

    # Closed network contracts to a scalar; every backend must agree.
    g = named_grid((3, 3))
    tn = random_tensornetwork(Float64, g; bond_dimension = 2)

    s_default = ITensors.contract(tn; alg = "exact")  # einexpr + Greedy (the default)
    s_optimal = ITensors.contract(tn; alg = "exact",
        contraction_sequence_kwargs = (; alg = "optimal"))
    s_treesa = ITensors.contract(tn; alg = "exact",
        contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = TreeSA()))
    s_greedy = ITensors.contract(tn; alg = "exact",
        contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = GreedyMethod()))

    @test s_optimal ≈ s_default
    @test s_treesa ≈ s_default
    @test s_greedy ≈ s_default

    # The new backend defaults to TreeSA when no optimizer is named.
    s_treesa_default = ITensors.contract(tn; alg = "exact",
        contraction_sequence_kwargs = (; alg = "omeinsum"))
    @test s_treesa_default ≈ s_default
end
end
