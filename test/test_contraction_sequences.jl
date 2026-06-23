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

    s_default = ITensors.contract(tn; alg = "exact")  # omeinsum + TreeSA (the default)
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

    # Open network: result is a tensor with dangling indices (iy non-empty).
    i, j, k, l, m = ITensors.Index(2), ITensors.Index(3), ITensors.Index(2), ITensors.Index(3), ITensors.Index(2)
    A = ITensors.random_itensor(i, j)
    B = ITensors.random_itensor(j, k, l)
    C = ITensors.random_itensor(l, m)
    open_tensors = [A, B, C]  # open indices: i, k, m

    seq_treesa = TensorNetworkQuantumSimulator.contraction_sequence(open_tensors; alg = "omeinsum", optimizer = TreeSA())
    seq_optimal = TensorNetworkQuantumSimulator.contraction_sequence(open_tensors; alg = "optimal")

    t_treesa = ITensors.contract(open_tensors; sequence = seq_treesa)
    t_optimal = ITensors.contract(open_tensors; sequence = seq_optimal)

    @test t_treesa ≈ t_optimal
end
end
