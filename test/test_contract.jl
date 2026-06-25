@eval module $(gensym())
using TensorNetworkQuantumSimulator.ITensorKit:
    Index, random_itensor, contract, contraction_sequence, array
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test

@testset "Test contract / contraction_sequence" begin
    Random.seed!(123)

    # A ring of tensors with one open leg, so the result is a rank-1 ITensorMap.
    i, j, k, l, m = Index(2), Index(2), Index(2), Index(2), Index(3)
    A = random_itensor(i, j)
    B = random_itensor(j, k)
    C = random_itensor(k, l)
    D = random_itensor(l, i, m)   # m is the single open leg
    ts = [A, B, C, D]

    # hand-written pairwise contraction (independent of any sequence finder)
    reference = array(((A * B) * C) * D)

    # default order
    @test array(contract(ts)) ≈ reference
    @test array(A * B * C * D) ≈ reference

    # honored sequences from both optimizers must agree with the default / reference
    for alg in ("optimal", "omeinsum")
        seq = contraction_sequence(ts; alg)
        @test array(contract(ts; sequence = seq)) ≈ reference
    end

    # passing an explicit (suboptimal) sequence still gives the same tensor
    @test array(contract(ts; sequence = [[[1, 2], 3], 4])) ≈ reference
    @test array(contract(ts; sequence = [[1, 4], [2, 3]])) ≈ reference

    # full (scalar) contraction
    sc = [random_itensor(i, j), random_itensor(j, k), random_itensor(k, i)]
    full = contract(sc)
    @test full[] ≈ ((sc[1] * sc[2]) * sc[3])[]
    for alg in ("optimal", "omeinsum")
        @test contract(sc; sequence = contraction_sequence(sc; alg))[] ≈ full[]
    end
end
end
