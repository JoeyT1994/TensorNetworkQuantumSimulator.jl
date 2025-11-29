@eval module $(gensym())
using Dictionaries: Dictionary
using ITensors: datatype
using Random
using Statistics
using TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test Sampling" begin
    Random.seed!(123)

    #Product State (all down) on hexagonal lattice
    g = named_hexagonal_lattice_graph(3, 3)
    ψ = tensornetworkstate(ComplexF64, v -> "↑", g)
    ψ = gauge_and_scale(ψ)
    bmps_sample = only(sample(ψ, 1; alg = "boundarymps", norm_mps_bond_dimension = 1, projected_mps_bond_dimension = 1, gauge_and_scale = false))
    @test all([bmps_sample[v] == 0 for v in vertices(g)])

    bp_sample = only(sample(ψ, 1; alg = "bp", gauge_state = false))
    @test all([bp_sample[v] == 0 for v in vertices(g)])

    #GHZ state on square grid
    g = named_grid((3, 3))
    s = siteinds("S=1/2", g)
    ψ1, ψ2 = tensornetworkstate(Float64, v -> "↑", g, s), tensornetworkstate(Float64, v -> "↓", g, s)
    ψ = ψ1 + ψ2
    #Set BP norm to 1 (sample will do this automatically unless you state gauge_and_scale = false)
    ψ = gauge_and_scale(ψ)

    nsamples = 5
    bp_samples = sample(ψ, nsamples; alg = "bp", gauge_state = false)
    @test bp_samples isa Vector{<:Dictionary{vertextype(g), Int}}
    @test length(bp_samples) == nsamples
    @test all([keys(bp_sample) == vertices(g) for bp_sample in bp_samples])
    bmps_certified_samples = sample_certified(ψ, nsamples; alg = "boundarymps", norm_mps_bond_dimension = 4, projected_mps_bond_dimension = 4, gauge_and_scale = false)
    p_qs = first.(bmps_certified_samples)
    bitstrings = last.(bmps_certified_samples)
    #GHZ state, so samples should be either all up or all down
    @test all([all([b[v] == 0 for v in vertices(g)]) || all([b[v] == 1 for v in vertices(g)]) for b in bitstrings])

    #We importance sampled with big enough mps_dimensions. Standard deviation should be small, mean should be norm of network.
    @test Statistics.std(p_qs) < 1.0e-8
    @test Statistics.mean(p_qs) ≈ norm_sqr(ψ; alg = "boundarymps", mps_bond_dimension = 4)
end

end
