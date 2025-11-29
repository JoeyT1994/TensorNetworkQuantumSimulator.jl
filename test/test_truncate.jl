@eval module $(gensym())
using Dictionaries: Dictionary
using ITensors: datatype
using Random
using Statistics
using TensorNetworkQuantumSimulator
using Test: @testset, @test
using NamedGraphs: NamedEdge, has_edge
using TensorNetworkQuantumSimulator: virtualinds, tensors


@testset "Test Sampling" begin
    Random.seed!(123)

    #Product State (all down) on hexagonal lattice
    g = named_hexagonal_lattice_graph(2, 2)
    ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 3)
    ψ = gauge_and_scale(ψ)

    ψ_trunc_bp = truncate(ψ; alg = "bp", maxdim = 2, cutoff = 1.0e-10, normalize_tensors = false)
    ψ_trunc_bmps = truncate(ψ; alg = "boundarymps", maxdim = 2, cutoff = 1.0e-10, normalize_tensors = false, gauge_state = false, mps_bond_dimension = 9)

    f_bp = inner(ψ_trunc_bp, ψ; alg = "exact") / sqrt(norm_sqr(ψ_trunc_bp; alg = "exact") * norm_sqr(ψ; alg = "exact"))
    f_bmps = inner(ψ_trunc_bmps, ψ; alg = "exact") / sqrt(norm_sqr(ψ_trunc_bmps; alg = "exact") * norm_sqr(ψ; alg = "exact"))

    f_bp = real(f_bp * conj(f_bp))
    f_bmps = real(f_bmps * conj(f_bmps))

    @test 0 ≤ f_bp ≤ 1
    @test 0 ≤ f_bmps ≤ 1
    @test f_bmps ≥ f_bp
    @test maxvirtualdim(ψ_trunc_bp) ≤ 2
    @test maxvirtualdim(ψ_trunc_bmps) ≤ 2

end

end
