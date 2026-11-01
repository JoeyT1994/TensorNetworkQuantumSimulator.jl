@eval module $(gensym())
using Dictionaries: Dictionary
using ITensors: datatype
using Random
using Statistics
using TensorNetworkQuantumSimulator
using Test: @testset, @test
using NamedGraphs: NamedEdge, has_edge
using TensorNetworkQuantumSimulator: virtualinds, tensors


@testset "Test Truncate" begin
    Random.seed!(123)

    #Product State (all down) on hexagonal lattice
    g = named_hexagonal_lattice_graph(2, 2)
    ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 3)
    ψ = gauge_and_scale(ψ)

    ψ_trunc_bp = truncate(ψ; alg = "bp", maxdim = 2, cutoff = 1.0e-10, normalize_tensors = false)
    ψ_trunc_ctmrg = truncate(ψ; alg = "ctmrg", maxdim = 2, cutoff = 1.0e-10, normalize_tensors = false, env_maxdim = 9)

    f_bp = inner(ψ_trunc_bp, ψ; alg = "exact") / sqrt(norm_sqr(ψ_trunc_bp; alg = "exact") * norm_sqr(ψ; alg = "exact"))
    f_ctmrg = inner(ψ_trunc_ctmrg, ψ; alg = "exact") / sqrt(norm_sqr(ψ_trunc_ctmrg; alg = "exact") * norm_sqr(ψ; alg = "exact"))

    f_bp = real(f_bp * conj(f_bp))
    f_ctmrg = real(f_ctmrg * conj(f_ctmrg))

    @test 0 ≤ f_bp ≤ 1
    @test 0 ≤ f_ctmrg ≤ 1
    @test f_ctmrg ≥ f_bp
    @test maxvirtualdim(ψ_trunc_bp) ≤ 2
    @test maxvirtualdim(ψ_trunc_ctmrg) ≤ 2

end

end
