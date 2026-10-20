@eval module $(gensym())
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: Dictionary, freenergy, map_tensors, ctmrg_rules
using Random
using Test: @testset, @test, @test_throws

# Positive random tensors, so ln Z is real and the exact contraction is the reference.
function positive_network(g; D = 2)
    tn = random_tensornetwork(Float64, g; bond_dimension = D)
    return map_tensors(t -> abs.(t), tn)
end

@testset "CTMRG is exact at lossless χ: $label" for (label, g) in [
        ("square", named_grid((4, 3))),
        ("honeycomb", named_hexagonal_lattice_graph(2, 2)),
        ("kagome", kagome_lattice(2, 2)),
        ("triangular", triangular_lattice(3, 3)),
    ]
    Random.seed!(3)
    tn = positive_network(g)
    lnZ = log(contract(tn; alg = "exact"))
    cache = update(CTMRGCache(tn, 256))
    @test freenergy(cache) ≈ lnZ atol = 1.0e-8
    @test contract(tn; alg = "ctmrg", maxdim = 256) ≈ exp(lnZ) rtol = 1.0e-8
end

@testset "CTMRG Ising free energy" begin
    # isotropic, anisotropic and non-square, at χ large enough to be lossless
    for (Lx, Ly, Kx, Ky) in [(5, 5, 0.3, 0.3), (5, 5, 0.3, 0.6), (4, 5, 0.3, 0.6)]
        g = named_grid((Lx, Ly))
        es = collect(edges(g))
        Js = Dictionary(es, [src(e)[2] == dst(e)[2] ? Kx : Ky for e in es])
        tn = ising_partitionfunction(g, 1.0; Js)
        @test freenergy(update(CTMRGCache(tn, 40))) ≈ log(contract(tn; alg = "exact")) atol = 1.0e-8
    end
    # near-critical and too large to be lossless: the error shrinks with χ
    tn = ising_partitionfunction(named_grid((6, 6)), 0.44)
    lnZ = log(contract(tn; alg = "exact"))
    errs = [abs(freenergy(update(CTMRGCache(tn, χ))) - lnZ) for χ in (2, 8)]
    @test errs[2] < errs[1]
    @test errs[2] < 1.0e-6
end

@testset "CTMRG update rules" begin
    # acyclic on the square lattice; the triangular lattice needs Jacobi passes
    @test !isnothing(ctmrg_rules(planar_embedding(named_grid((3, 3)))).order)
    @test isnothing(ctmrg_rules(planar_embedding(triangular_lattice(3, 3))).order)
end

@testset "CTMRG observables on a state" begin
    Random.seed!(7)
    g = named_hexagonal_lattice_graph(2, 2)
    ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 2)
    v = first(vertices(g))
    e = first(edges(g))
    obs = [("Z", [v]), ("X", [v]), ("ZZ", [src(e), dst(e)]), ("XY", [src(e), dst(e)])]
    exact = expect(ψ, obs; alg = "exact")
    @test expect(ψ, obs; alg = "ctmrg", maxdim = 64) ≈ exact atol = 1.0e-8

    cache = update(CTMRGCache(ψ, 64))
    @test expect(cache, ("Z", [v])) ≈ first(exact) atol = 1.0e-8
    @test partitionfunction(cache) ≈ norm_sqr(ψ; alg = "exact") rtol = 1.0e-8
    ρ = reduced_density_matrix(cache, [src(e), dst(e)])
    ρ_exact = reduced_density_matrix(ψ, [src(e), dst(e)]; alg = "exact")
    legs = TensorNetworkQuantumSimulator.inds(ρ_exact)
    @test TensorNetworkQuantumSimulator.array(ρ, legs...) ≈ TensorNetworkQuantumSimulator.array(ρ_exact, legs...) atol = 1.0e-8

    far = [w for w in vertices(g) if w != v && !(w in neighbors(g, v))]
    @test_throws ErrorException expect(cache, ("ZZ", [v, first(far)]))
end
end
