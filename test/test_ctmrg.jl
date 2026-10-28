@eval module $(gensym())
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: freenergy, map_tensors
using Random
using Test: @testset, @test

# Positive random tensors, so ln Z is real and the exact contraction is the reference.
function positive_network(g; D = 2)
    tn = random_tensornetwork(Float64, g; bond_dimension = D)
    return map_tensors(t -> abs.(t), tn)
end

# kagome 3×3 and Penrose need directional corners, whose regularised pseudo-inverses set a floor
# of about 1e-9 on lossless results; the others are exact to roundoff.
@testset "CTMRG is exact at lossless χ: $label" for (label, g, tol) in [
        ("square", named_grid((4, 3)), 1.0e-8),
        ("honeycomb", named_hexagonal_lattice_graph(2, 2), 1.0e-8),
        ("kagome", kagome_lattice(2, 2), 1.0e-8),
        ("triangular", named_triangular_lattice_graph(3, 4), 1.0e-8),
        ("kagome 3×3", kagome_lattice(3, 3), 1.0e-6),
        ("Penrose", penrose_tiling(2), 1.0e-6),
    ]
    Random.seed!(3)
    tn = positive_network(g)
    Z = contract(tn; alg = "exact")
    @test freenergy(update(CTMRGCache(tn, 256))) ≈ -log(Z) atol = tol
    @test contract(tn; alg = "ctmrg", maxdim = 256) ≈ Z rtol = tol
end

@testset "CTMRG below lossless χ: the error shrinks with χ" begin
    tn = ising_partitionfunction(named_grid((6, 6)), 0.44)
    lnZ = log(contract(tn; alg = "exact"))
    errs = [abs(freenergy(update(CTMRGCache(tn, χ))) + lnZ) for χ in (2, 8)]
    @test errs[2] < errs[1]
    @test errs[2] < 1.0e-6
end

@testset "CTMRG observables on a state" begin
    Random.seed!(7)
    g = named_hexagonal_lattice_graph(2, 2)
    ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 2)
    v, e = first(vertices(g)), first(edges(g))
    obs = [("Z", [v]), ("X", [v]), ("ZZ", [src(e), dst(e)]), ("XY", [src(e), dst(e)])]
    exact = expect(ψ, obs; alg = "exact")
    cache = update(CTMRGCache(ψ, 64))
    @test expect(cache, obs) ≈ exact atol = 1.0e-8
    @test expect(update(CTMRGCache(ψ, 64; projector = :eig)), obs) ≈ exact atol = 1.0e-8
    @test norm_sqr(cache; alg = "ctmrg") ≈ norm_sqr(ψ; alg = "exact") rtol = 1.0e-8
    ϕ = random_tensornetworkstate(ComplexF64, g, siteinds(ψ); bond_dimension = 2)
    @test inner(ψ, ϕ; alg = "ctmrg", maxdim = 64) ≈ inner(ψ, ϕ; alg = "exact") rtol = 1.0e-8
end
end
