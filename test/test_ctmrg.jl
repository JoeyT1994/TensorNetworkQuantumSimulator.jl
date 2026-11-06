@eval module $(gensym())
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: freenergy, map_tensors
using Random
using Random: Xoshiro
using LinearAlgebra: Diagonal, I, norm, opnorm, eigvals
using Test: @testset, @test, @test_nowarn

# Positive random tensors, so ln Z is real and the exact contraction is the reference.
function positive_network(g; D = 2, T = Float64)
    tn = random_tensornetwork(T, g; bond_dimension = D)
    return map_tensors(t -> abs.(t), tn)
end

# Triangular faces and trivalent vertices use pseudo-inverses of single corners, well conditioned
# here; kagome 3×3 and Penrose also need directional corners, whose regularised pseudo-inverses of
# corner chains set a floor of about 1e-9 on lossless results.
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
    @test contract(tn; alg = "ctmrg", env_maxdim = 256) ≈ Z rtol = tol
end

@testset "CTMRG below lossless χ: the error shrinks with χ" begin
    tn = ising_partitionfunction(named_grid((6, 6)), 0.44)
    lnZ = log(contract(tn; alg = "exact"))
    errs = [abs(freenergy(update(CTMRGCache(tn, χ))) + lnZ) for χ in (2, 8)]
    @test errs[2] < errs[1]
    @test errs[2] < 1.0e-6
end

@testset "CTMRG at χ = 1 (:eig) is BP: its free energy is the Bethe free energy" begin
    Random.seed!(5)
    tn = positive_network(named_grid((4, 4)))
    bp = update(BeliefPropagationCache(tn); maxiter = 500, tolerance = 1.0e-15)
    ct = update(CTMRGCache(tn, 1; projector = :eig); maxiter = 500, tolerance = 1.0e-14)
    @test freenergy(ct) ≈ freenergy(bp) rtol = 1.0e-10
end

@testset "CTMRG in single precision converges at its default tolerance" begin
    Random.seed!(6)
    tn = positive_network(named_grid((4, 3)); T = Float32)
    @test_nowarn update(CTMRGCache(tn, 64))
    @test_nowarn update(CTMRGCache(tn, 64); convergence = :freenergy)
end

@testset "The :eig ring solver: two-sided invariant subspaces, N = $N (dense below 400, Krylov above)" for N in (64, 600)
    rng = Xoshiro(N)
    # a ring of 4 blocks with a gap after its 5 dominant eigenvalues
    D = Diagonal([[4.0, 3.5, 3.0, 2.5, 2.0]; 0.5 .* rand(rng, N - 5)])
    S = randn(rng, N, N)
    R = S * D / S
    mats = [R, Matrix{Float64}(I, N, N), Matrix{Float64}(I, N, N), Matrix{Float64}(I, N, N)]
    V, W = TensorNetworkQuantumSimulator.two_sided_bases(mats, 5, N * eps() * opnorm(R))
    @test size(V, 2) == size(W, 2) == 5
    @test norm(R * V - V * (V' * R * V)) < 1.0e-8 * opnorm(R)
    @test norm(transpose(R) * W - W * (W' * transpose(R) * W)) < 1.0e-8 * opnorm(R)
    @test sort(real.(eigvals((transpose(W) * V) \ (transpose(W) * R * V)))) ≈ [2.0, 2.5, 3.0, 3.5, 4.0]
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
    @test inner(ψ, ϕ; alg = "ctmrg", env_maxdim = 64) ≈ inner(ψ, ϕ; alg = "exact") rtol = 1.0e-8
end
end
