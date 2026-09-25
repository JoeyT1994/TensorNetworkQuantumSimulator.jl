@eval module $(gensym())
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws
const TNQS = TensorNetworkQuantumSimulator

onsager(K; n = 400) = log(2) + sum(log(cosh(2K)^2 - sinh(2K) * (cos(2π * i / n) + cos(2π * j / n)))
                                   for i in 0:(n - 1), j in 0:(n - 1)) / (2 * n^2)
# a D = 1 boundary tensor with physical vector v
product_A(v) = (is = [TNQS.new_index(1) for _ in 1:4]; p = TNQS.new_index(length(v));
                TNQS.from_array(reshape(collect(Float64, v), 1, 1, 1, 1, length(v)), is..., p))

@testset "Boundary PEPS for 3D models" begin
    # Exact limits, where the dominant eigenvector of the layer transfer operator is a product
    # state: decoupled layers (Jz = 0) give Onsager, chains along z give ln 2cosh K. Measured
    # 2026-09-25: 4.4e-15 and 2.2e-16.
    site, legs, _ = ising3d_site(0.3; J = (1.0, 1.0, 0.0))
    bp = boundary_peps(site, legs, 1; maxdim = 8, init = product_A([1, 1] / sqrt(2)), maxiter = 0)
    @test abs(cvm_freenergy(bp) - onsager(0.3)) < 1.0e-12
    site, legs, _ = ising3d_site(0.4; J = (0.0, 0.0, 1.0))
    bp = boundary_peps(site, legs, 1; maxdim = 4, init = product_A([1, 1] / sqrt(2)), maxiter = 0)
    @test abs(cvm_freenergy(bp) - log(2cosh(0.4))) < 1.0e-12

    # The gradient is two one-site environments per network (ket and bra layer), no sum over
    # positions: against a 4-point finite difference along a random symmetric direction, 3D Ising
    # β = 0.22, D = 2, χ = 16 (measured 4.2e-13), and orthogonal to A (scale invariance).
    site, legs, mag = ising3d_site(0.22)
    al = Tuple(TNQS.new_index(2) for _ in 1:5)
    bl = Tuple(TNQS.new_index(2) for _ in 1:4)
    ctx = (; al, bl, site, legs = Tuple(legs), maxdim = 16, symmetrize = true, ctm_tolerance = 1.0e-12,
           ctm_maxiter = 2000, ctm_kwargs = NamedTuple())
    rng = Xoshiro(1)
    A = TNQS._bp_initial(site, legs, al, 2, [1.0, 0.0], 0.0, rng) + 0.1 * TNQS.random_tensor(rng, Float64, collect(al))
    A = TNQS._bp_symmetrize(A, al[1:4]); A = A / TNQS.norm(A)
    dA = TNQS._bp_symmetrize(TNQS.random_tensor(rng, Float64, collect(al)), al[1:4]); dA = dA / TNQS.norm(dA)
    f0, G, ln, ls = TNQS._bp_evaluate(A, ctx, nothing, nothing)
    h = 1.0e-3
    fs = Dict(k => TNQS._bp_evaluate(A + k * h * dA, ctx, ln, ls)[1] for k in (-2, -1, 1, 2))
    fd = (-fs[2] + 8fs[1] - 8fs[-1] + fs[-2]) / (12h)
    @test abs(real(TNQS.dot(G, dA)) - fd) < 1.0e-9
    @test abs(real(TNQS.dot(G, A))) < 1.0e-12

    # 3D Ising in the ordered phase, D = 2, χ = 16, 60 L-BFGS iterations from T applied to a
    # fixed-spin product state: m = 0.750930 against Talapov–Blöte's 0.750886 (3D CTMRG: 0.7575
    # at χ = 8), f = 0.8214065 (a lower bound; 3D CTMRG's χ = 4 value 0.8213841 is below it).
    site, legs, mag = ising3d_site(0.25)
    bp = boundary_peps(site, legs, 2; maxdim = 16, boundary = [1.0, 0.0], maxiter = 60)
    @test issorted(bp.history)
    @test abs(real(site_ratio(bp, mag)) - 0.750886) < 2.0e-4
    @test cvm_freenergy(bp) > 0.82140

    # argument checks
    @test_throws ArgumentError boundary_peps(site, legs, 0; maxdim = 4)
    @test_throws ArgumentError boundary_peps(site, legs[1:5], 2; maxdim = 4)
    s2, l2, _ = ising3d_site(0.25; J = (1.0, 0.5, 1.0))           # not C4v-invariant in x, y
    @test_throws ArgumentError boundary_peps(s2, l2, 2; maxdim = 4)
end
end
