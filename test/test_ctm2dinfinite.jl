@eval module $(gensym())
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws
using Logging: NullLogger, with_logger
const TNQS = TensorNetworkQuantumSimulator

onsager(K; n = 400) = log(2) + sum(log(cosh(2K)^2 - sinh(2K) * (cos(2π * i / n) + cos(2π * j / n)))
                                   for i in 0:(n - 1), j in 0:(n - 1)) / (2 * n^2)
yang(K) = (1 - sinh(2K)^-4)^(1 / 8)

# The 2D Ising site at coupling K on the legs `legs` (every call of `ising2d_site` makes new ones).
function ising_on(K, legs)
    s, l, m = ising2d_site(K)
    return TNQS.replaceinds(s, collect(l), collect(legs)), TNQS.replaceinds(m, collect(l), collect(legs))
end

@testset "Infinite 2D CTMRG" begin
    # Disordered 2D Ising (K = 0.3) against Onsager. Measured 2026-09-25: 1.3e-10 at χ = 4 and
    # 4e-15 at χ = 8 in 15 iterations, either pair; ⟨σ⟩ zero to 1e-15.
    site, legs, mag = ising2d_site(0.3)
    @test abs(cvm_freenergy(update(InfiniteCTM2D(site, legs, 4))) - onsager(0.3)) < 1.0e-9
    ic = update(InfiniteCTM2D(site, legs, 8))
    @test abs(cvm_freenergy(ic) - onsager(0.3)) < 1.0e-12
    @test abs(real(site_ratio(ic, mag))) < 1.0e-12
    iso = update(InfiniteCTM2D(site, legs, 8; pair = :isometric))
    @test abs(cvm_freenergy(iso) - onsager(0.3)) < 1.0e-12

    # Ordered (K = 0.5) from a fixed-spin seed against Yang's spontaneous magnetisation:
    # 3.1e-11 at χ = 8 (the truncation); 1.2e-12 at χ = 16 converged to 1e-12, exact after 200 more.
    site, legs, mag = ising2d_site(0.5)
    ic = update(InfiniteCTM2D(site, legs, 16; boundary = [1.0, 0.0]); tolerance = 1.0e-12)
    @test ic.stats[].converged
    @test abs(real(site_ratio(ic, mag)) - yang(0.5)) < 1.0e-11
    @test abs(cvm_freenergy(ic) - onsager(0.5)) < 1.0e-12
    # a fixed point stays put: 200 more iterations
    st = ic.state
    for _ in 1:200
        st = TNQS._i2_step(ic, st)
    end
    @test abs(real(site_ratio(TNQS._i2_setstate(ic, st), mag)) - real(site_ratio(ic, mag))) < 1.0e-11
    # `init` restarts from a state: from the converged one it is done at once, and from a nearby
    # coupling's it lands on the cold start's fixed point
    @test update(InfiniteCTM2D(site, legs, 16; boundary = [1.0, 0.0], init = ic); tolerance = 1.0e-12).stats[].iterations <= 3
    s2, _ = ising_on(0.51, legs)
    cold = update(InfiniteCTM2D(s2, legs, 16; boundary = [1.0, 0.0]))
    warm = update(InfiniteCTM2D(s2, legs, 16; boundary = [1.0, 0.0], init = ic))
    @test warm.stats[].converged
    @test abs(cvm_freenergy(warm) - cvm_freenergy(cold)) < 1.0e-12

    # Layers are never fused, and need not be: the norm network of a random nonnegative D = 2 PEPS,
    # as two layers and as one fused site, gives the same ln κ (measured: identical digits).
    Random.seed!(3)
    l, r, u, d, p = (TNQS.new_index(2; tags = n) for n in ("l", "r", "u", "d", "p"))
    A = TNQS.from_array(rand(2, 2, 2, 2, 2), l, r, u, d, p)
    lp, rp, up, dp = TNQS.prime(l), TNQS.prime(r), TNQS.prime(u), TNQS.prime(d)
    Ab = TNQS.replaceinds(A, [l, r, u, d], [lp, rp, up, dp])
    layered = update(InfiniteCTM2D([A, Ab], ([l, lp], [r, rp], [d, dp], [u, up]), 12))
    cs = [TNQS.combiner([x, y]) for (x, y) in ((l, lp), (r, rp), (d, dp), (u, up))]
    fused = update(InfiniteCTM2D(A * Ab * cs[1] * cs[2] * cs[3] * cs[4], Tuple(TNQS.combinedind(c) for c in cs), 12))
    @test abs(cvm_freenergy(layered) - cvm_freenergy(fused)) < 1.0e-12

    # The gradient of ln κ from ONE site's environment, d ln κ = ⟨E, da⟩ / z, against a 4-point
    # finite difference in K (measured: 4.9e-11 at K = 0.3, 9.5e-10 at K = 0.5, χ = 16).
    for (K, bnd, tol) in ((0.3, nothing, 1.0e-9), (0.5, [1.0, 0.0], 1.0e-8))
        _, legs, _ = ising2d_site(K)
        h = 1.0e-3
        sites = Dict(k => first(ising_on(K + k * h, legs)) for k in -2:2)
        # tight environment convergence: the gradient is only as good as the environment
        run(s) = with_logger(NullLogger()) do
            update(InfiniteCTM2D(s, legs, 16; boundary = bnd); tolerance = 1.0e-12, maxiter = 2000)
        end
        f = Dict(k => cvm_freenergy(run(sites[k])) for k in -2:2)
        fd = (-f[2] + 8f[1] - 8f[-1] + f[-2]) / (12h)
        E, z = site_environment(run(sites[0]), 1)
        da = (-sites[2] + 8 * sites[1] - 8 * sites[-1] + sites[-2]) / (12h)
        @test abs(TNQS.scalar(E * da) / z - fd) < tol
    end

    # The split (matrix-free) pair against the dense pair on a boundary-PEPS sandwich (3D Ising
    # layer, D = 2, χ = 16, interface n = 128 — at the subspace gate): the same fixed point.
    # Measured 2026-09-25: Δ ln κ 7e-16, Δm 2e-15, ‖ΔE‖/‖E‖ 2e-15, 13 iterations both ways.
    let
        site3, legs3, mag3 = ising3d_site(0.23)
        al = Tuple(TNQS.new_index(2) for _ in 1:5); bl = Tuple(TNQS.new_index(2) for _ in 1:4)
        A = TNQS._bp_initial(site3, legs3, al, 2, [1.0, 0.0], 0.05, Xoshiro(1))
        A = TNQS._bp_symmetrize(A, al[1:4]); A = A / TNQS.norm(A)
        sl, slegs = TNQS._bp_sandwich_layers(A, al, bl, site3, legs3)
        empty!(TNQS.CTM_SVD_STATS)
        split = update(InfiniteCTM2D(sl, slegs, 16); tolerance = 1.0e-10, maxiter = 400)
        @test get(TNQS.CTM_SVD_STATS, :i2_split, 0) > 0
        dense = update(InfiniteCTM2D(sl, slegs, 16; svd = :dense); tolerance = 1.0e-10, maxiter = 400)
        @test abs(cvm_freenergy(split) - cvm_freenergy(dense)) < 1.0e-12
        imp = Any[sl[1], mag3, sl[3]]
        @test abs(site_ratio(split, imp) - site_ratio(dense, imp)) < 1.0e-10
        Es, zs = site_environment(split, 1); Ed, zd = site_environment(dense, 1)
        @test TNQS.norm(Es / zs - Ed / zd) < 1.0e-10 * TNQS.norm(Ed / zd)
    end

    # argument checks
    site, legs, _ = ising2d_site(0.3)
    @test_throws ArgumentError InfiniteCTM2D(site, legs, 4; pair = :nonsense)
    @test_throws ArgumentError InfiniteCTM2D(site, legs, 4; projector = :cycle)
    @test_throws ArgumentError InfiniteCTM2D(site, (legs[1], legs[2], legs[3]), 4)
    @test_throws ArgumentError InfiniteCTM2D(site, (legs[1], legs[2], legs[3], TNQS.new_index(3)), 4)   # y± dims differ
    @test_throws ArgumentError InfiniteCTM2D(site, (legs[1], legs[2], legs[3], TNQS.new_index(2)), 4)   # not the site's legs
    @test_throws ArgumentError InfiniteCTM2D(site, legs, 4; init = update(InfiniteCTM2D(site, legs, 6)))
    @test_throws ArgumentError site_environment(update(InfiniteCTM2D(site, legs, 4)), 2)
end
end
