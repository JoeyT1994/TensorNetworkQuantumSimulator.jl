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
    # fixed-spin product state: m = 0.750930 against Talapov–Blöte's 0.750925 (3D CTMRG: 0.7575
    # at χ = 8), f = 0.8214065 (a lower bound; 3D CTMRG's χ = 4 value 0.8213841 is below it).
    site, legs, mag = ising3d_site(0.25)
    bp = boundary_peps(site, legs, 2; maxdim = 16, boundary = [1.0, 0.0], maxiter = 60)
    @test issorted(bp.history)
    @test abs(real(site_ratio(bp, mag)) - 0.750925) < 2.0e-4
    @test cvm_freenergy(bp) > 0.82140

    # THE STATIONARY (bilinear) boundary PEPS in an imaginary field, exact at D = 1, continued from
    # the real maximiser at θ = 0 with secant predictors. Measured 2026-09-26:
    # chains along z (K = 0.3): f = ln λ₁ to 2.8e-15 and m to 1.1e-9 (relative) down to 1e-3 from the
    # edge, and the edge sin θ_c = e^{−2K} from m⁻² → 0 (σ = −1/2 in 1D) to 1.8e-6;
    # planes (J_z = 0): f equals the 2D Ising ln κ at the same field to 3e-15.
    function continuation(β, J, θs; χ)
        _, lg, _ = ising3d_site(β; J)
        on(θ) = (x = ising3d_site(β; J, h = im * θ / β);
                 (TNQS.replaceinds(x[1], collect(x[2]), collect(lg)), TNQS.replaceinds(x[3], collect(x[2]), collect(lg))))
        s0 = ising3d_site(β; J)
        bp = boundary_peps(TNQS.replaceinds(s0[1], collect(s0[2]), collect(lg)), lg, 1; maxdim = χ, gtol = 1.0e-11, maxiter = 100)
        out = []; prev = bp; prevA = nothing; θp = 0.0; θ0 = 0.0; Jac = nothing
        for θ in θs
            st, mg = on(θ)
            A0 = isnothing(prevA) ? prev.A : prev.A + (prev.A - prevA) * ((θ - θ0) / (θ0 - θp))
            res, info = boundary_peps_stationary(st, lg, prev; maxdim = χ, A0, jacobian = Jac, tol = 1.0e-11)
            push!(out, (θ, cvm_freenergy(res), site_ratio(res, mg), info))
            info.converged || break
            prevA = prev.A; θp = θ0; θ0 = θ; prev = res; Jac = info.J
        end
        return out
    end
    let β = 0.3, K = 0.3
        θc = asin(exp(-2K))
        exact(θ) = (H = im * θ; s = sqrt(exp(2K) * sinh(H)^2 + exp(-2K)); λ = exp(K) * cosh(H) + s;
                    (log(λ), (exp(K) * sinh(H) + exp(2K) * sinh(H) * cosh(H) / s) / λ))
        vs = [0.9, 0.5, 0.25, 0.13, 0.065, 0.032, 0.016, 0.008, 0.004, 0.002, 0.001]
        out = continuation(β, (0.0, 0.0, 1.0), θc .* (1 .- vs); χ = 4)
        @test length(out) == length(vs) && all(o -> o[4].converged, out)
        @test maximum(abs(o[2] - real(exact(o[1])[1])) for o in out) < 1.0e-12
        @test maximum(abs(o[3] - exact(o[1])[2]) / abs(o[3]) for o in out) < 1.0e-8   # 1.1e-9 at v = 1e-3 (damped solver)
        a, b = out[end - 1], out[end]
        θe = b[1] + abs(b[3])^-2 * (b[1] - a[1]) / (abs(a[3])^-2 - abs(b[3])^-2)
        @test abs(θe - θc) < 1.0e-5
    end
    let β = 0.35
        out = continuation(β, (1.0, 1.0, 0.0), [0.004, 0.02]; χ = 16)
        for (θ, f, m, info) in out
            s2, l2, m2 = ising2d_site(β; h = im * θ / β)
            ic = update(InfiniteCTM2D(s2, l2, 16); tolerance = 1.0e-12, maxiter = 3000)
            @test abs(f - cvm_freenergy(ic)) < 1.0e-12
            @test abs(m - site_ratio(ic, m2)) < 1.0e-10
        end
    end

    # THE PROJECTED POWER METHOD (MP-BP bond projectors): exact where the dominant eigenvector is a
    # product (chains along z: ln 2cosh K to 1e-16 in 3 steps); on 3D Ising it converges fast (β = 0.25,
    # D = 2: 30 steps, ~7 s) but to a BIASED fixed point — bond-local truncation ignores the plane's
    # loops: f 4.7e-6 below the variational optimum, m = 0.75386 against 0.75093, and ∇f ≠ 0 there
    # (|g||c| = 3.7e-3). Measured 2026-09-26; at β = 0.22 it even orders (m = 0.40) where the D = 2
    # optimum is disordered.
    let K = 0.4
        s, l, _ = ising3d_site(K; J = (0.0, 0.0, 1.0))
        bp, info = boundary_peps_power(s, l, 1; maxdim = 4, tol = 1.0e-12)
        @test info.converged && abs(cvm_freenergy(bp) - log(2cosh(K))) < 1.0e-13
    end
    let β = 0.25
        s, l, m = ising3d_site(β)
        bp, info = boundary_peps_power(s, l, 2; maxdim = 16, boundary = [1.0, 0.0], tol = 1.0e-8)
        @test info.converged && info.asym < 1.0e-12
        @test -1.0e-5 < cvm_freenergy(bp) - 0.8214064836 < 0          # below the variational optimum
        @test 2.0e-3 < abs(site_ratio(bp, m)) - 0.7509291 < 4.0e-3     # the documented bias
        @test info.gnorm > 1.0e-3                                      # not stationary for f
    end

    # argument checks
    @test_throws ArgumentError boundary_peps(site, legs, 0; maxdim = 4)
    @test_throws ArgumentError boundary_peps(site, legs[1:5], 2; maxdim = 4)
    s2, l2, _ = ising3d_site(0.25; J = (1.0, 0.5, 1.0))           # not C4v-invariant in x, y
    @test_throws ArgumentError boundary_peps(s2, l2, 2; maxdim = 4)
end
end
