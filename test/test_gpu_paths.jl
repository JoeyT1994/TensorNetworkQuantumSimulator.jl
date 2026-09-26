@eval module $(gensym())
using Test: @test, @testset
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using LinearAlgebra: norm
using Random: Random
using Adapt: adapt

# GPU-path validation on real hardware: the same dense state on host and device through BP, exact
# contraction, gate application (both the copying and the consuming entry points), CTM (`:cut`,
# `:cycle`), boundary MPS, and the thermodynamic-limit engines (`InfiniteCTM2D`, the 3D boundary
# PEPS solvers) must agree to roundoff. Scalar indexing is disallowed so any silent
# host round-trip inside a device path throws. Skipped when CUDA is not functional (CI without a GPU).
const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end
HAS_CUDA || @info "CUDA not functional: skipping GPU-path checks"

if HAS_CUDA
    CUDA.allowscalar(false)
    @testset "GPU paths (CUDA)" begin
        Random.seed!(3)
        g = named_grid((4, 4))
        s = siteinds("S=1/2", g)
        ψ = random_tensornetworkstate(ComplexF64, g, s; bond_dimension = 3)
        ψg = adapt(CuArray, ψ)
        @test TNQS.TensorInterface.data(ψg[(1, 1)]) isa CuArray
        obs = ("Z", [(2, 2)])
        layer = Any[("Rzz", (src(e), dst(e)), 0.3) for e in edges(g)]
        append!(layer, ("Rx", [v], 0.2) for v in vertices(g))
        apply_kwargs = (; maxdim = 4, cutoff = 1.0e-12)

        z(x) = real(only(x))
        @test z(expect(ψg, obs; alg = "bp")) ≈ z(expect(ψ, obs; alg = "bp")) atol = 1.0e-12
        @test z(expect(ψg, obs; alg = "exact")) ≈ z(expect(ψ, obs; alg = "exact")) atol = 1.0e-12

        ψ2, _ = apply_gates(layer, ψ; apply_kwargs)
        ψ2g, _ = apply_gates(layer, ψg; apply_kwargs)
        @test z(expect(ψ2g, obs; alg = "bp")) ≈ z(expect(ψ2, obs; alg = "bp")) atol = 1.0e-12
        @test TNQS.TensorInterface.data(ψ2g[(2, 2)]) isa CuArray             # stays on the device
        # the consuming entry point (fused kernel + consumed destinations on device storage)
        bpc = BeliefPropagationCache(deepcopy(ψ)); bpcg = BeliefPropagationCache(deepcopy(ψg))
        bpc, _ = apply_gates!(layer, bpc; apply_kwargs, update_cache = true)
        bpcg, _ = apply_gates!(layer, bpcg; apply_kwargs, update_cache = true)
        @test z(expect(bpcg, obs)) ≈ z(expect(bpc, obs)) atol = 1.0e-12

        # CTM host/device agreement is checked on a LOSSLESS case (D = 2 at χ = 16 on a 4×4). The D = 3
        # state at χ = 16 is under-truncated and sits on a rank-decision threshold of `:cycle` that
        # roundoff (threaded BLAS, device reductions) tips either way: the same two answers, 5e-6
        # apart, were observed with host and device SWAPPED between runs (2026-09-17). That is a
        # property of the regime, not of the device path this test guards.
        ψ2d = random_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)
        ψ2dg = adapt(CuArray, ψ2d)
        for kw in ((;), (; projector = :cycle))
            ckw = haskey(kw, :projector) ? (; convergence = :marginal) : (;)
            c = TNQS.update(CTMEnvironmentCache(ψ2d, 16; kw...); maxiter = 6, tolerance = 1.0e-8, ckw...)
            cg = TNQS.update(CTMEnvironmentCache(ψ2dg, 16; kw...); maxiter = 6, tolerance = 1.0e-8, ckw...)
            @test z(expect(cg, obs)) ≈ z(expect(c, obs)) atol = 1.0e-10
            @test z(expect(cg, obs)) ≈ z(expect(ψ2d, obs; alg = "exact")) atol = 1.0e-8
        end

        # boundary MPS at a lossless χ (D² = 9 per bond, two bonds per cut on a 4-row column → 81)
        @test z(expect(ψg, obs; alg = "boundarymps", mps_bond_dimension = 81)) ≈
            z(expect(ψ, obs; alg = "boundarymps", mps_bond_dimension = 81)) atol = 1.0e-10

        # classical networks in the thermodynamic limit. 2D Ising in an imaginary field (complex data,
        # the c4v symmetric pair) and its correlation length:
        s2, l2, m2 = ising2d_site(0.4; h = im * 0.01)
        ic = TNQS.update(InfiniteCTM2D(s2, l2, 8; c4v = true); tolerance = 1.0e-10)
        icg = TNQS.update(InfiniteCTM2D(adapt(CuArray, s2), l2, 8; c4v = true); tolerance = 1.0e-10)
        @test abs(site_ratio(icg, adapt(CuArray, m2)) - site_ratio(ic, m2)) < 1.0e-9
        @test abs(first(correlation_length(icg)) - first(correlation_length(ic))) < 1.0e-6
        # the 3D boundary PEPS, a host state moved with `adapt`: L-BFGS and Newton–Krylov steps agree
        s3, l3, m3 = ising3d_site(0.25)
        bp0 = boundary_peps(s3, l3, 2; maxdim = 8, boundary = [1.0, 0.0], maxiter = 10)
        bp0g = adapt(CuArray, bp0)
        @test TNQS.TensorInterface.data(bp0g.A) isa CuArray
        b1 = boundary_peps(s3, l3, 2; maxdim = 8, init = bp0, maxiter = 3)
        b1g = boundary_peps(adapt(CuArray, s3), l3, 2; maxdim = 8, init = bp0g, maxiter = 3)
        @test abs(cvm_freenergy(b1g) - cvm_freenergy(b1)) < 1.0e-10
        k1, _ = boundary_peps_krylov(s3, l3, bp0; maxiter = 3)
        k1g, _ = boundary_peps_krylov(adapt(CuArray, s3), l3, bp0g; maxiter = 3)
        @test abs(cvm_freenergy(k1g) - cvm_freenergy(k1)) < 1.0e-10
        @test abs(site_ratio(k1g, m3) - site_ratio(k1, m3)) < 1.0e-8
        # … and the complex (Levenberg–Marquardt) path: chains along z in an imaginary field
        sr, lr, _ = ising3d_site(0.3; J = (0.0, 0.0, 1.0))
        bpr = boundary_peps(sr, lr, 1; maxdim = 4, maxiter = 50)
        x = ising3d_site(0.3; J = (0.0, 0.0, 1.0), h = im * 0.1)
        sc = TNQS.replaceinds(x[1], collect(x[2]), collect(lr)); mc = TNQS.replaceinds(x[3], collect(x[2]), collect(lr))
        kc, _ = boundary_peps_krylov(sc, lr, bpr; tol = 1.0e-10)
        kcg, _ = boundary_peps_krylov(adapt(CuArray, sc), lr, adapt(CuArray, bpr); tol = 1.0e-10)
        @test abs(site_ratio(kcg, mc) - site_ratio(kc, mc)) < 1.0e-9

        # ComplexF32 through `cu`, the example's route
        ψ32 = CUDA.cu(ψ)
        @test eltype(TNQS.TensorInterface.data(ψ32[(1, 1)])) == ComplexF32
        @test z(expect(ψ32, obs; alg = "bp")) ≈ z(expect(ψ, obs; alg = "bp")) atol = 1.0e-4
    end
end
end
