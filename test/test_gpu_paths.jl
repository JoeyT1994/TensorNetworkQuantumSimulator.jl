@eval module $(gensym())
using Test: @test, @testset
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using LinearAlgebra: norm
using Random: Random
using Adapt: adapt

# GPU-path validation on real hardware: the same dense state on host and device through BP, exact
# contraction, gate application (both the copying and the consuming entry points), CTM (`:cut`,
# `:cycle`) and boundary MPS must agree to roundoff. Scalar indexing is disallowed so any silent
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

        for kw in ((;), (; projector = :cycle))
            ckw = haskey(kw, :projector) ? (; convergence = :marginal) : (;)
            c = TNQS.update(CTMEnvironmentCache(ψ, 16; kw...); maxiter = 6, tolerance = 1.0e-8, ckw...)
            cg = TNQS.update(CTMEnvironmentCache(ψg, 16; kw...); maxiter = 6, tolerance = 1.0e-8, ckw...)
            @test z(expect(cg, obs)) ≈ z(expect(c, obs)) atol = 1.0e-10
        end

        # boundary MPS at a lossless χ (D² = 9 per bond, two bonds per cut on a 4-row column → 81)
        @test z(expect(ψg, obs; alg = "boundarymps", mps_bond_dimension = 81)) ≈
            z(expect(ψ, obs; alg = "boundarymps", mps_bond_dimension = 81)) atol = 1.0e-10

        # ComplexF32 through `cu`, the example's route
        ψ32 = CUDA.cu(ψ)
        @test eltype(TNQS.TensorInterface.data(ψ32[(1, 1)])) == ComplexF32
        @test z(expect(ψ32, obs; alg = "bp")) ≈ z(expect(ψ, obs; alg = "bp")) atol = 1.0e-4
    end
end
end
