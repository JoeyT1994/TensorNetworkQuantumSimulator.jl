@eval module $(gensym())
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws
const TNQS = TensorNetworkQuantumSimulator

@testset "3D CTMRG on cubic grids" begin
    # Dispatch and argument checks: a network on (x, y, z) vertices gets the cubic engine.
    tn2 = ising_partitionfunction(named_grid((2, 2, 2)), 0.2)
    @test CTMEnvironmentCache(tn2, 2) isa CTM3DEnvironmentCache
    @test_throws ArgumentError CTMEnvironmentCache(tn2, 2; plane_rest = :nonsense)
    @test_throws ArgumentError CTMEnvironmentCache(tn2, 2; projector = :cycle, plane_rest = :exact)
    @test_throws ErrorException CTMEnvironmentCache(ising_partitionfunction(named_grid((3, 3, 1)), 0.2), 2)

    # 1. Exact at lossless χ, where every pair is a full-rank isometry, on boxes whose widest face
    #    fits: 2×2×2 at χ = 2, 2×2×3 at 4, 3×3×3 at 16 (its largest quarter-plane has 2×2 bonds).
    for (dims, χ) in (((2, 2, 2), 2), ((2, 2, 3), 4), ((3, 3, 3), 16))
        tn = ising_partitionfunction(named_grid(dims), 0.2)
        lnZ = log(abs(contract(tn; alg = "exact")))
        for projector in (:cut, :cycle)
            @test abs(cvm_freenergy(update(CTMEnvironmentCache(tn, χ; projector))) - lnZ) < 1.0e-10
        end
    end
    # a signed, non-symmetric network: nothing may lean on the Ising model's symmetries
    Random.seed!(7)
    tnr = random_tensornetwork(Float64, named_grid((2, 3, 3)); bond_dimension = 2)
    @test abs(cvm_freenergy(update(CTMEnvironmentCache(tnr, 16))) - log(abs(contract(tnr; alg = "exact")))) < 1.0e-10

    # 2. Truncated: the 3×3×3 Ising box at χ = 2 and 3 (lossless needs 16). Measured 2026-09-24:
    #    |ΔF| 6.9e-9 / 1.1e-12 under :cut and 1.9e-7 / 2.7e-9 under :cycle; the bounds leave ~15×.
    tn3 = ising_partitionfunction(named_grid((3, 3, 3)), 0.2)
    lnZ3 = log(abs(contract(tn3; alg = "exact")))
    err(χ, projector) = abs(cvm_freenergy(update(CTMEnvironmentCache(tn3, χ; projector); maxiter = 15)) - lnZ3)
    @test err(2, :cut) < 1.0e-7
    @test err(3, :cut) < 1.0e-10
    @test err(2, :cycle) < 3.0e-6
    @test err(3, :cycle) < 5.0e-8
end

@testset "Infinite 3D CTMRG: exact limits" begin
    # A 1D chain (only Jx): ln κ = ln 2cosh K exactly, any χ.
    site, legs, _ = ising3d_site(0.4; J = (1.0, 0.0, 0.0))
    ic = update(InfiniteCTM3D(site, legs, 2); maxiter = 20)
    @test abs(cvm_freenergy(ic) - log(2cosh(0.4))) < 1.0e-12
    # Decoupled 2D layers (Jz = 0) against Onsager: measured 1.05e-6 at χ = 2, 1.3e-10 at χ = 4
    # (K = 0.3), both projectors identical there.
    K = 0.3
    n = 400
    onsager = log(2) + sum(log(cosh(2K)^2 - sinh(2K) * (cos(2π * i / n) + cos(2π * j / n)))
                           for i in 0:(n - 1), j in 0:(n - 1)) / (2 * n^2)
    site, legs, _ = ising3d_site(K; J = (1.0, 1.0, 0.0))
    @test abs(cvm_freenergy(update(InfiniteCTM3D(site, legs, 2); maxiter = 30)) - onsager) < 5.0e-6
    @test abs(cvm_freenergy(update(InfiniteCTM3D(site, legs, 4); maxiter = 30)) - onsager) < 1.0e-8
    # the argument checks
    @test_throws ArgumentError InfiniteCTM3D(site, legs, 2; plane_rest = :exact, projector = :cycle)
    @test_throws ArgumentError InfiniteCTM3D(site, legs[1:5], 2)
end
end
