@eval module $(gensym())
using Dictionaries: Dictionary
using TensorNetworkQuantumSimulator
using Test: @testset, @test
const TNQS = TensorNetworkQuantumSimulator

@testset "CTMEnvironmentCache" begin
    # Anisotropic (and isotropic, square and non-square) finite Ising partition
    # functions vs the library's exact contraction. χ large enough to be lossless.
    for (Lx, Ly, Kx, Ky) in [(5, 5, 0.3, 0.3), (5, 5, 0.3, 0.6), (4, 5, 0.3, 0.6)]
        g = named_grid((Lx, Ly))
        es = collect(edges(g))
        Js = Dictionary(es, [(src(e)[2] == dst(e)[2]) ? Kx : Ky for e in es])   # Kx horiz, Ky vert
        tn = ising_partitionfunction(g, 1.0; Js)
        z_exact = contract(tn; alg = "exact")

        cache = CTMEnvironmentCache(tn, 40)
        @test partitionfunction(cache) ≈ z_exact
        @test TNQS.freenergy(cache) ≈ log(z_exact)

        # row-environment sandwich (plain middle row) must also reproduce Z
        y = 3
        top, bot = row_environments(cache, y)
        midrow = [tn[(x, y)] for x in 1:Lx]
        @test contract_row(top, midrow, bot) ≈ z_exact
    end

    # χ-convergence: near-critical, too big to be lossless → error shrinks with χ
    g = named_grid((10, 10))
    es = collect(edges(g))
    Js = Dictionary(es, [0.44 for _ in es])
    tn = ising_partitionfunction(g, 1.0; Js)
    z_exact = contract(tn; alg = "exact")
    errs = [abs(partitionfunction(CTMEnvironmentCache(tn, χ)) - z_exact) for χ in (4, 16)]
    @test errs[2] < errs[1]
    @test errs[2] / abs(z_exact) < 1.0e-6
end

@testset "CTMEnvironmentCache double-layer (state norm)" begin
    # ⟨ψ|ψ⟩ of a PEPS folds to a norm network (bond dim D²); exact when χ is lossless.
    for (Lx, Ly) in [(3, 3), (4, 3)]
        g = named_grid((Lx, Ly))
        ψ = random_tensornetworkstate(Float64, g; bond_dimension = 2)
        @test partitionfunction(CTMEnvironmentCache(ψ, 40)) ≈ norm_sqr(ψ; alg = "exact")
    end
    # D=3 on 3×3 overflows small χ → error must shrink with χ
    ψ = random_tensornetworkstate(Float64, named_grid((3, 3)); bond_dimension = 3)
    nrm = norm_sqr(ψ; alg = "exact")
    errs = [abs(partitionfunction(CTMEnvironmentCache(ψ, χ)) - nrm) for χ in (4, 27)]
    @test errs[2] < errs[1]
end

@testset "CVM per-vertex environments" begin
    # Non-square as well as square: the `:S`/`:E` block families are keyed by their first
    # included row/column, so an off-by-one there leaves regions unclosable at the boundary.
    for (Lx, Ly) in [(3, 3), (4, 3)]
        tn = random_tensornetwork(Float64, named_grid((Lx, Ly)); bond_dimension = 2)
        lnZ = log(abs(real(contract(tn; alg = "exact"))))
        cache = update(CTMEnvironmentCache(tn, 100); maxiter = 3)
        @test !isnothing(environments(cache))       # `update` stores them on the cache

        # 1. lossless limit: EVERY region type contracts to the exact Z — interior, edge and
        # corner vertices, boundary edge strips, corner plaquettes.
        for cx in 1:0.5:Lx, cy in 1:0.5:Ly
            @test region_lnZ(cache, cx, cy) ≈ lnZ atol = 1.0e-8
        end
        # 2. Mobius identity: V − E + P = 1, so the weighted sum returns ln Z.
        @test cvm_freenergy(cache) ≈ lnZ atol = 1.0e-8
    end

    # Double layer, kept lazy: the sweep is generic over `bp_factors`, so ⟨ψ|ψ⟩ and the
    # equivalent QuadraticForm must both close their regions and agree.
    ψ = random_tensornetworkstate(Float64, named_grid((3, 3)); bond_dimension = 2)
    lnN = log(abs(real(norm_sqr(ψ; alg = "exact"))))
    for net in (ψ, QuadraticForm(ψ))
        cache = update(CTMEnvironmentCache(net, 200); maxiter = 2)
        @test cvm_freenergy(cache) ≈ lnN atol = 1.0e-8
        @test region_lnZ(cache, 2, 2) ≈ lnN atol = 1.0e-8      # interior vertex
        @test region_lnZ(cache, 1.5, 1.5) ≈ lnN atol = 1.0e-8  # corner plaquette
    end

    # The two-sided sweep must improve on the greedy one-sided pass and be monotone in χ.
    tn = random_tensornetwork(Float64, named_grid((4, 4)); bond_dimension = 3)
    lnZ = log(abs(real(contract(tn; alg = "exact"))))
    swept = Float64[]
    for χ in (6, 8)
        fresh = CTMEnvironmentCache(tn, χ)
        cache = update(fresh; maxiter = 30, tol = 1.0e-11)
        F = cvm_freenergy(cache)
        push!(swept, abs(F - lnZ))
        # An un-updated cache falls back to the greedy one-sided pass, which the sweep beats.
        @test swept[end] < abs(cvm_freenergy(fresh) - lnZ)
        # Stationary: sweeping the converged cache again barely moves F.
        @test cvm_freenergy(update(cache; maxiter = 2)) ≈ F atol = 1.0e-7
    end
    @test swept[2] < swept[1]
end
end
