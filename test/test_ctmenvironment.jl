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
end
