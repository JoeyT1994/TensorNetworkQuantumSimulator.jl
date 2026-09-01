@testset "fixed-storage local cycle CTM" begin
    TNQSLocal = TensorNetworkQuantumSimulator

    @test TNQSLocal._ctm_local_snake(3, 3) ==
          [(1, 1), (1, 2), (2, 2), (2, 1), (2, 2), (1, 2)]

    sites = [randn(2, 2, 2, 2) for _ in 1:3, _ in 1:3]
    S = TNQSLocal.CTMLocalCycleState(sites, 3)
    oldA = ntuple(k -> deepcopy(S.A[k]), 4)
    oldc = ntuple(k -> deepcopy(S.c[k]), 4)
    Ainner = ntuple(k -> (fill(Float64(10k + 1), 2, 3, 3),
                           fill(Float64(10k + 2), 2, 3, 3)), 4)
    cinner = ntuple(k -> fill(Float64(20k), 3, 3), 4)
    TNQSLocal._ctm_local_scatter!(S, Ainner, cinner, 1, 1)

    changedA = ntuple(4) do k
        Set((x, y) for x in 1:3, y in 1:3 if S.A[k][x, y] != oldA[k][x, y])
    end
    changedc = ntuple(4) do k
        Set((x, y) for x in 1:3, y in 1:3 if S.c[k][x, y] != oldc[k][x, y])
    end
    @test changedA == (Set([(1, 2), (1, 1)]), Set([(1, 1), (2, 1)]),
                       Set([(2, 1), (2, 2)]), Set([(2, 2), (1, 2)]))
    @test changedc == (Set([(1, 1)]), Set([(2, 1)]), Set([(2, 2)]), Set([(1, 2)]))
    @test all(k -> all(a -> size(a)[end-1:end] == (3, 3), S.A[k]), 1:4)
    @test all(k -> all(c -> size(c) == (3, 3), S.c[k]), 1:4)

    # Compare the factored enlarged-corner construction to its literal index sum.
    T = randn(2, 2, 2, 2)
    A = ntuple(_ -> randn(2, 2, 2), 4)
    c = randn(2, 2)
    for k in 1:4
        C = TNQSLocal._ctm_local_corner(T, A, c, k)
        order = (k, mod1(k + 1, 4), mod1(k + 2, 4), mod1(k + 3, 4))
        Tr = permutedims(T, order)
        literal = zeros(4, 4)
        for q3 in 1:2, i0 in 1:2, q2 in 1:2, j1 in 1:2,
            q0 in 1:2, q1 in 1:2, j0 in 1:2, i1 in 1:2
            literal[q3 + 2(i0 - 1), q2 + 2(j1 - 1)] +=
                Tr[q0, q1, q2, q3] * A[k][q0, i0, j0] * c[j0, i1] *
                A[mod1(k + 1, 4)][q1, i1, j1]
        end
        @test C ≈ literal rtol = 2e-14 atol = 2e-14
    end

    # The important closure regression: several overlapping local writes preserve the fixed storage
    # shape. This is exactly what failed when attempted on recursive C/T blocks.
    positive = [exp.(0.05 .* randn(2, 2, 2, 2)) for _ in 1:3, _ in 1:3]
    G = TNQSLocal.CTMLocalCycleState(positive, 2)
    result = TNQSLocal._ctm_local_sweep!(G)
    @test result.failures == 0
    @test result.updates == 6
    @test all(1 .<= G.rank .<= 2)
    @test all(k -> all(a -> size(a) == (2, 2, 2), G.A[k]), 1:4)
    @test all(k -> all(c -> size(c) == (2, 2), G.c[k]), 1:4)
    @test TNQSLocal._ctm_local_sweep!(G).failures == 0

    response = TNQSLocal._ctm_local_responses(G)
    # Optimized paired-GEMM response evaluation must equal the literal four-matrix trace.
    x, y = 2, 2
    q = ntuple(k -> size(G.A[k][x, y], 1), 4)
    B = ntuple(k -> [(@view G.A[k][x, y][a, :, :]) * G.c[k][x, y]
                     for a in axes(G.A[k][x, y], 1)], 4)
    literal_response = zeros(eltype(G.sites[x, y]), q)
    for a1 in 1:q[1], a2 in 1:q[2], a3 in 1:q[3], a4 in 1:q[4]
        literal_response[a1, a2, a3, a4] =
            TNQSLocal.tr(B[1][a1] * B[2][a2] * B[3][a3] * B[4][a4])
    end
    @test response[(x, y)] ≈ literal_response rtol = 3e-14 atol = 3e-14

    scaled = deepcopy(response)
    for key in keys(scaled)
        scaled[key] .*= -3.7
    end
    @test TNQSLocal._ctm_local_responsedist(response, scaled) < 1.0e-14

    # A real Schur conjugate pair must never be split. With budget two, retain the leading singlet,
    # skip the 2x2 rotation block which no longer fits, and take the next singlet instead.
    H = [6.0 0.0 0.0 0.0; 0.0 5.0 -2.0 0.0;
         0.0 2.0 5.0 0.0; 0.0 0.0 0.0 4.0]
    FH = TNQSLocal.schur(H)
    chosen, nkeep = TNQSLocal._ctm_local_schur_select(FH, 2)
    @test nkeep == 2
    @test count(chosen) == 2
    @test TNQSLocal.ordschur(FH, chosen).values[1:2] ≈ [6.0, 4.0]
    dropped, ndrop = TNQSLocal._ctm_local_schur_select(FH, 2; replace = false)
    @test ndrop == 1
    @test TNQSLocal.ordschur(FH, dropped).values[1] ≈ 6.0

    # The manuscript's QL construction is simultaneously biorthogonal, balanced, and triangular-
    # gauge preserving: VR differs from its orthogonal Schur basis by an upper-triangular factor.
    ZR = Matrix(TNQSLocal.qr(randn(8, 3)).Q)
    ZL = transpose(Matrix(TNQSLocal.qr(randn(8, 3)).Q))
    VRs, VLs = TNQSLocal._ctm_local_balance_schur(ZR, ZL)
    Xs = transpose(ZR) * VRs
    @test VLs * VRs ≈ Matrix{Float64}(TNQSLocal.I, 3, 3) rtol = 3e-14 atol = 3e-14
    @test transpose(VRs) * VRs ≈ VLs * transpose(VLs) rtol = 3e-14 atol = 3e-14
    @test TNQSLocal.norm(TNQSLocal.tril(Xs, -1)) < 3e-14 * TNQSLocal.norm(Xs)

    # Joint frame transport must change neither the oblique projector nor the balanced gauge.
    R = randn(8, 3)
    L = randn(3, 8)
    R, L = TNQSLocal._ctm_local_balance(R, L)
    O = Matrix(TNQSLocal.qr(randn(3, 3)).Q)
    Rm, Lm = R * O, transpose(O) * L
    Ra, La = TNQSLocal._ctm_local_align_pair(Rm, Lm, R, L)
    @test Ra * La ≈ R * L rtol = 2e-14 atol = 2e-14
    @test La * Ra ≈ Matrix{Float64}(TNQSLocal.I, 3, 3) rtol = 2e-14 atol = 2e-14
    @test transpose(Ra) * Ra ≈ La * transpose(La) rtol = 2e-14 atol = 2e-14
    @test TNQSLocal.norm(Ra - R) / TNQSLocal.norm(R) < 2e-14
    @test TNQSLocal.norm(La - L) / TNQSLocal.norm(L) < 2e-14

    # Production integration: the local state is converted to the existing C/T representation,
    # and the existing sum_R c_R log(Z_R) functional remains lossless on a finite Ising grid.
    tn = ising_partitionfunction(named_grid((3, 3)), 1.0)
    lnZ = log(real(contract(tn; alg = "exact")))
    cache = update(CTMEnvironmentCache(tn, 8; projector = :cycle, cycle_local = true);
                   maxiter = 10, tolerance = 1e-10)
    @test !isnothing(TNQSLocal.environments(cache))
    @test all(isfinite(TNQSLocal.region_lnZ(cache, cx, cy))
              for cx in 1:0.5:3, cy in 1:0.5:3)
    @test TNQSLocal.cvm_freenergy(cache) ≈ lnZ atol = 2e-12
    @test TNQSLocal.marginal_inconsistency(cache) < 2e-14

    @test TNQSLocal._ctm_use_local_cycle(CTMEnvironmentCache(tn, 32; projector = :cycle))
    @test !TNQSLocal._ctm_use_local_cycle(CTMEnvironmentCache(tn, 1; projector = :cycle))
    ψ = random_tensornetworkstate(Float64, named_grid((2, 2)); bond_dimension = 2)
    @test TNQSLocal._ctm_use_local_cycle(CTMEnvironmentCache(ψ, 8; projector = :cycle))
    @test !TNQSLocal._ctm_use_local_cycle(CTMEnvironmentCache(ψ, 9; projector = :cycle))
end
