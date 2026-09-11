@testset "graded (Z2) backend" begin
    Random.seed!(0x5eed_0002)

    function graded_digest(sectors)
        g = named_grid((3, 3))
        s = siteinds("S=1/2", g; sectors, symmetry = "Z2")
        ψ = tensornetworkstate(ComplexF64, v -> iseven(sum(v)) ? "↑" : "↓", g, s)
        layer = Any[("Rz", [v], 0.4) for v in vertices(g)]
        for ces in edge_color(g, 4)
            append!(layer, ("Rxx", pair, 0.7) for pair in ces)
            append!(layer, ("xx_plus_yy", pair, (0.7, 0.3)) for pair in ces)
            append!(layer, ("Rzz", pair, 0.2) for pair in ces)
        end
        circuit = reduce(vcat, [layer for _ in 1:2])
        bp_update_kwargs = (; maxiter = 30, tolerance = 1.0e-12)
        ψ, errs = apply_gates(
            circuit, ψ;
            apply_kwargs = (; maxdim = 4, cutoff = 1.0e-14), bp_update_kwargs
        )
        zs = expect(ψ, [("Z", [v]) for v in vertices(g)]; alg = "bp", cache_update_kwargs = bp_update_kwargs)
        zx = expect(ψ, ("Z", [(2, 2)]); alg = "exact")
        return ψ, real(sum(zs)), real(zx), sum(errs)
    end

    ψd, zd, zxd, ed = graded_digest(nothing)
    ψg, zg, zxg, eg = graded_digest([0 => 1, 1 => 1])
    @test Tensors.isgraded(ψg[(1, 1)])
    @test zd ≈ zg atol = 1e-10
    @test zxd ≈ zxg atol = 1e-10
    @test ed ≈ eg atol = 1e-12

    #Graded fitting needs a generic cold start, but cache construction must be exactly
    #repeatable and must not consume the caller's global RNG stream.
    Random.seed!(0x51a7_e001)
    expected_next = rand(UInt64)
    Random.seed!(0x51a7_e001)
    cold1 = BoundaryMPSCache(ψg, 4)
    @test rand(UInt64) == expected_next
    cold2 = BoundaryMPSCache(ψg, 4)
    @test !isempty(messages(cold1))
    @test all(message(cold1, e) ≈ message(cold2, e) for e in keys(messages(cold1)))

    nstored = sum(length(TI.data(ψg[v])) for v in vertices(ψg))
    nfull = sum(prod(Int[TI.dim(i) for i in TI.inds(ψg[v])]) for v in vertices(ψg))
    @test nstored <= 0.5 * nfull

    vg = (2, 2)
    cis = collect(TI.inds(ψg[vg]))[1:2]
    Cg = TI.combiner(cis)
    tc = ψg[vg] * Cg
    @test TI.dim(TI.combinedind(Cg)) == prod(TI.dim.(cis))
    trt = tc * TI.dag(Cg)
    @test trt ≈ ψg[vg] atol = 1e-13

    sg = only(TNQS.siteinds(TNQS.tensornetwork(ψg))[(1, 1)])
    @test_throws Exception TI.op("X", sg)
    @test_throws Exception TI.state("↓", sg)
    gz = named_grid((3, 3))
    sz = siteinds("S=1/2", gz; sectors = [0 => 1, 1 => 1], symmetry = "Z2")
    ψz = tensornetworkstate(ComplexF64, v -> v == (1, 1) ? "↓" : "↑", gz, sz)
    @test real(norm_sqr(ψz; alg = "exact")) ≈ 1.0

    ne = real(norm_sqr(ψg; alg = "exact"))
    nb = real(norm_sqr(ψg; alg = "boundarymps", mps_bond_dimension = 20))
    @test abs(nb / ne - 1) < 5e-3

    function amp2(ψt, bits)
        sd = TNQS.siteinds(ψt)
        tnp = copy(TNQS.tensornetwork(ψt))
        for v in vertices(tnp)
            P = TNQS.adapt_like(tnp[v], TI.projector(ComplexF64, only(sd[v]) => bits(v)))
            TNQS.setindex_preserve!(tnp, tnp[v] * P, v)
        end
        ts = [tnp[v] for v in vertices(tnp)]
        amp = TI.scalar(TI.contract(ts; sequence = TNQS.contraction_sequence(ts; alg = "optimal")))
        return abs2(amp) / real(norm_sqr(ψt; alg = "exact"))
    end

    bits = v -> iseven(v[1]) ? 1 : 2
    @test amp2(ψg, bits) ≈ amp2(ψd, bits) atol = 1.0e-10
    gsamples = sample_directly_certified(
        ψg, 2; alg = "boundarymps",
        norm_mps_bond_dimension = 12, projected_mps_bond_dimension = 12
    )
    @test all(x -> abs(real(x.poverq) - 1) < 0.1, gsamples)

    function thermal_logz(symmetry)
        gt = named_grid((2, 2))
        st = symmetry === nothing ? siteinds("S=1/2", gt; inds_per_site = 2) :
            siteinds("S=1/2", gt; inds_per_site = 2, symmetry)
        ψth = identity_tensornetworkstate(ComplexF64, gt, st)
        ψ_bpc = update(BeliefPropagationCache(ψth))
        gates = []
        for ces in edge_color(gt, 4)
            append!(gates, [op("Rxxyyzz", st[src(e)][1], st[dst(e)][1], θ = -0.01im) for e in ces])
        end
        logz = -TNQS.freenergy(ψ_bpc)
        rescale!(ψ_bpc)
        for _ in 1:2
            ψ_bpc, _ = apply_gates(gates, ψ_bpc; apply_kwargs = (; maxdim = 32, cutoff = 1.0e-14))
            logz -= TNQS.freenergy(ψ_bpc)
            rescale!(ψ_bpc)
        end
        return logz
    end

    @test thermal_logz("U1") ≈ thermal_logz(nothing) atol = 1.0e-11

    si = Tensors.Index(Tensors.graded_space("Z2", [0 => 1, 1 => 2]), "a")
    sj = Tensors.Index(Tensors.graded_space("Z2", [0 => 2, 1 => 1]), "b")
    T4 = TI.random_tensor(ComplexF64, si', sj', TI.dag(si), TI.dag(sj))
    svr = Ref{Any}(nothing)
    F1, F2, spec = TI.factorize_svd(T4, [si', sj']; ortho = "none", singular_values! = svr)
    @test norm(F1 * F2 - T4) < 1e-10
    @test spec.truncerr < 1e-13
    Q, R = qr(T4, [si', sj'])
    @test norm(Q * R - T4) < 1e-10

    # `pad_index`: the CTM cycle route pads a retained bond to width `kt`; on graded data the padded
    # bond's per-sector multiplicities must depend on the interface only, not on how the retained
    # bond happened to split its width (or the sweep-to-sweep unitary alignment declines).
    ins = [si, TI.dag(sj)]                                   # fused: Z2(0) => 4, Z2(1) => 5
    kt = 4
    sector_totals(w, z) = begin
        d = Dict{Any, Int}()
        for i in (w, z)
            i === nothing && continue
            for (c, n) in zip(Tensors.GA.sectors(Tensors.space(i)), Tensors.GA.blocklengths(Tensors.space(i)))
                d[c] = get(d, c, 0) + n
            end
        end
        d
    end
    w1 = Tensors.Index(Tensors.graded_space("Z2", [0 => 1, 1 => 1]), "w")
    w2 = Tensors.Index(Tensors.graded_space("Z2", [0 => 2]), "w")
    z1, z2 = TI.pad_index(w1, ins, kt), TI.pad_index(w2, ins, kt)
    @test TI.dim(w1) + TI.dim(z1) == kt && TI.dim(w2) + TI.dim(z2) == kt
    @test sector_totals(w1, z1) == sector_totals(w2, z2)
    @test Tensors.isdual(TI.pad_index(TI.dag(w1), ins, kt)) == true
    @test TI.pad_index(w1, ins, TI.dim(w1)) === nothing
    @test TI.pad_index(Tensors.Index(3, "d"), [Tensors.Index(2, "a")], 5) !== nothing   # dense: plain width

    @testset "random graded state: dual bond copies, BP, Hermitian eigen" begin
        # the two ends of every bond must carry mutually dual index copies, or the very first
        # BP contraction fails on mismatched axes (it did before the constructor dualised one end)
        gr = named_grid((3, 3))
        sr = siteinds("S=1/2", gr; sectors = [0 => 1, 1 => 1], symmetry = "Z2")
        ψr = random_tensornetworkstate(ComplexF64, gr, sr; bond_dimension = 4)
        for e in edges(gr)
            b = only(TI.commoninds(ψr[src(e)], ψr[dst(e)]))
            bs = TI.inds(ψr[src(e)])[findfirst(==(b), TI.inds(ψr[src(e)]))]
            bd = TI.inds(ψr[dst(e)])[findfirst(==(b), TI.inds(ψr[dst(e)]))]
            @test Tensors.isdual(bs) != Tensors.isdual(bd)
        end
        n_exact = real(norm_sqr(ψr; alg = "exact"))
        @test n_exact > 0
        bpc = update(BeliefPropagationCache(ψr); maxiter = 30, tolerance = 1.0e-10)
        @test isfinite(real(only(expect(bpc, ("Z", [(2, 2)])))))
        for e in Iterators.take(edges(TNQS.graph(bpc)), 3)
            M = TNQS.parity_message_gauge(TNQS.message(bpc, e))
            i1, i2 = TI.inds(M)
            @test norm(TI.array(Tensors._hermitian_part(M, [i1], [i2]), i1, i2) - TI.array(M)) < 1.0e-13 * norm(TI.array(M))
            Q, D, Qdag = TNQS.eigendecomp(M, i1, i2; ishermitian = true)
            @test norm(TI.array(Q * D * Qdag, i1, i2) - TI.array(M)) < 1.0e-10 * norm(TI.array(M))
        end
    end
end
