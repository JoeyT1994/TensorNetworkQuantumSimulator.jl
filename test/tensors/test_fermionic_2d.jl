@testset "fermionic 2D and loop corrections" begin
    Random.seed!(0x5eed_0006)

    @testset "2D grid ≡ dense JW (strings on vertical bonds)" begin
        g = named_grid((2, 3))
        vs = sort(collect(vertices(g)))
        n = length(vs)
        mode = Dict(v => i for (i, v) in enumerate(vs))
        s = TNQS.siteinds("Fermion", g)
        ψ = tensornetworkstate(ComplexF64, v -> "Emp", g, s)
        es = collect(edges(g))
        layer = Any[]
        append!(layer, ("F_pair", (src(e), dst(e)), 0.29) for e in es)
        append!(layer, ("F_hop", (src(e), dst(e)), 0.37) for e in es)
        append!(layer, ("F_phase", [v], 0.21) for v in vs)
        ψt, errs = apply_gates(layer, ψ; apply_kwargs = (; maxdim = 64))
        @test maximum(errs) < 1.0e-12
        cs = jw_ops(n)
        ψv = jw_evolve(layer, cs, mode, n)
        nrm = real(ψv' * ψv)
        occ_tn = [real(only(expect(ψt, ("N", [v]); alg = "exact"))) for v in vs]
        occ_jw = [real(ψv' * (cs[mode[v]]' * cs[mode[v]]) * ψv) / nrm for v in vs]
        @test maximum(abs.(occ_tn .- occ_jw)) < 1.0e-12
        e_v = first(filter(e -> src(e)[1] != dst(e)[1], es))
        v1, v2 = src(e_v), dst(e_v)
        j, k = mode[v1], mode[v2]
        hop = real(only(expect(ψt, ("hopping", (v1, v2)); alg = "exact")))
        @test hop ≈ real(ψv' * (cs[j]' * cs[k] + cs[k]' * cs[j]) * ψv) / nrm atol = 1.0e-12
        pr = real(only(expect(ψt, ("pairing", (v1, v2)); alg = "exact")))
        @test pr ≈ real(ψv' * (cs[j]' * cs[k]' + cs[k] * cs[j]) * ψv) / nrm atol = 1.0e-12

        samples = sample_directly_certified(
            ψt, 2; alg = "boundarymps",
            norm_mps_bond_dimension = 8, projected_mps_bond_dimension = 8
        )
        @test all(x -> abs(real(x.poverq) - 1) < 0.1, samples)
        @test all(x -> iseven(sum(values(x.bitstring))), samples)

        bmps = update(BoundaryMPSCache(ψt, 16))
        e_h = first(filter(e -> src(e)[1] == dst(e)[1], es))
        w1, w2 = src(e_h), dst(e_h)
        n_bmps = real(only(expect(bmps, ("N", [w1]); alg = "boundarymps")))
        @test n_bmps ≈ real(ψv' * (cs[mode[w1]]' * cs[mode[w1]]) * ψv) / nrm atol = 1.0e-10
        c_bmps = only(expect(bmps, ("CdagC", (w1, w2)); alg = "boundarymps"))
        @test c_bmps ≈ (ψv' * (cs[mode[w1]]' * cs[mode[w2]]) * ψv) / nrm atol = 1.0e-10
    end

    @testset "loop corrections at odd total parity (charge-leg gauge line)" begin
        g = named_grid((2, 3))
        s = TNQS.siteinds("Fermion", g)
        ψ = tensornetworkstate(ComplexF64, v -> isodd(sum(v)) ? "Occ" : "Emp", g, s)
        half = Any[]
        for ces in edge_color(g, 4)
            append!(half, ("F_hop", pair, -0.05) for pair in ces)
        end
        ψt, _ = apply_gates(
            vcat(half, reverse(half)), ψ;
            apply_kwargs = (; maxdim = 8, cutoff = 1.0e-12)
        )
        ne = real(norm_sqr(ψt; alg = "exact"))
        err(mcs) = abs(real(norm_sqr(ψt; alg = "loopcorrections", max_configuration_size = mcs)) - ne) / ne
        e0, e4, e8 = err(0), err(4), err(8)
        @test e4 < 0.01 * e0
        @test e8 < 1.0e-12
    end

    # REGRESSION (2026-09-10): eigen-decomposing a Hermitian message with its SECOND stored leg as the
    # codomain transposes two odd legs (a fermionic sign), so the odd-parity eigenvalues came back
    # negated — a positive block read as negative — and
    # `symmetric_gauge!` — and every fermionic boundary-MPS expectation on a state with alternating
    # bond arrows — died in `sqrt`. Both leg orders must give non-negative eigenvalues, reconstruct
    # the message in its own orientation, and the gauge must leave the state invariant.
    @testset "eigen of both message orientations (symmetric gauge)" begin
        g = named_grid((2, 2))
        s = TNQS.siteinds("Fermion", g)
        ψ = tensornetworkstate(ComplexF64, v -> v == (1, 1) ? "Occ" : "Emp", g, s)
        layer = Any[("F_hop", (src(e), dst(e)), 0.4) for e in edges(g)]
        ψ, _ = apply_gates(layer, ψ; apply_kwargs = (; maxdim = 4, cutoff = 1.0e-14))
        bpc = update(BeliefPropagationCache(ψ); maxiter = 50, tolerance = 1.0e-12)
        for e in edges(TNQS.graph(bpc))
            M = TNQS.parity_message_gauge(TNQS.message(bpc, e))
            i1, i2 = TI.inds(M)
            for (li, ri) in ((i1, i2), (i2, i1))
                # the Hermitian projection `eigen` applies must be the identity on an exactly
                # Hermitian fermionic message, in either orientation (map adjoint, not the bra)
                @test norm(TI.array(Tensors._hermitian_part(M, [li], [ri]), TI.inds(M)...) - TI.array(M)) < 1.0e-13 * norm(TI.array(M))
                Q, D, Qdag = TNQS.eigendecomp(M, li, ri; ishermitian = true)
                dv = real.([TI.array(D)[k, k] for k in 1:TI.dim(li)])
                @test minimum(dv) > -1.0e-10
                @test norm(TI.array(Q * D * Qdag, TI.inds(M)...) - TI.array(M)) < 1.0e-10 * norm(TI.array(M))
            end
        end
        n_before = real(norm_sqr(ψ; alg = "exact"))
        z_before = real(only(expect(ψ, ("N", [(1, 2)]); alg = "exact")))
        ψg = TNQS.network(TNQS.symmetric_gauge(bpc))
        @test real(norm_sqr(ψg; alg = "exact")) ≈ n_before rtol = 1.0e-8
        @test real(only(expect(ψg, ("N", [(1, 2)]); alg = "exact"))) ≈ z_before atol = 1.0e-10
        # the boundary-MPS path that gauges the state first
        @test real(only(expect(ψ, ("N", [(1, 2)]); alg = "boundarymps", mps_bond_dimension = 8))) ≈ z_before atol = 1.0e-8
    end
end
