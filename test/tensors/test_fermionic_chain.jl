@testset "fermionic chain backend" begin
    Random.seed!(0x5eed_0004)

    @testset "unit checks" begin
        s1 = Tensors.new_fermion_index(1, 1; tags = "s1")
        s2 = Tensors.new_fermion_index(1, 1; tags = "s2")
        t = TI.random_tensor(ComplexF64, s1, s2)
        @test real(TI.dag(t) * t) ≈ norm(t)^2
        F1, F2, _ = TI.factorize_svd(t, [s1])
        @test norm(TI.array(F1 * F2) - TI.array(t)) < 1.0e-13
        Q, R = qr(t, [s1])
        @test norm(TI.array(Q * R) - TI.array(t)) < 1.0e-13
        @test_throws Exception TI.state("Occ", s1)
    end

    @testset "Kitaev-chain quench ≡ dense JW (tree ⇒ BP exact)" begin
        n = 6
        g = NamedGraph(Graphs.path_graph(n))
        s = TNQS.siteinds("Fermion", g)
        ψ = tensornetworkstate(ComplexF64, v -> v in (2, 5) ? "Occ" : "Emp", g, s)
        @test real(norm_sqr(ψ; alg = "exact")) ≈ 1.0
        ψodd = tensornetworkstate(ComplexF64, v -> v == 1 ? "Occ" : "Emp", g, s)
        @test real(norm_sqr(ψodd; alg = "exact")) ≈ 1.0
        layer = Any[]
        for _ in 1:3
            append!(layer, ("F_pair", (v, v + 1), 0.29) for v in 1:2:(n - 1))
            append!(layer, ("F_hop", (v, v + 1), 0.37) for v in 2:2:(n - 1))
            append!(layer, ("F_hop", (v, v + 1), 0.37) for v in 1:2:(n - 1))
            append!(layer, ("F_phase", [v], 0.21) for v in 1:n)
        end
        ψt, errs = apply_gates(layer, ψ; apply_kwargs = (; maxdim = 32, cutoff = 1.0e-14))
        @test maximum(errs) < 1.0e-12
        cs = jw_ops(n)
        ψv = jw_evolve(layer, cs, Dict(v => v for v in 1:n), n; occupied = (2, 5))
        nrm = real(ψv' * ψv)
        for v in 1:n
            occ = real(only(expect(ψt, ("N", [v]); alg = "bp")))
            @test occ ≈ real(ψv' * (cs[v]' * cs[v]) * ψv) / nrm atol = 1.0e-12
        end
        hop = real(only(expect(ψt, ("hopping", (3, 4)); alg = "bp")))
        @test hop ≈ real(ψv' * (cs[3]' * cs[4] + cs[4]' * cs[3]) * ψv) / nrm atol = 1.0e-12
        pr = real(only(expect(ψt, ("pairing", (3, 4)); alg = "bp")))
        @test pr ≈ real(ψv' * (cs[3]' * cs[4]' + cs[4] * cs[3]) * ψv) / nrm atol = 1.0e-12
        for (name, mat, (v, w)) in (
                ("CdagC", (j, k) -> cs[j]' * cs[k], (1, 6)),
                ("CCdag", (j, k) -> cs[j] * cs[k]', (2, 4)),
                ("CdagCdag", (j, k) -> cs[j]' * cs[k]', (3, 5)),
                ("CC", (j, k) -> cs[j] * cs[k], (1, 4)),
            )
            tn = only(expect(ψt, (name, (v, w)); alg = "bp"))
            @test tn ≈ (ψv' * mat(v, w) * ψv) / nrm atol = 1.0e-12
        end

        ψc = tensornetworkstate(ComplexF64, v -> v in (2, 4, 5) ? "Occ" : "Emp", g, s)
        ψct, _ = apply_gates(layer, ψc; apply_kwargs = (; maxdim = 32, cutoff = 1.0e-14))
        ψcv = jw_evolve(layer, cs, Dict(v => v for v in 1:n), n; occupied = (2, 4, 5))
        nrmc = real(ψcv' * ψcv)
        occ3 = real(only(expect(ψct, ("N", [3]); alg = "bp")))
        @test occ3 ≈ real(ψcv' * (cs[3]' * cs[3]) * ψcv) / nrmc atol = 1.0e-12
        tnc = only(expect(ψct, ("CdagC", (1, 5)); alg = "bp"))
        @test tnc ≈ (ψcv' * (cs[1]' * cs[5]) * ψcv) / nrmc atol = 1.0e-12
        for x in ((0, 1, 0, 1, 1, 0), (1, 1, 1, 1, 1, 0))
            p_jw = abs2(ψcv[1 + sum(v -> x[v] * 2^(n - v), 1:n)]) / nrmc
            sd = TNQS.siteinds(ψct)
            tnp = copy(TNQS.tensornetwork(ψct))
            for v in vertices(tnp)
                P = TNQS.adapt_like(tnp[v], TI.projector(ComplexF64, only(sd[v]) => x[v] + 1))
                TNQS.setindex_preserve!(tnp, tnp[v] * P, v)
            end
            ts = [tnp[v] for v in vertices(tnp)]
            amp = TI.scalar(TI.contract(ts; sequence = TNQS.contraction_sequence(ts; alg = "optimal")))
            @test abs2(amp) / real(norm_sqr(ψct; alg = "exact")) ≈ p_jw atol = 1.0e-12
        end
    end

    @testset "fU1: number conservation is structural" begin
        #A fermionic message may use the parity gauge in which one symmetry block is
        #NSD. The PSD pseudoinverse must flip that whole block before taking roots,
        #rather than treating its negative spectrum as disposable roundoff.
        mi = Tensors.Index(Tensors.graded_space("fU1", [0 => 1, 1 => 1]), "message")
        msg = TI.delta(ComplexF64, mi, TI.prime(TI.dag(mi)))
        trees = collect(Tensors.TK.fusiontrees(msg.data))
        @test length(trees) == 2
        msg.data[last(trees)...] .*= -1
        msg_psd = Tensors.psd_gauge(msg)
        @test TI.array(msg_psd) ≈ Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
        msg_sqrt, msg_inv_sqrt = TNQS.pseudo_sqrt_inv_sqrt(msg; cutoff = 0.0)
        @test TI.array(msg_sqrt) ≈ Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
        @test TI.array(msg_inv_sqrt) ≈ Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)

        n = 4
        g = NamedGraph(Graphs.path_graph(n))
        s = TNQS.siteinds("Fermion", g; symmetry = "fU1")
        ψ = tensornetworkstate(ComplexF64, v -> v == 2 ? "Occ" : "Emp", g, s)
        @test real(norm_sqr(ψ; alg = "exact")) ≈ 1.0
        layer = Any[("F_hop", (v, v + 1), 0.37) for v in 1:(n - 1)]
        ψt, _ = apply_gates(layer, ψ; apply_kwargs = (; maxdim = 8, cutoff = 1.0e-14))
        cs = jw_ops(n)
        ψv = jw_evolve(layer, cs, Dict(v => v for v in 1:n), n; occupied = (2,))
        nrm = real(ψv' * ψv)
        for v in 1:n
            occ = real(only(expect(ψt, ("N", [v]); alg = "bp")))
            @test occ ≈ real(ψv' * (cs[v]' * cs[v]) * ψv) / nrm atol = 1.0e-12
        end
        c_tn = only(expect(ψt, ("CdagC", (1, 4)); alg = "bp"))
        @test c_tn ≈ (ψv' * (cs[1]' * cs[4]) * ψv) / nrm atol = 1.0e-12
        @test_throws Exception TI.op("F_pair", only(s[1]), only(s[2]); θ = 0.1)
    end
end
