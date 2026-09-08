@testset "spinful fermionic backend" begin
    Random.seed!(0x5eed_0005)

    @testset "Hubbard chain ≡ dense JW (2 modes/site)" for symm in ("fZ2", "fU1xU1")
        n = 4
        nm = 2n
        g = NamedGraph(Graphs.path_graph(n))
        s = TNQS.siteinds("SpinfulFermion", g; symmetry = symm)
        cs = jw_ops(nm)
        up(v) = 2v - 1
        dn(v) = 2v
        init = Dict(1 => "Up", 2 => "Dn", 3 => "UpDn", 4 => "Emp")
        ψ = tensornetworkstate(ComplexF64, v -> init[v], g, s)
        @test real(norm_sqr(ψ; alg = "exact")) ≈ 1.0
        layer = Any[]
        for _ in 1:2
            append!(layer, ("F_hop", (v, v + 1), 0.31) for v in 1:2:(n - 1))
            append!(layer, ("F_hop", (v, v + 1), 0.31) for v in 2:2:(n - 1))
            append!(layer, ("F_int", [v], 0.23) for v in 1:n)
            append!(layer, ("F_hop_up", (v, v + 1), 0.155) for v in 1:2:(n - 1))
        end
        ψt, errs = apply_gates(layer, ψ; apply_kwargs = (; maxdim = 32, cutoff = 1.0e-14))
        @test maximum(errs) < 1.0e-12
        hop_m(j, k) = cs[j]' * cs[k] + cs[k]' * cs[j]
        ψv = zeros(ComplexF64, 2^nm)
        ψv[1 + sum(m -> 2^(nm - m), [up(1), dn(2), up(3), dn(3)]; init = 0)] = 1.0
        ψv = foldl(layer; init = ψv) do ϕv, gate
            if gate[1] == "F_int"
                v = only(gate[2])
                exp(-im * gate[3] * (cs[up(v)]' * cs[up(v)] * cs[dn(v)]' * cs[dn(v)])) * ϕv
            else
                v, w = gate[2]
                H = gate[1] == "F_hop" ? hop_m(up(v), up(w)) + hop_m(dn(v), dn(w)) :
                    hop_m(up(v), up(w))
                exp(-im * gate[3] * H) * ϕv
            end
        end
        nrm = real(ψv' * ψv)
        for (name, mop) in (
                ("Nup", v -> cs[up(v)]' * cs[up(v)]),
                ("Ndn", v -> cs[dn(v)]' * cs[dn(v)]),
                ("NupNdn", v -> cs[up(v)]' * cs[up(v)] * cs[dn(v)]' * cs[dn(v)]),
            )
            for v in 1:n
                tn = only(expect(ψt, (name, [v]); alg = "bp"))
                @test tn ≈ (ψv' * mop(v) * ψv) / nrm atol = 1.0e-12
            end
        end
        c_tn = only(expect(ψt, ("CdagC_up", (1, 3)); alg = "bp"))
        @test c_tn ≈ (ψv' * (cs[up(1)]' * cs[up(3)]) * ψv) / nrm atol = 1.0e-12
        h_tn = only(expect(ψt, ("hopping_dn", (2, 3)); alg = "bp"))
        @test h_tn ≈ (ψv' * hop_m(dn(2), dn(3)) * ψv) / nrm atol = 1.0e-12
    end
end
