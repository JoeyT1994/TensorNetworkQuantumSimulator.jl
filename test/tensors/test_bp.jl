@testset "end-to-end BP digests" begin
    Random.seed!(0x5eed_0003)

    function digest(g)
        ψ = tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2")
        layer = Any[("Rx", [v], 0.6) for v in vertices(g)]
        for ces in edge_color(g, 4)
            append!(layer, ("Rzz", pair, 0.4) for pair in ces)
        end
        circuit = reduce(vcat, [layer for _ in 1:2])
        bp_update_kwargs = (; maxiter = 30, tolerance = 1.0e-12)
        ψ, errs = apply_gates(
            circuit, ψ;
            apply_kwargs = (; maxdim = 4, cutoff = 1.0e-14), bp_update_kwargs
        )
        zs = expect(ψ, [("Z", [v]) for v in vertices(g)]; alg = "bp", cache_update_kwargs = bp_update_kwargs)
        zzs = expect(
            ψ, [("ZZ", [src(e), dst(e)]) for e in edges(g)];
            alg = "bp", cache_update_kwargs = bp_update_kwargs
        )
        return ψ, real(sum(zs)), real(sum(zzs))
    end

    ψt, zt, zzt = digest(named_comb_tree((3, 3)))
    z_ex = sum(real, expect(ψt, [("Z", [v]) for v in vertices(ψt)]; alg = "exact"))
    zz_ex = sum(real, expect(ψt, [("ZZ", [src(e), dst(e)]) for e in edges(ψt)]; alg = "exact"))
    @test zt ≈ z_ex atol = 1e-9
    @test zzt ≈ zz_ex atol = 1e-9

    _, z33, zz33 = digest(named_grid((3, 3)))
    @test z33 ≈ 4.482750429812405 atol = 1e-8
    @test zz33 ≈ 3.930327161012339 atol = 1e-8
end
