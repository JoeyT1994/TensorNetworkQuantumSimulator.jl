@eval module $(gensym())
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: stale_bonds
using Test: @testset, @test

@testset "Full update carries its CTMRG cache across calls" begin
    g = named_grid((3, 3))
    ψ0 = tensornetworkstate(ComplexF64, v -> "+", g, "S=1/2")
    layer = vcat(
        [("Rx", [v], 0.3) for v in vertices(g)],
        [("Rzz", [src(e), dst(e)], 0.7) for c in edge_color(g, 4) for e in c],
    )
    cache0 = CTMRGCache(ψ0, 4)
    whole, _ = apply_gates_full_update(reduce(vcat, fill(layer, 3)), cache0; maxdim = 2)
    half, _ = apply_gates_full_update(layer, cache0; maxdim = 2)
    split, _ = apply_gates_full_update(reduce(vcat, fill(layer, 2)), half; maxdim = 2)
    ψw, ψs = network(whole), network(split)
    f = abs2(inner(ψw, ψs; alg = "exact")) / real(norm_sqr(ψw; alg = "exact") * norm_sqr(ψs; alg = "exact"))
    @test f ≈ 1 atol = 1.0e-8
    # the cache passed in is left as it was
    @test maxvirtualdim(network(cache0)) == 1
    # a returned cache whose bonds have grown since its blocks were built (here by the exact gates of
    # the first layer) updates and measures as a fresh one does
    obs = [("Z", [v]) for v in vertices(g)]
    @test stale_bonds(half)
    @test real.(expect(update(half), obs)) ≈ real.(expect(update(CTMRGCache(network(half), 4)), obs)) atol = 1.0e-8
end
end
