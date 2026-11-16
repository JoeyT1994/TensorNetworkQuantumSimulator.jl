@eval module $(gensym())
using TensorNetworkQuantumSimulator
using Test: @testset, @test

@testset "Full update" begin
    # a few Trotter steps of a transverse-field Ising quench on a 3x3 grid, truncated to bond
    # dimension 2: full update beats simple update (fidelities about 0.96 and 0.93)
    g = named_grid((3, 3))
    ψ0 = tensornetworkstate(ComplexF64, v -> "↓", g, "S=1/2")
    layer = vcat(
        [("Rx", [v], 0.5) for v in vertices(g)],
        [("Rzz", [src(e), dst(e)], 0.5) for c in edge_color(g, 4) for e in c],
    )
    circuit = reduce(vcat, fill(layer, 4))
    ψ_exact, _ = apply_gates(circuit, ψ0; apply_kwargs = (; maxdim = 64, cutoff = 1.0e-14))
    fidelity(ψ) = abs2(inner(ψ, ψ_exact; alg = "exact")) / real(norm_sqr(ψ; alg = "exact") * norm_sqr(ψ_exact; alg = "exact"))

    ψ_su, _ = apply_gates(circuit, ψ0; apply_kwargs = (; maxdim = 2, cutoff = 1.0e-14, normalize_tensors = true))
    ψ_fu, _ = apply_gates_full_update(circuit, ψ0; maxdim = 2, env_maxdim = 8)
    @test maxvirtualdim(ψ_fu) <= 2
    @test fidelity(ψ_fu) > fidelity(ψ_su)
end
end
