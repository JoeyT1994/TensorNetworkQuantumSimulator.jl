@eval module $(gensym())
using LinearAlgebra: qr
using Random: Random
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: Tensor, full_update, apply_gates_full_update,
    simple_update, incoming_messages, contract_optimal, contraction_sequence, GreedyMethod, contract, from_array,
    prime, dag, replaceinds, apply, scalar, update
using Test: @testset, @test

bra(t, sinds) = replaceinds(dag(prime(t)), prime.(sinds), sinds)

# The fidelity of the new tensors on the bond (v₁, v₂) against the gate applied exactly, in the
# exact environment of the bond, i.e. the fidelity of the whole new state.
function exact_fidelity(ψ, o, v₁, v₂, ψ₁′, ψ₂′)
    s = siteinds(ψ)
    ts = Tensor[t for v in vertices(ψ) if v ∉ (v₁, v₂) for t in (ψ[v], bra(ψ[v], s[v]))]
    E = contract(ts; sequence = contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod()))
    T = apply(o, ψ[v₁] * ψ[v₂])
    b₁, b₂ = bra(ψ₁′, s[v₁]), bra(ψ₂′, s[v₂])
    ov = scalar(contract_optimal(Tensor[E, T, b₁, b₂]))
    n′ = real(scalar(contract_optimal(Tensor[E, ψ₁′, ψ₂′, b₁, b₂])))
    nT = real(scalar(contract_optimal(Tensor[E, T, bra(T, vcat(s[v₁], s[v₂]))])))
    return abs2(ov) / (n′ * nT), E
end

function random_gate(rng, s₁, s₂)
    U = Matrix(qr(randn(rng, ComplexF64, 4, 4)).Q)
    return from_array(reshape(U, 2, 2, 2, 2), [prime(s₁), prime(s₂), s₁, s₂])
end

@testset "Full update environments" begin
    rng = Random.Xoshiro(42)
    Random.seed!(42)
    g = named_grid((3, 3))
    ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 2)
    u, v = (2, 2), (2, 3)
    s = siteinds(ψ)
    o = random_gate(rng, only(s[u]), only(s[v]))
    maxdim = 2

    bp = update(BeliefPropagationCache(ψ))
    ctm = update(CTMRGCache(ψ, 16))
    _, E = exact_fidelity(ψ, o, u, v, ψ[u], ψ[v])
    infidelity((a, b)) = 1 - first(exact_fidelity(ψ, o, u, v, a, b))

    su = infidelity(first(simple_update(o, [ψ[u], ψ[v]]; envs = incoming_messages(bp, [u, v]), maxdim)))

    a, b, d_exact = full_update(o, ψ[u], ψ[v], [E]; maxdim)
    exact = infidelity((a, b))
    ctmrg = infidelity(full_update(o, ψ[u], ψ[v], incoming_messages(ctm, [u, v]); maxdim)[1:2])
    bond_only = infidelity(full_update(o, ψ[u], ψ[v], incoming_messages(bp, [u, v]); maxdim)[1:2])

    # in the exact environment, the objective is the exact fidelity
    @test d_exact.fidelity ≈ 1 - exact rtol = 1.0e-8
    @test d_exact.negativity < 1.0e-10
    # the bond alone in its BP environment is simple update
    @test bond_only ≈ su rtol = 1.0e-6
    # the exact environment is the best, and CTMRG's beats simple update
    @test exact <= ctmrg + 1.0e-10
    @test ctmrg < su

    # without truncation the update is exact
    a, b, _ = full_update(o, ψ[u], ψ[v], incoming_messages(ctm, [u, v]); maxdim = 16)
    @test infidelity((a, b)) < 1.0e-10
end

@testset "Full update gates" begin
    g = named_grid((3, 3))
    ψ0 = tensornetworkstate(ComplexF64, v -> "+", g, "S=1/2")
    layer = vcat(
        [("Rx", [v], 0.3) for v in vertices(g)],
        [("Rzz", [src(e), dst(e)], 0.7) for c in edge_color(g, 4) for e in c],
    )
    circuit = reduce(vcat, fill(layer, 2))
    ψ_exact, _ = apply_gates(circuit, ψ0; apply_kwargs = (; maxdim = 64))
    for env_maxdim in (4, 16)
        ψ, diagnostics = apply_gates_full_update(circuit, ψ0; maxdim = 2, env_maxdim)
        @test maxvirtualdim(ψ) <= 2
        @test length(diagnostics) == 2 * length(edges(g))
        f = abs2(inner(ψ, ψ_exact; alg = "exact")) / real(norm_sqr(ψ; alg = "exact") * norm_sqr(ψ_exact; alg = "exact"))
        @test f > 0.9
    end
end
end
