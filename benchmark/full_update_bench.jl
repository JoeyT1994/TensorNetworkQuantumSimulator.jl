#=
Test bench for two-site gate updates with truncation, on lattices small enough that the exact
environment of a bond can be computed.

For each state, edge and random two-qubit gate, every method truncates the bond back to `maxdim`,
and is scored by the exact infidelity 1 - |⟨ψ′|Gψ⟩|² / (⟨ψ′|ψ′⟩⟨Gψ|Gψ⟩) of the truncated state ψ′
against the exactly updated state Gψ. With the exact environment E of the bond, this is the
fidelity in E, so no full state is ever rebuilt.

Methods:
  su_bp        simple update in the BP environment (what `apply_gates` does)
  fu_exact     `full_update` in the exact environment: the bound for any environment
  fu_ctmrg     `full_update` in the CTMRG environment, with corners truncated to `env_maxdim`

Gates are Haar-random two-qubit unitaries (`haar`) or exp(-i 0.1 H) for a random Hermitian H
(`weak`), closer to a Trotter step.

Run with: julia --project=<env with TensorNetworkQuantumSimulator> benchmark/full_update_bench.jl
=#
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: Tensor, simple_update, full_update,
    planar_embedding, incoming_messages, contraction_sequence, contract_optimal, GreedyMethod,
    from_array, prime, dag, replaceinds, apply, scalar, contract, inds, update
using LinearAlgebra: qr, Hermitian
using Printf: @printf
using Random: Random
using Statistics: mean, median

bra(t, sinds) = replaceinds(dag(prime(t)), prime.(sinds), sinds)

# The norm network with the vertices `vs` removed, contracted into one tensor.
function exact_environment(ψ, vs)
    s = siteinds(ψ)
    ts = Tensor[]
    for v in vertices(ψ)
        v in vs && continue
        push!(ts, ψ[v], bra(ψ[v], s[v]))
    end
    return contract(ts; sequence = contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod()))
end

# The fidelity in the environment E of the new tensors against the gate applied to the old ones.
function env_fidelity(E, o, ψ₁, ψ₂, ψ₁′, ψ₂′, s₁, s₂)
    T = apply(o, ψ₁ * ψ₂)
    b₁, b₂ = bra(ψ₁′, [s₁]), bra(ψ₂′, [s₂])
    ov = scalar(contract_optimal(Tensor[E, T, b₁, b₂]))
    n′ = real(scalar(contract_optimal(Tensor[E, ψ₁′, ψ₂′, b₁, b₂])))
    nT = real(scalar(contract_optimal(Tensor[E, T, bra(T, [s₁, s₂])])))
    return abs2(ov) / (n′ * nT)
end

function haar_gate(rng, s₁, s₂)
    U = Matrix(qr(randn(rng, ComplexF64, 4, 4)).Q)
    return from_array(reshape(U, 2, 2, 2, 2), [prime(s₁), prime(s₂), s₁, s₂])
end

function weak_gate(rng, s₁, s₂; dt = 0.1)
    A = randn(rng, ComplexF64, 4, 4)
    U = exp(-im * dt * Matrix(Hermitian((A + A') / 2)))
    return from_array(reshape(U, 2, 2, 2, 2), [prime(s₁), prime(s₂), s₁, s₂])
end

# A state with physical correlations: kicked Ising steps from all up, truncated to D with BP.
function evolved_state(g, D; nsteps = 4, θh = 0.8)
    layer = [("Rx", [v], θh) for v in vertices(g)]
    for colored_edges in edge_color(g, 4)
        append!(layer, ("Rzz", [src(e), dst(e)], -π / 2) for e in colored_edges)
    end
    ψ = tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2")
    for _ in 1:nsteps
        ψ, _ = apply_gates(layer, ψ; apply_kwargs = (; maxdim = D, cutoff = 1.0e-14, normalize_tensors = true))
    end
    return ψ
end

function run_case(ψ, e, o, maxdim, emb, ctm)
    v₁, v₂ = src(e), dst(e)
    s = siteinds(ψ)
    s₁, s₂ = only(s[v₁]), only(s[v₂])
    ψ₁, ψ₂ = ψ[v₁], ψ[v₂]
    E = exact_environment(ψ, [v₁, v₂])
    bpc = update(BeliefPropagationCache(ψ))
    score(ψ⃗) = 1 - env_fidelity(E, o, ψ₁, ψ₂, ψ⃗[1], ψ⃗[2], s₁, s₂)

    results = Dict{String, NamedTuple}()
    t = @elapsed u = first(simple_update(o, [ψ₁, ψ₂]; envs = incoming_messages(bpc, [v₁, v₂]), maxdim))
    results["su_bp"] = (; infidelity = score(u), time = t)
    for (name, envs) in (
            ("fu_exact", [E]), ("fu_ctmrg", incoming_messages(ctm, [v₁, v₂])),
        )
        t = @elapsed (a, b, diag) = full_update(o, ψ₁, ψ₂, envs; maxdim)
        results[name] = (; infidelity = score((a, b)), time = t, diag.sweeps, diag.negativity, diag.condition)
    end
    return results
end

const METHODS = ["su_bp", "fu_ctmrg", "fu_exact"]

function bench(; nx = 4, ny = 4, Ds = (2, 3), ngates = 3, seed = 1234, gates = (:haar, :weak), env_maxdim = 16)
    rng = Random.MersenneTwister(seed)
    g = named_grid((nx, ny))
    emb = planar_embedding(g)
    for gate in gates, D in Ds, (label, ψ) in (
            ("random", random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = D)),
            ("evolved", evolved_state(g, D)),
        )
        all = Dict(m => NamedTuple[] for m in METHODS)
        ctm = update(CTMRGCache(ψ, env_maxdim))
        for e in edges(g), _ in 1:ngates
            s = siteinds(ψ)
            o = (gate === :haar ? haar_gate : weak_gate)(rng, only(s[src(e)]), only(s[dst(e)]))
            r = run_case(ψ, e, o, D, emb, ctm)
            for m in METHODS
                push!(all[m], r[m])
            end
        end
        println("\n== $(nx)x$(ny), D = maxdim = $D, $label state, $gate gates, $(length(all["su_bp"])) gate updates ==")
        @printf("%-15s %12s %12s %10s\n", "method", "mean 1-F", "median 1-F", "time (ms)")
        su_mean, ex_mean = mean(getfield.(all["su_bp"], :infidelity)), mean(getfield.(all["fu_exact"], :infidelity))
        for m in METHODS
            inf = getfield.(all[m], :infidelity)
            @printf("%-15s %12.3e %12.3e %10.2f   gap closed %3.0f%%\n", m, mean(inf), median(inf),
                1000 * mean(getfield.(all[m], :time)), 100 * (su_mean - mean(inf)) / (su_mean - ex_mean))
        end
        for m in ("fu_ctmrg", "fu_exact")
            rs = all[m]
            @printf("%-15s sweeps mean %.1f max %d | negativity max %.1e | condition max %.1e\n",
                m, mean(getfield.(rs, :sweeps)), maximum(getfield.(rs, :sweeps)),
                maximum(getfield.(rs, :negativity)), maximum(getfield.(rs, :condition)))
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    bench()
end
