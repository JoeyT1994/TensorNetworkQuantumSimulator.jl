#=
Imaginary-time evolution of the square-lattice transverse-field Ising model
    H = -J Σ⟨uv⟩ Z_u Z_v - h Σ_v X_v
near its critical point (h/J ≈ 3.044 in the thermodynamic limit), on an open nx × ny lattice,
comparing simple update with BP environments against full update at the same bond dimension D.

All methods use second-order Trotter steps: half steps of exp(τ h X / 2) around the 4 edge colours
swept forward and back (see `trotter_sequence`). Full update (`apply_gates_full_update`) updates its
environment cache before each colour of the sweep, from its previous state; edges of one colour are
disjoint, so their gates commute and share the environment. Its environments come from CTMRG.

Energies are exact (full contraction) on lattices with at most 20 sites, else from CTMRG at each of
`measure_maxdims` (to check the measurement has converged).
On lattices with at most 20 sites the exact ground-state energy is computed with Lanczos.

Run with: julia --project=<env with TensorNetworkQuantumSimulator> benchmark/tfim_imaginary_time.jl
=#
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: Tensor, apply_gates_full_update, from_array, prime
using LinearAlgebra: normalize
using KrylovKit: eigsolve
using Printf: @printf

single_gate(s, τ, h) = from_array([cosh(τ * h) sinh(τ * h); sinh(τ * h) cosh(τ * h)], [prime(s), s])
function zz_gate(s₁, s₂, τ, J)
    d = [exp(τ * J), exp(-τ * J), exp(-τ * J), exp(τ * J)]
    M = zeros(4, 4)
    for i in 1:4
        M[i, i] = d[i]
    end
    return from_array(reshape(M, 2, 2, 2, 2), [prime(s₁), prime(s₂), s₁, s₂])
end

function energy(ψ, g, J, h; exact_energy, measure_maxdim)
    obs = vcat([("ZZ", [src(e), dst(e)]) for e in edges(g)], [("X", [v]) for v in vertices(g)])
    vals = exact_energy ? expect(ψ, obs; alg = "exact") : expect(ψ, obs; alg = "ctmrg", env_maxdim = measure_maxdim)
    ne = length(edges(g))
    return real(-J * sum(vals[1:ne]) - h * sum(vals[(ne + 1):end])) / length(vertices(g))
end

# Lanczos ground-state energy per site, for small lattices.
function exact_ground_energy(g, J, h)
    vs = collect(vertices(g))
    n = length(vs)
    bit = Dict(v => k - 1 for (k, v) in enumerate(vs))
    pairs = [(bit[src(e)], bit[dst(e)]) for e in edges(g)]
    function H(x)
        y = zero(x)
        for c in 0:(2^n - 1)
            xc = x[c + 1]
            iszero(xc) && continue
            diag = 0.0
            for (a, b) in pairs
                diag += ((c >> a) & 1) == ((c >> b) & 1) ? -J : J
            end
            y[c + 1] += diag * xc
            for a in 0:(n - 1)
                y[(c ⊻ (1 << a)) + 1] -= h * xc
            end
        end
        return y
    end
    vals, _ = eigsolve(H, randn(2^n), 1, :SR; ishermitian = true, tol = 1.0e-10)
    return first(vals) / n
end

# One second-order Trotter step of size τ: half X steps around the edge colours swept forward
# and back, the last colour taking a full step.
function trotter_sequence(colours, τ)
    n = length(colours)
    seq = Any[(:x, τ / 2)]
    for k in 1:(n - 1)
        push!(seq, (k, τ / 2))
    end
    push!(seq, (n, τ))
    for k in (n - 1):-1:1
        push!(seq, (k, τ / 2))
    end
    push!(seq, (:x, τ / 2))
    return seq
end

# The gates of the whole evolution, in order.
function evolution_gates(ψ, g, colours, J, h, schedule)
    s = siteinds(ψ)
    gates = Tensor[]
    for τ in schedule, (part, t) in trotter_sequence(colours, τ)
        if part === :x
            append!(gates, [single_gate(only(s[v]), t, h) for v in vertices(g)])
        else
            append!(gates, [zz_gate(only(s[src(e)]), only(s[dst(e)]), t, J) for e in colours[part]])
        end
    end
    return gates
end

su_evolve(ψ, gates, D) = first(apply_gates(gates, ψ; apply_kwargs = (; maxdim = D, cutoff = 1.0e-14, normalize_tensors = true)))

# Full update in CTMRG environments; the cache is updated from its previous state before each
# colour of the sweep. Returns the state and the largest negativity of an environment.
function fu_evolve(ψ, gates, D; env_maxdim = 16)
    ψ, diagnostics = apply_gates_full_update(gates, ψ; maxdim = D, env_maxdim)
    return ψ, maximum(d -> d.negativity, diagnostics)
end

function run(; nx = 4, ny = 4, J = 1.0, h = 3.04, Ds = (2, 3, 4), χ = 16, measure_maxdims = (32, 48),
        schedule = vcat(fill(0.1, 20), fill(0.05, 20), fill(0.02, 30), fill(0.01, 30), fill(0.005, 40)))
    g = named_grid((nx, ny))
    colours = edge_color(g, 4)
    exact_energy = nx * ny <= 20
    @printf("TFIM %dx%d, J = %.2f, h = %.3f, %d second-order Trotter steps, CTMRG env_maxdim %d\n", nx, ny, J, h, length(schedule), χ)
    if exact_energy
        e0 = exact_ground_energy(g, J, h)
        @printf("exact ground-state energy per site: %.8f\n", e0)
    end
    ψ0 = tensornetworkstate(Float64, v -> "+", g, "S=1/2")
    gates = evolution_gates(ψ0, g, colours, J, h, schedule)
    for D in Ds
        states = Pair{String, Any}[]
        t = @elapsed ψ = su_evolve(ψ0, gates, D)
        push!(states, "SU+BP" => ψ)
        @printf("D = %d  SU+BP %.0fs", D, t)
        t = @elapsed (ψ, neg) = fu_evolve(ψ0, gates, D; env_maxdim = χ)
        push!(states, "FU+CTMRG" => ψ)
        @printf(" | FU+CTMRG %.0fs (max env negativity %.1e)", t, neg)
        println()
        for m in (exact_energy ? (0,) : measure_maxdims)
            @printf("       %-9s", exact_energy ? "exact" : "CTMRG $m")
            for (label, ψ) in states
                e = energy(ψ, g, J, h; exact_energy, measure_maxdim = m)
                @printf("  %s: E = %.8f", label, e)
                exact_energy && @printf(" (rel err %.2e)", (e - e0) / abs(e0))
            end
            println()
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
