#=
Gate errors of simple update vs full update (CTMRG environments) in a real-time quench of the
square-lattice transverse-field Ising model
    H = J Σ⟨uv⟩ Z_u Z_v + h Σ_v X_v
from all spins down, on an open L × L lattice (the post-quench setup of Vovrosh et al., Phys. Rev.
Research 8, 023311 (2026), without a longitudinal field).

Each first-order Trotter step applies exp(-i dt h X) to every site, then exp(-i dt J ZZ) to the
edges of each of the 4 colours of an edge colouring. Both methods truncate every bond to `maxdim`:
  su  simple update, in BP environments (BP updated before each colour)
  fu  `full_update` in the CTMRG environment of the bond (CTMRG updated before each colour)
Each method keeps a CTMRG cache of its own state, updated before each colour, and every two-site
gate is scored by its fidelity against the exact gate in the CTMRG environment of the bond, so the
two methods' errors are measured in the same way. A bond keeps its index when its dimension is
unchanged, so the caches restart from their previous state.

Prints, per Trotter step, the summed gate infidelity Σ(1 - F), the largest gate infidelity, the
product of the gate fidelities so far and, every `measure_every` steps, the mean ⟨Z⟩.

Run with: julia --project=<env> benchmark/tfim_quench_gate_errors.jl su|fu
=#
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: Tensor, CTMRGCache, full_update, simple_update, incoming_messages,
    contract_environment, from_array, prime, dag, replaceinds, replaceind, apply, scalar, commonind, dim,
    update, setindex_preserve!, messages, NamedEdge
using LinearAlgebra: normalize
using Printf: @printf

x_gate(s, dt, h) = from_array(ComplexF64[cos(dt * h) -im*sin(dt * h); -im*sin(dt * h) cos(dt * h)], [prime(s), s])
function zz_gate(s₁, s₂, dt, J)
    M = zeros(ComplexF64, 4, 4)
    for (i, z) in enumerate([1, -1, -1, 1])
        M[i, i] = exp(-im * dt * J * z)
    end
    return from_array(reshape(M, 2, 2, 2, 2), [prime(s₁), prime(s₂), s₁, s₂])
end

bra(t, sinds) = replaceinds(dag(prime(t)), prime.(sinds), sinds)

# The mean ⟨Z⟩ from CTMRG converged at each environment bond dimension in `χs`, to check
# convergence in it.
mean_z(ψ, χs) = [real(sum(expect(update(CTMRGCache(ψ, χ)), [("Z", [v]) for v in vertices(ψ)]))) / length(vertices(ψ)) for χ in χs]

# The fidelity, in the environment `envs` of the bond, of the new tensors a, b against the gate
# applied exactly to the old ones. An environment is only fixed up to a scalar, hence the abs.
function env_fidelity(envs, o, ψ₁, ψ₂, a, b, s₁, s₂)
    T = apply(o, ψ₁ * ψ₂)
    ā, b̄ = bra(a, [s₁]), bra(b, [s₂])
    ov = scalar(contract_environment(vcat(envs, Tensor[T, ā, b̄])))
    nab = scalar(contract_environment(vcat(envs, Tensor[a, b, ā, b̄])))
    nT = scalar(contract_environment(vcat(envs, Tensor[T, bra(T, [s₁, s₂])])))
    return abs2(ov) / (abs(nab) * abs(nT))
end

function run(method::Symbol; L = 8, J = 1.0, h = 3.5, dt = 0.05, tmax = 2.0, maxdim = 6, env_maxdim = 16,
        ctm_tolerance = 1.0e-6, measure_every = 5, measure_maxdims = (16, 32))
    g = named_grid((L, L))
    colours = edge_color(g, 4)
    ψ = tensornetworkstate(ComplexF64, v -> "↓", g, "S=1/2")
    s = siteinds(ψ)
    vs = collect(vertices(g))
    bpc = method === :su ? update(BeliefPropagationCache(copy(ψ))) : nothing
    ctm = update(CTMRGCache(ψ, env_maxdim); tolerance = ctm_tolerance)
    rebuild = false
    log_fidelity = 0.0
    nsteps = round(Int, tmax / dt)
    @printf("# %s: %dx%d TFIM quench from all down, J = %.2f, h = %.2f, dt = %.3f, maxdim %d, CTMRG env_maxdim %d\n",
        method, L, L, J, h, dt, maxdim, env_maxdim)
    @printf("# %6s %14s %14s %14s %8s   <Z> from CTMRG at env_maxdim %s\n", "t", "sum(1-F)", "max(1-F)", "prod F", "time(s)", join(measure_maxdims, ", "))
    set!(t, v) = (ψ[v] = t; setindex_preserve!(ctm, t, v); isnothing(bpc) || setindex_preserve!(bpc, t, v))
    for step in 1:nsteps
        elapsed = @elapsed begin
            for v in vs
                set!(normalize(apply(x_gate(only(s[v]), dt, h), ψ[v])), v)
            end
            step_errors = Float64[]
            for colour in colours
                ctm = rebuild ? update(CTMRGCache(copy(ψ), env_maxdim); tolerance = ctm_tolerance) : update(ctm; tolerance = ctm_tolerance)
                rebuild = false
                isnothing(bpc) || (bpc = update(bpc))
                for e in colour
                    u, v = src(e), dst(e)
                    s₁, s₂ = only(s[u]), only(s[v])
                    o = zz_gate(s₁, s₂, dt, J)
                    bond = commonind(ψ[u], ψ[v])
                    envs = incoming_messages(ctm, [u, v])
                    if method === :fu
                        a, b, _ = full_update(o, ψ[u], ψ[v], envs; maxdim)
                    else
                        a, b = first(simple_update(o, [ψ[u], ψ[v]]; envs = incoming_messages(bpc, [u, v]), maxdim, normalize_tensors = false))
                        c = commonind(a, b)
                        dim(c) == dim(bond) && ((a, b) = (replaceind(a, c, bond), replaceind(b, c, bond)))
                    end
                    F = env_fidelity(envs, o, ψ[u], ψ[v], a, b, s₁, s₂)
                    push!(step_errors, 1 - F)
                    log_fidelity += log(F)
                    set!(normalize(a), u)
                    set!(normalize(b), v)
                    if commonind(a, b) != bond
                        rebuild = true
                        isnothing(bpc) || foreach(e -> haskey(messages(bpc), e) && delete!(messages(bpc), e), (NamedEdge(u => v), NamedEdge(v => u)))
                    end
                end
            end
        end
        mz = step % measure_every == 0 ? mean_z(copy(ψ), measure_maxdims) : Float64[]
        @printf("  %6.2f %14.6e %14.6e %14.8f %8.1f   %s\n", step * dt, sum(step_errors), maximum(step_errors), exp(log_fidelity), elapsed, join(round.(mz; digits = 6), "  "))
        flush(stdout)
    end
    return nothing
end

# Simple update alone (no CTMRG), at larger bond dimensions: per Trotter step the summed and
# largest BP truncation error of its gates and the largest bond dimension; for `maxdim` ≤ 8, every
# `measure_every` steps, the mean ⟨Z⟩ from CTMRG at each of `measure_maxdims` (converged CTMRG is
# out of reach at larger bond dimensions).
function run_su_bp(; L = 8, J = 1.0, h = 3.5, dt = 0.05, tmax = 2.0, maxdim = 16, measure_every = 5, measure_maxdims = (16, 32))
    g = named_grid((L, L))
    colours = edge_color(g, 4)
    ψ = tensornetworkstate(ComplexF64, v -> "↓", g, "S=1/2")
    s = siteinds(ψ)
    vs = collect(vertices(g))
    step_gates = vcat([x_gate(only(s[v]), dt, h) for v in vs], [zz_gate(only(s[src(e)]), only(s[dst(e)]), dt, J) for c in colours for e in c])
    @printf("# su_bp: %dx%d TFIM quench from all down, J = %.2f, h = %.2f, dt = %.3f, maxdim %d\n", L, L, J, h, dt, maxdim)
    @printf("# %6s %14s %14s %6s %8s   <Z> from CTMRG at env_maxdim %s\n", "t", "sum(err)", "max(err)", "D", "time(s)", join(measure_maxdims, ", "))
    bpc = update(BeliefPropagationCache(ψ))
    for step in 1:round(Int, tmax / dt)
        elapsed = @elapsed bpc, errors = apply_gates(step_gates, bpc; apply_kwargs = (; maxdim, cutoff = 1.0e-14, normalize_tensors = true))
        mz = maxdim <= 8 && step % measure_every == 0 ? mean_z(copy(network(bpc)), measure_maxdims) : Float64[]
        @printf("  %6.2f %14.6e %14.6e %6d %8.1f   %s\n", step * dt, sum(errors), maximum(errors), maxvirtualdim(network(bpc)), elapsed, join(round.(mz; digits = 6), "  "))
        flush(stdout)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    method, D = ARGS[1], parse(Int, ARGS[2])
    method == "su_bp" ? run_su_bp(; maxdim = D) : run(Symbol(method); maxdim = D)
end
