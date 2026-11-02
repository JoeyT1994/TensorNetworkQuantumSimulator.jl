#=
Real-time quench of the square-lattice transverse-field Ising model
    H = -J Σ⟨uv⟩ Z_u Z_v - h Σ_v X_v
from all spins up, on an open nx × ny lattice small enough (≤ 20 sites) to keep the exact state.

Each second-order Trotter step (half X steps around the 4 edge colours swept forward and back) is
applied to the exact state and, at bond dimension D, by simple update with BP environments
(`apply_gates`) and by full update in CTMRG environments (`apply_gates_full_update`).
At every checkpoint the exact fidelity |⟨ψ|ψ_exact⟩|² of each state is printed.

Run with: julia --project=<env with TensorNetworkQuantumSimulator> benchmark/tfim_quench.jl
=#
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: Tensor, apply_gates_full_update, from_array, prime, noprime, contract,
    contraction_sequence, GreedyMethod, scalar, dag, state
using Printf: @printf

x_gate(s, dt, h) = from_array(ComplexF64[cos(dt * h) im*sin(dt * h); im*sin(dt * h) cos(dt * h)], [prime(s), s])
function zz_gate(s₁, s₂, dt, J)
    d = exp.(im * dt * J * [1, -1, -1, 1])
    M = zeros(ComplexF64, 4, 4)
    for i in 1:4
        M[i, i] = d[i]
    end
    return from_array(reshape(M, 2, 2, 2, 2), [prime(s₁), prime(s₂), s₁, s₂])
end

function trotter_step(s, g, colours, dt, J, h)
    n = length(colours)
    parts = vcat([(:x, dt / 2)], [(k, dt / 2) for k in 1:(n - 1)], [(n, dt)], [(k, dt / 2) for k in (n - 1):-1:1], [(:x, dt / 2)])
    gates = Tensor[]
    for (part, t) in parts
        if part === :x
            append!(gates, [x_gate(only(s[v]), t, h) for v in vertices(g)])
        else
            append!(gates, [zz_gate(only(s[src(e)]), only(s[dst(e)]), t, J) for e in colours[part]])
        end
    end
    return gates
end

# The state as one dense tensor on its site indices.
function dense(ψ)
    ts = [ψ[v] for v in vertices(ψ)]
    return contract(ts; sequence = contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod()))
end

fidelity(a, b) = abs2(scalar(dag(a) * b)) / (real(scalar(dag(a) * a)) * real(scalar(dag(b) * b)))

function run(; nx = 4, ny = 5, J = 1.0, h = 3.04, dt = 0.05, nsteps = 40, every = 5, Ds = (3, 4),
        env_maxdim = 16)
    g = named_grid((nx, ny))
    colours = edge_color(g, 4)
    ψ0 = tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2")
    s = siteinds(ψ0)
    step = trotter_step(s, g, colours, dt, J, h)
    labels = ["SU+BP", "FU+CTMRG"]
    @printf("TFIM quench %dx%d from all up, J = %.2f, h = %.2f, dt = %.3f, %d steps\n", nx, ny, J, h, dt, nsteps)
    for D in Ds
        println("D = $D: exact fidelity at each checkpoint")
        @printf("%6s", "t")
        foreach(l -> @printf("%16s", l), labels)
        println()
        exact = contract([state("0", only(s[v])) for v in vertices(g)])
        states = Any[ψ0 for _ in labels]
        times = zeros(length(labels))
        for chunk in 1:div(nsteps, every)
            gates = reduce(vcat, fill(step, every))
            for o in gates
                exact = noprime(contract([o, exact]))
            end
            times[1] += @elapsed states[1] = first(apply_gates(gates, states[1]; apply_kwargs = (; maxdim = D, cutoff = 1.0e-14, normalize_tensors = true)))
            times[2] += @elapsed states[2] = first(apply_gates_full_update(gates, states[2]; maxdim = D, env_maxdim))
            @printf("%6.2f", chunk * every * dt)
            foreach(ψ -> @printf("%16.6f", fidelity(dense(ψ), exact)), states)
            println()
        end
        @printf("%6s", "time")
        foreach(t -> @printf("%15.0fs", t), times)
        println()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
