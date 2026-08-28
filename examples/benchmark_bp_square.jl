# Benchmark: BP simple update on a square-lattice PEPS, followed by BP measurements.
#
# Purpose: a fixed, deterministic workload for comparing tensor backends (walltime,
# allocations, and — crucially — agreement of the physics). Run it before and after any
# change to the tensor layer; the "agreement digest" printed at the end must match across
# backends to ~10 digits (BP convergence tolerance limits exact reproducibility, not the
# tensor arithmetic, which is float-deterministic for a fixed sweep schedule).
#
# Three timed phases:
#   1. evolve  — Trotterized TFIM circuit applied by BP-gauged simple update (the QR/SVD
#                gate kernel + interleaved BP cache updates).
#   2. bp      — an isolated BP cache update from fresh messages with a FIXED number of
#                sweeps (tolerance = nothing), the double-layer message-update kernel on a
#                deterministic workload. This is the primary backend-comparison number.
#   3. measure — BP measurements: single-site Z everywhere, ZZ on every edge.

using TensorNetworkQuantumSimulator
using Printf

function build_layer(g; J, hx, dt)
    layer = []
    append!(layer, ("Rx", [v], 2 * hx * dt) for v in vertices(g))
    for colored_edges in edge_color(g, 4)
        append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
    end
    return layer
end

function run_workload(; L, maxdim, nlayers, elt, J, hx, dt, bp_sweeps)
    g = named_grid((L, L))
    ψ0 = tensornetworkstate(elt, v -> "↑", g, "S=1/2")
    layer = build_layer(g; J, hx, dt)

    # Fixed BP schedule so every backend does the same work between gates.
    bp_update_kwargs = (; maxiter = 30, tolerance = 1.0e-10)
    apply_kwargs = (; maxdim, cutoff = 1.0e-12, normalize_tensors = true)

    # Phase 1: evolve
    ψ_bpc = BeliefPropagationCache(ψ0)
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    cum_trunc_err = 0.0
    evolve = @timed begin
        for _ in 1:nlayers
            ψ_bpc, errs = apply_gates(layer, ψ_bpc; apply_kwargs, bp_update_kwargs)
            cum_trunc_err += sum(errs)
        end
        ψ_bpc
    end
    ψ_bpc = evolve.value

    # Phase 2: isolated BP update, fresh messages, exactly `bp_sweeps` sweeps
    bpc_fresh = BeliefPropagationCache(network(ψ_bpc))
    bp = @timed update(bpc_fresh; maxiter = bp_sweeps, tolerance = nothing)

    # Phase 3: BP measurements
    obs_z = [("Z", [v]) for v in vertices(g)]
    obs_zz = [("ZZ", [src(e), dst(e)]) for e in edges(g)]
    measure = @timed (expect(ψ_bpc, obs_z; alg = "bp"), expect(ψ_bpc, obs_zz; alg = "bp"))
    zs, zzs = measure.value

    center = (cld(L, 2), cld(L, 2))
    digest = (;
        z_center = real(zs[findfirst(==(("Z", [center])), obs_z)]),
        mean_z = real(sum(zs)) / length(zs),
        sum_z = real(sum(zs)),
        sum_zz = real(sum(zzs)),
        cum_trunc_err,
        maxdim_reached = maxvirtualdim(ψ_bpc),
    )
    return (; evolve, bp, measure, digest, bp_sweeps, n_gates = nlayers * length(layer), n_edges = length(collect(edges(g))))
end

phase_report(label, t, extra = "") = @printf(
    "%-8s %8.2f s   %8.3f GiB allocated   %4.1f%% GC   %s\n",
    label, t.time, t.bytes / 2^30, 100 * t.gctime / t.time, extra
)

function main(; L = 10, maxdim = 8, nlayers = 10, elt = ComplexF64, J = 1.0, hx = 2.5, dt = 0.05, bp_sweeps = 20)
    # Warm-up at tiny size to exclude compilation from the timings.
    run_workload(; L = 3, maxdim = 2, nlayers = 1, elt, J, hx, dt, bp_sweeps = 2)

    res = run_workload(; L, maxdim, nlayers, elt, J, hx, dt, bp_sweeps)

    println("== BP square-lattice benchmark: L=$L (χ=$maxdim, $nlayers layers, $elt) ==")
    phase_report("evolve", res.evolve, "($(res.n_gates) gates, cum. gate err $(@sprintf("%.3e", res.digest.cum_trunc_err)))")
    phase_report("bp", res.bp, "($(res.bp_sweeps) sweeps × $(2 * res.n_edges) messages)")
    phase_report("measure", res.measure, "($(L * L) Z + $(res.n_edges) ZZ)")
    println("-- agreement digest (must match across backends) --")
    @printf("⟨Z⟩ center       = %.15f\n", res.digest.z_center)
    @printf("mean ⟨Z⟩         = %.15f\n", res.digest.mean_z)
    @printf("Σ ⟨Z⟩            = %.15f\n", res.digest.sum_z)
    @printf("Σ ⟨ZZ⟩           = %.15f\n", res.digest.sum_zz)
    @printf("max bond dim     = %d\n", res.digest.maxdim_reached)
    return res
end

# Reference digest at the default parameters (agreed with the original ITensors-backed
# implementation to ~1e-13 before its removal):
#   ⟨Z⟩ center = -0.16229418610571..., Σ ⟨Z⟩ = -24.7937337722159..., Σ ⟨ZZ⟩ = 62.8275875662148...
main()
