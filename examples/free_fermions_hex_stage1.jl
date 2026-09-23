# Stage 1 at scale: spinless fermions on the open honeycomb, ~60 sites, through the standard pipeline
# simple update → BP DMRG → CTMRG L-BFGS. At V = 0 the ground energy is EXACT at any size (sum of the
# lowest N single-particle eigenvalues of the hopping matrix), so this is the one 60-site test with a
# machine-precision reference; V ≠ 0 (t–V model, CDW transition near V/t ≈ 1.36) reuses everything
# except the reference.
#
#   julia -t 8 --project examples/free_fermions_hex_stage1.jl
#
# Edit the parameters below. Each stage checkpoints its state to a .jls file next to this script and
# is skipped on rerun if the checkpoint exists (delete the files to start over; change TAG to keep
# several runs). Energies at this size are read by boundary MPS (variational: lower is better).
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using LinearAlgebra, Random, Printf, Serialization

# ── parameters ───────────────────────────────────────────────────────────────────────────
const NX, NY     = 4, 5           # named_hexagonal_lattice_graph(NX, NY): (4,5) = 58 sites on a 10×6 box
const SYMMETRY   = "fU1"          # "fU1" (particle number fixed by the start state) or "fZ2"
const t, V       = 1.0, 0.0       # hopping and nearest-neighbour repulsion (V = 0: exact reference)
const D          = 3              # bond dimension
const SU_DTAU    = 0.02           # simple update: imaginary-time step
const SU_STEPS   = 200            # simple update: Trotter layers (τ = SU_DTAU · SU_STEPS)
const BP_SWEEPS  = 2              # BP DMRG sweeps
const CHI        = 32             # CTMRG interface dimension
const LBFGS_ITER = 10             # CTMRG L-BFGS iterations (per invocation; rerun to continue)
const STEP0      = 1 / 32         # first (Jacobi) step; fermions measured 1/32 (docs/dmrg.md)
const TIME_LIMIT = 3 * 3600       # seconds: no L-BFGS iteration starts after this
const CHI_BMPS   = 48             # boundary-MPS bond dimension for the energy readout
const SEED       = 1
const TAG        = "hex$(NX)x$(NY)_$(SYMMETRY)_V$(V)_D$(D)"

Random.seed!(SEED)
g = named_hexagonal_lattice_graph(NX, NY)
vs = collect(vertices(g)); nsites = length(vs)
s = siteinds("Fermion", g; symmetry = SYMMETRY)
println("spinless fermions on the open honeycomb: hex($NX,$NY) = $nsites sites, t = $t, V = $V, $SYMMETRY, D = $D, χ = $CHI")

# ── model ────────────────────────────────────────────────────────────────────────────────
H = Any[]
for e in edges(g)
    push!(H, ("hopping", (src(e), dst(e)), -t))       # -t (c†_v c_w + h.c.)
    iszero(V) || push!(H, ("NN", (src(e), dst(e)), V)) # V n_v n_w
end
_obs(term) = (term[1], term[2] isa Tuple ? collect(term[2]) : term[2], term[3])
# Boundary MPS measures a batch of observables only if they share a column or a row: vertical bonds
# (same x) go by column, horizontal ones (same y) by row, one batch per line.
function bmps_energy(ψ)
    groups = Dict{Tuple{Symbol, Int}, Vector{Tuple}}()
    for term in H
        v, w = term[2]
        key = v[1] == w[1] ? (:col, v[1]) : (:row, v[2])
        push!(get!(groups, key, Tuple[]), _obs(term))
    end
    E = 0.0
    for obs in values(groups)
        E += sum(real, expect(ψ, obs; alg = "boundarymps", mps_bond_dimension = CHI_BMPS))
    end
    return E
end
report(stage, ψ) = begin
    τ = @elapsed E = bmps_energy(ψ)
    @printf("%-24s E(bMPS χ=%d) = %.10f   per site %.8f   (readout %.0f s)\n", stage, CHI_BMPS, E, E / nsites, τ)
    E
end

# ── start: charge-density-wave product state on the two sublattices (half filling) ───────
# The honeycomb is bipartite; the hexagonal graph's (x, y) positions alternate sublattice with x + y.
occupied(v) = isodd(v[1] + v[2])
N = count(occupied, vs)
println("start: CDW product state with N = $N particles on $nsites sites")

# ── exact reference at V = 0: fill the N lowest single-particle levels ───────────────────
idx = Dict(v => k for (k, v) in enumerate(vs))
A = zeros(nsites, nsites)
for e in edges(g)
    A[idx[src(e)], idx[dst(e)]] = -t; A[idx[dst(e)], idx[src(e)]] = -t
end
ε = eigvals(Symmetric(A))
E_free = sum(ε[1:N])
if iszero(V)
    @printf("exact free-fermion ground energy at N = %d: %.12f   per site %.10f   (gap to N±1 levels: %.3e)\n",
            N, E_free, E_free / nsites, ε[N + 1] - ε[N])
else
    println("V ≠ 0: no exact reference; the V = 0 energy at this N would be $E_free")
end

# ── checkpoints ──────────────────────────────────────────────────────────────────────────
dir = @__DIR__
ckpt(stage) = joinpath(dir, "stage1_$(TAG)_$(stage).jls")
load(stage) = isfile(ckpt(stage)) ? deserialize(ckpt(stage)) : nothing
save(stage, x) = (serialize(ckpt(stage), x); println("  saved $(ckpt(stage))"))

# ── stage 1: simple update ───────────────────────────────────────────────────────────────
ψ_su = load("su")
if ψ_su === nothing
    ψ = tensornetworkstate(ComplexF64, v -> occupied(v) ? "Occ" : "Emp", g, s)
    layer = Any[]
    for ces in edge_color(g, 3)
        append!(layer, ("F_hop", (src(e), dst(e)), im * t * SU_DTAU) for e in ces)      # exp(-dτ (-t) hop)
        iszero(V) || append!(layer, ("F_nn", (src(e), dst(e)), -im * V * SU_DTAU) for e in ces)  # exp(-dτ V nn)
    end
    bpc = BeliefPropagationCache(ψ)
    τ = @elapsed for step in 1:SU_STEPS
        global bpc
        bpc, errs = apply_gates(layer, bpc; apply_kwargs = (; maxdim = D, cutoff = 1.0e-14))
        step % 50 == 0 && (println("  SU step $step of $SU_STEPS, max truncation error $(maximum(errs))"); flush(stdout))
    end
    global ψ_su = TNQS.network(bpc)
    println("simple update: $SU_STEPS steps of dτ = $SU_DTAU (τ = $(SU_STEPS * SU_DTAU)) in $(round(τ; digits = 1)) s")
    save("su", ψ_su)
else
    println("simple update: loaded checkpoint")
end
E_su = report("after simple update:", ψ_su)
flush(stdout)

# ── stage 2: BP DMRG ─────────────────────────────────────────────────────────────────────
ψ_bp = load("bp")
if ψ_bp === nothing
    τ = @elapsed global ψ_bp, Es_bp = dmrg(ψ_su, H; alg = "bp", nsweeps = BP_SWEEPS, verbose = false)
    println("BP DMRG: $BP_SWEEPS sweeps in $(round(τ; digits = 1)) s; Bethe energy $(first(Es_bp)) → $(last(Es_bp))")
    save("bp", ψ_bp)
else
    println("BP DMRG: loaded checkpoint")
end
E_bp = report("after BP DMRG:", ψ_bp)
flush(stdout)

# ── stage 3: CTMRG L-BFGS (chained across invocations through the checkpoint) ────────────
st = load("ctm")
ψ_start = st === nothing ? ψ_bp : st.ψ
Es_prev = st === nothing ? Float64[] : st.Es
st === nothing || println("CTMRG L-BFGS: resuming from checkpoint after $(length(Es_prev) - 1) iterations, E = $(last(Es_prev))")
caches = Ref{Any}(nothing)
τ = @elapsed ψ_ctm, Es = dmrg(ψ_start, H; alg = "ctmrg_lbfgs", maxdim = CHI, maxiter = LBFGS_ITER, step0 = STEP0,
                              ls_max = 5, time_limit = TIME_LIMIT, caches, verbose = true)
Es_all = isempty(Es_prev) ? Es : vcat(Es_prev, Es[2:end])
println("CTMRG L-BFGS: $(length(Es) - 1) accepted steps in $(round(τ; digits = 1)) s; FD-of-F energies ", round.(Es; digits = 8))
save("ctm", (; ψ = ψ_ctm, Es = Es_all))
E_ctm = report("after CTMRG L-BFGS:", ψ_ctm)

# ── summary ──────────────────────────────────────────────────────────────────────────────
println()
@printf("%-14s %16s %14s %14s\n", "stage", "E (bMPS)", "per site", iszero(V) ? "gap to exact" : "Δ vs SU")
ref = iszero(V) ? E_free : E_su
for (name, E) in (("SU", E_su), ("BP DMRG", E_bp), ("CTM L-BFGS", E_ctm))
    @printf("%-14s %16.10f %14.8f %14.3e\n", name, E, E / nsites, E - ref)
end
iszero(V) && @printf("exact          %16.10f %14.8f\n", E_free, E_free / nsites)
@printf("FD-of-F of the final state %.10f vs bMPS %.10f (diff %.1e); monotone %s\n", last(Es), E_ctm, abs(last(Es) - E_ctm), all(diff(Es) .< 0))
