# Stage 2 at scale: the spin-1/2 Heisenberg antiferromagnet on an open LX×LY square lattice through
# the standard pipeline simple update → BP DMRG → CTMRG L-BFGS, the last stage with the (μ,ν)
# response solve (src/response.jl) by default — one norm converge plus a linear polynomial
# iteration per energy evaluation, cost independent of the operator rank (r = 3 here).
#
#   julia -t 8 --project examples/heisenberg_square_stage2.jl
#
# Edit the parameters below (every one can also be set through an environment variable HSQ_<NAME>).
# Each stage checkpoints its state to a .jls file next to this script and is skipped on rerun if the
# checkpoint exists (delete the files to start over; change TAG to keep several runs). Energies are
# read by boundary MPS (variational for the state at hand: lower is better); the CTM stage also
# reports its own response energy per iteration. H = J Σ_⟨ij⟩ S_i·S_j with S = σ/2.
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using LinearAlgebra, Random, Printf, Serialization

# ── parameters ───────────────────────────────────────────────────────────────────────────
_env(name, default) = get(ENV, "HSQ_" * name, default)
const LX, LY     = parse(Int, _env("LX", "8")), parse(Int, _env("LY", "8"))
const J          = parse(Float64, _env("J", "1.0"))
const D          = parse(Int, _env("D", "4"))            # bond dimension
const SU_SCHEDULE = [(0.1, 100), (0.02, 100), (0.005, 50)] # simple update: (dτ, steps) stages
const BP_SWEEPS  = parse(Int, _env("BP_SWEEPS", "2"))    # BP DMRG sweeps
const CHI        = parse(Int, _env("CHI", "32"))         # CTMRG interface dimension (norm sector)
const RESPONSE   = parse(Bool, _env("RESPONSE", "true")) # response solve (true) or the ±λ pair (false)
const LBFGS_ITER = parse(Int, _env("LBFGS_ITER", "10"))  # CTMRG L-BFGS iterations (per invocation)
const STEP0      = parse(Float64, _env("STEP0", "0.5"))  # first (Jacobi) step
const TIME_LIMIT = parse(Float64, _env("TIME_LIMIT", string(6 * 3600)))  # seconds
const CHI_BMPS   = parse(Int, _env("CHI_BMPS", "64"))    # boundary-MPS bond dimension for the readout
const SEED       = parse(Int, _env("SEED", "1"))
const TAG        = _env("TAG", "sq$(LX)x$(LY)_D$(D)_chi$(CHI)" * (RESPONSE ? "_resp" : "_pm"))

Random.seed!(SEED)
g = named_grid((LX, LY))
vs = collect(vertices(g)); nsites = length(vs)
s = siteinds("S=1/2", g)
println("Heisenberg antiferromagnet on the open $(LX)×$(LY) square: $nsites sites, J = $J, D = $D, χ = $CHI, $(RESPONSE ? "response solve" : "±λ pair"), $(Threads.nthreads()) threads")

# ── model ────────────────────────────────────────────────────────────────────────────────
H = Any[]
for e in edges(g), o in ("XX", "YY", "ZZ")
    push!(H, (o, (src(e), dst(e)), J / 4))                # J S·S = (J/4)(XX + YY + ZZ)
end
_obs(term) = (term[1], term[2] isa Tuple ? collect(term[2]) : term[2], term[3])
# Boundary MPS measures a batch of observables only if they share a column or a row.
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
    flush(stdout)
    E
end
println("orientation only: the 2D thermodynamic-limit energy per site is −0.66944 (QMC); an open $(LX)×$(LY) cluster sits above it")

# ── checkpoints ──────────────────────────────────────────────────────────────────────────
dir = @__DIR__
ckpt(stage) = joinpath(dir, "stage2_$(TAG)_$(stage).jls")
load(stage) = isfile(ckpt(stage)) ? deserialize(ckpt(stage)) : nothing
save(stage, x) = (serialize(ckpt(stage), x); println("  saved $(ckpt(stage))"))

# ── stage 1: simple update from the Néel state ───────────────────────────────────────────
ψ_su = load("su")
if ψ_su === nothing
    ψ = tensornetworkstate(ComplexF64, v -> isodd(v[1] + v[2]) ? "Up" : "Dn", g, s)
    bpc = BeliefPropagationCache(ψ)
    for (dτ, steps) in SU_SCHEDULE
        layer = Any[]
        for ces in edge_color(g, 4)
            # exp(−dτ J S·S) = Rxxyyzz(θ) with θ = −i J dτ / 2   (Rxxyyzz = exp(−i θ (XX+YY+ZZ)/2))
            append!(layer, ("Rxxyyzz", (src(e), dst(e)), -im * J * dτ / 2) for e in ces)
        end
        τ = @elapsed for step in 1:steps
            global bpc
            bpc, errs = apply_gates(layer, bpc; apply_kwargs = (; maxdim = D, cutoff = 1.0e-12))
            step % 50 == 0 && (println("  SU dτ = $dτ step $step of $steps, max truncation error $(maximum(errs))"); flush(stdout))
        end
        println("simple update: $steps steps of dτ = $dτ in $(round(τ; digits = 1)) s")
    end
    global ψ_su = TNQS.network(bpc)
    save("su", ψ_su)
else
    println("simple update: loaded checkpoint")
end
E_su = report("after simple update:", ψ_su)

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

# ── stage 3: CTMRG L-BFGS (chained across invocations through the checkpoint) ────────────
st = load("ctm")
ψ_start = st === nothing ? ψ_bp : st.ψ
Es_prev = st === nothing ? Float64[] : st.Es
st === nothing || println("CTMRG L-BFGS: resuming from checkpoint after $(length(Es_prev) - 1) iterations, E = $(last(Es_prev))")
caches = Ref{Any}(nothing)
τ = @elapsed ψ_ctm, Es = dmrg(ψ_start, H; alg = "ctmrg_lbfgs", maxdim = CHI, maxiter = LBFGS_ITER, step0 = STEP0, ls_max = 5,
                              time_limit = TIME_LIMIT, caches, verbose = true, response = RESPONSE)
Es_all = isempty(Es_prev) ? Es : vcat(Es_prev, Es[2:end])
println("CTMRG L-BFGS: $(length(Es) - 1) accepted steps in $(round(τ; digits = 1)) s; energies ", round.(Es; digits = 8))
save("ctm", (; ψ = ψ_ctm, Es = Es_all))
E_ctm = report("after CTMRG L-BFGS:", ψ_ctm)

println("\nsummary ($(LX)×$(LY), D = $D, χ = $CHI): energy per site by boundary MPS")
@printf("  simple update   %.8f\n  BP DMRG         %.8f\n  CTMRG L-BFGS    %.8f  (%d iterations so far; rerun to continue)\n", E_su / nsites, E_bp / nsites, E_ctm / nsites, length(Es_all) - 1)
