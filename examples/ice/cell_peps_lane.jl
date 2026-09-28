# Ice Ic / Ih residual entropy by boundary PEPS on the bilayer cell (`ice_site`), one lane per D over
# a series of χ. Per (D, χ), per molecule (w = exp(f/2), the cell holding two):
#   hex — ln σ_max(M): bra_perm = inversion, norm unpermuted, maximised (L-BFGS, then Newton–Krylov)
#   sym — the real Rayleigh quotient ⟨A|M|A⟩/⟨A|A⟩ (Xu, Lin & Zhang's cubic estimator), maximised
#   cub — ln λ(M): both bras inverted, the stationary point (Newton–Krylov, residual merit)
#   F   — the inversion overlap per cell, ln κ⟨I(A)|A⟩ − ln κ⟨A|A⟩, of the hex and cub states
# Resumable: an atomic checkpoint after every stage; a stage cut off by the run's budget resumes
# from its partial state (and trust radius). Env: D, CHIS ("16,24,32"), BUDGET (380 s per run); SUFFIX
# (for a second lane at the same D) with INIT_FROM (a lane checkpoint) and INIT_CHI: start from that
# lane's hexagonal optimum (or INIT_EST's state) at INIT_CHI, skipping L-BFGS; START_STAGE picks the stage.
using TensorNetworkQuantumSimulator, Printf, LinearAlgebra, Serialization
using Logging: NullLogger, with_logger
BLAS.set_num_threads(1)
const T = TensorNetworkQuantumSimulator
const D = parse(Int, ENV["D"])
const BUDGET = parse(Float64, get(ENV, "BUDGET", "380"))
const INV, ID = (2, 1, 4, 3), (1, 2, 3, 4)
const TOL, NOISE = 1.0e-7, 1.0e-5                  # the ice gradient floors at ~2e-6 (χ = 16) from truncation
const GTOL_LBFGS = D >= 3 ? 1.0e-3 : 1.0e-5          # L-BFGS only to get close; Newton–Krylov converges
const TAG = "ice_D$(D)" * get(ENV, "SUFFIX", "")
const CKPT, OUT = TAG * ".jls", TAG * ".csv"
# the χ series: ENV CHIS, or a TAG.chis file (to re-plan a running lane)
const CHIS = parse.(Int, split(strip(isfile(TAG * ".chis") ? read(TAG * ".chis", String) : ENV["CHIS"]), ","))
t0 = time()
left() = BUDGET - (time() - t0)
quiet(f) = with_logger(f, NullLogger())
checkpoint(S) = (serialize(CKPT * ".tmp", S); mv(CKPT * ".tmp", CKPT; force = true))

S = if isfile(CKPT)
    deserialize(CKPT)
else
    W, wl = ice_site()
    open(io -> println(io, "D,chi,estimator,f,w,residual,evals,converged,seconds,F_inv"), OUT, "w")
    S0 = Dict{Symbol, Any}(:W => W, :wl => wl, :k => 1, :stage => :lbfgs, :hex => nothing, :partial => nothing,
                           :radius => 0.05, :spent => 0.0, :prev => nothing, :results => Dict{Any, Any}())
    if haskey(ENV, "INIT_FROM")
        bp0 = deserialize(ENV["INIT_FROM"])[:results][(parse(Int, ENV["INIT_CHI"]), get(ENV, "INIT_EST", "hex"))]
        S0[:W], S0[:wl], S0[:hex], S0[:stage] = bp0.site, bp0.legs, bp0, :hex
        haskey(ENV, "START_STAGE") && (S0[:stage] = Symbol(ENV["START_STAGE"]))   # e.g. cub: that stage only
    end
    S0
end
W, wl = S[:W], S[:wl]

function lnnorm(bp, χ, p)
    nl, lg = T._bp_norm_layers(bp.A, bp.Alegs, bp.blegs; perm = p)
    return T.cvm_freenergy(T.update(T.InfiniteCTM2D(nl, lg, χ); tolerance = 1.0e-12, maxiter = 4000))
end
overlap(bp, χ) = lnnorm(bp, χ, INV) - lnnorm(bp, χ, ID)  # ln κ⟨I(A)|A⟩ − ln κ⟨A|A⟩ per cell
record(χ, est, bp, info, F) = (S[:results][(χ, est)] = bp; open(io -> println(io, join((D, χ, est, cvm_freenergy(bp), exp(cvm_freenergy(bp) / 2),
                                                          info.residual, info.evals, info.converged, S[:spent], F), ",")),
                                   OUT, "a"))

# one Newton–Krylov stage; returns the result when done, `nothing` when cut off (state saved)
function krylov_stage(χ, init, name; kw...)
    from = isnothing(S[:partial]) ? init : S[:partial]
    tk = @elapsed bp, info = quiet(() -> boundary_peps_krylov(W, wl, from; maxdim = χ, symmetry = :diagonal, tol = TOL,
                                                                noise_tol = NOISE, radius = S[:radius], fd_ctm_tolerance = 1.0e-10,
                                                                time_limit = left(), kw...))
    S[:spent] += tk
    # a resumed stage has stalled — take it (flagged unconverged): a maximum when f no longer rises
    # (|g| alone misleads there: at D = 3 f still rose 2e-7 per radius-limited step at |g| ~ 1e-4), the
    # stationary cubic point when |g|, already < 10 noise_tol, gains < 20 %
    stuck = !isnothing(S[:partial]) && (name == "cub" ?
        info.residual > 0.8 * S[:partial].gnorm && info.residual < 10 * NOISE &&
            abs(cvm_freenergy(bp) - cvm_freenergy(S[:partial])) < 1.0e-9 :
        cvm_freenergy(bp) - cvm_freenergy(S[:partial]) < 1.0e-10)
    if !info.converged && left() <= 1 && !stuck
        @printf("D = %d χ = %d %s: cut off at |g| = %.1e, f = %.12f, after %.0f s\n", D, χ, name, info.residual,
                cvm_freenergy(bp), S[:spent])
        S[:partial], S[:radius] = bp, info.radius
        checkpoint(S)
        return nothing, info
    end
    @printf("D = %d χ = %d %-3s: f = %.12f  w = %.10f  |g| = %.1e  %s  %d evals  %.0f s\n", D, χ, name, cvm_freenergy(bp),
            exp(cvm_freenergy(bp) / 2), info.residual, info.converged ? "converged" : "NOT CONVERGED", info.evals, S[:spent])
    flush(stdout)
    S[:partial], S[:radius] = nothing, 0.05
    return bp, info
end

while S[:k] <= length(CHIS) && left() > 30
    χ = CHIS[S[:k]]
    st = S[:stage]
    if st === :lbfgs                                  # first χ only: L-BFGS for the hexagonal maximum
        init = isnothing(S[:partial]) ? S[:prev] : S[:partial]
        tk = @elapsed bp = quiet(() -> boundary_peps(W, wl, D; maxdim = χ, init, symmetry = :diagonal, bra_perm = INV,
                                                     gtol = GTOL_LBFGS, maxiter = 500, time_limit = left()))
        S[:spent] += tk
        if bp.gnorm > GTOL_LBFGS && left() <= 1
            @printf("D = %d χ = %d L-BFGS: cut off at |g| = %.1e (f = %.10f)\n", D, χ, bp.gnorm, cvm_freenergy(bp))
            S[:partial] = bp; checkpoint(S); break
        end
        @printf("D = %d χ = %d L-BFGS: f = %.12f |g| = %.1e (%.0f s)\n", D, χ, cvm_freenergy(bp), bp.gnorm, S[:spent])
        S[:hex], S[:partial], S[:stage], S[:spent] = bp, nothing, :hex, 0.0
        checkpoint(S)
    elseif st === :hex
        bp, info = krylov_stage(χ, S[:hex], "hex"; bra_perm = INV, norm_perm = ID, merit = :f)
        isnothing(bp) && break
        F = overlap(bp, χ)
        record(χ, "hex", bp, info, F)
        @printf("        inversion overlap per cell: ln F = %.3e\n", F)
        S[:hex], S[:stage], S[:spent] = bp, :sym, 0.0
        checkpoint(S)
    elseif st === :sym
        bp, info = krylov_stage(χ, S[:hex], "sym"; bra_perm = ID, norm_perm = ID, merit = :f)
        isnothing(bp) && break
        record(χ, "sym", bp, info, overlap(bp, χ))
        S[:stage], S[:spent] = :cub, 0.0
        checkpoint(S)
    elseif st === :cub
        bp, info = krylov_stage(χ, S[:hex], "cub"; bra_perm = INV, norm_perm = INV)
        isnothing(bp) && break
        F = overlap(bp, χ)
        record(χ, "cub", bp, info, F)
        @printf("        inversion overlap per cell: ln F = %.3e\n", F)
        S[:k] += 1; S[:stage] = :hex; S[:spent] = 0.0   # the next χ starts from this χ's hexagonal state
        checkpoint(S)
    end
end
S[:k] > length(CHIS) && (println("LANE DONE"); touch(TAG * ".done"))
