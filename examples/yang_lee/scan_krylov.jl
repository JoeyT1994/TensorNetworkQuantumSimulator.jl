# 3D Ising Yang–Lee edge map at fixed β by Newton–Krylov continuation in the imaginary field H = iθ,
# from the real maximiser at θ = 0 towards the finite-D fold. Each point: f, m, the sandwich's
# correlation length ξ, and the running fold estimate from (dm/dθ)⁻² → 0 through the last three points.
# Steps grow ×GROWTH per accepted point, capped at MAXSTEP·θf2 (θf2 the known D = 2 fold) and at half
# the distance to the fold estimate, and halve on failure. The predictor is the secant in θ, and in
# u = √(θ_f − θ) once a fold estimate exists (A ≈ A_f − a u near the fold). Resumable: atomic
# checkpoint after every point; a point cut off by the run's budget resumes from its partial state and
# trust radius, and one whose resume gains < 20 % in |g| counts as failed.
# Env: BETA, D (3), CHI (24), THF2 (the D = 2 fold), VSTOP (5e-3), BUDGET (380 s per run), GPU (1: run on the
# device; checkpoints stay on the host, so a run can resume on either).
using TensorNetworkQuantumSimulator, Printf, LinearAlgebra, Serialization
using Logging: NullLogger, with_logger
BLAS.set_num_threads(1)
const T = TensorNetworkQuantumSimulator
const GPU = get(ENV, "GPU", "0") == "1"
if GPU
    using CUDA, Adapt
end
dev(x) = (GPU && !isnothing(x)) ? adapt(CuArray, x) : x
host(x) = (GPU && !isnothing(x)) ? adapt(Array, x) : x
const β = parse(Float64, ENV["BETA"])
const D = parse(Int, get(ENV, "D", "3"))
const CHI = parse(Int, get(ENV, "CHI", "24"))
const THF2 = parse(Float64, ENV["THF2"])
const VSTOP = parse(Float64, get(ENV, "VSTOP", "5e-3"))
# The real start at θ = 0: its last stage's L-BFGS limits. At D ≥ 4 the defaults (150 iterations,
# 200 s) left |g| = 3.7e-4 at β = 0.21, and the first Newton–Krylov point had not converged after 65 minutes.
const T0ITER = parse(Int, get(ENV, "T0ITER", "150"))
const T0LIMIT = parse(Float64, get(ENV, "T0LIMIT", "200"))
const BUDGET = parse(Float64, get(ENV, "BUDGET", "380"))  # + load (~100 s) + one block and the trials
const TOL, NOISE = 1.0e-6, 5.0e-6                  # |g| ≲ 1e-6 gives m to ~1e-6 at D = 3; near the fold |g| floors at 3–4e-6
const GROWTH, MAXSTEP, FIRST = 1.3, 0.1, 0.1
const TAG = "ylk3_beta$(β)_D$(D)_chi$(CHI)"
const CKPT, OUT = TAG * ".jls", TAG * ".csv"
quiet(f) = with_logger(f, NullLogger())
hostify(S) = merge(S, (; cur = host(S.cur), prev = host(S.prev), partial = host(S.partial)))
devify(S) = merge(S, (; cur = dev(S.cur), prev = dev(S.prev), partial = dev(S.partial)))
checkpoint(S) = (serialize(CKPT * ".tmp", hostify(S)); mv(CKPT * ".tmp", CKPT; force = true))
t0 = time()
left() = BUDGET - (time() - t0)

if isfile(CKPT)
    S = devify(deserialize(CKPT))
else
    s0, legs, _ = ising3d_site(β)
    s0 = dev(s0)
    bp = nothing
    for dd in 2:D                                     # D = 2 at χ = 16, then D at χ = CHI (embedded)
        global bp = quiet(() -> boundary_peps(s0, legs, dd; maxdim = dd == D ? CHI : 16, init = bp,
                                                gtol = 1.0e-6, maxiter = dd == D ? T0ITER : 300, time_limit = T0LIMIT))
    end
    @printf("θ = 0: D = %d χ = %d f = %.12f |g| = %.1e (%.0f s)\n", D, CHI, cvm_freenergy(bp), bp.gnorm, time() - t0)
    open(io -> println(io, "beta,D,chi,theta,f,m_imag,m_real,xi,evals,residual,converged,seconds,theta_fold"), OUT, "w")
    S = (; legs, cur = bp, prev = nothing, θc = 0.0, θp = 0.0, step = FIRST * THF2, partial = nothing, spent = 0.0,
         fold = Tuple{Float64, Float64}[], θfold = Inf)
    checkpoint(S)
end
flush(stdout)
legs = S.legs
cur, prev, θc, θp, step, partial, spent, fold, θfold = S.cur, S.prev, S.θc, S.θp, S.step, S.partial, S.spent, S.fold, S.θfold
radius = get(S, :radius, 0.05)                        # the trust radius a cut-off point resumes with
state() = (; legs, cur, prev, θc, θp, step, partial, spent, fold, θfold, radius)
function predict(θ)
    isnothing(prev) && return cur.A
    if isfinite(θfold) && θ < θfold && θp > 0
        u, uc, up = sqrt(θfold - θ), sqrt(θfold - θc), sqrt(θfold - θp)
        return cur.A + (cur.A - prev.A) * ((u - uc) / (uc - up))
    end
    return cur.A + (cur.A - prev.A) * ((θ - θc) / (θc - θp))
end
function on(θ)
    x = ising3d_site(β; h = im * θ / β)
    return dev(T.replaceinds(x[1], collect(x[2]), collect(legs))), dev(T.replaceinds(x[3], collect(x[2]), collect(legs)))
end
done() = isfinite(θfold) && (θfold - θc) / θfold < VSTOP
while !done() && left() > 30
    θ = θc + step
    site, mag = on(θ)
    init, A0, r0 = isnothing(partial) ? (cur, predict(θ), 0.05) : (partial, partial.A, radius)
    tk = @elapsed rk, ik = quiet(() -> boundary_peps_krylov(site, legs, init; maxdim = CHI, A0, tol = TOL,
                                                             noise_tol = NOISE, radius = r0, time_limit = left()))
    global spent += tk
    stuck = !isnothing(partial) && ik.residual > 0.8 * partial.gnorm
    if !ik.converged && left() <= 1 && !stuck        # cut off by the budget: resume from here
        @printf("θ = %.10f  cut off at |g| = %.1e after %.0f s\n", θ, ik.residual, spent)
        global partial, radius = rk, ik.radius
        checkpoint(state())
        break
    end
    if !ik.converged                                 # a failed point: halve the step from θc
        @printf("θ = %.10f  UNCONVERGED at |g| = %.1e (%.0f s)%s: halving the step\n", θ, ik.residual, spent,
                stuck ? ", no gain on resume" : "")
        global step, partial, spent = step / 2, nothing, 0.0
        checkpoint(state())
        step < 1.0e-6 * max(θc, THF2) && break
        continue
    end
    m = site_ratio(rk, mag)
    ξ = first(correlation_length(rk.openv))
    push!(fold, (θ, imag(m)))
    if length(fold) >= 3                             # (dm/dθ)⁻² through the last three points, extrapolated to 0
        p = fold[(end - 2):end]
        xs = [(p[i][1] + p[i + 1][1]) / 2 for i in 1:2]
        ys = [((p[i + 1][2] - p[i][2]) / (p[i + 1][1] - p[i][1]))^-2 for i in 1:2]
        b = (ys[2] - ys[1]) / (xs[2] - xs[1])
        global θfold = b < 0 ? xs[2] - ys[2] / b : Inf
    end
    v = isfinite(θfold) ? (θfold - θ) / θfold : Inf
    @printf("θ = %.10f  f = %.13f  Im m = %.10f (Re %+.0e)  ξ = %6.3f  |g| = %.1e  %3d evals  %5.0f s  θ_f ≈ %.8f (v = %.1e)\n",
            θ, cvm_freenergy(rk), imag(m), real(m), ξ, ik.residual, ik.evals, spent, θfold, v)
    flush(stdout)
    open(io -> println(io, join((β, D, CHI, θ, cvm_freenergy(rk), imag(m), real(m), ξ, ik.evals, ik.residual,
                                 ik.converged, spent, θfold), ",")), OUT, "a")
    newstep = min(GROWTH * (θ - θc), MAXSTEP * THF2)
    isfinite(θfold) && (newstep = min(newstep, 0.5 * (θfold - θ)))
    global prev, cur, θp, θc, step, partial, spent = cur, rk, θc, θ, newstep, nothing, 0.0
    checkpoint(state())
end
done() && (println("REACHED v < $VSTOP"); touch(TAG * ".done"))
