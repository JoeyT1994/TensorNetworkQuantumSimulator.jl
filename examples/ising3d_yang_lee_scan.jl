# The Yang–Lee edge of the 3D Ising model by the stationary (bilinear) boundary PEPS
# (docs/yang_lee.md): at fixed β on the high-temperature side, continue the boundary state from θ = 0
# (the real maximiser) along the imaginary field H = iθ towards the finite-D fold, recording f, m and
# the sandwich's correlation length. The fold comes from m: m − m_f ∝ √(θ_f − θ) at a fold, so
# (dm/dθ)⁻² extrapolates linearly to θ_f.
#
# Each point is a `boundary_peps_stationary` solve from a secant predictor, reusing the previous
# point's Jacobian; the step grows while points converge and halves when one does not. RESUMABLE: the
# state is checkpointed after every accepted point (YL_CKPT), so a scan is a chain of runs, each within
# YL_BUDGET seconds.
#
# Environment: YL_BETA (0.18), YL_D (2), YL_CHI (16), YL_VSTOP (stop at this relative distance to the
# fold estimate, 1e-3), YL_BUDGET (480), YL_THETA0 (first θ, 0.002), YL_OUT (CSV), YL_CKPT (checkpoint).
# D = 3 needs smaller continuation steps and looser solver tolerances (its redundant bond directions);
# the defaults switch on D. Run with threads (julia -t 8) and BLAS on one thread.

using TensorNetworkQuantumSimulator
using LinearAlgebra, Printf, Serialization
using Logging: NullLogger, with_logger
BLAS.set_num_threads(1)
const TNQS = TensorNetworkQuantumSimulator

const BETA = parse(Float64, get(ENV, "YL_BETA", "0.18"))
const D = parse(Int, get(ENV, "YL_D", "2"))
const CHI = parse(Int, get(ENV, "YL_CHI", "16"))
const VSTOP = parse(Float64, get(ENV, "YL_VSTOP", "1e-3"))
const BUDGET = parse(Float64, get(ENV, "YL_BUDGET", "480"))
const TAG = "beta$(BETA)_D$(D)_chi$(CHI)"
const OUT = get(ENV, "YL_OUT", "yl3d_$TAG.csv")
const CKPT = get(ENV, "YL_CKPT", "yl3d_$TAG.jls")
# solver settings (D = 3: ~15 near-null reduced-Jacobian directions, a bending branch)
const SOLVER = D >= 3 ?
    (; tol = 1.0e-7, svd_rtol = 1.0e-5, step_tol = 1.0e-5, noise_tol = 1.0e-5, fd_step = 1.0e-4, fd_ctm_tolerance = 1.0e-10) :
    (; tol = 3.0e-8, svd_rtol = 1.0e-6, step_tol = 1.0e-7, noise_tol = 1.0e-7, fd_step = 1.0e-5, fd_ctm_tolerance = 1.0e-12)
const GROWTH = D >= 3 ? 1.3 : 2.0            # continuation step growth per accepted point …
const MAXSTEP = D >= 3 ? 0.004 : Inf         # … and cap
const BETA_C = 0.2216544

quiet(f) = with_logger(f, NullLogger())
t0 = time()
@printf("3D Ising Yang–Lee scan: β = %.6f (t = %.4f), D = %d, χ = %d, threads = %d\n", BETA,
        (BETA_C - BETA) / BETA_C, D, CHI, Threads.nthreads())

if isfile(CKPT)
    S = deserialize(CKPT)
    println("resumed from $CKPT at θ = $(S.θcur) ($(length(S.fold)) points)")
else
    s0, legs, _ = ising3d_site(BETA)
    # θ = 0: the real maximiser, climbing D = 2, …, D (the first Newton solve polishes it)
    bp = nothing
    for d in 2:D
        global bp = quiet() do
            boundary_peps(s0, legs, d; maxdim = max(8, ceil(Int, CHI * d^2 / D^2)), init = bp,
                          gtol = d == D ? 1.0e-6 : 1.0e-7, maxiter = d == D ? 150 : 300)
        end
        @printf("θ = 0: D = %d f = %.12f |g| = %.1e (%.0f s)\n", d, cvm_freenergy(bp), bp.gnorm, time() - t0)
    end
    if bp.normenv.maxdim != CHI
        bp = quiet(() -> boundary_peps(s0, legs, D; maxdim = CHI, init = bp, gtol = 1.0e-6, maxiter = 50))
    end
    open(io -> println(io, "beta,D,chi,theta,f,m_imag,m_real,xi,iters,residual,converged"), OUT, "w")
    S = (; legs, prev = bp, prevA = nothing, θprev = 0.0, θcur = 0.0, J = nothing,
         fold = Tuple{Float64, Float64}[], θ = parse(Float64, get(ENV, "YL_THETA0", "0.002")),
         stepmax = Inf, θfold = Inf)
    serialize(CKPT, S)
end
flush(stdout)
legs = S.legs
# the site and magnetisation at field iθ, on the scan's legs
function on(θ)
    x = ising3d_site(BETA; h = im * θ / BETA)
    return TNQS.replaceinds(x[1], collect(x[2]), collect(legs)), TNQS.replaceinds(x[3], collect(x[2]), collect(legs))
end
prev, prevA, θprev, θcur, J, fold, θ, stepmax, θfold = S.prev, S.prevA, S.θprev, S.θcur, S.J, S.fold, S.θ, S.stepmax, S.θfold
checkpoint() = serialize(CKPT, (; legs, prev, prevA, θprev, θcur, J, fold, θ, stepmax, θfold))

# No Jacobian yet (a fresh start): build it at the current point with a short Newton polish and
# checkpoint (at D = 3 the Jacobian and the first point do not fit in one 10-minute run).
if isnothing(J)
    tp = @elapsed (rp, ip) = quiet() do
        boundary_peps_stationary(on(θcur)[1], legs, prev; maxdim = CHI, maxiter = 2, SOLVER...)
    end
    J = ip.J
    ip.residual <= 10 * prev.gnorm + 1.0e-6 && (prev = rp)
    @printf("Jacobian built at θ = %.5f (%.0f s, residual %.1e)\n", θcur, tp, ip.residual)
    checkpoint()
end

tlast = 0.0
while time() - t0 < BUDGET && time() - t0 + 1.2 * tlast < BUDGET + 80    # no point may overrun much
    site, mag = on(θ)
    A0 = isnothing(prevA) ? prev.A : prev.A + (prev.A - prevA) * ((θ - θcur) / (θcur - θprev))  # secant
    tt = @elapsed res, info = quiet() do
        boundary_peps_stationary(site, legs, prev; maxdim = CHI, A0, jacobian = J, maxiter = 10,
                                 refresh_jacobian = D < 3, SOLVER...)
    end
    global tlast = tt
    m = site_ratio(res, mag)
    ξ = info.converged ? first(correlation_length(res.openv)) : NaN
    @printf("θ %.10e  f %.13f  m %+.10f i (re %+.0e)  ξ %7.3f  it %2d res %.0e %s %.1fs\n", θ,
            cvm_freenergy(res), imag(m), real(m), ξ, info.iterations, info.residual,
            info.converged ? "" : "UNCONVERGED", tt)
    flush(stdout)
    open(io -> println(io, join((BETA, D, CHI, θ, cvm_freenergy(res), imag(m), real(m), ξ, info.iterations,
                                 info.residual, info.converged), ",")), OUT, "a")
    if !info.converged                          # back off
        global stepmax = (θ - θcur) / 2
        global θ = θcur + stepmax
        stepmax < 1.0e-7 * θcur && break
        continue
    end
    global prevA, θprev, θcur, prev, J
    prevA, θprev, θcur, prev, J = prev.A, θcur, θ, res, info.J
    push!(fold, (θ, imag(m)))
    if length(fold) >= 3                        # (dm/dθ)⁻² through the last three points, extrapolated to 0
        p = fold[(end - 2):end]
        xs = [(p[i][1] + p[i + 1][1]) / 2 for i in 1:2]
        ys = [((p[i + 1][2] - p[i][2]) / (p[i + 1][1] - p[i][1]))^-2 for i in 1:2]
        b = (ys[2] - ys[1]) / (xs[2] - xs[1])
        global θfold = b < 0 ? xs[2] - ys[2] / b : Inf
    end
    v = isfinite(θfold) ? (θfold - θ) / θfold : Inf
    isfinite(v) && @printf("   fold estimate θ_f = %.10e, v = %.2e\n", θfold, v)
    step = min(GROWTH * (θcur - θprev), 1.5 * stepmax, MAXSTEP)
    isfinite(θfold) && (step = min(step, 0.5 * (θfold - θcur)))
    global stepmax = step
    global θ = θcur + step
    checkpoint()
    isfinite(v) && v < VSTOP && (println("reached v < $VSTOP"); break)
end
println("done in $(round(time() - t0)) s")
