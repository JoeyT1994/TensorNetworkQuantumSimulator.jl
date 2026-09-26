# Time-to-accuracy benchmark of the boundary-PEPS solvers (docs/boundary_peps.md) on 3D Ising: one
# solver, one (β, D, χ) per run, from a shared start state, within a wall-clock budget. Every run
# appends its trace (seconds since the solver started, f, |g|) and a summary row (final f, |g|, m) to
# CSV files, so runs of different solvers (and processes) compare directly.
#
# The start state is cached: D = 2 at the same β and χ converged to |g| < 1e-7 from the fixed-spin
# seed (ordered phase), then — for D > 2 — embedded at bond dimension D with relative noise 1e-2
# (seed 0) and its environments converged. Every solver starts from that same BoundaryPEPS.
#
# Solvers (BP_METHOD):
#   lbfgs           boundary_peps: preconditioned L-BFGS, adaptive CTMRG tolerance
#   krylov          boundary_peps_krylov: Newton–Krylov, subspace trust region on f
#   krylov_central  the same with central-difference products
#   hybrid          L-BFGS until |g| < BP_SWITCH (default 1e-4), then Newton–Krylov
#   newton_fd       boundary_peps_stationary: Newton with the full finite-difference Jacobian
#
# Environment: BP_METHOD, BP_BETA (0.2275), BP_D (3), BP_CHI (16), BP_BUDGET (seconds, 480),
# BP_TOL (target |g|, 1e-9), BP_SWITCH, BP_BLOCK and BP_RECYCLE (Newton–Krylov; 4 and 3),
# BP_ANDERSON (Anderson mixing in warm 2D CTMRG runs; unset: each solver's default), BP_CHECK
# (seconds between m checkpoints, 25; m(t) goes to *_mtrace.csv, evaluated after the run), BP_CACHE
# (directory for start states, "."), BP_OUT (prefix, "bench"), BP_SAVE=1 (serialize the final state),
# BP_INIT (resume from such a file) with BP_TOFFSET (seconds already spent, added to the trace).
# Run with threads (julia -t 8) and BLAS on one thread; loading, compiling and the m checkpoints add
# ~100 s to the budget. Results (D = 3 near β_c): docs/boundary_peps.md, "Routes to the boundary state".

using TensorNetworkQuantumSimulator
using LinearAlgebra, Printf, Serialization
using Logging: NullLogger, with_logger
BLAS.set_num_threads(1)

const METHOD = get(ENV, "BP_METHOD", "krylov")
const BETA = parse(Float64, get(ENV, "BP_BETA", "0.2275"))
const D = parse(Int, get(ENV, "BP_D", "3"))
const CHI = parse(Int, get(ENV, "BP_CHI", "16"))
const BUDGET = parse(Float64, get(ENV, "BP_BUDGET", "480"))
const TOL = parse(Float64, get(ENV, "BP_TOL", "1e-9"))
const SWITCH = parse(Float64, get(ENV, "BP_SWITCH", "1e-4"))
const CACHE = get(ENV, "BP_CACHE", ".")
const OUT = get(ENV, "BP_OUT", "bench")
const BLOCK = parse(Int, get(ENV, "BP_BLOCK", "4"))       # Newton–Krylov: products per Arnoldi step
const RECYCLE = parse(Int, get(ENV, "BP_RECYCLE", "3"))   # Newton–Krylov: directions carried between steps
# Anderson mixing in warm 2D CTMRG runs; unset: each solver's default (Newton–Krylov 5, L-BFGS 0)
const ANDERSON = haskey(ENV, "BP_ANDERSON") ? parse(Int, ENV["BP_ANDERSON"]) : nothing
const AKW = isnothing(ANDERSON) ? (;) : (; ctm_anderson = ANDERSON)
const LABEL = METHOD * (METHOD in ("krylov", "krylov_central", "hybrid") ? "_b$(BLOCK)r$(RECYCLE)" : "") *
              (isnothing(ANDERSON) ? "" : "_a$(ANDERSON)")
const VERBOSE = get(ENV, "BP_VERBOSE", "0") == "1"
const CHECK = parse(Float64, get(ENV, "BP_CHECK", "25"))    # seconds between magnetisation checkpoints
const INIT = get(ENV, "BP_INIT", "")        # resume from a saved final state
const TOFFSET = parse(Float64, get(ENV, "BP_TOFFSET", "0"))   # added to trace times (chained runs)
const SAVE = get(ENV, "BP_SAVE", "0") == "1"
const CTM_TOL = 1.0e-12

quiet(f) = with_logger(f, NullLogger())
site, legs, mag = ising3d_site(BETA)

function start_state()
    file = joinpath(CACHE, @sprintf("start_b%.5f_D%d_chi%d.jls", BETA, D, CHI))
    isfile(file) && return deserialize(file)
    bp = quiet() do
        b = boundary_peps(site, legs, 2; maxdim = CHI, boundary = [1.0, 0.0], maxiter = 2000, gtol = 1.0e-7,
                          ctm_tolerance = CTM_TOL)
        D == 2 ? b : boundary_peps(site, legs, D; maxdim = CHI, init = b, maxiter = 0, ctm_tolerance = CTM_TOL)
    end
    serialize(file, bp)
    return bp
end

bp0 = isempty(INIT) ? start_state() : deserialize(INIT)
println(@sprintf("%s β = %.5f D = %d χ = %d: start f = %.12f |g| = %.2e, budget %.0f s, threads %d", LABEL, BETA, D,
                 CHI, cvm_freenergy(bp0), bp0.gnorm, BUDGET, Threads.nthreads()))
flush(stdout)

trace = Tuple{Float64, Float64, Float64}[]            # (t, f, |g|)
t0 = Ref(time())
checkpoints = Any[]                                  # (t, A, Alegs) every CHECK seconds
nextcheck = Ref(0.0)
record(it, st) = (push!(trace, (TOFFSET + time() - t0[], st.f, st.res));
                  time() - t0[] >= nextcheck[] && (push!(checkpoints, (TOFFSET + time() - t0[], st.A, st.Alegs));
                                                   nextcheck[] += CHECK);
                  it % 10 == 0 && (@printf("  %7.1f s  it %4d  f = %.12f  |g| = %.2e\n", time() - t0[], it, st.f, st.res); flush(stdout)))

krylov(init, budget; kw...) = boundary_peps_krylov(site, legs, init; tol = TOL, maxiter = 10_000, time_limit = budget,
                                                   ctm_tolerance = CTM_TOL, block = BLOCK, recycle = RECYCLE,
                                                   verbose = VERBOSE, AKW...,
                                                   callback = record, kw...)[1]
lbfgs(init, budget; kw...) = boundary_peps(site, legs, D; maxdim = CHI, init, maxiter = 100_000, gtol = TOL,
                                           ctm_tolerance = CTM_TOL, time_limit = budget, callback = record,
                                           AKW..., kw...)

# compile both solvers' paths outside the timing (one short step each from the start state)
quiet() do
    boundary_peps(site, legs, D; maxdim = CHI, init = bp0, maxiter = 1, ctm_tolerance = CTM_TOL, AKW...)
    boundary_peps_krylov(site, legs, bp0; maxiter = 1, krylovdim = 2, block = min(BLOCK, 2), ctm_tolerance = CTM_TOL,
                         AKW...)
end
empty!(trace); empty!(checkpoints); nextcheck[] = 0.0
t0[] = time()
bp = quiet() do
    if METHOD == "lbfgs"
        lbfgs(bp0, BUDGET)
    elseif METHOD == "krylov"
        krylov(bp0, BUDGET)
    elseif METHOD == "krylov_central"
        krylov(bp0, BUDGET; central = true)
    elseif METHOD == "hybrid"
        b = lbfgs(bp0, BUDGET; gtol = SWITCH)
        krylov(b, BUDGET - (time() - t0[]))
    elseif METHOD == "newton_fd"
        boundary_peps_stationary(site, legs, bp0; tol = TOL, maxiter = 100, ctm_tolerance = CTM_TOL)[1]
    else
        error("unknown BP_METHOD $METHOD")
    end
end
elapsed = time() - t0[]
push!(trace, (TOFFSET + elapsed, cvm_freenergy(bp), bp.gnorm))
m = abs(real(site_ratio(bp, mag)))
@printf("%s done: %.1f s, f = %.13f, |g| = %.2e, m = %.10f\n", LABEL, elapsed, cvm_freenergy(bp), bp.gnorm, m)

tag = @sprintf("%s_b%.5f_D%d_chi%d", OUT, BETA, D, CHI)
open(tag * "_trace.csv", "a") do io
    for (t, f, r) in trace
        println(io, join((LABEL * "_t$(Threads.nthreads())", t, f, r), ","))
    end
end
open(tag * "_summary.csv", "a") do io
    println(io, join((LABEL, TOFFSET + elapsed, cvm_freenergy(bp), bp.gnorm, m, Threads.nthreads()), ","))
end
SAVE && serialize(tag * "_" * LABEL * "_final.jls", bp)

# m along the run: each checkpoint's A with its environments converged afresh (warm from the final ones)
function magnetisation(A, Alegs)
    fake = TensorNetworkQuantumSimulator.BoundaryPEPS(A, Tuple(Alegs), bp.blegs, bp.site, bp.legs, bp.normenv, bp.openv,
                                                      NaN, NaN, Float64[], false)
    b = quiet() do
        boundary_peps(site, legs, D; maxdim = CHI, init = fake, maxiter = 0, ctm_tolerance = CTM_TOL,
                      adaptive_tolerance = false)
    end
    return abs(real(site_ratio(b, mag)))
end
open(tag * "_mtrace.csv", "a") do io
    for (t, A, Alegs) in checkpoints
        println(io, join((LABEL * "_t$(Threads.nthreads())", t, magnetisation(A, Alegs)), ","))
    end
    println(io, join((LABEL * "_t$(Threads.nthreads())", TOFFSET + elapsed, m), ","))
end
