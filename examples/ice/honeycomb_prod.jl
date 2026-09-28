# PRODUCTION driver for the honeycomb ice campaign (docs/ice.md, docs/status_3d.md): the BP simple-update
# state at bond dimension D, then the three PAIRED networks ⟨ψ|ψ⟩, ⟨Iψ|ψ⟩, ⟨Iψ|Mψ⟩ by infinite CTMRG at χ.
# RESTARTABLE and DEVICE-AGNOSTIC: run it again with the same OUT and it resumes.
#
# Per (D, χ) in OUT: su_D<D>.jls (the state, built once), ctm_D<D>_chi<χ>_<kind>.jls (an atomic host
# checkpoint after every chunk: environment, iterations, converged, ln κ, s/iteration), a row in
# results.csv per converged network, xi_D<D>_chi<χ>.jls (ξ of ⟨ψ|ψ⟩), and a row in summary.csv
# (RQ → w_h = exp(RQ/2), ln F_I, ξ) once all three networks of (D, χ) are converged — whichever run
# converges the last one writes it, so the kinds can run in separate lanes. KINDS may add `mnorm`
# (⟨Mψ|Mψ⟩, raw bond 4D²): then residual.csv gets ln f = 2RQ(A) − RQ(A²), the eigenvector residual.
#
# ENV: D (4), CHI (D²), KINDS ("norm,inv,sand"), CHUNK (CTM iterations per checkpoint, 5), BUDGET (s, 540:
# a chunk starts only if its expected length fits), CONV (lnkappa | environment), TOL (lnkappa: 2e-14,
# relative to |ln κ| ≈ 50–100, i.e. ~1e-12 per iteration; environment: 1e-9 gives ln κ to ~1e-13), MINITS (15),
# MAXIT (600), DEVICE (cpu | gpu: CUDA, FP64), BLASN (4), OUT ("ice_prod"), SVD_OVERSAMPLE (ceil(1.3χ)).
#
# THE PROJECTOR'S OVERSAMPLING. Each pair is a subspace SVD of k = χ columns plus an oversampling block;
# when it cannot converge (a flat spectrum near k — the sandwich at χ = D² has one) it BAILS OUT to the
# dense route, which forms and factorises the n × n enlarged quadrant (n = χ·r): O(n³), and at large D
# ruinous (n ≈ 41 000 at D = 12). With the library default (16) the D = 6 sandwich bailed on 5 of 12
# pairs and the D = 7 one on 2 of 8: 177 s per iteration on the RTX 3070 against 9.2 s with 64 (1.3χ),
# every pair split, ln κ identical to 10 digits (2026-09-27). Every chunk logs its split/dense counts —
# a dense pair in production is a bug to chase, not noise.
#
# Costs measured on the local i9 (2026-09-27, s per CTM iteration, near convergence; early ones up to
# ~2× more): ⟨ψ|ψ⟩ D = 6 χ = 36 ~2–4, D = 7 χ = 49 ~6, D = 8 χ = 64 ~30; ⟨Iψ|Mψ⟩ (raw leg 2D², ~4× the
# norm's) D = 6 ~10–25, D = 7 ~50–100, D = 8 ~300–470. ~35–40 iterations to converge. Scaling ~χ³r² ~ D¹⁰
# at χ = D². χ below D² is NOT enough for F_I (D = 6: χ = 24 gives −5.3e-6, χ = 16 −1.5e-5 vs −5.19e-6).
include(joinpath(@__DIR__, "honeycomb.jl"))
using Serialization, Adapt
using Logging: NullLogger, with_logger
const DEVICE = get(ENV, "DEVICE", "cpu")
if DEVICE == "gpu"
    @eval using CUDA
    CUDA.allowscalar(false)
end
todev(x) = DEVICE == "gpu" ? adapt(CUDA.CuArray, x) : x
tohost(x) = DEVICE == "gpu" ? adapt(Array, x) : x
atomic_serialize(f, x) = (serialize(f * ".tmp", x); mv(f * ".tmp", f; force = true))
BLAS.set_num_threads(parse(Int, get(ENV, "BLASN", "4")))
const D = parse(Int, get(ENV, "D", "4")); const χ = parse(Int, get(ENV, "CHI", string(D^2)))
const KINDS = Symbol.(split(get(ENV, "KINDS", "norm,inv,sand"), ","))
const CHUNK = parse(Int, get(ENV, "CHUNK", "5")); const BUDGET = parse(Float64, get(ENV, "BUDGET", "540"))
const CONV = Symbol(get(ENV, "CONV", "lnkappa"))
const TOL = parse(Float64, get(ENV, "TOL", CONV === :lnkappa ? "2e-14" : "1e-9")); const MAXIT = parse(Int, get(ENV, "MAXIT", "600"))
const MINITS = parse(Int, get(ENV, "MINITS", "15"))           # no convergence before (the seed's rank still grows)
const OUT = get(ENV, "OUT", "ice_prod"); mkpath(OUT)
const OVERSAMPLE = parse(Int, get(ENV, "SVD_OVERSAMPLE", string(max(16, ceil(Int, 1.3χ)))))

function run_prod()
    t0 = time()
    sufile = joinpath(OUT, "su_D$(D).jls")
    if isfile(sufile)
        X, Y, w, suinfo = deserialize(sufile)
    else
        ts = time()
        X, Y, w, suinfo = hexstate(D)
        atomic_serialize(sufile, (X, Y, w, suinfo))
        @printf("D = %d: SU %d bilayers, Δw %.1e, discarded %.1e (%.1f s)\n", D, suinfo.steps, suinfo.δ, suinfo.err, time() - ts)
    end
    done = Dict{Symbol, Float64}(); newly = Symbol[]                 # rows only when something converged now
    for kind in KINDS
        ck = joinpath(OUT, "ctm_D$(D)_chi$(χ)_$(kind).jls")
        prev = isfile(ck) ? deserialize(ck) : nothing
        if !isnothing(prev) && prev.converged
            done[kind] = prev.lnk
            continue
        end
        site, legs, sv = paired(X, Y, w; kind)
        site = Any[todev(t) for t in site]
        ic = isnothing(prev) ? nothing : todev(prev.ic)
        its = isnothing(prev) ? 0 : prev.its
        conv = false; lnκ = NaN; chg = NaN
        sperit = isnothing(prev) ? 0.0 : get(prev, :sperit, 0.0)
        lastdur = CHUNK * sperit                                   # the next chunk's expected length
        while its < MAXIT && time() - t0 + lastdur < BUDGET
            tc = time()
            empty!(T.CTM_SVD_STATS)
            ic = with_logger(NullLogger()) do
                # miniter = 1 on a resumed chunk: `update` never reports convergence before miniter
                # iterations, so CHUNK = 1 with the default 2 would never converge
                T.update(T.InfiniteCTM2D(site, legs, χ; init = ic, boundary = sv, svd_oversample = OVERSAMPLE);
                         tolerance = TOL, maxiter = CHUNK, miniter = isnothing(ic) ? 2 : 1, convergence = CONV)
            end
            st = ic.stats[]
            its += st.iterations; conv = st.converged && its >= MINITS; chg = st.change
            lastdur = time() - tc; sperit = lastdur / st.iterations
            lnκ = T.cvm_freenergy(ic)
            atomic_serialize(ck, (; ic = tohost(ic), its, converged = conv, lnk = lnκ, change = chg, sperit))
            # dense pairs: the subspace gate declined (small n — fine) or the subspace SVD bailed (flag)
            nsplit = get(T.CTM_SVD_STATS, :i2_split, 0); nbail = get(T.CTM_SVD_STATS, :i2_split_bail, 0)
            @printf("  D = %d χ = %d %-4s: %4d its, ln κ = %+.13f, Δ %.1e%s  (%.1f s/it; pairs %d split, %d dense%s)\n",
                    D, χ, kind, its, lnκ, chg, conv ? " CONVERGED" : "", sperit, nsplit, 4st.iterations - nsplit,
                    nbail > 0 ? ", $nbail BAIL-OUTS — raise SVD_OVERSAMPLE" : "")
            flush(stdout)
            conv && break
        end
        if conv
            done[kind] = lnκ; push!(newly, kind)
            open(joinpath(OUT, "results.csv"), "a") do io
                println(io, join((D, χ, kind, @sprintf("%.14f", lnκ), its, @sprintf("%.1e", chg), DEVICE), ","))
            end
            if kind === :norm
                ξ, _ = T.correlation_length(ic)
                atomic_serialize(joinpath(OUT, "xi_D$(D)_chi$(χ).jls"), ξ)
            end
        elseif its >= MAXIT
            println("  MAXIT: $kind stopped unconverged at $its its (Δ $chg)")
            break
        else
            println("  (budget: $kind at $its its, resume to continue)")
            break
        end
    end
    for k in (:norm, :inv, :sand, :mnorm)                           # networks other lanes converged
        f = joinpath(OUT, "ctm_D$(D)_chi$(χ)_$(k).jls")
        if !haskey(done, k) && isfile(f)
            c = deserialize(f)
            c.converged && (done[k] = c.lnk)
        end
    end
    if !isempty(newly) && all(k -> haskey(done, k), (:norm, :inv, :sand))
        rq = done[:sand] - done[:norm]; lf = done[:inv] - done[:norm]
        xf = joinpath(OUT, "xi_D$(D)_chi$(χ).jls"); ξ = isfile(xf) ? deserialize(xf) : NaN
        @printf("SUMMARY D = %d χ = %d: RQ = %.12f  w_h = %.10f  ln F_I = %+.5e  ξ = %.4f\n", D, χ, rq, exp(rq / 2), lf, ξ)
        open(joinpath(OUT, "summary.csv"), "a") do io
            println(io, join((D, χ, @sprintf("%.14f", rq), @sprintf("%.11f", exp(rq / 2)), @sprintf("%.6e", lf),
                              @sprintf("%.5f", ξ), DEVICE), ","))
        end
    end
    if !isempty(newly) && all(k -> haskey(done, k), (:norm, :sand, :mnorm))   # the eigenvector residual
        rq = done[:sand] - done[:norm]; rq2 = done[:mnorm] - done[:norm]
        @printf("RESIDUAL D = %d χ = %d: RQ(A) = %.12f  RQ(A²) = %.12f  ln f = 2RQ − RQ₂ = %+.5e per cell\n", D, χ,
                rq, rq2, 2rq - rq2)
        open(joinpath(OUT, "residual.csv"), "a") do io
            println(io, join((D, χ, @sprintf("%.14f", rq), @sprintf("%.14f", rq2), @sprintf("%.6e", 2rq - rq2), DEVICE), ","))
        end
    end
    @printf("elapsed %.0f s\n", time() - t0)
end
run_prod()
