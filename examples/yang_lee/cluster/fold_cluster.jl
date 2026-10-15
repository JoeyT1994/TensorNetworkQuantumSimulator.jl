# The exact Yang–Lee fold (fit-free) for cluster jobs: pseudo-arclength through the turning point of the
# stationary bilinear boundary PEPS (../fold_pseudoarclength.jl), RESUMABLE and GPU-capable.
#
# Start (first run only), either
#   INIT=<a scan_krylov.jl checkpoint .jls>  — its two last converged states (prev at θp, cur at θc), or
#   THS=θ1,θ2,…                              — Newton–Krylov continuation from θ = 0 (D = 2 at χ = 16 embedded).
# Phase 1, the walk: adaptive arclength steps until the path has passed its highest θ (AFTER points below it).
# Phase 2, the refinement: back to the point before the highest one, and NREF short steps (an NREF-th of the coarse
# span over the turn) through the turn. θ_f is the maximum of a local CUBIC in arclength through the five refined
# points nearest the top (θ(s) is asymmetric about the turn: a quadratic there fell below points actually reached);
# the quadratic vertex is reported as a check, and the highest θ reached is a rigorous lower bound.
# Every accepted point is appended to fold_D<D>_beta<β>_chi<χ>.csv (phase 1 or 2) and the run is checkpointed
# (fold_…ckpt.jls, on the host); a rerun resumes. The result goes to fold_…result and a FOLD line.
# Env: BETA, D, CHI, GPU (1), INIT or THS, BUDGET (s; on running out prints "(budget" and exits 0), DTHETA (the
# first step in θ; default a quarter of the last θ spacing), REM (|Re m| guard, 1e-4), AFTER (3), NREF (8),
# MAXSTEPS (80).
include(joinpath(@__DIR__, "..", "fold_pseudoarclength.jl"))
using Serialization
using Logging: NullLogger, with_logger
const GPU = get(ENV, "GPU", "0") == "1"
if GPU
    using CUDA, Adapt
end
GPU || BLAS.set_num_threads(1)
dev(x) = (GPU && !isnothing(x)) ? adapt(CuArray, x) : x
host(x) = (GPU && !isnothing(x)) ? adapt(Array, x) : x
const β = parse(Float64, ENV["BETA"])
const D = parse(Int, ENV["D"])
const χ = parse(Int, ENV["CHI"])
const BUDGET = parse(Float64, get(ENV, "BUDGET", "Inf"))
const REM = parse(Float64, get(ENV, "REM", "1e-4"))
const AFTER = parse(Int, get(ENV, "AFTER", "3"))
const NREF = parse(Int, get(ENV, "NREF", "8"))
const MAXSTEPS = parse(Int, get(ENV, "MAXSTEPS", "80"))
const TAG = "fold_D$(D)_beta$(β)_chi$(χ)"
const CKPT, CSV, RES = TAG * ".ckpt.jls", TAG * ".csv", TAG * ".result"
quiet(f) = with_logger(f, NullLogger())
redirect_stderr(stdout)

hostst(s) = isnothing(s) ? s : FoldState(s.c, s.θ, s.g, s.f, host(s.ln), host(s.ls))
devst(s) = FoldState(s.c, s.θ, s.g, s.f, dev(s.ln), dev(s.ls))
checkpoint(W) = (serialize(CKPT * ".tmp", W); mv(CKPT * ".tmp", CKPT; force = true))

# θ_f from points (s, θ, m, ξ) around the turn: the vertex of a quadratic through the five nearest the top
function fit_turn(pts)
    θs = [p.θ for p in pts]; imax = argmax(θs)
    o = sortperm(abs.([p.s for p in pts] .- pts[imax].s)); sel = sort(o[1:min(5, length(pts))])
    S = [pts[i].s for i in sel]; Θ = θs[sel]
    X = [S .^ 2 S ones(length(S))]; a2, a1, a0 = X \ Θ
    sf = -a1 / (2a2); θq = a0 - a1^2 / (4a2)
    ev(c) = c[1] * sf^2 + c[2] * sf + c[3]
    mq, ξq = ev(X \ [pts[i].m for i in sel]), ev(X \ [pts[i].ξ for i in sel])
    Xc = [S .^ 3 S .^ 2 S ones(length(S))]; c = Xc \ Θ
    grid = range(minimum(S), maximum(S); length = 20001)
    θcub = maximum(c[1] .* grid .^ 3 .+ c[2] .* grid .^ 2 .+ c[3] .* grid .+ c[4])
    rms = sqrt(sum(abs2, X * [a2, a1, a0] - Θ) / length(S))
    return (; θq, θcub, θmax = θs[imax], m = mq, ξ = ξq, rms, interior = 1 < imax < length(pts))
end

function main()
    t0 = time()
    deadline = t0 + BUDGET
    if isfile(CKPT)
        W = deserialize(CKPT); legs = W.legs
        println("resuming $TAG ($(W.phase)): $(length(W.path)) path points, $(length(W.fine)) refined")
    else
        if haskey(ENV, "INIT")
            S0 = deserialize(ENV["INIT"]); legs = S0.legs
            prevbp, curbp, θp, θc = S0.prev, S0.cur, S0.θp, S0.θc
            println("start from $(ENV["INIT"]): θp = $θp, θc = $θc")
        else
            s0, legs, _ = ising3d_site(β); s0 = dev(s0)
            bp = quiet(() -> boundary_peps(s0, legs, 2; maxdim = 16, gtol = 1.0e-7, maxiter = 300))
            D > 2 && (bp = quiet(() -> boundary_peps(s0, legs, D; maxdim = χ, init = bp, gtol = 1.0e-6, maxiter = 600)))
            @printf("θ = 0: f = %.12f |g| = %.1e (%.0f s)\n", cvm_freenergy(bp), bp.gnorm, time() - t0); flush(stdout)
            prevbp, curbp, θp, θc = nothing, bp, 0.0, 0.0
            for θ in parse.(Float64, split(ENV["THS"], ","))
                site = dev((x = ising3d_site(β; h = im * θ / β); T.replaceinds(x[1], collect(x[2]), collect(legs))))
                A0 = isnothing(prevbp) ? curbp.A : curbp.A + (curbp.A - prevbp.A) * ((θ - θc) / (θc - θp))
                r, i = quiet(() -> boundary_peps_krylov(site, legs, curbp; maxdim = χ, A0, tol = 1.0e-9, noise_tol = 1.0e-7))
                @printf("θ = %.6f: Newton–Krylov |g| = %.1e, %d evals (%.0f s)\n", θ, i.residual, i.evals, time() - t0)
                flush(stdout)
                prevbp, curbp, θp, θc = curbp, r, θc, θ
            end
            prevbp, curbp = host(prevbp), host(curbp)
        end
        W = (; legs, start = (; prevbp, curbp, θp, θc), phase = :walk, path = NamedTuple[], fine = NamedTuple[],
             s = nothing, tc = nothing, tθ = 0.0, Δs = 0.0, spos = 0.0, after = 0, al = nothing, bl = nothing,
             recent = Dict{Int, Any}(), ref = nothing)
    end
    site_at(θ) = dev((x = ising3d_site(β; h = im * θ / β); T.replaceinds(x[1], collect(x[2]), collect(legs))))
    mag_at(θ) = dev((x = ising3d_site(β; h = im * θ / β); T.replaceinds(x[3], collect(x[2]), collect(legs))))
    if isnothing(W.s)                               # the first two points of the walk
        cur = dev(W.start.curbp); prev = dev(W.start.prevbp)
        al, bl = cur.Alegs, cur.blegs
        B0 = T._bp_c4v_basis(al)
        coords(b) = (c = T._bp_to_coords(b.A, B0, al); c / norm(c))
        tostate(b, θ) = (c = coords(b); (f, g, ln, ls) = evalg(site_at, legs, al, bl, B0, c, θ, χ, b.normenv, b.openv);
                         FoldState(c, θ, g, f, ln, ls))
        sa, sb = tostate(prev, W.start.θp), tostate(cur, W.start.θc)
        tc = sb.c - sa.c; tθ = sb.θ - sa.θ; nrm = sqrt(norm(tc)^2 + tθ^2); tc, tθ = tc / nrm, tθ / nrm
        dθ = parse(Float64, get(ENV, "DTHETA", string((W.start.θc - W.start.θp) / 4)))
        recent = Dict{Int, Any}(0 => (st = hostst(sb), tc, tθ, s = 0.0))
        W = merge(W, (; s = hostst(sb), tc, tθ, Δs = dθ / tθ, al, bl, recent, start = nothing))
        open(io -> println(io, "phase,s,theta,im_m,re_m,xi,f,seconds"), CSV, "w")
        checkpoint(W)
    end
    al, bl = W.al, W.bl
    B = T._bp_c4v_basis(al)
    bpof(st) = T.BoundaryPEPS(T._bp_from_coords(st.c, B, al, site_at(st.θ)), al, bl, site_at(st.θ), Tuple(legs), st.ln,
                              st.ls, st.f, norm(st.g), [st.f], true)
    # One arclength step from (s, tc, tθ) of length Δs: the accepted point and its secant, or nothing
    function step(s, tc, tθ, Δs, isfirst)
        new, _, _ = arcstep(site_at, legs, al, bl, B, χ, s, tc, tθ, Δs; noise = 1.0e-7, verbose = false)
        isnothing(new) && return nothing, "corrector failed"
        stc = new.c - s.c; stθ = new.θ - s.θ; nrm = sqrt(norm(stc)^2 + stθ^2); stc, stθ = stc / nrm, stθ / nrm
        ang = acos(clamp(real(dot(stc, tc)) + stθ * tθ, -1.0, 1.0))
        m = site_ratio(bpof(new), mag_at(new.θ))
        if (!isfirst && ang > 0.3) || abs(real(m)) > REM
            return nothing, @sprintf("rejected (θ = %.11f, turn %.3f, Re m %+.1e)", new.θ, ang, real(m))
        end
        ξ = first(correlation_length(new.ls))
        return (; new, stc, stθ, nrm, ang, m, ξ), ""
    end
    record(phase, p, sec) = (open(io -> println(io, join((phase, p.s, p.θ, p.m, p.rem, p.ξ, p.f, sec), ",")), CSV, "a");
                             @printf("%s s = %.6f: θ = %.11f  Im m = %.8f (Re %+.0e)  ξ = %.3f  (%.0f s)\n", phase, p.s,
                                     p.θ, p.m, p.rem, p.ξ, sec); flush(stdout))

    # ---- phase 1: the walk -----------------------------------------------------------------------
    if W.phase === :walk
        s, tc, tθ, Δs, spos, after, path, recent = devst(W.s), W.tc, W.tθ, W.Δs, W.spos, W.after, copy(W.path),
                                                   copy(W.recent)
        while after < AFTER && length(path) < MAXSTEPS
            time() > deadline && (println("(budget: walk, $(length(path)) path points, θ = $(s.θ); resume to continue)");
                                  return)
            r, why = step(s, tc, tθ, Δs, isempty(path))
            if isnothing(r)
                Δs /= 2; @printf("  %s: Δs → %.2e\n", why, Δs); flush(stdout)
                W = merge(W, (; Δs)); checkpoint(W); continue
            end
            spos += r.nrm
            pt = (; s = spos, θ = r.new.θ, m = imag(r.m), rem = real(r.m), ξ = r.ξ, f = r.new.f)
            push!(path, pt); record("walk", pt, time() - t0)
            s, tc, tθ = r.new, r.stc, r.stθ
            recent[length(path)] = (st = hostst(s), tc, tθ, s = spos)
            for k in collect(keys(recent)); k < length(path) - 6 && delete!(recent, k); end
            s.θ < maximum(p.θ for p in path) && (after += 1)
            Δs = min(r.ang < 0.1 ? 1.5Δs : r.ang > 0.2 ? 0.7Δs : Δs, 0.05)
            W = merge(W, (; s = hostst(s), tc, tθ, Δs, spos, after, path, recent)); checkpoint(W)
        end
        θs = [p.θ for p in path]; imax = argmax(θs)
        if after < AFTER || imax == length(path)
            println("the fold was not passed (MAXSTEPS = $MAXSTEPS)"); return
        end
        k = imax - 1                                 # refine from the point before the top
        haskey(recent, k) || (k = maximum(filter(<(imax), collect(keys(recent)))))
        span = path[imax + 1].s - recent[k].s
        W = merge(W, (; phase = :refine, ref = (; st = recent[k].st, tc = recent[k].tc, tθ = recent[k].tθ, s = recent[k].s,
                                                    Δs = span / NREF, n = 0)))
        checkpoint(W)
        @printf("passed the turn at θ ≈ %.11f; refining from s = %.5f in %d steps of %.2e\n", θs[imax], recent[k].s, NREF,
                span / NREF); flush(stdout)
    end

    # ---- phase 2: the refinement through the turn -------------------------------------------------
    if W.phase === :refine
        R = W.ref; s, tc, tθ, spos, Δs, n, fine = devst(R.st), R.tc, R.tθ, R.s, R.Δs, R.n, copy(W.fine)
        while n < NREF + 1
            time() > deadline && (println("(budget: refinement, $(length(fine)) refined points; resume to continue)"); return)
            r, why = step(s, tc, tθ, Δs, false)
            if isnothing(r)
                Δs /= 2; @printf("  %s: Δs → %.2e\n", why, Δs); flush(stdout)
                W = merge(W, (; ref = merge(R, (; Δs)))); R = W.ref; checkpoint(W); continue
            end
            spos += r.nrm; n += 1
            pt = (; s = spos, θ = r.new.θ, m = imag(r.m), rem = real(r.m), ξ = r.ξ, f = r.new.f)
            push!(fine, pt); record("refine", pt, time() - t0)
            s, tc, tθ = r.new, r.stc, r.stθ
            R = (; st = hostst(s), tc, tθ, s = spos, Δs, n)
            W = merge(W, (; ref = R, fine)); checkpoint(W)
        end
        W = merge(W, (; phase = :done)); checkpoint(W)
    end

    r = fit_turn(W.fine)
    line = @sprintf("FOLD D = %d β = %.4f χ = %d: θ_f = %.11f (local cubic); quadratic %.11f (rms %.1e); highest θ reached %.11f%s; Im m_f ≈ %.8f, ξ_f ≈ %.3f",
                    D, β, χ, max(r.θcub, r.θmax), r.θq, r.rms, r.θmax, r.interior ? "" : " (top at the END of the refined points: widen NREF)",
                    r.m, r.ξ)
    println(line); write(RES, line * "\n")
    @printf("total %.0f s this run\n", time() - t0)
end
main()
