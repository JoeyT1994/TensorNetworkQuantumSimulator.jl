# The fold prototype at 3D Ising, β = 0.20 (ENV BETA), D = 2 (ENV D), χ = 16 (ENV CHI): Newton–Krylov
# along θ to ~10 % from the fold, then pseudo-arclength through it. θ_f from the parabola through
# the three points around the turning point; against the D = 2 scan's six-point fit 0.0167357.
include(joinpath(@__DIR__, "fold_pseudoarclength.jl"))
using Serialization
using Logging: NullLogger, with_logger
BLAS.set_num_threads(1)
const β = parse(Float64, get(ENV, "BETA", "0.2"))
const D = parse(Int, get(ENV, "D", "2"))
const χ = parse(Int, get(ENV, "CHI", "16"))
const THS = parse.(Float64, split(get(ENV, "THS", "0.006,0.011,0.014,0.0155,0.0162"), ","))
const TLIMIT = parse(Float64, get(ENV, "TLIMIT", "510"))   # seconds for the whole run
quiet(f) = with_logger(f, NullLogger())
t0 = time()
s0, legs, _ = ising3d_site(β)
bp = quiet(() -> boundary_peps(s0, legs, 2; maxdim = 16, gtol = 1.0e-7, maxiter = 300))
if D > 2
    bp = quiet(() -> boundary_peps(s0, legs, D; maxdim = χ, init = bp, gtol = 1.0e-6, maxiter = 200))
end
site_at(θ) = (x = ising3d_site(β; h = im * θ / β); T.replaceinds(x[1], collect(x[2]), collect(legs)))
mag_at(θ) = (x = ising3d_site(β; h = im * θ / β); T.replaceinds(x[3], collect(x[2]), collect(legs)))
@printf("θ = 0: f = %.12f (%.0f s)\n", cvm_freenergy(bp), time() - t0); flush(stdout)
states = Any[]; prev = nothing; θp = 0.0; θc = 0.0; cur = bp
for θ in THS
    A0 = isnothing(prev) ? cur.A : cur.A + (cur.A - prev.A) * ((θ - θc) / (θc - θp))
    r, i = quiet(() -> boundary_peps_krylov(site_at(θ), legs, cur; maxdim = χ, A0, tol = 1.0e-9, noise_tol = 1.0e-7))
    @printf("θ = %.4f: Newton–Krylov |g| = %.1e, %d evals, Im m = %.10f\n", θ, i.residual, i.evals, imag(site_ratio(r, mag_at(θ))))
    flush(stdout)
    global prev, cur, θp, θc = cur, r, θc, θ
    push!(states, (θ, r))
end
al, bl = cur.Alegs, cur.blegs
B = T._bp_c4v_basis(al)
coords(bp) = (c = T._bp_to_coords(bp.A, B, al); c / norm(c))
function tostate(bp, θ)
    c = coords(bp)
    f, g, ln, ls = evalg(site_at, legs, al, bl, B, c, θ, χ, bp.normenv, bp.openv)
    return FoldState(c, θ, g, f, ln, ls)
end
sa, sb = tostate(states[end - 1][2], states[end - 1][1]), tostate(states[end][2], states[end][1])
tc = sb.c - sa.c; tθ = sb.θ - sa.θ; nrm = sqrt(norm(tc)^2 + tθ^2); tc, tθ = tc / nrm, tθ / nrm
@printf("secant tangent: t_θ = %.4f (Δθ per unit arclength), |Δc| per Δθ = %.1f\n", tθ, norm(sb.c - sa.c) / (sb.θ - sa.θ))
# SECANT predictor and step control: the direction from the last two converged points (the tangent
# solve with a Broyden-updated J was too noisy to gate steps on). A step is rejected (Δs halved) when
# the secant turns by more than ANGLE rad or the state leaves the PT-symmetric branch (|Re m| > REM):
# at the fold the PT-symmetric curve crosses the PT-broken one (m ≈ m_f + a√(θ_f − θ), real on one
# side, a conjugate pair on the other), and a long step lands on the wrong one.
const ANGLE, REM = 0.3, 1.0e-4
bpof(st) = T.BoundaryPEPS(T._bp_from_coords(st.c, B, al, site_at(st.θ)), al, bl, site_at(st.θ), Tuple(legs), st.ln, st.ls,
                          st.f, norm(st.g), [st.f], true)
Δs = 1.0e-4 / tθ                                       # first step ≈ Δθ = 1e-4
m0 = site_ratio(bpof(sb), mag_at(sb.θ))
path = [(s = 0.0, θ = sb.θ, tθ = tθ, m = imag(m0), rem = real(m0), ξ = first(correlation_length(sb.ls)))]
s = sb; spos = 0.0; after = 0
for k in 1:60
    time() - t0 > TLIMIT && (println("(time)"); break)
    new, J, Gθ = arcstep(site_at, legs, al, bl, B, χ, s, tc, tθ, Δs; noise = 1.0e-7, verbose = false)
    if isnothing(new)
        global Δs /= 2; @printf("  corrector failed: Δs → %.2e\n", Δs); continue
    end
    stc = new.c - s.c; stθ = new.θ - s.θ; nrm = sqrt(norm(stc)^2 + stθ^2); stc, stθ = stc / nrm, stθ / nrm
    ang = acos(clamp(real(dot(stc, tc)) + stθ * tθ, -1.0, 1.0))
    m = site_ratio(bpof(new), mag_at(new.θ))
    if (k > 1 && ang > ANGLE) || abs(real(m)) > REM
        global Δs /= 2
        @printf("  rejected (θ = %.11f, secant turn %.3f, Re m %+.1e): Δs → %.2e\n", new.θ, ang, real(m), Δs); continue
    end
    _, btθ, _ = tangent(al, B, new, J, Gθ, stc, stθ)    # the bordered tangent's θ-slope, for the fold fit
    global spos += nrm
    ξ = first(correlation_length(new.ls))
    push!(path, (s = spos, θ = new.θ, tθ = btθ, m = imag(m), rem = real(m), ξ))
    @printf("s = %.5f: θ = %.11f  dθ/ds = %+.5f  turn %.3f  Im m = %.8f (Re %+.0e)  ξ = %.3f  (%.0f s)\n", spos, new.θ, btθ,
            ang, imag(m), real(m), ξ, time() - t0)
    flush(stdout)
    global s, tc, tθ = new, stc, stθ
    new.θ < maximum(x.θ for x in path) && (global after += 1) >= 3 && break
    global Δs = min(ang < 0.1 ? 1.5Δs : ang > 0.2 ? 0.7Δs : Δs, 0.05)
end
serialize("fold_path_D$(D)_beta$(β).jls", path)
# θ_f: a quadratic in s through the highest point and its neighbours (both sides once passed)
imax = argmax([x.θ for x in path])
if imax == length(path) || imax == 1
    @printf("the fold was not passed (highest θ = %.11f at the %s)\n", path[imax].θ, imax == 1 ? "start" : "end")
else
    rng = max(1, imax - 2):min(length(path), imax + 2)
    S = [path[i].s for i in rng]; Θ = [path[i].θ for i in rng]
    X = [S .^ 2 S ones(length(S))]
    a2, a1, a0 = X \ Θ
    sf = -a1 / (2a2)
    mfit = X \ [path[i].m for i in rng]; ξfit = X \ [path[i].ξ for i in rng]
    ev(c) = c[1] * sf^2 + c[2] * sf + c[3]
    @printf("FOLD: θ_f = %.11f at s = %.5f (quadratic through %d points, rms %.1e), Im m_f ≈ %.8f, ξ_f ≈ %.3f\n",
            a0 - a1^2 / (4a2), sf, length(S), sqrt(sum(abs2, X * [a2, a1, a0] - Θ) / length(S)), ev(mfit), ev(ξfit))
end
@printf("total %.0f s\n", time() - t0)
