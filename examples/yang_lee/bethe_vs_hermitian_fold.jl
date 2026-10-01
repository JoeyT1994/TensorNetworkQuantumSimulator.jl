# The Bethe-vs-Hermitian showcase AT THE YANG–LEE FOLD (examples/yang_lee/bethe_vs_hermitian.jl there,
# away from it). Pseudo-arclength through the fold of the stationary bilinear boundary PEPS
# (fold_pseudoarclength.jl, as fold_run.jl), keeping every state; then at the turning point (the path's
# highest θ):
#   * both estimators along the path: f_B = ln κ⟨Rᵀ|T|R⟩ − ln κ⟨Rᵀ|R⟩, f_H = ln κ⟨R̄|T|R⟩ − ln κ⟨R̄|R⟩, Im m;
#   * R = R* + ηX for X random (C4v) and X = the null direction of the reduced Bethe Jacobian (the fold's
#     soft mode). Expected: Δf_B ∝ η² along X random, ∝ η³ along the null direction (the Hessian's zero
#     eigenvalue, exact at the fold); Δf_H ∝ η along both.
# Env: BETA (0.2), D (2), CHI (16), THS (continuation points before the arclength), TLIMIT (s).
include(joinpath(@__DIR__, "fold_pseudoarclength.jl"))
using Serialization, Random
using Logging: NullLogger, with_logger
BLAS.set_num_threads(1)
const β = parse(Float64, get(ENV, "BETA", "0.2"))
const D = parse(Int, get(ENV, "D", "2"))
const χ = parse(Int, get(ENV, "CHI", "16"))
const THS = parse.(Float64, split(get(ENV, "THS", "0.006,0.011,0.014,0.0155,0.0162"), ","))
const TLIMIT = parse(Float64, get(ENV, "TLIMIT", "5000"))
quiet(f) = with_logger(f, NullLogger())

function main()
    t0 = time()
    s0, legs, _ = ising3d_site(β)
    bp = quiet(() -> boundary_peps(s0, legs, 2; maxdim = 16, gtol = 1.0e-7, maxiter = 300))
    D > 2 && (bp = quiet(() -> boundary_peps(s0, legs, D; maxdim = χ, init = bp, gtol = 1.0e-6, maxiter = 200)))
    site_at(θ) = (x = ising3d_site(β; h = im * θ / β); T.replaceinds(x[1], collect(x[2]), collect(legs)))
    mag_at(θ) = (x = ising3d_site(β; h = im * θ / β); T.replaceinds(x[3], collect(x[2]), collect(legs)))
    cur, prev, θp, θc = bp, nothing, 0.0, 0.0
    for θ in THS
        A0 = isnothing(prev) ? cur.A : cur.A + (cur.A - prev.A) * ((θ - θc) / (θc - θp))
        r, i = quiet(() -> boundary_peps_krylov(site_at(θ), legs, cur; maxdim = χ, A0, tol = 1.0e-9, noise_tol = 1.0e-7))
        @printf("θ = %.4f: Newton–Krylov |g| = %.1e, %d evals\n", θ, i.residual, i.evals); flush(stdout)
        prev, cur, θp, θc = cur, r, θc, θ
    end
    al, bl = cur.Alegs, cur.blegs
    B = T._bp_c4v_basis(al)
    coords(b) = (c = T._bp_to_coords(b.A, B, al); c / norm(c))
    tostate(b, θ) = (c = coords(b); (f, g, ln, ls) = evalg(site_at, legs, al, bl, B, c, θ, χ, b.normenv, b.openv);
                     FoldState(c, θ, g, f, ln, ls))
    bpof(st) = T.BoundaryPEPS(T._bp_from_coords(st.c, B, al, site_at(st.θ)), al, bl, site_at(st.θ), Tuple(legs), st.ln,
                              st.ls, st.f, norm(st.g), [st.f], true)
    # both estimators at the coordinates c (environments warm from the state st)
    function both(st, c)
        A = T._bp_from_coords(c, B, al, site_at(st.θ))
        map((true, false)) do bil
            ctx = (; al, bl, site = site_at(st.θ), legs = Tuple(legs), maxdim = χ, group = T._BP_C4V, ctm_tolerance = 1.0e-12,
                   ctm_maxiter = 4000, ctm_kwargs = (; c4v = true), bilinear = bil)
            f, _, ln, ls = T._bp_evaluate(A, ctx, bil ? st.ln : nothing, bil ? st.ls : nothing)
            b = T.BoundaryPEPS(A, al, bl, ctx.site, Tuple(legs), ln, ls, real(f), 0.0, [real(f)], bil)
            (f = f, m = site_ratio(b, mag_at(st.θ)))
        end
    end
    sa, sb = tostate(prev, θp), tostate(cur, θc)
    tc = sb.c - sa.c; tθ = sb.θ - sa.θ; nrm = sqrt(norm(tc)^2 + tθ^2); tc, tθ = tc / nrm, tθ / nrm
    Δs = 1.0e-4 / tθ
    path = [sb]; s = sb; after = 0
    for k in 1:80
        time() - t0 > TLIMIT && (println("(time)"); break)
        new, _, _ = arcstep(site_at, legs, al, bl, B, χ, s, tc, tθ, Δs; noise = 1.0e-7, verbose = false)
        if isnothing(new)
            Δs /= 2; continue
        end
        stc = new.c - s.c; stθ = new.θ - s.θ; nrm = sqrt(norm(stc)^2 + stθ^2); stc, stθ = stc / nrm, stθ / nrm
        ang = acos(clamp(real(dot(stc, tc)) + stθ * tθ, -1.0, 1.0))
        m = site_ratio(bpof(new), mag_at(new.θ))
        if (k > 1 && ang > 0.3) || abs(real(m)) > 1.0e-4
            Δs /= 2; continue
        end
        push!(path, new)
        @printf("path: θ = %.11f  Im m = %.8f  (%.0f s)\n", new.θ, imag(m), time() - t0); flush(stdout)
        s, tc, tθ = new, stc, stθ
        # past the fold: shorter steps there, so that the highest point lies close to the turn
        new.θ < maximum(x.θ for x in path) && (after += 1) >= 2 && break
        Δs = min(ang < 0.1 ? 1.5Δs : ang > 0.2 ? 0.7Δs : Δs, 0.02)
    end
    θs = [x.θ for x in path]; imax = argmax(θs)
    imax == length(path) && println("WARNING: the fold was not passed; using the highest point")
    st = path[imax]
    @printf("\nfold point: θ* = %.11f (the path's highest θ)\n", st.θ)
    println("\nboth estimators along the path:")
    for x in path
        b, h = both(x, x.c)
        @printf("  θ = %.11f  f_B = %.12f  f_H = %.12f  Δ = %+.3e  Im m_B = %.7f  Im m_H = %.7f\n", x.θ, real(b.f),
                real(h.f), real(b.f - h.f), imag(b.m), imag(h.m))
    end
    flush(stdout)
    # the fold's soft mode: the smallest kept singular direction of the reduced Bethe Jacobian
    J, _ = jac(site_at, legs, al, bl, B, χ, st.c, st.θ, st.ln, st.ls)
    Q, W = T._bp_gauge_bases(st.c, B, T.dim(al[1]), T.dim(al[5]))
    F = svd(W' * J * Q)
    keep = findall(F.S .> 1.0e-6 * F.S[1])
    @printf("\nreduced Jacobian at θ*: σ_max = %.3e, smallest kept σ = %.3e (ratio %.1e), next %.3e\n", F.S[1],
            F.S[keep[end]], F.S[keep[end]] / F.S[1], F.S[keep[end - 1]])
    vnull = Q * F.V[:, keep[end]]
    rng = MersenneTwister(1)
    vrand = Q * (randn(rng, ComplexF64, size(Q, 2)))          # random, in the same gauge-free subspace
    b0, h0 = both(st, st.c)
    for (name, v) in (("random", vrand), ("null", vnull))
        v = v / norm(v)
        println("\nη-scan along the $name direction (|X| = |c| = 1):")
        for η in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
            b, h = both(st, st.c + η * v)
            @printf("  η = %.0e:  |Δf_B| = %.3e   |Δf_H| = %.3e\n", η, abs(b.f - b0.f), abs(h.f - h0.f))
            flush(stdout)
        end
    end
    serialize("fold_showcase_D$(D)_beta$(β).jls", (; path = [(x.θ, x.f) for x in path], θstar = st.θ, σ = F.S))
    @printf("total %.0f s\n", time() - t0)
end
main()
