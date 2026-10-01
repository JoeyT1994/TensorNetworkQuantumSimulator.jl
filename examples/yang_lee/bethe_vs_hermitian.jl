# Showcase: the Bethe (two-sided, bilinear) estimator against the Hermitian (conjugated-bra) one for the
# NON-HERMITIAN transfer matrix of 3D Ising in an imaginary field (the Yang–Lee setting).
#
#   f_B(R) = ln κ⟨Rᵀ|T|R⟩ − ln κ⟨Rᵀ|R⟩      stationary at the dominant eigenvector (T = Tᵀ):  error O(ε²)
#   f_H(R) = ln κ⟨R̄|T|R⟩ − ln κ⟨R̄|R⟩        not stationary (T ≠ T†):                         error O(ε)
#
# (1) STATIONARITY: R = R* + η X around the converged Bethe state R* (X a random C4v direction):
#     f_B(R) − f_B(R*) ∝ η², f_H(R) − f_H(R*) ∝ η.
# (2) TRUNCATION: Bethe states at D = 2, 3, … at the same θ; both estimators on each, saved for comparison
#     against larger D (the D = 4 states come from the GPU later).
# Env: BETA (0.21), DS ("2,3"), CHIS ("16,24"), THS ("0.002,0.004,0.0055"), PERTURB (1: run part 1 at D = 2).
using TensorNetworkQuantumSimulator, Printf, Serialization, LinearAlgebra, Random
using Logging: NullLogger, with_logger
const T = TensorNetworkQuantumSimulator
redirect_stderr(stdout)
quiet(f) = with_logger(f, NullLogger())
const β = parse(Float64, get(ENV, "BETA", "0.21"))
const DS = parse.(Int, split(get(ENV, "DS", "2,3"), ","))
const CHIS = parse.(Int, split(get(ENV, "CHIS", "16,24"), ","))
const THS = parse.(Float64, split(get(ENV, "THS", "0.002,0.004,0.0055"), ","))
const PERTURB = get(ENV, "PERTURB", "1") == "1"
const CTMTOL = 1.0e-12

s0, legs, _ = ising3d_site(β)
site_at(θ) = (x = ising3d_site(β; h = im * θ / β); T.replaceinds(x[1], collect(x[2]), collect(legs)))
mag_at(θ) = (x = ising3d_site(β; h = im * θ / β); T.replaceinds(x[3], collect(x[2]), collect(legs)))
ctx(bp, θ, bilinear) = (; al = bp.Alegs, bl = bp.blegs, site = site_at(θ), legs = Tuple(legs), maxdim = bp.normenv.maxdim,
                        group = T._BP_C4V, ctm_tolerance = CTMTOL, ctm_maxiter = 4000, ctm_kwargs = (; c4v = true), bilinear)
# both estimators (f and the impurity Im m) at the state A, environments warm-started from bp's
function both(bp, A, θ)
    out = map((true, false)) do bil
        c = ctx(bp, θ, bil)
        f, _, ln, ls = T._bp_evaluate(A, c, bil ? bp.normenv : nothing, bil ? bp.openv : nothing)
        st = T.BoundaryPEPS(A, bp.Alegs, bp.blegs, c.site, Tuple(legs), ln, ls, real(f), 0.0, [real(f)], bil)
        (f = f, m = site_ratio(st, mag_at(θ)))
    end
    return out[1], out[2]
end

function main()
results = Any[]
for (D, χ) in zip(DS, CHIS)
    t0 = time()
    bp = nothing
    for dd in 2:D
        bp = quiet(() -> boundary_peps(s0, legs, dd; maxdim = dd == D ? χ : 16, init = bp, gtol = 1.0e-8, maxiter = 600))
    end
    @printf("D = %d, χ = %d: θ = 0 converged, |g| = %.1e (%.0f s)\n", D, χ, bp.gnorm, time() - t0); flush(stdout)
    cur, prev, θc, θp = bp, nothing, 0.0, 0.0
    for θ in THS
        A0 = isnothing(prev) ? cur.A : cur.A + (cur.A - prev.A) * ((θ - θc) / (θc - θp))
        r, info = quiet(() -> boundary_peps_krylov(site_at(θ), legs, cur; maxdim = χ, A0, tol = 1.0e-9, noise_tol = 1.0e-7,
                                                   ctm_tolerance = CTMTOL))
        B, H = both(r, r.A, θ)
        @printf("  θ = %.5f: |g| = %.1e  f_B = %.12f%+.2ei  f_H = %.12f%+.2ei  Im m_B = %.9f  Im m_H = %.9f (%.0f s)\n",
                θ, info.residual, real(B.f), imag(B.f), real(H.f), imag(H.f), imag(B.m), imag(H.m), time() - t0)
        flush(stdout)
        push!(results, (; β, D, χ, θ, fB = B.f, fH = H.f, mB = B.m, mH = H.m, residual = info.residual))
        serialize("state_beta$(β)_D$(D)_chi$(χ)_theta$(θ).jls", r)
        serialize("results_beta$(β).jls", results)
        cur, prev, θp, θc = r, cur, θc, θ
        if PERTURB && D == DS[1] && θ == THS[end]
            println("  stationarity test at θ = $θ, D = $D:")
            Bc = T._bp_c4v_basis(r.Alegs)
            c0 = T._bp_to_coords(r.A, Bc, r.Alegs)
            rng = MersenneTwister(1)
            X = randn(rng, ComplexF64, length(c0)); X *= norm(c0) / norm(X)
            B0, H0 = B, H
            for η in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
                A = T._bp_from_coords(c0 + η * X, Bc, r.Alegs, site_at(θ))
                Bη, Hη = both(r, A, θ)
                @printf("    η = %.0e:  |Δf_B| = %.3e   |Δf_H| = %.3e   |ΔIm m_B| = %.3e   |ΔIm m_H| = %.3e\n", η,
                        abs(Bη.f - B0.f), abs(Hη.f - H0.f), abs(imag(Bη.m - B0.m)), abs(imag(Hη.m - H0.m)))
                flush(stdout)
                push!(results, (; β, D, χ, θ, η, dfB = Bη.f - B0.f, dfH = Hη.f - H0.f))
                serialize("results_beta$(β).jls", results)
            end
        end
    end
end
println("done")
end
main()
