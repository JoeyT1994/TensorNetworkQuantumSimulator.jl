# Prototype: the Yang–Lee fold of the stationary boundary PEPS by pseudo-arclength continuation.
# Unknowns (c, θ): c the C4v coordinates of A (complex, |c| = 1), θ the imaginary field, REAL. The
# stationarity g(c, θ) = ∇f is complex-linear in (dc, dθ), so W'JQ y + W'g_θ dθ = −W'g is solved by
# y = y0 + y1 dθ for any dθ, and the real arclength equation Re⟨t_c, c − c_k⟩ + t_θ(θ − θ_k) = Δs fixes
# a real dθ: θ stays on the real axis, and the bordered system
#
#     [W'JQ        W'g_θ] [y ]   [−W'g]
#     [Re⟨t_c, Q·⟩ t_θ  ] [dθ] = [−a  ]        (dc = Q y)
#
# stays regular through the fold, where W'JQ alone is singular; its redundant directions (unused
# bond dimension, σ < rtol·σ_max) are dropped by a truncated SVD. J by central finite differences
# (columns in parallel) at every new branch point; g_θ by a central difference in θ.
using TensorNetworkQuantumSimulator, LinearAlgebra, Printf
const T = TensorNetworkQuantumSimulator

mutable struct FoldState
    c::Vector{ComplexF64}; θ::Float64; g::Vector{ComplexF64}; f::Float64; ln; ls
end

function fold_ctx(site_at, legs, al, bl, θ, χ; ctm_tolerance = 1.0e-12, ctm_maxiter = 3000)
    site = site_at(θ)
    return (; al, bl, site, legs = Tuple(legs), maxdim = χ, group = T._BP_C4V, ctm_tolerance,
            ctm_maxiter, ctm_kwargs = (; c4v = true), bilinear = true)
end
function evalg(site_at, legs, al, bl, B, c, θ, χ, ln, ls)
    ctx = fold_ctx(site_at, legs, al, bl, θ, χ)
    f, G, ln2, ls2 = T._bp_evaluate(T._bp_from_coords(c, B, al, ctx.site), ctx, ln, ls, ctx.ctm_tolerance)
    return real(f), T._bp_to_coords(G, B, al), ln2, ls2
end
function jac(site_at, legs, al, bl, B, χ, c, θ, ln, ls; fd = 1.0e-5, dθ_fd = 1.0e-6)
    ctx = fold_ctx(site_at, legs, al, bl, θ, χ)
    J = T._bp_fd_jacobian(c, B, ctx, ln, ls, fd * norm(c), ctx.ctm_tolerance, ctx.site)
    Gθ = (evalg(site_at, legs, al, bl, B, c, θ + dθ_fd, χ, ln, ls)[2] -
          evalg(site_at, legs, al, bl, B, c, θ - dθ_fd, χ, ln, ls)[2]) / (2dθ_fd)
    return J, Gθ
end
function reduced(al, B, c, J; rtol = 1.0e-6)
    Q, W = T._bp_gauge_bases(c, B, T.dim(al[1]), T.dim(al[5]))
    F = svd(W' * J * Q)
    keep = F.S .> rtol * F.S[1]
    Pinv(r) = F.V[:, keep] * ((F.U[:, keep]' * r) ./ F.S[keep])
    return Q, W, Pinv, minimum(F.S[keep]) / F.S[1]
end

# One pseudo-arclength step of length Δs from the converged state s0 along (tc, tθ): predictor, then
# the bordered Newton corrector, J (and g_θ) computed at the predicted point and Broyden-updated
# ([J g_θ] (Δc, Δθ) = Δg) after every step. Accepted at |g| < tol, or at the gradient's noise floor:
# |g| < noise stagnating (a step gains < 30 %).
function arcstep(site_at, legs, al, bl, B, χ, s0::FoldState, tc, tθ, Δs; tol = 1.0e-9, noise = 1.0e-7, maxit = 12,
                 verbose = true)
    c = s0.c + Δs * tc; c /= norm(c); θ = s0.θ + Δs * tθ
    f, g, ln, ls = evalg(site_at, legs, al, bl, B, c, θ, χ, s0.ln, s0.ls)
    J, Gθ = jac(site_at, legs, al, bl, B, χ, c, θ, ln, ls)
    resprev = Inf
    for it in 1:maxit
        a = real(dot(tc, c - s0.c)) + tθ * (θ - s0.θ) - Δs
        res = norm(g) * norm(c)
        verbose && @printf("    corrector %d: |g| = %.2e, |a| = %.2e, θ = %.12f\n", it - 1, res, abs(a), θ)
        done = abs(a) < 1.0e-10 && (res < tol || (res < noise && res > 0.7 * resprev))
        done && return FoldState(c, θ, g, f, ln, ls), J, Gθ
        resprev = res
        Q, W, Pinv, _ = reduced(al, B, c, J)
        y0 = -Pinv(W' * g); y1 = -Pinv(W' * Gθ)
        dθ = (-a - real(dot(tc, Q * y0))) / (real(dot(tc, Q * y1)) + tθ)
        cn = c + Q * (y0 + y1 * dθ); cn /= norm(cn); θn = θ + dθ
        fn, gn, lnn, lsn = evalg(site_at, legs, al, bl, B, cn, θn, χ, ln, ls)
        Δc, Δθ, Δg = cn - c, θn - θ, gn - g                # good Broyden on the extended Jacobian
        r = Δg - J * Δc - Gθ * Δθ
        den = real(dot(Δc, Δc)) + Δθ^2
        J = J + (r * Δc') / den; Gθ = Gθ + r * (Δθ / den)
        c, θ, f, g, ln, ls = cn, θn, fn, gn, lnn, lsn
    end
    return nothing, J, Gθ
end

# The tangent at a converged point: the bordered system with right-hand side (0, 1), its last row the
# previous tangent; normalised |t_c|² + t_θ² = 1 and oriented along the previous tangent. Also the
# smallest kept singular value of the reduced Jacobian (relative), the fold's direction near the fold.
function tangent(al, B, s::FoldState, J, Gθ, tc0, tθ0)
    Q, W, Pinv, smin = reduced(al, B, s.c, J)
    y1 = -Pinv(W' * Gθ)
    dθ = 1 / (real(dot(tc0, Q * y1)) + tθ0)
    tc, tθ = Q * y1 * dθ, dθ
    nrm = sqrt(norm(tc)^2 + tθ^2)
    tc, tθ = tc / nrm, tθ / nrm
    if real(dot(tc, tc0)) + tθ * tθ0 < 0
        tc, tθ = -tc, -tθ
    end
    return tc, tθ, smin
end
