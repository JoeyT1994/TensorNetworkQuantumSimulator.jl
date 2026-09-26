# Boundary PEPS for 3D classical models in the thermodynamic limit (Vanderstraeten, Vanhecke &
# Verstraete, PRE 98, 042145 (2018); Nishino and coworkers' tensor product variational approach
# before them). One layer of the cubic network is a 2D tensor-network operator T: the site tensor
# with its z legs as physical in (z⁻) and out (z⁺) legs. Then Z = Tr T^{L_z}, and the free energy
# per site is set by T's dominant eigenvalue, approximated through a 1×1 infinite PEPS |Ψ(A)⟩ of
# bond dimension D:
#
#   f(A) = ln κ(⟨Ψ|T|Ψ⟩) − ln κ(⟨Ψ|Ψ⟩)                      (per site)
#
# both terms the free energy per site of an ordinary 2D network — three layers and two —
# contracted by `InfiniteCTM2D` in the Kikuchi form. f is the nested Bethe estimate: stationary at
# the boundary state, so its errors are second order. For a symmetric T (a symmetric bond split, as
# `ising3d_site`'s) f is variational: f ≤ ln κ₃D, approaching it as D grows.
#
# THE GRADIENT IS LOCAL. T is a product of site tensors, not a sum of local terms, so the derivative
# of either per-site free energy with respect to A is its normalised one-site environment
# (`site_environment`): no sum over positions, no channel environments. With A in the ket and the
# bra layer,
#
#   ∂f/∂A = (E_ket + E_bra)/z |_{⟨Ψ|T|Ψ⟩} − (E_ket + E_bra)/z |_{⟨Ψ|Ψ⟩}   (real A)
#
# f is invariant under A → cA, so the gradient is orthogonal to A. With `symmetrize = true` (a
# C4v-invariant site) A and every gradient are projected onto the C4v-symmetric subspace of the
# virtual legs. Three solvers for ∇f = 0 (measured against each other in docs/boundary_peps.md):
#
# * `boundary_peps` — L-BFGS on the unit sphere with the norm-metric preconditioner: cheap per
#   iteration and robust from a cold start; real data only.
# * `boundary_peps_krylov` — Newton–Krylov with a trust region in the Krylov subspace, real or complex
#   (bilinear) data: the fast route to a CONVERGED state, where observables near a critical point
#   are decided by soft modes that f and a loose gradient do not see.
# * `boundary_peps_stationary` — Newton with the full finite-difference Jacobian: small D and
#   continuations that reuse the Jacobian (the Yang–Lee scans, docs/yang_lee.md).

"""
    BoundaryPEPS

The result of [`boundary_peps`](@ref), [`boundary_peps_krylov`](@ref) or
[`boundary_peps_stationary`](@ref): the boundary tensor `A` on legs `Alegs = (x⁻, x⁺, y⁻, y⁺, p)`
(`p` the z bond), the 3D `site` and its `legs`, the converged 2D environments of ⟨Ψ|Ψ⟩ (`normenv`)
and ⟨Ψ|T|Ψ⟩ (`openv`), `lnkappa = f(A)`, the final gradient norm `gnorm`, the `history` of f, and
whether the bra is `A` itself (`bilinear`, the stationary solvers) or `conj(A)`. Any of the solvers
takes it as `init`; `adapt(CuArray, bp)` moves it (tensor, site and environments) to the GPU.
"""
struct BoundaryPEPS
    A::Any
    Alegs::NTuple{5, Any}
    blegs::NTuple{4, Any}          # the bra's virtual legs
    site::Any
    legs::NTuple{6, Any}
    normenv::Any
    openv::Any
    lnkappa::Float64
    gnorm::Float64
    history::Vector{Float64}
    bilinear::Bool                 # bra = A (a complex symmetric T: ⟨L| = |R⟩ᵀ), not conj(A)
end

Adapt.adapt_structure(to, bp::BoundaryPEPS) =
    BoundaryPEPS(adapt(to, bp.A), bp.Alegs, bp.blegs, adapt(to, bp.site), bp.legs, adapt(to, bp.normenv),
                 adapt(to, bp.openv), bp.lnkappa, bp.gnorm, bp.history, bp.bilinear)

# The 8 symmetries of the square acting on the legs (x⁻, x⁺, y⁻, y⁺): swap within either pair,
# and swap the pairs.
const _BP_C4V = ((1, 2, 3, 4), (2, 1, 3, 4), (1, 2, 4, 3), (2, 1, 4, 3),
                 (3, 4, 1, 2), (4, 3, 1, 2), (3, 4, 2, 1), (4, 3, 2, 1))

_bp_symmetrize(t, vl) = sum(replaceinds(t, collect(vl), [vl[π[i]] for i in 1:4]) for π in _BP_C4V) / length(_BP_C4V)

# Is the 3D site invariant under the square's symmetries of its (x±, y±) legs?
function _bp_isc4v(site, legs; atol = 1.0e-12)
    xy = collect(legs[1:4])
    ref = norm(site)
    return all(norm(replaceinds(site, xy, [xy[π[i]] for i in 1:4]) - site) <= atol * ref for π in _BP_C4V)
end

# ⟨Ψ|Ψ⟩ and ⟨Ψ|T|Ψ⟩ as layer lists with their (x⁻, x⁺, y⁻, y⁺) leg lists.
function _bp_norm_layers(A, al, bl; bilinear::Bool = false)
    ket = A
    bra = replaceinds(bilinear ? A : conj(A), collect(al[1:4]), collect(bl))
    return Any[ket, bra], ntuple(d -> Index[al[d], bl[d]], 4)
end

function _bp_sandwich_layers(A, al, bl, site, legs; bilinear::Bool = false)
    ket = replaceind(A, al[5], legs[5])                                         # A's p = T's z⁻
    bra = replaceinds(bilinear ? A : conj(A), vcat(collect(al[1:4]), [al[5]]), vcat(collect(bl), [legs[6]]))   # … z⁺
    return Any[ket, site, bra], ntuple(d -> Index[al[d], legs[d], bl[d]], 4)
end

# f(A), its gradient on A's legs, and the two converged environments (warm-started from `init`),
# the environments converged to `tol` (default: the context's fixed tolerance).
function _bp_evaluate(A, ctx, init_n, init_s, tol = ctx.ctm_tolerance)
    al, bl, site, legs = ctx.al, ctx.bl, ctx.site, ctx.legs
    bil = get(ctx, :bilinear, false)
    nl, nlegs = _bp_norm_layers(A, al, bl; bilinear = bil)
    sl, slegs = _bp_sandwich_layers(A, al, bl, site, legs; bilinear = bil)
    # The two environments are independent: converge them concurrently (each step's pairs and
    # blocks occupy at most 8 tasks, so on more threads the norm network rides along for free).
    # ANDERSON mixing (`ctx.anderson` iterates) only from a warm start: from the vacuum it can pick a
    # wrong fixed point while the kept rank grows (docs/boundary_peps.md). Warm, it cuts the CTM steps
    # per evaluation 1.7–1.8× (3D Ising β = 0.2275, D = 3, χ = 16, m = 5: 26.5 → 14.5 for a
    # finite-difference product, 35.8 → 21.2 after a 2e-2 step); the fixed point is unchanged.
    am = get(ctx, :anderson, 0)
    run_n() = update(InfiniteCTM2D(nl, nlegs, ctx.maxdim; init = init_n, ctx.ctm_kwargs...);
                     tolerance = tol, maxiter = ctx.ctm_maxiter, anderson = isnothing(init_n) ? 0 : am)
    run_s() = update(InfiniteCTM2D(sl, slegs, ctx.maxdim; init = init_s, ctx.ctm_kwargs...);
                     tolerance = tol, maxiter = ctx.ctm_maxiter, anderson = isnothing(init_s) ? 0 : am)
    if Threads.nthreads() > 1
        tn = Threads.@spawn run_n()
        ls = run_s()
        ln = fetch(tn)
    else
        ln = run_n(); ls = run_s()
    end
    f = cvm_freenergy(ls) - cvm_freenergy(ln)
    Eks, zs = site_environment(ls, 1)
    Ebs, _ = site_environment(ls, 3)
    Ekn, zn = site_environment(ln, 1)
    Ebn, _ = site_environment(ln, 2)
    G = (replaceind(Eks, legs[5], al[5]) + replaceinds(Ebs, vcat(collect(bl), [legs[6]]), collect(al))) / zs -
        (Ekn + replaceinds(Ebn, collect(bl), collect(al[1:4]))) / zn
    ctx.symmetrize && (G = _bp_symmetrize(G, al[1:4]))
    return f, G, ln, ls
end

# THE NORM-METRIC PRECONDITIONER. The one-site environment of ⟨Ψ|Ψ⟩ with both of the site's layers
# removed is the metric N (ket legs × bra legs) of the PEPS manifold at A — the Gram matrix of the
# tangent vectors ∂|Ψ⟩/∂A. Steps are taken in (N + τ λ_max)^{-1} g, a natural-gradient step (the
# identity on the physical leg), and L-BFGS uses it as its initial inverse Hessian. Measured
# (docs/boundary_peps.md): near β_c each iteration gains ~2× more f than the plain gradient step.
# The inverse is formed once per accepted A (n = D⁴, a dense n×n solve on the host).
function _bp_metric(ln, al, bl, τ)
    VX, _, env = _i2_shell(ln)
    old, new = _i2_site_relabelling(ln.legs, _I2_V, VX)
    N = _i2_relabel(env, new, old)                              # on al[1:4] and bl
    ka = collect(al[1:4]); kb = collect(bl)
    M = Array(array(N, ka..., kb...))
    n = prod(size(M)[1:4])
    M = reshape(M, n, n)
    M = (M + M') / 2
    λ = eigmax(Hermitian(M))
    (isfinite(λ) && λ > 0) || return nothing
    return inv(Hermitian(M + τ * λ * I)), ka
end

function _bp_precondition(Minv, ka, p, g)
    ga = Array(array(g, ka..., p))
    sz = size(ga)
    X = Minv * reshape(ga, size(Minv, 1), sz[end])
    return adapt_like(g, from_array(reshape(X, sz), ka..., p))
end

# The initial boundary tensor: T applied to the product state `b` on the z⁻ legs, its virtual legs
# embedded into (or cut down to) bond dimension D, plus a little noise so that padded directions
# have a gradient (they would otherwise stay exactly zero).
function _bp_initial(site, legs, al, D, b, noise, rng)
    elt = scalartype(site)
    bv = isnothing(b) ? ones(elt, dim(legs[5])) : convert(Vector{elt}, collect(b))
    t = site * adapt_like(site, from_array(bv, legs[5]))
    for d in 1:4
        dz = dim(legs[d])
        M = zeros(elt, dz, D)
        for j in 1:min(dz, D)
            M[j, j] = 1
        end
        t = t * adapt_like(site, from_array(M, legs[d], al[d]))
    end
    t = replaceind(t, legs[6], al[5])
    t = t / norm(t)
    if noise > 0
        t = t + noise * adapt_like(site, random_tensor(rng, elt, collect(al))) / sqrt(prod(dim.(collect(al))))
        t = t / norm(t)
    end
    return t
end

# A previous boundary tensor on legs `old` carried onto legs `new`: equal bond dimension is a
# relabelling; a larger one zero-pads every virtual leg and adds relative `noise` (the padded
# directions would otherwise have no gradient); a smaller one keeps the leading components.
function _bp_embed(t, old, new, noise, rng)
    Din, D = dim(old[1]), dim(new[1])
    Din == D && return (s = replaceinds(t, collect(old), collect(new)); s / norm(s))
    elt = scalartype(t)
    M = zeros(elt, Din, D)
    for j in 1:min(Din, D)
        M[j, j] = 1
    end
    for d in 1:4
        t = t * adapt_like(t, from_array(M, old[d], new[d]))
    end
    t = replaceind(t, old[5], new[5])
    t = t / norm(t)
    if D > Din && noise > 0
        t = t + noise * adapt_like(t, random_tensor(rng, elt, collect(new))) / sqrt(prod(dim.(collect(new))))
        t = t / norm(t)
    end
    return t
end

"""
    boundary_peps(site, legs, D; maxdim, init = nothing, boundary = nothing, symmetrize = true,
                  maxiter = 200, gtol = 1e-7, memory = 10, max_step = 0.2, ctm_tolerance = 1e-10,
                  ctm_maxiter = 1000, noise = 1e-2, seed = 0, precondition = true,
                  precondition_shift = 1e-2, adaptive_tolerance = true, ctm_anderson = 0, callback = nothing,
                  time_limit = Inf, verbose = false, kwargs...) -> BoundaryPEPS

Variational boundary PEPS for the translation-invariant cubic network of `site` (legs
`(x⁻, x⁺, y⁻, y⁺, z⁻, z⁺)`, as for [`InfiniteCTM3D`](@ref)): maximise the per-site Rayleigh quotient
`f(A) = ln κ(⟨Ψ|T|Ψ⟩) − ln κ(⟨Ψ|Ψ⟩)` of the layer transfer operator T over a 1×1 PEPS of bond
dimension `D`, both terms contracted by [`InfiniteCTM2D`](@ref) at `maxdim`. For a symmetric T
(e.g. [`ising3d_site`](@ref)'s) `f` is a variational lower bound on ln κ₃D.

* `init` — a `BoundaryPEPS` to start from (a nearby coupling, whose environments warm-start too
  when χ and D match; or a smaller D, zero-padded with relative `noise`), or a tensor on legs of
  the right dimensions; otherwise T applied to the product state `boundary` on z⁻ (ones by default;
  a fixed-spin vector selects a symmetry-broken phase), embedded at bond dimension `D` with
  relative `noise`.
* `symmetrize` — keep A and the gradient C4v-symmetric in the virtual legs; requires an invariant
  site.
* Stops when the tangent gradient norm (for normalised A) is below `gtol`, or after `maxiter`
  L-BFGS iterations, or after `time_limit` seconds, or when the line search fails.
* `callback(it, state)` — `state` a NamedTuple (A, Alegs, f, res) — runs after every accepted step.
* `ctm_anderson = m` — Anderson mixing of the last m iterates in every warm-started 2D CTMRG
  (1.7–1.8× fewer steps per evaluation at D = 3; the fixed point is unchanged); 0 is off.
* `precondition` — L-BFGS with the norm metric `(N + precondition_shift·λ_max)⁻¹` as its initial
  inverse Hessian (a natural-gradient step); `false` gives the plain `γ I`.
* `adaptive_tolerance` — converge the 2D environments to `1e-2·|g|`, clamped to
  `[ctm_tolerance, 1e-6]`, rather than always to `ctm_tolerance`.

Remaining keywords go to [`InfiniteCTM2D`](@ref) (e.g. `pair`). With `symmetrize = true` the 2D
environments use `c4v = true` (one pair and two blocks per step, the rest by symmetry; exact, 3–4×
less work) unless `c4v = false` is passed.
"""
function boundary_peps(site, legs, D::Integer; maxdim::Integer, init = nothing, boundary = nothing,
                       symmetrize::Bool = true, maxiter::Integer = 200, gtol::Real = 1.0e-7,
                       memory::Integer = 10, max_step::Real = 0.2, ctm_tolerance::Real = 1.0e-10,
                       ctm_maxiter::Integer = 1000, noise::Real = 1.0e-2, seed::Integer = 0,
                       ls_max::Integer = 12, precondition::Bool = true, precondition_shift::Real = 1.0e-2,
                       adaptive_tolerance::Bool = true, ctm_anderson::Integer = 0, callback = nothing,
                       time_limit::Real = Inf,
                       verbose::Bool = false, kwargs...)
    D >= 1 || throw(ArgumentError("D must be ≥ 1, got $D"))
    length(legs) == 6 || throw(ArgumentError("legs must be the six legs (x⁻, x⁺, y⁻, y⁺, z⁻, z⁺)"))
    issetequal(collect(inds(site)), collect(legs)) || throw(ArgumentError("the site tensor's indices must be exactly the six given legs"))
    dim(legs[5]) == dim(legs[6]) || throw(ArgumentError("the z legs must have equal dimensions"))
    symmetrize && !_bp_isc4v(site, legs) && throw(ArgumentError(
        "symmetrize = true needs a site invariant under the square's symmetries of its x, y legs"))
    al = (new_index(D; tags = "bp,xm"), new_index(D; tags = "bp,xp"), new_index(D; tags = "bp,ym"),
          new_index(D; tags = "bp,yp"), new_index(dim(legs[5]); tags = "bp,p"))
    bl = Tuple(new_index(D; tags = "bp,bra") for _ in 1:4)
    ctx = (; al, bl, site, legs = Tuple(legs), maxdim = Int(maxdim), symmetrize, ctm_tolerance,
           ctm_maxiter = Int(ctm_maxiter), ctm_kwargs = merge((; c4v = symmetrize), kwargs),
           anderson = Int(ctm_anderson))
    rng = Xoshiro(seed)
    A = if init isa BoundaryPEPS
        dim(init.Alegs[5]) == dim(al[5]) || throw(ArgumentError("init's physical dimension differs from the site's z legs"))
        _bp_embed(init.A, init.Alegs, al, noise, rng)
    elseif !isnothing(init)
        length(inds(init)) == 5 || throw(ArgumentError("init must be a BoundaryPEPS or a 5-leg tensor"))
        t = replaceinds(init, collect(inds(init)), collect(al))
        t / norm(t)
    else
        _bp_initial(site, legs, al, D, boundary, D > dim(legs[1]) ? noise : 0.0, rng)
    end
    symmetrize && (A = _bp_symmetrize(A, al[1:4]); A = A / norm(A))
    # the environments warm-start too when they fit (same χ and bond dimension)
    reuse = init isa BoundaryPEPS && init.normenv isa InfiniteCTM2D && init.normenv.maxdim == maxdim &&
            dim(init.Alegs[1]) == D
    init_n = reuse ? init.normenv : nothing
    init_s = reuse ? init.openv : nothing

    # minimise F = −f on the unit sphere
    tangent(g, x) = g - (real(dot(x, g)) / real(dot(x, x))) * x
    # ADAPTIVE CTMRG TOLERANCE: the environments are converged only as far as the current gradient
    # needs, 1e-2·|g| clamped to [ctm_tolerance, 1e-6]; ctm_tolerance is reached as |g| → gtol.
    # Measured near β_c (D = 3, χ = 24): half the CTM steps per iteration at no loss per iteration.
    ctmtol(gn) = adaptive_tolerance ? clamp(1.0e-2 * gn, ctm_tolerance, max(ctm_tolerance, 1.0e-6)) : ctm_tolerance
    tstart = time()
    f, G, ln, ls = _bp_evaluate(A, ctx, init_n, init_s, adaptive_tolerance ? max(ctm_tolerance, 1.0e-8) : ctm_tolerance)
    F, g = -f, tangent(-G, A)
    history = [f]
    verbose && (println("boundary_peps D=$D χ=$maxdim: start f = $f, |g| = $(norm(g))"); flush(stdout))
    Ss = Any[]; Ys = Any[]; ρs = Float64[]
    # H₀ = the norm-metric inverse at the current A (identity when `precondition = false`)
    metric(ln) = precondition ? _bp_metric(ln, al, bl, precondition_shift) : nothing
    Mk = metric(ln)
    Pinv(v) = isnothing(Mk) ? v : (w = _bp_precondition(Mk[1], Mk[2], al[5], v);
                                   symmetrize && (w = _bp_symmetrize(w, al[1:4])); tangent(w, A))
    gnorm = norm(g)
    for it in 1:maxiter
        gnorm = norm(g)
        gnorm < gtol && (verbose && println("boundary_peps: |g| = $gnorm below gtol"); break)
        time() - tstart > time_limit && (verbose && println("boundary_peps: time limit"); break)
        # two-loop recursion, H₀ = γ P⁻¹
        q = copy(g); αs = zeros(length(Ss))
        for i in length(Ss):-1:1
            αs[i] = ρs[i] * real(dot(Ss[i], q)); q = q - αs[i] * Ys[i]
        end
        r = Pinv(q)
        if !isempty(Ss)
            r = (real(dot(Ss[end], Ys[end])) / real(dot(Ys[end], Pinv(Ys[end])))) * r
        end
        for i in 1:length(Ss)
            β = ρs[i] * real(dot(Ys[i], r)); r = r + (αs[i] - β) * Ss[i]
        end
        d = tangent(-r, A)
        slope = real(dot(d, g))
        if !(slope < 0)
            empty!(Ss); empty!(Ys); empty!(ρs)
            d = tangent(-Pinv(g), A); slope = real(dot(d, g))
            if !(slope < 0)
                d = -g; slope = real(dot(d, g))
            end
        end
        α = isempty(Ss) ? min(1.0, max_step / norm(d)) : 1.0
        α * norm(d) > max_step && (α = max_step / norm(d))
        accepted = false
        local An, fn, Gn, lnn, lsn
        for k in 1:ls_max
            An = A + α * d
            symmetrize && (An = _bp_symmetrize(An, al[1:4]))
            An = An / norm(An)
            fn, Gn, lnn, lsn = _bp_evaluate(An, ctx, ln, ls, ctmtol(gnorm))
            if -fn <= F + 1.0e-4 * α * slope
                accepted = true
                break
            end
            verbose && println("    trial $k rejected: α = $α, Δf = $(fn - f)")
            α /= 2
        end
        if !accepted
            verbose && println("boundary_peps it $it: line search failed, stopping at f = $f")
            break
        end
        fn >= f - 1.0e-14 * abs(f) || error("boundary_peps: accepted a downhill step in f at iteration $it ($f → $fn)")
        gn = tangent(-Gn, An)
        s = An - A; y = gn - g; sy = real(dot(s, y))
        if sy > 0
            push!(Ss, s); push!(Ys, y); push!(ρs, 1 / sy)
            length(Ss) > memory && (popfirst!(Ss); popfirst!(Ys); popfirst!(ρs))
        end
        A, f, F, g, ln, ls = An, fn, -fn, gn, lnn, lsn
        Mk = metric(ln)
        push!(history, f)
        verbose && (println("boundary_peps it $it: f = $f, |g| = $(norm(g)), α = $α"); flush(stdout))
        isnothing(callback) || callback(it, (; A, Alegs = al, f, res = norm(g)))
    end
    return BoundaryPEPS(A, al, bl, site, Tuple(legs), ln, ls, f, norm(g), history, false)
end

"""
    cvm_freenergy(bp::BoundaryPEPS)

The boundary PEPS's estimate of ln κ, the log partition function per site of the 3D network:
`f(A) = ln κ(⟨Ψ|T|Ψ⟩) − ln κ(⟨Ψ|Ψ⟩)` — a lower bound for a symmetric transfer operator.
"""
cvm_freenergy(bp::BoundaryPEPS) = bp.lnkappa

"""
    site_ratio(bp::BoundaryPEPS, impurity)

`⟨Ψ|T_imp|Ψ⟩ / ⟨Ψ|T|Ψ⟩` for a 3D impurity site (same six legs as the site, e.g. the
magnetisation tensor of [`ising3d_site`](@ref)): the single-site expectation value in the bulk.
"""
function site_ratio(bp::BoundaryPEPS, impurity)
    sl, _ = _bp_sandwich_layers(bp.A, bp.Alegs, bp.blegs, bp.site, bp.legs; bilinear = bp.bilinear)
    return site_ratio(bp.openv, Any[sl[1], adapt_like(bp.site, impurity), sl[3]])
end

# --- the STATIONARY boundary PEPS: complex symmetric transfer operators ---------------------------
#
# For a site with complex weights whose layer operator is complex SYMMETRIC (Tᵀ = T, e.g. the Ising
# model in an imaginary field, whose bond splits are symmetric), the left eigenvector is the
# transpose of the right one, ⟨L| = |R⟩ᵀ, and the estimator is BILINEAR:
#
#   f(R) = ln κ(Rᵀ T R) − ln κ(Rᵀ R)          (the bra layer is R itself, not R̄)
#
# f is holomorphic in R and stationary — not maximal — at the dominant eigenvector, with first-order
# errors cancelling. There is no maximum principle, so it is solved as ∇f = 0 by Newton's method in
# the C4v-symmetric coordinates `c` (R = B c, B an orthonormal orbit basis; n = 12, 42, 110 at
# D = 2, 3, 4), with the Jacobian J = ∂g/∂c by central differences of the gradient (each a pair of
# warm-started 2D environments), refreshed by Broyden updates between recomputations.
#
# THE GAUGE. f(λR) = f(R), so g(λc) = g(c)/λ: J c = −g and cᵀ g = 0 — c is a right and (bilinearly)
# a left null vector of J at a stationary point. Newton works in the reduced Jacobian
# J_r = W† J Q, Q spanning c^⊥ and W spanning {y : cᵀy = 0}. Its smallest singular value vanishes
# where two stationary points merge: the FOLD, the finite-D image of the Yang–Lee edge (there the
# two leading eigenvectors of T coalesce, an exceptional point).

# An orthonormal real basis of the tensors on `al` invariant under the square's symmetries of the
# virtual legs: one normalised orbit sum per (orbit of the virtual multi-index, physical index).
function _bp_c4v_basis(al)
    D = dim(al[1]); dp = dim(al[5])
    seen = Set{NTuple{4, Int}}()
    orbits = Vector{Vector{NTuple{4, Int}}}()
    for I in CartesianIndices((D, D, D, D))
        i = Tuple(I)
        i in seen && continue
        orb = unique([(i[π[1]], i[π[2]], i[π[3]], i[π[4]]) for π in _BP_C4V])
        push!(orbits, orb); union!(seen, orb)
    end
    LI = LinearIndices((D, D, D, D, dp))
    B = zeros(D^4 * dp, length(orbits) * dp)
    col = 0
    for p in 1:dp, orb in orbits
        col += 1
        for o in orb
            B[LI[o..., p], col] = 1 / sqrt(length(orb))
        end
    end
    return B
end

_bp_from_coords(c, B, al, ref) = adapt_like(ref, from_array(reshape(B * c, Tuple(dim.(collect(al)))), al...))
_bp_to_coords(G, B, al) = transpose(B) * vec(ComplexF64.(Array(array(G, al...))))

# `M` applied to dimension `k` of the array `A`.
function _bp_modemul(A::AbstractArray, M::AbstractMatrix, k::Int)
    p = [k; setdiff(1:ndims(A), k)]
    Ap = permutedims(A, p)
    sz = size(Ap)
    Bp = reshape(M * reshape(Ap, sz[1], :), size(M, 1), sz[2:end]...)
    return permutedims(Bp, invperm(p))
end

# THE BOND GAUGE. A 1×1 C4v PEPS is invariant under R → (X ⊗ X ⊗ X ⊗ X) R for complex orthogonal X
# (X Xᵀ = 1, the same on every virtual leg): D(D−1)/2 more null directions of J, right and
# (bilinearly) left — the tangents a·R of the antisymmetric generators a, in coordinates.
function _bp_gauge_tangents(c, B, D::Int, dp::Int)
    A = reshape(B * c, D, D, D, D, dp)
    out = Vector{eltype(A)}[]
    for i in 1:D, j in (i + 1):D
        a = zeros(D, D); a[i, j] = 1; a[j, i] = -1
        δ = sum(_bp_modemul(A, a, leg) for leg in 1:4)
        push!(out, transpose(B) * vec(δ))
    end
    return out
end

# Newton's reduced system at `c`: Q spans the Hermitian complement of the null directions (c and
# the gauge tangents), W the bilinear one ({y : vᵀy = 0}).
function _bp_gauge_bases(c, B, D::Int, dp::Int)
    N = reduce(hcat, vcat([c], _bp_gauge_tangents(c, B, D, dp)))
    return nullspace(Matrix(N')), nullspace(Matrix(transpose(N)))
end

# Central-difference Jacobian of the gradient coordinates, columns in parallel.
function _bp_fd_jacobian(c, B, ctx, ln, ls, ε, tol, ref)
    n = length(c)
    cols = Vector{Vector{ComplexF64}}(undef, n)
    tasks = map(1:n) do k
        Threads.@spawn begin
            e = zeros(eltype(c), n); e[k] = ε
            _, Gp, _, _ = _bp_evaluate(_bp_from_coords(c + e, B, ctx.al, ref), ctx, ln, ls, tol)
            _, Gm, _, _ = _bp_evaluate(_bp_from_coords(c - e, B, ctx.al, ref), ctx, ln, ls, tol)
            (_bp_to_coords(Gp, B, ctx.al) - _bp_to_coords(Gm, B, ctx.al)) / (2ε)
        end
    end
    for k in 1:n
        cols[k] = fetch(tasks[k])
    end
    return reduce(hcat, cols)
end

"""
    boundary_peps_stationary(site, legs, init; maxdim, A0 = init.A, jacobian = nothing, tol = 1e-9,
                             maxiter = 12, fd_step = 1e-5, ctm_tolerance = 1e-12, ctm_maxiter = 2000,
                             ls_max = 8, noise_tol = 1e-7, svd_rtol = 1e-6, step_tol = 1e-7,
                             fd_ctm_tolerance = ctm_tolerance, refresh_jacobian = true,
                             verbose = false, kwargs...) -> (bp, info)

The STATIONARY 1×1 boundary PEPS of a (complex) symmetric layer operator (see the note above):
Newton's method on ∇f = 0 for the bilinear estimator `f(R) = ln κ(RᵀTR) − ln κ(RᵀR)` in the
C4v-symmetric coordinates, from `A0` (default `init.A`; a predictor in a continuation), with
`init`'s legs and warm-started environments; real data stay real. The full finite-difference
Jacobian costs 2n evaluations, so this suits small D and continuations that reuse it (`jacobian`);
otherwise [`boundary_peps_krylov`](@ref) is faster. `site` must carry `legs` (relabel a new
coupling's site onto them). The Jacobian is recomputed by central differences (step `fd_step`) at
the first iteration unless `jacobian` (a previous `info.J`) is passed, and Broyden-updated after every
accepted step; steps backtrack on the residual (up to `ls_max` halvings), a stale Jacobian is
recomputed once, and a fresh one that gives no descent means the gradient's noise floor (set by χ
and `ctm_tolerance`: measured 2e-9 at 3D Ising β = 0.18, 1e-8 at β = 0.21, D = 2, χ = 16) —
converged if the residual is below `noise_tol`.

`info`: `J` (the last Jacobian), `sv` (singular values of the reduced Jacobian, ascending),
`iterations`, `residual` (|g|·|c|), `converged`. The reduced Jacobian is ill-conditioned (3D Ising,
D = 2: singular values from 5e-6 to 2.6 — the flat directions of the boundary-PEPS landscape), so
`sv[1]` is not a fold indicator in practice; the magnetisation is (m − m_c ∝ √(θ_f − θ) at a fold).
"""
function boundary_peps_stationary(site, legs, init::BoundaryPEPS; maxdim::Integer = init.normenv.maxdim,
                                  A0 = init.A, jacobian = nothing, tol::Real = 1.0e-9,
                                  maxiter::Integer = 12, fd_step::Real = 1.0e-5,
                                  ctm_tolerance::Real = 1.0e-12, ctm_maxiter::Integer = 2000,
                                  ls_max::Integer = 8, noise_tol::Real = 1.0e-7, svd_rtol::Real = 1.0e-6,
                                  step_tol::Real = 1.0e-7, fd_ctm_tolerance::Real = ctm_tolerance,
                                  refresh_jacobian::Bool = true,
                                  verbose::Bool = false, kwargs...)
    al, bl = init.Alegs, init.blegs
    _bp_isc4v(site, legs) || throw(ArgumentError("boundary_peps_stationary needs a C4v-invariant site"))
    # real data stays real (the complex contraction costs 2–4× more)
    cplx = scalartype(site) <: Complex || scalartype(A0) <: Complex ||
           (!isnothing(jacobian) && eltype(jacobian) <: Complex)
    sitec = cplx && !(scalartype(site) <: Complex) ? site * complex(1.0) : site
    ctx = (; al, bl, site = sitec, legs = Tuple(legs), maxdim = Int(maxdim), symmetrize = true,
           ctm_tolerance, ctm_maxiter = Int(ctm_maxiter), ctm_kwargs = merge((; c4v = true), kwargs),
           bilinear = true)
    B = _bp_c4v_basis(al)
    ref = sitec
    coords(G) = (x = _bp_to_coords(G, B, al); cplx ? x : real(x))
    c = coords(A0)
    c /= norm(c)
    reuse = init.normenv isa InfiniteCTM2D && init.normenv.maxdim == maxdim
    ln = reuse ? (cplx ? _i2_complexify(init.normenv) : init.normenv) : nothing
    ls = reuse ? (cplx ? _i2_complexify(init.openv) : init.openv) : nothing
    J = jacobian
    f = NaN; g = zeros(eltype(c), length(c)); res = Inf; it = 0; converged = false
    history = Float64[]
    evalc(cc, n0, s0) = (r = _bp_evaluate(_bp_from_coords(cc, B, al, ref), ctx, n0, s0, ctm_tolerance);
                         (real(r[1]), coords(r[2]), r[3], r[4]))
    f, g, ln, ls = evalc(c, ln, ls)
    res = norm(g) * norm(c)
    push!(history, f)
    fresh = false                                   # is J a finite-difference Jacobian at c?
    for k in 1:(maxiter + 1)
        it = k - 1
        verbose && (println("  stationary it $it: f = $f, |g||c| = $res"); flush(stdout))
        if res < tol
            converged = true
            break
        end
        k == maxiter + 1 && break
        if isnothing(J)
            J = _bp_fd_jacobian(c, B, ctx, ln, ls, fd_step * norm(c), fd_ctm_tolerance, ref)
            cplx || (J = real(J))
            fresh = true
        end
        Q, W = _bp_gauge_bases(c, B, dim(al[1]), dim(al[5]))
        cplx || (Q = real(Q); W = real(W))
        # TRUNCATED-SVD Newton step: a boundary PEPS with more bond dimension than the state uses is
        # redundant — 3D Ising β = 0.18, D = 3: ~15 reduced-Jacobian singular values below 1e-8 (to
        # 1e-17) against a largest 2.6, and the plain solve stepped |dc| = 1.9e3. Directions below
        # `svd_rtol`·σ_max leave f (and the state) unchanged; they are dropped.
        F = svd(W' * J * Q)
        keep = F.S .> svd_rtol * F.S[1]
        dc = Q * (-(F.V[:, keep] * ((F.U[:, keep]' * (W' * g)) ./ F.S[keep])))
        # the residual along the dropped directions cannot be reduced (a floor ~1e-6 at D = 3):
        # converged once the step on the kept ones is negligible AND the residual is at the noise
        # level (near an exceptional point J grows as v^{-1/2}, so steps get small early: the D = 1
        # chain stopped with m 1.4e-6 off)
        if norm(dc) < step_tol && res < noise_tol
            converged = true
            break
        end
        # DAMPED: backtrack on the residual |g||c| (the undamped step from a distant start diverged,
        # 3D Ising β = 0.18: 0.11 → 0.13 → 5.3). Trials warm-start from the accepted environments;
        # every accepted step Broyden-updates J (J Δc = Δg).
        accepted = false
        α = 1.0
        for _ in 1:ls_max
            ct = c + α * dc
            ct /= norm(ct)
            ft, gt, lnt, lst = evalc(ct, ln, ls)
            rt = norm(gt) * norm(ct)
            if isfinite(rt) && rt < (1 - 1.0e-4 * α) * res
                Δc = ct - c; Δg = gt - g
                J = J + ((Δg - J * Δc) * Δc') / real(dot(Δc, Δc))
                c, g, f, ln, ls, res = ct, gt, ft, lnt, lst, rt
                push!(history, f)
                accepted = true; fresh = false
                break
            end
            verbose && (println("    trial α = $α rejected: |g||c| = $rt"); flush(stdout))
            α /= 2
        end
        if !accepted
            if fresh                                # a fresh Jacobian and no descent: the noise floor
                converged = res < noise_tol
                break
            end
            refresh_jacobian || break               # (a refresh costs 2n evaluations: ~450 s at D = 3)
            J = nothing                             # a stale Broyden Jacobian: recompute it
        end
    end
    converged = converged || res < noise_tol        # stopped at the gradient's noise floor
    Q, W = _bp_gauge_bases(c, B, dim(al[1]), dim(al[5]))
    sv = isnothing(J) ? Float64[] : sort(svdvals(W' * J * Q))
    A = _bp_from_coords(c, B, al, ref)
    bp = BoundaryPEPS(A, al, bl, sitec, Tuple(legs), ln, ls, f, norm(g), history, true)
    return bp, (; J, sv, iterations = it, residual = res, converged, c)
end

# --- NEWTON–KRYLOV WITH A SUBSPACE TRUST REGION --------------------------------------------------
#
# The recommended solver (docs/boundary_peps.md, "Routes to the boundary state"). The Newton method
# above builds all of J by finite differences (2n evaluations: n = 42 at D = 3, 110 at D = 4) before
# its first step. Newton–Krylov never forms J: it builds a subspace from Jacobian-vector products, each
# the finite difference of the gradient along a basis vector (one warm-started evaluation; two, in
# parallel, with `central`), in the reduced coordinates (the complement Q of the scale and bond-gauge
# null directions, W = Q̄, so the operator QᵀJQ is complex symmetric, and real symmetric — the Hessian
# of f — for real data). The subspace starts from −g and the directions recycled from the last step,
# and grows by block Arnoldi (`block` products at a time, on threads) until the model's residual
# outside it is below η|g| (Eisenstat–Walker forcing).
#
# WHY A TRUST REGION. The Hessian's spectrum is wide and soft at the bottom. 3D Ising β = 0.25, D = 2,
# χ = 16 (measured 2026-09-26): reduced eigenvalues −2.7 … −2.9e-6; at D = 3 they reach ±1e-13, some
# of them POSITIVE (the redundant bond dimension: the point is a saddle in directions that barely
# change the state). Along the soft modes the Newton step is long (0.063 along the λ = −1.6e-4 mode at
# |g| = 3.9e-5) and f is far from quadratic on that scale: the full step raised |g| 30× while f
# improved, and a line search on |g| stalled. So the step is the trust-region step of the model in the
# subspace: for real data the model of f (maximised; negative curvature, i.e. a positive eigenvalue,
# is followed to the boundary), the ratio of the actual to the predicted gain in f accepting it and
# adapting the radius; for complex data, with no maximum principle, the model of |g|²
# (Levenberg–Marquardt, singular values below `svd_rtol` dropped). A rejected step shrinks the radius
# and is re-solved in the SAME subspace: no new products.
#
# Measured and dropped: the norm metric as a preconditioner (at D = 3 GMRES needed 12 / 27 products to
# a 1e-1 / 1e-2 residual against 3 / 12 plain), and reusing a subspace for further steps without new
# products (it stalls at D = 2 unless recycling feeds it, and is slower when it does).

# max_y  −β y₁ + ½ yᵀ T y  over |y| ≤ Δ (T = the symmetrised subspace projection of the Hessian of f,
# −β e₁ the gradient of f there): Moré–Sorensen on the eigendecomposition of the small T, the hard case
# completed along the lowest mode. Returns (y, predicted gain in f).
function _bp_trs_max(T, β::Real, Δ::Real)
    M = -Symmetric(real((T + T') / 2))                # minimise q(y) = β y₁ + ½ yᵀ M y
    F = eigen(M)
    μ, U = F.values, F.vectors
    a = β .* U[1, :]
    ylen(λ) = norm(a ./ (μ .+ λ))
    λ = 0.0
    hard = false
    if !(μ[1] > 0 && ylen(0.0) <= Δ)
        lo = max(0.0, -μ[1])
        if ylen(lo + 1.0e-12 * max(1.0, abs(μ[end]))) <= Δ
            λ = lo; hard = true                          # a₁ ≈ 0: complete along the lowest mode
        else
            hi = lo + β / Δ + 1.0e-12                    # ylen(hi) ≤ β / (μ₁ + hi) ≤ Δ
            for _ in 1:200
                mid = (lo + hi) / 2
                ylen(mid) > Δ ? (lo = mid) : (hi = mid)
                hi - lo <= 1.0e-14 * max(1.0, hi) && break
            end
            λ = hi
        end
    end
    z = zeros(length(μ))
    for i in eachindex(μ)
        d = μ[i] + λ
        abs(d) > 1.0e-300 && (z[i] = -a[i] / d)
    end
    if hard
        z[1] = 0.0
        z[1] = sqrt(max(Δ^2 - sum(abs2, z), 0.0))
    end
    y = U * z
    return y, -(β * y[1] + dot(y, M * y) / 2)
end

# min_y ‖β e₁ − H y‖ over |y| ≤ Δ, singular values below rtol·σ_max dropped (Levenberg–Marquardt in
# the subspace). Returns (y, predicted decrease of |g|², unconstrained relative residual).
function _bp_trs_lsq(H, β::Real, Δ::Real, rtol::Real)
    F = svd(H)
    keep = F.S .> rtol * F.S[1]
    S = F.S[keep]; a = β .* conj.(F.U[1, keep])
    e1 = zeros(eltype(H), size(H, 1)); e1[1] = β
    ystep(λ) = F.V[:, keep] * (S .* a ./ (S .^ 2 .+ λ))
    y0 = ystep(0.0)
    free = norm(e1 - H * y0) / β
    λ = 0.0
    if norm(y0) > Δ
        lo = 0.0; hi = S[1] * norm(a) / Δ
        for _ in 1:200
            mid = (lo + hi) / 2
            norm(S .* a ./ (S .^ 2 .+ mid)) > Δ ? (lo = mid) : (hi = mid)
            hi - lo <= 1.0e-14 * max(1.0, hi) && break
        end
        λ = hi
    end
    y = λ == 0 ? y0 : ystep(λ)
    return y, β^2 - norm(e1 - H * y)^2, free
end

"""
    boundary_peps_krylov(site, legs, init; maxdim, A0 = init.A, merit = :auto, tol = 1e-9,
                         maxiter = 200, krylovdim = 40, radius = 0.05, max_radius = 0.5,
                         fd_step = 1e-5, central = false, ctm_tolerance = 1e-12,
                         fd_ctm_tolerance = ctm_tolerance, ctm_maxiter = 2000, max_rejects = 8,
                         svd_rtol = 1e-6, eta_max = 0.1, eta_min = 1e-4, noise_tol = 1e-7, block = 4,
                         recycle = 3, ctm_anderson = 5,
                         callback = nothing, time_limit = Inf, verbose = false, kwargs...) -> (bp, info)

The stationary boundary PEPS (∇f = 0 for the bilinear estimator, as [`boundary_peps_stationary`](@ref);
for real data the variational optimum of [`boundary_peps`](@ref)) by Newton–Krylov with a subspace
trust region (see the note above). `merit`: `:f` (maximise f; the default for real data) or
`:residual` (minimise |g|²; the default, and the only choice, for complex data). The Krylov basis
is built from finite-difference Jacobian-vector products (step `fd_step`, forward unless `central`,
the perturbed environments converged to `fd_ctm_tolerance`), at most `krylovdim` per step, until
the model residual outside it is below η|g|, η from Eisenstat–Walker forcing in
[`eta_min`, `eta_max`]; `block` products are computed at a time, concurrently (block Arnoldi), and
with `recycle = r > 0` the subspace starts from the last accepted step and the r − 1 Ritz vectors
it moved along most, besides −g. The trust radius starts at `radius` (on the unit-norm coordinates)
and is capped at `max_radius`; a step rejected `max_rejects` times running stops the solver. Stops at
|g|·|c| < `tol`, converged, or after `time_limit` seconds; otherwise converged if the residual is
below `noise_tol` (the gradient's noise floor). `callback(it, state)` — `state` a NamedTuple
(A, Alegs, f, res, evals, time) — runs before every step. `ctm_anderson` as for [`boundary_peps`](@ref).

`info`: `iterations`, `residual`, `converged`, `evals` (gradient evaluations), `ctm_steps` (2D CTMRG
steps over all evaluations), `krylov` (products per step), `trace` (per step: time, evals, res, f,
radius), `radius` (the final trust radius), `c` (the final coordinates).
"""
function boundary_peps_krylov(site, legs, init::BoundaryPEPS; maxdim::Integer = init.normenv.maxdim,
                              A0 = init.A, merit::Symbol = :auto, tol::Real = 1.0e-9,
                              maxiter::Integer = 200, krylovdim::Integer = 40, radius::Real = 0.05,
                              max_radius::Real = 0.5, fd_step::Real = 1.0e-5, central::Bool = false,
                              ctm_tolerance::Real = 1.0e-12, fd_ctm_tolerance::Real = ctm_tolerance,
                              ctm_maxiter::Integer = 2000, max_rejects::Integer = 8,
                              svd_rtol::Real = 1.0e-6, eta_max::Real = 0.1, eta_min::Real = 1.0e-4,
                              noise_tol::Real = 1.0e-7, block::Integer = 4, recycle::Integer = 3,
                              ctm_anderson::Integer = 5, callback = nothing, time_limit::Real = Inf,
                              verbose::Bool = false, kwargs...)
    al, bl = init.Alegs, init.blegs
    _bp_isc4v(site, legs) || throw(ArgumentError("boundary_peps_krylov needs a C4v-invariant site"))
    cplx = scalartype(site) <: Complex || scalartype(A0) <: Complex
    merit in (:auto, :f, :residual) || throw(ArgumentError("merit must be :auto, :f or :residual"))
    cplx && merit === :f &&
        throw(ArgumentError("merit = :f needs real data (no maximum principle for complex T)"))
    variational = merit === :f || (merit === :auto && !cplx)
    sitec = cplx && !(scalartype(site) <: Complex) ? site * complex(1.0) : site
    ctx = (; al, bl, site = sitec, legs = Tuple(legs), maxdim = Int(maxdim), symmetrize = true,
           ctm_tolerance, ctm_maxiter = Int(ctm_maxiter), ctm_kwargs = merge((; c4v = true), kwargs),
           bilinear = true, anderson = Int(ctm_anderson))
    B = _bp_c4v_basis(al)
    D, dp = dim(al[1]), dim(al[5])
    c = _bp_to_coords(A0, B, al)
    cplx || (c = real(c))
    c /= norm(c)
    reuse = init.normenv isa InfiniteCTM2D && init.openv isa InfiniteCTM2D && init.normenv.maxdim == maxdim
    ln = reuse ? (cplx ? _i2_complexify(init.normenv) : init.normenv) : nothing
    ls = reuse ? (cplx ? _i2_complexify(init.openv) : init.openv) : nothing
    nevals = Threads.Atomic{Int}(0); nsteps = Threads.Atomic{Int}(0)
    t0 = time()
    f, g, ln, ls = _bp_krylov_eval(c, B, ctx, ln, ls, ctm_tolerance, cplx, nevals, nsteps)
    res = norm(g)
    Δ = Float64(radius)
    resprev = NaN; η = Float64(eta_max)
    trace = [(; time = time() - t0, evals = nevals[], res, f, radius = Δ)]
    ksizes = Int[]; it = 0; converged = false; stall = 0
    recycled = Vector{Vector{eltype(c)}}()
    for k in 1:(maxiter + 1)
        it = k - 1
        verbose && (println("  krylov it $it: f = $f, |g| = $res, Δ = $Δ, evals $(nevals[]), ",
                            round(time() - t0; digits = 1), " s"); flush(stdout))
        isnothing(callback) || callback(it, (; A = _bp_from_coords(c, B, al, sitec), Alegs = al, f, res,
                                            evals = nevals[], time = time() - t0))
        if res < tol
            converged = true
            break
        end
        (k == maxiter + 1 || time() - t0 > time_limit) && break
        # forcing term (Eisenstat–Walker choice 2), never tighter than the last step needs
        if k > 1 && isfinite(resprev)
            ηn = 0.9 * (res / resprev)^2
            0.9 * η^2 > 0.1 && (ηn = max(ηn, 0.9 * η^2))
            η = clamp(ηn, eta_min, eta_max)
        end
        ηk = max(η, 0.5 * tol / res)
        # the subspace of the reduced operator: r₀ = −Qᵀg first, then the directions recycled from
        # the last step, then (block) Arnoldi — `block` products at a time, on threads. Any orthonormal
        # V with J V_k ⊂ span V works: T = H[1:k, 1:k] is the Galerkin projection, and the model's
        # residual outside span V_k is H[k+1:p, 1:k] y.
        N = reduce(hcat, vcat([c], _bp_gauge_tangents(c, B, D, dp)))
        Q = nullspace(Matrix(N'))
        cplx || (Q = real(Q))
        m = size(Q, 2)
        r0 = -(transpose(Q) * g)
        β0 = norm(r0)
        kmax = min(krylovdim, m)
        V = reshape(r0 / β0, m, 1)
        for v in recycled
            V = _bp_extend(V, Q' * v)
        end
        H = zeros(eltype(Q), m + 1, kmax)
        kk = 0; est = 1.0
        while kk < min(kmax, size(V, 2))
            batch = (kk + 1):min(kk + block, kmax, size(V, 2))
            Jus = _bp_krylov_products(c, g, [Q * V[:, j] for j in batch], B, ctx, ln, ls, fd_step, central,
                                      fd_ctm_tolerance, cplx, nevals, nsteps)
            for (i, j) in enumerate(batch)
                w = transpose(Q) * Jus[i]
                for _ in 1:2, l in 1:size(V, 2)       # modified Gram–Schmidt, twice
                    h = dot(V[:, l], w); H[l, j] += h; w -= h * V[:, l]
                end
                nw = norm(w)
                if nw > 1.0e-12 * β0 && size(V, 2) < m
                    V = hcat(V, w / nw); H[size(V, 2), j] = nw
                end
            end
            kk = last(batch)
            p = size(V, 2)
            # the model's residual outside the subspace, relative to |g|
            if variational
                y, _ = _bp_trs_max(H[1:kk, 1:kk], β0, Δ)
                est = norm(H[(kk + 1):p, 1:kk] * y) / β0
            else
                _, _, est = _bp_trs_lsq(H[1:p, 1:kk], β0, Inf, svd_rtol)
            end
            (est <= ηk || p == kk) && break
        end
        p = size(V, 2)
        push!(ksizes, kk)
        # the trust-region step in the fixed subspace; a rejection shrinks Δ and re-solves there
        accepted = false
        for _ in 1:max_rejects
            y, pred = variational ? _bp_trs_max(H[1:kk, 1:kk], β0, Δ) :
                                    _bp_trs_lsq(H[1:p, 1:kk], β0, Δ, svd_rtol)
            dc = Q * (V[:, 1:kk] * y)
            ct = (c + dc) / norm(c + dc)
            ft, gt, lnt, lst = _bp_krylov_eval(ct, B, ctx, ln, ls, ctm_tolerance, cplx, nevals, nsteps)
            rt = norm(gt)
            actual = variational ? ft - f : res^2 - rt^2
            # below the resolution of f (or of |g|²) the residual decides
            ρ = pred > 1.0e-13 * max(1.0, abs(f)) || !variational ? actual / pred : (rt < res ? 1.0 : -1.0)
            ok = isfinite(rt) && isfinite(ρ) && ρ > 1.0e-4
            verbose && (println("    $kk products, |y| = $(norm(y)) (Δ = $Δ): predicted $pred, actual $actual, ",
                                "ρ = $ρ, |g| → $rt", ok ? "" : "  REJECTED"); flush(stdout))
            if ρ < 0.25 || !isfinite(ρ)
                Δ = 0.25 * norm(y)
            elseif ρ > 0.75 && norm(y) > 0.99 * Δ
                Δ = min(2Δ, Float64(max_radius))
            end
            if ok
                recycled = _bp_recycle(dc, y, H, V, Q, kk, p, recycle, variational)
                resprev = res
                c, g, f, ln, ls, res = ct, gt, ft, lnt, lst, rt
                accepted = true
                break
            end
        end
        push!(trace, (; time = time() - t0, evals = nevals[], res, f, radius = Δ))
        accepted || break                              # no gain: the noise floor
        # accepted steps that no longer reduce |g| below `noise_tol`: the noise floor too (|g| crawled
        # at 5.1e-9 for 20 steps, 3D Ising β = 0.2275, D = 3, χ = 16, ctm_tolerance = 1e-12)
        stall = res < noise_tol && res > 0.5 * resprev ? stall + 1 : 0
        stall >= 3 && break
    end
    converged = converged || res < noise_tol
    A = _bp_from_coords(c, B, al, sitec)
    bp = BoundaryPEPS(A, al, bl, sitec, Tuple(legs), ln, ls, f, res, [t.f for t in trace], true)
    return bp, (; iterations = it, residual = res, converged, evals = nevals[], ctm_steps = nsteps[],
                krylov = ksizes, trace, radius = Δ, c)
end

function _bp_krylov_eval(c, B, ctx, ln, ls, τ, cplx, nevals, nsteps)
    f, G, lnn, lsn = _bp_evaluate(_bp_from_coords(c, B, ctx.al, ctx.site), ctx, ln, ls, τ)
    Threads.atomic_add!(nevals, 1)
    Threads.atomic_add!(nsteps, lnn.stats[].iterations + lsn.stats[].iterations)
    g = _bp_to_coords(G, B, ctx.al)
    return real(f), cplx ? g : real(g), lnn, lsn
end

# J u for every u in `us`, concurrently
function _bp_krylov_products(c, g, us, B, ctx, ln, ls, h, central, τ, cplx, nevals, nsteps)
    jv(u) = _bp_krylov_jv(c, g, u, B, ctx, ln, ls, h, central, τ, cplx, nevals, nsteps)
    length(us) == 1 && return [jv(only(us))]
    return fetch.([Threads.@spawn jv(u) for u in us])
end

# V with v appended, orthonormalised against it (unchanged if v is dependent on V's columns)
function _bp_extend(V, v)
    size(V, 2) >= size(V, 1) && return V
    nv = norm(v)
    nv > 0 || return V
    w = v - V * (V' * v)
    w -= V * (V' * w)
    norm(w) > 1.0e-8 * nv || return V
    return hcat(V, w / norm(w))
end

# The directions carried into the next step's subspace (full coordinates): the accepted step, then
# the Ritz vectors of the model it moved along most (the soft modes).
function _bp_recycle(dc, y, H, V, Q, kk, p, n, variational)
    out = Vector{eltype(dc)}[]
    n <= 0 && return out
    push!(out, dc)
    n == 1 && return out
    if variational
        F = eigen(Symmetric(real((H[1:kk, 1:kk] + H[1:kk, 1:kk]') / 2)))
        U = F.vectors
    else
        U = svd(H[1:p, 1:kk]).V
    end
    w = abs.(U' * y)
    for i in sortperm(w; rev = true)[1:min(n - 1, kk)]
        push!(out, Q * (V[:, 1:kk] * U[:, i]))
    end
    return out
end

# J u by a finite difference of the gradient (|u| = 1), warm-started from the environments at c
function _bp_krylov_jv(c, g, u, B, ctx, ln, ls, h, central, τ, cplx, nevals, nsteps)
    if central
        tp = Threads.@spawn _bp_krylov_eval(c + h * u, B, ctx, ln, ls, τ, cplx, nevals, nsteps)
        gm = _bp_krylov_eval(c - h * u, B, ctx, ln, ls, τ, cplx, nevals, nsteps)[2]
        return (fetch(tp)[2] - gm) / (2h)
    end
    return (_bp_krylov_eval(c + h * u, B, ctx, ln, ls, τ, cplx, nevals, nsteps)[2] - g) / h
end
