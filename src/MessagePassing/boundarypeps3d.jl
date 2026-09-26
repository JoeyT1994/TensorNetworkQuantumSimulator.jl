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
# contracted by `InfiniteCTM2D` in the Kikuchi form. For a symmetric T (a symmetric bond split, as
# `ising3d_site`'s) f is variational: f ≤ ln κ₃D, approaching it as D grows.
#
# THE GRADIENT IS LOCAL. T is a product of site tensors, not a sum of local terms, so the derivative
# of either per-site free energy with respect to A is its normalised one-site environment
# (`site_environment`): no sum over positions, no channel environments. With A in the ket and the
# bra layer,
#
#   ∂f/∂A = (E_ket + E_bra)/z |_{⟨Ψ|T|Ψ⟩} − (E_ket + E_bra)/z |_{⟨Ψ|Ψ⟩}   (real A)
#
# f is invariant under A → cA, so the gradient is orthogonal to A; the optimiser works on the unit
# sphere (steps projected tangent, A renormalised), with L-BFGS and Armijo backtracking as in the
# DMRG branch's `ctmrg_lbfgs`. With `symmetrize = true` (a C4v-invariant site) A and every gradient
# are projected onto the C4v-symmetric subspace of the virtual legs.

"""
    BoundaryPEPS

The result of [`boundary_peps`](@ref): the boundary tensor `A` on legs `Alegs = (x⁻, x⁺, y⁻, y⁺, p)`
(`p` the z bond), the 3D `site` and its `legs`, the converged 2D environments of ⟨Ψ|Ψ⟩ (`normenv`)
and ⟨Ψ|T|Ψ⟩ (`openv`), `lnkappa = f(A)` and the optimisation `history` of f.
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
                             maxiter = 12, fd_step = 1e-5, ctm_tolerance = 1e-12,
                             ctm_maxiter = 4000, verbose = false, kwargs...) -> (bp, info)

The STATIONARY 1×1 boundary PEPS of a complex symmetric layer operator (see the note above):
Newton's method on ∇f = 0 for the bilinear estimator `f(R) = ln κ(RᵀTR) − ln κ(RᵀR)` in the
C4v-symmetric coordinates, from `A0` (default `init.A`; a predictor in a continuation), with
`init`'s legs and warm-started environments. `site` must carry `legs` (relabel a new coupling's
site onto them). The Jacobian is recomputed by central differences (step `fd_step`) at the first
iteration unless `jacobian` (a previous `info.J`) is passed, and Broyden-updated after every
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
# The Newton method above builds all of J by finite differences (2n evaluations: n = 42 at D = 3,
# 110 at D = 4) before its first step. Newton–Krylov never forms J: the Krylov basis is built by
# Arnoldi from Jacobian-vector products, each the finite difference of the gradient along a basis
# vector (one warm-started evaluation; two, in parallel, with `central`), in the reduced coordinates
# (the complement Q of the scale and bond-gauge null directions, W = Q̄, so the operator QᵀJQ is
# complex symmetric, and real symmetric — the Hessian of f — for real data).
#
# WHY A TRUST REGION. The Hessian's spectrum is wide and soft at the bottom. 3D Ising β = 0.25, D = 2,
# χ = 16 (measured 2026-09-26): reduced eigenvalues −2.7 … −2.9e-6; at D = 3 they reach ±1e-13, some
# of them POSITIVE (the redundant bond dimension: the point is a saddle in directions that barely
# change the state). Along the soft modes the Newton step is long (0.063 along the λ = −1.6e-4 mode at
# |g| = 3.9e-5) and f is far from quadratic on that scale: the full step raised |g| 30× while f
# improved, and a line search on |g| stalled. So the step is the trust-region step of the model in the
# Krylov subspace: for real data the model of f (maximised; negative curvature, i.e. a positive
# eigenvalue, is followed to the boundary) and the ratio of the actual to the predicted gain in f
# accepts it and adapts the radius; for complex data, with no maximum principle, the model of |g|²
# (Levenberg–Marquardt, small singular values below `svd_rtol` dropped). A rejected step shrinks the
# radius and is re-solved in the SAME subspace: no new products. The basis grows until the model's
# residual outside it is below η|g| (Eisenstat–Walker forcing).
#
# The norm metric does not help here as a preconditioner: at D = 3 the metric-preconditioned
# operator needed more products to a 1e-1 / 1e-2 relative residual than the plain one (12 / 27
# against 3 / 12, of 38).

# max_y  −bᵀy + ½ yᵀ T y  over |y| ≤ Δ (T = the symmetrised Krylov projection of the Hessian of f, −b
# the gradient of f in the subspace; b = β e₁ for a fresh basis): Moré–Sorensen on the
# eigendecomposition of the small T, the hard case completed along the lowest mode. Returns
# (y, predicted gain in f).
_bp_trs_max(T, β::Number, Δ) = _bp_trs_max(T, [β; zeros(size(T, 1) - 1)], Δ)
function _bp_trs_max(T, b::AbstractVector, Δ)
    M = -Symmetric(real((T + T') / 2))                # minimise q(y) = bᵀy + ½ yᵀ M y
    F = eigen(M)
    μ, U = F.values, F.vectors
    a = U' * real(b)
    nb = norm(a)
    ylen(λ) = norm(a ./ (μ .+ λ))
    λ = 0.0
    hard = false
    if !(μ[1] > 0 && ylen(0.0) <= Δ)
        lo = max(0.0, -μ[1])
        if ylen(lo + 1.0e-12 * max(1.0, abs(μ[end]))) <= Δ
            λ = lo; hard = true                          # a₁ ≈ 0: complete along the lowest mode
        else
            hi = lo + nb / Δ + 1.0e-12
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
    return y, -(dot(real(b), y) + dot(y, M * y) / 2)
end

# min_y ‖b − H y‖ over |y| ≤ Δ (b = β e₁ for a fresh basis), singular values below rtol·σ_max
# dropped (Levenberg–Marquardt in the Krylov subspace). Returns (y, predicted decrease of |g|²,
# unconstrained relative residual).
_bp_trs_lsq(H, β::Number, Δ, rtol) = _bp_trs_lsq(H, [β; zeros(eltype(H), size(H, 1) - 1)], Δ, rtol)
function _bp_trs_lsq(H, b::AbstractVector, Δ, rtol)
    F = svd(H)
    keep = F.S .> rtol * F.S[1]
    S = F.S[keep]; a = F.U[:, keep]' * b
    ystep(λ) = F.V[:, keep] * (S .* a ./ (S .^ 2 .+ λ))
    y0 = ystep(0.0)
    free = norm(b - H * y0) / norm(b)
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
    return y, norm(b)^2 - norm(b - H * y)^2, free
end

"""
    boundary_peps_krylov(site, legs, init; maxdim, A0 = init.A, merit = :auto, tol = 1e-9,
                         maxiter = 50, krylovdim = 40, radius = 0.05, max_radius = 0.5,
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
                              maxiter::Integer = 50, krylovdim::Integer = 40, radius::Real = 0.05,
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
    cplx && merit === :f && throw(ArgumentError("merit = :f needs real data (no maximum principle for complex T)"))
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
        # residual outside span V_k is H[k+1:p, 1:k] y. (Reusing a subspace for further steps from the
        # new gradient, without new products, was tried and measured worse: 3D Ising β = 0.25, D = 2.)
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
            y, pred = variational ? _bp_trs_max(H[1:kk, 1:kk], β0, Δ) : _bp_trs_lsq(H[1:p, 1:kk], β0, Δ, svd_rtol)
            dc = Q * (V[:, 1:kk] * y)
            ct = (c + dc) / norm(c + dc)
            ft, gt, lnt, lst = _bp_krylov_eval(ct, B, ctx, ln, ls, ctm_tolerance, cplx, nevals, nsteps)
            rt = norm(gt)
            actual = variational ? ft - f : res^2 - rt^2
            # below the resolution of f (or of |g|²) the residual decides
            ρ = pred > 1.0e-13 * max(1.0, abs(f)) || !variational ? actual / pred : (rt < res ? 1.0 : -1.0)
            ok = isfinite(rt) && isfinite(ρ) && ρ > 1.0e-4
            verbose && (println("    $kk products, |y| = $(norm(y)) (Δ = $Δ): predicted $pred, actual $actual, ρ = $ρ, |g| → $rt",
                                ok ? "" : "  REJECTED"); flush(stdout))
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
    length(us) == 1 && return [_bp_krylov_jv(c, g, us[1], B, ctx, ln, ls, h, central, τ, cplx, nevals, nsteps)]
    tasks = [Threads.@spawn _bp_krylov_jv(c, g, u, B, ctx, ln, ls, h, central, τ, cplx, nevals, nsteps) for u in us]
    return fetch.(tasks)
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

# --- the NATURAL POWER METHOD: the full-environment update A ← N⁻¹ E_s --------------------------
#
# One application of T per step, the result fitted back into the bond-D manifold with the FULL
# environment rather than bond by bond: the new tensor maximises the fidelity |⟨X|T|A⟩|²/⟨X|X⟩ (in the
# bilinear sense for complex symmetric T), X = N⁻¹ E_s, with N the norm metric of ⟨A|A⟩ (its shell with
# both of the site's layers removed) and E_s the environment of the bra slot of ⟨A|T|A⟩. At a fixed
# point N A ∝ E_s, i.e. E_s/z_s = E_n/z_n: ∇f = 0 — the stationary point of the Bethe estimate, loops
# included (unlike the bond-projector power method below). No Jacobian, no Hessian, no line search, and
# it needs no variational principle, so it serves complex (non-Hermitian) T as it serves real. It is
# the natural-gradient step with the power method's step length; for a Hermitian T the preconditioned
# L-BFGS of `boundary_peps` accelerates exactly this. Near-null directions of N are regularised by
# `shift`·‖N‖ — in the correction only (see inside), which keeps the fixed point exact.
#
# ⚠️ MEASURED UNSTABLE as a plain iteration (2026-09-26, 3D Ising β = 0.25, D = 2, χ = 16): from the
# projected-power state (2e-3 from the optimum) |ΔA| grows 0.08 → 0.40 → 0.96 → ~1.3 and wanders (m ends
# at 0.49), with shift 1e-12 … 1e-5 alike. The step is computed in FROZEN environments and the true map
# is expansive — the environment response the local-eigenproblem update also tripped on
# (docs/boundary_peps.md). Kept for experiments; a damped version is fixed-step natural gradient, which
# the line-searched L-BFGS of `boundary_peps` already does better.
"""
    boundary_peps_natural(site, legs, init::BoundaryPEPS; maxdim = init.normenv.maxdim,
                          maxiter = 400, tol = 1e-9, shift = 1e-5, ctm_tolerance = 1e-11,
                          ctm_maxiter = 2000, verbose = false, kwargs...) -> (bp, info)

The natural power method (see the note above) from `init`'s tensor and environments (`site` carrying
`init`'s legs; real or complex, bilinear pairing). Stops at ‖ΔA‖ < `tol`. `info`: `iterations`,
`change`, `converged`, `history`, `gnorm` (|∇f|·|c| at the end, in the C4v coordinates).
"""
function boundary_peps_natural(site, legs, init::BoundaryPEPS; maxdim::Integer = init.normenv.maxdim,
                               maxiter::Integer = 400, tol::Real = 1.0e-9, shift::Real = 1.0e-5,
                               ctm_tolerance::Real = 1.0e-11, ctm_maxiter::Integer = 2000,
                               verbose::Bool = false, kwargs...)
    al, bl = init.Alegs, init.blegs
    _bp_isc4v(site, legs) || throw(ArgumentError("boundary_peps_natural needs a C4v-invariant site"))
    cplx = scalartype(site) <: Complex || scalartype(init.A) <: Complex
    sitec = cplx && !(scalartype(site) <: Complex) ? site * complex(1.0) : site
    ctx = (; al, bl, site = sitec, legs = Tuple(legs), maxdim = Int(maxdim), symmetrize = true, ctm_tolerance,
           ctm_maxiter = Int(ctm_maxiter), ctm_kwargs = merge((; c4v = true), kwargs), bilinear = true)
    A = cplx ? init.A * complex(1.0) : init.A
    A = A / norm(A)
    reuse = init.normenv isa InfiniteCTM2D && init.normenv.maxdim == maxdim
    ln = reuse ? (cplx ? _i2_complexify(init.normenv) : init.normenv) : nothing
    ls = reuse && init.openv isa InfiniteCTM2D ? (cplx ? _i2_complexify(init.openv) : init.openv) : nothing
    ka = collect(al[1:4]); kb = collect(bl); p = al[5]
    history = Float64[]; change = Inf; it = 0; converged = false
    f = NaN; G = nothing
    for k in 1:maxiter
        it = k
        f, G, ln, ls = _bp_evaluate(A, ctx, ln, ls, ctm_tolerance)
        # E_s: the bra slot's environment in ⟨A|T|A⟩, on A's legs
        Es, _ = site_environment(ls, 3)
        Es = replaceinds(Es, vcat(collect(bl), [legs[6]]), collect(al))
        # N: the norm metric (ket legs × bra legs), from the shell of ⟨A|A⟩
        VX, _, env = _i2_shell(ln)
        old, new = _i2_site_relabelling(ln.legs, _I2_V, VX)
        Nt = _i2_relabel(env, new, old)
        M = Array(array(Nt, ka..., kb...))
        n = prod(size(M)[1:4]); M = reshape(M, n, n)
        # N⁻¹E_s = (z_s/z_n)(A + (z_n/z_s) N⁻¹ r), r = E_s − (z_s/z_n) E_n: regularise the CORRECTION
        # only. N is so ill-conditioned that A has weight in its near-null directions, and a shifted
        # (N + s)X = E_s suppresses exactly those — measured at the variational optimum (β = 0.25,
        # D = 2, where r ≈ 4e-9): one step moved A by 2.4e-4 / 2.8e-3 / 0.19 at s = 1e-12 / 1e-8 / 1e-5.
        # This way the fixed point (r = 0) is exact for any shift.
        En = replaceinds(Nt * A, kb, ka)                                  # N A, on A's legs
        zs = scalar(Es * A); zn = scalar(En * A)
        r = Es - (zs / zn) * En
        R = reshape(Array(array(r, ka..., p)), n, :)
        δ = (transpose(M) + shift * opnorm(M, 1) * I) \ R                   # N δ = r over the bra legs
        Xt = A + (zn / zs) * adapt_like(A, from_array(reshape(δ, (dim.(ka)..., dim(p))), ka..., p))
        Xt = _bp_symmetrize(Xt, al[1:4]); Xt = Xt / norm(Xt)
        ph = dot(A, Xt); ph = iszero(ph) ? one(ph) : conj(ph) / abs(ph)
        change = norm(Xt * ph - A)
        push!(history, change)
        A = Xt * ph
        verbose && (println("  natural it $k: f = $f, |ΔA| = $change"); flush(stdout))
        if change < tol
            converged = true
            break
        end
    end
    f, G, ln, ls = _bp_evaluate(A, ctx, ln, ls, ctm_tolerance)
    B = _bp_c4v_basis(al)
    gn = norm(_bp_to_coords(G, B, al)) * norm(_bp_to_coords(A, B, al))
    bp = BoundaryPEPS(A, al, bl, sitec, Tuple(legs), ln, ls, real(f), gn, Float64[real(f)], true)
    return bp, (; iterations = it, change, converged, history, gnorm = gn)
end

# --- the PROJECTED POWER METHOD: z-direction eig-CTMRG ------------------------------------------
#
# The boundary state by iteration rather than optimisation: R ← Π(T R), T's layer raising the bond to
# D·d and a rank-D projector Π = V_R V_L on every bond cutting it back. The projector is the MP-BP
# choice (Woolls et al., MP-BP §V B, one dimension up): the approximation Z(Π) = ⟨L|Π(T R)⟩ is made
# STATIONARY under Π, i.e. V_R / V_L span the dominant right / left invariant subspaces of the bond
# environment K — the environment of one bond of the network ⟨L|R′⟩ (R′ = Π(T R) everywhere else) with
# the two sites next to the bond left untruncated on it. Gauge-equivariant, no Jacobian, no Hessian,
# and a power method converges to the dominant eigenvector whether or not T is Hermitian. For a C4v
# network K is (complex) symmetric and Π a Takagi pair, V_L = V_Rᵀ, so R stays C4v-symmetric.
#
# L = R (bilinear): the left eigenvector of a complex symmetric T, and for real data the Hermitian
# case as well. The Bethe estimator f = ln κ⟨R|T|R⟩ − ln κ⟨R|R⟩ is evaluated at the end, with its
# gradient: whether the power fixed point is ALSO the stationary point of f is not implied for PEPS
# messages (the bond projectors are a restricted family on a plane with loops).

# T·A as an array with fused (a, s) virtual legs, a fastest: dims (D ds, D ds, D ds, D ds, dp)
function _bp_TA(Aarr, Tarr)
    D = size(Aarr, 1); dp = size(Aarr, 5); ds = size(Tarr, 1)
    M = reshape(Aarr, D^4, dp) * reshape(permutedims(Tarr, (5, 1, 2, 3, 4, 6)), dp, ds^4 * dp)
    X = reshape(M, D, D, D, D, ds, ds, ds, ds, dp)
    return reshape(permutedims(X, (1, 5, 2, 6, 3, 7, 4, 8, 9)), D * ds, D * ds, D * ds, D * ds, dp)
end

# Π applied on the virtual legs `modes` of X (V_L on the minus legs 1, 3; V_Rᵀ on the plus legs 2, 4)
function _bp_project(X, VR, VL, modes)
    for k in modes
        X = _bp_modemul(X, isodd(k) ? VL : transpose(VR), k)
    end
    return X
end

# the symmetric square root of a (complex) symmetric matrix, real for real input
_bp_symsqrt(G) = eltype(G) <: Real ? real(sqrt(Symmetric(G))) : (R = sqrt(G); (R + transpose(R)) / 2)
# bilinear (Takagi) normalisation V ← V (VᵀV)^{-1/2}, so that V_L = Vᵀ is its biorthogonal partner
_bp_takagi(V) = V / _bp_symsqrt(transpose(V) * V)
# the complex-orthogonal rotation of V closest to Vref (the bilinear polar factor of VᵀVref)
function _bp_align(V, Vref)
    size(V) == size(Vref) || return V
    M = transpose(V) * Vref
    R = _bp_symsqrt(transpose(M) * M)
    all(isfinite, R) || return V
    return V * (M / R)
end

"""
    boundary_peps_power(site, legs, D; maxdim, init = nothing, boundary = nothing, maxiter = 400,
                        tol = 1e-9, ctm_tolerance = 1e-11, ctm_maxiter = 2000, noise = 1e-3,
                        seed = 0, evaluate = true, verbose = false, kwargs...) -> (bp, info)

The boundary PEPS of a C4v-symmetric cubic network by the projected power method with MP-BP bond
projectors (see the note above). Iterates to ‖ΔA‖ < `tol` (A normalised, phase-aligned). With
`evaluate`, the returned `BoundaryPEPS` carries the Bethe estimate f and the sandwich environment,
and `info.gnorm` is |∇f|·|c| at the fixed point. `info`: `iterations`, `change`, `converged`,
`history` (‖ΔA‖ per step), `asym` (the bond environment's asymmetry ‖K − Kᵀ‖/‖K‖), `spectrum` (the
kept and first dropped eigenvalues of K, normalised), `gnorm`.
"""
function boundary_peps_power(site, legs, D::Integer; maxdim::Integer, init = nothing, boundary = nothing,
                             maxiter::Integer = 400, tol::Real = 1.0e-9, ctm_tolerance::Real = 1.0e-11,
                             ctm_maxiter::Integer = 2000, noise::Real = 1.0e-3, seed::Integer = 0,
                             evaluate::Bool = true, verbose::Bool = false, kwargs...)
    length(legs) == 6 || throw(ArgumentError("legs must be the six legs (x⁻, x⁺, y⁻, y⁺, z⁻, z⁺)"))
    _bp_isc4v(site, legs) || throw(ArgumentError("boundary_peps_power needs a C4v-invariant site"))
    rng = Xoshiro(seed)
    ds, dp = dim(legs[1]), dim(legs[5])
    Tarr = Array(array(site, legs...))
    if init isa BoundaryPEPS && dim(init.Alegs[1]) == D
        al, bl = init.Alegs, init.blegs
        A = init.A
    else
        al = (new_index(D; tags = "bp,xm"), new_index(D; tags = "bp,xp"), new_index(D; tags = "bp,ym"),
              new_index(D; tags = "bp,yp"), new_index(dp; tags = "bp,p"))
        bl = Tuple(new_index(D; tags = "bp,bra") for _ in 1:4)
        A = init isa BoundaryPEPS ? _bp_embed(init.A, init.Alegs, al, noise, rng) :
            _bp_initial(site, legs, al, D, boundary, D > ds ? noise : 0.0, rng)
    end
    A = _bp_symmetrize(A, al[1:4]); A = A / norm(A)
    Aarr = Array(array(A, al...))
    elt = promote_type(scalartype(site), eltype(Aarr))
    Aarr = convert(Array{elt}, Aarr); Tarr = convert(Array{elt}, Tarr)
    # initial projector: the dominant left singular vectors of T·A across one virtual leg
    TA = _bp_TA(Aarr, Tarr)
    U = svd(reshape(permutedims(TA, (2, 1, 3, 4, 5)), size(TA, 2), :)).U[:, 1:D]
    VR = _bp_takagi(convert(Matrix{elt}, U))
    ckw = merge((; c4v = true), kwargs)
    runctm(nl, nlegs, init) = update(InfiniteCTM2D(nl, nlegs, Int(maxdim); init, ckw...);
                                     tolerance = ctm_tolerance, maxiter = ctm_maxiter)
    ln = nothing
    history = Float64[]; asym = NaN; spec = ComplexF64[]
    converged = false; it = 0; change = Inf
    iL = new_index(D * ds; tags = "bp,iL"); iR = new_index(D * ds; tags = "bp,iR")
    for k in 1:maxiter
        it = k
        # R′ = Π(T R), C4v-symmetrised and normalised, phase-aligned to R
        Anew = _bp_project(TA, VR, transpose(VR), 1:4)
        Anew = Array(array(_bp_symmetrize(from_array(Anew, al...), al[1:4]), al...))
        Anew ./= norm(Anew)
        ph = dot(Aarr, Anew); ph = iszero(ph) ? one(ph) : conj(ph) / abs(ph)
        change = norm(Anew * ph - Aarr)
        push!(history, change)
        Aarr = Anew * ph
        At = from_array(Aarr, al...)
        # the environment of ⟨R′|R′⟩ (L = R′, bilinear)
        nl, nlegs = _bp_norm_layers(At, al, bl; bilinear = true)
        ln = runctm(nl, nlegs, ln)
        # the bond environment: the pair (3,3)–(4,3) with T·R′ on the open bond, Π elsewhere
        TA = _bp_TA(Aarr, Tarr)
        Xl = from_array(_bp_project(TA, VR, transpose(VR), (1, 3, 4)), al[1], iL, al[3], al[4], al[5])
        Xr = from_array(_bp_project(TA, VR, transpose(VR), (2, 3, 4)), iR, al[2], al[3], al[4], al[5])
        VX, _, blocks = _i2_pair_blocks(ln)
        bra = ln.site[2]
        Nt = _ctm_contract(vcat(blocks, _i2_place_site(Any[Xl, bra], ln.legs, (3, 3), VX),
                                _i2_place_site(Any[Xr, bra], ln.legs, (4, 3), VX)), ln.options)
        K = transpose(Array(array(Nt, iL, iR)))
        asym = norm(K - transpose(K)) / norm(K)
        K = (K + transpose(K)) / 2
        F = eigen(K)
        o = sortperm(abs.(F.values); rev = true)
        spec = ComplexF64.(F.values[o[1:min(D + 1, end)]] ./ F.values[o[1]])
        Vn = _bp_takagi(convert(Matrix{elt}, F.vectors[:, o[1:D]]))
        VR = _bp_align(Vn, VR)
        if verbose
            println("  power it $k: |ΔA| = $change, asym $asym, |λ_{D+1}/λ_D| = ",
                    length(spec) > D ? abs(spec[D + 1] / spec[D]) : NaN, ", norm CTM ", ln.stats[].iterations, " its")
            flush(stdout)
        end
        if change < tol && k > 2
            converged = true
            break
        end
    end
    A = from_array(Aarr, al...)
    f = NaN; gn = NaN; ls = nothing
    if evaluate
        sitec = elt <: Complex && !(scalartype(site) <: Complex) ? site * complex(1.0) : site
        ctx = (; al, bl, site = sitec, legs = Tuple(legs), maxdim = Int(maxdim), symmetrize = true, ctm_tolerance,
               ctm_maxiter = Int(ctm_maxiter), ctm_kwargs = ckw, bilinear = true)
        f, G, ln, ls = _bp_evaluate(A, ctx, ln, nothing)
        B = _bp_c4v_basis(al)
        gn = norm(_bp_to_coords(G, B, al)) * norm(_bp_to_coords(A, B, al))
    end
    bp = BoundaryPEPS(A, al, bl, site, Tuple(legs), ln, ls, real(f), gn, Float64[real(f)], true)
    return bp, (; iterations = it, change, converged, history, asym, spectrum = spec, gnorm = gn)
end
