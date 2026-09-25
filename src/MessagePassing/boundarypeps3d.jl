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
function _bp_norm_layers(A, al, bl)
    ket = A
    bra = replaceinds(conj(A), collect(al[1:4]), collect(bl))
    return Any[ket, bra], ntuple(d -> Index[al[d], bl[d]], 4)
end

function _bp_sandwich_layers(A, al, bl, site, legs)
    ket = replaceind(A, al[5], legs[5])                                         # A's p = T's z⁻
    bra = replaceinds(conj(A), vcat(collect(al[1:4]), [al[5]]), vcat(collect(bl), [legs[6]]))   # … z⁺
    return Any[ket, site, bra], ntuple(d -> Index[al[d], legs[d], bl[d]], 4)
end

# f(A), its gradient on A's legs, and the two converged environments (warm-started from `init`).
function _bp_evaluate(A, ctx, init_n, init_s)
    al, bl, site, legs = ctx.al, ctx.bl, ctx.site, ctx.legs
    nl, nlegs = _bp_norm_layers(A, al, bl)
    sl, slegs = _bp_sandwich_layers(A, al, bl, site, legs)
    ln = update(InfiniteCTM2D(nl, nlegs, ctx.maxdim; init = init_n, ctx.ctm_kwargs...);
                tolerance = ctx.ctm_tolerance, maxiter = ctx.ctm_maxiter)
    ls = update(InfiniteCTM2D(sl, slegs, ctx.maxdim; init = init_s, ctx.ctm_kwargs...);
                tolerance = ctx.ctm_tolerance, maxiter = ctx.ctm_maxiter)
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

# The initial boundary tensor: T applied to the product state `b` on the z⁻ legs, its virtual legs
# embedded into (or cut down to) bond dimension D, plus a little noise so that padded directions
# have a gradient (they would otherwise stay exactly zero).
function _bp_initial(site, legs, al, D, b, noise, rng)
    elt = scalartype(site)
    bv = isnothing(b) ? ones(elt, dim(legs[5])) : convert(Vector{elt}, collect(b))
    t = site * from_array(bv, legs[5])
    for d in 1:4
        dz = dim(legs[d])
        M = zeros(elt, dz, D)
        for j in 1:min(dz, D)
            M[j, j] = 1
        end
        t = t * from_array(M, legs[d], al[d])
    end
    t = replaceind(t, legs[6], al[5])
    t = t / norm(t)
    if noise > 0
        t = t + noise * random_tensor(rng, elt, collect(al)) / sqrt(prod(dim.(collect(al))))
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
        t = t * from_array(M, old[d], new[d])
    end
    t = replaceind(t, old[5], new[5])
    t = t / norm(t)
    if D > Din && noise > 0
        t = t + noise * random_tensor(rng, elt, collect(new)) / sqrt(prod(dim.(collect(new))))
        t = t / norm(t)
    end
    return t
end

"""
    boundary_peps(site, legs, D; maxdim, init = nothing, boundary = nothing, symmetrize = true,
                  maxiter = 200, gtol = 1e-7, memory = 10, max_step = 0.2, ctm_tolerance = 1e-10,
                  ctm_maxiter = 1000, noise = 1e-2, seed = 0, verbose = false, kwargs...) -> BoundaryPEPS

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
  L-BFGS iterations, or when the line search fails.

Remaining keywords go to [`InfiniteCTM2D`](@ref) (e.g. `pair`).
"""
function boundary_peps(site, legs, D::Integer; maxdim::Integer, init = nothing, boundary = nothing,
                       symmetrize::Bool = true, maxiter::Integer = 200, gtol::Real = 1.0e-7,
                       memory::Integer = 10, max_step::Real = 0.2, ctm_tolerance::Real = 1.0e-10,
                       ctm_maxiter::Integer = 1000, noise::Real = 1.0e-2, seed::Integer = 0,
                       ls_max::Integer = 12, verbose::Bool = false, kwargs...)
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
           ctm_maxiter = Int(ctm_maxiter), ctm_kwargs = kwargs)
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
    f, G, ln, ls = _bp_evaluate(A, ctx, init_n, init_s)
    F, g = -f, tangent(-G, A)
    history = [f]
    verbose && (println("boundary_peps D=$D χ=$maxdim: start f = $f, |g| = $(norm(g))"); flush(stdout))
    Ss = Any[]; Ys = Any[]; ρs = Float64[]
    gnorm = norm(g)
    for it in 1:maxiter
        gnorm = norm(g)
        gnorm < gtol && (verbose && println("boundary_peps: |g| = $gnorm below gtol"); break)
        # two-loop recursion, H₀ = γ I
        q = copy(g); αs = zeros(length(Ss))
        for i in length(Ss):-1:1
            αs[i] = ρs[i] * real(dot(Ss[i], q)); q = q - αs[i] * Ys[i]
        end
        γ = isempty(Ss) ? 1.0 : real(dot(Ss[end], Ys[end])) / real(dot(Ys[end], Ys[end]))
        r = γ * q
        for i in 1:length(Ss)
            β = ρs[i] * real(dot(Ys[i], r)); r = r + (αs[i] - β) * Ss[i]
        end
        d = tangent(-r, A)
        slope = real(dot(d, g))
        if !(slope < 0)
            empty!(Ss); empty!(Ys); empty!(ρs)
            d = -g; slope = real(dot(d, g))
        end
        α = isempty(Ss) ? min(1.0, max_step / norm(d)) : 1.0
        α * norm(d) > max_step && (α = max_step / norm(d))
        accepted = false
        local An, fn, Gn, lnn, lsn
        for k in 1:ls_max
            An = A + α * d
            symmetrize && (An = _bp_symmetrize(An, al[1:4]))
            An = An / norm(An)
            fn, Gn, lnn, lsn = _bp_evaluate(An, ctx, ln, ls)
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
        push!(history, f)
        verbose && (println("boundary_peps it $it: f = $f, |g| = $(norm(g)), α = $α"); flush(stdout))
    end
    return BoundaryPEPS(A, al, bl, site, Tuple(legs), ln, ls, f, norm(g), history)
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
    sl, _ = _bp_sandwich_layers(bp.A, bp.Alegs, bp.blegs, bp.site, bp.legs)
    return site_ratio(bp.openv, Any[sl[1], impurity, sl[3]])
end
