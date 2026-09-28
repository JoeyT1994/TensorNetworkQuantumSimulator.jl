# Hexagonal-ice (Ih) boundary state on the HONEYCOMB — the coordination-4 formulation (docs/ice.md,
# "Honeycomb formulation"). Every O is its own tensor; nothing is fused into cubic-like cells.
#
# THE OPERATOR. The bilayer transfer M is a bond-2 PEPO on the honeycomb: a-vertices take the vertical
# bond from below plus 3 in-plane bonds (V_a), b-vertices 3 in-plane bonds plus the vertical bond above
# (V_b), both the ice rule (two of the O's four H near it). Hexagonal stacking returns to the same
# honeycomb every bilayer with the sublattices swapped, which is the in-plane inversion I: the power
# method iterates the SYMMETRIC operator I M, whose top eigenvector is the hexagonal boundary state
# (Mᵀ = I M I, so ln λ(I M) = 2 ln W(Ih) per cell of two molecules).
#
# THE STATE. A honeycomb PEPS: X[p, 1, 2, 3] on the sublattice carrying the physical (vertical) leg,
# Y[1, 2, 3] on the other, weights w_k on the three bond types (Vidal gauge). Cell R = (a(R), b(R)):
# a(R) bonds to b(R) [type 1], b(R − e1) [type 2], b(R − e2) [type 3] — a square topology.
#
# BP SIMPLE UPDATE (`hexstate`). One bilayer: apply M exactly (every bond D → 2D), bring the enlarged
# state to the Vidal (= BP) gauge by identity updates until the weights settle, truncate each bond by an
# SVD in that gauge, re-gauge, swap roles. The gauge fix is not optional: truncating with the old
# weights on the enlarged bonds (a non-canonical gauge) gave symmetry-broken states and a free energy
# that fell with D. `regauge` reproduces the library's BP gauge (`BeliefPropagationCache` +
# `symmetric_gauge`) bond spectra to 1e-10 (test/test_ice_honeycomb.jl). As D → ∞ nothing is truncated
# and the fixed point is exact; at finite D the state is a controlled approximation.
#
# GRAND CANONICAL, NOT U(1). The start is the product state (|0⟩ + |1⟩)^⊗N, a superposition of vertical
# flux sectors, and the fixed point stays one: its bond spectra carry no charges (only the arrow-
# reversal Z2 is exact). A zero-flux U(1) boundary PEPS has no finite-D fixed point — its virtual
# charge is the flux through the semi-infinite ribbon under the bond, whose variance grows by ~0.33 per
# bilayer (honeycomb_u1.jl). The efficient boundary state breaks U(1), and is gapless: ξ grows with D.
#
# THE NETWORKS, per cell, for ln κ by `InfiniteCTM2D`:
#   :norm ⟨ψ|ψ⟩;  :rev ⟨Rψ|ψ⟩ (R reverses every vertical arrow: exactly 1, a check);
#   :inv  ⟨Iψ|ψ⟩ — the inverted bra, its X's type-2/3 legs to +e1/+e2 and its Y's to −e1/−e2;
#   :sand ⟨Iψ|M|ψ⟩ = ⟨ψ|I M|ψ⟩.
# RQ = ln κ(:sand) − ln κ(:norm) is the Rayleigh quotient of the symmetric I M: a lower bound on
# 2 ln W(Ih) per cell (at converged χ) whatever the state; ln F_I = ln κ(:inv) − ln κ(:norm) is the
# inversion overlap, S_h − S_c ≈ −ln F_I per cell to second order (docs/ice.md).
# `network` builds them LAYERED (4 or 6 layers, raw legs [D, D] or [D, 2, D]); `paired` builds the
# same networks as 2-layer sites of 3-leg tensors on fused legs — in the inverted networks the ket's X
# and the bra's Y both point to −x/−y — the production form (values identical to 1e-14).
#
# THE CTM SEED. The default all-ones `boundary` mixes the Z2 sectors of the Vidal basis, and on the
# overlap networks the CTM then never converges (ln F_R = +0.31 against the exact 0 at D = 3): seed
# every state leg with e₁, the dominant Vidal basis vector (ones on a PEPO leg). Converges in 20–40
# iterations at every D measured.
using TensorNetworkQuantumSimulator, LinearAlgebra, Printf
const T = TensorNetworkQuantumSimulator

ice_rule(n...) = sum(n) == 2 ? 1.0 : 0.0
const VA = [ice_rule(p, e1, e2, e3) for p in 0:1, e1 in 0:1, e2 in 0:1, e3 in 0:1]                   # [p, e1, e2, e3]
const VB = [ice_rule(1 - e1, 1 - e2, 1 - e3, 1 - q) for e1 in 0:1, e2 in 0:1, e3 in 0:1, q in 0:1]   # [e1, e2, e3, q]

scaleleg(A, v, d) = A .* reshape(v, ntuple(i -> i == d ? length(v) : 1, ndims(A)))
invw(v) = [x > 1.0e-13 * maximum(v) ? 1 / x : 0.0 for x in v]

# --- BP simple update ----------------------------------------------------------------------------------

# Bond k of (X, Y) — leg k of each; X or Y may carry a trailing physical leg — truncated to ≤ D in the
# weight gauge: QR both sides, SVD of R_X w_k R_Yᵀ. Returns the new tensors, weights, discarded weight.
function su_bond(X, Y, w, k, D)
    o = [j for j in 1:3 if j != k]
    Xt, Yt = X, Y
    for j in o
        Xt = scaleleg(Xt, w[j], j); Yt = scaleleg(Yt, w[j], j)
    end
    pX = [setdiff(1:ndims(X), k); k]; pY = [setdiff(1:ndims(Y), k); k]
    QX, RX = qr(reshape(permutedims(Xt, pX), :, size(X, k)))
    QY, RY = qr(reshape(permutedims(Yt, pY), :, size(Y, k)))
    F = svd(RX * Diagonal(w[k]) * transpose(RY))
    n = min(D, count(>(1.0e-14 * F.S[1]), F.S))
    s = F.S[1:n]
    Xn = Matrix(QX) * F.U[:, 1:n]; Yn = Matrix(QY) * F.V[:, 1:n]
    sX = [size(X)...]; sX[k] = n; sY = [size(Y)...]; sY[k] = n
    Xn = permutedims(reshape(Xn, sX[pX]...), invperm(pX)); Yn = permutedims(reshape(Yn, sY[pY]...), invperm(pY))
    for j in o
        Xn = scaleleg(Xn, invw(w[j]), j); Yn = scaleleg(Yn, invw(w[j]), j)
    end
    w2 = copy(w); w2[k] = s / norm(s)
    return Xn, Yn, w2, sum(abs2, F.S[(n + 1):end]) / sum(abs2, F.S)
end

# The Vidal (= BP) gauge: identity updates bond after bond until the weights settle. With cap ≥ the
# bond dimensions nothing is truncated (only exact zeros dropped).
function regauge(X, Y, w, cap; gtol = 1.0e-12, gmax = 3000)
    for sweep in 1:gmax
        wold = w
        for k in 1:3
            X, Y, w, _ = su_bond(X, Y, w, k, cap)
        end
        Δ = length.(w) == length.(wold) ? maximum(maximum(abs.(w[k] - wold[k])) for k in 1:3) : Inf
        Δ < gtol && return X, Y, w, sweep
    end
    return X, Y, w, gmax
end

# M applied exactly: X·V_a (loses its physical leg) and Y·V_b (gains one); bond k becomes (old k, e_k),
# the old index fastest. Works for unequal bond dimensions.
function apply_bilayer(X, Y)
    d1, d2, d3 = size(Y)
    Xp = reshape(transpose(reshape(VA, 2, 8)) * reshape(X, 2, :), 2, 2, 2, d1, d2, d3)       # [e1,e2,e3,a1,a2,a3]
    Xp = reshape(permutedims(Xp, (4, 1, 5, 2, 6, 3)), 2d1, 2d2, 2d3)
    Yp = reshape(Y, d1 * d2 * d3, 1) * reshape(VB, 1, 16)
    Yp = reshape(permutedims(reshape(Yp, d1, d2, d3, 2, 2, 2, 2), (1, 4, 2, 5, 3, 6, 7)), 2d1, 2d2, 2d3, 2)
    return Xp, Yp
end

# One bilayer of BP simple update; returns (X, Y, w) with the roles swapped and the discarded weight.
function bilayer(X, Y, w, D; order = (1, 2, 3), gtol = 1.0e-12)
    Xp, Yp = apply_bilayer(X, Y)
    w2 = [vcat(w[k], w[k]) for k in 1:3]
    Xp, Yp, w2, _ = regauge(Xp, Yp, w2, typemax(Int); gtol)
    err = 0.0
    for k in order
        Xp, Yp, w2, e = su_bond(Xp, Yp, w2, k, D)
        err = max(err, e)
    end
    Xp, Yp, w2, _ = regauge(Xp, Yp, w2, D; gtol)
    Xn = permutedims(Yp, (4, 1, 2, 3)); Yn = Xp
    return Xn / norm(Xn), Yn / norm(Yn), w2, err
end

"""
    hexstate(D; steps = 1500, tol = 1e-13) -> (X, Y, w, info)

The hexagonal boundary state at bond dimension `D` by BP simple update from the product state, until
the sorted weights change by less than `tol` between even bilayers. `info = (; steps, δ, err)`.
"""
function hexstate(D; steps = 1500, tol = 1.0e-13, verbose = false)
    X = ones(2, 1, 1, 1); Y = ones(1, 1, 1); w = [ones(1) for _ in 1:3]
    wprev = nothing; δ = Inf; err = 0.0; it = 0
    for it_ in 1:steps
        it = it_
        X, Y, w, err = bilayer(X, Y, w, D; order = Tuple(circshift([1, 2, 3], it)))
        if iseven(it)                                  # compare every other bilayer (roles alternate)
            ws = sort(vcat(w...))
            if !isnothing(wprev) && length(ws) == length(wprev)
                δ = maximum(abs.(ws - wprev))
                verbose && it % 20 == 0 && @printf("  bilayer %d: Δw = %.1e, truncation %.1e\n", it, δ, err)
                δ < tol && break
            end
            wprev = ws
        end
    end
    iseven(it) || ((X, Y, w, err) = bilayer(X, Y, w, D))
    return X, Y, w, (; steps = it, δ, err)
end

# --- the networks ----------------------------------------------------------------------------------------

# √w absorbed on every leg of both tensors (the symmetric gauge the networks use)
function absorbed(X, Y, w)
    Xh, Yh = X, Y
    for k in 1:3
        Xh = scaleleg(Xh, sqrt.(w[k]), k + 1); Yh = scaleleg(Yh, sqrt.(w[k]), k)
    end
    return Xh, Yh
end

# LAYERED networks: (layers, legs (x⁻, x⁺, y⁻, y⁺)); the seed is `seed(X, kind)`.
function network(X, Y, w; kind = :norm)
    Xh, Yh = absorbed(X, Y, w)
    D = size(Xh, 2)
    p, q = T.new_index(2; tags = "p"), T.new_index(2; tags = "q")
    k1, kxm, kym, kxp, kyp = (T.new_index(D; tags = t) for t in ("k1", "kxm", "kym", "kxp", "kyp"))
    b1, bxm, bym, bxp, byp = (T.new_index(D; tags = t) for t in ("b1", "bxm", "bym", "bxp", "byp"))
    Xk = T.from_array(Xh, p, k1, kxm, kym); Yk = T.from_array(Yh, k1, kxp, kyp)
    legs2 = (T.Index[kxm, bxm], T.Index[kxp, bxp], T.Index[kym, bym], T.Index[kyp, byp])
    if kind in (:norm, :rev)
        Xb = T.from_array(kind === :rev ? Xh[[2, 1], :, :, :] : Xh, p, b1, bxm, bym); Yb = T.from_array(Yh, b1, bxp, byp)
        return Any[Xk, Yk, Xb, Yb], legs2
    end
    kind in (:inv, :sand) || throw(ArgumentError("kind must be :norm, :rev, :inv or :sand"))
    Xb = T.from_array(Xh, kind === :sand ? q : p, b1, bxp, byp); Yb = T.from_array(Yh, b1, bxm, bym)
    kind === :inv && return Any[Xk, Yk, Xb, Yb], legs2
    m1, mxm, mym, mxp, myp = (T.new_index(2; tags = t) for t in ("m1", "mxm", "mym", "mxp", "myp"))
    Va = T.from_array(VA, p, m1, mxm, mym); Vb = T.from_array(VB, m1, mxp, myp, q)
    return Any[Xk, Yk, Va, Vb, Xb, Yb],
           (T.Index[kxm, mxm, bxm], T.Index[kxp, mxp, bxp], T.Index[kym, mym, bym], T.Index[kyp, myp, byp])
end
function seed(X, kind)
    D = size(X, 2); e1 = zeros(D); e1[1] = 1
    return kind === :sand ? [e1, ones(2), e1] : [e1, e1]
end
function lnk(X, Y, w, χ; kind = :norm, tol = 1.0e-11, maxiter = 3000, init = nothing, boundary = seed(X, kind), kw...)
    site, legs = network(X, Y, w; kind)
    ic = T.update(T.InfiniteCTM2D(site, legs, χ; init, boundary, kw...); tolerance = tol, maxiter)
    return T.cvm_freenergy(ic), ic.stats[], ic
end

# PAIRED networks: a 2-layer site A[c, x⁻, y⁻], B[c, x⁺, y⁺] on fused legs (ket index fastest).
#   :norm/:rev  A = X·X̄ (over p), B = Y·Ȳ, c = (k1, b1)
#   :inv        A = X ⊗ Ȳ_inv, B = Y ⊗ X̄_inv, c = (p, k1, b1)
#   :sand       as :inv with the ket enlarged by M (bond 2D), the physical leg summed inside B
#   :mnorm      ⟨Mψ|Mψ⟩ = ⟨ψ|(I M)²|ψ⟩ (Mᵀ M = (I M)²): the :norm network of the enlarged state, whose
#               physical leg sits on the b tensor — for the eigenvector residual
#               ln f = 2 RQ(A) − RQ(A²) = 2 ln κ(:sand) − ln κ(:mnorm) − ln κ(:norm) ≤ 0 (0 iff an eigenvector)
# Returns (layers, legs, boundary): the seed e₁ ⊗ e₁ (with ones on the PEPO part). `paired_abs` takes the
# √w-absorbed tensors directly.
fuse(a, groups...) = reshape(permutedims(a, reduce(vcat, collect.(groups))), (prod(size(a, i) for i in g) for g in groups)...)
paired(X, Y, w; kind = :norm) = paired_abs(absorbed(X, Y, w)...; kind)
function paired_abs(Xh, Yh; kind = :norm)
    D = size(Xh, 2)
    if kind === :mnorm
        Kx, Ky = apply_bilayer(Xh, Yh)
        site, legs, _ = paired_abs(permutedims(Ky, (4, 1, 2, 3)), Kx; kind = :norm)
        v = zeros(2D); v[1] = 1; v[D + 1] = 1                                  # e₁ ⊗ ones(2) per enlarged leg
        return site, legs, [kron(v, v)]
    end
    e11 = [1.0; zeros(D^2 - 1)]
    if kind in (:norm, :rev)
        Xb = kind === :rev ? Xh[[2, 1], :, :, :] : Xh
        A = fuse(reshape(transpose(reshape(Xh, 2, :)) * reshape(Xb, 2, :), D, D, D, D, D, D), (1, 4), (2, 5), (3, 6))
        B = fuse(reshape(reshape(Yh, :, 1) * reshape(Yh, 1, :), D, D, D, D, D, D), (1, 4), (2, 5), (3, 6))
        sv = e11
    elseif kind === :inv
        A = fuse(reshape(reshape(Xh, :, 1) * reshape(Yh, 1, :), 2, D, D, D, D, D, D), (1, 2, 5), (3, 6), (4, 7))
        B = fuse(reshape(reshape(Yh, :, 1) * reshape(Xh, 1, :), D, D, D, 2, D, D, D), (4, 1, 5), (2, 6), (3, 7))
        sv = e11
    elseif kind === :sand
        Kx, Ky = apply_bilayer(Xh, Yh)                                        # Mψ: Kx[c1,c2,c3], Ky[c1,c2,c3,q]
        E = 2D
        A = fuse(reshape(reshape(Kx, :, 1) * reshape(Yh, 1, :), E, E, E, D, D, D), (1, 4), (2, 5), (3, 6))
        B = fuse(reshape(reshape(Ky, :, 2) * reshape(Xh, 2, :), E, E, E, D, D, D), (1, 4), (2, 5), (3, 6))
        sv = zeros(E * D); sv[1] = 1; sv[D + 1] = 1                           # (e₁ ⊗ ones(2)) ⊗ e₁
    else
        throw(ArgumentError("kind must be :norm, :rev, :inv, :sand or :mnorm"))
    end
    c = T.new_index(size(A, 1); tags = "c")
    xm, xp = T.new_index(size(A, 2); tags = "x-"), T.new_index(size(A, 2); tags = "x+")
    ym, yp = T.new_index(size(A, 3); tags = "y-"), T.new_index(size(A, 3); tags = "y+")
    return Any[T.from_array(A, c, xm, ym), T.from_array(B, c, xp, yp)], (T.Index[xm], T.Index[xp], T.Index[ym], T.Index[yp]), [sv]
end
function lnkp(X, Y, w, χ; kind = :norm, tol = 1.0e-11, maxiter = 3000, init = nothing, kw...)
    site, legs, sv = paired(X, Y, w; kind)
    ic = T.update(T.InfiniteCTM2D(site, legs, χ; init, boundary = sv, kw...); tolerance = tol, maxiter)
    return T.cvm_freenergy(ic), ic.stats[], ic
end

# --- the independent check: the same state as a cell PEPS through the library ---------------------------

# A[x⁻, x⁺, y⁻, y⁺, p]: a's type-1 bond to b contracted
function cellA(X, Y, w)
    Xh, Yh = absorbed(X, Y, w)
    D = size(Xh, 2)
    C = reshape(reshape(permutedims(Xh, (1, 3, 4, 2)), 2D^2, D) * reshape(Yh, D, D^2), 2, D, D, D, D)  # p xm ym xp yp
    return permutedims(C, (2, 4, 3, 5, 1))
end
# the library's f = ln κ⟨π(Ψ)|T|Ψ⟩ − ln κ⟨Ψ|Ψ⟩ with `ice_site`, π the inversion: equals RQ
function libf(X, Y, w, χ; tol = 1.0e-11, maxiter = 3000)
    site, legs = ice_site()
    A = cellA(X, Y, w); D = size(A, 1)
    al = (T.new_index(D; tags = "xm"), T.new_index(D; tags = "xp"), T.new_index(D; tags = "ym"),
          T.new_index(D; tags = "yp"), T.new_index(2; tags = "p"))
    bl = Tuple(T.new_index(D; tags = "b$k") for k in 1:4)
    ctx = (; al, bl, site, legs, maxdim = χ, group = ((1, 2, 3, 4),), ctm_tolerance = tol, ctm_maxiter = maxiter,
           ctm_kwargs = (;), bilinear = false, bra_perm = (2, 1, 4, 3), norm_perm = (1, 2, 3, 4))
    f, _, _, _ = T._bp_evaluate(T.from_array(A, al...), ctx, nothing, nothing)
    return real(f)
end
