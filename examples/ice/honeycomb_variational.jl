# Variational refinement of the honeycomb state (docs/ice.md, "Honeycomb formulation"): maximise
# RQ(Xh, Yh) = ln κ⟨Iψ|M|ψ⟩ − ln κ⟨ψ|ψ⟩ — the Rayleigh quotient of the symmetric I M, a lower bound on
# 2 ln W(Ih) per cell — by L-BFGS on the two unit spheres, the gradient from the CTM layer environments
# (ket + bra of X, of Y; RQ is invariant under rescaling either), both environments warm-started. Start:
# the BP simple-update state. Xh[p, 1, 2, 3], Yh[1, 2, 3] carry √w on every leg.
#
# Measured (2026-09-27): D = 2, χ = 16 from 1.5071868 (BP-SU) to 1.5073761 in 22 iterations (48 s),
# gradient check 3e-6 relative (0.14 % at χ = 8 = 2D²: finite-χ); D = 3, χ = 18 to 1.5074106 (not fully
# converged, |g| 6e-5). The cell-PEPS optima (a superset ansatz) are 1.5073953 and 1.5074448. ln F_I at
# the D = 2 optimum is −6.3e-5 against −1.4e-5 for the BP-SU state: F_I is FIRST order in the state
# error, RQ second — compare F_I only along a converging sequence of states.
#
# Run as a script: gradient check and optimisation (ENV D = 2, CHI = 16, BUDGET = 480).
isdefined(@__MODULE__, :hexstate) || include(joinpath(@__DIR__, "honeycomb.jl"))

struct HIdx
    p; q; k; b; m                                   # k, b: (1, xm, ym, xp, yp) of ket and bra; m the PEPO's
end
function hidx(D)
    mk(d, t) = Tuple(T.new_index(d; tags = "$t$s") for s in ("1", "xm", "ym", "xp", "yp"))
    return HIdx(T.new_index(2; tags = "p"), T.new_index(2; tags = "q"), mk(D, "k"), mk(D, "b"), mk(2, "m"))
end
# layers, legs, each state layer's canonical index order ((p,1,2,3) for X, (1,2,3) for Y) and positions;
# fixed indices, so environments warm-start across evaluations
function hnet(Xh, Yh, ix::HIdx; kind)
    p, q, k, b, m = ix.p, ix.q, ix.k, ix.b, ix.m
    kX = (p, k[1], k[2], k[3]); kY = (k[1], k[4], k[5])
    if kind === :norm
        bX = (p, b[1], b[2], b[3]); bY = (b[1], b[4], b[5])
    else                                            # inverted bra: X's type-2/3 legs to +e1/+e2
        bX = (kind === :sand ? q : p, b[1], b[4], b[5]); bY = (b[1], b[2], b[3])
    end
    L = Any[T.from_array(Xh, kX...), T.from_array(Yh, kY...)]
    legs = (T.Index[k[2], b[2]], T.Index[k[4], b[4]], T.Index[k[3], b[3]], T.Index[k[5], b[5]])
    if kind === :sand
        push!(L, T.from_array(VA, p, m[1], m[2], m[3]), T.from_array(VB, m[1], m[4], m[5], q))
        legs = (T.Index[k[2], m[2], b[2]], T.Index[k[4], m[4], b[4]], T.Index[k[3], m[3], b[3]], T.Index[k[5], m[5], b[5]])
    end
    push!(L, T.from_array(Xh, bX...), T.from_array(Yh, bY...))
    nl = length(L)
    return L, legs, (kX, kY, bX, bY), (1, 2, nl - 1, nl)
end
function hseed(D, kind)
    e1 = zeros(D); e1[1] = 1
    return kind === :sand ? [e1, ones(2), e1] : [e1, e1]
end
function hctm(Xh, Yh, ix, χ, kind; init = nothing, tol = 1.0e-10, maxiter = 600)
    L, legs, ids, pos = hnet(Xh, Yh, ix; kind)
    ic = T.update(T.InfiniteCTM2D(L, legs, χ; init, boundary = hseed(size(Xh, 2), kind)); tolerance = tol, maxiter)
    return ic, ids, pos
end
function hgrad(ic, ids, pos)
    g = map(1:4) do j
        E, z = T.site_environment(ic, pos[j])
        Array(T.array(E, ids[j]...)) / z
    end
    return g[1] + g[3], g[2] + g[4]                 # d ln κ / dXh, d ln κ / dYh
end
function hevaluate(Xh, Yh, ix, χ; initn = nothing, inits = nothing, tol = 1.0e-10)
    icn, idn, pn = hctm(Xh, Yh, ix, χ, :norm; init = initn, tol)
    ics, ids, ps = hctm(Xh, Yh, ix, χ, :sand; init = inits, tol)
    gXs, gYs = hgrad(ics, ids, ps); gXn, gYn = hgrad(icn, idn, pn)
    return T.cvm_freenergy(ics) - T.cvm_freenergy(icn), gXs - gXn, gYs - gYn, icn, ics
end
lnFI(Xh, Yh, ix, χ) = T.cvm_freenergy(hctm(Xh, Yh, ix, χ, :inv)[1]) - T.cvm_freenergy(hctm(Xh, Yh, ix, χ, :norm)[1])

# L-BFGS ascent on the product of spheres
function hoptimize(Xh, Yh, χ; maxiter = 400, gtol = 1.0e-8, memory = 10, max_step = 0.05, budget = Inf, verbose = true)
    ix = hidx(size(Xh, 2)); t0 = time()
    Xh = Xh / norm(Xh); Yh = Yh / norm(Yh)
    nx = length(Xh)
    pack(a, b) = vcat(vec(a), vec(b))
    unpack(v) = (reshape(v[1:nx], size(Xh)), reshape(v[(nx + 1):end], size(Yh)))
    function tang(g, x)
        gx, gy = unpack(g); xx, yy = unpack(x)
        return pack(gx - dot(xx, gx) * xx, gy - dot(yy, gy) * yy)
    end
    function renorm(v)
        a, b = unpack(v)
        return pack(a / norm(a), b / norm(b))
    end
    x = pack(Xh, Yh)
    f, gX, gY, icn, ics = hevaluate(Xh, Yh, ix, χ)
    g = tang(pack(gX, gY), x)
    hist = [f]
    verbose && @printf("  start RQ = %.12f, |g| = %.2e\n", f, norm(g))
    S = Vector{Vector{Float64}}(); Yv = Vector{Vector{Float64}}(); ρ = Float64[]
    for it in 1:maxiter
        norm(g) < gtol && break
        time() - t0 > budget && (verbose && println("  (budget)"); break)
        q = copy(g); α = zeros(length(S))                # two-loop recursion on −f
        for i in length(S):-1:1
            α[i] = ρ[i] * dot(S[i], q); q -= α[i] * Yv[i]
        end
        isempty(S) || (q *= dot(S[end], Yv[end]) / dot(Yv[end], Yv[end]))
        for i in 1:length(S)
            β = ρ[i] * dot(Yv[i], q); q += (α[i] - β) * S[i]
        end
        d = tang(q, x)
        if !(dot(d, g) > 0)
            empty!(S); empty!(Yv); empty!(ρ); d = g
        end
        a = min(1.0, max_step / norm(d)); ok = false
        local xn, fn, gXn, gYn, icnn, icsn
        for _ in 1:10
            xn = renorm(x + a * d)
            fn, gXn, gYn, icnn, icsn = hevaluate(unpack(xn)..., ix, χ; initn = icn, inits = ics)
            if fn >= f + 1.0e-4 * a * dot(d, g)
                ok = true; break
            end
            a /= 2
        end
        ok || (verbose && println("  line search failed (gradient noise floor)"); break)
        gn = tang(pack(gXn, gYn), xn)
        s = xn - x; y = -(gn - g)
        sy = dot(s, y)
        if sy > 0
            push!(S, s); push!(Yv, y); push!(ρ, 1 / sy)
            length(S) > memory && (popfirst!(S); popfirst!(Yv); popfirst!(ρ))
        end
        x, f, g, icn, ics = xn, fn, gn, icnn, icsn
        push!(hist, f)
        verbose && (it % 5 == 0 || it < 4) && @printf("  it %3d: RQ = %.12f (w_h %.9f), |g| = %.2e, step %.1e  (%.0f s)\n", it, f,
                                                     exp(f / 2), norm(g), a * norm(d), time() - t0)
        flush(stdout)
    end
    Xh, Yh = unpack(x)
    return Xh, Yh, f, norm(g), hist, ix
end

function main()
    BLAS.set_num_threads(4)
    D = parse(Int, get(ENV, "D", "2")); χ = parse(Int, get(ENV, "CHI", "16")); budget = parse(Float64, get(ENV, "BUDGET", "480"))
    X, Y, w, _ = hexstate(D)
    Xh, Yh = absorbed(X, Y, w); Xh /= norm(Xh); Yh /= norm(Yh)
    ix = hidx(D)
    f0, gX, gY, _, _ = hevaluate(Xh, Yh, ix, χ; tol = 1.0e-12)
    dX = randn(size(Xh)); dY = randn(size(Yh)); h = 1.0e-5
    fp = hevaluate(Xh + h * dX, Yh + h * dY, ix, χ; tol = 1.0e-12)[1]
    fm = hevaluate(Xh - h * dX, Yh - h * dY, ix, χ; tol = 1.0e-12)[1]
    @printf("D = %d χ = %d: BP-SU RQ = %.12f; gradient check: FD %.8e, analytic %.8e\n", D, χ, f0, (fp - fm) / 2h,
            dot(gX, dX) + dot(gY, dY))
    Xo, Yo, f, gn, hist, ix = hoptimize(Xh, Yh, χ; budget)
    @printf("D = %d χ = %d: variational RQ = %.12f (w_h %.9f; BP-SU %.9f), |g| %.1e, %d its; ln F_I = %+.4e\n", D, χ, f,
            exp(f / 2), exp(f0 / 2), gn, length(hist) - 1, lnFI(Xo, Yo, ix, χ))
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
