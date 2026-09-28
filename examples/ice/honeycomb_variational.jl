# Variational refinement of the honeycomb ice state (docs/ice.md, "Honeycomb formulation"): maximise
# RQ(Xh, Yh) = ln κ⟨Iψ|M|ψ⟩ − ln κ⟨ψ|ψ⟩ — the Rayleigh quotient of the symmetric I M, a lower bound on
# 2 ln W(Ih) per cell at converged χ — by L-BFGS on the two unit spheres (RQ is invariant under rescaling
# X or Y), the gradient from the CTM layer environments (ket + bra of X, of Y), every environment
# warm-started from the previous evaluation. Start: the BP simple-update state. Xh[p, 1, 2, 3], Yh[1, 2, 3]
# carry √w on every leg. LAYERED networks (each tensor its own layer, for its own gradient).
#
# RESUMABLE AND DEVICE-AGNOSTIC. After every accepted step an atomic checkpoint
# OUT/var_D<D>_chi<χ>.jls holds the state, the L-BFGS memory and host copies of both environments; a
# rerun resumes. The CTM converges on ln κ (`convergence = :lnkappa`, tolerance TOL relative) with the
# subspace oversampling at ~1.3χ (docs/status_3d.md "GPU readiness"), then POLISH more iterations so the
# environment gradients are accurate (ln κ is stationary; the environment lags). STOP at the gradient's
# noise floor: |g| < GTOL, or a line search that fails again right after a memory reset and a cold
# re-evaluation, or an RQ gain below FTOL3 (3e-9) over 3 accepted steps or FTOL over 10. Then the optimum is written as OUT/eval_D<D>_chi<χ>/su_D<D>.jls
# (X = Xh, Y = Yh, w = ones) for `honeycomb_prod.jl` (KINDS=norm,inv,sand,mnorm, same OUT) to evaluate
# w_h, ln F_I, ξ and the eigenvector residual resumably.
#
# ENV: D (3), CHI (2D²), DEVICE (cpu | gpu), OUT ("ice_var"), BUDGET (s, 540), SUDIR (a directory with the
# BP-SU su_D<D>.jls to start from; else built), GTOL (1e-8), FTOL (1e-11), MAXIT (500), TOL (2e-14),
# MAXSTEP (0.01: at D = 4 a 0.05 step left the CTM unconverged after 400 iterations; 0.0125 took 24–31),
# POLISH (3), GRADCHECK (0 | 1: a finite-difference check at the start), BLASN (4).
#
# Measured (CPU, 2026-09-27, the pre-resumable version): D = 2, χ = 16 from 1.5071868 (BP-SU) to 1.5073761
# in 22 iterations; D = 3, χ = 18 to 1.5074106 (|g| 6e-5, not converged). ln F_I at the D = 2 optimum
# −6.3e-5 against −1.4e-5 for BP-SU: F_I is FIRST order in the state error, RQ second.
isdefined(@__MODULE__, :hexstate) || include(joinpath(@__DIR__, "honeycomb.jl"))
using Serialization, Adapt
using Logging: NullLogger, with_logger
const VDEVICE = get(ENV, "DEVICE", "cpu")
if VDEVICE == "gpu"
    @eval using CUDA
    CUDA.allowscalar(false)
end
vdev(a) = VDEVICE == "gpu" ? adapt(CUDA.CuArray, a) : a
vhost(x) = VDEVICE == "gpu" ? adapt(Array, x) : x
vquiet(f) = with_logger(f, NullLogger())
vatomic(f, x) = (serialize(f * ".tmp", x); mv(f * ".tmp", f; force = true))

struct HIdx
    p; q; k; b; m                                   # k, b: (1, xm, ym, xp, yp) of ket and bra; m the PEPO's
end
function hidx(D)
    mk(d, t) = Tuple(T.new_index(d; tags = "$t$s") for s in ("1", "xm", "ym", "xp", "yp"))
    return HIdx(T.new_index(2; tags = "p"), T.new_index(2; tags = "q"), mk(D, "k"), mk(D, "b"), mk(2, "m"))
end
# layers (on the device), legs, each state layer's canonical index order ((p,1,2,3) for X, (1,2,3) for Y)
# and positions in the layer list
function hnet(Xh, Yh, ix::HIdx; kind)
    p, q, k, b, m = ix.p, ix.q, ix.k, ix.b, ix.m
    kX = (p, k[1], k[2], k[3]); kY = (k[1], k[4], k[5])
    if kind === :norm
        bX = (p, b[1], b[2], b[3]); bY = (b[1], b[4], b[5])
    else                                            # inverted bra: X's type-2/3 legs to +e1/+e2
        bX = (kind === :sand ? q : p, b[1], b[4], b[5]); bY = (b[1], b[2], b[3])
    end
    L = Any[T.from_array(vdev(Xh), kX...), T.from_array(vdev(Yh), kY...)]
    legs = (T.Index[k[2], b[2]], T.Index[k[4], b[4]], T.Index[k[3], b[3]], T.Index[k[5], b[5]])
    if kind === :sand
        push!(L, T.from_array(vdev(VA), p, m[1], m[2], m[3]), T.from_array(vdev(VB), m[1], m[4], m[5], q))
        legs = (T.Index[k[2], m[2], b[2]], T.Index[k[4], m[4], b[4]], T.Index[k[3], m[3], b[3]], T.Index[k[5], m[5], b[5]])
    end
    push!(L, T.from_array(vdev(Xh), bX...), T.from_array(vdev(Yh), bY...))
    nl = length(L)
    return L, legs, (kX, kY, bX, bY), (1, 2, nl - 1, nl)
end
function hseed(D, kind)
    e1 = zeros(D); e1[1] = 1
    return kind === :sand ? [e1, ones(2), e1] : [e1, e1]
end
Base.@kwdef struct VCfg
    χ::Int
    tol::Float64 = 2.0e-14
    polish::Int = 3
    maxiter::Int = 100                              # per evaluation: a trial the CTM cannot settle is rejected, not waited for
    precondition::Bool = true
    oversample::Int = 0                             # 0: ceil(1.3χ)
end
function hctm(Xh, Yh, ix, cfg::VCfg, kind; init = nothing)
    L, legs, ids, pos = hnet(Xh, Yh, ix; kind)
    opts = (; boundary = hseed(size(Xh, 2), kind), svd_oversample = cfg.oversample > 0 ? cfg.oversample : max(16, ceil(Int, 1.3 * cfg.χ)))
    ic = vquiet(() -> T.update(T.InfiniteCTM2D(L, legs, cfg.χ; init, opts...); tolerance = cfg.tol, maxiter = cfg.maxiter,
                               convergence = :lnkappa, miniter = isnothing(init) ? 15 : 3))
    n = ic.stats[].iterations
    if cfg.polish > 0                               # the environment lags ln κ: polish it for the gradient
        ic = vquiet(() -> T.update(T.InfiniteCTM2D(L, legs, cfg.χ; init = ic, opts...); tolerance = 0.0,
                                   maxiter = cfg.polish, miniter = cfg.polish, convergence = :lnkappa))
        n += cfg.polish
    end
    return ic, ids, pos, n
end
function hgrad(ic, ids, pos)
    g = map(1:4) do j
        E, z = T.site_environment(ic, pos[j])
        Array(T.array(E, ids[j]...)) / z
    end
    return g[1] + g[3], g[2] + g[4]                 # d ln κ / dXh, d ln κ / dYh
end
function hevaluate(Xh, Yh, ix, cfg; initn = nothing, inits = nothing, metric = false, τ = 1.0e-2)
    icn, idn, pn, nn = hctm(Xh, Yh, ix, cfg, :norm; init = initn)
    ics, ids, ps, ns = hctm(Xh, Yh, ix, cfg, :sand; init = inits)
    gXs, gYs = hgrad(ics, ids, ps); gXn, gYn = hgrad(icn, idn, pn)
    P = metric ? hmetric(icn, idn, pn, τ) : nothing
    return T.cvm_freenergy(ics) - T.cvm_freenergy(icn), gXs - gXn, gYs - gYn, icn, ics, nn + ns, P
end

# THE NORM-METRIC PRECONDITIONER (as `boundary_peps`'s): the ⟨ψ|ψ⟩ environment of a tensor's ket and bra
# layers together is the Gram matrix of ∂ψ/∂X — for X on its virtual legs (the physical leg is shared
# inside the pair), for Y on all three. Steps in (N + τλ_max)⁻¹ g; L-BFGS's initial inverse Hessian.
function hmetric(ic, ids, pos, τ)
    VX, tbl, env = T._i2_shell(ic)
    placed = tbl[T._I2_V]
    old, new = T._i2_site_relabelling(ic.legs, T._I2_V, VX)
    return map(((1, 3), (2, 4))) do (a, b)
        others = [placed[j] for j in eachindex(placed) if j != pos[a] && j != pos[b]]
        E = T._i2_relabel(T._ctm_contract(vcat(T.AbstractTensor[env], T.AbstractTensor[others...]), ic.options), new, old)
        ka = a == 1 ? ids[1][2:4] : ids[2]; kb = a == 1 ? ids[3][2:4] : ids[4]
        M = Array(T.array(E, ka..., kb...)); n = prod(size(M)[1:length(ka)])
        M = reshape(M, n, n); M = (M + M') / 2
        λ = eigmax(Hermitian(M))
        inv(Hermitian(M + τ * λ * I))
    end
end

# L-BFGS ascent on the product of spheres, resumable. `S` is the checkpoint state (a NamedTuple).
function hoptimize!(ck, S; cfg, budget, gtol, ftol, maxit, max_step, memory = 10, ftol3 = 3.0e-9)
    t0 = time()
    Xh, Yh = S.Xh, S.Yh
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
    function precond(v, P)                         # (N + τλ)⁻¹ per tensor; `nothing`: the identity
        isnothing(P) && return v
        vx, vy = unpack(v)
        wx = reshape(reshape(vx, size(vx, 1), :) * P[1], size(vx))   # X: on its virtual legs (P symmetric)
        wy = reshape(P[2] * vec(vy), size(vy))
        return pack(wx, wy)
    end
    ix = hidx(size(Xh, 2))
    x = pack(Xh, Yh); f = S.f; g = S.g; it = S.it; hist = S.hist; Sm = S.Sm; Ym = S.Ym; ρ = S.ρ
    P = get(S, :P, nothing); usemetric = cfg.precondition
    icn = isnothing(S.icn) ? nothing : vdev(S.icn); ics = isnothing(S.ics) ? nothing : vdev(S.ics)
    nfail = S.nfail; status = S.status; lastdur = S.lastdur
    evaldur = get(S, :evalwarm, 0.0); nrun = 0               # WARM evaluation time; evaluations this run
    lasttrials = 2                                 # evaluations the last line search needed
    function timed!(t)                             # the first evaluation of a run compiles: never the estimate
        nrun += 1
        if nrun > 1
            evaldur = t
        elseif evaldur == 0
            evaldur = 0.3 * t
        end
    end
    save() = vatomic(ck, (; Xh = unpack(x)[1], Yh = unpack(x)[2], f, g, it, hist, Sm, Ym, ρ, icn = vhost(icn),
                          ics = vhost(ics), nfail, status, lastdur, χ = cfg.χ, P, evalwarm = evaldur))
    if isnan(f)                                     # the first evaluation (BP-SU state)
        te = time()
        f, gX, gY, icn, ics, _, P = hevaluate(unpack(x)..., ix, cfg; metric = usemetric)
        timed!(time() - te)
        g = tang(pack(gX, gY), x); push!(hist, f)
        @printf("  start RQ = %.12f (w_h %.10f), |g| = %.2e  (%.0f s)\n", f, exp(f / 2), norm(g), time() - t0)
        flush(stdout)
        save()
    end
    while status === :running
        norm(g) < gtol && (status = :gtol; break)
        it >= maxit && (status = :maxit; break)
        if (length(hist) > 10 && hist[end] - hist[end - 10] < ftol) || (length(hist) > 3 && hist[end] - hist[end - 3] < ftol3)
            status = :stagnant; break                # RQ no longer moves: the gradient's noise floor
        end
        time() - t0 + 1.1 * max(2, lasttrials) * evaldur > budget && break   # the last search's trials, warm
        ts = time()
        q = copy(g); α = zeros(length(Sm))           # two-loop recursion on −f
        for i in length(Sm):-1:1
            α[i] = ρ[i] * dot(Sm[i], q); q -= α[i] * Ym[i]
        end
        q = precond(q, P)                            # H₀ = γ (N + τλ)⁻¹
        isempty(Sm) || (q *= dot(Sm[end], Ym[end]) / dot(Ym[end], precond(Ym[end], P)))
        for i in 1:length(Sm)
            β = ρ[i] * dot(Ym[i], q); q += (α[i] - β) * Sm[i]
        end
        d = tang(q, x)
        if !(dot(d, g) > 0)
            empty!(Sm); empty!(Ym); empty!(ρ); d = tang(precond(g, P), x)
            dot(d, g) > 0 || (d = g)
        end
        a = min(1.0, max_step / norm(d)); ok = false; nct = 0; aborted = false
        local xn, fn, gXn, gYn, icnn, icsn, Pn
        for trial in 1:8
            if trial > 1 && time() - t0 + 1.2 * evaldur > budget   # the next trial would not fit
                aborted = true; break
            end
            xn = renorm(x + a * d)
            te = time()
            fn, gXn, gYn, icnn, icsn, nc, Pn = hevaluate(unpack(xn)..., ix, cfg; initn = icn, inits = ics, metric = usemetric)
            timed!(time() - te); save()               # the timing survives a kill mid-search
            lasttrials = trial
            nct += nc
            if fn >= f + 1.0e-4 * a * dot(d, g)
                ok = true; break
            end
            a /= 2
        end
        if aborted                                   # out of budget inside the line search: resume later
            save(); break
        end
        if !ok                                       # consecutive failures: reset → cold restart → the floor
            nfail += 1
            if nfail == 1 && !isempty(Sm)
                empty!(Sm); empty!(Ym); empty!(ρ)    # the memory, then preconditioned steepest ascent
            elseif nfail <= 2                        # re-evaluate here from COLD environments (a warm start
                f, gX, gY, icn, ics, _, P = hevaluate(unpack(x)..., ix, cfg; metric = usemetric)   # may drift)
                g = tang(pack(gX, gY), x); empty!(Sm); empty!(Ym); empty!(ρ)
                @printf("  line search failed twice: cold re-evaluation, RQ = %.12f, |g| = %.2e\n", f, norm(g))
            else
                status = :noise_floor                # the gradient's noise floor
            end
            save(); continue
        end
        nfail = 0
        gn = tang(pack(gXn, gYn), xn)
        s = xn - x; y = -(gn - g)
        sy = dot(s, y)
        if sy > 0
            push!(Sm, s); push!(Ym, y); push!(ρ, 1 / sy)
            length(Sm) > memory && (popfirst!(Sm); popfirst!(Ym); popfirst!(ρ))
        end
        x, f, g, icn, ics, P = xn, fn, gn, icnn, icsn, Pn
        it += 1; push!(hist, f); lastdur = time() - ts
        @printf("  it %3d: RQ = %.12f (w_h %.10f), |g| = %.2e, step %.1e, %d CTM its  (%.0f s)\n", it, f, exp(f / 2),
                norm(g), a * norm(d), nct, time() - t0)
        flush(stdout)
        save()
    end
    save()
    return status, unpack(x)..., f, norm(g), it
end

function main()
    BLAS.set_num_threads(parse(Int, get(ENV, "BLASN", "4")))
    D = parse(Int, get(ENV, "D", "3")); χ = parse(Int, get(ENV, "CHI", string(2D^2)))
    out = get(ENV, "OUT", "ice_var"); mkpath(out)
    budget = parse(Float64, get(ENV, "BUDGET", "540")) - parse(Float64, get(ENV, "STARTUP_S", "100"))   # Julia's own load time
    cfg = VCfg(; χ, tol = parse(Float64, get(ENV, "TOL", "2e-14")), polish = parse(Int, get(ENV, "POLISH", "3")),
               precondition = get(ENV, "PRECONDITION", "1") == "1", oversample = parse(Int, get(ENV, "OVERSAMPLE", "0")))
    ck = joinpath(out, "var_D$(D)_chi$(χ).jls")
    evaldir = joinpath(out, "eval_D$(D)_chi$(χ)"); evalsu = joinpath(evaldir, "su_D$(D).jls")
    if isfile(ck)
        S = deserialize(ck)
        @printf("D = %d χ = %d: resuming at iteration %d (RQ %.12f, status %s)\n", D, χ, S.it, S.f, S.status)
    else
        f = joinpath(get(ENV, "SUDIR", ""), "su_D$(D).jls")
        X, Y, w = isfile(f) ? deserialize(f)[1:3] : hexstate(D)[1:3]
        Xh, Yh = absorbed(X, Y, w); Xh /= norm(Xh); Yh /= norm(Yh)
        S = (; Xh, Yh, f = NaN, g = Float64[], it = 0, hist = Float64[], Sm = Vector{Vector{Float64}}(),
             Ym = Vector{Vector{Float64}}(), ρ = Float64[], icn = nothing, ics = nothing, nfail = 0, status = :running,
             lastdur = 0.0, χ)
        @printf("D = %d χ = %d on %s: starting from the BP simple-update state\n", D, χ, VDEVICE)
        if get(ENV, "GRADCHECK", "0") == "1"
            ix = hidx(D); c2 = VCfg(; χ, tol = 1.0e-15, polish = parse(Int, get(ENV, "GC_POLISH", "10")), oversample = cfg.oversample)
            r0 = hevaluate(Xh, Yh, ix, c2); f0, gX, gY = r0[1], r0[2], r0[3]
            dX = randn(size(Xh)); dY = randn(size(Yh)); h = 1.0e-5
            fp = hevaluate(Xh + h * dX, Yh + h * dY, ix, c2)[1]; fm = hevaluate(Xh - h * dX, Yh - h * dY, ix, c2)[1]
            @printf("  gradient check: FD %.10e, analytic %.10e (relative %.1e)\n", (fp - fm) / 2h, dot(gX, dX) + dot(gY, dY),
                    abs((fp - fm) / 2h - dot(gX, dX) - dot(gY, dY)) / abs((fp - fm) / 2h))
        end
    end
    if S.status === :running
        status, Xo, Yo, f, gn, it = hoptimize!(ck, S; cfg, budget, gtol = parse(Float64, get(ENV, "GTOL", "1e-8")),
                                               ftol = parse(Float64, get(ENV, "FTOL", "1e-11")),
                                               ftol3 = parse(Float64, get(ENV, "FTOL3", "3e-9")),
                                               maxit = parse(Int, get(ENV, "MAXIT", "500")),
                                               max_step = parse(Float64, get(ENV, "MAXSTEP", "0.01")))
    else
        status, Xo, Yo, f, gn, it = S.status, S.Xh, S.Yh, S.f, norm(S.g), S.it
    end
    if status === :running
        @printf("  (budget: iteration %d, RQ %.12f, |g| %.2e — resume to continue)\n", it, f, gn)
    else
        @printf("DONE D = %d χ = %d: %s after %d iterations — RQ = %.12f, w_h = %.10f, |g| = %.2e\n", D, χ, status, it, f,
                exp(f / 2), gn)
        if !isfile(evalsu)
            mkpath(evaldir)
            vatomic(evalsu, (Xo, Yo, [ones(size(Yo, k)) for k in 1:3], (; steps = it, δ = gn, err = 0.0, status)))
            println("  optimum written to $evalsu — evaluate with honeycomb_prod.jl (OUT=$evaldir, KINDS=norm,inv,sand,mnorm)")
        end
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
