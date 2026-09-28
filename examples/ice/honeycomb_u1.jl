# U(1) simple update for the hexagonal-ice boundary state — the experiment that shows a zero-flux
# U(1) boundary PEPS has NO finite-D fixed point (docs/ice.md, "Honeycomb formulation").
#
# The ice rules conserve flux: an in-plane PEPO bond carries 1 − 2e (units of ½) from a to b, the
# vertical leg 2p − 1. A U(1) boundary PEPS puts a charge on every virtual basis state (the flux from
# the X site to the Y site) and each tensor conserves it: X: (2p − 1) + Σ_k q_k = c_X, Y: Σ_k q_k = −c_Y;
# zero total vertical flux ⟺ c_Y = 0, and c_X = c_Y = 0 is stable (they swap every bilayer). One
# bilayer: the enlarged bond (q, e) carries q + 1 − 2e, and the role swap flips every charge. QR and SVD
# go sector by sector; the truncation never splits the ±q pair (arrow reversal) or any degenerate set.
#
# RESULT (2026-09-27): the charge is the flux through the semi-infinite vertical ribbon under the bond —
# a plain sum over layers (the frame flips, the physical orientation does not) — and it random-walks:
# ⟨q²⟩ = 0.97, 1.22, 1.55, 1.88 exactly over the first four bilayers, still +0.33 per bilayer at n = 40
# with D = 40. At fixed D the charge support widens with D (±2 at D = 6, ±8 at D = 20, ~one state per
# sector), the discarded weight falls only ~1/D and the weights never settle. The grand-canonical start
# of honeycomb.jl (a superposition of flux sectors) avoids all of it: the efficient boundary state
# breaks U(1).
#
# Run as a script: the charge variance against depth (ENV D = 40, N = 40).
isdefined(@__MODULE__, :hexstate) || include(joinpath(@__DIR__, "honeycomb.jl"))

function su_bond_u1(X, Y, w, qs, k, D; degtol = 1.0e-9)
    o = [j for j in 1:3 if j != k]
    Xt, Yt = X, Y
    for j in o
        Xt = scaleleg(Xt, w[j], j); Yt = scaleleg(Yt, w[j], j)
    end
    pX = [setdiff(1:ndims(X), k); k]; pY = [setdiff(1:ndims(Y), k); k]
    Xm = reshape(permutedims(Xt, pX), :, size(X, k)); Ym = reshape(permutedims(Yt, pY), :, size(Y, k))
    q = qs[k]
    cand = Tuple{Float64, Int, Int}[]                   # (s, sector, index within it)
    blocks = Dict{Int, Any}()
    for c in sort(unique(q))
        r = findall(==(c), q)
        FX = qr(Xm[:, r]); FY = qr(Ym[:, r])
        F = svd(FX.R * Diagonal(w[k][r]) * transpose(FY.R))
        blocks[c] = (Matrix(FX.Q) * F.U, Matrix(FY.Q) * F.V, F.S)
        append!(cand, [(F.S[i], c, i) for i in eachindex(F.S)])
    end
    sort!(cand; by = first, rev = true)
    s1 = cand[1][1]
    n = min(D, count(x -> x[1] > 1.0e-14 * s1, cand))
    while 0 < n < length(cand) && cand[n + 1][1] > (1 - degtol) * cand[n][1]      # never split a multiplet
        n -= 1
    end
    kept = cand[1:n]
    Xn = hcat([blocks[c][1][:, i] for (_, c, i) in kept]...); Yn = hcat([blocks[c][2][:, i] for (_, c, i) in kept]...)
    s = [x[1] for x in kept]; qn = [x[2] for x in kept]
    sX = [size(X)...]; sX[k] = n; sY = [size(Y)...]; sY[k] = n
    Xn = permutedims(reshape(Xn, sX[pX]...), invperm(pX)); Yn = permutedims(reshape(Yn, sY[pY]...), invperm(pY))
    for j in o
        Xn = scaleleg(Xn, invw(w[j]), j); Yn = scaleleg(Yn, invw(w[j]), j)
    end
    w2 = copy(w); w2[k] = s / norm(s); qs2 = copy(qs); qs2[k] = qn
    err = sum(abs2, [x[1] for x in cand[(n + 1):end]]; init = 0.0) / sum(abs2, [x[1] for x in cand])
    return Xn, Yn, w2, qs2, err
end

function regauge_u1(X, Y, w, qs, cap; gtol = 1.0e-12, gmax = 3000)
    for sweep in 1:gmax
        wold = w
        for k in 1:3
            X, Y, w, qs, _ = su_bond_u1(X, Y, w, qs, k, cap)
        end
        Δ = length.(w) == length.(wold) ? maximum(maximum(abs.(w[k] - wold[k])) for k in 1:3) : Inf
        Δ < gtol && return X, Y, w, qs, sweep
    end
    return X, Y, w, qs, gmax
end

function bilayer_u1(X, Y, w, qs, D; gtol = 1.0e-12)
    Xp, Yp = apply_bilayer(X, Y)
    w2 = [vcat(w[k], w[k]) for k in 1:3]
    q2 = [vcat(qs[k] .+ 1, qs[k] .- 1) for k in 1:3]                         # flux a → b: q + 1 − 2e
    Xp, Yp, w2, q2, _ = regauge_u1(Xp, Yp, w2, q2, typemax(Int); gtol)
    err = 0.0
    for k in 1:3
        Xp, Yp, w2, q2, e = su_bond_u1(Xp, Yp, w2, q2, k, D)
        err = max(err, e)
    end
    Xp, Yp, w2, q2, _ = regauge_u1(Xp, Yp, w2, q2, D; gtol)
    Xn = permutedims(Yp, (4, 1, 2, 3)); Yn = Xp
    return Xn / norm(Xn), Yn / norm(Yn), w2, [-q for q in q2], err            # the swap flips q
end

# the zero-flux start: D = 3 on charges (−1, 0, 1), X and Y the indicators of their charge rules
function u1start()
    qv = [-1, 0, 1]
    X = [Float64((2p - 1) + qv[i] + qv[j] + qv[l] == 0) for p in 0:1, i in 1:3, j in 1:3, l in 1:3]
    Y = [Float64(qv[i] + qv[j] + qv[l] == 0) for i in 1:3, j in 1:3, l in 1:3]
    return X / norm(X), Y / norm(Y), [ones(3) / sqrt(3) for _ in 1:3], [copy(qv) for _ in 1:3]
end

# relative weight of the entries violating the charge rules (X, Y)
function u1violation(X, Y, qs)
    bad = sum(abs2(X[I]) for I in CartesianIndices(X) if (2(I[1] - 1) - 1) + qs[1][I[2]] + qs[2][I[3]] + qs[3][I[4]] != 0; init = 0.0)
    badY = sum(abs2(Y[I]) for I in CartesianIndices(Y) if qs[1][I[1]] + qs[2][I[2]] + qs[3][I[3]] != 0; init = 0.0)
    return sqrt(bad) / norm(X), sqrt(badY) / norm(Y)
end

function main()
    BLAS.set_num_threads(4)
    D = parse(Int, get(ENV, "D", "40")); N = parse(Int, get(ENV, "N", "40"))
    X, Y, w, qs = u1start()
    t0 = time()
    for n in 1:N
        X, Y, w, qs, err = bilayer_u1(X, Y, w, qs, D)
        p = w[1] .^ 2 / sum(w[1] .^ 2)
        v = sum(p .* qs[1] .^ 2) - sum(p .* qs[1])^2
        @printf("n = %2d: bond dim %d, charges %+d…%+d, ⟨q²⟩ = %.3f, discarded %.1e, U(1) violation %.0e  (%.0f s)\n", n,
                length(w[1]), minimum(qs[1]), maximum(qs[1]), v, err, maximum(u1violation(X, Y, qs)), time() - t0)
        flush(stdout)
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
