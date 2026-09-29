# Yang–Lee edge: finite-correlation-length scaling of the finite-D folds, and the t → 0 limit.
#
# Per β the finite-D fold θ_f(D) lies below the true edge θ_c. With ξ_f(D) the boundary state's
# correlation length at the fold (here, as in docs/yang_lee.md, at the last point with v ≥ 1e-2),
#
#     θ_c − θ_f(D) ∝ ξ_f(D)^−(3 − Δ_φ),   Δ_φ ≈ 0.215 (fuzzy sphere, series, 6-loop ε)
#
# so θ_c is the intercept of θ_f against ξ_f^−2.785. Then ζ = (B_c/χ)^{1/γ} θ_c^{−1/Δ} at each β, and
# ζ_c = lim_{t→0} ζ(t) with the leading correction a t^{ων}; |z_c| = ζ_c R_χ^{1/γ}.
#
# D = 2 and 3 come from docs/yang_lee.md (their scans' CSVs are not in the repo); larger D from the
# `ylk3_beta*_D*_chi*.csv` files of `scan_krylov.jl` in the working directory (or DIR), analysed with
# the same six-point fold fit as analyse_maps.jl. Run: julia fold_xi_scaling.jl  (DIR=… optional)
using LinearAlgebra, Printf
const γ = 1.23707; const βm = 0.326419; const δ = 4.78984; const Δ = βm * δ
const RCHI_G = 1.497; const BETAC = 0.2216544; const BC = 1.395; const OMEGANU = 0.8297 * 0.62998
const P = 3 - 0.215
const ZETA_FRG, Z_FRG = 1.621, 2.43
const DIR = get(ENV, "DIR", ".")
const VMAX = parse(Float64, get(ENV, "VMAX", "0.02"))

# (β, D) => (θ_f, ξ_f at v ≥ 1e-2, χ) from docs/yang_lee.md ("Results" and "D = 3 edge maps").
# D = 3's χ equals D = 2's to 0.1–0.6 %; it is backed out of the recorded ζ_eff.
χof(ζ, θf) = BC / (ζ * θf^(1 / Δ))^γ
const RECORDED = Dict(
    (0.20, 2) => (0.0167357, 2.48, 19.283),
    (0.20, 3) => (0.0170106, 3.73, χof(1.6214, 0.0170106)),
    (0.21, 2) => (0.0059898, 2.73, 42.046),
    (0.21, 3) => (0.0063160, 4.70, χof(1.6336, 0.0063160)),
)

function load(file)
    r = [split(l, ",") for l in readlines(file)[2:end]]
    r = [x for x in r if x[11] == "true"]
    isempty(r) && return nothing
    β = parse(Float64, r[1][1]); D = parse(Int, r[1][2])
    θ = [parse(Float64, x[4]) for x in r]; m = [parse(Float64, x[6]) for x in r]; ξ = [parse(Float64, x[8]) for x in r]
    o = sortperm(θ); θ, m, ξ = θ[o], m[o], ξ[o]
    k = [true; diff(θ) .> 1.0e-12]
    return β, D, θ[k], m[k], ξ[k]
end
function foldfit(θ, m)
    n = min(6, length(θ)); θl, ml = θ[(end - n + 1):end], m[(end - n + 1):end]
    rss(θf) = θf <= θl[end] ? Inf : (M = [ones(n) sqrt.(θf .- θl) (θf .- θl)]; c = M \ ml; sum(abs2, ml - M * c))
    lo, hi = θl[end] * (1 + 1.0e-9), θl[end] * 1.2
    for _ in 1:200
        a = lo + 0.382 * (hi - lo); b = lo + 0.618 * (hi - lo)
        rss(a) < rss(b) ? (hi = b) : (lo = a)
    end
    return (lo + hi) / 2
end
ζof(χ, θ) = (BC / χ)^(1 / γ) / θ^(1 / Δ)

data = Dict{Tuple{Float64, Int}, Any}()
for (k, v) in RECORDED
    data[k] = (θf = v[1], ξ = v[2], χ = v[3], vlast = NaN, src = "docs")
end
for f in filter(f -> occursin(r"^ylk3_beta.*_D\d+_chi\d+\.csv$", f), readdir(DIR))
    x = load(joinpath(DIR, f)); isnothing(x) && continue
    β, D, θ, m, ξ = x
    length(θ) < 6 && (println("  ($f: $(length(θ)) converged points, too few for the fold fit)"); continue)
    q = m ./ θ
    χH = q[1] - (q[2] - q[1]) / (θ[2]^2 - θ[1]^2) * θ[1]^2
    θf = foldfit(θ, m)
    i = findlast(j -> (θf - θ[j]) / θf >= 1.0e-2 && isfinite(ξ[j]), eachindex(θ))
    data[(β, D)] = (θf = θf, ξ = isnothing(i) ? NaN : ξ[i], χ = χH, vlast = (θf - θ[end]) / θf, src = f)
end

println("  β      t       D   θ_f           ξ_f     χ          ζ_eff    last v    source")
for k in sort(collect(keys(data)))
    d = data[k]
    @printf("  %.3f  %.4f  %d   %.8f   %6.3f  %9.4f  %.4f   %7.1e   %s\n", k[1], (BETAC - k[1]) / BETAC, k[2], d.θf, d.ξ, d.χ,
            ζof(d.χ, d.θf), d.vlast, d.src)
end

println("\nθ_c from θ_f = θ_c − a ξ_f^−$(P) (per β):")
ests = Dict{Float64, Any}()
for β in sort(unique(first.(collect(keys(data)))))
    # a map enters only once it has come within VMAX of its fold: the six-point fit from farther out is
    # biased low (β = 0.20, D = 4 at v = 0.079 read 0.01608, below the D = 2 fold)
    Ds = sort([k[2] for k in keys(data) if k[1] == β && isfinite(data[k].ξ) && !(data[k].vlast > VMAX)])
    length(Ds) < 2 && continue
    x = [data[(β, D)].ξ^(-P) for D in Ds]; y = [data[(β, D)].θf for D in Ds]
    fit(idx) = (M = [ones(length(idx)) -x[idx]]; c = M \ y[idx]; c[1])
    θc_all = fit(1:length(Ds))
    θc_top = fit((length(Ds) - 1):length(Ds))           # the two largest D only
    χbest = data[(β, Ds[end])].χ
    ests[β] = (θc_all, θc_top, χbest, Ds)
    @printf("  β = %.3f (t = %.4f), D = %s: θ_c = %.8f (all D), %.8f (two largest D); largest-D fold %.8f; ζ(θ_c) = %.4f, %.4f\n",
            β, (BETAC - β) / BETAC, join(Ds, ","), θc_all, θc_top, y[end], ζof(χbest, θc_all), ζof(χbest, θc_top))
end

bs = sort(collect(keys(ests)))
if length(bs) >= 2
    println("\nt → 0 (ζ = ζ_c + a t^{ων}, ων = $(round(OMEGANU; digits = 4)), through the two β):")
    for (label, j) in (("all D", 1), ("two largest D", 2))
        t = [(BETAC - β) / BETAC for β in bs]; ζ = [ζof(ests[β][3], ests[β][j]) for β in bs]
        M = [ones(length(bs)) t .^ OMEGANU]; c = M \ ζ
        @printf("  %-14s ζ(t) = %s  →  ζ_c = %.4f, |z_c| = %.3f   (FRG %.3f, %.2f)\n", label,
                join([@sprintf("%.4f", z) for z in ζ], ", "), c[1], c[1] * RCHI_G, ZETA_FRG, Z_FRG)
    end
end
