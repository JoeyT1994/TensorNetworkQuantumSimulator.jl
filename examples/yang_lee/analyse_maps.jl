# Yang–Lee edge maps, D = 2 (old stationary scans, ../yl3d_beta*_D2_chi16.csv) and D = 3 (Newton–Krylov
# scans, ./ylk3_beta*_D3_chi24.csv): per (β, D) the susceptibility χ (small-θ slope of Im m, θ² removed),
# the finite-D fold θ_f (m = m_f − a√(θ_f − θ) + b(θ_f − θ) through the last six points), ζ_eff =
# (B_c/χ)^{1/γ} θ_f^{−1/Δ}, and ξ near the fold. With PLOT=1: ζ_eff, the fold shift and ξ against t.
using LinearAlgebra, Printf
const γ = 1.23707; const βm = 0.326419; const δ = 4.78984; const Δ = βm * δ
const RCHI_G = 1.497; const BETAC = 0.2216544; const BC = 1.395; const OMEGANU = 0.8297 * 0.62998
const ZETA_FRG = 1.621
const PLOT = get(ENV, "PLOT", "0") == "1"
PLOT && @eval using Plots

function load(file)
    r = [split(l, ",") for l in readlines(file)[2:end]]
    r = [x for x in r if x[11] == "true"]
    β = parse(Float64, r[1][1]); D = parse(Int, r[1][2])
    θ = [parse(Float64, x[4]) for x in r]; m = [parse(Float64, x[6]) for x in r]; ξ = [parse(Float64, x[8]) for x in r]
    o = sortperm(θ); θ, m, ξ = θ[o], m[o], ξ[o]
    k = [true; diff(θ) .> 1.0e-12]
    return β, D, θ[k], m[k], ξ[k]
end
const VCUT = 1.0e-2                                   # the D = 3 scans stop near v ≈ 0.007–0.012
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
function analyse(β, D, θ, m, ξ)
    q = m ./ θ
    χH = q[1] - (q[2] - q[1]) / (θ[2]^2 - θ[1]^2) * θ[1]^2
    θf = foldfit(θ, m)
    k = (θf .- θ) ./ θf .>= VCUT                      # the same fit without the points closer than VCUT
    θfc = count(k) >= 6 && count(!, k) > 0 ? foldfit(θ[k], m[k]) : θf
    t = (BETAC - β) / BETAC
    ζ = (BC / χH)^(1 / γ) / θf^(1 / Δ)
    ξf = let i = findlast(isfinite, ξ); isnothing(i) ? NaN : ξ[i] end
    ξc = let i = findlast(j -> isfinite(ξ[j]) && k[j], eachindex(ξ)); isnothing(i) ? NaN : ξ[i] end  # at v ≥ VCUT
    return (; β, t, D, n = length(θ), χH, θf, θfc, ζ, vlast = (θf - θ[end]) / θf, ξf, ξc)
end

res = []
for f in vcat(joinpath.("..", filter(f -> occursin(r"^yl3d_beta.*_D2_chi16\.csv$", f), readdir(".."))),
              filter(f -> occursin(r"^ylk3_beta.*_D3_chi24\.csv$", f), readdir(".")))
    β, D, θ, m, ξ = load(f)
    length(θ) < 6 && (println("  (β = $β, D = $D: $(length(θ)) points, too few)"); continue)
    push!(res, analyse(β, D, θ, m, ξ))
end
sort!(res; by = r -> (r.D, r.β))
println("  β      t       D  pts   χ          θ_f          last v    ξ(last)  ζ_eff    θ_f(v ≥ $VCUT) − θ_f")
for r in res
    @printf("  %.3f  %.4f  %d  %3d   %9.4f  %.8f  %.1e   %6.3f   %.4f   %+.1e\n", r.β, r.t, r.D, r.n, r.χH, r.θf,
            r.vlast, r.ξf, r.ζ, r.θfc - r.θf)
end
println("\nfold shift D = 2 → 3:")
for r3 in filter(r -> r.D == 3, res)
    i = findfirst(r -> r.D == 2 && r.β == r3.β, res)
    isnothing(i) && continue
    r2 = res[i]
    @printf("  β = %.3f: θ_f %.8f → %.8f (%+.2e relative), ζ_eff %.4f → %.4f, ξ(v ≥ 1e-2) %.2f → %.2f\n", r3.β, r2.θf, r3.θf,
            r3.θf / r2.θf - 1, r2.ζ, r3.ζ, r2.ξc, r3.ξc)
end

if PLOT
    gr()
    p1 = plot(; xlabel = "t = 1 − β/β_c", ylabel = "ζ_eff", title = "Yang–Lee edge: ζ_eff(t)", framestyle = :box,
              legend = :bottomleft, xlims = (0, 0.3))
    hline!(p1, [ZETA_FRG]; ls = :dash, color = :black, label = "functional RG |ζ_c| = 1.621")
    p2 = plot(; xlabel = "t", ylabel = "ξ at the last point with v ≥ 1e-2", title = "correlation length near the fold", framestyle = :box,
              legend = :topright, xlims = (0, 0.3))
    for (D, mk, col) in ((2, :circle, 1), (3, :square, 2))
        rr = filter(r -> r.D == D && r.vlast < 0.02, res)        # fits within 2 % of the fold only
        isempty(rr) && continue
        scatter!(p1, [r.t for r in rr], [r.ζ for r in rr]; marker = mk, color = col, ms = 6, label = "D = $D")
        scatter!(p2, [r.t for r in rr], [r.ξc for r in rr]; marker = mk, color = col, ms = 6, label = "D = $D")
    end
    savefig(plot(p1, p2; layout = (1, 2), size = (1200, 480), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm),
            "yl_zeta_D2_D3.png")
    println("saved yl_zeta_D2_D3.png")
end
