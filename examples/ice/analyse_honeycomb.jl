# Collect and extrapolate the honeycomb ice results: summary.csv (RQ → w_h, ln F_I, ξ) and residual.csv
# (ln f = 2RQ(A) − RQ(A²)) from any number of run directories — a BP simple-update directory
# (honeycomb_prod.jl) or a variational evaluation directory (eval_D<D>_chi<χ>, whose su file records the
# optimiser's status) — plus the committed baseline examples/ice/results_honeycomb.csv. Rows are
# de-duplicated per (method, D, χ) (the last one wins). Then, per method, at the largest χ of each D:
# w_h = w∞ + a ξ^−p (p = 2, 3) and ln F_I = c∞ + b ξ^−p (p = 1, 2), least squares over the last 3 and
# last 4 D — the spread across these fits is the (rough) extrapolation uncertainty.
#
#   julia analyse_honeycomb.jl DIR [DIR …]        (default: ./ice_prod and ./ice_var/eval_*)
using Printf, Serialization, LinearAlgebra

const REF = [("Xu–Lin–Zhang raw D = 7, χ = 150 (w_h, Rayleigh)", 1.5074584), ("Xu–Lin–Zhang raw D = 7 (w_c)", 1.5074533),
             ("Xu–Lin–Zhang extrapolated S_h", exp(0.4104251)), ("Kolafa MC w_h ± 3.8e-6", 1.5074674),
             ("cell PEPS variational D = 3, χ = 32", 1.5074448)]

readcsv(f) = isfile(f) ? [split(strip(l), ",") for l in eachline(f) if !isempty(strip(l)) && !startswith(strip(l), "#")] : []
num(s) = (x = tryparse(Float64, strip(s)); isnothing(x) ? NaN : x)

function collect_rows(dirs)
    rows = Dict{Tuple{String, Int, Int}, Dict{Symbol, Any}}()
    row!(m, D, χ) = get!(rows, (m, D, χ), Dict{Symbol, Any}(:method => m, :D => D, :chi => χ))
    base = joinpath(@__DIR__, "results_honeycomb.csv")
    for r in readcsv(base)
        r[1] == "D" && continue
        d = row!("BP-SU", parse(Int, r[1]), parse(Int, r[2]))
        isnan(num(r[3])) || (d[:w] = num(r[3])); d[:lnF] = num(r[4]); d[:xi] = num(r[5])
    end
    for dir in dirs
        m = occursin("eval_", basename(normpath(dir))) ? "variational" : "BP-SU"
        for r in readcsv(joinpath(dir, "summary.csv"))
            d = row!(m, parse(Int, r[1]), parse(Int, r[2]))
            d[:w] = num(r[4]); d[:lnF] = num(r[5]); d[:xi] = num(r[6])
        end
        for r in readcsv(joinpath(dir, "residual.csv"))
            d = row!(m, parse(Int, r[1]), parse(Int, r[2]))
            d[:lnf] = num(r[5])
        end
        if m == "variational"
            for f in readdir(dir; join = true)
                occursin(r"su_D\d+\.jls$", f) || continue
                try
                    info = deserialize(f)[4]
                    for ((mm, D, χ), d) in rows
                        mm == m && occursin("eval_D$(D)_chi$(χ)", dir) && (d[:status] = "$(info.status), $(info.steps) its, |g| $(@sprintf("%.1e", info.δ))")
                    end
                catch
                end
            end
        end
    end
    return sort!(collect(values(rows)); by = d -> (d[:method], d[:D], d[:chi]))
end

fmtv(key, e) = key === :w ? @sprintf("%.8f", e) : @sprintf("%+.3e", e)

function fitlast(xs, ys, p, k)
    n = length(xs); n < k && return NaN
    x = xs[(n - k + 1):n] .^ (-p); y = ys[(n - k + 1):n]
    c = [ones(k) x] \ y
    return c[1]
end

function report(dirs)
    rows = collect_rows(dirs)
    @printf("%-12s %3s %4s %7s %13s %13s %13s  %s\n", "method", "D", "χ", "ξ", "w_h", "ln F_I/cell", "ln f/cell", "optimiser")
    for d in rows
        @printf("%-12s %3d %4d %7.3f %13s %13s %13s  %s\n", d[:method], d[:D], d[:chi], get(d, :xi, NaN),
                haskey(d, :w) ? @sprintf("%.10f", d[:w]) : "—", haskey(d, :lnF) ? @sprintf("%+.4e", d[:lnF]) : "—",
                haskey(d, :lnf) ? @sprintf("%+.4e", d[:lnf]) : "—", get(d, :status, ""))
    end
    println()
    for m in unique(d[:method] for d in rows)
        best = Dict{Int, Dict{Symbol, Any}}()                  # the largest χ of each D
        for d in rows
            d[:method] == m || continue
            (haskey(best, d[:D]) && best[d[:D]][:chi] > d[:chi]) || (best[d[:D]] = d)
        end
        Ds = sort(collect(keys(best)))
        for (key, ps, lab) in ((:w, (2, 3), "w_h"), (:lnF, (1, 2), "ln F_I"))
            pts = [(best[D][:xi], best[D][key]) for D in Ds if haskey(best[D], key) && !isnan(best[D][key]) && D >= 3]
            length(pts) < 3 && continue
            xs = first.(pts); ys = last.(pts)
            ests = [fitlast(xs, ys, p, k) for p in ps for k in (3, 4) if length(xs) >= k]
            @printf("%-12s %-7s from %d D (ξ %.2f–%.2f): last %s; ξ-extrapolations %s → %.8g ± %.1g\n", m, lab, length(xs),
                    minimum(xs), maximum(xs), key === :w ? @sprintf("%.10f", ys[end]) : @sprintf("%+.4e", ys[end]),
                    join([fmtv(key, e) for e in ests], ", "),
                    sum(ests) / length(ests), (maximum(ests) - minimum(ests)) / 2)
        end
    end
    println("\nreferences (w per molecule):")
    for (lab, w) in REF
        @printf("  %-50s %.7f\n", lab, w)
    end
end

dirs = isempty(ARGS) ? vcat(["ice_prod"], isdir("ice_var") ? filter(isdir, readdir("ice_var"; join = true)) : String[]) : ARGS
report(filter(isdir, dirs))
