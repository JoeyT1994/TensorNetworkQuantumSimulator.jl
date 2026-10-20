include(joinpath(@__DIR__, "dagtools.jl"))
include(joinpath(@__DIR__, "planar_proto2.jl"))
Base.include(PCTM, joinpath(@__DIR__, "planar_seq.jl"))
using Random
pdisc(g, r) = TNQS.rem_vertices(g, [v for v in vertices(g) if hypot(v...) > r])
trim2(g) = (vs = [v for v in vertices(g) if length(neighbors(g, v)) <= 2]; isempty(vs) ? g : TNQS.rem_vertices(g, vs))
postn(g; D = 2) = (tn = random_tensornetwork(Float64, g; bond_dimension = D); for v in vertices(g); tn[v] = abs.(tn[v]); end; tn)
function seq(label, g, χs; npass = 10, tnf = postn, proj = :biorth)
    d = design_minimax(g)
    emb = planar_embedding(g); ex = d.exitrule; exitf = (f, h, wh) -> ex((emb = emb,), f, h, wh)
    lv = isfinite(d.depth) ? d.levels : nothing
    Random.seed!(3); tn = tnf(g); lnZ = log(abs(contract(tn; alg = "exact")))
    for χ in χs
        Random.seed!(11)
        E = PCTM.init(tn, emb, exitf, d.choice; maxdim = χ)
        S = PCTM.Seq(E; levels = lv, proj)
        errs = Float64[]
        t = @elapsed for _ in 1:npass; PCTM.pass!(S); push!(errs, abs(PCTM.kikuchi(E) - lnZ)); end
        println(rpad(label, 14), " ", rpad(d.exit, 9), " χ=", lpad(χ, 4), "  |F − lnZ| per pass: ", join(string.(round.(errs; sigdigits = 2)), " "), "  (", round(t; digits = 1), " s)")
    end
end

function seqf(label, g, χs; npass = 12, tnf = postn)
    d = design_minimax(g)
    emb = planar_embedding(g); ex = d.exitrule; exitf = (f, h, wh) -> ex((emb = emb,), f, h, wh)
    lv = isfinite(d.depth) ? d.levels : nothing
    Random.seed!(3); tn = tnf(g); lnZ = log(abs(contract(tn; alg = "exact")))
    out = String[]
    for χ in χs
        Random.seed!(11)
        E = PCTM.init(tn, emb, exitf, d.choice; maxdim = χ); S = PCTM.Seq(E; levels = lv)
        errs = Float64[]
        for _ in 1:npass; PCTM.pass!(S); push!(errs, abs(PCTM.kikuchi(E) - lnZ)); end
        push!(out, "χ=$χ: " * string(round(errs[end]; sigdigits = 2)) * " (spread " * string(round(maximum(errs[(end - 2):end]); sigdigits = 1)) * ")")
    end
    println(rpad(label, 16), " ", rpad(d.exit, 9), join(out, "   "))
end
PCTM.PINV_RTOL[] = 1.0e-3
for (l, g, χs) in [("kagome 6x6", kagome_lattice(6, 6), (4, 8, 16)), ("Penrose 4", penrose_tiling(4), (8, 16))]
    t = @elapsed seqf(l, g, χs; npass = 14)
    println("   total ", round(t; digits = 1), " s")
end
