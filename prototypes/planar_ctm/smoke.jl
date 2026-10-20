# Quick check of the prototype (a few minutes): certified rule design + sequential passes,
# |F − ln Z| against exact contraction. Run with the repository project:
#   julia --project=. prototypes/planar_ctm/smoke.jl
include(joinpath(@__DIR__, "dagtools.jl"))
include(joinpath(@__DIR__, "planar_proto2.jl"))
Base.include(PCTM, joinpath(@__DIR__, "planar_seq.jl"))
using Random
pdisc(g, r) = TNQS.rem_vertices(g, [v for v in vertices(g) if hypot(v...) > r])
postn(g; D = 2) = (tn = random_tensornetwork(Float64, g; bond_dimension = D); for v in vertices(g); tn[v] = abs.(tn[v]); end; tn)
PCTM.PINV_RTOL[] = 1.0e-3
for (label, g, χs) in [("Penrose disc", pdisc(penrose_tiling(3), 0.45), (4, 1000)), ("kagome 3x3", kagome_lattice(3, 3), (4, 1000)),
                       ("Penrose 3", penrose_tiling(3), (8, 16))]
    d = design_minimax(g)
    emb = planar_embedding(g); ex = d.exitrule; exitf = (f, h, wh) -> ex((emb = emb,), f, h, wh)
    lv = isfinite(d.depth) ? d.levels : nothing
    Random.seed!(3); tn = postn(g); lnZ = log(abs(contract(tn; alg = "exact")))
    for χ in χs
        Random.seed!(11)
        E = PCTM.init(tn, emb, exitf, d.choice; maxdim = χ); S = PCTM.Seq(E; levels = lv)
        for _ in 1:12; PCTM.pass!(S); end
        println(rpad(label, 14), " exit ", rpad(d.exit, 9), " depth ", d.depth, "  χ=", lpad(χ, 4), "  |F − lnZ| = ", round(abs(PCTM.kikuchi(E) - lnZ); sigdigits = 2))
    end
end
