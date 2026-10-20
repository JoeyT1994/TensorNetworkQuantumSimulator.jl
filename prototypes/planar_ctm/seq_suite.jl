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
for rt in (1.0e-3, 1.0e-4)
    PCTM.PINV_RTOL[] = rt
    println("== pinv rtol ", rt)
    seqf("square 6x6", named_grid((6, 6)), (4, 8, 1000))
    seqf("honeycomb 4x4", named_hexagonal_lattice_graph(4, 4), (4, 8, 1000))
    seqf("triangular 5x5", triangular_lattice(5, 5), (4, 8, 1000))
    seqf("kagome 4x4 trim", trim2(kagome_lattice(4, 4)), (4, 8, 1000))
    seqf("Penrose 3", penrose_tiling(3), (4, 8, 16, 32, 1000))
end
