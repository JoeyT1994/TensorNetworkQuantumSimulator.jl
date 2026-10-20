# Static certification of a planar CTM rule set.
#   1. every block's TRUE signed region, from its cuts by winding numbers (no iteration);
#   2. every candidate update identity checked against those regions;
#   3. per block, a valid candidate whose parts are all strictly smaller (Σ|multiplicity|), so the
#      dependency graph is a DAG and the lossless sweep is exact after finitely many sweeps.
include(joinpath(@__DIR__, "rcg.jl"))

# the ray from face f across its edge (p → q) (f left of p → q): crossed edges, each oriented with
# the face it leaves on its left
function ray(P, f, p, q)
    out = Tuple{Any, Any}[]
    while true
        push!(out, (p, q))
        h = TNQS.face_of_dart(P.emb, q, p)
        isinner(h) || return out
        wh = rot_to(walk(P, h), p); n = length(wh)
        j = P.exit(P, f, h, wh)
        p, q, f = wh[j + 1], wh[mod1(j + 2, n)], h
        length(out) > 10 * TNQS.ne(P.emb.graph) && error("ray does not reach the outer face")
    end
end
# winding numbers of the closed curve [forward ray A] ∪ [backward ray B] (∪ the dual edge `bridge`)
# around the vertices, normalised to w(tip) = 1. Forward crossing of (x → y): w(y) − w(x) = +1.
function winding(P, tip, fwd, bwd, bridge)
    g = P.emb.graph
    δ = Dict{Tuple{Any, Any}, Int}()
    addδ(x, y, s) = (δ[(x, y)] = get(δ, (x, y), 0) + s; δ[(y, x)] = get(δ, (y, x), 0) - s)
    for (x, y) in fwd; addδ(x, y, 1); end
    for (x, y) in bwd; addδ(x, y, -1); end
    isnothing(bridge) || addδ(bridge..., 1)
    w = Dict{Any, Int}(tip => 1); q = Any[tip]
    while !isempty(q)
        x = popfirst!(q)
        for y in neighbors(g, x)
            wy = w[x] + get(δ, (x, y), 0)
            if haskey(w, y)
                w[y] == wy || error("inconsistent winding at $(x)–$(y)")
            else
                w[y] = wy; push!(q, y)
            end
        end
    end
    return MS(v => c for (v, c) in w if c != 0)
end
# C(p, f): forward along γ(f, (a → p)), backward along γ(f, (p → b)), a → p → b in f's walk
function true_C(P, p, f)
    w = rot_to(walk(P, f), p)
    return winding(P, p, ray(P, f, w[end], p), ray(P, f, p, w[2]), nothing)
end
# T(u → v): forward along γ(fA, (u_A → u)), the dual edge from fB to fA (w(u) − w(v) = +1),
# backward along γ(fB, (u → u_B)); a side on the outer face contributes no ray
function true_T(P, u, v)
    poly = vertex_polygon(P.emb, u); nb = first.(poly); z = length(nb); i = findfirst(==(v), nb)
    fA, fB = poly[i][2], poly[mod1(i - 1, z)][2]
    fwd = isinner(fA) ? ray(P, fA, nb[mod1(i + 1, z)], u) : Tuple{Any, Any}[]
    bwd = isinner(fB) ? ray(P, fB, u, nb[mod1(i - 1, z)]) : Tuple{Any, Any}[]
    return winding(P, u, fwd, bwd, (v, u))
end
Φ(r) = sum(abs, values(r); init = 0)

# symbolic parts of a candidate update: evaluate it on block-name singletons
function parts(P, upd)
    Tsave, Csave = P.T, P.C
    P.T = Dict{Any, MS}(k => MS(("T", k) => 1) for k in keys(Tsave)); P.C = Dict{Any, MS}(k => MS(("C", k) => 1) for k in keys(Csave))
    DEPMODE[] = true
    r = try upd(P) finally DEPMODE[] = false end
    P.T, P.C = Tsave, Csave
    return [(a, c) for (a, c) in r if a isa Tuple && length(a) == 2 && a[1] in ("T", "C")]
end
blockregion(P, a) = a[1] == "T" ? P.T[a[2]] : P.C[a[2]]

function certify(label, g; exitrule = straight, corner_candidates = (:diag, :prev, :next), verbose = false)
    P = newrc(g, exitrule)
    try
        for k in keys(P.T); P.T[k] = true_T(P, k...); end
        for k in keys(P.C); P.C[k] = true_C(P, k...); end
    catch err
        println(rpad(label, 22), "REGION ERROR: ", sprint(showerror, err)[1:min(end, 120)]); return false
    end
    full = MS(v => 1 for v in vertices(g))
    zok = all(v -> Zv(P, v) == full, vertices(g)) && all(e -> Ze(P, TNQS.src(e), TNQS.dst(e)) == full, TNQS.edges(g)) &&
        all(f -> Zf(P, f) == full, 1:length(TNQS.faces(P.emb)))
    cand(m) = m === :diag ? (P -> newC_diag(P, PCUR[]...)) : (P -> newC_dir(P, PCUR[]..., m))
    nTbad = 0; nTgrow = 0; nCnone = 0; choice = Dict{Any, Symbol}(); counts = Dict{Symbol, Int}()
    for (u, v) in keys(P.T)
        newT(P, u, v) == P.T[(u, v)] || (nTbad += 1; continue)
        all(((a, c),) -> Φ(blockregion(P, a)) < Φ(P.T[(u, v)]), parts(P, P -> newT(P, u, v))) || (nTgrow += 1)
    end
    for (p, f) in keys(P.C)
        PCUR[] = (p, f); found = false
        for m in corner_candidates
            upd = cand(m)
            upd(P) == P.C[(p, f)] || continue
            all(((a, c),) -> Φ(blockregion(P, a)) < Φ(P.C[(p, f)]), parts(P, upd)) || continue
            choice[(p, f)] = m; counts[m] = get(counts, m, 0) + 1; found = true; break
        end
        found || (nCnone += 1)
    end
    ok = zok && nTbad == 0 && nTgrow == 0 && nCnone == 0
    println(rpad(label, 22), ok ? "CERTIFIED" : "not certified", "  Z identities ", zok ? "ok" : "BAD",
        "  T: invalid ", nTbad, " non-decreasing ", nTgrow, "  C: no valid decreasing update ", nCnone, "  choices ", counts)
    return ok, P, choice
end
const PCUR = Ref{Any}(nothing)
function newC_dir(P, p, f, s)
    w = rot_to(walk(P, f), p); a = s === :next ? w[2] : w[end]
    pp, qq = oriented(P, f, p, a)
    h = TNQS.face_of_dart(P.emb, qq, pp)
    r = P.T[(p, a)]
    isinner(h) || return r
    wh = rot_to(walk(P, h), p)
    x = wh[2] == a ? wh[end] : wh[2]
    return add(r, P.C[(x, h)], projQ(P, f, p, a))
end
