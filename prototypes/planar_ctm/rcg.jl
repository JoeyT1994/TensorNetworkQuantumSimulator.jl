# General region oracle: tail-convention blocks with a per-dart EXIT rule. The ray from f across
# e = (p, q) (f's ccw order) enters h = [p, x₁, …, x_{n−2}, q] and leaves through edge
# (x_j, x_{j+1}) with x₀ = p, x_{n−1} = q, j ∈ 0:n−2. Message tails sit at j = 1 (p side) and
# j = n − 3 (q side); Q_left (p side) carries corners x₂…x_j, or C(x₁)⁻¹ for j = 0; Q_right (q side)
# corners x_{j+1}…x_{n−3}, or C(x_{n−2})⁻¹ for j = n − 2.
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
isinner(f) = f != 0
rot_to(w, p) = (k = findfirst(==(p), w); vcat(w[k:end], w[1:(k - 1)]))
const MS = Dict{Any, Int}
add(xs...) = (r = MS(); for x in xs, (k, c) in x; r[k] = get(r, k, 0) + c; end; filter!(kv -> kv[2] != 0, r))
const DEPMODE = Ref(false)        # dependency extraction: no signs, so nothing cancels
neg(x) = DEPMODE[] ? copy(x) : MS(k => -c for (k, c) in x)
mutable struct RC; emb; T::Dict{Any, MS}; C::Dict{Any, MS}; exit::Function; end
walk(P, f) = TNQS.face_vertices(P.emb, f)
chain(P, f, vs) = add((P.C[(p, f)] for p in vs)...)
R(P, p, g) = isinner(g) ? (w = rot_to(walk(P, g), p); chain(P, g, w[3:(end - 1)])) : MS()
function oriented(P, f, a, b)
    w = walk(P, f); n = length(w)
    for k in 1:n
        (w[k], w[mod1(k + 1, n)]) == (a, b) && return (a, b)
        (w[k], w[mod1(k + 1, n)]) == (b, a) && return (b, a)
    end
    error("edge not on face")
end
function Qsplit(P, f, p, q)
    h = TNQS.face_of_dart(P.emb, q, p)
    isinner(h) || return MS(), MS()
    wh = rot_to(walk(P, h), p); n = length(wh); x = wh[2:(n - 1)]
    j = P.exit(P, f, h, wh)
    left = j == 0 ? neg(P.C[(x[1], h)]) : chain(P, h, x[2:j])
    right = j == n - 2 ? neg(P.C[(x[n - 2], h)]) : chain(P, h, x[(j + 1):(n - 3)])
    return left, right
end
Qfull(P, f, p, q) = add(Qsplit(P, f, p, q)...)
function projQ(P, f, u, w)
    p, q = oriented(P, f, u, w)
    l, r = Qsplit(P, f, p, q)
    return u == p ? l : r
end
far_kept(P, p, excl) = [u for (u, _) in vertex_polygon(P.emb, p) if !(u in excl)]
function far_chain(P, p, excl)
    poly = vertex_polygon(P.emb, p); z = length(poly); nb = first.(poly)
    keep = [i for i in 1:z if !(nb[i] in excl)]
    isempty(keep) && return MS()
    start = something(findfirst(i -> !((mod1(i - 1, z)) in keep), keep), 1)
    order = [keep[mod1(start + j - 1, length(keep))] for j in 1:length(keep)]
    r = MS()
    for (j, i) in enumerate(order)
        r = add(r, P.T[(nb[i], p)])
        j < length(order) && (r = add(r, R(P, p, poly[i][2])))
    end
    return r
end
function enlarged(P, p, f)
    w = rot_to(walk(P, f), p)
    r = far_chain(P, p, (w[end], w[2]))
    if isempty(far_kept(P, p, (w[end], w[2])))
        g = only(fp for (u, fp) in vertex_polygon(P.emb, p) if fp != f)
        r = add(r, R(P, p, g))
    end
    return add(r, MS(p => 1))
end
function newC_diag(P, p, f)
    w = rot_to(walk(P, f), p)
    return add(enlarged(P, p, f), projQ(P, f, p, w[end]), projQ(P, f, p, w[2]))
end
const CORNER = Ref{Function}(newC_diag)
newC(P, p, f) = CORNER[](P, p, f)

function newT(P, u, v)
    poly = vertex_polygon(P.emb, u); nb = first.(poly); z = length(nb); i = findfirst(==(v), nb)
    fA, fB = poly[i][2], poly[mod1(i - 1, z)][2]
    excl = Any[v]; isinner(fA) && push!(excl, nb[mod1(i + 1, z)]); isinner(fB) && push!(excl, nb[mod1(i - 1, z)])
    r = far_chain(P, u, excl)
    if isempty(far_kept(P, u, excl)) && length(excl) == 3 && z == 3
        g = poly[mod1(i + 1, z)][2]
        isinner(g) && (r = add(r, neg(R(P, u, g))))
    end
    isinner(fA) && (r = add(r, projQ(P, fA, u, nb[mod1(i + 1, z)])))
    isinner(fB) && (r = add(r, projQ(P, fB, u, nb[mod1(i - 1, z)])))
    return add(r, MS(u => 1))
end
function sweep!(P)
    C2 = Dict{Any, MS}((p, f) => newC(P, p, f) for (p, f) in keys(P.C))
    T2 = Dict{Any, MS}((u, v) => newT(P, u, v) for (u, v) in keys(P.T))
    P.T = T2; P.C = C2
end
Zv(P, v) = add(MS(v => 1), (add(P.T[(u, v)], R(P, v, f)) for (u, f) in vertex_polygon(P.emb, v))...)
# Z_e: both far chains; across each adjacent inner face h the connector between the two tails is
# the full Q of h as seen from the face g on the other side of the edge.
function Ze(P, u, v)
    r = add(MS(u => 1, v => 1), far_chain(P, u, (v,)), far_chain(P, v, (u,)))
    f1, f2 = TNQS.face_of_dart(P.emb, u, v), TNQS.face_of_dart(P.emb, v, u)
    for (h, g) in ((f1, f2), (f2, f1))
        isinner(h) || continue
        if isinner(g)
            p, q = oriented(P, g, u, v); r = add(r, Qfull(P, g, p, q))
        else
            wh = rot_to(walk(P, h), u); wh[2] == v || (wh = rot_to(walk(P, h), v))
            n = length(wh); r = add(r, n >= 4 ? chain(P, h, wh[4:(n - 1)]) : neg(P.C[(wh[3], h)]))
        end
    end
    return r
end
function Zf(P, f)
    w = walk(P, f); n = length(w)
    return add((add(enlarged(P, w[k], f), Qfull(P, f, w[k], w[mod1(k + 1, n)])) for k in 1:n)...)
end
function newrc(g, exitrule)
    emb = planar_embedding(g)
    P = RC(emb, Dict{Any, MS}(), Dict{Any, MS}(), exitrule)
    for e in TNQS.edges(g), (u, v) in ((TNQS.src(e), TNQS.dst(e)), (TNQS.dst(e), TNQS.src(e))); P.T[(u, v)] = MS(); end
    for (f, w) in enumerate(TNQS.faces(emb)), p in w; P.C[(p, f)] = MS(); end
    return P
end
function deps(P)
    Tsave, Csave = P.T, P.C
    P.T = Dict{Any, MS}(k => MS(("T", k) => 1) for k in keys(Tsave)); P.C = Dict{Any, MS}(k => MS(("C", k) => 1) for k in keys(Csave))
    isblk(a) = a isa Tuple && length(a) == 2 && a[1] in ("T", "C")
    dep = Dict{Any, Vector{Any}}()
    for (p, f) in keys(Csave); dep[("C", (p, f))] = [a for a in keys(newC(P, p, f)) if isblk(a)]; end
    for (u, v) in keys(Tsave); dep[("T", (u, v))] = [a for a in keys(newT(P, u, v)) if isblk(a)]; end
    P.T, P.C = Tsave, Csave
    return dep
end
function has_cycle(P)
    dep = deps(P); color = Dict{Any, Int}()
    function dfs(b)
        color[b] = 1
        for d in dep[b]
            c = get(color, d, 0)
            c == 1 && return true
            c == 0 && dfs(d) && return true
        end
        color[b] = 2; return false
    end
    return any(b -> get(color, b, 0) == 0 && dfs(b), collect(keys(dep)))
end
function verdict(label, g, exitrule; nsweeps = 60, quiet = false)
    P = newrc(g, exitrule)
    cyc = has_cycle(P)
    for _ in 1:nsweeps; sweep!(P); end
    full = MS(v => 1 for v in vertices(g))
    bv = count(v -> Zv(P, v) != full, vertices(g))
    be = count(e -> Ze(P, TNQS.src(e), TNQS.dst(e)) != full, TNQS.edges(g))
    bf = count(f -> Zf(P, f) != full, 1:length(TNQS.faces(P.emb)))
    ok = bv == 0 && be == 0 && bf == 0
    quiet || println(rpad(label, 22), ok ? "PASS" : "fail", "  cycle=", cyc, "  bad Zv/Ze/Zf ", bv, "/", be, "/", bf, "  (V=", nv(g), ")")
    return ok, cyc
end

# --- exit rules
straight(P, f, h, wh) = 1
function dual_distance(emb)
    nf = length(TNQS.faces(emb)); d = fill(typemax(Int), nf); q = Int[]
    for f in 1:nf
        w = TNQS.face_vertices(emb, f)
        if any(k -> TNQS.face_of_dart(emb, w[mod1(k + 1, length(w))], w[k]) == 0, eachindex(w))
            d[f] = 1; push!(q, f)
        end
    end
    while !isempty(q)
        f = popfirst!(q); w = TNQS.face_vertices(emb, f)
        for k in eachindex(w)
            h = TNQS.face_of_dart(emb, w[mod1(k + 1, length(w))], w[k])
            if h != 0 && d[h] == typemax(Int); d[h] = d[f] + 1; push!(q, h); end
        end
    end
    return d
end
# geodesic: leave h through the edge whose far face is nearest the outer face; ties → straight (j = 1),
# then the smallest j
function make_geodesic(emb)
    d = dual_distance(emb); dd(h) = h == 0 ? 0 : d[h]
    return function (P, f, h, wh)
        n = length(wh); best = (typemax(Int), 0, 0); bj = 1
        for j in 0:(n - 2)
            a, b = wh[j + 1], wh[j + 2]
            key = (dd(TNQS.face_of_dart(P.emb, b, a)), j == 1 ? 0 : 1, j)
            key < best && (best = key; bj = j)
        end
        return bj
    end
end
