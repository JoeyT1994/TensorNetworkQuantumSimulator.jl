#=
Planar CTMRG prototype v2 (tensor level), on a single-layer TensorNetwork.

Blocks (tail convention): messages T[(u, v)] (legs bond(u,v), σ(fA,(u,u_A)), σ(fB,(u,u_B)) on inner
faces), turn corners C[(p, f)] (legs σ(f,(a,p)), σ(f,(p,b))). σ(f, e) is the bundle crossed by the
ray from f across e; the ray leaves the next face h through edge j = exitf(f, h, wh) (geodesic:
toward the outer face). The projector at (f, (p → q)) compresses [bond(p,q), σ(h, exit)] and is
computed from Bw = Ql·E(p, f), Be = Qr·E(q, f); Ql / Qr are the parts of h's corner chain on
either side of the exit (a turn next to the entry edge uses an inverse corner).

Corner updates per `mode[(p, f)]`: :diag  C' = P·E(p, f)·P  (enlarged corner)
                                   :prev / :next  C' = P·[T(p → a)·C(x, h)]   (directional)
The directional form keeps T(p → a)'s other side leg σ(f, (p, b)) from the previous sweep; once
dimensions are stable it is renamed to the current index (bases are deterministic functions of
converged inputs), before that the diagonal form is used.
=#
module PCTM
using TensorNetworkQuantumSimulator
using ITensors: ITensors, ITensor, Index, inds, commonind, dim, array, itensor, scalar, replaceind, replaceinds, hasind
using LinearAlgebra
using Random
const TNQS = TensorNetworkQuantumSimulator

ekey(a, b) = a < b ? (a, b) : (b, a)
isinner(f) = f != 0
rot_to(w, p) = (k = findfirst(==(p), w); vcat(w[k:end], w[1:(k - 1)]))

mutable struct Engine
    tn
    emb
    exitf::Function
    mode::Dict{Any, Symbol}
    bond::Dict{Any, Index}
    side::Dict{Any, Index}
    T::Dict{Any, ITensor}
    C::Dict{Any, ITensor}
    maxdim::Int
    opts
    stats::Dict{Symbol, Int}
    prev::Any                # the previous sweep's projectors (nothing before the first sweep)
    G::Dict{Any, ITensor}    # the last sweep's basis changes
end
walk(E, f) = TNQS.face_vertices(E.emb, f)
sid(E, f, a, b) = E.side[(f, ekey(a, b))]
other_nbr(E, h, u, v) = (w = rot_to(walk(E, h), u); w[2] == v ? w[end] : w[2])
stat!(E, k) = (E.stats[k] = get(E.stats, k, 0) + 1)

function contract_list(ts)
    ts = filter(!isnothing, collect(ts))
    isempty(ts) && return nothing
    length(ts) == 1 && return only(ts)
    length(ts) == 2 && return ts[1] * ts[2]
    return TNQS.TensorInterface.contract(ts; sequence = TNQS.contraction_sequence(ts; alg = "optimal"))
end
chain(E, f, vs) = isempty(vs) ? nothing : contract_list([E.C[(p, f)] for p in vs])
R(E, p, g) = isinner(g) ? (w = rot_to(walk(E, g), p); chain(E, g, w[3:(end - 1)])) : nothing

const PINV_RTOL = Ref(1.0e-12)
const CHECK = Ref(false)            # compare every directional corner with the diagonal one
const EXACT = Ref(false)            # identity projectors: uncompressed bundles (validation only)
const EXACT_ROT = Ref(false)        # exact mode with a fresh random rotation per interface per sweep
const CHECKLOG = Any[]
# the bilinear pseudo-inverse of a 2-leg connector: N[s1, s2] = pinv(A)[s2, s1], so A·N·A = A
function inv2(M, s1, s2; rtol = PINV_RTOL[])
    A = array(M, s1, s2)
    F = svd(A)
    cut = rtol * maximum(F.S)
    Ai = F.V * Diagonal([s > cut ? 1 / s : 0.0 for s in F.S]) * F.U'
    return itensor(Matrix(transpose(Ai)), s1, s2)
end

# the edge {a, b} of f in f's ccw order
function oriented(E, f, a, b)
    w = walk(E, f); n = length(w)
    for k in 1:n
        (w[k], w[mod1(k + 1, n)]) == (a, b) && return (a, b)
        (w[k], w[mod1(k + 1, n)]) == (b, a) && return (b, a)
    end
    error("edge not on face")
end
# h, its walk from p, and the exit j for the ray from f across (p → q)
function crossing(E, f, p, q)
    h = TNQS.face_of_dart(E.emb, q, p)
    isinner(h) || return (h, nothing, nothing)
    wh = rot_to(walk(E, h), p)
    return (h, wh, E.exitf(f, h, wh))
end
function Qsplit(E, f, p, q; left = true, right = true)
    h, wh, j = crossing(E, f, p, q)
    isinner(h) || return (nothing, nothing)
    n = length(wh); x = wh[2:(n - 1)]; xx(i) = wh[i + 1]      # xx(0) = p, xx(n − 1) = q
    Ql = !left ? nothing : if j == 0
        stat!(E, :inverse_turn); inv2(E.C[(x[1], h)], sid(E, h, p, x[1]), sid(E, h, x[1], xx(2)))
    else
        chain(E, h, x[2:j])
    end
    Qr = !right ? nothing : if j == n - 2
        stat!(E, :inverse_turn); inv2(E.C[(x[n - 2], h)], sid(E, h, x[n - 2], q), sid(E, h, xx(n - 3), x[n - 2]))
    else
        chain(E, h, x[(j + 1):(n - 3)])
    end
    return (Ql, Qr)
end
# the full connector across h between the two tails (independent of the exit)
function Qfull(E, h, p, q)          # h's walk has q → p (h is right of p → q)
    isinner(h) || return nothing
    wh = rot_to(walk(E, h), p); n = length(wh)
    n >= 4 && return chain(E, h, wh[3:(n - 2)])
    stat!(E, :inverse_triangle)
    return inv2(E.C[(wh[2], h)], sid(E, h, p, wh[2]), sid(E, h, wh[2], q))
end
function exit_index(E, f, p, q)
    h, wh, j = crossing(E, f, p, q)
    isinner(h) || return nothing
    n = length(wh)
    return sid(E, h, wh[j + 1], wh[mod1(j + 2, n)])
end

far_kept(E, p, excl) = [u for (u, _) in vertex_polygon(E.emb, p) if !(u in excl)]
function far_chain(E, p, excl)
    poly = vertex_polygon(E.emb, p); z = length(poly); nb = first.(poly)
    keep = [i for i in 1:z if !(nb[i] in excl)]
    isempty(keep) && return ITensor[]
    start = something(findfirst(i -> !((mod1(i - 1, z)) in keep), keep), 1)
    order = [keep[mod1(start + j - 1, length(keep))] for j in 1:length(keep)]
    ts = ITensor[]
    for (j, i) in enumerate(order)
        push!(ts, E.T[(nb[i], p)])
        if j < length(order)
            r = R(E, p, poly[i][2]); isnothing(r) || push!(ts, r)
        end
    end
    return ts
end
function enlarged(E, p, f)
    w = rot_to(walk(E, f), p)
    ts = far_chain(E, p, (w[end], w[2]))
    if isempty(far_kept(E, p, (w[end], w[2])))
        g = only(fp for (u, fp) in vertex_polygon(E.emb, p) if fp != f)
        r = R(E, p, g); isnothing(r) || push!(ts, r)
    end
    return contract_list(vcat(ts, [E.tn[p]]))
end

# Canonical gauge of a projector pair: with a fixed reference matrix Ref per interface (seeded on
# the interface and the dimensions), choose the kept basis so that Refᵀ·PA = 1. PA·PB is
# unchanged; the basis is then a continuous function of the kept subspace, so it converges with
# the blocks and a leg carried over from the previous sweep is in (nearly) the current basis.
const REFS = Dict{Any, Matrix{Float64}}()
const CANONICAL = Ref(false)
function canonical(key, PA, PB, wi, ins)
    CANONICAL[] || return PA, PB
    N = prod(dim.(ins)); r = dim(wi)
    A = reshape(array(PA, ins..., wi), N, r)
    B = reshape(array(PB, wi, ins...), r, N)
    Rf = get!(() -> randn(Random.MersenneTwister(hash((key, N, r))), N, r), REFS, (key, N, r))
    M = transpose(Rf) * A
    return itensor(reshape(A / M, dim.(ins)..., r), ins..., wi), itensor(reshape(M * B, r, dim.(ins)...), wi, ins...)
end
# Isometric projector onto the union of the ins-space ranges of Bw and Be (both sides kept): exact
# at lossless χ for either side, well-conditioned, PB = PAᵀ; truncation keeps the leading χ
# directions of [Bw/‖Bw‖  Beᵀ/‖Be‖]. Basis fixed by the canonical gauge (Refᵀ·U = upper-triangular
# positive is overkill; Refᵀ·U = 1 as for the biorthogonal pair).
const ISO = Ref(true)
const ISO_RTOL = Ref(1.0e-12)
const ISO_FULL = Ref(false)      # keep every direction (a rotation; validation)
function iso_projector(blocks, ins, maxdim)
    N = prod(dim.(ins))
    mat(B) = (rest = [i for i in inds(B) if !(i in ins)]; X = reshape(array(B, ins..., rest...), N, :); X / norm(X))
    M = reduce(hcat, [mat(B) for B in blocks if all(i -> i in inds(B), ins)])
    F = svd(M)
    r = ISO_FULL[] ? size(F.U, 2) : min(maxdim, count(>(ISO_RTOL[] * F.S[1]), F.S))
    U = F.U[:, 1:r]
    wi = Index(r, "σ")
    return itensor(reshape(U, dim.(ins)..., r), ins..., wi), itensor(reshape(Matrix(transpose(U)), r, dim.(ins)...), wi, ins...), wi
end
# The pre-projection block of T(u → v): T_u, the far messages and ring connectors (and the inverse
# at an interior trivalent vertex). Returns (Y, fA, uA, fB, uB): the side faces and side neighbours.
function msg_pre(E, u, v)
    poly = vertex_polygon(E.emb, u); nb = first.(poly); z = length(nb); i = findfirst(==(v), nb)
    fA, fB = poly[i][2], poly[mod1(i - 1, z)][2]
    uA, uB = nb[mod1(i + 1, z)], nb[mod1(i - 1, z)]
    excl = Any[v]; isinner(fA) && push!(excl, uA); isinner(fB) && push!(excl, uB)
    ts = far_chain(E, u, excl)
    if isempty(far_kept(E, u, excl)) && length(excl) == 3 && z == 3
        g = poly[mod1(i + 1, z)][2]
        if isinner(g)
            r = R(E, u, g)
            isnothing(r) && error("trivalent vertex whose far face is a triangle: unhandled")
            push!(ts, inv2(r, sid(E, g, uA, other_nbr(E, g, uA, u)), sid(E, g, uB, other_nbr(E, g, uB, u))))
            stat!(E, :inverse_trivalent)
        end
    elseif isempty(far_kept(E, u, excl)) && length(excl) == 3 && z == 2
        error("interior degree-2 vertex: unsupported")
    end
    return contract_list(vcat(ts, [E.tn[u]])), fA, uA, fB, uB
end
# the directional corner's pre-projection block T(p → a)·C(x, h) and the interface it is projected at
function dir_pre(E, p, f, m)
    w = rot_to(walk(E, f), p); a, b = w[end], w[2]
    aa, bb = m === :next ? (b, a) : (a, b)
    pp, qq = oriented(E, f, p, aa)
    h = TNQS.face_of_dart(E.emb, qq, pp)
    parts = ITensor[E.T[(p, aa)]]
    if isinner(h)
        wh = rot_to(walk(E, h), p); x = wh[2] == aa ? wh[end] : wh[2]
        push!(parts, E.C[(x, h)])
    end
    return contract_list(parts), aa, bb
end
function projectors!(E)
    pr = Dict{Any, Any}()
    pre = Dict{Any, Any}(k => msg_pre(E, k...) for k in keys(E.T))
    pr[(:pre, :E)] = pre
    dpre = Dict{Any, Any}(k => dir_pre(E, k..., m) for (k, m) in E.mode if m !== :diag && haskey(E.C, k))
    # the stale leg of each directional corner, expanded back to the current ins coordinates of its
    # interface, so that the new kept space contains it and the basis change carries it exactly
    stale = Dict{Any, Vector{ITensor}}()
    if !isnothing(E.prev)
        for ((c, fc), (Y, aa, bb)) in dpre
            I = (fc, ekey(c, bb)); haskey(E.prev, I) || continue
            old = E.prev[I]; ps = c == old.p
            GJ = isnothing(old.J) ? nothing : get(E.G, (old.J, ps), nothing)
            !isnothing(old.J) && isnothing(GJ) && continue
            push!(get!(() -> ITensor[], stale, I), contract_list([E.T[(c, aa)], ps ? old.PB : old.PA, GJ]))
        end
    end
    for (f, w) in enumerate(TNQS.faces(E.emb))
        n = length(w)
        Es = [enlarged(E, w[k], f) for k in 1:n]
        for k in 1:n
            p, q = w[k], w[mod1(k + 1, n)]
            Ql, Qr = Qsplit(E, f, p, q)
            ins = Index[E.bond[ekey(p, q)]]
            xi = exit_index(E, f, p, q); isnothing(xi) || push!(ins, xi)
            Bw = isnothing(Ql) ? Es[k] : Ql * Es[k]
            Be = isnothing(Qr) ? Es[mod1(k + 1, n)] : Qr * Es[mod1(k + 1, n)]
            if EXACT[]
                # lossless identity "projector": the combined bundle, no compression, no gauge
                N = prod(dim.(ins)); wi = Index(N, "σx")
                PA = itensor(reshape(Matrix{Float64}(LinearAlgebra.I, N, N), dim.(ins)..., N), ins..., wi)
                PB = itensor(reshape(Matrix{Float64}(LinearAlgebra.I, N, N), N, dim.(ins)...), wi, ins...)
                if EXACT_ROT[]
                    Qr_ = Matrix(qr(randn(N, N)).Q)
                    PA = itensor(reshape(Qr_, dim.(ins)..., N), ins..., wi)
                    PB = itensor(reshape(Matrix(transpose(Qr_)), N, dim.(ins)...), wi, ins...)
                end
            elseif ISO[]
                blocks = ITensor[Bw, Be]
                ap = (Y, side) -> push!(blocks, isnothing(side) ? Y : side * Y)
                pprev, qnext = w[mod1(k - 1, n)], w[mod1(k + 2, n)]
                haskey(pre, (p, pprev)) && ap(pre[(p, pprev)][1], Ql)
                haskey(pre, (q, qnext)) && ap(pre[(q, qnext)][1], Qr)
                for ((c, fc), (Y, aa, bb)) in dpre
                    fc == f && Set((c, aa)) == Set((p, q)) || continue
                    ap(Y, c == p ? Ql : Qr)
                end
                append!(blocks, get(stale, (f, ekey(p, q)), ITensor[]))
                PA, PB, wi = iso_projector(blocks, ins, E.maxdim)
                PA, PB = canonical((f, ekey(p, q)), PA, PB, wi, ins)
            else
                PA, PB, wi = TNQS._ctm_interface_proj2(Bw, Be, ins, E.maxdim, E.opts; subspace = false)
                PA, PB = canonical((f, ekey(p, q)), PA, PB, wi, ins)
            end
            h, wh, j = crossing(E, f, p, q)
            J = isinner(h) ? (h, ekey(wh[j + 1], wh[mod1(j + 2, length(wh))])) : nothing
            pr[(f, ekey(p, q))] = (p = p, q = q, PA = PA, PB = PB, Ql = Ql, Qr = Qr, w = wi, J = J, xin = xi)
        end
        pr[(f, :E)] = Es
    end
    return pr
end
# the projector of f at edge {u, w} as applied on u's side (connector first)
function side_proj(pr, f, u, w)
    x = pr[(f, ekey(u, w))]
    return u == x.p ? contract_list([x.Ql, x.PA]) : contract_list([x.Qr, x.PB])
end

# Change of basis σ_{s−1}(I) → σ_s(I) on either side of interface I, from the previous sweep's map
# at the interface J the ray continues into: p side G_I = PB_{s−1}·G_J·PA_s, q side
# G_I = PA_{s−1}·G_J·PB_s, with G_J : σ_{s−2}(J) → σ_{s−1}(J). At the boundary the ins is the bare
# bond and no G_J enters, so the maps become available from the boundary inward. Identity at the
# fixed point.
function basis_changes(E, pr)
    G = Dict{Any, ITensor}()
    isnothing(E.prev) && return G
    for (I, new) in pr
        I[2] === :E && continue
        haskey(E.prev, I) || continue
        old = E.prev[I]
        # the tail after Ql (p side) sits on the p side of J, after Qr (q side) on the q side
        for pside in (true, false)
            GJ = nothing
            if !isnothing(new.J)
                GJ = get(E.G, (new.J, pside), nothing)
                isnothing(GJ) && continue
            end
            G[(I, pside)] = pside ? contract_list([old.PB, GJ, new.PA]) : contract_list([old.PA, GJ, new.PB])
        end
    end
    return G
end
function sweep!(E; normalize = true)
    pr = projectors!(E)
    Gs = basis_changes(E, pr)
    newT = Dict{Any, ITensor}()
    for (u, v) in keys(E.T)
        Y, fA, uA, fB, uB = pr[(:pre, :E)][(u, v)]
        lst = ITensor[Y]
        isinner(fA) && push!(lst, side_proj(pr, fA, u, uA))
        isinner(fB) && push!(lst, side_proj(pr, fB, u, uB))
        t = contract_list(lst)
        newT[(u, v)] = normalize ? t / norm(t) : t
    end
    newC = Dict{Any, ITensor}()
    for (p, f) in keys(E.C)
        w = rot_to(walk(E, f), p); a, b = w[end], w[2]
        k = findfirst(==(p), walk(E, f))
        m = get(E.mode, (p, f), :diag)
        c = nothing
        if m !== :diag
            aa, bb = m === :next ? (b, a) : (a, b)          # aa: the directional neighbour; bb: the kept side
            pp, qq = oriented(E, f, p, aa)
            h = TNQS.face_of_dart(E.emb, qq, pp)
            I = (f, ekey(p, bb))
            if haskey(Gs, (I, p == pr[I].p))
                parts = ITensor[E.T[(p, aa)]]
                if isinner(h)
                    wh = rot_to(walk(E, h), p); x = wh[2] == aa ? wh[end] : wh[2]
                    push!(parts, E.C[(x, h)])
                end
                push!(parts, side_proj(pr, f, p, aa))
                push!(parts, Gs[(I, p == pr[I].p)])
                c = contract_list(parts)
                if CHECK[]
                    c0 = contract_list([side_proj(pr, f, p, a), pr[(f, :E)][k], side_proj(pr, f, p, b)])
                    cn, c0n = c / ITensors.norm(c), c0 / ITensors.norm(c0); push!(CHECKLOG, ((p, f), m, ITensors.norm(cn - c0n), ITensors.norm(cn + c0n)))
                end
                stat!(E, :directional)
            else
                stat!(E, :directional_deferred)
            end
        end
        if isnothing(c)
            c = contract_list([side_proj(pr, f, p, a), pr[(f, :E)][k], side_proj(pr, f, p, b)])
        end
        newC[(p, f)] = normalize ? c / norm(c) : c
    end
    for (key, x) in pr
        key[2] === :E && continue
        E.side[key] = x.w
    end
    E.T = newT; E.C = newC
    E.prev = Dict(k => v for (k, v) in pr if k[2] !== :E)
    E.G = Gs
    return E
end

function init(tn, emb, exitf, mode; maxdim = 10_000)
    g = TNQS.graph(emb)
    bond = Dict{Any, Index}(ekey(TNQS.src(e), TNQS.dst(e)) => commonind(tn[TNQS.src(e)], tn[TNQS.dst(e)]) for e in TNQS.edges(g))
    side = Dict{Any, Index}()
    for (f, w) in enumerate(TNQS.faces(emb)), k in eachindex(w)
        side[(f, ekey(w[k], w[mod1(k + 1, length(w))]))] = Index(1, "σ")
    end
    E = Engine(tn, emb, exitf, mode, bond, side, Dict{Any, ITensor}(), Dict{Any, ITensor}(), maxdim, TNQS.CTMOptions(), Dict{Symbol, Int}(), nothing, Dict{Any, ITensor}())
    for e in TNQS.edges(g), (u, v) in ((TNQS.src(e), TNQS.dst(e)), (TNQS.dst(e), TNQS.src(e)))
        legs = Index[bond[ekey(u, v)]]
        for h in (TNQS.face_of_dart(emb, u, v), TNQS.face_of_dart(emb, v, u))
            isinner(h) && push!(legs, sid(E, h, u, other_nbr(E, h, u, v)))
        end
        E.T[(u, v)] = ITensor(1.0, legs...)
    end
    for (f, w) in enumerate(TNQS.faces(emb)), k in eachindex(w)
        p = w[k]; a = w[mod1(k - 1, length(w))]; b = w[mod1(k + 1, length(w))]
        E.C[(p, f)] = ITensor(1.0, sid(E, f, a, p), sid(E, f, p, b))
    end
    return E
end

# --- Kikuchi free energy: c_f = 1, c_e = 1 − #inner faces on e, c_v = 1 + F_v − z_v
function Zv(E, v)
    ts = ITensor[E.tn[v]]
    for (u, f) in vertex_polygon(E.emb, v)
        push!(ts, E.T[(u, v)]); r = R(E, v, f); isnothing(r) || push!(ts, r)
    end
    return scalar(contract_list(ts))
end
function Ze(E, u, v)
    ts = vcat(ITensor[E.tn[u], E.tn[v]], far_chain(E, u, (v,)), far_chain(E, v, (u,)))
    for (p, q) in ((u, v), (v, u))
        q2 = Qfull(E, TNQS.face_of_dart(E.emb, q, p), p, q); isnothing(q2) || push!(ts, q2)
    end
    return scalar(contract_list(ts))
end
function Zf(E, f)
    w = walk(E, f); n = length(w); ts = ITensor[]
    for k in 1:n
        p, q = w[k], w[mod1(k + 1, n)]
        push!(ts, enlarged(E, p, f))
        q2 = Qfull(E, TNQS.face_of_dart(E.emb, q, p), p, q); isnothing(q2) || push!(ts, q2)
    end
    return scalar(contract_list(ts))
end
function kikuchi(E)
    g = TNQS.graph(E.emb); F = 0.0
    for v in vertices(g)
        poly = vertex_polygon(E.emb, v)
        c = 1 + count(((u, f),) -> isinner(f), poly) - length(poly)
        c == 0 || (F += c * log(abs(Zv(E, v))))
    end
    for e in TNQS.edges(g)
        u, v = TNQS.src(e), TNQS.dst(e)
        c = 1 - isinner(TNQS.face_of_dart(E.emb, u, v)) - isinner(TNQS.face_of_dart(E.emb, v, u))
        c == 0 || (F += c * log(abs(Ze(E, u, v))))
    end
    return F + sum(log(abs(Zf(E, f))) for f in 1:length(TNQS.faces(E.emb)); init = 0.0)
end
end # module
