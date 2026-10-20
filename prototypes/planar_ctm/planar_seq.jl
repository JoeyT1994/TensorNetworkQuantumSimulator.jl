#=
Sequential planar CTMRG (included into PCTM).

A pass = (1) one COHERENT projector set, (2) every block recomputed once, in dependency order.

(1) Interfaces are processed from the boundary inward along their rays: the projector at I compresses
[bond, σ(J)] with J the interface the ray continues into, and its inputs Bw, Be (built from the
current blocks, in the previous pass's coordinates) have their tail leg carried into J's NEW
coordinates by M_J first. M_I (old → new coordinates of σ(I), per side) is then
    p side  M_I = PB_old · M_J · PA_new        q side  M_I = PA_old · M_J · PB_new
so it is built in the same pass, boundary first. An imperfect M only makes a projector slightly
suboptimal: the network itself is always consistent, because

(2) with the projector set fixed, each block is built from parts that were themselves rebuilt
earlier in this pass (the minimum-depth levels order them), in the current coordinates. No leg is
ever carried over from a previous pass, so any projectors — the optimal two-sided ones included —
can be used, and directional corners need no basis maps.

The first pass uses random coherent isometries (full rank up to χ), so at lossless χ the very first
block pass is already exact.
=#
mutable struct Seq
    E::Engine
    J::Dict{Any, Any}          # interface → the interface its ray continues into (nothing at the boundary)
    pq::Dict{Any, Any}         # interface → (p, q), f's ccw orientation
    order::Vector{Any}         # interfaces, boundary first
    P::Dict{Any, Any}          # interface → (PA, PB, w)
    blocks::Union{Nothing, Vector{Any}}  # ("T", (u, v)) / ("C", (p, f)) in level order; nothing ⇒ cyclic, repeat
    nrep::Int                  # block passes per projector set when there is no level order
    proj::Symbol               # :biorth or :iso
    prevnb::Dict{Any, Any}     # interface → (p_prev, q_next) in f's walk (side messages T(p→p_prev), T(q→q_next))
end

function Seq(E::Engine; levels = nothing, nrep = 8, proj = :biorth)
    J = Dict{Any, Any}(); pq = Dict{Any, Any}(); pn = Dict{Any, Any}()
    for (f, w) in enumerate(TNQS.faces(E.emb)), k in eachindex(w)
        p, q = w[k], w[mod1(k + 1, length(w))]
        I = (f, ekey(p, q)); pq[I] = (p, q); pn[I] = (w[mod1(k - 1, length(w))], w[mod1(k + 2, length(w))])
        h, wh, j = crossing(E, f, p, q)
        J[I] = isinner(h) ? (h, ekey(wh[j + 1], wh[mod1(j + 2, length(wh))])) : nothing
    end
    dep = Dict{Any, Int}()
    d(I) = get!(dep, I) do; isnothing(J[I]) ? 1 : 1 + d(J[I]) end
    order = sort(collect(keys(J)); by = d)
    blocks = isnothing(levels) ? nothing : [b for (b, _) in sort(collect(levels); by = last)]
    return Seq(E, J, pq, order, Dict{Any, Any}(), blocks, nrep, proj, pn)
end

function random_projectors!(S::Seq)
    E = S.E
    for I in S.order
        ins = Index[E.bond[ekey(S.pq[I]...)]]
        isnothing(S.J[I]) || push!(ins, S.P[S.J[I]].w)
        N = prod(dim.(ins)); r = min(E.maxdim, N)
        U = Matrix(qr(randn(N, r)).Q)[:, 1:r]
        w = Index(r, "σ")
        S.P[I] = (PA = itensor(reshape(U, dim.(ins)..., r), ins..., w), PB = itensor(reshape(Matrix(transpose(U)), r, dim.(ins)...), w, ins...), w = w)
    end
    for I in S.order; E.side[I] = S.P[I].w; end
end

# Two-sided projector for SETS of blocks on either side of an interface. A = [left blocks] and
# B = [right blocks] matricised on the ins legs (each block normalised); Aᵀ B = U S Vᵀ, keep r:
#   PA = B V_r S_r^{-1/2},  PB = S_r^{-1/2} U_rᵀ Aᵀ    ⇒   Aᵀ PA PB B = Aᵀ B at full rank,
# lossless for every left/right pairing (not just the two enlarged corners); below full rank it is
# the truncation of the stacked environment product. :iso instead takes the orthogonal projector
# onto the union of all ranges (PB = PAᵀ).
const STACK_RTOL = Ref(1.0e-12)
function stacked_projector(left, right, ins, maxdim; kind = :biorth)
    N = prod(dim.(ins))
    mat(Bk) = (rest = [i for i in inds(Bk) if !(i in ins)]; X = reshape(array(Bk, ins..., rest...), N, :); X / norm(X))
    ok(Bk) = all(i -> i in inds(Bk), ins)
    A = reduce(hcat, [mat(x) for x in left if ok(x)]); B = reduce(hcat, [mat(x) for x in right if ok(x)])
    if kind === :iso
        F = svd(hcat(A, B)); r = min(maxdim, count(>(STACK_RTOL[] * F.S[1]), F.S))
        PAm = F.U[:, 1:r]; PBm = Matrix(transpose(PAm))
    else
        F = svd(transpose(A) * B); r = min(maxdim, count(>(STACK_RTOL[] * F.S[1]), F.S))
        s = F.S[1:r] .^ -0.5
        PAm = B * F.V[:, 1:r] * Diagonal(s); PBm = Diagonal(s) * transpose(F.U[:, 1:r]) * transpose(A)
    end
    w = Index(r, "σ")
    return itensor(reshape(PAm, dim.(ins)..., r), ins..., w), itensor(reshape(PBm, r, dim.(ins)...), w, ins...), w
end
function new_projectors!(S::Seq)
    E = S.E; newP = Dict{Any, Any}(); M = Dict{Any, ITensor}(); Ec = Dict{Any, ITensor}()
    enl(p, f) = get!(() -> enlarged(E, p, f), Ec, (p, f))
    for I in S.order
        f = I[1]; p, q = S.pq[I]; JI = S.J[I]
        Ql, Qr = Qsplit(E, f, p, q)
        Bw = contract_list([Ql, enl(p, f)]); Be = contract_list([Qr, enl(q, f)])
        ins = Index[E.bond[ekey(p, q)]]
        if !isnothing(JI)
            Bw = Bw * M[(JI, true)]; Be = Be * M[(JI, false)]
            push!(ins, newP[JI].w)
        end
        # the other blocks projected here: the side messages and directional corners, tails carried over
        left = ITensor[Bw]; right = ITensor[Be]
        carry(Y, side) = isnothing(JI) ? Y : Y * M[(JI, side)]
        pprev = S.prevnb[I][1]; qnext = S.prevnb[I][2]
        haskey(E.T, (p, pprev)) && push!(left, carry(contract_list([Qsplit(E, f, p, q; right = false)[1], msg_pre(E, p, pprev)[1]]), true))
        haskey(E.T, (q, qnext)) && push!(right, carry(contract_list([Qsplit(E, f, p, q; left = false)[2], msg_pre(E, q, qnext)[1]]), false))
        for c in (p, q), fc in (f,)
            m = get(E.mode, (c, fc), :diag); m === :diag && continue
            Y, aa, _ = dir_pre(E, c, fc, m)
            Set((c, aa)) == Set((p, q)) || continue
            c == p ? push!(left, carry(contract_list([Qsplit(E, f, p, q; right = false)[1], Y]), true)) :
                     push!(right, carry(contract_list([Qsplit(E, f, p, q; left = false)[2], Y]), false))
        end
        PA, PB, w = stacked_projector(left, right, ins, E.maxdim; kind = S.proj)
        newP[I] = (PA = PA, PB = PB, w = w)
        old = S.P[I]
        MJp = isnothing(JI) ? nothing : M[(JI, true)]; MJq = isnothing(JI) ? nothing : M[(JI, false)]
        M[(I, true)] = contract_list([old.PB, MJp, PA])
        M[(I, false)] = contract_list([old.PA, MJq, PB])
    end
    S.P = newP
    for I in S.order; E.side[I] = S.P[I].w; end
end

# the projector of f at edge {u, w} on u's side, with the connector built from the current corners
function sideP(S::Seq, f, u, w)
    I = (f, ekey(u, w)); p, q = S.pq[I]; x = S.P[I]
    Ql, Qr = Qsplit(S.E, f, p, q; left = u == p, right = u != p)
    return u == p ? contract_list([Ql, x.PA]) : contract_list([Qr, x.PB])
end
function update_block!(S::Seq, b)
    E = S.E
    if b[1] == "T"
        u, v = b[2]
        Y, fA, uA, fB, uB = msg_pre(E, u, v)
        lst = ITensor[Y]
        isinner(fA) && push!(lst, sideP(S, fA, u, uA))
        isinner(fB) && push!(lst, sideP(S, fB, u, uB))
        t = contract_list(lst); E.T[(u, v)] = t / norm(t)
    else
        p, f = b[2]
        w = rot_to(walk(E, f), p); a, bb0 = w[end], w[2]
        m = get(E.mode, (p, f), :diag)
        c = if m === :diag
            contract_list([sideP(S, f, p, a), enlarged(E, p, f), sideP(S, f, p, bb0)])
        else
            Y, aa, _ = dir_pre(E, p, f, m)
            contract_list([Y, sideP(S, f, p, aa)])
        end
        E.C[(p, f)] = c / norm(c)
    end
end
all_blocks(S::Seq) = vcat([("T", k) for k in keys(S.E.T)], [("C", k) for k in keys(S.E.C)])
function pass!(S::Seq)
    if isnothing(S.blocks)
        # cyclic (oracle-certified, diagonal corners only): the Jacobi sweep is consistent by itself
        old = ISO[]; ISO[] = false
        try sweep!(S.E) finally ISO[] = old end
        return S
    end
    isempty(S.P) ? random_projectors!(S) : new_projectors!(S)
    begin
        for b in S.blocks; update_block!(S, b); end
    end
    return S
end
