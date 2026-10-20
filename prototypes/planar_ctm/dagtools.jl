# Rule design for the planar CTM: geodesic exits + per-corner update modes making the block
# dependency graph acyclic (or, failing that, a rule set the region oracle proves convergent).
include(joinpath(@__DIR__, "rcs.jl"))

# geodesic exits: minimise the dual distance (to the outer face) of the face beyond the exit edge;
# ties → the most opposite edge, then the smaller j
function make_geo(emb)
    d = dual_distance(emb); dd(h) = h == 0 ? 0 : d[h]
    return function (P, f, h, wh)
        n = length(wh); best = nothing; bj = 0
        for j in 0:(n - 2)
            key = (dd(TNQS.face_of_dart(P.emb, wh[mod1(j + 2, n)], wh[j + 1])), abs(2j - (n - 2)), j)
            (isnothing(best) || key < best) && (best = key; bj = j)
        end
        return bj
    end
end
candC(m) = m === :diag ? ((P, p, f) -> newC_diag(P, p, f)) : ((P, p, f) -> newC_dir(P, p, f, m))
const MODES = (:diag, :prev, :next)
function find_cycle(dep)
    color = Dict{Any, Int}(); stack = Any[]
    function dfs(b)
        color[b] = 1; push!(stack, b)
        for d in dep[b]
            c = get(color, d, 0)
            c == 1 && return stack[findfirst(==(d), stack):end]
            if c == 0
                r = dfs(d); isnothing(r) || return r
            end
        end
        pop!(stack); color[b] = 2; return nothing
    end
    for b in keys(dep)
        get(color, b, 0) == 0 || continue
        r = dfs(b); isnothing(r) || return r
    end
    return nothing
end
# per-corner modes: a DAG if the switch search finds one, else all-diagonal if the oracle proves it
# converges; `nothing` if neither. Returns (choice, :dag | :oracle).
design_rules(g; kw...) = design_rules_with(g, make_geo(planar_embedding(g)); kw...)
function design_rules_with(g, ex; maxit = 5000)
    emb = planar_embedding(g)
    P = newrc(g, ex)
    Tdep = Dict(("T", k) => first.(parts(P, P -> newT(P, k...))) for k in keys(P.T))
    Cdep = Dict(("C", k) => Dict(m => first.(parts(P, P -> candC(m)(P, k...))) for m in MODES) for k in keys(P.C))
    choice = Dict(k => :diag for k in keys(Cdep))
    dep = merge(Tdep, Dict(k => Cdep[k][choice[k]] for k in keys(Cdep)))
    it = 0; tabu = Dict{Any, Int}()
    while (cyc = find_cycle(dep)) !== nothing
        it += 1
        cs = [b for b in cyc if b[1] == "C"]
        if it > maxit || isempty(cs)
            CORNER[] = (P, p, f) -> newC_diag(P, p, f)
            ok, _ = verdict("", g, ex; quiet = true, nsweeps = 4 * (length(TNQS.faces(emb)) + 4))
            return ok ? (Dict(k[2] => :diag for k in keys(Cdep)), :oracle) : (nothing, :none)
        end
        b = argmin(x -> get(tabu, x, 0), cs)
        tabu[b] = it
        choice[b] = MODES[mod1(findfirst(==(choice[b]), MODES) + 1, 3)]
        dep[b] = Cdep[b][choice[b]]
    end
    return Dict(k[2] => m for (k, m) in choice), :dag
end

# Exit rules in order of preference: straight (no turns outside triangles), opposite edge, geodesic
# (always terminates). The first whose rays all reach the outer face and that certifies wins.
const EXIT_RULES = [
    ("straight", emb -> ((P, f, h, wh) -> 1)),
    ("opposite", emb -> ((P, f, h, wh) -> (n = length(wh); n == 3 ? 1 : fld(n - 2, 2)))),
    ("geodesic", make_geo),
]
function rays_terminate(g, ex)
    P = newrc(g, ex)
    for (f, w) in enumerate(TNQS.faces(P.emb)), k in eachindex(w)
        try ray(P, f, w[k], w[mod1(k + 1, length(w))]) catch; return false end
    end
    return true
end
function design(g)
    emb = planar_embedding(g)
    for (name, mk) in EXIT_RULES
        ex = mk(emb)
        rays_terminate(g, ex) || continue
        choice, how = design_rules_with(g, ex)
        isnothing(choice) || return (exit = name, exitrule = ex, choice = choice, how = how)
    end
    return nothing
end

# Minimum-depth acyclic rule choice (AND/OR least fixed point): a message's level is 1 + the max
# over its parts, a corner's 1 + the min over its candidate updates of their max part level.
# Value iteration from ∞; finite levels are well-founded derivations, so the argmin choice is a DAG
# whose depth (the sweeps needed for exactness at lossless χ) is minimal for these candidates.
# Returns (choice, depth) or (nothing, Inf) when some block has no finite level.
const LEVELS = Ref{Any}(nothing)
function minimax_rules(g, ex)
    P = newrc(g, ex)
    Tdep = Dict(("T", k) => first.(parts(P, P -> newT(P, k...))) for k in keys(P.T))
    Cdep = Dict(("C", k) => Dict(m => first.(parts(P, P -> candC(m)(P, k...))) for m in MODES) for k in keys(P.C))
    L = Dict{Any, Float64}(b => Inf for b in Iterators.flatten((keys(Tdep), keys(Cdep))))
    lev(ds) = 1 + maximum((L[d] for d in ds); init = 0.0)
    choice = Dict{Any, Symbol}()
    for _ in 1:(length(L) + 1)
        changed = false
        for (b, ds) in Tdep
            v = lev(ds); v < L[b] && (L[b] = v; changed = true)
        end
        for (b, cs) in Cdep
            m, v = :diag, Inf
            for mm in MODES
                vv = lev(cs[mm])
                if vv < v
                    m = mm; v = vv
                end
            end
            if v < L[b]
                L[b] = v; choice[b[2]] = m; changed = true
            end
        end
        changed || break
    end
    depth = maximum(values(L))
    isfinite(depth) || return nothing, Inf
    LEVELS[] = L
    return choice, depth
end
function design_minimax(g)
    emb = planar_embedding(g); best = nothing
    for (name, mk) in EXIT_RULES
        ex = mk(emb)
        rays_terminate(g, ex) || continue
        choice, depth = minimax_rules(g, ex)
        isnothing(choice) && continue
        (isnothing(best) || depth < best.depth) && (best = (exit = name, exitrule = ex, choice = choice, how = :dag, depth = depth, levels = LEVELS[]))
    end
    if isnothing(best)          # no acyclic choice: the region oracle must certify the cyclic one
        d = design(g)
        isnothing(d) && return nothing
        return merge(d, (depth = Inf,))
    end
    return best
end
