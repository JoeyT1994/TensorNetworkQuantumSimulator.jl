# CTMRG on a planar embedding: its blocks, how each is built (a recipe), and the rules for updating
# them. The cache that holds the tensors is in ctmrg.jl.
#
# Every inner face f has a corner C(p, f) at each of its vertices p, and every dart u → v a message
# T(u → v). Each block is the contraction of a region of the network cut off along rays to the outer
# face. The cut across the edge (p → q) of a face (f on its left) is an interface, keyed by that dart;
# its ray continues into the face beyond, leaving it through the edge picked by the exit rule.
#
# Blocks and closed regions are recipes, lists of parts:
#   (:site, v), (:message, (u, v)), (:corner, (p, f)), (:enlarged, (p, f)),
#   (:inverse, corners, (I1, I2))   the pseudo-inverse of a chain of corners, between two interfaces,
#   (:projector, I, :A or :B)       one side of the projector pair at interface I.
# Contracting a recipe gives its tensor (ctmrg.jl). Summing it as a signed multiset of vertices gives
# its region, which is how the update rules are chosen and certified here without any tensors.

"""
    CTMRGRules

How a `CTMRGCache` updates its blocks, fixed by the geometry alone. See `ctmrg_rules`.
"""
struct CTMRGRules{V}
    exits::Dict{Tuple{V, V}, Int}                         # interface → its exit in the next face
    next::Dict{Tuple{V, V}, Union{Nothing, Tuple{V, V}}}  # interface → where its ray continues
    interfaces::Vector{Tuple{V, V}}                       # every interface, boundary first
    modes::Dict{Tuple{V, Int}, Symbol}                    # corner → :diag, :prev or :next
    order::Union{Nothing, Vector{Any}}                    # blocks in dependency order (nothing: Jacobi)
end

isinner(f) = f != 0

# f's counterclockwise walk, starting at p
function face_walk(emb::PlanarEmbedding, f, p)
    w = face_vertices(emb, f)
    k = findfirst(==(p), w)
    return vcat(w[k:end], w[1:(k - 1)])
end

# the dart of the edge {a, b} with face f on its left: the key of the interface (f, {a, b})
function interface_dart(emb::PlanarEmbedding, f, a, b)
    face_of_dart(emb, a, b) == f && return (a, b)
    face_of_dart(emb, b, a) == f && return (b, a)
    return error("CTMRG: the edge $a – $b is not on face $f")
end

interface_darts(emb::PlanarEmbedding) = [(w[k], w[mod1(k + 1, length(w))]) for w in faces(emb) for k in eachindex(w)]
darts(emb::PlanarEmbedding) = [d for e in edges(graph(emb)) for d in ((src(e), dst(e)), (dst(e), src(e)))]
corner_keys(emb::PlanarEmbedding) = [(p, f) for (f, w) in enumerate(faces(emb)) for p in w]
block_keys(emb::PlanarEmbedding) = Any[[(:message, d) for d in darts(emb)]; [(:corner, k) for k in corner_keys(emb)]]

# u's neighbour on face h other than v
other_neighbor(emb::PlanarEmbedding, h, u, v) = (w = face_walk(emb, h, u); w[2] == v ? w[end] : w[2])

# Number of faces between each inner face and the outer face.
function dual_distance(emb::PlanarEmbedding)
    d = fill(typemax(Int), length(faces(emb)))
    queue = Int[]
    for (f, w) in enumerate(faces(emb))
        if any(k -> !isinner(face_of_dart(emb, w[mod1(k + 1, length(w))], w[k])), eachindex(w))
            d[f] = 1
            push!(queue, f)
        end
    end
    while !isempty(queue)
        f = popfirst!(queue)
        w = face_vertices(emb, f)
        for k in eachindex(w)
            h = face_of_dart(emb, w[mod1(k + 1, length(w))], w[k])
            if isinner(h) && d[h] == typemax(Int)
                d[h] = d[f] + 1
                push!(queue, h)
            end
        end
    end
    return d
end

# An exit rule picks the edge (x_j, x_{j+1}) a ray leaves a face through, given the face's walk
# wh = [p = x_0, …, x_{n−1} = q] from the edge it entered by: straight on, the opposite edge, or
# toward the outer face.
function exit_rule(emb::PlanarEmbedding, rule::Symbol)
    rule === :straight && return wh -> 1
    rule === :opposite && return wh -> (n = length(wh); n == 3 ? 1 : fld(n - 2, 2))
    d = dual_distance(emb)
    distance(h) = isinner(h) ? d[h] : 0
    return function (wh)
        n = length(wh)
        return argmin(j -> (distance(face_of_dart(emb, wh[mod1(j + 2, n)], wh[j + 1])), abs(2j - (n - 2)), j), 0:(n - 2))
    end
end
# The exit rules, in the order ctmrg_rules prefers them.
const EXIT_RULES = (:straight, :opposite, :geodesic)

function exit_choices(emb::PlanarEmbedding{V}, rule::Symbol) where {V}
    exit = exit_rule(emb, rule)
    exits = Dict{Tuple{V, V}, Int}()
    next = Dict{Tuple{V, V}, Union{Nothing, Tuple{V, V}}}()
    for (p, q) in interface_darts(emb)
        h = face_of_dart(emb, q, p)
        if isinner(h)
            wh = face_walk(emb, h, p)
            j = exit(wh)
            exits[(p, q)] = j
            next[(p, q)] = (wh[j + 1], wh[mod1(j + 2, length(wh))])
        else
            next[(p, q)] = nothing
        end
    end
    return exits, next
end

# How many interfaces each ray crosses before it leaves; `nothing` if some ray never leaves.
function ray_lengths(next::Dict{K}) where {K}
    len = Dict{K, Int}()
    for I in keys(next)
        path = K[I]
        J = next[I]
        while !isnothing(J) && !haskey(len, J)
            J in path && return nothing
            push!(path, J)
            J = next[J]
        end
        n = isnothing(J) ? 0 : len[J]
        for L in reverse(path)
            n += 1
            len[L] = n
        end
    end
    return len
end

corner_parts(f, vs) = Any[(:corner, (x, f)) for x in vs]

# The corners of g around p, except those of p and of its two neighbours on g.
corner_ring(emb::PlanarEmbedding, p, g) = isinner(g) ? corner_parts(g, face_walk(emb, g, p)[3:(end - 1)]) : Any[]

# The messages into p from its neighbours outside `excl`, joined by the rings of the faces
# between consecutive ones.
function far_chain(emb::PlanarEmbedding, p, excl)
    poly = vertex_polygon(emb, p)
    z = length(poly)
    keep = [i for i in 1:z if !(first(poly[i]) in excl)]
    isempty(keep) && return Any[]
    start = something(findfirst(i -> !(mod1(i - 1, z) in keep), keep), 1)
    order = [keep[mod1(start + j - 1, length(keep))] for j in eachindex(keep)]
    parts = Any[]
    for (j, i) in enumerate(order)
        push!(parts, (:message, (first(poly[i]), p)))
        j < length(order) && append!(parts, corner_ring(emb, p, last(poly[i])))
    end
    return parts
end

# The connectors across the face h right of the interface (p, q): the corners of h on either side
# of the exit (or, when the exit is next to p or q, the inverse of the corner it skips).
function connectors(emb::PlanarEmbedding, exits, p, q)
    h = face_of_dart(emb, q, p)
    isinner(h) || return Any[], Any[]
    wh = face_walk(emb, h, p)
    n = length(wh)
    j = exits[(p, q)]
    x(i) = wh[i + 1]
    left = j == 0 ?
        Any[(:inverse, [(x(1), h)], (interface_dart(emb, h, p, x(1)), interface_dart(emb, h, x(1), x(2))))] :
        corner_parts(h, [x(k) for k in 2:j])
    right = j == n - 2 ?
        Any[(:inverse, [(x(n - 2), h)], (interface_dart(emb, h, x(n - 2), q), interface_dart(emb, h, x(n - 3), x(n - 2))))] :
        corner_parts(h, [x(k) for k in (j + 1):(n - 3)])
    return left, right
end

# The whole connector across the face h right of (p, q), independent of the exit.
function full_connector(emb::PlanarEmbedding, h, p, q)
    isinner(h) || return Any[]
    wh = face_walk(emb, h, p)
    n = length(wh)
    n >= 4 && return corner_parts(h, wh[3:(n - 2)])
    return Any[(:inverse, [(wh[2], h)], (interface_dart(emb, h, p, wh[2]), interface_dart(emb, h, wh[2], q)))]
end

# The projector of face f at its edge {u, x}, on u's side, with its connector.
function exit_side(emb::PlanarEmbedding, exits, f, u, x)
    I = interface_dart(emb, f, u, x)
    left, right = connectors(emb, exits, I...)
    return u == first(I) ? Any[left; (:projector, I, :A)] : Any[right; (:projector, I, :B)]
end

# Everything around p except its two neighbours on f: the enlarged corner of (p, f).
function enlarged_recipe(emb::PlanarEmbedding, p, f)
    w = face_walk(emb, f, p)
    excl = (w[end], w[2])
    parts = far_chain(emb, p, excl)
    if all(u -> u in excl, first.(vertex_polygon(emb, p)))
        g = only(h for (_, h) in vertex_polygon(emb, p) if h != f)
        append!(parts, corner_ring(emb, p, g))
    end
    return Any[parts; (:site, p)]
end

# T(u → v) before projection, and its two side faces with u's neighbours on them.
function message_core(emb::PlanarEmbedding, u, v)
    poly = vertex_polygon(emb, u)
    z = length(poly)
    i = findfirst(==(v), first.(poly))
    fA, fB = last(poly[i]), last(poly[mod1(i - 1, z)])
    uA, uB = first(poly[mod1(i + 1, z)]), first(poly[mod1(i - 1, z)])
    excl = Any[v]
    isinner(fA) && push!(excl, uA)
    isinner(fB) && push!(excl, uB)
    parts = far_chain(emb, u, excl)
    if length(excl) == 3 && all(x -> x in excl, first.(poly))
        z == 2 && error("CTMRG: interior vertex $u of degree 2 is not supported")
        g = last(poly[mod1(i + 1, z)])
        if isinner(g)
            ring = corner_ring(emb, u, g)
            isempty(ring) && error("CTMRG: trivalent vertex $u whose far face is a triangle is not supported")
            legs = (interface_dart(emb, g, uA, other_neighbor(emb, g, uA, u)), interface_dart(emb, g, uB, other_neighbor(emb, g, uB, u)))
            push!(parts, (:inverse, [last(c) for c in ring], legs))
        end
    end
    return Any[parts; (:site, u)], fA, uA, fB, uB
end

function message_recipe(emb::PlanarEmbedding, exits, u, v)
    parts, fA, uA, fB, uB = message_core(emb, u, v)
    isinner(fA) && append!(parts, exit_side(emb, exits, fA, u, uA))
    isinner(fB) && append!(parts, exit_side(emb, exits, fB, u, uB))
    return parts
end

# A directional corner C(p, f) = T(p → a) · C(x, h), from its neighbour a on f (the previous one
# for :prev, the next for :next), before projection; returns the parts and a.
function directional_core(emb::PlanarEmbedding, p, f, mode::Symbol)
    w = face_walk(emb, f, p)
    a = mode === :next ? w[2] : w[end]
    h = face_of_dart(emb, reverse(interface_dart(emb, f, p, a))...)
    parts = Any[(:message, (p, a))]
    if isinner(h)
        wh = face_walk(emb, h, p)
        push!(parts, (:corner, (wh[2] == a ? wh[end] : wh[2], h)))
    end
    return parts, a
end

function corner_recipe(emb::PlanarEmbedding, exits, p, f, mode::Symbol)
    w = face_walk(emb, f, p)
    mode === :diag && return Any[exit_side(emb, exits, f, p, w[end]); (:enlarged, (p, f)); exit_side(emb, exits, f, p, w[2])]
    parts, a = directional_core(emb, p, f, mode)
    return Any[parts; exit_side(emb, exits, f, p, a)]
end

function block_recipe(emb::PlanarEmbedding, exits, modes, block)
    kind, key = block
    kind === :message && return message_recipe(emb, exits, key...)
    return corner_recipe(emb, exits, key..., modes[key])
end

# The environments of one vertex and of the two ends of an edge: the messages and corners that
# close them.
function vertex_environment(emb::PlanarEmbedding, v)
    parts = Any[]
    for (u, f) in vertex_polygon(emb, v)
        push!(parts, (:message, (u, v)))
        append!(parts, corner_ring(emb, v, f))
    end
    return parts
end

function edge_environment(emb::PlanarEmbedding, u, v)
    parts = Any[far_chain(emb, u, (v,)); far_chain(emb, v, (u,))]
    for (p, q) in ((u, v), (v, u))
        append!(parts, full_connector(emb, face_of_dart(emb, q, p), p, q))
    end
    return parts
end

# The closed regions: the whole network, with one vertex, one edge or one face explicit.
function vertex_region(emb::PlanarEmbedding, v)
    parts = Any[(:site, v); vertex_environment(emb, v)]
    return factor_enlarged(emb, parts, [(v, f) for (_, f) in vertex_polygon(emb, v) if isinner(f)])
end

function edge_region(emb::PlanarEmbedding, u, v)
    parts = Any[(:site, u); (:site, v); edge_environment(emb, u, v)]
    fs = filter(isinner, [face_of_dart(emb, u, v), face_of_dart(emb, v, u)])
    parts = factor_enlarged(emb, parts, [(u, f) for f in fs])
    return factor_enlarged(emb, parts, [(v, f) for f in fs])
end

function face_region(emb::PlanarEmbedding, f)
    w = face_vertices(emb, f)
    parts = Any[]
    for k in eachindex(w)
        p, q = w[k], w[mod1(k + 1, length(w))]
        push!(parts, (:enlarged, (p, f)))
        append!(parts, full_connector(emb, face_of_dart(emb, q, p), p, q))
    end
    return parts
end

# Replace the parts of the first enlarged corner among `candidates` that a recipe contains (as a
# multiset) by that corner as one nested part. The recipe's value is unchanged.
function factor_enlarged(emb::PlanarEmbedding, parts, candidates)
    for k in candidates
        rest = copy(parts)
        contained = all(enlarged_recipe(emb, k...)) do part
            i = findfirst(isequal(part), rest)
            isnothing(i) && return false
            deleteat!(rest, i)
            return true
        end
        contained && return Any[(:enlarged, k); rest]
    end
    return parts
end

# Kikuchi counting numbers: 1 for a face, 1 − (inner faces on it) for an edge and
# 1 + (inner faces at it) − (its degree) for a vertex.
counting_number(emb::PlanarEmbedding, v) = 1 + count(isinner ∘ last, vertex_polygon(emb, v)) - length(vertex_polygon(emb, v))
counting_number(emb::PlanarEmbedding, u, v) = 1 - isinner(face_of_dart(emb, u, v)) - isinner(face_of_dart(emb, v, u))

# The sites and blocks a recipe reads, each with the sign it enters with (an inverse subtracts;
# projectors are the identity at lossless χ, so they read nothing).
function recipe_reads(emb::PlanarEmbedding, parts)
    out = Tuple{Any, Int}[]
    for part in parts
        kind = first(part)
        if kind === :site || kind === :message || kind === :corner
            push!(out, (part, 1))
        elseif kind === :enlarged
            append!(out, recipe_reads(emb, enlarged_recipe(emb, part[2]...)))
        elseif kind === :inverse
            append!(out, [((:corner, k), -1) for k in part[2]])
        end
    end
    return out
end

# The signed vertex multiset a recipe covers, given what the blocks cover.
function recipe_coverage(emb::PlanarEmbedding, parts, coverage)
    r = Dict{Any, Int}()
    for (part, sign) in recipe_reads(emb, parts), (x, c) in (first(part) === :site ? (part[2] => 1,) : coverage[part])
        r[x] = get(r, x, 0) + sign * c
    end
    return filter!(((x, c),) -> !iszero(c), r)
end

# The blocks a recipe reads.
recipe_dependencies(emb::PlanarEmbedding, parts) = unique!(Any[part for (part, _) in recipe_reads(emb, parts) if first(part) !== :site])

const CORNER_MODES = (:diag, :prev, :next)

# The corner updates that make the blocks' dependencies acyclic with the fewest levels: a message
# sits one level above the blocks it reads, a corner one above the best of its candidate updates.
# Returns the modes, the blocks in level order and the depth, or `nothing` if no acyclic choice exists.
function acyclic_corner_modes(emb::PlanarEmbedding{V}, exits) where {V}
    messages = Any[(:message, d) for d in darts(emb)]
    corners = Any[(:corner, k) for k in corner_keys(emb)]
    reads = Dict(b => recipe_dependencies(emb, message_recipe(emb, exits, b[2]...)) for b in messages)
    options = Dict(b => [m => recipe_dependencies(emb, corner_recipe(emb, exits, b[2]..., m)) for m in CORNER_MODES] for b in corners)
    level = Dict{Any, Float64}(b => Inf for b in [messages; corners])
    modes = Dict{Tuple{V, Int}, Symbol}(b[2] => :diag for b in corners)
    above(ds) = 1 + maximum((level[d] for d in ds); init = 0.0)
    for _ in 1:(length(level) + 1)
        changed = false
        for b in messages
            l = above(reads[b])
            l < level[b] && (level[b] = l; changed = true)
        end
        for b in corners
            m, ds = argmin(((m, ds),) -> above(ds), options[b])
            l = above(ds)
            l < level[b] && (level[b] = l; modes[b[2]] = m; changed = true)
        end
        changed || break
    end
    all(isfinite, values(level)) || return nothing
    return modes, sort(collect(keys(level)); by = b -> level[b]), maximum(values(level))
end

# The rules are exact at lossless χ if, starting from empty blocks, the passes (one in dependency
# order, or enough Jacobi passes) leave every closed region covering each vertex exactly once.
function certify_rules(emb::PlanarEmbedding, exits, modes, order)
    blocks = block_keys(emb)
    coverage = Dict{Any, Dict{Any, Int}}(b => Dict{Any, Int}() for b in blocks)
    if isnothing(order)
        # Jacobi passes until the coverage settles; each carries it one block further, so this many
        # bound it (blocks can also keep rotating while every closed region stays exact)
        for _ in 1:(4 * (length(faces(emb)) + 4))
            next = Dict{Any, Dict{Any, Int}}(b => recipe_coverage(emb, block_recipe(emb, exits, modes, b), coverage) for b in blocks)
            next == coverage && break
            coverage = next
        end
    else
        for b in order
            coverage[b] = recipe_coverage(emb, block_recipe(emb, exits, modes, b), coverage)
        end
    end
    g = graph(emb)
    whole = Dict{Any, Int}(v => 1 for v in vertices(g))
    closed = Iterators.flatten((
        (vertex_region(emb, v) for v in vertices(g)),
        (edge_region(emb, src(e), dst(e)) for e in edges(g)),
        (face_region(emb, f) for f in eachindex(faces(emb))),
    ))
    return all(r -> recipe_coverage(emb, r, coverage) == whole, closed)
end

"""
    ctmrg_rules(emb::PlanarEmbedding; jacobi = false)

Choose how a `CTMRGCache` on the embedding `emb` updates its blocks: the exit of every ray, the
update of every corner and the order of the blocks, certified exact at lossless bond dimension.

# Arguments
- `emb::PlanarEmbedding`: The planar embedding of the network's graph.

# Keyword Arguments
- `jacobi::Bool = false`: Use diagonal corners under Jacobi passes, as the eigenvalue projector needs.
  Otherwise the corners are chosen so the blocks can be recomputed once per pass in dependency
  order, with the fewest levels, and Jacobi passes are the fallback.

# Returns
- A `CTMRGRules`. Errors if no rule set certifies on this graph.
"""
function ctmrg_rules(emb::PlanarEmbedding{V}; jacobi::Bool = false) where {V}
    candidates = []
    for rule in EXIT_RULES
        exits, next = exit_choices(emb, rule)
        lengths = ray_lengths(next)
        isnothing(lengths) || push!(candidates, (; exits, next, lengths))
    end
    best = nothing
    if !jacobi
        for c in candidates
            found = acyclic_corner_modes(emb, c.exits)
            isnothing(found) && continue
            modes, order, depth = found
            certify_rules(emb, c.exits, modes, order) || continue
            if isnothing(best) || depth < best.depth
                best = (; c..., modes, order, depth)
            end
        end
    end
    if isnothing(best)
        diagonal = Dict{Tuple{V, Int}, Symbol}(k => :diag for k in corner_keys(emb))
        i = findfirst(c -> certify_rules(emb, c.exits, diagonal, nothing), candidates)
        isnothing(i) || (best = (; candidates[i]..., modes = diagonal, order = nothing))
    end
    if isnothing(best)
        jacobi && error("CTMRG: this graph needs directional corners, so `projector = :eig` is not available")
        error("CTMRG: no certified update rules for this graph")
    end
    interfaces = sort(collect(keys(best.next)); by = I -> best.lengths[I])
    return CTMRGRules{V}(best.exits, best.next, interfaces, best.modes, best.order)
end
