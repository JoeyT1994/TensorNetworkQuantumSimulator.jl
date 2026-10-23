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

_isinner(f) = f != 0

# f's counterclockwise walk, starting at p
function _walk(emb::PlanarEmbedding, f, p)
    w = face_vertices(emb, f)
    k = findfirst(==(p), w)
    return vcat(w[k:end], w[1:(k - 1)])
end

# the dart of the edge {a, b} with face f on its left: the key of the interface (f, {a, b})
function _interface(emb::PlanarEmbedding, f, a, b)
    face_of_dart(emb, a, b) == f && return (a, b)
    face_of_dart(emb, b, a) == f && return (b, a)
    return error("CTMRG: the edge $a – $b is not on face $f")
end

_interfaces(emb::PlanarEmbedding) = [(w[k], w[mod1(k + 1, length(w))]) for w in faces(emb) for k in eachindex(w)]
_darts(emb::PlanarEmbedding) = [d for e in edges(graph(emb)) for d in ((src(e), dst(e)), (dst(e), src(e)))]
_corner_keys(emb::PlanarEmbedding) = [(p, f) for (f, w) in enumerate(faces(emb)) for p in w]
_blocks(emb::PlanarEmbedding) = Any[[(:message, d) for d in _darts(emb)]; [(:corner, k) for k in _corner_keys(emb)]]

# u's neighbour on face h other than v
_other_neighbor(emb::PlanarEmbedding, h, u, v) = (w = _walk(emb, h, u); w[2] == v ? w[end] : w[2])

# Number of faces between each inner face and the outer face.
function _dual_distance(emb::PlanarEmbedding)
    d = fill(typemax(Int), length(faces(emb)))
    queue = Int[]
    for (f, w) in enumerate(faces(emb))
        if any(k -> !_isinner(face_of_dart(emb, w[mod1(k + 1, length(w))], w[k])), eachindex(w))
            d[f] = 1
            push!(queue, f)
        end
    end
    while !isempty(queue)
        f = popfirst!(queue)
        w = face_vertices(emb, f)
        for k in eachindex(w)
            h = face_of_dart(emb, w[mod1(k + 1, length(w))], w[k])
            if _isinner(h) && d[h] == typemax(Int)
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
function _exit_rule(emb::PlanarEmbedding, rule::Symbol)
    rule === :straight && return wh -> 1
    rule === :opposite && return wh -> (n = length(wh); n == 3 ? 1 : fld(n - 2, 2))
    d = _dual_distance(emb)
    distance(h) = _isinner(h) ? d[h] : 0
    return function (wh)
        n = length(wh)
        return argmin(j -> (distance(face_of_dart(emb, wh[mod1(j + 2, n)], wh[j + 1])), abs(2j - (n - 2)), j), 0:(n - 2))
    end
end
# The exit rules, in the order ctmrg_rules prefers them.
const _EXIT_RULES = (:straight, :opposite, :geodesic)

function _exits(emb::PlanarEmbedding{V}, rule::Symbol) where {V}
    exit = _exit_rule(emb, rule)
    exits = Dict{Tuple{V, V}, Int}()
    next = Dict{Tuple{V, V}, Union{Nothing, Tuple{V, V}}}()
    for (p, q) in _interfaces(emb)
        h = face_of_dart(emb, q, p)
        if _isinner(h)
            wh = _walk(emb, h, p)
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
function _ray_lengths(next::Dict{K}) where {K}
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

_corners(f, vs) = Any[(:corner, (x, f)) for x in vs]

# The corners of g around p, except those of p and of its two neighbours on g.
_ring(emb::PlanarEmbedding, p, g) = _isinner(g) ? _corners(g, _walk(emb, g, p)[3:(end - 1)]) : Any[]

# The messages into p from its neighbours outside `excl`, joined by the rings of the faces
# between consecutive ones.
function _far_chain(emb::PlanarEmbedding, p, excl)
    poly = vertex_polygon(emb, p)
    z = length(poly)
    keep = [i for i in 1:z if !(first(poly[i]) in excl)]
    isempty(keep) && return Any[]
    start = something(findfirst(i -> !(mod1(i - 1, z) in keep), keep), 1)
    order = [keep[mod1(start + j - 1, length(keep))] for j in eachindex(keep)]
    parts = Any[]
    for (j, i) in enumerate(order)
        push!(parts, (:message, (first(poly[i]), p)))
        j < length(order) && append!(parts, _ring(emb, p, last(poly[i])))
    end
    return parts
end

# The connectors across the face h right of the interface (p, q): the corners of h on either side
# of the exit (or, when the exit is next to p or q, the inverse of the corner it skips).
function _connectors(emb::PlanarEmbedding, exits, p, q)
    h = face_of_dart(emb, q, p)
    _isinner(h) || return Any[], Any[]
    wh = _walk(emb, h, p)
    n = length(wh)
    j = exits[(p, q)]
    x(i) = wh[i + 1]
    left = j == 0 ?
        Any[(:inverse, [(x(1), h)], (_interface(emb, h, p, x(1)), _interface(emb, h, x(1), x(2))))] :
        _corners(h, [x(k) for k in 2:j])
    right = j == n - 2 ?
        Any[(:inverse, [(x(n - 2), h)], (_interface(emb, h, x(n - 2), q), _interface(emb, h, x(n - 3), x(n - 2))))] :
        _corners(h, [x(k) for k in (j + 1):(n - 3)])
    return left, right
end

# The whole connector across the face h right of (p, q), independent of the exit.
function _full_connector(emb::PlanarEmbedding, h, p, q)
    _isinner(h) || return Any[]
    wh = _walk(emb, h, p)
    n = length(wh)
    n >= 4 && return _corners(h, wh[3:(n - 2)])
    return Any[(:inverse, [(wh[2], h)], (_interface(emb, h, p, wh[2]), _interface(emb, h, wh[2], q)))]
end

# The projector of face f at its edge {u, x}, on u's side, with its connector.
function _side(emb::PlanarEmbedding, exits, f, u, x)
    I = _interface(emb, f, u, x)
    left, right = _connectors(emb, exits, I...)
    return u == first(I) ? Any[left; (:projector, I, :A)] : Any[right; (:projector, I, :B)]
end

# Everything around p except its two neighbours on f: the enlarged corner of (p, f).
function _enlarged(emb::PlanarEmbedding, p, f)
    w = _walk(emb, f, p)
    excl = (w[end], w[2])
    parts = _far_chain(emb, p, excl)
    if all(u -> u in excl, first.(vertex_polygon(emb, p)))
        g = only(h for (_, h) in vertex_polygon(emb, p) if h != f)
        append!(parts, _ring(emb, p, g))
    end
    return Any[parts; (:site, p)]
end

# T(u → v) before projection, and its two side faces with u's neighbours on them.
function _message_core(emb::PlanarEmbedding, u, v)
    poly = vertex_polygon(emb, u)
    z = length(poly)
    i = findfirst(==(v), first.(poly))
    fA, fB = last(poly[i]), last(poly[mod1(i - 1, z)])
    uA, uB = first(poly[mod1(i + 1, z)]), first(poly[mod1(i - 1, z)])
    excl = Any[v]
    _isinner(fA) && push!(excl, uA)
    _isinner(fB) && push!(excl, uB)
    parts = _far_chain(emb, u, excl)
    if length(excl) == 3 && all(x -> x in excl, first.(poly))
        z == 2 && error("CTMRG: interior vertex $u of degree 2 is not supported")
        g = last(poly[mod1(i + 1, z)])
        if _isinner(g)
            ring = _ring(emb, u, g)
            isempty(ring) && error("CTMRG: trivalent vertex $u whose far face is a triangle is not supported")
            legs = (_interface(emb, g, uA, _other_neighbor(emb, g, uA, u)), _interface(emb, g, uB, _other_neighbor(emb, g, uB, u)))
            push!(parts, (:inverse, [last(c) for c in ring], legs))
        end
    end
    return Any[parts; (:site, u)], fA, uA, fB, uB
end

function _message_recipe(emb::PlanarEmbedding, exits, u, v)
    parts, fA, uA, fB, uB = _message_core(emb, u, v)
    _isinner(fA) && append!(parts, _side(emb, exits, fA, u, uA))
    _isinner(fB) && append!(parts, _side(emb, exits, fB, u, uB))
    return parts
end

# A directional corner C(p, f) = T(p → a) · C(x, h), from its neighbour a on f (the previous one
# for :prev, the next for :next), before projection; returns the parts and a.
function _directional_core(emb::PlanarEmbedding, p, f, mode::Symbol)
    w = _walk(emb, f, p)
    a = mode === :next ? w[2] : w[end]
    h = face_of_dart(emb, reverse(_interface(emb, f, p, a))...)
    parts = Any[(:message, (p, a))]
    if _isinner(h)
        wh = _walk(emb, h, p)
        push!(parts, (:corner, (wh[2] == a ? wh[end] : wh[2], h)))
    end
    return parts, a
end

function _corner_recipe(emb::PlanarEmbedding, exits, p, f, mode::Symbol)
    w = _walk(emb, f, p)
    mode === :diag && return Any[_side(emb, exits, f, p, w[end]); (:enlarged, (p, f)); _side(emb, exits, f, p, w[2])]
    parts, a = _directional_core(emb, p, f, mode)
    return Any[parts; _side(emb, exits, f, p, a)]
end

function _block_recipe(emb::PlanarEmbedding, exits, modes, block)
    kind, key = block
    kind === :message && return _message_recipe(emb, exits, key...)
    return _corner_recipe(emb, exits, key..., modes[key])
end

# The environments of one vertex and of the two ends of an edge: the messages and corners that
# close them.
function _vertex_environment(emb::PlanarEmbedding, v)
    parts = Any[]
    for (u, f) in vertex_polygon(emb, v)
        push!(parts, (:message, (u, v)))
        append!(parts, _ring(emb, v, f))
    end
    return parts
end

function _edge_environment(emb::PlanarEmbedding, u, v)
    parts = Any[_far_chain(emb, u, (v,)); _far_chain(emb, v, (u,))]
    for (p, q) in ((u, v), (v, u))
        append!(parts, _full_connector(emb, face_of_dart(emb, q, p), p, q))
    end
    return parts
end

# The closed regions: the whole network, with one vertex, one edge or one face explicit.
function _vertex_region(emb::PlanarEmbedding, v)
    parts = Any[(:site, v); _vertex_environment(emb, v)]
    return _factor_enlarged(emb, parts, [(v, f) for (_, f) in vertex_polygon(emb, v) if _isinner(f)])
end

function _edge_region(emb::PlanarEmbedding, u, v)
    parts = Any[(:site, u); (:site, v); _edge_environment(emb, u, v)]
    fs = filter(_isinner, [face_of_dart(emb, u, v), face_of_dart(emb, v, u)])
    parts = _factor_enlarged(emb, parts, [(u, f) for f in fs])
    return _factor_enlarged(emb, parts, [(v, f) for f in fs])
end

function _face_region(emb::PlanarEmbedding, f)
    w = face_vertices(emb, f)
    parts = Any[]
    for k in eachindex(w)
        p, q = w[k], w[mod1(k + 1, length(w))]
        push!(parts, (:enlarged, (p, f)))
        append!(parts, _full_connector(emb, face_of_dart(emb, q, p), p, q))
    end
    return parts
end

# Replace the parts of the first enlarged corner among `candidates` that a recipe contains (as a
# multiset) by that corner as one nested part. The recipe's value is unchanged.
function _factor_enlarged(emb::PlanarEmbedding, parts, candidates)
    for k in candidates
        rest = copy(parts)
        contained = all(_enlarged(emb, k...)) do part
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
_counting_number(emb::PlanarEmbedding, v) = 1 + count(_isinner ∘ last, vertex_polygon(emb, v)) - length(vertex_polygon(emb, v))
_counting_number(emb::PlanarEmbedding, u, v) = 1 - _isinner(face_of_dart(emb, u, v)) - _isinner(face_of_dart(emb, v, u))

# The signed vertex multiset a recipe covers, given what the blocks cover (projectors are the
# identity at lossless χ, so they cover nothing).
function _coverage(emb::PlanarEmbedding, parts, coverage)
    r = Dict{Any, Int}()
    add!(x, c) = (r[x] = get(r, x, 0) + c)
    for part in parts
        kind = first(part)
        if kind === :site
            add!(part[2], 1)
        elseif kind === :message || kind === :corner
            foreach(((x, c),) -> add!(x, c), coverage[part])
        elseif kind === :enlarged
            foreach(((x, c),) -> add!(x, c), _coverage(emb, _enlarged(emb, part[2]...), coverage))
        elseif kind === :inverse
            for k in part[2], (x, c) in coverage[(:corner, k)]
                add!(x, -c)
            end
        end
    end
    return filter!(((x, c),) -> !iszero(c), r)
end

# The blocks a recipe reads.
function _dependencies(emb::PlanarEmbedding, parts)
    deps = Any[]
    for part in parts
        kind = first(part)
        (kind === :message || kind === :corner) && push!(deps, part)
        kind === :enlarged && append!(deps, _dependencies(emb, _enlarged(emb, part[2]...)))
        kind === :inverse && append!(deps, [(:corner, k) for k in part[2]])
    end
    return unique!(deps)
end

const _CORNER_MODES = (:diag, :prev, :next)

# The corner updates that make the blocks' dependencies acyclic with the fewest levels: a message
# sits one level above the blocks it reads, a corner one above the best of its candidate updates.
# Returns the modes, the blocks in level order and the depth, or `nothing` if no acyclic choice exists.
function _acyclic_modes(emb::PlanarEmbedding{V}, exits) where {V}
    messages = Any[(:message, d) for d in _darts(emb)]
    corners = Any[(:corner, k) for k in _corner_keys(emb)]
    reads = Dict(b => _dependencies(emb, _message_recipe(emb, exits, b[2]...)) for b in messages)
    options = Dict(b => [m => _dependencies(emb, _corner_recipe(emb, exits, b[2]..., m)) for m in _CORNER_MODES] for b in corners)
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
function _certify(emb::PlanarEmbedding, exits, modes, order)
    blocks = _blocks(emb)
    coverage = Dict{Any, Dict{Any, Int}}(b => Dict{Any, Int}() for b in blocks)
    if isnothing(order)
        # each pass carries coverage one block further; this many crosses any face structure
        for _ in 1:(4 * (length(faces(emb)) + 4))
            coverage = Dict{Any, Dict{Any, Int}}(b => _coverage(emb, _block_recipe(emb, exits, modes, b), coverage) for b in blocks)
        end
    else
        for b in order
            coverage[b] = _coverage(emb, _block_recipe(emb, exits, modes, b), coverage)
        end
    end
    g = graph(emb)
    whole = Dict{Any, Int}(v => 1 for v in vertices(g))
    closed = Iterators.flatten((
        (_vertex_region(emb, v) for v in vertices(g)),
        (_edge_region(emb, src(e), dst(e)) for e in edges(g)),
        (_face_region(emb, f) for f in eachindex(faces(emb))),
    ))
    return all(r -> _coverage(emb, r, coverage) == whole, closed)
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
    for rule in _EXIT_RULES
        exits, next = _exits(emb, rule)
        lengths = _ray_lengths(next)
        isnothing(lengths) || push!(candidates, (; exits, next, lengths))
    end
    best = nothing
    if !jacobi
        for c in candidates
            found = _acyclic_modes(emb, c.exits)
            isnothing(found) && continue
            modes, order, depth = found
            _certify(emb, c.exits, modes, order) || continue
            if isnothing(best) || depth < best.depth
                best = (; c..., modes, order, depth)
            end
        end
    end
    if isnothing(best)
        diagonal = Dict{Tuple{V, Int}, Symbol}(k => :diag for k in _corner_keys(emb))
        i = findfirst(c -> _certify(emb, c.exits, diagonal, nothing), candidates)
        isnothing(i) || (best = (; candidates[i]..., modes = diagonal, order = nothing))
    end
    if isnothing(best)
        jacobi && error("CTMRG: this graph needs directional corners, so `projector = :eig` is not available")
        error("CTMRG: no certified update rules for this graph")
    end
    interfaces = sort(collect(keys(best.next)); by = I -> best.lengths[I])
    return CTMRGRules{V}(best.exits, best.next, interfaces, best.modes, best.order)
end
