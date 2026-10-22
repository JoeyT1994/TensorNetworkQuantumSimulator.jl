# Planar embeddings read off a straight-line drawing of the graph.
# Every dart u → v has one face on its left. Inner faces are traced counterclockwise, the outer
# face clockwise; around v the polygon alternates edge (v, uᵢ) and the face left of v → uᵢ.

"""
    PlanarEmbedding

A connected planar graph with a straight-line drawing: vertex positions, the counterclockwise
neighbour order around each vertex, and the faces. Inner faces are numbered `1:length(faces(emb))`,
the outer face is `0`. Build with [`planar_embedding`](@ref).
"""
struct PlanarEmbedding{V, G <: AbstractGraph{V}}
    graph::G
    positions::Dict{V, NTuple{2, Float64}}
    rotation::Dict{V, Vector{V}}           # neighbours of each vertex, counterclockwise
    faces::Vector{Vector{V}}               # inner faces as counterclockwise vertex walks
    outer::Vector{V}                       # the outer face's boundary walk (clockwise)
    dart_face::Dict{Tuple{V, V}, Int}      # face to the left of the dart u → v (0: outer)
end

graph(emb::PlanarEmbedding) = emb.graph
positions(emb::PlanarEmbedding) = emb.positions
faces(emb::PlanarEmbedding) = emb.faces
outer_face(emb::PlanarEmbedding) = emb.outer

# Vertex positions read off the vertex names, when every name is a 2-tuple of numbers
function default_positions(g::AbstractGraph)
    vs = collect(vertices(g))
    all(v -> v isa Tuple && length(v) == 2 && all(x -> x isa Real, v), vs) || throw(ArgumentError(
        "planar_embedding: the vertex names are not (x, y) coordinates; pass `positions` " *
            "(a dictionary vertex => (x, y)) explicitly"
    ))
    return Dict(v => (Float64(v[1]), Float64(v[2])) for v in vs)
end

"""
    planar_embedding(g; positions = default_positions(g))

Build the planar embedding of a connected graph from a straight-line drawing of it.
Errors if the drawing is not planar (crossing edges, a vertex on an edge, or coincident vertices).

# Arguments
- `g::AbstractGraph`: The connected graph.

# Keyword Arguments
- `positions`: Dictionary `vertex => (x, y)`. Defaults to the vertex names, which must then be 2-tuples of numbers.

# Returns
- `PlanarEmbedding`: The embedding.
"""
function planar_embedding(g::AbstractGraph{V}; positions = default_positions(g)) where {V}
    is_connected(g) || throw(ArgumentError("planar_embedding: the graph is not connected"))
    pos = Dict{V, NTuple{2, Float64}}(v => (Float64(positions[v][1]), Float64(positions[v][2])) for v in vertices(g))
    _check_drawing(g, pos)
    rotation = Dict{V, Vector{V}}(v => sort(collect(neighbors(g, v));
        by = u -> atan(pos[u][2] - pos[v][2], pos[u][1] - pos[v][1])) for v in vertices(g))
    # position of each neighbour in v's counterclockwise list
    slot = Dict{Tuple{V, V}, Int}((v, u) => i for v in vertices(g) for (i, u) in enumerate(rotation[v]))
    # Trace the face to the left of every dart: from u → v continue with v → w, w the neighbour of v
    # just before u in counterclockwise order (the sharpest left turn).
    walks = Vector{Vector{V}}()
    dart_walk = Dict{Tuple{V, V}, Int}()
    for v in vertices(g), u in rotation[v]
        haskey(dart_walk, (v, u)) && continue
        walk = V[]
        a, b = v, u
        while !haskey(dart_walk, (a, b))
            dart_walk[(a, b)] = length(walks) + 1
            push!(walk, a)
            nb = rotation[b]
            c = nb[mod1(slot[(b, a)] - 1, length(nb))]
            a, b = b, c
        end
        push!(walks, walk)
    end
    if nv(g) == 1
        return PlanarEmbedding{V, typeof(g)}(g, pos, rotation, Vector{V}[], V[only(vertices(g))], Dict{Tuple{V, V}, Int}())
    end
    # The outer walk runs clockwise (negative area; zero for a tree, whose one walk is the outer
    # face); every inner walk runs counterclockwise.
    areas = [_signed_area(w, pos) for w in walks]
    io = argmin(areas)
    all(i -> i == io || areas[i] > 0, eachindex(areas)) ||
        error("planar_embedding: more than one face walk is not counterclockwise")
    renumber = Dict(i => (i == io ? 0 : i - (i > io)) for i in eachindex(walks))
    inner = [walks[i] for i in eachindex(walks) if i != io]
    dart_face = Dict(d => renumber[i] for (d, i) in dart_walk)
    emb = PlanarEmbedding{V, typeof(g)}(g, pos, rotation, inner, walks[io], dart_face)
    # Euler: V − E + F = 2 counting the outer face
    nv(g) - ne(g) + length(inner) + 1 == 2 || error("planar_embedding: Euler characteristic check failed")
    return emb
end

_signed_area(walk, pos) = sum(
    pos[walk[i]][1] * pos[walk[mod1(i + 1, length(walk))]][2] -
        pos[walk[mod1(i + 1, length(walk))]][1] * pos[walk[i]][2] for i in eachindex(walk)
) / 2

_orient(p, q, r) = (q[1] - p[1]) * (r[2] - p[2]) - (q[2] - p[2]) * (r[1] - p[1])

# r on the closed segment p–q (assumes collinear)
_on_segment(p, q, r) = min(p[1], q[1]) <= r[1] <= max(p[1], q[1]) && min(p[2], q[2]) <= r[2] <= max(p[2], q[2])

# Planarity of the straight-line drawing: distinct points, no vertex inside an edge, no two edges
# crossing. Edges are bucketed on a uniform grid of cells about one typical edge length wide, so
# only nearby pairs are tested.
function _check_drawing(g::AbstractGraph{V}, pos) where {V}
    pts = collect(values(pos))
    length(unique(pts)) == length(pts) || throw(ArgumentError("planar_embedding: two vertices share a position"))
    es = [(src(e), dst(e)) for e in edges(g)]
    isempty(es) && return nothing
    len(e) = hypot(pos[e[1]][1] - pos[e[2]][1], pos[e[1]][2] - pos[e[2]][2])
    h = max(sum(len, es) / length(es), eps())
    cell(p) = (floor(Int, p[1] / h), floor(Int, p[2] / h))
    cells_of(p, q) = [(i, j) for i in min(cell(p)[1], cell(q)[1]):max(cell(p)[1], cell(q)[1])
                      for j in min(cell(p)[2], cell(q)[2]):max(cell(p)[2], cell(q)[2])]
    buckets = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (k, (a, b)) in enumerate(es), c in cells_of(pos[a], pos[b])
        push!(get!(() -> Int[], buckets, c), k)
    end
    vbucket = Dict{Tuple{Int, Int}, Vector{V}}()
    for v in vertices(g)
        push!(get!(() -> V[], vbucket, cell(pos[v])), v)
    end
    for (k, (a, b)) in enumerate(es)
        p, q = pos[a], pos[b]
        for c in cells_of(p, q)
            for v in get(vbucket, c, V[])
                (v == a || v == b) && continue
                r = pos[v]
                iszero(_orient(p, q, r)) && _on_segment(p, q, r) && throw(ArgumentError(
                    "planar_embedding: vertex $v lies on the edge $a – $b"))
            end
            for k2 in get(buckets, c, Int[])
                k2 <= k && continue
                c2, d2 = es[k2]
                length(unique((a, b, c2, d2))) == 4 || continue      # edges sharing an endpoint
                r, s = pos[c2], pos[d2]
                o1, o2, o3, o4 = _orient(p, q, r), _orient(p, q, s), _orient(r, s, p), _orient(r, s, q)
                if (o1 * o2 < 0) && (o3 * o4 < 0)
                    throw(ArgumentError("planar_embedding: the edges $a – $b and $c2 – $d2 cross"))
                end
            end
        end
    end
    return nothing
end

"""
    vertex_polygon(emb, v)

Get the polygon surrounding vertex `v` in the planar embedding.

# Arguments
- `emb::PlanarEmbedding`: The planar embedding.
- `v`: The vertex.

# Returns
- `Vector`: `(u, f)` pairs in counterclockwise order, one per incident edge, with `u` the neighbour
  across the edge and `f` the face between that edge and the next one (`0` for the outer face).
"""
function vertex_polygon(emb::PlanarEmbedding, v)
    return [(u, emb.dart_face[(v, u)]) for u in emb.rotation[v]]
end

"""
    face_vertices(emb, f)

The counterclockwise vertex walk of face `f` (`0`: the outer face, clockwise).
"""
face_vertices(emb::PlanarEmbedding, f::Integer) = f == 0 ? emb.outer : emb.faces[f]

"""
    face_centroid(emb, f)

The centroid of the vertices of inner face `f`.
"""
function face_centroid(emb::PlanarEmbedding, f::Integer)
    f == 0 && throw(ArgumentError("face_centroid: the outer face has no centroid"))
    w = emb.faces[f]
    return (sum(emb.positions[v][1] for v in w) / length(w), sum(emb.positions[v][2] for v in w) / length(w))
end

"""
    face_of_dart(emb, u, v)

The face to the left of the directed edge `u → v` (`0`: the outer face).
"""
face_of_dart(emb::PlanarEmbedding, u, v) = emb.dart_face[(u, v)]
