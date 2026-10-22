@eval module $(gensym())
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: NamedEdge, NamedGraph, add_edge, face_centroid, face_of_dart,
    face_vertices, faces, ne, outer_face
using Test: @testset, @test, @test_throws

# The embedding is read off a straight-line drawing; these lattices name their vertices by
# positions (or the default positions of their names). Expected inner-face sizes per lattice.
const LATTICES = [
    ("square", named_grid((5, 4)), [4]),
    ("honeycomb", named_hexagonal_lattice_graph(3, 2), [6]),
    ("heavy-hex", heavy_hexagonal_lattice(2, 2), [12]),
    ("triangular", named_triangular_lattice_graph(4, 6), [3]),
    ("kagome", kagome_lattice(2, 2), [3, 6]),
    ("Lieb", lieb_lattice(5, 5), [8]),
    ("Penrose", penrose_tiling(3), [4]),
]

signed_area(pts) = sum(pts[i][1] * pts[mod1(i + 1, end)][2] - pts[mod1(i + 1, end)][1] * pts[i][2] for i in eachindex(pts)) / 2

@testset "Planar embedding: $label" for (label, g, sizes) in LATTICES
    emb = planar_embedding(g)
    fs = faces(emb)
    # Euler for a connected planar graph, counting the outer face
    @test nv(g) - ne(g) + length(fs) + 1 == 2
    @test issetequal(unique(length.(fs)), sizes)
    pos = TensorNetworkQuantumSimulator.positions(emb)
    # inner faces counterclockwise, the outer one clockwise
    @test all(f -> signed_area([pos[v] for v in f]) > 0, fs)
    @test signed_area([pos[v] for v in outer_face(emb)]) < 0
    # every dart lies on exactly one face walk, and consecutive vertices of a walk are its darts
    darts = Set{Tuple{Any, Any}}()
    for (k, f) in enumerate(fs), i in eachindex(f)
        d = (f[i], f[mod1(i + 1, length(f))])
        @test face_of_dart(emb, d...) == k
        push!(darts, d)
    end
    w = outer_face(emb)
    for i in eachindex(w)
        @test face_of_dart(emb, w[i], w[mod1(i + 1, length(w))]) == 0
    end
    @test length(darts) + length(w) == 2 * ne(g)
    # the polygon around each vertex: one (neighbour, face) slot per edge, the face following each
    # edge counterclockwise contains both the vertex and that neighbour
    for v in vertices(g)
        poly = vertex_polygon(emb, v)
        @test length(poly) == length(neighbors(g, v))
        @test issetequal(first.(poly), neighbors(g, v))
        @test all(((u, f),) -> v in face_vertices(emb, f) && u in face_vertices(emb, f), poly)
    end
    # face centroids lie inside convex faces (all of these lattices' faces are convex)
    for f in eachindex(fs)
        c = face_centroid(emb, f)
        walk = fs[f]
        @test all(i -> signed_area([pos[walk[i]], pos[walk[mod1(i + 1, length(walk))]], c]) > 0, eachindex(walk))
    end
end

@testset "Planar embedding: input checks" begin
    # crossing edges
    g = add_edge(add_edge(named_grid((2, 2)), NamedEdge((1, 1) => (2, 2))), NamedEdge((1, 2) => (2, 1)))
    @test_throws ArgumentError planar_embedding(g)
    # a vertex on an edge
    h = NamedGraph([(0, 0), (2, 0), (1, 0), (1, 1)])
    h = add_edge(add_edge(add_edge(h, NamedEdge((0, 0) => (2, 0))), NamedEdge((1, 0) => (1, 1))), NamedEdge((0, 0) => (1, 1)))
    @test_throws ArgumentError planar_embedding(h)
    # names that are not coordinates need positions; supplied positions are used
    p = named_path_graph(3)
    @test_throws ArgumentError planar_embedding(p)
    emb = planar_embedding(p; positions = Dict(1 => (0, 0), 2 => (1, 0), 3 => (2, 1)))
    @test isempty(faces(emb))
    @test length(outer_face(emb)) == 4                     # the walk 1 → 2 → 3 → 2 around the path
    # the same graph embedded differently: a square drawn as a kite is still one 4-face
    sq = named_grid((2, 2))
    emb2 = planar_embedding(sq; positions = Dict((1, 1) => (0, 0), (2, 1) => (2, 0), (1, 2) => (0, 2), (2, 2) => (5, 5)))
    @test length(faces(emb2)) == 1
end

end
