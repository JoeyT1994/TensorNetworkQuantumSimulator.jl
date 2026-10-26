@eval module $(gensym())
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: faces, ne
using Test: @testset, @test, @test_throws

# Lattices whose vertex names are their positions, with the sizes of their inner faces.
@testset "Planar embedding: $label" for (label, g, sizes) in [
        ("square", named_grid((5, 4)), [4]),
        ("honeycomb", named_hexagonal_lattice_graph(3, 2), [6]),
        ("heavy-hex", heavy_hexagonal_lattice(2, 2), [12]),
        ("triangular", named_triangular_lattice_graph(4, 6), [3]),
        ("kagome", kagome_lattice(2, 2), [3, 6]),
        ("Lieb", lieb_lattice(5, 5), [8]),
        ("Penrose", penrose_tiling(3), [4]),
    ]
    fs = faces(planar_embedding(g))
    # Euler for a connected planar graph, counting the outer face
    @test nv(g) - ne(g) + length(fs) + 1 == 2
    @test issetequal(unique(length.(fs)), sizes)
end

@testset "Planar embedding: input checks" begin
    # crossing edges
    g = add_edge(add_edge(named_grid((2, 2)), NamedEdge((1, 1) => (2, 2))), NamedEdge((1, 2) => (2, 1)))
    @test_throws ArgumentError planar_embedding(g)
    # names that are not coordinates need positions
    p = named_path_graph(3)
    @test_throws ArgumentError planar_embedding(p)
    @test isempty(faces(planar_embedding(p; positions = Dict(1 => (0, 0), 2 => (1, 0), 3 => (2, 1)))))
end

end
