# Graphs

The starting point of most calculations is a `NamedGraph` from [NamedGraphs.jl](https://github.com/ITensor/NamedGraphs.jl) that encodes the geometry of your tensor network. Vertices correspond to tensors and edges correspond to pairs of tensors that share a bond in the tensor network.

## Built-in Lattice Constructors

Several common lattice geometries are provided out of the box:

```julia
g = named_grid((5, 5); periodic = true)         # 2D square lattice
g = named_grid((3, 3, 3); periodic = true)      # 3D cubic lattice
g = named_hexagonal_lattice_graph(4, 4; periodic = false)         # hexagonal lattice
g = heavy_hexagonal_lattice(5, 5)               # heavy-hexagonal lattice (IBM topology)
g = lieb_lattice(5, 5)                          # Lieb lattice
g = named_path_graph(10)                        # 1D chain (MPS)
g = named_comb_tree(4, 4)                       # comb tree
```

Vertices are typically integers or tuples. For instance, `named_grid((5, 5))` produces vertices `(1,1), (1,2), ..., (5,5)`, and `named_path_graph(10)` produces integer vertices `1, 2, ..., 10`.

## Custom Graphs

You can construct arbitrary graph topologies using [NamedGraphs.jl](https://github.com/ITensor/NamedGraphs.jl) directly:

```julia
using NamedGraphs

# Build a custom graph from an edge list
g = NamedGraph(["a", "b", "c", "d"])
add_edge(g, "a" => "b")
add_edge(g, "b" => "c")
add_edge(g, "c" => "d")
add_edge(g, "d" => "a")
```

Any graph that [NamedGraphs.jl](https://github.com/ITensor/NamedGraphs.jl) can represent can be used as the basis for a tensor network state.

## Graph Utilities

Standard graph queries are available and many algorithms defined in [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) are overloaded onto the `NamedGraph`:

```julia
vertices(g)          # all vertices
edges(g)             # all edges
neighbors(g, v)      # neighbors of vertex v
nv(g)                # number of vertices
degree(g, v)         # degree of vertex v
is_tree(g)           # check if the graph is a tree
```

## Edge Coloring

The `edge_color` function partitions the edges of the graph into groups of non-overlapping (independent) edges. This is used when building circuits to group two-site gates so that non-overlapping gates within each group can be applied without requiring intermediate BP cache updates:

```julia
ec = edge_color(g, 4)   # square lattice (max degree 4)
ec = edge_color(g, 3)   # hexagonal / heavy-hex (max degree 3)
ec = edge_color(g, 6)   # 3D cubic lattice (max degree 6)
```

The second argument should be the maximum vertex degree (coordination number) of the graph. The returned `ec` is a vector of vectors, where each inner vector contains non-overlapping edges.
