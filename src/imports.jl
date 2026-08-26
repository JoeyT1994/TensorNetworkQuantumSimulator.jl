using LinearAlgebra
using StatsBase

using Dictionaries: Dictionary, set!

using Graphs: simplecycles_limited_length, has_edge, SimpleGraph, center, steiner_tree, is_tree, vertices, nv

using SimpleGraphConverter
using SimpleGraphAlgorithms: edge_color

using NamedGraphs
using NamedGraphs:
    AbstractNamedGraph,
    AbstractGraph,
    AbstractEdge,
    position_graph,
    rename_vertices,
    edges,
    vertextype,
    add_vertex!,
    neighbors,
    leafless_edge_induced_subgraphs
using NamedGraphs.GraphsExtensions:
    src,
    dst,
    subgraph,
    is_connected,
    degree,
    add_edge,
    a_star,
    add_edge!,
    edgetype,
    leaf_vertices,
    post_order_dfs_edges,
    add_vertex,
    rem_edge,
    rem_vertex,
    add_edges,
    rem_vertex!

using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph, named_comb_tree, named_path_graph

# All tensor-level verbs come from the TensorInterface seam (see tensor_interface.jl) —
# never from a tensor library directly. `import` (not `using`) for the names this package
# extends with its own methods.
using .TensorInterface: Algorithm, @Algorithm_str
using .TensorInterface: inds, commonind, commoninds, unioninds, noncommonind, noncommoninds,
    hascommoninds, dim, plev, tags,
    dag, prime, noprime, sim, replaceind, replaceinds,
    onehot, delta, combiner, combinedind, random_itensor, directsum,
    op, state,
    scalar, apply, map_diag, map_diag!, factorize_svd,
    array, data, new_index, from_array
import .TensorInterface: contract, truncate, inner, uniqueinds, datatype, scalartype
using .KTensors: KTensors, KIndex, KTensor, register_op!

using Adapt: adapt

using TypeParameterAccessors: unspecify_type_parameters
