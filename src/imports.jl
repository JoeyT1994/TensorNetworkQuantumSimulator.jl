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
    rename_vertices,
    edges,
    vertextype,
    add_vertex!,
    neighbors,
    leafless_edge_induced_subgraphs,
    src,
    dst,
    subgraph,
    is_connected,
    degree,
    add_edge,
    a_star,
    add_edge!,
    leaf_vertices,
    post_order_dfs_edges,
    add_vertex,
    rem_edge,
    rem_vertex,
    add_edges,
    rem_vertex!

using NamedGraphs: named_grid, named_hexagonal_lattice_graph, named_triangular_lattice_graph, named_comb_tree, named_path_graph
using Graphs: ne

# Tensor-level functions come from TensorInterface, never from a tensor library directly.
# `import` for the names this package extends with its own methods.
using .TensorInterface: Algorithm, @Algorithm_str
using .TensorInterface: inds, commonind, commoninds, unioninds, noncommonind, noncommoninds,
    hascommoninds, dim, plev, tags, isgraded, isdual, space,
    dag, prime, noprime, sim, replaceind, replaceinds,
    new_index, pad_index, from_array, random_tensor, onehot, delta, dense, combiner, combinedind,
    directsum, charge_sectors,
    scalar, apply, map_diag, map_diag!, factorize_svd, truncated_svd, gram,
    TruncationStrategy, truncation_strategy, kept_count,
    array, data, scale!, fused_norm_message, fused_norm_scalar
import .TensorInterface: contract, truncate, inner, uniqueinds, datatype, scalartype

using Adapt: adapt

using TypeParameterAccessors: unspecify_type_parameters
