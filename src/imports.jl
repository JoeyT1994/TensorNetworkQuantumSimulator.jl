using LinearAlgebra
using Random: Random
using StatsBase

using Dictionaries: Dictionary, set!

using Graphs: has_edge, SimpleGraph, center, steiner_tree, is_tree, vertices, nv, ne

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

# Tensor-level functions come from TensorInterface, never from a tensor library directly.
# `import` for the names this package extends with its own methods.
using .TensorInterface: Algorithm, @Algorithm_str
using .TensorInterface: inds, commonind, commoninds, unioninds, noncommonind, noncommoninds,
    hascommoninds, dim, plev, tags, isgraded,
    dag, prime, noprime, sim, replaceind, replaceinds,
    new_index, from_array, random_tensor, onehot, delta, dense, combiner, combinedind, directsum,
    projector, charge_sectors, pad_index,
    scalar, apply, map_diag, map_diag!, factorize_svd, truncated_svd, truncation_strategy, gram,
    array, data, scale!, has_closure_gauge, site_op, site_state
import .TensorInterface: contract, truncate, inner, uniqueinds, datatype, scalartype

# Networks, their BP messages and norm networks come from ITensorNetworksNext; its name maps
# (e.g. `ITensorBase.prime` in `normnetwork`) and message wrappers are ITensorBase objects.
using ITensorBase: ITensorBase
using ITensorNetworksNext: ITensorNetworksNext as ITNN, ITensorNetwork
using DataGraphs: DataGraphs

using Adapt: adapt
using MatrixAlgebraKit: defaulttol

using TypeParameterAccessors: unspecify_type_parameters
