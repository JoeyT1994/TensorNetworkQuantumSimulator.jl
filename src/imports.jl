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
    edgeinduced_subgraphs_no_leaves,
    unique_cyclesubgraphs_limited_length
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
    decorate_graph_edges,
    add_vertex!,
    add_vertex,
    rem_edge,
    rem_vertex,
    add_edges,
    rem_vertices,
    rem_vertex!

using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph, named_comb_tree, named_path_graph

using TensorOperations

# Next-gen ITensor backend. The legacy `ITensors` / `ITensorMPS` API that TNQS was
# written against is provided by the compatibility layer in `itensor_compat/`
# (included right after this file), which re-publishes the legacy names in terms of
# `ITensorBase`, `TensorAlgebra`, and `MatrixAlgebraKit`.
using ITensorBase: ITensorBase, Index, ITensor

using Adapt: adapt

import PauliPropagation
const PP = PauliPropagation

using TypeParameterAccessors: unspecify_type_parameters
