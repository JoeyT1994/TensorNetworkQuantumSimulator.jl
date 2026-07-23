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

using OMEinsumContractionOrders: GreedyMethod, TreeSA

# ITensors is retained only as a matrix-data source for the operator/state
# catalogue (ITensorKit.op/state bridge into it) and for the Algorithm/OpName/
# SiteType dispatch machinery that the gate registry and message-update algorithms
# build on. All tensor operations go through ITensorKit (its `ITensor` is a
# `TensorMap`-backed wrapper).
using ITensors: ITensors, Algorithm, @Algorithm_str, OpName, @OpName_str, SiteType, @SiteType_str, truncate

using .ITensorKit: ITensorKit,
    Index, ITensor, itensor, random_itensor, onehot, delta, directsum,
    inds, plev, dim, space, dag, prime, noprime, sim, setprime,
    replaceind, replaceinds, swapind,
    commonind, commoninds, uniqueind, noncommonind, noncommoninds,
    unioninds, hascommoninds, hasind,
    combiner, combinedind, dense, denseblocks, hasqns,
    storagetype, scalar, array,
    map_diag, map_diag!, factorize_svd,
    op, state, apply, disable_warn_order, contraction_sequence

# Functions that TNQS adds methods to (extends) must be `import`ed, not `using`d.
import .ITensorKit: uniqueinds, datatype, scalartype, contract

using Adapt: adapt

using TypeParameterAccessors: unspecify_type_parameters
