using LinearAlgebra
using StatsBase

using Dictionaries: Dictionary, set!

using Graphs: simplecycles_limited_length, has_edge, SimpleGraph, center, steiner_tree

using NamedGraphs
using NamedGraphs:
    AbstractNamedGraph,
    AbstractGraph,
    position_graph,
    rename_vertices,
    edges,
    vertextype,
    add_vertex!,
    neighbors,
    edgeinduced_subgraphs_no_leaves
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
    rem_vertices

using NamedGraphs.PartitionedGraphs:
    PartitionedGraphs,
    partitioned_vertices,
    partitionedges,
    unpartitioned_graph,
    which_partition

using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph

using SimpleGraphConverter: UG
using SimpleGraphAlgorithms
using SimpleGraphAlgorithms: edge_color

using TensorOperations

using ITensors
using ITensors: Index, ITensor, inner, itensor, apply, map_diag!, @Algorithm_str, scalar, @OpName_str, @SiteType_str

using ITensorNetworks
using ITensorNetworks:
    AbstractBeliefPropagationCache,
    AbstractITensorNetwork,
    AbstractIndsNetwork,
    Indices,
    BeliefPropagationCache,
    QuadraticFormNetwork,
    PartitionedGraph,
    IndsNetwork,
    ITensorNetwork,
    inner_network,
    PartitionVertex,
    PartitionEdge,
    Algorithm,
    VidalITensorNetwork,
    expect,
    default_cache_construction_kwargs,
    cache,
    norm_sqr_network,
    update,
    updated_message,
    set_message,
    siteinds,
    vertices,
    dim,
    apply,
    neighbor_vertices,
    environment,
    incoming_messages,
    partitionedge,
    messages,
    update_factor,
    partitioned_tensornetwork,
    tensornetwork,
    operator_vertex,
    ket_vertex,
    update_factors,
    scalar_factors_quotient,
    default_cache_update_kwargs,
    partitionedges,
    region_scalar,
    partitionvertices,
    partitioned_graph,
    powerset,
    boundary_partitionedges,
    message,
    factors,
    contraction_sequence,
    group,
    partitionedges,
    linkinds,
    generic_state,
    setindex_preserve_graph!,
    edge_tag,
    default_edge_sequence,
    default_bp_maxiter,
    default_message_update,
    # update_message,
    tree_orthogonalize,
    gauge_walk,
    maxlinkdim,
    default_cache_construction_kwargs




using ITensorNetworks.ITensorsExtensions: map_eigvals

using EinExprs: Greedy

import PauliPropagation
const PP = PauliPropagation
