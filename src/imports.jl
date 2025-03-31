using LinearAlgebra

using Dictionaries: Dictionary, set!

using Graphs: simplecycles_limited_length, has_edge, SimpleGraph, center

using NamedGraphs
using NamedGraphs: AbstractNamedGraph, AbstractGraph, position_graph, rename_vertices, edges, vertextype, add_vertex!, neighbors
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
    post_order_dfs_edges

using NamedGraphs.PartitionedGraphs: PartitionedGraphs, partitioned_vertices, partitionedges, unpartitioned_graph

using SimpleGraphConverter: UG
using SimpleGraphAlgorithms
using SimpleGraphAlgorithms: edge_color

using ITensors
using ITensors: Index, ITensor, inner, itensor, apply, map_diag!

using ITensorNetworks
using ITensorNetworks:
    AbstractITensorNetwork,
    AbstractIndsNetwork,
    Indices,
    BeliefPropagationCache,
    QuadraticFormNetwork,
    PartitionedGraph,
    IndsNetwork,
    ITensorNetwork,
    PartitionVertex,
    PartitionEdge,
    Algorithm,
    VidalITensorNetwork,
    norm_sqr_network,
    update,
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
    default_message_update,
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
    # insert_linkinds,
    generic_state,
    setindex_preserve_graph!,
    edge_tag,
    default_edge_sequence,
    default_bp_maxiter,
    # update_message,
    tree_orthogonalize,
    gauge_walk,
    maxlinkdim




using ITensorNetworks.ITensorsExtensions: map_eigvals

using EinExprs: Greedy

import PauliPropagation
const PP = PauliPropagation

