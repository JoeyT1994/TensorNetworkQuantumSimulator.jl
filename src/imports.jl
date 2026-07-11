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

# TNQS-owned operator / named-state system. `op`/`state` are called qualified (`Ops.op`,
# `Ops.state`) so `state` does not clash with the unrelated `ITensorBase.state`, and gates
# are registered by extending `Ops.op`. The types and string macros are imported for
# unqualified use (gate definitions dispatch on bare `OpName"…"` / `SiteType"…"`).
using .Ops: OpName, SiteType, @OpName_str, @SiteType_str
using ITensorBase: ITensorBase, Index, ITensor, commonind, commoninds, hascommoninds, name,
    noprime, plev, prime, replaceinds, settags, sim, tags, trycommonind, trynoncommonind, unnamed
import ITensorBase: uniqueinds
using TensorAlgebra: trivialrange, matricize, scalar
import TensorAlgebra: datatype
import Base: truncate
using TensorAlgebra.MatrixAlgebra: sqrth_invsqrth_safe, sqrth_safe
using MatrixAlgebraKit: project_hermitian

using Adapt: adapt

using TypeParameterAccessors: unspecify_type_parameters
