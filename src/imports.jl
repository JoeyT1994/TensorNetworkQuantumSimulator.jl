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

# Legacy `ITensors` / `ITensorMPS` API, republished over the ITensorBase backend by the
# `ITensorsITensorBaseCompat` submodule (included before this file). It is aliased as
# `ITensors` so existing `ITensors.foo` call sites and `function ITensors.foo`
# extensions keep working unchanged, and its legacy names are imported for unqualified
# use. Each source file keeps its own `import`/`using` of this module exactly where it
# referenced `ITensors`, so the per-file imports mirror the original ITensors-based code.
import .ITensorsITensorBaseCompat as ITensors
using .ITensorsITensorBaseCompat:
    inds, commoninds, commonind, uniqueinds, noncommonind, noncommoninds, unioninds, hascommoninds, cat_inds,
    sim, dag, prime, noprime, replaceind, replaceinds, dim, swapind,
    itensor, random_itensor, scalar, delta, similar_map, onehot,
    qr, svd, svd_trunc, eigen, factorize, itensor_trunc,
    scalartype, datatype, array, data,
    denseblocks, dense, hasqns,
    contract, inner, apply, exp,
    directsum, disable_warn_order,
    Algorithm, @Algorithm_str,
    hastags,
    state, op, OpName, SiteType, @OpName_str, @SiteType_str
using ITensorBase: ITensorBase, Index, ITensor, name, plev, tags, unnamed
using TensorAlgebra: trivialrange
using TensorAlgebra.MatrixAlgebra: sqrth_invsqrth_safe, sqrth_safe
using MatrixAlgebraKit: project_hermitian

using Adapt: adapt

using TypeParameterAccessors: unspecify_type_parameters
