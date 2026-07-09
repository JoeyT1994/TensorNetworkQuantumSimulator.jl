using SimpleGraphAlgorithms: SimpleGraphAlgorithms

"""
    MultiGPUBeliefPropagationCache{V,B,P} <: AbstractBeliefPropagationCache{V}

Belief-propagation cache whose state is partitioned across several devices.

Wraps an ordinary `BeliefPropagationCache` (`bpc`) holding the
network, the per-edge message mirror, and contraction-sequence cache, together with
the device-resident partitions. The wrapped cache provides the standard
`AbstractBeliefPropagationCache` interface (messages, network, graph, …) and serves
as the scratch buffer for device-local gate application; the partitions hold the
authoritative device-resident tensors and messages during the BP sweep.

# Fields
- `bpc::B`: host/owner-side `BeliefPropagationCache` (interface + scratch).
- `partitions::Vector{P}`: per-device partition state; `P` is the device-backend
  partition type (`DevicePartition`, defined in `multigpu_cuda.jl`).
- `partition_map::Dictionary{V,Int}`: vertex → partition id (0-indexed).
- `edge_groups::Vector{Vector{NamedEdge}}`: edges grouped by proper edge color; all
  edges within a group can be updated in parallel without write conflicts.

"""
struct MultiGPUBeliefPropagationCache{V, B <: BeliefPropagationCache{V}, P} <:
       AbstractBeliefPropagationCache{V}
    bpc::B
    partitions::Vector{P}
    partition_map::Dictionary{V, Int}
    edge_groups::Vector{Vector{NamedEdge}}
end

# ── AbstractBeliefPropagationCache interface (forward onto the wrapped cache) ──

inner_cache(bpc::MultiGPUBeliefPropagationCache) = bpc.bpc
messages(bpc::MultiGPUBeliefPropagationCache) = messages(inner_cache(bpc))
network(bpc::MultiGPUBeliefPropagationCache) = network(inner_cache(bpc))
graph(bpc::MultiGPUBeliefPropagationCache) = graph(inner_cache(bpc))
contraction_sequences(bpc::MultiGPUBeliefPropagationCache) = contraction_sequences(inner_cache(bpc))

partitions(bpc::MultiGPUBeliefPropagationCache) = bpc.partitions
partition_map(bpc::MultiGPUBeliefPropagationCache) = bpc.partition_map
edge_groups(bpc::MultiGPUBeliefPropagationCache) = bpc.edge_groups

function Base.copy(bpc::MultiGPUBeliefPropagationCache)
    return MultiGPUBeliefPropagationCache(
        copy(inner_cache(bpc)),
        map(copy, partitions(bpc)),
        copy(partition_map(bpc)),
        copy(edge_groups(bpc)),
    )
end

function edge_scalar(bpc::MultiGPUBeliefPropagationCache, edge::AbstractEdge)
    return (message(bpc, edge) * message(bpc, reverse(edge)))[]
end

# ── Algorithmic defaults ──────────────────────────────────────────────────────

default_update_alg(::MultiGPUBeliefPropagationCache) = "multigpu_bp"
default_message_update_alg(::MultiGPUBeliefPropagationCache) = "contract"

default_multigpu_maxiter(bpc::MultiGPUBeliefPropagationCache) = default_bp_maxiter(graph(bpc))
default_multigpu_tolerance(bpc::MultiGPUBeliefPropagationCache) = default_tolerance(scalartype(network(bpc)))
default_multigpu_damping() = 1.0

function set_default_kwargs(alg::Algorithm"multigpu_bp", bpc::MultiGPUBeliefPropagationCache)
    maxiter = get(alg.kwargs, :maxiter, default_multigpu_maxiter(bpc))
    tolerance = get(alg.kwargs, :tolerance, default_multigpu_tolerance(bpc))
    α = get(alg.kwargs, :α, default_multigpu_damping())
    verbose = get(alg.kwargs, :verbose, false)
    enforce_hermiticity = get(alg.kwargs, :enforce_hermiticity, false)
    return Algorithm("multigpu_bp"; maxiter, tolerance, α, verbose, enforce_hermiticity)
end

# Used by the generic `apply_gates` driver to update the cache between gates.
default_bp_update_kwargs(bpc::MultiGPUBeliefPropagationCache) =
    (; maxiter = default_multigpu_maxiter(bpc), tolerance = default_multigpu_tolerance(bpc),
       α = default_multigpu_damping(), verbose = false)

# ── Edge coloring ─────────────────────────────────────────────────────────────

"""
    colored_edge_groups(g) -> Vector{Vector{NamedEdge}}

Group the edges of `g` into proper edge-color classes using
`SimpleGraphAlgorithms.edge_color`. Each returned group is a set of edges that
share no vertex, so all messages in a group can be updated concurrently.
"""
function colored_edge_groups(g::AbstractGraph)
    z = maximum(degree(g, v) for v in vertices(g))
    coloring = SimpleGraphAlgorithms.edge_color(g, z)
    return Vector{NamedEdge}[NamedEdge[e for e in grp] for grp in coloring]
end

# ── Graph partitioning ───────────────────────────────────────────────────────

"""
    partition_graph(g::AbstractGraph, n_parts::Integer; alg = "memory_balanced", kwargs...)
        -> Dictionary{vertex,Int}

Partition the vertices of `g` into `n_parts` blocks, returning a `Dictionary`
mapping each vertex to a 0-indexed partition id. The algorithm selects which
multi-GPU regime to optimize for:

- `"memory_balanced"` (default): equalize the per-block degree load (a device-memory
  proxy), spreading the high-degree bond-dimension hot spots evenly across devices.
  Prefer this when device memory is the binding constraint.
- `"min_cut"`: minimize inter-device communication by balancing vertex count only,
  letting the partitioner cut as few edges as possible. Prefer this when each device
  has ample memory.

Both accept `imbalance` (default `0.0`); keep it small so the balance constraint
stays tight. The implementations live in `multigpu_cuda.jl` (KaHyPar backend).
"""
function partition_graph(g::AbstractGraph, n_parts::Integer; alg = "memory_balanced", kwargs...)
    return partition_graph(Algorithm(alg; kwargs...), g, n_parts)
end

function partition_graph(alg::Algorithm, g::AbstractGraph, n_parts::Integer)
    return error("Unknown partitioning algorithm. Use \"memory_balanced\" or \"min_cut\".")
end
