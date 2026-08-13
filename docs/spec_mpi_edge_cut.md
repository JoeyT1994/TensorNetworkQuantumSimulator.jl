# Spec: edge-cut MPI partitioning with distributed boundary simple update

## Problem

`BeliefPropagationCacheMPI` currently partitions by **duplicating vertices**: a `shared_vertex` lives
on two ranks with identical tensors. Every graph edge therefore lies inside exactly one rank, and a
gate crossing a boundary is applied by whichever rank happens to hold both endpoints, after which the
whole shared tensor is shipped to the peer (`communicate_factors!`).

Two costs: the duplicated vertex tensor is the largest object in the calculation (χ^deg), and shipping
it after every boundary gate moves that much data. We want each factor on exactly one rank, and the
boundary gate to exchange only the small QR factors.

## Design

**Partitioning becomes an edge cut.** Every vertex is owned by one rank. For each cut edge `{v, w}`
with `v` local and `w` remote, the local rank holds two messages: the **outgoing** one on `v → w`,
which it computes, and the **incoming** one on `w → v`, which it receives. `messages_graph` already
carries a ghost vertex per remote neighbour, so it needs no structural change — what changes is that
the outgoing cut edges join this rank's `edge_sequence`, since no one else can compute them.

- Caller-facing API: `shared_vertices::Dictionary{V, Tuple{Int32,Int32}}` becomes
  `vertex_ranks::Dictionary{V, Int32}` — the owner of every super-graph vertex. Affects
  `BeliefPropagationCacheMPI`, `apply_gates_mpi`, `inner_mpi`.
- The cut bond's `Index` cannot be derived locally: `virtualinds(tn, e) = commoninds(tn[src], tn[dst])`
  needs both endpoints. The constructor exchanges each endpoint's `bp_factors` index list once and
  intersects, which reproduces `virtualinds` across the cut for plain, state and form networks alike,
  and seeds the boundary messages with the matching `delta`.
- `communicate_factors!` and `should_apply_gate` are deleted. Nothing but messages and small QR
  factors ever crosses a rank boundary.
- `freenergy` inverts: vertices now partition cleanly, cut edges are held by both ranks and are
  counted on the lower-ranked one.
- `default_bp_maxiter` gains an `Allreduce`. Ranks whose local partitions differ in tree-ness would
  otherwise run different sweep counts and deadlock in `communicate_messages!`. Pre-existing bug.
- `apply_gates_mpi` converts a tuple-form circuit itself, against the super graph and the site indices
  of every rank (one broadcast per rank, `Index` objects only). Its old route through
  `apply_gates`'s conversion used the rank-local graph, which under vertex duplication still held both
  endpoints of a boundary gate and now does not.

**Boundary gate.** Both ranks absorb their own √env and factor locally. The rank owning `v⃗[1]` — a
choice every rank can make from the circuit alone, so no negotiation — receives the partner's `R`,
applies the gate and the SVD, and sends back the partner's updated factor with the new bond index and
the singular values. Both then absorb `env^{-1/2}` and rebuild. Peak memory per rank is 2× its own
vertex tensor; the partner's never appears.

The exchange carries an explicit `Index` vector alongside each array, so neither side has to guess the
other's layout or the truncated bond dimension. Distinct MPI tag from the message exchange.

**Low-degree vertices.** A vertex with too few environment legs gives a wide matrix and no thin Q, and
across a cut there is no serial routine to fall back on. Then the whole env-absorbed tensor plays the
part of `R` (Q is the identity) and the env legs come back attached to whatever the SVD returned. Same
protocol, same code path; those tensors are small, so nothing is lost. A degree-2 vertex — every
interior site of a 1-D chain — takes this path, so it is not a corner case.

## Implementation order

1. `simple_update_dense.jl`: restore `mul_strided_batched!` in `absorb_matrix!` (an in-progress
   `@tensor` experiment left the package unable to precompile), factor the shared env/layout setup out
   of `simple_update_dense`, add `absorb_boundary_in!`/`absorb_boundary_out!`.
2. `simple_update_dense_boundary`: the ITensor-level protocol.
3. `beliefpropagation_mpi.jl`: struct, constructor, `edge_sequence`, `freenergy`, `gate_role`,
   `apply_gate!` boundary branch, delete the shared-vertex machinery.
4. Rewrite `test/mpi_beliefpropagation_worker.jl` for disjoint partitions; add a 2-D grid case so the
   tall path is exercised (all 1-D cases are wide). Serial tests for the new array primitives.

## Deferred

- Local (both-endpoints-local) two-site gates still go through `simple_update_mpi`. Routing them
  through `simple_update_dense` is a separate change.
- `rescale_messages!` still covers local edges only.
- Load balancing: the compute rank is fixed by gate vertex order, not chosen by tensor size.
