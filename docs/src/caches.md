# Caches

Two central data structures used for efficient tensor network contraction in TensorNetworkQuantumSimulator.jl are the `BeliefPropagationCache` and the `BoundaryMPSCache` which both subtype `AbstractBeliefPropagationCache`. Both wrap a tensor network together with auxiliary _message_ tensors that encode approximate information about the contraction of the rest of the network as seen from each edge. Understanding these caches — what they store, how they are updated, and why they matter — is key to getting the most out of the package as an advanced user.

## Belief Propagation Cache

### What it stores

A `BeliefPropagationCache` holds:

1. **The tensor network** (`network`): a `TensorNetwork` or `TensorNetworkState` whose tensors live on the vertices of a graph.
2. **Messages** (`messages`): a `Dictionary` mapping each directed edge ``e = (u \to v)`` to a message tensor. These messages form a basis with which to approximate the contraction of the entire network down to the given edge. In the case of a `TensorNetworkState` $\vert \psi \rangle$ the messages reflect the contraction of the norm $\langle \psi \vert \psi \rangle$ around that edge.
3. **An edge sequence** (`edge_sequence`): the order in which edges are visited during each BP iteration. Computed once from a forest cover of the graph [[Tindall2023]](index.md#references) and reused across iterations. Custom orders may slightly affect performance or the fixed point found --- although often the algorithm is robust to these choices.
4. **A contraction sequence cache** (`contraction_sequences`): a `Dictionary` that caches the optimal tensor contraction orderings discovered during message updates. This avoids recomputing contraction sequences when the local tensor structure hasn't changed and many iterations of BP are run. The cache is populated during each call to `update` and cleared at the start and end of every update to prevent stale sequences from leaking across operations.

### How belief propagation works

BP is an iterative message-passing algorithm [[Alkabetz2021]](index.md#references), [[Tindall2023]](index.md#references) for tensor network contraction. On each iteration, every directed edge ``e = (u \to v)`` updates its message by contracting the local tensor(s) at vertex ``u`` with all incoming messages _except_ the one from ``v``:

```math
m_{u \to v}^{(t+1)} = \text{contract}\bigl(\text{factors}(u),\; \{m_{w \to u}^{(t)} : w \in \mathcal{N}(u) \setminus v\}\bigr)
```

On a tree graph, BP converges in a single sweep (messages propagate from leaves to root and back). On loopy graphs, messages are iterated until convergence or a maximum number of iterations is reached.

### How the BP cache is used effectively 

- **Gate application**: Wrapping your `TensorNetworkState` in a `BeliefPropagationCache` before calling `apply_gates` allows the cache to persist across Trotter steps and be used to perform intermediate operations such as measurements. If not, the `apply_gates` function will build the cache internally by wrapping your state, running the circuit and relevant BP updates, and then unwrapping it and pass the updated state back to you.
The simple update algorithm uses the BP messages as approximate environments when performing SVD truncation after each two-site gate, and the cache is updated intelligently between non-overlapping groups of gates.
- **Fast expectation values**: Once the messages have converged, computing ``\langle O \rangle`` only requires contracting the local tensors around the observable's support together with the relevant incoming messages — an operation whose cost is independent of system size and trivially parallelizable with a converged cache.
- **As a prerequisite for other operations**: If a `TensorNetworkState` is wrapped in a `BoundaryMPSCache` it is, by default, first transformed into the symmetric gauge [[Tindall2023]](index.md#references) which is helpful for numerical stability.

### Basic usage

```julia
using TensorNetworkQuantumSimulator

g = named_grid((5, 5))
ψ = random_tensornetworkstate(ComplexF64, g; bond_dimension = 4)

# Construct and converge the cache
ψ_bpc = BeliefPropagationCache(ψ)
ψ_bpc = update(ψ_bpc)   # runs BP to convergence

# Compute an expectation value directly from the cache
sz = expect(ψ_bpc, [("Z", [(3, 3)])])

# Extract the underlying state
ψ = network(ψ_bpc)
```

### Controlling the BP update

The `update` function accepts several keyword arguments:

```julia
ψ_bpc = update(ψ_bpc;
    maxiter = 50,          # maximum number of BP iterations (default: 25 on loopy graphs, 1 on trees)
    tolerance = 1e-10,     # convergence threshold on message change (default depends on scalar type)
    verbose = true,        # print convergence info
)
```

### Reusing the cache for gate application

```julia
ψ_bpc = BeliefPropagationCache(ψ)
ψ_bpc = update(ψ_bpc)

# Apply gates — the cache is reused and updated between gate groups
ψ_bpc, errors = apply_gates(circuit, ψ_bpc; maxdim = 10, cutoff = 1e-10)

# Messages are still available for immediate measurements
sz = expect(ψ_bpc, [("Z", [(3, 3)])])
```

This is preferable to passing a bare `TensorNetworkState` to `apply_gates`, as taking the BP-based expectation value afterwards would otherwise require running BP from scratch.

### Symmetric gauge

The symmetric gauge transforms the tensors in the network so that all BP messages become diagonal with positive entries, without changing the physical state [[Tindall2023]](index.md#references). This improves numerical conditioning and is used automatically before sampling and boundary MPS operations:

```julia
# On a state
ψ_symm = symmetric_gauge(ψ)

# Within a BP cache
ψ_bpc = BeliefPropagationCache(ψ)
ψ_bpc = update(ψ_bpc)
ψ_bpc = symmetric_gauge(ψ_bpc)
```

## Boundary MPS Cache

### What it stores

A `BoundaryMPSCache` holds:

1. **The tensor network** (`network`): a `TensorNetwork` or `TensorNetworkState` whose tensors live on the vertices of a graph.
2. **Messages** (`messages`): a `Dictionary` mapping directed _partition_ edges to MPS tensors. Unlike BP messages (which are uncorrelated with each other — i.e. they don't share common indices), boundary MPS messages do and the messages along a row / column cut of the network connect together to form an implicit MPS.
3. **A partitioned graph** (`supergraph`): a `PartitionedGraph` that encodes how the 2D lattice has been partitioned into rows or columns.
4. **Sorted edges** (`sorted_edges`): for each partition edge, the ordered list of underlying graph edges, which determines the MPS structure and is reused for efficiency.
5. **MPS bond dimension** (`mps_bond_dimension`): the maximum dimension of the index connecting up the message tensors into an implicit MPS, controlling the accuracy–cost tradeoff.
6. **A contraction sequence cache** (`contraction_sequences`): same role as in the BP cache.

### How boundary MPS contraction works

The boundary MPS algorithm contracts a 2D tensor network by sweeping across the partitions (columns or rows) sequentially. At each step, the boundary — an MPS approximation of everything contracted so far — is updated by fitting a new MPS to the application of the next column of tensors to the current MPS. The MPS bond dimension controls how accurately this boundary MPS is represented.

The key steps are:

1. **Partition** the 2D lattice into columns (or rows).
2. **Initialize** boundary MPS message tensors on the the edges of the partition graph.
3. **Sweep** across columns / rows, updating each boundary MPS by contracting it with the local column tensors and compressing back to the target bond dimension.
4. **Extract** expectation values by sandwiching the observable between left and right boundary MPS messages.

Because each sweep involves MPS operations (contraction + SVD truncation), the cost typically scales as ``O(R^3)`` in the MPS bond dimension ``R``, and results converge to the exact answer as ``R \to \infty``.  By default the state is partitioned by its columns, use `partition_by = "row"` when constructing the cache or calling `expect` to change it.

### When to use it

- **Accurate expectation values on planar graphs**: Boundary MPS provides controllably accurate results on 2D lattices, converging to exact values as the MPS bond dimension increases. Use it when BP isn't accurate enough and you have a loop, correlated tensor network.
- **Sampling**: The `sample` function with `alg = "boundarymps"` uses a boundary MPS cache internally to compute conditional probabilities for sequential qubit projection.
- **Benchmarking BP**: Compare BP and boundary MPS results to assess the quality of BP approximations on your specific problem.

### Basic usage

```julia
# From a TensorNetworkState (constructs cache internally)
sz = expect(ψ, ("Z", (3, 3)); alg = "boundarymps", mps_bond_dimension = 16)

# Or construct the cache explicitly for reuse
ψ_bmps = BoundaryMPSCache(ψ, 16)
ψ_bmps = update(ψ_bmps)
sz = expect(ψ_bmps, [("Z", [(3, 3)])])
```

### Requirements

The boundary MPS algorithm requires the graph to be **planar** and partitionable into rows or columns where each partition forms a line graph. This is satisfied by all 2D regular lattices (square, hexagonal, heavy-hex, Lieb) but not by 3D lattices or graphs with long-range edges. The vertices of your graph should be named as tuples `(row, column)` and the algorithm will error if it detects a graph that does not satisfy its requirements.

### Partitioning

By default, the partitioning direction is inferred from the observable layout:

- If all observable vertices share the same row index, the network is partitioned by rows.
- If they share the same column index, it is partitioned by columns.

You can also specify the partitioning explicitly:

```julia
sz = expect(ψ, ("Z", (3, 3)); alg = "boundarymps", mps_bond_dimension = 16, partition_by = "col")
```

## Choosing between BP and Boundary MPS

| | Belief Propagation | Boundary MPS |
|---|---|---|
| **Graph requirement** | Any | Planar |
| **Accuracy** | Exact on trees, approximate on loopy graphs | Controllably accurate, converges to exact |
| **Cost** | Low (linear in graph size per iteration) | Moderate (cubic in MPS bond dimension) |
| **Tuning parameter** | Number of iterations | MPS bond dimension |
| **Best for** | Fast approximate results, gate application environments | High-accuracy measurements, sampling |

In practice, a common workflow is to use BP for gate application (where speed matters and approximate environments suffice because you can compensate the BP approximation by cranking up the bond dimension) and boundary MPS for final measurements (where accuracy matters and you don't have a tree tensor network):

```julia
# Fast gate application with BP
ψ_bpc = BeliefPropagationCache(ψ)
ψ_bpc, errors = apply_gates(circuit, ψ_bpc; maxdim = 10)

# Accurate measurement with boundary MPS
ψ = network(ψ_bpc)
sz_bmps = expect(ψ, ("Z", (3, 3)); alg = "boundarymps", mps_bond_dimension = 20)

#Compare to BP
sz_bp = expect(ψ_bpc, ("Z", (3, 3)))
```
