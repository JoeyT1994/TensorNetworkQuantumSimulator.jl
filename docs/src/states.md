# Tensor Networks

TensorNetworkQuantumSimulator provides two core tensor network types `TensorNetworkState` and `TensorNetwork`, both subtypes of `AbstractTensorNetwork`. They share the same graph-based interface but differ in that the former is expected to have physical (site) indices and thus defines a wavefunction whilst the latter isn't and thus defines a scalar. See [Graphs](graphs.md) for how to define the underlying graph.

## TensorNetwork

A `TensorNetwork` is the simpler of the two types: a collection of ITensors living on the vertices of a graph, connected by shared (virtual) indices along the graph's edges. Operationally it is expected that this tensor network represents a single `scalar`, i.e. if all tensors were contracted together you would get a single scalar value.

```julia
using TensorNetworkQuantumSimulator
using ITensors: Index, random_itensor

i,j = Index(2, "i"), Index(2, "j")
t_a, t_b, t_c = random_itensor(i), random_itensor(i,j),random_itensor(j)
# Construct from a dictionary of tensors (graph is inferred from shared indices)
tn = TensorNetwork(Dictionary(["a", "b", "c"], [t_a, t_b, t_c]))

# Should return a scalar
z = contract(tn; alg = "exact")

# Random tensor network with connectivity specified by the graph and virtual index size specified bond dimension
g = named_grid((3,3))
tn = random_tensornetwork(Float64, g; bond_dimension = 4)

# Should return a scalar
z = contract(tn; alg = "exact")
```

`TensorNetwork` is a useful type for representing objects like classical partition functions or the solutions to counting problems where you don't need the concept of a physical site index.

## TensorNetworkState

A `TensorNetworkState` extends `TensorNetwork` by additionally storing **site indices** — a vector of them per vertex — that represent the physical (local) degrees of freedom. This is the type we expect users to use most often, as it represents a quantum state ``|\psi\rangle`` on a lattice. The typical use case is one site index per vertex, but there is full flexibility here.

### Product States

```julia
# All qubits (spin half) pointing up (bond dimension 1)
ψ = tensornetworkstate(v -> "↑", g, "S=1/2")

# Specify element type (ComplexF32 for GPU-friendly single precision)
ψ = tensornetworkstate(ComplexF32, v -> "↑", g, "S=1/2")

# Spatially varying initial state
ψ = tensornetworkstate(v -> isodd(sum(v)) ? "↑" : "↓", g, "S=1/2")

# Pass site indices you constructed directly
s = siteinds("S=1/2", g)
ψ = tensornetworkstate(ComplexF64, v -> "↑", g, s)
```

The function `v -> "↑"` maps each vertex to a local state label. The site type string (e.g. `"S=1/2"`) determines the local Hilbert space dimension. Product states have bond dimension 1.

### Random States

```julia
# Random state with specified bond dimension
ψ = random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = 4)

# With explicit site indices
s = siteinds("S=1/2", g)
ψ = random_tensornetworkstate(ComplexF64, g, s; bond_dimension = 8)
```

Random states are useful for testing and benchmarking.

## More Complex Site Index Structures

Each vertex in a `TensorNetworkState` stores a `Vector{Index}` of site indices, not just a single one. The standard constructors create one site index per vertex, but you can pass custom site index dictionaries to have multiple physical indices per vertex. This is useful for representing:

- **Operators in the Heisenberg picture and density matrices**: Each site has a ket and a bra index.
- **Mixed systems**: Vertices consisting of multiple spins or bosons grouped together.

### Custom Multi-Index States

To construct a state with custom site indices per vertex, build a `Dictionary` mapping vertices to `Vector{Index}` and pass it directly:

```julia
# Two spin-1/2 indices per vertex (e.g. for a density matrix or bilayer system)
s = Dictionary(
    collect(vertices(g)),
    [Index[Index(2, "S=1/2,ket"), Index(2, "S=1/2,bra")] for _ in vertices(g)]
)
ψ = random_tensornetworkstate(ComplexF64, g, s; bond_dimension = 4)

# Access the site indices at a vertex
siteinds(ψ, (1, 1))  # returns [Index(2, "S=1/2,ket"), Index(2, "S=1/2,bra")]
```

### Supported Site Types

The built-in `siteinds(sitetype, g)` function supports:

| Site type | Aliases | Local dimension |
|-----------|---------|----------------|
| `"S=1/2"` | `"qubit"`, `"SpinHalf"` | 2 |
| `"S=1"` | `"qutrit"`, `"Spin1"` | 3 |
| `"Pauli"` | | 4 |

However, by simply defining your own indices and your own gate types, you can build whatever tensor network you wish.

## Shared Interface

Both `TensorNetwork` and `TensorNetworkState` support useful operations for accessing information about them:

```julia
graph(tn)             # underlying NamedGraph
vertices(tn)          # all vertices
neighbors(tn, v)      # Neighboring vertices to v
edges(tn)             # all edges
tn[v]                 # directly access the ITensor at vertex v
maxvirtualdim(tn)     # maximum bond dimension across all edges
scalartype(tn)        # element type of the tensors (e.g. ComplexF64)
datatype(tn)          # storage type (e.g. Array, CuArray)
virtualinds(tn, e)    # Indices connecting the tensors at tn[src(e)], tn[dst(e)]
setindex_preserve!(tn, t, v) # Set the tensor at vertex v to t. Assumes graph structure unchanged
istree(tn)            # Is the effective graph a tree? If yes, stick to BP exclusively as the contraction backend.
```

Additionally, `TensorNetworkState` provides:

```julia
siteinds(ψ)           # dictionary mapping vertices to their physical (site) indices
```

When working with caches, extract the underlying network with:

```julia
ψ = network(cache)    # extract TensorNetworkState (or TensorNetwork) from a cache
```

## Truncation

Reduce the bond dimension of a `TensorNetworkState` via belief propagation guided truncation:

```julia
ψ_truncated = truncate(ψ; alg = "bp", maxdim = 4, cutoff = 1e-10)
```
