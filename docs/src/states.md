# Tensor Networks

TensorNetworkQuantumSimulator provides two core types, `TensorNetworkState` and `TensorNetwork`, both subtypes of `AbstractTensorNetwork`. They share the same graph-based interface but differ in that `TensorNetworkState` carries physical (site) indices — representing a quantum wavefunction ``|\psi\rangle`` — while `TensorNetwork` does not, representing a scalar quantity. See [Graphs](graphs.md) for how to define the underlying graph.

## TensorNetwork

A `TensorNetwork` is the simpler of the two types: a collection of tensors living on the vertices of a graph, connected by shared (virtual) indices along the edges. Contracting all tensors produces a single scalar value.

```julia
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: new_index, random_tensor
using Dictionaries: Dictionary

i, j = new_index(2; tags = "i"), new_index(2; tags = "j")
t_a, t_b, t_c = random_tensor(Float64, i), random_tensor(Float64, i, j), random_tensor(Float64, j)
# Construct from a dictionary of tensors (graph is inferred from shared indices)
tn = TensorNetwork(Dictionary(["a", "b", "c"], [t_a, t_b, t_c]))

# Contracts to a scalar
z = contract(tn; alg = "exact")

# Random tensor network with graph-specified connectivity and given bond dimension
g = named_grid((3, 3))
tn = random_tensornetwork(Float64, g; bond_dimension = 4)

# Should return a scalar
z = contract(tn; alg = "exact")
```

`TensorNetwork` is a useful type for representing objects like classical partition functions or the solutions to counting problems where you don't need the concept of a physical site index.

The library ships with a built-in constructor for the classical Ising-model partition function on an arbitrary graph. Contracting it yields ``Z(β) = \sum_{\{σ\}} \exp(β \sum_{(u,v)} J_{uv}\, σ_u σ_v)``:

```julia
g = named_grid((4, 4))
β = 0.4
Z_tn = ising_partitionfunction(g, β)              # uniform J = 1
Z = contract(Z_tn; alg = "bp")                    # approximate via belief propagation

# Anisotropic couplings: pass a Dictionary keyed by edges
Js = Dictionary(edges(g), [isodd(src(e)[1]) ? 1.0 : 0.5 for e in edges(g)])
Z_tn = ising_partitionfunction(g, β; Js)
```

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

### Symmetric States

Pass `symmetry` to `siteinds` and the state is stored block-sparse over the conserved
sectors of an abelian symmetry — enforced structurally, so a symmetry-violating gate
raises an error rather than silently breaking conservation. Supported: `"Z2"`, `"U1"`,
and the fermionic gradings `"fZ2"`, `"fU1"`, `"fU1xU1"`.

```julia
# U(1)-symmetric spins (conserved magnetization)
s = siteinds("S=1/2", g; symmetry = "U1")
ψ = tensornetworkstate(ComplexF64, v -> iseven(sum(v)) ? "↑" : "↓", g, s)

# Custom sector layout (charge => dimension pairs) for the virtual structure
s = siteinds("S=1/2", g; symmetry = "Z2", sectors = [0 => 1, 1 => 1])
```

Everything downstream — gates, BP, boundary MPS, loop corrections, sampling — works
unchanged. Charged product states (nonzero total magnetization or particle number) are
handled automatically: site charges are routed through dimension-1 link components along
a spanning tree, and a nonzero total charge rides a dangling dimension-1 `"Charge"` leg.

### Fermionic States

Fermionic sites carry their statistics in the tensor backend: Jordan-Wigner strings
emerge from the graded algebra on any lattice, so gates and observables are local
operators with no string bookkeeping in user code.

```julia
# Spinless fermions; parity grading "fZ2" is the default,
# "fU1" additionally conserves particle number structurally
s = siteinds("Fermion", g; symmetry = "fU1")
ψ = tensornetworkstate(ComplexF64, v -> v ∈ occupied ? "Occ" : "Emp", g, s)

# Spinful (d = 4) sites: |0⟩, |↑⟩, |↓⟩, |↑↓⟩, with "fU1xU1" conserving N↑ and N↓
s = siteinds("SpinfulFermion", g; symmetry = "fU1xU1")
```

Fermionic gates include `"F_hop"` (hopping), `"F_pair"` (pairing), and `"F_phase"`;
observables include `"N"`, `"CdagC"` (any pair of vertices), `"hopping"`, and
`"pairing"`. See `examples/fermionic_dynamics.jl` for a quench benchmarked against the
exact free-fermion solution.

### Toric Code Ground State

For benchmarking and exploration of topologically ordered states, the library provides an exact bond-dimension-2 representation of Kitaev's toric code ground state on an `n × n` torus:

```julia
ψ = toriccode_groundstate(4)                      # 4 × 4 torus, S=1/2 sites by default

# Or supply your own site indices (one qubit per vertex of a periodic n × n grid)
g = named_grid((4, 4); periodic = true)
s = siteinds("S=1/2", g)
ψ = toriccode_groundstate(4, s)
```

The returned state lives on a periodic square lattice (qubits on vertices, not edges) and has bond dimension 2 by construction.

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
neighbors(tn, v)      # neighboring vertices of v
edges(tn)             # all edges
tn[v]                 # directly access the tensor at vertex v
maxvirtualdim(tn)     # maximum bond dimension across all edges
scalartype(tn)        # element type of the tensors (e.g. ComplexF64)
datatype(tn)          # storage type (e.g. Array, CuArray)
virtualinds(tn, e)    # Indices connecting the tensors at tn[src(e)], tn[dst(e)]
is_tree(tn)           # Is the effective graph a tree? If yes, stick to BP exclusively as the contraction backend.

# Internal, not exported — import explicitly as TensorNetworkQuantumSimulator.setindex_preserve!
setindex_preserve!(tn, t, v) # Set the tensor at vertex v to t. Assumes graph structure unchanged
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
