# Getting Started

This tutorial walks through a complete simulation: defining a lattice, preparing a state, building and applying a circuit, measuring observables, and drawing samples.

## Quick Start

```julia
using TensorNetworkQuantumSimulator

# 1. Define a graph (5x5 square lattice)
g = named_grid((5, 5))

# 2. Create an initial state (all spins up, ComplexF32 precision)
ψ = tensornetworkstate(ComplexF32, v -> "↑", g, "S=1/2")

# 3. Build a circuit layer (1st-order Trotter step of the transverse-field Ising model)
J, hx, dt = 1.0, 2.5, 0.01
layer = []
append!(layer, ("Rx", [v], 2 * hx * dt) for v in vertices(g))
ec = edge_color(g, 4)
for colored_edges in ec
    append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
end

# 4. Apply 50 layers of the circuit
apply_kwargs = (; maxdim = 10, cutoff = 1e-10, normalize_tensors = true)
circuit = reduce(vcat, [layer for _ in 1:50])
ψ, errors = apply_gates(circuit, ψ; apply_kwargs)

# 5. Measure an expectation value
sz_bp = expect(ψ, ("Z", (3, 3)); alg = "bp")
sz_bmps = expect(ψ, ("Z", (3, 3)); alg = "boundarymps", mps_bond_dimension = 10)
```

## Step-by-step Walkthrough

### 1. Define the Geometry

Every simulation starts with a graph. Vertices represent qubits (or other local degrees of freedom) and edges represent pairs of sites that directly interact:

```julia
g = named_grid((5, 5))  # 5x5 square lattice
```

See [Graphs](graphs.md) for the full list of built-in lattice constructors.

### 2. Create a State

A `TensorNetworkState` is a tensor network representation of your wavefunction on the graph:

```julia
ψ = tensornetworkstate(ComplexF32, v -> "↑", g, "S=1/2")
```

The first argument sets the element type. The function `v -> "↑"` maps each vertex to a local state label. `"S=1/2"` specifies the site type, which determines the local Hilbert space dimension (2 for spin-1/2).

### 3. Build a Circuit

Gates are specified as tuples `(gate_name, vertices, parameter)`. The `edge_color` function groups non-overlapping edges so that gates within each group can be applied without intermediate cache updates:

```julia
layer = []
append!(layer, ("Rx", [v], 2 * hx * dt) for v in vertices(g))
ec = edge_color(g, 4)
for colored_edges in ec
    append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
end
```

See [Gate Application](gates.md) for the full list of supported gates and details on edge coloring.

### 4. Apply Gates

The `apply_gates` function applies a sequence of gates using the simple update algorithm with SVD truncation. The `maxdim` parameter controls the maximum bond dimension:

```julia
apply_kwargs = (; maxdim = 10, cutoff = 1e-10, normalize_tensors = true)
circuit = reduce(vcat, [layer for _ in 1:50])
ψ, errors = apply_gates(circuit, ψ; apply_kwargs)
```

The returned `errors` vector contains the truncation error for each gate application.

### 5. Measure Observables

Use `expect` to compute expectation values. Several contraction algorithms are available:

```julia
# Belief propagation (fast, works on any graph)
sz_bp = expect(ψ, ("Z", (3, 3)); alg = "bp")

# Boundary MPS (more accurate, planar graphs only)
sz_bmps = expect(ψ, ("Z", (3, 3)); alg = "boundarymps", mps_bond_dimension = 10)
```

See [Expectation Values](expectation_values.md) for more options including norms, inner products, and reduced density matrices.

### 6. Sample Bitstrings

Draw samples from the Born distribution defined by the state:

```julia
bitstrings = sample(ψ, 100; alg = "boundarymps", norm_mps_bond_dimension = 10)
```

See [Sampling](sampling.md) for certified and directly certified sampling variants.

## Working with the BP Cache

For repeated operations (e.g. many Trotter steps), it is more efficient to wrap the state in a `BeliefPropagationCache` and reuse it:

```julia
ψ_bpc = BeliefPropagationCache(ψ)
ψ_bpc, errors = apply_gates(circuit, ψ_bpc; apply_kwargs)

# The cache already has messages — use them directly for expectations
sz = expect(ψ_bpc, [("Z", [(3, 3)])])

# Extract the underlying state when needed
ψ = network(ψ_bpc)
```

## Examples

See the [examples/](https://github.com/JoeyT1994/TensorNetworkQuantumSimulator.jl/tree/main/examples) directory for complete worked examples:

- **2D Ising dynamics** (`2dIsing_dynamics.jl`) — time evolution on a square lattice
- **3D Ising dynamics** (`3dIsing_dynamics.jl`) — time evolution on a periodic 3D cubic lattice
- **Heavy-hex Ising dynamics** (`heavyhexIsing_dynamics.jl`) — evolution, measurement, and sampling
- **Heisenberg picture** (`2dIsing_dynamics_Heisenbergpicture.jl`) — operator evolution in the Pauli basis
- **Boundary MPS** (`boundarymps.jl`) — comparing BP, boundary MPS, and exact contraction
- **Loop corrections** (`loopcorrections.jl`) — improving BP norm estimates
