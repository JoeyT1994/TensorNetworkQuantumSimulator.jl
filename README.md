# TensorNetworkQuantumSimulator

A Julia package for simulating quantum circuits, quantum dynamics and equilibrium physics with tensor networks (TNs) of near-arbitrary geometry. Built on top of [ITensors](https://github.com/ITensor/ITensors.jl) and [NamedGraphs](https://github.com/ITensor/NamedGraphs.jl).

The main workhorses of the simulation are _belief propagation_ (BP) and the _singular value decomposition_ for applying gates, and _BP_ or _boundary MPS_ for estimating expectation values and sampling.

## Installation

```julia
julia> using Pkg; Pkg.add("TensorNetworkQuantumSimulator")
```

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

## Usage Guide

### Defining the Geometry

The starting point of most calculations is a `NamedGraph` that encodes the geometry of your tensor network. Vertices correspond to qubits (or qutrits, bosonic sites, etc.) and edges correspond to pairs of sites that directly interact. Built-in constructors are provided for common lattices:

```julia
g = named_grid((5, 5))                         # 2D square lattice
g = named_grid((3, 3, 3); periodic = true)      # 3D periodic cubic lattice
g = named_hexagonal_lattice_graph(4, 4)         # hexagonal lattice
g = heavy_hexagonal_lattice(5, 5)               # heavy-hexagonal lattice
g = lieb_lattice(5, 5)                          # Lieb lattice
```

### Constructing a Tensor Network State

A `TensorNetworkState` (TNS) is a tensor network representation of your wavefunction, with the structure specified by the graph `g`. Product states can be constructed using string labels or numerical vectors:

```julia
# All spins up (bond dimension 1)
ψ = tensornetworkstate(v -> "↑", g, "S=1/2")

# Random state with bond dimension 4 and ComplexF32 elements
ψ = random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = 4)
```

The optional first argument sets the element type (`Float64` by default). Use `ComplexF32` or `ComplexF64` for complex-valued states.

### Heisenberg Picture

You can also work directly in the Heisenberg picture, representing a many-body operator as a TNS in the `"Pauli"` basis. Each site encodes the coefficients of I, X, Y, Z:

```julia
# Start with the Z operator on a single site, identity elsewhere
ψ = paulitensornetworkstate(ComplexF32, v -> v == v0 ? "Z" : "I", g)
```

Gates are then applied as conjugation by unitaries, and observables are extracted via traces (see `examples/2dIsing_dynamics_Heisenbergpicture.jl`).

### Building Circuits

A circuit is a `Vector` of gates to be applied sequentially. Each gate is specified as a tuple `(gate_string, vertices, parameter)` or as a raw `ITensor`:

```julia
layer = []
# Single-site rotations
append!(layer, ("Rx", [v], 2 * hx * dt) for v in vertices(g))
append!(layer, ("Rz", [v], 2 * hz * dt) for v in vertices(g))

# Two-site gates, grouped by edge coloring for efficiency
ec = edge_color(g, 4)  # 4 = coordination number of the square lattice
for colored_edges in ec
    append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
end
```

The `edge_color` function identifies groups of non-overlapping edges. Non-overlapping gates within a group are applied in parallel during the simulation. The second argument should be the coordination number (maximum vertex degree) of the graph.

### Applying Gates

Apply gates to the TNS using `apply_gates`. The `apply_kwargs` control bond dimension truncation during the SVD:

```julia
apply_kwargs = (; maxdim = 10, cutoff = 1e-10, normalize_tensors = true)
ψ, errors = apply_gates(circuit, ψ; apply_kwargs)
```

You can also pass a `BeliefPropagationCache` directly to reuse BP messages between gate applications:

```julia
ψ_bpc = update(BeliefPropagationCache(ψ))
ψ_bpc, errors = apply_gates(circuit, ψ_bpc; apply_kwargs)
```

### Expectation Values

Compute expectation values with `expect`, choosing from several contraction algorithms:

```julia
# Belief propagation (works on any graph, fast, approximate)
sz = expect(ψ, ("Z", (3, 3)); alg = "bp")

# Boundary MPS (planar graphs only, more accurate, adjustable precision)
sz = expect(ψ, ("Z", (3, 3)); alg = "boundarymps", mps_bond_dimension = 16)

# Exact contraction (only feasible for small systems)
sz = expect(ψ, ("Z", (3, 3)); alg = "exact")

# Multi-site observables
szz = expect(ψ, ("ZZ", [(3, 3), (3, 4)]); alg = "bp")

# Multiple observables at once
observables = [("Z", [v]) for v in vertices(g)]
sz_all = expect(ψ, observables; alg = "bp")
```

If you already have an updated `BeliefPropagationCache` or `BoundaryMPSCache`, pass it directly to avoid redundant cache construction.

### Norms, Inner Products and Reduced Density Matrices

```julia
# Squared norm
nsq = norm_sqr(ψ; alg = "bp")

# Norm (via LinearAlgebra)
using LinearAlgebra
n = norm(ψ; alg = "bp")

# Normalize the state
ψ = normalize(ψ; alg = "bp")

# Inner product between two states
ip = ITensors.inner(ψ, ϕ; alg = "bp")

# Reduced density matrix on a set of vertices
ρ = reduced_density_matrix(ψ, [(3, 3)]; alg = "bp")
```

All of these support the same algorithm options as `expect`: `"exact"`, `"bp"`, `"boundarymps"`, and (for norms) `"loopcorrections"`.

### Sampling

Draw bitstring samples from the probability distribution defined by the squared amplitudes of the state:

```julia
# Basic sampling (returns bitstrings only)
bitstrings = sample(ψ, 100; alg = "boundarymps", norm_mps_bond_dimension = 10)

# Directly certified sampling (returns bitstrings with p(x)/q(x) estimates)
results = sample_directly_certified(ψ, 100; alg = "boundarymps", norm_mps_bond_dimension = 10)

# Certified sampling (independent contraction for certification)
results = sample_certified(ψ, 100; alg = "boundarymps", norm_mps_bond_dimension = 10)
```

BP-based sampling is also available via `alg = "bp"`.

### Truncation

Reduce the bond dimension of an existing TNS:

```julia
ψ_truncated = truncate(ψ; alg = "bp", maxdim = 4, cutoff = 1e-10)
```

## Supported Gates

Gates are specified as tuples of the form `(gate_string, vertices)` or `(gate_string, vertices, parameter)`. The vertices can be a vector of one or two graph vertices, or a `NamedEdge` for two-qubit gates. All parameterised gates follow the Qiskit convention.

**One-qubit gates** (parameter in parentheses where required):

| Gate | Parameter | Description |
|------|-----------|-------------|
| `"X"`, `"Y"`, `"Z"` | -- | Pauli gates |
| `"H"` | -- | Hadamard |
| `"P"` | phase | Phase gate |
| `"Rx"`, `"Ry"`, `"Rz"` | angle | Pauli rotation |
| `"CRx"`, `"CRy"`, `"CRz"` | angle | Controlled Pauli rotation (single-qubit part) |

**Two-qubit gates:**

| Gate | Parameter | Description |
|------|-----------|-------------|
| `"CNOT"`, `"CX"`, `"CY"` | -- | Controlled gates |
| `"SWAP"`, `"iSWAP"`, `"√SWAP"`, `"√iSWAP"` | -- | Swap variants |
| `"Rxx"`, `"Ryy"`, `"Rzz"` | angle | Pauli-pair rotations |
| `"Rxxyy"`, `"Rxxyyzz"` | angle | Multi-Pauli rotations |
| `"CPHASE"` | phase | Controlled phase |

Custom gates can be defined by constructing the corresponding `ITensor` acting on the physical indices of the target qubits.

## GPU Support

GPU support is enabled for all operations. Load the relevant Julia GPU package (e.g. CUDA.jl or Metal.jl) and transfer the state or cache:

```julia
using TensorNetworkQuantumSimulator
using CUDA

g = named_grid((8, 8))
ψ_cpu = random_tensornetworkstate(ComplexF32, g; bond_dimension = 8)
ψ_gpu = CUDA.cu(ψ_cpu)

@time expect(ψ_cpu, ("Z", (1, 1)); alg = "boundarymps", mps_bond_dimension = 16)
@time expect(ψ_gpu, ("Z", (1, 1)); alg = "boundarymps", mps_bond_dimension = 16)
```

Caches can also be transferred: `CUDA.cu(BeliefPropagationCache(ψ))`. Significant speedups are seen on NVIDIA GPUs at moderate to large bond dimensions.

## Algorithm Guide

| Algorithm | Keyword | Graph requirement | Cost | Accuracy |
|-----------|---------|-------------------|------|----------|
| Belief propagation | `alg = "bp"` | Any | Low | Exact on trees, approximate on loopy graphs |
| Boundary MPS | `alg = "boundarymps"` | Planar | Moderate (tuneable via `mps_bond_dimension`) | Converges to exact with increasing bond dimension |
| Loop corrections | `alg = "loopcorrections"` | Any | Moderate | Systematic corrections to BP |
| Exact contraction | `alg = "exact"` | Any (small systems) | Exponential | Exact |

## Examples

See the [examples/](examples/) directory for complete worked examples:
- **2D Ising dynamics** (`2dIsing_dynamics.jl`) -- time evolution on a square lattice with BP and boundary MPS measurements
- **3D Ising dynamics** (`3dIsing_dynamics.jl`) -- time evolution on a periodic 3D cubic lattice
- **Heavy-hex Ising dynamics** (`heavyhexIsing_dynamics.jl`) -- evolution, measurement and sampling on a heavy-hexagonal lattice
- **Heisenberg picture** (`2dIsing_dynamics_Heisenbergpicture.jl`) -- operator evolution in the Pauli basis
- **Boundary MPS** (`boundarymps.jl`) -- comparing BP, boundary MPS and exact expectation values
- **Loop corrections** (`loopcorrections.jl`) -- improving BP norm estimates with loop corrections

We encourage users to read the literature listed below and explore the [tests](test/) and [source code](src/) to learn how the package works in detail.

## Relevant Literature

Helpful reading for understanding the algorithms and the kind of simulations the library has been used for:
- J. Tindall and M. Fishman, "Gauging tensor networks with belief propagation," SciPost Physics **15**, 222 (2023). [Link](https://www.scipost.org/SciPostPhys.15.6.222)
- J. Tindall, M. Fishman, E. M. Stoudenmire, and D. Sels, "Efficient Tensor Network Simulation of IBM's Eagle Kicked Ising Experiment," PRX Quantum **5**, 010308 (2024). [Link](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308)
- G. Evenbly, N. Pancotti, A. Milsted, J. Gray, and G. K.-L. Chan, "Loop Series Expansions for Tensor Networks," Physical Review Research **8**, 013245 (2026). [Link](https://arxiv.org/abs/2409.03108)
- J. Tindall, A. Mello, M. Fishman, M. Stoudenmire, and D. Sels, "Dynamics of disordered quantum systems with two- and three-dimensional tensor networks," arXiv:2503.05693 (2025). [Link](https://arxiv.org/abs/2503.05693)
- M. S. Rudolph and J. Tindall, "Simulating and Sampling from Quantum Circuits with 2D Tensor Networks," arXiv:2507.11424 (2025). [Link](https://arxiv.org/abs/2507.11424)

If you use this library in your research, please cite at minimum either:
- M. S. Rudolph and J. Tindall, "Simulating and Sampling from Quantum Circuits with 2D Tensor Networks," arXiv:2507.11424 (2025). [Link](https://arxiv.org/abs/2507.11424)

or
- J. Tindall and M. Fishman, "Gauging tensor networks with belief propagation," SciPost Physics **15**, 222 (2023). [Link](https://www.scipost.org/SciPostPhys.15.6.222)

## Upcoming Features

- Applying gates to distant nodes of the TN via SWAP gates.
- Infinite tensor network states 
- Finite temperature and ground state examples

## Authors and Acknowledgements

The package was developed by Joseph Tindall ([JoeyT1994](https://github.com/JoeyT1994)), an Associate Research Scientist at the Center for Computational Quantum Physics, Flatiron Institute NYC, and Manuel S. Rudolph ([MSRudolph](https://github.com/MSRudolph)), a PhD Candidate at EPFL, Switzerland, during a research stay at the Center for Computational Quantum Physics, Flatiron Institute NYC.

The package was strongly influenced by [ITensorNetworks](https://github.com/ITensor/ITensorNetworks.jl), a general tensor network package developed by Matt Fishman ([mtfishman](https://github.com/mtfishman)), Joseph Tindall ([JoeyT1994](https://github.com/JoeyT1994)) and others. The next generation of ITensorNetworks is currently being developed [here](https://github.com/ITensor/ITensorNetworksNext.jl). A quantum simulation package such as this will hopefully then be able to utilize many of its general features for working with tensor networks.
