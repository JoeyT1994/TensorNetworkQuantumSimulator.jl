```@meta
CurrentModule = TensorNetworkQuantumSimulator
```

# TensorNetworkQuantumSimulator.jl

A Julia package for simulating quantum circuits, quantum dynamics and equilibrium physics with tensor networks of near-arbitrary geometry. Built on top of [ITensors.jl](https://github.com/ITensor/ITensors.jl) and [NamedGraphs.jl](https://github.com/ITensor/NamedGraphs.jl).

![Overview of TensorNetworkQuantumSimulator.jl](mainfig.svg)

## Features

- **Tensor Networks of Arbitrary geometry**: 2D and 3D lattices (square, hexagonal, heavy-hex, Lieb, cubic), trees, and custom graphs via NamedGraphs.jl
- **Custom Site Types**: Multiple spin and bosonic modes possible on each tensor in the network. Completely flexible over where the physical degrees of freedom live in your TN.
- **Gate application**: Apply large numbers of gates seamlessly with control over truncation parameters via the simple update procedure using belief propagation computed environments. Extremely fast, simple and robust. Full update with boundary MPS environments also supported for planar graphs but, naturally, much slower so only useful if your bond dimensions are expected to remain low. Both real time and imaginary time evolution are supported.
- **Expectation values**: Belief propagation, boundary MPS and exact contraction backends for computing expectation values of multi-point observables.
- **Sampling**: Sample from planar tensor network states using Boundary MPS, with the mps bond dimension controlling the sample quality. Options to also compute the importance sample ratio $p(x)/q(x)$ to directly assess sample quality.
- **Operators**: Operator evolution in the Heisenberg picture and density matrix representation.
- **GPU support**: GPU acceleration via CUDA.jl or Metal.jl. CUDA.jl highly recommend for large bond dimension simulations as it can provide dramatic speedups that determine whether a simulation is even viable or not.
- **Arbitrary precision**: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`, and other numeric types

## Algorithm Overview

| Algorithm | Keyword | Graph Requirement | Cost | Accuracy |
|-----------|---------|-------------------|------|----------|
| Belief propagation | `alg = "bp"` | Any | Low | Exact on trees, approximate on loopy graphs |
| Boundary MPS | `alg = "boundarymps"` | Planar | Variable (generally scales cubically via `mps_bond_dimension`) | Controllably and will converge with increasing MPS bond dimension |
| Loop corrections | `alg = "loopcorrections"` | Any | Moderate | Systematic corrections to BP, accurate when correlations are exponentially decaying  |
| Exact contraction | `alg = "exact"` | Any (small systems only) | Exponential in system size | Exact |

## Installation

```julia
julia> using Pkg; Pkg.add("TensorNetworkQuantumSimulator")
```

## Documentation Outline

- **[Getting Started](getting_started.md)**: A complete walkthrough from lattice definition to defining and running circuits to measurement
- **[Graphs](graphs.md)**: Defining lattice geometries and graph operations
- **[Tensor Networks](states.md)**: `TensorNetwork` and `TensorNetworkState` types and their construction
- **[Gate Application](gates.md)**: Building circuits and applying gates
- **[Expectation Values](expectation_values.md)**: Computing observables, norms, and overlaps
- **[Sampling](sampling.md)**: Drawing bitstring samples from a planar tensor network state with optional certification
- **[Caches](caches.md)**: How the `BeliefPropagationCache` and `BoundaryMPSCache` work and why they matter
- **[Advanced Topics](advanced.md)**: GPU support, loop corrections, precision control, etc
- **[API Reference](api.md)**: Complete list of documented functions

## References

- **[Alkabetz2021]** N. Alkabetz and I. Arad, "Tensor networks contraction and the belief propagation algorithm," Physical Review Research **3**, 023073 (2021). [Link](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023073)
- **[Tindall2023]** J. Tindall and M. Fishman, "Gauging tensor networks with belief propagation," SciPost Physics **15**, 222 (2023). [Link](https://www.scipost.org/SciPostPhys.15.6.222)
- **[Tindall2024]** J. Tindall, M. Fishman, E. M. Stoudenmire, and D. Sels, "Efficient Tensor Network Simulation of IBM's Eagle Kicked Ising Experiment," PRX Quantum **5**, 010308 (2024). [Link](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308)
- **[Evenbly2026]** G. Evenbly, N. Pancotti, A. Milsted, J. Gray, and G. K.-L. Chan, "Loop Series Expansions for Tensor Networks," Physical Review Research **8**, 013245 (2026). [Link](https://arxiv.org/abs/2409.03108)
- **[Tindall2025]** J. Tindall, A. Mello, M. Fishman, M. Stoudenmire, and D. Sels, "Dynamics of disordered quantum systems with two- and three-dimensional tensor networks," arXiv:2503.05693 (2025). [Link](https://arxiv.org/abs/2503.05693)
- **[Rudolph2025]** M. S. Rudolph and J. Tindall, "Simulating and Sampling from Quantum Circuits with 2D Tensor Networks," arXiv:2507.11424 (2025). [Link](https://arxiv.org/abs/2507.11424)
- **[Ferris2021]** A. J. Ferris and G. Vidal, "Perfect sampling with unitary tensor networks," Physical Review B **104**, 235141 (2021). [Link](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.235141)

If you use this library in your research, please cite at minimum either:
- M. S. Rudolph and J. Tindall, "Simulating and Sampling from Quantum Circuits with 2D Tensor Networks," arXiv:2507.11424 (2025). [Link](https://arxiv.org/abs/2507.11424)

or

- J. Tindall and M. Fishman, "Gauging tensor networks with belief propagation," SciPost Physics **15**, 222 (2023). [Link](https://www.scipost.org/SciPostPhys.15.6.222)
