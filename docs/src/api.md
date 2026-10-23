# API Reference

```@meta
CurrentModule = TensorNetworkQuantumSimulator
```

## Tensor Network States

```@docs
TensorNetworkState
tensornetworkstate
random_tensornetworkstate
zerostate
identity_tensornetworkstate
toriccode_groundstate
```

## Classical Partition Functions

```@docs
ising_partitionfunction
```

## Gate Application

```@docs
apply_gates
simple_update
full_update
circuit_from_qasm
circuit_from_qiskit
```

## Custom Gate Registration

```@docs
register_gate!
register_alias!
unregister_gate!
```

## Expectation Values and Observables

```@docs
expect
inner
norm_sqr
reduced_density_matrix
```

## Entanglement Entropy

```@docs
renyi_entropy
von_neumann_entanglement_entropy
second_renyi_entanglement_entropy
```

## Normalization and Truncation

```@docs
normalize
truncate
```

## Sampling

```@docs
sample
sample_directly_certified
sample_certified
```

## Graph Constructors

```@docs
heavy_hexagonal_lattice
lieb_lattice
kagome_lattice
penrose_tiling
```

## Planar Embeddings

```@docs
PlanarEmbedding
planar_embedding
vertex_polygon
```

## Message Passing

```@docs
CTMRGCache
update
update_iteration!
```

## Utilities

```@docs
safe_eigen
add
fidelity
optimise_p_q
TensorNetworkQuantumSimulator.TensorInterface.Algorithm
```

## Operators and States

The package's own operator and state library, for S=1/2 and S=1 sites. Parameterised operators
follow qiskit's conventions (e.g. `op("Rzz", s1, s2; θ)` is `exp(-iθ/2 Z⊗Z)`); multi-site matrices
take the first site as the most significant. Add your own with `register_op!`.

```@docs
register_op!
op
state
```

## Index

```@index
```
