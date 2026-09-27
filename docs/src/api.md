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
siteinds
```

## Classical Partition Functions

```@docs
ising_partitionfunction
```

## Gate Application

```@docs
apply_gates
apply_gates!
simple_update
full_update
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
```

## Message Passing

```@docs
update
update_iteration!
```

## Corner Transfer Matrix Environments

```@docs
CTMOptions
CTMEnvironmentCache
CTM3DEnvironmentCache
environments
options
vertex_environments
sweep_vertex_environments
region_lnZ
vertex_window
vertex_ring
region_ring
marginal_inconsistency
cvm_freenergy
```

## Classical Models in the Thermodynamic Limit

```@docs
InfiniteCTM2D
InfiniteCTM3D
site_ratio
site_environment
pair_ratio
correlation_length
ising2d_site
ising3d_site
ice_site
BoundaryPEPS
boundary_peps
boundary_peps_krylov
boundary_peps_stationary
```

## Utilities

```@docs
add
fidelity
optimise_p_q
```

## Custom Gate Definitions

```@docs
TensorNetworkQuantumSimulator.Tensors.register_op!
```

## Tensor Backend Internals

```@docs
TensorNetworkQuantumSimulator.TensorInterface.Algorithm
TensorNetworkQuantumSimulator.TensorInterface.factorize_svd
TensorNetworkQuantumSimulator.Tensors.fused_norm_message
TensorNetworkQuantumSimulator.Tensors.graded_space
```

## Index

```@index
```
