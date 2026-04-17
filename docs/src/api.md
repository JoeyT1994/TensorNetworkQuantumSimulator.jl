# API Reference

```@meta
CurrentModule = TensorNetworkQuantumSimulator
```

## Tensor Network States

```@docs
tensornetworkstate
random_tensornetworkstate
zerostate
paulitensornetworkstate
identitytensornetworkstate
```

## Gate Application

```@docs
apply_gates
simple_update
full_update
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

## Utilities

```@docs
paulirotationmatrix
safe_eigen
add
fidelity
optimise_p_q
```

## Custom Gate Definitions

```@docs
ITensors.SiteTypes.op(::ITensors.SiteTypes.OpName"Rxxyy", ::ITensors.SiteTypes.SiteType"S=1/2", ::ITensors.Index, ::ITensors.Index)
ITensors.SiteTypes.op(::ITensors.SiteTypes.OpName"Rxxyyzz", ::ITensors.SiteTypes.SiteType"S=1/2", ::ITensors.Index, ::ITensors.Index)
```

## Index

```@index
```
