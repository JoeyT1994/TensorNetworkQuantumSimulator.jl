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

## All Other Documented Symbols

The following collects all remaining documented symbols not listed above.

```@autodocs
Modules = [TensorNetworkQuantumSimulator]
```

## Index

```@index
```
