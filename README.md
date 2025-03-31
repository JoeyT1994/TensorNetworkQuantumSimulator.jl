# TensorNetworkQuantumSimulator

A package for simulating quantum circuits with tensor networks (TNs) of near-arbitrary geometry. 

The main workhorses of the simulation are _belief propagation_ (BP) for gauging the TNs, _simple update_ for applying the gates, and BP with _loop corrections_ or _boundary MPS_ for estimating expectation values. This package is an experimental compilation of state-of-the-art features before some of them get integrated into [ITensorNetworks](https://github.com/ITensor/ITensorNetworks.jl) over time, with a focus on simulating quantum circuits.

The workflow is that you pass a `NamedGraph` object to the tensor network constructor, which is a graph describing the desired connectivity of your tensor network. Then you  define and subsequently apply the desired gates to the TN (truncating the bonds of the tensor network down to some desired threshold), estimating expectation values with any of the available techniques along the way. These techniques make different levels of approximation and have different control paramaters. The relevant literature describes these in more detail.

## Upcoming Features
- Gates beyond Pauli rotations, for example, Clifford gates.
- Applying gates to distant nodes of the TN via SWAP gates.
- Sampling bitstrings from loopy networks.

## Relevant Literature
- [Gauging tensor networks with belief propagation](https://www.scipost.org/SciPostPhys.15.6.222?acad_field_slug=chemistry)
- [Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308)
- [Loop Series Expansions for Tensor Networks](https://arxiv.org/abs/2409.03108)
- [Dynamics of disordered quantum systems with two- and three-dimensional tensor networks](https://arxiv.org/abs/2503.05693)

## Authors
The TN methods were developed and written by Joseph Tindall ([JoeyT1994](https://github.com/JoeyT1994)), a Postdoctoral Researcher at the Center for Computational Quantum Physics, Flatiron Institute NYC.

The package was developed with Manuel S. Rudolph ([MSRudolph](https://github.com/MSRudolph)), a PhD Candidate at EPFL, Switzerland, during a research stay at the Center for Computational Quantum Physics, Flatiron Institute NYC.

