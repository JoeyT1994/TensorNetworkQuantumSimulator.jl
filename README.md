# TensorNetworkQuantumSimulator

A package for simulating quantum circuits and quantum dynamics with tensor networks (TNs) of near-arbitrary geometry. This package is built on top of [ITensors](https://github.com/ITensor/ITensors.jl) and [NamedGraphs](https://github.com/ITensor/NamedGraphs.jl).

The main workhorses of the simulation are _belief propagation_ (BP) and the _Singular Value Decomposition_ for applying gates, and _BP_ or _boundary MPS_ for estimating expectation values and sampling. 

# Workflow

The starting point of most calculations is that you will define a `NamedGraph` object `g` that encodes the geometry of your problem and how you want your tensor network structured. This is just a list of vertices and edges between pairs of vertices. Convenient constructors for a number of lattices, such as regular 1/2/3D grids, hexagonal lattices etc are provided.
The vertices of this graph will then correspond to the qubits (or qutrits or bosonic lattice sites) in your setup and the edges the pairs of qubits that directly interact in your system (e.g. those that can have two-site gates applied to them). 

You can then initialise an `TensorNetworkState ψ` object (TNS) as your chosen starting state,  with user-friendly constructors for various different product states. This is a tensor network representation of your state, with the structure specified by the graph `g` of the state of your system. Most likely your TNS will represent the many-body wavefunction, but you can also work directly in the Heisenberg picture and have it represent the many-body operator instead by using a `"Pauli"` basis.

Now you define the _circuit_ which you will apply to your tensor network. A circuit, or layer of a circuit, simply takes the form `Vector{<:Tuple}` or `Vector{<:ITensor}` of a list of one or two-site gates to be applied in sequential order to the TNS. These gates can either be specified as a Tuple `(Gate_string::String, vertices_gate_acts_on::Vector, optional_gate_parameter::Number)` using the pre-defined gates listed below or as specific ITensors for more custom gate options. Your circuit may encode the Trotterised real- or imaginary-time dynamics of a Hamiltonian, a generic quantum circuit or something more exotic - you can build absolutely any list of one and two-site gates.    

You will now apply the desired gates to your TNS with the `apply_gates` function. Keyword arguments, i.e. the `apply_kwargs` should be passed to indicate the desired level of truncation of the virtual bonds to use when applying the circuits. At any point during the simulation, expectation values, samples or other information (e.g. overlaps) can be extracted from the `TensorNetworkState` with different options for the algorithm to use and the various hyperparameters that control their complexity. The docstrings and relevant literature describes these in more detail. 

This code is a very powerful quantum simulation tool, but should not be treated as a black box. We therefore strongly encourage users to read the literature listed below and look through the examples in the folder [here](examples/) to learn how the code works in more detail so they can effectively deploy their own simulations.

## Supported Gates
Gates can take the form of ITensors or Tuples of length two or three, i.e.
`(gate_string::String, qubits_to_act_on::Union{Vector, NamedEdge})` or `(gate_string::String, qubits_to_act_on::Union{Vector, NamedEdge}, optional_parameter::Number)` depending on whether the gate type supports an optional parameter. The qubits_to_act on can be a vector of one or two vertices of the network where the gate acts. In the case of a two-qubit gate an edge of the network can also be passed. 

Pre-Defined One qubit gates (brackets indicates the optional rotation parameter which must be specified). These are consistent with the qiskit definitions.
- "X", "Y", "Z', "Rx" (θ), "Ry" (θ), "Rz" (θ), "CRx" (θ), "CRy" (θ), "CRz" (θ), "P", "H"

Pre-Defined Two qubit gates (brackets indicates the optional rotation angle parameter which must be specified). These are consistent with the qiskit definitions.
- "CNOT", "CX", "CY", "SWAP", "iSWAP", "√SWAP", "√iSWAP", "Rxx" (θ), "Ryy" (θ), "Rzz" (θ), "Rxxyy" (θ), "Rxxyyzz" (θ), "CPHASE" (θ)

If the user wants to instead define a custom gate, they can do so by creating the corresponding `ITensor` which acts on the physical indices for the qubit or pair of qubits they wish it to apply to.

## GPU Support
GPU support is enabled for all operations. Simply load in the relevant GPU Julia module (Metal.jl or CUDA.jl for example) and transform the tensor network state, beliefpropagationcache or boundarympscache with `ψ = CUDA.cu(ψ)`. Now when you perform the desired operation e.g. `sample_directly_certified(ψ. 10; norm_message_rank = 10)` it will run on GPU. Dramatic speedups are seen on NVidia GPUs for moderate to large bond dimension states. 

## Relevant Literature
- [Simulating and sampling quantum circuits with 2D tensor networks](https://arxiv.org/abs/2507.11424)
- [Gauging tensor networks with belief propagation](https://www.scipost.org/SciPostPhys.15.6.222?acad_field_slug=chemistry)
- [Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308)
- [Loop Series Expansions for Tensor Networks](https://arxiv.org/abs/2409.03108)
- [Dynamics of disordered quantum systems with two- and three-dimensional tensor networks](https://arxiv.org/abs/2503.05693)

## Upcoming Features
- Applying gates to distant nodes of the TN via SWAP gates.

## Authors
The package was developed by Joseph Tindall ([JoeyT1994](https://github.com/JoeyT1994)), an Associate Research Scientist at the Center for Computational Quantum Physics, Flatiron Institute NYC and Manuel S. Rudolph ([MSRudolph](https://github.com/MSRudolph)), a PhD Candidate at EPFL, Switzerland, during a research stay at the Center for Computational Quantum Physics, Flatiron Institute NYC.

