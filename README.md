# TensorNetworkQuantumSimulator

A package for simulating quantum circuits, quantum dynamics and equilibrium physics with tensor networks (TNs) of near-arbitrary geometry. This package is built on top of [ITensors](https://github.com/ITensor/ITensors.jl) and [NamedGraphs](https://github.com/ITensor/NamedGraphs.jl).

The main workhorses of the simulation are _belief propagation_ (BP) and the _Singular Value Decomposition_ for applying gates, and _BP_ or _boundary MPS_ for estimating expectation values and sampling. 

# How to use the package

The starting point of most calculations is that you will define a `NamedGraph` object `g` that encodes the geometry of your problem and how you want your tensor network structured. This is just a list of vertices and edges between pairs of vertices. Convenient constructors for a number of lattices, such as regular 1/2/3D grids, hexagonal lattices etc are provided, for instance:

```julia
julia> using TensorNetworkQuantumSimulator

julia> g = named_grid((5,5))
```

will construct you a 5 x 5 square lattice.


The vertices of this graph will then correspond to the qubits (or qutrits or bosonic lattice sites) in your setup and the edges the pairs of qubits that directly interact in your system (e.g. those that can have two-site gates applied to them). 

You can then initialise an `TensorNetworkState ψ` object (TNS) as your chosen starting state,  with user-friendly constructors for various different product states and physical bases. For instance, you could construct a random, spin 1/2, TNS via

```julia
julia> s = siteinds("S=1/2", g)
julia> ψ = random_tensornetworkstate(g, s; bond_dimension = 2)
```

or the bond dimension 1 all down state

```julia
julia> ψ = tensornetworkstate(v -> "↑", g, s)
```

This `TensorNetworkState` is a tensor network representation of your wavefunction, with the structure specified by the graph `g` of the state of your system. Most likely your TNS will represent the many-body wavefunction, but you can also work directly in the Heisenberg picture and have it represent the many-body operator instead by using a `"Pauli"` basis.

Now you define the _circuit_ which you will apply to your tensor network. A circuit, or layer of a circuit, simply takes the form `Vector{<:Tuple}` or `Vector{<:ITensor}` of a list of one or two-site gates to be applied in sequential order to the TNS. These gates can either be specified as a Tuple `(Gate_string::String, vertices_gate_acts_on::Vector, optional_gate_parameter::Number)` using the pre-defined gates listed below or as specific ITensors for more custom gate options. Your circuit may encode the Trotterised real- or imaginary-time dynamics of a Hamiltonian, a generic quantum circuit or something more exotic - you can build absolutely any list of one and two-site gates. For instance

```julia
julia> J, hx, hz, dt = 1.0, 2.5, 0.2, 0.01
julia> layer = []
julia> append!(layer, ("Rx", [v], 2 * hx * dt) for v in vertices(g))
julia> append!(layer, ("Rz", [v], 2 * hz * dt) for v in vertices(g))

julia> #For two site gates do an edge coloring to Trotterise the circuit
julia> ec = edge_color(g, 4)
julia> for colored_edges in ec
        append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
       end
```

Will build you the circuit corresponding to a 1st order Trotterization of the Ising model propagator with both longitudinal and transverse fields. The `edge_color` provides a nice way to identify groups of non-overlapping gates - which will help maintain efficiency during the simulation as non-overlapping gates are applied in parallel. This kind of grouping can be done on any bipartite lattice, where the second argument should be the co-ordination number (maximum degree of any of the vertices) of the graph.

You will now apply the desired gates to your TNS with the `apply_gates` function. Keyword arguments, i.e. the `apply_kwargs` should be passed to indicate the desired level of truncation of the virtual bonds to use when applying the circuits. At any point during the simulation, expectation values, samples or other information (e.g. overlaps) can be extracted from the `TensorNetworkState` with different options for the algorithm to use and the various hyperparameters that control their complexity. The docstrings and relevant literature describes these in more detail. Here we apply 50 layers of our Ising circuit:

```julia

julia> apply_kwargs = (; maxdim = 10, cutoff = 1e-10, normalize_tensors = true)
julia> circuit = reduce(vcat, [layer for i in 1:50])
julia> ψ, errors = apply_gates(circuit, ψ; apply_kwargs, verbose = false)
```
Now we can measure something on a vertex, with various backends supported

```julia

julia> sz_bp = expect(ψ, ("Z", (2,2)); alg = "bp")
#Boundary MPS only allowed on a planar graph
julia> sz_bmps = expect(ψ, ("Z", (2,2)); alg = "boundarymps", mps_bond_dimension = 10)
```

This code is a very powerful quantum simulation tool, but should not be treated as a black box. We therefore strongly encourage users to read the literature listed below and look through the tests [here](test/), examples [here](examples/) and even source code [here](src/) to learn how the code works in more detail so they can effectively deploy their own simulations and achieve results like those in the literature provided below.

## Supported Gates
Gates can take the form of ITensors or Tuples of length two or three, i.e.
`(gate_string::String, qubits_to_act_on::Union{Vector, NamedEdge})` or `(gate_string::String, qubits_to_act_on::Union{Vector, NamedEdge}, optional_parameter::Number)` depending on whether the gate type supports an optional parameter. The qubits_to_act on can be a vector of one or two vertices of the network where the gate acts. In the case of a two-qubit gate an edge of the network can also be passed. 

Pre-Defined One qubit gates (brackets indicates the optional rotation parameter which must be specified). These are consistent with the qiskit definitions.
```
- "X", "Y", "Z', "Rx"(θ), "Ry"(θ), "Rz"(θ), "CRx"(θ), "CRy"(θ), "CRz"(θ), "P", "H"
```

Pre-Defined Two qubit gates (brackets indicates the optional rotation angle parameter which must be specified). These are consistent with the qiskit definitions.
```
- "CNOT", "CX", "CY", "SWAP", "iSWAP", "√SWAP", "√iSWAP", "Rxx" (θ), "Ryy"(θ), "Rzz"(θ), "Rxxyy"(θ), "Rxxyyzz"(θ), "CPHASE"(θ)
```

If the user wants to instead define a custom gate, they can do so by creating the corresponding `ITensor` which acts on the physical indices for the qubit or pair of qubits they wish it to apply to.

## GPU Support
GPU support is enabled for all operations. Simply load in the relevant GPU Julia module (Metal.jl or CUDA.jl for example) and transform the tensor network state, beliefpropagationcache or boundarympscache with `ψ = CUDA.cu(ψ)`. Now when you perform the desired operation e.g. `sample_directly_certified(ψ. 10; norm_message_rank = 10)` it will run on GPU. Dramatic speedups are seen on NVidia GPUs for moderate to large bond dimension states. 
For instance, try something like the following and see for yourself (FYI you need an NVidia GPU to use CUDA)
```julia
julia> using TensorNetworkQuantumSimulator
julia> using CUDA

julia> g = named_grid((8,8))
julia> ψ_cpu = random_tensornetworkstate(ComplexF32, g; bond_dimension = 8)
julia> ψ_gpu = CUDA.cu(ψ_cpu)
julia> @time sz_bmps = expect(ψ_cpu, ("Z", (1,1)); alg = "boundarymps", mps_bond_dimension = 16)
julia> @time sz_bmps = expect(ψ_gpu, ("Z", (1,1)); alg = "boundarymps", mps_bond_dimension = 16)
```

## Relevant Literature
Helpful reading for understanding the machinery inside the library and the kind of simulations its been used for. 
- [Gauging tensor networks with belief propagation](https://www.scipost.org/SciPostPhys.15.6.222?acad_field_slug=chemistry)
- [Efficient Tensor Network Simulation of IBM’s Eagle Kicked Ising Experiment](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308)
- [Loop Series Expansions for Tensor Networks](https://arxiv.org/abs/2409.03108)
- [Dynamics of disordered quantum systems with two- and three-dimensional tensor networks](https://arxiv.org/abs/2503.05693)
- [Simulating and sampling quantum circuits with 2D tensor networks](https://arxiv.org/abs/2507.11424)

If you use this library in your research paper please cite, at minimum, either
- [Simulating and sampling quantum circuits with 2D tensor networks](https://arxiv.org/abs/2507.11424)
or 
- [Gauging tensor networks with belief propagation](https://www.scipost.org/SciPostPhys.15.6.222?acad_field_slug=chemistry)

## Upcoming Features
- Applying gates to distant nodes of the TN via SWAP gates.

## Authors and acknowledgements
The package was developed by Joseph Tindall ([JoeyT1994](https://github.com/JoeyT1994)), an Associate Research Scientist at the Center for Computational Quantum Physics, Flatiron Institute NYC and Manuel S. Rudolph ([MSRudolph](https://github.com/MSRudolph)), a PhD Candidate at EPFL, Switzerland, during a research stay at the Center for Computational Quantum Physics, Flatiron Institute NYC.

The package was strongly influenced by [ITensorNetworks](https://github.com/ITensor/ITensorNetworks.jl), a general tensor network package developed by Matt Fishman ([mtfishman](https://github.com/mtfishman)), Joseph Tindall ([JoeyT1994](https://github.com/JoeyT1994)) and others. The next generation of ITensorNetworks is currently being developed [here](https://github.com/ITensor/ITensorNetworksNext.jl). A quantum simulation package such as this will hopefully then be able to utilize many of its general features for working with tensor networks. 




