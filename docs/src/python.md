# Using from Python

TensorNetworkQuantumSimulator.jl can be called from Python through [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/), without writing a Julia script. Together with OpenQASM import (see [Circuits from Qiskit](qiskit.md)), a circuit built in Python qiskit can be simulated from the same Python session.

## Installation

Install juliacall with pip. It downloads Julia itself if none is installed:

```bash
pip install juliacall
```

Then add TensorNetworkQuantumSimulator.jl to the Julia environment that juliacall manages. This only needs to be done once:

```python
import juliapkg
juliapkg.add("TensorNetworkQuantumSimulator", "4de3b72a-362e-43dd-83ff-3f381eda9f9c")
juliapkg.resolve()
```

The first `import` of the package in a session takes some time while Julia compiles it; later calls are fast.

## A circuit from qiskit

Build the circuit in qiskit, hand it to Julia as an OpenQASM 2 string, and call the package's functions through `jl`:

```python
from juliacall import Main as jl
from qiskit import QuantumCircuit, qasm2

jl.seval("using TensorNetworkQuantumSimulator")

qc = QuantumCircuit(4)
qc.h(0)
for q in range(3):
    qc.cx(q, q + 1)
qc.rz(0.3, 3)

circuit = jl.circuit_from_qasm(qasm2.dumps(qc))
g = jl.build_graph_from_circuit(circuit)

psi0 = jl.tensornetworkstate(jl.ComplexF64, jl.seval('v -> "↑"'), g, "S=1/2")
psi = jl.apply_gates(circuit, psi0, apply_kwargs=jl.seval("(; maxdim = 16, cutoff = 1e-12)"))

observables = jl.seval('[("Z", [1]), ("ZZ", [1, 4]), ("XXXX", [1, 2, 3, 4])]')
values = [complex(v) for v in jl.expect(psi, observables, alg="bp")]
print(values)  # [0j, (1+0j), (0.955336489125606+0j)]
```

The circuit prepares a four-qubit GHZ state and rotates the last qubit, so ``\langle Z_1 \rangle = 0``, ``\langle Z_1 Z_4 \rangle = 1`` and ``\langle XXXX \rangle = \cos(0.3)``.

A few things to keep in mind when crossing between the two languages:

- **Qubit numbering.** Qiskit's qubit `k` is vertex `k + 1`, since Julia counts from 1.
- **Julia values.** Strings, numbers and booleans convert automatically. Anything Julia-specific, such as an anonymous function, a named tuple of keyword arguments or a vector of `(name, vertices)` tuples, is easiest to write as Julia code with `jl.seval`.
- **Results.** Julia arrays come back as array-like objects that can be iterated, indexed or passed to `numpy.asarray`. Expectation values are complex numbers.
- **Keyword arguments** are passed as Python keyword arguments, as with `alg="bp"` above.

## Larger workflows

For anything more than a few calls, write the simulation as a Julia function inside `jl.seval` and call it from Python. Loops and keyword arguments then stay in Julia, and Python only passes in the circuit and gets back numbers:

```python
jl.seval("""
function simulate(qasm; maxdim = 16)
    circuit = circuit_from_qasm(qasm)
    g = build_graph_from_circuit(circuit)
    ψ0 = tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2")
    ψ = apply_gates(circuit, ψ0; apply_kwargs = (; maxdim, cutoff = 1e-12))
    return [real(only(expect(ψ, [("Z", [v])]; alg = "bp"))) for v in circuit.qubits]
end
""")

magnetisations = list(jl.simulate(qasm2.dumps(qc), maxdim=32))
print(magnetisations)  # all zero, up to rounding
```

Every function in this documentation can be used this way, including sampling (see [Sampling](sampling.md)) and the other contraction algorithms (see [Caches](caches.md)).
