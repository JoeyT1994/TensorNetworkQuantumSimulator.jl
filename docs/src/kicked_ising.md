# IBM's Kicked Ising Experiment

In 2023 IBM ran a kicked Ising circuit on all 127 qubits of the *ibm_kyiv* Eagle processor and, with error mitigation, measured magnetisations beyond the reach of exact classical simulation [Kim2023]. Tensor networks with belief propagation (BP) reproduce these results on a laptop [Tindall2024]. This page does so with TensorNetworkQuantumSimulator.jl in a few dozen lines: build the device's qubit layout, apply the circuit, and measure the magnetisations the experiment reported.

## The device

The Eagle processor has 127 qubits on a heavy-hex lattice: seven rows of qubits joined by four "bridge" qubits between each pair of rows. We label the vertices with the device's own qubit numbers, so that qubit 62 of the experiment is vertex `62` here. The edges below are exactly ibm_kyiv's 144 couplings, as listed in the `FakeKyiv` backend of `qiskit-ibm-runtime`.

```julia
using TensorNetworkQuantumSimulator
using Statistics: mean

function eagle_graph()
    rows = [0:13, 18:32, 37:51, 56:70, 75:89, 94:108, 113:126]
    # (bridge qubit, qubit in the row above, qubit in the row below)
    bridges = [
        (14, 0, 18), (15, 4, 22), (16, 8, 26), (17, 12, 30),
        (33, 20, 39), (34, 24, 43), (35, 28, 47), (36, 32, 51),
        (52, 37, 56), (53, 41, 60), (54, 45, 64), (55, 49, 68),
        (71, 58, 77), (72, 62, 81), (73, 66, 85), (74, 70, 89),
        (90, 75, 94), (91, 79, 98), (92, 83, 102), (93, 87, 106),
        (109, 96, 114), (110, 100, 118), (111, 104, 122), (112, 108, 126),
    ]
    g = NamedGraph(collect(0:126))
    for row in rows, q in row[1:(end - 1)]
        g = add_edge(g, q => q + 1)
    end
    for (b, above, below) in bridges
        g = add_edge(add_edge(g, above => b), b => below)
    end
    return g
end

g = eagle_graph()
```

## The circuit

Starting from every qubit in ``|0\rangle``, each Trotter step of the experiment applies ``R_X(\theta_h)`` to every qubit followed by ``R_{ZZ}(-\pi/2)`` on every coupled pair. The couplings are split into three layers of non-overlapping pairs with `edge_color`, as on the device. The gate registry follows qiskit's conventions, ``R_{ZZ}(\theta) = e^{-i\theta Z \otimes Z / 2}``, so the angles are the experiment's.

```julia
function kicked_ising_layer(g, θh)
    layer = [("Rx", [v], θh) for v in vertices(g)]
    for colored_edges in edge_color(g, 3)
        append!(layer, ("Rzz", [src(e), dst(e)], -π / 2) for e in colored_edges)
    end
    return layer
end

# Evolve `ψ0` for `nsteps` Trotter steps with bond dimension `maxdim`. Returns the BP cache, which
# holds the state and its environment.
function kicked_ising(g, θh, nsteps, ψ0; maxdim)
    layer = kicked_ising_layer(g, θh)
    ψ = BeliefPropagationCache(ψ0)
    for _ in 1:nsteps
        ψ = apply_gates(layer, ψ; apply_kwargs = (; maxdim, cutoff = 1.0e-12, normalize_tensors = true))
    end
    return ψ
end

ψ0 = tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2")
```

Gates are applied with the BP-gauged simple update and observables are measured with BP, which is exact on trees. The smallest loop of the heavy-hex lattice has 12 qubits, so BP is very accurate here.

## Magnetisation after 5 steps

The experiment's first test was the average magnetisation ``M_z = \frac{1}{127}\sum_q \langle Z_q \rangle`` after 5 Trotter steps, across ``\theta_h`` from 0 to ``\pi/2``.

```julia
for θh in (0.0, π / 8, π / 4, 3π / 8, π / 2)
    ψ = kicked_ising(g, θh, 5, ψ0; maxdim = 32)
    mz = mean(real, expect(ψ, [("Z", [v]) for v in vertices(g)]))
    println("θh = $(round(θh; digits = 3)): Mz = $(round(mz; digits = 4))")
end
```

| ``\theta_h`` | 0 | ``\pi/8`` | ``\pi/4`` | ``3\pi/8`` | ``\pi/2`` |
|---|---|---|---|---|---|
| ``M_z`` | 1.0 | 0.9048 | 0.4528 | 0.0373 | 0.0 |

Each ``R_{ZZ}`` gate at most doubles the bond dimension between two qubits, so after 5 steps no bond needs more than ``2^5 = 32``. With `maxdim = 32` no bond is cut down by `maxdim`, and the only approximation left is BP's estimate of the environment. At ``\theta_h = \pi/2`` the circuit is Clifford and ``M_z`` is exactly zero, which BP reproduces. The whole sweep takes a few seconds.

## A single qubit after 20 steps

The experiment's deepest circuits ran 20 Trotter steps and measured ``\langle Z_{62} \rangle``, on a qubit in the middle of the device. The bond dimension could now reach ``2^{20}``, so the state must be truncated, and the result should be checked as `maxdim` grows.

```julia
for maxdim in (4, 8, 16, 32)
    ψ = kicked_ising(g, π / 8, 20, ψ0; maxdim)
    z62 = real(only(expect(ψ, [("Z", [62])])))
    println("maxdim = $maxdim: <Z_62> = $(round(z62; digits = 4))")
end
```

| `maxdim` | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| ``\theta_h = \pi/8`` | 0.8669 | 0.8552 | 0.8539 | 0.8538 |
| ``\theta_h = \pi/4`` | 0.7034 | 0.6614 | 0.5549 | 0.4266 |

At ``\theta_h = \pi/8`` the result has converged by `maxdim = 16`: ``\langle Z_{62} \rangle`` changes only in the fourth decimal place from `maxdim = 16` to `maxdim = 32`. At ``\theta_h = \pi/4`` the state is far more entangled: ``\langle Z_{62} \rangle`` still changes in the first decimal place from `maxdim = 16` to `maxdim = 32`, so a converged answer needs a larger bond dimension (see [Tindall2024]). Each run at `maxdim = 32` takes about half a minute on a laptop.

## Going further

- **Other observables.** `expect` takes any Pauli string, such as `("XYZ", [37, 41, 52])`, so the multi-qubit observables measured in the experiment can be computed from the same state.
- **Larger bond dimensions.** Pass a larger `maxdim`; for speed use `ComplexF32`, or a GPU (see [Advanced Topics](advanced.md)).
- **Loop corrections.** BP neglects correlations around loops of the lattice. [Advanced Topics](advanced.md) shows how to correct for them.
- **The device's own circuit.** A transpiled circuit exported from qiskit as OpenQASM 2 can be run directly; see [Circuits from Qiskit](qiskit.md).

## References

- **[Kim2023]** Y. Kim, A. Eddins, S. Anand, *et al.*, "Evidence for the utility of quantum computing before fault tolerance," Nature **618**, 500 (2023). [Link](https://doi.org/10.1038/s41586-023-06096-3)
- **[Tindall2024]** J. Tindall, M. Fishman, E. M. Stoudenmire, and D. Sels, "Efficient Tensor Network Simulation of IBM's Eagle Kicked Ising Experiment," PRX Quantum **5**, 010308 (2024). [Link](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010308)
