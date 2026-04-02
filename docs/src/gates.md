# Gate Application

## Building Circuits

A circuit is a `Vector` of gates to be applied sequentially. Each gate is specified as a tuple `(gate_string, vertices, parameter)` or `(gate_string, vertices)` if no parameter is associated with that gate:

```julia
layer = []
# Single-site rotations
append!(layer, ("Rx", [v], 2 * hx * dt) for v in vertices(g))
append!(layer, ("Rz", [v], 2 * hz * dt) for v in vertices(g))

# Two-site gates, grouped by edge coloring for maximum BP update efficiency when using `apply_gates`
ec = edge_color(g, 4)  # 4 = coordination number of the square lattice
for colored_edges in ec
    append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
end
```

### Edge Coloring

The `edge_color` function partitions edges into groups of non-overlapping edges. Non-overlapping gates within a group are applied without requiring intermediate BP cache updates, which is significantly more efficient. An edge coloring into `k` groups is always possible on a bipartite graph of degree `k` so in that case the second argument should be the coordination number (maximum vertex degree) of the graph:

```julia
ec = edge_color(g, 4)   # square lattice (degree 4)
ec = edge_color(g, 3)   # hexagonal / heavy-hex lattice (degree 3)
ec = edge_color(g, 6)   # 3D cubic lattice (degree 6)
```

## Applying Gates

Apply gates to a `TensorNetworkState` or a `BeliefPropagationCache`:

```julia
apply_kwargs = (; maxdim = 10, cutoff = 1e-10, normalize_tensors = true)

# Apply to a TensorNetworkState (constructs a BP cache internally)
ψ, errors = apply_gates(circuit, ψ; apply_kwargs)

# Apply to a BeliefPropagationCache (reuses existing messages)
ψ_bpc = BeliefPropagationCache(ψ)
ψ_bpc, errors = apply_gates(circuit, ψ_bpc; apply_kwargs)
```

### Keyword Arguments

- `apply_kwargs`: A `NamedTuple` controlling bond dimension truncation:
  - `maxdim`: Maximum bond dimension after SVD truncation.
  - `cutoff`: Singular value cutoff for truncation.
  - `normalize_tensors`: Whether to locally normalize tensors after each gate application (recommended for numerical stability).
- `bp_update_kwargs`: Keyword arguments controlling the BP message update between overlapping gate groups.
- `update_cache`: Whether to update BP messages between gate groups (default `true`).

### Return Values

`apply_gates` returns a tuple `(ψ_updated, errors)` where:
- `ψ_updated` is the updated state (or cache).
- `errors` is a vector of truncation errors, one per gate.

The product `prod(1 .- errors)` gives the approximate overall fidelity from applying the given gates.

## Simple Update Algorithm

Under the hood, each two-site gate is applied via the _simple update_ algorithm [[Tindall2024]](index.md#references) [[Rudolph2025]](index.md#references):

1. Gauge the state locally with the square roots of the BP environment messages
2. Perform a QR decomposition to efficiently isolate the two `R` tensors.
3. Apply the gate
4. Perform an SVD and truncate the singular values to the desired bond dimension.
5. Multiply the `Q` tensors back in and ungauge the state with the inverse square root messages.
4. Update the BP messages (both directions) on the affected bond with the singular value matrix `S`.

Single-site gates are applied by direct contraction with the site tensor (no truncation needed). If the gate is unitary the BP messages will be unchanged.

## Supported Gates

All parameterised gates follow the Qiskit convention.

### One-qubit Gates

| Gate | Parameter | Description |
|------|-----------|-------------|
| `"X"`, `"Y"`, `"Z"` | -- | Pauli gates |
| `"H"` | -- | Hadamard |
| `"P"` | phase | Phase gate |
| `"Rx"`, `"Ry"`, `"Rz"` | angle | Pauli rotation |
| `"CRx"`, `"CRy"`, `"CRz"` | angle | Controlled Pauli rotation (single-qubit part) |

### Two-qubit Gates

| Gate | Parameter | Description |
|------|-----------|-------------|
| `"CNOT"`, `"CX"`, `"CY"` | -- | Controlled gates |
| `"SWAP"`, `"iSWAP"`, `"√SWAP"`, `"√iSWAP"` | -- | Swap variants |
| `"Rxx"`, `"Ryy"`, `"Rzz"` | angle | Pauli-pair rotations |
| `"Rxxyy"`, `"Rxxyyzz"` | angle | Multi-Pauli rotations |
| `"CPHASE"` | phase | Controlled phase |

### Custom Gates

Custom gates can be defined by constructing the corresponding `ITensor` acting on the physical indices of the target qubits and directly passing to `apply_gates`:

```julia
s = siteinds(ψ)
# Custom gates
gate1 = ITensor(my_local_matrix, s[v1], s[v1])
gate2 = ITensor(my_nn_gate, s[v1], s[v2], s[v1]', s[v2]')
ψ, errors = apply_gates([gate1, gate2], ψ; apply_kwargs)
```
