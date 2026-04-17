# Expectation Values

## Computing Expectation Values

The `expect` function computes expectation values of local observables from a `TensorNetworkState` using belief propagation [[Tindall2023]](index.md#references), boundary MPS [[Rudolph2025]](index.md#references), or exact contraction.

### Observable Format

Observables are specified as tuples with 2 or 3 elements:

```
(operator, vertices)
(operator, vertices, coefficient)
```

- **`operator`**: Either a `String` or a `Vector{String}`. A single string is interpreted as a sequence of single-character operator names, one per vertex — e.g. `"ZZ"` on two vertices means ``Z \otimes Z``. A `Vector{String}` assigns one (possibly multi-character) operator per vertex, e.g. `["Sx", "Sy"]`.
- **`vertices`**: A single vertex, a `Vector` of vertices, or a `NamedEdge` (which expands to its two endpoints). Vertices must match the vertex type of your graph (e.g. `(3, 3)` for a 2D grid) and align with the number of operators passed.
- **`coefficient`** (optional): A scalar multiplier, defaults to `1`.

```julia
# Single-site: ⟨Z⟩ at vertex (3,3)
("Z", (3, 3))

# Two-site correlator: ⟨Z⊗Z⟩ using a string with one character per site
("ZZ", [(3, 3), (3, 4)])

# Two-site with explicit vector of operator strings
(["Z", "Z"], [(3, 3), (3, 4)])

# Two-site via a NamedEdge (expands to src and dst)
("ZZ", NamedEdge((3, 3) => (3, 4)))

# With a coefficient: 0.5 * ⟨Z⟩
("Z", (3, 3), 0.5)
```

To measure multiple observables at once, pass a `Vector` of tuples. This avoids redundant cache construction when no cache is available — `expect` builds the cache once and evaluates all observables against it.

```julia
observables = [("Z", [v]) for v in vertices(g)]
sz_all = expect(ψ, observables; alg = "bp")
```

!!! note
    When using `alg = "boundarymps"`, all observables in a batch must be aligned along the same row or the same column of the lattice, since the boundary MPS contraction partitions in one direction. 

### Examples

```julia
# Single-site observable with BP
sz = expect(ψ, ("Z", (3, 3)); alg = "bp")

# Two-site correlator with BP
szz = expect(ψ, ("ZZ", [(3, 3), (3, 4)]); alg = "bp")

# Multiple observables at once with BP
observables = [("Z", [v]) for v in vertices(g)]
sz_all = expect(ψ, observables; alg = "bp")
```

### Algorithm Options

```julia
# Belief propagation (works on any graph, fast, approximate on loopy graphs)
sz = expect(ψ, ("Z", (3, 3)); alg = "bp")

# Boundary MPS (planar graphs only, more accurate, adjustable precision with mps_bond_dimension)
sz = expect(ψ, ("Z", (3, 3)); alg = "boundarymps", mps_bond_dimension = 16)

# Exact contraction of the tensor network (only feasible for small systems)
sz = expect(ψ, ("Z", (3, 3)); alg = "exact")
```

### Using Caches Directly

If you already have an updated `BeliefPropagationCache` (e.g. from `apply_gates`) or `BoundaryMPSCache`, pass it directly to avoid redundant cache construction:

```julia
ψ_bpc = BeliefPropagationCache(ψ)
ψ_bpc = update(ψ_bpc)
sz = expect(ψ_bpc, [("Z", [(3, 3)])])
```
