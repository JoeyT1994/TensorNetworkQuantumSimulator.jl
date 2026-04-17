# Entanglement Entropy

The von Neumann and Rényi entanglement entropies can be computed via several backends.

## Bond Entanglement Entropy (BP, fast)

For a single bond, the BP messages provide a cheap estimate of the entanglement spectrum without constructing a full reduced density matrix. This is the recommended approach when computing entanglement across many bonds iteratively, as it is fast and scales linearly in system size:

```julia
e = NamedEdge((2, 3) => (3, 3))

# Von Neumann entropy across the bond
s = von_neumann_entanglement_entropy(ψ, e; alg = "bp")

# Second Rényi entropy across the bond
s2 = second_renyi_entanglement_entropy(ψ, e; alg = "bp")
```

If you already have a converged cache, pass it directly to avoid rebuilding BP:

```julia
ψ_bpc = BeliefPropagationCache(ψ)
ψ_bpc = update(ψ_bpc)

# Compute entropy across every bond
s_all = [von_neumann_entanglement_entropy(ψ_bpc, e) for e in edges(ψ_bpc)]
```

## Subsystem Entanglement Entropy (via RDM)

For a subsystem of multiple vertices, the entropy is computed from the reduced density matrix. This supports `"bp"`, `"boundarymps"`, and `"exact"` backends:

```julia
verts = [(2, 2), (2, 3), (3, 2), (3, 3)]

# Via BP
s = von_neumann_entanglement_entropy(ψ, verts; alg = "bp")

# Via boundary MPS (more accurate on loopy graphs)
s = von_neumann_entanglement_entropy(ψ, verts; alg = "boundarymps", mps_bond_dimension = 16)
```

## General Rényi Entropy

Use `renyi_entropy` directly for an arbitrary Rényi index `α`:

```julia
# α = 1 → von Neumann, α = 2 → second Rényi, etc.
s = renyi_entropy(ψ, e; alg = "bp", α = 3)
s = renyi_entropy(ψ, verts; alg = "bp", α = 2)
```

The Rényi entropy of order `α` is defined as

```math
S_\alpha(\rho) = \frac{1}{1 - \alpha} \log \operatorname{tr}(\rho^\alpha)
```

with the limit ``\alpha \to 1`` giving the von Neumann entropy ``S = -\operatorname{tr}(\rho \log \rho)``.
