# Sampling

Draw bitstring samples from the probability distribution defined by the squared amplitudes of a tensor network state [[Ferris2021]](index.md#references) [[Rudolph2025]](index.md#references).

## Basic Sampling

```julia
# Returns bitstrings only (each bitstring is a Dictionary mapping vertices to 0/1)
bitstrings = sample(ψ, 100; alg = "boundarymps", norm_mps_bond_dimension = 10)
```

BP-based sampling is also available and works on non-planar graphs, but scales quadratically with system size per sample rather than linearly:

```julia
bitstrings = sample(ψ, 100; alg = "bp")
```

## Directly Certified Sampling

Samples are drawn from an approximate distribution ``q(x)`` and for each sample ``\langle x|\psi\rangle`` is calculated on-the-fly to estimate ``p(x)/q(x)``:

```julia
results = sample_directly_certified(ψ, 100;
    alg = "boundarymps",
    norm_mps_bond_dimension = 10,
)
```

Each result is a `NamedTuple` with fields:
- `poverq`: Approximate value of ``p(x)/q(x)`` for the sampled bitstring ``x``.
- `logq`: Log probability of drawing the bitstring under ``q``.
- `bitstring`: The sampled bitstring as a `Dictionary` mapping vertices to configurations (0, 1, ..., d-1).

The `projected_mps_bond_dimension` keyword controls the bond dimension used for the projected boundary MPS messages during contraction of `⟨x|ψ⟩`. It defaults to `5 * maxvirtualdim(ψ)`.

## Certified Sampling

Similar to directly certified sampling, but performs an independent contraction of `⟨x|ψ⟩` for each sample after all samples have been drawn:

```julia
results = sample_certified(ψ, 100;
    alg = "boundarymps",
    norm_mps_bond_dimension = 10,
    certification_mps_bond_dimension = 50,
)
```

Each result contains:
- `poverq`: Value of ``p(x)/q(x)`` computed via independent contraction.
- `bitstring`: The sampled bitstring.

## Importance Sampling for Observables

The ``p(x)/q(x)`` ratios from certified sampling can be used for importance-sampled estimation of observables:

```julia
results = sample_directly_certified(ψ, nsamples;
    alg = "boundarymps",
    norm_mps_bond_dimension = 10,
)

# Estimate ⟨Z⟩ on a vertex using importance sampling
v = (3, 3)
sampled_sz = sum(r.poverq * (-2 * r.bitstring[v] + 1) for r in results) / sum(r.poverq for r in results)
```

## Keyword Arguments

### Boundary MPS Sampling (`alg = "boundarymps"`)

| Keyword | Description |
|---------|-------------|
| `norm_mps_bond_dimension` | Bond dimension of the boundary MPS messages for contracting ``\langle\psi|\psi\rangle``. |
| `projected_mps_bond_dimension` | Bond dimension for the projected MPS messages (default: `5 * maxvirtualdim(ψ)`). |
| `norm_cache_message_update_kwargs` | Additional kwargs for the norm cache message update. |
| `partition_by` | How to partition the graph (`"row"` by default). |
| `gauge_state` | Whether to gauge the state before sampling (default `true`). |

### BP Sampling (`alg = "bp"`)

| Keyword | Description |
|---------|-------------|
| `bp_update_kwargs` | Keyword arguments for updating the BP cache between projections. |
| `gauge_state` | Whether to gauge the state before sampling (default `true`). |
