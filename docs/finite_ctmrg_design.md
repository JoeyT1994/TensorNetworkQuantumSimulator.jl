# Finite CTMRG design

This document records the current design, not the experiment history. See
[`ctmrg_status.md`](ctmrg_status.md) for measured behavior and known limits.

## Region graph

Each occupied grid vertex owns a position-resolved environment ring with four corners and four edge
tensors. Sparse square embeddings support square, hexagonal, and heavy-hexagonal geometries. The CVM
functional is

```text
ln Z_CVM = Σ_vertices ln Z_v - Σ_edges ln Z_e + Σ_plaquettes ln Z_p.
```

All terms, local observables, and RDMs consume the same converged cache.

## Local move

A sweep grows each corner through one neighboring site and truncates the enlarged shared interfaces.
Every interface is represented by a left/right projector pair `(P_A, P_B)` satisfying the retained
biorthogonality relation. Gauge alignment to the preceding sweep makes successive environments
comparable without changing the represented subspace.

The implementation keeps double-layer states lazy: a site remains `[ket, bra]` or
`[ket, operator, bra]`, inward ket/bra legs remain separate, and contraction-order optimization avoids
materializing a fused D² site tensor.

## Cut projector

`:cut` treats one interface at a time. Thin QR factorizations compress the two enlarged sides, an SVD
of their small triangular product selects the leading χ directions, and symmetric inverse-square-root
whitening produces the two projectors. This is the optimal rank-χ approximation for that bipartition,
but independent cuts do not make the global CVM functional stationary.

## Cycle projector

`:cycle` forms the action of the four-corner product without materializing the full product. Right and
left invariant subspaces are solved independently and biorthogonalized through their overlap. The
result enforces consistency around a plaquette and makes the converged CVM environment stationary.

The production solver uses matrix-free Krylov/Schur extraction. The optional block-subspace solver
uses only products and QR factorizations and is therefore attractive for GPU batching. It may reuse
the previous sweep's compatible full bases; cold deterministic blocks are used for initialization or
after a shape change.

A cycle may resolve fewer than χ numerical directions. Those directions are retained and only the
fixed-width storage index is padded. A cycle that cannot be defined geometrically or numerically is
declined for the whole plaquette and its interfaces are supplied by the cut pass with a warning.

## Fixed-point iteration

One projector depends on its complementary environment, so construction is a nonlinear fixed-point
problem:

1. Bootstrap all interfaces with the cut projector.
2. Sweep plaquettes in deterministic order, deriving the requested projector family.
3. Align compatible projector gauges to the preceding sweep.
4. Rebuild the position-resolved `C/T` environments.
5. Stop on the requested free-energy or environment signal.

Observable convergence compares the gauge-invariant interface maps `P_A P_B`; tensor entries alone
are not a valid signal because their gauge is arbitrary. The map distance itself is compared with
the requested tolerance. Two full traversals are required before certification so a changed boundary
can propagate through the lattice.

The current C/T sweep is synchronous: all plaquette projectors are derived from the preceding global
environment before any block is rebuilt. On a double-layer `TensorNetworkState`, optional
`gauge_state = true` first moves the PEPS to Vidal/BP symmetric-gauge coordinates; this is a pure
bond-gauge preconditioner and is the supported way to suppress finite-χ input-gauge sensitivity.

## Correctness invariants

- Projector pairs and environment tensors retain the network's scalar type and device.
- All random starts are deterministic for reproducibility.
- No per-interface mixture of cut and cycle is allowed within a defined plaquette.
- A local operator never participates in environment convergence.
- `marginal_inconsistency` is reported as a stationarity residual, never as an accuracy certificate.
- Exact or converged boundary-MPS contraction is the reference whenever lossless CTMRG is unavailable.

The implementation is in `src/MessagePassing/ctmenvironmentcache.jl`; focused coverage is in
`test/test_ctmenvironment.jl`.
