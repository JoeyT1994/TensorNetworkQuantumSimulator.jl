# Native planar CTMRG: status (2026-10-17)

This note covers the general planar CTMRG engine. It works on any planar graph, using the planar
embedding in `src/Planar/planar_embedding.jl` (commit 7ea4015). It is meant to replace the grid
engine eventually.

The engine is a working **prototype** that has not yet been ported to `src/`. Its scratch files are
listed at the end.

## Where it stands

- **Lossless χ:** exact to machine precision on every lattice tested. That covers square,
  honeycomb, triangular, kagome (any boundary shape), Penrose generations 2–4, and disc-shaped
  Penrose patches.
- **Truncated χ:** stable and accurate, including Penrose and kagome (table below).
- **Proof of exactness:** a combinatorial check proves exactness per graph before any tensor is
  touched.

## Formulation (tail convention)

- **Messages** `T(u→v)` live on directed edges. Their legs are `bond(u,v)`, plus `σ(f,(u,u_f))` on
  each adjacent inner face `f`, where `u_f` is u's other neighbour on `f`.
- **Turn corners** `C(p,f)` sit at (vertex, face). Their legs are f's two sides at `p`.
- **`σ(f,e)`** is the bundle of bonds crossed by a *ray*: a dual path from `f` across `e` to the
  outer face. In the next face, the ray leaves by an exit rule.
- **Connectors.** The projector at `(f, e)` compresses `[bond(e), σ(h, exit)]`. Corner chains of `h`
  on either side of the exit act as connectors `Ql`/`Qr`. A turn next to the entry edge, and every
  triangle, uses an inverse corner.
- **Messages at trivalent vertices** use the inverse ring connector of the far face.
- **Kikuchi free energy:** `F = Σ_v c_v ln Z_v + Σ_e c_e ln Z_e + Σ_f ln Z_f`.
  - The counting numbers are `c_e = 1 − #inner faces on e` and `c_v = 1 + F_v − z_v`.
  - They are zero on the boundary. Flat ±1 weights do not cancel the block normalisations there.

## The three problems solved

1. **Cuts must reach the boundary.**
   - Exit rules are tried in order: straight (j = 1), opposite edge, then geodesic (toward the
     nearest outer face, which always terminates).
   - When every ray terminates, each block's *true signed region* follows from winding numbers.
   - Checked statically on every lattice: the true regions then satisfy **every** update identity.
     That covers all of Z_v/Z_e/Z_f, every message, and every corner candidate.
2. **Zero-holonomy corner rings** (Penrose and kagome).
   - Rings of 5–6 corners around positively curved spots satisfy the diagonal update identically.
     The local equations therefore leave them undetermined, which caused the boundary-shape
     failures on kagome and all the Penrose failures.
   - Fix: each corner picks one of three valid updates.
     - `:diag`: enlarged corner.
     - `:prev` / `:next`: directional, `C(p,f) = T(p→a)·C(x,h)·Q`.
   - A min-max least fixed point on the AND/OR dependency graph picks them. It gives a DAG of
     **minimum depth**, the number of sweeps needed for exactness: Penrose 4 is 17, kagome 6×6 is 15.
   - Triangular has no DAG. A region oracle certifies its cyclic diagonal-only rules instead.
3. **Consistent tensor iteration.** A *pass* has two stages.
   1. Build one coherent projector set, boundary-inward along the rays. Each projector's tail is
      carried into its next interface's new coordinates by
      `M_I = PB_old · M_J · PA_new` (q side: `PA_old · M_J · PB_new`).
   2. Recompute every block once, in dependency-level order. Its parts are then already in
      current coordinates, so no leg is carried over from a previous pass.
   - The projectors are **stacked two-sided**. Each side stacks its enlarged corner, side message
     and directional pre-blocks; then `AᵀB = USVᵀ`, `PA = BVS^{-1/2}`, `PB = S^{-1/2}UᵀAᵀ`. This is
     lossless for every block at full rank.
   - The trivalent / turn **pseudo-inverse is regularised** (relative cutoff 1e-3).

## Results

Random positive tensors, D = 2, pseudo-inverse cutoff 1e-3. Entries are |F − ln Z|.

| lattice | χ = 4 | χ = 8 | χ ≥ 16 / lossless |
|---|---|---|---|
| square 6×6 | 1e-14 | 1e-14 | 0 |
| honeycomb 4×4 | 4e-15 | 1e-14 | 4e-15 |
| triangular 5×5 | 7e-9 | 8e-12 | 6e-11 |
| kagome 4×4 (trimmed) | 8e-8 | 5e-11 | 5e-11 |
| kagome 6×6 | 6e-7 | 6e-11 | 6e-11 |
| Penrose 3 | 9e-5 | 2e-8 | 3e-10 |

**Exact mode** replaces every projector by the identity. It reproduces the region identities to
1e-14 and is the validation tool.

## Caveats / open

- **Precision floor.** The pseudo-inverse cutoff sets a floor: about 3e-10 at 1e-3, about 1e-11 at
  1e-4. The smaller cutoff converges worse at small χ. A cutoff that tightens during convergence is
  untried.
- **Positive tensors.** These limit unregularised inverse precision to about 1e-7, because corner
  spectra decay fast. Signed tensors reach 1e-12.
- **Triangular** runs on the older per-sweep (Jacobi) update, since it has no acyclic rule set. That
  update is consistent there because it uses diagonal corners only.
- **Penrose 4 (191 sites)** did not finish within 15 minutes. It is untimed: the exact reference
  contraction or the passes could be the bottleneck.
- **Not yet done:**
  - Ising / PEPS (double-layer) networks;
  - a comparison with the grid engine on the square lattice;
  - interior degree-2 vertices (absorb them first);
  - a trivalent vertex whose far face is a triangle.
- **Inverse-free corner gauge.** This would remove the floor and the pseudo-inverses. It is a
  separate research item; Lukin–Sotnikov, arXiv:2610.00631, report no systematic method.

## Next: production (on the maintainer's go)

Planned file: `src/MessagePassing/planarctm.jl`. It would contain:
- rule design and certification (exit portfolio, static regions, min-depth DAG, oracle fallback);
- the sequential pass with stacked projectors;
- the Kikuchi F and convergence check;
- expectation values and reduced density matrices on vertex rings;
- double-layer support;
- tests on the lattices above.

After that: the comparison with the grid engine, then replacing it.

## Prototype files (session scratchpad, not in the repo)

| file | contents |
|---|---|
| `planar_proto2.jl` | `PCTM` tensor engine: blocks, connectors, Jacobi sweep, Kikuchi F, exact mode |
| `planar_seq.jl` | sequential pass, stacked projectors |
| `rcg.jl`, `rcs.jl` | region oracle (vertex multisets), static true regions from winding numbers |
| `dagtools.jl` | exit rules, `design_minimax` (min-depth rule choice) |
| `seq_suite.jl`, `seq_big.jl` | the drivers behind the table above |
