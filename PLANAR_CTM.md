# Native planar CTMRG: status (2026-10-20)

This note covers the general planar CTMRG engine. It works on any planar graph, using the planar
embedding in `src/Planar/planar_embedding.jl` (commit 7ea4015). It is meant to replace the grid
engine eventually.

**In `src/` since 2026-10-20** as `CTMRGCache` (`alg = "ctmrg"`), the package's only CTMRG engine:
the grid engine (`CTMEnvironmentCache`, `ctmenvironmentcache.jl`) was retired once this one
reproduced it (see [Production port](#production-port)).
`src/MessagePassing/ctmrg_rules.jl` (geometry, recipes, rule design and certification) and
`src/MessagePassing/ctmrg.jl` (the cache, passes, free energy, observables); tests in
`test/test_ctmrg.jl`. See [Production port](#production-port) for what changed from the
prototype, which stays in `prototypes/planar_ctm/` for reference.

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
  - interior degree-2 vertices (absorb them first);
  - a trivalent vertex whose far face is a triangle.
- **Inverse-free corner gauge.** This would remove the floor and the pseudo-inverses. It is a
  separate research item; Lukin–Sotnikov, arXiv:2610.00631, report no systematic method.

## Production port

- **One description of the geometry.** The prototype described every block twice, as tensors
  (`planar_proto2.jl`) and as vertex multisets (`rcg.jl`). In `src/` each block, and each closed
  region (vertex, edge, face), is one *recipe*: a list of parts (sites, messages, corners,
  pseudo-inverses, projector sides). Contracting a recipe gives the tensor; summing it as a signed
  multiset gives the region, and its block parts are the dependencies.
- **Certified on every graph, at construction.** `planar_ctm_rules` tries the exit rules, picks the
  minimum-depth acyclic corner modes (the least fixed point of `design_minimax`), and certifies
  them: one symbolic pass in dependency order must leave every vertex, edge and face region covering
  each vertex exactly once. With no acyclic choice it certifies diagonal corners under Jacobi
  passes instead (the triangular lattice). The prototype certified only the lattices it was tried on.
- **Projectors reduce to the grid engine's.** An interface's projector is the grid `:cut` projector
  of its two enlarged corners. Only where a directional corner is
  projected, which has no enlarged corner behind it, are the directional corners and side messages
  stacked in as well, keeping each lossless. On the square lattice no corner is directional, and the
  engine reproduces grid `:cut` exactly: ⟨Z⟩ errors 0.0096 / 0.0018 at χ = 4 / 8 on a random 6×6
  D = 2 PEPS, both engines. (The prototype stacked side messages everywhere: 2× worse at χ = 4 there,
  and 100× worse on kagome.) On a random 10×10 D = 3 PEPS the two engines agree on ⟨Z⟩ to 1e-7
  at χ = 16 and 32, and the planar one is as fast or faster (57 / 109 s against 137 / 136 s per
  `update`, the first grid run including compilation; the planar engine took 15 / 30 min before
  these changes).
- **Enlarged corners are nested parts** of the recipes (diagonal corners, face regions, and factored
  out of vertex and edge regions), contracted once and reused. This keeps every list short, which is
  what matters for cost: a cold `update` was ~93% JIT compilation of the many distinct intermediate
  shapes long lists produce, not contraction or order search.
- **Dropped:** the per-sweep basis-change machinery (`prev`, `G`, stale legs), the isometric and
  canonical-gauge projectors, the `CHECK`/`EXACT` switches. The sequential pass made them unnecessary.
- **Double layer.** A site's factors are a list (`bp_factors`), and an edge's bonds are all the legs
  its two sites share, so ⟨ψ|ψ⟩ and forms run lazily, never forming the ket ⊗ bra tensor.
- **Observables** on one vertex (its ring) or on the two ends of an edge (the edge region): `expect`,
  `reduced_density_matrix`; `freenergy`, `partitionfunction`, and `contract(tn; alg = "ctmrg")`.
- **Through the seam.** Every tensor operation is a seam verb; the stacked projector is the grid
  engine's `_ctm_biorth` on direct sums of the blocks, and contractions share its sequence cache.

Validation against exact contraction (same positive D = 2 networks as the table above):
square, honeycomb, Penrose disc ~1e-14 at every χ; triangular 7e-9 / 7e-12 / 6e-11 at χ = 4 / 8 /
lossless (as the prototype); kagome 4×4 1.9e-8 / 6e-12 / 6e-12; Penrose 3 8.4e-7 / 6.7e-9 /
2.3e-9 at χ = 4 / 8 / lossless (the floor is the pseudo-inverse regularisation; the prototype's
was 2.9e-10, its χ = 4 / 8 errors 9e-5 / 2e-8). Double-layer ⟨Z⟩, ⟨X⟩, nearest-neighbour ⟨ZZ⟩, ⟨XY⟩ and ⟨ψ|ψ⟩ on random complex
square and honeycomb states: ≤ 1e-15 at lossless χ.

- **Convergence on every closed region.** `update` stops when no vertex, edge or face region
  changes its ln Z by more than `tolerance`, not when the free energy settles: F settles first (the
  Kikuchi sum cancels), and stopping there left single-site observables short (4.3e-5 against
  2.6e-5 on the 6×6 PEPS at χ = 16). With it, the 6×6 PEPS gives 0.0096 / 0.0018 / 2.6e-5 at
  χ = 4 / 8 / 16, grid `:cut` exactly, in ~22–24 passes.
- **Retiring the grid engine.** Only grid `:cut` had been reproduced; grid `:cycle` (the
  four-corner invariant-subspace projector) went with it. It was the most accurate where it
  converged (3.2e-6 at χ = 16 above) but failed to converge on the 10×10 D = 3 PEPS (⟨Z⟩ 0.0738
  against 0.0537). Its planar generalisation (a cycle around each face) is open; the grid code is
  in the history before the retirement commit.

## Face cycle projector (`projector = :cycle`, experimental, 2026-10-21)

The grid engine's `:cycle` generalised to faces of any size. Face f's interfaces cut the ring of its
enlarged corners, Z_f = tr(K₁ ⋯ Kₙ); each interface I_k is truncated to the spectral projector of
the ring's transfer matrix M_k = K_{k+1} ⋯ K_k onto its maxdim largest eigenvalues. Right and left
bases come from ONE Schur decomposition (left from a Sylvester solve), so they belong to the same
eigenvalues; computing them by two separate decompositions picked different eigenvalue sets
wherever the choice was ambiguous and broke at χ = 16 (⟨Z⟩ off by 0.4). Numerically zero
eigenvalues are never kept and clusters are never split. Needs diagonal corners under Jacobi
passes (square, honeycomb, triangular, small Penrose; not kagome or larger Penrose). Dense: the
ring is formed as N × N matrices, O(N³) per face against the grid engine's matrix-free O(N²χ).

Convergence: `update(...; convergence = :messages)` (the default, for observables) waits for every
vertex's environment to settle (BP's message distance on it); `:freenergy` waits for F only.

Results (max error of ⟨Z⟩, ⟨X⟩ and a nearest-neighbour ⟨ZZ⟩ against exact, TFIM quench
hx = 2, J = 1, t = 1 on 5×5):

| | χ = 4 | χ = 8 | χ = 16 |
|---|---|---|---|
| D = 2, `:cut` / `:cycle` | 0.0032 / 0.0027 | 1.4e-4 / **4.1e-5** | exact / exact |
| D = 3, `:cut` / `:cycle` | **0.029** / 0.047 | 0.0056 / **3.4e-4** | 5.8e-5 / 5.2e-5 |

`:cycle` wins where χ binds (3–16×), loses at severe truncation (D = 3, χ = 4), where it also does
not converge (as the grid's did). Ising 8×8 at β = 0.44: both projectors agree on F (2.4e-10 at
χ = 4, exact from χ = 8), converging in 3–5 passes. On random complex PEPS it matches the grid's
`:cycle` (6×6 D = 2: 0.0025 / 5.1e-4 / 3.2e-6 at χ = 4 / 8 / 16).

Open: a matrix-free (Krylov–Schur) ring for the cost; directional corners; the triangular lattice
at lossless χ, where `:cycle` sits at ~1e-8 and does not converge (`:cut` is exact).

`pinv_rtol` is a real trade-off: at 1e-6 or 1e-9 the lossless floor vanishes (Penrose 3 4e-14,
kagome 7e-15), but truncated χ breaks (Penrose 3 at χ = 8: 5e-3 against 7e-9; kagome at χ = 4:
4e-3 against 2e-8). 1e-3 stays the default; a cutoff that tightens as the truncation becomes
lossless is the open refinement.

## Prototype files (`prototypes/planar_ctm/`, superseded by `src/`)

Run `julia --project=. prototypes/planar_ctm/smoke.jl` from the repository root. It takes a few minutes and
prints |F − ln Z| for the Penrose disc, kagome 3×3 and Penrose 3 at truncated and lossless χ.

| file | contents |
|---|---|
| `smoke.jl` | quick end-to-end check from the repository project |
| `planar_proto2.jl` | `PCTM` tensor engine: blocks, connectors, Jacobi sweep, Kikuchi F, exact mode |
| `planar_seq.jl` | sequential pass, stacked projectors |
| `rcg.jl`, `rcs.jl` | region oracle (vertex multisets), static true regions from winding numbers |
| `dagtools.jl` | exit rules, `design_minimax` (min-depth rule choice) |
| `seq_suite.jl`, `seq_big.jl`, `seq_run.jl` | the drivers behind the table above |
