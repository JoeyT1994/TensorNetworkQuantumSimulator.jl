# 3D CTMRG: finite cubic boxes and the infinite lattice

*Started 2026-09-24 (FixesV2). Everything here is measured; the numbers are dated.*

Two engines, one construction:

* **`CTM3DEnvironmentCache`** (`src/MessagePassing/ctm3denvironmentcache.jl`) — position-resolved
  finite CTMRG on an open L×L×L (or rectangular) box: the 2D engine's region-graph (CVM)
  construction one dimension up. `CTMEnvironmentCache(net, χ)` returns it for a network on
  `(x, y, z)` vertices.
* **`InfiniteCTM3D`** (`src/MessagePassing/ctm3dinfinite.jl`) — the same moves on a 1×1×1 unit cell
  for the thermodynamic limit, built from one six-leg site tensor.

```julia
tn = ising_partitionfunction(named_grid((4, 4, 4)), 0.2)
c = update(CTMEnvironmentCache(tn, 4; projector = :cut))
cvm_freenergy(c)                        # ≈ ln Z

site, legs, mag = ising3d_site(0.25)    # cubic Ising, J = (1, 1, 1)
ic = update(InfiniteCTM3D(site, legs, 4; boundary = [1.0, 0.0]))
cvm_freenergy(ic), site_ratio(ic, mag)  # ln κ per site, ⟨σ⟩ — both exact contractions, χ ≲ 5

# larger χ: converge on the blocks, read observables through the edge region (~χ⁸)
ic8 = update(InfiniteCTM3D(site, legs, 8; boundary = [1.0, 0.0]); convergence = :blocks, tolerance = 1e-8)
site_ratio(ic8, mag; method = :edge)
```

## Geometry

Per axis a block is `coord < p` (sign −1), `coord == p` (0) or `coord ≥ p` (+1). A vertex's
environment SHELL is 26 blocks — 6 half-lines T (one nonzero sign), 12 quarter-planes E (two), 8
octants C (three) — which with the vertex tile the box. A block's legs are its FACES: the bonds
along one axis over the block's transverse extent. Both transverse signs nonzero: a PLANE interface
(a quarter-plane of bonds, octant ↔ quarter-plane block); one zero: a LINE interface (a half-line,
quarter-plane ↔ half-line block, exactly the 2D engine's interfaces); both zero: a half-line's
single bond into its vertex, never truncated. Every interface is shared by four blocks, two per
side, so its projector pair is one object: `P_A` on the low side, `P_B` on the high side.

`F` is the Möbius sum over the half-integer region grid, weight (−1)^(number of half-integer
coordinates): vertex +1, edge −1, face +1, cube −1. The Euler characteristic of the box's cell
complex is 1, so at lossless χ `F = ln Z`, and every block's scale cancels (a block's regions carry
weights summing to zero), exactly as in 2D. Per site in the infinite lattice this is
`ln κ = ln Z_v − Σ ln Z_e + Σ ln Z_f − ln Z_c` over one vertex, three edge, three face and one cube
region — the 3D form of the 2D CTMRG partition-function-per-site formula.

## The move and the projectors

A new block is the previous state's block one slab further out plus the slab next to its new
boundary: an octant grows from 8 old pieces (C, 3E, 3T, vertex), a quarter-plane from 4, a
half-line from 2. A new plane face is then χ·χ·χ·D wide (the old plane leg, the two line legs of
the new strips, the new corner bond), a new line face χ·D.

**The rest problem.** A pair is derived from the two enlarged blocks on either side of its
interface with the same transverse extent (two octants of one cube for a plane interface), as the
2D engine derives a column interface from its NW and NE enlarged corners. In 2D the "rest" legs of
those blocks are χD wide and stay open. An octant's two other faces are each χ³D wide, so the
2D-style block would be (χ³D)² × χ³D — only feasible for χ ≲ 5. The plane rest faces are therefore
COMPRESSED by the previous sweep's pair for the same face, leaving each block χ² × χ³D (line rest
faces stay open). For that to be consistent the previous pair must live on the current legs, so
every interface has a FIXED kept index, created once with width min(χ, full bond dimension) and
reused by every sweep (derived pairs are zero-padded to it and Procrustes-aligned to the previous
pair within it). `plane_rest = :exact` keeps the rest open — a small-χ reference.

The 2D record (finite_ctmrg_design.md, "Gauss-Seidel sweeping") found that deriving a `:cut`
projector from a block whose rest was already truncated degrades it, because it optimises the
truncated bipartition. In 3D there is no alternative beyond χ ≈ 5, and here it works: 4×4×4 Ising
at χ = 4 lands 2e-8 from exact (below).

**Seeding.** From an empty state (open boundary: missing blocks are the region outside the box),
uncompressed sweeps at `seed_maxdim` fill the blocks one slab per sweep. The state is then
zero-padded onto the target χ's kept indices and the compressed sweeps take over; a compressed
rest face of rank r caps the next plane pair at rank r², so the kept rank grows quadratically.
The infinite engine seeds from the open-boundary VACUUM instead (every plane and line leg starts
one-dimensional, a half-line's vertex leg carries the `boundary` vector).

**`:cut`** — the 2D biorthogonal pair from triangular factors (`_ctm_twosided_projector_qr`) for
every interface.

**`:cycle`** — for plane interfaces, the dominant invariant subspace of the interface's CUBE: the
other six octants closed around it, every other face of the cube truncated by THIS sweep's `:cut`
pair. With the two open octants `a`, `b` (χ² × n) and the other six `e` (χ² × χ²), the interface
operator ρ = aᵀ e b has rank ≤ χ², and its nonzero spectrum is that of the χ²×χ² matrix
M = e b aᵀ; one Schur form of M gives both invariant bases (right `aᵀY`, left `Uᵀ e b`, made
biorthonormal by a Sylvester solve) with no n-dimensional eigensolve. `P_A` spans the left basis,
`P_B` the right, so that Z_cube(Π) = Tr(ρ Πᵀ) keeps the dominant eigenvalues. Line interfaces keep
their `:cut` pair.

### What it took to get here — 2026-09-24

1. **Scalar gauge runaway (fixed).** A pair has the gauge P_A → tP_A, P_B → P_B/t that leaves
   Π = P_A P_B, every block and F unchanged. A pair derived from two compressed octants inherited
   the ratio of their magnitudes, which came from the previous pairs, so the log-imbalance doubled
   every sweep: max ‖P‖ 4 → 17 → 79 → 3e3 → 2.5e6 → … → 1e192 and overflow at sweep 13, with F
   converged at 2e-8 the whole time. Fixed by normalising both blocks before deriving and
   balancing ‖P_A‖ = ‖P_B‖ afterwards (`_c3_finish`); 15 sweeps then hold F at 2.15e-8 with max ‖P‖
   2.01.
2. **`:cycle` closed with the previous `:cycle` pairs (replaced).** Each cube's 12 pairs then feed
   back into each other through one Jacobi step. On 4×4×4 at χ = 4, F reached 2.4e-8 at sweep 1
   and wandered between −3e-5 and +1e-5 for 15 sweeps (also after the gauge fix). The 2D cycle
   has no such loop: the plaquette's ring makes the consistency condition closed-form for all
   four interfaces at once, and the cube graph is not a ring.
3. **`:cycle` closed with this sweep's `:cut` pairs (current).** Converges monotonically and
   smoothly — but to a WORSE fixed point for F: 7.5e-7 on 4×4×4 at χ = 4 against 2.15e-8 for `:cut`.

## Validation — finite boxes, against exact contraction

Exact contraction reaches 4×4×4 (ln Z in ~54 s at β = 0.2).

**Lossless χ is exact** (every pair a full-rank isometry): 2×2×2 at χ = 2 (0), 2×2×3 at 4
(1.8e-15), 3×3×3 at 16 (1.1e-14), both projectors, and a random signed 2×3×3 network — the
geometry, faces, Möbius weights, pairing and seed promotion are right.

**Truncated, 3D Ising at β = 0.2 (T = 5, T_c ≈ 4.51):**

| box | χ | `:cut` \|F − ln Z\| | `:cycle` \|F − ln Z\| |
|---|---|---|---|
| 3×3×3 | 2 | 6.9e-9 | 1.9e-7 |
| 3×3×3 | 3 | 1.1e-12 | 2.7e-9 |
| 3×3×3 | 4 | 3.6e-15 | 1.1e-14 |
| 4×4×4 | 2 | 1.9e-4 | — |
| 4×4×4 | 4 | 2.2e-8 | 7.5e-7 |

**4×4×4 across the transition** (*2026-09-25*). The comparison is against exact contraction:
ln Z = 45.091477831124, 47.430137672829, 48.201585859063 and 52.186497852932 at β = 0.1, 0.2,
0.2217 and 0.3. Up to 30 sweeps; |F − ln Z|:

| β | χ | `:cut`, `:biorth` | `:cut`, `:isometric` | `:cycle` |
|---|---|---|---|---|
| 0.1 | 2 | 1.3e-6 | 1.3e-6 | 1.3e-6 |
| 0.1 | 3 | 1.2e-7 | 8.6e-8 | **3.84** |
| 0.1 | 4 | 1.0e-12 | 1.1e-12 | 3.7e-11 |
| 0.2 | 2 | 1.9e-4 | 1.9e-4 | 1.8e-4 |
| 0.2 | 3 | 2.1e-5 | 1.8e-5 | 2.8e-5 |
| 0.2 | 4 | 2.2e-8 | 1.6e-8 | 7.4e-7 |
| 0.2217 | 2 | 3.5e-4 | 3.5e-4 | 3.3e-4 |
| 0.2217 | 3 | 3.3e-5 | 3.2e-5 | 4.2e-5 |
| 0.2217 | 4 | 1.2e-7 | 8.1e-8 | 1.9e-6 |
| 0.3 | 2 | 2.1e-3 | 2.4e-3 | 1.4e-3 |
| 0.3 | 3 | 1.3e-5 | 1.1e-4 | 5.7e-5 |
| 0.3 | 4 | 1.1e-4 | 1.5e-5 | 7.7e-5 |

* **Accuracy by phase.** A finite box is easy in the disordered phase and at β_c (a 4-site face
  holds little entanglement). The ordered side is the hard one: the free-boundary box is a cat
  state, and there χ = 4 `:biorth` is worse than χ = 3.
* **Pair choice.** Neither pair wins everywhere. `:isometric` is better at χ = 4 (up to 7× at
  β = 0.3) but 8× worse at β = 0.3, χ = 3, so the finite default stays `:biorth`.
* **`:cycle` at χ = 3, β = 0.1 fails outright.** χ = 3 cuts through a degenerate multiplet of the
  cube spectrum, where the invariant subspace is ill-defined. `:cut` at the same χ is fine. Keep
  `:cycle` to χ at a spectral gap, or set `degtol`.

**An observable** (4×4×4, β = 0.2 with a field h = 0.1 so ⟨σ⟩ ≠ 0; exact ⟨σ⟩ = 0.2574 at the
corner, 0.4734 at (2,2,2)), read through the vertex shell as an impurity ratio:

| χ | projector | \|ΔF\| | \|Δσ\| (1,1,1) | \|Δσ\| (2,2,2) |
|---|---|---|---|---|
| 2 | `:cut` | 1.1e-4 | 6.1e-4 | 8.4e-4 |
| 2 | `:cycle` | 1.1e-4 | 8.2e-4 | 6.8e-4 |
| 4 | `:cut` | 9.6e-9 | 2.1e-5 | 1.0e-5 |
| 4 | `:cycle` | 2.7e-7 | 7.4e-5 | 1.3e-5 |

So in 3D, as it stands, `:cycle` does not beat `:cut` on F or on this observable. Single-site
observables are also much less accurate than F (1e-5 against 1e-8 at χ = 4): F's Möbius sum
cancels errors a single shell does not — the same asymmetry the 2D engine shows.

**The `:cycle` convention is the right one.** Swapping it — `P_A` on the right basis — gives
|ΔF| = 1.75e-2 and |Δσ| = 4.6e-2 / 3.0e-2 on the same χ = 4 case, 4–5 orders worse: the pair keeps
the subspace the cube actually uses. The criterion, not its implementation, is what does not
beat `:cut` here.

## Validation — the infinite engine, against exact limits

* **1D chain** (only Jx, K = 0.4): ln κ = ln 2cosh K to 2e-16 at χ = 2.
* **Decoupled 2D layers** (Jz = 0, K = 0.3) against Onsager: |Δ ln κ| = 1.05e-6 at χ = 2 and
  1.3e-10 at χ = 4, `:cut` and `:cycle` identical (the z bonds are rank one).
* χ = 8 ran out of memory in ln κ — see "THE CEILING" below.

## The infinite iteration: stability, and what its fixed point is worth — *2026-09-25*

### A side-asymmetric instability, and the isometric pair

**Symptom.** 3D Ising (J = 1), fixed-spin seed, `:cut`: at β = 0.25 every χ from 2 to 6 converged
and then blew up. At χ = 2, m = 0.75805743 and ln κ to 1e-10 by iteration 20, unchanged at 80,
m = 0.6465 at 100 and −1.81 at 150. β = 0.15 held for 200 iterations. The first 3D Ising scan
(χ = 4: m = −0.36 at β = 0.215 from a + seed, ln κ 0.03 off at 0.22) read states in the middle
of this.

**Diagnosis.**

* Invisible to anything symmetric. From iteration 25 to 65, m and all 24 interface spectra held
  to 1e-16 while the largest block change grew ×8 every 5 iterations (2e-8 → 1e-4). Only the
  plane pairs and the E and C blocks moved; the line pairs and T blocks stayed put until
  precision went.
* Not gauge drift. Every Procrustes alignment succeeded and the pairs stayed perfectly
  conditioned (κ(P_A) = 1.00). Replacing Procrustes by the pair's own sign-matched singular
  basis left the growth rate unchanged, and added glitches where two singular values nearly
  cross, so Procrustes stays.
* A mirror mode. The octant C(+,+,+) stayed exactly symmetric under permutations of its own axes.
  But it and its mirror image C(−,+,+) (sorted entry magnitudes) drifted apart ×1.52 per
  iteration from round-off at iteration 5: 3e-16, 2e-15, 1.7e-14, …, 5e-10 at 40, 1e-3 at 75. The
  mode was there from the start, hidden under the converging symmetric modes.

The biorthogonal pair amplifies exactly this kind of perturbation, one that makes an interface's
two sides differ. It builds `P_A` from the high side's triangular factor and `P_B` from the low
side's, so a side difference becomes `P_A ≠ P_Bᵀ` at first order and feeds into both sides' next
blocks.

**Fix: `pair = :isometric`** (the `InfiniteCTM3D` default):

* **Construction.** `P` is the dominant eigenvectors of `Ac†Ac + Bc†Bc` on the interface legs
  (the right singular vectors of the two triangular factors stacked), with `P_A = P`,
  `P_B = P†`.
* **Why it has the same fixed point.** At a mirror-symmetric fixed point `R_A = R_B`, so the
  biorthogonal pair *is* this isometry (hence κ(P_A) = 1.00 above).
* **Why it is stable.** A perturbation that is antisymmetric between the two sides changes
  `Ac†Ac + Bc†Bc` only at second order.

Measured results:

* **β = 0.25, χ = 2:** converges to the same m = 0.7580574315 and holds it through iteration
  150; the mirror octants agree to 2e-16 throughout.
* **β = 0.25, χ = 4 and 6:** converge smoothly.
* **β = 0.15, χ = 4:** ln κ = 0.72853629 with either pair.

The finite engine keeps `:biorth` as its default: its boxes are not mirror-symmetric at every
interface, and its sweeps settle within ~15.

### Two things that are not references

* **`plane_rest = :exact` in the infinite engine.** Open rest faces expose the two octants' full
  entanglement across a quarter-plane that grows every iteration. The spectrum flattens without
  bound: at β = 0.15, χ = 4, iteration 40, the discarded weight is 5e-3 and s₅/s₄ = 0.88, against
  1e-5 compressed. ln κ drifts with it, 5e-4 below the compressed value at iteration 40 and still
  falling. In the infinite system the compression is what makes a fixed point exist.
* **A χ that splits a near-degenerate pair.** The plane spectrum at β = 0.15 is
  (1, 0.052, 0.043, 0.043, 0.0025, …). χ = 3 cuts through the pair and lands 4.5e-5 away from
  both χ = 2 and χ = 4 (ln κ 0.72853461 / 0.72857951 / 0.72853630). Choose χ at a gap in the
  spectrum.

### What the fixed point is worth: 3D Ising

Isometric pairs, fixed-spin seed, iterated until the largest block change is ≤ 1e-9 or 400
iterations have run; m is read by the edge estimator (below). The Talapov–Blöte fit to Monte Carlo is
m = t^0.32694 (1.69190 − 0.34358 t^0.50842 − 0.42572 t), with t = 1 − β_c/β and β_c = 0.2216544:

| β | χ = 2 | χ = 4 | χ = 6 | χ = 8 | Talapov–Blöte |
|---|---|---|---|---|---|
| 0.20 | 0 | 0 † | | | 0 |
| 0.21 | 0.2953 ‡ | † | | | 0 |
| 0.215 | 0.3996 | † | | | 0 |
| 0.22 | 0.4807 | 0.4074 | | | 0 |
| 0.2217 | 0.5050 | 0.4488 | | | 0.1052 |
| 0.225 | 0.5485 | 0.5150 | | | 0.4156 |
| 0.23 | 0.6057 | 0.6019 | | | 0.5454 |
| 0.24 | 0.6944 | 0.6936 | | | 0.6758 |
| 0.25 | 0.7581 | 0.7579 | 0.7577 | 0.7575 | 0.7509 |

† No stable fixed point. The plane pair problem collapses to rank 4 with an exact doublet
(spectrum 1, 0.2226, 0.2226, 0.0496, 4e-9), the blocks rotate by a constant 4–6% per iteration,
and m flows smoothly from +0.06 through zero to −0.36 (β = 0.215). The (non-physical) negative
values of the pre-fix scan were this too.

‡ 400 iterations, still moving at 5e-6.

What the table shows:

* **The finite-χ transition is mean-field-like and hot.** At χ = 2, m ∝ (β − β_c)^½ fits m(0.21),
  m(0.215) and m(0.22) with β_c(2) = 0.2039 ± 0.0001, a transition 8% hotter than the true one.
  χ = 4's transition lies in 0.215–0.22, where its iteration has no stable fixed point.
* **Deep in the ordered phase the bias barely moves with χ.** At β = 0.25 the fixed point is 0.9%
  high at χ = 2, 4, 6 and 8 (0.75806, 0.75788, 0.75769, 0.75754): −1.8e-4 per step of 2 in χ,
  so closing the 6.6e-3 gap this way would take χ ≈ 80 (χ = 8 alone: ~5 s per iteration on 8
  threads, 200 iterations). That is the construction, not convergence. One χ-dimensional index
  per quarter-plane of bonds, fed back through the rest compression, is a mean-field-like
  treatment of a 2D boundary.
* **Convergence is slow.** χ = 2 needs 26–135 iterations away from its transition. At χ = 4, 400
  iterations leave 2e-6 to 9e-4 of block change at β ≥ 0.22, although m is steady to ~1e-7 at
  β = 0.25 by iteration 150.

## THE CEILING: exact vertex regions cost ~χ¹⁰–χ¹²

A vertex region is the vertex plus its 26-block shell. The shell's graph is the cube's face
lattice (8 corners — 12 edges — 6 faces, every link χ), and its contraction does not separate
along an 8-leg equator as the caps alone suggest: absorbing the 8-block belt between the two caps
carries 10 χ-legs at once. Measured contraction complexity (OMEinsumContractionOrders, the infinite
engine's actual region networks with every χ-leg rescaled; log₂ entries / flops):

| region | tensors | space χ=4 | space χ=8 | time χ=8 | greedy space χ=8 |
|---|---|---|---|---|---|
| vertex (shell + site) | 27 | 2^24 | **2^36** | 2^48 | 2^39 |
| shell alone, 6 site legs open | 26 | 2^24 | 2^36 | 2^49 | — |
| edge | 18 | 2^16 | 2^24 | 2^36 | 2^24 |
| face | 12 | 2^12 | 2^18 | 2^25 | 2^18 |
| cube | 8 | 2^8 | 2^12 | 2^19 | 2^12 |

TreeSA's best vertex order is ~χ¹² in space; a hand order (top cap χ⁸ → belt → bottom cap) peaks
at ~χ¹⁰·D⁵. Either way the exact vertex region — and with it the Möbius `F` and every single-site
observable read through the shell — stops at χ ≈ 5 (2^36 entries at χ = 8 is 0.5 TB). The sweeps
themselves do not have this problem: plane pairs are ~O(χ⁷), block rebuilds similar, cube and face
regions cheap. In the finite engine the boundary blocks are thin, which is why 4×4×4 at χ = 4 was
cheap; a bulk vertex of a larger box hits the same wall.

So larger χ in 3D needs either an APPROXIMATE shell contraction, or estimators that avoid the full
shell. For single-site observables the second works — `site_ratio(ic, impurity; method = :edge)`:

### Cheap single-site observables: through LINE interfaces, never plane ones — *2026-09-24*

Measured on decoupled 2D layers (Jz = 0, K = 0.5, fixed-spin seed) against Yang's exact
spontaneous magnetisation (1 − sinh⁻⁴2K)^(1/8) = 0.91131938:

| χ | `:shell` (vertex shell) | octant regrown (removed) | `:edge` (edge region, half-line regrown) |
|---|---|---|---|
| 2 | 2.0e-4 | 4.0e-3 | 2.0e-4 |
| 4 | 2.9e-6 | 3.9e-3 | 3.2e-6 |
| 6 | unaffordable | 3.9e-3 | **4.3e-9** |
| 8 | unaffordable | 3.9e-3 | — |

* **Regrowing an OCTANT** to expose its corner site (~χ⁷) put the site's three outer bonds through
  the octant's PLANE projectors, and sat 3.9e-3 off at every χ from 2 to 8. A quarter-plane of
  bonds is an area-law object — here literally a product over infinitely many layers — that no
  fixed χ holds. Removed.
* **Regrowing a HALF-LINE** inside the 18-block edge region across an axis keeps the site's bond
  along that axis raw and sends its four transverse bonds through LINE projectors — 1D
  interfaces, compressed as well as the 2D engine's. It tracks the shell estimator where both run
  and converges exponentially beyond; the edge region costs ~χ⁸ (2^24 entries at χ = 8).

The same lesson applies to anything read off the 3D environment: keep the quantity's own bonds
on half-line blocks.

For CONVERGENCE at larger χ, `update(ic; convergence = :blocks)` watches the largest phase-free
change of any block (their kept indices are fixed and aligned, so they compare directly) and
never contracts a vertex region.

Iteration cost (infinite engine, single-threaded, 3D Ising): χ = 6 1.1 s (12 plane pairs 0.52 s,
12 line pairs 0.15 s, 26 blocks 0.41 s); χ = 8 8.0 s (4.6 / 1.1 / 2.3 s) — the χ⁷ of the plane
pairs. The pairs, blocks and regions of an iteration are independent and run on Julia threads
(single-threaded BLAS, as in 2D).

## Costs and limits

Measured on 4×4×4 at χ = 4 (warm, single thread): 1.0 s per sweep (324 plane pairs at 1.4 ms, 432
line pairs at 0.8 ms, 936 blocks) and 0.5 s per F (the 64 vertex regions 0.38 s of it). The
earlier ~50 s per run was compilation.

Scaling (dense, single layer, bond dimension D):

* **Plane pairs** — an enlarged octant with two faces compressed has χ⁵D entries and costs
  ~O(χ⁷D) to form; the `:cut` QR or `:cycle`'s `b aᵀ` another O(χ⁷D).
* **Memory** — ~26·L³ blocks, E and T of χ⁴ and χ⁴D entries: ~2 GB per state at L = 6, χ = 16.
* **Region contractions (F, observables)** — a vertex region is a closed shell of 26 blocks, and
  its exact contraction needs ~χ¹⁰–χ¹² memory (THE CEILING above; the ~χ⁸ once estimated from an
  8-leg equator does not survive the belt). This, not the sweep, caps the exact CVM free energy at
  χ ≈ 5. Edge regions cost ~χ⁸, face ~χ⁶, cube ~χ⁴.

The infinite engine removes the L³: one iteration derives 24 pairs and grows 26 blocks, whatever
the lattice size.

## Open problems

1. **The vertex-region ceiling (χ ≈ 5 for the exact Möbius F).** Single-site observables have a
   way round it (`:edge`); the free energy per site does not yet — an approximate shell
   contraction, or thermodynamic integration of a bond energy read through line interfaces, are
   the candidates.
2. **`:cycle` in 3D.** The cut-closed cube converges but to a worse F than `:cut`, and is not
   better on the corner/centre observable. Whether a joint cube solve (all 12 interfaces at once,
   as the 2D ring does for 4) would change that is untested; its cost is ~12× a sweep. Its plane
   pairs are biorthogonal, so in the infinite engine the side-asymmetric instability presumably
   applies to them too (untested).
3. Observables through a single shell are 3 orders less accurate than F at χ = 4.
4. Dense data only; the double layer (3D PEPS norms) and graded tensors are untested.
5. **The infinite fixed point is biased, and χ barely helps.** For 3D Ising: m is 0.9% high at
   β = 0.25 for χ = 2–6, and the transition is mean-field-like and 8% hot at χ = 2. At χ = 4 there
   is no stable fixed point just below its own transition. A single χ-index per quarter-plane of
   bonds cannot hold a 2D boundary.

   For the thermodynamic limit near criticality, the natural next construction is a
   **boundary iPEPS**: the 3D network's layer transfer operator acting on a 2D PEPS boundary
   state of bond dimension D_b, with every expectation value contracted by the existing 2D
   CTMRG. The boundary then carries a 2D tensor network instead of one index.
6. **Convergence speed.** Even where it converges, the infinite iteration contracts at only
   ~0.9–0.95 per iteration away from criticality (χ = 4: ~150 iterations to steady m). An
   accelerated fixed-point solve (Anderson/DIIS on the aligned blocks) is the obvious candidate.
