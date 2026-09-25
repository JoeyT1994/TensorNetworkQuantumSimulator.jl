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
* χ = 8 ran out of memory in ln κ — see the next section.

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
* **Region contractions (F, observables)** — a vertex region is a closed shell of 26 blocks, a
  SPHERE, whose balanced separators cut ~8 legs: an exact contraction needs ~χ⁸ memory. This, not
  the sweep, is the ceiling on χ for the CVM free energy (χ ≈ 10). Cube regions (8 octants) are
  only ~χ⁴.

The infinite engine removes the L³: one iteration derives 24 pairs and grows 26 blocks, whatever
the lattice size.

## Open problems

1. **The vertex-region ceiling (χ ≈ 5 for the exact Möbius F).** Single-site observables have a
   way round it (`:edge`); the free energy per site does not yet — an approximate shell
   contraction, or thermodynamic integration of a bond energy read through line interfaces, are
   the candidates.
2. **`:cycle` in 3D.** The cut-closed cube converges but to a worse F than `:cut`, and is not
   better on the corner/centre observable. Whether a joint cube solve (all 12 interfaces at once,
   as the 2D ring does for 4) would change that is untested; its cost is ~12× a sweep.
3. Observables through a single shell are 3 orders less accurate than F at χ = 4.
4. Dense data only; the double layer (3D PEPS norms) and graded tensors are untested.
