> **⚠️ START WITH [`ctmrg_status.md`](ctmrg_status.md).** This file is the full chronological record:
> it is long, it contains claims that were later retracted (marked as such where they appear), and
> several of its measured tables predate fixes that changed them. The status doc is the current
> state, the current numbers, and the current next steps. Come back here for the derivations and for
> the record of what was tried and failed.

# Finite CTMRG with CVM regions — design and current state

## Goal

A genuine **finite, position-resolved CTMRG**: a 4C+4T environment on *every vertex*, grown
and projected by local corner moves, with the free energy taken from the region-graph (CVM)
functional

```
F  =  Σ_v ln Z_v   −   Σ_e ln Z_e   +   Σ_p ln Z_p            (Möbius numbers +1 / −1 / +1)

Z_v : vertex ring     = 4C + 4T + a        (parent region)
Z_e : edge strip      = 4C + 2T            (edge overlap)
Z_p : plaquette loop  = 4C                 (corner overlap)
```

made stationary. This is **not** boundary MPS and is intended to supersede it: the regions are
local per-vertex objects, there is no planarity requirement, and the corner tier is
information that BP and boundary MPS do not carry.

Exactness check: for a disk, `V − E + P = 1` (on `L×L`: `L² − 2L(L−1) + (L−1)² = 1`), so when
every region contracts to the exact `Z` the weighted sum returns `ln Z`. At finite `χ` the
region errors cancel to a large degree — that cancellation is the point of the method
(measured earlier: per-region ~1e-5 collapsing to ~1e-9 in the sum).

## Removed: the row-absorption contractor (do not bring it back)

An earlier version of `ctmenvironmentcache.jl` (commits `97f4092`, `9542756`) absorbed the
lattice **row by row** and exposed **row** environments, via `partitionfunction`, `freenergy`,
`row_environments` and `contract_row`. That is a boundary-MPS-shaped contraction, not per-vertex
CTMRG. It was accurate as a contractor but it was the wrong object, and having it sitting in the
same file made the engine read as boundary MPS. **It has been deleted.**

Why it could never support this method:

* environments are rows, not per-vertex 4C+4T rings;
* there are no corner tensors at all, so no CVM regions and no `Σ_v/Σ_e/Σ_p` sum;
* a "region" carved out of it cannot differ between vertices — splitting a fixed chain at
  different columns returns the same scalar, so the Möbius sum degenerates to `(V−E+P)·ln Z`.

Genuine regions require each corner to be an **independently truncated, few-leg object**,
closing into a small ring (~χ⁴), *not* a slice of a whole-lattice contraction.

What survived the deletion, because the CVM path uses it: `_ctm_eigsolve`,
`_ctm_eig_projector`, `_ctm_psd_factor`, `_ctm_twosided_projector`, `_ctm_contract` and the
tuning knobs (since migrated to `CTMOptions`, see below). The `CTM_TWOSIDED` flag went with it —
it only ever switched the chain sweep, and the CVM path is unconditionally two-sided.

If you need an independent reference number, use `contract(tn; alg="boundarymps",
mps_bond_dimension=χ)` or `alg="exact"`, not a resurrected row engine. The cache's `grid` field
is grid *geometry* (`grid[y][x]`), not a row decomposition.

## Target design

### Objects (all position-resolved)

```
C_NW[x,y] = block of all vertices with col<x, row<y        (and NE / SW / SE)
T_N[x,y]  = column x, rows<y     T_S[x,y] = column x, rows≥y
T_W[x,y]  = cols<x, row y        T_E[x,y] = cols≥x, row y
```

### Interface projector families (nested chains, each derived exactly once)

```
PH_N[x,y] : {hb[(x,j)] : j < y}    nested in y     shared by C_NW.right, C_NE.left, T_N.left/right
PH_S[x,y] : {hb[(x,j)] : j ≥ y}    nested down     shared by C_SW.right, C_SE.left, T_S.left/right
PV_W[x,y] : {vb[(i,y)] : i < x}    nested in x     shared by C_NW.down, C_SW.up, T_W.up/down
PV_E[x,y] : {vb[(i,y)] : i ≥ x}    nested down     shared by C_NE.down, C_SE.up, T_E.up/down
```

Each interface is shared by several blocks, so its projector **must be a single object** —
deriving it twice from different sides yields inconsistent bases and silently corrupts the
contraction. Assign one deriving corner per family (NW derives PH_N and PV_W; SE derives PH_S
and PV_E; NE and SW consume).

### The move — grow, then project

Corner growth absorbs the two adjoining edge tensors and the vertex tensor (this is the local
CTMRG move; it is what the collaborator specified):

```
C̃_NW(x+1,y+1) = C_NW(x,y) · T_N(x,y) · T_W(x,y) · a(x,y)
```

Its two open interfaces are exactly `PH_N[x,y+1]` (right) and `PV_W[x+1,y]` (down), each a
nested extension of the previous level. Eigendecompose to get the projectors, then

```
C_NW(x+1,y+1) = C̃ · P_right · P_down
```

Edges grow analogously: `T_N(x,y+1) = T_N(x,y)·a(x,y)`, projected on both sides.

### Projector: two-sided, from a Hermitian eigendecomposition (no SVD)

With `ρ_L = A†A` and `ρ_R = B†B` for the two half-environments of an interface, `H = B ρ_L Bᵀ`
is Hermitian PSD with eigenvalues `S²` and eigenvectors `V`:

```
P_A = Bᵀ V S^(-1/2)          P_B = S^(-3/2) Vᵀ B ρ_L
```

`A (P_A P_B) Bᵀ = A Bᵀ` exactly at full rank. Both come from **one** eigenbasis — using two
separate decompositions for `U` and `V` invites per-vector sign/phase mismatch and arbitrary
rotation inside degenerate clusters, and double-layer corners *do* have degeneracies
(`λ_ij = λ_ji` under ket↔bra exchange).

Hard-won numerical points (all measured, see `ctmenvironmentcache.jl`):

* **`pinv_cutoff ≈ 1e-8 ≈ √eps`, not 1e-12.** The inverse powers of `S` amplify roundoff,
  and `S` comes from an eig of a squared object so it is only resolved to ~√eps relatively. A
  null-space-only cutoff leaves a hard, χ-independent error floor (measured: 9.5e-7).
* **Krylov needs an isometry guard.** `eigsolve` returns non-orthonormal vectors on degenerate
  clusters, silently corrupting the projector (err 3.9 vs 0.027). Check `‖V'V − I‖` and fall
  back to dense.
* **One-sided truncation is not a valid variational choice** and makes the error non-monotonic
  in χ. This was the root cause of a long chain of confusing results.

### Triangular/QR projector — implemented, accuracy-NEUTRAL, kept for GPU (`opts.qr`, default ON)

Take a thin QR of each bounding block instead of forming `ρ` and eigendecomposing it back into a
square root: `Bw = Q_A R_A`, `Be = Q_B R_B`, so `R_A† R_A = ρ_L` and `R_B† R_B = ρ_R` *exactly*,
with no squaring. The `Q`s are isometries, so the whole problem reduces to the small triangular
product `W = R_A R_B†`, and with `W = U S V†`:

```
P_A = R_B† V S^(-1/2)          P_B = S^(-1/2) U† R_A
```

`A (P_A P_B) B† = A B†` exactly at full rank (verified to 1e-15), and the `S^(-3/2)` on `P_B`
collapses to a symmetric `S^(-1/2)`. Still one decomposition, so `U` and `V` stay in a consistent
basis — the reason the ρ route avoided separate eigen-solves.

**Do not expect an accuracy win — measured, it is a wash.** It matches the ρ route to 3
significant figures on 18 moderate-χ configurations and 10 near-lossless ones, at cutoffs
1e-8/1e-11/1e-13/1e-15 alike, and passes the CTM suite 100/100 as a drop-in default.

**Why, and this is the useful part:** *precision is not the binding constraint — χ is.* The
retained spectrum has median `S_k/S_1` of 1e-1…1e-2 (measured over the 200–384 solves in a
sweep) and **0%** of retained directions fall below 1e-8. In the ρ route a direction at
`S_k/S_1 = 1e-8` carries relative error `~eps·(S_1/S_k)² ≈ 50%`, which is exactly why
`pinv_cutoff` sits at √eps: **the cutoff and the squaring are two faces of one constraint.**
QR makes directions down to ~1e-15 usable, but they carry no weight. So accuracy per χ cannot be
bought with better arithmetic — only by changing *which subspace is kept*.

The reason to keep it is GPU: `geqrf`/`gesvd` have batched GPU implementations where batched
Hermitian eig support is thin, and a sweep is 200–384 **independent tiny** factorizations
(`n ≤ 128`) — a batching problem, not a big-linear-algebra one. Note the remaining blocker in
*both* routes: `Array(ρ, b, bp)` and `_ctm_block_matrix` materialise host `Array`s, so every
projector currently round-trips through the CPU. QR alone does not deliver GPU.

### Arnoldi is dormant on the CVM path at small D (measured)

`arnoldi = true`, but the `n > 4k` gate almost never fires here, because a CVM interface is
only `χ · D_layer` (`D_layer` = D single-layer, D² double) so `n > 4χ` needs `D_layer > 4`:

| network | interface `n` | Arnoldi fired |
|---|---|---|
| single-layer D=3, χ=8 / χ=16 | ≤ 24 / ≤ 27 | **0%** |
| double-layer D=2, χ=8 | ≤ 32 | **0%** (misses by one — the test is strict `>`) |
| double-layer D=3, χ=8 | ≤ 72 | 66% |
| double-layer D=4, χ=8 | ≤ 128 | 67% |

The `4k` heuristic was tuned for the deleted row engine, where `n` was the whole chain bond and
could be huge. Retuning it for CVM interface sizes is open work — and note Arnoldi is hostile to
the batching above (iterative, data-dependent iteration counts, a `randn` per call).

`degtol = 0.0`, i.e. **eigenvalue pair-keeping is off**, consistent with it having measured
as a no-op on single layer and marginal on double layer.


### Lanczos warnings at larger D, continuation in χ, and why observables lag

**The KrylovKit warnings are diagnostic, not corruption.** `Invariant subspace of dimension 6 …
howmany == 10` says the interface's *effective rank* is 6 while χ=10 was requested — routine once
D grows. `_ctm_eigsolve` already falls through to dense whenever `converged < k`, and the dense
result is **bit-identical and deterministic** (three runs with different RNG streams: spread
~1e-14; `arnoldi` on vs off: identical). Now silenced with `verbosity = 0`, since warning on
every such call buries real problems.

**Continuation in χ buys nothing — the fixed point is unique.** Warm-starting a χ run from the
converged environments of the previous χ (`_ctm_setenv`) gives results **bit-identical** to
starting from the greedy seed, at every χ on three networks. The sweep's fixed point is
seed-independent. Good robustness property; not an accuracy lever. It can only ever save sweeps.

**`marginal_inconsistency` is monotone in χ where `|F − ln Z|` is not.** Double 3×3 D=4:

| χ | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| `\|F − ln Z\|` | 4.6e-2 | 5.9e-3 | **1.0e-2** | 3.0e-3 | 1.7e-4 |
| `marginal_inconsistency` | 1.13e-2 | 7.5e-3 | 4.9e-3 | 3.3e-3 | 1.9e-3 |

The bump at χ=8 is cancellation noise in `F`; the diagnostic decays smoothly throughout. Use it.

**Why single-site observables lag boundary MPS — a structural limit, not a tuning problem.**
Möbius cancellation requires a quantity to appear in several regions with opposite signs. Every
`C` and `T` does (4 regions `+1−1−1+1`, or 2 with `+1−1`), which is exactly why `F` is so accurate.
A **site** does not:

| region | weight | contains a site? |
|---|---|---|
| vertex | +1 | **yes** |
| h-edge | −1 | no |
| v-edge | −1 | no |
| plaquette | +1 | no |

Only the vertex region holds a site, so a single-site observable is read off **one** region with
weight `+1` and gets **no cancellation at all** — it inherits the raw per-region error while `F`
gets the cancelled one. That is the whole gap. Closing it needs a **finer region graph** whose
child regions also contain the site (so the site's marginal is over-counted and then corrected),
not a better projector or more sweeps.

### Strip estimator for observables — TESTED, no gain. The deficit is truncation, not geometry.

Boundary MPS gets local observables right by taking the whole strip containing the site *exactly*
and putting compressed messages only above and below. CTMRG can build the same object out of
pieces it already has: `T_N[x,y]` is column `x`, rows `< y`, so `{T_N(x,y)}_{x=1..Lx}` tiles
everything above row `y` and the `PH[:N,x,y]` interfaces chain them into **exactly an MPS**;
likewise `{T_S(x,y+1)}` below. The strip is then

```
{T_N(x,y)}ₓ  ·  {a(x,y)}ₓ  ·  {T_S(x,y+1)}ₓ          (row y kept EXACT)
```

versus the ring, which compresses the within-row context into `T_W(x,y)` and `T_E(x+1,y)`. It is
~10 lines and validated: observables from it are exact at lossless χ. (Its absolute value is *not*
`Z` — block renormalisation again, so only ratios are meaningful, same caveat as `region_lnZ`.)

**Measured, it does not help.** 4×4 D=2, error in `⟨Z⟩`:

| χ | vertex | ring | strip | bMPS | best |
|---|---|---|---|---|---|
| 4 | (2,2) | 1.03e-3 | 1.72e-3 | **2.27e-4** | bMPS |
| 4 | (3,2) | 6.07e-4 | 3.11e-4 | **2.28e-5** | bMPS |
| 8 | (2,2) | 1.034e-4 | 1.032e-4 | **4.78e-5** | bMPS |
| 8 | (3,2) | 1.25e-4 | 1.10e-4 | **3.90e-6** | bMPS |

**strip ≈ ring ≪ bMPS.** Adopting boundary MPS's geometry does not recover its accuracy, so the
gap is **not** the ring-vs-strip shape. It is *how the `T`s were truncated*: `PH[:N,x,y]` is
derived from the two enlarged **corners** `C̃_NW(x+1,y) | C̃_NE(x+1,y)`, i.e. optimised for the
corner–corner contraction, whereas boundary MPS **variationally fits** its row MPS to the very
contraction it will be used for.

**The tension this exposes is structural.** Each interface must carry *one* projector — deriving
it twice from different sides gives inconsistent bases and corrupts the contraction (see the top
of this document). But the projector that makes the CVM regions mutually consistent (corner-derived)
is *not* the projector that makes the strip accurate (fit-derived). **One projector set cannot be
optimal for both.** Any real fix has to choose: either fit the `T`-chain for the strip and give up
region consistency, or keep the corner-derived projectors and accept that local observables inherit
the raw per-region error. This — together with the fact that only the vertex region contains a site,
so observables get no Möbius cancellation — is the complete account of why CVM beats boundary MPS
on `ln Z` and loses on `⟨O⟩`.

Not added to the engine: it is not better, and a second observable path that is sometimes worse is
worse than none. Recover from this commit if a future region graph changes the trade-off.

### Speed vs boundary MPS at matched χ (measured)

`update` + `cvm_freenergy` against `contract(...; alg="boundarymps")` / `norm_sqr(...;
alg="boundarymps")`. **Each configuration is timed twice and the second reported** — the netcon
sequence cache is keyed on shape, so the first call at a new lattice size pays all the
optimisation and JIT. Getting this wrong made two earlier benchmarks in this document misleading.

| case | χ | CTM s / err | bMPS s / err | faster | more accurate |
|---|---|---|---|---|---|
| single 6×6 Ising K=.44 | 4 | 0.082 / 1.5e-11 | 0.012 / 2.5e-9 | bMPS 7.0× | CTM 166× |
| | 8 | 0.018 / 1.1e-14 | 0.020 / 0 | CTM 1.1× | bMPS |
| | 16 | 0.020 / 3.6e-15 | 0.012 / 0 | bMPS 1.8× | bMPS |
| single 5×5 random D=3 | 4 | 0.310 / 4.6e-1 | 0.008 / 1.4e0 | bMPS 38× | CTM 3× |
| | 8 | 0.212 / 8.1e-4 | 0.008 / 4.2e-1 | bMPS 25× | **CTM 510×** |
| | 16 | 0.025 / 7.1e-15 | 0.009 / 3.6e-15 | bMPS 2.9× | bMPS |
| double 4×4 PEPS D=2 | 4 | 0.131 / 3.4e-4 | 0.067 / 2.3e-4 | bMPS 2.0× | bMPS |
| | 8 | 0.091 / 6.6e-6 | 0.077 / 3.8e-6 | bMPS 1.2× | bMPS |
| | 16 | 0.015 / 0 | 0.047 / 0 | CTM 3.1× | — |
| double 4×4 PEPS D=3 | 4 | 0.318 / 4.8e-2 | 0.420 / 3.7e-2 | CTM 1.3× | bMPS |
| | 8 | 0.423 / 1.1e-3 | 0.624 / 1.8e-3 | **CTM 1.5×** | **CTM** |
| | 16 | 0.517 / 2.6e-5 | 0.696 / 5.5e-5 | **CTM 1.3×** | **CTM** |

**Single layer: boundary MPS wins on speed, decisively at small χ** — up to 38×, because CTM pays
8–30 sweeps regardless of χ while boundary MPS does essentially one pass. CTM's cost barely falls
as χ drops, so its floor is the sweep count, not the linear algebra.

**Double layer at D=3: CTM is faster (1.3–1.5×) AND more accurate at χ ≥ 8.** This is the regime
the method is actually for — the crossover is with `D_layer = D²`, where CTM's corner tier starts
paying for itself.

**But on time-to-target-accuracy, boundary MPS still usually wins.** The most CTM-favourable cell
is single 5×5 D=3 at χ=8, where CTM is 510× more accurate — yet boundary MPS reaches machine
precision at χ=16 in 0.009 s, faster than CTM's 0.025 s. Matched-χ accuracy is the wrong headline
unless χ is the binding constraint (memory, or an observable that needs a specific environment).

**Actionable:** the single-layer gap is the sweep count, not arithmetic. That is exactly what
gauge fixing unlocked and nobody has spent yet — Anderson acceleration on the now-well-posed
fixed point is the highest-value remaining performance work, and it attacks the one axis where
boundary MPS is 25–38× ahead.

### Sparse (x,y) grids: hexagonal and heavy-hexagonal — WORKS with the unit-square regions

Hex and heavy-hex are laid out on an `(x,y)` grid with vertices *and* edges missing. All edges are
grid-adjacent (displacements only `±(1,0)`, `±(0,1)`), so the embedding is faithful and the engine
needed only sparse-grid plumbing: `grid` is now a `Dict` of occupied positions plus an explicit
bounding box, with `nothing` at empty slots.

**I first claimed this would degenerate to BP, on the grounds that no unit square is a face of a
hex lattice. That was WRONG on two counts and the measurements below refute both.**

1. **The Möbius identity has nothing to do with graph faces.** It is a telescoping identity on the
   **bounding box**: `Lx·Ly − (Lx−1)Ly − Lx(Ly−1) + (Lx−1)(Ly−1) = 1`, independent of which slots
   are filled. And every region still contracts to `Z`, because `C_NW = {col<x, row<y}` and the `T`
   strips partition positions by **comparison**, not occupancy — an empty vertex slot simply has no
   site factor to insert, and `T_W(x,y) ∪ T_E(x+1,y)` still covers row `y`.
2. **The plaquette region is the four-quadrant overlap of a CUT, not a lattice face.** A cut does
   not care whether the unit square it straddles happens to be a face. The corner tensors remain
   genuine 2D blocks carrying information BP does not have.

Measured, `|F − ln Z|`, `bond_dimension = 2`, χ=40:

| lattice | V | E | bounding box | holes | error |
|---|---|---|---|---|---|
| hex 2×2 | 16 | 19 | 6×3 | 2 | 1.3e-15 |
| hex 3×3 | 30 | 38 | 8×4 | 2 | 3.2e-13 |
| heavy-hex 2×2 | 35 | 38 | 11×5 | **20** | 4.4e-16 |

Exact even with 20 of 55 slots empty. And against BP at `D=3`:

| lattice | BP error | CVM χ=2 | χ=4 | χ=8 |
|---|---|---|---|---|
| hex 3×3 | 8.6e-2 | 21× better | 46× | **2.7e8×** (3.2e-10) |
| hex 4×4 | 1.47e1 | 73× | 12× | 114× |
| heavy-hex 3×3 | 3.10 | 12× | 4.2e5× | **2.2e14×** (1.4e-14) |

Better than BP by 12–73× even at χ=2. **No hexagonal region graph is required for correctness or
for beating BP.**

What is still open is *accuracy*, not correctness: whether a face-based region graph (regions per
vertex / per edge / per hexagon, with hexagons being 2×1 bricks in this embedding) would do better
than the cut-based unit-square one at matched χ, since hexagons are 6-cycles. That is now a
measurable question against a working baseline rather than a prerequisite.

Coverage: `test_ctmenvironment.jl`, testset "CVM on sparse (x,y) grids".

### Observable accuracy at fixed χ: the WINDOW estimator — this one works

The observable used the 1×1 window (`4C + 4T + a`). The same cut construction works for any
rectangular window — cuts at `(xL, xR, yT, yB)` give

```
4C  +  T_N/T_S on columns xL … xR−1  +  T_W/T_E on rows yT … yB−1  +  interior sites except v
```

which tiles the lattice for any window, so it stays exact at lossless χ, and `w = 0` reproduces the
ring exactly. **Every block already exists in the cache**, so a larger window costs only a larger
contraction — no extra sweeps and no extra truncation of the blocks. `vertex_window(cache, v, w)`,
with `expect`/`rdm` taking a `window` keyword.

Measured, 6×6 D=2 PEPS, error in `⟨Z⟩`:

| χ | vertex | w=0 (ring) | w=1 (3×3) | bMPS |
|---|---|---|---|---|
| 2 | (3,3) | 9.96e-2 | **1.11e-2** | 2.16e-2 |
| 2 | (4,3) | 8.06e-2 | **7.07e-3** | 5.68e-2 |
| 4 | (2,2) | 4.25e-3 | **6.83e-4** | 3.11e-3 |
| 6 | (3,3) | 1.45e-3 | **1.02e-3** | 1.49e-3 |
| 6 | (4,3) | 2.75e-3 | **6.71e-4** | 1.37e-3 |
| 6 | (2,2) | 2.51e-4 | **9.26e-5** | 1.47e-4 |

`w = 1` beats the ring at **8 of 9** (site, χ) combinations by 1.4×–11.4×, and beats boundary MPS
at **6 of 9 — including all three sites at χ=6.** The single exception is a near-boundary site at
χ=2, where the ring was barely truncated and the extra interfaces cost more than the exact context
buys.

**⚠️ BUT MATCHED-χ IS THE WRONG METRIC, and I made exactly the mistake I flagged for boundary
MPS.** Raising χ dominates the window. 6×6 D=2, mean `|err|` over three sites:

| config | `update` s | obs ms/site | mean \|err\| |
|---|---|---|---|
| χ=6, w=0 | 1.90 | 0.4 | 1.48e-3 |
| χ=6, w=1 | 0.41 | 1.9 | 5.95e-4 |
| **χ=10, w=0** | **0.37** | **0.3** | **5.59e-5** |
| χ=24, w=0 | 1.05 | 0.3 | 3.03e-6 |

**χ=10 with the plain ring is 10× more accurate than χ=6 with `w=1`, and cheaper on both axes.**
(`update` times are noisy from shape-cache effects; the error column and the per-site column are
solid.) Warmed, `w=1` costs ~5× per observable — an earlier 380× figure was mostly first-call
sequence optimisation, so the cost was overstated *and* the benefit was.

**So `window` stays `0` by default, and that is now a data-backed choice rather than caution.** It
is worth reaching for only when χ is the binding constraint — memory-bound runs, or an environment
you cannot rebuild — which is the same narrow condition under which matched-χ comparisons against
boundary MPS mean anything.

**This still corrects an earlier over-pessimistic conclusion in this document.** The observable deficit
was diagnosed as structural — only the vertex region contains a site, so no Möbius cancellation is
available, and closing the gap would need a finer region graph. That diagnosis of the *mechanism*
was right, but the *conclusion* was wrong: you do not need cancellation, you need more exact
context, and the existing blocks already supply it. No new region graph required.

Implementation note: `alg = "optimal"` is ExhaustiveSearch netcon, exponential in **tensor count**,
and it hangs outright on the ~25-tensor lists a `w = 1` window produces. `optimal_max` (12)
gates it to the greedy optimiser above that — a feasibility gate, not the performance tweak of the
same shape that was tried and reverted earlier. `expect`/`rdm` now route through `_ctm_contract` so
they get both that gate and the sequence cache.

### Full re-derivation of the projector and sweep: no missed accuracy win, and why

**The projector truncation is provably OPTIMAL for the bipartition it is given.** Substituting the
truncated pair back:

```
A P_A^(k) P_B^(k) B†  =  Q_A [W V_k S_k⁻¹ U_k†] W Q_B†  =  Q_A U_k S_k V_k† Q_B†  =  Q_A W_k Q_B†
```

using `W V_k = U_k S_k` and `U_k† W = S_k V_k†`. `Q_A` and `Q_B` are isometries, so this is the
**Eckart–Young optimal rank-k approximation of `A B†`**. Verified numerically against an explicit
truncated SVD of `A B†` on four shapes: error ratio **1.0000000000** to ten decimals. So no better
choice of subspace exists for that objective, which independently confirms the swap experiment.

**The one assumption in that derivation is the bipartition — and there is no better one available.**
`(C̃_NW | C̃_NE)` tiles only rows `< y`; the south closure does not weight the truncation. The
natural fix would be an environment-aware ("full update") truncation. It is not available as a
drop-in, because **the interface is never a bipartition of any region.** Measured on the plaquette,
whose four corners each share exactly one χ-dim interface:

```
NW–NE, NW–SW, NE–SE, SW–SE     — a 4-CYCLE of blocks
cut PH[:N] alone → NW–SW still connected, NE–SE still connected
```

So cutting one interface does not separate west from east even in the *smallest* region. The four
truncations around a plaquette are **coupled in a loop**: there is no reduced density matrix for a
single interface to be optimal against, and an environment-weighted scheme would have to optimise
all four jointly. That is a variational full-update scheme — substantial new machinery, not a
missed easy win. It is also the honest reason the corner-pair choice is the right local one: the
two enlarged corners are the unique pair that exactly tiles the part of the lattice the interface
lives in.

**Sweep side, re-checked:** the Jacobi structure only affects convergence rate, since the fixed
point is unique and seed-independent (measured, including χ-continuation). Block rescaling is
gauge-invariant (`F` unchanged to 1 ulp). The nested interface chains do accumulate truncation —
level `y` is built on level `y−1`'s truncated basis — which is a genuine error source, but the
remedy is larger χ, and χ is cheap here because sweep count *falls* as χ rises.

**Conclusion: the remaining accuracy lever is χ.** Five independent attempts now agree —
arithmetic route, Möbius-stationary projector, subspace swaps, harder convergence, and the window
estimator — and the projector is provably optimal for its objective. The only routes left are a
joint variational treatment of the coupled truncation loop, or a different region graph.

### Is the projector optimal for OBSERVABLES? No — but the reason is cancellation, not the criterion

The ring is exactly `∂Z/∂a_v` (punch the site out, legs open), so the natural question is whether a
projector chosen to minimise `‖A B† − A P_A P_B B†‖_F` — which targets **Z** — is also right for
`E_v`. It is not: an interface error `δ` reaches `Z` weighted by `∂Z/∂(interface)` but reaches `E_v`
weighted by `∂²Z/∂a_v∂(interface)`. Different weightings, so optimal for one is generically not
optimal for the other.

**Measured, 6×6 D=2, site (3,3):**

| χ | rel err `ln Z` | err `⟨Z⟩` | err rdm | ratio |
|---|---|---|---|---|
| 2 | 6.18e-4 | 9.96e-2 | 8.02e-2 | 161× |
| 4 | 1.30e-5 | 2.78e-3 | 2.39e-3 | 214× |
| 6 | 4.63e-7 | 1.45e-3 | 1.28e-3 | **3127×** |
| 8 | 1.03e-6 | 6.03e-4 | 4.41e-4 | 585× |
| 12 | 2.36e-7 | 4.16e-5 | 2.95e-5 | 176× |

`E_v` is represented 160–3000× worse than `Z`. The rdm error tracks `⟨Z⟩` because the rdm *is*
normalised `E_v`.

**But the derivative argument shows the criterion is not the main culprit.** With a field `h`
coupling to `O` at `v`, `⟨O⟩ = ∂lnZ/∂h`, so applying it to the estimator:

```
∂F/∂h|₀  =  Σ_R c_R (∂Z_R/∂h) / Z_R
```

Only regions **containing v** contribute, and only the vertex region contains a site. So `∂F/∂h`
collapses to exactly `(∂Z_v/∂h)/Z_v` — the ring estimator we already compute. **The accurate
free-energy estimator, differentiated, gives back the inaccurate observable estimator.** There is no
free lunch to extract by reformulating.

So the gap is not a suboptimal subspace choice; it is that `F` earns its accuracy from Möbius
cancellation across many regions while a single-site observable lands in exactly one region and gets
none. That is consistent with the window estimator recovering only 1.4–11.4× — better local
treatment cannot touch a 3000× cancellation deficit — and with the strip estimator measuring
`strip ≈ ring`.

**The only route that addresses it is a region graph in which a site appears in several regions with
opposite Möbius signs.** Candidates: 2×2 block regions as parents (a site then sits in up to four,
with edge/plaquette overlaps that also contain sites), which restores cancellation for local
quantities the same way `V − E + P = 1` does for `Z`. Worth checking on paper that the counting
works before building: the requirement is that every *site* be covered with total weight 1 while
every *block* still is.

### Current state of the engine (the clean slate)

| piece | state |
|---|---|
| two-sided biorthogonal projector | **on** — the default path, measured optimal (see the headroom re-run) |
| interface projector | ONE route: thin QR + one SVD of the triangular product. The `ρ`-route alternative is deleted |
| `gauge` unitary gauge fixing | **on by default** — `F` invariant to 1e-14, gives the state distance |
| `marginal_inconsistency` | **live diagnostic** — the only `ln Z`-free quality measure |
| Möbius-stationary projector | **deleted** — made results worse |
| row-absorption contractor | **deleted** — wrong object |
| `degtol` pair-keeping | present, `0.0` (off) — measured a no-op on single layer, marginal on double |
| `arnoldi` | present, on, but dormant unless `D_layer > 4` |

Verified unchanged by the deletions, 4×4 D=3: `|F − ln Z|` = 5.203e-3 / 1.556e-3 / 6.455e-6 /
5.33e-15 at χ = 4 / 6 / 8 / 12, with `marginal_inconsistency` 8.02e-3 / 3.88e-4 / 1.14e-5 / 6e-17.

### The CYCLE projector — prototyped, and the answer is "it cannot be done cheaply"

The collaborator derives all four of a plaquette's projectors from ONE cyclic problem: the four
enlarged corners compose into a map on one bond, and its dominant invariant subspace gives the
projector there, propagated round the loop. Ours solves four independent two-block cuts. Theirs
enforces consistency AROUND THE LOOP; ours optimises each cut in isolation.

**The geometry maps onto ours exactly**, which is worth recording. At a cut `(X,Y)` each of the four
enlarged corners has exactly two open interfaces — the two it shares with its cyclic neighbours — so
with bonds ordered `(W, N, E, S)`:

```
C0 = E_NW : N -> W    C1 = E_NE : E -> N    C2 = E_SE : S -> E    C3 = E_SW : W -> S
```

and `M = C0 C1 C2 C3` acts on the west bond. The four projectors land on our existing keys —
`PH[:N,X-1,Y]`, `PH[:S,X-1,Y]`, `PV[:W,X,Y-1]`, `PV[:E,X,Y-1]` — so only the derivation would change,
never the consumers. No restructuring needed.

**Two prototypes, both dead, and the diagnostics say why.** Measured on square 4×4 D=3 at χ=8:

| plaquette | cycle dim | gap ratio `\|λ_{χ+1}/λ_χ\|` | condition number |
|---|---|---|---|
| (3,3) | 72 | **0.869** | **4.4e+15** |
| (2,2) | 9 | 0.396 | 1.4e+04 |
| (4,4) | 72 | 0.509 | **5.4e+18** |

* **Explicit product + `eigen`/Schur**: condition number reaches 5.4e18, past the Float64 limit. Dead.
* **Periodic subspace iteration** (apply one factor, re-orthogonalise, never form the product,
  warm-started from our pairwise projector): converges at the gap ratio per cycle, so at 0.869 three
  iterations buy a factor of 0.66 and ~100 would be needed. Dead.

Measured quality, for the record — `marg` 3.0e-4 (cut) against 3.7e-1 (cycle, 0 iters) and 5.5e-1
(3 iters) at square χ=8. The tell is that **zero iterations is already ruined**: that is no
eigensolve at all, just our own pairwise projector propagated round the loop. Propagation only makes
sense once the basis IS the invariant subspace; short of that it replaces three optimal projectors
with derived ones.

**This does NOT condemn the criterion.** The prototype omits everything their implementation treats
as load-bearing: the structured `V_from_Ac` warm start built from the corner solve equations rather
than from a pairwise projector, rank masking, the Schur gauge with exact structural zeros, and above
all a periodic Krylov–Schur solver. The two failure modes measured above are *precisely* what that
machinery exists to defeat — periodic Schur removes the conditioning, Krylov removes the dependence
on the spectral gap. Their 2,200 lines are not incidental.

**So the cost is now known rather than guessed.** Answering "is the cycle criterion better than the
cut?" requires a periodic (product) eigensolver. KrylovKit has `schursolve` but not the periodic
variant — which is why the collaborator wrote 2200 lines of Cython/SLICOT/JAX-FFI.

#### …but in Julia that solver already exists: `PeriodicSchurDecompositions.jl`

Their own `linalg/periodic_schur/julia_version.py` is a "Julia reference backend" that does
`using PeriodicSchurDecompositions` and calls `pschur!` / `ordschur!`. They needed the Cython/SLICOT
path only because they are in Python/JAX and wanted a compiled FFI. **We are in the language their
reference implementation is written in.** Registered, one small dependency
(`MatrixFactorizations`), precompiles in 2 s. Validated against what the cycle projector needs:

| check | result |
|---|---|
| `pschur!(As, :L; wantZ=true)` eigenvalues vs the explicit product `A4·A3·A2·A1` | 2.4e-13, product never formed |
| `ordschur!(P, sel; wantZ=true)` moves the dominant χ to the leading block | works (conjugate pairs kept together, as real Schur requires) |
| `A_l · Z_l[:,1:k]` lies in `span(Z_{l+1}[:,1:k])` | **3.4e-15** |

The last row is the payload: `Z_l[:, 1:k]` is an orthonormal basis at EVERY bond satisfying the
cyclic invariant-subspace relation exactly — i.e. the four projectors, in one call. Note the
convention: `:L` means the product `A_p ⋯ A_1`, so our chain must be handed over in reverse.

This defeats both failure modes measured above: no product is formed (kills the 5.4e18 conditioning)
and it is a direct decomposition (kills the dependence on the 0.869 gap ratio).

**Caveats before anyone gets excited.** The validation above is on well-conditioned random matrices;
real CTM corners are rank-deficient with condition numbers to 5.4e18, and `pschur!` may struggle
there too — test that first. And `pschur!` is DENSE, O(n³) per plaquette at n = χ·D_layer; the
collaborator's Krylov variant exists for taking only the top `k` of a large `n`. Dense is
nonetheless entirely adequate to answer whether the CRITERION beats the cut, which is the question
that gates all the rest.

#### Step 1 run: the solver is free, the RANK/RECTANGULARITY handling is not

Gate was "does `pschur!` survive real corners". It does not, as-is, and the reason is architectural
rather than numerical. Everything below is measured.

**Conventions, pinned empirically (keep these, they are not in the docstrings).** `:L` means the
product `A_p ⋯ A_1`. Searching all index combinations against the defining relation gives, at 1e-15
and with every `T` exactly upper triangular:

```
Z_{l+1}' A_l Z_l = T_l          A_l : space l -> space l+1
As = [E_SW, E_SE, E_NE, E_NW]   spaces (1,2,3,4) = (W, S, E, N)
```

so `Z_l[:, 1:k]` is the projector at bond `l`, and `A_l Z_l[:,1:k] ⊆ span(Z_{l+1}[:,1:k])` follows
from upper-triangularity.

**`ordschur!` needs a conjugate-pair-safe selection, and fails SILENTLY without one.** Cutting
between the two halves of a 2×2 real-Schur block gave a span residual of 1.7e-1 with no error
raised; backing `nev` off by one gives 2e-15. Guard the selection.

**Real corners break both routes, for two different reasons.**

* *Dense `pschur!` + `ordschur!`*: `pschur!` succeeds everywhere, but `ordschur!` fails on **every**
  real plaquette with `unexpected subdiag in triang factor 2 at 70: -6.03e-30` → "ordschur algorithm
  bug". That is its internal consistency check defeated by dynamic range: the cycle spectrum spans
  **38 orders** (max 6.93e-01, min nonzero 9.06e-39), because a 4-fold product is roughly the 4th
  power of one corner's already-wide spectrum. At 1e-30 a "subdiagonal residue" is O(1), not roundoff.
* *Krylov `partial_pschur`*: needs no reordering, and where the geometry cooperates it is excellent —
  span residual 3.8e-11, isometry 1.4e-14, zero leakage into padded rows. But it works on **1 of 9**
  plaquettes.

**Why only 1 of 9: the bonds are RECTANGULAR, and unavoidably so.** A bond's dimension is
`k_prev · D_layer`. At the lattice boundary `k_prev = 1`, in the interior `k_prev = χ`, so a boundary
bond is `D²` and an interior one is `χ·D²` — different for any χ > 1. Only plaquettes at least two
steps from every edge have all four equal: 1/9 on 4×4, 1/25 on 6×6, 9/49 on 8×8. Zero-padding to a
common size is mathematically exact (dense `pschur!` reproduces the product's eigenvalues to 2.96e-15
with the padded directions exactly 0) but makes every factor **singular**, which is what breaks the
Arnoldi recursion — `SingularException`, `convergence failed at level 20`, `PKSFailure`.

**The architectural difference this exposes.** Their `CTMState.init` allocates *every* corner and
edge at `(chi, chi)` regardless of position, one-hot at the boundary, and carries a separate `rank`
field for the effective dimension. So their cycle factors are square **by construction** and the rank
machinery — `active_cols`, rank-masked QR, `_stochastic_expand_range`, the Schur gauge's exact
structural zeros — exists to track what is actually live inside those fixed-size arrays. We use exact
ITensor indices that shrink at boundaries, which is cleaner for everything else in this engine and is
precisely what makes the cycle factors rectangular.

**So the honest cost, third revision.** `PeriodicSchurDecompositions.jl` gives us the eigensolver for
free — that part of my earlier estimate stands, and both `pschur!` and the Krylov `partial_pschur`
work. What it does not give is the rank/rectangularity handling, and that is a real slice of their
2200 lines. Adopting the cycle projector means adopting fixed-χ padded storage with explicit rank
tracking, which is an architectural change to our environment representation, not a swap of one
function for another.

#### RESOLVED: matrix-free Krylov on the rectangular chain — no padding needed

The padding was never necessary. It was an artefact of trying to hand the whole chain to `pschur!`,
which demands square equal-size factors. **We never need that.** What the projector needs is

1. the dominant `k`-dim invariant subspace of `M = A₄A₃A₂A₁` at one bond, and
2. the bases at the other three,

and (2) is just propagation, `V_{l+1} = orth(A_l V_l)`, which is rectangular-safe already. For (1)
only the *action* of `M` on a vector is required — four matvecs through rectangular matrices, well
defined and never forming the product. So a matrix-free Krylov solve does it:

```julia
fwd(v) = As[4] * (As[3] * (As[2] * (As[1] * v)))          # W -> W, product never formed
schursolve(fwd, v0, k, :LM, Arnoldi(; krylovdim = max(4k, 20)))
```

`schursolve` also returns a real orthonormal basis, which sidesteps the conjugate-pair trap that
`ordschur!` fails silently on. Measured on real corners:

| | plaquettes passing | residual range | isometry |
|---|---|---|---|
| square 4×4 real D=3, χ=8 | **9/9** | 1.3e-15 – 2.9e-14 | ~4e-15 |
| hex 4×4 complex D=2, χ=8 | **32/32** | 4.6e-16 – 4.4e-14 | ~2e-15 |

Bonds like `[9,72,72,9]` and `[4,16,32,16]` are handled natively. **No padding, no
`PeriodicSchurDecompositions` dependency, no fixed-χ storage, no architectural change** — and
KrylovKit is already a dependency. The conditioning objection does not apply either: it was about
forming the product to resolve *small* eigenvalues, and we only ever want the top χ.

The lesson is that their fixed-χ padded storage with rank tracking is a consequence of needing
square factors for a *dense* periodic Schur. Our adaptive bond dimensions are a feature, and the
matrix-free formulation embraces them rather than fighting them.

**Two things still open before this is a projector rather than a subspace.**

* Only the RIGHT bases are validated above. The left/biorthogonal side, and the per-bond whitening
  that makes `P_B P_A = I`, are not yet checked.
* `k = min(χ, narrowest bond)` is a real design decision. On hex `[4,16,32,16]` the cycle rank is
  bottlenecked at 4, so a bond of dimension 32 is truncated to 4 where the pairwise projector keeps
  8. Physically the loop *is* bottlenecked, but these projectors are also consumed by region
  contractions that do not go round the loop. This is what per-bond rank tracking manages in their
  code, and it should be measured, not assumed.

#### VERDICT: the cycle projector is exact on square, INVALID on hex, and worse where valid

Built it end to end — matrix-free Krylov, both bases, per-bond biorthogonalisation, wired into the
sweep on the existing keys with fallthrough to the cut. The machinery is right; the criterion loses.

*Left bases and sides, for the record.* The left basis propagates DOWNWARD, `V_L[l] ∝ V_L[l+1] A_l`,
seeded from `schursolve` on the transposed action `A_1ᵀA_2ᵀA_3ᵀA_4ᵀ`. Per bond, the factor attached
to the tensor CONSUMING it is the right basis and the one attached to the producer is the left, so
against our west/north = `P_A` convention W and S take `P_A = V_L` while E and N take `P_A = V_R`.
Verified by the insertion identity `Bp (P_A P_B) Bc = Bp Bc` at 1.1e-14.

| case | χ | cut `marg` | cycle `marg` | cut `⟨Z⟩` err | cycle `⟨Z⟩` err |
|---|---|---|---|---|---|
| square 4×4 D=3 | 4 | **1.8e-3** | 2.5e-1 | 2.3e-2 | 1.6e-2 |
| | 8 | **3.0e-4** | 1.1e-1 | **1.1e-3** | 2.3e-1 |
| hex 4×4 cplx D=2 | 4 | 1.1e-8 | **2.6e-10** | 2.5e-5 | **7.2e-7** |
| | 8 | 9.9e-10 | **1.3e-11** | 3.6e-7 | **1.6e-9** |
| **hex, correctness gate** | 16 | **2.8e-16** | **6.6e-04** | | |
| | 40 | **3.5e-16** | **1.2e-03** (worse) | | |

**The hex win is illusory and the gate is what caught it.** At χ = 4–8 the cycle looks 79–222×
better; at lossless χ it saturates near 1e-3 and DEGRADES with χ — the systematic-error signature,
exactly as in the complex-projector bug. Square is exact at lossless χ (3.55e-15, matching the cut)
but 100–400× worse at finite χ.

**Root cause: the rank rule.** `k = min(χ, narrowest bond)` is lossless *for the loop* — a corner's
rank is bounded by its narrower leg — and the 3×3 exactness test passed on that basis. But hex has
plaquettes with a bond of dimension **1** (a missing lattice link), so all four bonds there are cut
to a single direction forever, whatever χ is. The flaw is the one flagged and then talked out of:
these projectors are also consumed by region contractions that do NOT go round the loop, and the
loop's rank is not a bound for those.

**What is reusable.** The matrix-free cycle solver itself is correct and cheap: `schursolve` on the
four-matvec action handles rectangular bonds natively at 1e-15 on 9/9 square and 32/32 hex
plaquettes, needs no padding, no new dependency and no architectural change. If the criterion is
ever revisited, the fix to try is a PER-BOND rank rule (each bond keeps `min(χ, its own dim)` with
the cycle used only to choose the subspace, not the dimension) — but note the square result says the
criterion is worse even where it is perfectly valid, so the rank rule is not the only problem.

Code reverted. Fourth port attempt from `joey_ctmrg_bp`, fourth rejection.
### HEAD-TO-HEAD on the collaborator's own 5×5 Ising PEPS

Their code DOES run here (an earlier note in this document said it could not — that was wrong). The
compiled SLICOT extension is only reached by the periodic-Schur paths; the demo's default
`"eig one sided"` method never touches it, so `pip install jax einops` in a venv is enough. Their
demo converges in 3 sweeps to `max dV = 8.9e-16`.

Loading their `isingZZX_5x5_D3_g3.04438.npz` into our engine as a fused double layer, exact
reference `ln Z = -6.217866847854575` (numpy column sweep and ITensors exact contraction agree):

Errors in `ln Z`:

| χ | Julia finite CTMRG | Python finite CTMRG | Julia boundary MPS with BP estimator |
|---|---|---|---|
| 4 | **2.12e-04** | 2.42e-04 | 4.00e-04 |
| 9 | **9.29e-09** | 1.75e-08 | 4.85e-06 |
| 16 | **6.71e-12** | 1.55e-10 | 8.45e-09 |
| 32 | **1.07e-14** | 2.40e-14 | 3.08e-12 |

**Both CTMRG implementations converge cleanly to machine precision.** Julia is consistently ahead —
1.1× at χ=4, 1.9× at χ=9, 23× at χ=16 — and both beat the boundary-MPS/BP estimator by orders of
magnitude from χ=9 on. There is no qualitative difference between the two schemes on this benchmark,
which is a reassuring result for both and is consistent with the four ported ideas all being
rejected: neither implementation has an accuracy advantage worth importing.

#### RETRACTED: the cycle-augmented projector is NOT robust — code removed

Implemented the union design, it passed three gates, and then a broader scan destroyed it. Recording
the whole arc because the failure mode is the one this document keeps re-learning.

**What passed.** Exactness at lossless χ (square 3×3 `3.55e-15`, hex 3×3 `1.85e-15`); stationarity
(hex 3×3 χ=40 `marg` **2.42e-16** against the cut's `6.80e-11`); and `⟨X⟩` on the collaborator's 5×5
Ising PEPS reproducing the Python engine **to four significant figures at χ=9** (5.132e-08 both), with
4.169e-05 vs their 5.218e-05 at χ=4 and 9.202e-10 vs 9.279e-10 at χ=16. On adaptive bonds, no
padding. All of that is real and repeatable.

**What the broader scan showed.** Ratio = cut error / cycle error, so **below 1 means the cycle route
is WORSE**:

| lattice | χ=2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| square 4×4 real D=2 | 0.14 | 0.64 | 8.89 | 2.12 | 0.16 |
| square 4×4 real D=3 | — | 0.64 | 0.34 | 0.74 | 0.54 |
| hex 4×4 complex D=2 | 1.13 | **74.65** | 0.00 | 0.00 | 0.00 |

Worse at EVERY χ on square D=3. On hex 4×4 it is catastrophic from χ=8, with `marg` stuck at 1.1e-2,
7.4e-3, 7.1e-3 — **not converging at all**, the same saturation signature as the pre-union prototype.
So the union rule fixed hex 3×3 and not hex 4×4, and the 5×5 Ising match plus the 74× win at hex χ=4
were lucky configurations.

**Why the gates missed it.** All three were single-configuration: one square lattice, one hex lattice,
at ONE χ each. The failure needs a 4×4 hex at χ≥8 to show up. A gate suite that samples one point per
axis cannot distinguish "correct" from "correct here".

**Two mechanisms understood, one not.**

* *Fixed:* the merge must use OBLIQUE DEFLATION against the cycle pair, not an independent QR on each
  side. Independent QR destroys the pairing between column `j` of `A` and row `j` of `B`, so the
  overlap stops being near-identity and whitening mixes directions. Deflation gives `B_cyc Ad = 0`
  and `Bd A_cyc = 0` identically, so the merged overlap is block diagonal with the cycle block exactly
  `I`. This cleaned up `marg` (χ=32 to 9.4e-17) but did NOT change the observable — so it was a real
  defect and not the dominant one.
* *Fixed:* `_ctm_biorth` must TRUNCATE the overlap singular values, not floor them at `eps`. Moved
  hex χ=4 `marg` from 3.4e-3 to 8.9e-7.
* *NOT understood:* why forcing `k_cyc` cycle directions in costs so much once many cut directions are
  appended. The damage tracks the ratio: pure cycle at `k_cyc = k_b` (χ=4, 9) is excellent, 9+7 at
  χ=16 is fine, 9+23 at χ=32 is 20× worse than the plain cut. And separately, why hex 4×4 stops
  converging at χ≥8 while hex 3×3 is exact.

**Verdict: code removed.** Leaving it default-off was tempting, but the accompanying tests asserted
only the passing configurations and would have given the next person false confidence in a route that
is worse in most of the scan. The design reasoning above and the two mechanism fixes are worth keeping;
the realization is not.

**If revisited:** the gate suite must scan χ AND lattice size AND bond dimension before any positive
claim, and hex 4×4 at χ ≥ 8 belongs in it as a specific regression case. The unexplained
`k_cyc / k_b` dependence is the thing to understand first — it is probably the whole story.

#### OBSERVABLES: the stationary projector wins, decisively

`Z` is not the whole story. Single-site `⟨Z⟩` at the centre site of the same 5×5, exact
`4.2177326741e-05` (numpy and ITensors agree to 4.4e-16 — an independent cross-check on the
transfer, after the two bugs below):

| χ | Julia (cut / Corboz SVD) | Python (stationary / eig) | ratio |
|---|---|---|---|
| 4 | 1.62e-04 *(wrong sign)* | **1.70e-05** | 9.5× |
| 9 | 1.90e-07 | **5.72e-09** | 33× |
| 16 | 5.30e-09 | **3.91e-10** | 14× |
| 32 | 3.20e-11 | 2.46e-13 | 130× |   ⚠️ SUSPECT: the ⟨X⟩ twin of this row was wrong by 16×

**Set against the `ln Z` table, this is a clean split** and it confirms Zaletel's hypothesis:

* `ln Z` — a wash, slightly favouring the CUT projector (2–23× to Julia).
* 1-point functions — strongly favouring the STATIONARY projector (10–130× to Python).

The mechanism is the one the stationarity discussion predicts. `∂F/∂B = 0` is marginal consistency:
the marginal read from one region agrees with the marginal read from an overlapping one. Their
projector has it by construction; ours does not, and our own `marginal_inconsistency` has been
reporting that all along. `Z` hides the defect because the Möbius sum cancels it (~4000×); a
single-region ratio cannot.

**This is the first thing from `joey_ctmrg_bp` that is worth taking, and it is not a detail — it is
the projector criterion.** It also reframes the four earlier rejections: those were all judged on
`marginal_inconsistency` and `Z`, and the cycle projector's own rank rule was broken. The criterion
itself, implemented properly as they do, buys 1–2 orders on observables.

Repeated with `⟨X⟩`, which is O(1) here rather than nearly zero (exact `0.916900598128483`, numpy
and ITensors agreeing to 1.2e-14) — so relative and absolute error coincide and the near-zero
objection does not apply:

| χ | Julia (cut / Corboz SVD) | Python (stationary / eig) | ratio |
|---|---|---|---|
| 4 | 1.515e-04 | **5.218e-05** | 2.9× |
| 9 | 4.240e-07 | **5.132e-08** | 8.3× |
| 16 | 6.255e-09 | **9.279e-10** | 6.7× |
| 32 | 7.403e-12 | 8.149e-14 | 91× |   ⚠️ WRONG: re-measured, theirs is 1.33e-12, so 5.6× to **us**

Same direction on both operators, at every χ. The margin is smaller on `⟨X⟩` at small and moderate χ
(2.9–8.3× rather than 9.5–33×) but widest of all at χ=32, where ours plateaus at 7.4e-12 while theirs
reaches 8.1e-14 — and our `marginal_inconsistency` is 4.4e-17 there, so that is not a convergence
failure but the projector extracting less from the same χ.

Remaining caveats: one state, one site. Worth repeating across sites and on a second state before it
is load-bearing, though two operators agreeing at four χ each makes a reversal unlikely.

**TWO data-transfer bugs, both of which produced confident wrong conclusions.** Recording them
because each was caught only by an independent computation, never by internal consistency.

1. *Byte order.* `ndarray.tofile()` always writes C order regardless of the array's memory layout, so
   `np.asfortranarray(B).tofile(...)` handed Julia C-ordered bytes that were read column-major,
   transposing every tensor. This produced `ln Z = -6.9713` against the true `-6.2179` — and our
   exact contraction, our CVM and our boundary MPS ALL AGREED on it, because all three were
   correctly contracting the same wrong network.
2. *Silent downcast.* JAX downcasts float64 to float32 on unpickling unless x64 is enabled. Their
   `configure_jax()` enables it; my dump scripts did not call it, so I exported float32-truncated
   tensors (`0.02549903` against the true `0.02549904`). The resulting reference was wrong at ~4e-8,
   and I published a claim that the PYTHON code "saturates at 7.5e-8". **Retracted** — that floor
   was mine. With the true float64 data their χ=32 error is 2.4e-14.

**Agreement among our own methods is no check at all on the input.** Both bugs were caught by
recomputing on the far side of the transfer — the first by a numpy contraction, the second by
comparing dtypes across the x64 boundary.

**A methodological warning from this exercise.** The first head-to-head said we were RIGHT and they
were wrong by a factor of 2.1. That was a bug in my data transfer: `ndarray.tofile()` always writes
C order regardless of the array's memory layout, so `np.asfortranarray(B).tofile(...)` handed Julia
C-ordered bytes that were then read column-major, transposing every tensor. Our exact contraction,
our CVM and our boundary MPS all agreed with each other on the wrong answer — because they were all
correctly contracting the same wrong network. **Agreement among our own methods is no check at all
on the input.** What caught it was an independent computation in the OTHER language, on the other
side of the transfer.

### EVALUATED, NOTHING TO PORT: `max dV` — we already have it

The collaborator converges on `max dV`, the largest change in the projector BASES between sweeps.
We converge on `_ctm_statedist`, the largest relative change in the C/T BLOCKS. Traced side by side:

| sweep | square 4×4 D=3 χ=8: `\|ΔF\|` / statedist / projdist | hex 4×4 cplx χ=8 |
|---|---|---|
| 4 | 6.0e-05 / 4.8e-01 / 4.4e-01 | 2.8e-07 / 8.99e-01 / 8.5e-01 |
| 6 | 4.7e-05 / 7.1e-03 / 2.3e-02 | 1.2e-10 / 2.4e-02 / 6.1e-02 |
| 10 | 2.7e-08 / 4.7e-05 / 1.0e-04 | 5.0e-13 / 2.2e-05 / 5.0e-05 |

They agree within a factor of 2–3 at every sweep, become available at the same sweep, and both catch
what `|ΔF|` misses — at sweep 4 of the square case `|ΔF|` reads 6e-5 while the true error is *rising*
and both state signals read 0.44. Projector distance is consistently ~2–3× more conservative, so it
would stop marginally later; there is no failure mode where the block distance misses something.

Conclusion: our criterion IS theirs, measured on the outputs rather than the variables. A second
redundant signal is complexity for no gain, so nothing was added.

Also visible in the trace, and worth remembering: on the square case the true `|F − ln Z|` is
2.1e-3 at sweep 2 and 2.9e-3 at convergence — the converged answer is WORSE than an intermediate
one. Cancellation again, and one more reason never to tune against `F`.

### TESTED AND REJECTED: Gauss-Seidel sweeping — it breaks the projector's optimality

Ported from the collaborator's JAX code (`joey_ctmrg_bp`), which sweeps plaquette-by-plaquette with
`lax.scan` so each local update sees the previous one's output, where our sweep derives every
projector from `S` and rebuilds everything at once (Jacobi).

A full per-plaquette Gauss-Seidel is not available in our geometry — the four corner families grow
from four different directions, so no single ordering leaves all of them fresh. What the geometry
does admit is **two stages**: derive the `PH` families from the raw enlarged corners, apply them, then
derive the `PV` families from the already-horizontally-truncated corners. That is what directional
CTMRG does, and it is cheaper too (the QR sees a "rest" leg of χ instead of χ·D_layer).

**It is worse on both axes.** Exact at lossless χ (3.55e-15, `marg` 3e-16) so the machinery is right;
the deficit is entirely the truncation criterion.

| case | χ | jacobi `marg` | GS `marg` | jacobi `⟨Z⟩` err | GS `⟨Z⟩` err |
|---|---|---|---|---|---|
| square 4×4 real D=3 | 4 | 1.8e-3 | **1.5e-1** | 2.3e-2 | **7.07e+00** |
| | 6 | 7.3e-4 | 6.3e-2 | 1.4e-3 | 3.3e-1 |
| | 8 | 3.0e-4 | 9.8e-2 | 1.1e-3 | 1.4e-2 |
| | 12 | 8.7e-5 | 5.6e-2 | 5.0e-3 | 4.7e-3 |

Jacobi's `marg` falls monotonically; **Gauss-Seidel's plateaus at 5–15e-2 and does not improve with
χ** — a degraded fixed point, not a slower one. It also needs MORE sweeps, not fewer (31 vs 13 on hex
4×4 complex D=2, and it failed to converge in 100 at χ=8). Hex showed an apparent 35× observable win
at χ=4 that reverses at χ=8 and 16; it was noise.

**Why, and this is the transferable part.** This document already records that the projector
truncation is *provably optimal for the bipartition it is given*. Deriving the vertical projector
from a horizontally-truncated corner hands it the WRONG bipartition — it optimises retention of
`A_truncated Bᵀ` rather than `A Bᵀ`, and cannot see what the horizontal projection already discarded.
Concretely the block matrix stops being square: "rest" is χ while the interface is still χ·D_layer,
so `R_A` has rank ≤ χ, `W = R_A R_Bᵀ` is rank-deficient, and its singular vectors do not carry enough
to choose well.

**So the collaborator's Gauss-Seidel is inseparable from their cycle projector.** Theirs is the
dominant invariant subspace of the four-corner cycle `C0 C1 C2 C3`, an eigenproblem that is well
posed against the current state whatever its truncation status. Ours is a two-block cut whose
optimality argument *requires both blocks untruncated*. The sweep ordering and the projector are
coupled: porting the ordering alone does not merely fail to help, it removes the property that makes
our projector good. If Gauss-Seidel is wanted, the cycle projector has to come with it.

### REMOVED: the `ρ`-route projector (`qr = false`) — one projector now, not two

The density-matrix route (`ρ_L = A†A`, `ρ_R = B†B`, eigendecomposing `ρ_R` back into a square-root
factor, selected by `qr = false`) is gone, along with `_ctm_psd_factor` and the `pinv_cutoff` option.

It was sesquilinear by construction — it needs `ρ = A†A` Hermitian PSD to have a square root at all —
while the sweep contracts the corners bilinearly. So it was wrong for complex tensors and could not
be repaired in place: `Aᵀ A` is complex *symmetric* and has no PSD root, so making it bilinear meant
replacing its machinery outright. It had been guarded to error on complex input; now it simply does
not exist.

Its remaining value had been as an independent cross-check for real tensors, and **that job is now
done better** by the full-rank identity assertion (`A (P_A P_B) Bᵀ = A Bᵀ`), which tests the pair
against what it must satisfy rather than against a second implementation that could be wrong the same
way. Note the history: the two routes *did* agree with each other for years, on complex networks,
while both were wrong. Agreement between implementations is a weaker signal than an invariant.

`opts.qr` and `opts.pinv_cutoff` are gone with it, so `CTMOptions` is down to six fields and
`qr_cutoff` is the only cutoff. Passing `qr = false` (or any removed field) is now a `MethodError`,
which is asserted. Default-path numbers are unchanged except for a 1.2e-15 relative shift on one hex
case, from floating-point reassociation after the struct layout changed.

### Complex-path audit — CLEAN, and now covered by tests

The projector bug was found by testing an invariant rather than by reading code, so every other
operation that is "a no-op for real tensors" got the same treatment. All four passed first run.

| path | invariant tested | result |
|---|---|---|
| greedy one-sided projector (`ρ = Bc·prime(dag(Bc))`, applied as `P` / `dag(P)`) | exact at lossless χ | complex 0.0e+00, real 1.3e-15 |
| gauge fixing (`svd(a' * ao)`, a sesquilinear Procrustes) | `F` invariant to the gauge | 7.1e-15 / 3.6e-15, same as real |
| `expect` / `rdm` | exact at lossless χ; Hermitian; positive | 2.8e-16; 1.7e-16; min eigval 0.38 |
| hex / heavy-hex `nothing` paths | `F` and `⟨Z⟩` vs exact | 0.0e+00 and 1e-16 |
| `marginal_inconsistency` | monotone in χ, → 0 when lossless | 4.0e-3 → 9.6e-5 → 1.8e-6 → 2.1e-16 |

Two notes. The greedy projector's sesquilinear `ρ` is **suboptimal but not wrong** for complex — `P`
is unitary at full rank so `P dag(P) = I` exactly against the bilinear pairing, and at finite χ it is
only ever a seed that `update` sweeps away. And the RDM picks up ~1e-12 non-hermiticity at
intermediate χ because the truncation does not respect the ket↔bra structure; that is the genuine
symmetry defect, nine orders below the truncation error, and the reason the section below concludes
what it does.

### REMOVED: ket↔bra symmetry in the projector — TESTED, no gain on speed OR accuracy

The double layer has an exact symmetry the projector ignores. With σ the involution swapping every
ket leg with its bra partner, any contracted region satisfies (verified to 1e-16, real and complex):

```
B[σ(r), σ(i)] = conj(B[r, i])
```

Real tensors: a linear Z₂ symmetry, so the interface splits into σ = ±1 sectors. Complex tensors:
antilinear, `M = A Bᵀ` commutes with `K = σ∘conj` with `K² = +1` — no block split, but an explicit
basis in which every block is **real**: take the σ = +1 vectors as they are and multiply the σ = −1
vectors by `i`. Verified: an enlarged corner with all legs rotated comes out real to 3.1e-17.

Two traps found while deriving it, worth keeping:

* **The rotation is not the same on both sides.** `U` is unitary but not orthogonal — the `i` makes
  `U Uᵀ ≠ I` — and the pairing is BILINEAR, so inserting the same `U` on both blocks silently changes
  the contraction. It must be `U` on the west/north block and `conj(U)` on the east/south one, giving
  `Σ_α (Bw U)(Be conj U) = Bw (U U†) Be = Bw Be`. Both realify, since `U[σ(a),α] = conj(U[a,α])`.
* **The invariant must hold through the projector, not locally.** A block's array is real only if
  EVERY leg is K-adapted. An enlarged corner carrying a non-K-real kept index sits at ~0.4 imaginary.
  `P_A = U · P_A_real` keeps the kept index K-real, so the induction closes with no grading
  bookkeeping — `K = σ ⊗ conj` factorises over `w ⊗ (ℓ_ket, ℓ_bra)`.

**Implemented behind a `symmetric` option, measured, and removed.**

*Speed* — complex 4×4, best-of-3: D=2 χ=8 0.73×, D=3 χ=8 **0.49×**, D=3 χ=16 **0.40×**. The real
LAPACK is genuinely happening; it just cannot pay for four rotation contractions, two
`norm(imag(·))` passes and two `real(·)` allocations per interface per sweep. The only way to make
the rotation free is to keep the environment permanently in the K basis, which means rotating a raw
pair when it is absorbed — i.e. **fusing** `(ℓ_ket, ℓ_bra)` into one D² leg. So the symmetry lives on
the fused pair and laziness exists to keep it unfused: **the two goals are in direct tension**, and
this engine has chosen laziness.

*Accuracy* — the real reason it is not worth it. Measured on 4×4 D=3, single-site RDM at (2,2):

| | min eigval | ‖ρ−ρ†‖/‖ρ‖ | ‖ρ−ρ_exact‖ |
|---|---|---|---|
| Float64, χ=2…8 | 0.47 … 0.49 | 1e-16 | 3.5e-2 … 9.4e-3 |
| ComplexF64, χ=2…8 | 0.40 … 0.40 | 2.7e-7 … 1.2e-11 | 3.0e-2 … 1.2e-2 |

Positivity is **not** violated, so there is no PSD-preservation target. Hermiticity **is** violated
for complex — that is the genuine symmetry defect — but at 1e-11 against a truncation error of
1e-2, i.e. **nine orders below what dominates**. Enforcing the symmetry cannot move a measurable
number, and indeed `F` was identical to the default route at χ=8. `(ρ + ρ†)/2` in `rdm` would
capture the entire measurable benefit in one line.

So "χ is the binding constraint, not arithmetic" extends to STRUCTURE, not just precision.

Also settled while measuring: **the observable error is not monotone in χ, and that is not a bug.**
4×4 D=3, `marginal_inconsistency` is strictly monotone (6.6e-3, 1.8e-3, 7.3e-4, 3.0e-4, 8.7e-5,
5.2e-5, 1.9e-5 at χ = 2…24) while `|⟨Z⟩ − exact|` bounces (1.4e-3 at χ=6, 5.0e-3 at χ=12) — and
boundary MPS bounces identically. The observable error is a signed quantity passing through zero, so
its magnitude need not fall monotonically even as the environment improves uniformly. Judge
environment quality with `marginal_inconsistency`, never with a single observable's error.

The remaining untried variant is the REAL-tensor ± block split: no rotation, so no per-call overhead
to lose, worth ~2.3× (D=2) to 4× (D→∞) on the factorizations alone (~1.2–1.35× end-to-end). It needs
the ± grading recursion and is irrelevant to complex networks.

### FIXED: `update` could "converge" after ONE sweep, returning the greedy environment

Found from `examples/ctm_test.jl`: on a complex hex 4×4 D=2 state the single-site `⟨Z⟩` was at
machine precision at χ = 16 and 32, jumped to **7.0e-4 at χ = 64**, and was fine again at χ = 128.
Non-monotone in χ, with the norm exact to 1.3e-15 throughout.

**Cause, and it is not about χ at all.** `_ctm_statedist` returns `nothing` on the first sweep (the
interface bases are still bootstrapping), so `crit = max(Δ, sd²)` degenerates to `Δ` alone. `F` is a
signed Möbius sum whose cancellation is worth ~4000×, so it can already sit at its final value while
the state is still the one-sided **greedy seed**. At χ=64 sweep 1 happened to report
`|ΔF| = 2.2e-16`, the loop exited immediately, and the returned cache was the greedy environment —
3–4 orders worse and non-monotone, exactly as this document says of it. χ=64 was not special; `Δ`
just got unlucky. That is the point: **a single `Δ` carries no information about the state.**

`marginal_inconsistency` confirmed it independently — 2.9e-6 at χ=64 against 8.7e-10 at χ=32 and
χ=128, i.e. a genuinely worse fixed point, not a bad readout.

**Fix.** Require positive evidence the state stopped moving: at least two sweeps, plus a real
`_ctm_statedist` whenever the gauge makes one available. With `gauge = false` there is no state
distance to be had and `Δ` remains the only signal, unchanged. After the fix the χ sweep is monotone
and `⟨Z⟩` is ≤ 2.8e-16 at every lossless χ, with `marginal_inconsistency` a flat 8.747e-10.

**Two lessons worth keeping.** First, `|ΔF|` is structurally unfit as a sole convergence signal here
— the cancellation that makes `F` accurate is exactly what makes it blind to the state. Second, the
norm cannot detect this class of bug at all; a **single-region observable** can, because it is a
ratio over one region with no cancellation available. Diagnose with `⟨Z⟩` and
`marginal_inconsistency`, never with `|F − ln Z|` — which this document already says, for a
different reason.

Several tests were quietly relying on the old shortcut: they passed their `F` assertions with
`maxiter` of 2, 3 and 6 because `F` converges as `sd²` and lands long before the state. Their
budgets are now 30, and the "sweeping again barely moves `F`" check calls
`sweep_vertex_environments` directly rather than `update(...; maxiter = 2)`, which cannot certify by
construction.

### FIXED: the projector was sesquilinear, the network is bilinear (complex tensors were wrong)

**Complex networks gave silently wrong answers, at every χ, until this was fixed.** Symptom on a
4×4 complex double layer: `cvm_freenergy` sat 3.663e-3 from the exact norm at χ=16 **and χ=64**,
while `inner(ψ, ψ; alg="boundarymps")` was exact to 0.000e+00 at every χ. The saturation in χ is
the tell — a truncation error shrinks, a wrong subspace does not.

**Cause.** The sweep contracts the two enlarged corners *plainly*: `Bw * Be` conjugates nothing.
But both projector routes were derived from conjugated objects — `_ctm_block_matrix` returned
`conj(...)` (commented "a no-op for real tensors", which is exactly why it hid), and the ρ route
built `ρL = Bwc * prime(dag(Bwc), io)`. So the pair was optimised to preserve the sesquilinear
`Bw Be†` while the engine applied it to the bilinear `Bw Be`. Identical for real tensors, 11% wrong
for complex — measured directly against the pair's own full-rank identity:

| eltype | route | worst ‖Bw·P_A·P_B·Be − Bw·Be‖/‖Bw·Be‖ at full rank |
|---|---|---|
| Float64 | QR | 1.3e-15 |
| Float64 | ρ | 3.3e-13 |
| ComplexF64 | QR | **0.111** → **1.8e-15** after the fix |
| ComplexF64 | ρ | **0.111** → now refuses (see below) |

**The corrected QR derivation.** With `A`, `B` the blocks as (rest × interface) matrices, so that
the tensor contraction is `A Bᵀ`:

```
A = Q_A R_A,  B = Q_B R_B          (thin QR, NO conjugation)
A Bᵀ = Q_A (R_A R_Bᵀ) Q_Bᵀ    ⇒    W = R_A R_Bᵀ          -- transpose, not adjoint
W = U S V†                    ⇒    P_A = R_Bᵀ V S^(-1/2),  P_B = S^(-1/2) U† R_A
```

`R_A P_A P_B R_Bᵀ = U S^(1/2) · S^(1/2) V† = W`, so `A (P_A P_B) Bᵀ = A Bᵀ` exactly at full rank.
`Q_A† Q_A = I` and `Q_Bᵀ (Q_Bᵀ)† = I`, so `W`'s singular values are those of `A Bᵀ` and the
truncation is optimal for the product the network actually forms.

**The ρ route cannot be fixed this way and now errors on complex input.** It is sesquilinear by
construction — it needs `ρ = A†A` Hermitian PSD to have a square-root factor at all — and `Aᵀ A` is
complex *symmetric* with no PSD root. Making it bilinear means replacing the machinery outright, so
it refuses rather than returning a plausible wrong number. `qr = true` (the default) is exact for
both element types.

**Second, independent bug found alongside it.** `region_lnZ` used `log(abs(real(Z_R)))`. For a
genuinely complex single-layer `Z` the Möbius sum then telescoped to `log|Re Z|` rather than
`log|Z|` — verified: the error equalled `log|Z| − log|Re Z|` exactly. Now `log(abs(Z_R))`, which
gives `Σ c_R log|Z_R| = Re(Σ c_R log Z_R) = log|Z|`, confirmed to 1.8e-15. A no-op for real tensors
and for the double-layer norm (both real positive).

**Why it went unnoticed:** 2×2 and 3×3 were exact for complex; only 4×4 and larger failed. Any
test small enough to check by hand passed. The regression test therefore uses 4×4, asserts the
value at two lossless χ (to catch the saturation signature), and checks the pair's full-rank
identity directly rather than only end-to-end.

**Real tensors are bit-identical across both fixes** — verified over `F` at three χ single-layer,
`F`/`⟨Z⟩`/`marginal_inconsistency` at two χ double-layer, the `w=1` window estimator and a hex grid.

### The knobs are per-cache (`CTMOptions`), not global `Ref`s

All eight tuning flags used to be module-level `const … = Ref(…)`. They are now fields of a
`CTMOptions` struct stored ON the cache and passed as keywords to the constructor:

```julia
cache = update(CTMEnvironmentCache(tn, 8; qr = false, degtol = 1e-9))
```

Renamed `CTM_QR` → `qr`, `CTM_GAUGE` → `gauge`, `CTM_ARNOLDI` → `arnoldi`, `CTM_DEGTOL` →
`degtol`, `CTM_PINV_CUTOFF` → `pinv_cutoff`, `CTM_QR_CUTOFF` → `qr_cutoff`, `CTM_KRYLOV_MIN` →
`krylov_min`, `CTM_OPTIMAL_MAX` → `optimal_max`. Defaults are unchanged, so every measurement in
this document still describes the default path. Each flag's rationale stays in the comment next to
the code it governs; `CTMOptions`' docstring is a one-line index into those.

Three reasons this mattered:

* **A run was not reproducible from its call site.** `cvm_freenergy(cache)` meant something
  different depending on globals set arbitrarily far away, including by a previous test.
* **The test that exercised the ρ route flipped `CTM_QR[]` with no `try`/`finally`.** Any throw
  inside it — the `update` call, say — would have left the non-default route on for every later
  testset in the file, silently. It now builds two caches instead, and there is nothing to restore.
* **It foreclosed threading**, on top of the `CTM_SEQ_CACHE` issue.

`CTM_SEQ_CACHE` deliberately stays a global: it is a pure memo keyed on tensor *shape*, so entries
are valid for any network of the same geometry, and sharing it across caches is the point. The
gate's verdict (`length(ts) ≤ optimal_max`) is now part of its key, so two caches with different
`optimal_max` cannot trade sequences and each end up with whichever optimiser happened to run
first. It remains not thread-safe.

### The greedy fallback warns

An un-updated cache still evaluates — `cvm_freenergy`, `region_lnZ`, `vertex_window` and hence
`expect`/`rdm` all fall back to the greedy single pass — but it now **warns**, and the same number
comes back, so nothing breaks.

This was modelled on the `BeliefPropagationCache` convention, and that was the wrong analogy. An
un-updated BP cache gives an unconverged answer from the *same* algorithm; this gives an answer
from a *different* one, whose error is 3–4 orders larger and **non-monotone in χ** (a flat ~2.5e-3
floor at every χ on the PEPS norm). So a forgotten `update` does not present as "not converged
yet" — it presents as a plausible number that refuses to improve when you raise χ, which costs far
more to diagnose than an early warning costs to read. The fallback also rebuilds the entire
environment set per call, so a loop over regions pays a full greedy build each time.

`update` itself seeds from the greedy pass, so the warning lives on the *read-out* paths only
(`_ctm_env_checked`), not in `_ctm_env` — otherwise every normal `update` would warn. Asking for
greedy on purpose is silent: `cvm_freenergy(vertex_environments(cache), cache)`, which is what the
beats-greedy test and the `cvm_vs_boundarymps` example use. No `maxlog`: each occurrence is a
separate wrong number over a separate full rebuild.

**Next step, now well-posed:** Anderson acceleration of the sweep. The gauge makes iterates
linearly combinable, and the Picard rate is ≈0.35/sweep over 8–12 sweeps, so there is real room.

### Toward a stationary projector — measured findings and the derivation

The current projector maximises the fidelity of **one** local interface contraction (the two
bounding corners closing directly, i.e. the plaquette-like pairing). But each interface actually
appears in **six** regions. For `PH[:N,x,y]`:

| region | weight |
|---|---|
| plaquette `(x+½, y−½)` | +1 |
| h-edge `(x+½, y)` | −1 |
| vertex `(x, y)` / vertex `(x+1, y)` | +1 / +1 |
| v-edge `(x, y−½)` / v-edge `(x+1, y−½)` | −1 / −1 |

The weights sum to zero (this is the scale-cancellation above). Each `Z_R` is *linear* in
`Π = P_A P_B`, so `Z_R = Tr[E_R Π]` and

```
∂F/∂Π  =  Σ_R c_R E_R / Z_R  ≡  G
```

Stationarity over rank-`k` `Π`: writing `Π = Σ_j a_j b_j†` biorthogonally, `δF = Tr[G δΠ]`
vanishes for all variations mixing kept with discarded directions iff **the kept subspace is an
invariant subspace of `G`**. That is the target condition — note `G` is *signed and
non-symmetric*, so it is a partial Schur/invariant-subspace problem, not a top-eigenvector one.

**Finding 1 — `|F − ln Z|` is an INVALID objective. Do not tune against it.** Overriding which
SVD directions one interface keeps, the best single swap improved `|F − ln Z|` by **18×** — and
made single-site observables *worse* (`⟨Z⟩` at `(3,2)`: 4.6e-3 → 1.1e-2). A different interface
gave a 2.9× gain that *did* improve 2 of 3 observables. So some apparent gains are cancellation
artifacts of the signed Möbius sum, and optimising `F` against the exact answer chases them. Any
criterion must be `ln Z`-free; stationarity is.

**Finding 2 — the converged point is far from stationary (`ln Z`-free measurement).** Swapping
the marginal kept direction (`k` ↔ `k+1`) on a *single* interface moves `F` by **2–23× more than
`F`'s own error**: on 4×4 D=3, χ=4 → error 5.2e-3 but max `|ΔF|` 1.2e-1; χ=8 → error 6.5e-6, max
`|ΔF|` 2.9e-5. Where the contraction is lossless (3×3 at χ≥3) `ΔF` is exactly 0, so the
diagnostic correctly reports stationarity when there is nothing to truncate.

**Finding 3 — sensitivity to the *minimal* swap is concentrated, but do not over-read it.**
Median `|ΔF|` over interfaces is **exactly 0**: on a 4×4 only **4 of 36** interfaces respond to
the `k` ↔ `k+1` swap, always the same ones — `PH[:N,2,3]`, `PH[:S,2,3]`, `PV[:W,3,2]`,
`PV[:E,3,2]`, the **maximally balanced central cuts**, which carry the most rank.

⚠️ **Corrected later.** That "4 of 36" is a property of *that one minimal perturbation*, not a
general statement that only 4 interfaces matter. The exact residual below shows **12** interfaces
truncate on the same system and **all 12** are far from stationary (residual 0.31–1.05). A
full subspace change is a much stronger move than swapping one adjacent direction, so both
measurements are consistent — but any claim of the form "only N interfaces matter" must name the
perturbation it was measured under.

**Finding 4 (negative) — explicit non-uniform χ buys nothing.** Since `k = min(maxdim, rank)`,
low-rank interfaces *already* use less than χ; per-interface χ allocation is already implicit.
Raising χ on only the 4 central interfaces gives **bit-identical** `F` to raising it everywhere.

**Finding 5 — sweep count collapses with χ, so large χ is cheaper AND better.** 4×4 D=3, warmed:

| χ | sweeps to converge | \|F − exact\| | total time |
|---|---|---|---|
| 4 | 24 | 5.2e-3 | 0.46 s |
| 6 | 18 | 1.6e-3 | 0.16 s |
| 8 | 16 | 6.5e-6 | 0.13 s |
| 12 | **1** | 5.3e-15 | 0.02 s |

χ=8 is 3.5× faster *and* 800× more accurate than χ=4. At lossless χ the projectors are exact so
the fixed point is reached in one sweep. **Small χ is the expensive regime as well as the
inaccurate one** — the usual accuracy/cost tradeoff is inverted here, so do not reach for small
χ to save time.

### The stationary (partial-Schur) projector — derived, prototyped, VALIDATED, not yet landed

The prototype that validated this (`examples/ctm_stationary_projector_prototype.jl`) has been
**deleted** — it documented a disproven approach and its Schur step had come to error with
`LAPACKException` after later cleanups, so it was a committed example that did not run. The full
derivation, the carrier table and the sign trap are all recorded below; recover the script from git
(`7bb0b2f`) if it is ever wanted.

**Stationarity condition.** Variations that preserve rank-`k` projector structure are
`δΠ = (I−Π) X Π + Π Y (I−Π)`, so `δF = Tr[GᵀδΠ]` vanishes for all `X, Y` iff

```
Π Gᵀ (I−Π) = 0   and   (I−Π) Gᵀ Π = 0      ⟺   [Π, Gᵀ] = 0
```

i.e. **the kept subspace is a `Gᵀ`-invariant subspace** — the partial Schur problem.

**Construction.** Schur-decompose `Gᵀ = Z T Zᵀ`, `ordschur` the `k` dominant eigenvalues to the
front, partition `T = [[T11,T12],[0,T22]]`, and solve one Sylvester equation. Then

```
Π = Z · [[I, X],[0, 0]] · Zᵀ            P_A = Z[:,1:k]     P_B = [I  X] Zᵀ
```

**SIGN TRAP — this cost a debug cycle.** The commutator vanishes iff `T11 X − X T22 = T12`, and
Julia's `sylvester(A,B,C)` solves `A X + X B + C = 0`, so the call is

```julia
X = sylvester(T11, -T22, -T12)      # NOT +T12
```

With `+T12` the "stationary" projector comes out with a residual **2× worse** than the current
one — it looks like the whole idea has failed rather than like a sign error.

**Assembling `G` — the fiddly part.** `E_R` is region `R` contracted with the projector pair
removed and the raw interface legs left open (prime the east copy). The trap is that the two
blocks *carrying* the interface are **not always two corners**. For `PH[:N,x,y]`:

| region | west carrier | east carrier |
|---|---|---|
| plaquette `(x+½,y−½)` | `C_NW(x+1,y)` | `C_NE(x+1,y)` |
| h-edge `(x+½,y)` | `C_NW(x+1,y)` | `C_NE(x+1,y)` |
| vertex `(x,y)` | `T_N(x,y)` | `C_NE(x+1,y)` |
| vertex `(x+1,y)` | `C_NW(x+1,y)` | `T_N(x+1,y)` |
| v-edge `(x,y−½)` | `T_N(x,y)` | `C_NE(x+1,y)` |
| v-edge `(x+1,y−½)` | `C_NW(x+1,y)` | `T_N(x+1,y)` |

Swapping west/east silently pairs `P_B` with `P_B`. Also: build **every** block in the region
from the enlarged pieces plus the *same* projector set, and use the **fresh** pair for the target
interface in `Z_R` too — mixing the stored `S.PH` pair with freshly-enlarged corners mismatches
the previous level's `w` index and gives silently wrong numbers.

**Two validations, both passing.** Per region, `Tr[E_Rᵀ Π] == Z_R` exactly (2e-16). And a
finite-difference check on the weighted log-sum: `ΔF/predicted` = 0.999873, 0.999987, 0.999999 at
`ε` = 1e-5, 1e-6, 1e-7.

**Measured on the central interface `PH[:N,2,3]` (4×4 D=3, χ=6, n=9, k=6):**

| | residual `(‖ΠGᵀQ‖+‖QGᵀΠ‖)/‖G‖` |
|---|---|
| current two-sided projector | **0.443** |
| partial-Schur projector | **6.3e-15** |

The current projector is far from stationary, the Schur one is exactly stationary by
construction, and the subspaces genuinely differ — principal angles `0°, 0°, 0°, 7.2°, 8.5°,
40.5°`. `Π` is confirmed a true idempotent (`‖Π²−Π‖ = 1.2e-15`, rank 6).

### REMOVED: the Möbius-stationary projector (kept here as a record of why)

Was generalised to all four families and wired in behind `CTM_STATIONARY[]`; **the code has
since been deleted** (`CTM_STATIONARY`, `_ctm_carriers`, `_ctm_gradient`, `_ctm_schur_projector`)
because it measurably made results worse. `_ctm_region_desc` and `_ctm_block` were kept — they now
serve [`marginal_inconsistency`](#), the one trustworthy diagnostic. Recover the deleted pieces
from git (`783fb28`) rather than rewriting; the prototype in
`examples/ctm_stationary_projector_prototype.jl` also remains. The region set per interface is found by
*enumerating candidate centres and keeping those holding one carrier from each side*, rather than
hand-coding 24 cases — which also handles boundaries (fewer than six regions) with no special
case.

**The machinery is correct.** Every truncating interface, all four families, 4×4 D=3 χ=6:

| interface | n | k | residual, base pair | residual, stationary pair |
|---|---|---|---|---|
| `(:N,1,3)` | 9 | 3 | 4.85e-1 | **1.6e-15** |
| `(:N,2,3)` | 9 | 6 | 3.13e-1 | **2.8e-15** |
| `(:N,2,4)` | 18 | 6 | 6.74e-1 | **8.2e-15** |
| `(:S,2,2)` | 18 | 6 | 9.49e-1 | **4.0e-15** |
| all 12 truncating | | | 0.31 – 1.05 | 1e-15 – 1e-14 |

**And `F` gets much worse anyway:**

| χ | base | stationary |
|---|---|---|
| 4 | 5.20e-3 | 7.24e-3 |
| 6 | 1.56e-3 | 7.01e-1 |
| 8 | 6.46e-6 | 6.21e-3 |

So this is not a bug — the projector is *exactly* stationary and the answer is *worse*.

**Why, and it is worth internalising.** `[Π, Gᵀ] = 0` holds for **any** `Gᵀ`-invariant subspace,
and there are combinatorially many. `F` is not variational (its error changes sign), so those
stationary points are mostly saddles and spurious branches. Selecting the **dominant-|λ|** branch
measured **113× worse** than the input; re-selecting the branch *nearest* the incoming projector
(by eigenvector overlap — a continuation step rather than a jump) recovered most of that but is
still worse. Free stationarity simply does not single out the physical solution.

**What to try instead.** In region-graph/CVM theory the free energy is stationary at the
consistent solution **subject to marginal-matching constraints** — a parent region's marginal on a
child's variables must equal the child's. That is *constrained* stationarity. Freely extremising
`F` over the projectors, which is what this implements, is a different and evidently wrong
problem. The natural next attempt is to impose parent/child consistency directly rather than
`∂F/∂Π = 0`.

**Kept, not deleted, despite being off by default:** `_ctm_gradient` computes the exact `∂F/∂Π`
(validated two independent ways) and is the only way to *measure* the stationarity residual, which
is the one `ln Z`-free diagnostic available. Any constrained scheme will need it. Rip it out if it
proves dead weight.

### ⚠️ RETRACTED — the section below was wrong. `F` IS the Bethe/Kikuchi functional.

**The claim that follows ("`F` is an estimator, not a variational functional", based on
`cos(M_v,M_e) ≈ 0.845` at lossless χ) was an artifact of an index bug and is FALSE.** Kept only
because the bug is instructive.

**The bug.** The diagnostic rebuilt blocks with `_ctm_block(S, tbl, k->S.PH[k], k->S.PV[k], …)`.
But `S.PH`/`S.PV` are the projectors derived during the sweep that *produced* `S`, so their legs
reference the **pre-`S`** indices, whereas `_ctm_enlarged(S,…)` produces `S`'s indices:

```
enlarged C_NW(3,3) open inds: (3, 3, 3, 3)
stored PH[:N,2,3] P_A inds:   (3, 3, 9)     → only ONE index in common
```

So the projector contracted over one leg instead of two and left danglers. The earlier
`Tr[E_Rᵀ Π] == Z_R` check did not catch it because both sides were built the same wrong way — it
verified *internal consistency*, not correctness. **Lesson: a self-consistency check between two
objects you built with the same helper proves nothing about that helper.** Always tie the check to
an independent reference (here: the Möbius sum against `ln Z`).

**Corrected measurement**, using the freshly-derived (index-consistent) projector set — the same
ones `sweep_vertex_environments` applies:

| χ | Möbius sum of rebuilt regions vs `ln Z` | `cos(M_v,M_e)` min / median |
|---|---|---|
| 6 | 11.9235029 vs 11.9219469 (= the known χ=6 error) | 0.993098 / 1.000000 |
| 8 | agrees to 6.5e-6 (= the known χ=8 error) | 0.999837 / 1.000000 |
| 12 | **exact** | **1.000000 / 1.000000** |
| 16 | **exact** | **1.000000 / 1.000000** |

**So the exact solution IS a stationary point: the marginals are exactly parallel.** The BP
analogy holds precisely — at bond dimension 1 this *is* BP, `M_e` is the reverse message, `M_v` is
the vertex factor times the other incoming messages, and `M_v ∥ M_e` is the BP fixed-point
equation. `F = Σ_v ln Z_v − Σ_e ln Z_e + Σ_p ln Z_p` is the Bethe/Kikuchi free energy and BP/GBP
fixed points are its stationary points.

**Two things this gives us.**

1. **A validated `ln Z`-free convergence diagnostic.** `1 − cos(M_v, M_e)` over the edge-like
   blocks is 7e-3 at χ=6, 1.6e-4 at χ=8, and exactly 0 at lossless χ. It needs no reference value,
   so unlike `|F − ln Z|` it is safe to optimise against.
2. **A recalibration of the headroom.** The current algorithm is *already nearly stationary*
   (`1 − cos ≲ 1e-3`). The remaining finite-χ error is therefore mostly **truncation**, not
   non-stationarity — consistent with Finding 5, where sweep count collapses and accuracy jumps as
   soon as χ reaches the lossless range. Do not expect a large win from enforcing stationarity
   harder; expect it from spending χ where the rank actually is.

`schursolve` (KrylovKit) remains the natural tool for the *block fixed point* — `C ∝
project(grow(C))` is a non-symmetric eigenproblem, so Krylov–Schur replaces fixed-point iteration
and should cut the 8–12 sweeps. That is a convergence-rate win, not an accuracy win, given (2).

### Gauge fixing — LANDED, `gauge` default ON

The pair has an exact gauge freedom `P_A → P_A R`, `P_B → R⁻¹ P_B`, leaving `Π = P_A P_B` and so
every region value untouched. The sweep used to pick that gauge — and a fresh `Index` — arbitrarily
each iteration, so `S` and `sweep(S)` sat in different bases. That is what blocked every
accelerator and left `|ΔF|` as the only convergence signal.

**The gauge must be UNITARY — measured, this is not optional.** Canonicalising by QR (pushing the
triangular factor into `P_B`) changes `F`: 1.1e-2 at χ=4, 2.6e-3 at χ=6, 8.7e-6 at χ=8, invariant
only at lossless χ=12. A triangular `R` is not unitary, so it changes the metric on the interface
and the *next* level's SVD truncation then picks a different subspace. **Truncation is not gauge
invariant**; only inner-product-preserving changes of basis are safe.

So `_ctm_align` uses orthogonal **Procrustes**: with `M = P_A_newᵀ P_A_old = U S Vᵀ`, take
`R = U Vᵀ`, the nearest unitary. Reusing the old `Index` then makes successive blocks comparable.
`F` comes out invariant to 1e-14 at χ = 4/6/8/12, and identical to all printed digits against the
pre-gauge default at χ = 4/6/8/12/16.

**Payoff, χ=6:**

| | comparable blocks | state distance |
|---|---|---|
| gauge off | **0 / 84** every sweep | never available |
| gauge on | 0 → 20 → 48 → **84 / 84** | 2.8e-2 → 9.3e-3 → 2.9e-3 → 1.4e-3 → 6.4e-4 → 9.7e-5 → 2.6e-5 |

It bootstraps over ~3 sweeps — `ins` contains the lower level's kept index, so index stability
propagates upward — then every block is comparable and the distance decays cleanly at ≈0.35/sweep.

**Unexpected bonus: `|ΔF|` is an unreliable stopping criterion.** Over sweeps 8–10 it *rises*:
1.2e-7 → 3.4e-7 → 5.4e-7, oscillating at the roundoff floor of a signed log-sum. The state distance
decays monotonically throughout. `update` now converges on `max(|ΔF|, statedist)` via
`_ctm_statedist`, which both stops later when it should and actually certifies convergence.

**What this unlocks.** Anderson/JFNK/Krylov acceleration is now well-posed — iterates share a
vector space, so they can be linearly combined. At ≈0.35/sweep the current Picard rate needs ~8–12
sweeps; acceleration should cut that materially. Continuation in χ (warm-starting a large-χ run
from a converged small-χ one) also becomes possible. Neither is implemented yet.

Caveat: `_ctm_statedist` returns `nothing` while bases bootstrap, and `_ctm_align` falls through to
the unaligned pair whenever the previous projector does not live on the current `ins` (first
gauge-fixed sweep) or the rank changed. Both are silent by design; a rank change resets that
interface's basis.

### No convergence headroom at fixed χ — and why `schursolve` has no home here

Tracking both `|F − ln Z|` and `mean(1 − cos)` out to 22 sweeps, 4×4 D=3:

| χ | `mean(1−cos)` @ sweep 8 | @ sweep 22 | `\|ΔF\|` @ 22 |
|---|---|---|---|
| 4 | 8.0217e-3 | 8.0230e-3 | 7.5e-11 |
| 6 | 3.8825e-4 | 3.8825e-4 | 1.1e-14 |
| 8 | 1.1387e-5 | 1.1387e-5 | 1.1e-14 |

**Both measures plateau at the same sweep (~8)**, while `|ΔF|` keeps falling to 1e-14 with no
further gain in either. So the residual marginal inconsistency is a **truncation floor, not
incomplete convergence**, and its size tracks `F`'s own error (χ=4: 8.0e-3 vs 5.2e-3; χ=8: 1.1e-5
vs 6.5e-6). There is **no accuracy headroom at fixed χ from converging harder** — the fixed point
is the best state available at that χ.

Confirming the artifact story once more: at χ=8, sweep 1 gives `|F − ln Z| = 2.2e-6`, *better* than
the converged 6.5e-6 — but its cosine is worse. Early stopping "wins" on `F` are cancellation, not
accuracy. Converge properly.

**Why `schursolve` does not apply to this engine.** Every Krylov/Anderson/JFNK accelerator needs
the iterates to live in a common vector space so they can be linearly combined. They do not here:
each sweep calls `Index(k)` afresh and re-derives the kept *subspace*, so `S` and `sweep(S)` are
expressed in different bases. Linear mixing of successive states is ill-defined without a
gauge-fixing step that pins the interface bases across sweeps — that gauge fixing is the real
prerequisite, not the eigensolver.

Nor is there a non-symmetric eigenproblem to hand it. On a **finite, position-resolved** lattice
each `C_NW(x,y)` is a *different* object, so corner growth is a recursion (a DP), not a
`C ∝ project(grow(C))` eigenproblem — that form only appears with translation invariance, i.e. in
*infinite* CTMRG, which is exactly what this engine deliberately is not. The one non-symmetric
eigenproblem in the pipeline is `Gᵀ` in `_ctm_schur_projector`, and that path is disproven. In the
QR route the small object `W = R_A R_B†` needs its **SVD** (Eckart–Young) for optimal rank-`k`
truncation, not a Schur form; if that ever becomes a bottleneck the right tool is
`KrylovKit.svdsolve`, and at the current sizes (`n ≤ 128`) dense is faster anyway.

**Given all of the above, the levers that remain for fixed-BD accuracy are:** a bulk/infinite
variant (where the corner fixed point *is* an eigenproblem and Krylov–Schur genuinely belongs);
gauge-fixing the interface bases across sweeps, which would unlock acceleration and cut the ~8
sweeps; and non-uniform χ driven by *effective* rank rather than `maxdim` (see the gate caveat
below). Accuracy per χ is truncation-limited — three independent attempts at the projector all
came back negative.

### Re-run headroom against the trusted objective: THE CURRENT PROJECTOR IS ALREADY OPTIMAL

Repeating the projector-headroom experiment with `mean(1 − cos)` as the objective instead of
`|F − ln Z|`. 4×4 D=3, χ=6; baseline `mean(1−cos) = 3.88e-4`, `|F − ln Z| = 1.56e-3`. Swapping the
marginal kept direction on each genuinely truncating interface:

| interface | mean(1−cos) | cos gain | F gain |
|---|---|---|---|
| `(:W,4,2)` | 4.19e-4 | 0.93 | 1.00 |
| `(:S,2,3)` | 5.32e-4 | 0.73 | 2.39 |
| `(:S,2,2)` | 5.38e-4 | 0.72 | 1.00 |
| `(:W,3,2)` | 6.06e-4 | 0.64 | **4.34** |
| `(:E,2,2)` | 9.41e-4 | 0.41 | 1.00 |
| `(:E,3,2)` | 1.34e-3 | 0.29 | 0.41 |

**Every swap makes marginal consistency worse** — the top-`k` two-sided/QR projector is already
optimal by the criterion we trust. There is no headroom in the projector.

**And the cross-check that settles the earlier confusion:** the `|F − ln Z|`-optimal swap
`(:W,3,2)`, worth 4.34× on `F`, is simultaneously **0.64× worse** on the cosine. Rank correlation
between the two objectives across swaps is only 0.49 and the extrema disagree. This is the third
independent confirmation that `|F − ln Z|` gains are cancellation artifacts (the first two: they
degrade single-site observables, and they degrade the stationarity residual).

**Conclusion for where effort should go.** The projector is not the bottleneck; the remaining
finite-χ error is truncation. Accuracy per χ cannot be bought with a better subspace choice, a
better arithmetic route (the QR result), or harder stationarity enforcement. It has to come from
spending χ where the rank is — and per Finding 5 large χ is *cheaper* anyway, since sweep count
collapses. `schursolve` on the block fixed point remains worthwhile purely to cut the 8–12 sweeps.

Gate caveat for anyone re-running this: `k >= prod(dim.(ins))` does **not** identify the
truncating interfaces. Many have *effective* rank below χ, so the swap is a silent no-op (20 of 32
here). Gate on the number of non-negligible singular values instead.

### SUPERSEDED (wrong, see above): `F = Σ c_R ln Z_R` is an ESTIMATOR, not a variational functional

Stationarity w.r.t. the **blocks** (the `C`s and `T`s) rather than w.r.t. a projector is the right
instinct — it is what "self-consistency among marginals" means. `Z_R` is *linear* in every block
(each region contains a given block once), so `Z_R = Tr[M_R B]` with `M_R` = region minus `B`, and

```
∂F/∂B = Σ_{R∋B} c_R M_R / Z_R = 0
```

For an **edge tensor** — exactly two regions, weights `+1, −1` — this collapses beautifully:

```
M_v / Z_v = M_e / Z_e     ⟺     M_v ∥ M_e
```

(parallelism is *equivalent*, not merely necessary: the constant is forced to `Z_v/Z_e` because
`Tr[M_R B] = Z_R`.) The parent (vertex) and child (edge) environments must be parallel — literally
marginal consistency. Note `B` itself drops out, so block-stationarity is a condition on
*everything else*; this is why it cannot be met by choosing a subspace for one projector, and why
the `∂F/∂Π` route was structurally wrong rather than merely ill-conditioned.

**But measured, the exact solution does NOT satisfy it.** Cosines `cos(M_v, M_e)` over 48 edges,
4×4 D=3:

| χ | \|F − ln Z\| | min | median | max |
|---|---|---|---|---|
| 8 | 6.5e-6 | 0.083 | 0.857 | 1.000 |
| **12** | **5.3e-15** | 0.019 | **0.845** | 1.000 |
| 16 | 5.3e-15 | 0.019 | 0.845 | 1.000 |

At lossless χ, `F` is exact to machine precision while the marginals remain badly inconsistent,
and the residual is **flat in χ**. So:

* `F = Σ c_R ln Z_R` has **no entropy terms and its variables are tensors, not normalised
  beliefs** — it is not the CVM variational functional. It is an estimator that returns `ln Z`
  whenever every region is *individually* exact, courtesy of `V − E + P = 1`.
* Therefore **"make `F` stationary" is not a valid target in any form** — w.r.t. projectors or
  w.r.t. blocks. Both attempts degraded the answer, and this is why.
* Getting each region's *scalar* right is far weaker than getting its *marginals* consistent.

**This is the same phenomenon as the observable gap.** CVM beats boundary MPS on `ln Z` but loses
on single-site observables. Both follow from the above: the Möbius cancellation fixes the scalar
sum, but nothing enforces marginal consistency, and an observable read off a single vertex ring
gets no cancellation. **Marginal consistency is the stronger and more useful target**, and
enforcing it should fix observables — not just `ln Z`.

### Proposed next algorithm: the block fixed point, solved by partial Schur

The correct reading of "stationarity w.r.t. `C`" is not `∂F/∂C = 0` for the estimator above, but
the genuine **fixed-point condition** that CTMRG is built on:

```
C  ∝  project( grow(C) )          i.e.   C is an eigenvector of grow-then-project
```

with the projector taken as the **invariant subspace of that (non-symmetric) growth map**, rather
than from a density matrix. That is exactly where a **partial Schur** decomposition is the right
tool — a symmetric eigendecomposition does not apply because the four corners are distinct and
non-symmetric — and it is almost certainly what the collaborator means. `KrylovKit.schursolve`
provides Krylov–Schur directly, and `arnoldi` already pulls KrylovKit in.

This is a different algorithm from the current biorthogonal/density-matrix projector, not a tweak
to it. Enforce parent/child consistency directly (GBP-style) rather than deriving it from the
estimator, and judge by the marginal cosines above plus observables — never by `|F − ln Z|`.

### Two-sided projectors inside the per-vertex DP

Each interface is bounded by exactly two corners, which are its two half-environments:

| interface | west/north half | east/south half |
|---|---|---|
| `PH[:N,x,y]` | `C_NW(x+1,y)` | `C_NE(x+1,y)` |
| `PH[:S,x,y]` | `C_SW(x+1,y)` | `C_SE(x+1,y)` |
| `PV[:W,x,y]` | `C_NW(x,y+1)` | `C_SW(x,y+1)` |
| `PV[:E,x,y]` | `C_NE(x,y+1)` | `C_SE(x,y+1)` |

So `ρ_L`, `ρ_R` for the biorthogonal pair come from those two corners, and the pair is applied
by **side**: `P_A` to everything touching the interface from the west/north
(`C_NW(x+1,y).right`, `T_N(x,y).right`), `P_B` to everything touching it from the east/south
(`C_NE(x+1,y).left`, `T_N(x+1,y).left`). Every contraction across the interface then pairs one
`P_A` with one `P_B`, for all three region types (in a plaquette the two corners meet directly;
in a vertex region the edge tensor sits between them).

Both enlarged corners expose the *same* interface index set, which is what makes the pair
well defined — e.g. for `PH[:N,x,y+1]`:

```
C̃_NW(x+1,y+1) = C_NW(x,y)·T_N(x,y)·T_W(x,y)·a(x,y)            → right ins = (PH[:N,x,y].w, hlink(x,y))
C̃_NE(x+1,y+1) = C_NE(x+2,y)·T_N(x+1,y)·T_E(x+2,y)·a(x+1,y)    → left  ins = (PH[:N,x,y].w, hlink(x,y))
```

**Important negative result:** the projector must be applied *at the moment of growth*, when the
interface is `χ·D`-dimensional. Re-truncating an already-projected interface index in place is a
**no-op** — its dimension is already ≤ χ, so the projector is the identity and the sweep changes
nothing. A cheap "refine the existing basis" iteration therefore does not work: each sweep must
**regrow** the corners from the previous state's blocks and project the fresh, enlarged
interfaces. That is why the iteration and the two-sided projector are one piece of work.

### Stationarity (the iteration)

A two-sided projector inside the per-vertex DP needs the complement environment, which needs
the other corners — genuinely circular. So it must be iterated (Jacobi): within a sweep,
**every** tensor is built from the *previous* sweep's state, and the new projectors are derived
from the enlarged corners. Mixing new corners with old edges produces index/basis mismatch
("not a valid combiner contraction"). Store the projector dicts in the state so the nested
interfaces can reference the previous level's indices. Iterate until `F` stops changing.

The API follows the `BeliefPropagationCache` convention — the cache *carries* its environments
and `update` returns a cache with converged ones:

```julia
cache = update(CTMEnvironmentCache(ψ, χ))     # runs the two-sided sweep to stationarity
F     = cvm_freenergy(cache)                  # read the number off
lnZ_v = region_lnZ(cache, x, y)               # or a single region
```

`cvm_freenergy` on an **un-updated** cache falls back to the greedy one-sided pass. That is
deliberate (same as evaluating an un-updated BP cache) but it is a trap worth knowing: the two
numbers differ by 3–4 orders and the greedy one is non-monotone in χ. An earlier API returned
`(env, F, iters)` from a `cvm_environments` function, which conflated running the algorithm with
reporting the answer and made exactly this difference look like a bug.

**The tail is slow — do not stop at 2–3 sweeps.** Sweep 1 does nearly all the work (it is what
replaces the greedy pass's one-sided cuts, worth 3–4 orders on its own), but `|ΔF|` then decays
over roughly **8–12** sweeps before reaching 1e-8. Truncating the iteration at 2–4 lands
mid-transient, where `|F − ln Z|` bounces by an order of magnitude and reads convincingly like
a limit cycle. It is not one: run it out and `|ΔF|` reaches 1e-10/1e-11 monotonically.

Block *scale* is not part of the fixed point and does not need fixing: every corner appears in
exactly four regions with Möbius weights `+1 −1 −1 +1` (vertex, h-edge, v-edge, plaquette) and
every edge tensor in two with `+1 −1`, so any per-block rescaling cancels from `F` identically.

This is what "make `Z_v` stationary" means operationally, and it lands *together with* the
two-sided projector rather than as a separate feature.

### Double layer, kept lazy

A vertex's factors stay a `Vector{ITensor}` from `bp_factors` (`[ket, bra]`, or
`[ket, op, bra]`); environments keep ket and bra legs separate at dimension D and are never
fused to D². Each absorption contracts the flat list `[env; factors…]` with
`contraction_sequence(...; alg="optimal")`. Note the measured lesson: this representation alone
buys ~1× — the win is in *not* forming the fat site tensor, while the dominant cost is the
merged per-column object. Absorbing one layer at a time is ~40× faster but far less accurate,
so it is not used.

## Validation plan (each step must pass before the next)

1. **Lossless limit**: at large χ every region type (vertex / h-edge / v-edge / plaquette)
   contracts to the exact `Z`. Test interior, edge *and* corner vertices — boundary regions are
   where index bookkeeping breaks.
2. **Möbius identity**: `V − E + P = 1`, so `F = ln Z` at large χ.
3. **Cancellation**: at moderate χ, per-region error ≫ error of the weighted sum.
4. **Monotonicity in χ** — non-monotonic error means the projector is wrong (this is the
   canary that caught the one-sided bug).
5. **Stationarity**: `F` converges under sweeps.
6. **Against boundary MPS** at matched χ and D, on a **random non-symmetric** network
   (a symmetric Ising model can be passed by accident via symmetry crutches).

Reference values: `contract(tn; alg="exact")`, `norm_sqr(ψ; alg="exact")`, and
`contract(tn; alg="boundarymps", mps_bond_dimension=χ)`.

## Status

| piece | state |
|---|---|
| two-sided eig projector + Arnoldi + cutoff discipline | done, committed, validated |
| lazy double-layer factors | done, committed |
| row-absorption contractor | **DELETED** — it was the wrong object (see below) and kept reading as boundary MPS. `contract(tn; alg="boundarymps")` is the reference contractor. |
| single-site observables | **done** — `vertex_ring`, plus `expect`/`rdm` with `alg="ctmrg"` |
| block renormalization | **done** — every C/T rescaled at build; gauge cancels from the Möbius sum |
| per-vertex C/T DP (grow + project) | **done** — `vertex_environments`, single-pass greedy, generic over `bp_factors` |
| region contraction + Möbius sum | **done** — `region_lnZ`, `cvm_freenergy` |
| stationary sweep with two-sided projectors | **done, validated** — `sweep_vertex_environments`, driven to convergence by `update(cache)` |
| re-measure vs boundary MPS | **done** — now competitive-to-better, see below |

Test coverage: `test/test_ctmenvironment.jl`, testset "CVM per-vertex environments" — every
region type on a square *and* a non-square grid, the Möbius identity, sweep convergence,
beats-greedy, and monotonicity in χ.

### Fixed: the `:S`-family off-by-one in `sweep_vertex_environments`

The symptom was two leftover dim-2 indices out of `region_lnZ` (an interface projected on one
side but not the other). The cause was an index-key off-by-one, but not in a single key: the
**whole `:S` family** was shifted. `:S` and `:E` blocks are keyed by their *first included*
row/column (`T_S[x,y] = rows ≥ y`), so the `:S` family lives at `y ∈ 2:Ly`, with `y = Ly+1` the
empty block. All four `:S` rebuild loops (`PH[:S]`, `C[:SW]`, `C[:SE]`, `T[:S]`) ran over
`y ∈ 1:(Ly-1)` instead: they built a useless `y = 1` and **never built `y = Ly`**. The loop
*bodies* were correct throughout, as were the `P_A`/`P_B` side assignments and both `PV`
families (the `:E` ranges were already right). Fix = the four ranges.

This predicts the observed symptom exactly: on a 3×3, region `(1,2)` needs `C_SE(2,3)` and
`T_S(1,3)`, so `vl(1,2)` and the `PV[:E,2,2]` interface are left unconsumed — two dim-2 indices.

**The diff-against-the-greedy-oracle diagnosis worked and is worth reusing.** Compare, per
block key, a *signature* of the index set — sorted dims, each tagged as a raw lattice link or a
projector index — between `vertex_environments` and the rebuild. Two notes on reading it:

* Filter to **present/absent** mismatches (`nothing` vs a tensor). Those are the bug.
* **Differing dimensions are expected and are not bugs.** The two-sided projector legitimately
  truncates below the greedy one-sided cut when the *complement* is low rank — e.g. on a 3×3,
  `C̃_NE(3,3)` reduces to `T_N(3,2)·a(3,2)`, rank ≤ 2 across a dim-4 interface, so `k = 2` is
  lossless where greedy keeps 4.

### Measured

All regions contract to exact `Z` at large χ — interior, edge and corner vertices, boundary
edges, corner plaquettes (0 – 2.0e-14) — and the Möbius sum returns `ln Z`, confirming
`V − E + P = 1`. Cancellation is real: random 4×4, D=3, χ=8 → single region 2.6e-2, CVM sum
**6.5e-6** (~4000×).

Two-sided sweep vs greedy one-sided pass, and vs boundary MPS at matched χ and D. Random
**non-symmetric** networks (`random_itensor`, so signed — a symmetric Ising model can be passed
by accident via symmetry crutches):

| case | χ | CVM greedy | CVM swept | boundary MPS |
|---|---|---|---|---|
| positive 4×4 D=3 | 4 | 2.7e-5 | **1.2e-8** | 6.9e-8 |
| | 6 | 2.9e-6 | **5.0e-10** | 1.1e-9 |
| | 8 | 1.2e-7 | **5.7e-13** | 1.7e-10 |
| signed 4×4 D=3 | 6 | **1.1e-2** | 3.8e-2 | 6.5e-2 |
| | 8 | 5.7e-3 | **1.1e-5** | 1.0e-3 |
| signed 5×5 D=3 | 6 | 2.1e0 | 2.3e-2 | **7.6e-3** |
| | 8 | 2.8e-1 | **1.7e-4** | 2.1e-2 |
| PEPS norm 4×4 D=2 | 8 | 3.0e-3 | **6.6e-6** | 1.2e-3 (row-CTM) |
| | 12 | 2.3e-3 | **8.0e-8** | 5.0e-5 (row-CTM) |

So the doc's earlier verdict is **reversed**: on the recorded comparison point (random positive
4×4, D=3, χ=8) CVM now reads 5.7e-13 against boundary MPS's 1.7e-10 — a ~300× win, where the
greedy pass lost by ~4 orders. The sweep is worth 3–4 orders over greedy across the board. CVM
wins at most χ and loses occasionally (5×5 D=3 at χ=6, and at χ=2 where nothing is converged);
call it competitive-to-better rather than uniformly better.

Monotonicity in χ (validation step 4, the projector canary) now holds for the swept result —
`5.2e-3, 1.6e-3, 6.5e-6, ~1e-14` on 4×4 D=3 at χ = 4, 6, 8, 12 — while the greedy pass is still
visibly non-monotone (`1.4e0, 2.5e-1, 6.1e-2` with a bump at χ=4, and on the PEPS norm a flat
~2.5e-3 floor at every χ that the sweep breaks straight through).

### LANDED: the two-option engine — `projector = :cut` or `:cycle`

`CTMOptions(; projector = :cut | :cycle)` selects the interface projector; `cycle_fill` tunes the
one sub-choice inside `:cycle`. Both land on the same interface keys, so every consumer — the sweep,
the regions, `cvm_freenergy`, `expect`, `rdm` — is shared, and `:cut` is the graceful fallback for
any plaquette `:cycle` declines. Two passes in `sweep_vertex_environments`, the cut pass backfilling
whatever the cycle pass left unset. 180 tests green, new testset "Two-projector engine".

**Verified head-to-head**, 5×5 Ising PEPS D=3, `⟨X⟩` error against `alg="exact"`, their engine
measured *here* through `contract_Z11((X·t, t), A_local, c_local)` rather than quoted:

| χ | `:cut` | `:cycle` | their engine | `marg` `:cut` → `:cycle` |
|---|---|---|---|---|
| 4 | 1.52e-04 | **5.18e-05** | 5.22e-05 | 2.0e-07 → 2.8e-09 |
| 9 | 4.24e-07 | **5.13e-08** | 5.13e-08 | 2.9e-11 → **3.3e-16** |
| 16 | 6.26e-09 | **9.30e-10** | 9.28e-10 | 2.6e-14 → **8.7e-15** |
| 32 | 7.39e-12 | 7.39e-12 | 1.33e-12 | 6.4e-17 → 6.4e-17 |

`:cycle` matches their engine to within 2% at χ ≤ 16 and is stationary to machine precision there.
At χ=32 the gate (below) declines and the result equals the cut.

⚠️ **Correction to the older tables.** The Python `⟨X⟩` entry at χ=32 recorded as `8.149e-14` is
wrong; measured, it is **1.33e-12**. The `⟨Z⟩` table's `2.46e-13` / "130×" was never re-measured and
should be assumed to carry the same error.

#### LANDED: zero-pad the retained index to a uniform width

`kres` — the rank `schursolve` actually resolves — fluctuates from sweep to sweep and from plaquette
to plaquette (measured 1–4 on heavy-hex, 18–22 on the 5×5 interior). Every such change RESIZES the
interface, which breaks `_ctm_align`'s dimension guard, throws away the gauge, and hands the next
sweep a basis it cannot compare with the last. **That is the instability underneath the whole cycle
route**, and it is why five different fillers all failed in five different ways.

The fix is bookkeeping, not physics: build the biorthogonal pair at `kres`, then embed it in a
uniform-width index with the remaining columns exactly ZERO. `Π = P_A P_B` still has rank `kres`, so
every region value is identical to simply shrinking — but the index dimension stops moving. This is
what their fixed-χ storage plus explicit `rank` field buys them, and the reason it looks like a silly
implementation detail is that it *is* one; it just happens to be load-bearing. The padding must be
applied AFTER `_ctm_biorth`, never before — whitening a pair with null columns inverts a singular
overlap, the `S^(-1/2)` amplification `qr_cutoff` exists to guard against.

Result on the 5×5, `⟨X⟩` error and stationarity:

| χ | `:cut` | `:cycle` | their engine | `marg`, `:cycle` |
|---|---|---|---|---|
| 9 | 4.24e-07 | **5.13e-08** | 5.13e-08 | **3.3e-16** |
| 16 | 6.26e-09 | **9.11e-10** | 9.28e-10 | **3.3e-16** |
| 32 | **7.39e-12** | 5.24e-11 | 1.33e-12 | **4.5e-16** |

**`F` is now stationary at every χ with the cycle applied everywhere** — 3–5e-16 throughout, where
before χ=32 only looked stationary because the plaquettes were declining to the cut. χ=32 also
improved from 1.16e-10 (shrink) to 5.24e-11. And the saturation is gone: heavy-hex now runs
3.7e-06 → 1.3e-10 from χ=8 → 32, improving rather than degrading.

**Honest remaining gaps** — this is stabilised, not finished:

| | `:cut` | `:cycle` | |
|---|---|---|---|
| 5×5 χ=32 `⟨X⟩` | 7.39e-12 | 5.24e-11 | 7× behind the cut, 40× behind their engine |
| heavy-hex χ=32 `⟨Z⟩` | 1.11e-16 | 1.34e-10 | cut wins outright on sparse grids |
| square 4×4 D=3 χ=8 `⟨Z⟩` | 5.23e-03 | 5.86e-02 | 11× behind; `marg` 2.9e-03, not stationary |
| hex 4×4 χ=32 | 5.55e-17 | 1.28e-09 | `marg` 1.9e-05, not stationary |

So stationarity holds on the 5×5 at every χ but NOT uniformly across lattices, and the cut still wins
on sparse grids and on random D=3. The next step is the one the five-filler table points at: a **block
Arnoldi on a χ-wide warm-started seed, taking the top-k Ritz vectors without a convergence
requirement** — which is what their `ritz_rank` of 29–32 actually is, and which the uniform padded
width is now the precondition for.

**Status, stated plainly: `F` is stationary where the gate passes, and not where it declines.** Full
stationarity at every χ and lattice is NOT achieved.

### FIXED: the engine was irreproducible run to run

Every Krylov solve took its start vector from the **global** RNG (`eigsolve` via an explicit
`randn`, `svdsolve` via KrylovKit's internal default). Measured spread across identical runs: `⟨X⟩`
at χ=16 over 8.1e-10 – 9.3e-10, `|F − ln Z|` at χ=32 over 8.9e-16 – 6.2e-15. That is larger than the
difference between the two projectors, i.e. **the comparison above was not measurable before this
was fixed.** All three sites now draw from a locally seeded `Xoshiro` (`_ctm_startvec`, and a
per-plaquette seed in `_ctm_cycle_projectors`), which also means a sweep no longer perturbs the
caller's `Random.seed!`. Verified bit-identical across runs and independent of global stream
position.

### ⚠️ The reference values ARE correct — and how they were nearly discarded

`ln⟨ψ|ψ⟩ = -6.217866847854575` and `⟨X⟩ = 0.916900598128483` are right, now confirmed by numpy and
ITensors at 3.6e-15 / 2.2e-16 **on a verified-good transfer**.

I spent most of a session concluding they were wrong by 7.5e-8 and 2.1e-9, and that their `Z()` was
a defective estimator. That was entirely an artefact of my own export. **The npz PICKLES JAX
ARRAYS**, so `np.load(..., allow_pickle=True)` runs jax on unpickle and, without x64 enabled,
returns everything as **float32** (`0.24756516516` against the true `0.24756516631`).
`configure_jax()` must run *before* `np.load`. Avoiding jax in the export script does not dodge
trap 2 of the handoff — **it triggers it**, and my script carried a comment asserting the opposite.

Every cross-check I ran then agreed, because numpy, ITensors, brute force and their own loader were
all reading the same degraded bytes. What finally broke it was running *their* engine on a small
random network where the exact answer was independently constructible: it matched brute force to
0.0–3.6e-15, which no defective estimator could. `_compute_Z` (`ctm_primitives.py:276`) *is* an
inclusion–exclusion functional, `ΣZ11 − ΣZ01 − ΣZ10 + ΣZ00` — that observation was correct and is
worth knowing — but it converges to the exact answer, as a Möbius sum must.

The lesson is sharper than "recompute on the far side of the transfer": that is what I did, four
times. **Validate the transfer against a case whose answer you can construct independently of the
transfer**, and treat any discrepancy at the 1e-8 level on float32-capable data as a dtype bug until
proven otherwise. `examples/ctm_ising5x5_benchmark.jl` now asserts both reference values, so a
degraded export fails loudly instead of surfacing as a physics result.

### THE ACCURACY FLOOR WAS OUR OWN ARNOLDI TOLERANCE — worth 590×

Both engines appeared to floor on observables (ours 5.2e-11 at χ=32, theirs 1.33e-12, against an
`ln Z` that reached 4.4e-15). The cause is a **fourth-power dynamic-range squeeze**, and it is now
measured rather than argued.

**The cycle spectrum is the product of the four factor spectra.** On the 5×5 at χ=32, for an
under-resolved plaquette:

```
cycle   s_k/s_1 :  1 → 4.4e-09 (k=10) → 4.0e-12 (k=22) → 4.2e-14 (k=32)
factors s_32/s_1:  2.25e-04, 6.39e-04, 4.75e-04, 5.17e-04   →  product 3.5e-14
```

`3.5e-14` against a measured `4.2e-14` — the four-fold product spends three quarters of float64's
dynamic range, leaving ~4 digits of usable corner spectrum.

**And the tolerance sat above the spectrum it was resolving.** `Arnoldi(tol = 1e-13)` is ABSOLUTE on
the residual, so with `s_22/s_1 = 4.0e-12` the solver declared an invariant subspace at k≈19-22 while
directions out to k=32 were still four orders above machine epsilon. The projector silently discarded
them, and that — not the criterion, not the rank rule, not any filler — was the floor.

**Fix:** normalise the cycle action by its dominant singular value (five power iterations; the
invariant subspace is scale-invariant, so it costs nothing) and the tolerance becomes relative. Then
`tol = 1e-16`:

| 5×5, χ=32 | `:cut` | `:cycle` tol 1e-13 | 1e-15 | 1e-16 | **normalised + 1e-16** | their engine |
|---|---|---|---|---|---|---|
| `⟨X⟩` err | 7.39e-12 | 5.24e-11 | 7.56e-13 | 8.87e-14 | **4.77e-14** | 1.33e-12 |
| `lnN` err | 4.4e-15 | 1.52e-11 | 2.58e-13 | 2.13e-14 | **1.07e-14** | — |
| `marg` | 6.4e-17 | 4.5e-16 | 4.2e-16 | 4.2e-16 | **3.4e-16** | — |

**155× better than our cut projector and 28× better than their engine, while staying stationary.**
χ=9 and χ=16 are unchanged and still match their engine (5.132e-08 exactly; 9.277e-10 against 9.279e-10).

Sparse grids improve just as sharply, and the earlier saturation is gone entirely:

| `⟨Z⟩` err, χ=32 | `:cut` | `:cycle` before | `:cycle` now |
|---|---|---|---|
| heavy-hex 2×2 D=2 | 1.11e-16 | 1.34e-10 | **0.00** (exact) |
| hex 4×4 cplx D=2 | 5.55e-17 | 1.28e-09 | **3.89e-16** |
| square 4×4 D=3, χ=8 | 5.23e-03 | 5.86e-02 | **6.55e-03** |

**Dead end recorded: factor-norm balancing.** The idea that motivated this — rescaling the four
corners so a badly-scaled product stops losing range — has nothing to fix. Measured spread of the
four factor norms is only 2.2× (median, 8.9× worst), because the C/T blocks are already renormalised
at build. Osborne-style *diagonal* balancing was therefore not attempted either; the squeeze is
intrinsic to the spectrum being a fourth power, not to scaling.

**What remains.** Two weak spots, both now narrow:

* heavy-hex at χ=8: 3.7e-06 against a cut that is exact (1.11e-16). Here `kcyc = min(χ, narrowest
  bond) = 4`, so the loop is STRUCTURALLY bottlenecked and the extra directions come from padding
  rather than from the criterion. This is the case the `G`-based projector should address (below).
* square 4×4 D=3: 2.97e-04 against the cut's 1.30e-04 at χ=32, ~2.3×.

**Still to try: partial Schur on `G` rather than the four-corner cycle.** Same stationarity condition
(`[Π, Gᵀ] = 0`), but `G = Σ_R c_R E_R / Z_R` is a **sum**, so there is no fourth-power squeeze and no
per-bond bottleneck — exactly the two things that limit the remaining cases. `_ctm_gradient` (38
lines), `_ctm_schur_projector` (49) and `_ctm_carriers` are recoverable from `783fb28`, and their
dependencies `_ctm_region_desc`/`_ctm_block` are still live. It was rejected once, but that was
measured before the engine was made reproducible run-to-run, when the spread between runs exceeded
the effect. NOT yet attempted.

### ⚠️ RETRACTED: the "structural bottleneck" explanation of `:cycle` on sparse grids

Several sections above attribute `:cycle`'s weakness on hex/heavy-hex to `kcyc = min(χ, narrowest
bond)` bottlenecking the loop below what the interfaces carry. **Measured, that is backwards.** At
χ=8 on heavy-hex the cut keeps rank 1–4 at interfaces of raw dimension 8/16/32 — those interfaces
genuinely have that rank, so the cut is lossless and exact (1.1e-16) — while the cycle keeps 8. The
cycle keeps MORE, not fewer.

What is actually true: the cycle map's numerical rank there is **1** (spectrum `1.0, 0.0, 0.0, 0.0`),
so the loop pins one direction and propagation supplies the rest, spanning an arbitrary subspace.

This also retracts the reading of the `G`-projector experiment: `G` getting heavy-hex exactly was
taken as confirming the bottleneck diagnosis, and it cannot confirm a mechanism that is not there.
`G`'s heavy-hex result stands as a measurement; its interpretation does not.

Three consequences of the corrected picture were tested and all fail — do not retry:

| attempt | result |
|---|---|
| truncate to the cycle's numerical rank (their `CTM_eig_cutoff` analogue) | much worse: heavy-hex χ=32 `0.00 → 2.0e-06`, hex χ=8 `3.9e-08 → 2.2e-03` |
| independent per-bond eigensolves (their stated robustness trick) | byte-identical — where the cycle is rank-deficient the independent solve cannot converge `kcyc` either and falls back to propagation |
| defer to the cut on non-truncating interfaces | neutral to worse (heavy-hex χ=32 `0.00 → 1.1e-16`, hex χ=8 `marg` 4.6e-12 → 1.1e-09) |

The open question is therefore narrower and better posed than before: **when the cycle map's rank is
far below the interface's, what should supply the remaining directions?** Propagation currently does,
and it is arbitrary. Every local rule tried so far (five fillers, plus the three above) fails.

### TRIED AND FALSIFIED: extracting the cycle subspace from `range(M)` instead of the eigen ranking

On sparse grids the four-corner map is nearly NILPOTENT — measured on hex, per-factor ranks 4–5 but
only ONE nonzero eigenvalue, with the product of the factors' own `s₂/s₁` at ~1e-07 (far above eps,
so degeneracy and not precision). `schursolve(…, :LM)` therefore pins one direction and calls the
rest invariant. Since `range(M)` is itself `M`-invariant (`M·Mx = M²x ∈ range(M)`) with dimension
`rank(M)`, the subspace we want exists and is simply not eigenvalue-ranked — this looked like exactly
the "schur vs eig matters for very non-Hermitian networks" case Zaletel names.

Implemented via `svdsolve` on the matrix-free action, with two triggers. Both fail.

| trigger | result |
|---|---|
| switch whenever the eigen spectrum is degenerate (`nnz < kcyc`) | 5×5 χ=16 regressed 9.28e-10 → 1.95e-09, losing the match with their engine |
| switch only when the eigen route UNDER-SPANS (`rank(M) > nnz`) | byte-identical — the 5×5 under-spans too, so it does not discriminate |

Multi-seed against pure eigen, the second variant is worse essentially everywhere: square 5×5 D=2 at
χ=8 `4/6 med 2.52 → 2/6 med 0.19`; square 4×4 D=3 at χ=4 `2/6 med 0.63 → 0/6 med 0.03`; and it does
NOT fix hex χ=16 across seeds (0/2, median 0.00) even though seed 1001 alone looked fixed
(1.15e-11 → 3.33e-16).

**The lesson, now stated three ways from three experiments: the loop's spectrum does not tell you
what the interface needs, and neither does its range.** Dropping the low-eigenvalue directions is
worse, keeping arbitrary ones is worse, and re-ranking by singular value is worse. The eigen-ranked
invariant subspace is better for accuracy even while it spans fewer directions than `rank(M)`.
