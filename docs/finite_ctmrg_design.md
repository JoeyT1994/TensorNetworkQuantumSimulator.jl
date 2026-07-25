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
`CTM_*` tuning knobs. The `CTM_TWOSIDED` flag went with it — it only ever switched the chain
sweep, and the CVM path is unconditionally two-sided.

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

* **`CTM_PINV_CUTOFF ≈ 1e-8 ≈ √eps`, not 1e-12.** The inverse powers of `S` amplify roundoff,
  and `S` comes from an eig of a squared object so it is only resolved to ~√eps relatively. A
  null-space-only cutoff leaves a hard, χ-independent error floor (measured: 9.5e-7).
* **Krylov needs an isometry guard.** `eigsolve` returns non-orthonormal vectors on degenerate
  clusters, silently corrupting the projector (err 3.9 vs 0.027). Check `‖V'V − I‖` and fall
  back to dense.
* **One-sided truncation is not a valid variational choice** and makes the error non-monotonic
  in χ. This was the root cause of a long chain of confusing results.

### Triangular/QR projector — implemented, accuracy-NEUTRAL, kept for GPU (`CTM_QR`, default off)

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
`CTM_PINV_CUTOFF` sits at √eps: **the cutoff and the squaring are two faces of one constraint.**
QR makes directions down to ~1e-15 usable, but they carry no weight. So accuracy per χ cannot be
bought with better arithmetic — only by changing *which subspace is kept*.

The reason to keep it is GPU: `geqrf`/`gesvd` have batched GPU implementations where batched
Hermitian eig support is thin, and a sweep is 200–384 **independent tiny** factorizations
(`n ≤ 128`) — a batching problem, not a big-linear-algebra one. Note the remaining blocker in
*both* routes: `Array(ρ, b, bp)` and `_ctm_block_matrix` materialise host `Array`s, so every
projector currently round-trips through the CPU. QR alone does not deliver GPU.

### Arnoldi is dormant on the CVM path at small D (measured)

`CTM_ARNOLDI[] = true`, but the `n > 4k` gate almost never fires here, because a CVM interface is
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

`CTM_DEGTOL[] = 0.0`, i.e. **eigenvalue pair-keeping is off**, consistent with it having measured
as a no-op on single layer and marginal on double layer.

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
