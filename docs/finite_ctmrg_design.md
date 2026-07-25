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


### Lanczos warnings at larger D, continuation in χ, and why observables lag

**The KrylovKit warnings are diagnostic, not corruption.** `Invariant subspace of dimension 6 …
howmany == 10` says the interface's *effective rank* is 6 while χ=10 was requested — routine once
D grows. `_ctm_eigsolve` already falls through to dense whenever `converged < k`, and the dense
result is **bit-identical and deterministic** (three runs with different RNG streams: spread
~1e-14; `CTM_ARNOLDI` on vs off: identical). Now silenced with `verbosity = 0`, since warning on
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

### Current state of the engine (the clean slate)

| piece | state |
|---|---|
| two-sided biorthogonal projector | **on** — the default path, measured optimal (see the headroom re-run) |
| `CTM_QR` triangular/QR route | **on by default** — accuracy-neutral, chosen for GPU batching |
| `CTM_GAUGE` unitary gauge fixing | **on by default** — `F` invariant to 1e-14, gives the state distance |
| `marginal_inconsistency` | **live diagnostic** — the only `ln Z`-free quality measure |
| Möbius-stationary projector | **deleted** — made results worse |
| row-absorption contractor | **deleted** — wrong object |
| `CTM_DEGTOL` pair-keeping | present, `0.0` (off) — measured a no-op on single layer, marginal on double |
| `CTM_ARNOLDI` | present, on, but dormant unless `D_layer > 4` |

Verified unchanged by the deletions, 4×4 D=3: `|F − ln Z|` = 5.203e-3 / 1.556e-3 / 6.455e-6 /
5.33e-15 at χ = 4 / 6 / 8 / 12, with `marginal_inconsistency` 8.02e-3 / 3.88e-4 / 1.14e-5 / 6e-17.

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

Prototype: `examples/ctm_stationary_projector_prototype.jl` (runs, self-checking). It covers one
interface family on a 4×4; what remains is the other three families and plumbing into the sweep.

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

### Gauge fixing — LANDED, `CTM_GAUGE` default ON

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
provides Krylov–Schur directly, and `CTM_ARNOLDI` already pulls KrylovKit in.

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
