# Finite CTMRG: current status

**Read this first.** `finite_ctmrg_design.md` is the full record (~1900 lines, chronological, and
contains retracted claims clearly marked as such); `ctmrg_cycle_projector_handoff.md` is an older
handoff carrying a correction banner. This file is the current state.

---

## What the engine is

Position-resolved finite CTMRG framed as a region-graph (CVM) free energy: a 4C+4T ring on every
vertex, grown by local corner moves, with

```
F  =  Σ_v ln Z_v  −  Σ_e ln Z_e  +  Σ_p ln Z_p            (Möbius +1 / −1 / +1)
```

One cache, one sweep, one set of consumers (`cvm_freenergy`, `expect`, `rdm`, `region_lnZ`), and
**two interchangeable interface projectors**:

```julia
cache = update(CTMEnvironmentCache(ψ, χ; projector = :cut))     # default
cache = update(CTMEnvironmentCache(ψ, χ; projector = :cycle))
```

| | what it optimises | stationary? |
|---|---|---|
| `:cut` | the best rank-χ truncation of ONE bipartition, each interface independently | no |
| `:cycle` | the dominant invariant subspace of the four-corner cycle — consistency AROUND the plaquette | **yes** |

Both are two-sided and biorthogonal, both write the same interface keys, and both are exact at
lossless χ.

**Both are PURE — no per-interface mixing.** `:cycle` derives every interface from the four-corner
cycle; it does not fall back to `:cut` on a per-interface basis.

⚠️ **The reason is design, not measurement, and an earlier version of this file overstated it.** A
rank-based deferral WAS implemented and measured multi-seed, and on accuracy it is at least as good
as pure: it wins three cells (including hex χ=16, which pure fails outright), pure wins three, three
tie. The claim that "a mixture is worse than either pure method" rested on ONE seed at ONE χ (square
4×4 D=3 at χ=32: 9.17e-04 vs 2.97e-04 pure-cycle and 1.30e-04 pure-cut) and is NOT corroborated by
the multi-seed data at neighbouring χ.

Purity is chosen for three design reasons: a mixed lattice is not stationary, so "is `:cycle`
stationary?" stops being a well-posed question — and stationarity is the whole reason the cycle
projector exists; the switch is a discontinuous rank comparison, so behaviour can flip on a small
change in the environment; and two formulations you can reason about separately are worth more than
a third thing wearing one of their names. If the mixture is ever wanted, the rule is "defer whenever
`rank(cut) > rank(cycle)`" and it is a ten-line change.

The one remaining fallback is structural, not a choice: a plaquette whose cycle is undefined (a
corner carrying more than its two interfaces, or a `schursolve` that throws) declines wholesale, and
a `@warn` reports how many did — silence would read as "the cycle ran everywhere".

**Both are convergent.** `⟨X⟩` error on the 5×5 Ising at χ=16, by sweep count:

```
sweeps   1       2       3       4       5       6       7       8
:cut     1.7e-09 4.9e-09 6.1e-09 6.2e-09 6.2e-09 6.3e-09 6.3e-09 6.3e-09
:cycle   9.7e-10 9.3e-10 9.3e-10 9.3e-10 9.3e-10 9.3e-10 9.3e-10 9.3e-10
```

`:cycle` settles in **two** sweeps and is flat thereafter; `:cut` takes ~4 and lands 6.8× worse.

⚠️ **Both corrected 2026-08-09 — read the last section of this file before trusting either claim.**

1. **"`:cycle` settles in two sweeps" is measured on an INTERIOR observable and does not generalise.**
   The boundary of the same lattice needs more (6×6 corner: 3; heavy-hex: 5), and `update` was
   stopping before it got there. Settling fast is precisely what walked `:cycle` into the bug.
2. **"Both are convergent" is FALSE on 8×8 D=2.** `:cycle` there is a genuine limit cycle at χ=16 and
   χ=32 — `F` fluctuating in the 5th decimal, observables oscillating between 1e-4 and 1e-6 forever.
   This is currently the single biggest open problem with `:cycle`.
3. The `~3e-11 _ctm_statedist` plateau was blamed on "basis wander in the diagnostic". The diagnostic
   was in fact **sampling 13% of the state**; the residual was real, in the blocks it never compared.
   Sign-flip wander was later tested directly on 8×8 and **ruled out** (0 of 58 moving blocks repaired
   by a flip). `|ΔF|` remains useless for both, as stated.

---

## Accuracy, against verified exact references

Collaborator's 5×5 Ising PEPS, D=3. `⟨X⟩` at their site (2,2) = our (3,3). Their engine measured
here through `contract_Z11((X·t, t), A_local, c_local)` at matched χ — **not** quoted from the older
docs, one of whose entries was wrong by 16×.

### Observables — the metric that matters

| χ | `:cut` | **`:cycle`** | their cycle | ours vs theirs |
|---|---|---|---|---|
| 4 | 1.515e-04 | **4.767e-05** | 5.218e-05 | **1.1× ahead** |
| 9 | 4.241e-07 | **5.132e-08** | 5.132e-08 | equal |
| 16 | 6.255e-09 | **9.277e-10** | 9.279e-10 | equal |
| 32 | 7.392e-12 | **4.774e-14** | 1.330e-12 | **28× ahead** |

`:cycle` beats `:cut` at **every** χ (3.2× / 8.3× / 6.7× / 152×) and **matches or beats their
engine at every χ**. The χ=4 entry was previously recorded as 8.277e-05 / "1.6× behind"; that was
measured before the start vector was seeded on plaquette POSITION only, and never re-measured.

### Free energy — `:cut` wins, and that is expected

| χ | `:cut` | `:cycle` |
|---|---|---|
| 4 | **2.120e-04** | 3.399e-04 |
| 9 | **9.286e-09** | 1.747e-08 |
| 16 | **6.698e-12** | 1.550e-10 |
| 32 | **4.441e-15** | 1.066e-14 |

The Möbius sum cancels per-region error ~4000×, which flatters `:cut`. **Do not judge the projectors
on `|F − ln Z|`** — it is accurate and blind. Judge on observables and `marginal_inconsistency`.

### Stationarity — `marginal_inconsistency`

| χ | `:cut` | `:cycle` |
|---|---|---|
| 4 | **2.030e-07** | 1.790e-04 |
| 9 | 2.889e-11 | **3.400e-16** |
| 16 | 2.605e-14 | **2.220e-16** |
| 32 | 6.384e-17 | **3.428e-16** |

Machine precision for χ ≥ 9 — `∂F/∂B = 0`, the same condition BP satisfies.

### Random states — MULTI-SEED, 6 seeds per cell, PURE formulations

Ratio = `:cut` error / `:cycle` error on `⟨Z⟩`, so **>1 means `:cycle` is better**. Cells where both
reach machine precision are dropped (`n` is how many seeds remained informative).

| case | χ=4 | χ=8 | χ=16 |
|---|---|---|---|
| square 5×5 D=2 | **1.99** (4/6) | **1.32** (4/6) | **8.49** (2/2) |
| square 4×4 D=2 | **2.87** (4/6) | **5.39** (4/6) | **6.50** (2/3) |
| square 4×4 D=3 | 0.19 (1/6) | 0.17 (1/6) | 0.24 (1/6) |
| hex 4×4 D=2 | **1.3e+03** (6/6) | **1.3e+03** (6/6) | **1.85** (2/4) |
| heavy-hex 2×2 D=2 | **0.00** (0/6) | **0.00** (0/6) | 18.0 (1/1) |

Hex across the transition that used to fail (6 seeds): χ=8 **1.3e+03** (6/6), χ=12 **6.7e+03** (6/6),
χ=14 **8.6e+03** (6/6), χ=16 **1.85** (2/4), χ=20 6.12, χ=24 2.80, χ=32 113.

⚠️ **These numbers SUPERSEDE the earlier table, and the change is large.** The previous table recorded
square 4×4 D=2 at 0.56 (2/6) and 1.4 (3/6), and hex χ=16 as an outright failure at "0/4, median 0.00".
Both improved because the Krylov start vector was later seeded on plaquette POSITION only, and that
change was never re-measured outside the 5×5 benchmark. **The hex failure regime no longer exists** —
`:cycle` now wins or ties at every χ on hex.

**Where `:cycle` wins:** hex (10³ at χ≤14), both D=2 squares (2-8×), and the 5×5 Ising at every χ.
**Where it loses:** far from convergence (below), and heavy-hex 2×2 at χ ≤ 8 where `:cut` is already
exact (1.1e-16) and `:cycle` sits at 1e-05.

#### The real story: `:cycle` fails at BOUNDARY sites, and on random D=3 states

⚠️ Two earlier framings in this file were wrong and are retracted. "Square 4×4 D=3 is a standing ~5×
loss" was measured only inside the unconverged regime. Its replacement — "`:cycle` overtakes once
`:cut` reaches ~1e-3" — was written up before being tested and FAILS as a prediction on 5×5 D=3 and
6×6 D=2. What the follow-up actually found is more specific and more useful.

🛑 **RETRACTED 2026-08-09 — the boundary table below is an artefact of the convergence bug.** The
whole "`:cycle` fails at boundary sites" finding, and every mechanism proposed for it (D² caps,
`kcyc` bottlenecks, bad seeds, χ-relative-to-L), was `update` stopping before the boundary converged.
The boundary-adjacent projectors were measured **bit-identical** between the two backends, which
should have been the tell. With the test fixed, the 6×6 corner goes −2.5 → −5.3 (equal to `:cut`) and
the edge to −6.1 (**better** than `:cut`). Kept for the record; do not cite these ratios.

**1. `:cycle` degrades at boundary sites and is excellent in the bulk.** 6×6 D=2, 5 seeds, ratio
`:cut`/`:cycle` (>1 = `:cycle` better), same states, three different observable sites:

| χ | interior (3,3) | edge (6,3) | corner (1,1) |
|---|---|---|---|
| 4 | **1.48** | 1.19 | 0.61 |
| 8 | **2.65** | 0.12 | 0.80 |
| 16 | **2.27** | **0.00** | 0.01 |
| 24 | **11.09** | **0.00** | 0.04 |

At the edge site `:cycle` runs 1.5e-03 → 3.7e-04 → 2.1e-03 → 1.3e-03: it saturates near 1e-3 and
never converges, while `:cut` reaches 6.8e-07. In the bulk `:cycle` converges monotonically
(1.95e-03 → 6.08e-04 → 2.00e-04 → 4.99e-05 → 2.80e-06 → 1.13e-07) and ends 11× ahead.

**This contaminates most earlier scans in this document.** They picked the observable site as
`collect(vertices(g))[n÷2]`, which on a square grid is a BOUNDARY vertex, while the 5×5 Ising
benchmark uses the interior (3,3). So "`:cycle` loses" results were largely measured at the boundary
and "`:cycle` wins" results in the bulk. Re-measured at interior sites, D=2 squares are clean wins:
6×6 2.65/2.27/11.09, 4×4 3.07/26.90.

**2. Random D=3 states remain a genuine loss, even in the bulk.** sq 4×4 D=3 at (2,2): 0.06 → 0.33 →
0.84 for χ = 8/16/32; sq 5×5 D=3 at the centre (3,3): 0.25 → 1.13 → 0.69. Improving with χ but not
a win. Note the contrast that isolates the variable: the PHYSICAL 5×5 Ising PEPS is also D=3, also
measured at (3,3), and `:cycle` beats both `:cut` and the collaborator's engine there at every χ. So
the discriminator is the STATE — structured/physical versus random signed — not D, not the site.

Neither effect is explained yet. The boundary one is the more actionable: it is a clean saturation
signature localised to sites whose vertex ring touches the lattice edge.

**Practical guidance:** prefer `:cycle` once the calculation is anywhere near converged — say `:cut`
error ≲ 1e-3, which is the measured crossover. It wins on hex, on D=2 squares, on the 5×5 benchmark at
every χ, and it is 15.7× faster at 8×8 D=3. Use `:cut` when χ is far too small for the problem (the
unconverged regime, where the criterion strains) and on heavy-hex at small χ. A cheap way to tell
which regime you are in: run `:cut` first; if its error estimate is still ≫1e-3, you are below the
crossover and should raise χ rather than switch projector.

---

## The three things that made `:cycle` work

Everything else in the long design doc is failed attempts. These three are the content.

### 1. Zero-pad the retained index to a uniform width

`kres` — the rank `schursolve` actually resolves — fluctuates sweep to sweep and plaquette to
plaquette (1–4 on heavy-hex, 18–22 on the 5×5 interior). Every change **resizes the interface**,
which trips `_ctm_align`'s dimension guard, discards the gauge, and hands the next sweep a basis it
cannot compare with the last. That was the instability under the whole route.

Build the biorthogonal pair at `kres`, then embed it in a uniform-width index with the remaining
columns exactly **zero**. `Π = P_A P_B` still has rank `kres`, so every region value is unchanged —
this is bookkeeping, not physics, and it is what their fixed-χ storage plus `rank` field buys them.
**Pad AFTER `_ctm_biorth`, never before**: whitening a pair with null columns inverts a singular
overlap.

### 2. A scale-free Arnoldi tolerance — worth 155×

The cycle spectrum is the **product** of the four factors' spectra, verified quantitatively:

```
factors s₃₂/s₁ :  2.25e-04 · 6.39e-04 · 4.75e-04 · 5.17e-04  →  3.5e-14
cycle   s₃₂/s₁ :                                                4.2e-14   (measured)
```

A four-fold product spends three quarters of float64's dynamic range. KrylovKit's `tol` is
**absolute** on the residual, so a fixed `tol = 1e-13` sat *above* the eigenvalues being resolved
(`s₂₂/s₁` is already 4.0e-12): Arnoldi declared an invariant subspace at k≈19–22 while directions out
to k=32 were four orders above machine epsilon, and the projector silently dropped them.

Fix: normalise the cycle action by its dominant singular value (five power iterations — free, the
invariant subspace is scale-invariant), then `tol = 1e-16` is relative. χ=32 `⟨X⟩`:
5.2e-11 → **4.8e-14**.

**This is very likely the floor the collaborator reports.** Their `CTM_eig_cutoff = 2e-14` and
`rank_from_c`'s `rtol = 1e-14` are the same mistake in their idiom — a relative cutoff on a fourth
power, so 1e-14 on the cycle discards single-corner directions at ~3e-4. Measured on their engine:
loosening those cutoffs improved `ln Z` ~6× and moved their rank ceiling off 32 (it had been pinned
at 32 for χ = 32, 48, 64 **and** 81). Worth telling them.

### 3. Never keep less than the cut would

The zero padding of item 1 leaves `Π` at rank `kres`, which is harmless while the interface is
truncating anyway. Once χ is large enough for the interface to be **lossless**, the cut keeps
everything and the cycle's shortfall becomes a real loss. Measured on hex 4×4 D=2: at χ=14 only 3% of
interfaces have `kres` below the cut's rank; at χ=16 it is **16%**, with the cycle keeping 4 where the
cut keeps all 16. That is exactly where `marg` jumps eight orders (2e-15 → 1.2e-07) and never
recovers, while the cut reaches machine precision — a clean failure bracketed by two regimes where
`:cycle` wins by 10²–10⁴.

The cut is the optimal rank-k truncation of that interface, so falling short of its rank is strictly
lossy. Compute it and defer whenever `rank(cut) > rank(cycle)`. Hex χ=16 goes
**1.15e-11 → 4.44e-16**, and multi-seed the whole cell moves from `0/6 wins, median 0.00` to both
methods at machine precision.

The cost, recorded honestly: two already-coin-flip cells shift ~2× in the median (square 4×4 D=2 at
χ=8, `3/6 med 1.44 → 2/6 med 0.60`; square 4×4 D=3 at χ=16, `3/6 med 1.16 → 2/6 med 0.70`), and
square 4×4 D=3 at χ=32 goes 2.97e-04 → 9.17e-04 with `marg` 3.4e-16 → 2.9e-06. That last one is worse
than BOTH the pure cycle and the pure cut, which suggests a mixed-family inconsistency rather than a
truncation loss — the interfaces that defer and those that do not are not mutually consistent.
Trading a catastrophic failure mode for a 2× median shift is the right call for robustness, but the
mixture effect is a real open item.

---

## Known limitations

* **χ=4 is NOT stationary** (`marg` 1.79e-04 against 2-3e-16 for χ ≥ 9), even though its observable
  now beats both `:cut` (3.2×) and their engine (1.1×). A fully resolved rank-4 invariant subspace is
  a worse *stationary point* than an under-resolved one — the criterion straining at severe
  truncation. See "under-convergence as a regulariser" below: it is fixable at χ=4 but the fix costs
  accuracy at χ=8, so it is not applied.
* ~~**Sparse grids at small χ.**~~ **RESOLVED 2026-08-09 — this was not a projector defect.** The
  whole of the bullet below (and its nilpotency analysis) was diagnosing an observable error that came
  from `update` STOPPING AFTER 4 SWEEPS, not from the cycle map. With the convergence test fixed,
  heavy-hex 2×2 D=2 at χ=8 goes **3.7e-06 → 4.0e-16**, matching the cut. See "The convergence test was
  certifying off a 13% sample" at the end of this document. The nilpotency measurement itself still
  stands as a property of that cycle map; it was simply never the cause of the error.

* **Sparse grids at small χ.** heavy-hex at χ=8 is 3.7e-06 against a cut that is EXACT (1.1e-16).
  ⚠️ This was previously attributed to `kcyc = min(χ, narrowest bond)` structurally bottlenecking the
  loop. **That is wrong** — measured, the cycle keeps MORE than the cut there, not fewer:

  ```
  χ=8, heavy-hex, raw interface dim → rank kept
    CUT:    8→[1,4]   16→[1,4]   32→[1]
    CYCLE:  8→[8]     16→[8]
  ```

  Those interfaces genuinely have rank 1–4 and the cut detects it, so its truncation is lossless. The
  cycle's own numerical **eigen**-rank there is **1** (spectrum `1.0, 0.0, 0.0, 0.0`), so only one
  direction is pinned by the loop and the rest come from propagation, spanning an arbitrary subspace.

  **Why the eigen-rank is 1 — the sharpest statement of the open problem.** It is NOT a precision
  effect: measured per-factor ranks are 4.4–5.2, and the product of the four factors' own `s₂/s₁` is
  ~1e-07, far above eps. The cycle map there is nearly **NILPOTENT** — rank 4–5, but carrying a single
  nonzero eigenvalue — so eigenvalue magnitude does not rank the directions and `schursolve(…, :LM)`
  returns a 1-dimensional dominant subspace and calls the rest invariant. This is precisely the regime
  Zaletel flags: "schur vs eig should only matter for … very non-Hermitian networks."

  The invariant subspace we want DOES exist: `range(M)` is `M`-invariant (`M·Mx = M²x ∈ range(M)`) with
  dimension `rank(M)` = 4–5. Taking it from the singular vectors instead of the eigen ranking was
  implemented and is a PARTIAL win — square 4×4 D=3 at χ=8 improved 4.6× (6.55e-03 → 1.41e-03, beating
  the cut) — but it is not landed: at χ=32 the same case degrades 4× and `marg` collapses from 3.4e-16
  to 4.0e-06, and heavy-hex does not move at all. A construction that takes the range where the
  spectrum is degenerate AND keeps stationarity where it is not is the open problem. Three fixes
  for this were tried and all fail: truncating to the cycle's numerical rank is much worse
  (heavy-hex χ=32 `0.00 → 2.0e-06`, hex χ=8 `3.9e-08 → 2.2e-03` — the extra directions are NOT junk,
  they carry region weight the loop's spectrum knows nothing about); independent per-bond eigensolves
  (the collaborator's own robustness trick) are byte-identical, because where the cycle is
  rank-deficient they cannot converge `kcyc` either and fall back to propagation; and deferring to the
  cut on non-truncating interfaces is neutral-to-worse.
* **Random D=3** square lattices: ~2.3× behind `:cut`.
* **`marginal_inconsistency` is not uniformly at machine precision** across lattices (hex 4×4 at
  χ=32 reads 1.9e-05 despite a machine-precision observable — probably a degenerate measurement,
  unverified).

Five fillers for the unresolved cycle rank were implemented and falsified, each for a different
reason — see the table in the design doc before attempting a sixth. The short version: on a
bottlenecked loop the missing directions must be simultaneously loop-invariant and weight-carrying,
and those sets are nearly orthogonal.

---

## Next steps, in order

1. **Tell the collaborator about the fourth-power cutoff** (section 2 above). One email, and it may
   be their floor.
2. **Precision diagnostic** — rerun the two weak cases in `Double64`/`BigFloat`. If the floor moves
   with mantissa bits it is conditioning; if not it is genuine truncation. Decides whether anything
   below is worth doing.
3. **Block Arnoldi on a χ-wide warm-started seed**, top-k Ritz vectors with no convergence
   requirement. This is what their `ritz_rank` of 29–32 actually is — unconverged Arnoldi directions
   kept because `dominant_eigenspace_one_sided` filters on `|λ|`, never on residual. The uniform
   padded width is the precondition.

**TRIED AND REJECTED: partial Schur on `G` instead of the four-corner cycle.** Restored from
`783fb28`, adapted to current signatures, wired as `projector = :gradient` (cut base + `G`
refinement at truncating interfaces), and measured with BOTH selection rules. Not landed.

| | `:cut` | `:cycle` | `:gradient` |
|---|---|---|---|
| 6×6 D=2 χ=4, `marg` | 1.9e-05 | **5.7e-17** | **1.0e-01** |
| heavy-hex 2×2 D=2 χ=8, `⟨Z⟩` | 1.1e-16 | 3.7e-06 | **0.00** (exact, `marg` 3.8e-17) |
| square 4×4 D=3 χ=8, `⟨Z⟩` | 5.23e-03 | 6.55e-03 | 5.23e-03 (identical to cut) |
| cost | 1× | **0.05×** | 10–30× |

Three findings, all worth keeping:

* **The original selection rule is self-referential.** It scored eigenvectors of `Gᵀ` by overlap with
  the CURRENT pair, so it reproduces whatever it is given — instrumented, it fired on all 720
  interfaces of a 4×4 D=3 and returned a result identical to the cut it started from. It never bailed
  and never truncated, so there was no protective fallback either. Selecting by `|λ|` (the actual
  criterion — the dominant invariant subspace) is a one-line fix and does change the answer.
* **`G` CONFIRMS the bottleneck diagnosis.** On heavy-hex, where `:cycle` is limited by
  `kcyc = min(χ, narrowest bond) = 4`, `G` — which has no per-bond bottleneck — gets the observable
  EXACTLY and stationary. That is direct evidence the diagnosis of `:cycle`'s remaining weakness is
  right, even though this implementation is not viable.
* **But exact stationarity is wrong at severe truncation.** `marg` 1.0e-01 on a 6×6 at χ=4, worse
  than doing nothing. This is the SAME lesson as `:cycle` losing at χ=2/4: at severe truncation the
  stationary subspace is not the accurate one. A projector that is wildly non-stationary in any
  regime cannot ship, and gating it by χ would be the tuned-threshold trap this project keeps
  relearning.

Not worth doing: filling the unresolved rank by any local rule (five falsified); factor-norm
balancing (measured spread is only 2.2×, the C/T blocks are already renormalised at build);
`G`-refinement as a projector (above — but its heavy-hex result says a bottleneck-free construction
is the right target).

---

## Reproducing the benchmark

```bash
python3 examples/export_ising5x5.py          # writes examples/peps5x5.bin
julia --project=. --startup-file=no examples/ctm_ising5x5_benchmark.jl
```

The exporter needs `joey_ctmrg_bp` on the path (`JOEY_CTMRG_BP` env var) and `jax` installed. The
benchmark asserts both reference values, so a degraded transfer fails loudly.

**⚠️ The npz PICKLES JAX ARRAYS**, so `np.load(..., allow_pickle=True)` runs jax on unpickle and
without x64 returns everything as **float32** (`0.24756516516` against the true `0.24756516631`).
`configure_jax()` must run *before* `np.load`. Avoiding jax does not dodge this — it triggers it. A
float32 export shifts `ln Z` by 7.5e-8 and `⟨X⟩` by 2.1e-9, which is small enough to look like a
real finding about the collaborator's engine, and it cost a full session's detour. Verified
references:

```
ln⟨ψ|ψ⟩ = -6.217866847854575        ⟨X⟩ at their (2,2) = 0.916900598128483
```

---

## Testing

`test/test_ctmenvironment.jl`, testset "Two-projector engine: :cut and :cycle" — option validation,
exactness at lossless χ on square **and** hex, stationarity, the observable win, no-saturation-in-χ
on heavy-hex, and determinism.

The whole file takes ~4–5 minutes. **Do not use it as an inner-loop check** — iterate with a small
purpose-built script on one lattice and 2–3 χ values, and run the file once before declaring done.

The engine is deterministic: every Krylov solve takes a locally seeded start vector, so runs are
bit-reproducible and a sweep does not perturb the caller's global RNG. Before that was fixed the
run-to-run spread exceeded the difference between the two projectors.

---

## Measured 2026-08-09: performance, and a χ-dependent accuracy lever

### `:cut` is 19× slower than `:cycle` at D=3, for identical output

Per-sweep wall clock, warm, PEPS states:

| L | D | χ | `:cut` | `:cycle` | ratio |
|---|---|---|---|---|---|
| 8 | 3 | 8 | **1.467 s** | 0.093 s | **15.7×** |
| 8 | 3 | 16 | 1.884 s | 0.248 s | 7.6× |
| 8 | 2 | 24 | 0.407 s | 0.105 s | 3.9× |
| 6 | 2 | 16-24 | 0.078-0.102 s | 0.033-0.041 s | 2.4-2.5× |
| 5 | 2 | 8-32 | 0.011-0.015 s | 0.016-0.020 s | 0.5-0.9× (cut faster) |

The decisive control: at 8×8 D=3 χ=8 both backends return **196 projectors with identical retained
dimensions** (mean 8.0, total 1568), so `:cycle` is not winning by truncating harder — it is genuinely
cheaper for the same output. The cost is the projector DERIVATION, not the corner/edge rebuild (which
both share): `:cut` forms dense QR factors of the enlarged corners (`_ctm_tri_factor`), while
`:cycle` is matrix-free. A full `update` at 8×8 D=3 takes 25.7 s with `:cut`.

Other timing facts: a sweep is 10-120 ms at these sizes; steady-state `update` is pure sweep cost
with no measurable waste; first use of a new tensor shape costs 5-14× steady state (contraction
sequence search), amortised after ~3 calls and shared across caches, so it is a warmup cost not an
algorithmic one. ⚠️ An earlier single-seed run of this table showed 19× at 6×6 χ=24; multi-seeded it
is 2.5×. Single seeds remain worthless for timing as well as accuracy.

### Under-convergence as a regulariser: real, but χ-dependent, so NOT applied

The collaborator runs only three block-Arnoldi iterations; we use `krylovdim = max(4·kcyc+8, 24)`.
Restricting it to `kcyc+2` was tested (5 seeds per cell, χ=4):

| case | `marg` default → kd+2 | accuracy (median `:cut`/`:cycle`, wins) |
|---|---|---|
| sq 5×5 D=2 | 9.18e-05 → **6.55e-06** | 2.85 (3/5) → 2.73 (3/5) |
| sq 4×4 D=3 | 1.07e-01 → **1.52e-03** | **0.06 (1/5) → 1.09 (3/5)** |
| hex 4×4 D=2 | 2.83e-08 → 2.83e-08 | 134 (5/5) → 134 (5/5) |

At χ=4 it is a clear win — it turns `:cycle`'s worst regime from an 17× loss into a coin flip, and on
the 5×5 Ising it takes `marg` from 1.79e-04 to **1.58e-16**. But at χ=8 the effect REVERSES, in the
same direction on both square cases: sq 5×5 1.25 (3/5) → 0.51 (2/5), sq 4×4 D=3 0.21 (1/5) → 0.17
(0/5). So the right Krylov depth depends on how severe the truncation is, and hard-coding it either
way is the tuned threshold this project keeps re-learning not to trust. Not applied. If someone wants
it, it is a one-line change to the `Arnoldi(; krylovdim = ...)` call and the numbers above say exactly
what it buys and costs.


## Measured 2026-08-09: the convergence test was certifying off a 13% sample

**This was the root cause of every `:cycle` boundary failure recorded above.** It was not the
projector, not the seed, not the D² boundary caps, and not `kcyc`.

`_ctm_statedist` compared only blocks whose index set was unchanged and **silently skipped the rest**.
That sample is biased: interface widths stabilise from the bulk outward, so the blocks comparable on
early sweeps are exactly the ones that settled first. Square 6×6 D=2 at χ=16, `:cycle`:

```
it   |ΔF|      statedist    coverage
1    2.83e-03  0.000e+00      0/220
2    2.13e-14  1.010e-15     28/220   <- certified convergence here, on 13% of the state
3    2.84e-14  1.021e-01     64/220   <- the other 87% was still moving by 10%
6    0.00e+00  1.542e-03    220/220
```

`update` stopped at sweep 2. The bulk and `F` were converged; the **boundary ring was not**, and it
was left ~3 orders wrong — *χ-independently*, which is exactly why it read as a systematic projector
defect rather than an early exit (χ=16/32/48 all floored at ~1e-3 while `:cut` reached 1e-9.2).

`:cut` escaped **only by chance** — its 28-block subset still read 2.6e-1 at sweep 2, so it never
tripped the test. `:cycle` walked into it *because* it settles the bulk in one sweep. The bug is as
old as the function and affects both projectors.

**Fix:** a block that appeared, vanished, or changed index set has definitively changed, so it now
returns `nothing` ("no distance exists") instead of being dropped. `update`'s existing `certified`
guard already refused to converge on `nothing` — its comment claimed the function behaved this way,
and it never did. No change to `update` was needed.

### Effect — log10 |⟨Z⟩ − exact|, seed 1

| case | `:cut` | `:cycle` before | `:cycle` after |
|---|---|---|---|
| heavy-hex 2×2 D=2, χ=8 | −15.7 | **−5.3** | **−15.4** |
| square 6×6 D=2 χ=16, corner (1,1) | −5.3 | **−2.5** | **−5.3** |
| square 6×6 D=2 χ=16, edge (6,3) | −5.2 | — | **−6.1** |
| hex 4×4 D=2, χ=16 | −15.6 | −14.6 | −15.0 |
| square 4×4 D=3, χ=16 | −2.8 | −2.3 | −2.3 (unaffected) |

`:cycle` still converges in fewer sweeps than `:cut` (6×6: 7 against 11). The 5×5 Ising benchmark is
unchanged to all printed digits except χ=32 `⟨X⟩`, 4.863e-14 → 4.818e-14. Full suite: 180 pass.

Square 4×4 D=3 is **not** this bug — forced sweeps 1..8 leave it flat at −2.3. That 3× gap to `:cut`
is a separate, genuine, and much smaller effect.

### Open, and newly visible: `:cycle` does not converge on 8×8

Fixing the test exposed what it had been hiding. Square 8×8 D=2 at χ=16, `:cycle`, sweeps 20–30:

```
it   stateDist   F             cornerErr  interiorErr
20   1.569       118.0609105   -5.5       -5.1
23   1.518       118.0619714   -3.9       -5.1
24   1.477       118.0607724   -5.8       -2.6
30   1.859       118.0609187   -5.4       -4.8
```

`F` fluctuates in the 5th decimal and observables bounce between 1e-4 and 1e-6 indefinitely — a
genuine limit cycle, **not** gauge wander (58 of 420 blocks move by >0.5 and **none** is repaired by a
sign flip; ruled out explicitly). It persists at χ=32 (final sd 1.62). `:cut` converges in 13 sweeps.

This invalidates the earlier claim that "8×8 boundary is fine at every χ" — that measured one
arbitrary point of this orbit. It also explains the old asymmetry where the *failing* 6×6 converged
silently while the *working* 8×8 warned: the 8×8's non-convergence was accidentally protecting it by
forcing all 30 sweeps.

**`degtol` cannot be the fix as it stands: it is a silent no-op for `:cycle`.** It is read only in the
`:cut` singular-value truncations (`_ctm_statedist`'s file, the two `while k > 1` backoffs) and never
in `_ctm_cycle_projectors`. Measured, 8×8 at degtol 0 / 1e-8 / 1e-4: identical to every digit. The
option table advertises it without qualification, which is now wrong.
