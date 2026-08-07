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
⚠️ Do NOT read convergence off `_ctm_statedist`: it is gauge-sensitive and plateaus at ~3e-11 for
`:cycle` while the observable does not move at all. That plateau is basis wander in the diagnostic,
not a residual — it is insensitive to the Arnoldi `tol` across 1e-10/1e-13/1e-16, which is what ruled
out solver noise. `|ΔF|` is useless for both (1e-15 from sweep 1, even while `:cut`'s state distance
is 2e-01).

---

## Accuracy, against verified exact references

Collaborator's 5×5 Ising PEPS, D=3. `⟨X⟩` at their site (2,2) = our (3,3). Their engine measured
here through `contract_Z11((X·t, t), A_local, c_local)` at matched χ — **not** quoted from the older
docs, one of whose entries was wrong by 16×.

### Observables — the metric that matters

| χ | `:cut` | **`:cycle`** | their cycle | ours vs theirs |
|---|---|---|---|---|
| 4 | 1.515e-04 | 8.277e-05 | **5.218e-05** | 1.6× behind |
| 9 | 4.241e-07 | **5.132e-08** | 5.132e-08 | equal |
| 16 | 6.255e-09 | **9.277e-10** | 9.279e-10 | equal |
| 32 | 7.392e-12 | **4.774e-14** | 1.330e-12 | **28× ahead** |

`:cycle` beats `:cut` at **every** χ (1.8× / 8.3× / 6.7× / 155×).

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
| 4 | **2.030e-07** | 1.815e-04 |
| 9 | 2.889e-11 | **3.400e-16** |
| 16 | 2.605e-14 | **2.220e-16** |
| 32 | 6.384e-17 | **3.428e-16** |

Machine precision for χ ≥ 9 — `∂F/∂B = 0`, the same condition BP satisfies.

### Random states — MULTI-SEED, 6 seeds per cell, PURE formulations

Ratio = `:cut` error / `:cycle` error on `⟨Z⟩`, so **>1 means `:cycle` is better**. Cells where both
reach machine precision are dropped as uninformative.

| case | χ=4 | χ=8 | χ=16 |
|---|---|---|---|
| square 5×5 D=2 | 5/6, med 1.4 (0.60–173) | 4/6, med 2.5 (0.10–22.7) | both exact |
| square 4×4 D=2 | 2/6, med 0.56 (0.26–4.5) | 3/6, med 1.4 (0.45–44.7) | both exact |
| square 4×4 D=3 | 2/6, med 0.63 (0.08–3.0) | 1/6, med 0.51 (0.09–5.5) | 2/6, med 0.93 (0.30–21.3) |
| hex 4×4 D=2 | **6/6**, med **86** (4.3–809) | **6/6**, med **531** (231–14999) | **0/4, med 0.00** |

⚠️ **A single seed per cell is worthless here** — ranges span 3–4 orders inside one cell
(0.10–22.7, 231–14999). An earlier single-seed table claimed "wins 6 of 9 on squares"; that did not
survive.

**Clear wins:** hex at χ=4–8 (median 86× and 531×). **Coin flip:** square lattices generally.
**Known failure:** hex at χ=16 — see below.

#### The one known failure regime of pure `:cycle`

hex 4×4 D=2 at χ=16: 0/4 seeds, median ratio 0.00. It is a narrow, diagnosable window — `:cycle`
wins by 10²–10⁴ at χ=4–8 and both are exact by χ=32 — and it sits exactly at the χ where the
environment becomes **lossless**. The mechanism is measured: at χ=14 only 3% of interfaces have the
cycle's resolved rank below the cut's rank; at χ=16 it is **16%**, with the cycle keeping 4 where the
cut keeps all 16. `marg` jumps eight orders there (2e-15 → 1.2e-07) and never recovers, while the cut
reaches machine precision.

Deferring to the cut on exactly those interfaces FIXES it (hex χ=16 → both exact, 1.15e-11 →
4.44e-16) and also lifts square 5×5 D=2 (med 1.4 → 3.0 at χ=4, 2.5 → 6.4 at χ=8). It is not shipped
because it makes the lattice carry a mixture, and mixtures were measured worse than either pure
method (square 4×4 D=3 at χ=32: 9.17e-04 vs 2.97e-04 pure-cycle and 1.30e-04 pure-cut). **The
formulations are kept pure by choice; this is the price.** If the mixture is ever wanted, the rule is
"defer whenever `rank(cut) > rank(cycle)`" and it is a ten-line change.

**Practical guidance:** `:cycle` for hex/heavy-hex, and on square lattices measure both — the
variance across seeds is larger than most of the effects. Check `:cut` whenever the environment is
near-lossless, which is where `:cycle`'s known failure lives.

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

* **χ=4 costs 1.6× against their engine.** A fully resolved rank-4 invariant subspace is a worse
  projector than an under-resolved one — the criterion losing at severe truncation, which every scan
  shows (`:cycle` also loses at χ=2). It explains their three-iteration Arnoldi: under-convergence is
  a regulariser at small χ. Two fixes were tried and both are worse; see the design doc. `:cycle`
  still beats *our* `:cut` there (8.3e-05 vs 1.5e-04).
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
