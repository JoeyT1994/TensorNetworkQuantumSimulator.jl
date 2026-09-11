# Finite CTMRG: current status

**This file is the current state: what is true now, and what is still open.** For the derivations,
the full chronological record, and the long list of approaches that were tried and failed, see
[`finite_ctmrg_design.md`](finite_ctmrg_design.md). `ctmrg_cycle_projector_handoff.md` is an older
handoff, superseded.

Every measurement here is dated. Anything measured before **2026-08-09** was produced with a
convergence test that stopped early (see [The convergence test](#the-convergence-test)); tables that
predate the fix and were not re-measured are marked as such.

---

## What the engine is

Position-resolved finite CTMRG framed as a region-graph (CVM) free energy: a 4C+4T ring on every
vertex, grown by local corner moves. No row absorption, no whole-lattice chain.

```
F  =  Σ_v ln Z_v  −  Σ_e ln Z_e  +  Σ_p ln Z_p            (Möbius +1 / −1 / +1)
```

One cache, one sweep, one set of consumers (`cvm_freenergy`, `expect`, `rdm`, `region_lnZ`), and two
interchangeable interface projectors:

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
cycle and never falls back to `:cut` on a per-interface basis. The one exception is *structural*, not
a choice: a plaquette whose cycle is undefined (a corner carrying more than its two interfaces, a
rank-collapsed hex plaquette, a `schursolve` that throws) declines **wholesale**, those interfaces
are backfilled by the cut pass, and a `@warn` reports how many — silence would read as "the cycle ran
everywhere".

Purity is a **design** choice, not a measured one. A rank-based deferral was implemented and measured
multi-seed, and on accuracy it is at least as good as pure. It is not used because a mixed lattice is
not stationary — which makes "is `:cycle` stationary?" ill-posed, and stationarity is the entire
reason the cycle projector exists — and because the switch is a discontinuous rank comparison that
can flip on a small change in the environment. ⚠️ An earlier version of this file claimed the mixture
was *worse than either pure method*; that rested on one seed at one χ and is **retracted**.

---

## Choosing a projector

**Default to `:cut`.** It has no known failure regime and is the longer-tested path. `:cut` is *not*
cheaper — `:cycle` is markedly faster at scale (below) — so cost is not the reason.

**Use `:cycle` when you want stationarity or speed**, and when χ is adequate for the lattice. It is the
better projector on hex, on D=2 squares, and on the physical 5×5 benchmark at every χ, and it is
matrix-free so it is much cheaper per sweep. Its weaknesses: random D=3 states, and a **truncation-induced
limit cycle when χ is below the lossless threshold** — now understood and flagged by the convergence
check (both `|ΔF|` and the worst-region signal move on it; O(1) = raise χ), no longer an unexplained
failure. See [Open problems](#open-problems) and [The convergence test](#the-convergence-test).

---

## Accuracy

### The collaborator's 5×5 Ising PEPS, D=3 — verified against exact contraction

`⟨X⟩` at their site (2,2) = our (3,3). Their engine measured here through
`contract_Z11((X·t, t), A_local, c_local)` at matched χ. *Measured 2026-08-09, post-fix.*

| χ | `:cut` | **`:cycle`** | their engine | `marg`, `:cut` → `:cycle` |
|---|---|---|---|---|
| 4 | 1.515e-04 | **4.767e-05** | 5.218e-05 | 2.0e-07 → 1.8e-04 |
| 9 | 4.241e-07 | **5.132e-08** | 5.132e-08 | 2.9e-11 → **3.2e-16** |
| 16 | 6.255e-09 | **9.279e-10** | 9.279e-10 | 2.6e-14 → **2.4e-16** |
| 32 | 7.392e-12 | **4.818e-14** | 1.330e-12 | 6.4e-17 → **3.9e-16** |

`:cycle` beats `:cut` at every χ and matches or beats their engine at every χ (28× at χ=32).

**Free energy is the wrong metric here — do not judge the projectors on `|F − ln Z|`.** The Möbius
sum cancels per-region error ~4000×, which flatters `:cut` (it wins every row). `F` is accurate and
blind. Judge on observables and `marginal_inconsistency`.

### Random states, multi-seed — *re-measured 2026-08-09, post-fix*

Ratio = `:cut` error / `:cycle` error on `⟨Z⟩`, so **>1 means `:cycle` is better**. Median over 4
seeds. "both exact" = both below 1e-14, so the comparison carries no information. Reported at an
interior *and* a boundary site, because earlier scans conflated the two.

| case | χ=8 interior | χ=8 boundary | χ=16 interior | χ=16 boundary |
|---|---|---|---|---|
| square 5×5 D=2 | 1.20 | 1.27 | both exact | both exact |
| square 4×4 D=2 | **4.53** | **21.99** | both exact | both exact |
| square 4×4 D=3 | **0.04** | 1.45 | **0.35** | 0.78 |
| hex 4×4 D=2 | **1208** | **137** | both exact | both exact |
| heavy-hex 2×2 D=2 | both exact | both exact | both exact | both exact |

Read this as: `:cycle` is a large win on hex, a solid win on D=2 squares, and **loses on random D=3
in the interior** (0.04 at χ=8, recovering to 0.35 at χ=16). Everything reaches machine precision
once χ makes the environment lossless.

**The boundary is no longer a weakness.** It used to look like one; that was the convergence bug.

⚠️ The D=3 loss is real and is NOT the convergence bug — forcing sweeps 1..8 leaves square 4×4 D=3
flat at 2.3e-03. Note the contrast that isolates the variable: the *physical* 5×5 Ising PEPS is also
D=3, also measured at an interior site, and `:cycle` wins there at every χ. The discriminator is the
STATE — structured/physical versus random signed — not D and not the site.

---

## Performance

Per-sweep wall clock, warm, PEPS states. *Measured 2026-08-09.*

| L | D | χ | `:cut` | `:cycle` | ratio |
|---|---|---|---|---|---|
| 8 | 3 | 8 | **1.467 s** | 0.093 s | **15.7×** |
| 8 | 3 | 16 | 1.884 s | 0.248 s | 7.6× |
| 8 | 2 | 24 | 0.407 s | 0.105 s | 3.9× |
| 6 | 2 | 16–24 | 0.078–0.102 s | 0.033–0.041 s | 2.4–2.5× |
| 5 | 2 | 8–32 | 0.011–0.015 s | 0.016–0.020 s | 0.5–0.9× (cut faster) |

The decisive control: at 8×8 D=3 χ=8 both backends return **196 projectors with identical retained
dimensions** (mean 8.0, total 1568), so `:cycle` is not winning by truncating harder — it is genuinely
cheaper for the same output. The cost is the projector DERIVATION, not the shared corner/edge rebuild:
`:cut` forms dense QR factors of the enlarged corners (`_ctm_tri_factor`) where `:cycle` is
matrix-free.

First use of a new tensor shape costs 5–14× steady state (contraction-sequence search), amortised
after ~3 calls and shared across caches — a warmup cost, not an algorithmic one. Always time each
configuration twice and report the second.

---

## GPU / CUDA compatibility — *added 2026-08-19*

Both projectors run on-device (CUDA.jl) end to end, matching the boundary-MPS path. Nothing in the
algorithm is device-specific; the changes were mechanical — remove host-forcing conversions and move
scalar work off the device:

- `Base.Array(::ITensor, inds…)` FORCES a host `Matrix` (it typeasserts `Matrix{Float64}` and throws
  on a device tensor) → `ITensors.array`, which preserves the storage type.
- Hardcoded host allocations, rebuilt on the network's device/eltype: the identity isometry
  (`Matrix{Float64}(I)` → `adapt_like(B, dense(delta(…)))`), the zero-padding, and every Krylov start
  vector (same seed, drawn on CPU then moved on-device — a bit-identical draw, so CPU numerics are
  unchanged). Helpers: `_ctm_to_device`, `_ctm_zeros_like`, `_ctm_eye_like`.
- `Matrix(qr().Q/.R)` → device-generic: drop `Matrix()` on `.R`; the thin Q is `qr(X).Q · thinI`,
  which on CUDA yields the m×kk result directly via `ormqr` and never materializes the full m×m
  (also the one incidental memory win).
- Scalar reads of the singular/eigen value vectors (`S[k]`, `sortperm(vals)`, `count(>, S)`) → done
  on a host copy of the O(n) values, then the device matrices are sliced/gathered by the computed
  rank. The whitening scale is anchored to the SVD factors' device (`_ctm_svd_topk` returns host `S`
  but device `U`/`V`).

**Validation without a GPU.** The whole engine was run on a `JLArray`-backed network (the reference
`AbstractGPUArray`, the same interface `CuArray` implements) under `allowscalar(false)`, which throws
on any scalar index or host round-trip. `:cut`/`:cycle` × single/double layer all match the CPU
result to ~1e-15; the forced large-χ Krylov paths (`_ctm_svd_topk`, `_ctm_eigsolve`) run matrix-free
on device and match dense. The one thing JLArray cannot exercise is CUSOLVER itself (`svd`/`qr`/
`eigen` on a real `CuArray`) — shimmed in the harness, so a single `adapt(CuArray, tn)` run on
hardware still needs to close that gap. The harness is a scratch script, not yet in the repo.

### Precision — the memory lever

The engine is now eltype-generic throughout, so `Float32`/`ComplexF32` runs natively at ~2× less
memory (at the data-dominated large-χ scales where memory binds; the error is χ-truncation, not
precision). ONE blocker was fixed: the default convergence tolerance was a hardcoded `1e-10`,
UNREACHABLE below Float64 (Float32 roundoff on `|ΔF|`/state distance is ~1e-7), so a Float32 run spun
to `maxiter`. `update` now defaults it precision-aware, `max(1e-10, 1e3·eps(real eltype))` — Float64
stays at exactly 1e-10 (unchanged), Float32 gets ~1.2e-4 and converges. Single-layer 6×6 churn per
`update`: Float32 **782 MB → 136 MB**; Float64 142 MB, untouched.

**Memory profile (2026-08-19).** The engine is already lean, so no structural memory rewrite was
made. Persistent env footprint is small and metadata-bound (2.4–2.8 MB at 6×6 D=3 χ=12 double-layer);
churn is dominated by `_ctm_contract` (~50%) at ~1.8 KB average per contraction — many small
contractions, so peak VRAM is low. Churn micro-optimisation (the seq-key cache etc.) is low-yield on
peak and was left alone, consistent with the buffer-reuse rejection elsewhere in this project.

---

## The convergence test

`update` sweeps until

```
certified  =  it ≥ 2  AND  a real state signal exists
crit       =  max(|ΔF|, state signal)  ≤  tolerance · max(1, |F|)
```

`|ΔF|` **is not a certificate on its own.** `F` is a signed Möbius sum whose cancellation is worth
~4000×, so it can sit at its final value while the state is still the greedy seed — measured, complex
hex 4×4 D=2 at χ=64 reported `|ΔF| = 2.2e-16` on sweep 1 with a still-greedy environment. Hence the
`certified` guard (≥2 sweeps), and, for `:cut`, the second term.

**The signal depends on the projector, and for `:cycle` on the `convergence` kwarg (2026-08-20):**

* `:cut` → `crit = max(|ΔF|, _ctm_statedist²)`, the raw-tensor distance between successive C/T blocks
  (gauge on to make them comparable), squared because `F` is stationary in the state (`|ΔF| ~ sd²`).
  Correct for `:cut`, which is not a stationary point and so has no vanishing residual to watch instead.
  `convergence = :worst_region` ADDS the worst-region term on top of this pair (2026-08-21) — the
  region terms are projector-independent, so the kwarg is honored rather than silently ignored. That
  also closes a hole: with `gauge = false`, `:cut` previously certified on `|ΔF|` alone (the
  documented false-certify mode); `:worst_region` gives that configuration a safe, full-coverage
  signal. ⚠️ Rarely useful for `:cut` otherwise — measured on lossless heavy-hex χ=8, the `:cut`
  worst-region signal floors at ~8e-6 (statedist ~3e-6) while `⟨Z⟩` is exact to 1e-17, so it warns
  spuriously; `:cut`'s own pair is already observable-tight. Hence the observable paths inject
  worst-region only for `:cycle`.
* `:cycle`, `convergence = :free_energy` (**DEFAULT**) → `crit = |ΔF|` alone. Converging `F` *is* the
  target when the endpoint is a scalar (lnZ, an overlap, a free energy), and there `|ΔF|` is both
  correct and tightest: it ignores the gauge / near-null-mode wander a raw state distance sees (that
  wander spuriously GROWS with χ once χ exceeds the state's rank, though `F` is at machine precision),
  yet still MOVES on a genuine under-truncation limit cycle. Usually stops in ~2 sweeps. ⚠️ **It is NOT
  a stationarity certificate.** `F` is a *stationary functional* of the messages, so its value is
  correct to second order in the message error — it sits at its final value while the messages are
  still moving. Measured: heavy-hex χ=8, `|ΔF| = 2e-14` from sweep 1 while the worst region is still
  **2.79** and a local `⟨Z⟩` is 5e-6 wrong until sweep 5; square 4×4 D=2, `|ΔF|` hits machine-zero at
  sweep 2 while the messages do not settle until sweep 4. So `:free_energy` certifies before a
  boundary-lagged single-site observable settles.
* `:cycle`, `convergence = :worst_region` → `crit = max(|ΔF|, worst-region |Δ lnZ_r|)`, the largest
  change across the Möbius regions `F` already sums (`_ctm_region_terms` returns them in the same pass,
  so it is FREE — same per-sweep cost as `:free_energy`, it just does not stop as early). This is the
  faithful **message-stationarity** signal, and it is built from the *same* free-energy pieces as
  `|ΔF|` — the region terms `lnZ_r`, read BEFORE the ±1 Möbius signs cancel them, rather than after.
  Two properties earn it. GAUGE INVARIANT: each region is a CLOSED contraction, so the interface gauge
  `R`/`R⁻¹` cancels inside it (`_ctm_statedist`, by contrast, is not gauge invariant — at a stationary
  `:cycle` fixed point whose interface basis merely rotates it reports ~1, measured 8×8 single-layer
  while `marginal_inconsistency` sat at 1e-16, and would reject a converged state forever). NON-
  CANCELLING: the MAX over regions exposes a laggy region that `|ΔF|` misses (measured heavy-hex χ=8:
  worst region ~6e-2 through the sweep where `⟨Z⟩` is still 5e-6 wrong, falling to ~1e-10 only once
  that region settles). Cost: it over-warns on OVER-parametrised states (χ > rank), where the surplus
  null modes wander and the residual plateaus ~1e-4 though `F` and the observable are converged.

**The observable paths pick `:worst_region` for you — on `:cycle` caches.**
`expect(::TensorNetworkState; alg = "ctmrg")` and `reduced_density_matrix(...; alg = "ctmrg")` build
and solve the cache internally; when `ctm_options` selects `projector = :cycle` they default that
internal solve to `convergence = :worst_region` — an observable read off the ring needs the messages
actually stationary, not just `F`. `:cut` caches keep their statedist pair, which is already
observable-tight (and worst-region warns spuriously there, see above). Either choice is overridable
through `cache_update_kwargs`. Driving `update` yourself keeps the `:free_energy` default.

Rejected alternatives (2026-08-19), each falsified by a cheap test: `marginal_inconsistency` alone
(the BP message-overlap `1-|⟨M_v,M_e⟩|/‖·‖` — false-earlies via the local lag above, broke the heavy-hex
exactness test at err 4.96e-6); a gauge-invariant corner-spectrum distance (plateaus at the
truncation-tail level, 1e-7…1e-2); a per-corner cosine (needs gauge-fixing an open index, which the
closed-region metric sidesteps). See `_ctm_region_terms` and `update`.

### FIXED 2026-08-09: the test was certifying off a 13% sample

**This was the root cause of every `:cycle` boundary failure previously recorded here.** Not the
projector, not the seed, not the D² boundary caps, not `kcyc`.

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

`update` stopped at sweep 2. `F` and the bulk were converged; the **boundary ring was not**, and it
was left ~3 orders wrong — *χ-independently* (χ=16/32/48 all floored near 1e-3 while `:cut` reached
1e-9.2), which is exactly why it read as a systematic projector defect rather than an early exit.

`:cut` escaped **only by chance** — its 28-block subset still read 2.6e-1 at sweep 2. `:cycle` walked
into it *because* it settles the bulk in one sweep. The bug was as old as the function and affected
both projectors.

**Fix:** a block that appeared, vanished, or changed index set has definitively changed, so
`_ctm_statedist` now returns `nothing` ("no distance exists") instead of dropping it. `update` needed
no change — its `certified` guard already refused to converge on `nothing`; the function had simply
never honoured the contract its own comment claimed.

Effect, log10 |⟨Z⟩ − exact|, seed 1:

| case | `:cut` | `:cycle` before | `:cycle` after |
|---|---|---|---|
| heavy-hex 2×2 D=2, χ=8 | −15.7 | **−5.3** | **−15.4** |
| square 6×6 D=2 χ=16, corner (1,1) | −5.3 | **−2.5** | **−5.3** |
| square 6×6 D=2 χ=16, edge (6,3) | −5.2 | — | **−6.1** |
| hex 4×4 D=2, χ=16 | −15.6 | −14.6 | −15.0 |
| square 4×4 D=3, χ=16 | −2.8 | −2.3 | −2.3 (unaffected) |

`:cycle` still converges in fewer sweeps than `:cut` (6×6: 7 against 11). Full suite: 180 pass.

**Method lesson.** Four mechanisms were proposed and falsified for this — D² caps, `kcyc`
bottlenecks, bad seeds, χ-relative-to-L — while the cause was the stopping rule. The tell was
available the whole time and ignored: the boundary-adjacent projectors were measured **bit-identical**
between the two backends, so the projector could not be the difference. When two methods differ in
output but agree in the object under suspicion, suspect the loop around it. Separately, every
diagnostic script wrote stderr to `/dev/null`, hiding the convergence warnings that pointed here.

---

## The cycle-spectrum anatomy and the `cycle_gapcut` fix — *2026-08-21*

Instrumenting `_ctm_cycle_projectors` on an interior plaquette (engine-identical construction,
dense ground-truth spectra) split the `:cycle` pathologies into three distinct diseases:

| case | cycle spectrum | largest cliff | fwd/bwd retained-spectrum mismatch | biorth σmin/σmax |
|---|---|---|---|---|
| over-param (TFIM 5×5 nl=3, χ=16) | 4 real modes, then noise at 1e-11 | **1.8e5** at j=4 | **0.44** (kres=14) | 2.8e-2 |
| deep modes (TFIM 4×4 nl=4, χ=16) | smooth decay to 3.4e-14 | 4.5e2 | 4.3e-5 | **1.0** |
| under-trunc (random 5×5, χ=8) | smooth, no cliff; cut lands ON a degenerate pair (`\|λ_8\| = \|λ_9\|`) | 27 | 3.4e-13 | 1.7e-2 |
| healthy (random 5×5, χ=16) | smooth to 7.5e-9 | 1.2e2 | 2.0e-9 | 1.7e-2 |

1. **The over-param wander is retained noise, drawn differently left and right.** The engine kept
   kres=14 of which 10 were noise, and the two `schursolve`s retained spectral sets differing by
   44% — in that block `Π = P_A P_B` is not a spectral projector of anything and is redrawn every
   sweep. **Fixes (both in `_ctm_cycle_projectors`): the always-on left/right spectral-consistency
   guard** (keep the longest prefix on which the two solves agree to 1e-3; healthy cases measure
   ≤ 4.3e-5, the failure 0.44) **and the `cycle_gapcut` noise-cliff cut** (truncate below the first
   cliff that is both steep, ≤ 1e-4×, and tiny, ≤ √eps·|λ₁|). Cliffs separate what magnitudes
   cannot: junk sits at 1.6e-11·|λ₁| in one case while REAL weight sits at 6.4e-13·|λ₁| in another
   (why every fixed `cycle_rankcut` failed), but the measured cliff into noise is 1.8e5 against
   ≤ 4.5e2 anywhere inside a physical decay. A/B (25 sweeps, median of last 10 worst-region
   residuals): 5×5 nl=3 goes **1.4e-7/7.4e-3 (χ=8/16) → 3.8e-13/2.7e-13** — flat in χ, a genuine
   fixed point; deep-modes/random cases bit-identical. Cost: the observable absorbs the cut weight
   (1.4e-14 → 1.7e-11 ≈ |λ₅|/|λ₁| on the over-param case) — the old "cycle-null modes carry vertex
   weight" lesson, now a controlled trade instead of an uncontrolled wander.
2. **The under-truncation limit cycle is a forced cluster split.** At χ=8 the cut lands exactly on
   a degenerate pair — a "dominant invariant subspace" that must split a cluster is ill-defined and
   re-resolves differently each sweep, so the map has no stable fixed point. `degtol` now applies to
   the cycle cut with the same back-off semantics as `:cut` (retiring open problem #4's silent
   no-op), though the fundamental fix remains χ.
3. **The pipeline is exact when fed a spectrally clean subspace** — the deep-modes case biorth
   overlap is conditioned at exactly 1.0.
4. **Random states at tight χ fail on the CRITERION, not the solver, and nothing tunes it away
   (measured on `ctm_test.jl`'s gauged 4×4 D=2, seed 1234).** The cycle spectrum of a random state
   is quasi-degenerate exactly where a small χ forces the cut (χ=4 lands inside a
   {4.6e-3, 4.6e-3, 4.4e-3} cluster; χ=2 splits a 1.2-ratio pair), so there is NO good rank-χ
   invariant subspace: at χ=2 it limit-cycles, at χ=4 it converges to a GENUINE stationary point
   (worst-region 1e-13) that is still 18× worse than `:cut` on lnZ. Ruled out by direct A/B:
   the gauge (cycle spectra and errors bit-identical gauged vs ungauged — the cycle spectrum is
   gauge-invariant by similarity; meanwhile the SAME gauge improves `:cut`'s lnZ 30×, because the
   BP gauge is precisely the basis where per-bipartition truncation is optimal) and `cycle_gapcut`
   (inert). `degtol` back-off DOES help — but only after a wiring fix (2026-08-21, same day): the
   first cycle-degtol implementation compared against a magnitude list truncated AT the cut, so the
   guard could never fire and the initial "inert" measurement was vacuous. Fixed (compare against
   the full returned Ritz list), `degtol = 0.3` takes the seed-1234 χ=2 lnZ from 5.2e-2 to 2.9e-3
   (18×) and the observable from 3.6e-3 to 4.1e-4 — within 8× of `:cut` on lnZ while 30× better on
   the observable. It is NOT a default: the value that helps χ=2 slightly worsens the χ=4
   observable on the same state — no single value is right even within one example, because the
   ill-posed cut has no right answer, only different trades. The DEFAULT gained a free structural
   guard from the fix: at `degtol = 0` the `≤` fires on EXACT magnitude ties, i.e. a (λ, conj λ)
   swap pair straddling the cut is never split. Measured on the one known pair-straddling case
   (random 5×5 χ=8, `|λ_8| = |λ_9|`): oscillation 1.9 → 0.78, obs 1.2e-3 → 6.9e-4, strictly better;
   flagship 5×5 Ising and all other battery cases bit-identical; 183/183. The trace-extremal criterion ranks directions by plaquette-loop weight, which
   at shallow, structureless spectra does not match what the Möbius sum needs. The SAME case shows
   the flip side: `:cycle` beats both `:cut` and bMPS on the corner observable at χ=2 (3.6e-3
   against 1.2e-2 / 5.5e-3) while losing lnZ by 150× — stationarity buys single-region ratios, the
   optimal bipartition cut buys the global signed sum. **Guidance: for lnZ on random/unstructured
   states at tight χ, gauge + `:cut`; `:cycle`'s domain is observables and adequate χ.** Physical
   states have decaying, gapped cycle spectra, which is why they do not hit this.

Also derived by hand (recorded so nobody re-derives it): extremising the plaquette trace
`Z_p = tr(Π₄A₄Π₃A₃Π₂A₂Π₁A₁)` over rank-k oblique projectors forces `range(Π)`/`ker(Π)` to be
right/left invariant subspaces — the extremals ARE the spectral projectors of the cycle, dominant =
maximal `|Z_p|`. The construction is sound; every disease above is numerical realisation, not
formulation.

### FALSIFIED 2026-08-21: "the ket↔bra swap gives the left subspace for free" (one-solve `:cycle`)

The hypothesis was `Mᵀ = S·M·S` on ⟨ψ|ψ⟩ norm networks (`S` = ket↔bra swap on the bond), which
would make the backward `schursolve` redundant. Probed dense on a 2×2 lattice where all four cycle
bonds are raw (ket, bra) pairs and `S` is exactly constructible through the engine's own combiners.
**It fails at O(1)**: `‖Mᵀ − SMS‖/‖M‖` = 0.94 on a complex random state, 0.41 real random (the
7e-4 on a TFIM-evolved state is only because that cycle is near-identity). The identity that DOES
hold — to machine precision on every case, per corner and for the cycle — is

```
conj(M) = S · M · S        (from the CP structure  E = Σ_p A_p ⊗ conj(A_p))
```

with three verified corollaries: (1) **real norm networks: `[M, S] = 0`** — the cycle commutes with
the swap, so both solves could in principle be restricted to a swap-parity sector (~2× smaller,
and it would remove cross-sector degeneracies), at the cost of maintaining swap-covariant kept
indices recursively; (2) the retained `:LM` invariant subspace is exactly closed under
`v ↦ S·conj(v)` (worst principal angle cos = 1.000000 on all cases) — the structural origin of the
systematic double-layer degeneracies `degtol` guards; (3) the same relation holds for `Mᵀ`, so the
left subspace is equally structured — but left and right stay two independent problems. Do not
re-attempt one-solve-via-swap; if a one-sided eig is ever justified on physical states it must rest
on something else (approximate Hermiticity of the environment, not this symmetry).

---

## 2026-09-09: the QR skip is dense-only

Change 3 below ("the QR of a block only pays when it shrinks it") was exact on dense data but broke
graded `:cut`: on the fermionic 3×3 D=3 state, lossless χ=16 read `⟨N⟩` 4.2e-8 off (1.6e-19 the day
before), independent of `gauge`, `svd` route, sweep count and `qr_cutoff`, while the bosonic Z2 twin
stayed exact. Cause: without the triangular factors the whitening contracts `dag(V)`/`dag(U)` over the
block's OWN legs, which carry mixed orientations, and on fermionic tensors that dag-then-contract picks
up parity twists (the same mechanism as the indefinite Gram matrix of 2026-09-08). Through the QR the
contraction runs over one fresh bond of a single orientation. Fix: skip the QR only on dense blocks
(`_ctm_twosided_projector_qr`). Fermionic χ=16 is exact again (5e-20); χ=4 back to 1.87e-9.
Rule of thumb for graded/fermionic code paths: a dag'd factor may be contracted over a fresh
factorization bond, never over a block's own mixed-orientation legs; use `gram` for inner products.

## 2026-09-08 (night): `:cut` at boundary-MPS cost, and two certification fixes

Starting point, measured on a 9×9 D=3 TFIM PEPS (imaginary-time simple update at g = 3.04438,
`BLAS` single-threaded), `update` at the default tolerance 1e-10:

| χ | `:cut` before | bMPS | `:cut` after (`svd = :auto`) | `:cut` after (`svd = :dense`) |
|---|---|---|---|---|
| 32 | 21.7 s | 14.7 s | **16 s** | — |
| 48 | **508 s** (ran to `maxiter`) | 5.6 s | **31 s** (7 sweeps) | 39 s |

Four changes, in order of impact.

### 1. `_ctm_align` rejected every lossless interface — `update` never certified

The Procrustes guard compared `P_B P_A` before and after the rotation. That product is `𝟙 + E`
with `E` the pair's own biorthogonality defect, `‖E‖ ~ eps·σ₁/σ_k`, which reaches ~1e-3 at the
default `qr_cutoff = 1e-13` whenever σ_k sits at the cutoff — i.e. on every lossless interface —
and `‖R†(𝟙+E)R − (𝟙+E)‖ ~ ‖E‖` for a perfectly unitary `R`. So the deepest interfaces were never
aligned, everything above them re-indexed, `_ctm_statedist` had no distance, and `update` ran to
`maxiter` (90 s for a 5 s solve on 6×6 χ=48; 508 s on 9×9 χ=48). A unitary preserves the norm of
each factor, so that is what the guard tests now; the `replaceinds` sector-structure check stays.

### 2. Index bootstrapping: one level per sweep → all levels from sweep 2

The greedy seed stored `(P, w)` pairs, so the first sweep could not align to it and minted fresh
indices everywhere; a level can only align once the level below it kept its index in the previous
sweep, so the bases stabilised one lattice level per sweep — 9 sweeps on 9×9 while `|ΔF|` was at
1e-14 from sweep 2. Now the seed stores `(P, dag(P), w)` triples (also the first sweep's warm
start), and a level that cannot keep its index records the transition maps `P_B·P_A⁰`, `P_B⁰·P_A`
in its triple so the level above transports its previous projector onto the new basis
(`_ctm_transport`, `_ctm_remint`; both passes walk each chain edge-inward, `_ctm_each_interface`).
Measured 9×9: 33 rank mismatches after the seed, **256/256 aligned from sweep 2**, certification
at sweep 7 with a genuine ~17× per-sweep contraction of the criterion.

### 3. The `:cut` projector without the `O(χ³D⁶)` SVD

Per interface the dense route did two `n×n` QRs and a full `n×n` SVD, `n = χ·D²`, to keep `χ`
triplets — 91% of a 9×9 χ=48 sweep. Two things:

* The QR of a block only pays when it shrinks it (`rest > ins`); every bulk interface has
  `rest = ins`, so it is skipped and the overlap formed directly. Exact, 1.3× on the sweep.
* `svd = :auto` (default): a matrix-free block subspace iteration on the enlarged corners
  themselves, warm-started from the previous sweep's projector (`_ctm_subspace_svd`), cost
  `O(K·n²·χ) = O(χ³D⁴)` — the boundary-MPS scaling — with a convergence residual restricted to the
  retained directions, a Ritz/observed-rate extrapolation that **bails out to the dense route** on
  flat spectra (random D=3 states: every attempt bails, the memo `cache.route` backs off, overhead
  5%), and a per-interface memo. Reproduces the dense fixed point to 1e-14 in every check (5×5 D=2
  χ=4/8/16, complex 4×4 χ=16/64, Z2 3×3). On the physical 9×9 all 80 eligible interfaces take it
  at ~1 iteration each once warm (`CTM_SVD_STATS`); the other 176 are too small for it to pay
  (`_ctm_use_subspace`) and stay dense. Sweep 4.9 s → 3.3 s. Note `n/χ = D²` is fixed on a double
  layer, so the gain is the warm start, not an asymptotic one: ~3× per projector, no more.

### 4. `:cycle` certification: `convergence = :marginal`

On a near-lossless state `:worst_region` floors (6×6 D=3 TFIM χ=32: worst region stuck at 1.5e-7
while `|ΔF| ~ 1e-15` and `⟨X⟩` agreed with `:cut` to 4e-14), so `expect(...; projector = :cycle)`
ran to `maxiter` at 12× the converged cost. `:marginal` watches each vertex's normalised reduced
density matrix from its own ring (`_ctm_vertex_marginals`) — what an observable actually reads,
gauge invariant, full coverage — and certifies once the observable is stationary. `expect`/`rdm`
now inject it for `:cycle`. Measured, `:cycle`:

| state | χ | `:free_energy` | `:worst_region` | **`:marginal`** | `⟨X⟩` vs `:cut` |
|---|---|---|---|---|---|
| 6×6 D=3 | 16 | 1.1 s | 3.6 s | 3.4 s | 1.6e-10 |
| 6×6 D=3 | 32 | 3.2 s | >17 s, not converged | 6.1 s | 3.6e-14 |
| 9×9 D=3 | 32 | 14 s, `⟨X⟩` 3e-9 off | — | 52 s | 1.7e-12 |

`:free_energy` alone stops before the observable settles (the documented lag): 3e-9 on 9×9.

Where the remaining `:cut` time goes (9×9 χ=48, warm sweep 3.1 s): projectors 1.65 s (80
subspace + 176 small dense), enlarged corners 0.73 s, corner/edge rebuilds 0.68 s, region terms
0.18 s; ~35% of the sweep is `permutedims` inside TensorOperations. Seven sweeps to 1e-10 is the
outer fixed-point rate, not bootstrapping any more. bMPS at χ=48 is 5.6 s because its fitting
converges in a couple of iterations on this near-lossless state; at χ=32 the two are equal.

Also fixed: `_ctm_marginal_distance` and `_ctm_statedist`-style direction distances computed as
`√(2 − 2|⟨a,b⟩|)` cannot resolve below ~1e-8 (cancellation at machine precision); the marginal
signal uses the phase-corrected difference `‖a − e^{iφ}b‖` instead.

Not addressed: the untracked `examples/ctm_ising9x9_energy.jl` and
`examples/collect_mike_tfim_sampled_single_site_svd.jl` include files that are not in the repo
and pass `gauge_state`/`convergence = :environment`, which the current constructors reject.

---

## Open problems

Ranked by how much they should worry you.

### 1. `:cycle` under-truncation limit cycle — DIAGNOSED 2026-08-19 (was "does not converge on 8×8")

**Resolution: it is a truncation-insufficiency signal, not a bug, and the convergence check now flags
it** (the `|ΔF|` default fails to certify because `F` genuinely oscillates on the cycle; the
worst-region signal — the observable-path default — makes the O(1) floor explicit). Reframed
2026-08-19 with the gauge-invariant worst-region `|Δ lnZ_r|`:

* **Truncation-induced, sharp χ threshold.** Same 6×6 D=2 state, worst-region floor by χ: χ=2→3.5,
  χ=4→1.0, χ=8→1.7, **χ=16→2.2e-8, χ=32→1.6e-8**. Below the lossless χ it limit-cycles at O(1); at/above
  it converges to ~1e-8. 8×8 needs χ>16 to be lossless, so 8×8 at χ=16 is *under-truncated* and cycles,
  while 6×6 at χ=16 is lossless and converges. Amplitude ∝ degree of under-truncation (8×8 D=2 χ=16
  observable oscillation ≈ 4e-5, nearly fine; severely under-truncated 6×6 χ=8 seed7 ≈ 6.6e-2).
* **A wandering instability, not a periodic orbit.** The worst region hops around the lattice each
  sweep; the observable oscillation *straddles* the truncated value (some sweeps closer to exact, some
  farther), so there is no clean sub-oscillation fixed point.
* **Not fixable by solver tricks** (each falsified by a cheap test, 2026-08-19): a continuous warm-started
  block seed — the tractable core of the collaborator's `V_from_Ac` — does NOT break it (both cold and
  warm oscillate O(1)); a better *seed* cannot stabilize a *map* with no stable fixed point at that χ.
  Cycle-averaging is unreliable (in every case with an exact reference the best single sweep beat the
  average; severe seeds never centred), and Anderson's premise (a clean fixed point to converge to)
  therefore does not robustly hold. **The fix is χ**; worst-region tells you when χ is insufficient
  (O(1) = raise χ, →0 = converged).

The detailed record below (the plateau tables and the earlier falsified mechanisms) is kept as the
pre-diagnosis history — it is how the χ-threshold picture was reached.

---

Square 8×8 D=2 at χ=16, `:cycle`, sweeps 20–30:

```
it   stateDist   F             cornerErr  interiorErr
20   1.569       118.0609105   -5.5       -5.1
23   1.518       118.0619714   -3.9       -5.1
24   1.477       118.0607724   -5.8       -2.6
30   1.859       118.0609187   -5.4       -4.8
```

(That trace is a `random_tensornetworkstate`, i.e. a DOUBLE-LAYER PEPS norm.) `:cut` converges in 13
sweeps. This violates the requirement that both formulations be convergent algorithms.

**It is a PLATEAU, not slow convergence, and not a size threshold.** Measured 2026-08-09 across
network types at 8×8, χ=16, single layer, state distance by sweep out to 40:

```
                       sweep 10    20      30      40      verdict
8×8 random SIGNED       8.7e-10  4.8e-10 1.3e-09 8.2e-10  plateau BELOW tolerance -> "converges" (9 sweeps)
8×8 random POSITIVE     3.0e-01  7.0e-01 7.1e-01 5.3e-01  plateau
8×8 Ising β=0.44        9.5e-04  7.2e-04 8.4e-04 8.4e-04  plateau
8×8 Ising β=0.20        2.5e-03  7.9e-04 1.3e-03 1.1e-03  plateau
6×6 (all three types)                                     converges, 6 sweeps
```

Every case plateaus at a nonzero residual; none decreases, so more sweeps never help. The only
difference between "converges" and "doesn't" is where the plateau sits relative to
`√(tolerance·max(1,|F|))`. ⚠️ **This kills the hypothesis that it is a random-signed-state pathology**
(the sister of the D=3 weakness): random signed is the case that *works*, and the failures are the
POSITIVE networks, including the physical Ising at two temperatures.

**`F` is unaffected throughout** — `|F − ln Z|` is 2e-14 even where the state distance is 0.5. A
genuine 50% change in the environments could not leave `F` at machine precision, so the wander is in a
direction `F` barely sees: the retained subspace is still invariant, just rotated, and the Möbius sum
cancels ~4000× of what is left. A single-region observable has no such cancellation, which is why it
moves at 1e-4. **So `:cycle` at 8×8 is sound for free energies and unreliable for observables.**

`degtol` cannot be the fix as it stands (see 4).

#### TRIED AND FALSIFIED: pivoted QR in `_ctm_orthcols`

The 2026-08-09 audit found that `_ctm_orthcols` uses an **unpivoted** `qr` with no rank check, so for
a rank-deficient `X` it returns a full-width `Q` whose surplus columns lie **entirely outside**
`range(X)` (a 10×3 matrix of rank 2 gives a third column with `‖(I−P)q₃‖ = 1.0000`). Its inputs are
rank-deficient exactly in the near-degenerate regime, and the shortfall guard checks only the width,
so nothing declines. That is the right *shape* of mechanism for a limit cycle, and it was tested:
`qr(X, ColumnNorm())` plus truncation to the numerical rank, harmonised across the four bonds.

**It does not fix the 8×8** — still 30 sweeps, final state distance 1.44 against 1.86. And it is not
worth landing on its own merits (8 seeds, interior sites):

| case | χ | unpivoted | pivoted | wins |
|---|---|---|---|---|
| square 4×4 D=3 | 8 | 0.289 | **0.268** | 3/8 → **1/8** |
| square 4×4 D=3 | 12 | 2.219 | 2.219 | 6/8 → 6/8 |
| square 5×5 D=3 | 8 | 0.019 | 0.019 | 0/8 → 0/8 |
| square 4×4 D=2 | 8 | 7.522 | 7.522 | 7/8 → 7/8 |

Bit-identical on three of four — the deficiency essentially never fires — and slightly **worse** where
it does. That reproduces the sign of an already-falsified family: truncating to the cycle's numerical
rank was tried before and was much worse, because *the extra directions are not junk, they carry
region weight the loop's spectrum knows nothing about*. The 5×5 control was unaffected (χ=32 `⟨X⟩`
4.774e-14 against 4.818e-14). Reverted; the latent risk is documented at `_ctm_orthcols` instead.

⚠️ A 4-seed version of this measurement showed the D=3 case improving 6.4× (0.16 → 1.03). That was an
artefact of picking the observable site as `vertices(g)[n÷2]`, which is a BOUNDARY vertex. At an
interior site over 8 seeds the effect reverses sign. Same trap as every other retraction in this file.

#### TRIED AND FALSIFIED: under-relaxation

Mixing the new environment with the old (`env ← α·new + (1−α)·old` on the C/T blocks, new projectors
kept). 8×8 Ising β=0.44, χ=16, `:cycle`, state distance by sweep:

| α | sd@10 | sd@20 | sd@30 | sd@40 | \|F−lnZ\| |
|---|---|---|---|---|---|
| 1.0 | 9.5e-04 | 7.2e-04 | 8.4e-04 | 8.4e-04 | 2.1e-14 |
| 0.7 | 2.2e-04 | 3.6e-04 | 4.7e-04 | 2.5e-04 | 7.1e-15 |
| 0.5 | 5.1e-04 | 8.4e-04 | 4.5e-04 | 6.0e-04 | 7.1e-15 |
| 0.3 | 1.3e-04 | 1.5e-04 | 2.2e-04 | 2.0e-04 | 7.1e-15 |
| 0.1 | 7.5e-05 | 5.6e-05 | 5.4e-05 | 3.8e-05 | 2.1e-14 |

Every α plateaus — but the plateau **scales with α** (22× down for a 10× smaller mixing; noisy in the
middle, clear at the endpoints). **This rules out an unstable feedback loop**, which damping would
have converged. A residual proportional to the step size is the signature of a FIXED-SIZE
PERTURBATION INJECTED EVERY SWEEP, which damping attenuates but never eliminates.

#### The remaining hypothesis: the Krylov solve is cold-started

`_ctm_cycle_projectors` seeds `schursolve` from a vector hashed on plaquette POSITION only — identical
every sweep, by design, because that is what made the engine deterministic. But the cycle matrix
drifts from sweep to sweep, so a cold restart from a fixed vector can return a different basis for a
near-degenerate invariant subspace each time: a constant-amplitude kick, exactly what the α-scaling
shows. It also explains the otherwise puzzling pair of facts — `F` stays exact to 7e-15 (the subspace
is still invariant, merely rotated, and the Möbius sum cancels the rest) while a single-region
observable moves at 1e-4 because it reads one region with no cancellation.

**Tested 2026-08-09 and FALSIFIED — in one specific form.** Warm-starting `schursolve` from the
previous sweep's projector on the same bond (`P_A`'s leading column, plus a 1e-3 hashed admixture to
avoid landing exactly inside an invariant subspace and closing the Krylov space early), falling back
to the hash on the first sweep and on any index mismatch:

| case | cold | warm |
|---|---|---|
| 8×8 Ising β=0.44 | 8.4e-04 | **4.6e-02** (55× worse) |
| 8×8 random positive | 5.3e-01 | 6.2e-01 |

`|F − ln Z|` unchanged at 2.1e-14 either way. Reverted.

⚠️ **Scope this correctly.** The vector used was column 1 of `P_A`, which is the WHITENED projector
`A·V·S^{-1/2}` — its columns are deliberately not orthonormal and the inverse square root can scale
them badly, so it is a poor representative of the previous subspace. What is falsified is
"warm-start from the stored projector", NOT "warm-starting cannot work". The principled version starts
from the previous sweep's orthonormal Schur basis `VR[1]`, which is **not stored anywhere** and would
need a new field on `CTMVertexEnvironments`. That remains untested and is the natural next attempt.

#### How the collaborator's engine differs — and what that rules in

Their code (`examples/python_ctmrg`) does **not** solve the cycle from a random vector. Every sweep it
builds a **χ-wide block basis in closed form from the current corners**, `V_from_Ac`, by solving the
defining cyclic relations with a padded pseudo-inverse:

```
VL[j] = c_{j-1,int}⁻¹ · A_{j-1} · c_{j-1,ext}
VR[j] = c_{j,ext} · A_{j+1} · c_{j,int}⁻¹
```

then refines it with `krylov_eig_one_sided`, patching only rank-deficient columns stochastically and
only for the first two sweeps (`V_guess_stochastic_num_iter = 2`). The start is therefore a pure
function of the CURRENT state — history-free, deterministic, and continuous as the state drifts.

Two corrections to earlier readings of their code. (1) They **compose the four-corner product too**
(`for k in 3:-1:0: V = C[k] @ V`), so composition is not the distinguishing feature; the periodic
Krylov-Schur in `linalg/periodic_krylov_schur.py` is a SELECTABLE method, not their default
(`DEFAULT_PROJECTOR_METHOD = "eig one sided"`). (2) Their convergence criterion is `dVL`, the change
in the projector bases themselves, not a state distance over environment blocks.

#### TRIED AND FALSIFIED: a state-derived SINGLE start vector

The cheap analogue of the above: the power iteration that computes the scale factor already produces
a state-derived direction, and it was being thrown away while `schursolve` got the raw hashed random
vector. Feeding the power-iterated direction in instead (8×8, χ=16, plateau = min state distance over
the last 10 of 30 sweeps):

| start | Ising β=0.44 | positive | `|F−lnZ|` positive |
|---|---|---|---|
| cold (current) | 6.10e-04 | 1.55e-01 | 4.97e-14 |
| pure power | 1.25e-02 | 2.48e-01 | **4.43e-04** |
| mix, 1.0 random | 9.32e-04 | 2.64e-01 | 7.11e-15 |
| mix, 0.1 random | 2.63e-04 | 3.26e-01 | 7.11e-15 |

Nothing clears the plateau; the best is 2.3× on one case and worse on the other. **The pure-power row
is the useful result**: `|F − ln Z|` degrades by TEN ORDERS. A start already converged onto the
dominant eigenvector closes the Krylov space immediately, collapsing `kres` and losing the projector
silently. That is direct evidence the **block width is load-bearing**, not an implementation detail —
a single vector cannot stand in for a χ-wide basis however well it is chosen. Reverted.

#### TRIED: periodic Schur via `PeriodicSchurDecompositions.jl` — blocked by a REPRESENTATION mismatch

Julia already has the tool their 2000-line `periodic_krylov_schur.py` implements:
`PeriodicSchurDecompositions.partial_pschur(As, nev, which; u1, maxdim, vrand!, ...)` returns a
partial periodic Schur of the product `Aₚ⋯A₁` **without forming it**, giving an orthonormal basis at
EVERY site. That would replace both the composed-product `schursolve` and the propagation loop with
`_ctm_orthcols` in one step. ⚠️ Note the package's own default `which = LM()` references a name it
does not import — pass `ArnoldiMethod.LM()` explicitly.

Validated first on synthetic factors: the site convention is `As[l]·span(Z[l]) ⊆ span(Z[l+1])`
(residual 1e-15), matching our `As[l] : io[l] → io[l+1]` exactly, and on CTM-like strongly-decaying
spectra the cycle invariance at site 1 is **3.7e-17**. (Comparing against `eigen` of the *formed*
product is meaningless in that regime — the product spans ~1e-21 — which is the whole argument for
periodic Schur.)

**It does not fit our data structure.** `partial_pschur` requires four SQUARE factors of equal size.
Measured over 48 plaquettes of a 5×5 D=2 at χ=8:

```
raw interface dims per plaquette: [4,32,32,4]  [4,32,32,16]  [4,16,32,32]  [16,32,32,16] ...
square: 0 of 48        padded-and-failed: 48 of 48        999x "NaN in ujl"
```

**No plaquette is square.** Zero-padding a rank-4 block into a 32×32 factor makes it severely
singular and the periodic Arnoldi breaks down. Every plaquette then declined and the run silently
became pure `:cut` — which "converged" and made the 8×8 plateau look 8 orders better (6.10e-04 →
7.83e-12) while the 5×5 control returned *exactly* the `:cut` value, 4.657e-04. **That is the
canonical false positive for this project: a fix that converges by ceasing to be `:cycle`.** Only the
paired control caught it.

Root cause is architectural, not a bug: their engine stores every corner at uniform `(χ,χ)` with an
explicit `rank` field, so square periodic Schur is natural for them; our raw interfaces are genuinely
rectangular. Two ways forward, both real work: (a) pre-compress each bond to a common `kcyc` square
block and run periodic Schur on that — the cycle is already capped at `kcyc = min(χ, narrowest bond)`
so nothing is lost in principle, but the pre-compression must not smuggle the cut back in; or
(b) adopt uniform-width storage plus a rank field throughout the cache.

#### Where this leaves the 8×8 problem

Four candidate causes falsified: rank-blind QR in `_ctm_orthcols`, an under-damped iteration, a warm
start from the stored projector, and a state-derived single start vector. What is established:

* it is a PLATEAU, not slow convergence — flat to sweep 40, so `maxiter` cannot help;
* it is not a size threshold and not a random-state artefact — 6×6 converges on all three network
  types, 8×8 fails on the POSITIVE ones (including physical Ising) and succeeds on random signed;
* the residual scales with the damping factor, so it is a per-sweep injected perturbation rather than
  an unstable feedback loop;
* `F` is unaffected at 2e-14 throughout, so the perturbation lies almost entirely in directions the
  Möbius sum cancels — the retained subspace stays invariant and rotates.

The last two together still point at the projector derivation being discontinuous in the sweep.

**The one untested lever left is the real thing: a χ-wide BLOCK start basis built in closed form from
the current corners**, i.e. a port of `V_from_Ac`. Every cheap single-vector surrogate for it has now
failed, and the pure-power collapse shows why. This is a substantial change, not a tweak:
`KrylovKit.schursolve` takes a single start vector, so it needs either block Arnoldi / subspace
iteration or a different solver, plus an analogue of their `A`/`c` split (our `As[l]` are the fused
corners, with no separate small-`c` factor to invert). Budget it as real work.

### 2. Random D=3 states

`:cycle` is ~3–25× behind `:cut` on random signed D=3 squares in the interior, improving with χ
(0.04 → 0.35 from χ=8 to 16). Not the convergence bug. Unexplained; the physical-vs-random contrast
above is the sharpest clue.

### 3. χ=4 is not stationary

`marg` 1.79e-04 against 2–3e-16 for χ ≥ 9, even though the χ=4 observable beats both `:cut` and their
engine. A fully resolved rank-4 invariant subspace is a worse *stationary point* than an
under-resolved one — the criterion straining at severe truncation. Restricting `krylovdim` fixes it at
χ=4 but costs accuracy at χ=8, so it is deliberately not applied; gating by χ would be the
tuned-threshold trap this project keeps relearning.

### 4. `degtol` is a silent no-op under `:cycle`

It is read only in the `:cut` singular-value truncations, never in `_ctm_cycle_projectors`. Measured
on 8×8 at degtol 0 / 1e-8 / 1e-4: identical to every digit. Either wire it in or say so in the option
table.

### 5. `marginal_inconsistency` is not uniformly at machine precision

hex 4×4 at χ=32 reads 1.9e-05 despite a machine-precision observable. Probably a degenerate
measurement; unverified.

---

### 6. `:cycle` on heavy-hex: a state-dependent, χ-independent observable floor — *found 2026-09-08*

On the random heavy-hex(2,2) D=2 state drawn as `Random.seed!(1)` on the `Tensors` backend
(`test_ctmenvironment.jl`, partial-coverage guard), `:cycle` at χ = 4/8/12/16 gives
`⟨Z⟩` errors 1.1e-8 / 2.9e-8 / 8.2e-11 / 8.2e-11, while `:cut` is exact (≤1e-15) at every χ.
The cache is fully stationary (`marginal_inconsistency` 5e-17) and 200 sweeps at
tolerance 1e-13 change nothing: the worst-region `|Δ lnZ|` sits on a 2.5e-9 limit cycle.
8 of 40 plaquettes decline (the usual heavy-hex geometry). The SAME tensors loaded into the
ITensors-backend build reproduce every value to 1e-15, so this is the engine, not roundoff or
the backend port. A second seed floors at 4.3e-10 at every χ; a third is exact. The test's
`:cycle` bound is 1e-7 for this reason (30× under the 3.7e-6 partial-coverage bug it guards).
Not diagnosed. Candidates: the declined-plaquette backfill (cut interfaces inside a cycle
lattice are not stationary with respect to each other), or a `cycle_gapcut` cut landing in a
real tail on this state.

## Backend port — *2026-09-08*

The engine now runs on the `Tensors` backend of the `Fixes` branch (ITensors is gone from the
package). The port was a rename: `ITensors.array` → `array`, `ITensor(A, i, j)` → `from_array`,
`ITensor` → `AbstractTensor`, plus two seam methods added for it — `array(t, inds...)`
(permuted dense view) and `prime(t, inds...)` (prime only the listed indices). Verified on
IDENTICAL tensors dumped from one backend and loaded into the other: `:cut` and `:cycle`, lnZ
and `⟨Z⟩`, heavy-hex and square, all agree to ~1e-15. Deterministic Ising cases in
`examples/ctm_environment.jl` reproduce the recorded floors (χ=8/16 at 4e-15..1e-14).
Random-state numbers in the examples and tests DIFFER from the ones recorded in this file
because the two backends draw random tensors from different streams — same statistics,
different states. Two seed-specific assertions were rewritten to be state-independent
(`window = 1` gain, cut-vs-cycle 2× win → median over seeds).

**`:cut` is now written in tensor verbs — no unwrapping (same day).** The two-sided projector is
`qr` of each enlarged corner over its non-interface legs, a bilinear contraction of the two
triangular factors, one `svd`, and a `map_diag` whitening; the greedy one-sided isometry is the
left singular vectors of the block (no `ρ = B B†` squaring); the Procrustes gauge alignment is
`svd(P_A† P_A⁰)`. The truncation rule (rank ≤ χ, `qr_cutoff` relative cutoff, `degtol`
back-off) is a MatrixAlgebraKit truncation strategy (`truncation_strategy` on the seam,
`RankGapTruncation` in the backend) applied INSIDE the truncated SVD, with a sector-merged
implementation for graded spectra — so the identical code path is what will run on symmetric
tensors; only the `:cycle` path still touches raw matrices. Verified on the identical-state dump:
agreement with the ITensors reference to ~1e-12 (the one 1e-5-relative case is a truncated,
tolerance-limited sweep). RETIRED with the matrix code: the `arnoldi`/`krylov_min` options and
the dense top-k Krylov SVD (`_ctm_svd_topk`, 6–12× on n ≥ 288 blocks); the "Performance" notes
on `krylov_min` below are historical. Bring it back as a dense-backend hook if the large-χ
benchmark shows the loss.

**`:cut` RUNS ON SYMMETRIC TENSORS (same day).** A Z2-conserving 4×4 state (product state + conserving
circuit) and its dense twin give the SAME truncated CTMRG numbers: |ΔF| 4e-15 / 2e-14 and |Δ⟨Z⟩|
1e-14 at D = 4, χ = 4 / 8; at D = 2 both are exact at lossless χ = 16 (F to 4e-15, ⟨Z⟩ to 2e-15).
U(1): identical to dense at D = 2 (χ = 4 truncated, χ = 16 exact); at D = 4 the two runs follow
DIFFERENT truncation paths (|ΔF| 8e-4 at χ = 4) because the double-layer corners carry exact
cross-sector degeneracies (measured relative gaps 1e-15..1e-16, the ket↔bra pairs) that dense and
sector-merged sorting break differently — both are valid CTMRG fixed points, and `degtol` is the
knob that makes the two agree. Three things had to change in the seed pass for graded arrows: the
one-sided isometry is `dag(U)` of `svd(B, ins)` (the RIGHT singular basis, which is also the
arrow-compatible copy — the old `P = conj(V)` convention was only correct for real data), the
block that derives `P` absorbs `P` and every block at the OTHER end of the interface absorbs
`dag(P)` (the row strips absorbed `P` at both ends), and `combiner` now dispatches on index
content like `delta`. Test: "CVM on graded (Z2-symmetric) tensors". `:cycle` on graded tensors
remains TO DO (per-sector Krylov via charged start vectors; it still goes through `array`).

## Backend port, part 2 — *2026-09-08 (afternoon)*

**`:cycle` is in tensor form too.** The Krylov vectors are tensors on the west legs with a dim-1
"charge" leg (trivial on dense data, one per sector on graded data — a symmetric tensor on the bond
legs alone lives in the charge-zero sector, and the cycle map conserves charge, so the graded cycle
problem is a direct sum of per-sector problems). One `schursolve` per sector and side, sector spectra
merged before every rank decision, Ritz vectors stacked into a basis by direct sum along the charge
legs (`directsum(p1, p2)` two-pair form, new on the seam; the retained bond gets one count per sector
for free), propagation by `qr`, the biorthogonal whitening shared with `:cut` (`_ctm_biorth`). Nothing
is unwrapped. New seam verbs: `charge_sectors`, two-pair `directsum`, seeded `random_tensor`, `gram`;
VectorInterface on `GradedTensor`. Retired: `cycle_subspace`/`cycle_iters` (block subspace iteration
needs a non-Hermitian eigen the seam does not have). Validated: identical dense states agree with the
matrix version to ≤1e-11 (`sq`/`hh` battery), two-projector testset 18/18, Z2-graded D=2 4×4 matches
dense to all printed digits at χ=4 and χ=16 (the χ=16 D=2 `:cycle` floor of 4.3e-9 is pre-existing:
same number from the matrix version).

**OPEN: graded `:cycle` at D=4 disagrees with dense** (Z2 4×4 D=4, χ=4: |F−lnN| 3.0e-2 graded vs
5.9e-4 dense; χ=8: 1.4e-3 vs 4.7e-4; neither certifies). The per-plaquette spectra are identical to six
digits on both backends, so the sector solves are right; the difference is downstream. Diagnosed so
far: (a) with `degtol = 1e-8` graded χ=8 read F off by 3.8 until the alignment guard below — the
zero-padding lands its columns in whatever sectors `new_index` allots, so the padded bond's sector
structure differs from the previous sweep's and a Procrustes rotation between them is not unitary;
now caught (alignment declined) but not cured; (b) forcing the left side to the right side's
per-sector counts changed nothing; (c) the cross-sector exact ties (ket↔bra pairs) at the cut back
off differently on the two backends (kres 3 vs 4 on one plaquette). Candidates left: pad per sector
of the RETAINED bond (so padding never changes sector structure), or drop padding for graded bonds
and let `_ctm_align` compare spaces. Not a port-blocker for `:cut`.

**Fermionic (fZ2) `:cut` — one genuine bug found and fixed, one non-bug.** (1) `_ctm_align` built its
Gram matrix as `dag(P_A) * P_A⁰`. On fermionic tensors that is the BILINEAR pairing (correct for
inserting projector pairs into the network — ground-truthed: the full-rank pair changes ⟨ψ|ψ⟩ by
1e-9), not the Hilbert inner product: the Gram came out Hermitian but indefinite (trace −1.0 against
‖P_A‖² = 5.9), Procrustes rotated onto a sign-twisted target, and every sweep's blocks came back as
±themselves — `|ΔF|` at 1e-15, state distance 2.0, never certified, 30 sweeps every solve. Fixed with
the `gram(a, b, legs)` seam verb (TensorKit adjoint composition; `dag(a) * b` on dense) and a
phase-immune state distance (`2 − 2|⟨ta,tb⟩|` on norm-1 blocks; equals the old one where no phase
flips occur). Fermionic 3×3 D=3 χ=16 now certifies in 0.56 s; χ=4 still stops at maxiter with F
stationary to 3e-9 and a 1.4% residual — the 9→4 heavy-truncation limit-cycle regime, not a
convention error. (2) `P_B P_A` viewed as a dense matrix through `array` looks like diag(±1) on
fermions; that view is convention-laden, the network insertion is the test that counts.

**Fermionic `:cycle` — a parity twist in the propagated bases, found and fixed (2026-09-09).** Symptom:
on a 3×3 fU1/fZ2 CDW quench (D=4, 50 second-order Trotter steps), `:cycle` read `⟨N⟩` 0.5936 at
χ=1, 4 AND the lossless χ=16 while `:cut` at χ=16 and boundary MPS agreed on 0.48705 — i.e. `:cycle`
sat at the BP value (0.5941) regardless of χ, with every pair full rank and no plaquette declined.
It was NOT convergence (an exact fixed point after two sweeps, |ΔF| = 0), NOT rank collapse, NOT
geometry. The plaquette insertion identity `Z(with all four pairs) = Z` was the discriminating
test: 1e-15 for `:cut` and for `:cycle` on dense tensors, **9.65% off** for `:cycle` on either
fermionic grading. Isolating the propagation step by step: bases propagated FORWARD
(`V_R[l+1] ∝ A_l V_R[l]`) through `_ctm_orthbasis` (the `Q` of a QR, whose fresh bond is DUAL in
the graded backend) twist; the same `Q` on the backward-propagated `V_L` is exact; raw
(unorthonormalised) propagation is exact on both sides. Swapping to a non-dual-bond basis (`V` of
an SVD, or `R` of a QR) on the right side alone restores 1e-15 on every interface; swapping the
left side too breaks it again. So the fresh bond's orientation has to match the propagation
direction — the same handedness `_delta_pair` documents for a bare identity. Fix:
`_ctm_orthbasis_fwd` (SVD `V`) for `V_R`, `_ctm_orthbasis` unchanged for `V_L`; dense tensors
are unaffected. Regression: the 2×2 fZ2/fU1 CDW case with `projector = :cycle` next to the `:cut`
one. Lesson, again: `P_B P_A = 𝟙` is necessary and useless as a check on fermions — only the network
insertion counts.

Two things found in the same session, neither fermion-specific:

* **kres = 1 always declined** (`:cycle` at χ = 1 declined 4 of 4 on the fermionic 3×3, dense would
  too). `_ctm_stack` returned a lone Krylov vector unchanged, with its dim-1 charge leg as the bond;
  the left start vector carries `dag` of that same leg, so left and right bases shared an index id
  and `_ctm_biorth`'s overlap contracted it along with the interface, down to a scalar. Fixed: a
  single vector gets a fresh `Link,cyc` bond like the direct sum mints for two or more.
* **A single-vector Krylov solve cannot resolve a DEGENERATE eigenvalue — and double-layer cycles
  are full of them.** Symptom: on the 2×2 fZ2/fU1 CDW test state at the lossless χ = 16, `:cycle`
  resolved 13 (fZ2) / 10 (fU1) of 16 directions, the plaquette insertion identity was off by
  7.6e-4 / 4.0e-4 (against 1e-15 for `:cut`), and the observable by 3e-4 / 9e-4. Instrumenting
  per sector: the fZ2 even sector (dim 8) "closed" at 5 of 8 with the three missing directions being
  partners of a 3-fold-degenerate eigenvalue (1.1e-5) whose weight, 3.4e-5, is exactly the deficit
  in Z. A Krylov space grown from one start vector holds exactly one vector per DISTINCT eigenvalue
  (the projections of the start onto the eigenspaces), so multiplicity is invisible to it and
  `schursolve` reports a closed invariant subspace. Ket↔bra exchange, lattice symmetries of the
  state and graded sector structure all produce exact multiplicities. This is almost certainly the
  "closes at ~19-22 of a requested 32" on the 5×5 benchmark that motivated the falsified fillers —
  the right filler is the degenerate partners. Fix: `_ctm_cycle_schur` restarts `schursolve` on the
  deflated operator `P M P`, `P = 1 − QQ†` over the Schur vectors found so far, from a FRESH random
  start (the closed Krylov space contains the vector it grew from, so deflating that one is exactly
  zero); the compression to the complement of an invariant subspace has exactly the remaining
  eigenvalues and the union is again invariant. Each restart recovers ONE more copy of a degenerate
  eigenvalue (the same one-per-distinct-eigenvalue limit applies to the complement), so a
  multiplicity-m eigenvalue costs m − 1 small restarts — measured: three restarts of one vector each
  on the 2×2, after which kres = 16 and the inserted plaquette reproduces Z to 1e-16. The deflated
  eigenvalues come back as ~0 and are removed by the existing noise-cliff and left/right guards.
  Same signature on a DENSE 4×4 D=2 state before the fix: insertion identity 1e-15, observable
  stuck at 2.4e-7 at the lossless χ = 16. ⚠️ The dense `array` of a fermionic tensor is NOT a faithful
  matrix of the categorical map — its `eigvals` did not match the Krylov spectrum (they came out as
  geometric means of the true clusters); do not use it to cross-check graded spectra.
* **`cycle_gapcut` default → 0 (2026-09-09).** The noise-cliff cut was the χ-independent 1e-10…1e-8
  observable floor listed as open below: a cycle eigenvalue is the fourth power of a corner's, so
  the √eps tininess floor lets the cut remove modes with ~1e-2 corner weight (dense 4×4 D=2, lossless
  χ = 16: plaquette (2,4) spectrum 1, 3e-3, 1e-4, 4e-9 — the last is steep and tiny, cut, observable
  2.4e-7 off while `marginal_inconsistency` reads 7e-17; without the cut 1.7e-15). A/B with the
  cut on/off over dense 4×4 χ=8/16, the suite's over-parametrised random 4×4 χ=32, random 5×5
  χ=8/16 and TFIM 5×5 nl=3 χ=8/16/32: off is never worse and better three times (2.4e-7 → 1.7e-15,
  2.3e-7 → 9e-9, 2.3e-4 → 1.5e-5); the over-parametrised case is bit-identical, i.e. the
  left/right consistency guard alone removes the surplus noise the cut was added for. The knob
  stays for a case the guard demonstrably misses. Lesson: stationarity metrics do not see this
  loss — `marg` was 7e-17 on the 2.4e-7 case — only a comparison against an exact observable does.
* **The `:cut` subspace route never converges on the fermionic states tried** (2×2 CDW, 3×3 CDW
  quench, weakly-hopped 3×3 CDW; χ = 2…8, forced with `svd = :subspace`): every interface bails to
  the dense route on the rate extrapolation, so results are identical to `:dense` by construction.
  Fermion safety of that route is therefore UNTESTED, not established. Low risk today because
  `:auto` never selects it on graded data.

**Geometry guard.** `CTMEnvironmentCache` now rejects any bond between non-adjacent grid positions
(periodic lattices, e.g. `named_hexagonal_lattice_graph(...; periodic = true)`): a wraparound bond
rides uncontracted inside every block and the sweep's contractions grow exponentially at any χ.
That was the "insanely slow at χ=1" on the hexagonal thermal-state example.

**Performance notes (2026-09-10 measurement).** On the fermionic (fU1) 3×3 quench, `:cycle` at χ = 1
took 384 s for a 30-sweep `update` in a fresh process and **1.2 s** for the identical call in the same
process with the contraction-sequence cache emptied (0.4 s warm). Steady-state sweeps are 0.08 s
(χ = 1) and 1.0 s (χ = 16). Netcon is NOT the cost: `ExhaustiveSearch` takes 0.1 ms on a 5-tensor
corner list and 1–2 ms on a 10-tensor ring, and delivers orders 1.4–3× cheaper in flops than
`GreedyMethod` (TreeSA matches it at 200–600 ms). The first-run cost is Julia method specialisation
per new contraction pattern on the graded backend, ~0.2–2 s per pattern, and the graded double
layer produces many patterns (99 sequence-cache entries for one 3×3 `update`). The remedy remains a
PrecompileTools workload covering a dense, a graded and a fermionic CTM `update`; nothing in the
sweep itself is worth optimising before that. Original notes follow.

**Performance notes.** JIT dominates fresh sessions: first dense CTM solve ~18 s, first graded ~65 s,
first fermionic ~88 s, graded `apply_gates` compile ~116–130 s; the solves themselves are 0.05–1 s.
Steady-state dense `:cut` is unchanged against the committed matrix version (0.38/0.27/0.86 s vs
0.38/0.37/0.67 s per sweep at χ=8/16/32, 5×5 D=3). A PrecompileTools workload covering a dense and
a graded update would remove most of the fresh-session latency. On a 3×3, boundary MPS at χ=48
(0.03 s) is ~20× cheaper than CTM at χ=16 (0.5 s) — per-vertex 4C+4T rings are more work than one
boundary MPS on a lattice that small.

## Backend switch: TensorKit → ITensorBase/GradedArrays — *2026-09-10*

`src/Tensors/ITensorBackend.jl` replaces `Tensors.jl` + `gradedtensor.jl` behind the same
`TensorInterface` seam; the generic CTM/BP/BMPS code is unchanged except where noted. Dense and
graded tensors are the same type (`ITensor` over `Array` or `AbelianGradedArray`), so the two
paths share every code line. Dense: all test files pass, CTM is faster than the TensorKit backend
(two-projector testset 25 s vs 65 s). Graded (Z2, U1, fZ2, fU1, fU1×U1): constructors, gates,
exact/BP/BMPS/CTM observables, fermionic chain/spinful/2D against dense Jordan–Wigner.

**What differs, and the traps found (all measured in a scratch env, then gated by the suite).**

1. *Two adjoints.* GradedArrays contracts in the supertrace convention: the identity inserted on a
   DUAL leg is the parity-twisted identity. Consequences: (a) `conj` is a homomorphism over
   contraction and is the right BRA for a network (`dag(t)`); the only correction is a twist on
   dangling dual "Charge" legs, without which the norm of an odd-parity product state is −1.
   (b) `conj` is NOT the adjoint of an isometry with mixed-orientation legs: `B·conj(U)·U ≠ B` for
   `U = svd(B, ins).U` when `ins` mixes arrows (every CTM interface: ket bond + bra bond). The
   projector adjoint is `conj` plus a twist on the dual legs of the codomain and the non-dual legs
   of the domain (the two sets agree up to the trivial twist of a flux-zero block) — new seam verb
   `dag(t, cod)`; `gram(a, b, legs) = dag(a, legs) * b`. No bond-arrow convention or pre-twisted
   factor can make `(conj(U), U)` a projector pair on a mixed interface (proved by exhaustion over
   the leg-class rules), so the CTM sites that mean the map adjoint now say so: one-sided
   `P = dag(U, ins)`, the greedy `dag(P_A, [w])`, whitening `dag(V, [v])`, `dag(U, [u])`,
   `dag(isk, [u])`, the subspace `applyadj` (`dag(Brow, rows)`, `dag(Acol, ins)`, `dag(B, rows)`),
   `dag(PBo, [wo])`, the alignment `dag(R, [wo])`. Ground truth: plaquette insertion identity,
   `:cut`/`:cycle` at lossless χ equal to exact contraction to 1e-15 on Z2, fZ2 and fU1.
2. *Combiner.* The graded fusion isometry is the identity on the fused space with its domain leg
   split by `unmatricize` (needs a `FusedGradedMatrix`); `t * C` fuses, `(t*C) * dag(C, [c])`
   splits. The lossless CTM branch returns `(co, dag(co, [c]), c)`.
3. *Boundary-MPS bra rail.* The one-site ALS is exact only with `fit_adjoint(m, crossing) = conj`
   plus a twist on the NON-dual crossing legs, never on the MPS bonds (exhaustive search against an
   exact MPS fit; direction-independent; the TensorKit backend's recipe). Restored on the seam.
4. *Link sectors.* `GA.dual` on a `SectorRange` sets an arrow flag and is not `==` to the base
   label, so the BMPS link allocation looked up `wr[dual(q)]`, found nothing, and fell back to a
   trivial-sector-only link: graded boundary MPS silently returned the BP value at every χ (Z2 and
   fermions alike). Fixed with a label-based `dual_sector`; Z2 4×4 D=2 BMPS now hits the exact norm
   to 1e-15 at χ ≥ 4.
5. *In-place scaling.* `rmul!(data(t), c)` is a no-op on block-sparse storage (`data` is a copy);
   graded gate application was not normalising tensors (Z2 state vs dense twin norms disagreed by
   e¹⁶, observables agreed). New seam verb `scale!(t, c)`.
6. *Diagonal maps* must touch the diagonal only (a `copyto!` into a `FusedGradedMatrix` hits
   forbidden blocks); `state`/`op` reject charged states/operators on graded sites (the old error).

7. *BP hot path.* The generic sequence path matricises both operands of every pairwise
   contraction (a permuted F-sized copy each), measured at 8–9 F allocated and 3× the time per
   message at D = 12 against the old arena kernel. Restored as `fused_norm_message` /
   `fused_norm_scalar` (dense only): messages absorbed along their bond by slice GEMMs into a
   ping-pong buffer, ψ̄ folded into the closing GEMM — ψ + two buffers live (3F), no permutations.
   Agrees with the generic path to 1e-15; anything unrecognised falls back.

8. *Hermitian eigen on fermionic bonds.* Matricising a 2-leg tensor with the SECOND stored leg as
   codomain transposes the legs, and a transposition of two odd legs carries the fermionic sign, so
   `eigh_full` returns the odd-parity block negated: a positive message came back with eigenvalues
   −0.087, −0.297 and `symmetric_gauge!` (hence every fermionic boundary-MPS expectation on a state
   whose stored leg order puts the unprimed bond second) died in `sqrt`. `pseudo_sqrt_inv_sqrt` was
   unaffected only because it passes the stored order. (A first fix keyed on arrow duality instead
   of leg order broke the Jordan–Wigner tests — occupations off by 0.15 — and was caught the same
   night.) The seam's `eigen` now decomposes in stored order and maps back (`conj(V)` onto the
   requested leg); verified `Ul·D·dag(U) = M` in M's orientation and `ψ·√M·dag(√M⁻¹) = ψ` to 1e-16.
   Regression test: fermionic 2D testset "eigen of both message orientations".

9. *Graded `:cycle` at over-parametrised χ.* Fermionic 4×4 D=3: χ=24 certifies in 4 sweeps (3.6e-8)
   but χ=36 had F stable to 1e-10 while the marginal signal grew sweep by sweep and ⟨N⟩ drifted to
   6e-3 after 40 sweeps. The zero-padding of the retained bond drew its sectors from `new_index`,
   so a padded bond's graded structure varied between sweeps. New seam verb `pad_index(w, ins, kt)`
   fixes the per-sector target from the fused interface space; χ=36 now converges (9.5e-11 at 40
   sweeps). `_ctm_align` outcomes are counted in `CTM_SVD_STATS`: on graded data the only declines
   are in sweep 1, against the greedy seed (`:align_nonunitary` 26 fermionic / 36 Z2, then zero);
   the dense twin never declines. The Z2 D=4 χ=8 case does not certify on EITHER backend (F flips by
   5e-5 between two sweep parities; graded 3e-4 vs dense 2e-6 off exact) — an under-parametrised
   `:cycle` limitation, not a graded defect; at χ=16 both read 3.04e-6.

10. *Graded `:cycle` speed.* Z2 4×4 D=4 χ=16, 3 sweeps: 62 s vs 22 s on the dense twin. Krylov
    matvecs were 2382 vs 1386 because every charge sector requested the full χ; proportional per-
    sector requests with a saturation re-solve cut that to 1950 at identical retained spectra, but
    only 5% of the time: the remainder is GradedArrays' per-operation cost on the small blocks a
    D=4 Z2 state has (the fermionic 4×4 D=3 `:cut` shows the other side of the same coin — 10 s
    here against 37 s on the TensorKit backend). `:cycle_matvec` is counted in `CTM_SVD_STATS`.

11. *BLAS threads.* The fused BP kernel issues many small GEMMs: with 8 OpenBLAS threads a D=12
    grid runs 0.36 s vs 0.26 s single-threaded (threading overhead), while D=40 runs 9.9 s vs 16.7 s.
    Set `BLAS.set_num_threads(1)` for small-D sweeps; no code-level heuristic was added (global
    BLAS state is the caller's).

12. *6×6 cross-method stress (no exact reference).* ⟨N⟩ on a gate-built fZ2 6×6 D=3 state: `:cut`,
    `:cycle` and boundary MPS agree to 7e-5 at χ=8, 5e-7 at χ=32, 3e-8 at χ=48 (145 s / 162 s / 14 s).
    On a random dense 6×6 D=3 state (flat spectrum) the spread is 4e-3 at χ=8 and 4e-5 at χ=48, and
    `:cycle` there costs 430 s at χ=48 against 37 s for `:cut` — the Krylov route is not for flat
    spectra, as the subspace-route notes already say. Element types: ComplexF32, Float32 and Float64
    states run through BP, gate application, CTM and boundary MPS without error.

**Upstream-bound costs (not fixable in this package without custom contraction code).** TensorAlgebra
matricises a contraction operand by a permuted copy whenever its contracted legs are not contiguous
at an end (measured 1 F vs 2 F per product), zero-fills every allocated output, and `mul!` into a
preallocated destination is slower than the allocating product on contiguous legs (3.5×); GradedArrays
pays a per-block bookkeeping cost that dominates at D ≤ 4. These account for the remaining 10–30%
gaps to the TensorKit backend on CTM/BMPS and for GC's ~17–20% share of a sweep.

Compile latency dominates graded runs (package precompile ~4–6 min after a source edit); probe new
conventions in a scratch environment with the ITensorBase stack alone (seconds) before the suite.

## Performance audit after the backend switch — *2026-09-10 (night)*

Harnesses in the session scratchpad (`bench_peak.jl`, `bench_ctm.jl`, `bench_bmps.jl`,
`bench_graded.jl`): a `named_comb_tree((3, 3))` whose centre tensor (2·D³, F = 500 MB at D = 250)
dominates every other tensor, so the resident-set high-water mark over an operation reads in
units of F; CTM/BMPS on random 6×6 / 8×8 grids. Single BLAS thread. Controls: `Fixes` (TensorKit
backend) and `main` (ITensors backend), in worktrees.

| operation (D = 250 comb) | main | Fixes | FixesV2 before | FixesV2 now |
|---|---|---|---|---|
| BP, 3 iterations: time / peak | 44 s / 7.1 F | 30.6 s / 0.1 F | 29.5 s / 0.5 F | same |
| simple update, 1 layer: time / peak / churn | 46 s / 6.3 F / 41 F | 42.5 s / 2.8 F / 2.9 F | 49.5 s / 5.4 F / 28 F | 42.2 s / 3.3 F / 15 F |

*BP.* The generic sequence path matricises both operands of every pairwise contraction (TensorAlgebra
skips the copy only when the contracted legs are already contiguous at an end — measured 1 F vs 2 F
per product), so a message cost 8–9 F and 3× the time. The fused kernel (`fused_norm_message`,
`fused_norm_scalar`) absorbs each message along its bond by GEMM into two pooled buffers: ψ + 2F
live. It applies to any dense state vertex with standard doubled messages; graded, MPS-link and ρ
cases fall back.

*Simple update.* Three seams were allocating stubs after the port: `absorb_chain` (environment
chain), `left_orthogonalize(consume_input)`, `contract(...; dest)`. Now: chains `mul!` into the
consumed input's storage ping-ponging with one pooled buffer, laid out so the next factor
contracts by a free reshape; the QR factorises the matricised workspace in place (`qr_compact!`)
and writes Q into the consumed storage; a two-factor `contract` writes into `dest`. Peak 3.3 F: the
remaining 0.3 F is TensorAlgebra materialising a permuted-output temporary; without the layout trick
it is 3.05 F at 50 s/layer. The QR itself is 8 of the 14 s of a centre gate (LAPACK Householder,
inherent at one thread).

*CTM.* L=6 D=4 χ=64 `:cut`, 3 sweeps: 81 s (Fixes 79 s), peak over baseline 403 MB (Fixes 623 MB).
At L=6 D=3 χ=32 the new backend was 10–17% slower (small-matrix overheads). Profile of that sweep:
contractions 40% (of which permuted copies ~11% and output zero-fill ~3%, both inside TensorAlgebra),
SVD 21%, QR 8%, `_hdot` 6% (an `aligndims` copy even when leg orders agree — removed), GC ~20%.
(A micro-benchmark suggested MatrixAlgebraKit's default "safe" SVD driver was 6× slower than plain
divide-and-conquer at 288×288; measured cleanly through the tensor interface both take 0.04 s and the
CTM sweep did not change — the default stays.)

*Boundary MPS.* L=6 D=3 χ=32, 3 iterations: 106 s (Fixes 80 s), peak over baseline 492 MB (Fixes
547 MB), ~145 GB churn on both. Profile of one update: GEMM 37%, GC 17%, TensorAlgebra permuted
operand copies ~20%, output zero-fill 8%, QR 6%. Contracting into an uninitialised destination via
`mul!` was measured and rejected (3.5× slower on contiguous legs). The 25% gap is TensorAlgebra
internals; dense boundary MPS agrees with exact contraction to 1e-15 at lossless χ on both backends.
Harnesses are in `benchmarks/`. Graded boundary MPS has no convergence floor: fermionic 4×4 D=3 is
exact (2e-16) at the lossless χ=81, and the 1e-7 readings at χ=24/36 are truncation (unchanged by a
1e-14 fitting tolerance or 200 iterations); the fermionic example's 3.7e-7 plateau for χ ≥ 16 is the
same — its two-row message needs χ=256.

*Graded (fZ2 4×4 D=3, Z2 4×4 D=2).* fermionic CTM `:cut` 6 sweeps 10.4 s (Fixes 36.8 s); `:cycle`
11.4 s (9.2 s); exact contraction 50 s (108 s); Z2 BMPS χ=16 2.4 s (9.8 s); Z2 BP 1.3 s (3.0 s).
Observables identical to the digits printed.

## What made `:cycle` work

Two things. Everything else in the design doc is failed attempts.

### 1. A scale-free Arnoldi tolerance — worth ~1000× at χ=32

The cycle spectrum is the **product** of the four factors' spectra, verified quantitatively:

```
factors s₃₂/s₁ :  2.25e-04 · 6.39e-04 · 4.75e-04 · 5.17e-04  →  3.5e-14
cycle   s₃₂/s₁ :                                                4.2e-14   (measured)
```

A four-fold product spends three quarters of float64's dynamic range. KrylovKit's `tol` is
**absolute** on the residual, so a fixed `tol = 1e-13` sat *above* the eigenvalues being resolved
(`s₂₂/s₁` is already 4.0e-12): Arnoldi declared an invariant subspace at k≈19–22 while directions out
to k=32 were four orders above machine epsilon, and the projector silently dropped them.

Fix: normalise the cycle action by its dominant eigenvalue magnitude — the spectral radius, which is
what power iteration converges to, not σ_max (five iterations — free, the invariant subspace is
scale-invariant). Then `tol = 1e-16` is relative. χ=32 `⟨X⟩`: 5.2e-11 → 4.8e-14.

**This is very likely the floor the collaborator reports, and is worth telling them.** Their
`CTM_eig_cutoff = 2e-14` and `rank_from_c`'s `rtol = 1e-14` are the same mistake in their idiom — a
relative cutoff on a fourth power, so 1e-14 on the cycle discards single-corner directions at ~3e-4.
Measured on their engine: loosening those cutoffs improved `ln Z` ~6× and moved their rank ceiling off
32 (it had been pinned at 32 for χ = 32, 48, 64 **and** 81).

### 2. Zero-pad the retained index to a uniform width

`kres` — the rank `schursolve` actually resolves — fluctuates sweep to sweep and plaquette to
plaquette. Every change **resizes the interface**, which trips `_ctm_align`'s dimension guard,
discards the gauge, and hands the next sweep a basis it cannot compare with the last.

Build the biorthogonal pair at `kres`, then embed it in a uniform-width index with the remaining
columns exactly **zero**. `Π = P_A P_B` still has rank `kres`, so every region value is unchanged —
bookkeeping, not physics. **Pad AFTER `_ctm_biorth`, never before**: whitening a pair with null
columns inverts a singular overlap.

### Considered and NOT implemented: rank-based deferral

⚠️ Earlier versions of this file listed "never keep less than the cut would" as a third enabler and
prescribed *"compute the cut's rank and defer whenever `rank(cut) > rank(cycle)`"*. **That rule is not
in the code** — the cut pass backfills only keys the cycle pass left unset, with no rank comparison —
and it is deliberately not there, because it produces a mixed, non-stationary lattice (see
[What the engine is](#what-the-engine-is)). Kept here only so the idea is not re-proposed as a
description of current behaviour.

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
float32 export shifts `ln Z` by 7.5e-8 and `⟨X⟩` by 2.1e-9, small enough to look like a real finding
about the collaborator's engine, and it cost a full session's detour. Verified references:

```
ln⟨ψ|ψ⟩ = -6.217866847854575        ⟨X⟩ at their (2,2) = 0.916900598128483
```

---

## Testing

`test/test_ctmenvironment.jl`, testset "Two-projector engine: :cut and :cycle" — option validation,
exactness at lossless χ on square **and** hex, stationarity, the observable win, no-saturation-in-χ on
heavy-hex, determinism, and a regression guard on the convergence test. The observable-accuracy
assertions go through the high-level `expect(::TensorNetworkState; alg = "ctmrg")` path on purpose, so
they exercise its `convergence = :worst_region` default; the stationarity assertions that drive
`update` directly name `:worst_region` explicitly (the direct-`update` default is `:free_energy`).

The whole file takes ~4–5 minutes. **Do not use it as an inner-loop check** — iterate with a small
purpose-built script on one lattice and 2–3 χ values, and run the file once before declaring done.

The engine is deterministic: every Krylov solve takes a locally seeded start vector, so runs are
bit-reproducible and a sweep does not perturb the caller's global RNG. Before that was fixed, the
run-to-run spread exceeded the difference between the two projectors.

**Measurement discipline, learned the hard way.** Never conclude anything from one seed — this
document has had to retract single-seed claims at least five times (a "19× slower" that was 2.5×, a
hex failure regime that did not exist, a `krylovdim` win, a "union was the bug"). Use ≥4 seeds, and
report which cells were informative rather than averaging in cells where both methods are exact.

---

## Retracted claims

Kept deliberately: each was believed, written down, and falsified. Do not re-derive them.

| claim | why it was wrong |
|---|---|
| "`:cycle` fails at boundary sites" | The convergence test stopped before boundaries converged. Boundary-adjacent projectors were bit-identical between backends. |
| "`:cycle` overtakes once `:cut` reaches ~1e-3" | Written up as a rule *before* being tested; fails on 5×5 D=3 and 6×6 D=2. |
| "Sparse grids fail at small χ" / the nilpotency analysis | Same early-stop bug. heavy-hex χ=8 is now exact. The nilpotency measurement stands as a property of that cycle map, but it was never the cause. |
| "A mixture is worse than either pure method" | One seed at one χ; not corroborated multi-seed. Purity is a design choice. |
| "`:cut` is 19× slower at 6×6 χ=24" | Single seed; multi-seeded it is 2.5×. |
| "hex at lossless χ is a known `:cycle` failure regime" | Fixed by seeding the Krylov start vector on plaquette position only; never re-measured until 2026-08-09, when hex came out exact at χ=16. |
| "`:cycle` settles in two sweeps" | Measured on an *interior* observable. Boundaries need more (6×6 corner: 3; heavy-hex: 5), and settling fast is what walked it into the early-stop bug. |
| "`_ctm_statedist`'s ~3e-11 plateau is basis wander in the diagnostic" | The diagnostic was sampling 13% of the state; the residual was real, in the blocks it never compared. |
| their engine's `⟨X⟩` at χ=32 = 8.149e-14 | Measured directly it is 1.330e-12. |

## Measured 2026-08-11: why the periodic-Schur ADAPTER cannot work, and what that implies

`PeriodicSchurDecompositions.jl` (`pschur`, `partial_pschur`, `ordschur!`) is the right tool and it
works: on synthetic CTM-like factors the cycle invariance at site 1 is 3.7e-17. Julia has it; nothing
needs writing. Four attempts to use it via an ADAPTER around the current representation all failed,
and the reason is structural rather than a bug to be found.

**The killer measurement.** Capture real plaquette factors (6×6 D=2, χ=8), zero-pad to square, run
dense `pschur`, `ordschur!` to the dominant `k`, truncate rows back to the true bond dimensions, and
test whether the cyclic relation `As[l]·span(VR[l]) ⊆ span(VR[l+1])` still holds:

```
plaquette nsp=[4,32,32,4]        relative residual
  As[1]*VR[1] -> VR[2]           8.44e-01     VIOLATED
  As[2]*VR[2] -> VR[3]           6.92e-01     VIOLATED
  As[3]*VR[3] -> VR[4]           1.52e-16     ok
  As[4]*VR[4] -> VR[1]           3.37e-16     ok
```

The relations that break are exactly those leaving a bond that had to be row-truncated (bonds 1 and 4
are `nsp=4` inside `n=32`; bonds 2 and 3 need no truncation). Measured directly, the Schur vectors
carry norm **1.4–1.75 outside the true rows** — the invariant subspace of the padded operator really
does live in the padded coordinates, and its projection onto the true coordinates is NOT an invariant
subspace of the rectangular problem. No choice of left-bond mapping repairs this; all four candidates
were tested and none is clean.

⚠️ An earlier check reported "subspace agreement 0.000" and was used to conclude the mathematics was
fine. That measured ONE bond (bond 1) on plaquettes where `kcyc = min(nsp)` made selection trivial. It
did not generalise. Also note a genuine bug found along the way and worth keeping: row-truncated Schur
vectors are **not orthonormal** (‖ZᵀZ−I‖ ≈ 1.3) and must be re-orthonormalised by SVD with a rank
check — but fixing that did not rescue the approach.

**Implication: option (b) is not the expensive alternative, it is the only one that can work.** Their
engine does not pad-then-unpad; it NEVER un-pads. Every corner and edge is stored at uniform width
with an explicit `rank`, so the four cycle factors are square by construction and the invariant
subspace is computed and consumed in the same space, with no projection step to destroy the relations.
Any attempt to keep our rectangular storage and adapt at the call site reintroduces the truncation
that this measurement shows is fatal.

**Also established, and independent of periodic Schur:** the 8×8 plateau is a CONTINUITY problem. A
propagation-derived subspace (a smooth function of the state) converges the sweep to machine precision
— but to a NON-stationary point (`marg` 1e-16 → 5e-7, χ=32 `⟨X⟩` 2500× worse). So the target is a
solve that is continuous in the state AND stationary; propagation gives the first without the second,
the cycle eigensolve gives the second without the first. Under the stated priorities — stationarity
and observable/partition-function accuracy first — a plateau fix that costs stationarity is strictly
bad, and was rejected on that basis.

### FALSIFIED 2026-08-11: the plateau is NOT a projector discontinuity, and option (b) is dead

Direct measurement, not inference. Capture real plaquette factors (8×8 Ising β=0.44, χ=16), perturb
all four along a fixed random direction by relative size ε, and measure how far the retained subspace
moves (‖ΔP‖) for the current solver and for dense periodic Schur:

```
plaquette nsp=[2,32,32,2] k=2   spectral gap at the cut  lam[k]/lam[k+1] = 1.8e+13
 eps      dP pschur     dP schursolve (current)
 1e-10    2.614e-10     7.07e-16
 1e-08    2.614e-08     1.08e-15
 1e-06    2.614e-06     4.10e-16
 1e-04    2.614e-04     2.36e-16
 1e-02    2.613e-02     2.51e-16
```

Same on all three plaquettes sampled. **Both solvers are continuous** — `pschur` scales linearly
(‖ΔP‖ ≈ 2.6ε) — so there is no discontinuity for a periodic-Schur rewrite to remove. Worse for the
hypothesis: **the current `schursolve` is INSENSITIVE**, ‖ΔP‖ ~ 1e-16 at every ε including a 1%
perturbation, because the spectral gap at the truncation cut is 1e12–1e13. By this measure the
existing solver is *more* stable than the replacement would be.

**Consequences.**

* **Option (b) — uniform-width storage plus a rank field — is not justified.** Its entire purpose was
  to enable periodic Schur, whose value rested on supplying continuity the current solve lacks. It
  does not lack it. (b) would be a representation-wide refactor with real performance cost, buying
  cleanliness only. Do not start it on this rationale.
* **The plateau's cause is again unknown.** The under-relaxation α-scaling is solid and still says a
  fixed-size perturbation enters every sweep — but it does NOT come from the cycle eigensolve.
  Remaining suspects, in the sweep but outside the solve: `_ctm_biorth`'s `S^{-1/2}` whitening, the
  zero-padding, `_ctm_align`'s Procrustes gauge fixing, and `_ctm_orthcols`' unpivoted QR (which IS a
  discontinuous operation, though pivoting it was separately measured not to fix the plateau).

⚠️ Three mechanisms have now been proposed for this plateau and rejected by measurement: rank-blind
QR, an under-damped iteration, and projector discontinuity. Propose the next one only with a cheap
falsifying test attached, as here — this check cost twenty minutes and prevented a multi-session
refactor built on a false premise.

## Measured 2026-08-11: performance audit — the defaults are validated, not improved

**Where a `:cut` sweep goes.** Timed inside the projector (6×6 D=3): the two thin QRs plus the SVD
are **>80% of the sweep**; the combiner contractions are 0.7%, and everything shared with `:cycle`
(enlarged-corner build, block rebuild) is the rest. Enlarged corners are already memoised per sweep
(`get!(enl, …)`), so there is no recomputation to remove. `:cut` is 9.3× slower than `:cycle` on the
same lattice (2.457 s against 0.264 s per sweep) and essentially all of that gap is the QR+SVD the
cycle path does not do. **The speed lever is choosing `:cycle`, not tuning `:cut`.**

**`krylov_min = 128` — verified, do NOT lower.** In isolation a top-k Krylov SVD beats a dense one
well below 128, with singular values agreeing to ~1e-15:

```
n     k    dense      topk       speedup
48    8    5.75e-04   1.45e-04   3.97
72    8    5.99e-04   3.43e-04   1.75
144   8    2.65e-03   9.78e-04   2.71
288   8    1.80e-01   3.65e-03   49.17
144/200/288 at k=32          FAILED (info.converged < k)
```

and 40–48 of 100 interfaces at χ=8/16 are blocked from that path *only* by `krylov_min`. Nevertheless
**end to end `krylov_min = 48` is a net LOSS**: 3.3× slower at χ=8, 2.2× slower at χ=32, 1.36× faster
only at χ=16. The isolated benchmark skipped the failures; in the engine a failed Krylov attempt costs
the attempt *plus* the dense fallback, and the real `W` has a decaying spectrum that synthetic
triangular test matrices do not. Accuracy is bit-identical either way, so this is purely a speed
question and the answer is no.

**`qr_cutoff = 1e-13` — inert.** Over 4 seeds, both projectors, 5×5 D=2 at χ=4 and 8, `⟨Z⟩` is
identical to every digit for cutoffs from 1e-15 to 1e-7. No retained direction ever falls below the
threshold, matching the source's own note that the retained spectrum has median `S_k/S_1` ~1e-1…1e-2.
It is insurance against `S^(-1/2)` amplification, not a lever.

⚠️ **Methodology note, since it produced a wrong answer here.** A synthetic micro-benchmark of a
kernel is not evidence about the engine: it excluded the failure path and used matrices with the wrong
spectral profile, and pointed the opposite way from the end-to-end measurement. Benchmark the sweep.

⚠️ **Instrumentation can dominate what it measures.** A first pass recorded block sizes by calling
`_ctm_block_matrix` a second time per interface; that inflated the sweep ~9× and made the QR/SVD
percentages read low. The qualitative conclusion survived, the percentages did not.

### FIXED 2026-08-11: `_ctm_svd_topk` threw on every large interface

`KrylovDefaults.krylovdim` is 30, and `svdsolve(W, x0, k, :LR)` **THROWS** for `k ≥ 30` rather than
returning unconverged. `_ctm_svd_topk` caught it and returned `nothing`, so every interface with
`kw ≥ 30` paid a doomed Krylov attempt AND the dense SVD — worst on the largest blocks, which are
exactly what the top-k path exists to accelerate. That is why the χ=32 instrumentation read
"topk fired 0, DENSE 100": the gate was passing and the solve was failing, not declining.

Fixed by passing `krylovdim = max(2k + 10, 30)`. That alone would make things WORSE, because with the
solve now succeeding the old `nW > 4kw` gate admits cases where top-k loses badly. Measured
(min-of-N, triangular products like the real `W`):

```
n     k    dense      topk       speedup
288   8    4.01e-02   3.19e-03   12.56x
288   16   3.24e-02   4.67e-03    6.94x
288   32   1.05e-01   1.65e-02    6.34x
200   32   2.25e-02   6.03e-03    3.74x
144   8    2.72e-03   8.55e-04    3.18x
144   16   2.76e-03   1.55e-03    1.79x
144   32   2.82e-03   2.07e-02    0.14x   <- 7x SLOWER
288   48   1.80e-02   6.32e-02    0.28x   <- slower
```

The win needs `nW/kw ≳ 8`, so the gate is now `nW > 8kw`. It retains every large speedup above and
declines both losing cases. `krylov_min = 128` is unchanged (re-verified separately: lowering it is a
net loss because of the failure-path cost).

**Accuracy is bit-identical** — `⟨Z⟩`, `marg` and `F` match to every digit across both projectors at
χ = 4/8/16, which is expected since top-k returns the same singular triplets (agreement ~1e-15). 183
assertions pass.

⚠️ **The end-to-end speed gain is NOT verified.** Sweep timings on this machine varied 7× for
identical work (6×6 D=3 χ=16 measured at 0.392 s, 1.034 s, 1.210 s, 2.150 s, 2.892 s across the
session) because three unrelated Julia processes were running throughout; two benchmark runs were also
OOM-killed. The χ=8 arm flipped from "2× slower" to "2× faster" between two runs of the same code.
**Re-measure on a quiet machine before quoting any sweep-level number.** What is defensible without a
stopwatch: the old code did work that could not possibly succeed, and the new gate is backed by
repeated millisecond-scale microbenchmarks that are far less load-sensitive than sweep timings.
