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

**Use `:cycle` when you want stationarity or speed**, and when the lattice is not large. It is the
better projector on hex, on D=2 squares, and on the physical 5×5 benchmark at every χ, and it is
matrix-free so it is much cheaper per sweep. Its two known weaknesses are random D=3 states and, more
seriously, **non-convergence on larger lattices** — see [Open problems](#open-problems).

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

## The convergence test

`update` sweeps until

```
certified  =  it ≥ 2  AND  a real state distance exists
crit       =  max(|ΔF|, statedist²)  ≤  tolerance · max(1, |F|)
```

`|ΔF|` **is not a certificate on its own.** `F` is a signed Möbius sum whose cancellation is worth
~4000×, so it can sit at its final value while the state is still the greedy seed — measured, complex
hex 4×4 D=2 at χ=64 reported `|ΔF| = 2.2e-16` on sweep 1 with a still-greedy environment. Hence the
two-term criterion and the `certified` guard. `statedist²` rather than `statedist` because `F` is
stationary in the state at the fixed point, so `|ΔF| ~ sd²`; holding both to the same tolerance is
dimensionally inconsistent and measured ~3× the sweeps for no accuracy.

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

## Open problems

Ranked by how much they should worry you.

### 1. `:cycle` does not converge on 8×8 — the top issue

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
heavy-hex, determinism, and a regression guard on the convergence test.

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
