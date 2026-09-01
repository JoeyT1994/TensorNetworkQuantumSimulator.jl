# Finite CTMRG status

Updated 2026-08-27. This is the authoritative summary of the implementation and its known limits.
The compact derivation is in [`finite_ctmrg_design.md`](finite_ctmrg_design.md).

## API

Finite CTMRG stores a position-resolved `4C + 4T` environment around every occupied grid vertex.
The two projector primitives share the same cache and consumers:

```julia
cut = update(CTMEnvironmentCache(tn, χ; projector = :cut))
cycle = update(CTMEnvironmentCache(tn, χ; projector = :cycle))
cycle_gauged = update(CTMEnvironmentCache(ψ, χ; projector = :cycle, gauge_state = true))

lnZ = cvm_freenergy(cycle)
value = expect(cycle, ("X", [site]))
ρ = reduced_density_matrix(cycle, [site])
```

`expect(...; alg = "ctmrg")` and `reduced_density_matrix(...; alg = "ctmrg")` build and update the
cache automatically. Cycle observables on a `TensorNetworkState` use the Vidal/BP
`symmetric_gauge` preconditioner by default; direct cache construction remains explicit through
`gauge_state = true` so raw-gauge benchmarks are still possible.

## Projectors

| projector | construction | main use |
|---|---|---|
| `:cut` | best rank-χ truncation of each bipartition independently | conservative default |
| `:cycle` | dominant left/right invariant spaces of the four-corner cycle | stationary environments and often better observables |

Both are two-sided and biorthogonal. A numerically low-rank cycle keeps its resolved subspace and
zero-pads only the bookkeeping index; it does not silently switch the sweep to `:cut`. A genuinely
undefined plaquette is declined as a whole, backfilled by `:cut`, and reported by a warning.

The default `:cycle` solver is matrix-free Krylov/Schur. `cycle_subspace = true` selects an
experimental block-subspace backend. That backend can reuse the preceding sweep's compatible
left/right bases with `cycle_warmstart = true`; bootstrap and changed index spaces fall back to the
deterministic cold start.

## Stopping criteria

Use the stopping signal that matches the requested quantity:

- `convergence = :free_energy` watches the CVM `ln Z`; use it for free-energy-only runs.
- `convergence = :environment` uses a gauge-invariant adaptive test for observables and RDMs. A
  stable retained-interface projector is the cheap witness and retains a conservative two-lattice-
  traversal propagation floor. If high-rank near-null directions make that projector wander, the
  solver automatically compares normalized local response environments instead: one-site RDMs for
  PEPS and open local-factor environments for flat networks. These responses determine every local
  observable and are blind to retained-bond gauge rotations. Two strict passes certify; a tiny
  finite-χ limit cycle can also certify after four consecutive passes within four times tolerance,
  returning the iterate with the smallest response residual.
- `convergence = :worst_region` is a stricter diagnostic that also watches individual CVM terms.

At least two complete sweeps are required, and cycle free energies retain a five-sweep bootstrap
floor. `expect` and `reduced_density_matrix` automatically use environment convergence for `:cycle`.
The `:cut` path already includes its gauge-fixed state-distance signal. A local observable is always
inserted after converging the unperturbed network; its numerator and denominator use the same
surrounding environment.

`marginal_inconsistency` measures CVM stationarity, not contraction accuracy. An under-ranked
stationary environment can have a wrong observable, while a lossless nonstationary basis can contract
exactly. Check convergence in χ against exact contraction or a converged boundary MPS reference.

## Current benchmarks

### Random-bond Ising model

The 10×10 benchmark again reaches roundoff in `ln Z` and a local magnetisation for both projectors
once χ is sufficient. Across five disorder samples, the χ=8 mean relative local-observable errors were
`2.7e-13` (`:cycle`), `8.6e-10` (`:cut`), and `2.2e-11` (matched-rank bMPS). The distinct `:cut` and
`:cycle` curves confirm that no whole-sweep fallback is masking the cycle result.

The paper-matched 20×20 benchmark uses `βJ = 0.88`, `βJ′ = 0.17`, equal bond probability, and
`h = 0.01`. Plotting scripts and generated data live outside this repository in the requested
`CTMRG/_PLOTS` analysis folder.

### 9×9 D=3 Ising PEPS

Converged boundary-MPS references are `ln Z = -38.024120943923315` and lattice-average
`⟨X⟩ = 0.9277470295539833`. The high-χ SVD/cut checks are:

| χ | `ln Z` absolute error | average-X absolute error |
|---:|---:|---:|
| 72 | `2.13e-14` | `2.56e-14` |
| 80 | `2.84e-14` | `2.96e-14` |
| 96 | `4.26e-14` | `1.22e-15` |

For five reproducible random computational-basis configurations, single-layer amplitudes
`⟨s|ψ⟩` use lossless bMPS χ=81/96 references. SVD-CTMRG reaches relative error
`2.3e-13` or better for every sample by χ=8, and remains at roundoff through χ=48.

For the double-layer cycle solver at χ=32:

| cycle solver | `ln Z` absolute error | one-point absolute error | time |
|---|---:|---:|---:|
| default Schur | `1.10e-11` | `7.30e-11` | — |
| cold block, 20 steps | `3.04e-12` | `4.64e-11` | 60.0 s |
| warm block, 2 steps | `1.37e-11` | `1.05e-11` | 48.6 s |

Warm start lowers block-solver time by about 19% and improves the observable, but it does not improve
the double-layer norm. Four warm steps moved the nonlinear sweep to a poorer finite-χ fixed point
(`7.62e-10` norm error), so warm reuse is deliberately limited to two local steps.

The former projector-only environment criterion became pathological once χ entered the
over-parametrised tail: at χ=56 its projector distance oscillated between `0.04` and `0.11` even
though the physical local-response distance was below `1e-10` by sweep 8. The adaptive response
fallback now returns `ok` in `102.1 s` instead of warning after `433.9 s`. Selecting the
minimum-response iterate gives average `X = 0.9277470295540583`, only `7.5e-14` from the reference;
the old 80-sweep value was `2.0e-14` away. The norm remains at roundoff.

A fresh χ=36:4:64 tail with this criterion returned `ok` at every rank. The χ=64 endpoint took
`71.1 s`, with `7.1e-15` absolute error in `ln Z` and `4.4e-16` in average X. These regenerated
values replace the former χ=56,60,64 `not_converged` rows in the collaborator-facing CSV and plots.

This is consistent with the collaborator's observation that SVD/cut-style contractions can be an
O(1) factor more accurate for `Z`, while the stationary eig/cycle construction is especially useful
for local derivatives and observables. The remaining double-layer norm difference is an algorithmic
finite-χ effect, not a marginal-consistency or stopping bug.

### PEPS bond-gauge attack

With independent κ=5 bond gauges on the 5×5 D=3 state, the preconditioned production scan now gives
`lnZ` spreads `4.71e-14` at χ=4 and `4.35e-14` at χ=8 across original, Vidal/BP, and two random
gauges. These are the local A/c points: overlapping 2×2 plaquettes are written immediately in a
symmetric Gauss–Seidel schedule, then converted back to the public C/T region representation for all
readout. After restoring complete-multiplet back-off in the synchronous Schur path, the former
`1.45e-9` χ=12 outlier is gone: every sampled χ=12,16,…,32 spread is between `4.17e-14` and
`5.15e-14`.

The supported mitigation is `gauge_state = true`, which solves in Vidal/BP symmetric-gauge
coordinates. This is labeled as a preconditioner in generated figures rather than conflated with
intrinsic invariance of every synchronous high-rank fixed point.

Expanding each observable contraction to an exact 3×3 window is not a useful substitute. At χ=8 it
still left a `5.7e-9` Vidal/original average-X split, while producing high-order contractions and a
large runtime increase, so that experiment is not exposed as an option in the maintained example.

## Known limits

- `:cycle` can limit-cycle when χ is too small. Increasing χ is the reliable remedy.
- Random signed D=3 states can favour `:cut`; physical D=3 PEPS and flat RBIM networks often favour
  `:cycle`. Select using more than one representative state.
- The optional block-subspace backend is useful on GPU-oriented paths but is not uniformly more
  accurate than Schur.
- `cycle_rankcut` and `cycle_gapcut` are experimental and disabled by default because valid deep
  directions overlap spectral noise on strongly non-Hermitian networks.
- Double-layer `ln Z` remains the main accuracy weakness of `:cycle`; observables are typically much
  stronger.
- Raw low-χ `:cycle` fixed points can remain gauge sensitive on double-layer PEPS. Use
  `gauge_state = true` when representation sensitivity itself is not the quantity under study.
- `cycle_local = true` is the default production routing. The fixed-storage A/c map uses atomic
  12-variable writes, complete real-Schur multiplets, QL balanced biorthogonalization, and a
  symmetric in-place plaquette schedule. Its convergence witness is the gauge-invariant local
  response, and its values are still read through the unchanged `sum_R c_R log(Z_R)` functional.
  A local attempt is bounded to 12 sweeps and must certify; otherwise `update` falls back silently
  to the established synchronous cycle rather than returning an unconverged A/c state. The local
  path is selected for flat networks and for double layers with `2 <= chi <= 8`; `chi = 1` cannot
  retain a complete real-Schur multiplet, while higher-rank PEPS norms have substantially better
  one-point accuracy under the established synchronous sweep.
- The manuscript's rule against splitting real complex-conjugate multiplets was the decisive low-χ
  fix. On the 9×9 D=3 double layer, requested χ=4 drops a crossing pair to local rank 3 and response
  motion reaches `7.7e-14` in nine sweeps; requested χ=8 reaches about `1.5e-12` in ten. Requested
  χ=7 is an isolated awkward cut on this example: some local spectra have no dominant real
  seven-dimensional invariant subspace, so use χ=6 or include the complete pair at χ=8. This is not
  treated as evidence for a generic CTMRG instability and no damping/tolerance knob is added. The
  synchronous Schur path now applies the same complete-multiplet rule: a tied boundary in either
  left or right spectrum backs the retained rank off to the preceding gap. On a focused 9×9 rerun,
  χ=9,12,13,15 all changed from `not_converged` to `ok`; the χ=13 `ln Z` error fell from
  `9.92e-6` to `1.41e-7`. The complete χ=1:32 production scan still flags χ=7,16,24 honestly:
  χ=7 is the expected awkward cut, while χ=16 and χ=24 settle into small period-four response
  cycles (respectively about `2.5e-8` and `5.0e-10` one-sweep response motion) rather than literal
  fixed points. Their final `ln Z` errors are `1.68e-9` and `4.66e-10`; the stopping tolerance is
  not relaxed merely to relabel those rows.
  Focused local-cycle tests cover the fixed storage, response contraction, complete-multiplet rule,
  balanced gauge, C/T bridge, exact region functional, stationarity diagnostic, and production
  routing.

### Finite hexagonal Heisenberg thermal state

The instrumented 2×2-cell open honeycomb purification has 16 sites, final PEPS `D=3`, and
`beta=0.2` after five `d_beta=0.02` imaginary-time steps. Against converged bMPS `chi=32`,
`ln Z = 11.1411782496936` (`-ln Z/N = -0.696323640605848`). Absolute `ln Z` errors are:

| χ | bMPS | `:cut` | `:cycle` |
|---:|---:|---:|---:|
| 2 | 2.41e-8 | 5.62e-8 | 6.55e-8 |
| 4 | 0 | 1.78e-15 | 1.78e-15 |
| 6 | 1.78e-15 | 1.78e-15 | 3.55e-15 |
| 8 | 1.78e-15 | 3.55e-15 | 3.55e-15 |
| 12 | 0 | 0 | 1.78e-15 |
| 16 | 0 | 0 | 1.78e-15 |

Thus both projector families are accurate and χ-stable once χ=4. `:cut`'s conservative raw-state
guard does not certify at χ=6/8 even though `ln Z` and the marginal diagnostic are at roundoff,
exposing an over-parametrized null-space stopping issue rather than a scalar error. The original
`:cycle` scan regressed to about 1.6e-8 at χ=12/16: the narrowest cycle bond had dimension nine, but
Arnoldi resolved only 3–5 modes when asked for its complete space. Production now detects when χ can
hold the full narrowest bond, starts from that identity space, and propagates it around the cycle
without an eigensolve. This restores machine precision and monotonicity without a magnitude cutoff;
the seed-101 10×10 RBIM remains at machine precision through χ=16. Sparse geometry still makes 2 of
10 rectangular CVM cycles undefined, so those boundary plaquettes correctly use `:cut`; the run is
necessarily hybrid. The local Gauss--Seidel bridge is rectangular-only and declines this sparse grid.

The scaled observable benchmark uses a 4×4-cell open honeycomb (48 sites), final purification
`D=4`, `beta=1`, and a weak staggered field `h_s=0.05`. The field is necessary because the finite
spin-symmetric state has exactly zero one-point staggered magnetisation without a pin. The measured
quantity is `m_s = sum_i eta_i <S_i^z>/N`; a bMPS χ=48 reference gives
`m_s = 0.0234599085661912`. At matched χ=4,8,12,16, eig-CTMRG absolute errors are respectively
`2.11e-10`, `5.88e-11`, `5.75e-11`, and `1.57e-11`, versus SVD-CTMRG errors `2.29e-9`,
`6.21e-10`, `5.63e-10`, and `5.21e-10`. Thus the stationary estimator is about 10× better at
χ=4–12 and 33× better at χ=16; beyond that both CTMRG curves are at a roughly `1e-11`–`1e-10`
numerical/reference floor. The χ=32 bMPS value agrees with its χ=48 reference to `2.3e-15`, so this
separation is not caused by an unconverged reference. The CSV and matching plot are
kept with their plotting script in the analysis project's `CTMRG/thermal_plots` directory.

A deeper incremental temperature scan keeps the same 48-site lattice, raises the purification to
`D=6`, and fixes every tested contractor at χ=12. Six checkpoints cover `T/J=2.5` down to
`T/J=0.4167`; each is now referenced independently with bMPS χ=160 and cross-checked at χ=128.
The formerly reference-limited `beta J=1.2` point moves by only `1.57e-13` between those reference
ranks. For the pinned staggered magnetisation, eig-CTMRG is 5.4× more accurate than SVD-CTMRG at
`beta J=0.8`, 28.8× at `beta J=1.2`, and 8.5–9.2× at `beta J=1.6...2.4`. At `beta J=0.4` all
estimators are already at the `1e-12` floor. The free-energy panel shows the expected much smaller
separation between the two CTMRG estimators. Production data and plots are in
`CTMRG/thermal_plots/hexagonal_heisenberg_temperature_scan.{csv,png}`; the maximum simple-update
gate truncation reported by the final `beta=2.4` checkpoint is `1.63e-6`.

A matching positive random-bond scan uses `J_e = J(1 + W u_e)`, `u_e in [-1,1]`, with
`W=0.5` and seed `271828` (the realized range is `J_e/J=0.5381...1.4774`). Keeping every
bond antiferromagnetic preserves the bipartite staggered observable. Disorder makes the estimator
separation especially clear at `beta J=0.8`: after strengthening the reference to χ=160/128, the
fixed-χ eig error in `m_s` is about 62× smaller than SVD-CTMRG. From `beta J=1.2` through `2.4`, eig
is 9.8–10.9× more accurate, while SVD-CTMRG retains a modest advantage for `ln Z`. The χ=128 and
χ=160 references agree exactly to displayed precision through `beta J=0.8`; their `m_s` drift at
the colder checkpoints remains 24–38× below the χ=20 eig error. This is one reproducible disorder
realization rather than a disorder average.
Its data and plot are
`CTMRG/thermal_plots/hexagonal_heisenberg_temperature_disorder_w0.5_seed271828.{csv,png}`;
both clean and disordered plots now use inverse temperature `beta J` on the horizontal axis.

A matched χ=20 follow-up covers the five nontrivial checkpoints `beta J=0.8...2.4`. On clean bonds,
eig improves by factors 1.6–1.7 over χ=12 at the three coldest points and remains 10.5–13× more
accurate than SVD at χ=20. On the disordered realization, eig improves by factors 6.3, 3.8, 3.1,
and 2.8 at `beta J=1.2,1.6,2.0,2.4`; the two coldest eig rows require more than 60 sweeps and are
stored after converging with an extended 180-sweep cap. The `beta J=0.8` eig row is genuinely
non-monotone in χ. Disordered SVD does not satisfy the environment criterion within 60 sweeps for
`beta J=1.2...2.4`; those provisional rows remain flagged and crossed out in the comparison plot,
not silently accepted. The files are
`hexagonal_heisenberg_temperature_{scan,disorder_w0.5_seed271828}_chi20.csv` and
the fixed-χ `hexagonal_heisenberg_temperature_chi12_{lnZ,magnetisation}.png` method-comparison
figures in `CTMRG/thermal_plots`. χ=12 is used for the figures because it gives the clearest
method separation and the strongest converged comparison points; any remaining conservative SVD
environment flags are marked explicitly. Higher-χ data remain available in CSV form.

## GPU status

Finite CTMRG now uses a hybrid CUDA path. The latency-bound greedy C/T seed is built on the CPU and
transferred once; equal-shape cut interfaces use exact batched CUDA QR/SVD when the SVD is at most
32×32, while dense interfaces up to 96 are copied as one shape batch, factored by LAPACK, and returned
as completed thin projectors. The expensive enlarged-corner contractions and sweep absorption remain
on-device. The 96 cutoff is conservative: on the RTX 3070, 48 warmed ComplexF32 72×72 SVDs take
0.077 s on CPU versus 0.200 s on CUDA, while each complete batched PCIe transfer takes below 0.5 ms.
Completed equal-rank projectors return in contiguous batches, and C/T normalization reduces to a
singleton device array rather than synchronizing a host scalar for every block.

Interfaces above the host-batch cutoff are streamed one at a time through exact CUDA QR/SVD. This is
load-bearing at large D: collecting every nominally same-shape matrix filled the RTX 3070 even though
cuSOLVER has no exact batched path for those dimensions. The PEPS ket and bra remain separate lazy
factors throughout; the implementation never materializes a D⁸ double-layer site tensor.

On an RTX 3070, a 6×6 random ComplexF32 PEPS with D=3 and four fixed sweeps now gives:

| χ | projector | CPU | GPU | CPU/GPU |
|---:|---|---:|---:|---:|
| 8 | `:cut` | 1.22 s | 2.13 s | 0.57× |
| 8 | block `:cycle` | 0.80 s | 4.64 s | 0.17× |
| 16 | `:cut` | 2.08 s | 9.53 s | 0.22× |
| 16 | block `:cycle` | 1.91 s | 5.43 s | 0.35× |

The cut values are numerically consistent with the CPU at ComplexF32 precision (a few parts in
10⁻⁶ after four fixed sweeps). The random complex benchmark is not a cycle-convergence benchmark:
its block-subspace iterates diverge on both CPU and GPU, so the cycle rows above are throughput only.
Use the maintained real PEPS/RBIM examples for cycle accuracy.

The intended high-rank crossover is now demonstrated on the full lazy double layer:

| lattice | PEPS D | χ | projector | CPU / sweep | GPU / sweep | speedup |
|---:|---:|---:|---|---:|---:|---:|
| 6×6 | 10 | 20 | `:cut` | 67.75 s | 32.46 s | **2.09×** |
| 6×6 | 10 | 20 | `:cycle` | 47.29 s | 42.20 s | **1.12×** |

Both devices start from the identical prebuilt greedy environment; the timed region is one complete
sweep. The ComplexF32 CPU/GPU `ln Z` differences are `7.09e-6` for cut and `1.79e-6` for cycle.
The cycle row is the production direct Schur/Krylov route (`cycle_subspace = false`); it crosses over,
but its sequential two-sided eigensolver leaves substantially less dense work for CUDA than cut's
large QR/SVD projectors. Before streaming large cut-projector workspaces, the same cut GPU sweep took
85.15 s (0.80× CPU) and peaked near the 8 GB device limit.

Against the resumed all-device cut path, the hybrid route lowers the χ=8 two-sweep time from 1.56 s
to 0.99 s and the eight-sweep time by 1.68×. A matched CUPTI trace reduced kernel launches from
33,252 to 7,126, stream synchronizations from 3,604 to 2,028, device-to-host copies from 1,801 to
1,013, and host CUDA-API time from 930 ms to 396 ms. A CUDA-only regression test exercises a
72-wide D=3 interface and checks the resulting free energy against the CPU.

CPU is still faster at the latency-bound D=3, χ≤16 points, while CUDA wins once the contractions are
large enough. A direct attempt to pack the small corner rebuild GEMMs was reverted: packing cost made
χ=8 35% slower. The remaining structural work is to reduce edge-absorption metadata traffic and reuse
persistent workspaces without materializing the double layer. The internal
`CTM_CUDA_HOST_FACTOR_MAX` environment variable exists only for crossover diagnostics; the default
requires no CTM hyperparameter tuning.

Sweep-wide batching of CVM readout, distance reductions, and Procrustes alignment was measured and
reverted: ordinary queued ITensor operations retained enough temporary storage to destroy the D=10
crossover. Revisit these only behind persistent, explicitly bounded workspaces.

## Reproduction and validation

- `examples/ctm_rbim10_precision.jl`: 10×10 precision and convergence diagnostics.
- `examples/ctm_rbim20_jjprime.jl`: paper-matched 20×20 disorder average.
- `examples/ctm_ising5x5_benchmark.jl`: exact-contractible PEPS check.
- `examples/ctm_ising9x9_benchmark.jl`: χ scan against boundary MPS.
- `examples/ctm_ising9x9_amplitudes.jl`: random `⟨s|ψ⟩` SVD convergence against lossless bMPS.
- `examples/ctm_ising5x5_gauge.jl`: κ-controlled raw and symmetric-gauge PEPS attacks.
- `examples/collect_mike_rbim20_data.jl`: method-separated RBIM collaborator CSV export.
- `examples/collect_mike_tfim9_norm_highchi.jl`: 9×9 norm and site-averaged `X` scan through χ=64.
- `examples/collect_mike_tfim_amplitude_samples.jl`: five-seed TFIM amplitude CSV export.
- `examples/ctm_ising9x9_single_gauge.jl`: focused matched-error study for one adverse gauge.
- `examples/hexagonal_heisenbergmodel_thermalstate.jl`: finite-temperature honeycomb χ scan with
  matched bMPS, SVD-CTMRG, and eig-CTMRG `ln Z` and pinned staggered-magnetisation CSV output.
- `examples/export_peps.py`: converts the ignored local `.npz` fixtures into raw Julia buffers.
- `examples/ctm_gpu_benchmark.jl`: synchronized CPU/GPU throughput and CUPTI profiling baseline.

The focused fixed-storage local-cycle suite passes 38/38 tests and the focused CTM environment file
passes 195/195 on the current implementation. The focused CUDA test passes 2/2, including a 72-wide
D=3 host-batched interface. The non-CTMRG package suite was not rerun for this change; the last full
package run passed 534/534.
