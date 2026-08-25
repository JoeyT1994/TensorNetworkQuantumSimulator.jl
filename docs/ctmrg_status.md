# Finite CTMRG status

Updated 2026-08-25. This is the authoritative summary of the implementation and its known limits.
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
- `convergence = :environment` additionally compares gauge-invariant interface projectors between
  sweeps; use it before reading observables or RDMs. The projector motion itself—not its square—is
  compared with `tolerance`.
- `convergence = :worst_region` is a stricter diagnostic that also watches individual CVM terms.

At least two complete sweeps are required. `expect` and `reduced_density_matrix` automatically use
environment convergence for `:cycle`. The `:cut` path already includes its gauge-fixed state-distance
signal. A local observable is always inserted after converging the unperturbed network; its numerator
and denominator use the same surrounding environment.

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

This is consistent with the collaborator's observation that SVD/cut-style contractions can be an
O(1) factor more accurate for `Z`, while the stationary eig/cycle construction is especially useful
for local derivatives and observables. The remaining double-layer norm difference is an algorithmic
finite-χ effect, not a marginal-consistency or stopping bug.

### PEPS bond-gauge attack

With independent κ=5 bond gauges on the 5×5 D=3 state, raw synchronous `:cycle` fixed points remain
gauge sensitive at low χ (lnZ spread `8.98e-5` at χ=4 and `3.94e-7` at χ=8). Increasing Arnoldi
depth, forcing the requested Schur rank, warm starts, block subspace iteration, and stricter stopping
do not remove that spread. The difference from the collaborator's implementation is structural:
their fixed-size A/c environment supports immediate overlapping 2×2 Gauss–Seidel writes, while the
current global C/T region representation is synchronous and is not closed under a single local
write without retaining stale interfaces.

The supported mitigation is `gauge_state = true`, which solves in Vidal/BP symmetric-gauge
coordinates. On the same κ=5 test it reduces the χ=4 spread to `7.17e-10` and the χ=8 spread to
`3.37e-11`; curves are visually coincident on the Mike-style log plot. This is labeled as a
preconditioner in generated figures, not presented as intrinsic gauge invariance of the synchronous
fixed-point map.

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
- Raw low-χ `:cycle` fixed points are gauge sensitive on double-layer PEPS. Use `gauge_state = true`
  for robust observables; matching intrinsic MP-BP gauge invariance requires a fixed-storage local
  A/c environment refactor.

## GPU status

Finite CTMRG is device-compatible but not yet GPU-accelerated in practice. On an RTX 3070, a 6×6
random ComplexF32 PEPS with D=3 and four fixed sweeps gave:

| χ | projector | CPU | GPU | CPU/GPU |
|---:|---|---:|---:|---:|
| 8 | `:cut` | 0.86 s | 7.66 s | 0.11× |
| 8 | block `:cycle` | 0.58 s | 10.14 s | 0.06× |
| 16 | `:cut` | 1.92 s | 29.27 s | 0.07× |
| 16 | block `:cycle` | 1.49 s | 24.36 s | 0.06× |

A CUPTI trace of one small cut sweep captured 4,492 kernel launches, 1,038 stream synchronizations,
518 device-to-host copies, and only 16.6% GPU-busy time. The sweep currently dispatches hundreds of
small QR/SVD/contraction operations separately and copies each small spectrum to the CPU for rank
decisions. Moving the existing scalar algorithm to a GPU therefore increases latency.

The required optimization is structural: group same-shape interfaces, batch their contractions and
factorizations, keep rank decisions on-device, and reuse persistent workspaces. Until that lands, use
CPU finite CTMRG at these ranks; `ComplexF32` alone does not overcome launch overhead.

## Reproduction and validation

- `examples/ctm_rbim10_precision.jl`: 10×10 precision and convergence diagnostics.
- `examples/ctm_rbim20_jjprime.jl`: paper-matched 20×20 disorder average.
- `examples/ctm_ising5x5_benchmark.jl`: exact-contractible PEPS check.
- `examples/ctm_ising9x9_benchmark.jl`: χ scan against boundary MPS.
- `examples/ctm_ising9x9_amplitudes.jl`: random `⟨s|ψ⟩` SVD convergence against lossless bMPS.
- `examples/ctm_ising5x5_gauge.jl`: κ-controlled raw and symmetric-gauge PEPS attacks.
- `examples/export_peps.py`: converts the ignored local `.npz` fixtures into raw Julia buffers.
- `examples/ctm_gpu_benchmark.jl`: synchronized CPU/GPU throughput and CUPTI profiling baseline.

The focused CTM suite passes 183/183 tests on the current implementation; the last full package run
passed 534/534.
