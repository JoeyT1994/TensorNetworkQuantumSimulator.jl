# 3D classical tensor networks: status and handoff

*Written 2026-09-27 (evening) for the next agent. Branch `FixesV2`. Everything below is measured and
dated; where a number lives in another doc, that doc has the detail.*

## Morning summary — overnight 2026-09-28/29 (local only: the A6000 and the workstation CPUs)

**1. Yang–Lee edge, D = 4, χ = 48** (`examples/yang_lee/scan_krylov.jl`, now with `GPU=1`).
- **β = 0.21: done.** 13 points to v = 0.012 of the fold, ξ at every point (2.4 → 5.2).
  Data: `examples/yang_lee/data/ylk3_beta0.21_D4_chi48.csv`.
- **β = 0.20: done.** On 8 CPU threads to v = 0.056 at 1–3 h per point, then on the GPU from 06:40 at 30 min per point,
  to v = 0.012. Data: `examples/yang_lee/data/ylk3_beta0.2_D4_chi48.csv`.
- **The finding: the fold stays put from D = 3 to D = 4 at both β**, while ξ at the fold keeps growing:

  | | θ_f D = 3 | θ_f D = 4 | shift | ξ_f D = 3 → 4 |
  |---|---|---|---|---|
  | β = 0.20 (t = 0.098) | 0.0170106 | 0.0169843 | −0.16 % | 3.73 → 3.96 |
  | β = 0.21 (t = 0.053) | 0.0063160 | 0.0063139 | −0.03 % | 4.70 → 5.20 |

  - D = 2 → 3 had moved the fold +1.6 % and +5.5 %, so θ_c − θ_f ∝ ξ_f^−2.785 does not describe D ≥ 3.
  - The D = 2, 3 extrapolation (ζ_c ≈ 1.61–1.62) is withdrawn.
  - Over three treatments of θ_c, the t → 0 limit gives ζ_c = 1.636–1.663 and |z_c| = 2.45–2.49, 1–2.5 % above the
    FRG 2.43(4). That is consistent, but not a controlled number.
  - Detail: docs/yang_lee.md, "D = 4 edge map at β = 0.21" and the β = 0.20 part after it.
- **The χ question is answered: χ = 48 is converged at the fold.** At β = 0.21, θ = 0.0061915 (v = 1.9 %), χ = 64 moves
  Im m by −3.5e-5. That is equivalent to shifting θ by 1e-7 (0.0015 %), 20× below the D = 3 → 4 shift. ξ grows 2 %
  (4.935 → 5.031).
  - The run is a GPU copy of the check, 1.5 h. The CPU copy, 18 h into its first point, was stopped.
  - The last point (v = 1.2 %) agrees: ΔIm m = −2.4e-5, ξ 5.203 → 5.323 (1.6 h on the GPU, finished 03:50).
  - So the stalled fold is not a χ artefact. Detail: yang_lee.md, "The χ = 64 check".

**2. Three-state Potts at D = 4, χ = 48: done. β_t = 0.550505, Q = 0.1678.**

| | β_t | Q |
|---|---|---|
| Monte Carlo (Janke–Villanova) | 0.550565(10) | 0.16160(47) |
| D = 3, χ = 27 | 0.550408 (−0.029 %) | 0.1891 (+17 %) |
| **D = 4, χ = 48** | **0.550505 (−0.011 %)** | **0.1678 (+3.8 %)** |

- Both branches have converged points at β = 0.5504 and 0.5506, on either side of the crossing. Everything is
  interpolated linearly between those two rows. Detail: boundary_peps.md, "3D three-state Potts".
  - A third disordered row, at 0.5508, allows a quadratic interpolation: β_t = 0.550507, Q = 0.1686.
  - Quoted with the interpolation error: **β_t = 0.550506(2), Q = 0.168(1)**.
- **Ordered branch (GPU): finished at 16:45.** It ran 0.565 → 0.551 on the coarse grid, then 0.5508 → 0.5504 on the fine
  grid, at 1–7 h per point. At 0.5502 it collapsed to m = 0.002, so its metastable window is narrower than at D = 3.
- **Disordered branch (GPU, not the CPU as the goal said).**
  - The CPU run's first point, a cold D = 3 → 4 climb at β = 0.550, had not converged after 27 h. That is at 16 threads,
    on a workstation shared with two other jobs.
  - Once the ordered run freed the A6000, a GPU disordered run started at 16:55. Its first point, at β = 0.5504, took
    4.1 h cold; the one at 0.5506 took 25 min warm-started.
  - The collapsed ordered point at 0.5502 had predicted the disordered f at 0.5504 to within 2e-7.
- Still running, as a cross-check only: the GPU disordered run, now past 0.5508, heading to 0.5512.
- The CPU disordered run was stopped at 22:40, still on its first point after 29 h. The GPU run had made it redundant,
  and it was taking cores from the χ = 64 Yang–Lee check.

**What went wrong overnight, and the fixes (all committed):**
- **GPU memory.** Two jobs on the A6000 ran it out of memory three times. CUDA.jl's pools do not return memory, and
  `JULIA_CUDA_SOFT_MEMORY_LIMIT` did not hold them (a 16 GB cap reached 24 GB). One crash went unnoticed for 2.5 h:
  the log watcher filtered it out.
  - `_bp_evaluate` now runs the norm and sandwich environments one after the other on a GPU. Concurrently, their
    working sets added and each task opened its own stream: one crash was out of memory in cuStreamCreate, one a
    segfault in cuStreamIsCapturing.
  - The Yang–Lee maps run as a loop of 30-minute resumable chunks, so a crash costs one chunk, and the driver's
    cut-off/resume step control works as designed.
- **The θ = 0 start.** With the old 200 s limit on the real start (|g| = 3.7e-4) the first D = 4 point had not converged
  after 65 min. `T0ITER`/`T0LIMIT` fix that: a converged start, then 5–13 min per point away from the fold.
- **The coarse Potts ordered run was stopped** at β = 0.550 (35 min into that point) to free the GPU. It was replaced
  by the fine-grid run above.

**Next:**
1. The χ = 64 check is done: χ = 48 holds at the fold, so no redo is needed.
2. If χ = 48 holds, the folds have stalled in D at ξ ≈ 4–5. Then the question is whether D = 5 moves them at all — the
   cluster, or a free GPU — and a third β (0.215) sharpens the t → 0 limit more than a fifth D.
3. For Potts, raise χ (64) so the gradient's noise floor drops below the gtol, or accept |g| ≈ 5e-6 with a time cap per
   point.

---

Where this can go — the literature, the levers ranked, the targets ranked, and a sequence:
[`roadmap_3d.md`](roadmap_3d.md).

**What precision is reachable, and what to aim at (2026-09-28): [`campaign_3d.md`](campaign_3d.md).**
- Ice w_h to 8 d.p. is out of reach: D ≈ 10 at χ = 2D² is the single-GPU ceiling, and convergence is
  algebraic in ξ.
- The Ih–Ic gap is to be measured directly: −ln F_I, and 2(S_h − S_sym) from an inversion-symmetric ansatz.
- Local gates come first: G0 the symmetric ansatz, G1 larger exact tori (they give ~3e-7 against the PEPS's
  ~2e-6), G2 implicit gradients.
- Targets re-ranked: ice, the Yang–Lee edge location, and the 3D Ising interface tension as the ± fixed-point
  overlap (new, low risk), then gauge–Higgs and complex-field Potts.

**3D three-state Potts at zero field (2026-09-28):** [`boundary_peps.md`](boundary_peps.md), "3D three-state
Potts". A two-branch scan (ordered and disordered continuations, `examples/potts3d_boundary_peps.jl`) gives,
at D = 3, β_t = 0.550408 (MC 0.550565) and latent heat Q = 0.1891 (MC 0.16160; the tensor product
variational approach 0.228). D = 4 on the GPU is running. This is the coexistence solver target 4 asks for,
and the real-field anchor of target 5.

---

## Morning summary — overnight 2026-09-27/28 (local machine only; nothing pushed)

Detail: docs/ice.md, "How far BP simple update is from the eigenvector — and variational states";
numbers: `examples/ice/results_honeycomb.csv`; run everything with `julia examples/ice/analyse_honeycomb.jl`.

1. **BP simple update is ~1e-5 per cell from an eigenvector, whatever BP says.** The true truncation
   error, ln f = 2 RQ(A) − RQ(A²) per cell (A = I M; 0 only for an eigenvector; the new `:mnorm`
   network), for D = 3–8 at χ = D²: −8.3e-5, −2.4e-5, −3.7e-5, −1.15e-5, −1.12e-5, −1.15e-5 — against BP's
   own discarded weight per bilayer of 5e-7 … 1.5e-14. The Bethe metric misjudges the loop (ice-rule)
   correlations it truncates by up to nine orders of magnitude; that is why w_h(BP-SU) creeps.
2. **Variational states (GPU, resumable, `honeycomb_variational.jl`)**, from BP-SU, at the practical
   noise floor:

   | D | w_h | ln F_I per cell | ξ | ln f per cell |
   |---|---|---|---|---|
   | 3 | 1.5074106 (BP-SU 1.5072791) | −4.25e-6 (BP-SU −7.06e-6) | 2.16 (1.31) | −3.2e-5 (−8.3e-5) |
   | 4 | 1.5074547 (1.5073835) | −5.40e-6 (−5.17e-6) | 4.80 (1.97) | −2.0e-6 (−2.4e-5) |
   | 5 | 1.5074530 (1.5073588) — not the optimum | −3.04e-6 (−4.74e-6) | 4.40 (2.04) | −3.4e-6 (−3.7e-5) |

   (χ = 2χ_opt; cell PEPS D = 3: 1.5074448; Xu–Lin–Zhang raw D = 7, χ = 150: 1.5074584; Kolafa
   1.5074674(38).) The variational states are 10× closer to an eigenvector and twice as correlated. D = 5
   stalled below D = 4: optimised at χ = 32 < 2D², its gradient was too noisy — the reason the campaign
   must optimise at χ ≥ 2D².
3. **Verdict on ln F_I ≈ −5e-6 per cell: survives in sign and order of magnitude, not in value.** Every
   state — BP-SU D = 4–9 (−4.5 … −5.2e-6; D = 9: −4.48e-6) and variational D = 3–5 (−3.0 … −5.4e-6) — has a
   nonzero inversion asymmetry: S_h − S_c ≈ 1.5–2.7e-6 per molecule. But the variational values scatter
   ±1.2e-6 without a trend (F_I is first order in the state error; the D ≥ 4 optima are noise-limited).
   Settling it needs converged optima at D = 5–8 with χ_opt ≥ 2D², then a ξ-extrapolation.
4. **Cluster kit ready** (`examples/ice/cluster/`): pinned environment (Project + Manifest, repo by relative
   path), `setup.sh`, `ice_array.slurm` (job array, resume, self-requeue, dry-run mode), `jobs_ice.txt`
   (13 tasks) and `jobs_smoke.txt`, `cost_table.md`, README; analysis `examples/ice/analyse_honeycomb.jl`.
   Tested locally: dry runs of every task type, and both smoke tasks through the real script on the local
   GPU (D = 4 reproduced exactly). Before submitting, update `jobs_ice.txt`: variational χ_opt = 2D² (the
   list has it) — and expect ~1 accepted step per 10 minutes only on this card; on an H100 far more.
5. **Engineering found overnight** (all committed): the optimiser's step budget must use warm evaluation
   times (the first evaluation of a GPU run compiles for ~130 s); steps ≤ 0.01 and CTM ≤ 100 iterations
   per evaluation (a 0.05 step left the CTM unconverged after 400); start each line search near the last
   accepted step; stop when 3 steps gain < 1e-7 in RQ; the production driver's chunk must shrink to fit
   its budget; two GPU processes on the 8-GB card stall each other (CUDA.jl's allocation retries) — one
   at a time.

---

## Ground rules (the user's standing instructions)

* **Commit on FixesV2; never push** — the user pushes. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
* **Runs ≤ ~10 minutes each, at most 2 Julia processes at once**, BLAS on 1 thread when Julia threads
  are used (otherwise ≤ 6 BLAS threads per process with two running). Long jobs are chains of
  resumable ten-minute runs (every driver here checkpoints).
* **Stop means stop**: on an interrupt kill every background job — Julia runs, bash lane loops,
  monitors — by command line, and do not resume until asked. On Windows/MSYS stopping a background
  task kills only its wrapper: find loops with
  `Get-CimInstance Win32_Process | ? CommandLine -match '<script name>'` and kill them, then their julia.
* Do not touch the user's untracked files (`examples/collect_mike_*.jl`, `examples/ctm_ising9x9_energy.jl`,
  `examples/hexagonal_heisenberg_spin_glass_response.jl`, `tmp/`).
* The GPU cluster (H100/H200, Slurm, ~16 GPUs at once) is for "something substantial" only — ask
  before preparing a cluster campaign; the user pushes the code there.
* When truncating a PEPS after applying an operator, fix the gauge (BP/Vidal) first — the user treats
  this as obvious.

**Environment.** The committed `Manifest.toml` is stale. Use an environment that `dev`s this repo;
the working one (Julia 1.11) is
`C:\Users\Joey\AppData\Local\Temp\claude\C--Users-Joey--julia-dev-TensorNetworkQuantumSimulator\2c7647bb-3e4d-45af-be25-3fbd5b06dd24\scratchpad\gpuenv`
(deps: TensorNetworkQuantumSimulator (dev), CUDA, Adapt, Dictionaries, KrylovKit, Statistics). The
backend packages come from the ITensor registry (`https://github.com/ITensor/ITensorRegistry`); set
`JULIA_PKG_USE_CLI_GIT=true` if cloning hangs.

---

## The engines (library, `src/MessagePassing/`)

| piece | what it is | status | doc |
|---|---|---|---|
| `InfiniteCTM2D` | 1×1 infinite 2D CTMRG; a site is a LIST of layers, never fused; Kikuchi ln κ; `site_environment` gradients; split (matrix-free) pairs; `c4v`; GPU via `adapt` | production | boundary_peps.md |
| `InfiniteCTM3D`, `CTM3DEnvironmentCache` | infinite / finite 3D CTMRG (corner/edge/face blocks) | works; infinite fixed point biased ~1 % (one χ-index per quarter-plane cannot hold a 2D boundary); vertex regions cost ~χ¹⁰ | ctmrg3d.md |
| `boundary_peps` (L-BFGS), `boundary_peps_krylov` (Newton–Krylov, subspace trust region), `boundary_peps_stationary` (FD Jacobian) | 3D models by a 1×1 boundary iPEPS of the layer transfer operator, f = ln κ⟨Ψ\|T\|Ψ⟩ − ln κ⟨Ψ\|Ψ⟩; complex/bilinear for imaginary fields; symmetry groups (`:c4v`, `:diagonal`, `:none`); permuted bras (`bra_perm`, `norm_perm`) | production; Newton–Krylov is the fastest accurate route (docs/boundary_peps.md "Routes") | boundary_peps.md |
| `ising3d_site`, `ice_site` | 3D Ising layer (real/complex field); the ice bilayer cell | tested | ice.md |

Tests: `test/test_ctm2dinfinite.jl`, `test_boundarypeps3d.jl` (30), `test_ice.jl` (17, ~6 min),
`test_ice_honeycomb.jl` (17, ~4 min), `test_gpu_paths.jl` (skipped without CUDA). Last full pass of the
3D/ice files: 2026-09-27.

---

## Project 1: the Yang–Lee edge of the 3D Ising model (docs/yang_lee.md)

**Done.** The stationary bilinear (complex) boundary PEPS follows the dominant eigenvector into the
imaginary field and ends in a finite-D fold. D = 2 maps at β = 0.16–0.21; D = 3 maps at β = 0.18,
0.20, 0.21 (χ = 24): the fold moves up with D (+0.1 %, +1.6 %, +5.5 % at t = 0.19, 0.10, 0.05) and
ζ_eff flattens; two-point D extrapolation gives ζ ≈ 1.61–1.62 (FRG 1.621(4)) — uncontrolled until
D ≥ 4. At D ≤ 3 the fold is mean-field-like (ξ ≈ 3–5).

**In progress: the fold solver.** `examples/yang_lee/fold_pseudoarclength.jl` (+ `fold_run.jl`):
pseudo-arclength continuation through the fold — θ real, the complex-linear stationarity equations
bordered by a real arclength row, truncated SVD dropping the redundant (gauge/unused-bond) directions,
Broyden-updated [J g_θ], secant predictor with step control on the secant's turn angle and a
|Re m| guard (at the fold the PT-symmetric branch crosses the PT-broken one). **Works at D = 2,
β = 0.20: θ_f = 0.0167381** (quadratic fit through the turning point, rms 2.9e-6; against the scan's
0.0167357), Im m_f ≈ 0.6666, ξ_f ≈ 3.19, in one 516-s run. Not yet in the library.

Next: port into the library (resumable, chunked), run at D = 3 β = 0.20, then D = 4–5 at
t ≈ 0.05–0.10 with ξ at the fold, for θ_c − θ_f ∝ ξ_f^{−(3−Δ_φ)} (GPU for D ≥ 4). Other scripts:
`examples/yang_lee/scan_krylov.jl` (the D = 3 map driver), `analyse_maps.jl`, `probe_cost.jl`,
and `examples/ising3d_yang_lee_scan.jl` (the D = 2 stationary scans). Data (CSV/JLS) in the old
scratchpad's `ylk/` directory (path as the env above, `…\scratchpad\ylk`).

---

## Project 2: hexagonal vs cubic ice residual entropy (docs/ice.md)

The question: is Onsager's S_h ≥ S_c strict? Kolafa's MC and Xu–Lin–Zhang's tensor networks
(arXiv:2511.22477, D ≤ 7) do not resolve a difference; XLZ claim equality.

**Established.**
* Mᵀ = I M I (I the in-plane inversion) — exact, verified by brute force. Hence S_h = ln λ(I M)
  (symmetric), S_c = ln λ(M), and XLZ's Rayleigh estimator is S_sym = λ((M+Mᵀ)/2), which equals S_h iff
  S_c = S_h — a valid test with half the gap. (M+Mᵀ)/2 = P₊AP₊ − P₋AP₋ with A = I M.
* Zero-flux tori (up to 4 × 4, 2 × 9): S_h > S_c strictly (4 × 4: 3.3e-7 per molecule; 2 × n strips grow
  to 2.1e-6); S_h − S_c ≈ −ln F_I per cell holds to 1–3 % (`examples/ice/ice_exact.jl`,
  `ice_cross_sections.jl`).
* Cell PEPS (library, `ice_site`) D ≤ 3: finite-D errors (~5e-5) exceed the gap; ln F_I −1.15e-5 (D = 2),
  −7.6e-6 (D = 3, χ = 32) (`examples/ice/cell_peps_lane.jl`).
* **Honeycomb formulation** (`examples/ice/`, docs/ice.md "Honeycomb formulation"): BP simple update
  (Vidal gauge = library BP gauge to 1e-10) + infinite CTMRG of the paired networks. Results
  (`examples/ice/results_honeycomb.csv`):

  | D | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|
  | ξ | 1.86 | 1.92 | 2.36 | 2.40 | 2.51 |
  | w_h | 1.5073816 | 1.5073565 | 1.5074139 | 1.5074191 | 1.5074212 |
  | ln F_I per cell | −5.10e-6 | −4.83e-6 | −5.19e-6 | −4.74e-6 | −4.84e-6 |

  ln F_I is flat at ≈ −5e-6 per cell from D = 4 to 8 (S_h − S_c ≈ 2.4–2.6e-6 per molecule if it
  survives D → ∞); w_h is still 3.7e-5 below the literature at D = 8 with shrinking increments (+2.1e-6
  from D = 7): absolute entropies need the variational refinement; ξ grows slowly with D.
* **U(1) is broken by the boundary state** (any update): a zero-flux U(1) PEPS has no finite-D fixed
  point — its virtual charge (the flux through the ribbon under a bond) random-walks, ⟨q²⟩ +0.33 per
  bilayer. Only the arrow-reversal Z2 is exact. The boundary state looks gapless (ξ grows with D).

**Caveats.** F_I is first order in the state error (at D = 2 it ranges −1.4e-5 … −6.3e-5 across
states) and needs χ ≥ D². BP simple update is not variational; a variational pass from it recovers
most of the w_h gap at small D (`honeycomb_variational.jl`).

**Data (2026-09-28 morning).** All numbers are in `examples/ice/results_honeycomb.csv`; the raw
outputs and every checkpoint stay in the old scratchpad (`…\scratchpad\ice\`):
`prod\` — BP-SU states `su_D<D>.jls` (D = 3–10, 12), CTM checkpoints `ctm_D<D>_chi<χ>_<kind>.jls`,
`summary.csv`, `residual.csv`, lane logs; `var\` — optimiser checkpoints `var_D<D>_chi<χ>.jls`
(D = 3 at χ 18, 4 at 32, 5 at 32 and a partial 5 at 50) and per optimum `eval_D<D>_chi<χ>\` (the state,
its CTM checkpoints at χ_opt and 2χ_opt, `summary.csv`, `residual.csv`). Any of it resumes with the
committed drivers (same OUT). Not done locally: BP-SU D = 9 sandwich; variational D = 6+.

**Next — revised 2026-09-28 ([`campaign_3d.md`](campaign_3d.md) §3.4).** Before any cluster time, the local
gates: G0, an inversion-symmetric ansatz for 2(S_h − S_sym) against −ln F_I at D = 2–4; G1, exact tori
4 × 5 and 4 × 6, and a zero-flux-sector product for 5 × 6 and 4 × 8, to resolve the torus ~3e-7 against the
PEPS ~2e-6; G2, implicit gradients. Then the list below, with D up to 10.

**Next (as of the morning of 2026-09-28).** (1) The first cluster campaign
(`examples/ice/cluster/jobs_ice.txt`): variational optima at D = 5–8 with χ_opt = 2D², evaluated at 2χ_opt; BP-SU D = 9–11 and residuals for the record; (2) the
ξ-extrapolation of w_h and ln F_I (`analyse_honeycomb.jl`); (3) S_sym at the same D (the inversion-
symmetric restriction) as the independent check of the gap; (4) a better optimiser (implicit
differentiation of the CTM fixed point for gradients below the current ~1e-4–1e-3 floor); (5) Z2 block
sparsity in the CTM.

---

## GPU readiness (2026-09-27, measured on the local RTX 3070 — FP64 ~1/64 rate, 8 GB)

`examples/ice/honeycomb_profile.jl` (s per warm CTM iteration, the pair route taken, optional
`CUDA.@profile` breakdown; working set by `JULIA_CUDA_HARD_MEMORY_LIMIT`).

* **Correct.** `test/test_gpu_paths.jl` 20/20 on the card (so the "not yet validated" warning in
  `docs/src/advanced.md` is stale — updated). The ice driver on GPU reproduces the CPU summaries exactly.
  With `CUDA.allowscalar(false)` nothing falls back to the host.
* **The pair route.** With the library's default `svd_oversample = 16` the ⟨Iψ|M|ψ⟩ pairs BAIL OUT of
  the subspace SVD (flat spectrum at χ = D²) to the dense route, which forms the n × n quadrant
  (n = χr): 5 of 12 pairs at D = 6, 2 of 8 at D = 7 — **177 s per iteration at D = 7 against 9.2 s with
  `svd_oversample = 64` (1.3χ)**, every pair split, ln κ identical to 10 digits. At D = 12 (n ≈ 41 000) one
  bail-out would stall a job. The driver now defaults to ceil(1.3χ) and logs split/dense/bail-out counts
  per chunk ("BAIL-OUTS — raise SVD_OVERSAMPLE"). This also explains the 300–470 s/it CPU D = 8 sandwich.
* **Memory.** `update`'s default `convergence = :environment` contracts the site environment with all
  four raw legs open — r⁴ = 16D⁸ doubles for the sandwich: 0.2 GB (D = 6), 0.7 GB (D = 7), ~13 GB (D = 10),
  ~55 GB (D = 12) — the largest single allocation. `convergence = :lnkappa` (the driver's default now,
  TOL 2e-14 relative, MINITS 15) never forms it: D = 5 reproduced to 1e-12 in ln κ, identical w_h, ln F_I,
  ξ, in ~25 % fewer iterations. Then the D = 7 sandwich fits in 1.5 GiB and **D = 8 in 5 GiB**. The
  remaining large term is the pair's subspace block, ~2.3χ²r² doubles (still ~D⁸): ~7 GB (D = 10), ~16 GB
  (D = 11), ~32 GB (D = 12) for the sandwich, a working set of a few times that — D ≤ 11 on an 80-GB H100,
  D = 12 on an H200, beyond that batch the block's columns (an easy engine change).
* **Speed (3070, FP64).** Norm / sandwich s per iteration: D = 6 0.57 / 2.3, D = 7 5.9 / 6.8, D = 8 — / 25.5
  (CPU: 2–4 / 10–25, ~6 / 50–100, ~30 / 300–470). At D = 6 the GPU is busy ~65 % of an iteration: ~8 000
  tiny host→device copies, ~13 000 kernel launches, ~500 stream syncs per iteration, and the small SVDs
  run latency-bound in cuSOLVER (thousands of `lasr`/`ormtr` kernels). On an H100 (~100–200× the 3070's
  FP64) that host overhead dominates up to D ≈ 9 and is negligible from D ≈ 10. Rough H100 estimate
  (±3×): sandwich ~2–3 s/it at D = 10, ~10–15 s/it at D = 12; 20–40 iterations per network.
* **Before large D, worth doing:** small factorisations on the host (or batched Jacobi), fewer host
  round-trips per contraction, column batching of the subspace block.
* **Locally now:** D = 8 w_h on the 3070 is ~15 minutes in chunks (`DEVICE=gpu`), no longer out of reach.

## Lessons that cost time (keep)

* **The CTM seed.** `InfiniteCTM2D`'s default all-ones seed fails on overlap networks of Vidal-gauge
  states (ln κ jumping by O(1), ln F_R = +0.31 where the exact value is 0). Seed with e₁ per state leg.
* **`miniter`.** `update` never reports convergence before `miniter` (default 2) iterations: chunked
  runs with `maxiter = 1` need `miniter = 1`.
* **χ ≥ D² for overlaps.** Fast-decaying BP weights do not mean a small environment: ln F_I at D = 6
  moves from −1.5e-5 (χ = 16) to −5.2e-6 (χ = 36).
* **Truncate only in the BP/Vidal gauge.** A stale gauge on the enlarged bonds broke C3 and Z2 and
  made the free energy fall with D.
* **The budget check.** A chunk must be expected to fit (checkpoints record s/iteration) or the run
  is killed mid-chunk and the work lost; the lane treats a timeout kill as "resume".
* **Stationary solves near soft modes wander** (the cubic ice estimator, the Yang–Lee fold): trust f
  and a converged residual, not a stalled |g|.
* **Julia parse traps**: `(a, b = f(); …)` and `(e.ls, e.ln = …)` parse as named tuples — use blocks or
  functions; `@__FILE__ && main()` needs `(@__FILE__)`.

**Checked 2026-09-27 (see "GPU readiness"):** `docs/src/advanced.md` said GPU execution "is not yet validated",
while `test/test_gpu_paths.jl` covers BP, CTM, `InfiniteCTM2D` and the boundary-PEPS solvers on CUDA
(and the ice driver's CUDA path matched the CPU to 13 digits on the local RTX 3070). The test file
passed 20/20 on the RTX 3070 and the page was updated.

---

## Costs (local i9-9900K, 8 cores; per 2D-CTM iteration near convergence, χ = D²)

| network (raw bond) | D = 6 | D = 7 | D = 8 |
|---|---|---|---|
| ⟨ψ\|ψ⟩, ⟨Iψ\|ψ⟩ (D²) | 2–4 s | ~6 s | ~27–30 s |
| ⟨Iψ\|M\|ψ⟩ (2D²) | 10–25 s | 50–100 s | 300–470 s |

~35–40 iterations to converge; cost ~χ³r² ≈ D¹⁰ at χ = D². The Yang–Lee sandwich (cell PEPS, bond 2D²):
~D^9.3, 41.5 s per step at D = 5, χ = 72 (docs/boundary_peps.md "Costs").
