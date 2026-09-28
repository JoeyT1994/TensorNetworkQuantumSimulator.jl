# 3D classical tensor networks: status and handoff

*Written 2026-09-27 (evening) for the next agent. Branch `FixesV2`. Everything below is measured and
dated; where a number lives in another doc, that doc has the detail.*

Where this can go — the literature, the levers ranked, the targets ranked, and a sequence:
[`roadmap_3d.md`](roadmap_3d.md).

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
  | w_h | 1.5073816 | 1.5073565 | 1.5074139 | 1.5074191 | — |
  | ln F_I per cell | −5.10e-6 | −4.83e-6 | −5.19e-6 | −4.74e-6 | −4.84e-6 |

  ln F_I is flat at ≈ −5e-6 per cell from D = 4 to 8 (S_h − S_c ≈ 2.4–2.6e-6 per molecule if it
  survives D → ∞); w_h is still 4e-5 below the literature at D = 7; ξ grows slowly with D.
* **U(1) is broken by the boundary state** (any update): a zero-flux U(1) PEPS has no finite-D fixed
  point — its virtual charge (the flux through the ribbon under a bond) random-walks, ⟨q²⟩ +0.33 per
  bilayer. Only the arrow-reversal Z2 is exact. The boundary state looks gapless (ξ grows with D).

**Caveats.** F_I is first order in the state error (at D = 2 it ranges −1.4e-5 … −6.3e-5 across
states) and needs χ ≥ D². BP simple update is not variational; a variational pass from it recovers
most of the w_h gap at small D (`honeycomb_variational.jl`).

**Running at handoff.** Lane A (`ice_prodlane.sh` in the old scratchpad, same code as the committed
driver) is computing D = 9 ⟨ψ|ψ⟩ (χ = 81) and stops by itself after it. Its output and every
checkpoint are in `…\scratchpad\ice\prod\` (`summary.csv`, `results.csv`, `ctm_D<D>_chi<χ>_<kind>.jls`,
`su_D<D>.jls` for D = 6–10, 12, lane logs). Partial: D = 8 ⟨Iψ|M|ψ⟩ at 4 iterations (≈ 470 s per
iteration here — does not fit ten-minute runs), D = 9 ⟨Iψ|ψ⟩ at a few iterations. Resume any of them
with the committed driver:

    OUT=<that prod dir> D=9 CHI=81 KINDS=inv CHUNK=1 TOL=1e-9 \
      julia --project=<env> examples/ice/honeycomb_prod.jl       # or examples/ice/honeycomb_lane.sh

**Next.** (1) D = 8–12 (χ ≥ D², and 1.5D² checks) on GPUs — `honeycomb_prod.jl` with `DEVICE=gpu` as a job
array over (D, χ, kind); CUDA path validated on the local RTX 3070 (identical to 13 digits); (2) the
ξ-extrapolation of w_h and ln F_I (finite-correlation-length scaling; the boundary state is gapless);
(3) S_sym at the same D (the inversion-symmetric restriction) as the independent check of the gap;
(4) Z2 block sparsity in the CTM.

---

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

**Possibly stale, not yet checked:** `docs/src/advanced.md` says GPU execution "is not yet validated",
while `test/test_gpu_paths.jl` covers BP, CTM, `InfiniteCTM2D` and the boundary-PEPS solvers on CUDA
(and the ice driver's CUDA path matched the CPU to 13 digits on the local RTX 3070). Re-run
`test_gpu_paths.jl` on hardware before editing the user-facing page.

---

## Costs (local i9-9900K, 8 cores; per 2D-CTM iteration near convergence, χ = D²)

| network (raw bond) | D = 6 | D = 7 | D = 8 |
|---|---|---|---|
| ⟨ψ\|ψ⟩, ⟨Iψ\|ψ⟩ (D²) | 2–4 s | ~6 s | ~27–30 s |
| ⟨Iψ\|M\|ψ⟩ (2D²) | 10–25 s | 50–100 s | 300–470 s |

~35–40 iterations to converge; cost ~χ³r² ≈ D¹⁰ at χ = D². The Yang–Lee sandwich (cell PEPS, bond 2D²):
~D^9.3, 41.5 s per step at D = 5, χ = 72 (docs/boundary_peps.md "Costs").
