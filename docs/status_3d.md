# 3D classical tensor networks: current state

*Branch `FixesV2`. Rewritten 2026-10-01 as the single current-state document for the 3D work. It replaces
the earlier overnight summaries and `roadmap_3d.md`, `campaign_3d.md` and `paper_3d.md` (all in the git history
before this rewrite). Everything here is measured and dated; the detail is in the topic docs:*

| doc | what |
|---|---|
| [`boundary_peps.md`](boundary_peps.md) | the method: `InfiniteCTM2D`, the boundary-PEPS solvers, validation, 3D Ising, 3D three-state Potts |
| [`yang_lee.md`](yang_lee.md) | the Yang–Lee edge in an imaginary field; the Bethe-vs-Hermitian showcase |
| [`ice.md`](ice.md) | hexagonal vs cubic ice residual entropy |
| [`ctmrg3d.md`](ctmrg3d.md) | the direct 3D CTMRG engines (finite boxes, `InfiniteCTM3D`) — biased, parked |

The **2D** finite CTMRG (`CTMEnvironmentCache`) has its own docs, `ctmrg_status.md` and `finite_ctmrg_design.md`.
The 3D work does not change it (see "Scope" below).

---

## Scope: what the 3D work touches in the library

- **New files only:** `src/MessagePassing/ctm2dinfinite.jl` (`InfiniteCTM2D`), `boundarypeps3d.jl`,
  `ctm3dinfinite.jl`, `ctm3denvironmentcache.jl`, their tests, and the exports in `src/TensorNetworkQuantumSimulator.jl`.
- **One shared-code change:** `CTMEnvironmentCache(net, χ)` sends a network whose vertices are all 3-tuples to
  `CTM3DEnvironmentCache`. The 2D engine accepts only 2-tuple grid vertices, so no 2D call is rerouted.
- Docstring-only edits in `src/Apply/simple_update.jl` and `src/Tensors/ITensorBackend.jl` (docs build).
- Audit, 2026-10-01: loading the package with `--warn-overwrite=yes` reports no method overwrites. The core test
  files ran with 4 Julia threads:
  - constructors, forms, expect, boundarymps, beliefpropagation, apply, sampling, truncate and contraction_sequences
    pass;
  - `test_ctmenvironment.jl` has 4 `@test_logs` failures. In each, the expected warning is present, plus one
    extra info message, "CTM runs its sweeps serially while BLAS is multi-threaded". The 2D engine
    (`ctmenvironmentcache.jl:392`, commit 584601b) logs it only when Julia has more than one thread. This is an
    artefact of the threaded run, not a behaviour change. Rerun single-threaded to confirm;
  - tensors and ctm2dinfinite were still running when this was committed.

---

## Ground rules (the user's standing instructions)

- Commit on `FixesV2`; **never push** — the user pushes. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Slurm: read-only queries only** (`squeue -u $USER`, `sacct`), never in a loop. Never `sbatch`/`scancel`: the user
  submits (Flatiron policy). Prepare job scripts and give the exact `sbatch` line.
- Do not touch the user's untracked files.
- The user runs flat `.jl` scripts; do not build precompile machinery.
- Kill processes by explicit PID, never by a pattern that could match your own command line.
- On a GPU, one job at a time: two jobs on one card ran it out of memory (CUDA.jl's pools do not shrink).
- When truncating a PEPS after applying an operator, fix the gauge (BP/Vidal) first.

**Environment.**
- Linux workstation (A6000, 32 cores): the Julia 1.12.7 binary
  `~/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia` with `--project=<repo>` (the juliaup launcher is
  broken there). CUDA and Adapt come from the default environment `~/.julia/environments/v1.12`.
- The home directory and depot are shared with the Rusty cluster, so cluster jobs run this checkout directly.
- The ice cluster kit's pinned environment (`examples/ice/cluster/Manifest.toml`) is stale against the repo
  (2026-10-01: fails on LogExpFunctions); the Potts kit uses the repo environment instead.
- Windows machine (earlier sessions): an environment that `dev`s this repo, plus CUDA and Adapt; the backend
  packages come from the ITensor registry; set `JULIA_PKG_USE_CLI_GIT=true` if cloning hangs.

---

## The engines (`src/MessagePassing/`)

| piece | what it is | status |
|---|---|---|
| `InfiniteCTM2D` | 1×1 infinite 2D CTMRG; a site is a LIST of layers, never fused; Kikuchi ln κ; `site_environment` gradients; split (matrix-free subspace SVD) pairs, χ³D⁴ per step; `c4v`; `projector = :cut` / `:cycle`; Anderson mixing (opt-in); GPU via `adapt` | production |
| `boundary_peps` (L-BFGS, norm-metric preconditioner, adaptive CTM tolerance, `miniter`), `boundary_peps_krylov` (Newton–Krylov, subspace trust region), `boundary_peps_stationary` (FD Jacobian) | 3D models by a 1×1 boundary iPEPS of the layer transfer operator: the **Bethe estimator** f = ln κ⟨L\|T\|R⟩ − ln κ⟨L\|R⟩ with ⟨L\| = ⟨R̄\| (real symmetric T: variational), Rᵀ (complex symmetric, `bilinear`), or π(R) (permuted bras); symmetry groups `:c4v`, `:diagonal`, `:none` | production; Newton–Krylov is the fastest accurate route |
| `ising3d_site`, `potts3d_site`, `ice_site` | 3D Ising (real or complex field), q-state Potts (with the exact ∂site/∂β), the ice bilayer cell | tested |
| `InfiniteCTM3D`, `CTM3DEnvironmentCache` | direct 3D CTMRG | works, but the infinite fixed point is biased ~1 % (one χ-index per quarter-plane cannot hold a 2D boundary); parked |

**Drivers (`examples/`).**
- 3D Ising: `ising3d_boundary_peps_benchmark.jl`, `ising3d_solver_benchmark.jl`.
- Potts: `potts3d_boundary_peps.jl` — one branch per run; `PT_SAVE` (states per β), `PT_INIT` (warm or embedded
  start), `PT_BUDGET` (resumable).
- Yang–Lee (`examples/yang_lee/`):
  - `scan_krylov.jl` — edge maps, GPU, resumable chunks;
  - `fold_pseudoarclength.jl` + `fold_run.jl` — the exact fold;
  - `bethe_vs_hermitian.jl`, `bethe_vs_hermitian_fold.jl` — the estimator showcase;
  - `fold_xi_scaling.jl`, `analyse_maps.jl` — fits to the maps; `probe_cost.jl` — CTM-step cost against D and χ.
- Ice: `examples/ice/`, see `ice.md`.

**Cluster kits.**
- `examples/potts_cluster/`: job array, resumable, self-requeue. Job lists `jobs_potts.txt` (D = 5), `jobs_d5b.txt`
  (at the crossing), smoke list.
- `examples/ice/cluster/`: its environment is stale, see above.

**Tests.** `test/test_ctm2dinfinite.jl`, `test_boundarypeps3d.jl`, `test_ctm3denvironment.jl`, `test_ice.jl` (~6 min),
`test_ice_honeycomb.jl` (~4 min), `test_gpu_paths.jl` (skipped without CUDA).

---

## Results (dated)

### 3D Ising (`boundary_peps.md`, "3D Ising")

- m(β) against Monte Carlo in the thermodynamic limit, D = 2–4.
- D = 2 (χ = 16) has Monte Carlo quality for β ≥ 0.25. Its pseudo-T_c lies between β = 0.220 and 0.221 (Vanderstraeten et
  al.'s D = 2 value).
- Near β_c, D = 4 (χ = 48, GPU) halves D = 3's error: at β = 0.2225, 4.3e-4 against 1.5e-3.

### 3D three-state Potts, the first-order transition (`boundary_peps.md`, "3D three-state Potts")

The method: two branches (ordered and disordered continuations); β_t is where their free energies cross; Q is
e_dis − e_ord there.

| D, χ | β_t | Q | where |
|---|---|---|---|
| 3, 27 | 0.550408 | 0.1891 | CPU |
| 4, 48 | 0.550506(2) | 0.168(1) | A6000; crossing bracketed by both branches |
| 4, 64 | ≈ 0.550522 | ≈ 0.169 | A6000; the disordered branch is ~10× more χ-sensitive than the ordered one |
| 5, 75 | ≈ 0.550548 | 0.169–0.170 (extrapolated) | Rusty H200, job 7142983; the ordered spinodal lies within ~1e-4 of β_t |
| Monte Carlo (Janke–Villanova 1997; Bazavov–Berg 2007) | 0.550565(10) | 0.1614(3); 0.1643(8) | |

- β_t is converging onto Monte Carlo: the D = 3, 4, 5 shifts are 9.8e-5 and 4.2e-5.
- β_t needs χ convergence at each D: the χ shift at D = 4 (1.7e-5) equals D = 5's remaining gap.
- Q does not improve from D = 4 to D = 5. The finite-D ordered branch collapses just below β_t, where e_ord is steep.
- **Running:** `jobs_d5b.txt` (ordered and disordered points at β = 0.5505 and 0.55045, 3000 iterations) turns the
  extrapolation into an interpolation.

### The Yang–Lee edge (`yang_lee.md`)

- **Maps.**
  - D = 2 at β = 0.16–0.21; D = 3 at β = 0.18, 0.20, 0.21; D = 4 at β = 0.20 and 0.21 (χ = 48; a χ = 64 check moves Im m by
    ≤ 3.5e-5 at the fold).
  - Data: `examples/yang_lee/data/`.
- **The Bethe estimator against the Hermitian one** (2026-10-01; D = 2, β = 0.20–0.21). T is complex symmetric, not Hermitian.
  - **Stationarity:** under R = R* + ηX, the Bethe f changes as η² and the Hermitian f as η. At η = 1e-4 the Bethe
    change is 6300× smaller (14 000× at the fold).
  - **The edge:** the Hermitian impurity Im m is 7–9× off, and it is smooth through the fold. The Hermitian estimator
    does not see the Yang–Lee edge.
  - At the fold, the soft mode's curvature is ~2000× flatter than a random direction's.
- **The fold fit is the weak point.**
  - Six-point square-root fits drift by 0.4–0.8 % with the fit window, more than the D = 3 → 4 shifts (−0.03 %,
    −0.16 %). "The fold stays put from D = 3 to 4" is therefore **not established**.
  - The same applies to |z_c| = 2.45–2.49, against the functional-RG value 2.43(4).
  - The impurity m and the Bethe f agree (d Re f/dθ = −Im m to the quadrature error).
- **The exact fold** (pseudo-arclength through the turning point, no fit): θ_f(D = 2, β = 0.20) = 0.016738(2), in 19 min on
  the CPU.
- **Running:** the exact fold at D = 3, β = 0.20 (CPU); both estimators at D = 3 (CPU) and D = 4 (A6000) on the same θ, for
  the truncation version of the showcase.

### Ice Ih vs Ic (`ice.md`)

- Established: Mᵀ = I M I; S_h > S_c strictly on zero-flux tori.
- Honeycomb BP-SU: ln F_I ≈ −5e-6 per cell, flat over D = 4–8.
- Variational states at D = 3–5 have a nonzero inversion asymmetry, but they scatter ±1.2e-6 without a trend.
- Xu–Lin–Zhang (arXiv:2511.22477) argue S_h = S_c exactly (the operator is normal), using a symmetrised Rayleigh estimator
  λ((M+Mᵀ)/2) — not a two-sided Bethe functional.
- Data: `examples/ice/results_honeycomb.csv`. Paused.

---

## Prior work and positioning (literature check 2026-09-27 and 2026-10-01)

**3D q = 3 Potts** (β = 1/T):

| work | method | β_t | Q |
|---|---|---|---|
| Janke & Villanova 1997 | Monte Carlo, L = 36 | 0.550565(10) | 0.1614(3) |
| Bazavov & Berg, PRD 75, 094506 (2007) | Monte Carlo, L = 50 | 0.5505653(58) | 0.1643(8) |
| Monte Carlo 1991 | L = 36 | 0.550523(11) | 0.16062(52) |
| Gendiar & Nishino 2002 | TPVA | 0.5496 | 0.228 |
| Wang, Xie, Chen, Normand & Xiang, CPL 31, 070503 (2014), arXiv:1405.1179 | HOTRG, D = 21 | 0.55048(15) | 0.2029 (D = 14) |
| Jha, arXiv:2201.01789 (2022) | triad TRG | 0.55021(45) | — |

- Our D = 5 β_t is ~9× closer to Monte Carlo than HOTRG's.
- The Monte Carlo latent heats disagree with each other: 0.1614(3) and 0.1643(8) differ by 3.4σ. A Q converged in D would
  arbitrate.
- Chen, Liu, Deng & Zhang (arXiv:2509.23945, 2025): tensor-network MCMC crosses the 3D Potts barrier at finite size,
  up to 64³.

**3D tensor-network methods:**

| family | best recent result | lesson |
|---|---|---|
| boundary PEPS, variational | Vanderstraeten–Vanhecke–Verstraete 2018 (1805.10598), D ≤ 4; scaling collapse in ξ(D, χ), Vanhecke et al. 2022 (2102.03143) | χ need not be converged with a ξ collapse |
| boundary PEPS, split CTMRG | **Xu–Lin–Zhang, PRB 112, 134403 (2025), arXiv:2506.19339**: split CTMRG for the triple layer, O(χ³D³d³), L-BFGS, 3D Ising D = 6, χ = 100, T_c 3e-4 off; ice (2511.22477), D ≤ 7 | **"split CTMRG" is theirs first** — cite them; compare our pair split |
| boundary PESS by simple update | Yang, Fu, Xie, Xiang 2023 (2210.09896): 3D Ising D = 20, T_c to 1e-4 | single-layer overlap networks |
| coarse graining | HOTRG (1405.1179); ATRG (1906.02007) | good T_c, poor exponents |
| BP and loop series | 2409.03108, 2510.02290; degrade at criticality (2604.03228) | warm starts only |
| GPU CTMRG | QR-CTMRG (2505.00494): D = 8, χ = 700 on one H100; SI-CTMRG (2607.15158) | decompositions can be made negligible; what remains is FP64 GEMM |

**What is new here:**
- **The Bethe (two-sided) estimator on non-Hermitian 3D transfer matrices:** ε² against ε, demonstrated; the
  Hermitian estimator misses the Yang–Lee edge.
- **The first converged 3D tensor-network first-order transition:** β_t at Monte Carlo precision, metastable branches and
  spinodals directly.
- **Scale:** GPU, C4v and the preconditioner; D = 5 points in hours on an H200; D = 8, χ = 128 at 25 s per step on an A6000.
- **No tensor-network study of the 3D Yang–Lee edge exists;** Monte Carlo cannot reach it. The universal location has only
  functional RG (2.43(4), 2203.16651) and analytic continuation (2.429(56)).

---

## Paper plan and next steps

**Framing (the user's, 2026-09-30):**
- **Centre:** better estimators, large GPUs and new algorithmic ideas raise what 3D tensor-network contraction can do.
- **Headline:** one well-converged result.
- **Support:** several smaller results.

As of 2026-10-01 the user's priority is the **Bethe estimator and a non-Hermitian 3D network under control**
(Yang–Lee), or the paper is incremental.

**Figures:**
1. The cost and scaling of the contraction (split pairs, C4v, GPU).
2. The Bethe vs Hermitian stationarity scans (η² against η), away from the fold and at it.
3. Yang–Lee: Im m through the fold, Bethe against Hermitian; exact folds against D; ζ(t) → |z_c|.
4. Potts: f_ord − f_dis, the latent heat, β_t and Q against D (and χ).
5. Ising m(β) at D = 2–4 (5–6).

**Next, in order:**
1. Exact folds at D = 3 and 4 (D = 5 on the cluster) at β = 0.20 and 0.21, plus a third β (0.215). Two things are missing:
   - a bordered Newton fold solve, to converge onto the turn;
   - a resumable GPU port of the fold solver.

   This decides whether |z_c| becomes a controlled number (~1 %, the first lattice value).
2. The truncation version of the showcase: Bethe and Hermitian errors at D = 2, 3 against D = 4.
3. Potts:
   - the `jobs_d5b.txt` interpolation;
   - χ = 100 at D = 5;
   - the ordered branch near its spinodal (Q depends on it).
4. Before claiming novelty, check whether Xu–Lin–Zhang use a two-sided functional anywhere.
5. An H100 scaling and memory benchmark; Ising at D = 5–6.

**Targets beyond the paper** (from the 2026-09-28 campaign):
- **The Yang–Lee edge against a first-order transition** — edge and spinodal compared in the 3D Potts model. In mean-field
  theory they coincide (An–Mesterházy–Stephanov, 1707.06447); here both are computable on the same footing.
- **Validation in the Yang–Lee class:** the cubic monomer–dimer model at negative activity, z₀ = −0.0520268(2)
  (Butera–Pernici); a sign problem for Monte Carlo, none for us.
- **3D Ising interface tension and roughening:** the overlap of the ± fixed points; check first in 2D against Onsager.
- **Three-state Potts at complex field:** the heavy-dense QCD effective model (2601.06446).
- **The 3D Z2 gauge–Higgs self-dual line:** XY* or a new CFT (2112.01824, 2012.15845); needs new tensors and multi-site cells.

---

## GPU readiness (2026-09-27, measured on the local RTX 3070 — FP64 ~1/64 rate, 8 GB)

The profiling script is `examples/ice/honeycomb_profile.jl`: seconds per warm CTM iteration, the pair route taken, an
optional `CUDA.@profile` breakdown, and the working set via `JULIA_CUDA_HARD_MEMORY_LIMIT`.

- **Correct.**
  - `test/test_gpu_paths.jl` passes 20/20 on the card.
  - The ice driver reproduces the CPU summaries exactly on the GPU.
  - With `CUDA.allowscalar(false)`, nothing falls back to the host.
- **The pair route.** With `svd_oversample = 16`, the ⟨Iψ|M|ψ⟩ pairs bail out of the subspace SVD at χ = D² (flat
  spectrum) to the dense route.
  - Cost: 177 s per iteration at D = 7, against 9.2 s with `svd_oversample` = 1.3χ.
  - The ice driver now defaults to ⌈1.3χ⌉ and logs bail-outs.
- **Memory.** `update`'s default `convergence = :environment` forms the site environment with four raw legs open:
  16D⁸ doubles for the sandwich, ~13 GB at D = 10. `convergence = :lnkappa` never forms it.
  - The remaining large term is the pair's subspace block, ~2.3χ²r² doubles.
  - That allows D ≤ 11 on an 80-GB H100 and D = 12 on an H200.
- **Speed (3070).** Norm / sandwich, s per iteration: D = 6 0.57 / 2.3; D = 7 5.9 / 6.8; D = 8 — / 25.5. Host overhead
  (small copies, kernel launches, latency-bound small SVDs) dominates up to D ≈ 9 on an H100.
- **Measured 2026-09-30 (Potts):**
  - D = 3, χ = 27: an A100 or H100 converges a cold point in 5–6 min, against 13–28 min on the workstation CPU.
  - D = 4: an H200 reaches the A6000's 7-hour ordered point in about 1 hour.
  - D = 5, χ = 75: 2–7 h per point on an H200.
- On the GPU, `_bp_evaluate` runs the norm and sandwich environments one after the other. Run concurrently, two jobs
  sharing a card died in `cuStreamCreate` and `cuStreamIsCapturing`.

## Lessons that cost time (keep)

- **The CTM seed.** `InfiniteCTM2D`'s all-ones seed fails on overlap networks of Vidal-gauge states. Seed with e₁ per state
  leg.
- **`miniter`.**
  - `update` never reports convergence before `miniter` iterations.
  - A smaller-D optimum embedded into a larger D is nearly stationary, so `boundary_peps` needs `miniter`
    (default 20 after an embedding) or it stops at once.
- **χ ≥ D² for overlaps;** χ convergence can move a first-order β_t by as much as one step in D (Potts, above).
- **Truncate only in the BP/Vidal gauge.**
- **Fit-free beats fitted.** A square-root fit to a fold drifts with its window by more than the effect being measured.
  Locate turning points from the stationary equations.
- **Soft scope in flat scripts:** top-level loops that assign globals lose their results, or fail on `global` inside a
  loop. Wrap drivers in functions.
- **Thread oversubscription:** CPU drivers need `BLAS.set_num_threads(1)` when Julia threads are used. Without it one run
  took ~16 cores and slowed everything 10×.
- **Log watchers:** the filter must match every terminal state (`ERROR`, `Segmentation`, `Killed`, `Out of GPU memory`).
  One crash went unnoticed for 2.5 h, and finished rows were missed when the filter was too narrow.
- **Julia parse traps:** `(a, b = f(); …)` parses as a named tuple; `@__FILE__ && main()` needs `(@__FILE__)`.

## Costs

- Workstation i9-9900K (8 cores), per 2D-CTM iteration near convergence, χ = D²:
  - ⟨ψ|ψ⟩ (bond D²): 2–4 s at D = 6, ~6 s at D = 7, ~27–30 s at D = 8;
  - ⟨Iψ|M|ψ⟩ (2D²): 10–25 s, 50–100 s, 300–470 s respectively;
  - ~35–40 iterations to converge; cost ~χ³r² ≈ D¹⁰ at χ = D².
- The A6000 (2026-09): 12–38× the CPU from D = 4 up; with C4v, D = 6, χ = 72 runs at 0.86 s per step.
