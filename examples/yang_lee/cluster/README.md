# Cluster kit: the Yang–Lee folds and the estimator showcase

Physics and status: `docs/yang_lee.md`, `docs/status_3d.md`.

**Why.** The six-point fits that located the D = 3 and D = 4 folds drift with their window by 0.4–0.8 %. That is
more than the D = 3 → 4 shift they were meant to measure, so whether the finite-D fold moves with D (and hence
|z_c|) is open. `fold_cluster.jl` locates each fold with no fit to the maps: it walks the stationary branch
through the turning point by pseudo-arclength, then refines over the turn in steps ~6× shorter.

- Validated at D = 2, β = 0.20 (2026-10-15, CPU): θ_f = 0.01673710; local-cubic rms 4e-8; the highest point reached
  is 0.01673709.
- The old six-point fit gave 0.0167357.

| file | what |
|---|---|
| `fold_cluster.jl` | the exact fold: resumable (checkpoint after every accepted step), GPU or CPU, start from `INIT` (a `scan_krylov.jl` checkpoint) or `THS` (continuation from θ = 0) |
| `yl_array.slurm` | the job array: one line of the job list per GPU; fold tasks requeue themselves at the budget |
| `jobs_yl.txt` | folds at D = 2 (β = 0.21), D = 3 (β = 0.20, 0.21), D = 4 (β = 0.20, 0.21, from the saved maps); the D = 3 and D = 4 showcase |
| `seeds/` (git-ignored) | the D = 4 map checkpoints `ylk3_beta{0.2,0.21}_D4_chi48.jls`, copied from `~/3d_runs_2026-10-01/yl/` |

## Steps (Claude does not submit jobs)

From this directory (the repo and the depot are in the shared home):

    sbatch --array=1-7 yl_array.slurm

Watch the jobs:

    squeue -u $USER
    tail -f logs/yanglee_<jobid>_<task>.out

A fold task prints walk points, then refine points, then a `FOLD …` line. That line is also written to
`runs/fold_D<D>_beta<β>_chi<χ>.result`, and every point goes to the matching `.csv`.

## Expected costs (unmeasured on the cluster)

- D = 2: minutes. D = 3: about an hour.
- D = 4: each step needs a finite-difference Jacobian, 110 columns × 2 evaluations, done one after another on a GPU.
  Expect 0.5–1 h per step on an H100/H200, and ~15–25 steps, so a requeue or two.
- The showcase tasks are not resumable. D = 4 (the θ = 0 climb plus three points) took ~1.5 h of climbing on the
  A6000 alone; expect a few hours on an H100.

## Notes

- `REM` (1e-4) rejects a step that leaves the PT-symmetric branch (|Re m| too large). At D = 4 near the fold, Re m drifts
  to 1–4e-5 at the gradient's noise floor. If a D = 4 task keeps rejecting steps with Re m between 1e-4 and 1e-3, set
  `REM=1e-3` in the job's environment.
- The refinement starts from the point before the highest one and spans to the point after it. If the top lands on
  the last refined point, the FOLD line says so; then raise `NREF`.
