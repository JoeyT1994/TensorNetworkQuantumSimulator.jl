# Cluster kit: the 3D three-state Potts campaign (D = 5, 6)

The paper's headline result is the Potts first-order transition (β_t, latent heat Q) converged in D.

| D, χ | β_t | Q | where |
|---|---|---|---|
| 3, 27 | 0.550408 | 0.1891 | workstation CPU |
| 4, 48 | 0.550506(2) | 0.168(1) | A6000 |
| Monte Carlo | 0.550565(10) | 0.16160(47) | Janke & Villanova 1997 |

This kit runs D = 5 (and then 6) on the Rusty GPU partition. Each task runs one branch through three β
around the crossing, on one GPU. Physics and the D ≤ 4 data: `docs/boundary_peps.md`, "3D three-state
Potts".

| file | what |
|---|---|
| `potts_array.slurm` | the job-array script: one line of a job list per task; resumes; requeues itself at the budget |
| `jobs_potts.txt` | the campaign: D = 5, χ = 75, both branches (2 tasks); the second wave commented out |
| `jobs_smoke.txt` | 2 cold D = 3 tasks with known answers (~15–30 min) |

## Steps (Claude does not submit jobs; these are for you to run)

The repo and the Julia depot live in the shared home, so nothing needs to be copied. The job uses this
checkout (`--project` = the repo) and the pinned Julia 1.12.7 binary under `~/.julia/juliaup`. Set
`JULIA=...` to override the binary.

1. Smoke test, from this directory. Compare f with the numbers in `jobs_smoke.txt`:

       sbatch --array=1-2 --time=01:00:00 --export=ALL,JOBS=$PWD/jobs_smoke.txt,WALL_S=3600 potts_array.slurm

2. The campaign. `--time` and `WALL_S` must agree; the default is 2 days:

       sbatch --array=1-2 potts_array.slurm

3. Watch:

       squeue -u $USER
       tail -f logs/potts3d_<jobid>_<task>.out

   One row prints per converged β. `(budget: …)` means the task saved an unfinished β and requeued itself.
4. Results go to `runs/`:
   - `d5_D5_chi75_{ordered,disordered}.csv`, one row per β;
   - `*_beta<β>.jls`, the states;
   - `*_partial.jls`, unfinished points.

   The crossing is reported by the driver once both CSVs exist. Or run
   `PT_TAG=d5 PT_D=5 PT_CHI=75 PT_BETAS=` … by hand. Tell Claude when rows land and it will do the analysis.

## Notes

- **GPUs.** Constrained to H100/H200/A100-80GB, all strong in FP64. Not the RTX Pro 6000 Blackwell nodes, or
  the A6000-class cards, which run FP64 at 1/64 rate. The cost scales as χ³D⁴: D = 5, χ = 75 is about 9× D = 4,
  χ = 48 per step. D = 4 took 25 min to 7 h per point on the A6000, so expect well under that per point on an
  H100 once warm.
- **Memory.** Not yet measured at D = 5; 80 GB is the assumption. If a task dies out of memory, add `h200` only
  (141 GB) with `--constraint=h200`.
- **Seeds.** The commented lines in `jobs_potts.txt` warm-start D = 5 from saved D = 4, χ = 64 states (being made
  on the workstation). That is cheaper than the cold D = 3 → 4 → 5 climb. Put the files in `seeds/`, which
  is ignored by git like `runs/` and `logs/`.
- **Resume.** A task stops at `PT_BUDGET` (the wall time minus 17 min) between L-BFGS iterations. It saves the
  unfinished β and requeues. The rerun continues from that state, with the L-BFGS memory restarted.
  Resubmitting the same array is safe.
- Dry run without Slurm or Julia:

      DRYRUN=1 SLURM_ARRAY_TASK_ID=1 bash potts_array.slurm
