# Cluster kit: the honeycomb ice campaign

Everything needed to run the ice networks on a Slurm GPU cluster (H100/H200), one (D, χ, network) or
one variational optimum per GPU, resumable across allocations. Physics and status: `docs/ice.md`,
`docs/status_3d.md`; costs: [`cost_table.md`](cost_table.md).

| file | what |
|---|---|
| `Project.toml`, `Manifest.toml` | the pinned environment (this repo by relative path, CUDA, Adapt) |
| `setup.sh` | one-time, on a login node: registries (General + ITensor), instantiate, precompile, CUDA runtime |
| `ice_array.slurm` | the job-array script: one line of the job list per task, resumes, requeues itself near the wall |
| `jobs.sh` | `jobs.sh count FILE`, `jobs.sh line FILE N` (the list without comments) |
| `jobs_ice.txt` | the first campaign (13 tasks) |
| `jobs_smoke.txt` | a 2-task smoke test (~5 min) with known answers |
| `cost_table.md` | measured and extrapolated s/iteration and memory per (D, χ, network) |
| `../analyse_honeycomb.jl` | collect `summary.csv`/`residual.csv` from the run directories, tabulate, ξ-extrapolate |

## Steps

1. Get the code there (the user pushes FixesV2; `git pull` on the cluster). The kit lives in
   `examples/ice/cluster/`; the environment refers to the repo as `../../..`.
2. On a login node: `./setup.sh` (set `JULIA_DEPOT_PATH` to a shared filesystem first if compute nodes
   do not see your home).
3. Smoke test (compare with the numbers in `jobs_smoke.txt`):

       sbatch --array=1-2 --time=00:30:00 --export=ALL,JOBS=$PWD/jobs_smoke.txt,WALL_S=1800 ice_array.slurm

4. The campaign (≤ 16 GPUs at once; the `--time` and `WALL_S` must agree, in seconds):

       sbatch --array=1-$(./jobs.sh count jobs_ice.txt)%16 ice_array.slurm

   Tasks near the wall time requeue themselves (`REQUEUE=1`); resubmitting the same array is also safe —
   finished jobs exit in seconds.
5. Watch: `squeue -u $USER`, `tail -f logs/ice-honeycomb_<jobid>_<task>.out`. Every CTM chunk prints its
   pair routes; "BAIL-OUTS — raise SVD_OVERSAMPLE" means a pair fell back to the dense route (fix before
   continuing: it is ruinous at large D).
6. Collect: `julia ../analyse_honeycomb.jl runs/prod runs/var/eval_*`.

## Layout of `runs/`

`runs/prod/`: BP simple-update states `su_D<D>.jls`, per-network checkpoints `ctm_D<D>_chi<χ>_<kind>.jls`,
`results.csv` (one row per converged network), `summary.csv` (w_h, ln F_I, ξ), `residual.csv` (ln f).
`runs/var/`: optimiser checkpoints `var_D<D>_chi<χ>.jls`, and `eval_D<D>_chi<χ>/` — the optimum as a state
plus its own `summary.csv`/`residual.csv`.

## Knobs (environment of `honeycomb_prod.jl` / `honeycomb_variational.jl`)

`SVD_OVERSAMPLE` (⌈1.3χ⌉), `CONV` (`lnkappa`) and `TOL` (2e-14 relative), `MINITS` (15), `CHUNK` (CTM
iterations per checkpoint), `BLASN`; variational: `GTOL`, `FTOL`, `POLISH`, `MAXSTEP`, `PRECONDITION`.
Dry run of a task without Julia: `DRYRUN=1 SLURM_ARRAY_TASK_ID=3 ./ice_array.slurm`.
