#!/bin/bash
# Slurm job array for the 3D three-state Potts campaign (examples/potts3d_boundary_peps.jl): one line of
# the job list per array task, one GPU each. Every task RESUMES (PT_SAVE=1: finished β are read back, an
# unfinished β continues from its partial state), so resubmitting the same array continues the unfinished
# tasks and finished ones exit in minutes. Submit from this directory:
#
#   sbatch --array=1-$(bash ../ice/cluster/jobs.sh count jobs_potts.txt) potts_array.slurm
#
# A task runs the driver with PT_BUDGET = the wall time minus MARGIN_S; when the driver reports an
# unfinished β ("(budget") the task requeues itself if REQUEUE=1. Env overrides: JOBS (jobs_potts.txt),
# RUNDIR (./runs), WALL_S (must match --time, in s), MARGIN_S (900), REQUEUE (1), DRYRUN (1: print only).
#SBATCH --job-name=potts3d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100-80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -u
KIT=${KIT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
REPO=$(cd "$KIT/../.." && pwd)
JOBS=${JOBS:-$KIT/jobs_potts.txt}
RUNDIR=${RUNDIR:-$KIT/runs}; mkdir -p "$RUNDIR" "$KIT/logs"
WALL_S=${WALL_S:-172800}; MARGIN_S=${MARGIN_S:-900}
TASK=${SLURM_ARRAY_TASK_ID:?"run as an array task (or set SLURM_ARRAY_TASK_ID)"}
LINE=$(bash "$REPO/examples/ice/cluster/jobs.sh" line "$JOBS" "$TASK")
[ -z "$LINE" ] && { echo "task $TASK: no such job line"; exit 1; }
read -r BRANCH D CHI BETAS INIT TAG <<< "$LINE"
# the repo's own environment, as on the workstation (home and the depot are shared with the cluster).
# Not the ice kit's pinned one: its Manifest is stale against the repo (2026-09-30: LogExpFunctions).
export JULIA_PROJECT="$REPO"
INITPATH=""; [ "$INIT" != "-" ] && INITPATH=$(cd "$KIT" && realpath "$INIT")
BUDGET=$((WALL_S - MARGIN_S - 120))        # 120 s for Julia to load
# the juliaup launcher is unreliable here: default to the pinned binary in the (shared) home, else `julia`
JL=${JULIA:-$HOME/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia}; [ -x "$JL" ] || JL=julia
CMD=(env PT_BRANCH=$BRANCH PT_D=$D PT_CHI=$CHI PT_BETAS=$BETAS PT_TAG=$TAG PT_GPU=1 PT_SAVE=1
     PT_INIT=$INITPATH PT_BUDGET=$BUDGET "$JL" --startup-file=no -t 4 "$REPO/examples/potts3d_boundary_peps.jl")
echo "$(date) task $TASK on $(hostname): $LINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN (in $RUNDIR): ${CMD[*]}"; exit 0; fi
cd "$RUNDIR"
# (PT_INIT only seeds a β with no state on disk: on a resume the finished and partial states take over)
"${CMD[@]}" 2>&1 | tee "task$TASK.last"
if grep -q "(budget" "task$TASK.last" && [ "${REQUEUE:-1}" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "$(date) task $TASK: β unfinished at the budget, requeueing"
  scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK}"
fi
echo "$(date) task $TASK: finished this allocation"
