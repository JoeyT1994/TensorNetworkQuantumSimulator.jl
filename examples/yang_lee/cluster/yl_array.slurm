#!/bin/bash
# Slurm job array for the Yang–Lee jobs (jobs_yl.txt): one line per array task, one GPU each. Submit from this
# directory:
#
#   sbatch --array=1-$(bash ../../ice/cluster/jobs.sh count jobs_yl.txt) yl_array.slurm
#
# fold tasks RESUME (fold_cluster.jl checkpoints every accepted arclength step): a task stops at its budget (the
# wall time minus MARGIN_S), prints "(budget", and requeues itself if REQUEUE=1; resubmitting the array is safe.
# showcase tasks are not resumable (a few hours on an H100/H200 at D = 4). Env overrides: JOBS, RUNDIR (./runs),
# WALL_S (must match --time, s), MARGIN_S (900), REQUEUE (1), DRYRUN (1: print only).
#SBATCH --job-name=yanglee
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200|a100-80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -u
KIT=${KIT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
REPO=$(cd "$KIT/../../.." && pwd)
JOBS=${JOBS:-$KIT/jobs_yl.txt}
RUNDIR=${RUNDIR:-$KIT/runs}; mkdir -p "$RUNDIR" "$KIT/logs"
WALL_S=${WALL_S:-172800}; MARGIN_S=${MARGIN_S:-900}
TASK=${SLURM_ARRAY_TASK_ID:?"run as an array task (or set SLURM_ARRAY_TASK_ID)"}
LINE=$(bash "$REPO/examples/ice/cluster/jobs.sh" line "$JOBS" "$TASK")
[ -z "$LINE" ] && { echo "task $TASK: no such job line"; exit 1; }
read -r KIND D CHI BETA SPEC EXTRA <<< "$LINE"
EXTRA_ENV=(); [ -n "${EXTRA:-}" ] && IFS=, read -r -a EXTRA_ENV <<< "$EXTRA"   # optional 6th column: K=V,K=V
export JULIA_PROJECT="$REPO"               # the repo's environment (home and depot shared with the workstation)
JL=${JULIA:-$HOME/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia}; [ -x "$JL" ] || JL=julia
BUDGET=$((WALL_S - MARGIN_S - 120))
case "$KIND" in
  fold)
    case "$SPEC" in
      ths:*)  START=(THS="${SPEC#ths:}") ;;
      init:*) START=(INIT="$(cd "$KIT" && realpath "${SPEC#init:}")") ;;
      *) echo "fold SPEC must be ths:… or init:…, got $SPEC"; exit 1 ;;
    esac
    WORK="$RUNDIR"
    CMD=(env GPU=1 BETA=$BETA D=$D CHI=$CHI BUDGET=$BUDGET "${START[@]}" "${EXTRA_ENV[@]}"
         "$JL" --startup-file=no -t 4 "$REPO/examples/yang_lee/cluster/fold_cluster.jl") ;;
  showcase)
    WORK="$RUNDIR/showcase_D${D}_beta${BETA}"; mkdir -p "$WORK"
    CMD=(env GPU=1 BETA=$BETA DS=$D CHIS=$CHI THS=$SPEC PERTURB=0 "${EXTRA_ENV[@]}"
         "$JL" --startup-file=no -t 4 "$REPO/examples/yang_lee/bethe_vs_hermitian.jl") ;;
  *) echo "unknown KIND $KIND"; exit 1 ;;
esac
echo "$(date) task $TASK on $(hostname): $LINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN (in $WORK): ${CMD[*]}"; exit 0; fi
cd "$WORK"
"${CMD[@]}" 2>&1 | tee "task$TASK.last"
if grep -q "(budget" "task$TASK.last" && [ "${REQUEUE:-1}" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "$(date) task $TASK: unfinished at the budget, requeueing"
  scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK}"
fi
echo "$(date) task $TASK: finished this allocation"
