#!/bin/bash
# Slurm job array for the honeycomb ice campaign: one line of the job list per array task, one GPU each.
# Every job RESUMES from its checkpoints, so resubmitting the same array continues the unfinished ones
# and finished ones exit in seconds. Submit from this directory:
#
#   sbatch --array=1-$(./jobs.sh count jobs_ice.txt)%16 ice_array.slurm          # ≤ 16 GPUs at once
#
# A task runs its driver with a budget up to the wall time minus MARGIN_S, loops while the driver
# reports an unfinished budget-stop, and on SIGUSR1 (sent MARGIN_S before the end) requeues itself if
# REQUEUE=1. Env overrides: JOBS (jobs_ice.txt), RUNDIR (./runs), WALL_S (must match --time, in s),
# MARGIN_S (600), REQUEUE (1), BLASN (8), DRYRUN (1: print the command, run nothing).
#SBATCH --job-name=ice-honeycomb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --signal=B:USR1@600
set -u
KIT=${KIT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
REPO=$(cd "$KIT/../../.." && pwd)
JOBS=${JOBS:-$KIT/jobs_ice.txt}
RUNDIR=${RUNDIR:-$KIT/runs}; mkdir -p "$RUNDIR" "$KIT/logs"
WALL_S=${WALL_S:-21600}; MARGIN_S=${MARGIN_S:-600}
TASK=${SLURM_ARRAY_TASK_ID:?"run as an array task (or set SLURM_ARRAY_TASK_ID)"}
LINE=$("$KIT/jobs.sh" line "$JOBS" "$TASK")
[ -z "$LINE" ] && { echo "task $TASK: no such job line"; exit 1; }
read -r MODE D CHI KINDS CHUNK <<< "$LINE"
export JULIA_PROJECT="$KIT" DEVICE=gpu BLASN=${BLASN:-8}
START=$(date +%s); END=$((START + WALL_S - MARGIN_S))
requeue() {
  echo "$(date) task $TASK: wall time near"
  if [ "${REQUEUE:-1}" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK}" && echo "requeued"
  fi
  exit 0
}
trap requeue USR1
run() {                                   # run one driver invocation with the remaining budget
  local budget=$((END - $(date +%s)))
  [ "$budget" -le 60 ] && return 2
  if [ "${DRYRUN:-0}" = 1 ]; then echo "DRYRUN: BUDGET=$budget $*"; return 0; fi
  env BUDGET=$budget "$@" &               # background + wait, so the USR1 trap fires at once
  wait $!
}
echo "$(date) task $TASK on $(hostname): $LINE (budget up to $((END - START)) s)"
case "$MODE" in
  prod)                                   # BP simple-update state + the networks in KINDS
    OUT="$RUNDIR/prod"
    for pass in $(seq 1 1000); do
      run D=$D CHI=$CHI KINDS=$KINDS CHUNK=$CHUNK OUT=$OUT julia "$REPO/examples/ice/honeycomb_prod.jl" \
        > "$RUNDIR/task$TASK.last" 2>&1 || true
      cat "$RUNDIR/task$TASK.last"
      [ "${DRYRUN:-0}" = 1 ] && break
      grep -q "(budget" "$RUNDIR/task$TASK.last" || break
      [ $((END - $(date +%s))) -le 60 ] && break
    done ;;
  var)                                    # variational optimum from the BP-SU state, then its evaluation
    OUT="$RUNDIR/var"
    for pass in $(seq 1 1000); do
      run D=$D CHI=$CHI OUT=$OUT SUDIR="$RUNDIR/prod" julia "$REPO/examples/ice/honeycomb_variational.jl" \
        > "$RUNDIR/task$TASK.last" 2>&1 || true
      cat "$RUNDIR/task$TASK.last"
      [ "${DRYRUN:-0}" = 1 ] && break
      grep -q "^DONE" "$RUNDIR/task$TASK.last" && break
      [ $((END - $(date +%s))) -le 60 ] && break
    done
    if [ "${DRYRUN:-0}" = 1 ] || grep -q "^DONE" "$RUNDIR/task$TASK.last"; then
      for pass in $(seq 1 1000); do
        run D=$D CHI=$CHI KINDS=norm,inv,sand,mnorm CHUNK=$CHUNK OUT="$OUT/eval_D${D}_chi${CHI}" \
          julia "$REPO/examples/ice/honeycomb_prod.jl" > "$RUNDIR/task$TASK.last" 2>&1 || true
        cat "$RUNDIR/task$TASK.last"
        [ "${DRYRUN:-0}" = 1 ] && break
        grep -q "(budget" "$RUNDIR/task$TASK.last" || break
        [ $((END - $(date +%s))) -le 60 ] && break
      done
    fi ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
echo "$(date) task $TASK: finished this allocation"
