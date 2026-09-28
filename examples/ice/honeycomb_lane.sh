#!/bin/bash
# A lane of honeycomb_prod.jl jobs, each resumed in ≤ 10-minute runs until its networks converge.
#   usage: honeycomb_lane.sh TAG "D CHI KINDS CHUNK [TOL]" ["D CHI KINDS CHUNK [TOL]" ...]
#   env:   JULIA_PROJECT (an environment that `dev`s this repo, plus Adapt/CUDA for DEVICE=gpu),
#          OUT (ice_prod), BLASN (6 with two lanes), DEVICE (cpu)
# Logs: $OUT/$TAG.log (filtered, appended per run) and $OUT/$TAG.last (the current run).
# A run killed by the timeout (a chunk that did not fit) resumes; an ERROR or MAXIT stops the job.
# STOPPING: killing this script's task does NOT stop its loop on Windows/MSYS — kill by command line
# (Get-CimInstance Win32_Process | ? CommandLine -match 'honeycomb_lane'), then the julia runs.
TAG=$1; shift
DIR=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-ice_prod}; mkdir -p "$OUT"
for job in "$@"; do
  set -- $job
  for attempt in $(seq 1 100); do
    D=$1 CHI=$2 KINDS=$3 CHUNK=$4 TOL=${5:-1e-11} OUT=$OUT BLASN=${BLASN:-6} BUDGET=560 \
      timeout 598 julia "$DIR/honeycomb_prod.jl" > "$OUT/$TAG.last" 2>&1
    rc=$?
    grep -v "Warning\|@ Tensor\|^\s*$" "$OUT/$TAG.last" >> "$OUT/$TAG.log"
    if grep -q "ERROR\|MAXIT" "$OUT/$TAG.last"; then echo "=== stopped: $job" >> "$OUT/$TAG.log"; break; fi
    [ $rc -eq 124 ] && { echo "(timeout kill, resuming)" >> "$OUT/$TAG.log"; continue; }
    grep -q "(budget" "$OUT/$TAG.last" || break
  done
  echo "=== job $job finished (attempt $attempt)" >> "$OUT/$TAG.log"
done
echo "=== lane $TAG done" >> "$OUT/$TAG.log"
