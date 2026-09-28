#!/bin/bash
# A lane of variational honeycomb optima: per job, honeycomb_variational.jl in ≤ 10-minute runs until it
# reports DONE, then honeycomb_prod.jl on the optimum (OUT/eval_D<D>_chi<χ>) for ⟨ψ|ψ⟩, ⟨Iψ|ψ⟩, ⟨Iψ|M|ψ⟩
# and ⟨Mψ|Mψ⟩ at the same χ (summary.csv and residual.csv there).
#   usage: honeycomb_variational_lane.sh TAG "D CHI CHUNK" ["D CHI CHUNK" ...]   (CHUNK: the evaluation's)
#   env:   JULIA_PROJECT, OUT (ice_var), DEVICE (cpu), BLASN (4), SUDIR (BP-SU states to start from)
# STOPPING: kill by command line (Get-CimInstance Win32_Process | ? CommandLine -match 'variational_lane').
TAG=$1; shift
DIR=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-ice_var}; mkdir -p "$OUT"
for job in "$@"; do
  set -- $job
  D=$1; CHI=$2; CHUNK=$3
  for attempt in $(seq 1 150); do
    D=$D CHI=$CHI OUT=$OUT BLASN=${BLASN:-4} BUDGET=540 timeout 598 julia "$DIR/honeycomb_variational.jl" \
      > "$OUT/$TAG.last" 2>&1
    rc=$?
    grep -v "Warning\|@ Tensor\|^\s*$" "$OUT/$TAG.last" >> "$OUT/$TAG.log"
    if grep -q "ERROR" "$OUT/$TAG.last"; then echo "=== stopped (error): $job" >> "$OUT/$TAG.log"; continue 2; fi
    [ $rc -eq 124 ] && { echo "(timeout kill, resuming)" >> "$OUT/$TAG.log"; continue; }
    grep -q "^DONE" "$OUT/$TAG.last" && break
  done
  EV="$OUT/eval_D${D}_chi${CHI}"
  for attempt in $(seq 1 100); do
    D=$D CHI=$CHI KINDS=norm,inv,sand,mnorm CHUNK=$CHUNK OUT=$EV BLASN=${BLASN:-4} BUDGET=560 \
      timeout 598 julia "$DIR/honeycomb_prod.jl" > "$OUT/$TAG.last" 2>&1
    rc=$?
    grep -v "Warning\|@ Tensor\|^\s*$" "$OUT/$TAG.last" >> "$OUT/$TAG.log"
    if grep -q "ERROR\|MAXIT" "$OUT/$TAG.last"; then echo "=== eval stopped: $job" >> "$OUT/$TAG.log"; break; fi
    [ $rc -eq 124 ] && { echo "(timeout kill, resuming)" >> "$OUT/$TAG.log"; continue; }
    grep -q "(budget" "$OUT/$TAG.last" || break
  done
  echo "=== job $job finished" >> "$OUT/$TAG.log"
done
echo "=== lane $TAG done" >> "$OUT/$TAG.log"
