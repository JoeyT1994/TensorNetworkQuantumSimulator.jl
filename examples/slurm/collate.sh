#!/bin/bash
# Table of Δ ln|A| (CTMRG − exact) per depth and χ from the disBatch outputs.
cd "$(dirname "$0")/out" || exit 1
printf "%6s %14s %8s %12s %12s\n" depth "ln|A| exact" chi cut cycle
for f in circ_d*_chi*.out; do
    [ -f "$f" ] || continue
    d=${f#circ_d}; d=${d%%_*}; c=${f##*chi}; c=${c%.out}
    grep -E "^ +$d +-" "$f" | awk -v d="$d" -v c="$c" '{printf "%6s %14s %8s %12s %12s\n", d, $2, c, $3, $4}'
done | sort -n -k1 -k3
