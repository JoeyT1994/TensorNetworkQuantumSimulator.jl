#!/bin/bash
# jobs.sh count FILE | jobs.sh line FILE N — the job list without comments and blank lines
f=$2
case "$1" in
  count) grep -v '^[[:space:]]*#' "$f" | grep -cv '^[[:space:]]*$' ;;
  line)  grep -v '^[[:space:]]*#' "$f" | grep -v '^[[:space:]]*$' | sed -n "${3}p" ;;
esac
