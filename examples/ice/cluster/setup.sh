#!/bin/bash
# One-time setup on a LOGIN node (internet access): registries, the pinned environment, precompilation,
# and CUDA's runtime artifacts (compute nodes may be offline). If the compute nodes do not share your
# home directory, first point JULIA_DEPOT_PATH at a shared filesystem, here and in the job script's env.
set -e
KIT=$(cd "$(dirname "$0")" && pwd)
export JULIA_PKG_USE_CLI_GIT=true          # libgit2 hung for 20+ minutes on the ITensor registry clone
julia --project="$KIT" -e '
using Pkg
names = [r.name for r in Pkg.Registry.reachable_registries()]
"General" in names || Pkg.Registry.add("General")
"ITensorRegistry" in names || Pkg.Registry.add(Pkg.RegistrySpec(url = "https://github.com/ITensor/ITensorRegistry"))
Pkg.instantiate()
Pkg.precompile()
println("environment OK: ", Pkg.project().path)'
julia --project="$KIT" -e '
using CUDA
try; CUDA.precompile_runtime(); catch err; @warn "precompile_runtime" err; end
CUDA.functional() ? CUDA.versioninfo() : println("no GPU on this node (fine on a login node); check with the smoke job")'
echo "next: sbatch --array=1-2 --time=00:30:00 --export=ALL,JOBS=$KIT/jobs_smoke.txt,WALL_S=1800 ice_array.slurm"
