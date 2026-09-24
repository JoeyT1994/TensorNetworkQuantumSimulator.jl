#!/bin/bash
# Sycamore-gate-set depth sweep of examples/circuit_ctmrg.jl on a 20-qubit chain: 12 independent
# (depth, χ) tasks through disBatch, 8 cores each. Submit from the repository root:
#   sbatch examples/slurm/circuit_sweep.sbatch
# then collate with:  bash examples/slurm/collate.sh
#SBATCH -J circ_ctmrg
#SBATCH -p ccq
#SBATCH -n 12
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH -t 04:00:00
#SBATCH -o examples/slurm/out/slurm_%j.log
# The compiled cache in ~/.julia is shared with the workstation, whose CPU differs from the nodes':
# fix the target so each machine gets its own cache entries, and precompile ONCE before the tasks
# (twelve tasks precompiling the stack at the same time is slow and tripped "Unable to find
# compatible target in cached code image").
export JULIA_CPU_TARGET="generic;skylake-avx512,clone_all;znver3,clone_all"
JULIA=/mnt/home/jtindall/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia
$JULIA --startup-file=no --project=. -e 'using TensorNetworkQuantumSimulator' || exit 1
module load disBatch
disBatch examples/slurm/circuit_tasks.txt
