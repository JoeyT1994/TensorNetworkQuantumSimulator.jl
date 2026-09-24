# 1+1D circuits as space–time rectangles — *2026-09-24*

`examples/circuit_ctmrg.jl` contracts a brickwork circuit on N qubits and DEPTH layers as an
N × DEPTH rectangle with the finite CTMRG: each two-qubit gate is split by operator-Schmidt into two
halves carrying a horizontal bond (rank ≤ 4), the |0⟩ boundary and the observable are folded into the
first and last layers, and the CVM free energy gives ln|amplitude|. Two quantities: the return
amplitude ⟨0…0|U|0…0⟩ (single layer, D = 2) and ⟨Z_k⟩ (folded double layer, D = 4). Constants are
overridable through `CIRC_*` environment variables; `examples/slurm/circuit_sweep.sbatch` runs a
(depth, χ) sweep through disBatch (one precompile for the node before the tasks, fixed CPU target
because the compiled cache is shared with the workstation).

**Haar brickwork, N = 10, depth 8, against the statevector.** The exact network contraction agrees
to 1e-15. Return amplitude: `:cut` exact (1e-14) from χ = 16, `:cycle` from χ = 8. ⟨Z₅⟩ (folded):

| χ | :cut | :cycle |
|---|---|---|
| 8 | 2.7e-2 | 3.7e-1 |
| 16 | 4.4e-5 | 4e-14 |
| 32 | 7.2e-6 | 2e-16 |

**Sycamore gate set (fSim(π/2, π/6) couplers, random √X/√Y/√W singles), N = 20 chain, return
amplitude, Δ ln|A| against the statevector** (workstation to depth 12, Slurm job 7101487 beyond):

| depth | ln\|A\| exact | cut χ=8 | cut 16 | cut 32 | cut 64 | cut 128 | cycle 32 | cycle 64 | cycle 128 |
|---|---|---|---|---|---|---|---|---|---|
| 2–8 | −6.9 | 1e-14 | 1e-14 | 1e-14 | 1e-14 | | 1e-14 | 1e-14 | |
| 12 | −7.773 | 8.6 | 1.2 | 1e-14 | 2e-14 | 2e-14 | 1e-13 | 2e-14 | 2e-13 |
| 16 | −6.485 | | | 0.55 | 3.0e-2 | 3e-14 | 17.5 | 10.9 | 8e-14 |
| 20 | −7.538 | | | 1.35 | 4.4 | 0.60 | 33.6 | 25.0 | 18.2 |
| 24 | −6.813 | | | −21.0 | −7.9 | −4.65 | 25.7 | 25.2 | 24.2 |

Reading. Up to depth 8 the space-time cut has not filled the 20-qubit width and every χ is exact.
The exact threshold then moves fast — χ ≤ 32 at depth 12, between 64 and 128 at depth 16, above 128
at depth 20 — roughly doubling every four layers, the volume-law growth of the cut for a scrambling
circuit. Below the threshold `:cut` degrades gracefully (3e-2 at depth 16/χ = 64, 0.6 at depth
20/χ = 128) while `:cycle` fails by 10–30 in ln|A| wherever it is truncated: the cycle's stationary
problem has no good truncated fixed point, as in the DMRG runs at hard truncation. For circuits the
cut is the projector to use; the cycle is exact where the cut is still creeping (⟨Z⟩ at χ = 16 above)
but only above the rank.

Caveats in the script: the free energy is ln|z| per region, so signed observables take their sign
from the exact value here (signed region contractions are a small engine addition); Sycamore itself
is a 2D array, whose space–time network is 3D and outside the 2D engine.
