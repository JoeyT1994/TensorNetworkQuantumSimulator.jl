# Benchmarks

Timing and peak-memory harnesses used for the 2026-09-10 performance audit (see
`docs/ctmrg_status.md`, "Performance audit after the backend switch"). All run single-threaded BLAS
so numbers are comparable across machines and branches; run them from another branch's worktree
with `--project=<that worktree>` to compare backends.

| script | what it measures | usage |
|---|---|---|
| `peak_memory.jl` | BP update and one simple-update layer on a `named_comb_tree((3, 3))` whose centre tensor dominates (F = its size); reports time, allocation churn and resident-set high-water mark in units of F | `julia --project=. benchmarks/peak_memory.jl su 250`, `... bp 250` |
| `ctm.jl` | CTM `:cut` / `:cycle` sweeps on a random L×L grid | `julia --project=. benchmarks/ctm.jl cut 6 4 64` |
| `boundarymps.jl` | boundary-MPS iterations on a random L×L grid | `julia --project=. benchmarks/boundarymps.jl 6 3 32` |
| `graded.jl` | fermionic CTM and Z2 boundary MPS / BP timings on gate-built graded states | `julia --project=. benchmarks/graded.jl` |
| `gpu.jl` | CPU vs GPU (CUDA) speed of one BP iteration and one centre-bond gate on the comb tree, both precisions, best of two | `julia --project=. benchmarks/gpu.jl 60,120,200` |
| `gpu_peak.jl` | peak LIVE device memory of one BP iteration / one gate / one layer on the comb tree, as pass–fail under a hard CUDA.jl memory limit (bisect the limit; the smallest passing headroom over the baseline of state + messages is the peak, in units of F) | `JULIA_CUDA_HARD_MEMORY_LIMIT=1513000000 julia --project=. benchmarks/gpu_peak.jl 250 f32 gate` |
| `robustness_graded.jl` | CTM `:cut`/`:cycle` and boundary MPS against exact contraction as χ grows, fermionic and Z2 (with its dense twin) | `julia --project=. benchmarks/robustness_graded.jl` |

The peak-memory reading is the process high-water mark over the timed operation minus the
resident set just before it; it is only meaningful when the operation's tensors are much larger
than the Julia runtime's own footprint (hence D = 250 on the comb tree, F = 500 MB). "NaN" or
"below earlier high-water mark" means the run stayed under a previous peak.
