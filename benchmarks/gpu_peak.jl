# Device peak live memory, measured as "does it fit": run under a hard CUDA.jl memory limit
#
#   JULIA_CUDA_HARD_MEMORY_LIMIT=<bytes> julia --project=. benchmarks/gpu_peak.jl 250 f32 gate
#
# CUDA.jl collects garbage and retries before it throws OutOfGPUMemoryError, so the smallest limit
# under which the operation passes is the high-water mark of LIVE device memory — the quantity the
# "≤ 3F + change" target speaks about (the pool's own high-water counters include uncollected
# garbage and are not comparable). Baseline = the state plus its BP messages; the script prints
# baseline and headroom in units of F, the centre tensor's size. Bisect by repeating with several
# limits (each run ~1.5 min including the small warm-up at D = 16): 2026-09-11 results in
# docs/ctmrg_status.md, "GPU (CUDA.jl)".
using TensorNetworkQuantumSimulator, LinearAlgebra, Random, Printf, CUDA, Adapt
const TNQS = TensorNetworkQuantumSimulator; const CC = CUDA.CUDACore
CUDA.allowscalar(false)
D = parse(Int, ARGS[1]); elt = ARGS[2] == "f32" ? ComplexF32 : ComplexF64; mode = ARGS[3]
function setup(elt, D)
    Random.seed!(1); g = named_comb_tree((3, 3))
    ψ = random_tensornetworkstate(elt, g, "S=1/2"; bond_dimension = D)
    c = first(center(g)); nb = first(neighbors(g, c))
    F = sizeof(elt) * prod(TNQS.dim.(TNQS.inds(ψ[c])))
    return adapt(CuArray, ψ), Any[("Rxx", (c, nb), 0.1)], F, g
end
ψw, gw, _, _ = setup(elt, 16); bpcw = TNQS.update(BeliefPropagationCache(ψw); maxiter = 1)
apply_gates!(gw, bpcw; apply_kwargs = (; maxdim = 16, cutoff = nothing), update_cache = false, verbose = false)
ψw = bpcw = nothing; GC.gc(); CUDA.reclaim()
ψ, gate, F, g = setup(elt, D)
bpc = TNQS.update(BeliefPropagationCache(ψ); maxiter = 1)
GC.gc(); GC.gc(); CUDA.reclaim(); CUDA.synchronize()
base = CC.used_memory()
lim = CC.memory_limits().hard
layer = Any[]; for ces in edge_color(g, 3); append!(layer, ("Rxx", pair, 0.1) for pair in ces); end
op = mode == "bp" ? (() -> TNQS.update(bpc; maxiter = 1)) :
     mode == "gate" ? (() -> apply_gates!(gate, bpc; apply_kwargs = (; maxdim = D, cutoff = nothing), update_cache = false, verbose = false)) :
     (() -> apply_gates!(layer, bpc; apply_kwargs = (; maxdim = D, cutoff = nothing), update_cache = false, verbose = false))
status = try; op(); CUDA.synchronize(); "OK "; catch e; e isa CUDA.OutOfGPUMemoryError || (e isa TaskFailedException && e.task.result isa CUDA.OutOfGPUMemoryError) ? "OOM" : rethrow(); end
@printf("%-6s D=%d %s  baseline %.2f F  hard limit %.2f F (headroom %.2f F): %s\n", mode, D, ARGS[2], base / F, lim / F, (lim - base) / F, status)
