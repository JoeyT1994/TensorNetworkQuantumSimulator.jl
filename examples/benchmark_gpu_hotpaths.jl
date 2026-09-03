# WARMED, operation-local CUDA pool high-water benchmark for the two dominant hot paths.
# Compilation is paid on a separate χ=8 state before the target is constructed.
#
#   julia --project=. examples/benchmark_gpu_hotpaths.jl [χ] [BP repetitions]
#   julia --project=. examples/benchmark_gpu_hotpaths.jl 500 3
#
# F is the byte size of the dominant centre tensor. The intended steady-state resident
# baseline is its F-sized data plus the reusable 2F contraction arena. Result and vendor
# workspace are reported as transient change above that 3F baseline. `CUDA.@allocated` is
# allocation volume, not a peak; this file reads CUDA's resettable memory-pool watermark.

using TensorNetworkQuantumSimulator
using CUDA
using UUIDs: UUID

const TNQS = TensorNetworkQuantumSimulator
const CUDACore = Base.loaded_modules[
    Base.PkgId(UUID("bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"), "CUDACore")
]

pool() = CUDACore.pool_create(CUDACore.device())
pool_used(p) = Int(CUDACore.attribute(UInt64, p, CUDACore.MEMPOOL_ATTR_USED_MEM_CURRENT))
pool_high(p) = Int(CUDACore.attribute(UInt64, p, CUDACore.MEMPOOL_ATTR_USED_MEM_HIGH))
reset_pool_high!(p) = CUDACore.attribute!(p, CUDACore.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))

function warmup(g, gate)
    χ = 8
    kw = (; maxdim = χ, cutoff = nothing, normalize_tensors = true)
    ψ = CUDA.cu(random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = χ))
    bpc = CUDA.@sync TNQS.update(BeliefPropagationCache(ψ))
    CUDA.@sync apply_gates!([gate], bpc; apply_kwargs = kw, update_cache = false)
    return nothing
end

function bp_probe(bpc, v, w, F, p, reps)
    e = NamedEdge(v => w)
    incoming = TNQS.incoming_messages(bpc, v; ignore_edges = (reverse(e),))
    factor = network(bpc)[v]
    f() = TNQS.norm_message_kernel(network(bpc), v, incoming; normalize = true)

    for _ in 1:3
        m = CUDA.@sync f()
        CUDA.unsafe_free!(TNQS.data(m))
    end
    CUDA.synchronize()

    arena = TNQS.Tensors._kernel_buffer(TNQS.data(factor))
    base = pool_used(p)
    best, extra, volume = Inf, 0, 0
    for _ in 1:reps
        reset_pool_high!(p)
        timed = CUDA.@timed f()
        CUDA.synchronize()
        best = min(best, timed.time)
        extra = max(extra, pool_high(p) - base)
        volume = max(volume, timed.gpu_bytes)
        CUDA.unsafe_free!(TNQS.data(timed.value))
    end
    return (; base, peak = base + extra, extra, arena = length(arena), best, volume,
        peak_over_F = 1 + length(arena) / F + extra / F)
end

function gate_probe(bpc, gate, v, F, p, χ)
    kw = (; maxdim = χ, cutoff = nothing, normalize_tensors = true)
    GC.gc(true)
    CUDA.synchronize()
    base = pool_used(p)
    reset_pool_high!(p)
    timed = CUDA.@timed apply_gates!(
        [gate], bpc; apply_kwargs = kw, update_cache = false, verbose = false,
    )
    CUDA.synchronize()
    bpc = first(timed.value)
    arena = TNQS.Tensors._kernel_buffer(TNQS.data(network(bpc)[v]))
    return bpc, (; base, peak = pool_high(p), extra = pool_high(p) - base,
        arena = length(arena), seconds = timed.time, volume = timed.gpu_bytes,
        peak_over_F = pool_high(p) / F)
end

function main()
    CUDA.allowscalar(false)
    χ = isempty(ARGS) ? 64 : parse(Int, ARGS[1])
    reps = length(ARGS) < 2 ? 5 : parse(Int, ARGS[2])
    p = pool()
    g = named_comb_tree((3, 3))
    v = first(center(g))
    w = first(neighbors(g, v))
    gate = ("Rxx", (v, w), -0.5)

    warmup(g, gate)
    GC.gc(true)
    CUDA.synchronize()

    ψ = CUDA.cu(random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = χ))
    bpc = BeliefPropagationCache(ψ)
    F = sizeof(eltype(TNQS.data(network(bpc)[v]))) * length(TNQS.data(network(bpc)[v]))

    bp = bp_probe(bpc, v, w, F, p, reps)
    bpc = CUDA.@sync TNQS.update(bpc)
    bpc, gate_result = gate_probe(bpc, gate, v, F, p, χ)

    println("CUDA hot paths: χ=$χ, F=$F bytes")
    println("  BP:    arena/F=$(bp.arena / F), change/F=$(bp.extra / F), " *
        "peak-bound/F=$(bp.peak_over_F), best=$(bp.best)s, volume=$(bp.volume)")
    println("  gate:  arena/F=$(gate_result.arena / F), change/F=$(gate_result.extra / F), " *
        "pool-peak/F=$(gate_result.peak_over_F), time=$(gate_result.seconds)s, " *
        "volume=$(gate_result.volume)")
end

main()
