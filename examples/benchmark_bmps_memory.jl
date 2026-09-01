# Warmed CUDA memory-pool watermark for boundary-MPS fitting. Compilation and cuTENSOR
# planning are paid on a throwaway cache with the target shape before the measured cache
# is constructed.
#
#   julia --project=. examples/benchmark_bmps_memory.jl [state bond D] [MPS bond chi]

using TensorNetworkQuantumSimulator
using CUDA
using LinearAlgebra: norm, rmul!
using UUIDs: UUID

const TNQS = TensorNetworkQuantumSimulator
const CUDACore = Base.loaded_modules[
    Base.PkgId(UUID("bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"), "CUDACore")
]

pool() = CUDACore.pool_create(CUDACore.device())
pool_used(p) = Int(CUDACore.attribute(UInt64, p, CUDACore.MEMPOOL_ATTR_USED_MEM_CURRENT))
pool_high(p) = Int(CUDACore.attribute(UInt64, p, CUDACore.MEMPOOL_ATTR_USED_MEM_HIGH))
reset_pool_high!(p) = CUDACore.attribute!(p, CUDACore.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))

tensor_bytes(t) = sizeof(eltype(TNQS.data(t))) * length(TNQS.data(t))
message_bytes(cache) = sum(tensor_bytes, values(messages(cache)); init = 0)
largest_message(cache) = maximum(tensor_bytes, values(messages(cache)); init = 0)

function make_cache(D, chi)
    g = named_grid((3, 3))
    psi = CUDA.cu(random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = D))
    return BoundaryMPSCache(psi, chi)
end

function fitting_update!(cache)
    pe = first(TNQS.all_quotientedges(cache))
    alg = TNQS.set_default_kwargs(TNQS.Algorithm("fitting"; niters = 1), cache)
    TNQS.update_message!(alg, cache, pe)
    return cache
end

fitting_alg(cache) = TNQS.set_default_kwargs(TNQS.Algorithm("fitting"; niters = 1), cache)

function gauge_once!(cache)
    pe = first(TNQS.all_quotientedges(cache))
    es = TNQS.sorted_edges(cache, pe)
    TNQS.gauge_step!(fitting_alg(cache), cache, reverse(es[end]), reverse(es[end - 1]))
    return cache
end

function extract_once(cache)
    pe = first(TNQS.all_quotientedges(cache))
    return TNQS.extracter(fitting_alg(cache), cache, first(TNQS.sorted_edges(cache, pe)))
end

function adjoint_once(cache)
    pe = first(TNQS.all_quotientedges(cache))
    e = first(TNQS.sorted_edges(cache, pe))
    return TNQS.fit_adjoint_message(cache, e, message(cache, e))
end

function probe(label, operation, D, chi, p)
    cache = make_cache(D, chi)
    CUDA.synchronize()
    Fm = largest_message(cache)
    baseline = pool_used(p)
    reset_pool_high!(p)
    timed = CUDA.@timed operation(cache)
    CUDA.synchronize()
    peak = pool_high(p)
    println("  $label: change=$(peak - baseline) bytes " *
        "($(round((peak - baseline) / Fm; digits = 2)) Fm), " *
        "time=$(round(timed.time; digits = 3))s, volume=$(timed.gpu_bytes) bytes")
    timed = nothing
    cache = nothing
    GC.gc(true)
    CUDA.reclaim()
    return nothing
end

function trace_step!(label, operation, cache, p)
    CUDA.synchronize()
    baseline = pool_used(p)
    reset_pool_high!(p)
    operation()
    CUDA.synchronize()
    current = pool_used(p)
    high = pool_high(p)
    println("    $label: baseline=$baseline current=$current high=$high " *
        "change=$(high - baseline), largest-message=$(largest_message(cache))")
    return nothing
end

function describe_message_update(cache, e)
    incoming = TNQS.incoming_messages(cache, TNQS.src(e); ignore_edges = (reverse(e),))
    factors = vcat(incoming, TNQS.bp_factors(cache, TNQS.src(e)))
    alg = TNQS.set_default_kwargs(TNQS.Algorithm("contract"; normalize = false), cache)
    sequence = TNQS.contraction_sequence(factors; alg = alg.kwargs.sequence_alg)
    elsize = sizeof(mapreduce(eltype, promote_type, factors))
    _, buffered_bytes = TNQS.Tensors._seq_temp_bytes(factors, sequence, elsize)
    println("      edge=$e factor-bytes=$(tensor_bytes.(factors)) sequence=$sequence " *
        "buffered-tree-bytes=$buffered_bytes")
    return nothing
end

function trace_fitting!(cache, p)
    pe = first(TNQS.all_quotientedges(cache))
    alg = fitting_alg(cache)
    es = TNQS.sorted_edges(cache, pe)
    g = TNQS.partition_graph(cache, TNQS.src(pe))
    update_seq = vcat(es, @view(es[(end - 1):-1:2]), es[1])

    trace_step!("delete local", () -> TNQS.delete_partition_messages!(cache, TNQS.src(pe)), cache, p)
    trace_step!("switch", () -> TNQS.switch_messages!(cache, pe), cache, p)
    for i in length(es):-1:2
        e1, e2 = reverse(es[i]), reverse(es[i - 1])
        trace_step!("initial gauge $i", () -> TNQS.gauge_step!(alg, cache, e1, e2), cache, p)
    end
    for (i, e) in enumerate(TNQS.post_order_dfs_edges(g, TNQS.src(first(update_seq))))
        describe_message_update(cache, e)
        trace_step!("initial partition $i", () -> TNQS.update_partition!(cache, [e]), cache, p)
    end

    prev_e = nothing
    for (i, update_e) in enumerate(update_seq)
        if prev_e !== nothing
            trace_step!("updater $i", () -> TNQS.updater!(alg, cache, g, prev_e, update_e), cache, p)
        end
        holder = Ref{Any}()
        trace_step!("extract $i", () -> (holder[] = TNQS.extracter(alg, cache, update_e)), cache, p)
        m = holder[]
        n = norm(m)
        !iszero(n) && rmul!(TNQS.data(m), inv(n))
        trace_step!("insert $i", () -> TNQS.inserter!(alg, cache, update_e, m), cache, p)
        prev_e = update_e
    end
    return cache
end

function main()
    CUDA.allowscalar(false)
    D = isempty(ARGS) ? 2 : parse(Int, ARGS[1])
    chi = length(ARGS) < 2 ? 8 : parse(Int, ARGS[2])

    warm = make_cache(D, chi)
    CUDA.@sync fitting_update!(warm)
    warm = nothing
    GC.gc(true)
    CUDA.reclaim()

    cache = make_cache(D, chi)
    CUDA.synchronize()
    p = pool()
    Fpsi = maximum(tensor_bytes, values(TNQS.tensors(TNQS.network(cache))); init = 0)
    Fm = largest_message(cache)
    message_total = message_bytes(cache)
    baseline = pool_used(p)
    println("CUDA bMPS fitting: D=$D, chi=$chi")
    println("  Fpsi=$Fpsi bytes, largest-message Fm=$Fm bytes")
    println("  resident messages=$message_total bytes ($(round(message_total / Fm; digits = 2)) Fm)")
    println("  pool baseline=$baseline bytes ($(round(baseline / Fm; digits = 2)) Fm)")
    cache = nothing
    GC.gc(true)
    CUDA.reclaim()

    probe("gauge QR", gauge_once!, D, chi, p)
    probe("target extraction", extract_once, D, chi, p)
    probe("fit adjoint", adjoint_once, D, chi, p)
    probe("full fitting update", fitting_update!, D, chi, p)
    if length(ARGS) >= 3 && ARGS[3] == "trace"
        println("  traced full fitting update:")
        cache = make_cache(D, chi)
        trace_fitting!(cache, p)
    end
end

main()
