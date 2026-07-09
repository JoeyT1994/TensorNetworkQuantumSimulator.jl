# ──────────────────────────────────────────────────────────────────────────────
# CUDA device backend for MultiGPUBeliefPropagationCache.
#
#   1. The TN graph is partitioned across N devices; each device holds its
#      partition's vertex tensors and all messages incident to it in VRAM.
#   2. Edges are colored; within a color all updates are conflict-free.
#   3. Per color: each device batches and contracts its messages (with optional
#      damping), then a P2P halo exchange propagates fresh boundary messages.
#   4. Gates are applied on the device owning a "home" vertex, with an OOM→CPU
#      fallback for the QR/SVD at very large bond dimension.
#
# ──────────────────────────────────────────────────────────────────────────────

using CUDA
using CUDA: CuArray, CuDevice, OutOfGPUMemoryError
using KaHyPar
using SparseArrays: sparse
using ITensors: dim
using ITensors.NDTensors: Dense

# ── 1. Device-resident partition state ────────────────────────────────────────

"""
    DevicePartition{V}

All data owned by a single device: assigned vertex tensors, the physical site
indices per vertex (CPU metadata for BP-factor construction), all messages for
edges incident to the partition, and the list of outgoing boundary edges.
"""
mutable struct DevicePartition{V}
    device_id::Int
    vertices::Vector{V}
    vertex_set::Set{V}
    tensors::Dictionary{V, ITensor}
    siteinds_map::Dictionary{V, Vector{Index}}
    messages::Dictionary{NamedEdge, ITensor}
    boundary_edges_out::Vector{NamedEdge}
end

# Copy the containers to avoid aliasing the source cache's mutable state
function Base.copy(part::DevicePartition)
    return DevicePartition(
        part.device_id,
        copy(part.vertices),
        copy(part.vertex_set),
        copy(part.tensors),
        copy(part.siteinds_map),
        copy(part.messages),
        copy(part.boundary_edges_out),
    )
end

# ── 2. Device-aware tensor movement ───────────────────────────────────────────

"""
    tensor_to_device(t, target; complex_type=ComplexF32) -> ITensor

Move an ITensor to CUDA device `target` (Int or `CuDevice`). Uses explicit
known-source-device staging to avoid retaining a foreign device pointer (which
later breaks CUBLAS calls with `ERROR_INVALID_VALUE`).
"""
function tensor_to_device(t::ITensor, target::Union{Int, CuDevice}; complex_type = ComplexF32)
    target_device = target isa Int ? CuDevice(target) : target
    arr = ITensors.NDTensors.data(ITensors.NDTensors.tensor(t))

    if arr isa CuArray && CUDA.device(arr) == target_device
        CUDA.device!(target_device)
        return adapt(CuArray{complex_type}, t)
    end

    CUDA.device!(target_device)
    dst_data = CuArray{complex_type}(undef, dim(inds(t)))
    arr isa CuArray ? copyto!(dst_data, arr) : copyto!(dst_data, adapt(Array, arr))
    return itensor(Dense{complex_type, typeof(dst_data)}(dst_data), inds(t))
end

tensor_to_cpu(t::ITensor) = adapt(Array, t)

# ── 3. Peer access ────────────────────────────────────────────────────────────

const ENABLED_PEER_ACCESS_PAIRS = Set{Tuple{Int, Int}}()

"""
    ensure_peer_access!(partitions)

Enable CUDA peer access between every pair of partition devices so boundary
messages can be copied device-to-device. Idempotent; falls back silently to the
normal copy path (possibly via host) where P2P is unsupported.
"""
function ensure_peer_access!(partitions::Vector{<:DevicePartition})
    for src_part in partitions, dst_part in partitions
        src_part.device_id == dst_part.device_id && continue
        pair = (src_part.device_id, dst_part.device_id)
        pair in ENABLED_PEER_ACCESS_PAIRS && continue
        try
            CUDA.device!(src_part.device_id)
            CUDA.cuCtxEnablePeerAccess(CUDA.context(CuDevice(dst_part.device_id)), 0)
        catch
            # Already enabled or unsupported; tensor_to_device still works.
        end
        push!(ENABLED_PEER_ACCESS_PAIRS, pair)
    end
    CUDA.device!(0)
    return nothing
end

# ── 4. Construction ───────────────────────────────────────────────────────────

"""
    MultiGPUBeliefPropagationCache(bpc::BeliefPropagationCache; n_devices, complex_type, partition_alg, partition_kwargs)

Partition the graph, color its edges, and build device-local state for each
partition. State-network only.
"""
function MultiGPUBeliefPropagationCache(
        bpc::BeliefPropagationCache;
        n_devices::Integer = length(CUDA.devices()),
        complex_type = ComplexF32,
        partition_alg = "memory_balanced",
        partition_kwargs = (;),
        verbose::Bool = false,
    )
    ψ = network(bpc)
    ψ isa TensorNetworkState ||
        error("MultiGPUBeliefPropagationCache currently supports TensorNetworkState networks only.")
    g = graph(bpc)

    n_avail = length(CUDA.devices())
    n_devices = min(n_devices, n_avail)
    n_devices >= 1 || error(
        "MultiGPUBeliefPropagationCache requires at least one CUDA device, but " *
        "resolved n_devices=$n_devices (CUDA.devices() reports $n_avail). Ensure a " *
        "GPU is visible and CUDA.functional() returns true.",
    )
    verbose && @info "Multi-GPU BP: using $n_devices of $n_avail CUDA devices"

    pmap = partition_graph(g, n_devices; alg = partition_alg, partition_kwargs...)
    groups = colored_edge_groups(g)
    parts = build_device_partitions(bpc, pmap, n_devices; complex_type)
    ensure_peer_access!(parts)

    inner = copy(bpc)
    cache = MultiGPUBeliefPropagationCache(inner, parts, pmap, groups)
    sync_inner_from_partitions!(cache)
    return cache
end

"""
    build_device_partitions(bpc, partition_map, n_devices; complex_type) -> Vector{DevicePartition}

Move state tensors and incident messages to each partition's device VRAM and
classify boundary edges.
"""
function build_device_partitions(
        bpc::BeliefPropagationCache, partition_map::Dictionary, n_devices::Integer;
        complex_type = ComplexF32,
    )
    ψ = network(bpc)
    g = graph(bpc)
    V = eltype(vertices(g))
    tns_siteinds = siteinds(ψ)

    device_verts = [V[] for _ in 1:n_devices]
    for v in vertices(g)
        push!(device_verts[partition_map[v] + 1], v)
    end

    partitions = DevicePartition[]
    for dev_id in 0:(n_devices - 1)
        CUDA.device!(dev_id)
        verts = device_verts[dev_id + 1]
        vset = Set(verts)

        sinds_map = Dictionary{V, Vector{Index}}()
        dev_tensors = Dictionary{V, ITensor}()
        for v in verts
            set!(sinds_map, v, Vector{Index}(tns_siteinds[v]))
            set!(dev_tensors, v, tensor_to_device(ψ[v], dev_id; complex_type))
        end

        # Classify edges incident to this partition into local / boundary, and
        # collect the directed edges whose messages this device must hold.
        boundary_out = NamedEdge[]
        incident = NamedEdge[]
        for e in edges(g)
            s, d = src(e), dst(e)
            s_in, d_in = s in vset, d in vset
            (s_in || d_in) || continue
            push!(incident, NamedEdge(s => d), NamedEdge(d => s))
            if s_in ⊻ d_in
                # boundary: outgoing edge starts at the in-partition vertex
                out, inn = s_in ? (s, d) : (d, s)
                push!(boundary_out, NamedEdge(out => inn))
            end
        end

        dev_messages = Dictionary{NamedEdge, ITensor}()
        for e in unique(incident)
            set!(dev_messages, e, tensor_to_device(message(bpc, e), dev_id; complex_type))
        end

        push!(partitions, DevicePartition(
            dev_id, verts, vset, dev_tensors, sinds_map, dev_messages, boundary_out,
        ))
    end

    CUDA.device!(0)
    return partitions
end

# ── 5. Host mirror ↔ partition synchronization ────────────────────────────────

"""
    sync_inner_from_partitions!(cache) -> cache

Refresh the wrapped host cache's tensor/message references from the device
partitions (a lightweight metadata sync; tensors stay on their owning device).
"""
function sync_inner_from_partitions!(cache::MultiGPUBeliefPropagationCache)
    bpc = inner_cache(cache)
    for part in partitions(cache)
        for v in part.vertices
            setindex_preserve!(bpc, part.tensors[v], v)
        end
        for (e, m) in pairs(part.messages)
            setmessage!(bpc, e, m)
        end
    end
    return cache
end

# ── 6. Per-device batched message update ──────────────────────────────────────

function partition_factors(partition::DevicePartition, v)
    tnv = partition.tensors[v]
    tnv_dag = dag(prime(tnv))
    sinds = partition.siteinds_map[v]
    tnv_dag = replaceinds(tnv_dag, prime.(sinds), sinds)
    return ITensor[tnv, tnv_dag]
end

"""
    batched_update_messages!(partition, directed_edges, α; enforce_hermiticity, normalize_msgs, sequence_alg)

Update every message in `directed_edges` whose source is owned by `partition`, on
the partition's device. Contracts incoming messages with the vertex BP factors,
optionally enforces hermiticity and normalizes, then applies a damped update. Falls
back to CPU for an individual message on GPU OOM.
"""
function batched_update_messages!(
        partition::DevicePartition, directed_edges::Vector{NamedEdge}, α::Real;
        enforce_hermiticity::Bool = false, normalize_msgs::Bool = true,
        sequence_alg::String = "optimal",
    )
    CUDA.device!(partition.device_id)

    incoming_edges = Dict{Any, Vector{NamedEdge}}()
    for edge in keys(partition.messages)
        push!(get!(incoming_edges, dst(edge), NamedEdge[]), edge)
    end

    for edge in directed_edges
        v = src(edge)
        v in partition.vertex_set || continue

        incoming = ITensor[
            partition.messages[ne] for ne in get(incoming_edges, v, NamedEdge[]) if ne != reverse(edge)
        ]
        contract_list = ITensor[incoming; partition_factors(partition, v)]

        # At very high bond dimension the GPU contraction workspace can exceed
        # VRAM even though the resulting rank-2 message fits.
        used_cpu_retry = false
        local new_msg
        try
            seq = contraction_sequence(contract_list; alg = sequence_alg)
            new_msg = ITensors.contract(contract_list; sequence = seq)
        catch err
            err isa OutOfGPUMemoryError || rethrow()
            @warn "GPU OOM during BP message contraction; retrying on CPU." edge device = partition.device_id
            GC.gc(); CUDA.reclaim()
            cpu_list = tensor_to_cpu.(contract_list)
            seq = contraction_sequence(cpu_list; alg = sequence_alg)
            new_msg = ITensors.contract(cpu_list; sequence = seq)
            used_cpu_retry = true
        end

        enforce_hermiticity && (new_msg = make_hermitian(new_msg))

        if normalize_msgs
            message_sum = sum(new_msg)
            iszero(message_sum) || (new_msg = new_msg / message_sum)
        end

        # Damped update; only valid when link indices still match (after a gate
        # the link indices may have changed, making old messages incompatible).
        if haskey(partition.messages, edge)
            old_msg = partition.messages[edge]
            if Set(ITensors.inds(old_msg)) == Set(ITensors.inds(new_msg))
                old_msg = used_cpu_retry ? tensor_to_cpu(old_msg) : old_msg
                T = real(ITensors.eltype(new_msg))
                α_T = T(α)
                new_msg = α_T * new_msg + (one(T) - α_T) * old_msg
            end
        end

        used_cpu_retry &&
            (new_msg = tensor_to_device(new_msg, partition.device_id; complex_type = ITensors.eltype(new_msg)))
        set!(partition.messages, edge, new_msg)
    end

    CUDA.synchronize()
    return nothing
end

# ── 7. P2P halo exchange ──────────────────────────────────────────────────────

"""
    p2p_halo_exchange!(partitions, partition_map)

Broadcast each freshly-computed boundary message to the partition that holds the
other copy of that directed edge, keyed identically (`s => d`). The destination is
the partition owning `dst(edge)`, looked up directly from `partition_map`.
"""
function p2p_halo_exchange!(partitions::Vector{<:DevicePartition}, partition_map::Dictionary)
    for src_part in partitions
        for edge in src_part.boundary_edges_out
            haskey(src_part.messages, edge) || continue
            src_msg = src_part.messages[edge]
            dst_part = partitions[partition_map[dst(edge)] + 1]
            haskey(dst_part.messages, edge) || continue
            set!(dst_part.messages, edge, tensor_to_device(
                src_msg, dst_part.device_id; complex_type = ITensors.eltype(src_msg),
            ))
        end
    end
    for part in partitions
        CUDA.device!(part.device_id)
        CUDA.synchronize()
    end
    CUDA.device!(0)
    return nothing
end

# ── 8. update: colored parallel sweep + halo exchange ─────────────────────────

function update(alg::Algorithm"multigpu_bp", cache::MultiGPUBeliefPropagationCache)
    parts = partitions(cache)
    pmap = partition_map(cache)
    groups = edge_groups(cache)

    max_iterations = alg.kwargs.maxiter
    tolerance = alg.kwargs.tolerance
    α = alg.kwargs.α
    verbose = alg.kwargs.verbose
    enforce_hermiticity = alg.kwargs.enforce_hermiticity
    check_convergence = !isnothing(tolerance) && tolerance > 0.0

    for iter in 1:max_iterations
        old_msgs = Dict{NamedEdge, ITensor}()
        if check_convergence
            for part in parts, edge in keys(part.messages)
                if src(edge) in part.vertex_set
                    old_msgs[edge] = part.messages[edge]
                end
            end
        end

        for color_edges in groups
            directed = NamedEdge[]
            for e in color_edges
                push!(directed, e, reverse(e))
            end
            @sync for part in parts
                Threads.@spawn batched_update_messages!(part, directed, α; enforce_hermiticity)
            end
            p2p_halo_exchange!(parts, pmap)
        end

        if check_convergence
            total_diff = 0.0
            n_edges = 0
            for (edge, old_m) in old_msgs
                part = parts[pmap[src(edge)] + 1]
                haskey(part.messages, edge) || continue
                CUDA.device!(part.device_id)
                new_m = part.messages[edge]
                if Set(ITensors.inds(new_m)) == Set(ITensors.inds(old_m))
                    total_diff += message_diff(new_m, old_m)
                else
                    total_diff += 1.0  # indices changed → not converged
                end
                n_edges += 1
            end
            avg_diff = n_edges > 0 ? total_diff / n_edges : 0.0
            verbose && @info "Multi-GPU BP iteration $iter: avg message diff = $avg_diff"
            if avg_diff <= tolerance
                verbose && @info "Multi-GPU BP converged after $iter iterations."
                break
            end
        elseif verbose
            @info "Multi-GPU BP iteration $iter complete (convergence check skipped)."
        end
    end

    CUDA.device!(0)
    sync_inner_from_partitions!(cache)
    return cache
end

# ── 9. Device-local gate application with CPU QR/SVD fallback ──────────────────

# adapt_gate is a no-op for the multi-GPU cache
adapt_gate(gate::ITensor, ::MultiGPUBeliefPropagationCache) = gate

function apply_gate!(
        gate::ITensor, cache::MultiGPUBeliefPropagationCache;
        v⃗, apply_kwargs = (;), complex_type = ComplexF32,
    )
    bpc = inner_cache(cache)
    parts = partitions(cache)
    pmap = partition_map(cache)

    if length(v⃗) == 1
        v = only(v⃗)
        dev = pmap[v]
        CUDA.device!(dev)
        setindex_preserve!(bpc, parts[dev + 1].tensors[v], v)
        gate = tensor_to_device(gate, dev; complex_type)
        bpc, err = apply_gate!(gate, bpc; v⃗, apply_kwargs)
        set!(parts[dev + 1].tensors, v, network(bpc)[v])
        return cache, err
    end

    v1, v2 = v⃗
    home_dev = pmap[v1]
    guest_dev = pmap[v2]
    home_part = parts[home_dev + 1]
    guest_part = parts[guest_dev + 1]

    # Place the two acted tensors on the home device in the scratch cache.
    setindex_preserve!(bpc, home_part.tensors[v1], v1)
    if home_dev != guest_dev
        setindex_preserve!(bpc, tensor_to_device(guest_part.tensors[v2], home_dev; complex_type), v2)
    else
        setindex_preserve!(bpc, home_part.tensors[v2], v2)
    end

    # Place env messages on the home device so the core apply_gate! finds them.
    CUDA.device!(home_dev)
    g = graph(bpc)
    for v in v⃗, n in neighbors(g, v)
        n in v⃗ && continue
        e_in = NamedEdge(n => v)
        src_part = parts[pmap[n] + 1]
        m = haskey(src_part.messages, e_in) ? src_part.messages[e_in] : message(bpc, e_in)
        setmessage!(bpc, e_in, tensor_to_device(m, home_dev; complex_type))
    end

    gate_cpu_source = gate
    gate = tensor_to_device(gate, home_dev; complex_type)

    local err
    try
        bpc, err = apply_gate!(gate, bpc; v⃗, apply_kwargs)
    catch err_obj
        err_obj isa OutOfGPUMemoryError || rethrow()
        @warn "GPU OOM during 2-qubit gate QR/SVD; retrying on CPU." vertices = v⃗ device = home_dev
        GC.gc(); CUDA.reclaim()
        prepare_cpu_gate_retry!(cache, v⃗)
        bpc, err = apply_gate!(tensor_to_cpu(gate_cpu_source), bpc; v⃗, apply_kwargs)
        sync_cpu_gate_retry!(cache, v⃗; complex_type)
        return cache, err
    end

    # Move the updated tensors back to their owning devices and update caches.
    set!(home_part.tensors, v1, network(bpc)[v1])
    t2_done = network(bpc)[v2]
    if home_dev != guest_dev
        t2_back = tensor_to_device(t2_done, guest_dev; complex_type)
        setindex_preserve!(bpc, t2_back, v2)
        set!(guest_part.tensors, v2, t2_back)
    else
        set!(home_part.tensors, v2, t2_done)
    end

    # Propagate the SVD singular-value messages on the gate edge (set by the core
    # apply_gate!) to every partition that tracks them.
    broadcast_gate_edge_messages!(cache, v1, v2; complex_type)

    return cache, err
end

"""
    broadcast_gate_edge_messages!(cache, v1, v2; to_cpu, complex_type)

Propagate the gate-edge messages (set on the host cache by the core `apply_gate!`)
to every partition that tracks them, and back into the host mirror. Pass
`to_cpu = true` after a CPU OOM retry so the host message is densified on the CPU.
"""
function broadcast_gate_edge_messages!(
        cache::MultiGPUBeliefPropagationCache, v1, v2; to_cpu::Bool = false,
        complex_type = ComplexF32,
    )
    bpc = inner_cache(cache)
    parts = partitions(cache)
    for e in (NamedEdge(v1 => v2), NamedEdge(v2 => v1))
        m = denseblocks(to_cpu ? tensor_to_cpu(message(bpc, e)) : message(bpc, e))
        for part in parts
            haskey(part.messages, e) &&
                set!(part.messages, e, tensor_to_device(m, part.device_id; complex_type))
        end
        setmessage!(bpc, e, m)
    end
    return cache
end

"""
    prepare_cpu_gate_retry!(cache, v⃗)

Stage the acted vertices' tensors and their incoming env messages onto the host in
the scratch cache, so the core `apply_gate!` can be retried on CPU after a GPU OOM.
"""
function prepare_cpu_gate_retry!(cache::MultiGPUBeliefPropagationCache, v⃗)
    bpc = inner_cache(cache)
    parts = partitions(cache)
    pmap = partition_map(cache)
    for v in v⃗
        setindex_preserve!(bpc, tensor_to_cpu(parts[pmap[v] + 1].tensors[v]), v)
    end
    g = graph(bpc)
    for v in v⃗, n in neighbors(g, v)
        (n in v⃗) && continue
        e_in = NamedEdge(n => v)
        src_part = parts[pmap[n] + 1]
        m = haskey(src_part.messages, e_in) ? src_part.messages[e_in] : message(bpc, e_in)
        setmessage!(bpc, e_in, tensor_to_cpu(m))
    end
    return cache
end

"""
    sync_cpu_gate_retry!(cache, v⃗; complex_type)

After a CPU gate retry, move the updated tensors and the gate-edge messages back
onto their owning devices and into the partition caches.
"""
function sync_cpu_gate_retry!(cache::MultiGPUBeliefPropagationCache, v⃗; complex_type = ComplexF32)
    bpc = inner_cache(cache)
    parts = partitions(cache)
    pmap = partition_map(cache)
    ψ = network(bpc)
    for v in v⃗
        dev = pmap[v]
        t_gpu = tensor_to_device(ψ[v], dev; complex_type)
        setindex_preserve!(bpc, t_gpu, v)
        set!(parts[dev + 1].tensors, v, t_gpu)
    end

    v1, v2 = v⃗
    broadcast_gate_edge_messages!(cache, v1, v2; to_cpu = true, complex_type)
    CUDA.device!(0)
    return cache
end

# ── 10. Collection helpers ────────────────────────────────────────────────────

"""
    collect_to_cpu!(cache) -> BeliefPropagationCache

Pull all partition tensors and messages to host memory and return the wrapped
host `BeliefPropagationCache` for measurement. Use this for large final
measurements where collecting onto a single device would exceed VRAM.
"""
function collect_to_cpu!(cache::MultiGPUBeliefPropagationCache)
    bpc = inner_cache(cache)
    for part in partitions(cache)
        CUDA.device!(part.device_id)
        for v in part.vertices
            setindex_preserve!(bpc, tensor_to_cpu(part.tensors[v]), v)
        end
        for (e, m) in pairs(part.messages)
            setmessage!(bpc, e, tensor_to_cpu(m))
        end
    end
    CUDA.device!(0)
    return bpc
end

"""
    collect_to_device0!(cache; complex_type) -> BeliefPropagationCache

Copy all partition tensors and messages to device 0 and return the wrapped host
cache for measurement. Only suitable when the whole state fits on one device.
"""
function collect_to_device0!(cache::MultiGPUBeliefPropagationCache; complex_type = ComplexF32)
    bpc = inner_cache(cache)
    CUDA.device!(0)
    for part in partitions(cache)
        for v in part.vertices
            setindex_preserve!(bpc, tensor_to_device(part.tensors[v], 0; complex_type), v)
        end
        for (e, m) in pairs(part.messages)
            setmessage!(bpc, e, tensor_to_device(m, 0; complex_type))
        end
    end
    CUDA.device!(0)
    return bpc
end

# ── 11. Graph partitioning backends (KaHyPar) ─────────────────────────────────

"""
    partition_graph(::Algorithm"memory_balanced", g, n_parts; imbalance = 0.0)

Partition `g` into `n_parts` blocks, balancing the per-block *degree load* so the
high-degree vertices (the bond-dimension memory hot spots) are spread evenly across
devices. Returns a `Dictionary` mapping each vertex to a 0-indexed partition id.
"""
function partition_graph(alg::Algorithm"memory_balanced", g::AbstractGraph, n_parts::Integer)
    return _kahypar_partition(
        g, n_parts; weight = v -> degree(g, v), imbalance = get(alg.kwargs, :imbalance, 0.0)
    )
end

"""
    partition_graph(::Algorithm"min_cut", g, n_parts; imbalance = 0.0)

Partition `g` into `n_parts` blocks, balancing vertex count only so KaHyPar is free
to minimize the inter-device edge cut as aggressively as possible. Returns a
`Dictionary` mapping each vertex to a 0-indexed partition id.
"""
function partition_graph(alg::Algorithm"min_cut", g::AbstractGraph, n_parts::Integer)
    return _kahypar_partition(
        g, n_parts; weight = v -> 1, imbalance = get(alg.kwargs, :imbalance, 0.0)
    )
end

# Run KaHyPar with the given per-vertex `weight` function and balance `imbalance`,
function _kahypar_partition(g::AbstractGraph, n_parts::Integer; weight, imbalance = 0.0)
    nodes = collect(vertices(g))
    edges_list = collect(edges(g))
    row_of = Dict(v => i for (i, v) in enumerate(nodes))

    # Vertex × hyperedge incidence: each graph edge is a column with its two
    # incident vertex rows set to 1.
    rows = [row_of[v] for e in edges_list for v in (src(e), dst(e))]
    cols = [j for j in eachindex(edges_list) for _ in 1:2]
    A = sparse(rows, cols, ones(Int, length(rows)), length(nodes), length(edges_list))

    node_weights = [max(1, Int(weight(v))) for v in nodes]
    hypergraph = KaHyPar.HyperGraph(A, node_weights, ones(Int, length(edges_list)))
    parts = redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            KaHyPar.partition(hypergraph, n_parts; imbalance, configuration = :edge_cut)
        end
    end

    assignment = Dictionary{eltype(nodes), Int}()
    for (i, v) in enumerate(nodes)
        set!(assignment, v, Int(parts[i]))
    end
    return assignment
end
