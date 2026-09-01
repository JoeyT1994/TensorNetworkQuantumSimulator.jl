# CUDA specialization for independent cut projectors. Equal-shape interfaces are factored together;
# unsupported shapes retain the scalar device path from `ctmenvironmentcache.jl`.

import CUDA

const _ctm_cuda_reported_shapes = Ref(false)

# Greedy C/T construction is a long dependency chain of tiny contractions, decompositions,
# normalizations, and scalar rank decisions.  On CUDA that means tens of thousands of launches and
# host/device synchronizations before the first genuine CTMRG sweep.  Build it on the CPU and cross
# the device boundary once in each direction; subsequent sweeps and readout remain device-resident.
function _ctm_initial_env_backend(::Type{<:CUDA.CuArray}, cache::CTMEnvironmentCache)
    cpu_network = adapt(Array, network(cache))
    cpu_cache = CTMEnvironmentCache(cpu_network, cache.grid, cache.coords, cache.dims,
                                    cache.maxdim, nothing, cache.options)
    return adapt(CUDA.CuArray, vertex_environments(cpu_cache))
end

# `norm(::CuArray)` returns a host scalar and therefore synchronizes every C/T rebuild.  Keep the
# singleton reduction on-device and broadcast the safe divisor back over the tensor.  Zero and
# non-finite norms retain the historical no-rescale behavior by selecting a divisor of one.
function _ctm_rescale_backend(::Type{<:CUDA.CuArray}, tensor::ITensor)
    get(ENV, "CTM_CUDA_DEVICE_RESCALE", "true") == "true" || return nothing
    order = collect(inds(tensor))
    A = ITensors.array(tensor, order...)
    dims = ntuple(identity, ndims(A))
    scale = sqrt.(sum(abs2, A; dims))
    safe_scale = ifelse.(isfinite.(scale) .& .!iszero.(scale), scale,
                         one(eltype(scale)))
    return ITensor(A ./ safe_scale, order...)
end

function _ctm_cuda_cut_prepare(request)
    co = combiner(request.ins...)
    io = combinedind(co)
    A = _ctm_block_matrix(request.Bw * co, io)
    B = _ctm_block_matrix(request.Be * co, io)
    return (; A, B, co, io)
end

function _ctm_cuda_cut_shape(request)
    interface = Set(request.ins)
    n = prod((ITensors.dim(index) for index in request.ins); init = 1)
    mA = prod((ITensors.dim(index) for index in inds(request.Bw) if !(index in interface)); init = 1)
    mB = prod((ITensors.dim(index) for index in inds(request.Be) if !(index in interface)); init = 1)
    return ((mA, n), (mB, n), eltype(request.Bw))
end

function _ctm_cuda_batched_r(matrices, rows, cols, ::Type{T}) where {T}
    work = CUDA.CuMatrix{T}[copy(matrix) for matrix in matrices]
    CUDA.CUBLAS.geqrf_batched!(work)
    rank = min(rows, cols)
    return CUDA.CuMatrix{T}[triu(matrix[1:rank, :]) for matrix in work]
end

function _ctm_cuda_group_svd(W)
    if max(size(W, 1), size(W, 2)) <= 32
        return CUDA.CUSOLVER.gesvdj!('V', copy(W))
    end

    # CUSOLVER's exact batched Jacobi solver is limited to 32×32. Its approximate `gesvda`
    # and a grouped-QR/scalar-SVD hybrid were both measured here and are dramatically slower for
    # CTMRG's rectangular double-layer interfaces, so preserve the established scalar exact path.
    return nothing
end

# Small dense factorizations are latency-bound on CUDA.  On the reference RTX 3070, a warmed
# batch of 48 ComplexF32 72x72 SVDs takes 0.20 s on CUDA versus 0.077 s in LAPACK, while moving
# the complete 4 MB batch across PCIe takes 0.0004 s.  Keep the tensor contractions on-device but
# factor equal-shape small interfaces as one host batch.  The crossover becomes hardware-sensitive
# around n=128, so use the conservative n<=96 region where the measured win is unambiguous.
_ctm_cuda_host_factor_max() = parse(Int, get(ENV, "CTM_CUDA_HOST_FACTOR_MAX", "96"))

function _ctm_cuda_stack_device(matrices, rows, cols, ::Type{T}) where {T}
    matrices = collect(matrices)
    stacked = CUDA.CuArray{T}(undef, rows, cols, length(matrices))
    for (batch, matrix) in enumerate(matrices)
        copyto!(@view(stacked[:, :, batch]), matrix)
    end
    return stacked
end

_ctm_cuda_stack_to_host(matrices, rows, cols, ::Type{T}) where {T} =
    Array(_ctm_cuda_stack_device(matrices, rows, cols, T)) # one sync for the complete shape group

function _ctm_cuda_cut_host_group!(output, prepared, indices, maxdim, opts)
    length(indices) < 2 && return false
    first_item = prepared[first(indices)]
    mA, nA = size(first_item.A)
    mB, nB = size(first_item.B)
    rA, rB = min(mA, nA), min(mB, nB)
    max(rA, rB) <= _ctm_cuda_host_factor_max() || return false

    T = eltype(first_item.A)
    As = _ctm_cuda_stack_to_host((prepared[i].A for i in indices), mA, nA, T)
    Bs = _ctm_cuda_stack_to_host((prepared[i].B for i in indices), mB, nB, T)
    host_projectors = Vector{Any}(undef, length(indices))
    rank_groups = Dict{Int, Vector{Int}}()
    for (batch, index) in enumerate(indices)
        item = prepared[index]
        PA, PB, w = _ctm_twosided_projector_qr(
            @view(As[:, :, batch]), @view(Bs[:, :, batch]), item.io, maxdim, opts)
        host_projectors[batch] = (PA, PB, w)
        push!(get!(rank_groups, ITensors.dim(w), Int[]), batch)
    end

    # Move each equal-rank projector family in two contiguous transfers instead of sending every
    # tiny PA/PB separately.  The per-interface device copies below only carve independent storage
    # out of the batch; no pageable-host transfer or synchronization remains in that loop.
    for (rank, batches) in rank_groups
        PA_host = cat((ITensors.array(host_projectors[b][1], prepared[indices[b]].io,
                                     host_projectors[b][3]) for b in batches)...; dims = 3)
        PB_host = cat((ITensors.array(host_projectors[b][2], host_projectors[b][3],
                                     prepared[indices[b]].io) for b in batches)...; dims = 3)
        PA_device = CUDA.CuArray(PA_host)
        PB_device = CUDA.CuArray(PB_host)
        for (local_batch, batch) in enumerate(batches)
            index = indices[batch]
            item = prepared[index]
            w = host_projectors[batch][3]
            PA = ITensor(copy(@view(PA_device[:, :, local_batch])), item.io, w)
            PB = ITensor(copy(@view(PB_device[:, :, local_batch])), w, item.io)
            output[index] = (PA * item.co, PB * item.co, w)
        end
    end
    return true
end

function _ctm_cuda_cut_group!(output, prepared, indices, maxdim, opts)
    first_item = prepared[first(indices)]
    mA, nA = size(first_item.A)
    mB, nB = size(first_item.B)
    T = eltype(first_item.A)
    rA, rB = min(mA, nA), min(mB, nB)
    # The exact batched Jacobi SVD is limited to matrices no larger than 32 in either
    # direction.  Reject unsupported groups before launching batched QR/GEMM; previously
    # those results were thrown away and the complete scalar projector was then repeated.
    if length(indices) < 2 || max(rA, rB) > 32
        return false
    end

    RAs = _ctm_cuda_batched_r((prepared[i].A for i in indices), mA, nA, T)
    RBs = _ctm_cuda_batched_r((prepared[i].B for i in indices), mB, nB, T)
    RA3 = cat(RAs...; dims = 3)
    RB3 = cat(RBs...; dims = 3)
    W = CUDA.CUBLAS.gemm_strided_batched('N', 'T', RA3, RB3)
    F = _ctm_cuda_group_svd(W)
    isnothing(F) && return false
    U, S, V = F
    spectra = Array(S)  # one synchronization for the whole shape group

    rank_groups = Dict{Int, Vector{Int}}()
    for batch in eachindex(indices)
        rank = _ctm_twosided_rank(@view(spectra[:, batch]), maxdim, opts)
        push!(get!(rank_groups, rank, Int[]), batch)
    end
    for (rank, batches) in rank_groups
        RA_batch = RA3[:, :, batches]
        RB_batch = RB3[:, :, batches]
        U_batch = U[:, 1:rank, batches]
        V_batch = V[:, 1:rank, batches]
        invsqrt = inv.(sqrt.(S[1:rank, batches]))
        PA_batch = CUDA.CUBLAS.gemm_strided_batched('T', 'N', RB_batch, V_batch)
        PB_batch = CUDA.CUBLAS.gemm_strided_batched('C', 'N', U_batch, RA_batch)
        PA_batch .*= reshape(invsqrt, 1, rank, :)
        PB_batch .*= reshape(invsqrt, rank, 1, :)
        for (local_batch, batch) in enumerate(batches)
            index = indices[batch]
            item = prepared[index]
            w = Index(rank)
            PA = ITensor(copy(@view(PA_batch[:, :, local_batch])), item.io, w)
            PB = ITensor(copy(@view(PB_batch[:, :, local_batch])), w, item.io)
            output[index] = (PA * item.co, PB * item.co, w)
        end
    end
    return true
end

function _ctm_interface_proj2_batch_backend(::Type{<:CUDA.CuArray}, requests,
                                            maxdim::Integer, opts::CTMOptions)
    groups = Dict{Any, Vector{Int}}()
    for (index, request) in enumerate(requests)
        key = _ctm_cuda_cut_shape(request)
        push!(get!(groups, key, Int[]), index)
    end
    if get(ENV, "CTM_CUDA_REPORT_SHAPES", "false") == "true" && !_ctm_cuda_reported_shapes[]
        _ctm_cuda_reported_shapes[] = true
        for (shape, indices) in sort!(collect(groups); by = first)
            println("CTM CUDA cut group: count=$(length(indices)), A=$(shape[1]), B=$(shape[2]), eltype=$(shape[3])")
        end
    end

    output = Vector{Any}(undef, length(requests))
    prepared = Vector{Any}(undef, length(requests))
    for (shape, indices) in groups
        mA, nA = shape[1]
        mB, nB = shape[2]
        maxrank = max(min(mA, nA), min(mB, nB))
        grouped = length(indices) >= 2 &&
                  (maxrank <= 32 || maxrank <= _ctm_cuda_host_factor_max())
        if grouped
            for index in indices
                prepared[index] = _ctm_cuda_cut_prepare(requests[index])
            end
            _ctm_cuda_cut_group!(output, prepared, indices, maxdim, opts) && continue
            _ctm_cuda_cut_host_group!(output, prepared, indices, maxdim, opts) && continue
        end

        # Large exact SVDs have no supported batched cuSOLVER path.  Prepare and finish one
        # interface at a time instead of retaining every O((chi*D_layer)^2) matrix concurrently.
        # At D=10, chi=20 this is the difference between a streamed 2000-wide workspace and filling
        # essentially the entire 8 GB device with matrices that will only be factored serially.
        for index in indices
            item = grouped ? prepared[index] : _ctm_cuda_cut_prepare(requests[index])
            PA, PB, w = _ctm_twosided_projector_qr(
                item.A, item.B, item.io, maxdim, opts)
            output[index] = (PA * item.co, PB * item.co, w)
        end
    end
    return output
end
