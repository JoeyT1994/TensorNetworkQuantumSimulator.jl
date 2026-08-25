# CUDA specialization for independent cut projectors. Equal-shape interfaces are factored together;
# unsupported shapes retain the scalar device path from `ctmenvironmentcache.jl`.

import CUDA

const _ctm_cuda_reported_shapes = Ref(false)

function _ctm_cuda_cut_prepare(request)
    co = combiner(request.ins...)
    io = combinedind(co)
    A = _ctm_block_matrix(request.Bw * co, io)
    B = _ctm_block_matrix(request.Be * co, io)
    return (; A, B, co, io)
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

function _ctm_cuda_cut_group!(output, prepared, indices, maxdim, opts)
    first_item = prepared[first(indices)]
    mA, nA = size(first_item.A)
    mB, nB = size(first_item.B)
    T = eltype(first_item.A)
    rA, rB = min(mA, nA), min(mB, nB)
    if length(indices) < 2
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
    prepared = [_ctm_cuda_cut_prepare(request) for request in requests]
    groups = Dict{Any, Vector{Int}}()
    for (index, item) in enumerate(prepared)
        key = (size(item.A), size(item.B), eltype(item.A))
        push!(get!(groups, key, Int[]), index)
    end
    if get(ENV, "CTM_CUDA_REPORT_SHAPES", "false") == "true" && !_ctm_cuda_reported_shapes[]
        _ctm_cuda_reported_shapes[] = true
        for (shape, indices) in sort!(collect(groups); by = first)
            println("CTM CUDA cut group: count=$(length(indices)), A=$(shape[1]), B=$(shape[2]), eltype=$(shape[3])")
        end
    end

    output = Vector{Any}(undef, length(requests))
    for indices in values(groups)
        _ctm_cuda_cut_group!(output, prepared, indices, maxdim, opts) && continue
        for index in indices
            request = requests[index]
            output[index] = _ctm_interface_proj2(
                request.Bw, request.Be, request.ins, maxdim, opts)
        end
    end
    return output
end
