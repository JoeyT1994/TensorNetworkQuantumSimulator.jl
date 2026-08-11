# GPU compatible dense simple update designed in minimise peak memory usage.

function absorb_first_matrix!(dst_tensor, src_tensor, matrix)

    @assert size(dst_tensor) == size(src_tensor)
    chi = size(src_tensor, 1)
    @assert chi == size(matrix, 1) == size(matrix, 2)

    mul!(reshape(dst_tensor, chi, :), matrix, reshape(src_tensor, chi, :))

    return dst_tensor
end

function absorb_matrix!(dst_tensor, src_tensor, matrix, ind)

    ind > 1 || return absorb_first_matrix!(dst_tensor, src_tensor, transpose(matrix))

    lead = size(src_tensor, 1)

    for i in 2:(ind - 1)
        lead *= size(src_tensor, i)
    end

    chi = size(src_tensor, ind)

    # mul_strided_batched!(
    #     reshape(dst_tensor, lead, chi, :),
    #     reshape(src_tensor, lead, chi, :),
    #     matrix,
    # )

    C = reshape(dst_tensor, lead, chi, :)
    A = reshape(src_tensor, lead, chi, :)
    B = matrix

    @tensor backend = backend C[lead, chi, trail] = A[lead, chi, trail] * B[chi, chi]


    return dst_tensor
end

# CPU
function mul_strided_batched!(C::AbstractArray, A::AbstractArray, B::AbstractMatrix)
    _, _, trailA = size(A)
    _, _, trailC = size(C)

    @assert trailA == trailC

    for t in 1:trailC
        mul!(view(C, :, :, t), view(A, :, :, t), B)
    end

    return C
end

# GPU methods live in ext/TensorNetworkQuantumSimulatorCUDAExt.jl, which routes this to
# `CUBLAS.gemm_strided_batched!` in one launch instead of one per slice.

# Absorb each matrix in `matrices` along the first `length(matrices)` indices of `tensor`,
# alternating between `tensor` and `scratch`. Returns both in their final roles: the buffer holding
# the result, then the one left free.
function absorb_chain!(tensor::AbstractArray, scratch::AbstractArray, matrices; op = identity)
    for (ind, matrix) in enumerate(matrices)
        absorb_matrix!(scratch, tensor, op(matrix), ind)
        tensor, scratch = scratch, tensor
    end
    return tensor, scratch
end

function absorb_matrices!(
        tensor::AbstractArray,
        matrices,
        inds::NTuple,
        other_inds::NTuple = Tuple(setdiff(collect(1:ndims(tensor)), inds));
        scratch = nothing,
        kwargs...
    )

    dims = map(i -> size(tensor, i), inds)
    qrdims = map(i -> size(tensor, i), other_inds)

    newsize = (dims..., qrdims...)

    # PERF: this is the only large allocation in this function if scratch not provided.
    supplied = !isnothing(scratch)
    scratch = supplied ? reshape(scratch, newsize) : similar(tensor, newsize)

    permutedims!(scratch, tensor, (inds..., other_inds...))

    # Actual tensor now lives in `scratch`; also repurpose the input as the scratch buffer.
    reused = reshape(tensor, newsize)
    live, free = absorb_chain!(scratch, reused, matrices; kwargs...)

    # A supplied `scratch` comes back free, and the result lands in the input's storage. An even
    # number of matrices ends the alternation the other way round, and a caller that lent out part
    # of a larger buffer needs all of it back.
    if supplied && live !== reused
        copyto!(reused, live)
        live, free = reused, live
    end

    return live, free
end

# Absorbs each matrix in `matrices` into each respective `inds` of `tensor`, and performs a
# QR factorization `qrinds` as the right-most indices. `qrinds` is optional and by default
# will be the complementary indices of `inds`, in ascending order.
# The third return is the buffer left free. It is the same length as `Q`, so it can go straight
# back as `absorb_matrices_mul!`'s `scratch` and spare that call its own allocation.
function absorb_matrices_qr!(
        tensor::AbstractArray,
        matrices,
        inds::NTuple,
        qrinds::NTuple = Tuple(setdiff(collect(1:ndims(tensor)), inds));
        kwargs...
    )

    dims = map(i -> size(tensor, i), inds)
    qrdims = map(i -> size(tensor, i), qrinds)

    tensor, free = absorb_matrices!(tensor, matrices, inds, qrinds; kwargs...)

    _, R = thin_qr_matrix!(reshape(tensor, prod(dims), prod(qrdims)))

    Q = reshape(tensor, dims..., size(R, 1))
    R = reshape(R, size(R, 1), qrdims...)

    return Q, R, free
end

# Counterpart to `absorb_matrices_qr!`, consuming its `Q` and whatever its `R` has become. The
# inverse direction wants `op = transpose`, contracting each matrix's second index, not its first.
# The second return is the buffer the final gemm read and no longer needs, full size rather than
# the result's, so a caller chaining another update can pass it straight back as `scratch`.
function absorb_matrices_mul!(
        tensor::AbstractArray,
        matrices,
        R::AbstractArray; # R is chi x chi'.
        scratch = nothing,
        kwargs...
    )

    dims = ntuple(i -> size(tensor, i), ndims(tensor) - 1)
    chi = size(tensor, ndims(tensor))
    coldims = ntuple(i -> size(R, i + 1), ndims(R) - 1)

    @assert chi == size(R, 1)

    scratch = isnothing(scratch) ? similar(tensor) : reshape(scratch, size(tensor))

    absorbed, free = absorb_chain!(tensor, scratch, matrices; kwargs...)

    # `R`'s trailing extent is not `chi`: an SVD that truncated makes it smaller, one that grew the
    # bond makes it larger. Take the front of `free` when the result fits, which is the truncating
    # and steady-state cases, and allocate only when it does not.
    rows, cols = prod(dims), prod(coldims)
    out = if rows * cols <= length(free)
        reshape(view(free, 1:(rows * cols)), rows, cols)
    else
        similar(free, rows, cols)
    end

    mul!(out, reshape(absorbed, rows, chi), reshape(R, chi, cols))

    return reshape(out, dims..., coldims...), absorbed
end

# Two-site simple update on dense vertex tensors. `middle!` owns the gate and the truncation:
# `(R1, R2) -> (R1, R2, svals, err)` over the two small QR factors.
# Per side, `qrinds` names the axes to the right of the QR -- (site, shared bond) -- and the
# environment legs are the remaining axes in ascending order. Both inputs are consumed, and each
# output is laid out (environment legs..., trailing axes of the `R` that `middle!` returned).
# One buffer is allocated, sized to the larger side, and every later scratch is carved out of it,
# so the footprint is 2x the larger vertex tensor plus 1x the smaller.
function simple_update_dense!(
        tensors, matrices, inv_matrices, qrinds, middle!;
        normalize_tensors = true,
    )

    # Larger side first, so its leftover buffer can host the smaller side's scratch. Reversing a
    # 2-tuple is its own inverse, so `order` both applies that and undoes it. Index 1 below is the
    # larger side, not the caller's side 1.
    flip = length(tensors[1]) < length(tensors[2])
    order(x1, x2) = flip ? reverse((x1, x2)) : (x1, x2)

    tensors, matrices, inv_matrices, qrinds =
        order(tensors...), order(matrices...), order(inv_matrices...), order(qrinds...)
    inds = ntuple(i -> Tuple(setdiff(1:ndims(tensors[i]), qrinds[i])), 2)

    Q1, R1, scratch = absorb_matrices_qr!(tensors[1], matrices[1], inds[1], qrinds[1])

    Q2, R2, _ = absorb_matrices_qr!(
        tensors[2], matrices[2], inds[2], qrinds[2];
        scratch = view(scratch, 1:length(tensors[2])),
    )

    R1, R2, svals, err = middle!(order(R1, R2)...)
    R1, R2 = order(R1, R2)

    # Deliberate rebinding of `scratch`.
    u1, scratch = absorb_matrices_mul!(Q1, inv_matrices[1], R1; op = transpose, scratch)

    u2, _ = absorb_matrices_mul!(
        Q2, inv_matrices[2], R2;
        op = transpose, scratch = view(scratch, 1:length(Q2)),
    )

    if normalize_tensors
        rmul!(u1, inv(norm(u1)))
        rmul!(u2, inv(norm(u2)))
        isnothing(svals) || (svals = normalize(svals))
    end

    return order(u1, u2), svals, err
end

function simple_update_dense_boundary!(
        tensor, matrices, inv_matrices, qrinds, middle!;
        normalize_tensors = true, compute_rank, other_rank, comm
    )


    inds = setdiff(1:ndims(tensor), qrinds)

    Q, R, scratch = absorb_matrices_qr!(tensor, matrices, inds, qrinds)

    if MPI.Comm_rank(comm) == other_rank
        MPI.Send(R, comm; dest = compute_rank)
        Q2 = Q

        chi = size(R2, 1)
        R2 = reshape(R, chi, length(R2) ÷ chi)

        MPI.Recv!()

        u2, _ = absorb_matrices_mul!(
            Q2, inv_matrices, R2;
            op = transpose, scratch = view(scratch, 1:length(Q2)),
        )

        if normalize_tensors
            rmul!(u2, inv(norm(u2)))
        end


    else MPI.Comm_rank(comm) == compute_rank
        R2 = MPI.Recv(comm; source = other_rank)
        Q1 = Q
        R1 = R

        chi = size(R1, ndims(R1))

        R1 = reshape(R1, length(R1) ÷ chi, chi)
        R2 = reshape(R2, chi, length(R2) ÷ chi)

        R1, R2, svals, err = MatrixAlgebraKit.svd_trunc(R1 * R2; trunc)

        MPI.Isend(R2, comm; dest = other_rank)
        MPI.Isend(svals, comm; dest = other_rank)
        MPI.Isend(err, comm; dest = other_rank)

        u1, scratch = absorb_matrices_mul!(Q1, inv_matrices, R1; op = transpose, scratch)

        if normalize_tensors
            rmul!(u1, inv(norm(u1)))
        end

    end
end

# Signature-compatible with `simple_update`, but CONSUMES `ψ⃗`: the dense path writes over the
# storage behind those ITensors, which is what holds the peak at 2x the larger vertex tensor rather
# than 4x. Callers must not read `ψ⃗` afterwards, and a shallow `copy` of a network or cache does not
# protect it -- that shares the ITensor objects and hence their buffers.
#
# The gate and the SVD stay in ITensor land on the small QR factors, so the truncation semantics are
# `factorize_svd`'s rather than a second copy of them.
#
# Falls back to `simple_update` for one-site gates, blocked storage, and any side whose environment
# legs are too short to give the QR a tall matrix. Every such check runs before `ψ⃗` is touched.
function simple_update_dense(
        o::ITensor, ψ⃗::Vector{<:ITensor};
        envs, normalize_tensors = true, sqrt_cutoff = nothing,
        apply_kwargs...
    )

    fallback() = simple_update(
        o, ψ⃗; envs, normalize_tensors, sqrt_cutoff, apply_kwargs...
    )

    length(ψ⃗) == 2 || return fallback()
    any(hasqns, ψ⃗) && return fallback()

    sqrt_cutoff_ref = isempty(envs) ? first(ψ⃗) : first(envs)
    sqrt_cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(sqrt_cutoff_ref))) : sqrt_cutoff

    lb = only(commoninds(ψ⃗[1], ψ⃗[2]))
    sinds = ntuple(i -> collect(commoninds(ψ⃗[i], o)), 2)
    side_envs = ntuple(i -> filter(env -> hascommoninds(env, ψ⃗[i]), envs), 2)
    @assert all(ndims(env) == 2 for env in vcat(side_envs...))
    legs = ntuple(i -> [only(commoninds(e, ψ⃗[i])) for e in side_envs[i]], 2)

    # The thin QR needs the environment legs to outnumber the site and bond legs on both sides.
    rows(i) = prod(dim, legs[i]; init = 1)
    cols(i) = prod(dim, sinds[i]; init = 1) * dim(lb)
    all(i -> !isempty(legs[i]) && rows(i) >= cols(i), 1:2) || return fallback()

    roots = ntuple(i -> pseudo_sqrt_inv_sqrt.(side_envs[i]; cutoff = sqrt_cutoff), 2)
    # Each environment matrix as an array ordered (leg, leg'), matching `absorb_matrix!`'s
    # convention that `op = identity` contracts the first index.
    mats(es, ls) = [ITensors.array(es[j], ls[j], prime(ls[j])) for j in eachindex(ls)]
    matrices = ntuple(i -> mats(first.(roots[i]), legs[i]), 2)
    inv_matrices = ntuple(i -> mats(dag.(last.(roots[i])), legs[i]), 2)

    # Environment legs lead so the QR's row block needs no permute of its own. `array` hands back a
    # view of the ITensor's own storage when that order already matches, and a permuted copy when it
    # does not; either is fine to overwrite, which is why `ψ⃗` is spent from here on.
    tensors = ntuple(i -> ITensors.array(ψ⃗[i], legs[i]..., sinds[i]..., lb), 2)
    qrinds = ntuple(i -> Tuple((length(legs[i]) + 1):(length(legs[i]) + length(sinds[i]) + 1)), 2)

    # The gate and the SVD act on the two O(chi^2) factors, so they go back through ITensors. The
    # site indices come out primed, and the new bond is whatever `factorize_svd` chose, so both are
    # read off the results rather than assumed and handed back out for the rewrap.
    outinds = Ref{Any}(nothing)
    function middle!(R1, R2)
        qb = ntuple(i -> Index(size((R1, R2)[i], 1), "Link,qr"), 2)
        Rs = ntuple(i -> ITensors.itensor((R1, R2)[i], qb[i], sinds[i]..., lb), 2)
        singular_values! = Ref(ITensor())
        factored = factorize_svd(
            ITensors.apply(o, Rs[1] * Rs[2]), unioninds(Index[qb[1]], sinds[1]);
            ortho = "none", singular_values!, apply_kwargs...,
        )
        isnothing(factored) && error(
            "simple_update_dense: the SVD did not converge. Pass a different algorithm via the " *
                "`alg` apply kwarg -- on GPU use alg = \"jacobi_algorithm\".",
        )
        L, Rr, spec = factored
        newbond = only(commoninds(L, Rr))
        souts = ntuple(i -> collect(uniqueinds((L, Rr)[i], qb[i], newbond)), 2)
        outinds[] = (souts, newbond)
        return ITensors.array(L, qb[1], souts[1]..., newbond),
            ITensors.array(Rr, qb[2], souts[2]..., newbond),
            singular_values![], spec.truncerr
    end

    (u1, u2), s_values, err = simple_update_dense!(
        tensors, matrices, inv_matrices, qrinds, middle!; normalize_tensors
    )

    souts, newbond = outinds[]
    # `itensor`, not `ITensor`: the capitalised constructor copies, which would undo the buffering.
    updated_tensors = ITensor[
        ITensors.itensor(u1, legs[1]..., souts[1]..., newbond),
        ITensors.itensor(u2, legs[2]..., souts[2]..., newbond),
    ]

    return noprime.(updated_tensors), s_values, err
end
