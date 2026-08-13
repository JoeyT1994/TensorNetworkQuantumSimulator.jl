# GPU compatible dense simple update designed in minimise peak memory usage.
# See docs/spec_dense_update.md for the memory measurements behind the buffer reuse.

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

    mul_strided_batched!(
        reshape(dst_tensor, lead, chi, :),
        reshape(src_tensor, lead, chi, :),
        matrix,
    )

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

# Absorbs each matrix along the leading indices of `tensor`, alternating it with `scratch`. Returns
# both in their final roles: the buffer holding the result, then the one left free.
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

    # An even number of matrices ends the alternation with the result in the supplied `scratch`; a
    # caller that lent out part of a larger buffer needs all of it back free.
    if supplied && live !== reused
        copyto!(reused, live)
        live, free = reused, live
    end

    return live, free
end

# Thin QR of a tall matrix (m >= n) with Q written into `A`, returning `(Q, R)` where `Q === A`, or
# `nothing` when no in-place method exists for this array type. See docs/spec_dense_update.md.
function thin_qr_matrix!(A::AbstractMatrix)
    eltype(A) <: LinearAlgebra.BlasFloat || return nothing
    applicable(LAPACK.geqrf!, A) || return nothing
    n = size(A, 2)
    A, tau = LAPACK.geqrf!(A)
    R = triu!(A[1:n, :])                          # n × n; must precede orgqr!
    return LAPACK.orgqr!(A, tau), R
end

# Absorbs each matrix into its respective `inds`, then QRs with `qrinds` as the right-most indices.
# The third return is a free buffer the same length as `Q`, for `absorb_matrices_mul!`'s `scratch`.
function absorb_matrices_qr!(
        tensor::AbstractArray,
        matrices,
        inds::NTuple,
        qrinds::NTuple = Tuple(setdiff(collect(1:ndims(tensor)), inds));
        kwargs...
    )

    dims = map(i -> size(tensor, i), inds)
    qrdims = map(i -> size(tensor, i), qrinds)

    # Checked before the permute overwrites the input, which would leave a fallback nothing to read.
    prod(dims) >= prod(qrdims) || throw(
        DimensionMismatch(
            "absorb_matrices_qr!: $(dims) against $(qrdims) is a wide matrix, whose thin Q does " *
                "not fill the input's storage. Split it the other way, or use " *
                "`absorb_boundary_in!`, which carries the wide case."
        )
    )
    applicable(LAPACK.geqrf!, similar(tensor, 2, 2)) && eltype(tensor) <: LinearAlgebra.BlasFloat ||
        error(
        "absorb_matrices_qr!: no in-place QR for $(typeof(tensor)) with element type " *
            "$(eltype(tensor)).",
    )

    tensor, free = absorb_matrices!(tensor, matrices, inds, qrinds; kwargs...)

    _, R = thin_qr_matrix!(reshape(tensor, prod(dims), prod(qrdims)))

    Q = reshape(tensor, dims..., size(R, 1))
    R = reshape(R, size(R, 1), qrdims...)

    return Q, R, free
end

# Counterpart to `absorb_matrices_qr!`, consuming its `Q` and whatever its `R` has become; the
# inverse direction wants `op = transpose`. The second return is free and full size, not result size.
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

    # `R`'s trailing extent is not `chi`: truncation shrinks it, a grown bond enlarges it. The front
    # of `free` serves whenever the result fits, so only a grown bond allocates.
    rows, cols = prod(dims), prod(coldims)
    out = if rows * cols <= length(free)
        reshape(view(free, 1:(rows * cols)), rows, cols)
    else
        similar(free, rows, cols)
    end

    mul!(out, reshape(absorbed, rows, chi), reshape(R, chi, cols))

    return reshape(out, dims..., coldims...), absorbed
end

# `absorb_matrices_qr!` needs a tall matrix, which a low-degree vertex does not give. Then the whole
# absorbed tensor stands in for `R`, and `Q === nothing` signals that to `absorb_boundary_out!`.
function absorb_boundary_in!(
        tensor::AbstractArray,
        matrices,
        inds::NTuple,
        qrinds::NTuple = Tuple(setdiff(collect(1:ndims(tensor)), inds));
        kwargs...
    )

    rows = prod(i -> size(tensor, i), inds; init = 1)
    cols = prod(i -> size(tensor, i), qrinds; init = 1)

    rows >= cols || return (nothing, absorb_matrices!(tensor, matrices, inds, qrinds; kwargs...)...)

    return absorb_matrices_qr!(tensor, matrices, inds, qrinds; kwargs...)
end

# Counterpart to `absorb_boundary_in!`. Without a `Q`, `R` still carries the environment legs as its
# leading axes, so the chain runs over it directly and there is nothing left to multiply.
function absorb_boundary_out!(Q, matrices, R::AbstractArray, scratch; kwargs...)

    isnothing(Q) || return first(absorb_matrices_mul!(Q, matrices, R; scratch, kwargs...))

    # A truncation that grew the bond makes `R` longer than the buffer the QR side left behind.
    buffer = if length(R) <= length(scratch)
        reshape(view(scratch, 1:length(R)), size(R))
    else
        similar(R)
    end

    return first(absorb_chain!(R, buffer, matrices; kwargs...))
end

# Two-site simple update on dense vertex tensors, consuming both. `middle!` owns the gate and the
# truncation as `(R1, R2) -> (R1, R2, svals, err)`; `qrinds` names the axes right of each side's QR.
function simple_update_dense!(
        tensors, matrices, inv_matrices, qrinds, middle!;
        normalize_tensors = true,
    )

    # Larger side first, so its leftover buffer can host the smaller side's scratch; index 1 below is
    # that side, not the caller's. Reversing a 2-tuple is its own inverse, so `order` also undoes it.
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

# Read before anything is consumed: the caller needs these to decide whether the dense path applies.
function dense_update_legs(o::ITensor, ψᵥ::ITensor, side_envs)
    sinds = collect(commoninds(ψᵥ, o))
    legs = Index[only(commoninds(e, ψᵥ)) for e in side_envs]
    return legs, sinds
end

# Environment roots as (leg, leg') arrays, and `ψᵥ` laid out (environment legs..., site legs..., bond)
# so the QR's row block needs no permute. CONSUMES `ψᵥ`: `array` may return a view of its storage.
function dense_update_setup(ψᵥ::ITensor, side_envs, legs, sinds, lb, sqrt_cutoff)
    roots = pseudo_sqrt_inv_sqrt.(side_envs; cutoff = sqrt_cutoff)
    mats(es) = [ITensors.array(es[j], legs[j], prime(legs[j])) for j in eachindex(legs)]
    return (;
        tensor = ITensors.array(ψᵥ, legs..., sinds..., lb),
        matrices = mats(first.(roots)),
        inv_matrices = mats(dag.(last.(roots))),
        qrinds = Tuple((length(legs) + 1):(length(legs) + length(sinds) + 1)),
    )
end

# Where the gate left each site leg, ordered as the array was built; `uniqueinds` promises no order.
gate_image(t::ITensor, sinds) = Index[only(commoninds(t, Index[s, prime(s)])) for s in sinds]

# Keeps the factor exchange clear of `communicate_messages!`, which uses the default tag 0.
const _BOUNDARY_GATE_TAG = 1

# The array moves as a buffer, which CUDA-aware MPI can do device to device; the index vector goes
# separately, so neither side predicts the other's layout nor the bond dimension chosen.
function send_factor(t::ITensor, is, comm; dest)
    MPI.send(collect(is), comm; dest, tag = _BOUNDARY_GATE_TAG)
    MPI.Send(ITensors.array(t, is...), comm; dest, tag = _BOUNDARY_GATE_TAG)
    return t
end

function recv_factor(like::AbstractArray, comm; source)
    is = MPI.recv(comm; source, tag = _BOUNDARY_GATE_TAG)
    array = similar(like, dim.(Tuple(is)))
    MPI.Recv!(array, comm; source, tag = _BOUNDARY_GATE_TAG)
    return ITensors.itensor(array, is...), is
end

# Two-site update across a rank boundary; `compute` picks the rank running the gate and the SVD, and
# CONSUMES `ψᵥ`. Both ranks must reach this for the same gates in order, or the sends deadlock.
function simple_update_dense_boundary(
        o::ITensor, ψᵥ::ITensor;
        envs, lb::Index, compute::Bool, other_rank::Integer, comm::MPI.Comm,
        normalize_tensors = true, sqrt_cutoff = nothing, apply_kwargs...
    )

    @assert all(ndims(env) == 2 for env in envs)
    sqrt_cutoff_ref = isempty(envs) ? ψᵥ : first(envs)
    sqrt_cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(sqrt_cutoff_ref))) : sqrt_cutoff

    legs, sinds = dense_update_legs(o, ψᵥ, envs)
    (; tensor, matrices, inv_matrices, qrinds) =
        dense_update_setup(ψᵥ, envs, legs, sinds, lb, sqrt_cutoff)

    Q, R, scratch = absorb_boundary_in!(tensor, matrices, Tuple(eachindex(legs)), qrinds)

    # Without a thin Q the environment legs remain in the row block and pass through the gate.
    rowinds = isnothing(Q) ? legs : Index[Index(size(R, 1), "Link,qr")]
    Rt = ITensors.itensor(R, rowinds..., sinds..., lb)

    if compute
        Rother, isother = recv_factor(R, comm; source = other_rank)
        # The partner's site legs are the indices of `o` this side does not carry.
        sother = collect(commoninds(o, Rother))
        rowother = setdiff(collect(isother), Index[sother...; lb])

        singular_values! = Ref(ITensor())
        factored = factorize_svd(
            ITensors.apply(o, Rt * Rother), unioninds(rowinds, sinds);
            ortho = "none", singular_values!, apply_kwargs...,
        )
        isnothing(factored) && error(
            "simple_update_dense_boundary: the SVD did not converge. Pass a different algorithm " *
                "via the `alg` apply kwarg -- on GPU use alg = \"jacobi_algorithm\".",
        )
        L, Rr, spec = factored
        err, svals = spec.truncerr, singular_values![]
        newbond = only(commoninds(L, Rr))

        # Returned in the layout the partner sent, so `absorb_boundary_out!` needs no permute there.
        send_factor(
            Rr, Index[rowother...; gate_image(Rr, sother)...; newbond], comm; dest = other_rank
        )
        MPI.send((svals, err), comm; dest = other_rank, tag = _BOUNDARY_GATE_TAG)
        souts = gate_image(L, sinds)
        Rp = ITensors.array(L, rowinds..., souts..., newbond)
    else
        send_factor(Rt, Index[rowinds...; sinds...; lb], comm; dest = other_rank)
        Rpt, isp = recv_factor(R, comm; source = other_rank)
        svals, err = MPI.recv(comm; source = other_rank, tag = _BOUNDARY_GATE_TAG)
        # The row block returns unchanged and in the order sent, so the middle axes are the site legs.
        newbond = last(isp)
        souts = collect(isp)[(length(rowinds) + 1):(end - 1)]
        Rp = ITensors.array(Rpt, isp...)
    end

    u = absorb_boundary_out!(Q, inv_matrices, Rp, scratch; op = transpose)

    if normalize_tensors
        rmul!(u, inv(norm(u)))
        svals = normalize(svals)
    end

    # `itensor`, not `ITensor`: the capitalised constructor copies, undoing the buffering.
    return noprime(ITensors.itensor(u, legs..., souts..., newbond)), svals, err
end

# Signature-compatible with `simple_update`, but CONSUMES `ψ⃗` unless one of the fallbacks below fires
# first: it writes over the input's storage, which a shallow cache `copy` does not protect.
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
    side_envs = ntuple(i -> filter(env -> hascommoninds(env, ψ⃗[i]), envs), 2)
    @assert all(ndims(env) == 2 for env in vcat(side_envs...))
    sides = ntuple(i -> dense_update_legs(o, ψ⃗[i], side_envs[i]), 2)
    legs, sinds = first.(sides), last.(sides)

    # The thin QR needs the environment legs to outnumber the site and bond legs on both sides.
    rows(i) = prod(dim, legs[i]; init = 1)
    cols(i) = prod(dim, sinds[i]; init = 1) * dim(lb)
    all(i -> !isempty(legs[i]) && rows(i) >= cols(i), 1:2) || return fallback()

    setups = ntuple(
        i -> dense_update_setup(ψ⃗[i], side_envs[i], legs[i], sinds[i], lb, sqrt_cutoff), 2
    )
    tensors = ntuple(i -> setups[i].tensor, 2)
    matrices = ntuple(i -> setups[i].matrices, 2)
    inv_matrices = ntuple(i -> setups[i].inv_matrices, 2)
    qrinds = ntuple(i -> setups[i].qrinds, 2)

    # The new bond and the gate's image of the site legs are read off the SVD result and passed back
    # through `outinds` for the rewrap.
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
        souts = ntuple(i -> gate_image((L, Rr)[i], sinds[i]), 2)
        outinds[] = (souts, newbond)
        return ITensors.array(L, qb[1], souts[1]..., newbond),
            ITensors.array(Rr, qb[2], souts[2]..., newbond),
            singular_values![], spec.truncerr
    end

    (u1, u2), s_values, err = simple_update_dense!(
        tensors, matrices, inv_matrices, qrinds, middle!; normalize_tensors
    )

    souts, newbond = outinds[]
    # `itensor`, not `ITensor`: the capitalised constructor copies, undoing the buffering.
    updated_tensors = ITensor[
        ITensors.itensor(u1, legs[1]..., souts[1]..., newbond),
        ITensors.itensor(u2, legs[2]..., souts[2]..., newbond),
    ]

    return noprime.(updated_tensors), s_values, err
end
