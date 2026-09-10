using MatrixAlgebraKit: DivideAndConquer, Householder, Jacobi, QRIteration, default_driver,
    diagview, geqrf!, notrunc, qr_compact!, svd_trunc!, truncerror, truncrank, ungqr!,
    uppertriangular!
using TensorOperations: TensorOperations as TO

_as_eltype(::Type{T}, x::AbstractArray) where {T} = eltype(x) === T ? x : convert.(T, x)

_slot(flat::AbstractArray, dims) =
    reshape(prod(dims) == length(flat) ? flat : view(flat, 1:prod(dims)), dims)

function _scratch_slot(own::AbstractArray, allocator, n::Integer = length(own))
    ttype = TO.tensoradd_type(eltype(own), own, ((1,), ()), false)
    return TO.tensoralloc(ttype, (n,), Val(true), allocator)
end

function _step!(dst, A, pA, B, pB, pAB, backend, allocator)
    cp = TO.allocator_checkpoint!(allocator)
    try
        TO.tensorcontract!(
            dst, A, pA, false, B, pB, false, pAB, TO.One(), TO.Zero(), backend, allocator
        )
    finally
        TO.allocator_reset!(allocator, cp)
    end
    return dst
end

function absorb_into!(dst, src, mat, k, transposed, backend, allocator)
    N = ndims(src)
    c = N + 1
    IA = ntuple(i -> i == k ? c : i, N)
    IB = transposed ? (k, c) : (c, k)
    pA, pB, pAB = TO.contract_indices(IA, IB, ntuple(identity, N))
    return _step!(dst, src, pA, mat, pB, pAB, backend, allocator)
end

function bond_into!(dst, Q, R, backend, allocator)
    nq, nr = ndims(Q), ndims(R)
    c = nq + nr
    IA = ntuple(i -> i == nq ? c : i, nq)
    IB = ntuple(i -> i == 1 ? c : nq + i - 2, nr)
    pA, pB, pAB = TO.contract_indices(IA, IB, ntuple(identity, nq + nr - 2))
    return _step!(dst, Q, pA, R, pB, pAB, backend, allocator)
end

function qr_compact_inplace!(M::AbstractMatrix)
    m, n = size(M)
    k = min(m, n)
    driver = default_driver(Householder, M)
    M, tau = geqrf!(driver, M)
    R = similar(M, (k, n))
    copyto!(R, view(M, 1:k, 1:n))
    uppertriangular!(R)
    return ungqr!(driver, M, tau), R
end

function qr_forward!(own, scratch, A, perm, matrices, backend, allocator)
    tdims = ntuple(i -> size(A, perm[i]), ndims(A))
    cur = _slot(scratch, tdims)
    cp = TO.allocator_checkpoint!(allocator)
    try
        TO.tensoradd!(cur, A, (perm, ()), false, TO.One(), TO.Zero(), backend, allocator)
    finally
        TO.allocator_reset!(allocator, cp)
    end
    in_scratch = true
    for k in eachindex(matrices)
        dst = _slot(in_scratch ? own : scratch, tdims)
        absorb_into!(dst, cur, matrices[k], k, false, backend, allocator)
        cur, in_scratch = dst, !in_scratch
    end
    m = length(matrices)
    dims, qrdims = tdims[1:m], tdims[(m + 1):end]
    M = reshape(cur, prod(dims), prod(qrdims))
    inplace = size(M, 1) >= size(M, 2)
    Q, R = inplace ? qr_compact_inplace!(M) : qr_compact!(M)
    return reshape(Q, dims..., size(Q, 2)), reshape(R, size(R, 1), qrdims...),
        inplace ? in_scratch : nothing
end

function qr_backward!(own, scratch, Q, Rp, inv_matrices, q_in_scratch, backend, allocator)
    m = length(inv_matrices)
    udims = (size(Q)[1:(end - 1)]..., size(Rp)[2:end]...)
    n = prod(udims)
    if n <= min(length(own), length(scratch))
        a, b, to_a = own, scratch, isnothing(q_in_scratch) ? iseven(m) : q_in_scratch
    else
        a, b, to_a = similar(own, n), _scratch_slot(own, allocator, n), isodd(m + 1)
    end
    cur = bond_into!(_slot(to_a ? a : b, udims), Q, Rp, backend, allocator)
    to_a = !to_a
    for k in eachindex(inv_matrices)
        dst = _slot(to_a ? a : b, udims)
        absorb_into!(dst, cur, inv_matrices[k], k, true, backend, allocator)
        cur, to_a = dst, !to_a
    end
    return cur
end

const DENSE_SVD_KWARGS = (:maxdim, :mindim, :cutoff, :alg)

function truncation_strategy(; maxdim = nothing, mindim = nothing, cutoff = nothing)
    bounds = (
        (isnothing(maxdim) ? () : (truncrank(maxdim),))...,
        (isnothing(cutoff) ? () : (truncerror(; rtol = sqrt(cutoff)),))...,
    )
    isempty(bounds) && return notrunc()
    trunc = reduce(&, bounds)
    return isnothing(mindim) ? trunc : trunc | truncrank(mindim)
end

function svd_algorithm(alg::AbstractString)
    alg == "divide_and_conquer" && return DivideAndConquer()
    alg == "qr_iteration" && return QRIteration()
    alg == "jacobi_algorithm" && return Jacobi()
    return error(
        "svd_algorithm: no MatrixAlgebraKit counterpart for alg $(repr(alg)). Pass " *
            "\"divide_and_conquer\", \"qr_iteration\", \"jacobi_algorithm\", or a " *
            "MatrixAlgebraKit algorithm."
    )
end
svd_algorithm(alg) = alg

function gate_array(o::ITensor, sinds)
    ins = prime.(sinds)
    (!hasqns(o) && ndims(o) == 2 * length(sinds)) || return nothing
    issetequal(inds(o), Index[sinds..., ins...]) || return nothing
    return ITensors.array(o, sinds..., ins...)
end

function gate_split(
        gate::AbstractArray, R1::AbstractArray, R2::AbstractArray;
        maxdim = nothing, mindim = nothing, cutoff = nothing, alg = nothing,
        allocator = TO.DefaultAllocator(), backend = TO.DefaultBackend(), kwargs...
    )
    ns1, ns2 = ndims(R1) - 2, ndims(R2) - 2
    bond = ns1 + ns2 + 1
    @assert size(R1, ndims(R1)) == size(R2, ndims(R2))
    @assert ndims(gate) == 2 * (ns1 + ns2)

    r1labels = (-1, (1:ns1)..., bond)
    r2labels = (-(2 + ns1), ((ns1 + 1):(ns1 + ns2))..., bond)
    gatelabels = (
        (-i for i in 2:(1 + ns1))...,
        (-i for i in (3 + ns1):(2 + ns1 + ns2))...,
        (1:(ns1 + ns2))...,
    )
    tlabels = (-1, (1:ns1)..., -(2 + ns1), ((ns1 + 1):(ns1 + ns2))...)
    outlabels = ntuple(i -> -i, 2 + ns1 + ns2)

    TC = TO.promote_contract(TO.scalartype(R1), TO.scalartype(R2), TO.scalartype(gate))
    A1, A2, G = _as_eltype(TC, R1), _as_eltype(TC, R2), _as_eltype(TC, gate)
    q1, q2 = size(R1, 1), size(R2, 1)

    cp = TO.allocator_checkpoint!(allocator)
    U, S, Vt, discarded, d1, d2 = try
        pA, pB, pAB = TO.contract_indices(r1labels, r2labels, tlabels)
        T = TO.tensoralloc_contract(TC, A1, pA, false, A2, pB, false, pAB, Val(true), allocator)
        TO.tensorcontract!(
            T, A1, pA, false, A2, pB, false, pAB, TO.One(), TO.Zero(), backend, allocator
        )
        qA, qB, qAB = TO.contract_indices(tlabels, gatelabels, outlabels)
        M = TO.tensoralloc_contract(TC, T, qA, false, G, qB, false, qAB, Val(true), allocator)
        TO.tensorcontract!(
            M, T, qA, false, G, qB, false, qAB, TO.One(), TO.Zero(), backend, allocator
        )
        e1, e2 = size(M)[2:(1 + ns1)], size(M)[(3 + ns1):end]
        u, sv, vt, disc = svd_trunc!(
            reshape(M, q1 * prod(e1), q2 * prod(e2));
            trunc = truncation_strategy(; maxdim, mindim, cutoff), alg = svd_algorithm(alg),
        )
        (u, sv, vt, disc, e1, e2)
    finally
        TO.allocator_reset!(allocator, cp)
    end

    svals = diagview(S)
    k = length(svals)
    err = iszero(discarded) ? zero(discarded) :
        discarded^2 / (norm(svals)^2 + discarded^2)
    root = Diagonal(sqrt.(svals))

    return reshape(U * root, q1, d1..., k),
        reshape(permutedims(root * Vt, (2, 1)), q2, d2..., k),
        svals, err
end

function onesite_update!(o::ITensor, ψᵥ::ITensor; allocator = TO.DefaultAllocator())
    is = collect(inds(ψᵥ))
    sinds = collect(commoninds(ψᵥ, o))
    (hasqns(ψᵥ) || isempty(sinds)) && return nothing
    gate = gate_array(o, sinds)
    isnothing(gate) && return nothing
    pos = [findfirst(==(s), is) for s in sinds]
    any(isnothing, pos) && return nothing

    N, ns = length(is), length(sinds)
    IA = ntuple(i -> (k = findfirst(==(i), pos); isnothing(k) ? i : N + k), N)
    IB = ntuple(j -> j <= ns ? pos[j] : N + (j - ns), 2ns)
    pA, pB, pAB = TO.contract_indices(IA, IB, ntuple(identity, N))

    own = ITensors.data(ψᵥ)
    dims = ntuple(i -> dim(is[i]), N)
    cp = TO.allocator_checkpoint!(allocator)
    try
        dst = _slot(_scratch_slot(own, allocator), dims)
        _step!(
            dst, ITensors.array(ψᵥ), pA, _as_eltype(eltype(own), gate), pB, pAB,
            TO.DefaultBackend(), allocator
        )
        copyto!(_slot(own, dims), dst)
    finally
        TO.allocator_reset!(allocator, cp)
    end
    return ITensors.itensor(_slot(own, dims), is...)
end

function simple_update_dense!(
        tensors, matrices, inv_matrices, middle!;
        normalize_tensors = true, allocator = TO.DefaultAllocator(),
    )
    backend = TO.DefaultBackend()
    owns = ntuple(i -> vec(tensors[i]), 2)
    cp = TO.allocator_checkpoint!(allocator)
    try
        scratches = ntuple(i -> _scratch_slot(owns[i], allocator), 2)
        fwd = ntuple(
            i -> qr_forward!(
                owns[i], scratches[i], tensors[i], ntuple(identity, ndims(tensors[i])),
                matrices[i], backend, allocator
            ), 2
        )
        R1, R2, svals, err = middle!(fwd[1][2], fwd[2][2])
        us = ntuple(
            i -> qr_backward!(
                owns[i], scratches[i], fwd[i][1], (R1, R2)[i], inv_matrices[i], fwd[i][3],
                backend, allocator
            ), 2
        )
        if normalize_tensors
            foreach(u -> rmul!(u, inv(norm(u))), us)
            isnothing(svals) || (svals = normalize(svals))
        end
        return us, svals, err
    finally
        TO.allocator_reset!(allocator, cp)
    end
end

function dense_update_legs(o::ITensor, ψᵥ::ITensor, side_envs)
    sinds = collect(commoninds(ψᵥ, o))
    legs = Index[only(commoninds(e, ψᵥ)) for e in side_envs]
    return legs, sinds
end

# `own` is `ψᵥ`'s own storage, which the chain uses as one of its two slots and overwrites -- the
# caller must have given up every other reference to it.
function dense_update_setup(ψᵥ::ITensor, side_envs, legs, sinds, lb, sqrt_cutoff)
    roots = pseudo_sqrt_inv_sqrt.(side_envs; cutoff = sqrt_cutoff)
    T = eltype(ITensors.data(ψᵥ))
    mats(es) = [
        _as_eltype(T, ITensors.array(es[j], legs[j], prime(legs[j]))) for j in eachindex(legs)
    ]
    is = collect(inds(ψᵥ))
    target = Index[legs..., sinds..., lb]
    return (;
        array = ITensors.array(ψᵥ),
        perm = ntuple(i -> findfirst(==(target[i]), is), length(is)),
        own = ITensors.data(ψᵥ),
        matrices = mats(first.(roots)),
        inv_matrices = mats(dag.(last.(roots))),
    )
end

bond_values(svals, newbond::Index) = ITensors.diag_itensor(svals, newbond, sim(newbond))

const _BOUNDARY_GATE_TAG = 1

# `MPI.Send`/`MPI.Recv!` on a device buffer are issued outside the stream the contraction kernels
# run on. A GPU caller closes that gap with
# `TensorNetworkQuantumSimulator.mpi_device_synchronize!() = CUDA.synchronize()`.
mpi_device_synchronize!() = nothing

function send_factor(a::AbstractArray, is, comm; dest)
    MPI.send(collect(is), comm; dest, tag = _BOUNDARY_GATE_TAG)
    mpi_device_synchronize!()
    MPI.Send(a, comm; dest, tag = _BOUNDARY_GATE_TAG)
    return a
end

function recv_factor(like::AbstractArray, comm; source)
    is = MPI.recv(comm; source, tag = _BOUNDARY_GATE_TAG)
    a = similar(like, dim.(Tuple(is)))
    mpi_device_synchronize!()
    MPI.Recv!(a, comm; source, tag = _BOUNDARY_GATE_TAG)
    mpi_device_synchronize!()
    return a, is
end

# `compute` picks the rank running the gate and the SVD. Both ranks must reach this for the same
# gates in the same order, or the sends deadlock.
function simple_update_dense_boundary(
        o::ITensor, ψᵥ::ITensor;
        envs, lb::Index, compute::Bool, other_rank::Integer, comm::MPI.Comm,
        normalize_tensors = true, sqrt_cutoff = nothing,
        allocator = TO.DefaultAllocator(), apply_kwargs...
    )
    @assert all(ndims(env) == 2 for env in envs)
    all(in(DENSE_SVD_KWARGS), keys(apply_kwargs)) || error(
        "simple_update_dense_boundary: cannot honour apply kwargs " *
            "$(setdiff(keys(apply_kwargs), DENSE_SVD_KWARGS)); this path implements " *
            "$(DENSE_SVD_KWARGS) only.",
    )
    sqrt_cutoff_ref = isempty(envs) ? ψᵥ : first(envs)
    sqrt_cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(sqrt_cutoff_ref))) : sqrt_cutoff

    legs, sinds = dense_update_legs(o, ψᵥ, envs)
    setup = dense_update_setup(ψᵥ, envs, legs, sinds, lb, sqrt_cutoff)
    backend = TO.DefaultBackend()

    cp = TO.allocator_checkpoint!(allocator)
    scratch = _scratch_slot(setup.own, allocator)
    Q, R, q_in_scratch = qr_forward!(
        setup.own, scratch, setup.array, setup.perm, setup.matrices, backend, allocator
    )

    if compute
        Rother, isother = recv_factor(R, comm; source = other_rank)
        sother = collect(isother)[2:(end - 1)]
        gate = gate_array(o, Index[sinds...; sother...])
        isnothing(gate) && error(
            "simple_update_dense_boundary: $(o) is not a plain dense operator on the site indices " *
                "$(Index[sinds...; sother...]) and their primes.",
        )
        L, Rr, svals, err = gate_split(gate, R, Rother; allocator, apply_kwargs...)
        newbond = Index(length(svals), "Link,l")
        send_factor(Rr, Index[first(isother); sother...; newbond], comm; dest = other_rank)
        MPI.send((svals, err), comm; dest = other_rank, tag = _BOUNDARY_GATE_TAG)
        Rp = L
    else
        send_factor(R, Index[Index(size(R, 1), "Link,qr"); sinds...; lb], comm; dest = other_rank)
        Rp, isp = recv_factor(R, comm; source = other_rank)
        svals, err = MPI.recv(comm; source = other_rank, tag = _BOUNDARY_GATE_TAG)
        newbond = last(isp)
    end

    u = try
        qr_backward!(
            setup.own, scratch, Q, Rp, setup.inv_matrices, q_in_scratch, backend, allocator
        )
    finally
        TO.allocator_reset!(allocator, cp)
    end

    if normalize_tensors
        rmul!(u, inv(norm(u)))
        svals = normalize(svals)
    end
    return ITensors.itensor(u, legs..., sinds..., newbond), bond_values(svals, newbond), err
end

function simple_update_dense(
        o::ITensor, ψ⃗::Vector{<:ITensor};
        envs, normalize_tensors = true, sqrt_cutoff = nothing,
        allocator = TO.DefaultAllocator(), apply_kwargs...
    )
    fallback() = simple_update(o, ψ⃗; envs, normalize_tensors, sqrt_cutoff, apply_kwargs...)

    length(ψ⃗) == 2 || return fallback()
    any(hasqns, ψ⃗) && return fallback()
    all(in(DENSE_SVD_KWARGS), keys(apply_kwargs)) || return fallback()

    sqrt_cutoff_ref = isempty(envs) ? first(ψ⃗) : first(envs)
    sqrt_cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(sqrt_cutoff_ref))) : sqrt_cutoff

    lb = only(commoninds(ψ⃗[1], ψ⃗[2]))
    side_envs = ntuple(i -> filter(env -> hascommoninds(env, ψ⃗[i]), envs), 2)
    @assert all(ndims(env) == 2 for env in vcat(side_envs...))
    sides = ntuple(i -> dense_update_legs(o, ψ⃗[i], side_envs[i]), 2)
    legs, sinds = first.(sides), last.(sides)

    all(i -> !isempty(legs[i]), 1:2) || return fallback()
    gate = gate_array(o, Index[sinds[1]...; sinds[2]...])
    isnothing(gate) && return fallback()

    setups = ntuple(
        i -> dense_update_setup(ψ⃗[i], side_envs[i], legs[i], sinds[i], lb, sqrt_cutoff), 2
    )
    fill!(ψ⃗, ITensor())
    backend = TO.DefaultBackend()

    cp = TO.allocator_checkpoint!(allocator)
    try
        scratches = ntuple(i -> _scratch_slot(setups[i].own, allocator), 2)
        fwd = ntuple(
            i -> qr_forward!(
                setups[i].own, scratches[i], setups[i].array, setups[i].perm,
                setups[i].matrices, backend, allocator
            ), 2
        )
        R1, R2, svals, err = gate_split(gate, fwd[1][2], fwd[2][2]; allocator, apply_kwargs...)
        us = ntuple(
            i -> qr_backward!(
                setups[i].own, scratches[i], fwd[i][1], (R1, R2)[i],
                setups[i].inv_matrices, fwd[i][3], backend, allocator
            ), 2
        )
        if normalize_tensors
            foreach(u -> rmul!(u, inv(norm(u))), us)
            isnothing(svals) || (svals = normalize(svals))
        end
        newbond = Index(length(svals), "Link,l")
        updated_tensors = ITensor[
            ITensors.itensor(us[1], legs[1]..., sinds[1]..., newbond),
            ITensors.itensor(us[2], legs[2]..., sinds[2]..., newbond),
        ]
        return updated_tensors, bond_values(svals, newbond), err
    finally
        TO.allocator_reset!(allocator, cp)
    end
end
