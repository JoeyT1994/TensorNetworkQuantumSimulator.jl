# GPU compatible dense simple update. Each absorption is one `ncon` network, so the intermediates are
# the backend's business; `allocator` and `backend` kwargs reach `ncon`.

using MatrixAlgebraKit: DivideAndConquer, Jacobi, QRIteration, diagview, notrunc, qr_compact!,
    svd_trunc!, truncerror, truncrank
using TensorOperations: TensorOperations as TO

# `transposed` contracts the matrix's second index rather than its first.
matrix_labels(k::Int, transposed::Bool) = transposed ? [-k, k] : [k, -k]

function absorb_matrices(tensor::AbstractArray, matrices; transposed = false, kwargs...)

    m = length(matrices)
    @assert m <= ndims(tensor)

    labels = Int[1:m..., (-i for i in (m + 1):ndims(tensor))...]
    network = Vector{Int}[labels, (matrix_labels(k, transposed) for k in 1:m)...]

    return TO.ncon((tensor, matrices...), network; order = collect(1:m), kwargs...)
end

# Extents of the absorbed axes and of the rest, which are the QR's rows and columns.
absorbed_split(tensor::AbstractArray, matrices) =
    size(tensor)[1:length(matrices)], size(tensor)[(length(matrices) + 1):end]

# The QR runs on the absorbed copy with the absorbed axes as its rows, so `tensor` is untouched.
function absorb_matrices_qr(tensor::AbstractArray, matrices; kwargs...)

    dims, qrdims = absorbed_split(tensor, matrices)

    absorbed = absorb_matrices(tensor, matrices; kwargs...)

    Q, R = qr_compact!(reshape(absorbed, prod(dims), prod(qrdims)))

    return reshape(Q, dims..., size(Q, 2)), reshape(R, size(R, 1), qrdims...)
end

# Counterpart to `absorb_matrices_qr`; the inverse direction wants `transposed = true`.
function absorb_matrices_mul(
        Q::AbstractArray,
        matrices,
        R::AbstractArray;
        transposed = false,
        kwargs...
    )

    nq, nr, m = ndims(Q), ndims(R), length(matrices)
    bond = m + 1

    @assert m < nq
    @assert size(Q, nq) == size(R, 1)

    # Output axes are `Q`'s up to the bond, then `R`'s past it, so `R`'s j-th trailing axis lands at
    # output position `npass + j`.
    npass = nq - 1
    qlabels = Int[1:m..., (-i for i in (m + 1):npass)..., bond]
    rlabels = Int[bond, (-(npass + j) for j in 1:(nr - 1))...]

    network = Vector{Int}[qlabels, (matrix_labels(k, transposed) for k in 1:m)..., rlabels]

    return TO.ncon((Q, matrices..., R), network; order = collect(1:bond), kwargs...)
end

# The truncation controls this path implements; `factorize_svd` honours more, so the rest falls back.
const DENSE_SVD_KWARGS = (:maxdim, :mindim, :cutoff, :alg)

# ITensors' `cutoff` bounds the discarded weight -- the squared norm of the discarded singular values
# -- relative to the total, where `truncerror` bounds that norm itself.
function truncation_strategy(; maxdim = nothing, mindim = nothing, cutoff = nothing)
    bounds = (
        (isnothing(maxdim) ? () : (truncrank(maxdim),))...,
        (isnothing(cutoff) ? () : (truncerror(; rtol = sqrt(cutoff)),))...,
    )
    isempty(bounds) && return notrunc()
    trunc = reduce(&, bounds)
    return isnothing(mindim) ? trunc : trunc | truncrank(mindim)
end

# The names ITensors' `svd` takes, so one set of `apply_kwargs` also serves the fallback.
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

# The gate as (site outs..., site ins...), or `nothing` when it is not a plain dense operator.
function gate_array(o::ITensor, sinds)
    ins = prime.(sinds)
    (!hasqns(o) && ndims(o) == 2 * length(sinds)) || return nothing
    issetequal(inds(o), Index[sinds..., ins...]) || return nothing
    return ITensors.array(o, sinds..., ins...)
end

# Gates the two factors' site axes and splits with a truncated SVD, the square root of the singular
# values going to each side. Every factor here is (row block, site axes..., bond).
function gate_split(
        gate::AbstractArray, R1::AbstractArray, R2::AbstractArray;
        maxdim = nothing, mindim = nothing, cutoff = nothing, alg = nothing, kwargs...
    )

    ns1, ns2 = ndims(R1) - 2, ndims(R2) - 2
    bond = ns1 + ns2 + 1

    @assert size(R1, ndims(R1)) == size(R2, ndims(R2))
    @assert ndims(gate) == 2 * (ns1 + ns2)

    # Negative labels are output positions: each factor's row block followed by its gated site axes.
    r1labels = Int[-1, 1:ns1..., bond]
    r2labels = Int[-(2 + ns1), (ns1 + 1):(ns1 + ns2)..., bond]
    gatelabels = Int[
        (-i for i in 2:(1 + ns1))...,
        (-i for i in (3 + ns1):(2 + ns1 + ns2))...,
        1:(ns1 + ns2)...,
    ]

    M = TO.ncon(
        (R1, R2, gate), Vector{Int}[r1labels, r2labels, gatelabels];
        order = collect(1:bond), kwargs...
    )

    q1, q2 = size(R1, 1), size(R2, 1)
    d1, d2 = size(M)[2:(1 + ns1)], size(M)[(3 + ns1):end]

    U, S, Vt, discarded = svd_trunc!(
        reshape(M, q1 * prod(d1), q2 * prod(d2));
        trunc = truncation_strategy(; maxdim, mindim, cutoff), alg = svd_algorithm(alg),
    )

    svals = diagview(S)
    k = length(svals)
    # `discarded` is the norm of the dropped values; `factorize_svd` reports the weight they carried.
    err = iszero(discarded) ? zero(discarded) :
        discarded^2 / (norm(svals)^2 + discarded^2)
    root = Diagonal(sqrt.(svals))

    return reshape(U * root, q1, d1..., k),
        reshape(permutedims(root * Vt, (2, 1)), q2, d2..., k),
        svals, err
end

# Two-site simple update on dense vertex tensors, each laid out with its environment legs first.
# `middle!` owns the gate and the truncation as `(R1, R2) -> (R1, R2, svals, err)`.
function simple_update_dense!(
        tensors, matrices, inv_matrices, middle!;
        normalize_tensors = true,
    )

    factors = ntuple(i -> absorb_matrices_qr(tensors[i], matrices[i]), 2)

    R1, R2, svals, err = middle!(last(factors[1]), last(factors[2]))

    us = ntuple(
        i -> absorb_matrices_mul(
            first(factors[i]), inv_matrices[i], (R1, R2)[i]; transposed = true
        ), 2
    )

    if normalize_tensors
        foreach(u -> rmul!(u, inv(norm(u))), us)
        isnothing(svals) || (svals = normalize(svals))
    end

    return us, svals, err
end

# The caller needs these to decide whether the dense path applies.
function dense_update_legs(o::ITensor, ψᵥ::ITensor, side_envs)
    sinds = collect(commoninds(ψᵥ, o))
    legs = Index[only(commoninds(e, ψᵥ)) for e in side_envs]
    return legs, sinds
end

# Environment roots as (leg, leg') arrays, and `ψᵥ` laid out (environment legs..., site legs..., bond)
# so the QR's row block needs no permute. `array` may return a view of `ψᵥ`'s storage, read only.
function dense_update_setup(ψᵥ::ITensor, side_envs, legs, sinds, lb, sqrt_cutoff)
    roots = pseudo_sqrt_inv_sqrt.(side_envs; cutoff = sqrt_cutoff)
    mats(es) = [ITensors.array(es[j], legs[j], prime(legs[j])) for j in eachindex(legs)]
    return (;
        tensor = ITensors.array(ψᵥ, legs..., sinds..., lb),
        matrices = mats(first.(roots)),
        inv_matrices = mats(dag.(last.(roots))),
    )
end

# As `setbondmessages!` reads them: diagonal over the new bond and a dummy it contracts away.
bond_values(svals, newbond::Index) = ITensors.diag_itensor(svals, newbond, sim(newbond))

# Keeps the factor exchange clear of `communicate_messages!`, which uses the default tag 0.
const _BOUNDARY_GATE_TAG = 1

# The array moves as a buffer, which CUDA-aware MPI can do device to device; the index vector goes
# separately: it carries the new bond `Index` itself, so both ranks name the cut bond by the same
# `Index` id and `commoninds(message(ψ_bpc, e_in), ψᵥ)` still finds it on the next gate.
function send_factor(a::AbstractArray, is, comm; dest)
    MPI.send(collect(is), comm; dest, tag = _BOUNDARY_GATE_TAG)
    MPI.Send(a, comm; dest, tag = _BOUNDARY_GATE_TAG)
    return a
end

function recv_factor(like::AbstractArray, comm; source)
    is = MPI.recv(comm; source, tag = _BOUNDARY_GATE_TAG)
    a = similar(like, dim.(Tuple(is)))
    MPI.Recv!(a, comm; source, tag = _BOUNDARY_GATE_TAG)
    return a, is
end

# Two-site update across a rank boundary; `compute` picks the rank running the gate and the SVD.
# Both ranks must reach this for the same gates in order, or the sends deadlock.
function simple_update_dense_boundary(
        o::ITensor, ψᵥ::ITensor;
        envs, lb::Index, compute::Bool, other_rank::Integer, comm::MPI.Comm,
        normalize_tensors = true, sqrt_cutoff = nothing, apply_kwargs...
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
    (; tensor, matrices, inv_matrices) =
        dense_update_setup(ψᵥ, envs, legs, sinds, lb, sqrt_cutoff)

    Q, R = absorb_matrices_qr(tensor, matrices)

    if compute
        Rother, isother = recv_factor(R, comm; source = other_rank)
        # The partner sent (row block, site axes..., bond), so its site legs are the middle indices.
        sother = collect(isother)[2:(end - 1)]

        gate = gate_array(o, Index[sinds...; sother...])
        isnothing(gate) && error(
            "simple_update_dense_boundary: $(o) is not a plain dense operator on the site indices " *
                "$(Index[sinds...; sother...]) and their primes.",
        )

        L, Rr, svals, err = gate_split(gate, R, Rother; apply_kwargs...)
        newbond = Index(length(svals), "Link,l")

        # Returned in the layout the partner sent, so its own `sinds` still name its site axes.
        send_factor(Rr, Index[first(isother); sother...; newbond], comm; dest = other_rank)
        MPI.send((svals, err), comm; dest = other_rank, tag = _BOUNDARY_GATE_TAG)
        Rp = L
    else
        send_factor(R, Index[Index(size(R, 1), "Link,qr"); sinds...; lb], comm; dest = other_rank)
        Rp, isp = recv_factor(R, comm; source = other_rank)
        svals, err = MPI.recv(comm; source = other_rank, tag = _BOUNDARY_GATE_TAG)
        newbond = last(isp)
    end

    u = absorb_matrices_mul(Q, inv_matrices, Rp; transposed = true)

    if normalize_tensors
        rmul!(u, inv(norm(u)))
        svals = normalize(svals)
    end

    # `itensor`, not `ITensor`: the capitalised constructor copies an array that is already private.
    return ITensors.itensor(u, legs..., sinds..., newbond), bond_values(svals, newbond), err
end

# Signature-compatible with `simple_update`, falling back to it on any shape the dense path cannot
# take: quantum numbers, a vertex count other than two, a vertex with no environment legs, a gate
# that is not a plain dense operator, or an apply kwarg whose truncation this path does not implement.
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
    tensors = ntuple(i -> setups[i].tensor, 2)
    matrices = ntuple(i -> setups[i].matrices, 2)
    inv_matrices = ntuple(i -> setups[i].inv_matrices, 2)

    middle!(R1, R2) = gate_split(gate, R1, R2; apply_kwargs...)

    (u1, u2), svals, err = simple_update_dense!(
        tensors, matrices, inv_matrices, middle!; normalize_tensors
    )

    newbond = Index(length(svals), "Link,l")
    # `itensor`, not `ITensor`: the capitalised constructor copies an array that is already private.
    updated_tensors = ITensor[
        ITensors.itensor(u1, legs[1]..., sinds[1]..., newbond),
        ITensors.itensor(u2, legs[2]..., sinds[2]..., newbond),
    ]

    return updated_tensors, bond_values(svals, newbond), err
end
