using ITensors: array, hasqns, permute
using ITensors.NDTensors: NDTensors
using LinearAlgebra: lmul!, qr!

#
# A memory-bounded two-site gate application, selected with `blocked_gates!(true)`.
#
# Mathematically identical to `simple_update`'s two-site branch; the difference is which
# factor-sized arrays exist. For a degree-3 vertex every one of the site tensor, its
# environment-gauged copy, the QR's Q, and the result is S·χ³ elements -- the same size -- so peak
# memory is decided purely by how many coexist.
#
# `simple_update` goes through ITensors' `qr`, which permutes into a matrix, runs LAPACK,
# materialises Q, and permutes Q back out. That is roughly four factor-sized arrays live inside
# one call, and it is the dominant unbounded term (measured at 7.2x the factor in allocation
# churn per site).
#
# This path instead:
#   * permutes into a matrix once, deliberately, and hands that buffer to `qr!` -- which
#     overwrites it in place with the Householder reflectors, so no second copy appears;
#   * never forms Q. `lmul!(F.Q, C)` applies the reflectors directly to the output buffer, so the
#     only m-by-something array is the result itself;
#   * never permutes back. Storage order is not observable through an ITensor, so the result is
#     labelled to match the layout it already has.
#
# That bounds a two-site update at three factor-sized arrays with no hidden internals.
#
# `qr!` and `lmul!` are generic LinearAlgebra, deliberately: on a `CuArray` they dispatch to
# CUSOLVER's geqrf/ormqr, so nothing here names a vendor library. What cannot be checked without
# real hardware is whether that dispatch exists and reaches the vendor path rather than a generic
# fallback -- see `test_gpu_paths.jl` for what the JLArray tests do and do not prove.
#

const _BLOCKED_GATES = Ref(false)

"""
    blocked_gates() -> Bool
    blocked_gates!(enabled::Bool)

Whether two-site gates are applied through the memory-bounded path in `blocked_gate.jl` (`true`)
or through `simple_update`'s standard branch (`false`, the default).

The two are mathematically equivalent -- the tests assert agreement to machine precision -- and
the blocked path exists only to bound peak memory, which matters when a single site tensor is a
large fraction of a GPU. It falls back to the standard branch for anything it does not specialise
(QN/block-sparse tensors, non-matrix environments), so it is safe to enable globally.

See also [`gpu_direct_mpi!`](@ref).
"""
blocked_gates() = _BLOCKED_GATES[]

function blocked_gates!(enabled::Bool)
    _BLOCKED_GATES[] = enabled
    return enabled
end

# Flatten `t` into an `m x n` matrix whose row space is `rowinds` and column space is `colinds`,
# in exactly that order, as a buffer that may be overwritten. `permute` copies (the one copy this
# path makes) and `array` is a reshaped view onto that copy's storage, so no further allocation
# happens here.
function _matrixize(t::ITensor, rowinds, colinds)
    m = prod(dim, rowinds; init = 1)
    n = prod(dim, colinds; init = 1)
    return reshape(array(permute(t, rowinds..., colinds...)), m, n)
end

# Small, so a contiguous copy of the triangular factor costs nothing and avoids reshaping a view.
function _dense_r(F, prototype)
    R = F.R
    out = similar(prototype, size(R, 1), size(R, 2))
    copyto!(out, R)
    return out
end

# `Q * Rn`, where Q is only ever present as the reflectors inside `F`.
#
# `lmul!` applies the full m-by-m Q, so the target has to have m rows: `Rn` is written into the
# leading `r` rows of an m-by-n buffer and the rest zeroed. Because the trailing rows are zero and
# Q's leading r columns are the thin Q, the product is exactly `thin_Q * Rn` -- and the buffer is
# the result, so the padding costs nothing extra.
function _lmul_q(F, Rn::ITensor, r::Index, m::Integer, rowinds)
    colinds = filter(!isequal(r), collect(inds(Rn)))
    rr, nn = dim(r), prod(dim, colinds; init = 1)
    Rmat = _matrixize(Rn, [r], colinds)

    C = similar(Rmat, m, nn)
    fill!(C, zero(eltype(C)))
    # Broadcast rather than `copyto!`: the destination is a non-contiguous view, and `copyto!`
    # between a strided view and an array has no specialised method on a device array, so it would
    # fall back to an elementwise loop and hit disallowed scalar indexing.
    view(C, 1:rr, :) .= Rmat

    lmul!(F.Q, C)
    return itensor(
        reshape(C, dim.(rowinds)..., dim.(colinds)...), rowinds..., colinds...
    )
end

# Returns `nothing` when the inputs are not of the specialised form, so the caller falls back.
function blocked_two_site_update(
        o::ITensor, ψ⃗::Vector{<:ITensor};
        envs, normalize_tensors, sqrt_cutoff, consume_inputs, apply_kwargs...
    )
    length(ψ⃗) == 2 || return nothing
    # The matrix view assumes a flat dense buffer whose length is the product of the dimensions,
    # and `dag` being plain conjugation. Neither holds for block-sparse or QN tensors.
    all(t -> ITensors.storage(t) isa NDTensors.Dense, ψ⃗) || return nothing
    any(hasqns, ψ⃗) && return nothing
    isempty(envs) && return nothing

    all(env -> ndims(env) == 2, envs) || return nothing
    cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(first(envs)))) : sqrt_cutoff
    envs_v1 = filter(env -> hascommoninds(env, ψ⃗[1]), envs)
    envs_v2 = filter(env -> hascommoninds(env, ψ⃗[2]), envs)

    sq1 = pseudo_sqrt_inv_sqrt.(envs_v1; cutoff)
    sq2 = pseudo_sqrt_inv_sqrt.(envs_v2; cutoff)
    sqrt_envs_v1, inv_sqrt_envs_v1 = first.(sq1), last.(sq1)
    sqrt_envs_v2, inv_sqrt_envs_v2 = first.(sq2), last.(sq2)

    sᵥ₁ = commoninds(ψ⃗[1], o)
    sᵥ₂ = commoninds(ψ⃗[2], o)

    # A vertex whose only bond is the gate bond has an empty Q-side, so there is nothing to
    # factorise. Checked here, on the raw tensors, rather than after the environments are absorbed:
    # absorbing relabels these indices but cannot change how many there are, and bailing out below
    # -- after `consume_inputs` has emptied `ψ⃗` -- would hand the caller's fallback empty tensors.
    (
        isempty(uniqueinds(uniqueinds(ψ⃗[1], ψ⃗[2]), sᵥ₁)) ||
            isempty(uniqueinds(uniqueinds(ψ⃗[2], ψ⃗[1]), sᵥ₂))
    ) && return nothing

    # Same release discipline as `simple_update`: each factor-sized array is dropped as soon as
    # its successor exists.
    ψᵥ₁ = contract([ψ⃗[1]; sqrt_envs_v1])
    consume_inputs && (ψ⃗[1] = ITensor())
    ψᵥ₂ = contract([ψ⃗[2]; sqrt_envs_v2])
    consume_inputs && (ψ⃗[2] = ITensor())

    qinds₁ = collect(uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
    qinds₂ = collect(uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
    rinds₁ = filter(i -> i ∉ qinds₁, collect(inds(ψᵥ₁)))
    rinds₂ = filter(i -> i ∉ qinds₂, collect(inds(ψᵥ₂)))

    m₁ = prod(dim, qinds₁; init = 1)
    m₂ = prod(dim, qinds₂; init = 1)
    M₁ = _matrixize(ψᵥ₁, qinds₁, rinds₁)
    ψᵥ₁ = ITensor()
    M₂ = _matrixize(ψᵥ₂, qinds₂, rinds₂)
    ψᵥ₂ = ITensor()

    # In place: M now holds the reflectors, and F keeps it alive. No Q anywhere.
    F₁ = qr!(M₁)
    F₂ = qr!(M₂)

    rᵥ₁ = Index(size(F₁.R, 1))
    rᵥ₂ = Index(size(F₂.R, 1))
    Rᵥ₁ = itensor(
        reshape(_dense_r(F₁, M₁), dim(rᵥ₁), dim.(rinds₁)...), rᵥ₁, rinds₁...
    )
    Rᵥ₂ = itensor(
        reshape(_dense_r(F₂, M₂), dim(rᵥ₂), dim.(rinds₂)...), rᵥ₂, rinds₂...
    )

    # The coupled problem is small (the QR ranks are S·χ, not χ²), so it keeps going through
    # `factorize_svd` -- matching the standard path's truncation semantics exactly rather than
    # reimplementing maxdim/cutoff/mindim handling.
    oR = ITensors.apply(o, Rᵥ₁ * Rᵥ₂)
    singular_values! = Ref(ITensor())
    Rᵥ₁, Rᵥ₂, spec = factorize_svd(
        oR, unioninds(rᵥ₁, sᵥ₁); ortho = "none", singular_values!, apply_kwargs...
    )
    s_values = singular_values![]

    out₁ = _lmul_q(F₁, Rᵥ₁, rᵥ₁, m₁, qinds₁)
    F₁ = M₁ = nothing            # reflectors dead; release before the second site's buffer
    for env in inv_sqrt_envs_v1
        out₁ = out₁ * dag(env)
    end

    out₂ = _lmul_q(F₂, Rᵥ₂, rᵥ₂, m₂, qinds₂)
    F₂ = M₂ = nothing
    for env in inv_sqrt_envs_v2
        out₂ = out₂ * dag(env)
    end

    updated_tensors = ITensor[out₁, out₂]
    if normalize_tensors
        s_values = normalize(s_values)
        for ψᵥ in updated_tensors
            rmul!(ITensors.data(ψᵥ), inv(norm(ψᵥ)))
        end
    end
    return noprime.(updated_tensors), s_values, spec.truncerr
end
