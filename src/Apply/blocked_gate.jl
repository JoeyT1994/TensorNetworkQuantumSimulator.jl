using ITensors: array, hasqns, permute
using ITensors.NDTensors: NDTensors
using LinearAlgebra: LinearAlgebra, lmul!, mul!, qr!

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
#   * absorbs the environments with plain gemms folded into that same permute, rather than through
#     `contract` -- see the header above `GaugeLeg` for why that is where the peak actually was;
#   * never forms Q. `lmul!(F.Q, C)` applies the reflectors directly to the output buffer, so the
#     only m-by-something array is the result itself;
#   * never permutes back. Storage order is not observable through an ITensor, so the result is
#     labelled to match the layout it already has.
#
# Two factor-sized arrays are live at a time in each phase, and no phase has hidden internals.
# Measured against the standard branch on a degree-3 site at χ=256: 5.9 factors of allocation
# versus 14.9, and a peak resident footprint of 3.8 factors versus 4.5.
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
(QN/block-sparse tensors, non-matrix environments, and a state whose scalar type the gate would
promote), so it is safe to enable globally.

Note the last of those: the state constructors default to `Float64` while every standard rotation
is complex, so on a `zerostate` the specialised path never runs and this flag does nothing. Build
the state with a complex scalar type to get the benefit.

See also [`gpu_direct_mpi!`](@ref).
"""
blocked_gates() = _BLOCKED_GATES[]

function blocked_gates!(enabled::Bool)
    _BLOCKED_GATES[] = enabled
    return enabled
end

# cuSOLVER's dense `unmqr`/`ungqr` are indexed with 32-bit ints, so applying Q to a large enough
# matrix fails with CUSOLVER_STATUS_INVALID_VALUE out of the buffer-size query. `geqrf` has a
# 64-bit variant and succeeds, so without care the failure surfaces one call later, at `lmul!`,
# with nothing in the trace pointing at the size. For a degree-3 vertex the matrix is χ²×S·χ, so
# the quantity at risk is `lda*n = S·χ³`.
#
# So the QR is done in row blocks (TSQR) whenever the matrix is too big for one call: `p` blocks
# divide that product by `p`, which is exactly the right lever.
#
# The limit is **not** 2^31. Setting it there says "anything that fits in an int is fine", which
# assumes cuSOLVER's largest internal index is exactly `lda*n` and never `lda*n + something`.
# Measured against that assumption: χ=750 (S·χ³ = 1.69e9, 21% below 2^31) runs, χ=800
# (2.05e9, 4.6% below) fails -- and at 2^31 neither would have been split at all. So the usable
# ceiling sits below the representable one, and the default keeps a factor of two in hand.
#
# Raising it back is a one-liner (`qr_block_limit!`) if a future CUDA stack proves more generous;
# lowering it costs only a slightly smaller block buffer, since that buffer is F/p.
default_qr_block_limit() = 1 << 30
const _QR_BLOCK_LIMIT = Ref(default_qr_block_limit())

"""
    qr_block_limit() -> Int
    qr_block_limit!(n::Integer)

Largest number of entries handed to a single QR call by the memory-bounded gate path. Above this
the QR is split into row blocks (see `_tall_skinny_qr!`), which divides the count by the number of
blocks.

Defaults to `2^30`, half of what cuSOLVER's 32-bit dense API can represent. The margin is
deliberate and was measured: on one CUDA stack a degree-3 vertex at χ=750 (`S·χ³` = 1.69e9, 21%
below `typemax(Int32)`) runs, while χ=800 (2.05e9, 4.6% below) fails — so the usable ceiling sits
below the representable one. Raise it if a stack proves more generous; lower it to exercise the
blocked path on the host, which is what the tests do.
"""
qr_block_limit() = _QR_BLOCK_LIMIT[]
qr_block_limit!(n::Integer) = (_QR_BLOCK_LIMIT[] = Int(n); n)

# Householder QR of a tall-skinny matrix in row blocks. `M = [M₁; …; M_p]`, each `Mᵢ = QᵢRᵢ`, then
# `[R₁; …; R_p] = Q̃R`, so `Q = blockdiag(Q₁…Q_p)·Q̃` and `M = QR` exactly.
#
# Every factor is orthonormal by construction, so this is as stable as the direct Householder QR
# -- and unlike forming `Q = M·R⁻¹` it never inverts anything, which matters because a padded
# product state makes `R` singular.
struct TallSkinnyQR{F, G}
    blocks::Vector{F}
    rows::Vector{UnitRange{Int}}
    top::G            # second-level factorization of the stacked R factors
    ncols::Int
end

# `qr!` overwrites its argument, so each block is copied into its own contiguous buffer: a strided
# view of `M` would keep the parent's `lda`, and it is `lda` that overflows.
function _tall_skinny_qr!(M::AbstractMatrix, nrow_blocks::Integer)
    m, n = size(M)
    bounds = round.(Int, range(0, m; length = nrow_blocks + 1))
    rows = [(bounds[i] + 1):bounds[i + 1] for i in 1:nrow_blocks]
    # Each block is copied into its own contiguous buffer: a strided view of `M` would keep the
    # parent's `lda`, and `lda` is exactly what overflows. Broadcast, not `copyto!` -- the source
    # is a strided view, which `copyto!` would walk elementwise.
    blocks = map(rows) do r
        block = similar(M, length(r), n)
        block .= view(M, r, :)
        qr!(block)
    end
    stacked = similar(M, nrow_blocks * n, n)
    fill!(stacked, zero(eltype(M)))
    for (i, F) in enumerate(blocks)
        Ri = F.R
        view(stacked, ((i - 1) * n + 1):((i - 1) * n + size(Ri, 1)), :) .= Ri
    end
    top = qr!(stacked)
    return TallSkinnyQR(blocks, rows, top, n)
end

# One QR when it fits, TSQR when it does not.
function _qr_tall!(M::AbstractMatrix)
    m, n = size(M)
    # Only worth splitting a genuinely tall matrix, and only into blocks that still have at least
    # as many rows as columns -- a block with fewer would have a rank-deficient Rᵢ and the
    # two-level product would no longer be a QR of `M`. A degree-2 vertex gives a wide matrix and
    # always lands here.
    (m < 2n || m * n <= max(qr_block_limit(), 1)) && return qr!(M)
    nblocks = min(cld(m * n, max(qr_block_limit(), 1)), fld(m, n))
    nblocks <= 1 && return qr!(M)
    return _tall_skinny_qr!(M, nblocks)
end

_qr_rank(F, M) = min(size(M)...)
_qr_r(F, M) = F.R
# Every block has at least `n` rows, so the second-level `R` is the full `n x n` factor.
_qr_rank(F::TallSkinnyQR, M) = size(F.top.R, 1)
_qr_r(F::TallSkinnyQR, M) = F.top.R

#
# Environment gauging as a chain of gemms.
#
# Absorbing a χ×χ environment into one leg of a factor-sized tensor is a single matrix product,
# but `contract`/`*` reaches it through `permutedims`: measured here, one such contraction costs
# ~2.1 factors of transient memory -- a permute scratch plus the output -- to produce one factor
# of result. A two-site gate does four of them, two gauging in and two ungauging out, and it is
# those, not the QR, that set the peak.
#
# The absorbed leg keeps its dimension, so the product can instead be written as a gemm that
# *rotates* that leg from one end of the storage order to the other:
#
#   A[k, rest…] · env[k, j]  =  (reshape(A, k, :)ᵀ · env)   → [rest…, j]    front to back
#   A[rest…, k] · env[k, j]  =  (envᵀ · reshape(A, :, k)ᵀ)  → [j, rest…]    back to front
#
# Chaining one form absorbs `d` legs in `d` gemms with no `permutedims` at all: the rotation is
# what carries the next leg into position. Going back-to-front leaves the absorbed legs at the
# front, in order, which is exactly the row space the QR wants -- so the gauging and the flattening
# into that matrix become a single pass. Two arrays are live at a time regardless of how many legs
# are absorbed.
#
# Measured on the degree-3 site at χ=256: gauging plus flattening went from 5.95 to 2.09 factors of
# allocation, and the whole two-site gate from 9.7 to 5.9.
#

# An environment resolved against the tensor it acts on: `mat` is the χ×χ matrix with the
# contracted leg as its rows, `from` is that leg and `to` is the one left behind.
struct GaugeLeg{M, I}
    mat::M
    from::I
    to::I
end

# `sqrt_envs` and `inv_sqrt_envs` carry the same index pair, so one pass resolves both directions:
# the forward leg absorbs what `t` carries, and the reverse leg absorbs what the forward one left
# behind and restores the original index. `dag` on the reverse side is plain conjugation -- the
# caller has already rejected QN tensors. Returns `nothing` if the environments are not of the
# expected shape, so the caller can fall back.
function _gauge_legs(t::ITensor, sqrt_envs, inv_sqrt_envs)
    length(sqrt_envs) == length(inv_sqrt_envs) || return nothing
    resolved = map(zip(sqrt_envs, inv_sqrt_envs)) do (sq, isq)
        shared = commonind(sq, t)
        isnothing(shared) && return nothing
        rest = filter(!isequal(shared), collect(inds(sq)))
        length(rest) == 1 || return nothing
        other = only(rest)
        (shared ∈ inds(isq) && other ∈ inds(isq)) || return nothing
        return (
            GaugeLeg(array(permute(sq, shared, other)), shared, other),
            GaugeLeg(conj(array(permute(isq, other, shared))), other, shared),
        )
    end
    any(isnothing, resolved) && return nothing
    return first.(resolved), last.(resolved)
end

# Flatten `t` into the `(rowinds, colinds)` matrix the QR consumes, absorbing one environment per
# gauged leg on the way. Returns the matrix and the row order it actually ended up with: the
# gauged legs rotate to the front, so a row leg carrying no environment is pushed behind them.
#
# Whether the permuted copy may alias the network's tensor turns on what happens next. With no
# environments this returns that copy directly and `qr!` overwrites it, so it must be a copy. With
# environments the last gemm's output is what gets overwritten instead and the permuted array is
# only ever read, so aliasing is safe and saves a factor whenever the layout already matches.
function _gauge_matrixize(tref::Base.RefValue, rowinds, colinds, legs)
    plain = filter(i -> !any(l -> l.from == i, legs), collect(rowinds))
    # Gauged legs trail, in absorption order: each gemm takes the last one and rotates it to the
    # front, so after all of them the order is `[to₁, …, to_d, plain…, colinds…]`.
    A = array(
        permute(
            tref[], plain..., colinds..., (l.from for l in legs)...;
            allow_alias = !isempty(legs)
        )
    )
    # The site tensor is dead the moment it has been permuted -- `A` is either a fresh copy of it
    # or aliases its buffer, and nothing below reads it again. It comes in by `Ref` so it can be
    # dropped here instead of on return: held to the end it would sit alongside the permuted copy
    # and the first gemm's output, which is three factor-sized arrays where two suffice. On a
    # degree-3 site that one reference is the difference between a 3F and a 2F peak.
    tref[] = nothing
    for leg in Iterators.reverse(legs)
        M = reshape(A, :, size(leg.mat, 1))
        out = similar(M, size(leg.mat, 2), size(M, 1))
        mul!(out, transpose(leg.mat), transpose(M))
        A = out                  # the input is dead as soon as its successor exists
    end
    newrows = vcat([l.to for l in legs], plain)
    return reshape(A, prod(dim, newrows; init = 1), prod(dim, colinds; init = 1)), newrows
end

# `Q * Rn`, where Q is only ever present as the reflectors inside `F`.
#
# `lmul!` applies the full m-by-m Q, so the target has to have m rows: `Rn` is written into the
# leading `r` rows of an m-by-n buffer and the rest zeroed. Because the trailing rows are zero and
# Q's leading r columns are the thin Q, the product is exactly `thin_Q * Rn` -- and the buffer is
# the result, so the padding costs nothing extra.
function _lmul_q(F, Rn::ITensor, r::Index, rowinds)
    colinds = filter(!isequal(r), collect(inds(Rn)))
    m = prod(dim, rowinds; init = 1)
    # Read-only, so `array` may alias when the layout already matches.
    Rmat = reshape(array(Rn, r, colinds...), dim(r), prod(dim, colinds; init = 1))
    rr, nn = size(Rmat)

    C = similar(Rmat, m, nn)
    fill!(C, zero(eltype(C)))
    # Broadcast rather than `copyto!`: the destination is a non-contiguous view, and `copyto!`
    # between a strided view and an array has no specialised method on a device array, so it would
    # fall back to an elementwise loop and hit disallowed scalar indexing.
    view(C, 1:rr, :) .= Rmat

    _apply_q!(F, C)
    return C, colinds
end

# `Q · Rn`, with the environments ungauged off Q's legs on the way out.
#
# Two things are released here rather than by the caller, and both are worth a full factor:
#
#   * `_lmul_q`'s buffer never escapes. Held by the caller it would stay reachable for the whole
#     ungauging chain, so a second gauged leg would put three factor-sized arrays live at once.
#   * the factorization is dropped the moment `_lmul_q` returns. Its reflectors live in the
#     matrix `_gauge_matrixize` produced -- a full factor -- and nothing needs them once Q has
#     been applied, but a caller-held reference would keep them alongside the ungauging buffers.
#     It comes in by `Ref` for exactly that reason: clearing the caller's binding is the point.
#
# `rowinds` leads the layout `_lmul_q` produced and the gauged legs lead `rowinds` (that is the
# order `_gauge_matrixize` returned), so this is the front-to-back rotation: each gemm takes the
# leading leg and sends it to the back.
function _close_and_ungauge(Fref::Base.RefValue, Rn::ITensor, r::Index, rowinds, legs)
    A, colinds = _lmul_q(Fref[], Rn, r, rowinds)
    Fref[] = nothing
    for leg in legs
        M = reshape(A, size(leg.mat, 1), :)
        out = similar(M, size(M, 2), size(leg.mat, 2))
        mul!(out, transpose(M), leg.mat)
        A = out
    end
    plain = filter(i -> !any(l -> l.from == i, legs), collect(rowinds))
    newinds = vcat(plain, colinds, [l.to for l in legs])
    return itensor(reshape(A, dim.(newinds)...), newinds...)
end

_apply_q!(F, C::AbstractMatrix) = lmul!(F.Q, C)

# `out = blockdiag(Q₁…Q_p) · (Q̃ · C)`. `C` arrives with `R_new` in its leading `n` rows and zeros
# below, the same convention `lmul!` relies on for the single-block case: the trailing zeros are
# what make the full Q act as the thin Q.
function _apply_q!(F::TallSkinnyQR, C::AbstractMatrix)
    n, nn, p = F.ncols, size(C, 2), length(F.blocks)

    # Second level, small: Y = Q̃ · R_new, padded to p·n rows so `lmul!` sees the full Q̃.
    Y = similar(C, p * n, nn)
    fill!(Y, zero(eltype(C)))
    view(Y, 1:n, :) .= view(C, 1:n, :)
    lmul!(F.top.Q, Y)

    # First level: block i of the output is Qᵢ times its slice of Y, by the same padded `lmul!`.
    #
    # The buffer handed to `lmul!` must be a *concrete* device matrix, which is why this is a
    # reshaped contiguous prefix of a flat buffer and not the obvious `view(buf, 1:rows, :)` or
    # `view(C, r, :)`. A row-range view of a matrix is not contiguous, so GPUArrays leaves it a
    # `SubArray`; CUDA declares its Q-multiply as
    #
    #     lmul!(::QRPackedQ{T, <:CuArray, <:CuArray}, ::CuVecOrMat{T})
    #
    # which a `SubArray` does not match, so dispatch falls through to LinearAlgebra's
    # `StridedMatrix` method, which ccalls host LAPACK and throws on a device pointer. A
    # contiguous range does collapse back to the concrete array type, so a flat prefix is safe.
    #
    # JLArray cannot catch this class of bug: it has no GPU `qr!`/`lmul!` override, so the whole
    # factorization runs on host LAPACK and every view works by accident. `test_gpu_paths.jl`
    # therefore asserts the buffer's *type* directly rather than relying on the call succeeding.
    #
    # One block tall and reused across blocks, so this costs F/nblocks -- 0.33 F at χ=1024, S=4.
    buf = similar(C, maximum(length, F.rows) * nn)
    for (i, (block, r)) in enumerate(zip(F.blocks, F.rows))
        b = _block_buffer(buf, length(r), nn)
        fill!(b, zero(eltype(C)))
        view(b, 1:n, :) .= view(Y, ((i - 1) * n + 1):(i * n), :)
        lmul!(block.Q, b)
        view(C, r, :) .= b          # broadcast, so a non-contiguous destination is fine
    end
    return C
end

# A `rows x nn` scratch matrix backed by the front of `buf`. Split out so the tests can assert it
# stays a concrete device array rather than degrading to a `SubArray` -- see `_apply_q!`.
function _block_buffer(buf::AbstractVector, rows::Integer, nn::Integer)
    n = rows * nn
    flat = n == length(buf) ? buf : view(buf, 1:n)
    return reshape(flat, rows, nn)
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

    # Everything here is built at the site tensor's scalar type, but the buffer `_apply_q!`
    # multiplies into is sized from the *post-gate* SVD output. Nothing promotes between them and
    # `lmul!` has no mixed-eltype method, so a real state under a complex gate is a `MethodError`
    # rather than a fallback -- and that is the canonical `zerostate` workflow, since the state
    # constructors default to `Float64` and every standard rotation is complex.
    #
    # Handed back rather than promoted: promoting means an extra full copy of the site tensor,
    # which is the one thing this path exists to avoid, and the reference branch gets the same
    # promotion for free inside its `Q * R`.
    elt = promote_type(scalartype(o), scalartype.(ψ⃗)..., scalartype.(envs)...)
    all(t -> scalartype(t) === elt, ψ⃗) || return nothing

    cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(first(envs)))) : sqrt_cutoff
    envs_v1 = filter(env -> hascommoninds(env, ψ⃗[1]), envs)
    envs_v2 = filter(env -> hascommoninds(env, ψ⃗[2]), envs)

    sq1 = pseudo_sqrt_inv_sqrt.(envs_v1; cutoff)
    sq2 = pseudo_sqrt_inv_sqrt.(envs_v2; cutoff)
    sqrt_envs_v1, inv_sqrt_envs_v1 = first.(sq1), last.(sq1)
    sqrt_envs_v2, inv_sqrt_envs_v2 = first.(sq2), last.(sq2)

    sᵥ₁ = commoninds(ψ⃗[1], o)
    sᵥ₂ = commoninds(ψ⃗[2], o)

    # Every guard is evaluated on the raw tensors, before anything is consumed: bailing out later
    # -- after `consume_inputs` has emptied `ψ⃗` -- would hand the caller's fallback empty tensors.
    qraw₁ = collect(uniqueinds(uniqueinds(ψ⃗[1], ψ⃗[2]), sᵥ₁))
    qraw₂ = collect(uniqueinds(uniqueinds(ψ⃗[2], ψ⃗[1]), sᵥ₂))
    # A vertex whose only bond is the gate bond has an empty Q-side: nothing to factorise.
    (isempty(qraw₁) || isempty(qraw₂)) && return nothing
    rinds₁ = filter(i -> i ∉ qraw₁, collect(inds(ψ⃗[1])))
    rinds₂ = filter(i -> i ∉ qraw₂, collect(inds(ψ⃗[2])))

    legs₁ = _gauge_legs(ψ⃗[1], sqrt_envs_v1, inv_sqrt_envs_v1)
    legs₂ = _gauge_legs(ψ⃗[2], sqrt_envs_v2, inv_sqrt_envs_v2)
    (isnothing(legs₁) || isnothing(legs₂)) && return nothing
    fwd₁, rev₁ = legs₁
    fwd₂, rev₂ = legs₂
    # Each environment has to act on a leg of the row space. One acting on a column leg would be
    # rotated out of the row space by its own gemm, so the matrix reaching the QR would no longer
    # be the gauged tensor. Messages only ever land on bonds leaving `v⃗`, so this holds in
    # practice; it is checked rather than assumed because the fallback is free.
    all(l -> l.from ∈ qraw₁, fwd₁) || return nothing
    all(l -> l.from ∈ qraw₂, fwd₂) || return nothing

    # Gauging and flattening in one pass. The site tensors are handed over by `Ref` and the
    # caller's own reference dropped first, so under `consume_inputs` the old factor is freed
    # inside the call, as soon as it has been permuted, rather than surviving until it returns.
    tref₁, tref₂ = Ref{Any}(ψ⃗[1]), Ref{Any}(ψ⃗[2])
    consume_inputs && (ψ⃗[1] = ψ⃗[2] = ITensor())
    M₁, qinds₁ = _gauge_matrixize(tref₁, qraw₁, rinds₁, fwd₁)
    M₂, qinds₂ = _gauge_matrixize(tref₂, qraw₂, rinds₂, fwd₂)

    # In place: M now holds the reflectors, and F keeps it alive. No Q anywhere.
    # Released as soon as each factorization is built, not at the end: `qr!` keeps the reflectors
    # inside `M`, but TSQR copied them into its own blocks, so `M` is dead the moment it returns.
    # Holding both `M`s alongside both sets of blocks would be four factor-sized arrays at once.
    F₁ = _qr_tall!(M₁)
    F₁ isa TallSkinnyQR && (M₁ = similar(M₁, 0, 0))
    F₂ = _qr_tall!(M₂)
    F₂ isa TallSkinnyQR && (M₂ = similar(M₂, 0, 0))

    # `F.R` already allocates a fresh contiguous upper-triangular matrix, so it is reshaped
    # directly -- and the rank is read off `M` rather than by materialising `R` a second time.
    rᵥ₁ = Index(_qr_rank(F₁, M₁))
    rᵥ₂ = Index(_qr_rank(F₂, M₂))
    Rᵥ₁ = itensor(reshape(_qr_r(F₁, M₁), dim(rᵥ₁), dim.(rinds₁)...), rᵥ₁, rinds₁...)
    Rᵥ₂ = itensor(reshape(_qr_r(F₂, M₂), dim(rᵥ₂), dim.(rinds₂)...), rᵥ₂, rinds₂...)

    # The coupled problem is small (the QR ranks are S·χ, not χ²), so it keeps going through
    # `factorize_svd` -- matching the standard path's truncation semantics exactly rather than
    # reimplementing maxdim/cutoff/mindim handling.
    oR = ITensors.apply(o, Rᵥ₁ * Rᵥ₂)
    singular_values! = Ref(ITensor())
    Rᵥ₁, Rᵥ₂, spec = factorize_svd(
        oR, unioninds(rᵥ₁, sᵥ₁); ortho = "none", singular_values!, apply_kwargs...
    )
    s_values = singular_values![]

    # From here each `M` is only its factorization's backing store, so dropping these bindings
    # costs nothing and leaves `_close_and_ungauge` the sole owner -- which is what lets it free a
    # factor's worth of reflectors before allocating the buffers that replace them.
    M₁ = M₂ = nothing
    Fref₁, Fref₂ = Ref{Any}(F₁), Ref{Any}(F₂)
    F₁ = F₂ = nothing
    out₁ = _close_and_ungauge(Fref₁, Rᵥ₁, rᵥ₁, qinds₁, rev₁)
    out₂ = _close_and_ungauge(Fref₂, Rᵥ₂, rᵥ₂, qinds₂, rev₂)

    updated_tensors = ITensor[out₁, out₂]
    if normalize_tensors
        s_values = normalize(s_values)
        for ψᵥ in updated_tensors
            rmul!(ITensors.data(ψᵥ), inv(norm(ψᵥ)))
        end
    end
    return noprime.(updated_tensors), s_values, spec.truncerr
end
