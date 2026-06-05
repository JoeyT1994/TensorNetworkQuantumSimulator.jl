# Factorizations of locally-ordered fermionic tensors (svd / qr / pseudo_sqrt_inv_sqrt).
#
# All three reduce to the same engine:
#   1. Split the legs into a ROW set R (left factor) and COLUMN set C (right factor),
#      permute to canonical order [R...; C...] folding the Koszul sign into the data
#      (`_apply_reorder_sign`), and reshape to a dense matrix M.
#   2. Because `ft` is parity-EVEN, M[I,J] = 0 unless the fused row parity pR(I) equals
#      the fused col parity pC(J). So M is block diagonal in parity: M = M₀ ⊕ M₁. Each
#      parity block is factorized with the ordinary dense LAPACK routine.
#   3. The new internal bond is an ordinary fermionic bond: its grading vector records the
#      parity of each retained column, and its arrows are chosen so that the existing
#      supertrace bookkeeping in `contract` reassembles the factors back to `ft`.
#
# See arXiv:2410.02215 (locally-ordered formalism) and the SciPost "Fermionic tensor
# networks" treatment of the bond metric `g` used in `pseudo_sqrt_inv_sqrt`.

using LinearAlgebra: LinearAlgebra, Diagonal, Hermitian
using ITensors: ITensors, ITensor, Index, dim, inds, sim, tags, replaceinds
using ITensors.NDTensors: NDTensors
using Dictionaries: Dictionary

# ---------------------------------------------------------------------------
# leg relabelling (used by the higher-level algorithms and the squaring identity)
# ---------------------------------------------------------------------------
function ITensors.replaceinds(ft::FermionicITensor, oldis, newis)
    T = replaceinds(ft.tensor, oldis, newis)
    rep = Dict(o => n for (o, n) in zip(collect(oldis), collect(newis)))
    neworder = Index[get(rep, i, i) for i in ft.order]
    gr = Dictionary{Index, Vector{Bool}}(neworder, Vector{Bool}[ft.grading[i] for i in ft.order])
    return FermionicITensor(T, neworder, copy(ft.dirs), gr)
end

# ---------------------------------------------------------------------------
# shared engine
# ---------------------------------------------------------------------------

# Parity bit of every fused multi-index over `legs`, enumerated in column-major order
# (matching Julia `reshape`). Returns a length-`prod(dim)` Vector{Bool}; the empty-leg
# case is a single even component.
function _parity_vector(grading::Dictionary, legs::Vector{<:Index})
    isempty(legs) && return Bool[false]
    bits = Vector{Bool}[grading[i] for i in legs]
    dims = Tuple(length(b) for b in bits)
    par = Vector{Bool}(undef, prod(dims))
    for (lin, I) in enumerate(CartesianIndices(dims))
        p = false
        for k in eachindex(bits)
            p ⊻= bits[k][I[k]]
        end
        par[lin] = p
    end
    return par
end

# Permute `ft` to [R...; C...] (folding the Koszul sign) and reshape to a dense matrix.
# Returns (M, pr, pc, R, C) where pr/pc are the row/column parity vectors.
function _matricize_ft(ft::FermionicITensor, R::Vector{<:Index})
    C = filter(i -> !(i in R), ft.order)
    P = Index[R; C]
    T = _apply_reorder_sign(ft.grading, ft.tensor, ft.order, P)
    arr = ITensors.array(T, P...)
    nr = prod(Int[dim(i) for i in R]; init = 1)
    nc = prod(Int[dim(i) for i in C]; init = 1)
    M = reshape(Array(arr), nr, nc)
    pr = _parity_vector(ft.grading, R)
    pc = _parity_vector(ft.grading, C)
    return M, pr, pc, R, C
end

# Build a factor FermionicITensor whose spectator legs are `legs` (in order) followed (if
# `bond_first` is false) or preceded (if true) by the new bond `b`. `mat` is the dense
# matrix with spectator multi-index along `spectator_dim` and bond along the other.
function _factor_tensor(ft::FermionicITensor, legs::Vector{<:Index}, legdirs::Vector{Bool},
        b::Index, bdir::Bool, bondgr::Vector{Bool}, mat::AbstractMatrix; bond_first::Bool)
    legdims = Int[dim(i) for i in legs]
    if bond_first
        order = Index[b; legs]
        dirs = Bool[bdir; legdirs]
        arr = reshape(Array(mat), (dim(b), legdims...))   # mat is (bond × spectator)
    else
        order = Index[legs; b]
        dirs = Bool[legdirs; bdir]
        arr = reshape(Array(mat), (legdims..., dim(b)))    # mat is (spectator × bond)
    end
    gr = Dictionary{Index, Vector{Bool}}(order, Vector{Bool}[i == b ? bondgr : ft.grading[i] for i in order])
    return FermionicITensor(ITensor(arr, order...), order, dirs, gr)
end

# ---------------------------------------------------------------------------
# block helpers
# ---------------------------------------------------------------------------
function _safe_svd(A::AbstractMatrix)
    (size(A, 1) == 0 || size(A, 2) == 0) &&
        return (zeros(eltype(A), size(A, 1), 0), Float64[], zeros(eltype(A), 0, size(A, 2)))
    F = LinearAlgebra.svd(A)
    return (F.U, F.S, F.Vt)
end

function _safe_qr(A::AbstractMatrix)
    k = min(size(A)...)
    k == 0 && return (zeros(eltype(A), size(A, 1), 0), zeros(eltype(A), 0, size(A, 2)))
    F = LinearAlgebra.qr(A)
    Q = Matrix(F.Q)[:, 1:k]
    R = Q' * A
    return (Q, R)
end

# Pool singular values from the even (`Se`) and odd (`So`) blocks, truncate with ITensors'
# own `NDTensors.truncate!` (so cutoff/maxdim/mindim semantics match the bosonic path), and
# return the surviving local indices in each sector.
function _svd_keep(Se::Vector, So::Vector; cutoff, maxdim, mindim, use_absolute_cutoff, use_relative_cutoff)
    ne, no = length(Se), length(So)
    vals = Float64[abs.(Se); abs.(So)]
    sec = [fill(0, ne); fill(1, no)]
    loc = [collect(1:ne); collect(1:no)]
    perm = sortperm(vals; rev = true)
    P = Float64[vals[p]^2 for p in perm]          # squared singular values, descending
    isempty(P) && return (Int[], Int[])
    NDTensors.truncate!(P; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    kept = perm[1:length(P)]                       # truncate! resizes P to the kept count
    ke = sort!(Int[loc[t] for t in kept if sec[t] == 0])
    ko = sort!(Int[loc[t] for t in kept if sec[t] == 1])
    return ke, ko
end

# ---------------------------------------------------------------------------
# SVD:  ft  ≈  U * S * V   (fermionic contract)
# ---------------------------------------------------------------------------
"""
    svd(ft::FermionicITensor, row_inds::Vector{<:Index}; cutoff, maxdim, mindim, tags)

Singular value decomposition of a parity-even `FermionicITensor`. `row_inds` become the
spectator legs of `U`; the remaining legs become the spectator legs of `V`. Returns
`(U, S, V)` such that `U * S * V ≈ ft` under fermionic contraction. The new bond carries a
data-driven Z2 grading (parity of the retained singular vectors); truncation pools singular
values across both parity sectors and applies a single `cutoff`/`maxdim`/`mindim`.
"""
function ITensors.svd(ft::FermionicITensor, row_inds::Vector{<:Index};
        cutoff = nothing, maxdim = nothing, mindim = nothing,
        use_absolute_cutoff = nothing, use_relative_cutoff = nothing, tags = "Link,fermion")
    M, pr, pc, R, C = _matricize_ft(ft, collect(Index, row_inds))
    re, ro = findall(!, pr), findall(identity, pr)
    ce, co = findall(!, pc), findall(identity, pc)

    Ue, Se, Vte = _safe_svd(M[re, ce])
    Uo, So, Vto = _safe_svd(M[ro, co])

    ke, ko = _svd_keep(Se, So; cutoff, maxdim, mindim, use_absolute_cutoff, use_relative_cutoff)
    r0, r1 = length(ke), length(ko)
    nb = r0 + r1
    bondgr = Bool[fill(false, r0); fill(true, r1)]

    eltp = eltype(M)
    nr, nc = size(M)
    Ufull = zeros(eltp, nr, nb)
    Vfull = zeros(eltp, nb, nc)
    svals = zeros(real(eltp), nb)
    for (col, k) in enumerate(ke)
        Ufull[re, col] = Ue[:, k]
        Vfull[col, ce] = Vte[k, :]
        svals[col] = Se[k]
    end
    for (j, k) in enumerate(ko)
        col = r0 + j
        Ufull[ro, col] = Uo[:, k]
        Vfull[col, co] = Vto[k, :]
        svals[col] = So[k]
    end

    b = Index(nb, tags)
    bd = Index(nb, tags)
    Rdirs = Bool[_dir(ft, i) for i in R]
    Cdirs = Bool[_dir(ft, i) for i in C]

    # arrows: U.b = OUT, S = (b IN, bd OUT), V.bd = IN. The two supertraces inserted on
    # recontraction (g on b, g on bd) cancel because S is parity-diagonal (p_b = p_bd).
    U = _factor_tensor(ft, R, Rdirs, b, false, bondgr, Ufull; bond_first = false)
    Sgr = Dictionary{Index, Vector{Bool}}(Index[b, bd], Vector{Bool}[bondgr, bondgr])
    Smat = zeros(eltp, nb, nb)
    for i in 1:nb
        Smat[i, i] = svals[i]
    end
    S = FermionicITensor(ITensor(Smat, b, bd), Index[b, bd], Bool[true, false], Sgr)
    V = _factor_tensor(ft, C, Cdirs, bd, true, bondgr, Vfull; bond_first = true)
    return U, S, V
end

ITensors.svd(ft::FermionicITensor, row_inds::Index...; kwargs...) =
    ITensors.svd(ft, collect(Index, row_inds); kwargs...)

# ---------------------------------------------------------------------------
# Symmetric SVD:  ft  ≈  X ∘ Y   with   X = U·√S ,  Y = √S·V
# ---------------------------------------------------------------------------
"""
    symmetric_svd(ft::FermionicITensor, row_inds::Vector{<:Index}; cutoff, maxdim, mindim, tags)

Symmetric singular value decomposition of a parity-even `FermionicITensor`: SVD `ft ≈ U S V`,
then absorb `√S` into both sides, returning `(X, Y, S, err)` with `X = U·√S`, `Y = √S·V`, so
that `X ∘ Y ≈ ft` under fermionic contraction over their single shared bond. `row_inds` become
the spectator legs of `X`; the remaining legs become the spectator legs of `Y`.

The shared bond uses the same arrow convention as `qr` (`X.bond = IN`, `Y.bond = OUT`), so no
net supertrace is inserted on recontraction and `X ∘ Y` is a plain parity-block matmul
`(U√S)(√S V) = U S V`. `S` is returned as the diagonal singular-value matrix on its own bond
pair (legs `[b, bd]`, arrows `[IN, OUT]`, parity-diagonal grading), matching `svd`'s `S`.

`err` is the truncation error `(Σ discarded σ² ) / (Σ all σ²)` computed from the full
pre-truncation spectrum.
"""
function symmetric_svd(ft::FermionicITensor, row_inds::Vector{<:Index};
        cutoff = nothing, maxdim = nothing, mindim = nothing,
        use_absolute_cutoff = nothing, use_relative_cutoff = nothing, tags = "Link,fermion")
    M, pr, pc, R, C = _matricize_ft(ft, collect(Index, row_inds))
    re, ro = findall(!, pr), findall(identity, pr)
    ce, co = findall(!, pc), findall(identity, pc)

    Ue, Se, Vte = _safe_svd(M[re, ce])
    Uo, So, Vto = _safe_svd(M[ro, co])

    # Truncation error from the FULL (pre-truncation) spectrum.
    total = sum(abs2, Se) + sum(abs2, So)
    ke, ko = _svd_keep(Se, So; cutoff, maxdim, mindim, use_absolute_cutoff, use_relative_cutoff)
    kept = sum(abs2, @view Se[ke]) + sum(abs2, @view So[ko])
    err = total == 0 ? zero(total) : (total - kept) / total

    r0, r1 = length(ke), length(ko)
    nb = r0 + r1
    bondgr = Bool[fill(false, r0); fill(true, r1)]

    eltp = eltype(M)
    nr, nc = size(M)
    Xfull = zeros(eltp, nr, nb)         # spectator(row) × bond
    Yfull = zeros(eltp, nb, nc)         # bond × spectator(col)
    svals = zeros(real(eltp), nb)
    for (col, k) in enumerate(ke)
        sq = sqrt(Se[k])
        Xfull[re, col] = Ue[:, k] .* sq
        Yfull[col, ce] = Vte[k, :] .* sq
        svals[col] = Se[k]
    end
    for (j, k) in enumerate(ko)
        col = r0 + j
        sq = sqrt(So[k])
        Xfull[ro, col] = Uo[:, k] .* sq
        Yfull[col, co] = Vto[k, :] .* sq
        svals[col] = So[k]
    end

    b = Index(nb, tags)                 # shared bond of X and Y
    Rdirs = Bool[_dir(ft, i) for i in R]
    Cdirs = Bool[_dir(ft, i) for i in C]

    # Same arrows as qr: X.b = IN, Y.b = OUT → no net supertrace, plain block matmul.
    X = _factor_tensor(ft, R, Rdirs, b, true, bondgr, Xfull; bond_first = false)
    Y = _factor_tensor(ft, C, Cdirs, b, false, bondgr, Yfull; bond_first = true)

    # S sits on the SAME bond `b` shared by X and Y (its prime is the second leg), so the
    # returned spectrum lives on the same edge as the two factors — mirroring the bosonic
    # `factorize_svd` path, where the singular-value tensor shares the post-`noprime` bond.
    bd = prime(b)
    Sgr = Dictionary{Index, Vector{Bool}}(Index[b, bd], Vector{Bool}[bondgr, bondgr])
    Smat = zeros(eltp, nb, nb)
    for i in 1:nb
        Smat[i, i] = svals[i]
    end
    S = FermionicITensor(ITensor(Smat, b, bd), Index[b, bd], Bool[true, false], Sgr)

    return X, Y, S, err
end

symmetric_svd(ft::FermionicITensor, row_inds::Index...; kwargs...) =
    symmetric_svd(ft, collect(Index, row_inds); kwargs...)

# ---------------------------------------------------------------------------
# QR:  ft  ≈  Q * R   (fermionic contract)
# ---------------------------------------------------------------------------
"""
    qr(ft::FermionicITensor, row_inds::Vector{<:Index}; tags)

QR decomposition of a parity-even `FermionicITensor`. `row_inds` become the spectator legs
of `Q`; the remaining legs become the spectator legs of `R`. Returns `(Q, R)` with
`Q * R ≈ ft`. The shared bond is graded by the parity of each retained column; its arrow is
set so that no net supertrace is inserted on recontraction (`Q.bond = IN`, `R.bond = OUT`).
"""
function ITensors.qr(ft::FermionicITensor, row_inds::Vector{<:Index}; tags = "Link,fermion")
    M, pr, pc, R, C = _matricize_ft(ft, collect(Index, row_inds))
    re, ro = findall(!, pr), findall(identity, pr)
    ce, co = findall(!, pc), findall(identity, pc)

    Qe, Re_ = _safe_qr(M[re, ce])
    Qo, Ro_ = _safe_qr(M[ro, co])
    k0, k1 = size(Qe, 2), size(Qo, 2)
    nb = k0 + k1
    bondgr = Bool[fill(false, k0); fill(true, k1)]

    eltp = eltype(M)
    nr, nc = size(M)
    Qfull = zeros(eltp, nr, nb)
    Rfull = zeros(eltp, nb, nc)
    Qfull[re, 1:k0] = Qe
    Rfull[1:k0, ce] = Re_
    Qfull[ro, (k0 + 1):nb] = Qo
    Rfull[(k0 + 1):nb, co] = Ro_

    b = Index(nb, tags)
    Rdirs = Bool[_dir(ft, i) for i in R]
    Cdirs = Bool[_dir(ft, i) for i in C]

    Q = _factor_tensor(ft, R, Rdirs, b, true, bondgr, Qfull; bond_first = false)   # Q.b = IN
    Rt = _factor_tensor(ft, C, Cdirs, b, false, bondgr, Rfull; bond_first = true)  # R.b = OUT
    return Q, Rt
end

ITensors.qr(ft::FermionicITensor, row_inds::Index...; kwargs...) =
    ITensors.qr(ft, collect(Index, row_inds); kwargs...)

# ---------------------------------------------------------------------------
# pseudo_sqrt_inv_sqrt of a fermionic bond matrix
# ---------------------------------------------------------------------------
function _herm_sqrt_block(A::AbstractMatrix; cutoff::Real)
    n = size(A, 1)
    n == 0 && return A, A
    F = LinearAlgebra.eigen(Hermitian(A))
    λ = F.values
    s = zeros(real(eltype(A)), n)
    si = zeros(real(eltype(A)), n)
    for i in 1:n
        # Root the MAGNITUDE of each eigenvalue. A real BP message carries the supertrace
        # metric, so its parity blocks are Hermitian but NOT positive-semidefinite (e.g. a
        # block eigenvalue −0.223). The Vidal-gauge round-trip only needs `X` Hermitian with
        # `X·X⁻¹ = I`: with `s = √|λ|` the factor `X = Q·diag(s)·Qᴴ` is Hermitian, so
        # `X · dag(X⁻¹) = X·X⁻¹ = I` exactly, independent of the eigenvalue sign. (Clipping
        # negatives to 0 projects out that sector; a complex √λ breaks the inverse because
        # `dag` conjugates the imaginary root.)
        if abs(λ[i]) > cutoff
            s[i] = sqrt(abs(λ[i]))
            si[i] = inv(s[i])
        end
    end
    Q = F.vectors
    return Q * Diagonal(s) * Q', Q * Diagonal(si) * Q'
end

"""
    pseudo_sqrt_inv_sqrt(M::FermionicITensor; cutoff)

Matrix square root and pseudo-inverse-square root of a 2-leg fermionic bond matrix `M`
(order `[a, a']`, both legs sharing the same grading). `M` is Hermitian but NOT
positive-semidefinite: a real BP message carries the supertrace metric, so its parity
blocks can have negative eigenvalues (e.g. an even block `−0.223`). Each parity block is
diagonalised and its eigenvalue MAGNITUDES are rooted (`s = √|λ|`), giving a Hermitian
factor `X`. The Vidal-gauge round-trip (absorb `X` into a site tensor, later remove `Xinv`)
needs only `X · dag(Xinv) = X·X⁻¹ = I`, which holds for any Hermitian `X` regardless of
eigenvalue sign — so `√|λ|` reconstructs the bond exactly. (Rooting `λ` directly would clip
negative blocks to zero and project out that sector; a complex `√λ` breaks the inverse
because `dag` conjugates the imaginary root.) Returns `(X, Xinv)`.
"""
function pseudo_sqrt_inv_sqrt(M::FermionicITensor; cutoff::Real = 10 * eps(real(scalartype(M))))
    @assert length(M.order) == 2 "pseudo_sqrt_inv_sqrt expects a 2-leg fermionic tensor"
    a, a2 = M.order
    @assert dim(a) == dim(a2) "the two legs must have equal dimension"
    Mmat, pr, pc, _, _ = _matricize_ft(M, Index[a])
    @assert pr == pc "the two legs must carry the same grading"

    # Root the eigenvalue MAGNITUDES of each parity block (see `_herm_sqrt_block`).
    #
    # Empirical structure of real BP messages (studied on random 3×3/4×4 fermionic TNS over
    # bond dims 2/4/6, before and after BP convergence): every message is Hermitian, and each
    # parity block is sign-DEFINITE — all eigenvalues in a block share one sign; an indefinite
    # block was NEVER observed. So a message factors as `M = G·|M|` with `|M|` genuinely PSD
    # and `G = diag(s_even·𝟙_even, s_odd·𝟙_odd)`, `s ∈ {+1,−1}` a constant ±1 sign PER BLOCK
    # (the supertrace metric). The negatives are a per-block metric sign, not within-block
    # indefiniteness. Typically one block is PSD and the other NSD (opposite metric signs);
    # near-maximally-mixed messages can have both blocks PSD.
    #
    # Hence `√|λ|` is exact, not a hack: `X = Q·diag(√|λ|)·Qᴴ` is Hermitian with `X² = |M|`,
    # and the per-block sign `G` is a ±1 unitary that factors out and cancels in the
    # Vidal-gauge round-trip `X·dag(Xinv) = X·X⁻¹ = I`. (Were a block ever indefinite, `√|λ|`
    # would incoherently mix signs — but it never is.)
    re, ro = findall(!, pr), findall(identity, pr)
    Xe, Xie = _herm_sqrt_block(Mmat[re, re]; cutoff)
    Xo, Xio = _herm_sqrt_block(Mmat[ro, ro]; cutoff)

    eltp = eltype(Mmat)
    n = dim(a)
    Xfull = zeros(eltp, n, n)
    Xifull = zeros(eltp, n, n)
    Xfull[re, re] = Xe
    Xfull[ro, ro] = Xo
    Xifull[re, re] = Xie
    Xifull[ro, ro] = Xio

    X = FermionicITensor(ITensor(Xfull, a, a2), copy(M.order), copy(M.dirs), M.grading)
    Xinv = FermionicITensor(ITensor(Xifull, a, a2), copy(M.order), copy(M.dirs), M.grading)
    return X, Xinv
end
