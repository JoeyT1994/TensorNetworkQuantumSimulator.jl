# Fixed-storage local Gauss--Seidel backend for finite cycle CTMRG.
#
# This deliberately does not reuse `CTMVertexEnvironments`. Those C/T blocks recursively contain
# retained indices created at earlier positions in a sweep, so overwriting one plaquette can leave
# neighbouring blocks in incompatible index epochs. Here the mutable variables are instead the
# local boundary ring used by the variational functional:
#
#     A[k][x,y][a_k,i_k,j_k],   c[k][x,y][j_k,i_{k+1}],   k = 1,...,4.
#
# Every retained axis has the fixed allocation `chi`. A plaquette atomically owns eight A slots and
# four c slots. Consequently a local write is closed, and the next plaquette can immediately consume
# it: this is a genuine Gauss--Seidel update rather than a partially asynchronous rebuild of the
# recursive cache. The code remains internal while its functional and observables are cross-checked
# against the established backend.

const _CTM_LOCAL_OFFSETS = ((1, 1), (0, 1), (0, 0), (1, 0))

"""
    CTMLocalCycleState

Fixed-storage state for local cycle CTMRG. `sites[x,y]` is a rank-four local factor in
counter-clockwise leg order `(right, up, left, down)`. For a PEPS norm this is the fused double-layer
factor, so each leg has dimension `D^2`. Boundary factors are zero-padded to the same leg dimensions
as the bulk; the finite boundary is represented by the one-hot exterior `A,c` tensors.
"""
struct CTMLocalCycleState{T}
    sites::Matrix{Array{T, 4}}
    A::NTuple{4, Matrix{Array{T, 3}}}
    c::NTuple{4, Matrix{Matrix{T}}}
    rank::Matrix{Int}
    chi::Int
end

function CTMLocalCycleState(sites::AbstractMatrix{<:AbstractArray{T, 4}}, chi::Integer) where {T}
    chi > 0 || throw(ArgumentError("chi must be positive"))
    Nx, Ny = size(sites)
    (Nx >= 2 && Ny >= 2) || throw(ArgumentError("local CTM needs at least a 2x2 grid"))
    local_sites = Matrix{Array{T, 4}}(undef, Nx, Ny)
    for x in 1:Nx, y in 1:Ny
        local_sites[x, y] = Array(sites[x, y])
    end
    A = ntuple(4) do k
        field = Matrix{Array{T, 3}}(undef, Nx, Ny)
        for x in 1:Nx, y in 1:Ny
            q = size(local_sites[x, y], k)
            a = zeros(T, q, chi, chi)
            a[1, 1, 1] = one(T)
            field[x, y] = a
        end
        field
    end
    c = ntuple(4) do _
        field = Matrix{Matrix{T}}(undef, Nx, Ny)
        for x in 1:Nx, y in 1:Ny
            corner = zeros(T, chi, chi)
            corner[1, 1] = one(T)
            field[x, y] = corner
        end
        field
    end
    return CTMLocalCycleState(local_sites, A, c, ones(Int, Nx - 1, Ny - 1), Int(chi))
end

# Alternating columns followed by the reverse interior.  Omitting both turn-around endpoints gives
# the standard symmetric Gauss--Seidel schedule with 2N-2 updates: the final plaquette of one sweep
# is adjacent to the first of the next, and no plaquette is accidentally updated twice in a row.
function _ctm_local_snake(Nx::Integer, Ny::Integer)
    nx, ny = Nx - 1, Ny - 1
    (nx >= 1 && ny >= 1) || throw(ArgumentError("local CTM needs at least one plaquette"))
    forward = Tuple{Int, Int}[]
    for x in 1:nx
        ys = isodd(x) ? (1:ny) : (ny:-1:1)
        append!(forward, ((x, y) for y in ys))
    end
    length(forward) == 1 && return forward
    return vcat(forward, reverse(forward[2:(end - 1)]))
end

_ctm_local_snake(S::CTMLocalCycleState) = _ctm_local_snake(size(S.sites)...)

_ctm_local_site(x::Int, y::Int, xy) = (x + xy[1], y + xy[2])

# Only these twelve slots belong to the plaquette. Keeping ownership in one routine prevents a
# later local solver change from accidentally overwriting exterior context.
function _ctm_local_scatter!(S::CTMLocalCycleState, Ainner, cinner, x::Int, y::Int;
                             damping::Real = 1)
    Nx, Ny = size(S.sites)
    (1 <= x < Nx && 1 <= y < Ny) || throw(BoundsError(S.rank, (x, y)))
    0 < damping <= 1 || throw(ArgumentError("local damping must lie in (0,1]"))
    for k in 1:4
        xy0 = _CTM_LOCAL_OFFSETS[mod1(k + 1, 4)]
        xy1 = _CTM_LOCAL_OFFSETS[mod1(k + 2, 4)]
        for (slot, xy) in enumerate((xy0, xy1))
            a = Ainner[k][slot]
            na = norm(a)
            (isfinite(na) && na > 0) || throw(ArgumentError("local A update is singular"))
            site = _ctm_local_site(x, y, xy)
            anew = a / na
            mixed = damping == 1 ? anew : (1 - damping) * S.A[k][site...] + damping * anew
            nm = norm(mixed)
            (isfinite(nm) && nm > 0) || throw(ArgumentError("damped local A update is singular"))
            S.A[k][site...] = mixed / nm
        end
        xyc = _CTM_LOCAL_OFFSETS[mod1(k + 2, 4)]
        c = cinner[k]
        nc = norm(c)
        (isfinite(nc) && nc > 0) || throw(ArgumentError("local c update is singular"))
        sitec = _ctm_local_site(x, y, xyc)
        cnew = c / nc
        mixedc = damping == 1 ? cnew : (1 - damping) * S.c[k][sitec...] + damping * cnew
        ncm = norm(mixedc)
        (isfinite(ncm) && ncm > 0) || throw(ArgumentError("damped local c update is singular"))
        S.c[k][sitec...] = mixedc / ncm
    end
    return S
end

function _ctm_local_patch(S::CTMLocalCycleState, x::Int, y::Int)
    Ap = ntuple(4) do k
        [S.A[k][x + dx, y + dy] for dx in 0:1, dy in 0:1]
    end
    cp = ntuple(4) do k
        [S.c[k][x + dx, y + dy] for dx in 0:1, dy in 0:1]
    end
    return Ap, cp
end

# Dense reference construction of the kth enlarged corner map. The production version can replace
# the final dense multiplication with a matrix-free action without changing state ownership.
function _ctm_local_corner(T::Array{R, 4}, A, c::AbstractMatrix, k::Int) where {R<:Real}
    kp1 = mod1(k + 1, 4)
    order = (k, kp1, mod1(k + 2, 4), mod1(k + 3, 4))
    Tr = permutedims(T, order)
    q0, q1, q2, q3 = size(Tr)
    chi = size(c, 1)
    size(c) == (chi, chi) || throw(DimensionMismatch("corner must be square"))
    size(A[k]) == (q0, chi, chi) || throw(DimensionMismatch("A[$k] has the wrong shape"))
    size(A[kp1]) == (q1, chi, chi) || throw(DimensionMismatch("A[$kp1] has the wrong shape"))

    # A_k c_k A_{k+1}, arranged as (q0,q1) x (i_k,j_{k+1}).
    A0c = reshape(A[k], q0 * chi, chi) * c
    A1m = reshape(permutedims(A[kp1], (2, 1, 3)), chi, q1 * chi)
    ring = reshape(A0c * A1m, q0, chi, q1, chi)
    ringm = reshape(permutedims(ring, (1, 3, 2, 4)), q0 * q1, chi * chi)

    # T contracts q0,q1; the open map is (q3,i_k) x (q2,j_{k+1}).
    Tm = reshape(permutedims(Tr, (4, 3, 1, 2)), q3 * q2, q0 * q1)
    raw = reshape(Tm * ringm, q3, q2, chi, chi)
    return reshape(permutedims(raw, (1, 3, 2, 4)), q3 * chi, q2 * chi)
end

function _ctm_local_orth(X::AbstractMatrix, r::Int; cutoff = 1.0e-14)
    isempty(X) && return zeros(eltype(X), size(X, 1), 0)
    F = svd(X)
    isempty(F.S) && return zeros(eltype(X), size(X, 1), 0)
    keep = min(r, count(s -> s > cutoff * F.S[1], F.S))
    return Matrix(F.U[:, 1:keep])
end

# Whiten a left/right pair so L*R = I. This is the oblique analogue of canonicalising an MPS bond.
function _ctm_local_biorth(R::AbstractMatrix, L::AbstractMatrix; cutoff = 1.0e-14)
    F = svd(L * R)
    isempty(F.S) && return nothing
    keep = findall(s -> s > cutoff * F.S[1], F.S)
    isempty(keep) && return nothing
    U = F.U[:, keep]
    V = adjoint(F.Vt)[:, keep]
    W = Diagonal(inv.(sqrt.(F.S[keep])))
    return R * V * W, W * adjoint(U) * L
end

# Put an oblique projector in the balanced gauge used by MP--BP.  If P=R*L and L*R=I, the compact
# SVD P=U*S*V' gives the factorisation Rb=U*sqrt(S), Lb=sqrt(S)*V'.  Idempotency then implies
# Lb*Rb=I while Rb'*Rb = Lb*Lb' = S.  This is considerably better conditioned than independently
# whitening the four bonds, and is invariant under the incoming retained-bond gauge.
function _ctm_local_balance(R::AbstractMatrix, L::AbstractMatrix; cutoff = 1.0e-14)
    pair = _ctm_local_biorth(R, L; cutoff)
    isnothing(pair) && return nothing
    R0, L0 = pair
    F = svd(R0 * L0)
    isempty(F.S) && return nothing
    keep = count(s -> s > cutoff * F.S[1], F.S)
    keep < 1 && return nothing
    root = Diagonal(sqrt.(F.S[1:keep]))
    return Matrix(F.U[:, 1:keep] * root), Matrix(root * F.Vt[1:keep, :])
end

# Balanced-Schur gauge from the numerical-implementation section. `ZR` has orthonormal columns and
# `ZL` orthonormal rows.  For ZL*ZR = U*s*V', a QL factorisation
#
#     s^(-1/2) V' = Q L
#
# gives VR = ZR*L' and VL = Q'*s^(-1/2)*U'*ZL. Then VL*VR=I, the two Gram matrices agree, and the
# right gauge L' is upper triangular, so triangular/quasi-triangular reduced corner factors remain
# triangular under the gauge change. Julia has no public `ql`, hence the reversal-to-QR identity.
function _ctm_local_balance_schur(ZR::AbstractMatrix, ZL::AbstractMatrix;
                                  cutoff = 1.0e-14)
    F = svd(ZL * ZR)
    isempty(F.S) && return nothing
    keep = count(s -> s > cutoff * F.S[1], F.S)
    keep == length(F.S) || return nothing
    Sinv = Diagonal(inv.(sqrt.(F.S)))
    A = Sinv * F.Vt
    r = size(A, 1)
    J = Matrix{eltype(A)}(I, r, r)[:, end:-1:1]
    QRF = qr(J * A * J)
    Q = Matrix(QRF.Q)
    R = Matrix(QRF.R)
    QL = J * Q * J
    Ltri = J * R * J
    VR = ZR * transpose(Ltri)
    VL = transpose(QL) * Sinv * transpose(F.U) * ZL
    return Matrix(VR), Matrix(VL)
end

# Orthogonally transport a newly balanced pair to the frame already stored on this bond.  The joint
# Procrustes objective uses both right and left factors.  R->R*O, L->O'*L leaves L*R, R*L, the
# balanced Gram equality, and every reduced corner invariant, while removing arbitrary Schur signs
# and rotations before overlapping plaquettes overwrite shared tensors.
function _ctm_local_align_pair(R::AbstractMatrix, L::AbstractMatrix,
                               Rold::AbstractMatrix, Lold::AbstractMatrix)
    r = size(R, 2)
    (size(L, 1) == r && size(Rold, 1) == size(R, 1) &&
     size(Lold, 2) == size(L, 2) && size(Rold, 2) >= r && size(Lold, 1) >= r) ||
        return R, L
    M = transpose(R) * Rold[:, 1:r] + L * transpose(Lold[1:r, :])
    F = svd(M)
    O = F.U * F.Vt
    return Matrix(R * O), Matrix(transpose(O) * L)
end

function _ctm_local_start_block(G, n::Int, r::Int, rng; cutoff = 1.0e-14)
    cols = Vector{Vector{eltype(G)}}()
    if size(G, 1) == n
        scale = maximum((norm(@view G[:, j]) for j in axes(G, 2)); init = zero(real(eltype(G))))
        if scale > 0
            for j in axes(G, 2)
                norm(@view G[:, j]) > cutoff * scale && push!(cols, collect(@view G[:, j]))
            end
        end
    end
    X = isempty(cols) ? zeros(eltype(G), n, 0) : reduce(hcat, cols)
    X = hcat(X, randn(rng, eltype(G), n, r))
    return _ctm_local_orth(X, r; cutoff)
end

# One block-Arnoldi expansion on each member of a periodic chain. `direction = :right` follows
# C_k : bond(k+1) -> bond(k); `:left` follows C_k' in the opposite direction. Reorthogonalising
# twice is cheap at the intended K=3--4 and prevents an almost-closed Krylov space being counted
# twice at low chi.
function _ctm_local_periodic_krylov(C, starts; depth::Int = 4, direction::Symbol = :right,
                                    cutoff = 1.0e-14)
    depth >= 1 || throw(ArgumentError("periodic Krylov depth must be positive"))
    Q = [copy(B) for B in starts]
    last = [copy(B) for B in starts]
    for _ in 2:depth
        candidate = Vector{Matrix{eltype(C[1])}}(undef, 4)
        if direction === :right
            for k in 1:4
                candidate[k] = C[k] * last[mod1(k + 1, 4)]
            end
        elseif direction === :left
            for k in 1:4
                candidate[mod1(k + 1, 4)] = transpose(C[k]) * last[k]
            end
        else
            throw(ArgumentError("periodic Krylov direction must be :right or :left"))
        end
        any_added = false
        for k in 1:4
            W = candidate[k]
            for _ in 1:2
                W -= Q[k] * (transpose(Q[k]) * W)
            end
            B = _ctm_local_orth(W, size(starts[k], 2); cutoff)
            last[k] = B
            if !isempty(B)
                Q[k] = hcat(Q[k], B)
                any_added = true
            end
        end
        any_added || break
    end
    return Tuple(Q)
end

# Manuscript block-Arnoldi space K={V,ΛV,...,Λ^K V}, formed without materialising Λ.  Repeated
# thin SVD orthogonalisation is deterministic and robust for the small depth K=3--4 used here.
function _ctm_local_product_krylov(C, start::AbstractMatrix; depth::Int = 4,
                                   direction::Symbol = :right, cutoff = 1.0e-14)
    depth >= 1 || throw(ArgumentError("block Krylov depth must be positive"))
    Q = _ctm_local_orth(start, size(start, 2); cutoff)
    isempty(Q) && return Q
    last = Q
    apply = if direction === :right
        X -> C[1] * (C[2] * (C[3] * (C[4] * X)))
    elseif direction === :left
        X -> transpose(C[4]) * (transpose(C[3]) *
             (transpose(C[2]) * (transpose(C[1]) * X)))
    else
        throw(ArgumentError("block Krylov direction must be :right or :left"))
    end
    blockrank = size(start, 2)
    for _ in 2:depth
        W = apply(last)
        for _ in 1:2
            W -= Q * (transpose(Q) * W)
        end
        last = _ctm_local_orth(W, blockrank; cutoff)
        isempty(last) && break
        Q = hcat(Q, last)
    end
    return Q
end

# Two-sided Petrov--Galerkin extraction on the compressed block-Krylov problem.  The trial/test
# spaces are first biorthogonalised, after which ordinary real Schur decompositions of H and H' give
# matched right/left invariant spaces. Complete real Schur blocks are retained at the cut.
function _ctm_local_block_bases(C::NTuple{4, <:AbstractMatrix}, chi::Int;
                                seed::UInt = 0x12345678, cutoff = 1.0e-14,
                                guess = nothing, krylov_depth::Int = 4)
    n = ntuple(k -> size(C[k], 1), 4)
    all(k -> size(C[k], 2) == n[mod1(k + 1, 4)], 1:4) ||
        throw(DimensionMismatch("the four local corner maps do not form a cycle"))
    r = min(chi, minimum(n))
    rngR = Xoshiro(seed)
    rngL = Xoshiro(seed + 0x9e3779b9)
    GR = isnothing(guess) ? zeros(eltype(C[1]), n[1], 0) : guess.VR[1]
    GL = isnothing(guess) ? zeros(eltype(C[1]), n[1], 0) : transpose(guess.VL[1])
    startR = _ctm_local_start_block(GR, n[1], r, rngR; cutoff)
    startL = _ctm_local_start_block(GL, n[1], r, rngL; cutoff)
    QR = _ctm_local_product_krylov(C, startR; depth = krylov_depth,
                                   direction = :right, cutoff)
    QL = _ctm_local_product_krylov(C, startL; depth = krylov_depth,
                                   direction = :left, cutoff)
    m = min(size(QR, 2), size(QL, 2))
    m >= r || return nothing
    QR = QR[:, 1:m]; QL = QL[:, 1:m]
    overlap = svd(transpose(QL) * QR)
    keep = count(s -> s > cutoff * overlap.S[1], overlap.S)
    keep >= r || return nothing
    rootinv = Diagonal(inv.(sqrt.(overlap.S[1:keep])))
    QRb = QR * adjoint(overlap.Vt)[:, 1:keep] * rootinv
    QLb = QL * overlap.U[:, 1:keep] * rootinv
    applyR(X) = C[1] * (C[2] * (C[3] * (C[4] * X)))
    H = transpose(QLb) * applyR(QRb)
    FR = schur(H)
    FL = schur(transpose(H))
    selectR, nr = _ctm_local_schur_select(FR, r; replace = false)
    selectL, nl = _ctm_local_schur_select(FL, r; replace = false)
    kres = min(nr, nl)
    kres >= 1 || return nothing
    # If the two compressed orderings back off differently, reorder again at their common complete
    # rank. This avoids pairing different spectral sets merely because one Schur form listed a block
    # boundary first.
    selectR, nr = _ctm_local_schur_select(FR, kres; replace = false)
    selectL, nl = _ctm_local_schur_select(FL, kres; replace = false)
    kres = min(nr, nl)
    kres >= 1 || return nothing
    Fro = ordschur(FR, selectR)
    Flo = ordschur(FL, selectL)
    ZR = Vector{Matrix{eltype(C[1])}}(undef, 4)
    ZL = Vector{Matrix{eltype(C[1])}}(undef, 4)
    ZR[1] = Matrix(QRb * Fro.Z[:, 1:kres])
    ZL[1] = Matrix(transpose(QLb * Flo.Z[:, 1:kres]))
    for k in (4, 3, 2)
        F = qr(C[k] * ZR[mod1(k + 1, 4)])
        ZR[k] = Matrix(F.Q[:, 1:kres])
    end
    for k in 1:3
        F = qr(transpose(ZL[k] * C[k]))
        ZL[k + 1] = Matrix(transpose(F.Q[:, 1:kres]))
    end
    VR = Vector{Matrix{eltype(C[1])}}(undef, 4)
    VL = Vector{Matrix{eltype(C[1])}}(undef, 4)
    for k in 1:4
        pair = _ctm_local_balance_schur(ZR[k], ZL[k]; cutoff)
        isnothing(pair) && return nothing
        VR[k], VL[k] = pair
    end
    reduced = ntuple(k -> VL[k] * C[k] * VR[mod1(k + 1, 4)], 4)
    return (VR = Tuple(VR), VL = Tuple(VL), c = reduced, rank = kres)
end

# Extract the dominant rank-r periodic invariant spaces from a compressed cyclic operator.  The
# four roots associated with each product eigenvalue have equal magnitude, hence selecting 4r Schur
# directions retains r physical directions on every bond.  The row block of the ordered real Schur
# space is reduced by an SVD; this also handles real 2x2 blocks without manually pairing conjugates.
function _ctm_local_periodic_schur(C, Q, r::Int; direction::Symbol = :right,
                                   cutoff = 1.0e-14)
    dims = collect(map(q -> size(q, 2), Q))
    any(==(0), dims) && return nothing
    firsts = cumsum(vcat(1, dims[1:3]))
    ranges = ntuple(k -> firsts[k]:(firsts[k] + dims[k] - 1), 4)
    H = zeros(eltype(C[1]), sum(dims), sum(dims))
    if direction === :right
        for k in 1:4
            kp1 = mod1(k + 1, 4)
            H[ranges[k], ranges[kp1]] .= transpose(Q[k]) * C[k] * Q[kp1]
        end
    elseif direction === :left
        for k in 1:4
            kp1 = mod1(k + 1, 4)
            H[ranges[kp1], ranges[k]] .= transpose(Q[kp1]) * transpose(C[k]) * Q[k]
        end
    else
        throw(ArgumentError("periodic Schur direction must be :right or :left"))
    end
    F = schur(H)
    values = F.values
    nkeep = min(4r, length(values))
    nkeep >= 4 || return nothing
    order = sortperm(abs.(values); rev = true)
    threshold = abs(values[order[nkeep]])
    # Include a roundoff tie at the boundary. This guarantees that a real Schur conjugate pair is
    # never split and is harmless because the per-bond SVD below returns exactly r directions.
    select = abs.(values) .>= threshold * (1 - 100eps(real(eltype(H))))
    Fo = try
        ordschur(F, select)
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
    Z = Matrix(Fo.Z[:, 1:count(select)])
    out = Vector{Matrix{eltype(H)}}(undef, 4)
    for k in 1:4
        B = @view Z[ranges[k], :]
        U = _ctm_local_orth(B, r; cutoff)
        size(U, 2) == r || return nothing
        out[k] = Q[k] * U
    end
    return Tuple(out)
end

# Coupled periodic solve for all four left/right invariant spaces.  This follows the collaborator's
# plaquette CTMRG construction: short cyclic block-Krylov spaces, periodic Schur on the compressed
# problem, and balanced biorthogonal factors.  In contrast to forming C1*C2*C3*C4, the spectrum is
# never raised to the fourth power; in contrast to propagating four bases independently, all bonds
# are selected by one periodic invariant space.
function _ctm_local_cycle_bases(C::NTuple{4, <:AbstractMatrix}, chi::Int;
                                seed::UInt = 0x12345678, cutoff = 1.0e-14,
                                guess = nothing, krylov_depth::Int = 4)
    n = ntuple(k -> size(C[k], 1), 4)
    all(k -> size(C[k], 2) == n[mod1(k + 1, 4)], 1:4) ||
        throw(DimensionMismatch("the four local corner maps do not form a cycle"))
    r = min(chi, minimum(n))
    rngR = Xoshiro(seed)
    rngL = Xoshiro(seed + 0x9e3779b9)
    startsR = ntuple(4) do k
        G = isnothing(guess) ? zeros(eltype(C[1]), n[k], 0) : guess.VR[k]
        _ctm_local_start_block(G, n[k], r, rngR; cutoff)
    end
    startsL = ntuple(4) do k
        G = isnothing(guess) ? zeros(eltype(C[1]), n[k], 0) : transpose(guess.VL[k])
        _ctm_local_start_block(G, n[k], r, rngL; cutoff)
    end
    QR = _ctm_local_periodic_krylov(C, startsR; depth = krylov_depth,
                                    direction = :right, cutoff)
    QL = _ctm_local_periodic_krylov(C, startsL; depth = krylov_depth,
                                    direction = :left, cutoff)
    right = _ctm_local_periodic_schur(C, QR, r; direction = :right, cutoff)
    left = _ctm_local_periodic_schur(C, QL, r; direction = :left, cutoff)
    (isnothing(right) || isnothing(left)) && return nothing
    VR = collect(right)
    VL = [Matrix(transpose(left[k])) for k in 1:4]
    kres = r
    for k in 1:4
        pair = _ctm_local_balance(VR[k], VL[k]; cutoff)
        isnothing(pair) && return nothing
        VR[k], VL[k] = pair
        if !isnothing(guess)
            VR[k], VL[k] = _ctm_local_align_pair(
                VR[k], VL[k], guess.VR[k], guess.VL[k])
        end
        kres = min(kres, size(VR[k], 2))
    end
    # A rare rank drop during whitening is handled uniformly instead of leaving different widths on
    # different sides of the same plaquette.
    VR = [v[:, 1:kres] for v in VR]
    VL = [v[1:kres, :] for v in VL]
    reduced = ntuple(k -> VL[k] * C[k] * VR[mod1(k + 1, 4)], 4)
    rr = maximum(norm(C[k] * VR[mod1(k + 1, 4)] - VR[k] * reduced[k]) /
                 max(norm(C[k] * VR[mod1(k + 1, 4)]), eps(Float64)) for k in 1:4)
    lr = maximum(norm(VL[k] * C[k] - reduced[k] * VL[mod1(k + 1, 4)]) /
                 max(norm(VL[k] * C[k]), eps(Float64)) for k in 1:4)
    br = maximum(norm(transpose(VR[k]) * VR[k] - VL[k] * transpose(VL[k])) /
                 max(norm(transpose(VR[k]) * VR[k]), eps(Float64)) for k in 1:4)
    return (VR = Tuple(VR), VL = Tuple(VL), c = reduced, rank = kres,
            right_residual = rr, left_residual = lr, balance_residual = br)
end

# Select complete real-Schur blocks within a fixed rank budget.  In particular, if a two-dimensional
# complex-conjugate block straddles the budget, skip it and admit the next one-dimensional block. This
# is the manuscript's fixed-rank alternative to dropping the whole final multiplet. `ordschur` then
# performs the orthogonal reordering; merely taking a non-prefix subset of Schur vectors would not
# produce an invariant subspace for a non-normal matrix.
function _ctm_local_schur_select(F, budget::Int; replace::Bool = true)
    n = length(F.values)
    budget = min(budget, n)
    blocks = UnitRange{Int}[]
    i = 1
    while i <= n
        paired = i < n && !iszero(F.T[i + 1, i])
        push!(blocks, i:(paired ? i + 1 : i))
        i += paired ? 2 : 1
    end
    sort!(blocks; by = b -> maximum(abs, @view F.values[b]), rev = true)
    selected = falses(n)
    remaining = budget
    for block in blocks
        if length(block) > remaining
            replace ? continue : break
        end
        selected[block] .= true
        remaining -= length(block)
        remaining == 0 && break
    end
    return selected, budget - remaining
end

function _ctm_local_ordered_subspace(apply, vectors, converged::Int, budget::Int;
                                     replace::Bool = true)
    m = min(converged, length(vectors))
    m >= 1 || return nothing
    Q = reduce(hcat, vectors[1:m])
    F = schur(transpose(Q) * apply(Q))
    selected, keep = _ctm_local_schur_select(F, budget; replace)
    keep >= 1 || return nothing
    Fo = try
        ordschur(F, selected)
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
    return Matrix(Q * Fo.Z[:, 1:keep]), keep
end

# Dominant invariant subspace of the four-corner product. The product is applied matrix-free both
# in Arnoldi and in the reduced real-Schur projection: forming the dense product gave identical
# spaces but made the otherwise robust route scale cubically in the enlarged dimension.
function _ctm_local_product_bases(C::NTuple{4, <:AbstractMatrix}, chi::Int;
                                  seed::UInt = 0x12345678, cutoff = 1.0e-14,
                                  guess = nothing)
    n = ntuple(k -> size(C[k], 1), 4)
    r = min(chi, minimum(n))
    all(k -> size(C[k], 2) == n[mod1(k + 1, 4)], 1:4) ||
        throw(DimensionMismatch("the four local corner maps do not form a cycle"))
    # Dense multiplication is deliberately retained for modest enlarged spaces: in an exactly
    # rank-deficient lossless problem, changing matrix-matrix to matrix-vector association can move
    # a roundoff-degenerate Schur block across the cut. Above this crossover the dense product is
    # the dominant cost, while the retained subspace is genuinely truncated and matrix-free Arnoldi
    # is both the scalable and stable route.
    if n[1] <= 192
        M = C[1] * C[2] * C[3] * C[4]
        scale = max(opnorm(M, Inf), eps(real(eltype(M))))
        fwd = v -> (M * v) / scale
        bwd = v -> (transpose(M) * v) / scale
    else
        scales = ntuple(k -> max(opnorm(C[k], Inf), eps(real(eltype(C[k])))), 4)
        scale = prod(scales)
        fwd = v -> (C[1] * (C[2] * (C[3] * (C[4] * v)))) / scale
        bwd = v -> (transpose(C[4]) * (transpose(C[3]) *
                    (transpose(C[2]) * (transpose(C[1]) * v)))) / scale
    end
    v0 = randn(Xoshiro(seed), eltype(C[1]), n[1])
    l0 = randn(Xoshiro(seed + 0x9e3779b9), eltype(C[1]), n[1])
    # Resolve a few blocks past the requested cut so the real-Schur structure immediately across
    # the boundary is known; detecting a crossing doublet requires seeing both of its members.
    nev = min(n[1], r + 8)
    alg = Arnoldi(; krylovdim = min(n[1], max(4nev + 8, 24)), tol = 1.0e-16,
                  verbosity = 0)
    _, rv, _, ir = schursolve(fwd, v0, nev, :LM, alg)
    _, lv, _, il = schursolve(bwd, l0, nev, :LM, alg)
    # Use the manuscript's first option here: drop a crossing multiplet. The smaller active block is
    # embedded in the fixed-chi A/c allocation by exact zero padding, giving spatially varying active
    # rank without resizing a retained index during the sweep.
    right = _ctm_local_ordered_subspace(fwd, rv, ir.converged, r; replace = false)
    left = _ctm_local_ordered_subspace(bwd, lv, il.converged, r;
                                       replace = false)
    (isnothing(right) || isnothing(left)) && return nothing
    R1, nr = right
    L1, nl = left
    kres = min(nr, nl)
    kres >= 1 || return nothing
    # Periodically propagate orthogonal Schur bases with QR, not SVD. The resulting right reduced
    # factors are upper/quasi-triangular; `_ctm_local_balance_schur` preserves that property.
    ZR = Vector{Matrix{eltype(C[1])}}(undef, 4)
    ZL = Vector{Matrix{eltype(C[1])}}(undef, 4)
    ZR[1] = R1[:, 1:kres]
    ZL[1] = Matrix(transpose(L1[:, 1:kres]))
    for k in (4, 3, 2)
        F = qr(C[k] * ZR[mod1(k + 1, 4)])
        ZR[k] = Matrix(F.Q[:, 1:kres])
    end
    for k in 1:3
        F = qr(transpose(ZL[k] * C[k]))
        ZL[k + 1] = Matrix(transpose(F.Q[:, 1:kres]))
    end
    VR = Vector{Matrix{eltype(C[1])}}(undef, 4)
    VL = Vector{Matrix{eltype(C[1])}}(undef, 4)
    for k in 1:4
        pair = _ctm_local_balance_schur(ZR[k], ZL[k]; cutoff)
        isnothing(pair) && return nothing
        VR[k], VL[k] = pair
        kres = min(kres, size(VR[k], 2))
    end
    VR = [v[:, 1:kres] for v in VR]
    VL = [v[1:kres, :] for v in VL]
    reduced = ntuple(k -> VL[k] * C[k] * VR[mod1(k + 1, 4)], 4)
    return (VR = Tuple(VR), VL = Tuple(VL), c = reduced, rank = kres)
end

# Dense cyclic-lift oracle for the true periodic Schur problem. This never forms the four-factor
# product and is used to validate/repair the short-Krylov implementation at low chi; its O((sum n)^3)
# cost is intentionally not the production default.
function _ctm_local_dense_periodic_bases(C::NTuple{4, <:AbstractMatrix}, chi::Int;
                                         cutoff = 1.0e-14, guess = nothing)
    n = ntuple(k -> size(C[k], 1), 4)
    r = min(chi, minimum(n))
    Q = ntuple(k -> Matrix{eltype(C[1])}(I, n[k], n[k]), 4)
    right = _ctm_local_periodic_schur(C, Q, r; direction = :right, cutoff)
    left = _ctm_local_periodic_schur(C, Q, r; direction = :left, cutoff)
    (isnothing(right) || isnothing(left)) && return nothing
    VR = collect(right)
    VL = [Matrix(transpose(left[k])) for k in 1:4]
    kres = r
    for k in 1:4
        pair = _ctm_local_balance(VR[k], VL[k]; cutoff)
        isnothing(pair) && return nothing
        VR[k], VL[k] = pair
        if !isnothing(guess)
            VR[k], VL[k] = _ctm_local_align_pair(
                VR[k], VL[k], guess.VR[k], guess.VL[k])
        end
        kres = min(kres, size(VR[k], 2))
    end
    VR = [v[:, 1:kres] for v in VR]
    VL = [v[1:kres, :] for v in VL]
    reduced = ntuple(k -> VL[k] * C[k] * VR[mod1(k + 1, 4)], 4)
    rr = maximum(norm(C[k] * VR[mod1(k + 1, 4)] - VR[k] * reduced[k]) /
                 max(norm(C[k] * VR[mod1(k + 1, 4)]), eps(Float64)) for k in 1:4)
    lr = maximum(norm(VL[k] * C[k] - reduced[k] * VL[mod1(k + 1, 4)]) /
                 max(norm(VL[k] * C[k]), eps(Float64)) for k in 1:4)
    return (VR = Tuple(VR), VL = Tuple(VL), c = reduced, rank = kres,
            right_residual = rr, left_residual = lr)
end

# Recover the current cyclic coordinates from the fixed A/c ring. Besides providing a warm Krylov
# start, these identities are the local consistency contract that makes A/c sufficient state.
function _ctm_local_coordinates(Ap, cp; cutoff = 1.0e-14)
    chi = size(cp[1][1, 1], 1)
    VL = Vector{Matrix{eltype(cp[1][1, 1])}}(undef, 4)
    VR = similar(VL)
    for j in 1:4
        edge = mod1(j - 1, 4)
        xy = _CTM_LOCAL_OFFSETS[j]
        xyi = _CTM_LOCAL_OFFSETS[mod1(j + 1, 4)]
        A = Ap[edge][xy[1] + 1, xy[2] + 1]
        cext = cp[edge][xy[1] + 1, xy[2] + 1]
        cint = cp[edge][xyi[1] + 1, xyi[2] + 1]
        blocks = cat((pinv(cint; rtol = cutoff) * A[a, :, :] * cext
                      for a in axes(A, 1))...; dims = 3)       # (i,h,q)
        VL[j] = reshape(permutedims(blocks, (1, 3, 2)), chi, :) # (i,(q,h))
    end
    for j in 1:4
        edge = mod1(j + 1, 4)
        xy = _CTM_LOCAL_OFFSETS[mod1(j + 3, 4)]
        xyi = _CTM_LOCAL_OFFSETS[mod1(j + 2, 4)]
        A = Ap[edge][xy[1] + 1, xy[2] + 1]
        cext = cp[j][xy[1] + 1, xy[2] + 1]
        cint = cp[j][xyi[1] + 1, xyi[2] + 1]
        blocks = cat((cext * A[a, :, :] * pinv(cint; rtol = cutoff)
                      for a in axes(A, 1))...; dims = 3)       # (i,h,q)
        VR[j] = reshape(permutedims(blocks, (3, 1, 2)), :, chi) # ((q,i),h)
    end
    return (VL = Tuple(VL), VR = Tuple(VR))
end

function _ctm_local_reconstruct(Ap, cp, bases, x::Int, y::Int)
    chi = size(cp[1][1, 1], 1)
    r = bases.rank
    # Embed the resolved subspace into fixed-chi storage. The inactive tail is exactly zero.
    VL = ntuple(4) do k
        out = zeros(eltype(bases.VL[k]), chi, size(bases.VL[k], 2))
        out[1:r, :] .= bases.VL[k]
        out
    end
    VR = ntuple(4) do k
        out = zeros(eltype(bases.VR[k]), size(bases.VR[k], 1), chi)
        out[:, 1:r] .= bases.VR[k]
        out
    end
    cinner = ntuple(4) do k
        out = zeros(eltype(bases.c[k]), chi, chi)
        out[1:r, 1:r] .= bases.c[k]
        out
    end

    # Use the newly reduced inner corners and the untouched exterior corners in the coordinate
    # identities A*c_ext = c_inner*VL and c_ext*A = VR*c_inner.
    cpnew = ntuple(4) do k
        field = copy(cp[k])
        xy = _CTM_LOCAL_OFFSETS[mod1(k + 2, 4)]
        field[xy[1] + 1, xy[2] + 1] = cinner[k]
        field
    end
    Ainner = [Vector{Array{eltype(cinner[1]), 3}}(undef, 2) for _ in 1:4]
    for j in 1:4
        edge = mod1(j - 1, 4)
        xy = _CTM_LOCAL_OFFSETS[j]
        xyi = _CTM_LOCAL_OFFSETS[mod1(j + 1, 4)]
        cext = cpnew[edge][xy[1] + 1, xy[2] + 1]
        cint = cpnew[edge][xyi[1] + 1, xyi[2] + 1]
        q = size(VL[j], 2) ÷ chi
        rhs = permutedims(reshape(cint * VL[j], chi, q, chi), (2, 1, 3))
        Ainner[edge][1] = cat((rhs[a, :, :] * pinv(cext) for a in 1:q)...; dims = 3) |>
                              z -> permutedims(z, (3, 1, 2))
    end
    for j in 1:4
        edge = mod1(j + 1, 4)
        xy = _CTM_LOCAL_OFFSETS[mod1(j + 3, 4)]
        xyi = _CTM_LOCAL_OFFSETS[mod1(j + 2, 4)]
        cext = cpnew[j][xy[1] + 1, xy[2] + 1]
        cint = cpnew[j][xyi[1] + 1, xyi[2] + 1]
        q = size(VR[j], 1) ÷ chi
        Vj = reshape(VR[j], q, chi, chi)
        rhs = cat((Vj[a, :, :] * cint for a in 1:q)...; dims = 3)
        rhs = permutedims(rhs, (3, 1, 2))
        Ainner[edge][2] = cat((pinv(cext) * rhs[a, :, :] for a in 1:q)...; dims = 3) |>
                              z -> permutedims(z, (3, 1, 2))
    end
    return Tuple((Tuple(a) for a in Ainner)), cinner
end

"""
    _ctm_local_update!(state, x, y)

Perform one Gauss--Seidel plaquette update. Returns the resolved local rank, or zero if
the invariant-space solve could not produce a biorthogonal pair. Failure leaves the state unchanged.
"""
function _ctm_local_bases(S::CTMLocalCycleState{T}, x::Int, y::Int;
                          seed::UInt = hash((x, y)), solver::Symbol = :product) where {T<:Real}
    Ap, cp = _ctm_local_patch(S, x, y)
    C = ntuple(4) do k
        xy = _CTM_LOCAL_OFFSETS[k]
        gx, gy = _ctm_local_site(x, y, xy)
        Ak = ntuple(j -> Ap[j][xy[1] + 1, xy[2] + 1], 4)
        ck = cp[k][xy[1] + 1, xy[2] + 1]
        _ctm_local_corner(S.sites[gx, gy], Ak, ck, k)
    end
    guess = _ctm_local_coordinates(Ap, cp)
    bases = solver === :block ? _ctm_local_block_bases(C, S.chi; seed, guess) :
            solver === :periodic ? _ctm_local_cycle_bases(C, S.chi; seed, guess) :
            solver === :dense_periodic ? _ctm_local_dense_periodic_bases(C, S.chi; guess) :
            solver === :product ? _ctm_local_product_bases(C, S.chi; seed, guess) :
            throw(ArgumentError(
                "local cycle solver must be :block, :product, :periodic, or :dense_periodic"))
    return bases, Ap, cp
end

function _ctm_local_update!(S::CTMLocalCycleState{T}, x::Int, y::Int;
                            seed::UInt = hash((x, y)), solver::Symbol = :product,
                            damping::Real = 1) where {T<:Real}
    bases, Ap, cp = _ctm_local_bases(S, x, y; seed, solver)
    isnothing(bases) && return 0
    Ainner, cinner = _ctm_local_reconstruct(Ap, cp, bases, x, y)
    _ctm_local_scatter!(S, Ainner, cinner, x, y; damping)
    S.rank[x, y] = bases.rank
    return bases.rank
end

# Snapshot the gauge-invariant oblique projectors Π=VR*VL selected by every local problem. Unlike a
# raw A/c tensor difference, this is blind to the exact retained-bond gauge and therefore suitable as
# an environment convergence diagnostic.
function _ctm_local_projectors(S::CTMLocalCycleState; solver::Symbol = :product)
    Nx, Ny = size(S.sites)
    out = Dict{Tuple{Int, Int, Int}, Matrix{eltype(S.sites[1, 1])}}()
    for x in 1:(Nx - 1), y in 1:(Ny - 1)
        bases, _, _ = _ctm_local_bases(S, x, y; solver)
        isnothing(bases) && return nothing
        for k in 1:4
            out[(x, y, k)] = bases.VR[k] * bases.VL[k]
        end
    end
    return out
end

function _ctm_local_projectordist(a, b)
    (isnothing(a) || isnothing(b) || keys(a) != keys(b)) && return nothing
    worst = 0.0
    for key in keys(a)
        worst = max(worst, norm(a[key] - b[key]) / max(norm(a[key]), eps(Float64)))
    end
    return worst
end

# Gauge-invariant local response surrounding every site.  For fixed open virtual labels a_k this is
# tr(A_1[a_1] c_1 ... A_4[a_4] c_4), i.e. the tensor contracted with the local factor to obtain its
# contribution. It determines arbitrary one-site insertions and is therefore the correct convergence
# witness for observables; raw A/c tensors and retained projectors can move inside null directions.
function _ctm_local_responses(S::CTMLocalCycleState)
    Nx, Ny = size(S.sites)
    out = Dict{Tuple{Int, Int}, Array{eltype(S.sites[1, 1]), 4}}()
    for x in 1:Nx, y in 1:Ny
        q = ntuple(k -> size(S.A[k][x, y], 1), 4)
        B = ntuple(4) do k
            A = S.A[k][x, y]
            c = S.c[k][x, y]
            [(@view A[a, :, :]) * c for a in axes(A, 1)]
        end
        # Pair adjacent halves with GEMM, then close the two remaining retained indices in one
        # matrix product. This replaces q^4 separate chi^3 products by 2q^2 products plus one
        # q^2-by-q^2 contraction and is essential for D_P=3, chi=64 observable scans.
        B1 = permutedims(cat(B[1]...; dims = 3), (3, 1, 2))
        B2 = permutedims(cat(B[2]...; dims = 3), (3, 1, 2))
        B3 = permutedims(cat(B[3]...; dims = 3), (3, 1, 2))
        B4 = permutedims(cat(B[4]...; dims = 3), (3, 1, 2))
        chi = size(B1, 2)
        P12m = reshape(B1, q[1] * chi, chi) *
               reshape(permutedims(B2, (2, 1, 3)), chi, q[2] * chi)
        P12 = permutedims(reshape(P12m, q[1], chi, q[2], chi), (1, 3, 2, 4))
        P34m = reshape(B3, q[3] * chi, chi) *
               reshape(permutedims(B4, (2, 1, 3)), chi, q[4] * chi)
        P34 = permutedims(reshape(P34m, q[3], chi, q[4], chi), (1, 3, 4, 2))
        response = reshape(reshape(P12, q[1] * q[2], chi^2) *
                           transpose(reshape(P34, q[3] * q[4], chi^2)), q)
        out[(x, y)] = Array(response)
    end
    return out
end

function _ctm_local_responsedist(a, b)
    keys(a) == keys(b) || return nothing
    worst = 0.0
    for key in keys(a)
        old, new = a[key], b[key]
        no, nn = norm(old), norm(new)
        (no == 0 || nn == 0) && return Inf
        overlap = dot(old, new) / (no * nn)
        phase = iszero(overlap) ? one(overlap) : overlap / abs(overlap)
        worst = max(worst, norm(new / nn - phase * old / no))
    end
    return worst
end

function _ctm_local_sweep!(S::CTMLocalCycleState; schedule = _ctm_local_snake(S),
                           solver::Symbol = :product, damping::Real = 1)
    failures = 0
    for (x, y) in schedule
        _ctm_local_update!(S, x, y; solver, damping) == 0 && (failures += 1)
    end
    return (; state = S, failures, updates = length(schedule))
end

# Production bridge. The local update owns fixed-width dense A/c storage, but the public cache keeps
# the established `CTMVertexEnvironments`. Conversion uses the current C/T blocks solely as an index-
# topology template; `cvm_freenergy`, `region_lnZ`, and `expect` remain completely unchanged and read
# the converted blocks through the existing sum_R c_R log(Z_R) implementation.
struct CTMLocalCycleBridge{T}
    state::CTMLocalCycleState{T}
    template::CTMVertexEnvironments
    factors::Matrix{ITensor}
    leginds::Matrix{NTuple{4, Vector{Index}}}
    qmax::NTuple{4, Int}
end

# The local directions are (right, up, left, down). In the established finite-grid C/T
# convention, `S` is grown down from the upper boundary and is therefore the environment seen
# across a site's up leg; analogously `N` is seen across its down leg. Keep this mapping explicit:
# a geometric compass-name guess silently rotates the four local corners.
_ctm_local_cdesc(k::Int, x::Int, y::Int) =
    k == 1 ? (:SE, x + 1, y + 1) : k == 2 ? (:SW, x, y + 1) :
    k == 3 ? (:NW, x, y) : (:NE, x + 1, y)
_ctm_local_adesc(k::Int, x::Int, y::Int) =
    k == 1 ? (:E, x + 1, y) : k == 2 ? (:S, x, y + 1) :
    k == 3 ? (:W, x, y) : (:N, x, y)

function _ctm_local_fused_array(t::ITensor, groups::NTuple{4, Vector{Index}},
                                qmax::NTuple{4, Int})
    is = reduce(vcat, groups; init = Index[])
    Set(is) == Set(inds(t)) || return nothing
    q = ntuple(k -> isempty(groups[k]) ? 1 : prod(ITensors.dim, groups[k]), 4)
    raw = isempty(is) ? fill(scalar(t), 1, 1, 1, 1) : reshape(ITensors.array(t, is...), q)
    out = zeros(eltype(raw), qmax)
    out[ntuple(k -> 1:q[k], 4)...] .= raw
    return out
end

function _ctm_local_bridge(cache::CTMEnvironmentCache, template::CTMVertexEnvironments)
    Lx, Ly = cache.dims
    all(haskey(cache.grid, (x, y)) for x in 1:Lx, y in 1:Ly) || return nothing
    factors = Matrix{ITensor}(undef, Lx, Ly)
    for x in 1:Lx, y in 1:Ly
        ts = bp_factors(network(cache), cache.grid[(x, y)])
        factors[x, y] = length(ts) == 1 ? only(ts) : reduce(*, ts)
        hasqns(factors[x, y]) && return nothing
    end
    # Local tensor leg order is (right, up, left, down); the plaquette offsets use y+1 as up.
    offsets = ((1, 0), (0, 1), (-1, 0), (0, -1))
    leginds = Matrix{NTuple{4, Vector{Index}}}(undef, Lx, Ly)
    legdims = Matrix{NTuple{4, Int}}(undef, Lx, Ly)
    for x in 1:Lx, y in 1:Ly
        groups = ntuple(4) do k
            xn, yn = x + offsets[k][1], y + offsets[k][2]
            1 <= xn <= Lx && 1 <= yn <= Ly ?
                collect(commoninds(factors[x, y], factors[xn, yn])) : Index[]
        end
        leginds[x, y] = groups
        legdims[x, y] = ntuple(k -> isempty(groups[k]) ? 1 : prod(ITensors.dim, groups[k]), 4)
    end
    qmax = ntuple(k -> maximum(legdims[x, y][k] for x in 1:Lx, y in 1:Ly), 4)
    rawsites = Matrix{Any}(undef, Lx, Ly)
    for x in 1:Lx, y in 1:Ly
        rawsites[x, y] = _ctm_local_fused_array(factors[x, y], leginds[x, y], qmax)
        isnothing(rawsites[x, y]) && return nothing
    end
    Traw = promote_type((eltype(rawsites[x, y]) for x in 1:Lx, y in 1:Ly)...)
    sites = if Traw <: Complex
        scale = maximum(maximum(abs, rawsites[x, y]) for x in 1:Lx, y in 1:Ly)
        imagmax = maximum(maximum(abs, imag.(rawsites[x, y])) for x in 1:Lx, y in 1:Ly)
        imagmax <= 256eps(real(Traw)) * max(scale, one(scale)) || return nothing
        [real.(rawsites[x, y]) for x in 1:Lx, y in 1:Ly]
    elseif Traw <: Real
        [Array(rawsites[x, y]) for x in 1:Lx, y in 1:Ly]
    else
        return nothing
    end
    bridge = CTMLocalCycleBridge(CTMLocalCycleState(sites, cache.maxdim), template,
                                 factors, leginds, qmax)
    _ctm_local_seed_from_vertex!(bridge) || return nothing
    return bridge
end

_ctm_local_groupdim(is) = isempty(is) ? 1 : prod(ITensors.dim, is)

function _ctm_local_array_itensor(a::AbstractArray,
                                  groups::AbstractVector{<:AbstractVector{<:Index}})
    dims = map(_ctm_local_groupdim, groups)
    slices = ntuple(k -> 1:dims[k], length(dims))
    data = a[slices...]
    is = reduce(vcat, groups; init = Index[])
    isempty(is) && return ITensor(only(data))
    return ITensor(reshape(data, map(ITensors.dim, is)...), is)
end

function _ctm_local_itensor_array(t::ITensor,
                                  groups::AbstractVector{<:AbstractVector{<:Index}})
    is = reduce(vcat, groups; init = Index[])
    Set(is) == Set(inds(t)) || return nothing
    dims = map(_ctm_local_groupdim, groups)
    isempty(is) && return fill(scalar(t), dims...)
    return reshape(ITensors.array(t, is...), dims...)
end

# Import the current C/T environment into fixed-width A/c storage. A fresh production cache supplies
# the established greedy pass; a cache carrying previous environments supplies those instead. This
# is a genuine warm start and avoids selecting a different stationary branch merely because the
# otherwise arbitrary retained tensors began as one-hot identities.
function _ctm_local_seed_from_vertex!(B::CTMLocalCycleBridge)
    S, template = B.state, B.template
    Lx, Ly = size(S.sites)
    for x in 1:Lx, y in 1:Ly, k in 1:4
        ad = _ctm_local_adesc(k, x, y)
        at = _ctm_nn(template.T, ad)
        if !isnothing(at)
            cp = _ctm_nn(template.C, _ctm_local_cdesc(mod1(k - 1, 4), x, y))
            cn = _ctm_nn(template.C, _ctm_local_cdesc(k, x, y))
            ip = isnothing(cp) ? Index[] : collect(commoninds(at, cp))
            in = isnothing(cn) ? Index[] : collect(commoninds(at, cn))
            raw = _ctm_local_itensor_array(at, [B.leginds[x, y][k], ip, in])
            isnothing(raw) && return false
            fill!(S.A[k][x, y], zero(eltype(S.A[k][x, y])))
            slices = ntuple(j -> 1:size(raw, j), 3)
            S.A[k][x, y][slices...] .= raw
        end
        cd = _ctm_local_cdesc(k, x, y)
        ct = _ctm_nn(template.C, cd)
        if !isnothing(ct)
            ap = _ctm_nn(template.T, _ctm_local_adesc(k, x, y))
            an = _ctm_nn(template.T, _ctm_local_adesc(mod1(k + 1, 4), x, y))
            ip = isnothing(ap) ? Index[] : collect(commoninds(ct, ap))
            in = isnothing(an) ? Index[] : collect(commoninds(ct, an))
            raw = _ctm_local_itensor_array(ct, [ip, in])
            isnothing(raw) && return false
            fill!(S.c[k][x, y], zero(eltype(S.c[k][x, y])))
            slices = ntuple(j -> 1:size(raw, j), 2)
            S.c[k][x, y][slices...] .= raw
        end
    end
    return true
end

function _ctm_local_to_vertex(B::CTMLocalCycleBridge)
    S, template = B.state, B.template
    Lx, Ly = size(S.sites)
    C = Dict{Tuple{Symbol, Int, Int}, Any}()
    T = Dict{Tuple{Symbol, Int, Int}, Any}()
    for x in 1:Lx, y in 1:Ly
        for k in 1:4
            ad = _ctm_local_adesc(k, x, y)
            at = _ctm_nn(template.T, ad)
            if !isnothing(at)
                cp = _ctm_nn(template.C, _ctm_local_cdesc(mod1(k - 1, 4), x, y))
                cn = _ctm_nn(template.C, _ctm_local_cdesc(k, x, y))
                ip = isnothing(cp) ? Index[] : collect(commoninds(at, cp))
                in = isnothing(cn) ? Index[] : collect(commoninds(at, cn))
                phys = B.leginds[x, y][k]
                Set(reduce(vcat, (phys, ip, in); init = Index[])) == Set(inds(at)) || return nothing
                T[ad] = _ctm_rescale(_ctm_local_array_itensor(S.A[k][x, y],
                                                              [phys, ip, in]))
            end
            cd = _ctm_local_cdesc(k, x, y)
            ct = _ctm_nn(template.C, cd)
            if !isnothing(ct)
                ap = _ctm_nn(template.T, _ctm_local_adesc(k, x, y))
                an = _ctm_nn(template.T, _ctm_local_adesc(mod1(k + 1, 4), x, y))
                ip = isnothing(ap) ? Index[] : collect(commoninds(ct, ap))
                in = isnothing(an) ? Index[] : collect(commoninds(ct, an))
                Set(vcat(ip, in)) == Set(inds(ct)) || return nothing
                C[cd] = _ctm_rescale(_ctm_local_array_itensor(S.c[k][x, y], [ip, in]))
            end
        end
    end
    return CTMVertexEnvironments(C, T, template.PH, template.PV, Lx, Ly)
end

function _ctm_local_update_cache(cache::CTMEnvironmentCache; maxiter::Integer = 30,
                                 tolerance::Real = _ctm_default_tol(cache),
                                 verbose::Bool = false,
                                 convergence::Symbol = :free_energy,
                                 require_converged::Bool = false,
                                 warn_unconverged::Bool = true)
    template = _ctm_env(cache)
    bridge = _ctm_local_bridge(cache, template)
    isnothing(bridge) && return nothing
    responses = _ctm_local_responses(bridge.state)
    prevF = nothing
    prevterms = nothing
    convergence_streak = 0
    converged = false
    finalenv = nothing
    finalres = Inf
    bestscore = Inf
    bestenv = nothing
    for sweep in 1:maxiter
        info = _ctm_local_sweep!(bridge.state; solver = :product)
        info.failures == 0 || break
        nextresponses = _ctm_local_responses(bridge.state)
        finalres = _ctm_local_responsedist(responses, nextresponses)
        env = _ctm_local_to_vertex(bridge)
        isnothing(env) && return nothing
        F, terms = _ctm_region_terms(env, cache) # unchanged sum_R c_R log(Z_R)
        dF = isnothing(prevF) ? Inf : abs(F - prevF)
        worst = isnothing(prevterms) ? Inf : maximum(abs.(terms .- prevterms))
        Fpass = dF <= tolerance * max(1, abs(F))
        Rpass = finalres <= tolerance
        verbose && @info "local cycle sweep $sweep: F=$F |dF|=$dF response=$finalres worst=$worst rank=$(minimum(bridge.state.rank)):$(maximum(bridge.state.rank))"
        finalenv = env
        pass = convergence === :free_energy ? Fpass :
               convergence === :environment ? (Fpass && Rpass) :
               (Fpass && Rpass && worst <= tolerance * max(1, abs(F)))
        convergence_streak = pass ? convergence_streak + 1 : 0
        score = convergence === :worst_region ?
                max(finalres, worst / max(1, abs(F))) : finalres
        if score < bestscore && !isnothing(prevF)
            bestscore, bestenv = score, env
        end
        responses, prevF, prevterms = nextresponses, F, terms
        if sweep >= 5 && (convergence === :free_energy ? pass : convergence_streak >= 2)
            converged = true
            break
        end
    end
    if convergence !== :free_energy && !isnothing(bestenv)
        finalenv = bestenv
    end
    isnothing(finalenv) && return nothing
    if !converged
        warn_unconverged && @warn "Local cycle CTMRG did not converge to tolerance $tolerance after $maxiter sweeps (response residual=$finalres)."
        require_converged && return nothing
    end
    return _ctm_setenv(cache, finalenv)
end


"""Fuse a real PEPS site `(physical,right,up,left,down)` into its rank-four norm factor."""
function _ctm_local_double_layer(t::AbstractArray{T, 5}) where {T<:Real}
    p, q0, q1, q2, q3 = size(t)
    tf = reshape(t, p, :)
    E = reshape(transpose(tf) * tf, q0, q1, q2, q3, q0, q1, q2, q3)
    E = permutedims(E, (1, 5, 2, 6, 3, 7, 4, 8))
    return reshape(E, q0^2, q1^2, q2^2, q3^2)
end
