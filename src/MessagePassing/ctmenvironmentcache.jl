# Finite directional CTMRG over a grid-structured TensorNetwork.
#
# Rows of the grid are absorbed one at a time; each horizontal bond is truncated to
# `maxdim` by a biorthogonal projector pair built from BOTH half-environment ("corner")
# density matrices — always via a Hermitian EIGENDECOMPOSITION (Arnoldi/Lanczos where it
# pays), never an SVD. Works for anisotropic / non-square grids and free boundaries.
# `partitionfunction` / `freenergy` contract the whole network; `row_environments` +
# `contract_row` expose the top/bottom environments for row-local observables.
#
# Double-layer networks (⟨ψ|ψ⟩, ⟨ψ|O|ψ⟩) are handled LAZILY, as in the boundary-MPS
# backend: a vertex's factors stay a `Vector{ITensor}` ([ket, bra], or [ket, op, bra]) and
# the environment tensors keep their inward ket and bra legs separate (dimension D each,
# never fused to D²). Each absorption contracts the flat list [environment; factors…] in a
# netcon-optimal order, so the fat ket⊗bra site tensor is never materialised.
#
# Validated against exact contraction and brute force in examples/ctm_finite_aniso.jl
# (free energy: exact + χ-convergent) and examples/ctm_observable.jl (⟨sᵢsⱼ⟩ incl.
# boundary bonds, ~1e-11).

using LinearAlgebra: eigen, Hermitian, diagm, norm, I, Diagonal
using KrylovKit: eigsolve

"""
    CTMEnvironmentCache(tn::AbstractTensorNetwork, maxdim::Integer)

Directional-CTMRG environment for a 2D grid `TensorNetwork` (vertices `(x, y)`).
Contract it with [`partitionfunction`](@ref); bond truncation to `maxdim` uses the
eigendecomposition of the accumulated corner density matrix.
"""
struct CTMEnvironmentCache{V, N}
    network::N
    rows::Vector{Vector{V}}     # grid vertices grouped into rows, sorted within each row
    maxdim::Int
end

network(cache::CTMEnvironmentCache) = cache.network
graph(cache::CTMEnvironmentCache) = graph(network(cache))

# Works for a single-layer `TensorNetwork`, a `TensorNetworkState` (⟨ψ|ψ⟩) or an
# `AbstractForm` (⟨ψ|O|ψ⟩) — all of them expose their per-vertex tensors through
# `bp_factors`, which is how the double layer is kept LAZY (see `_ctm_row`).
function CTMEnvironmentCache(net, maxdim::Integer)
    vs = collect(vertices(graph(net)))
    all(v -> (v isa Tuple || v isa CartesianIndex) && length(v) == 2, vs) ||
        error("CTMEnvironmentCache requires a 2D grid network (vertices as (x, y)).")
    ys = sort(unique(last.(vs)))
    rows = [sort(filter(v -> last(v) == y, vs); by = first) for y in ys]
    allequal(length.(rows)) || error("CTMEnvironmentCache requires a rectangular grid.")
    return CTMEnvironmentCache(net, rows, Int(maxdim))
end

# --- the move --------------------------------------------------------------------
# eig projector from a density matrix ρ(bnd, bnd'): the top-`maxdim` eigenvectors.
# Relative cutoff gap below which the truncation is judged to split a (near-)degenerate
# multiplet; the cut is then backed off to a real gap. Matters for DOUBLE-LAYER networks,
# whose corner spectra carry systematic 2-fold degeneracies from ket↔bra exchange
# (λ_ij = λ_ji). 0 disables it.
const CTM_DEGTOL = Ref(0.0)
# Spectrum of the last/largest corner solved, for diagnostics (relative to λ₁).
const CTM_SPECTRUM = Ref(Float64[])

# Only the top `maxdim` eigenpairs are needed, so for a corner much larger than `maxdim`
# an Arnoldi/Lanczos solve (KrylovKit) costs O(maxdim·n²) instead of dense eigen's O(n³).
# Set to `false` to force dense. Falls back to dense if Krylov does not converge.
const CTM_ARNOLDI = Ref(true)

# Build each bond's projector from BOTH half-environments (biorthogonal pair) rather than
# from the accumulated left block alone. See `_ctm_sweep_twosided!`. A one-sided cut is not
# a well-defined variational choice and makes the error non-monotonic in `maxdim`; set this
# to `false` only to reproduce that older behaviour for comparison.
const CTM_TWOSIDED = Ref(true)

# Relative cutoff on the S values inverted by the biorthogonal projector. Those inverse
# powers amplify roundoff, so tiny directions must be dropped or they impose a hard,
# χ-independent error floor. 1e-8 ≈ √eps is the right scale (not 1e-12): the projector gets
# S from a Hermitian eig of a squared object, which resolves it to only ~√eps relatively.
# Measured optimum — 6×6 Ising: 5.1e-13 here vs 3.1e-10 at 1e-12 and 3.4e-9 at 1e-4.
const CTM_PINV_CUTOFF = Ref(1.0e-8)

# Top-`k` eigenpairs of a Hermitian matrix: Krylov when it pays off, else dense.
function _ctm_eigsolve(ρs::Hermitian, k::Integer)
    n = size(ρs, 1)
    if CTM_ARNOLDI[] && n > 4k
        try
            v0 = randn(eltype(ρs), n)
            vals, vecs, info = eigsolve(x -> ρs * x, v0, k, :LR;
                                        ishermitian = true, krylovdim = max(2k + 8, 20))
            if info.converged >= k && length(vecs) >= k
                V = reduce(hcat, @view(vecs[1:k]))
                eltype(ρs) <: Real && (V = real.(V))
                # The projector must be a genuine isometry: degenerate clusters (which
                # double-layer corners have, from ket↔bra exchange) can otherwise come back
                # non-orthonormal and silently corrupt the truncation.
                if norm(V' * V - I) < 1.0e-8
                    return real.(vals[1:k]), V
                end
            end
        catch
            # fall through to dense
        end
    end
    F = eigen(ρs)
    return F.values, F.vectors
end

function _ctm_eig_projector(ρ::ITensor, bnd::Index, maxdim::Integer)
    bp = prime(bnd)
    ρm = Array(ρ, bnd, bp)
    ρs = Hermitian((ρm + ρm') / 2)
    vals, vecs = _ctm_eigsolve(ρs, Int(maxdim))
    order = sortperm(vals; rev = true)
    sv = vals[order]
    length(sv) > length(CTM_SPECTRUM[]) && (CTM_SPECTRUM[] = abs.(sv) ./ abs(sv[1]))
    k = min(Int(maxdim), length(sv), size(vecs, 2))
    while k > 1 && k < length(sv) && abs(sv[k] - sv[k + 1]) ≤ CTM_DEGTOL[] * abs(sv[k])
        k -= 1                              # don't split a degenerate multiplet
    end
    keep = order[1:k]
    w = Index(k)
    return ITensor(vecs[:, keep], bnd, w), w, vals[keep]
end

# Contract a small flat list in a good order (netcon). This is what keeps the double layer
# lazy: the list is [environment; ket; (operator;) bra] and the optimizer interleaves them,
# so the fat ket⊗bra site tensor (legs of dimension D²) is never formed.
function _ctm_contract(ts::Vector{ITensor})
    length(ts) == 1 && return only(ts)
    return contract(ts; sequence = contraction_sequence(ts; alg = "optimal"))
end

# Absorb `top` row into `nxt` (contract vertical bonds), combine the doubled horizontal
# bonds, then truncate each to `maxdim` via the accumulated corner density matrix (eig).
# `nxt` holds one *group* of factors per column (2 tensors for a state norm, 3 for a form,
# 1 for a single-layer network); the environment tensors in `top` keep their downward ket
# and bra legs SEPARATE (dimension D each), exactly like a boundary-MPS message.
function _ctm_absorb_row(top::Union{Nothing, Vector{ITensor}}, nxt::Vector{Vector{ITensor}},
                         maxdim::Integer)
    n = length(nxt)
    merged = ITensor[_ctm_contract(isnothing(top) ? nxt[x] : ITensor[top[x]; nxt[x]])
                     for x in 1:n]
    return _ctm_truncate_chain!(merged, maxdim)
end

# Combine the doubled horizontal bonds, then truncate each bond left-to-right to `maxdim`.
#
# The accumulated corner ("left block") is carried as an UPPER-TRIANGULAR factor `Rc` with
# Rc†Rc = the corner density matrix, rather than as the density matrix itself. The
# projector is then the leading right singular vectors of the triangular factor — the same
# subspace the density matrix's top eigenvectors would give, but obtained without ever
# forming ρ = M†ρM, which squares the condition number and loses half the available
# precision. This is what fixes the non-monotonic-in-χ error of the ρ-based sweep.
function _ctm_truncate_chain!(merged::Vector{ITensor}, maxdim::Integer)
    n = length(merged)
    for x in 1:(n - 1)
        shared = commoninds(merged[x], merged[x + 1])
        if length(shared) > 1
            C = combiner(shared...)
            merged[x] = merged[x] * C
            merged[x + 1] = merged[x + 1] * C
        end
    end
    CTM_TWOSIDED[] && return _ctm_sweep_twosided!(merged, maxdim)
    return _ctm_sweep_density!(merged, maxdim)
end

# --- two-sided (biorthogonal) truncation ------------------------------------------
# A one-sided cut (keep the top eigenvectors of the LEFT block alone) is not a well-defined
# variational choice and makes the error non-monotonic in χ. The proper CTMRG projector uses
# BOTH half-environments: with ρ_L = A†A and ρ_R = B†B (PSD square-root factors) and
# A·Bᵀ = U S V†, the pair
#     P_A = Bᵀ V S^{-1/2}   (bond → kept),   P_B = S^{-1/2} U† A   (kept → bond)
# satisfies A (P_A P_B) Bᵀ = A Bᵀ exactly at full rank, and truncating S is the optimal
# rank-χ choice for the whole contraction rather than for the left block only.
function _ctm_sweep_twosided!(merged::Vector{ITensor}, maxdim::Integer)
    n = length(merged)
    n < 2 && return merged
    bonds = Index[]
    for x in 1:(n - 1)
        b = commonind(merged[x], merged[x + 1])
        isnothing(b) && return _ctm_sweep_density!(merged, maxdim)   # not a simple chain
        push!(bonds, b)
    end
    # Right half-environment density matrices, from the still-untouched right part.
    ρR = Vector{ITensor}(undef, n - 1)
    accR = nothing
    for x in (n - 1):-1:1
        M = merged[x + 1]
        pr = x + 1 <= n - 1 ? Index[bonds[x], bonds[x + 1]] : Index[bonds[x]]
        Mp = prime(dag(M), pr...)
        accR = isnothing(accR) ? M * Mp : accR * M * Mp
        ρR[x] = accR
    end
    ρL = nothing
    for x in 1:(n - 1)
        b = bonds[x]
        M = merged[x]
        isnothing(ρL) && (ρL = M * prime(dag(M), b))
        PA, PB, w = _ctm_twosided_projector(ρL, ρR[x], b, maxdim)
        merged[x] = M * PA
        merged[x + 1] = merged[x + 1] * PB
        if x < n - 1                                   # advance the left environment
            ρLp = ρL * PA * prime(dag(PA))
            Mn = merged[x + 1]
            ρL = ρLp * Mn * prime(dag(Mn), bonds[x + 1], w)
        end
    end
    return merged
end

# PSD square-root factor: returns A with A†A = ρ (rows = rank, cols = bond).
function _ctm_psd_factor(ρm::AbstractMatrix)
    F = eigen(Hermitian((ρm + ρm') / 2))
    λ = max.(real.(F.values), zero(real(eltype(ρm))))
    tol = 1.0e-30 * (isempty(λ) ? one(eltype(λ)) : maximum(λ))
    keep = findall(>(tol), λ)
    isempty(keep) && (keep = [argmax(λ)])
    return Diagonal(sqrt.(λ[keep])) * F.vectors[:, keep]'
end

# Built from a single HERMITIAN EIGENDECOMPOSITION, no SVD: with ρ_L = A†A and ρ_R = B†B,
# H = B ρ_L Bᵀ is Hermitian PSD with eigenvalues S² and eigenvectors V, giving
#     P_A = Bᵀ V S^{-1/2},    P_B = S^{-3/2} Vᵀ B ρ_L
# and A(P_A P_B)Bᵀ = A Bᵀ exactly at full rank. Taking both from ONE eigenbasis also avoids
# the sign/phase mismatch (and arbitrary rotation inside degenerate clusters) that separate
# decompositions for U and V would introduce.
function _ctm_twosided_projector(ρL::ITensor, ρR::ITensor, b::Index, maxdim::Integer)
    bp = prime(b)
    ρLm = Array(ρL, b, bp); ρLm = (ρLm + ρLm') / 2
    B = _ctm_psd_factor(Array(ρR, b, bp))
    H = Hermitian((H0 = B * ρLm * B'; (H0 + H0') / 2))
    λ, V = _ctm_eigsolve(H, Int(maxdim))
    ord = sortperm(real.(λ); rev = true)
    S = sqrt.(max.(real.(λ[ord]), zero(real(eltype(ρLm)))))
    # S^{-1/2}/S^{-3/2} amplify roundoff in small values, which puts a hard floor on the
    # achievable error, so tiny directions must be dropped (not merely the null ones).
    k = min(Int(maxdim), length(S), size(V, 2))
    while k > 1 && S[k] ≤ CTM_PINV_CUTOFF[] * S[1]
        k -= 1
    end
    while k > 1 && k < length(S) && abs(S[k] - S[k + 1]) ≤ CTM_DEGTOL[] * abs(S[k])
        k -= 1                                      # don't split a degenerate multiplet
    end
    Vk = V[:, ord[1:k]]; Sk = S[1:k]
    PAm = B' * Vk * Diagonal(1 ./ sqrt.(Sk))        # (bond × kept)
    PBm = Diagonal(Sk .^ (-3 / 2)) * Vk' * B * ρLm  # (kept × bond)
    w = Index(k)
    return ITensor(PAm, b, w), ITensor(PBm, w, b), w
end

# Truncate via the upper-triangular corner factor (default — numerically stable).

# Top-`maxdim` eigenvectors of ρ = R†R for an upper-triangular factor R with legs
# (qlink, bnd), returned as a projector on `bnd`. Same index convention as
# `_ctm_eig_projector`, so it is interchangeable with the density-matrix sweep.

# Eigenpairs of R†R without forming it: Arnoldi on x ↦ R†(R x), else a dense SVD of R
# (whose right singular vectors are exactly those eigenvectors — still no squaring).

# Truncate via the accumulated corner DENSITY MATRIX and its eigendecomposition (which may
# use Arnoldi, see `_ctm_eigsolve`). Kept for comparison: forming ρ squares the condition
# number, which makes the error non-monotonic in `maxdim` on double-layer networks.
function _ctm_sweep_density!(merged::Vector{ITensor}, maxdim::Integer)
    n = length(merged)
    ρL = ITensor(one(ITensors.NDTensors.scalartype(merged[1])))
    for x in 1:(n - 1)
        bnd = commonind(merged[x], merged[x + 1])
        isnothing(bnd) && continue
        M = merged[x]
        lb = commonind(M, ρL)
        Mp = isnothing(lb) ? prime(dag(M), bnd) : prime(dag(M), bnd, lb)
        ρext = ρL * M * Mp
        P, w, λ = _ctm_eig_projector(ρext, bnd, maxdim)
        merged[x] = M * P
        merged[x + 1] = merged[x + 1] * dag(P)
        ρL = ITensor(diagm(λ), w, prime(w))
    end
    return merged
end

# Row `y` as one *group* of factors per column — unfolded, so the double layer stays lazy.
_ctm_row(cache::CTMEnvironmentCache, y::Integer) =
    Vector{ITensor}[Vector{ITensor}(bp_factors(network(cache), v)) for v in cache.rows[y]]

# --- partition function / free energy --------------------------------------------
function partitionfunction(cache::CTMEnvironmentCache)
    cur = _ctm_absorb_row(nothing, _ctm_row(cache, 1), cache.maxdim)
    for y in 2:length(cache.rows)
        cur = _ctm_absorb_row(cur, _ctm_row(cache, y), cache.maxdim)
    end
    return scalar(reduce(*, cur))
end

freenergy(cache::CTMEnvironmentCache) = log(partitionfunction(cache))

# =================================================================================
# Per-vertex CVM environments: a 4C+4T ring on EVERY vertex.
#
#   C[:NW,x,y] = all vertices with col<x, row<y      (likewise :NE :SW :SE)
#   T[:N,x,y]  = column x, rows<y                    T[:S,x,y] = column x, rows≥y
#   T[:W,x,y]  = cols<x, row y                       T[:E,x,y] = cols≥x, row y
#
# Corners are GROWN with their two adjoining edge tensors and the vertex tensor,
#   C̃_NW(x+1,y+1) = C_NW(x,y) · T_N(x,y) · T_W(x,y) · a(x,y),
# and the two open interfaces of C̃ are then PROJECTED. Each interface is shared by
# several blocks, so its projector must be a single object — derived once, consumed
# elsewhere. Interface families, each a nested chain of isometries:
#
#   PH[:N,x,y] : horizontal links at column x, rows<y   (C_NW.right, C_NE.left, T_N sides)
#   PH[:S,x,y] : horizontal links at column x, rows≥y   (C_SW.right, C_SE.left, T_S sides)
#   PV[:W,x,y] : vertical links at row y, cols<x        (C_NW.down, C_SW.up, T_W sides)
#   PV[:E,x,y] : vertical links at row y, cols≥x        (C_NE.down, C_SE.up, T_E sides)
#
# See docs/finite_ctmrg_design.md.
struct CTMVertexEnvironments
    C::Dict{Tuple{Symbol, Int, Int}, Any}
    T::Dict{Tuple{Symbol, Int, Int}, Any}
    PH::Dict{Tuple{Symbol, Int, Int}, Any}
    PV::Dict{Tuple{Symbol, Int, Int}, Any}
    Lx::Int
    Ly::Int
end

_ctm_nn(d, k) = get(d, k, nothing)
_ctm_mul(a, b) = isnothing(a) ? b : (isnothing(b) ? a : _ctm_contract(ITensor[a, b]))
_ctm_widx(d, k) = (t = get(d, k, nothing); isnothing(t) ? nothing : t[2])

# Isometry truncating index set `ins` of block `B` to `maxdim`, from the eigendecomposition
# of B's reduced density matrix on those indices. Returns (P, w) with P legs (ins…, w).
function _ctm_interface_proj(B, ins::Vector{<:Index}, maxdim::Integer)
    (isnothing(B) || isempty(ins)) && return nothing
    co = combiner(ins...); io = combinedind(co)
    d = ITensors.dim(io); k = min(Int(maxdim), d)
    if k == d                              # nothing to truncate: keep the basis intact
        w = Index(d)
        return ITensor(Matrix{Float64}(I, d, d), io, w) * co, w
    end
    Bc = B * co
    ρ = Bc * prime(dag(Bc), io)
    P, w, _ = _ctm_eig_projector(ρ, io, k)
    return P * co, w
end

# Grid geometry / lazy factors ----------------------------------------------------
_ctm_dims(cache::CTMEnvironmentCache) = (length(first(cache.rows)), length(cache.rows))
_ctm_vertex(cache::CTMEnvironmentCache, x::Int, y::Int) = cache.rows[y][x]

function _ctm_factor_table(cache::CTMEnvironmentCache)
    Lx, Ly = _ctm_dims(cache)
    tbl = Dict{Tuple{Int, Int}, Vector{ITensor}}()
    for y in 1:Ly, x in 1:Lx
        tbl[(x, y)] = Vector{ITensor}(bp_factors(network(cache), _ctm_vertex(cache, x, y)))
    end
    return tbl
end

# Links between neighbouring vertices: ONE index for a single layer, TWO (ket+bra) for a
# double layer — discovered from the tensors, never fused.
function _ctm_links(tbl, a::Tuple{Int, Int}, b::Tuple{Int, Int})
    is = Index[]
    for t1 in tbl[a], t2 in tbl[b]
        append!(is, commoninds(t1, t2))
    end
    return unique(is)
end

"""
    vertex_environments(cache::CTMEnvironmentCache)

Build the position-resolved corner/edge environments (a 4C+4T ring on every vertex) by
growing corners with their adjoining edge tensors and projecting each shared interface.
Feeds [`region_lnZ`](@ref) and the CVM free energy.
"""
function vertex_environments(cache::CTMEnvironmentCache)
    Lx, Ly = _ctm_dims(cache)
    χ = cache.maxdim
    tbl = _ctm_factor_table(cache)
    hl(x, y) = _ctm_links(tbl, (x, y), (x + 1, y))      # horizontal link cols x|x+1 at row y
    vl(x, y) = _ctm_links(tbl, (x, y), (x, y + 1))      # vertical link rows y|y+1 at col x
    a(x, y) = tbl[(x, y)]

    C = Dict{Tuple{Symbol, Int, Int}, Any}()
    T = Dict{Tuple{Symbol, Int, Int}, Any}()
    PH = Dict{Tuple{Symbol, Int, Int}, Any}()
    PV = Dict{Tuple{Symbol, Int, Int}, Any}()

    # ---- W strips (y increasing, x increasing): derives PV[:W] ----
    for y in 1:Ly, x in 1:(Lx - 1)
        raw = _ctm_mul(_ctm_nn(T, (:W, x, y)), _ctm_contract(a(x, y)))
        if y > 1
            P = _ctm_nn(PV, (:W, x + 1, y - 1))
            !isnothing(P) && (raw = raw * P[1])
        end
        if y < Ly
            ins = Index[]
            w = _ctm_widx(PV, (:W, x, y)); !isnothing(w) && push!(ins, w)
            append!(ins, vl(x, y))
            pr = _ctm_interface_proj(raw, ins, χ)
            if !isnothing(pr)
                PV[(:W, x + 1, y)] = pr
                raw = raw * pr[1]
            end
        end
        T[(:W, x + 1, y)] = raw
    end
    # ---- E strips (x decreasing): derives PV[:E] ----
    for y in 1:Ly, x in Lx:-1:2
        raw = _ctm_mul(_ctm_contract(a(x, y)), _ctm_nn(T, (:E, x + 1, y)))
        if y > 1
            P = _ctm_nn(PV, (:E, x, y - 1))
            !isnothing(P) && (raw = raw * P[1])
        end
        if y < Ly
            ins = Index[]
            append!(ins, vl(x, y))
            w = _ctm_widx(PV, (:E, x + 1, y)); !isnothing(w) && push!(ins, w)
            pr = _ctm_interface_proj(raw, ins, χ)
            if !isnothing(pr)
                PV[(:E, x, y)] = pr
                raw = raw * pr[1]
            end
        end
        T[(:E, x, y)] = raw
    end
    # ---- C[:NW] (y increasing): derives PH[:N] ----
    for x in 2:Lx, y in 1:(Ly - 1)
        raw = _ctm_mul(_ctm_nn(C, (:NW, x, y)), _ctm_nn(T, (:W, x, y)))
        ins = Index[]
        w = _ctm_widx(PH, (:N, x - 1, y)); !isnothing(w) && push!(ins, w)
        append!(ins, hl(x - 1, y))
        pr = _ctm_interface_proj(raw, ins, χ)
        if !isnothing(pr)
            PH[(:N, x - 1, y + 1)] = pr
            raw = raw * pr[1]
        end
        C[(:NW, x, y + 1)] = raw
    end
    # ---- C[:SW] (y decreasing): derives PH[:S] ----
    for x in 2:Lx, y in Ly:-1:2
        raw = _ctm_mul(_ctm_nn(C, (:SW, x, y + 1)), _ctm_nn(T, (:W, x, y)))
        ins = Index[]
        append!(ins, hl(x - 1, y))
        w = _ctm_widx(PH, (:S, x - 1, y + 1)); !isnothing(w) && push!(ins, w)
        pr = _ctm_interface_proj(raw, ins, χ)
        if !isnothing(pr)
            PH[(:S, x - 1, y)] = pr
            raw = raw * pr[1]
        end
        C[(:SW, x, y)] = raw
    end
    # ---- C[:NE] / C[:SE]: consume PH ----
    for x in 2:Lx
        for y in 1:(Ly - 1)
            raw = _ctm_mul(_ctm_nn(C, (:NE, x, y)), _ctm_nn(T, (:E, x, y)))
            P = _ctm_nn(PH, (:N, x - 1, y + 1))
            !isnothing(P) && (raw = raw * dag(P[1]))
            C[(:NE, x, y + 1)] = raw
        end
        for y in Ly:-1:2
            raw = _ctm_mul(_ctm_nn(C, (:SE, x, y + 1)), _ctm_nn(T, (:E, x, y)))
            P = _ctm_nn(PH, (:S, x - 1, y))
            !isnothing(P) && (raw = raw * dag(P[1]))
            C[(:SE, x, y)] = raw
        end
    end
    # ---- N / S column strips: consume PH ----
    for x in 1:Lx
        for y in 1:(Ly - 1)
            raw = _ctm_mul(_ctm_nn(T, (:N, x, y)), _ctm_contract(a(x, y)))
            P = _ctm_nn(PH, (:N, x - 1, y + 1)); !isnothing(P) && (raw = raw * dag(P[1]))
            Q = _ctm_nn(PH, (:N, x, y + 1));     !isnothing(Q) && (raw = raw * Q[1])
            T[(:N, x, y + 1)] = raw
        end
        for y in Ly:-1:2
            raw = _ctm_mul(_ctm_contract(a(x, y)), _ctm_nn(T, (:S, x, y + 1)))
            P = _ctm_nn(PH, (:S, x - 1, y)); !isnothing(P) && (raw = raw * dag(P[1]))
            Q = _ctm_nn(PH, (:S, x, y));     !isnothing(Q) && (raw = raw * Q[1])
            T[(:S, x, y)] = raw
        end
    end
    return CTMVertexEnvironments(C, T, PH, PV, Lx, Ly)
end

# Biorthogonal (two-sided) projector pair for the interface shared by two complementary
# enlarged corners. Returns (P_A, P_B, w): P_A goes on the west/north block, P_B on the
# east/south one, so every contraction across the interface pairs one with the other.
function _ctm_interface_proj2(Bw, Be, ins::Vector{<:Index}, maxdim::Integer)
    (isnothing(Bw) || isnothing(Be) || isempty(ins)) && return nothing
    co = combiner(ins...); io = combinedind(co)
    Bwc = Bw * co; Bec = Be * co
    ρL = Bwc * prime(dag(Bwc), io)
    ρR = Bec * prime(dag(Bec), io)
    PA, PB, w = _ctm_twosided_projector(ρL, ρR, io, maxdim)
    return PA * co, PB * co, w
end

# Enlarged corner: the quadrant cut at (x,y), grown one vertex out of the PREVIOUS state's
# blocks (so all indices are in a consistent basis) with its two adjoining edges and vertex.
function _ctm_enlarged(S::CTMVertexEnvironments, tbl, sym::Symbol, x::Int, y::Int)
    A(i, j) = (haskey(tbl, (i, j)) ? _ctm_contract(tbl[(i, j)]) : nothing)
    m4(a, b, c, d) = _ctm_mul(_ctm_mul(_ctm_mul(a, b), c), d)
    if sym === :NW          # cols<x, rows<y  — grown from vertex (x-1, y-1)
        return m4(_ctm_nn(S.C, (:NW, x - 1, y - 1)), _ctm_nn(S.T, (:N, x - 1, y - 1)),
                  _ctm_nn(S.T, (:W, x - 1, y - 1)), A(x - 1, y - 1))
    elseif sym === :NE      # cols≥x, rows<y  — grown from vertex (x, y-1)
        return m4(_ctm_nn(S.C, (:NE, x + 1, y - 1)), _ctm_nn(S.T, (:N, x, y - 1)),
                  _ctm_nn(S.T, (:E, x + 1, y - 1)), A(x, y - 1))
    elseif sym === :SW      # cols<x, rows≥y  — grown from vertex (x-1, y)
        return m4(_ctm_nn(S.C, (:SW, x - 1, y + 1)), _ctm_nn(S.T, (:S, x - 1, y + 1)),
                  _ctm_nn(S.T, (:W, x - 1, y)), A(x - 1, y))
    else                    # :SE  cols≥x, rows≥y — grown from vertex (x, y)
        return m4(_ctm_nn(S.C, (:SE, x + 1, y + 1)), _ctm_nn(S.T, (:S, x, y + 1)),
                  _ctm_nn(S.T, (:E, x + 1, y)), A(x, y))
    end
end

"""
    sweep_vertex_environments(cache, S) -> CTMVertexEnvironments

!!! warning "Work in progress — does not yet produce closable regions"
    An interface is being projected on one side but not the other, so raw links survive and
    `region_lnZ` fails with leftover indices. `vertex_environments` (the greedy single pass)
    is unaffected. See the "Open bug" section of `docs/finite_ctmrg_design.md` for the
    symptom, the three suspected causes and the suggested diff-against-the-greedy-builder
    diagnosis.

One pass round the lattice, vertex to vertex: at each cut, grow the four enlarged corners out
of `S`, take a TWO-SIDED (biorthogonal) projector for each interface from the two corners that
bound it, and rebuild the corners and edges with it. Interfaces must be projected at growth,
when they are χ·D dimensional — re-projecting an already-truncated interface is a no-op — so a
sweep regrows rather than refines. Iterate until [`cvm_freenergy`](@ref) stops moving.
"""
function sweep_vertex_environments(cache::CTMEnvironmentCache, S::CTMVertexEnvironments)
    Lx, Ly = S.Lx, S.Ly
    χ = cache.maxdim
    tbl = _ctm_factor_table(cache)
    C = Dict{Tuple{Symbol, Int, Int}, Any}()
    T = Dict{Tuple{Symbol, Int, Int}, Any}()
    PH = Dict{Tuple{Symbol, Int, Int}, Any}()
    PV = Dict{Tuple{Symbol, Int, Int}, Any}()
    enl = Dict{Tuple{Symbol, Int, Int}, Any}()
    E(sym, x, y) = get!(enl, (sym, x, y)) do
        _ctm_enlarged(S, tbl, sym, x, y)
    end
    # --- derive every interface projector pair from its two bounding corners -------
    for x in 1:(Lx - 1), y in 2:Ly            # PH[:N,x,y]: C_NW(x+1,y) | C_NE(x+1,y)
        Bw = E(:NW, x + 1, y); Be = E(:NE, x + 1, y)
        (isnothing(Bw) || isnothing(Be)) && continue
        pr = _ctm_interface_proj2(Bw, Be, commoninds(Bw, Be), χ)
        !isnothing(pr) && (PH[(:N, x, y)] = pr)
    end
    for x in 1:(Lx - 1), y in 1:(Ly - 1)      # PH[:S,x,y]: C_SW(x+1,y) | C_SE(x+1,y)
        Bw = E(:SW, x + 1, y); Be = E(:SE, x + 1, y)
        (isnothing(Bw) || isnothing(Be)) && continue
        pr = _ctm_interface_proj2(Bw, Be, commoninds(Bw, Be), χ)
        !isnothing(pr) && (PH[(:S, x, y)] = pr)
    end
    for x in 2:Lx, y in 1:(Ly - 1)            # PV[:W,x,y]: C_NW(x,y+1) | C_SW(x,y+1)
        Bn = E(:NW, x, y + 1); Bs = E(:SW, x, y + 1)
        (isnothing(Bn) || isnothing(Bs)) && continue
        pr = _ctm_interface_proj2(Bn, Bs, commoninds(Bn, Bs), χ)
        !isnothing(pr) && (PV[(:W, x, y)] = pr)
    end
    for x in 1:(Lx - 1), y in 1:(Ly - 1)      # PV[:E,x,y]: C_NE(x,y+1) | C_SE(x,y+1)
        Bn = E(:NE, x + 1, y + 1); Bs = E(:SE, x + 1, y + 1)
        (isnothing(Bn) || isnothing(Bs)) && continue
        pr = _ctm_interface_proj2(Bn, Bs, commoninds(Bn, Bs), χ)
        !isnothing(pr) && (PV[(:E, x + 1, y)] = pr)
    end
    # --- rebuild corners: P_A on the west/north side, P_B on the east/south side ----
    apA(t, pr) = isnothing(pr) || isnothing(t) ? t : t * pr[1]
    apB(t, pr) = isnothing(pr) || isnothing(t) ? t : t * pr[2]
    for x in 2:Lx, y in 2:Ly
        C[(:NW, x, y)] = apA(apA(E(:NW, x, y), _ctm_nn(PH, (:N, x - 1, y))),
                             _ctm_nn(PV, (:W, x, y - 1)))
    end
    for x in 1:(Lx - 1), y in 2:Ly
        C[(:NE, x + 1, y)] = apA(apB(E(:NE, x + 1, y), _ctm_nn(PH, (:N, x, y))),
                                 _ctm_nn(PV, (:E, x + 1, y - 1)))
    end
    for x in 2:Lx, y in 1:(Ly - 1)
        C[(:SW, x, y)] = apB(apA(E(:SW, x, y), _ctm_nn(PH, (:S, x - 1, y))),
                             _ctm_nn(PV, (:W, x, y - 1)))
    end
    for x in 1:(Lx - 1), y in 1:(Ly - 1)
        C[(:SE, x + 1, y)] = apB(apB(E(:SE, x + 1, y), _ctm_nn(PH, (:S, x, y))),
                                 _ctm_nn(PV, (:E, x + 1, y - 1)))
    end
    # --- rebuild edges from the previous state, projected on both sides -------------
    for x in 1:Lx, y in 2:Ly                  # T_N: left = east side, right = west side
        raw = _ctm_mul(_ctm_nn(S.T, (:N, x, y - 1)), _ctm_contract(tbl[(x, y - 1)]))
        T[(:N, x, y)] = apA(apB(raw, _ctm_nn(PH, (:N, x - 1, y))), _ctm_nn(PH, (:N, x, y)))
    end
    for x in 1:Lx, y in 1:(Ly - 1)            # T_S
        raw = _ctm_mul(_ctm_contract(tbl[(x, y)]), _ctm_nn(S.T, (:S, x, y + 1)))
        T[(:S, x, y)] = apA(apB(raw, _ctm_nn(PH, (:S, x - 1, y))), _ctm_nn(PH, (:S, x, y)))
    end
    for x in 2:Lx, y in 1:Ly                  # T_W: up = south side, down = north side
        raw = _ctm_mul(_ctm_nn(S.T, (:W, x - 1, y)), _ctm_contract(tbl[(x - 1, y)]))
        T[(:W, x, y)] = apA(apB(raw, _ctm_nn(PV, (:W, x, y - 1))), _ctm_nn(PV, (:W, x, y)))
    end
    for x in 1:(Lx - 1), y in 1:Ly            # T_E
        raw = _ctm_mul(_ctm_contract(tbl[(x + 1, y)]), _ctm_nn(S.T, (:E, x + 2, y)))
        T[(:E, x + 1, y)] = apA(apB(raw, _ctm_nn(PV, (:E, x + 1, y - 1))),
                                _ctm_nn(PV, (:E, x + 1, y)))
    end
    return CTMVertexEnvironments(C, T, PH, PV, Lx, Ly)
end

"""
    region_lnZ(env::CTMVertexEnvironments, cache, cx, cy)

Region free energy `ln Z_R`. Integer `(cx,cy)` → vertex ring (4C+4T+a); half-integer in one
axis → edge strip (4C+2T); both half-integer → plaquette loop (4C).
"""
function region_lnZ(env::CTMVertexEnvironments, cache::CTMEnvironmentCache, cx::Real, cy::Real)
    Lx, Ly = env.Lx, env.Ly
    rL = ceil(Int, cx); rR = floor(Int, cx) + 1
    tT = ceil(Int, cy); tB = floor(Int, cy) + 1
    xint = rL < rR; yint = tT < tB
    ts = Any[_ctm_nn(env.C, (:NW, rL, tT)), _ctm_nn(env.C, (:NE, rR, tT)),
             _ctm_nn(env.C, (:SW, rL, tB)), _ctm_nn(env.C, (:SE, rR, tB))]
    if xint
        push!(ts, _ctm_nn(env.T, (:N, Int(cx), tT)))
        push!(ts, _ctm_nn(env.T, (:S, Int(cx), tB)))
    end
    if yint
        push!(ts, _ctm_nn(env.T, (:W, rL, Int(cy))))
        push!(ts, _ctm_nn(env.T, (:E, rR, Int(cy))))
    end
    if xint && yint
        append!(ts, bp_factors(network(cache), _ctm_vertex(cache, Int(cx), Int(cy))))
    end
    keep = ITensor[t for t in ts if !isnothing(t)]
    isempty(keep) && return 0.0
    return log(abs(real(scalar(_ctm_contract(keep)))))
end

"""
    cvm_freenergy(cache::CTMEnvironmentCache)

Region-graph (CVM) free energy `F = Σ_v ln Z_v − Σ_e ln Z_e + Σ_p ln Z_p`. Exact when the
environments are lossless, since `V − E + P = 1` for a disk.
"""
function cvm_freenergy(cache::CTMEnvironmentCache; env = vertex_environments(cache))
    Lx, Ly = env.Lx, env.Ly
    F = 0.0
    for x in 1:Lx, y in 1:Ly;           F += region_lnZ(env, cache, x, y);             end
    for x in 1:(Lx - 1), y in 1:Ly;     F -= region_lnZ(env, cache, x + 0.5, y);       end
    for x in 1:Lx, y in 1:(Ly - 1);     F -= region_lnZ(env, cache, x, y + 0.5);       end
    for x in 1:(Lx - 1), y in 1:(Ly - 1); F += region_lnZ(env, cache, x + 0.5, y + 0.5); end
    return F
end

# --- row-local observables -------------------------------------------------------
# Top / bottom environments around row `y` (each `nothing` at the boundary).
function row_environments(cache::CTMEnvironmentCache, y::Integer)
    Ly = length(cache.rows)
    top = y == 1 ? nothing :
        foldl((c, yy) -> _ctm_absorb_row(c, _ctm_row(cache, yy), cache.maxdim),
              2:(y - 1); init = _ctm_absorb_row(nothing, _ctm_row(cache, 1), cache.maxdim))
    bot = y == Ly ? nothing :
        foldl((c, yy) -> _ctm_absorb_row(c, _ctm_row(cache, yy), cache.maxdim),
              (Ly - 1):-1:(y + 1); init = _ctm_absorb_row(nothing, _ctm_row(cache, Ly), cache.maxdim))
    return top, bot
end

# Sandwich (top env) · (row tensors) · (bottom env) → scalar. Insert operator-weighted
# tensors into `rowts` (sharing the network's bond indices) to build an expectation.
function contract_row(top, rowts::Vector{<:ITensor}, bot)
    merged = ITensor[]
    for x in eachindex(rowts)
        t = rowts[x]
        top !== nothing && (t = t * top[x])
        bot !== nothing && (t = t * bot[x])
        push!(merged, t)
    end
    return scalar(reduce(*, merged))
end
