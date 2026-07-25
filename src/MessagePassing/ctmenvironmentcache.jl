# Finite, position-resolved CTMRG over a grid-structured TensorNetwork, framed as a
# region-graph (CVM) free energy. This is NOT boundary MPS and is intended to supersede it:
# every vertex carries its own 4C+4T environment ring, grown and projected by LOCAL corner
# moves. There is no row absorption and no whole-lattice chain anywhere in here.
#
#   C[:NW,x,y] = all vertices with col<x, row<y      (likewise :NE :SW :SE)
#   T[:N,x,y]  = column x, rows<y                    T[:S,x,y] = column x, rows≥y
#   T[:W,x,y]  = cols<x, row y                       T[:E,x,y] = cols≥x, row y
#
#   F = Σ_v ln Z_v − Σ_e ln Z_e + Σ_p ln Z_p     (Möbius numbers +1 / −1 / +1)
#
# Each shared interface is truncated to `maxdim` by a biorthogonal projector PAIR built from
# BOTH bounding corners. The default route uses a Hermitian EIGENDECOMPOSITION (Arnoldi/Lanczos
# where it pays) of the corner density matrices; `CTM_QR` selects an equivalent triangular/QR
# route that never squares and batches better on GPU (accuracy-neutral — see its comment). The
# pair needs the complement environment, so the build is a fixed-point iteration: `update` sweeps
# it to stationarity. Works for anisotropic / non-square grids and free boundaries.
#
# Entry points: `update` (run it), `cvm_freenergy` (ln Z), `vertex_ring` / `expect` / `rdm`
# (single-site observables from a vertex's own ring).
#
# Double-layer networks (⟨ψ|ψ⟩, ⟨ψ|O|ψ⟩) are handled LAZILY: a vertex's factors stay a
# `Vector{ITensor}` ([ket, bra], or [ket, op, bra]) and the environment tensors keep their
# inward ket and bra legs separate (dimension D each, never fused to D²). Each absorption
# contracts the flat list [environment; factors…] in a netcon-optimal order, so the fat
# ket⊗bra site tensor is never materialised.
#
# See docs/finite_ctmrg_design.md for the derivations and the measured comparisons.

using LinearAlgebra: eigen, Hermitian, diagm, norm, I, Diagonal, qr, svd
using KrylovKit: eigsolve

"""
    CTMEnvironmentCache(tn::AbstractTensorNetwork, maxdim::Integer)

Position-resolved CTMRG environment for a 2D grid `TensorNetwork` (vertices `(x, y)`): a
`4C + 4T` ring on every vertex, with each shared interface truncated to `maxdim` by a two-sided
(biorthogonal) projector pair.

A freshly built cache carries **no** per-vertex CVM environments. [`update`](@ref) runs the
two-sided stationary sweep and returns a cache holding the converged ones; [`cvm_freenergy`](@ref)
and [`region_lnZ`](@ref) then read them off. As with `BeliefPropagationCache`, evaluating an
un-updated cache gives you the un-converged answer (here the greedy single pass) rather than an
error, so `update` before you trust the number.
"""
struct CTMEnvironmentCache{V, N, E}
    network::N
    grid::Vector{Vector{V}}     # vertices by grid position: grid[y][x]. Geometry only.
    maxdim::Int
    environments::E             # `nothing`, or the CVM blocks from `update`
end

network(cache::CTMEnvironmentCache) = cache.network
graph(cache::CTMEnvironmentCache) = graph(network(cache))

"""
    environments(cache::CTMEnvironmentCache)

The cache's per-vertex CVM environments, or `nothing` if it has not been [`update`](@ref)d.
"""
environments(cache::CTMEnvironmentCache) = cache.environments

# Works for a single-layer `TensorNetwork`, a `TensorNetworkState` (⟨ψ|ψ⟩) or an
# `AbstractForm` (⟨ψ|O|ψ⟩) — all of them expose their per-vertex tensors through
# `bp_factors`, which is how the double layer is kept LAZY.
function CTMEnvironmentCache(net, maxdim::Integer)
    vs = collect(vertices(graph(net)))
    all(v -> (v isa Tuple || v isa CartesianIndex) && length(v) == 2, vs) ||
        error("CTMEnvironmentCache requires a 2D grid network (vertices as (x, y)).")
    ys = sort(unique(last.(vs)))
    rows = [sort(filter(v -> last(v) == y, vs); by = first) for y in ys]
    allequal(length.(rows)) || error("CTMEnvironmentCache requires a rectangular grid.")
    return CTMEnvironmentCache(net, rows, Int(maxdim), nothing)
end

# Same network/grid/maxdim, different CVM environments.
_ctm_setenv(cache::CTMEnvironmentCache, env) =
    CTMEnvironmentCache(cache.network, cache.grid, cache.maxdim, env)

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

# Relative cutoff on the S values inverted by the biorthogonal projector. Those inverse
# powers amplify roundoff, so tiny directions must be dropped or they impose a hard,
# χ-independent error floor. 1e-8 ≈ √eps is the right scale (not 1e-12): the projector gets
# S from a Hermitian eig of a squared object, which resolves it to only ~√eps relatively.
# Measured optimum — 6×6 Ising: 5.1e-13 here vs 3.1e-10 at 1e-12 and 3.4e-9 at 1e-4.
const CTM_PINV_CUTOFF = Ref(1.0e-8)

# TRIANGULAR (QR) route for the interface projector. Instead of forming ρ_L = A†A and
# ρ_R = B†B and then eigendecomposing ρ_R back into a square-root factor — squaring the
# condition number and undoing it — take a thin QR of each block directly:
#
#   Bw = Q_A R_A,  Be = Q_B R_B      ⇒  R_A† R_A = ρ_L,  R_B† R_B = ρ_R    (exactly)
#
# The R's ARE the square-root factors, obtained without ever squaring. Both Q's are isometries,
# so the singular values of the full cross-interface object `Bw Be†` are those of the small
# triangular product `W = R_A R_B†`, and with `W = U S V†`:
#
#   P_A = R_B† V S^(-1/2)            P_B = S^(-1/2) U† R_A
#
# which is A(P_A P_B)B† = A B† exactly at full rank (verified to 1e-15). The `S^(-3/2)` on
# `P_B` also collapses to a symmetric `S^(-1/2)`, so the worst inverse power is gone.
#
# It uses ONE svd of a small triangular product, so U and V come from a single decomposition in
# a consistent basis — the reason the ρ route avoided separate eigen-solves for U and V. It is
# NOT an svd of a squared object.
#
# WHY IT IS HERE — GPU / BATCHING, NOT ACCURACY. Measured: this is accuracy-NEUTRAL. On 18
# moderate-χ configurations (3 seeds single-layer D=3, 2 double-layer D=2, 1 double-layer D=3;
# χ = 4/6/8) and 10 near-lossless ones it matches the ρ route to 3 significant figures, and at
# cutoffs 1e-8/1e-11/1e-13/1e-15 alike. The reason is that precision is not the binding
# constraint: the RETAINED spectrum has median `S_k/S_1` of 1e-1…1e-2 (measured over 200–384
# solves per sweep) and 0% of retained directions fall below 1e-8. In the ρ route a direction at
# `S_k/S_1 = 1e-8` carries relative error `~eps·(S_1/S_k)² ≈ 50%`, which is exactly why
# `CTM_PINV_CUTOFF` sits at √eps — the cutoff and the squaring are two faces of one constraint.
# QR makes directions down to ~1e-15 usable, but they carry no weight. **χ is the binding
# constraint, not arithmetic.** Do not expect an accuracy win from this; the win is that
# geqrf/gesvd have batched GPU implementations where batched Hermitian eig support is thin, and
# a sweep is 200–384 INDEPENDENT tiny factorizations (n ≤ 128) — a batching problem, not a
# big-linear-algebra one.
#
# Remaining GPU blocker, in BOTH routes: `Array(ρ, b, bp)` / `_ctm_block_matrix` materialise a
# host `Array`, so every projector round-trips through the CPU. Fixing that is separate work.
#
# DEFAULT since the GPU rationale landed. The eig route is kept reachable (`CTM_QR[] = false`)
# because it is the long-standing reference path and the two are numerically interchangeable;
# the general preference for eig-over-SVD elsewhere in this file still stands, and the reason
# behind it — one decomposition, one consistent basis for U and V — is satisfied here.
const CTM_QR = Ref(true)

# Relative cutoff for the QR route. Can sit far below `CTM_PINV_CUTOFF` precisely because `S`
# is no longer read off a squared object.
const CTM_QR_CUTOFF = Ref(1.0e-13)

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



# A block as a plain (rest × interface) matrix, CONJUGATED so that the triangular factor `R`
# from its QR satisfies `R†R = ρ` under this file's ρ convention
# (`ρ[i,j] = Σ_r B[r,i] conj(B[r,j])`, i.e. ρ = conj(B†B)). A no-op for real tensors.
function _ctm_block_matrix(B::ITensor, io::Index)
    rest = collect(uniqueinds(B, io))
    isempty(rest) && return reshape(conj(Array(B, io)), 1, ITensors.dim(io))
    return reshape(conj(Array(B, rest..., io)), :, ITensors.dim(io))
end

# Triangular factor of a block: R with R†R = ρ, never forming ρ. See `CTM_QR`.
_ctm_tri_factor(B::ITensor, io::Index) = Matrix(qr(_ctm_block_matrix(B, io)).R)

# Biorthogonal pair from the TRIANGULAR factors of the two bounding blocks — no squaring
# anywhere. See `CTM_QR` for the derivation.
function _ctm_twosided_projector_qr(Bw::ITensor, Be::ITensor, io::Index, maxdim::Integer)
    RA = _ctm_tri_factor(Bw, io)
    RB = _ctm_tri_factor(Be, io)
    F = svd(RA * RB')                       # ONE decomposition → consistent U, S, V
    S = F.S
    k = min(Int(maxdim), length(S))
    while k > 1 && S[k] ≤ CTM_QR_CUTOFF[] * S[1]
        k -= 1
    end
    while k > 1 && k < length(S) && abs(S[k] - S[k + 1]) ≤ CTM_DEGTOL[] * abs(S[k])
        k -= 1                                      # don't split a degenerate multiplet
    end
    Sk = S[1:k]; isk = Diagonal(1 ./ sqrt.(Sk))
    PAm = RB' * F.V[:, 1:k] * isk           # (bond × kept)
    PBm = isk * F.U[:, 1:k]' * RA           # (kept × bond)
    w = Index(k)
    return ITensor(PAm, io, w), ITensor(PBm, w, io), w
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

# Every C and T is renormalized as it is built, as in standard CTMRG — blocks span O(L²)
# vertices, so their raw magnitude grows like exp(c·L²) and would otherwise overflow.
#
# The CVM functional is INVARIANT under this. Each corner occurs in exactly four regions with
# Möbius weights +1 −1 −1 +1 (vertex, h-edge, v-edge, plaquette) and each edge tensor in two
# with +1 −1, so per-block scale cancels from `F` identically; every block for which that count
# would fail at the boundary is `nothing` and absent anyway. Single-site observables are ratios
# over one shared ring, so it cancels there too.
#
# CONSEQUENCE: an individual `region_lnZ` no longer equals `ln Z` — its scale is arbitrary.
# Only the Möbius-weighted SUM (`cvm_freenergy`) is meaningful.
_ctm_rescale(t) = isnothing(t) ? t :
    (n = norm(t); (iszero(n) || !isfinite(n)) ? t : t / n)

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
_ctm_dims(cache::CTMEnvironmentCache) = (length(first(cache.grid)), length(cache.grid))
_ctm_vertex(cache::CTMEnvironmentCache, x::Int, y::Int) = cache.grid[y][x]

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
        T[(:W, x + 1, y)] = _ctm_rescale(raw)
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
        T[(:E, x, y)] = _ctm_rescale(raw)
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
        C[(:NW, x, y + 1)] = _ctm_rescale(raw)
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
        C[(:SW, x, y)] = _ctm_rescale(raw)
    end
    # ---- C[:NE] / C[:SE]: consume PH ----
    for x in 2:Lx
        for y in 1:(Ly - 1)
            raw = _ctm_mul(_ctm_nn(C, (:NE, x, y)), _ctm_nn(T, (:E, x, y)))
            P = _ctm_nn(PH, (:N, x - 1, y + 1))
            !isnothing(P) && (raw = raw * dag(P[1]))
            C[(:NE, x, y + 1)] = _ctm_rescale(raw)
        end
        for y in Ly:-1:2
            raw = _ctm_mul(_ctm_nn(C, (:SE, x, y + 1)), _ctm_nn(T, (:E, x, y)))
            P = _ctm_nn(PH, (:S, x - 1, y))
            !isnothing(P) && (raw = raw * dag(P[1]))
            C[(:SE, x, y)] = _ctm_rescale(raw)
        end
    end
    # ---- N / S column strips: consume PH ----
    for x in 1:Lx
        for y in 1:(Ly - 1)
            raw = _ctm_mul(_ctm_nn(T, (:N, x, y)), _ctm_contract(a(x, y)))
            P = _ctm_nn(PH, (:N, x - 1, y + 1)); !isnothing(P) && (raw = raw * dag(P[1]))
            Q = _ctm_nn(PH, (:N, x, y + 1));     !isnothing(Q) && (raw = raw * Q[1])
            T[(:N, x, y + 1)] = _ctm_rescale(raw)
        end
        for y in Ly:-1:2
            raw = _ctm_mul(_ctm_contract(a(x, y)), _ctm_nn(T, (:S, x, y + 1)))
            P = _ctm_nn(PH, (:S, x - 1, y)); !isnothing(P) && (raw = raw * dag(P[1]))
            Q = _ctm_nn(PH, (:S, x, y));     !isnothing(Q) && (raw = raw * Q[1])
            T[(:S, x, y)] = _ctm_rescale(raw)
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
    if CTM_QR[]
        PA, PB, w = _ctm_twosided_projector_qr(Bwc, Bec, io, maxdim)
    else
        ρL = Bwc * prime(dag(Bwc), io)
        ρR = Bec * prime(dag(Bec), io)
        PA, PB, w = _ctm_twosided_projector(ρL, ρR, io, maxdim)
    end
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

One pass round the lattice, vertex to vertex: at each cut, grow the four enlarged corners out
of `S`, take a TWO-SIDED (biorthogonal) projector for each interface from the two corners that
bound it, and rebuild the corners and edges with it. Interfaces must be projected at growth,
when they are χ·D dimensional — re-projecting an already-truncated interface is a no-op — so a
sweep regrows rather than refines. Call [`update`](@ref) rather than this directly — it iterates
until [`cvm_freenergy`](@ref) stops moving.
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
    # Every `:S` block is keyed by its FIRST included row (`T_S[x,y] = rows ≥ y`), so the
    # family lives at `y ∈ 2:Ly` — `y = Ly+1` is the empty block. All four `:S` loops below
    # (this one, C_SW, C_SE, T_S) must use that range: `1:(Ly-1)` builds a useless `y = 1` and
    # never builds `y = Ly`, leaving the bottom interface of every region unconsumed.
    for x in 1:(Lx - 1), y in 2:Ly            # PH[:S,x,y]: C_SW(x+1,y) | C_SE(x+1,y)
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
        C[(:NW, x, y)] = _ctm_rescale(apA(apA(E(:NW, x, y), _ctm_nn(PH, (:N, x - 1, y))),
                                          _ctm_nn(PV, (:W, x, y - 1))))
    end
    for x in 1:(Lx - 1), y in 2:Ly
        C[(:NE, x + 1, y)] = _ctm_rescale(apA(apB(E(:NE, x + 1, y), _ctm_nn(PH, (:N, x, y))),
                                              _ctm_nn(PV, (:E, x + 1, y - 1))))
    end
    for x in 2:Lx, y in 2:Ly
        C[(:SW, x, y)] = _ctm_rescale(apB(apA(E(:SW, x, y), _ctm_nn(PH, (:S, x - 1, y))),
                                          _ctm_nn(PV, (:W, x, y - 1))))
    end
    for x in 1:(Lx - 1), y in 2:Ly
        C[(:SE, x + 1, y)] = _ctm_rescale(apB(apB(E(:SE, x + 1, y), _ctm_nn(PH, (:S, x, y))),
                                              _ctm_nn(PV, (:E, x + 1, y - 1))))
    end
    # --- rebuild edges from the previous state, projected on both sides -------------
    for x in 1:Lx, y in 2:Ly                  # T_N: left = east side, right = west side
        raw = _ctm_mul(_ctm_nn(S.T, (:N, x, y - 1)), _ctm_contract(tbl[(x, y - 1)]))
        T[(:N, x, y)] = _ctm_rescale(apA(apB(raw, _ctm_nn(PH, (:N, x - 1, y))), _ctm_nn(PH, (:N, x, y))))
    end
    for x in 1:Lx, y in 2:Ly                  # T_S
        raw = _ctm_mul(_ctm_contract(tbl[(x, y)]), _ctm_nn(S.T, (:S, x, y + 1)))
        T[(:S, x, y)] = _ctm_rescale(apA(apB(raw, _ctm_nn(PH, (:S, x - 1, y))), _ctm_nn(PH, (:S, x, y))))
    end
    for x in 2:Lx, y in 1:Ly                  # T_W: up = south side, down = north side
        raw = _ctm_mul(_ctm_nn(S.T, (:W, x - 1, y)), _ctm_contract(tbl[(x - 1, y)]))
        T[(:W, x, y)] = _ctm_rescale(apA(apB(raw, _ctm_nn(PV, (:W, x, y - 1))), _ctm_nn(PV, (:W, x, y))))
    end
    for x in 1:(Lx - 1), y in 1:Ly            # T_E
        raw = _ctm_mul(_ctm_contract(tbl[(x + 1, y)]), _ctm_nn(S.T, (:E, x + 2, y)))
        T[(:E, x + 1, y)] = _ctm_rescale(apA(apB(raw, _ctm_nn(PV, (:E, x + 1, y - 1))),
                                             _ctm_nn(PV, (:E, x + 1, y))))
    end
    return CTMVertexEnvironments(C, T, PH, PV, Lx, Ly)
end

"""
    region_lnZ(cache::CTMEnvironmentCache, cx, cy)
    region_lnZ(env::CTMVertexEnvironments, cache, cx, cy)

Region free energy `ln Z_R`. Integer `(cx,cy)` → vertex ring (4C+4T+a); half-integer in one
axis → edge strip (4C+2T); both half-integer → plaquette loop (4C).

!!! note "Scale is arbitrary"
    The C/T blocks are renormalized as they are built, so a single `region_lnZ` is offset from
    `ln Z` by a per-block gauge and is **not** meaningful on its own — even at lossless `maxdim`.
    Only the Möbius-weighted sum, [`cvm_freenergy`](@ref), is: the offsets cancel there exactly
    (`+1 −1 −1 +1` per corner, `+1 −1` per edge). Ratios over a single fixed region — a
    single-site observable, say — are also well defined.

The cache form uses the cache's own environments, so [`update`](@ref) it first.
"""
region_lnZ(cache::CTMEnvironmentCache, cx::Real, cy::Real) =
    region_lnZ(_ctm_env(cache), cache, cx, cy)

# The C/T blocks bounding a region, with boundary `nothing`s dropped. No vertex factors — the
# caller supplies those, which is what lets an observable be inserted (see `vertex_ring`).
function _ctm_region_blocks(env::CTMVertexEnvironments, cx::Real, cy::Real)
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
    return ITensor[t for t in ts if !isnothing(t)]
end

function region_lnZ(env::CTMVertexEnvironments, cache::CTMEnvironmentCache, cx::Real, cy::Real)
    ts = _ctm_region_blocks(env, cx, cy)
    if isinteger(cx) && isinteger(cy)     # vertex ring: close it with the vertex's own factors
        append!(ts, bp_factors(network(cache), _ctm_vertex(cache, Int(cx), Int(cy))))
    end
    isempty(ts) && return 0.0
    return log(abs(real(scalar(_ctm_contract(ts)))))
end

# The cache's environments, falling back to the greedy single pass when it has not been
# `update`d. Matches the BP convention: an un-updated cache evaluates, it just is not converged.
_ctm_env(cache::CTMEnvironmentCache) =
    isnothing(environments(cache)) ? vertex_environments(cache) : environments(cache)

"""
    vertex_ring(cache::CTMEnvironmentCache, v) -> Vector{ITensor}

The `4C + 4T` ring enclosing vertex `v = (x, y)` — the corners `C_NW(x,y)`, `C_NE(x+1,y)`,
`C_SW(x,y+1)`, `C_SE(x+1,y+1)` and the edges `T_N(x,y)`, `T_S(x,y+1)`, `T_W(x,y)`,
`T_E(x+1,y)`, with blocks that fall off the lattice omitted. This is the CTMRG analogue of
`incoming_messages` for a BP cache: it is `v`'s environment, and it contains **no** factor from
`v` itself, so the caller closes it with whatever they want at the site.

Its open legs are exactly the ket and bra virtual indices of `v`'s own tensors (kept separate at
dimension `D`, never fused), so it pairs directly with `norm_factors(ψ, v; op_strings)`:

```julia
cache = update(CTMEnvironmentCache(ψ, χ))
ring  = vertex_ring(cache, v)
num   = scalar(contract([norm_factors(ψ, v; op_strings = _ -> "Z"); ring]))
den   = scalar(contract([norm_factors(ψ, v; op_strings = _ -> "I"); ring]))
Zexp  = num / den
```

[`expect`](@ref) with `alg = "ctmrg"` does exactly this. [`update`](@ref) the cache first —
otherwise the ring comes from the greedy single pass.
"""
vertex_ring(cache::CTMEnvironmentCache, v) = vertex_ring(_ctm_env(cache), cache, v)

function vertex_ring(env::CTMVertexEnvironments, cache::CTMEnvironmentCache, v)
    x, y = _ctm_coords(cache, v)
    return _ctm_region_blocks(env, x, y)
end

# Grid position of a vertex, by lookup rather than by trusting `v == (x, y)` — the cache sorts
# its rows, and a network's vertices need not be 1-based or contiguous.
function _ctm_coords(cache::CTMEnvironmentCache, v)
    for (y, row) in enumerate(cache.grid)
        x = findfirst(==(v), row)
        isnothing(x) || return (x, y)
    end
    return error("vertex $v is not in the CTMEnvironmentCache's grid.")
end

"""
    cvm_freenergy(cache::CTMEnvironmentCache)

Region-graph (CVM) free energy `F = Σ_v ln Z_v − Σ_e ln Z_e + Σ_p ln Z_p`, read off the cache's
environments. Exact when they are lossless, since `V − E + P = 1` for a disk.

[`update`](@ref) the cache first. On an un-updated cache this falls back to the greedy single
pass ([`vertex_environments`](@ref)), whose one-sided cuts are 3–4 orders worse and
**non-monotone in `maxdim`** — the two numbers differing is that, not a bug.
"""
cvm_freenergy(cache::CTMEnvironmentCache) = cvm_freenergy(_ctm_env(cache), cache)

function cvm_freenergy(env::CTMVertexEnvironments, cache::CTMEnvironmentCache)
    Lx, Ly = env.Lx, env.Ly
    F = 0.0
    for x in 1:Lx, y in 1:Ly;           F += region_lnZ(env, cache, x, y);             end
    for x in 1:(Lx - 1), y in 1:Ly;     F -= region_lnZ(env, cache, x + 0.5, y);       end
    for x in 1:Lx, y in 1:(Ly - 1);     F -= region_lnZ(env, cache, x, y + 0.5);       end
    for x in 1:(Lx - 1), y in 1:(Ly - 1); F += region_lnZ(env, cache, x + 0.5, y + 0.5); end
    return F
end

"""
    update(cache::CTMEnvironmentCache; maxiter = 30, tol = 1e-10, verbose = false)

Run the two-sided CVM sweep on `cache` to stationarity and return a cache carrying the
converged per-vertex environments. Extract numbers from it with [`cvm_freenergy`](@ref) or
[`region_lnZ`](@ref):

```julia
cache = update(CTMEnvironmentCache(ψ, χ))
F = cvm_freenergy(cache)
```

Seeds from the greedy single pass ([`vertex_environments`](@ref)), then applies
[`sweep_vertex_environments`](@ref) until `cvm_freenergy` stops moving. The two-sided projector
needs the complement environment, which needs the other corners, so this is a genuine
fixed-point map rather than a one-shot build. The first sweep does almost all the work (it is
what replaces the greedy pass's one-sided cuts), but the tail is slow: `|ΔF|` typically needs
~8–12 sweeps to reach 1e-8. Stopping at 2–3, as an earlier iteration did, lands mid-transient
and reads as a limit cycle. Warns if `tol` is not met within `maxiter`.
"""
function update(cache::CTMEnvironmentCache; maxiter::Integer = 30, tol::Real = 1.0e-10,
                verbose::Bool = false)
    env = _ctm_env(cache)
    F = cvm_freenergy(env, cache)
    converged, Δ = false, Inf
    for it in 1:maxiter
        env = sweep_vertex_environments(cache, env)
        Fnew = cvm_freenergy(env, cache)
        Δ = abs(Fnew - F)
        F = Fnew
        verbose && @info "CVM sweep $it: F = $F, |ΔF| = $Δ"
        if Δ ≤ tol * max(one(Δ), abs(F))
            converged = true
            verbose && @info "CVM sweep converged after $it sweeps."
            break
        end
    end
    if !converged
        msg = "CVM sweep did not converge to tolerance $tol after $maxiter sweeps " *
              "(final |ΔF| = $Δ)."
        verbose ? println(msg) : @warn(msg)
    end
    return _ctm_setenv(cache, env)
end
