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

using LinearAlgebra: eigen, Hermitian, norm, dot, I, Diagonal, qr, svd
using KrylovKit: eigsolve, svdsolve

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
    grid::Dict{Tuple{Int, Int}, V}   # OCCUPIED positions only — holes allowed (hex, heavy-hex)
    dims::Tuple{Int, Int}            # bounding box (Lx, Ly)
    maxdim::Int
    environments::E                  # `nothing`, or the CVM blocks from `update`
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
    # NO rectangularity requirement. Holes are fine: `C_NW = {col<x, row<y}` and the `T`
    # strips partition grid positions by COMPARISON, not occupancy, so the 4C+4T tiling and the
    # Möbius identity both survive. The identity is a telescoping one on the BOUNDING BOX —
    # Lx·Ly − (Lx−1)Ly − Lx(Ly−1) + (Lx−1)(Ly−1) = 1 — and is independent of which slots are
    # filled. An empty vertex slot simply has no site factor to insert. This is what lets
    # hexagonal and heavy-hexagonal lattices (laid out on (x,y) with vertices/edges missing) use
    # the same engine.
    grid = Dict{Tuple{Int, Int}, eltype(vs)}((Int(v[1]), Int(v[2])) => v for v in vs)
    Lx = maximum(first.(keys(grid))); Ly = maximum(last.(keys(grid)))
    return CTMEnvironmentCache(net, grid, (Lx, Ly), Int(maxdim), nothing)
end

# Same network/grid/maxdim, different CVM environments.
_ctm_setenv(cache::CTMEnvironmentCache, env) =
    CTMEnvironmentCache(cache.network, cache.grid, cache.dims, cache.maxdim, env)

# --- the move --------------------------------------------------------------------
# eig projector from a density matrix ρ(bnd, bnd'): the top-`maxdim` eigenvectors.
# Relative cutoff gap below which the truncation is judged to split a (near-)degenerate
# multiplet; the cut is then backed off to a real gap. Matters for DOUBLE-LAYER networks,
# whose corner spectra carry systematic 2-fold degeneracies from ket↔bra exchange
# (λ_ij = λ_ji). 0 disables it.
const CTM_DEGTOL = Ref(0.0)

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

# Smallest interface at which the Krylov SVD beats a dense one; see `_ctm_svd_topk` for the
# measured crossover. Below this, dense LAPACK wins even when only `k` of `n` triplets are used.
const CTM_KRYLOV_MIN = Ref(128)

# Top-`k` eigenpairs of a Hermitian matrix: Krylov when it pays off, else dense.
function _ctm_eigsolve(ρs::Hermitian, k::Integer)
    n = size(ρs, 1)
    if CTM_ARNOLDI[] && n > 4k
        try
            v0 = randn(eltype(ρs), n)
            # `verbosity = 0`: when the interface's effective rank is below `k` — routine at
            # larger D — KrylovKit reports "invariant subspace of dimension r < howmany" and
            # "stopped without convergence". Both are expected here, not errors: we fall through
            # to dense below, and the dense result is bit-identical (verified) and deterministic.
            # Warning on every such call buries real problems in noise.
            vals, vecs, info = eigsolve(x -> ρs * x, v0, k, :LR;
                                        ishermitian = true, verbosity = 0,
                                        krylovdim = max(2k + 8, 20))
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
    k = min(Int(maxdim), length(sv), size(vecs, 2))
    while k > 1 && k < length(sv) && abs(sv[k] - sv[k + 1]) ≤ CTM_DEGTOL[] * abs(sv[k])
        k -= 1                              # don't split a degenerate multiplet
    end
    keep = order[1:k]
    w = Index(k)
    return ITensor(vecs[:, keep], bnd, w), w
end

# Contract a small flat list in a good order (netcon). This is what keeps the double layer
# lazy: the list is [environment; ket; (operator;) bra] and the optimizer interleaves them,
# so the fat ket⊗bra site tensor (legs of dimension D²) is never formed.
# Netcon (`alg = "optimal"`) is expensive, and the lattice geometry is fixed — so the SAME
# einsum recurs once per sweep for 8–12 sweeps. Measured 4000–9000 `_ctm_contract` calls per
# `update()` with a 97–99% repeat rate, netcon accounting for 17–31% of contraction time on
# double-layer runs and 89% on a lossless single-layer one.
#
# Cache the sequence on a STRUCTURAL key: each tensor's indices relabelled by order of first
# appearance across the list, paired with their dimensions. For a fixed tensor ordering that is a
# canonical form, and `contract(ts; sequence)` addresses tensors by POSITION, so a cached sequence
# is exactly what netcon would have returned. Keys are shape-only, so different networks of the
# same geometry share entries and the cache is bounded by the number of distinct shapes.
#
# NOT thread-safe (plain `Dict`); this engine is single-threaded.
const CTM_SEQ_CACHE = Dict{Any, Any}()

function _ctm_seq_key(ts::Vector{ITensor})
    seen = Dict{Any, Int}()
    label(i) = (get!(seen, i, length(seen) + 1), ITensors.dim(i))
    return Tuple(Tuple(label(i) for i in inds(t)) for t in ts)
end

# Above this tensor count, fall back to the greedy optimiser. This is a FEASIBILITY gate, not a
# performance one: `alg = "optimal"` is ExhaustiveSearch netcon, exponential in the number of
# tensors, and it hangs outright on the ~25-tensor lists a `vertex_window` observable produces.
# (Tried and reverted as a *perf* tweak earlier — it bought ~1.5% on sweep-sized lists.)
const CTM_OPTIMAL_MAX = Ref(12)

function _ctm_contract(ts::Vector{ITensor})
    length(ts) == 1 && return only(ts)
    length(ts) == 2 && return ts[1] * ts[2]          # no sequence to choose
    seq = get!(CTM_SEQ_CACHE, _ctm_seq_key(ts)) do
        length(ts) <= CTM_OPTIMAL_MAX[] ?
            contraction_sequence(ts; alg = "optimal") :
            contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod())
    end
    return contract(ts; sequence = seq)
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
# Top-`k` singular triplets of `W`, by Golub–Kahan–Lanczos when it pays. `W` is `n×n` with
# `n = χ·D_layer` and only `k = maxdim` triplets are ever used, so a full dense SVD discards
# everything past column `k` — measured 45% of wall at 5×5 D=4 χ=12.
#
# GATE: `n ≥ CTM_KRYLOV_MIN` **and** `n > 4k`. The ratio test alone is not enough — measured
# crossover (dense vs `svdsolve`, ms per solve): n=72 0.62/0.50, n=96 1.06/3.31, n=128 2.13/1.11,
# n=192 5.70/2.49, n=256 12.22/3.95. So Krylov wins 1.9–3.1× from n≈128 and *loses* below it;
# gating on the ratio alone made 4×4 D=3 χ=8 (n≤72) 1.2× SLOWER. Falls back to dense on
# non-convergence; agreement with dense measured to 1.4e-15 on matrices captured from a real run.
function _ctm_svd_topk(W::AbstractMatrix, k::Integer)
    try
        vals, lvecs, rvecs, info = svdsolve(W, k, :LR)
        (info.converged >= k && length(lvecs) >= k && length(rvecs) >= k) || return nothing
        U = reduce(hcat, @view(lvecs[1:k])); V = reduce(hcat, @view(rvecs[1:k]))
        eltype(W) <: Real && (U = real.(U); V = real.(V))
        return (; S = real.(vals[1:k]), U, V)
    catch
        return nothing                                # fall through to dense
    end
end

function _ctm_twosided_projector_qr(Bw::ITensor, Be::ITensor, io::Index, maxdim::Integer)
    RA = _ctm_tri_factor(Bw, io)
    RB = _ctm_tri_factor(Be, io)
    W = RA * RB'
    kw = min(Int(maxdim), min(size(W)...))
    nW = min(size(W)...)
    F = (CTM_ARNOLDI[] && nW >= CTM_KRYLOV_MIN[] && nW > 4kw) ? _ctm_svd_topk(W, kw) : nothing
    isnothing(F) && (F = svd(W))            # ONE decomposition → consistent U, S, V
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
_ctm_mul(a, b) = isnothing(a) ? b : (isnothing(b) ? a : a * b)   # 2 tensors: no netcon needed
# Kept index of a stored projector. The greedy pass stores `(P, w)` and the sweep stores
# `(P_A, P_B, w)`, so index from the END — `t[2]` would silently mean `P_B` on a swept dict.
_ctm_widx(d, k) = (t = get(d, k, nothing); isnothing(t) ? nothing : t[end])

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
    P, w = _ctm_eig_projector(ρ, io, k)
    return P * co, w
end

# Grid geometry / lazy factors ----------------------------------------------------
_ctm_dims(cache::CTMEnvironmentCache) = cache.dims
# `nothing` at an unoccupied grid position.
_ctm_vertex(cache::CTMEnvironmentCache, x::Int, y::Int) = get(cache.grid, (x, y), nothing)

# Contracted site tensor at a grid position, or `nothing` if unoccupied.
_ctm_site(tbl, x::Int, y::Int) = haskey(tbl, (x, y)) ? _ctm_contract(tbl[(x, y)]) : nothing

function _ctm_factor_table(cache::CTMEnvironmentCache)
    Lx, Ly = _ctm_dims(cache)
    tbl = Dict{Tuple{Int, Int}, Vector{ITensor}}()
    for y in 1:Ly, x in 1:Lx
        v = _ctm_vertex(cache, x, y)
        isnothing(v) && continue                      # unoccupied position (hex etc.)
        tbl[(x, y)] = Vector{ITensor}(bp_factors(network(cache), v))
    end
    return tbl
end

# Links between neighbouring vertices: ONE index for a single layer, TWO (ket+bra) for a
# double layer — discovered from the tensors, never fused.
function _ctm_links(tbl, a::Tuple{Int, Int}, b::Tuple{Int, Int})
    (haskey(tbl, a) && haskey(tbl, b)) || return Index[]   # one end unoccupied: no link
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
    a(x, y) = get(tbl, (x, y), ITensor[])

    C = Dict{Tuple{Symbol, Int, Int}, Any}()
    T = Dict{Tuple{Symbol, Int, Int}, Any}()
    PH = Dict{Tuple{Symbol, Int, Int}, Any}()
    PV = Dict{Tuple{Symbol, Int, Int}, Any}()

    # ---- W strips (y increasing, x increasing): derives PV[:W] ----
    for y in 1:Ly, x in 1:(Lx - 1)
        raw = _ctm_mul(_ctm_nn(T, (:W, x, y)), _ctm_site(tbl, x, y))
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
        raw = _ctm_mul(_ctm_site(tbl, x, y), _ctm_nn(T, (:E, x + 1, y)))
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
            raw = _ctm_mul(_ctm_nn(T, (:N, x, y)), _ctm_site(tbl, x, y))
            P = _ctm_nn(PH, (:N, x - 1, y + 1)); !isnothing(P) && (raw = raw * dag(P[1]))
            Q = _ctm_nn(PH, (:N, x, y + 1));     !isnothing(Q) && (raw = raw * Q[1])
            T[(:N, x, y + 1)] = _ctm_rescale(raw)
        end
        for y in Ly:-1:2
            raw = _ctm_mul(_ctm_site(tbl, x, y), _ctm_nn(T, (:S, x, y + 1)))
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


# --- gauge fixing --------------------------------------------------------------------
# The pair (P_A, P_B) has an exact gauge freedom  P_A -> P_A R,  P_B -> R⁻¹ P_B: it leaves
# Π = P_A P_B, hence every region value and `F`, untouched. But the sweep picks that gauge
# arbitrarily (and a fresh `Index`) every iteration, so `S` and `sweep(S)` sit in DIFFERENT bases.
# That is what blocks every accelerator — Anderson, JFNK and Krylov all need to linearly combine
# iterates — and it leaves `|ΔF|` as the only convergence signal instead of a state distance.
#
# THE GAUGE MUST BE UNITARY. Measured: canonicalising by QR (pushing the triangular factor into
# `P_B`) changes `F` at finite χ — 1.1e-2 at χ=4, 2.6e-3 at χ=6, invariant only at lossless χ=12.
# A triangular R is not unitary, so it changes the metric on the interface, and the NEXT level's
# SVD truncation then selects a different subspace. Truncation is not gauge invariant; only
# inner-product-preserving changes of basis are safe.
#
# So align to the previous sweep with the nearest UNITARY (orthogonal Procrustes): with
# M = P_A_newᵀ P_A_old = U S Vᵀ, take R = U Vᵀ. This preserves Π and every inner product, so `F`
# is exactly invariant, while rotating the new basis as close to the old one as a unitary can.
# Reusing the old `Index` then makes successive blocks directly comparable.
#
# Bootstrapping note: `ins` itself contains the previous level's kept index, so alignment only
# becomes possible once the lower levels are already index-stable. The guard below falls through
# to the unaligned pair whenever the old projector does not live on the current `ins`, which is
# what happens on the first gauge-fixed sweep.
# DEFAULT ON: `F` is exactly invariant (verified to 1e-14 at χ = 4/6/8/12), the cost is one k×k
# SVD per interface per sweep, and it turns `|ΔF|` — which oscillates at the roundoff floor of a
# signed log-sum, measured rising 1.2e-7 -> 3.4e-7 -> 5.4e-7 over sweeps 8..10 — into a monotone
# state distance. It is also the prerequisite for any accelerator.
const CTM_GAUGE = Ref(true)

function _ctm_align(pr, ins, prev)
    (isnothing(prev) || length(prev) < 3) && return pr
    PA, PB, w = pr
    PAo, _, wo = prev
    k = ITensors.dim(w)
    ITensors.dim(wo) == k || return pr
    issetequal(collect(inds(PAo)), vcat(collect(ins), [wo])) || return pr   # same raw space?
    co = combiner(ins...); io = combinedind(co)
    Am, Ao, Bm, R = try
        a  = Array(PA * co, io, w)
        ao = Array(PAo * co, io, wo)
        b  = Array(PB * co, w, io)
        F  = svd(a' * ao)
        a, ao, b, F.U * F.Vt                          # nearest unitary
    catch
        return pr                                     # any trouble: keep the unaligned pair
    end
    (all(isfinite, Am) && all(isfinite, Ao) && all(isfinite, Bm) && all(isfinite, R)) || return pr
    return (ITensor(Am * R, io, wo) * co, ITensor(R' * Bm, wo, io) * co, wo)
end

# Largest relative change of any block between two states, over blocks that share an index set.
# Meaningful only with `CTM_GAUGE[]` on; returns `nothing` while the bases are still bootstrapping
# (the first ~3 sweeps), since `ins` carries the lower level's index and stability propagates up.
function _ctm_statedist(a::CTMVertexEnvironments, b::CTMVertexEnvironments)
    n = 0; worst = 0.0
    for (d1, d2) in ((a.C, b.C), (a.T, b.T)), (k, ta) in d1
        tb = _ctm_nn(d2, k)
        (isnothing(ta) || isnothing(tb)) && continue
        Set(inds(ta)) == Set(inds(tb)) || continue
        na = norm(ta)
        na > 0 && (worst = max(worst, norm(ta - tb) / na))
        n += 1
    end
    return n == 0 ? nothing : worst
end

# Enlarged corner: the quadrant cut at (x,y), grown one vertex out of the PREVIOUS state's
# blocks (so all indices are in a consistent basis) with its two adjoining edges and vertex.
function _ctm_enlarged(S::CTMVertexEnvironments, tbl, sym::Symbol, x::Int, y::Int)
    A(i, j) = (haskey(tbl, (i, j)) ? _ctm_contract(tbl[(i, j)]) : nothing)
    # ONE netcon over all four rather than the hardcoded left fold `((C·T)·T)·a`. Measured: at
    # these sizes netcon PICKS THAT SAME ORDER, so this buys no speed (within noise on four
    # benchmarks) — it is here so the order stops being an unverified assumption, which matters
    # because the enlarged corner is the hottest object in the sweep. Free, given the sequence
    # cache. Boundary blocks arrive as `nothing` and drop out.
    m4(args...) = (ts = ITensor[t for t in args if !isnothing(t)];
                   isempty(ts) ? nothing : _ctm_contract(ts))
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

# =================================================================================
# Region/block reconstruction, used by the marginal-consistency diagnostic below.
#
# `_ctm_region_desc` gives a region's block descriptors, Möbius weight and centre vertex;
# `_ctm_block` rebuilds any one block from the enlarged pieces plus a supplied projector set.
#
# CRITICAL: the projector set passed to `_ctm_block` must be the one derived FROM `S`'s enlarged
# corners, i.e. the *next* sweep's `PH`/`PV` — not `S.PH`/`S.PV`, which were derived during the
# sweep that produced `S` and whose legs reference the pre-`S` indices. Mixing them shares only
# one index of two, contracts over the wrong leg and silently returns garbage. That mistake
# produced a completely wrong conclusion once; see the retraction in docs/finite_ctmrg_design.md.

# A region's block descriptors, its Möbius weight, and its centre vertex (if a vertex region).
function _ctm_region_desc(cx::Real, cy::Real)
    rL = ceil(Int, cx); rR = floor(Int, cx) + 1
    tT = ceil(Int, cy); tB = floor(Int, cy) + 1
    xint = rL < rR; yint = tT < tB
    ds = Any[(:C, :NW, rL, tT), (:C, :NE, rR, tT), (:C, :SW, rL, tB), (:C, :SE, rR, tB)]
    if xint
        push!(ds, (:T, :N, Int(cx), tT)); push!(ds, (:T, :S, Int(cx), tB))
    end
    if yint
        push!(ds, (:T, :W, rL, Int(cy))); push!(ds, (:T, :E, rR, Int(cy)))
    end
    nhalf = (xint ? 0 : 1) + (yint ? 0 : 1)
    return ds, (iseven(nhalf) ? 1 : -1), (xint && yint ? (Int(cx), Int(cy)) : nothing)
end

# One block, rebuilt from `S`'s enlarged pieces plus the projector set `P` — mirrors
# `sweep_vertex_environments` exactly, which is the correctness argument for it.
function _ctm_block(S::CTMVertexEnvironments, tbl, P::CTMVertexEnvironments, d)
    phg(k) = _ctm_nn(P.PH, k)
    pvg(k) = _ctm_nn(P.PV, k)
    kind, sym, i, j = d
    aA(t, p) = (isnothing(p) || isnothing(t)) ? t : t * p[1]
    aB(t, p) = (isnothing(p) || isnothing(t)) ? t : t * p[2]
    fac(a, b) = haskey(tbl, (a, b)) ? _ctm_contract(tbl[(a, b)]) : nothing
    if kind === :C
        E = _ctm_enlarged(S, tbl, sym, i, j)
        isnothing(E) && return nothing
        sym === :NW && return aA(aA(E, phg((:N, i - 1, j))), pvg((:W, i, j - 1)))
        sym === :NE && return aA(aB(E, phg((:N, i - 1, j))), pvg((:E, i, j - 1)))
        sym === :SW && return aB(aA(E, phg((:S, i - 1, j))), pvg((:W, i, j - 1)))
        return aB(aB(E, phg((:S, i - 1, j))), pvg((:E, i, j - 1)))
    end
    if sym === :N
        r = _ctm_mul(_ctm_nn(S.T, (:N, i, j - 1)), fac(i, j - 1))
        return isnothing(r) ? nothing : aA(aB(r, phg((:N, i - 1, j))), phg((:N, i, j)))
    elseif sym === :S
        r = _ctm_mul(fac(i, j), _ctm_nn(S.T, (:S, i, j + 1)))
        return isnothing(r) ? nothing : aA(aB(r, phg((:S, i - 1, j))), phg((:S, i, j)))
    elseif sym === :W
        r = _ctm_mul(_ctm_nn(S.T, (:W, i - 1, j)), fac(i - 1, j))
        return isnothing(r) ? nothing : aA(aB(r, pvg((:W, i, j - 1))), pvg((:W, i, j)))
    end
    r = _ctm_mul(fac(i, j), _ctm_nn(S.T, (:E, i + 1, j)))
    return isnothing(r) ? nothing : aA(aB(r, pvg((:E, i, j - 1))), pvg((:E, i, j)))
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
        ins = commoninds(Bw, Be)
        pr = _ctm_interface_proj2(Bw, Be, ins, χ)
        CTM_GAUGE[] && !isnothing(pr) && (pr = _ctm_align(pr, ins, _ctm_nn(S.PH, (:N, x, y))))
        !isnothing(pr) && (PH[(:N, x, y)] = pr)
    end
    # Every `:S` block is keyed by its FIRST included row (`T_S[x,y] = rows ≥ y`), so the
    # family lives at `y ∈ 2:Ly` — `y = Ly+1` is the empty block. All four `:S` loops below
    # (this one, C_SW, C_SE, T_S) must use that range: `1:(Ly-1)` builds a useless `y = 1` and
    # never builds `y = Ly`, leaving the bottom interface of every region unconsumed.
    for x in 1:(Lx - 1), y in 2:Ly            # PH[:S,x,y]: C_SW(x+1,y) | C_SE(x+1,y)
        Bw = E(:SW, x + 1, y); Be = E(:SE, x + 1, y)
        (isnothing(Bw) || isnothing(Be)) && continue
        ins = commoninds(Bw, Be)
        pr = _ctm_interface_proj2(Bw, Be, ins, χ)
        CTM_GAUGE[] && !isnothing(pr) && (pr = _ctm_align(pr, ins, _ctm_nn(S.PH, (:S, x, y))))
        !isnothing(pr) && (PH[(:S, x, y)] = pr)
    end
    for x in 2:Lx, y in 1:(Ly - 1)            # PV[:W,x,y]: C_NW(x,y+1) | C_SW(x,y+1)
        Bn = E(:NW, x, y + 1); Bs = E(:SW, x, y + 1)
        (isnothing(Bn) || isnothing(Bs)) && continue
        ins = commoninds(Bn, Bs)
        pr = _ctm_interface_proj2(Bn, Bs, ins, χ)
        CTM_GAUGE[] && !isnothing(pr) && (pr = _ctm_align(pr, ins, _ctm_nn(S.PV, (:W, x, y))))
        !isnothing(pr) && (PV[(:W, x, y)] = pr)
    end
    for x in 1:(Lx - 1), y in 1:(Ly - 1)      # PV[:E,x,y]: C_NE(x,y+1) | C_SE(x,y+1)
        Bn = E(:NE, x + 1, y + 1); Bs = E(:SE, x + 1, y + 1)
        (isnothing(Bn) || isnothing(Bs)) && continue
        ins = commoninds(Bn, Bs)
        pr = _ctm_interface_proj2(Bn, Bs, ins, χ)
        CTM_GAUGE[] && !isnothing(pr) && (pr = _ctm_align(pr, ins, _ctm_nn(S.PV, (:E, x + 1, y))))
        !isnothing(pr) && (PV[(:E, x + 1, y)] = pr)
    end
    # --- rebuild corners: P_A on the west/north side, P_B on the east/south side ----
    apA(t, pr) = isnothing(pr) || isnothing(t) ? t : t * pr[1]
    apB(t, pr) = isnothing(pr) || isnothing(t) ? t : t * pr[2]
    # Horizontal projector takes P_A on the west corners and P_B on the east; vertical takes
    # P_A on the north and P_B on the south. Keys are uniformly (fam, x−1, y) and (fam, x, y−1).
    for (sym, hfam, hA, vfam, vA) in ((:NW, :N, true,  :W, true), (:NE, :N, false, :E, true),
                                      (:SW, :S, true,  :W, false), (:SE, :S, false, :E, false))
        for x in 2:Lx, y in 2:Ly
            t = (hA ? apA : apB)(E(sym, x, y), _ctm_nn(PH, (hfam, x - 1, y)))
            C[(sym, x, y)] = _ctm_rescale((vA ? apA : apB)(t, _ctm_nn(PV, (vfam, x, y - 1))))
        end
    end
    # --- rebuild edges from the previous state, projected on both sides -------------
    for x in 1:Lx, y in 2:Ly                  # T_N: left = east side, right = west side
        raw = _ctm_mul(_ctm_nn(S.T, (:N, x, y - 1)), _ctm_site(tbl, x, y - 1))
        T[(:N, x, y)] = _ctm_rescale(apA(apB(raw, _ctm_nn(PH, (:N, x - 1, y))), _ctm_nn(PH, (:N, x, y))))
    end
    for x in 1:Lx, y in 2:Ly                  # T_S
        raw = _ctm_mul(_ctm_site(tbl, x, y), _ctm_nn(S.T, (:S, x, y + 1)))
        T[(:S, x, y)] = _ctm_rescale(apA(apB(raw, _ctm_nn(PH, (:S, x - 1, y))), _ctm_nn(PH, (:S, x, y))))
    end
    for x in 2:Lx, y in 1:Ly                  # T_W: up = south side, down = north side
        raw = _ctm_mul(_ctm_nn(S.T, (:W, x - 1, y)), _ctm_site(tbl, x - 1, y))
        T[(:W, x, y)] = _ctm_rescale(apA(apB(raw, _ctm_nn(PV, (:W, x, y - 1))), _ctm_nn(PV, (:W, x, y))))
    end
    for x in 1:(Lx - 1), y in 1:Ly            # T_E
        raw = _ctm_mul(_ctm_site(tbl, x + 1, y), _ctm_nn(S.T, (:E, x + 2, y)))
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
# One block by descriptor, or `nothing` at the boundary.
_ctm_fetch(env::CTMVertexEnvironments, d) =
    (kind = d[1]; _ctm_nn(kind === :C ? env.C : env.T, (d[2], d[3], d[4])))

function _ctm_region_blocks(env::CTMVertexEnvironments, cx::Real, cy::Real)
    ds, _, _ = _ctm_region_desc(cx, cy)
    return ITensor[t for t in (_ctm_fetch(env, d) for d in ds) if !isnothing(t)]
end

function region_lnZ(env::CTMVertexEnvironments, cache::CTMEnvironmentCache, cx::Real, cy::Real)
    ts = _ctm_region_blocks(env, cx, cy)
    if isinteger(cx) && isinteger(cy)     # vertex ring: close it with the vertex's own factors
        v = _ctm_vertex(cache, Int(cx), Int(cy))
        isnothing(v) || append!(ts, bp_factors(network(cache), v))
    end
    isempty(ts) && return 0.0
    return log(abs(real(scalar(_ctm_contract(ts)))))
end

# The cache's environments, falling back to the greedy single pass when it has not been
# `update`d. Matches the BP convention: an un-updated cache evaluates, it just is not converged.
_ctm_env(cache::CTMEnvironmentCache) =
    isnothing(environments(cache)) ? vertex_environments(cache) : environments(cache)

"""
    vertex_window(cache::CTMEnvironmentCache, v, w::Integer = 0) -> Vector{ITensor}

The environment of vertex `v` from a rectangular window of half-width `w`, as the block list to
close with `v`'s own factors. `w = 0` is the `4C + 4T` ring; `w = 1` keeps the surrounding 3×3
patch **exact** and pushes the truncated environment one site further out; and so on.

Every block already exists in the cache, so a larger window costs only a larger contraction — no
extra sweeps, no extra truncation of the blocks themselves. With cuts at `(xL, xR, yT, yB)` the
window is

```
4C  +  T_N/T_S on columns xL … xR−1  +  T_W/T_E on rows yT … yB−1  +  interior sites except v
```

which tiles the lattice for any window, so it is exact at lossless `maxdim` like the ring.

**This is the lever for observable accuracy at fixed χ.** Measured on a 6×6 D=2 PEPS, `w = 1`
against `w = 0`: better at 8 of 9 (site, χ) combinations by 1.4×–11.4×, and better than boundary
MPS at 6 of 9 — including all three sites at χ=6. The exception is a near-boundary site at χ=2,
where the ring was barely truncated and the extra interfaces cost more than the exact context buys.

Note the site is *excluded* from the returned list, so the caller supplies it — that is what lets
an operator be inserted. See [`expect`](@ref) with `alg = "ctmrg"` and its `window` keyword.
"""
function vertex_window(cache::CTMEnvironmentCache, v, w::Integer = 0)
    env = _ctm_env(cache)
    tbl = _ctm_factor_table(cache)
    Lx, Ly = _ctm_dims(cache)
    x, y = _ctm_coords(cache, v)
    xL, xR = max(1, x - w), min(Lx, x + w) + 1
    yT, yB = max(1, y - w), min(Ly, y + w) + 1
    ts = ITensor[]
    for b in (_ctm_nn(env.C, (:NW, xL, yT)), _ctm_nn(env.C, (:NE, xR, yT)),
              _ctm_nn(env.C, (:SW, xL, yB)), _ctm_nn(env.C, (:SE, xR, yB)))
        isnothing(b) || push!(ts, b)
    end
    for c in xL:(xR - 1), b in (_ctm_nn(env.T, (:N, c, yT)), _ctm_nn(env.T, (:S, c, yB)))
        isnothing(b) || push!(ts, b)
    end
    for r in yT:(yB - 1), b in (_ctm_nn(env.T, (:W, xL, r)), _ctm_nn(env.T, (:E, xR, r)))
        isnothing(b) || push!(ts, b)
    end
    for c in xL:(xR - 1), r in yT:(yB - 1)
        (c, r) == (x, y) && continue
        haskey(tbl, (c, r)) && append!(ts, tbl[(c, r)])
    end
    return ts
end

"""
    vertex_ring(cache::CTMEnvironmentCache, v) -> Vector{ITensor}

The `4C + 4T` ring enclosing `v` — [`vertex_window`](@ref) at `w = 0`. Its open legs are exactly
`v`'s ket and bra virtual indices, so it pairs directly with `norm_factors(ψ, v; op_strings)`.
"""
vertex_ring(cache::CTMEnvironmentCache, v) = vertex_window(cache, v, 0)

function vertex_ring(env::CTMVertexEnvironments, cache::CTMEnvironmentCache, v)
    x, y = _ctm_coords(cache, v)
    return _ctm_region_blocks(env, x, y)
end

# Grid position of a vertex, by lookup rather than by trusting `v == (x, y)` — the cache sorts
# its rows, and a network's vertices need not be 1-based or contiguous.
function _ctm_coords(cache::CTMEnvironmentCache, v)
    for (pos, w) in cache.grid
        w == v && return pos
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
    # The half-integer grid enumerates every region exactly once — integer/integer is a vertex
    # (+1), one half-integer an edge (−1), both a plaquette (+1) — and `_ctm_region_desc`
    # already knows the Möbius weight, so it is not spelled out a second time here.
    F = 0.0
    for cx in 1.0:0.5:Lx, cy in 1.0:0.5:Ly
        F += _ctm_region_desc(cx, cy)[2] * region_lnZ(env, cache, cx, cy)
    end
    return F
end

"""
    marginal_inconsistency(cache::CTMEnvironmentCache) -> Real

How far the cache is from a genuine CVM/BP fixed point, as `mean(1 − cos(M_v, M_e))` over the
edge-like blocks. **This is the only `ln Z`-free quality measure available, and the only one safe
to optimise against.**

Each edge-like block sits in exactly two regions with Möbius weights `+1, −1`, so `Z_R` being
linear in it gives `∂F/∂B = M_v/Z_v − M_e/Z_e`, which vanishes iff `M_v ∥ M_e` — the parent and
child marginals are parallel. At bond dimension 1 this *is* the BP fixed-point equation (`M_e` is
the reverse message, `M_v` the vertex factor times the other incoming messages), and `F` is the
Bethe/Kikuchi free energy.

Measured behaviour on 4×4 D=3: exactly `0` at lossless χ, 1.1e-5 at χ=8, 3.9e-4 at χ=6, 8.0e-3 at
χ=4 — i.e. it tracks the truncation error, and it plateaus at the same sweep `F` does.

Do **not** use `|F − ln Z|` to judge changes to this algorithm. Measured three independent ways,
its apparent gains are cancellation artifacts of the signed Möbius sum: the swap that improves it
4.3× simultaneously degrades this diagnostic 0.64×, degrades single-site observables, and degrades
the stationarity residual.
"""
function marginal_inconsistency(cache::CTMEnvironmentCache)
    env = _ctm_env(cache)
    nxt = sweep_vertex_environments(cache, env)      # its PH/PV are consistent with `env`
    tbl = _ctm_factor_table(cache)
    Lx, Ly = _ctm_dims(cache)
    memo = Dict{Any, Any}()
    blk(d) = get!(memo, d) do
        _ctm_block(env, tbl, nxt, d)
    end
    # descriptors only: this diagnostic is weight-free, it just needs the two regions a block
    # sits in and whether they carry a centre site
    regs = [_ctm_region_desc(cx, cy) for cx in 1.0:0.5:Lx for cy in 1.0:0.5:Ly]
    gaps = Float64[]
    for sym in (:N, :S, :W, :E), i in 1:(Lx + 1), j in 1:(Ly + 1)
        d = (:T, sym, i, j)
        isnothing(blk(d)) && continue
        rs = filter(r -> d in r[1], regs)
        length(rs) == 2 || continue
        Ms = ITensor[]
        for (ds, _, ctr) in rs
            full = ITensor[]; minus = ITensor[]
            for e in ds
                t = blk(e); isnothing(t) && continue
                push!(full, t); e == d || push!(minus, t)
            end
            if !isnothing(ctr) && haskey(tbl, ctr)
                append!(full, tbl[ctr]); append!(minus, tbl[ctr])
            end
            (isempty(full) || isempty(minus)) && continue
            Z = try scalar(_ctm_contract(full)) catch; continue end
            (!isfinite(Z) || iszero(Z)) && continue
            push!(Ms, _ctm_contract(minus) / Z)
        end
        length(Ms) == 2 || continue
        a, b = Ms
        Set(inds(a)) == Set(inds(b)) || continue
        is = inds(a)
        va = vec(Array(a, is...)); vb = vec(Array(b, is...))
        na = norm(va); nb = norm(vb)
        (iszero(na) || iszero(nb)) && continue
        # clamp: `cos` can marginally exceed 1 in roundoff, and this is a distance
        push!(gaps, max(zero(Float64), 1 - abs(dot(va, vb)) / (na * nb)))
    end
    return isempty(gaps) ? 0.0 : sum(gaps) / length(gaps)
end

"""
    update(cache::CTMEnvironmentCache; maxiter = 30, tolerance = 1e-10, verbose = false)

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
function update(cache::CTMEnvironmentCache; maxiter::Integer = 30,
                tolerance::Real = 1.0e-10, verbose::Bool = false)
    env = _ctm_env(cache)
    F = cvm_freenergy(env, cache)
    converged, Δ = false, Inf
    for it in 1:maxiter
        prev = env
        env = sweep_vertex_environments(cache, env)
        Fnew = cvm_freenergy(env, cache)
        Δ = abs(Fnew - F)
        F = Fnew
        # Use the state distance alongside `|ΔF|`: the latter oscillates at the roundoff floor
        # of a signed log-sum and can RISE late in the iteration, so on its own it both stops
        # early and fails to certify convergence. See `_ctm_statedist`.
        #
        # Compare `sd²`, not `sd`. `F` is stationary in the state at the fixed point, so
        # `|ΔF| ~ sd²` — holding both to the same tolerance is dimensionally inconsistent and
        # measured ~3x the sweeps for no accuracy (5×5 D=2 χ=8: 30 sweeps / 21 s against 11
        # sweeps / 2.2 s, same `F` to 12 digits). This is equivalent to `sd ≤ √tolerance`.
        sd = CTM_GAUGE[] ? _ctm_statedist(env, prev) : nothing
        crit = isnothing(sd) ? Δ : max(Δ, sd^2)
        verbose && @info "CVM sweep $it: F = $F, |ΔF| = $Δ, state Δ = $(something(sd, NaN))"
        if crit ≤ tolerance * max(one(crit), abs(F))
            converged = true
            verbose && @info "CVM sweep converged after $it sweeps."
            break
        end
    end
    if !converged
        msg = "CVM sweep did not converge to tolerance $tolerance after $maxiter sweeps " *
              "(final |ΔF| = $Δ)."
        verbose ? println(msg) : @warn(msg)
    end
    return _ctm_setenv(cache, env)
end
