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
# BOTH bounding corners, via a thin QR of each block and one SVD of the small triangular product —
# never squaring, and batching well on GPU (see `_ctm_twosided_projector_qr`). The pair needs the
# complement environment, so the build is a fixed-point iteration: `update` sweeps it to
# stationarity. Works for anisotropic / non-square grids and free boundaries.
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
using Random: Xoshiro
using KrylovKit: eigsolve, svdsolve, schursolve, Arnoldi

"""
    CTMOptions(; kwargs...)

Numerical strategy for a [`CTMEnvironmentCache`](@ref). Carried BY the cache, so every derived
quantity — `update`, `cvm_freenergy`, `expect`, `rdm` — uses the same settings the cache was
built with, and two caches with different settings can coexist. Pass them as keywords to the
cache constructor: `CTMEnvironmentCache(tn, 8; degtol = 1e-9)`.

The defaults are the measured-best route; each field's rationale lives next to the code it
governs, referenced below.

| field | default | what it selects |
|---|---|---|
| `gauge` | `true` | fix the projector pair's gauge to the previous sweep by orthogonal Procrustes, making iterates comparable — see `_ctm_align`. Prerequisite for any accelerator. |
| `arnoldi` | `true` | use Krylov (Arnoldi/Lanczos) for top-`k` eigen/singular triplets where it pays; `false` forces dense — see `_ctm_eigsolve`. |
| `degtol` | `0.0` | relative gap below which a truncation is judged to split a near-degenerate multiplet, and is backed off. `0` disables. Matters for double-layer corners (ket↔bra exchange gives `λ_ij = λ_ji`). |
| `qr_cutoff` | `1e-13` | relative cutoff on the `S` values the projector inverts. It can sit this low because `S` comes off a triangular product rather than a squared object — see `_ctm_twosided_projector_qr`. |
| `krylov_min` | `128` | smallest interface at which the Krylov SVD beats a dense one; below it dense LAPACK wins — see `_ctm_svd_topk` for the crossover. |
| `optimal_max` | `12` | tensor count above which contraction-order search falls back from exhaustive netcon to greedy. A FEASIBILITY gate — see `_ctm_contract`. |
| `projector` | `:cut` | which interface projector to derive: `:cut` (optimal rank-χ truncation of one bipartition) or `:cycle` (four-corner cycle, which makes `F` stationary). See "Choosing a projector" below. |

## Choosing a projector

Both are two-sided and biorthogonal, both land on the same interface keys, both are exact at lossless
χ, and both are PURE — no per-interface mixing (see `sweep_vertex_environments`). They differ in what
they optimise:

* `:cut` optimises **each interface in isolation** — the best rank-χ truncation of that one
  bipartition. It is not a stationary point of `F`, so `marginal_inconsistency` stays nonzero.
* `:cycle` enforces consistency **around the plaquette**, which IS stationarity of `F`, i.e. marginal
  consistency — the same condition BP satisfies, and what a single-region observable ratio needs.

On the collaborator's 5×5 Ising PEPS (D=3), `⟨X⟩` error against an exact contraction, with their
engine measured here through `contract_Z11` at matched χ:

| χ | `:cut` | `:cycle` | their engine | `marg`, `:cut` → `:cycle` |
|---|---|---|---|---|
| 4 | 1.52e-04 | **4.77e-05** | 5.22e-05 | 2.0e-07 → 1.8e-04 |
| 9 | 4.24e-07 | **5.13e-08** | 5.13e-08 | 2.9e-11 → **3.0e-16** |
| 16 | 6.26e-09 | **9.28e-10** | 9.28e-10 | 2.6e-14 → **2.4e-16** |
| 32 | 7.39e-12 | **4.86e-14** | 1.33e-12 | 6.4e-17 → **1.8e-16** |

`:cycle` beats `:cut` at every χ here (3.2× / 8.3× / 6.7× / 152×) and matches or beats their engine
at every χ, by 27× at χ=32. It is stationary to machine precision for χ ≥ 9 — but NOT at χ=4, where a
fully resolved rank-4 invariant subspace is a worse stationary point than an under-resolved one. That
is a property of the criterion at severe truncation, not a bug; restricting `krylovdim` fixes it at
χ=4 but costs accuracy at χ=8, so it is deliberately not done (see docs/ctmrg_status.md).

`:cycle` is also markedly CHEAPER: at 8×8 D=3 it is 15.7× faster per sweep than `:cut` for identical
retained dimensions, because it is matrix-free where `:cut` forms dense QR factors of the enlarged
corners.

⚠️ **This is one physical state. On random states the picture is far more mixed** — measured over 6
seeds per cell, `:cycle` is a large win on hex (median 86× at χ=4, 531× at χ=8) and roughly a coin
flip on square lattices, with seed-to-seed ranges spanning 3–4 orders. It also has one known failure
regime: hex at the χ where the environment becomes lossless. Do not choose between these on a single
configuration; see `docs/ctmrg_status.md` for the multi-seed tables and the failure analysis.

`:cut` remains the default because it is cheaper, has no known failure regime, and is the
longer-tested path.

"""
Base.@kwdef struct CTMOptions
    gauge::Bool = true
    arnoldi::Bool = true
    degtol::Float64 = 0.0
    qr_cutoff::Float64 = 1.0e-13
    krylov_min::Int = 128
    optimal_max::Int = 12
    projector::Symbol = :cut

    function CTMOptions(gauge, arnoldi, degtol, qr_cutoff, krylov_min, optimal_max,
                        projector)
        projector in (:cut, :cycle) || throw(ArgumentError(
            "projector must be :cut or :cycle, got $(repr(projector))"))
        return new(gauge, arnoldi, degtol, qr_cutoff, krylov_min, optimal_max, projector)
    end
end

"""
    CTMEnvironmentCache(tn::AbstractTensorNetwork, maxdim::Integer; kwargs...)

Position-resolved CTMRG environment for a 2D grid `TensorNetwork` (vertices `(x, y)`): a
`4C + 4T` ring on every vertex, with each shared interface truncated to `maxdim` by a two-sided
(biorthogonal) projector pair.

A freshly built cache carries **no** per-vertex CVM environments. [`update`](@ref) runs the
two-sided stationary sweep and returns a cache holding the converged ones; [`cvm_freenergy`](@ref)
and [`region_lnZ`](@ref) then read them off.

Evaluating an un-updated cache falls back to the greedy single pass
([`vertex_environments`](@ref)) rather than erroring, but **warns**, because that pass is a
different algorithm — 3–4 orders less accurate and non-monotone in `maxdim`, so the number will
not improve when you raise χ. `update` first. To ask for the greedy pass deliberately, pass its
environments explicitly (`cvm_freenergy(vertex_environments(cache), cache)`); that is silent.

Keyword arguments set the numerical strategy and are stored on the cache; see
[`CTMOptions`](@ref) for the list.
"""
struct CTMEnvironmentCache{V, N, E}
    network::N
    grid::Dict{Tuple{Int, Int}, V}   # OCCUPIED positions only — holes allowed (hex, heavy-hex)
    dims::Tuple{Int, Int}            # bounding box (Lx, Ly)
    maxdim::Int
    environments::E                  # `nothing`, or the CVM blocks from `update`
    options::CTMOptions              # numerical strategy, fixed at construction
end

network(cache::CTMEnvironmentCache) = cache.network
graph(cache::CTMEnvironmentCache) = graph(network(cache))

"""
    environments(cache::CTMEnvironmentCache)

The cache's per-vertex CVM environments, or `nothing` if it has not been [`update`](@ref)d.
"""
environments(cache::CTMEnvironmentCache) = cache.environments

"""
    options(cache::CTMEnvironmentCache)

The [`CTMOptions`](@ref) the cache was built with.
"""
options(cache::CTMEnvironmentCache) = cache.options

# Works for a single-layer `TensorNetwork`, a `TensorNetworkState` (⟨ψ|ψ⟩) or an
# `AbstractForm` (⟨ψ|O|ψ⟩) — all of them expose their per-vertex tensors through
# `bp_factors`, which is how the double layer is kept LAZY.
function CTMEnvironmentCache(net, maxdim::Integer; kwargs...)
    opts = CTMOptions(; kwargs...)
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
    return CTMEnvironmentCache(net, grid, (Lx, Ly), Int(maxdim), nothing, opts)
end

# Same network/grid/maxdim/options, different CVM environments.
_ctm_setenv(cache::CTMEnvironmentCache, env) =
    CTMEnvironmentCache(cache.network, cache.grid, cache.dims, cache.maxdim, env, cache.options)

# --- the move --------------------------------------------------------------------
# `opts.degtol` — relative cutoff gap below which the truncation is judged to split a
# (near-)degenerate multiplet; the cut is then backed off to a real gap. Matters for
# DOUBLE-LAYER networks, whose corner spectra carry systematic 2-fold degeneracies from
# ket↔bra exchange (λ_ij = λ_ji). 0 disables it.
#
# `opts.arnoldi` — only the top `maxdim` eigenpairs are needed, so for a corner much larger
# than `maxdim` an Arnoldi/Lanczos solve (KrylovKit) costs O(maxdim·n²) instead of dense
# eigen's O(n³). `false` forces dense. Falls back to dense if Krylov does not converge.
#

# THE interface projector, via a TRIANGULAR (QR) factorization of each block.
#
# There used to be a second, `ρ`-based route (`ρ_L = A†A`, `ρ_R = B†B`, eigendecomposing `ρ_R` back
# into a square-root factor) selected by a `qr` option, kept as the long-standing reference path.
# It is GONE. It was sesquilinear by construction — it needs `ρ = A†A` Hermitian PSD to have a
# square root at all — while the sweep contracts the corners bilinearly, so it was simply wrong for
# complex tensors and could not be repaired without replacing its machinery (`Aᵀ A` is complex
# SYMMETRIC and has no PSD root). Its value had been as an independent cross-check, and that job is
# now done better by the full-rank identity test below, which checks the pair against what it is
# supposed to satisfy rather than against another implementation that might be wrong the same way.
# Its `pinv_cutoff` option went with it; `opts.qr_cutoff` is now the only cutoff.
#
# THE PAIRING IS BILINEAR, NOT SESQUILINEAR. The sweep contracts the two enlarged corners plainly:
# `Bw * Be` conjugates nothing. So with `A`, `B` the blocks as (rest × interface) matrices, the
# object the pair must preserve is `A Bᵀ` — TRANSPOSE, not adjoint. Getting this wrong is invisible
# on real tensors and catastrophic on complex ones: the earlier conjugated version optimised
# `A B†` and was 11% off its own full-rank identity on a complex 4×4, so every truncation sat in
# the wrong subspace and raising χ never helped.
#
#   A = Q_A R_A,  B = Q_B R_B   (thin QR, no conjugation)
#   A Bᵀ = Q_A (R_A R_Bᵀ) Q_Bᵀ           ⇒   W = R_A R_Bᵀ
#
# `Q_A† Q_A = I` and `Q_Bᵀ (Q_Bᵀ)† = I`, so `W`'s singular values ARE those of `A Bᵀ` and the
# truncation is optimal for the product the network actually forms. With `W = U S V†`:
#
#   P_A = R_Bᵀ V S^(-1/2)            P_B = S^(-1/2) U† R_A
#
# giving `R_A P_A P_B R_Bᵀ = U S^(1/2) · S^(1/2) V† = W`, i.e. `A (P_A P_B) Bᵀ = A Bᵀ` exactly at
# full rank — the identity the regression test asserts directly, since it is what caught the
# sesquilinear bug. Note the symmetric `S^(-1/2)` on both sides: no worse inverse power appears.
#
# It uses ONE svd of a small triangular product, so U and V come from a single decomposition in a
# consistent basis, which is what keeps degenerate clusters from picking up a relative rotation.
# It is NOT an svd of a squared object: `S` is resolved to ~eps relatively rather than ~√eps, which
# is why `opts.qr_cutoff` can sit at 1e-13.
#
# WHY QR AND NOT AN EIGENDECOMPOSITION — GPU / BATCHING, NOT ACCURACY. Measured accuracy-NEUTRAL
# against the (now removed) ρ route: on 18 moderate-χ configurations (3 seeds single-layer D=3, 2
# double-layer D=2, 1 double-layer D=3; χ = 4/6/8) and 10 near-lossless ones it matched to 3
# significant figures, at cutoffs 1e-8/1e-11/1e-13/1e-15 alike. Precision is not the binding
# constraint: the RETAINED spectrum has median `S_k/S_1` of 1e-1…1e-2 (measured over 200–384 solves
# per sweep) and 0% of retained directions fall below 1e-8. **χ is the binding constraint, not
# arithmetic** — and as the removed-symmetry section of the design doc records, that extends to
# structure too, not just precision. The win is that geqrf/gesvd have batched GPU implementations
# where batched Hermitian eig support is thin, and a sweep is 200–384 INDEPENDENT tiny
# factorizations (n ≤ 128) — a batching problem, not a big-linear-algebra one.
#
# Remaining GPU blocker: `_ctm_block_matrix` materialises a host `Array`, so every projector
# round-trips through the CPU. Fixing that is separate work.

# Deterministic start vector for every Krylov solve in this file.
#
# KrylovKit draws its own start from the GLOBAL RNG when none is supplied, which makes the whole
# engine irreproducible run to run — measured on the 5×5, `⟨X⟩` at χ=16 wandered over
# 8.1e-10 – 9.3e-10 and `|F − ln Z|` at χ=32 over 8.9e-16 – 6.2e-15. That is fatal for regression
# testing and for comparing two projectors, where the difference under test can be smaller than the
# run-to-run spread. Seeding locally from the problem's shape makes each solve reproducible AND
# leaves the caller's global stream untouched (a CTM sweep must not perturb a user's `Random.seed!`).
_ctm_startvec(T, n::Integer, tag) = randn(Xoshiro(hash(tag, hash(n))), T, n)

# Top-`k` eigenpairs of a Hermitian matrix: Krylov when it pays off, else dense.
function _ctm_eigsolve(ρs::Hermitian, k::Integer, opts::CTMOptions)
    n = size(ρs, 1)
    if opts.arnoldi && n > 4k
        try
            v0 = _ctm_startvec(eltype(ρs), n, :eig)
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

function _ctm_eig_projector(ρ::ITensor, bnd::Index, maxdim::Integer, opts::CTMOptions)
    bp = prime(bnd)
    ρm = Array(ρ, bnd, bp)
    ρs = Hermitian((ρm + ρm') / 2)
    vals, vecs = _ctm_eigsolve(ρs, Int(maxdim), opts)
    order = sortperm(vals; rev = true)
    sv = vals[order]
    k = min(Int(maxdim), length(sv), size(vecs, 2))
    while k > 1 && k < length(sv) && abs(sv[k] - sv[k + 1]) ≤ opts.degtol * abs(sv[k])
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

# Above `opts.optimal_max` tensors, fall back to the greedy optimiser. This is a FEASIBILITY
# gate, not a performance one: `alg = "optimal"` is ExhaustiveSearch netcon, exponential in the
# number of tensors, and it hangs outright on the ~25-tensor lists a `vertex_window` observable
# produces. (Tried and reverted as a *perf* tweak earlier — it bought ~1.5% on sweep-sized lists.)
#
# The gate's verdict joins the cache key: `optimal_max` is per-cache, so two caches sharing a
# lattice shape would otherwise trade sequences and each get whichever optimiser ran first.
function _ctm_contract(ts::Vector{ITensor}, opts::CTMOptions)
    length(ts) == 1 && return only(ts)
    length(ts) == 2 && return ts[1] * ts[2]          # no sequence to choose
    use_optimal = length(ts) <= opts.optimal_max
    seq = get!(CTM_SEQ_CACHE, (_ctm_seq_key(ts), use_optimal)) do
        use_optimal ?
            contraction_sequence(ts; alg = "optimal") :
            contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod())
    end
    return contract(ts; sequence = seq)
end



# A block as a plain (rest × interface) matrix. NO conjugation.
#
# It used to conjugate, so that `R†R` reproduced this file's Hermitian `ρ` convention. That was
# wrong for complex networks and invisible for real ones: the sweep contracts the two enlarged
# corners PLAINLY (`Bw * Be` conjugates nothing), so the projector must preserve the BILINEAR
# product `A Bᵀ`, not the sesquilinear `A B†`. Conjugating here optimised the wrong pairing and
# broke the pair's exactness at full rank by ~11% on a complex 4×4. See `_ctm_twosided_projector_qr`.
function _ctm_block_matrix(B::ITensor, io::Index)
    rest = collect(uniqueinds(B, io))
    isempty(rest) && return reshape(Array(B, io), 1, ITensors.dim(io))
    return reshape(Array(B, rest..., io), :, ITensors.dim(io))
end

# Triangular factor of a block: `R` with `A = Q R`, never forming `ρ`. See the projector note above.
_ctm_tri_factor(B::ITensor, io::Index) = Matrix(qr(_ctm_block_matrix(B, io)).R)

# Biorthogonal pair from the TRIANGULAR factors of the two bounding blocks — no squaring
# anywhere. See the projector note above for the derivation.
# Top-`k` singular triplets of `W`, by Golub–Kahan–Lanczos when it pays. `W` is `n×n` with
# `n = χ·D_layer` and only `k = maxdim` triplets are ever used, so a full dense SVD discards
# everything past column `k` — measured 45% of wall at 5×5 D=4 χ=12.
#
# GATE: `n ≥ opts.krylov_min` **and** `n > 4k`. The ratio test alone is not enough — measured
# crossover (dense vs `svdsolve`, ms per solve): n=72 0.62/0.50, n=96 1.06/3.31, n=128 2.13/1.11,
# n=192 5.70/2.49, n=256 12.22/3.95. So Krylov wins 1.9–3.1× from n≈128 and *loses* below it;
# gating on the ratio alone made 4×4 D=3 χ=8 (n≤72) 1.2× SLOWER. Falls back to dense on
# non-convergence; agreement with dense measured to 1.4e-15 on matrices captured from a real run.
function _ctm_svd_topk(W::AbstractMatrix, k::Integer)
    try
        x0 = _ctm_startvec(eltype(W), size(W, 2), :svd)
        vals, lvecs, rvecs, info = svdsolve(W, x0, k, :LR)
        (info.converged >= k && length(lvecs) >= k && length(rvecs) >= k) || return nothing
        U = reduce(hcat, @view(lvecs[1:k])); V = reduce(hcat, @view(rvecs[1:k]))
        eltype(W) <: Real && (U = real.(U); V = real.(V))
        return (; S = real.(vals[1:k]), U, V)
    catch
        return nothing                                # fall through to dense
    end
end

function _ctm_twosided_projector_qr(Bw::ITensor, Be::ITensor, io::Index, maxdim::Integer,
                                   opts::CTMOptions)
    RA = _ctm_tri_factor(Bw, io)
    RB = _ctm_tri_factor(Be, io)
    RBt = transpose(RB)                     # TRANSPOSE: the pairing is `A Bᵀ`, not `A B†`
    W = RA * RBt
    kw = min(Int(maxdim), min(size(W)...))
    nW = min(size(W)...)
    F = (opts.arnoldi && nW >= opts.krylov_min && nW > 4kw) ? _ctm_svd_topk(W, kw) : nothing
    isnothing(F) && (F = svd(W))            # ONE decomposition → consistent U, S, V
    S = F.S
    k = min(Int(maxdim), length(S))
    while k > 1 && S[k] ≤ opts.qr_cutoff * S[1]
        k -= 1
    end
    while k > 1 && k < length(S) && abs(S[k] - S[k + 1]) ≤ opts.degtol * abs(S[k])
        k -= 1                                      # don't split a degenerate multiplet
    end
    Sk = S[1:k]; isk = Diagonal(1 ./ sqrt.(Sk))
    PAm = RBt * F.V[:, 1:k] * isk           # (bond × kept)
    PBm = isk * F.U[:, 1:k]' * RA           # (kept × bond)
    w = Index(k)
    return ITensor(PAm, io, w), ITensor(PBm, w, io), w
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
function _ctm_interface_proj(B, ins::Vector{<:Index}, maxdim::Integer, opts::CTMOptions)
    (isnothing(B) || isempty(ins)) && return nothing
    co = combiner(ins...); io = combinedind(co)
    d = ITensors.dim(io); k = min(Int(maxdim), d)
    if k == d                              # nothing to truncate: keep the basis intact
        w = Index(d)
        return ITensor(Matrix{Float64}(I, d, d), io, w) * co, w
    end
    Bc = B * co
    ρ = Bc * prime(dag(Bc), io)
    P, w = _ctm_eig_projector(ρ, io, k, opts)
    return P * co, w
end

# Grid geometry / lazy factors ----------------------------------------------------
_ctm_dims(cache::CTMEnvironmentCache) = cache.dims
# `nothing` at an unoccupied grid position.
_ctm_vertex(cache::CTMEnvironmentCache, x::Int, y::Int) = get(cache.grid, (x, y), nothing)

# Site factors at a grid position as a LIST — `[ket, bra]` (or `[ket, op, bra]`) for a double
# layer, `[a]` for a single one, empty if unoccupied.
#
# NEVER pre-contracted. `ket * bra` is the fat ket⊗bra site tensor the lazy double layer exists to
# avoid: for a 4-link D=3 vertex it is D^8 = 6561 entries against 162 per factor, and forming it
# also denies netcon the chance to interleave the two layers with the environment blocks. Every
# absorption below therefore passes this list straight into `_ctm_contract`.
_ctm_facs(tbl, x::Int, y::Int) = get(tbl, (x, y), ITensor[])

# Flatten mixed arguments — `ITensor`, `nothing`, or a factor list — into one contraction list.
function _ctm_list(args...)
    ts = ITensor[]
    for a in args
        isnothing(a) && continue
        a isa ITensor ? push!(ts, a) : append!(ts, a)
    end
    return ts
end

# ONE netcon over [core; extras], or `nothing` when the core is empty.
#
# `core` is the environment blocks and site factors; `extras` are isometries (projectors). The
# split matters only at the boundary: with no core there is nothing to absorb, and the extras are
# dropped rather than contracted on their own — which is what the `_ctm_mul`/`apA` chain this
# replaces did by short-circuiting on `nothing`.
#
# Putting the projectors in the SAME netcon call as the growth is the second half of the fix: the
# optimiser may now apply an isometry BEFORE the site factors, truncating an interface before it is
# grown rather than after.
function _ctm_absorb(opts::CTMOptions, core::Vector{ITensor}, extras...)
    isempty(core) && return nothing
    ts = copy(core)
    for e in extras
        isnothing(e) || push!(ts, e)
    end
    return _ctm_contract(ts, opts)
end

# `P_A` / `P_B` of a stored projector, or `nothing`. The greedy pass stores `(P, w)` so `p[1]` is
# its only isometry; the sweep stores `(P_A, P_B, w)`.
_ctm_pA(d, k) = (p = _ctm_nn(d, k); isnothing(p) ? nothing : p[1])
_ctm_pB(d, k) = (p = _ctm_nn(d, k); isnothing(p) ? nothing : p[2])
# `dag(P_A)`, which is how the east/south blocks of the GREEDY pass consume a projector derived on
# their west/north partner. (The sweep uses a genuine biorthogonal `P_B` instead.)
_ctm_pAdag(d, k) = (p = _ctm_pA(d, k); isnothing(p) ? nothing : dag(p))

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
    opts = cache.options
    tbl = _ctm_factor_table(cache)
    hl(x, y) = _ctm_links(tbl, (x, y), (x + 1, y))      # horizontal link cols x|x+1 at row y
    vl(x, y) = _ctm_links(tbl, (x, y), (x, y + 1))      # vertical link rows y|y+1 at col x
    a(x, y) = get(tbl, (x, y), ITensor[])

    C = Dict{Tuple{Symbol, Int, Int}, Any}()
    T = Dict{Tuple{Symbol, Int, Int}, Any}()
    PH = Dict{Tuple{Symbol, Int, Int}, Any}()
    PV = Dict{Tuple{Symbol, Int, Int}, Any}()

    # ---- W strips (y increasing, x increasing): derives PV[:W] ----
    # The growth is ONE netcon over [edge; ket; bra; incoming isometry]. `raw` must be materialised
    # before the interface projector below, since that projector is derived FROM it — but the
    # absorption itself no longer pre-contracts `ket * bra`.
    for y in 1:Ly, x in 1:(Lx - 1)
        raw = _ctm_absorb(opts, _ctm_list(_ctm_nn(T, (:W, x, y)), _ctm_facs(tbl, x, y)),
                          y > 1 ? _ctm_pA(PV, (:W, x + 1, y - 1)) : nothing)
        if y < Ly
            ins = Index[]
            w = _ctm_widx(PV, (:W, x, y)); !isnothing(w) && push!(ins, w)
            append!(ins, vl(x, y))
            pr = _ctm_interface_proj(raw, ins, χ, opts)
            if !isnothing(pr)
                PV[(:W, x + 1, y)] = pr
                raw = raw * pr[1]
            end
        end
        T[(:W, x + 1, y)] = _ctm_rescale(raw)
    end
    # ---- E strips (x decreasing): derives PV[:E] ----
    for y in 1:Ly, x in Lx:-1:2
        raw = _ctm_absorb(opts, _ctm_list(_ctm_facs(tbl, x, y), _ctm_nn(T, (:E, x + 1, y))),
                          y > 1 ? _ctm_pA(PV, (:E, x, y - 1)) : nothing)
        if y < Ly
            ins = Index[]
            append!(ins, vl(x, y))
            w = _ctm_widx(PV, (:E, x + 1, y)); !isnothing(w) && push!(ins, w)
            pr = _ctm_interface_proj(raw, ins, χ, opts)
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
        pr = _ctm_interface_proj(raw, ins, χ, opts)
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
        pr = _ctm_interface_proj(raw, ins, χ, opts)
        if !isnothing(pr)
            PH[(:S, x - 1, y)] = pr
            raw = raw * pr[1]
        end
        C[(:SW, x, y)] = _ctm_rescale(raw)
    end
    # ---- C[:NE] / C[:SE]: consume PH ----
    for x in 2:Lx
        for y in 1:(Ly - 1)
            C[(:NE, x, y + 1)] = _ctm_rescale(_ctm_absorb(opts,
                _ctm_list(_ctm_nn(C, (:NE, x, y)), _ctm_nn(T, (:E, x, y))),
                _ctm_pAdag(PH, (:N, x - 1, y + 1))))
        end
        for y in Ly:-1:2
            C[(:SE, x, y)] = _ctm_rescale(_ctm_absorb(opts,
                _ctm_list(_ctm_nn(C, (:SE, x, y + 1)), _ctm_nn(T, (:E, x, y))),
                _ctm_pAdag(PH, (:S, x - 1, y))))
        end
    end
    # ---- N / S column strips: consume PH ----
    # One netcon over [edge; ket; bra; both isometries] — the site factors stay a list and the
    # optimiser is free to truncate either interface before growing.
    for x in 1:Lx
        for y in 1:(Ly - 1)
            T[(:N, x, y + 1)] = _ctm_rescale(_ctm_absorb(opts,
                _ctm_list(_ctm_nn(T, (:N, x, y)), _ctm_facs(tbl, x, y)),
                _ctm_pAdag(PH, (:N, x - 1, y + 1)), _ctm_pA(PH, (:N, x, y + 1))))
        end
        for y in Ly:-1:2
            T[(:S, x, y)] = _ctm_rescale(_ctm_absorb(opts,
                _ctm_list(_ctm_facs(tbl, x, y), _ctm_nn(T, (:S, x, y + 1))),
                _ctm_pAdag(PH, (:S, x - 1, y)), _ctm_pA(PH, (:S, x, y))))
        end
    end
    return CTMVertexEnvironments(C, T, PH, PV, Lx, Ly)
end

# Biorthogonal (two-sided) projector pair for the interface shared by two complementary
# enlarged corners. Returns (P_A, P_B, w): P_A goes on the west/north block, P_B on the
# east/south one, so every contraction across the interface pairs one with the other.
function _ctm_interface_proj2(Bw, Be, ins::Vector{<:Index}, maxdim::Integer, opts::CTMOptions)
    (isnothing(Bw) || isnothing(Be) || isempty(ins)) && return nothing
    co = combiner(ins...); io = combinedind(co)
    PA, PB, w = _ctm_twosided_projector_qr(Bw * co, Be * co, io, maxdim, opts)
    return PA * co, PB * co, w
end


# --- CYCLE projector (`opts.projector === :cycle`) -------------------------------------
#
# WHY. The cut projector optimises ONE bipartition per interface, independently. That is not a
# stationary point of `F` — `marginal_inconsistency` measures exactly that residual. Deriving all
# four of a plaquette's projectors from the dominant invariant subspace of the four-corner cycle
# enforces consistency AROUND the loop, which IS stationarity, i.e. marginal consistency, which is
# what a single-region observable ratio needs. Worth 3–8× on single-site observables where χ binds.
#
# GEOMETRY. With bonds ordered (W, S, E, N), each enlarged corner maps one bond to the next:
#   A1 = E_SW : W->S    A2 = E_SE : S->E    A3 = E_NE : E->N    A4 = E_NW : N->W
# so `M = A4 A3 A2 A1` acts on the west bond. The four projectors land on the EXISTING keys
# PH[:N,X-1,Y], PH[:S,X-1,Y], PV[:W,X,Y-1], PV[:E,X,Y-1] — only the derivation changes, never the
# consumers. Left bases propagate DOWNWARD, `V_L[l] ∝ V_L[l+1] A_l`. Per bond, the factor on the
# CONSUMING tensor is the right basis and the one on the producer is the left, so against our
# west/north = `P_A` convention W and S take `P_A = V_L` while E and N take `P_A = V_R`.
#
# NO PADDING. Bonds are rectangular in general (`k_prev · D_layer`, and `k_prev = 1` at the
# boundary). Their engine pads everything to a fixed χ with a separate `rank` field because a DENSE
# periodic Schur needs square equal-size factors. We only ever need the ACTION of the cycle on a
# vector — four matvecs, product never formed — so a matrix-free `schursolve` handles adaptive bonds
# natively and returns a real orthonormal basis, sidestepping the conjugate-pair handling a dense
# `ordschur` fails silently on.
#
# RANK, AND THE HONEST DOMAIN LIMIT. `schursolve` terminates when its Krylov space closes, so the
# RESOLVED cycle rank `kres` can fall well short of what a bond could hold: the four-fold product's
# spectrum is roughly the 4th power of one corner's, and on a bottlenecked loop `kcyc` itself is
# capped by the narrowest bond. Directions past `kres` carry no cycle weight and the criterion says
# nothing about them, so this projector is used ONLY where `kres` already fills every bond's target
# (the gate below). Elsewhere the plaquette declines to the cut.
#
# That gate is a real restriction, and lifting it is the open problem. Filling the shortfall was
# tried two ways and BOTH are destructive on random states, in the pattern the retracted union
# showed — `⟨Z⟩` error rising with χ, the systematic-error signature:
#
#   filler                     heavy-hex 2×2 D=2, χ = 2 / 8 / 32        (cut: 5.8e-05 / 1.1e-16 / 1.1e-16)
#   random (their _stochastic_expand_range)   —                          cold-started here, re-drawn
#                                                                        every sweep, never settles
#   deflated cut directions    4.2e-08 / 3.5e-06 / 3.2e-04               DEGRADES with χ
#
# The deflated-cut filler is excellent on the collaborator's 5×5 Ising PEPS (χ=32 `⟨X⟩` 3.9e-12
# against the cut's 7.4e-12, their engine's 1.3e-12) and catastrophic on sparse grids, so it is not
# landed. Diagnosed mechanism: the merged pair becomes severely ill-conditioned — relative leakage of
# `M v` outside the kept space reaches 6e12, i.e. `a b` stops being a projector at all — which is the
# `S^(-1/2)` amplification `qr_cutoff` guards against, not a failure of invariance. A filler chosen
# INSIDE the numerical null space of `M` (minimising `‖M v‖` over the deflated cut span) would keep
# invariance without that amplification and is the thing to try next; it is not yet validated.
#
# CONSEQUENCE, stated plainly: `F` is stationary only where the gate passes. On the 5×5 that is
# χ ≤ 16 (`marginal_inconsistency` 3.3e-16 at χ=9, 8.7e-15 at χ=16); at χ=32 most plaquettes decline
# and the result equals the cut. Full stationarity at every χ and every lattice is NOT yet achieved.
#
# RELATION TO THE RETRACTED UNION. That also merged cut directions in and was withdrawn as
# unreliable; two things differ and both matter. (1) It seeded the cycle block at
# `min(χ, narrowest bond)` — 9 at χ=32 — forcing in 9 cycle directions and taking 23 from the cut.
# Here the cycle block is the rank the eigensolve actually RESOLVED, so the cut only ever supplies
# directions the cycle genuinely does not span. (2) It was never exercised at the χ where it was
# judged: it demanded the full requested rank and so DECLINED every interior plaquette at χ=32,
# falling back to the cut — which is why later removing the union appeared to change nothing.
#
# DOMAIN. A hex/heavy-hex plaquette can have a bond of dimension 1 (a missing lattice link), which
# pins `kcyc = 1` at every χ. Such plaquettes decline and fall back to the cut rather than
# collapsing the interface; their engine never meets this, requiring all four T-space dims equal.
#
# DETERMINISM. `schursolve` needs a start vector, drawn from an RNG seeded PER PLAQUETTE from its
# position and bond dimensions; the fill is deterministic already. Without this the sweep is
# irreproducible run to run (measured: ⟨X⟩ at χ=16 wandered over 8.1e-10 – 9.3e-10, which is larger
# than the gap between the two projectors) and useless as a regression target. The local RNG also
# leaves the caller's global stream untouched.
_ctm_orthcols(X, k) = (Q = Matrix(qr(X).Q); Q[:, 1:min(k, size(Q, 2))])

# Whiten a pair so that `B A = I`, via the SVD of their overlap.
#
# The overlap MUST be truncated, not merely floored at `eps`: near-null overlap directions get
# multiplied by `S^(-1/2)`, amplifying pure noise — the same failure `qr_cutoff` guards against in
# the cut projector. Dropping below `cutoff · S[1]` shrinks `k` instead.
function _ctm_biorth(A::AbstractMatrix, B::AbstractMatrix, cutoff::Real)
    F = svd(B * A)
    k = count(>(cutoff * (isempty(F.S) ? one(eltype(F.S)) : F.S[1])), F.S)
    k < 1 && return nothing
    isq = Diagonal(1 ./ sqrt.(F.S[1:k]))
    return A * F.V[:, 1:k] * isq, isq * F.U[:, 1:k]' * B
end

function _ctm_cycle_projectors(ENW, ENE, ESE, ESW, maxdim::Integer, opts::CTMOptions,
                               seed::UInt)
    any(isnothing, (ENW, ENE, ESE, ESW)) && return nothing
    ins = (collect(commoninds(ENW, ESW)), collect(commoninds(ESW, ESE)),
           collect(commoninds(ENE, ESE)), collect(commoninds(ENW, ENE)))     # W, S, E, N
    any(isempty, ins) && return nothing
    cs = ntuple(l -> combiner(ins[l]...), 4)
    io = ntuple(l -> combinedind(cs[l]), 4)
    As = try
        [Array((ESW * cs[1]) * cs[2], io[2], io[1]), Array((ESE * cs[2]) * cs[3], io[3], io[2]),
         Array((ENE * cs[3]) * cs[4], io[4], io[3]), Array((ENW * cs[4]) * cs[1], io[1], io[4])]
    catch
        return nothing                          # a corner carrying more than its two interfaces
    end
    nsp = [ITensors.dim(io[l]) for l in 1:4]
    kcyc = min(Int(maxdim), minimum(nsp))
    kcyc < 1 && return nothing
    # Seeded on plaquette POSITION only, so the start vector is bit-identical every sweep. Seeding
    # on the bond dimensions too let it move whenever a rank shifted, which showed up as sweep-to-
    # sweep basis wander (state distance floor 1e-10 rather than 3e-11).
    v0 = randn(Xoshiro(seed), eltype(As[1]), nsp[1])
    # SCALE-FREE TOLERANCE — this was the algorithm's accuracy floor, worth ~1000× at χ=32.
    #
    # KrylovKit's `tol` is ABSOLUTE on the residual, and the cycle spectrum is the PRODUCT of the four
    # factors' spectra, so it spans ~14 orders: measured on the 5×5 at χ=32, `s_k/s_1` runs
    # 1 → 4.4e-09 (k=10) → 4.0e-12 (k=22) → 4.2e-14 (k=32), against a per-factor `s_32/s_1` of ~5e-04
    # whose fourth power is ~3.5e-14. A fixed `tol = 1e-13` therefore sat ABOVE the eigenvalues being
    # resolved: Arnoldi declared an invariant subspace at k≈19-22 while directions out to k=32 were
    # still orders above machine epsilon, and the projector silently lost them.
    #
    # Normalising the action by its dominant singular value (five power iterations — the invariant
    # subspace is scale-invariant, so it is free) makes the tolerance relative. Measured `⟨X⟩` at
    # χ=32: tol 1e-13 → 5.2e-11, 1e-15 → 7.6e-13, 1e-16 → 4.9e-14.
    #
    # Do not try to make `tol` χ-adaptive. Varying it alone changes NOTHING at χ=4 (identical at
    # 1e-13/1e-14/1e-15/1e-16), and tying it to `s_kcyc/s_1` via a loose first pass collapses χ=32 to
    # 9.0e-09, because a loose pass cannot resolve 32 eigenvalues and so reads the tail off the wrong
    # one. The χ=4 cost that remains is the criterion, not the solver — see the docstring.
    scale = let v = v0 / max(norm(v0), eps()), sc = one(real(eltype(As[1])))
        for _ in 1:5
            w = As[4] * (As[3] * (As[2] * (As[1] * v)))
            nw = norm(w)
            (isfinite(nw) && nw > 0) || break
            v, sc = w / nw, nw
        end
        sc > 0 && isfinite(sc) ? sc : one(sc)
    end
    fwd(v) = (As[4] * (As[3] * (As[2] * (As[1] * v)))) / scale
    bwd(u) = (transpose(As[1]) * (transpose(As[2]) *
              (transpose(As[3]) * (transpose(As[4]) * u)))) / scale
    alg = Arnoldi(; krylovdim = max(4kcyc + 8, 24), tol = 1.0e-16)
    local VRv, VLv, iR, iL
    try
        _, VRv, _, iR = schursolve(fwd, v0, kcyc, :LM, alg)
        _, VLv, _, iL = schursolve(bwd, v0, kcyc, :LM, alg)
    catch
        return nothing                          # fall through to the pairwise cut
    end
    # Solve the cycle at the rank it can actually RESOLVE. `schursolve` terminates when the Krylov
    # space closes, which at an interior plaquette is ~19 of a requested 32: the four-fold product's
    # spectrum is ~the 4th power of one corner's, so directions past that carry no cycle weight.
    kres = min(kcyc, iR.converged, iL.converged, length(VRv), length(VLv))
    kres < 1 && return nothing
    VR = Vector{Any}(undef, 4); VL = Vector{Any}(undef, 4)
    VR[1] = reduce(hcat, VRv[1:kres]); VL[1] = permutedims(reduce(hcat, VLv[1:kres]))
    for l in 1:3
        VR[l + 1] = _ctm_orthcols(As[l] * VR[l], kres)
    end
    for l in (4, 3, 2)
        VL[l] = permutedims(_ctm_orthcols(transpose(As[l]) * permutedims(VL[mod1(l + 1, 4)]), kres))
    end
    # Each bond keeps what it can support. Forcing all four to the plaquette's narrowest instead —
    # which is what their engine's `rank` field reports — measured immaterial (3.883e-12 against
    # 3.885e-12 on the 5×5 at χ=32), so it is not a knob. NOTE that comparison was made in an earlier
    # configuration and has not been re-checked since; it is a "do not bother" note, not a result.
    target(l) = min(Int(maxdim), nsp[l])
    # ZERO-PAD the retained index to a uniform width instead of letting it track `kres`.
    #
    # This is NOT an accuracy device: the padded columns are exactly zero, so `Π = P_A P_B` still has
    # rank `kres` and every region value is identical to simply shrinking. What it buys is a STABLE
    # INDEX DIMENSION. `kres` fluctuates from sweep to sweep and from plaquette to plaquette (measured
    # 1-4 on heavy-hex, 18-22 on the 5×5 interior), and every such change resizes the interface,
    # which breaks `_ctm_align`'s dimension guard, discards the gauge, and hands the next sweep a
    # basis it cannot compare with the last. That is the instability underneath the whole cycle route.
    # Their engine gets uniform widths for free from fixed-χ storage plus an explicit `rank` field;
    # this is the same trick, and it is bookkeeping rather than physics.
    #
    # The padding must be applied AFTER `_ctm_biorth`, never before: whitening a pair with null
    # columns inverts a singular overlap, which is the `S^(-1/2)` amplification `qr_cutoff` guards
    # against. Build the pair at `kres`, then embed.
    out = Vector{Any}(undef, 4)
    for l in 1:4
        (size(VR[l], 2) == kres && size(VL[l], 1) == kres) || return nothing
        Acol = (l <= 2) ? permutedims(VL[l]) : VR[l]         # (dim x kres), the P_A side
        Brow = (l <= 2) ? permutedims(VR[l]) : VL[l]         # (kres x dim), the P_B side
        ab = _ctm_biorth(Acol, Brow, opts.qr_cutoff)
        isnothing(ab) && return nothing
        a, b = ab                                            # b * a = I exactly
        (all(isfinite, a) && all(isfinite, b)) || return nothing
        kt = target(l)
        if size(a, 2) < kt                              # embed at rank, pad the rest with zeros
            T = eltype(a)
            a = hcat(a, zeros(T, size(a, 1), kt - size(a, 2)))
            b = vcat(b, zeros(T, kt - size(b, 1), size(b, 2)))
        end
        w = Index(size(a, 2))
        out[l] = (ITensor(a, io[l], w) * cs[l], ITensor(b, w, io[l]) * cs[l], w, ins[l])
    end
    return (W = out[1], S = out[2], E = out[3], N = out[4])
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
# `opts.gauge` DEFAULTS ON: `F` is exactly invariant (verified to 1e-14 at χ = 4/6/8/12), the cost
# is one k×k SVD per interface per sweep, and it turns `|ΔF|` — which oscillates at the roundoff
# floor of a signed log-sum, measured rising 1.2e-7 -> 3.4e-7 -> 5.4e-7 over sweeps 8..10 — into a
# monotone state distance. It is also the prerequisite for any accelerator.

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

# Largest relative change of any block between two states. `nothing` means NO DISTANCE EXISTS, not
# "converged" — `update` refuses to certify on it. Meaningful only with `opts.gauge` on.
#
# A block that appeared, vanished, or changed index set has DEFINITIVELY changed, so it aborts the
# comparison instead of being dropped from the sample. Skipping structurally-changed blocks silently
# measured a subset, and the subset is biased: interface widths stabilise from the bulk outward, so
# the blocks that compare early are exactly the ones that settled early. Measured, square 6×6 D=2 at
# χ=16 with `:cycle`: sweep 2 compared 28 of 220 blocks, found them equal to 1.0e-15, and stopped —
# while sweep 3, at 64 of 220, still moved by 1.0e-1. Full coverage arrived only at sweep 6. That
# stop left the whole boundary ring ~3 orders wrong (corner ⟨Z⟩ error 3e-3 against 5e-6 once allowed
# to run), χ-INDEPENDENTLY, which is what made it read as a projector defect rather than an early
# exit. `:cut` escaped only by chance — its 28-block subset still read 2.6e-1 at sweep 2 — and
# `:cycle` walked into it precisely BECAUSE it settles the bulk in one sweep.
function _ctm_statedist(a::CTMVertexEnvironments, b::CTMVertexEnvironments)
    n = 0; worst = 0.0
    (length(a.C) == length(b.C) && length(a.T) == length(b.T)) || return nothing
    for (d1, d2) in ((a.C, b.C), (a.T, b.T)), (k, ta) in d1
        tb = _ctm_nn(d2, k)
        isnothing(ta) && isnothing(tb) && continue
        (isnothing(ta) || isnothing(tb)) && return nothing
        Set(inds(ta)) == Set(inds(tb)) || return nothing
        na = norm(ta)
        na > 0 && (worst = max(worst, norm(ta - tb) / na))
        n += 1
    end
    return n == 0 ? nothing : worst
end

# Enlarged corner: the quadrant cut at (x,y), grown one vertex out of the PREVIOUS state's
# blocks (so all indices are in a consistent basis) with its two adjoining edges and vertex.
function _ctm_enlarged(S::CTMVertexEnvironments, tbl, sym::Symbol, x::Int, y::Int,
                      opts::CTMOptions)
    # ONE netcon over the corner, both edges and the site's factor LIST, rather than the hardcoded
    # left fold `((C·T)·T)·a`. Measured earlier: at these sizes netcon picks that same order for
    # the *blocks*, so the four-way fold bought no speed on its own — it is here so the order stops
    # being an unverified assumption, which matters because the enlarged corner is the hottest
    # object in the sweep. Free, given the sequence cache. Boundary blocks arrive as `nothing`.
    #
    # The site enters as `_ctm_facs`, NOT as a pre-contracted `ket * bra`. Handing netcon the fused
    # site tensor was throwing away the lazy double layer exactly where the sweep spends its time.
    grow(blocks, facs) = _ctm_absorb(opts, _ctm_list(blocks..., facs))
    if sym === :NW          # cols<x, rows<y  — grown from vertex (x-1, y-1)
        return grow((_ctm_nn(S.C, (:NW, x - 1, y - 1)), _ctm_nn(S.T, (:N, x - 1, y - 1)),
                     _ctm_nn(S.T, (:W, x - 1, y - 1))), _ctm_facs(tbl, x - 1, y - 1))
    elseif sym === :NE      # cols≥x, rows<y  — grown from vertex (x, y-1)
        return grow((_ctm_nn(S.C, (:NE, x + 1, y - 1)), _ctm_nn(S.T, (:N, x, y - 1)),
                     _ctm_nn(S.T, (:E, x + 1, y - 1))), _ctm_facs(tbl, x, y - 1))
    elseif sym === :SW      # cols<x, rows≥y  — grown from vertex (x-1, y)
        return grow((_ctm_nn(S.C, (:SW, x - 1, y + 1)), _ctm_nn(S.T, (:S, x - 1, y + 1)),
                     _ctm_nn(S.T, (:W, x - 1, y))), _ctm_facs(tbl, x - 1, y))
    else                    # :SE  cols≥x, rows≥y — grown from vertex (x, y)
        return grow((_ctm_nn(S.C, (:SE, x + 1, y + 1)), _ctm_nn(S.T, (:S, x, y + 1)),
                     _ctm_nn(S.T, (:E, x + 1, y))), _ctm_facs(tbl, x, y))
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
function _ctm_block(S::CTMVertexEnvironments, tbl, P::CTMVertexEnvironments, d,
                   opts::CTMOptions)
    kind, sym, i, j = d
    aA(t, p) = (isnothing(p) || isnothing(t)) ? t : t * p[1]
    aB(t, p) = (isnothing(p) || isnothing(t)) ? t : t * p[2]
    # Edge blocks absorb in ONE netcon over [edge; ket; bra; P_B; P_A] — no eager `ket * bra`,
    # matching the edge rebuilds in `sweep_vertex_environments` term for term. The corner branch
    # gets its laziness from `_ctm_enlarged`, and keeps the sequential `aA`/`aB` because the
    # enlarged corner is memoised by the sweep and reused across interfaces.
    edge(block, facs, pB, pA) = _ctm_absorb(opts, _ctm_list(block, facs), pB, pA)
    if kind === :C
        E = _ctm_enlarged(S, tbl, sym, i, j, opts)
        isnothing(E) && return nothing
        sym === :NW && return aA(aA(E, _ctm_nn(P.PH, (:N, i - 1, j))), _ctm_nn(P.PV, (:W, i, j - 1)))
        sym === :NE && return aA(aB(E, _ctm_nn(P.PH, (:N, i - 1, j))), _ctm_nn(P.PV, (:E, i, j - 1)))
        sym === :SW && return aB(aA(E, _ctm_nn(P.PH, (:S, i - 1, j))), _ctm_nn(P.PV, (:W, i, j - 1)))
        return aB(aB(E, _ctm_nn(P.PH, (:S, i - 1, j))), _ctm_nn(P.PV, (:E, i, j - 1)))
    end
    if sym === :N
        return edge(_ctm_nn(S.T, (:N, i, j - 1)), _ctm_facs(tbl, i, j - 1),
                    _ctm_pB(P.PH, (:N, i - 1, j)), _ctm_pA(P.PH, (:N, i, j)))
    elseif sym === :S
        return edge(_ctm_nn(S.T, (:S, i, j + 1)), _ctm_facs(tbl, i, j),
                    _ctm_pB(P.PH, (:S, i - 1, j)), _ctm_pA(P.PH, (:S, i, j)))
    elseif sym === :W
        return edge(_ctm_nn(S.T, (:W, i - 1, j)), _ctm_facs(tbl, i - 1, j),
                    _ctm_pB(P.PV, (:W, i, j - 1)), _ctm_pA(P.PV, (:W, i, j)))
    end
    return edge(_ctm_nn(S.T, (:E, i + 1, j)), _ctm_facs(tbl, i, j),
                _ctm_pB(P.PV, (:E, i, j - 1)), _ctm_pA(P.PV, (:E, i, j)))
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
    opts = cache.options
    tbl = _ctm_factor_table(cache)
    C = Dict{Tuple{Symbol, Int, Int}, Any}()
    T = Dict{Tuple{Symbol, Int, Int}, Any}()
    PH = Dict{Tuple{Symbol, Int, Int}, Any}()
    PV = Dict{Tuple{Symbol, Int, Int}, Any}()
    enl = Dict{Tuple{Symbol, Int, Int}, Any}()
    E(sym, x, y) = get!(enl, (sym, x, y)) do
        _ctm_enlarged(S, tbl, sym, x, y, opts)
    end
    # --- projector pass 1 of 2, `:cycle` only: all four of a plaquette's interfaces from ONE
    # cyclic problem, writing the SAME keys as the pairwise pass below.
    #
    # `:cycle` is a PURE formulation: there is no per-interface fallback to the cut, because a
    # lattice carrying a MIXTURE of the two families is not stationary, which would make the one
    # question `:cycle` exists to answer ill-posed. The only fallback is structural — a plaquette
    # whose cycle is undefined (a corner carrying more than its two interfaces, a rank-collapsed
    # hex plaquette, a `schursolve` that throws) declines wholesale, and the warning below says so.
    if opts.projector === :cycle
        ncyc = ndec = 0
        for X in 2:Lx, Y in 2:Ly
            cyc = _ctm_cycle_projectors(E(:NW, X, Y), E(:NE, X, Y), E(:SE, X, Y), E(:SW, X, Y),
                                        χ, opts, hash((X, Y)))
            if isnothing(cyc)
                ndec += 1
                continue
            end
            ncyc += 1
            for (fam, isH, key) in ((cyc.N, true, (:N, X - 1, Y)), (cyc.S, true, (:S, X - 1, Y)),
                                    (cyc.W, false, (:W, X, Y - 1)), (cyc.E, false, (:E, X, Y - 1)))
                pr = (fam[1], fam[2], fam[3])
                opts.gauge && (pr = _ctm_align(pr, fam[4], _ctm_nn(isH ? S.PH : S.PV, key)))
                isH ? (PH[key] = pr) : (PV[key] = pr)
            end
        end
        # Silence here would read as "the cycle projector was used everywhere", which is the one
        # thing a reader must not assume when comparing the two options.
        ndec > 0 && @warn "projector = :cycle declined $ndec of $(ncyc + ndec) plaquettes, whose \
            cycle is undefined (a corner carrying more than its two interfaces, a rank-collapsed \
            hex plaquette, or a `schursolve` that threw). Those interfaces used the cut, so this \
            run is NOT a pure `:cycle` result — see `_ctm_cycle_projectors`." maxlog = 1
    end
    # --- projector pass 2 of 2: the CUT projector, from each interface's two bounding corners.
    # Under `:cut` this owns everything; under `:cycle` it backfills whatever pass 1 declined.
    for x in 1:(Lx - 1), y in 2:Ly            # PH[:N,x,y]: C_NW(x+1,y) | C_NE(x+1,y)
        haskey(PH, (:N, x, y)) && continue
        Bw = E(:NW, x + 1, y); Be = E(:NE, x + 1, y)
        (isnothing(Bw) || isnothing(Be)) && continue
        ins = commoninds(Bw, Be)
        pr = _ctm_interface_proj2(Bw, Be, ins, χ, opts)
        opts.gauge && !isnothing(pr) && (pr = _ctm_align(pr, ins, _ctm_nn(S.PH, (:N, x, y))))
        !isnothing(pr) && (PH[(:N, x, y)] = pr)
    end
    # Every `:S` block is keyed by its FIRST included row (`T_S[x,y] = rows ≥ y`), so the
    # family lives at `y ∈ 2:Ly` — `y = Ly+1` is the empty block. All four `:S` loops below
    # (this one, C_SW, C_SE, T_S) must use that range: `1:(Ly-1)` builds a useless `y = 1` and
    # never builds `y = Ly`, leaving the bottom interface of every region unconsumed.
    for x in 1:(Lx - 1), y in 2:Ly            # PH[:S,x,y]: C_SW(x+1,y) | C_SE(x+1,y)
        haskey(PH, (:S, x, y)) && continue
        Bw = E(:SW, x + 1, y); Be = E(:SE, x + 1, y)
        (isnothing(Bw) || isnothing(Be)) && continue
        ins = commoninds(Bw, Be)
        pr = _ctm_interface_proj2(Bw, Be, ins, χ, opts)
        opts.gauge && !isnothing(pr) && (pr = _ctm_align(pr, ins, _ctm_nn(S.PH, (:S, x, y))))
        !isnothing(pr) && (PH[(:S, x, y)] = pr)
    end
    for x in 2:Lx, y in 1:(Ly - 1)            # PV[:W,x,y]: C_NW(x,y+1) | C_SW(x,y+1)
        haskey(PV, (:W, x, y)) && continue
        Bn = E(:NW, x, y + 1); Bs = E(:SW, x, y + 1)
        (isnothing(Bn) || isnothing(Bs)) && continue
        ins = commoninds(Bn, Bs)
        pr = _ctm_interface_proj2(Bn, Bs, ins, χ, opts)
        opts.gauge && !isnothing(pr) && (pr = _ctm_align(pr, ins, _ctm_nn(S.PV, (:W, x, y))))
        !isnothing(pr) && (PV[(:W, x, y)] = pr)
    end
    for x in 1:(Lx - 1), y in 1:(Ly - 1)      # PV[:E,x,y]: C_NE(x,y+1) | C_SE(x,y+1)
        haskey(PV, (:E, x + 1, y)) && continue
        Bn = E(:NE, x + 1, y + 1); Bs = E(:SE, x + 1, y + 1)
        (isnothing(Bn) || isnothing(Bs)) && continue
        ins = commoninds(Bn, Bs)
        pr = _ctm_interface_proj2(Bn, Bs, ins, χ, opts)
        opts.gauge && !isnothing(pr) && (pr = _ctm_align(pr, ins, _ctm_nn(S.PV, (:E, x + 1, y))))
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
    # ONE netcon per edge over [previous edge; ket; bra; P_B; P_A]. The site's factors go in as a
    # LIST — pre-contracting `ket * bra` here built the fat site tensor on every one of the
    # 4·Lx·Ly absorptions per sweep, which is where this engine spends its time. Folding both
    # isometries into the same call also lets the optimiser truncate an interface before growing
    # across it. `_ctm_block` mirrors these four term for term; keep them in step.
    edge(block, facs, pB, pA) = _ctm_rescale(_ctm_absorb(opts, _ctm_list(block, facs), pB, pA))
    for x in 1:Lx, y in 2:Ly                  # T_N: left = east side, right = west side
        T[(:N, x, y)] = edge(_ctm_nn(S.T, (:N, x, y - 1)), _ctm_facs(tbl, x, y - 1),
                             _ctm_pB(PH, (:N, x - 1, y)), _ctm_pA(PH, (:N, x, y)))
    end
    for x in 1:Lx, y in 2:Ly                  # T_S
        T[(:S, x, y)] = edge(_ctm_nn(S.T, (:S, x, y + 1)), _ctm_facs(tbl, x, y),
                             _ctm_pB(PH, (:S, x - 1, y)), _ctm_pA(PH, (:S, x, y)))
    end
    for x in 2:Lx, y in 1:Ly                  # T_W: up = south side, down = north side
        T[(:W, x, y)] = edge(_ctm_nn(S.T, (:W, x - 1, y)), _ctm_facs(tbl, x - 1, y),
                             _ctm_pB(PV, (:W, x, y - 1)), _ctm_pA(PV, (:W, x, y)))
    end
    for x in 1:(Lx - 1), y in 1:Ly            # T_E
        T[(:E, x + 1, y)] = edge(_ctm_nn(S.T, (:E, x + 2, y)), _ctm_facs(tbl, x + 1, y),
                                 _ctm_pB(PV, (:E, x + 1, y - 1)), _ctm_pA(PV, (:E, x + 1, y)))
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
    region_lnZ(_ctm_env_checked(cache), cache, cx, cy)

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
    opts = cache.options
    ts = _ctm_region_blocks(env, cx, cy)
    if isinteger(cx) && isinteger(cy)     # vertex ring: close it with the vertex's own factors
        v = _ctm_vertex(cache, Int(cx), Int(cy))
        isnothing(v) || append!(ts, bp_factors(network(cache), v))
    end
    isempty(ts) && return 0.0
    return log(abs(scalar(_ctm_contract(ts, opts))))
end

# The cache's environments, falling back to the greedy single pass when it has not been
# `update`d. SILENT, because `update` seeds from this — the fallback is the intended path there.
_ctm_env(cache::CTMEnvironmentCache) =
    isnothing(environments(cache)) ? vertex_environments(cache) : environments(cache)

# Same, but WARNS on the fallback. Use this on every path that hands a number to the caller.
#
# The BP convention this was modelled on does not transfer. An un-updated `BeliefPropagationCache`
# evaluates to an unconverged answer from the SAME algorithm; this falls back to a DIFFERENT one —
# the one-sided greedy pass, measured 3–4 orders worse and, crucially, **non-monotone in `maxdim`**
# (a flat ~2.5e-3 floor at every χ on the PEPS norm, which the sweep breaks straight through). So a
# forgotten `update` does not read as "not converged yet": it reads as a plausible number that
# refuses to improve when you raise χ, which is a much more expensive mistake to diagnose.
# It also rebuilds the entire environment set on every call, so a loop over regions pays a full
# greedy build per region.
#
# No `maxlog`: each occurrence is a separate wrong number over a separate full rebuild. Correct
# usage never triggers it, and asking for the greedy pass on purpose —
# `cvm_freenergy(vertex_environments(cache), cache)` — is silent, which is what the beats-greedy
# comparisons in the tests and `examples/ctm_environment.jl` use.
function _ctm_env_checked(cache::CTMEnvironmentCache)
    isnothing(environments(cache)) && @warn(
        "CTMEnvironmentCache has not been `update`d — falling back to the greedy single pass, " *
        "which is 3–4 orders less accurate and NON-MONOTONE in `maxdim`, and is rebuilt on " *
        "every call. Use `update(cache)` first. If you meant the greedy pass, ask for it " *
        "explicitly with `vertex_environments(cache)` and this warning goes away."
    )
    return _ctm_env(cache)
end

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
    env = _ctm_env_checked(cache)
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

[`update`](@ref) the cache first. On an un-updated cache this **warns** and falls back to the
greedy single pass ([`vertex_environments`](@ref)), whose one-sided cuts are 3–4 orders worse and
**non-monotone in `maxdim`** — the two numbers differing is that, not a bug. For the greedy number
on purpose, and without the warning, use the two-argument form:
`cvm_freenergy(vertex_environments(cache), cache)`.
"""
cvm_freenergy(cache::CTMEnvironmentCache) = cvm_freenergy(_ctm_env_checked(cache), cache)

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
    env = _ctm_env_checked(cache)
    opts = cache.options
    nxt = sweep_vertex_environments(cache, env)      # its PH/PV are consistent with `env`
    tbl = _ctm_factor_table(cache)
    Lx, Ly = _ctm_dims(cache)
    memo = Dict{Any, Any}()
    blk(d) = get!(memo, d) do
        _ctm_block(env, tbl, nxt, d, opts)
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
            Z = try scalar(_ctm_contract(full, opts)) catch; continue end
            (!isfinite(Z) || iszero(Z)) && continue
            push!(Ms, _ctm_contract(minus, opts) / Z)
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
    opts = cache.options
    F = cvm_freenergy(env, cache)
    converged, Δ, crit = false, Inf, Inf
    sd = nothing                       # hoisted: the warning below reports both terms
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
        sd = opts.gauge ? _ctm_statedist(env, prev) : nothing
        crit = isnothing(sd) ? Δ : max(Δ, sd^2)
        verbose && @info "CVM sweep $it: F = $F, |ΔF| = $Δ, state Δ = $(something(sd, NaN))"
        # `|ΔF|` IS NOT A CERTIFICATE ON ITS OWN, and least of all on the first sweep. `F` is a
        # signed Möbius sum whose cancellation is worth ~4000×, so it can already sit at its final
        # value while the state is still the GREEDY seed. Measured, complex hex 4×4 D=2 at χ=64:
        # sweep 1 reported `|ΔF| = 2.2e-16`, the loop exited after ONE sweep, and the returned
        # cache was still the one-sided greedy environment — norm exact to 1.3e-15 (all
        # cancellation), but `⟨Z⟩` 7.0e-4 wrong and `marginal_inconsistency` 2.9e-6 against 8.7e-10
        # at χ=32 and χ=128. Nothing was special about χ=64 except that `Δ` got unlucky; that is
        # the point — a single `Δ` carries no information about the state.
        #
        # So require positive evidence that the STATE stopped moving: at least two sweeps, and a
        # real `_ctm_statedist` when the gauge makes one available (it returns `nothing` while the
        # interface bases bootstrap, which is exactly when `Δ` is least trustworthy). With the gauge
        # off there is no state distance to be had and `Δ` remains the only signal, as before.
        certified = it >= 2 && (!opts.gauge || !isnothing(sd))
        if certified && crit ≤ tolerance * max(one(crit), abs(F))
            converged = true
            verbose && @info "CVM sweep converged after $it sweeps."
            break
        end
    end
    if !converged
        # Report BOTH terms of the criterion. `|ΔF|` alone is actively misleading: it routinely
        # bottoms out at ~1e-14 while `sd²` is still the binding term, so the message read
        # "did not converge to tolerance 1e-10 (final |ΔF| = 1.4e-14)" — a number four orders
        # BELOW the tolerance it claimed to have missed.
        msg = "CVM sweep did not converge to tolerance $tolerance after $maxiter sweeps " *
              "(final |ΔF| = $Δ, state Δ = $(something(sd, NaN)); " *
              "binding criterion max(|ΔF|, state Δ²) = $crit)."
        verbose ? println(msg) : @warn(msg)
    end
    return _ctm_setenv(cache, env)
end
