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
# Each shared interface is truncated to `maxdim` by a biorthogonal projector PAIR built from BOTH
# bounding corners. TWO projectors are selectable via `CTMOptions.projector` (see below):
#
#   :cut    (default) the optimal rank-χ truncation of that one bipartition, from a thin QR of each
#           block and one SVD of the small triangular product — never squaring, batches well on GPU
#           (`_ctm_twosided_projector_qr`).
#   :cycle  the dominant invariant subspace of the four-corner cycle, solved matrix-free
#           (`_ctm_cycle_projectors`). This makes `F` stationary; `:cut` is not stationary.
#
# Either way the pair needs the complement environment, so the build is a fixed-point iteration:
# `update` sweeps it to stationarity. Works for anisotropic / non-square grids and free boundaries.
#
# Entry points: `update` (run it), `cvm_freenergy` (ln Z), `vertex_ring` / `expect` / `rdm`
# (single-site observables from a vertex's own ring).
#
# Double-layer networks (⟨ψ|ψ⟩, ⟨ψ|O|ψ⟩) are handled LAZILY: a vertex's factors stay a
# `Vector{AbstractTensor}` ([ket, bra], or [ket, op, bra]) and the environment tensors keep their
# inward ket and bra legs separate (dimension D each, never fused to D²). Each absorption
# contracts the flat list [environment; factors…] in a netcon-optimal order, so the fat
# ket⊗bra site tensor is never materialised.
#
# See docs/ctmrg_status.md for the current numbers and the open problems;
# docs/finite_ctmrg_design.md for the derivations and the full record of what was tried.

using LinearAlgebra: norm, dot, qr, svd, diag
using Random: Xoshiro
using KrylovKit: schursolve, Arnoldi

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
| `degtol` | `0.0` | relative gap below which a truncation is judged to split a near-degenerate multiplet, and is backed off. `0` disables. Matters for double-layer corners (ket↔bra exchange gives `λ_ij = λ_ji`). Applies to the `:cut` singular-value truncations AND (2026-08-21) to the `:cycle` spectrum cut in `_ctm_cycle_projectors` — an invariant subspace forced to split a cluster is ill-defined and re-resolves differently each sweep. |
| `qr_cutoff` | `1e-13` | relative cutoff on the `S` values the projector inverts. It can sit this low because `S` comes off a triangular product rather than a squared object — see `_ctm_twosided_projector_qr`. Measured 2026-08-11 (4 seeds, both projectors, 5×5 D=2 at χ=4/8): **INERT** — `⟨Z⟩` is identical to every digit from 1e-15 to 1e-7, i.e. the guard never fires at these sizes. It is insurance against `S^(-1/2)` amplification, not a tuning lever. |
| `optimal_max` | `12` | tensor count above which contraction-order search falls back from exhaustive netcon to greedy. A FEASIBILITY gate — see `_ctm_contract`. |
| `projector` | `:cut` | which interface projector to derive: `:cut` (optimal rank-χ truncation of one bipartition) or `:cycle` (four-corner cycle, which makes `F` stationary). See "Choosing a projector" below. |
| `cycle_rankcut` | `0.0` | ⚠️ **`:cycle` only.** Relative cutoff on the four-corner cycle spectrum: retained modes with `abs(λ) ≤ cycle_rankcut · abs(λ_max)` are dropped. Guards the OVER-parametrised regime (χ above the state's rank), where the surplus near-null modes are arbitrary and wander sweep to sweep. `0` disables — deliberately the default: a fixed MAGNITUDE cutoff that fixes over-parametrised cases breaks higher-entanglement ones (measured 2026-08-19: 1e-10 repairs 5×5 nl=3 but degrades 4×4 nl=4 from 1e-15 to 7.6e-11, and no smaller value threads the needle — junk and real weight OVERLAP in magnitude across cases). `cycle_gapcut` below is the gap-based rule that does thread it. Distinct from `qr_cutoff`, which cuts the biorthogonal OVERLAP. |
| `cycle_gapcut` | `1e-4` | ⚠️ **`:cycle` only.** Noise-cliff rank cut: truncate the trailing spectral block below the first cliff that is BOTH steep (`abs(λ_{j+1}) ≤ cycle_gapcut · abs(λ_j)`) and genuinely tiny (`abs(λ_{j+1}) ≤ √eps · abs(λ_1)`). Gap-based where `cycle_rankcut` is magnitude-based: measured cliffs are 1.8e5 into a noise block against ≤ 4.5e2 anywhere inside a physical decay, so the rule kills the over-parametrised wander while leaving deep-but-real modes (down to 3.4e-14 relative, measured to carry observable weight) untouched. `0` disables. The left/right spectral-consistency guard (see `_ctm_cycle_projectors`) is always on and independent of this knob. |
| `svd` | `:auto` | **`:cut` only.** How the truncated SVD behind each interface projector is computed. `:dense` is the dense route — thin QR of each enlarged corner and a FULL SVD of their `n × n` overlap, `n = χ·D²` on a double layer, costing `O(n³) = O(χ³D⁶)` per interface although only `χ` triplets are used. `:subspace` is matrix-free — block subspace iteration on the enlarged corners themselves, warm-started from the previous sweep's projector (see `_ctm_subspace_svd`), costing `O(n²χ) = O(χ³D⁴)` per interface, the boundary-MPS scaling. `:auto` (default) picks `:subspace` on dense tensors whenever the block is large enough for it to win (see `_ctm_use_subspace`) and `:dense` otherwise — including on graded (symmetric) tensors, where a cold start cannot yet allocate its block across sectors; force `:subspace` there explicitly. |
| `svd_oversample` | `16` | `:subspace` only. Extra block columns beyond `χ`. The retained `χ`-dimensional subspace converges like `(σ_{χ+p+1}/σ_χ)²` per iteration, so oversampling buys convergence on slowly decaying spectra; each column costs one more `n²` GEMM per application. |
| `svd_maxiter` | `10` | `:subspace` only. Cap on block iterations per projector; the route bails out to the dense SVD as soon as the residual decay predicts the cap will not suffice (see `_ctm_subspace_svd`). Warm-started interfaces typically exit after 1–3; a cold start needs more, and a cap of 6 measured too tight for the first sweeps on a 9×9 D=3 PEPS at χ=48. |
| `svd_tol` | `1e-10` | `:subspace` only. Stop when the block is invariant to this relative residual, `‖O†Q − X X†O†Q‖ / ‖O†Q‖` — a first-order measure of the subspace error θ (singular VALUES converge only as θ²). The truncation error grows only as `θ²·(σ_χ/σ_{χ+1})²`, so `1e-10` is far below anything the outer sweep can resolve. |
| `svd_min` | `64` | `:subspace` under `:auto` only. Smallest interface dimension at which the subspace route is taken; below it LAPACK's dense SVD on a tiny matrix is as fast. The `:auto` gate also requires `n ≥ 4(χ + svd_oversample)`, since the subspace route's cost is linear in the block width. |

## Choosing a projector

Both are two-sided and biorthogonal, both land on the same interface keys, both are exact at lossless
χ, and both are PURE *per plaquette* — `:cycle` never mixes the two families on a single plaquette
(see `sweep_vertex_environments`). A plaquette whose cycle is undefined declines WHOLESALE and its
four interfaces are backfilled by the cut pass, so a `:cycle` lattice can still carry both families
across different plaquettes; `update` warns when that happens. They differ in what they optimise:

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
| 32 | 7.39e-12 | **4.82e-14** | 1.33e-12 | 6.4e-17 → **3.9e-16** |

`:cycle` beats `:cut` at every χ here (3.2× / 8.3× / 6.7× / 152×) and matches or beats their engine
at every χ, by 27× at χ=32. It is stationary to machine precision for χ ≥ 9 — but NOT at χ=4, where a
fully resolved rank-4 invariant subspace is a worse stationary point than an under-resolved one. That
is a property of the criterion at severe truncation, not a bug; restricting `krylovdim` fixes it at
χ=4 but costs accuracy at χ=8, so it is deliberately not done (see docs/ctmrg_status.md).

`:cycle` is also markedly CHEAPER: at 8×8 D=3 it is 15.7× faster per sweep than `:cut` for identical
retained dimensions, because it is matrix-free where `:cut` forms dense QR factors of the enlarged
corners.

⚠️ **This is one physical state.** On random states, multi-seed, `⟨Z⟩` ratio `:cut`/`:cycle` at χ=8
(>1 means `:cycle` better): hex 4×4 **1208×**, square 4×4 D=2 **4.5×**, square 5×5 D=2 1.2×,
heavy-hex exact both ways — but square 4×4 **D=3 is 0.04**, i.e. `:cycle` loses by 25× there,
recovering only to 0.35 by χ=16. The discriminator is the STATE (structured versus random signed),
not D and not the observable site. Never choose between these on a single configuration.

⚠️ **Below the lossless χ threshold, `:cycle` limit-cycles instead of converging — a
truncation-insufficiency signal, DIAGNOSED 2026-08-19, and the fix is χ.** (This includes the 8×8
cases that were long the top open problem: 8×8 D=2 needs χ > 16 to be lossless, so 8×8 at χ=16
under-truncates and cycles while 6×6 at the same χ is lossless and converges to ~1e-8.) The map has
no stable fixed point at insufficient χ; the oscillation amplitude tracks the degree of
under-truncation, and it is NOT fixable by solver tricks — warm-started seeds, cycle-averaging and
Anderson were each falsified (a better seed cannot stabilise a map with no stable fixed point).

`F` IS BARELY AFFECTED even mid-cycle — the Möbius sum cancels ~4000× of the oscillation, while a
single-site observable reads ONE region with no cancellation and oscillates at ~1e-4. So at
insufficient χ `:cycle` stays sound for free energies and unreliable for observables. `update`'s
convergence check flags the situation either way: `|ΔF|` genuinely moves on the cycle, and the
`convergence = :worst_region` signal makes the O(1) floor explicit (O(1) = raise χ, →0 = converged).
See `docs/ctmrg_status.md` for the full diagnosis and the falsified-fixes record.

`:cut` remains the default because it has no known failure regime and is the longer-tested path.
The timing note above predates `svd = :auto` (2026-09-08): with the dense SVD route `:cut` was the
dearer projector; with the subspace route its projectors cost `O(χ³D⁴)` and a `:cut` `update`
measured cheaper than a `:cycle` one on a physical 9×9 D=3 PEPS at χ=32 (16 s against 52 s under
`convergence = :marginal`, which `:cycle` observables need), see docs/ctmrg_status.md.

"""
Base.@kwdef struct CTMOptions
    gauge::Bool = true
    degtol::Float64 = 0.0
    qr_cutoff::Float64 = 1.0e-13
    optimal_max::Int = 12
    projector::Symbol = :cut
    # Relative cutoff on the four-corner cycle SPECTRUM: drop retained modes with |λ| below
    # `cycle_rankcut · |λ_max|`. `0` disables. Guards the OVER-parametrised regime (χ larger than the
    # state's actual rank), where the surplus near-null modes are arbitrary and wander sweep-to-sweep,
    # making the residual grow with χ. Distinct from `qr_cutoff` (which cuts the biorth OVERLAP).
    cycle_rankcut::Float64 = 0.0
    # NOISE-CLIFF rank cut on the cycle spectrum: truncate the trailing block below the first cliff
    # that is BOTH steep (|λ_{j+1}| ≤ cycle_gapcut·|λ_j|) and genuinely tiny (|λ_{j+1}| ≤ √eps·|λ_1|).
    # `0` disables. Unlike `cycle_rankcut`, this is gap-based, which is what separates noise from
    # deep-but-real modes — see the derivation note in `_ctm_cycle_projectors`.
    cycle_gapcut::Float64 = 1.0e-4
    # Truncated-SVD route for the `:cut` projector — see the table above and `_ctm_subspace_svd`.
    svd::Symbol = :auto
    svd_oversample::Int = 16
    svd_maxiter::Int = 10
    svd_tol::Float64 = 1.0e-10
    svd_min::Int = 64

    function CTMOptions(gauge, degtol, qr_cutoff, optimal_max,
                        projector, cycle_rankcut, cycle_gapcut,
                        svd, svd_oversample, svd_maxiter, svd_tol, svd_min)
        projector in (:cut, :cycle) || throw(ArgumentError(
            "projector must be :cut or :cycle, got $(repr(projector))"))
        0 <= cycle_gapcut < 1 || throw(ArgumentError(
            "cycle_gapcut is a relative gap ratio and must lie in [0, 1), got $cycle_gapcut"))
        svd in (:auto, :dense, :subspace) || throw(ArgumentError(
            "svd must be :auto, :dense or :subspace, got $(repr(svd))"))
        svd_oversample >= 0 || throw(ArgumentError("svd_oversample must be ≥ 0, got $svd_oversample"))
        svd_maxiter >= 1 || throw(ArgumentError("svd_maxiter must be ≥ 1, got $svd_maxiter"))
        svd_tol >= 0 || throw(ArgumentError("svd_tol must be ≥ 0, got $svd_tol"))
        return new(gauge, degtol, qr_cutoff, optimal_max, projector, cycle_rankcut, cycle_gapcut,
                   svd, svd_oversample, svd_maxiter, svd_tol, svd_min)
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
    coords::Dict{V, Tuple{Int, Int}} # inverse of `grid`: vertex → position, for O(1) lookup
    dims::Tuple{Int, Int}            # bounding box (Lx, Ly)
    maxdim::Int
    environments::E                  # `nothing`, or the CVM blocks from `update`
    options::CTMOptions              # numerical strategy, fixed at construction
    # Per-interface memo for the `:cut` projector's subspace route: `key => (skip, backoff)` after
    # the route bailed out on that interface — `skip` further sweeps go straight to the dense
    # route, then it is retried, and a second bail doubles `backoff` (capped). A success clears the
    # entry. Shared by every cache `_ctm_setenv` derives from this one, so the memory persists
    # across the sweeps of one `update`. See `sweep_vertex_environments`.
    route::Dict{Tuple{Symbol, Int, Int}, Tuple{Int, Int}}
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
    length(grid) == length(vs) || error("CTMEnvironmentCache: two vertices share a grid position.")
    coords = Dict{eltype(vs), Tuple{Int, Int}}(v => pos for (pos, v) in grid)
    # Every bond must join grid NEIGHBOURS. The corner moves project the links between adjacent
    # columns/rows and nothing else: a bond between non-adjacent positions (a periodic wraparound,
    # a long-range coupling) would ride along uncontracted inside every block, so each sweep adds
    # legs and the region contractions grow exponentially — at any χ, including χ = 1. Reject it
    # here with a reason rather than let the caller discover it as an endless run.
    for e in edges(graph(net))
        (x1, y1) = coords[src(e)]; (x2, y2) = coords[dst(e)]
        abs(x1 - x2) + abs(y1 - y2) == 1 || error(
            "CTMEnvironmentCache: bond $(src(e)) – $(dst(e)) joins non-adjacent grid positions " *
            "$((x1, y1)) and $((x2, y2)). CTMRG needs an OPEN lattice whose bonds connect grid " *
            "neighbours (periodic lattices, e.g. `named_hexagonal_lattice_graph(...; periodic = true)`, " *
            "are not supported).")
    end
    Lx = maximum(first.(keys(grid))); Ly = maximum(last.(keys(grid)))
    return CTMEnvironmentCache(net, grid, coords, (Lx, Ly), Int(maxdim), nothing, opts,
                               Dict{Tuple{Symbol, Int, Int}, Tuple{Int, Int}}())
end

# Same network/grid/maxdim/options (and route memo), different CVM environments.
_ctm_setenv(cache::CTMEnvironmentCache, env) =
    CTMEnvironmentCache(cache.network, cache.grid, cache.coords, cache.dims, cache.maxdim, env,
                        cache.options, cache.route)

# --- the move --------------------------------------------------------------------
# `opts.degtol` — relative cutoff gap below which the truncation is judged to split a
# (near-)degenerate multiplet; the cut is then backed off to a real gap. Matters for
# DOUBLE-LAYER networks, whose corner spectra carry systematic 2-fold degeneracies from
# ket↔bra exchange (λ_ij = λ_ji). 0 disables it.
#

# THE interface projector, via a TRIANGULAR (QR) factorization of each block.
#
# A second `ρ`-based route was removed (it was sesquilinear by construction and therefore wrong for
# complex tensors); `opts.qr_cutoff` is now the only cutoff. See docs/finite_ctmrg_design.md.
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
# WHY QR AND NOT AN EIGENDECOMPOSITION — GPU BATCHING, NOT ACCURACY. Measured accuracy-neutral
# against the removed ρ route across 28 configurations and four cutoffs. χ is the binding constraint,
# not arithmetic (retained spectrum median `S_k/S_1` ~1e-1…1e-2; 0% below 1e-8). The win is that
# geqrf/gesvd batch well on GPU where batched Hermitian eig support is thin, and a sweep is 200–384
# INDEPENDENT tiny factorizations (n ≤ 128) — a batching problem, not a big-linear-algebra one.
#
# DEVICE / BACKEND NOTE. Both projectors are written entirely in tensor verbs — QR, SVD, diagonal
# whitening, direct sums, and Krylov iteration on tensor vectors — so they run on whatever backend
# and device the network lives on (MatrixAlgebraKit factorizations, TensorKit for graded data). The
# only scalar work is O(χ): the truncation rule inside the SVD and the cycle's spectral guards. See
# docs/ctmrg_status.md "GPU / CUDA compatibility".


# Truncation rule shared by every `:cut` factorization — rank ≤ χ, drop singular values at or
# below `qr_cutoff · s₁`, never split a multiplet degenerate to `degtol`. It is a backend truncation
# STRATEGY, so the decision runs inside the truncated SVD: on dense tensors over the sorted
# spectrum, on graded tensors over the merged spectrum of all sectors (the retained bond then gets
# one count per sector for free).
_ctm_trunc(maxdim::Integer, opts::CTMOptions; rtol::Real = opts.qr_cutoff) =
    truncation_strategy(; maxdim = Int(maxdim), rtol, degtol = opts.degtol)

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

function _ctm_seq_key(ts::Vector{<:AbstractTensor})
    seen = Dict{Any, Int}()
    label(i) = (get!(seen, i, length(seen) + 1), dim(i))
    return Tuple(Tuple(label(i) for i in inds(t)) for t in ts)
end

# Above `opts.optimal_max` tensors, fall back to the greedy optimiser. This is a FEASIBILITY
# gate, not a performance one: `alg = "optimal"` is ExhaustiveSearch netcon, exponential in the
# number of tensors, and it hangs outright on the ~25-tensor lists a `vertex_window` observable
# produces. (Tried and reverted as a *perf* tweak earlier — it bought ~1.5% on sweep-sized lists.)
#
# The gate's verdict joins the cache key: `optimal_max` is per-cache, so two caches sharing a
# lattice shape would otherwise trade sequences and each get whichever optimiser ran first.
function _ctm_contract(ts::Vector{<:AbstractTensor}, opts::CTMOptions)
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



# Triangular factor of a block over its interface `ins`: `B = Q·R` with `R` on (b, ins…). Never
# forms `ρ`. A block that IS its interface (no other legs) is its own factor, on a width-1 bond.
function _ctm_tri_factor(B, ins::Vector{<:Index})
    rest = uniqueinds(B, ins)
    if isempty(rest)
        b = new_index(B, 1; tags = "Link,qr")
        return B * adapt_like(B, delta(scalartype(B), b))
    end
    return last(qr(B, rest))
end

# Biorthogonal pair from the TRIANGULAR factors of the two bounding blocks — no squaring
# anywhere, and NO conjugation in the pairing: the sweep contracts the two enlarged corners
# PLAINLY (`Bw * Be`), so the projector must preserve the BILINEAR product `A Bᵀ`, not the
# sesquilinear `A B†`. (An earlier version conjugated; that optimised the wrong pairing and broke
# the pair's exactness at full rank by ~11% on a complex 4×4.)
#
#   Bw = Q_A R_A,  Be = Q_B R_B                    R on (b, ins…)
#   W  = R_A R_Bᵀ = U S Vᵀ                         bilinear contraction over `ins`
#   P_A = R_Bᵀ V S^{-1/2}  (ins… → w)              P_B = S^{-1/2} Uᴴ R_A  (w → ins…)
#   P_B P_A = S^{-1/2} Uᴴ R_A R_Bᵀ V S^{-1/2} = 𝟙_w
#
# Everything stays in tensor form: QR, SVD and the diagonal whitening are backend verbs, so this
# runs unchanged on dense and graded (symmetric) tensors and on whatever device the network lives
# on. Truncation (rank, relative cutoff, degeneracy back-off) is `_ctm_trunc`, applied INSIDE the
# SVD. The dense top-k Krylov SVD that once lived here (`_ctm_svd_topk`, measured 6–12× on
# n ≥ 288 blocks) was retired with the matrix code; it can return as a dense-backend hook if the
# large-χ benchmark asks for it.
#
# The QR of a block only pays when it SHRINKS it — `R` is `min(rest, ins) × ins`, so a block whose
# rest legs are no wider than its interface (every bulk interface: both are `χ·D²`) gets a same-size
# triangular factor for the price of an `n×n` QR, and `R_A R_Bᵀ` has exactly the singular values of
# `A Bᵀ` (the `Q`s are isometries). Measured at n = 288: the two QRs were 10.7 ms of a 25 ms
# projector, the SVD 12.6 ms. So factor only the blocks the QR actually reduces; the overlap of the
# rest is formed directly.
function _ctm_twosided_projector_qr(Bw, Be, ins::Vector{<:Index}, maxdim::Integer, opts::CTMOptions)
    # The QR is skipped where it does not shrink the block (`rest ≤ ins`) — exact on dense data. NOT
    # on graded data: the whitening then contracts `dag(V)`/`dag(U)` over the block's OWN legs, which
    # carry mixed orientations, and on fermionic tensors that dag-then-contract picks up parity
    # twists (measured: fermionic 3×3 D=3 at the lossless χ=16 read ⟨N⟩ 4.2e-8 off against 1.6e-19
    # through the triangular factors, whose single fresh bond has one orientation).
    skip(B) = !_ctm_isgraded(B) && dim(uniqueinds(B, ins)) <= dim(ins)
    RA = skip(Bw) ? Bw : _ctm_tri_factor(Bw, ins)
    RB = skip(Be) ? Be : _ctm_tri_factor(Be, ins)
    # `_ctm_biorth` forms `R_A R_Bᵀ` (bilinear, over `ins`), takes its truncated SVD, and whitens.
    return _ctm_biorth(RB, RA, ins, _ctm_trunc(maxdim, opts))
end

# --- the SUBSPACE route (`opts.svd`) ------------------------------------------------------
#
# WHY. The dense route above is the classic svd-CTMRG bottleneck: with `n = dim(ins) = χ·D²` on a
# double layer, every interface pays two `n×n` QRs and a FULL `n×n` SVD — `O(χ³D⁶)` — to keep `χ`
# triplets. Measured on a 9×9 D=3 PEPS at χ=48 that was 91% of a sweep (2·QR 23 ms + SVD 33 ms per
# interface, 256 interfaces), and it is exactly the `D²` (plus the SVD-versus-GEMM constant) by
# which this engine trailed boundary MPS.
#
# WHAT. The truncated SVD of `O = Bw·Be` (bilinear over `ins`) by block subspace iteration on the
# ENLARGED CORNERS THEMSELVES — the product is never formed and nothing `n×n` is ever factorised:
#
#   X  ← orthonormal block on Be's rest legs           (n × k′, k′ = χ + oversample)
#   repeat:  Q ← orth(Bw (Be X)),  Z ← Be† (Bw† Q),  X ← orth(Z)     4 GEMMs of n²k′ each
#   until    ‖Z − X X†Z‖ ≤ svd_tol · ‖Z‖              (X invariant under O†O to first order)
#   O ≈ Q Q†O = Q Z†  ⇒  svd(Z) gives U = Q·V_Z, S, V = conj(U_Z)      one n × k′ SVD
#
# then the same whitening as the dense route (`_ctm_whiten`), so the pair is the same object at
# convergence. Cost `O(K·n²·χ) = O(χ³D⁴)` per interface, the boundary-MPS scaling, in GEMMs.
#
# WARM START — the paper's block-Krylov idea (Woolls et al., Sec. V.D). The outer CTMRG sweep is a
# fixed-point iteration, so the previous sweep's projector already spans (nearly) the invariant
# subspace: at a converged environment `O ≈ Bw Π Be` with `Π = P_A P_B`, hence
# `range(O†) ⊂ range(Be† P_B†)`. `X₀ = Be† P_B†` therefore starts the iteration where it will end,
# and as the outer sweep converges the block exits after ONE iteration with a residual at roundoff.
# The previous projector lives on the current `ins` only once the interface bases are index-stable,
# i.e. under `gauge = true` (`_ctm_align` reuses the kept index) — the same guard `_ctm_align` uses.
# Without it every sweep is a cold start (random block, `svd_maxiter` iterations), still cheaper
# than dense but slower to settle. Oversampling columns are always random, seeded on the interface
# POSITION so a run is reproducible; drawn on the host and moved to the network's device.
#
# ACCURACY. Subspace iteration converges the retained subspace at `(σ_{k′+1}/σ_χ)²` per iteration; a
# subspace error θ costs `θ²·(σ_χ/σ_{χ+1})²` relative to the optimal truncation error — second order,
# and a gapped spectrum (large ratio) is precisely where the iteration is fastest. The nested
# outer/inner iteration converges to the SAME fixed point as the dense route: at that fixed point
# the warm start is exact and one iteration reproduces it. Lossless χ (rank(O) ≤ k′) is exact in
# one iteration from any generic start.
#
# GRADED DATA. Everything below is written in seam verbs (contraction, `qr`, `svd`, `directsum`,
# `charge_sectors`), so it runs on symmetric tensors — but a cold start must give the block columns
# in every sector the top-χ subspace needs, and the round-robin allocation in `_ctm_random_block`
# is a guess. `:auto` therefore keeps graded tensors on the dense route; `:subspace` opts in.

_ctm_isgraded(t) = t isa Tensors.GradedTensor

# Running counts for the subspace route — projectors by outcome (`:subspace`, `:dense` = bailed,
# `:declined` = gate, `:skipped` = memo) and block iterations spent — so a run can be checked for
# "did the cheap route actually run, and how hard did it work" without a profiler. Diagnostic
# only; reset it yourself (`empty!`). NOT thread-safe, like `CTM_SEQ_CACHE`.
const CTM_SVD_STATS = Dict{Symbol, Int}()
_ctm_stat!(k::Symbol, n::Integer = 1) = (CTM_SVD_STATS[k] = get(CTM_SVD_STATS, k, 0) + n; nothing)

# The diagonal of a (u, v) singular-value tensor as a host vector of magnitudes, descending. O(k²)
# through the dense array, on either backend (a graded diagonal comes out in sector order, hence
# the sort); only ever read for an O(1) rate decision.
function _ctm_diagvals(S)
    is = collect(inds(S))
    return sort!(abs.(Array(diag(array(S, is...)))); rev = true)
end

# Gate for `opts.svd === :auto`: the subspace route wins when the block it keeps is a small
# fraction of an interface that is not tiny. Its per-iteration cost is `8·n²·k′` flops in GEMMs
# against ~`25·n³` for the dense route, so it pays from `n ≳ k′`; the factor 4 covers the `n·k′²`
# small factorizations and the fixed overheads, `svd_min` the regime where LAPACK on a tiny matrix
# is fast regardless.
function _ctm_use_subspace(opts::CTMOptions, ts, nrows::Integer, ncols::Integer, nins::Integer,
                           kp::Integer)
    opts.svd === :dense && return false
    nmin = min(nrows, ncols, nins)
    nmin >= 1 && kp >= 1 || return false
    opts.svd === :subspace && return nmin > kp     # anything smaller is the dense problem itself
    any(_ctm_isgraded, ts) && return false
    return nmin >= max(opts.svd_min, 4kp)
end

# A random orthonormal-to-be block on `legs` with `kp` columns: one random vector per column,
# allocated round-robin over the sectors `charge_sectors(legs)` can reach (one trivial sector on
# dense data) and stacked along their charge legs (`_ctm_stack`). Host-drawn from `rng`; the caller
# moves it on-device. `nothing` if no column could be drawn.
function _ctm_random_block(rng, elt::Type, legs::Vector{<:Index}, kp::Integer)
    # Dense legs: one draw on a plain block leg. (The sector-stacked path below is the same thing one
    # column at a time, and its `kp` direct sums measured 0.9 ms against 0.05 ms for the draw.)
    all(i -> Tensors.space(i) isa Integer, legs) &&
        return random_tensor(rng, elt, vcat(legs, [new_index(legs, kp; tags = "Link,blk")]))
    secs = charge_sectors(legs)
    vs = Any[]; slots = Any[]
    j = 0
    while length(vs) < kp && j < 4kp + length(secs)
        c = secs[mod1(j += 1, length(secs))]
        v = random_tensor(rng, elt, vcat(legs, [c]))
        norm(v) > 0 || continue                    # a sector the legs cannot reach
        push!(vs, v); push!(slots, c)
    end
    isempty(vs) && return nothing
    return _ctm_stack(vs, slots)
end

# Leading singular triplets of the linear map `apply` (block on `colsd` → block on `rows`, with
# `applyadj` its adjoint) by warm-started block subspace iteration — see the section comment.
# `colsd` are the column legs AS THE BLOCK CARRIES THEM (the dual of the operator's own), `X0` the
# start block on `(colsd…, c)`. Returns `(U, S, V)` in the seam's `svd` convention — `U (rows…, u)`,
# `S (u, v)`, `V (cols…, v)` with `U·S·V` the bilinear reconstruction, i.e. `V` is the CONJUGATE of
# the right singular vectors, exactly what `svd(O, rows)` returns — truncated by `trunc`. `nothing`
# on any numerical trouble; the caller falls back to the dense route.
#
# BAIL-OUT. Subspace iteration converges at `ρ = (σ_{k′+1}/σ_χ)²` per iteration, and on a FLAT
# spectrum — a random D=3 PEPS at χ=32 measured ρ ≈ 0.75 — it never reaches `svd_tol` within any
# sensible cap. Accepting the unconverged block is not an option: its leftover error re-enters the
# next sweep through the corners, and the outer sweep then wanders instead of converging (measured:
# state distance flat at 2e-3 for ten sweeps while the dense route fell to 4e-5). So the residual
# decay is extrapolated after each iteration and, once it predicts the tolerance will not be met
# within `svd_maxiter`, the block is abandoned to the dense route — `nothing`. A cold start's first
# ratio is not used: a random block sheds its component along the dominant directions in one
# iteration and the ratio reads far better than the asymptotic rate. The abandoned work is two to
# three iterations, a fraction of the dense cost; a decaying spectrum is exactly where the
# extrapolation says "continue".
#
# Two rate estimates feed the extrapolation. The RITZ estimate `(s_{k′}/s_χ)²` from the singular
# values of `O·X` — available from the first iteration, and accurate as soon as the block is
# close (a warm start) — is the classical bound on the rate at which subspace iteration resolves
# the χ-th direction, with `s_{k′} ≥ σ_{k′+1}` making it conservative. The OBSERVED ratio of
# successive residuals replaces it once two trustworthy residuals exist. Orthonormalising `O·X` by
# an SVD rather than a QR costs ~0.2 ms more per iteration at n = 288 and is what makes the Ritz
# values free.
function _ctm_subspace_svd(apply, applyadj, rows::Vector{<:Index}, colsd::Vector{<:Index},
                           X0, k::Integer, trunc, opts::CTMOptions; warm::Bool = false)
    X = _ctm_orthbasis(X0, colsd)
    local Q, Z
    converged = false
    rprev = NaN
    for it in 1:opts.svd_maxiter
        Q, Sy, _ = svd(apply(X), rows)               # Q spans O·X; Sy holds the Ritz values
        Z = applyadj(Q)
        s = _ctm_diagvals(Sy)
        # First-order invariance residual of the CURRENT block, before it is replaced — over the
        # RETAINED directions only. The oversampling tail (and, at a lossless χ, the null tail
        # inside the retained block) spans directions with σ ≈ 0 whose basis is arbitrary and never
        # settles; measured on a converged 6×6 TFIM PEPS it floored the whole-block residual at
        # ~1e-8 and made every interface bail. Masking to the χ leading Ritz columns weights each
        # direction by its own σ_j (the columns of `Z = O†Q` scale as σ_j), which is exactly the
        # weight it carries in the truncated product `Bw Π Be`.
        thr = length(s) >= k ? s[k] : zero(eltype(s))
        mask = map_diag(x -> abs(x) >= thr ? one(x) : zero(x), Sy)   # (u, v): keep the top k
        Zk = Z * mask                                 # top-k columns of Z, on the mask's v leg
        nz = norm(Zk)
        (isfinite(nz) && nz > 0) || return nothing
        resid = norm(Zk - X * gram(X, Zk, colsd)) / nz
        _ctm_stat!(:iterations)
        X = _ctm_orthbasis(Z, colsd)
        if resid <= opts.svd_tol
            converged = true
            break
        end
        ρ = NaN
        if it >= (warm ? 2 : 3)
            ρ = resid / rprev
        elseif length(s) > k && s[k] > 0
            ρ = (s[end] / s[k])^2
        end
        if !isnan(ρ)
            # A ratio above ½ is a flat spectrum whatever the extrapolation says — the observed
            # ratio only grows as the fast directions die out (measured 0.22, 0.62, 0.74, 0.81 on
            # a random D=3 state), so a generous cap would run out the budget before bailing.
            ρ >= 0.5 && return nothing
            # iterations still needed at this rate
            need = log(opts.svd_tol / resid) / log(ρ)
            it + need > opts.svd_maxiter && return nothing
        end
        rprev = resid
    end
    converged || return nothing
    # O ≈ Q Q†O = Q Z†. With Z = U_Z S V_Zᵀ (seam convention: V_Z is conj of the right vectors),
    # O[a,b] = Σ_u (Q·conj(V_Z))[a,u] S_u conj(U_Z)[b,u], so U = Q·dag(V_Z) and V = dag(U_Z).
    Uz, Sz, Vz = svd(Z, colsd; trunc)
    a = only(commoninds(Uz, Sz)); b = only(commoninds(Vz, Sz))
    U = replaceind(Q * dag(Vz), b, a)            # (rows…, a): the seam's U leg
    V = replaceind(dag(Uz), a, b)                # (cols…, b): the seam's V leg, cols in O's orientation
    return U, Sz, V
end

# Two-sided projector by the subspace route. `prev = (P_A, P_B, w)` from the previous sweep seeds
# the block when it lives on the current `ins`. `seed` fixes the random oversampling columns.
function _ctm_twosided_projector_subspace(Bw, Be, ins::Vector{<:Index}, maxdim::Integer,
                                          opts::CTMOptions, prev, seed::UInt)
    Brow, Acol = Bw, Be                              # the dense route's roles: O = Brow · Acol
    rows = _ctm_legs_of(Brow, uniqueinds(Brow, ins))
    cols = _ctm_legs_of(Acol, uniqueinds(Acol, ins))
    colsd = dag(cols)                                # as a block contracting INTO Acol carries them
    k = min(Int(maxdim), dim(rows), dim(cols), dim(ins))
    kp = min(k + opts.svd_oversample, dim(rows), dim(cols))
    # `missing`: the gate declined (too small, graded under `:auto`, or `svd = :dense`) — not a
    # bail-out, so the sweep's memo must not count it as one.
    _ctm_use_subspace(opts, (Bw, Be), dim(rows), dim(cols), dim(ins), kp) || return missing
    dBrow = dag(Brow); dAcol = dag(Acol)             # conjugated ONCE per projector, not per iteration
    apply(X) = Brow * (Acol * X)
    applyadj(Q) = dAcol * (dBrow * Q)
    elt = scalartype(Brow)
    X0 = nothing
    if !isnothing(prev) && length(prev) >= 3
        PBo, wo = prev[2], prev[3]
        # Be† P_B† spans the previous sweep's right invariant subspace (see the section comment).
        issetequal(collect(inds(PBo)), vcat(collect(ins), [wo])) && (X0 = dAcol * dag(PBo))
    end
    nrand = isnothing(X0) ? kp : max(0, kp - dim(only(uniqueinds(X0, colsd))))
    if nrand > 0
        R = _ctm_random_block(Xoshiro(seed), elt, colsd, nrand)
        isnothing(R) && return nothing
        R = adapt_like(Acol, R)
        if isnothing(X0)
            X0 = R
        else
            # Oversample the warm block with the random columns. On a graded backend the two block
            # legs can carry opposite duality (the warm leg is the dual of `P_B`'s kept index, the
            # random one is whatever the sector stacking minted) and the direct sum then refuses;
            # the warm block alone is a valid, if unpadded, start — so fall back to it.
            X0 = try
                directsum(X0 => only(uniqueinds(X0, colsd)), R => only(uniqueinds(R, colsd)); tags = "Link,blk")
            catch err
                err isa InterruptException && rethrow()
                X0
            end
        end
    end
    F = try
        _ctm_subspace_svd(apply, applyadj, rows, colsd, X0, k, _ctm_trunc(maxdim, opts), opts;
                          warm = !isnothing(prev) && nrand < kp)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    isnothing(F) && return nothing
    pr = _ctm_whiten(Acol, Brow, ins, F...)
    all(isfinite, (norm(pr[1]), norm(pr[2]))) || return nothing
    return pr
end

# One-sided truncation isometry by the subspace route — `svd(B, ins)`'s `U` (the leading left
# singular vectors of `B` matricized as `ins × rest`), for the greedy pass. Cold start only (there
# is no previous projector to seed from); `nothing` when the gate declines.
function _ctm_onesided_subspace(B, ins::Vector{<:Index}, maxdim::Integer, opts::CTMOptions,
                                seed::UInt)
    rows = _ctm_legs_of(B, ins)
    cols = _ctm_legs_of(B, uniqueinds(B, ins))
    colsd = dag(cols)
    k = min(Int(maxdim), dim(rows), dim(cols))
    kp = min(k + opts.svd_oversample, dim(rows), dim(cols))
    _ctm_use_subspace(opts, (B,), dim(rows), dim(cols), dim(rows), kp) || return nothing
    dB = dag(B)
    apply(X) = B * X
    applyadj(Q) = dB * Q
    X0 = _ctm_random_block(Xoshiro(seed), scalartype(B), colsd, kp)
    isnothing(X0) && return nothing
    F = try
        _ctm_subspace_svd(apply, applyadj, rows, colsd, adapt_like(B, X0), k,
                          _ctm_trunc(maxdim, opts; rtol = 0.0), opts)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    isnothing(F) && return nothing
    U = F[1]
    isfinite(norm(U)) || return nothing
    return U
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
# Kept index of a stored projector — always the THIRD entry: `(P_A, P_B, w)`, the greedy pass's
# `(P, dag(P), w)`, or `(P_A, P_B, w, M, M′)` with transition maps (see `_ctm_transport`).
_ctm_widx(d, k) = (t = get(d, k, nothing); isnothing(t) ? nothing : t[3])

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

# Isometry truncating index set `ins` of block `B` to `maxdim`: `B ≈ B P P†` with `P` the leading
# RIGHT singular vectors of `B` on those legs (⇔ the eigenvectors of its reduced density matrix,
# without squaring). Returns (P, w) with P legs (ins…, w).
#
# Pairing convention for the greedy pass: the block that DERIVED `P` absorbs `P`; every block at
# the OTHER end of the interface absorbs `dag(P)` (`_ctm_pAdag`). That is `B P P† B′ᵀ`, the
# optimal truncation for complex data too, and it is what makes the arrows pair up on a graded
# backend: `dag(U)` carries the legs of `ins` with the orientation opposite to `B`'s. The lossless
# branch returns the combiner, which has the same orientation convention.
# No relative cutoff here — a null direction is a harmless zero column of an isometry, whereas the
# two-sided pair INVERTS its spectrum.
function _ctm_interface_proj(B, ins::Vector{<:Index}, maxdim::Integer, opts::CTMOptions,
                             seed::UInt = UInt(0))
    (isnothing(B) || isempty(ins)) && return nothing
    #
    # Stored as the SAME triple the sweep stores, `(P_A, P_B, w)` with `P_B = dag(P_A)` (the pair a
    # one-sided isometry is), so that the first sweep can `_ctm_align` to the seed and warm-start
    # its subspace iteration from it. With a bare `(P, w)` the first sweep minted fresh kept
    # indices at every level, and since a level can only align once the level below it kept its
    # index in the previous sweep, the bases stabilised one level per sweep — `_ctm_statedist`
    # had no distance, and `update` could not certify, before sweep ~L (measured 9 sweeps on a
    # 9×9 lattice whose `|ΔF|` was at 1e-14 from sweep 2).
    if Int(maxdim) >= dim(ins)                 # nothing to truncate: keep the basis intact
        co = adapt_like(B, combiner(ins))      # the reshape isometry, on B's device/eltype (vector, not
                                               # splat: keeps the graded dispatch)
        return co, dag(co), combinedind(co)
    end
    # The subspace route when the gate takes it (`opts.svd`), the dense SVD otherwise.
    U = _ctm_onesided_subspace(B, ins, maxdim, opts, seed)
    isnothing(U) && (U = first(svd(B, ins; trunc = _ctm_trunc(maxdim, opts; rtol = 0.0))))
    P = dag(U)                                 # conj: the seam's U is conj(V) for B viewed as (rest × ins)
    return P, U, only(uniqueinds(P, ins))
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
_ctm_facs(tbl, x::Int, y::Int) = get(tbl, (x, y), AbstractTensor[])

# Flatten mixed arguments — a tensor, `nothing`, or a factor list — into one contraction list.
function _ctm_list(args...)
    ts = AbstractTensor[]
    for a in args
        isnothing(a) && continue
        a isa AbstractTensor ? push!(ts, a) : append!(ts, a)
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
function _ctm_absorb(opts::CTMOptions, core::Vector{<:AbstractTensor}, extras...)
    isempty(core) && return nothing
    ts = copy(core)
    for e in extras
        isnothing(e) || push!(ts, e)
    end
    return _ctm_contract(ts, opts)
end

# `P_A` / `P_B` of a stored projector, or `nothing`. Both passes store `(P_A, P_B, w)`; for the
# greedy pass `P_B = dag(P_A)`.
_ctm_pA(d, k) = (p = _ctm_nn(d, k); isnothing(p) ? nothing : p[1])
_ctm_pB(d, k) = (p = _ctm_nn(d, k); isnothing(p) ? nothing : p[2])
# `dag(P_A)`, which is how a block at the OTHER end of an interface consumes a projector derived by
# the block at the near end in the GREEDY pass — east/south corners and strips for a west/north
# projector, and the next-row strip for a row strip's top projector. (The sweep uses a genuine
# biorthogonal `P_B` instead.)
_ctm_pAdag(d, k) = (p = _ctm_pA(d, k); isnothing(p) ? nothing : dag(p))

function _ctm_factor_table(cache::CTMEnvironmentCache)
    Lx, Ly = _ctm_dims(cache)
    tbl = Dict{Tuple{Int, Int}, Vector{AbstractTensor}}()
    for y in 1:Ly, x in 1:Lx
        v = _ctm_vertex(cache, x, y)
        isnothing(v) && continue                      # unoccupied position (hex etc.)
        tbl[(x, y)] = Vector{AbstractTensor}(bp_factors(network(cache), v))
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
                          y > 1 ? _ctm_pAdag(PV, (:W, x + 1, y - 1)) : nothing)
        if y < Ly
            ins = Index[]
            w = _ctm_widx(PV, (:W, x, y)); !isnothing(w) && push!(ins, w)
            append!(ins, vl(x, y))
            pr = _ctm_interface_proj(raw, ins, χ, opts, hash((:W, x + 1, y)))
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
                          y > 1 ? _ctm_pAdag(PV, (:E, x, y - 1)) : nothing)
        if y < Ly
            ins = Index[]
            append!(ins, vl(x, y))
            w = _ctm_widx(PV, (:E, x + 1, y)); !isnothing(w) && push!(ins, w)
            pr = _ctm_interface_proj(raw, ins, χ, opts, hash((:E, x, y)))
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
        pr = _ctm_interface_proj(raw, ins, χ, opts, hash((:N, x - 1, y + 1)))
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
        pr = _ctm_interface_proj(raw, ins, χ, opts, hash((:S, x - 1, y)))
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
#
# `prev` is the interface's projector from the previous sweep (`(P_A, P_B, w)` or `nothing`) and
# `seed` a per-interface hash: both feed the subspace route's warm start and its reproducible
# oversampling (`_ctm_twosided_projector_subspace`). That route declines — `nothing` — when
# `opts.svd`'s gate says the block is too small to profit, or on numerical trouble; the dense
# QR+SVD route then does the work.
#
# `route`, if a `Ref{Symbol}`, reports which route produced the pair (`:subspace` or `:dense`);
# `subspace = false` skips the attempt outright (the sweep's memo, see below).
function _ctm_interface_proj2(Bw, Be, ins::Vector{<:Index}, maxdim::Integer, opts::CTMOptions,
                              prev = nothing, seed::UInt = UInt(0);
                              subspace::Bool = true, route = nothing)
    (isnothing(Bw) || isnothing(Be) || isempty(ins)) && return nothing
    if subspace
        pr = _ctm_twosided_projector_subspace(Bw, Be, ins, maxdim, opts, prev, seed)
        if !(pr isa Union{Nothing, Missing})
            isnothing(route) || (route[] = :subspace)
            _ctm_stat!(:subspace)
            return pr
        end
        isnothing(route) || (route[] = ismissing(pr) ? :declined : :dense)
        _ctm_stat!(ismissing(pr) ? :declined : :dense)
    else
        isnothing(route) || (route[] = :skipped)
        _ctm_stat!(:skipped)
    end
    return _ctm_twosided_projector_qr(Bw, Be, ins, maxdim, opts)
end

# The sweep's per-interface memo (`cache.route`): decide whether to attempt the subspace route on
# `key` this sweep, and record the outcome. A bail-out costs one to three abandoned iterations —
# 20–40% of the dense projector at n = 288 — and a flat spectrum stays flat, so after a bail the
# interface skips the attempt for `skip` sweeps, doubling up to 4 on repeated bails; a success
# clears it. On a decaying spectrum nothing is ever recorded.
function _ctm_route_try!(memo::Dict, key)
    e = get(memo, key, nothing)
    isnothing(e) && return true
    skip, backoff = e
    skip > 0 || return true
    memo[key] = (skip - 1, backoff)
    return false
end
function _ctm_route_record!(memo::Dict, key, outcome::Symbol)
    if outcome === :subspace
        delete!(memo, key)
    elseif outcome === :dense                         # a genuine bail-out; `:declined` is not one
        backoff = last(get(memo, key, (0, 1)))
        memo[key] = (backoff, min(2backoff, 4))
    end
    return memo
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
# MATRIX-FREE AND IN TENSOR FORM, so bonds may be rectangular and the corners may be graded. Bonds
# are `k_prev · D_layer` in general, and `k_prev = 1` at the boundary. Their engine pads every corner
# to a fixed χ with a separate `rank` field because a DENSE periodic Schur needs square equal-size
# factors; we only need the cycle's ACTION on a vector — four contractions, product never formed —
# so `schursolve` handles adaptive bonds natively and, on real inputs, returns a real orthonormal
# basis (no conjugate-pair handling to get wrong). The Krylov vectors ARE tensors on the west legs
# (KrylovKit needs only the VectorInterface, which both backends provide), each carrying a dim-1
# "charge" leg: trivial on dense data, one per sector on graded data — a symmetric tensor on the
# bond legs alone would live in the charge-zero sector only, and the cycle map conserves charge, so
# the graded cycle problem IS a direct sum of per-sector problems. The solve therefore runs once per
# sector (once, for dense) and the sector spectra are MERGED before every rank decision below; the
# retained bond gets one count per sector for free when the Ritz vectors are stacked by direct sum
# along their charge legs. Nothing is ever unwrapped to a matrix.
#
# RANK. `schursolve` stops when its Krylov space closes, so the resolved rank `kres` can fall short of
# what a bond could hold — the four-fold spectrum is ~the 4th power of one corner's. A shortfall is
# ZERO-PADDED, never declined: `Π = P_A P_B` keeps rank `kres` while the index width stays stable
# across sweeps (see the padding note in `_ctm_cycle_finish`). Filling the shortfall instead was tried
# two ways and both degrade with χ — see "the falsified fillers" in docs/finite_ctmrg_design.md.
#
# Consequently `F` is stationary only in the subspace the cycle RESOLVES. Measured on the 5×5 that
# suffices for machine-precision stationarity at every χ ≥ 9 (`marginal_inconsistency` 3.2e-16 /
# 2.4e-16 / 3.9e-16 at χ = 9 / 16 / 32); χ=4 reads 1.8e-04, where a fully resolved rank-4 invariant
# subspace is a worse stationary point than an under-resolved one.
#
# DOMAIN. A hex/heavy-hex plaquette can have a bond of dimension 1 (a missing lattice link), pinning
# `kcyc = 1` at every χ. Those plaquettes decline to the cut rather than collapse the interface.
#
# DETERMINISM. The start vector is seeded per plaquette from POSITION ONLY (`hash((X, Y))`) — never
# the bond dimensions, which would move the seed whenever a rank shifted. Without this the sweep is
# irreproducible run to run (⟨X⟩ at χ=16 wandered 8.1e-10 – 9.3e-10, wider than the gap between the
# two projectors). The local RNG leaves the caller's global stream untouched.

# The legs of `is` as tensor `t` carries them (same identities, `t`'s orientations).
_ctm_legs_of(t, is) = filter(i -> i ∈ is, collect(inds(t)))

# Stack vectors that agree on every leg but a dim-1 slot leg into one basis tensor — the direct
# sum along the slot legs, one column per vector. Returns the basis; its bond is the leg not in
# `vs[1]`'s other legs. (On graded data the slot legs carry the vectors' sectors, so the bond
# comes out with one count per sector.)
function _ctm_stack(vs::AbstractVector, slots::AbstractVector)
    acc, w = vs[1], slots[1]
    for j in 2:length(vs)
        acc = directsum(acc => w, vs[j] => slots[j]; tags = "Link,cyc")
        w = only(uniqueinds(acc, vs[1]))
    end
    return acc
end

# Orthonormal basis of the range of `X` over the legs `keep`: the Q of `X = Q R`.
#
# ⚠️ `qr` is UNPIVOTED with no rank check: for rank-deficient `X` the surplus columns are arbitrary
# completions outside `range(X)`, and the width-based guard below cannot detect it. Deliberate —
# pivoting it was measured (8 seeds) to be bit-identical where the deficiency never fires and slightly
# WORSE where it does, and it does not fix the 8×8 plateau. See docs/ctmrg_status.md.
_ctm_orthbasis(X, keep::Vector{<:Index}) = first(qr(X, keep))

# Whiten a pair into a biorthogonal projector pair over the interface `ins`.
#
#   O = Brow · Acol = U S Vᴴ          (bilinear contraction over `ins`)
#   P_A = Acol V S^{-1/2}  (ins… → w),   P_B = S^{-1/2} Uᴴ Brow  (w → ins…),   P_B P_A = 𝟙_w
#
# The truncation strategy owns the rank: near-null overlap directions get multiplied by `S^{-1/2}`,
# amplifying pure noise, so they must be DROPPED (the strategy's relative cutoff), never floored.
# Seam convention: `U * S * V` (bilinear) reconstructs `O`, so the returned `V` is the CONJUGATE
# of the right singular vectors — hence `dag(V)` (inert for real data). `dag(isk)` is the copy of
# S^{-1/2} (real, so only the arrows change) whose legs pair with `dag(U)`/`dag(V)` on a graded
# backend; on dense tensors arrows are inert and this is just the algebra above.
function _ctm_biorth(Acol, Brow, ins::Vector{<:Index}, trunc)
    bB = uniqueinds(Brow, ins)                           # one QR bond, or a raw block's rest legs
    isempty(bB) && return nothing                        # a block that IS its interface: `_ctm_tri_factor` it first
    O = Brow * Acol                                      # (bB…, bA…)
    U, S, V = svd(O, bB; trunc)
    return _ctm_whiten(Acol, Brow, ins, U, S, V)
end

# The whitening tail shared by the dense and subspace routes: `(U, S, V)` in the seam's `svd`
# convention for `O = Brow · Acol` (bilinear over `ins`), out come `(P_A, P_B, w)`.
function _ctm_whiten(Acol, Brow, ins::Vector{<:Index}, U, S, V)
    isk = map_diag(x -> inv(sqrt(x)), S)                 # S^{-1/2} on S's (u, v)
    PA = (Acol * dag(V)) * dag(isk)                      # (ins…, u)
    PB = (dag(U) * Brow) * dag(isk)                      # (v, ins…)
    uA = only(uniqueinds(PA, ins))
    PB = replaceind(PB, only(uniqueinds(PB, ins)), dag(uA))   # the opposite copy of P_A's bond
    return PA, PB, uA
end

function _ctm_cycle_projectors(ENW, ENE, ESE, ESW, maxdim::Integer, opts::CTMOptions,
                               seed::UInt)
    any(isnothing, (ENW, ENE, ESE, ESW)) && return nothing
    As = (ESW, ESE, ENE, ENW)                    # A_l : bond l -> bond l+1  (W->S, S->E, E->N, N->W)
    ins = (collect(commoninds(ENW, ESW)), collect(commoninds(ESW, ESE)),
           collect(commoninds(ENE, ESE)), collect(commoninds(ENW, ENE)))     # W, S, E, N
    any(isempty, ins) && return nothing
    for l in 1:4                                 # a corner carrying more than its two interfaces
        isempty(uniqueinds(As[l], vcat(ins[l], ins[mod1(l + 1, 4)]))) || return nothing
    end
    nsp = [dim(ins[l]) for l in 1:4]
    kcyc = min(Int(maxdim), minimum(nsp))
    kcyc < 1 && return nothing
    elt = scalartype(ESW)
    # Right vectors live on the west legs as ENW carries them (so `x * A₁` contracts), left vectors
    # as ESW carries them (so `u * A₄` contracts): `u · M · x` is a BILINEAR pairing, no conjugation.
    wR = _ctm_legs_of(ENW, ins[1]); wL = _ctm_legs_of(ESW, ins[1])
    act(x) = (((x * As[1]) * As[2]) * As[3]) * As[4]
    tca(u) = (((u * As[4]) * As[3]) * As[2]) * As[1]
    # Seeded on plaquette POSITION only, so the start vectors are bit-identical every sweep. Seeding
    # on the bond dimensions too let them move whenever a rank shifted, which showed up as sweep-to-
    # sweep basis wander (state distance floor 1e-10 rather than 3e-11). One charge leg per sector;
    # a sector the west legs cannot reach gives a zero start vector and is skipped.
    rng = Xoshiro(seed)
    starts = Tuple{Any, Any, Any}[]              # (charge leg, right start, left start)
    for c in charge_sectors(wR)
        xR = random_tensor(rng, elt, vcat(wR, [c]))
        xL = random_tensor(rng, elt, vcat(wL, [dag(c)]))
        (norm(xR) > 0 && norm(xL) > 0) || continue
        push!(starts, (c, xR, xL))
    end
    isempty(starts) && return nothing
    # SCALE-FREE TOLERANCE — this was the algorithm's accuracy floor, worth ~1000× at χ=32.
    #
    # KrylovKit's `tol` is ABSOLUTE on the residual, and the cycle spectrum is the PRODUCT of the four
    # factors' spectra, so it spans ~14 orders: measured on the 5×5 at χ=32, `s_k/s_1` runs
    # 1 → 4.4e-09 (k=10) → 4.0e-12 (k=22) → 4.2e-14 (k=32), against a per-factor `s_32/s_1` of ~5e-04
    # whose fourth power is ~3.5e-14. A fixed `tol = 1e-13` therefore sat ABOVE the eigenvalues being
    # resolved: Arnoldi declared an invariant subspace at k≈19-22 while directions out to k=32 were
    # still orders above machine epsilon, and the projector silently lost them.
    #
    # Normalising the action by its dominant EIGENVALUE MAGNITUDE — the spectral radius, which is
    # what power iteration converges to, NOT σ_max (five power iterations per sector — the invariant
    # subspace is scale-invariant, so it is free; the largest over sectors is the radius) makes the
    # tolerance relative. Measured `⟨X⟩` at χ=32: tol 1e-13 → 5.2e-11, 1e-15 → 7.6e-13,
    # 1e-16 → 4.9e-14.
    #
    # Do not try to make `tol` χ-adaptive. Varying it alone changes NOTHING at χ=4 (identical at
    # 1e-13/1e-14/1e-15/1e-16), and tying it to `s_kcyc/s_1` via a loose first pass collapses χ=32 to
    # 9.0e-09, because a loose pass cannot resolve 32 eigenvalues and so reads the tail off the wrong
    # one. The χ=4 cost that remains is the criterion, not the solver — see the docstring.
    scale = zero(real(elt))
    for (_, xR, _) in starts
        v = xR / norm(xR); sc = zero(real(elt))
        for _ in 1:5
            w = act(v); nw = norm(w)
            (isfinite(nw) && nw > 0) || break
            v, sc = w / nw, nw
        end
        scale = max(scale, sc)
    end
    (scale > 0 && isfinite(scale)) || (scale = one(real(elt)))
    fwd(x) = act(x) / scale
    bwd(u) = tca(u) / scale
    # `verbosity = 0`: `schursolve` stopping on a closed invariant subspace smaller than `kcyc` is
    # ROUTINE here (the four-fold spectrum runs out before the bond does), handled by `kres` below —
    # KrylovKit's per-call warning for it buries real problems in noise.
    alg = Arnoldi(; krylovdim = max(4kcyc + 8, 24), tol = 1.0e-16, verbosity = 0)
    # One solve per sector and side. Every entry: (|λ|, sector, position in that sector's Schur
    # list, converged?) — the sector lists are then MERGED by magnitude for all the rank decisions.
    vecsR = Vector{Any}(undef, length(starts)); vecsL = Vector{Any}(undef, length(starts))
    entR = NamedTuple{(:mag, :s, :j, :ok), Tuple{Float64, Int, Int, Bool}}[]
    entL = similar(entR)
    try
        for (si, (c, xR, xL)) in enumerate(starts)
            k = min(kcyc, length(data(xR)))     # a sector holds at most its own dimension
            _, VRv, valsR, iR = schursolve(fwd, xR, k, :LM, alg)
            _, VLv, valsL, iL = schursolve(bwd, xL, k, :LM, alg)
            vecsR[si] = VRv; vecsL[si] = VLv
            append!(entR, ((mag = Float64(abs(valsR[j])), s = si, j = j, ok = j <= iR.converged) for j in eachindex(valsR)))
            append!(entL, ((mag = Float64(abs(valsL[j])), s = si, j = j, ok = j <= iL.converged) for j in eachindex(valsL)))
        end
    catch err
        err isa InterruptException && rethrow()
        return nothing                          # fall through to the pairwise cut
    end
    sort!(entR; by = e -> -e.mag); sort!(entL; by = e -> -e.mag)
    # Solve the cycle at the rank it can actually RESOLVE. `schursolve` terminates when the Krylov
    # space closes, which at an interior plaquette is ~19 of a requested 32: the four-fold product's
    # spectrum is ~the 4th power of one corner's, so directions past that carry no cycle weight.
    # Values beyond `info.converged` are unconverged Ritz estimates — fine for a gap test below,
    # never used as retained modes.
    nv = min(length(entR), length(entL))
    nv < 1 && return nothing
    aR = [entR[j].mag for j in 1:nv]; aL = [entL[j].mag for j in 1:nv]
    kres = min(kcyc, nv, something(findfirst(e -> !e.ok, entR), nv + 1) - 1,
               something(findfirst(e -> !e.ok, entL), nv + 1) - 1)
    kres < 1 && return nothing
    # LEFT/RIGHT SPECTRAL CONSISTENCY (always on). `λ(Mᵀ) = λ(M)`, so the two solves must retain
    # the SAME spectrum; where their magnitudes disagree, the "pair" spans two DIFFERENT spectral
    # sets and `Π = P_A·P_B` is not a spectral projector of anything — it is an arbitrary oblique
    # projector redrawn every sweep. Measured (over-parametrised 5×5 TFIM χ=16): kres = 14 with a
    # 44% magnitude mismatch — ten noise modes, drawn differently on each side, which IS the
    # residual-grows-with-χ wander. Keep the longest agreeing prefix. The 1e-3 tolerance is ~20×
    # looser than the worst healthy case measured (4.3e-5) and ~400× tighter than the failure.
    # Magnitudes over the FULL merged lists, not just the retained prefix: the degtol back-off
    # below must compare the retained boundary `aR[kres]` against the first DROPPED value
    # `aR[kres+1]`, which only exists if the list extends past the cut.
    for j in 1:kres
        if abs(aR[j] - aL[j]) > 1.0e-3 * max(aR[j], aL[j])
            kres = j - 1
            break
        end
    end
    kres < 1 && return nothing
    # NOISE-CLIFF RANK CUT (`opts.cycle_gapcut`, 0 disables). Truncate the trailing block below
    # the first cliff that is BOTH steep (`aR[j+1] ≤ gapcut·aR[j]`) and genuinely tiny
    # (`aR[j+1] ≤ √eps·aR[1]`). Magnitude alone cannot separate noise from deep-but-real modes —
    # the falsified fixed `cycle_rankcut` default: junk sits at 1.6e-11·|λ_1| on one measured
    # case while REAL weight sits at 6.4e-13·|λ_1| on another — but the CLIFF can: measured
    # 1.8e5 into the noise block against ≤ 4.5e2 anywhere inside a physical decay. The two
    # conditions guard each other: a genuine spectral gap of ~1e4 with real modes below it fails
    # the tininess floor (the cycle spectrum is ~ the 4th power of a corner's, so modest corner
    # gaps make large cycle cliffs), and a smooth decay into tininess fails the cliff.
    if opts.cycle_gapcut > 0
        fl = sqrt(eps(real(elt))) * aR[1]
        for j in 1:(kres - 1)
            if aR[j + 1] <= opts.cycle_gapcut * aR[j] && aR[j + 1] <= fl
                kres = j
                break
            end
        end
    end
    # `degtol` back-off — the SAME semantics as the cut path: never split a near-degenerate cluster
    # at the cut. An invariant subspace that must split a cluster is ill-defined, and the sweep
    # re-resolves it differently each time — the under-truncation limit cycle lands EXACTLY on one
    # (measured `|λ_8| = |λ_9|` to displayed digits on random 5×5 at χ=8). At the default
    # `degtol = 0` the `≤` still fires on EXACT magnitude ties — which on ⟨ψ|ψ⟩ networks are the
    # (λ, conj λ) pairs the swap identity `conj(M) = S·M·S` forces on the spectrum (see
    # docs/ctmrg_status.md, the falsified swap-symmetry entry) — so a conjugate pair straddling the
    # cut is never split even with the knob off. On graded data a tie across two sectors is
    # harmless in itself (the sector label separates the modes), but it is backed off all the same.
    while kres > 1 && kres < length(aR) && abs(aR[kres] - aR[kres + 1]) <= opts.degtol * abs(aR[kres])
        kres -= 1
    end
    # RANK-CAP (opts.cycle_rankcut > 0): drop the near-null tail of the cycle spectrum so that a χ
    # larger than the state's rank does not carry arbitrary null modes. Off by default → committed.
    if opts.cycle_rankcut > 0
        capR = count(>(opts.cycle_rankcut * aR[1]), @view aR[1:kres])
        capL = count(>(opts.cycle_rankcut * aL[1]), @view aL[1:kres])
        kres = min(kres, capR, capL)
        kres < 1 && return nothing
    end
    # Retained bases: the leading `kres` of the merged RIGHT list, stacked along their charge legs;
    # the left side takes the SAME number of modes per sector (its own leading ones in each sector).
    # Per sector the two solves see one block and its transpose, so their spectra coincide and the
    # counts must agree — letting the left side pick its own leading `kres` would let a cross-sector
    # near-tie at the cut hand the two sides different sector contents, leaving `P_B P_A` rank-
    # deficient in one sector and over-complete in another.
    keepR = entR[1:kres]
    keepL = eltype(entL)[]
    for si in eachindex(starts)
        nR = count(e -> e.s == si, keepR)
        candL = filter(e -> e.s == si && e.ok, entL)
        length(candL) >= nR || return nothing        # the left solve resolved fewer modes here
        append!(keepL, candL[1:nR])
    end
    VR = Vector{Any}(undef, 4); VL = Vector{Any}(undef, 4)
    VR[1] = _ctm_stack([vecsR[e.s][e.j] for e in keepR], [starts[e.s][1] for e in keepR])
    VL[1] = _ctm_stack([vecsL[e.s][e.j] for e in keepL], [dag(starts[e.s][1]) for e in keepL])
    # Propagate the invariant subspace around the plaquette: right bases forward, `V_R[l+1] ∝ A_l V_R[l]`;
    # left bases backward, `V_L[l] ∝ V_L[l+1] A_l` — each re-orthonormalised.
    for l in 1:3
        VR[l + 1] = _ctm_orthbasis(As[l] * VR[l], ins[l + 1])
    end
    for l in (4, 3, 2)
        VL[l] = _ctm_orthbasis(VL[mod1(l + 1, 4)] * As[l], ins[l])
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
    # this is the same trick, and it is bookkeeping rather than physics. (On graded data the padding
    # lands in whatever sectors `new_index` allots — zero columns carry no weight, so which sector
    # they sit in is immaterial to every region value.)
    #
    # The padding must be applied AFTER `_ctm_biorth`, never before: whitening a pair with null
    # columns inverts a singular overlap, which is the `S^(-1/2)` amplification `qr_cutoff` guards
    # against. Build the pair at `kres`, then embed.
    out = Vector{Any}(undef, 4)
    for l in 1:4
        (dim(only(uniqueinds(VR[l], ins[l]))) == kres && dim(only(uniqueinds(VL[l], ins[l]))) == kres) ||
            return nothing
        Acol = (l <= 2) ? VL[l] : VR[l]                      # the P_A side
        Brow = (l <= 2) ? VR[l] : VL[l]                      # the P_B side
        ab = try
            _ctm_biorth(Acol, Brow, ins[l], truncation_strategy(; maxdim = kres, rtol = opts.qr_cutoff))
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
        isnothing(ab) && return nothing
        a, b, w = ab                                         # b * a = 𝟙 exactly
        (isfinite(norm(a)) && isfinite(norm(b))) || return nothing
        kt = target(l); k = dim(w)
        if k < kt                                            # embed at rank, pad the rest with zeros
            z = new_index(a, kt - k; tags = "Link,pad")
            z = z.dual == w.dual ? z : dag(z)                # pad leg oriented like P_A's bond
            za = random_tensor(elt, vcat(_ctm_legs_of(a, ins[l]), [z])) * zero(elt)
            zb = random_tensor(elt, vcat(_ctm_legs_of(b, ins[l]), [dag(z)])) * zero(elt)
            a = directsum(a => w, za => z; tags = "Link,cyc")
            b = directsum(b => w, zb => dag(z); tags = "Link,cyc")
            w = only(uniqueinds(a, ins[l]))
            b = replaceind(b, only(uniqueinds(b, ins[l])), dag(w))
        end
        out[l] = (a, b, w, ins[l])
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

#
# Interface bases are nested: `ins` of one interface carries the kept index of the interface one
# level closer to the lattice edge in the same chain. So a level whose kept index changed (a rank
# change, a failed alignment) used to force EVERY level above it to mint a fresh index in the next
# sweep — the previous projector there lived on the old lower index — and the change crawled
# inward one level per sweep. Measured on a 9×9 lattice: 33 rank mismatches between the greedy
# seed and the first sweep, and `_ctm_statedist` had no distance until sweep 8 although `|ΔF|`
# had been 1e-14 since sweep 2.
#
# The remedy is a TRANSITION MAP recorded at the moment a level re-mints. With the old pair
# `(P_A⁰, P_B⁰, w⁰)` and the new pair `(P_A, P_B, w)` on the same raw legs, `M = P_B·P_A⁰` (w ← w⁰)
# and `M′ = P_B⁰·P_A` (w⁰ ← w) map between the two kept bases, and `_ctm_remint` stores them as a
# fourth and fifth entry of the new triple. The next sweep's corners reach the level above carrying
# `w`, while that level's previous projector still consumes `w⁰`; the west corner there is
# `B·P_A⁰` in the old basis and `B·P_A` in the new, and since `P_A P_B` is the identity on what the
# new pair keeps, `B·P_A⁰·P_A_prev ≈ B·P_A·(P_B P_A⁰)·P_A_prev = B·P_A·M·P_A_prev`. So
# `_ctm_transport` re-expresses the previous projector as `M·P_A_prev` and, by the mirror argument,
# `P_B_prev·M′` — exact when the two truncations keep the same subspace, a sound approximation
# otherwise, and all it seeds is the warm start and the gauge reference; the projector itself is
# derived afresh. On a graded backend the orientations pair up because every contraction joins a
# `P_A` leg with a `P_B` leg of the same interface. A level whose own raw legs changed (the level
# below IT re-minted without a map) records no map, and the change crawls one level that sweep, as
# before.
#
# Tuple layout, everywhere: `(P_A, P_B, w)` or `(P_A, P_B, w, M, M′)`; `w` is always the third.
#
# The four interface families, each walked from the lattice edge inward along its nested chain,
# so that a level is visited only after the level whose kept index it carries. Calls
# `f(isH, key, below, cornerA, cornerB)` with `isH` selecting the `PH`/`PV` family, `below` the
# key of the lower level (absent at the edge), and the two enlarged-corner descriptors `(sym, x, y)`
# bounding the interface. Every `:S` block is keyed by its FIRST included row (`T_S[x,y] = rows ≥ y`),
# so that family lives at `y ∈ 2:Ly`; `:E` likewise at `x ∈ 2:Lx`.
function _ctm_each_interface(f, Lx::Int, Ly::Int)
    for x in 1:(Lx - 1), y in 2:Ly            # PH[:N,x,y]: C_NW(x+1,y) | C_NE(x+1,y); chain grows with y
        f(true, (:N, x, y), (:N, x, y - 1), (:NW, x + 1, y), (:NE, x + 1, y))
    end
    for x in 1:(Lx - 1), y in Ly:-1:2         # PH[:S,x,y]: C_SW(x+1,y) | C_SE(x+1,y); chain grows as y falls
        f(true, (:S, x, y), (:S, x, y + 1), (:SW, x + 1, y), (:SE, x + 1, y))
    end
    for x in 2:Lx, y in 1:(Ly - 1)            # PV[:W,x,y]: C_NW(x,y+1) | C_SW(x,y+1); chain grows with x
        f(false, (:W, x, y), (:W, x - 1, y), (:NW, x, y + 1), (:SW, x, y + 1))
    end
    for x in Lx:-1:2, y in 1:(Ly - 1)         # PV[:E,x,y]: C_NE(x,y+1) | C_SE(x,y+1); chain grows as x falls
        f(false, (:E, x, y), (:E, x + 1, y), (:NE, x, y + 1), (:SE, x, y + 1))
    end
    return nothing
end

function _ctm_transport(prev, below)
    (isnothing(prev) || isnothing(below) || length(below) < 5) && return prev
    M, Mt = below[4], below[5]
    wold = only(uniqueinds(M, [below[3]]))              # the lower index `prev` was derived on
    wold ∈ inds(prev[1]) || return prev                  # not on the old lower index: nothing to do
    PA = M * prev[1]                                     # (w, w⁰) · (w⁰, links…, kept)
    PB = prev[2] * Mt                                    # (kept, w⁰, links…) · (w⁰, w)
    (isfinite(norm(PA)) && isfinite(norm(PB))) || return prev
    return (PA, PB, prev[3])
end

# The new pair `pr` could not keep `prev`'s index: attach the transition maps when both live on the
# same raw legs `ins` (otherwise the level above cannot be helped this sweep).
function _ctm_remint(pr, ins, prev)
    (isnothing(prev) || length(prev) < 3 || isnothing(pr)) && return pr
    pr[3] == prev[3] && return pr                         # aligned after all
    issetequal(collect(inds(prev[1])), vcat(collect(ins), [prev[3]])) || return pr
    M = pr[2] * prev[1]                                   # (w, w⁰): P_B · P_A⁰ over the raw legs
    Mt = prev[2] * pr[1]                                  # (w⁰, w): P_B⁰ · P_A
    (isfinite(norm(M)) && isfinite(norm(Mt))) || return pr
    return (pr[1], pr[2], pr[3], M, Mt)
end

function _ctm_align(pr, ins, prev)
    (isnothing(prev) || length(prev) < 3) && return pr
    PA, PB, w = pr
    PAo, _, wo = prev
    dim(wo) == dim(w) || return pr
    issetequal(collect(inds(PAo)), vcat(collect(ins), [wo])) || return pr   # same raw space?
    R = try
        M = gram(PA, PAo, ins)                           # (w, wo) = P_A† P_A⁰ over the raw legs (Hilbert, any backend)
        U, S, V = svd(M, [w])
        # nearest unitary U Vᴴ: the seam's `V` is already conj(V_true), so the product is bilinear;
        # relabel V's bond to the copy opposite U's so the two contract on any backend.
        uU = only(uniqueinds(U, [w]))
        U * replaceind(V, only(uniqueinds(V, [wo])), dag(uU))
    catch err
        err isa InterruptException && rethrow()
        return pr                                        # any other trouble: keep the unaligned pair
    end
    all(isfinite, (norm(PA), norm(PAo), norm(PB), norm(R))) || return pr
    # The alignment must preserve `Π = P_A P_B`, i.e. R must be unitary. Equal total dimension does
    # not guarantee that on a graded bond: the old and new bonds can distribute the same width over
    # the sectors differently (zero-padding lands wherever `new_index` puts it), and a Procrustes map
    # between two sector structures is rectangular blockwise. Measured without this guard: graded
    # `:cycle` at χ=8 with `degtol = 1e-8` read `F` off by 3.8 (Z2 4×4 D=4). Checked on the pair's
    # own product rather than against a bare identity, so fermionic parity conventions cannot
    # trip it: `P_B P_A` before and after, relabelled onto the same bond, must agree.
    PAn, PBn = PA * R, dag(R) * PB
    #
    # Unitarity is tested through what a unitary preserves — the norm of each factor — rather than
    # through `P_B P_A` before and after. That product is `𝟙 + E` with `E` the pair's own
    # biorthogonality defect, `‖E‖ ~ eps · σ₁/σ_k`, which at the default `qr_cutoff` reaches
    # ~1e-3 on a lossless interface (σ_k at the cutoff), and `‖R†(𝟙+E)R − (𝟙+E)‖ = ‖R†ER − E‖` is
    # then ~‖E‖ for a perfectly unitary `R`. Measured (6×6 D=3 TFIM PEPS, χ=48): that comparison
    # rejected every correctly aligned dim-48 interface, the level above then re-indexed, and
    # `_ctm_statedist` had no distance for 100 sweeps. The product is still formed, relabelled onto
    # the old bond, because `replaceinds` is what detects a sector-structure mismatch.
    ok = try
        before = replaceind(PB, w, prime(w)) * PA                         # (w', w)
        after = replaceind(PBn, wo, prime(wo)) * PAn                      # (wo', wo)
        after = replaceinds(after, [prime(wo), wo], [prime(w), w])        # errors if the sector structures differ
        isfinite(norm(after - before)) &&
            abs(norm(PAn) - norm(PA)) <= 1.0e-8 * norm(PA) &&
            abs(norm(PBn) - norm(PB)) <= 1.0e-8 * norm(PB)
    catch err
        err isa InterruptException && rethrow()
        false
    end
    ok || return pr
    return (PAn, PBn, wo)
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
        na = norm(ta); nb = norm(tb)
        (na > 0 && nb > 0) || continue
        # Direction distance, immune to a block's overall phase: blocks are norm-1 (`_ctm_rescale`),
        # so `|ta − tb|² = 2 − 2 Re⟨ta,tb⟩`, and we use `2 − 2|⟨ta,tb⟩|` instead. A block's phase is
        # gauge — `F` and every observable are ratios/products of blocks that appear an even number
        # of times, so a sign flip of a block changes nothing physical — but `|ta − tb|` read it as a
        # change of 2. Measured on the fermionic 3×3 D=3 (fZ2) at χ=4/16: half the interior blocks
        # came back as minus themselves every sweep (Re⟨ta,tb⟩ = −0.9999 with |⟨ta,tb⟩| = 0.9999)
        # while |ΔF| sat at 1e-15, so the sweep never certified and ran to `maxiter` every time.
        # Where no phase flips occur (dense data) this equals the old distance.
        ov = min(abs(dot(ta, tb)) / (na * nb), 1.0)
        d = sqrt(max(0.0, 2 - 2ov))
        worst = max(worst, d)
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
function sweep_vertex_environments(cache::CTMEnvironmentCache, S::CTMVertexEnvironments,
                                   tbl = _ctm_factor_table(cache))
    Lx, Ly = S.Lx, S.Ly
    χ = cache.maxdim
    opts = cache.options
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
    #
    # Interface bases are nested chains (see `_ctm_transport`), and both passes finish every pair
    # the same way — transport the previous projector onto the current lower basis, align to it,
    # record transition maps if the index could not be kept — walking each chain from the lattice
    # edge inward so the lower level is always finished first. `_ctm_each_interface` is that walk.
    function finish!(dnew, key, pr, ins, prev)
        isnothing(pr) && return nothing
        if opts.gauge
            pr = _ctm_align(pr, ins, prev)
            pr = _ctm_remint(pr, ins, prev)
        end
        dnew[key] = pr
        return nothing
    end
    if opts.projector === :cycle
        ncyc = ndec = 0
        cyc_pairs = Dict{Tuple{Symbol, Int, Int}, Any}()   # key => (pair, ins), finished below
        for X in 2:Lx, Y in 2:Ly
            cyc = _ctm_cycle_projectors(E(:NW, X, Y), E(:NE, X, Y), E(:SE, X, Y), E(:SW, X, Y),
                                        χ, opts, hash((X, Y)))
            if isnothing(cyc)
                ndec += 1
                continue
            end
            ncyc += 1
            for (fam, key) in ((cyc.N, (:N, X - 1, Y)), (cyc.S, (:S, X - 1, Y)),
                               (cyc.W, (:W, X, Y - 1)), (cyc.E, (:E, X, Y - 1)))
                cyc_pairs[key] = ((fam[1], fam[2], fam[3]), fam[4])
            end
        end
        _ctm_each_interface(Lx, Ly) do isH, key, below, _, _
            haskey(cyc_pairs, key) || return nothing
            dnew, dold = isH ? (PH, S.PH) : (PV, S.PV)
            pr, ins = cyc_pairs[key]
            finish!(dnew, key, pr, ins, _ctm_transport(_ctm_nn(dold, key), _ctm_nn(dold, below)))
        end
        # Silence here would read as "the cycle projector was used everywhere", which is the one
        # thing a reader must not assume when comparing the two options.
        ndec > 0 && @warn "projector = :cycle declined $ndec of $(ncyc + ndec) plaquettes, whose \
            cycle is undefined. Causes are GEOMETRIC (a corner carrying more than its two \
            interfaces, a rank-collapsed hex plaquette, an empty interface) or NUMERICAL (a \
            `schursolve` that threw, an orthonormalisation shortfall, no above-cutoff overlap \
            direction in `_ctm_biorth`, or a non-finite whitened pair) — do not assume geometry. \
            Those interfaces used the cut, so this run is NOT a pure `:cycle` result — see \
            `_ctm_cycle_projectors`." maxlog = 1
    end
    # --- projector pass 2 of 2: the CUT projector, from each interface's two bounding corners.
    # Under `:cut` this owns everything; under `:cycle` it backfills whatever pass 1 declined.
    # Each interface hands its previous-sweep projector to the derivation (the subspace route's
    # warm start, see `_ctm_twosided_projector_subspace`) and then to `_ctm_align`; the seed is
    # the interface's position, so the oversampling draw is reproducible sweep to sweep. The
    # cache's route memo decides whether the subspace route is attempted at all on this interface
    # and records how it went (`_ctm_route_try!` / `_ctm_route_record!`).
    #
    # The previous projector (transported onto the current lower basis) seeds the subspace
    # route's warm start before the derivation and the alignment after it; the cache's route memo
    # decides whether the subspace route is attempted on this interface at all.
    route = Ref(:dense)
    _ctm_each_interface(Lx, Ly) do isH, key, below, ca, cb
        dnew, dold = isH ? (PH, S.PH) : (PV, S.PV)
        haskey(dnew, key) && return nothing            # `:cycle` pass 1 owns it
        Ba = E(ca...); Bb = E(cb...)
        (isnothing(Ba) || isnothing(Bb)) && return nothing
        ins = commoninds(Ba, Bb)
        prev = _ctm_transport(_ctm_nn(dold, key), _ctm_nn(dold, below))
        attempt = _ctm_route_try!(cache.route, key)
        pr = _ctm_interface_proj2(Ba, Bb, ins, χ, opts, prev, hash(key); subspace = attempt, route)
        attempt && _ctm_route_record!(cache.route, key, route[])
        finish!(dnew, key, pr, ins, prev)
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
    return AbstractTensor[t for t in (_ctm_fetch(env, d) for d in ds) if !isnothing(t)]
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
    vertex_window(cache::CTMEnvironmentCache, v, w::Integer = 0) -> Vector{AbstractTensor}

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
    ts = AbstractTensor[]
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
    vertex_ring(cache::CTMEnvironmentCache, v) -> Vector{AbstractTensor}

The `4C + 4T` ring enclosing `v` — [`vertex_window`](@ref) at `w = 0`. Its open legs are exactly
`v`'s ket and bra virtual indices, so it pairs directly with `norm_factors(ψ, v; op_strings)`.
"""
vertex_ring(cache::CTMEnvironmentCache, v) = vertex_window(cache, v, 0)

# Grid position of a vertex, by lookup rather than by trusting `v == (x, y)` — a network's
# vertices need not be 1-based or contiguous. O(1) via the `coords` inverse map (a linear scan
# here made a lattice-wide observable pass O(V²)).
function _ctm_coords(cache::CTMEnvironmentCache, v)
    pos = get(cache.coords, v, nothing)
    isnothing(pos) && error("vertex $v is not in the CTMEnvironmentCache's grid.")
    return pos
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

# One pass over the region grid returning BOTH the Möbius free energy and the raw per-region ln Z
# values, so the convergence test can watch the WORST region's change (see `update`) at no extra
# cost. The half-integer grid enumerates every region exactly once — integer/integer is a vertex
# (+1), one half-integer an edge (−1), both a plaquette (+1) — and `_ctm_region_desc` knows the
# Möbius weight. `region_lnZ` returns a real `Float64` (a `log(abs(...))`) and `0.0` for absent
# regions, so `vals` is a fixed-length real vector every sweep.
function _ctm_region_terms(env::CTMVertexEnvironments, cache::CTMEnvironmentCache)
    RT = _ctm_real_eltype(cache)        # keep the network's working precision (Float32 stays Float32)
    Lx, Ly = env.Lx, env.Ly
    vals = RT[]
    F = zero(RT)
    for cx in 1.0:0.5:Lx, cy in 1.0:0.5:Ly
        z = convert(RT, region_lnZ(env, cache, cx, cy))
        push!(vals, z)
        F += convert(RT, _ctm_region_desc(cx, cy)[2]) * z
    end
    return F, vals
end

cvm_freenergy(env::CTMVertexEnvironments, cache::CTMEnvironmentCache) =
    _ctm_region_terms(env, cache)[1]

# Normalised single-vertex marginals, the stationarity witness behind `convergence = :marginal`.
#
# For a `TensorNetworkState` this is the vertex's reduced density matrix as its own ring produces
# it — ring × ket × bra with the site legs left open — normalised; for any other network (a
# single-layer partition function, a form) it is the ring alone, open on the vertex's virtual
# legs, normalised: the environment every local quantity at `v` is read from. Both are
# GAUGE-INVARIANT (a closed contraction up to the open site legs: the interface gauge cancels, and
# the per-block rescaling is a scalar the normalisation removes) and FULL-COVERAGE (every vertex,
# every sweep). Cost: one ring contraction per vertex, a small addition to `_ctm_region_terms`.
#
# WHY NOT THE REGION VALUES. `:worst_region` watches the per-region `|Δ lnZ_r|`, and once χ
# exceeds an interface's rank the surplus modes wander from sweep to sweep. That wander is
# invisible to any observable — the ring's numerator and denominator share it — but a region's
# `lnZ_r` is a single number that does not, so the signal floors: measured on a converged 6×6 D=3
# TFIM PEPS at χ=32 with `:cycle`, `|ΔF| ~ 1e-15`, `⟨X⟩` agreeing with `:cut` to 4e-14, and the
# worst region stuck at 1.5e-7 for ever, so `update` ran to `maxiter` at 12× the converged cost.
# The marginal is what the observable sees, so it settles when the observable does.
function _ctm_vertex_marginals(env::CTMVertexEnvironments, cache::CTMEnvironmentCache)
    net = network(cache); opts = cache.options
    out = Dict{Any, Any}()
    for ((x, y), v) in cache.grid
        ts = _ctm_region_blocks(env, x, y)
        isempty(ts) && continue
        net isa TensorNetworkState && append!(ts, norm_factors(net, [v]; op_strings = _ -> "ρ"))
        m = try
            _ctm_contract(ts, opts)
        catch err
            err isa InterruptException && rethrow()
            continue
        end
        n = norm(m)
        (isfinite(n) && n > 0) || continue
        out[v] = m / n
    end
    return out
end

# Largest distance between two marginal sets, immune to a block's sign/phase gauge like
# `_ctm_statedist` — but as the norm of `a − e^{iφ} b` with the phase read off `⟨a, b⟩`, not as
# `√(2 − 2|⟨a, b⟩|)`: that form cancels at machine precision and cannot read below ~1e-8, which is
# above the tolerances this signal certifies. `nothing` if the sets do not cover the same
# vertices on the same legs.
function _ctm_marginal_distance(a::Dict, b::Dict)
    worst = 0.0; n = 0
    for (v, ma) in a
        mb = get(b, v, nothing)
        isnothing(mb) && return nothing
        Set(inds(ma)) == Set(inds(mb)) || return nothing
        ov = dot(ma, mb)
        ph = abs(ov) > 0 ? ov / abs(ov) : one(ov)
        worst = max(worst, norm(ma - ph * mb))
        n += 1
    end
    return n == 0 ? nothing : worst
end

"""
    marginal_inconsistency(cache::CTMEnvironmentCache) -> Real

How far the cache is from a genuine CVM/BP fixed point, as `mean(1 − |cos(M_v, M_e)|)` over the
edge-like blocks. **This is the only `ln Z`-free quality measure available, and the only one safe
to optimise against.**

The absolute value is deliberate: `_ctm_rescale` leaves a sign gauge, so anti-parallel marginals are
as consistent as parallel ones. Returns `NaN` — never `0.0` — when no block could be measured, since
`0.0` is this metric's BEST value and would report perfect stationarity from an empty sample.

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
    tbl = _ctm_factor_table(cache)
    nxt = sweep_vertex_environments(cache, env, tbl) # its PH/PV are consistent with `env`
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
        Ms = AbstractTensor[]
        for (ds, _, ctr) in rs
            full = AbstractTensor[]; minus = AbstractTensor[]
            for e in ds
                t = blk(e); isnothing(t) && continue
                push!(full, t); e == d || push!(minus, t)
            end
            if !isnothing(ctr) && haskey(tbl, ctr)
                append!(full, tbl[ctr]); append!(minus, tbl[ctr])
            end
            (isempty(full) || isempty(minus)) && continue
            Z = try
                scalar(_ctm_contract(full, opts))
            catch err
                err isa InterruptException && rethrow()
                continue
            end
            (!isfinite(Z) || iszero(Z)) && continue
            push!(Ms, _ctm_contract(minus, opts) / Z)
        end
        length(Ms) == 2 || continue
        a, b = Ms
        Set(inds(a)) == Set(inds(b)) || continue
        is = inds(a)
        va = vec(array(a, is...)); vb = vec(array(b, is...))
        na = norm(va); nb = norm(vb)
        (iszero(na) || iszero(nb)) && continue
        # clamp: `cos` can marginally exceed 1 in roundoff, and this is a distance
        push!(gaps, max(zero(Float64), 1 - abs(dot(va, vb)) / (na * nb)))
    end
    # NOT `0.0` when nothing was measured. `0` is this metric's BEST value, so an empty sample would
    # report PERFECT stationarity — and the sample is self-selected: the `continue`s above (including
    # the swallowed contraction failure) each drop a block silently. That is exactly the defect that
    # let `_ctm_statedist` certify convergence off 13% of the state, in the metric this file calls
    # the only one safe to optimise against. `NaN` makes every `<` threshold test evaluate `false`,
    # so a caller fails loudly instead of reading "perfectly stationary".
    return isempty(gaps) ? NaN : sum(gaps) / length(gaps)
end

"""
    update(cache::CTMEnvironmentCache; maxiter = 30, tolerance = _ctm_default_tol(cache),
           convergence = :free_energy, verbose = false)

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
and reads as a limit cycle. Warns if `tolerance` is not met within `maxiter` — except under
`verbose = true`, where the same message is `println`ed rather than warned.

The default `tolerance` is precision-aware — `max(1e-10, 1e3·eps)` of the network's working real
type — so a Float32 network converges at its own roundoff floor instead of spinning to `maxiter`.

`convergence` selects the stopping signal (see docs/ctmrg_status.md, "The convergence test"):

- `:free_energy` (default): stop when the free energy settles. For `:cut` that is
  `max(|ΔF|, statedist²)` (the raw C/T state distance, available when `gauge` is on); for
  `:cycle` it is `|ΔF|` alone. Right — and tightest — when the endpoint is a scalar (`ln Z`, an
  overlap, a free energy). ⚠️ `F` is a *stationary functional* of the messages, so it settles a
  few sweeps BEFORE a boundary-lagged single-site observable does.
- `:worst_region`: additionally require the worst region's `|Δ lnZ_r|` — over the same Möbius
  regions `F` sums, computed in the same pass, so no extra per-sweep cost — to settle. This is
  the message-stationarity signal a `:cycle` observable needs; [`expect`](@ref) and
  [`reduced_density_matrix`](@ref) select it automatically for `:cycle` caches they build
  internally. Honored for `:cut` too (added on top of its statedist pair), but rarely useful
  there: `:cut`'s own pair is already observable-tight, and the worst-region signal can floor
  above tolerance while the observable is exact (measured ~8e-6 on lossless heavy-hex). It also
  over-warns on over-parametrised `:cycle` states (χ above the state's rank), where surplus null
  modes wander although `F` and the observable are converged.
- `:marginal`: additionally require every vertex's normalised single-site marginal — its reduced
  density matrix as its own ring produces it (the ring alone, on the vertex's virtual legs, for a
  network without site legs) — to stop changing (`_ctm_vertex_marginals`). Gauge invariant and
  full coverage like `:worst_region`, but it watches exactly what an observable reads, so the
  surplus-mode wander that floors `:worst_region` on over-parametrised states cancels out of it
  (measured: 6×6 D=3 TFIM PEPS at χ=32 with `:cycle`, `:worst_region` floors at 1.5e-7 and runs
  to `maxiter` while `:marginal` certifies once `⟨X⟩` is stationary). This is what
  [`expect`]() and [`reduced_density_matrix`]() select for the `:cycle` caches they build
  internally. One extra ring contraction per vertex per sweep.
"""
# Working real precision of the stored network — sets the reachable convergence floor.
_ctm_real_eltype(cache::CTMEnvironmentCache) = real(eltype(datatype(network(cache))))
# Default convergence tolerance, PRECISION-AWARE. `|ΔF|` and the state distance bottom out at the
# working roundoff (~1e-7 in Float32), so a fixed 1e-10 is UNREACHABLE below Float64 and the sweep
# spins to `maxiter` — wasting sweeps (and, on GPU, the transient memory they churn). `1e3·eps`
# recovers ~1.2e-4 for Float32 while leaving Float64 at exactly 1e-10 (1e3·eps(Float64) ≈ 2e-13).
_ctm_default_tol(cache::CTMEnvironmentCache) = max(1.0e-10, 1.0e3 * eps(_ctm_real_eltype(cache)))

function update(cache::CTMEnvironmentCache; maxiter::Integer = 30,
                tolerance::Real = _ctm_default_tol(cache), verbose::Bool = false,
                convergence::Symbol = :free_energy)
    convergence in (:free_energy, :worst_region, :marginal) || throw(ArgumentError(
        "convergence must be :free_energy (default), :worst_region or :marginal; got $(repr(convergence))"))
    env = _ctm_env(cache)
    opts = cache.options
    tbl = _ctm_factor_table(cache)     # geometry is fixed across sweeps; build the table once
    # CONVERGENCE SIGNAL — see the docstring and docs/ctmrg_status.md, "The convergence test".
    #
    # Base signal, per projector. `:cut` pairs `|ΔF|` with `_ctm_statedist²` (raw C/T distance,
    # gauge on) — `|ΔF|` alone false-certifies, since `F` is a signed Möbius sum whose ~4000×
    # cancellation can sit at its final value while the state is still the greedy seed. `:cycle`
    # uses `|ΔF|` alone: a raw state distance is the WRONG metric there — not gauge invariant, so at
    # a stationary cycle fixed point whose interface basis merely rotates it reports ~1 forever, and
    # its wander spuriously GROWS with χ once χ exceeds the state's rank.
    #
    # `convergence = :worst_region` ADDS the worst region's `|Δ lnZ_r|` to the base signal, for
    # EITHER projector (the region terms are projector-independent). Gauge invariant (each region is
    # a CLOSED contraction, so the interface gauge cancels) and non-cancelling (the MAX exposes a
    # laggy region the signed `F`-sum hides), it waits for genuine message stationarity — which is
    # what an observable read off one region needs, and which `F` reaches a few sweeps LATER than
    # its own value settles, because `F` is a stationary functional of the messages. Same per-sweep
    # cost: `_ctm_region_terms` returns the terms in the pass that computes `F` anyway.
    cyc = opts.projector === :cycle
    wr = convergence === :worst_region
    mg = convergence === :marginal
    local vprev
    mprev = mg ? _ctm_vertex_marginals(env, cache) : nothing
    F = if wr
        f, vprev = _ctm_region_terms(env, cache); f
    else
        cvm_freenergy(env, cache)
    end
    converged, Δ, crit = false, Inf, Inf
    sd = nothing                       # `:cut` state distance — reported in the warning
    wrd = nothing                      # worst region's |Δ lnZ| — reported in the warning
    mgd = nothing                      # worst vertex-marginal change — reported in the warning
    for it in 1:maxiter
        prev = env
        env = sweep_vertex_environments(cache, env, tbl)
        if wr
            Fnew, vnow = _ctm_region_terms(env, cache)
            wrd = maximum(abs.(vnow .- vprev))
            vprev = vnow
        else
            Fnew = cvm_freenergy(env, cache)
        end
        Δ = abs(Fnew - F); F = Fnew
        !cyc && opts.gauge && (sd = _ctm_statedist(env, prev))
        if mg
            mnow = _ctm_vertex_marginals(env, cache)
            mgd = _ctm_marginal_distance(mnow, mprev)
            mprev = mnow
        end
        crit = Δ
        isnothing(sd) || (crit = max(crit, sd^2))    # sd² ~ |ΔF|; `max(1,|F|)` below loosens by √|F|
        isnothing(wrd) || (crit = max(crit, wrd))
        isnothing(mgd) || (crit = max(crit, mgd))
        verbose && @info "CVM sweep $it: F = $F, |ΔF| = $Δ, state = $(something(sd, NaN)), " *
                         "worst region = $(something(wrd, NaN)), marginal = $(something(mgd, NaN))"
        # Positive evidence of convergence: ≥2 sweeps (guards the sweep-1 `|ΔF|` cancellation
        # coincidence), plus a real state signal where one is expected — `:cut` under gauge needs a
        # full-coverage `_ctm_statedist`, unless worst-region (full-coverage by construction, every
        # region is always computable) stands in.
        certified = it >= 2 && (mg ? !isnothing(mgd) : (wr || cyc || !opts.gauge || !isnothing(sd)))
        if certified && crit ≤ tolerance * max(one(crit), abs(F))
            converged = true
            verbose && @info "CVM sweep converged after $it sweeps."
            break
        end
    end
    if !converged
        extra = wr ? ", worst region |Δ lnZ| = $(something(wrd, NaN))" : ""
        mg && (extra *= ", worst marginal change = $(something(mgd, NaN))")
        cyc || (extra *= ", state distance = $(something(sd, NaN))")
        msg = "CVM sweep did not converge to tolerance $tolerance after $maxiter sweeps " *
              "(final |ΔF| = $Δ$extra; binding criterion = $crit)."
        verbose ? println(msg) : @warn(msg)
    end
    return _ctm_setenv(cache, env)
end
