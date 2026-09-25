# Finite, position-resolved CTMRG on a 3D CUBIC grid: the 2D engine's region-graph (CVM)
# construction one dimension up. Every vertex carries a 26-block environment shell, grown and
# projected by local moves, and `F` is the Möbius sum over the half-integer region grid.
#
# BLOCKS. Per axis a block is `coord < p` (sign −1), `coord == p` (0) or `coord ≥ p` (+1); a block
# is a sign triple (not all 0) with a position triple, keyed `(s1, s2, s3, p1, p2, p3)`. By its
# number of nonzero signs it is a half-line T (1; six around a vertex), a quarter-plane E (2;
# twelve) or an octant C (3; eight). Vertex (x, y, z)'s shell takes, per axis, sign −1 at x, 0 at
# x and +1 at x + 1, and together with the vertex tiles the box.
#
# INTERFACES. A block's legs are its FACES — the bonds along axis `a` between `k` and `k + 1` over
# the block's transverse extent — keyed `(a, k, sb, pb, sc, pc)` with `(b, c)` the other two axes
# in increasing order. Both transverse signs nonzero: a PLANE interface (a quarter-plane of bonds,
# between an octant and a quarter-plane block); one zero: a LINE interface (a half-line, between a
# quarter-plane and a half-line block); both zero: the single raw bond of a T into its vertex,
# never truncated. Every interface is shared by exactly four blocks, two on each side, so its
# projector pair is ONE object — derived once, consumed four times, `P_A` on the low side and
# `P_B` on the high side as in 2D — and its kept index `W[F]` is FIXED: created once with
# dimension min(χ, full bond dimension of the face) and reused by every sweep. A previous sweep's
# pair therefore always lives on the current blocks' legs, which the compressed derivation below
# relies on; the gauge within `W[F]` is aligned to the previous pair by orthogonal Procrustes
# (`_ctm_align`) as in 2D.
#
# THE MOVE. A new block is the union of the previous state's blocks one slab further out and the
# slab next to its new boundary: for every subset of its nonzero axes, the sub-block with those
# axes collapsed onto the slab (all collapsed = the vertex). An octant grows from eight old
# pieces (C, 3E, 3T, vertex), a quarter-plane from four (E, 2T, vertex), a half-line from two
# (T, vertex). A new PLANE face is then χ·χ·χ·D wide (the old plane leg, the two line legs of the
# new strips, the new corner bond), a new LINE face χ·D, and both are projected to `W`.
#
# PROJECTORS. Each pair is derived from the two enlarged blocks on either side of the interface
# with the same transverse extent — the two octants of one cube that meet across a plane
# interface, the two quarter-planes that meet across a line interface — as the 2D engine derives
# a column interface from its NW and NE enlarged corners. The REST is what differs: an octant's two
# other faces are each χ³D wide, so the 2D-style block `A` (rest × interface) would be
# (χ³D)² × χ³D. Plane rest faces are therefore COMPRESSED by the previous sweep's pair for the same
# face, leaving `A` χ² × χ³D; line rest faces stay open, as in 2D. (`plane_rest = :exact` keeps
# plane rest faces open too — small χ only, a reference for what the compression costs.)
#
#   :cut    the 2D biorthogonal pair from triangular factors of the two blocks
#           (`_ctm_twosided_projector_qr`), for every interface.
#   :cycle  for PLANE interfaces, the dominant invariant subspace of the interface's cube
#           environment: the other six octants closed around it, every other face of the cube
#           truncated by THIS sweep's `:cut` pair. With `a` (χ² × n) and `b` (χ² × n) the two
#           open octants as matrices and `e` (χ² × χ²) the other six, the interface operator
#           ρ = aᵀ e b has rank ≤ χ², and its nonzero spectrum is that of the χ²×χ² matrix
#           M = e b aᵀ; one Schur form of `M` gives both invariant bases (`aᵀY` right, `Uᵀ e b`
#           left, biorthonormal through a Sylvester solve) without an n-dimensional eigensolve.
#           `P_A` (low side) spans the LEFT basis, `P_B` the right, so that Z_cube(Π) = Tr(ρ Πᵀ)
#           keeps the dominant eigenvalues. This is the 3D reading of the 2D cycle's consistency
#           around the plaquette; the 2D ring makes that condition closed-form for all four
#           interfaces at once, the cube graph is not a ring, and closing the cube with the
#           previous sweep's `:cycle` pairs instead made the pairs feed back into each other and
#           the sweep never settled (see `_c3_cycle_cube`). Line interfaces keep their `:cut` pair.
#
# SEEDING. From an empty state, `seed_sweeps` sweeps at `seed_maxdim` with nothing compressed
# fill the blocks one slab per sweep (every face is at most seed_maxdim³·D wide there). The state
# and its pairs are then zero-padded onto the target χ's kept indices, and the compressed sweeps
# take over: a compressed rest face of rank r limits the next plane pair to rank r², so the
# retained rank grows ~quadratically per sweep until it reaches χ.
#
# COSTS (dense, single layer, bond dimension D). An enlarged octant with two faces compressed has
# χ⁵D entries and costs ~O(χ⁷D) to form; each plane pair costs O(χ⁷D) more (the QR, or `b aᵀ`).
# There are ~26·L³ blocks, the E and T of χ⁴ and χ⁴D entries. Region contractions for `F`: a
# vertex region is a closed shell of 26 blocks (a sphere) whose balanced separators cut ~8 legs,
# so its exact contraction needs ~χ⁸ memory — the practical ceiling on χ for the CVM free energy.

using LinearAlgebra: LinearAlgebra, sylvester

"""
    CTM3DEnvironmentCache

Finite CTMRG environments of a tensor network on a 3D cubic grid (vertices `(x, y, z)`), the
three-dimensional counterpart of [`CTMEnvironmentCache`](@ref), which constructs it when the
network's vertices are 3-tuples. [`update`](@ref) runs the sweep to stationarity and
[`cvm_freenergy`](@ref) returns the Möbius sum over the region grid — `ln Z` at lossless `maxdim`.

Keyword arguments, beyond the [`CTMOptions`](@ref) ones that apply (`projector`, `gauge`,
`degtol`, `qr_cutoff`, `optimal_max`):

| keyword | default | meaning |
|---|---|---|
| `seed_maxdim` | `2` | χ of the uncompressed seed sweeps that fill the blocks from an empty state |
| `plane_rest` | `:compress` | `:compress`: an octant's two other faces are compressed by the previous sweep's pairs when a plane pair is derived (χ² × χ³D blocks); `:exact`: kept open ((χ³D)² × χ³D — small χ only, and `:cut` only) |
| `pair` | `:biorth` | `:biorth`: the 2D engine's biorthogonal `:cut` pair; `:isometric`: an orthogonal projector from both sides' interface Gram matrices, the same pair wherever the two sides are mirror images and immune to the side-asymmetric instability of the infinite iteration (`_c3_isometric`) |

`projector = :cycle` derives every PLANE interface from its cube environment (see the source
notes); LINE interfaces use `:cut` under either option. Dense networks only.
"""
struct CTM3DEnvironmentCache{V, N, E}
    network::N
    grid::Dict{NTuple{3, Int}, V}      # OCCUPIED positions only
    coords::Dict{V, NTuple{3, Int}}
    dims::NTuple{3, Int}
    maxdim::Int
    environments::E                    # `nothing`, or a `CTM3DEnvironments` from `update`
    options::CTMOptions
    seed_maxdim::Int
    plane_rest::Symbol
    pair::Symbol                       # :biorth or :isometric, see `_c3_isometric`
    freenergy::Base.RefValue{Any}
end

# Blocks, the projector pair of every interface (the ones the blocks were built with), the fixed
# kept index of every interface, and — under `:cycle` — the same sweep's `:cut` pairs, which the
# next sweep's `:cut` stage compresses with (under `:cut`, `Q === P`).
struct CTM3DEnvironments
    B::AbstractDict{NTuple{6, Int}, Any}
    P::AbstractDict{NTuple{6, Int}, Any}
    W::AbstractDict{NTuple{6, Int}, Any}
    chi::Int
    Q::AbstractDict{NTuple{6, Int}, Any}
end
CTM3DEnvironments(B, P, W, chi) = CTM3DEnvironments(B, P, W, chi, P)

network(cache::CTM3DEnvironmentCache) = cache.network
graph(cache::CTM3DEnvironmentCache) = graph(network(cache))
environments(cache::CTM3DEnvironmentCache) = cache.environments
options(cache::CTM3DEnvironmentCache) = cache.options

function CTM3DEnvironmentCache(net, maxdim::Integer; seed_maxdim::Integer = 2,
                               plane_rest::Symbol = :compress, pair::Symbol = :biorth, kwargs...)
    opts = CTMOptions(; kwargs...)
    maxdim >= 1 || throw(ArgumentError("maxdim must be ≥ 1, got $maxdim"))
    seed_maxdim >= 1 || throw(ArgumentError("seed_maxdim must be ≥ 1, got $seed_maxdim"))
    plane_rest in (:compress, :exact) || throw(ArgumentError(
        "plane_rest must be :compress or :exact, got $(repr(plane_rest))"))
    (plane_rest === :exact && opts.projector === :cycle) && throw(ArgumentError(
        "projector = :cycle closes each cube with the previous sweep's pairs, so it needs " *
        "plane_rest = :compress"))
    pair in (:biorth, :isometric) || throw(ArgumentError("pair must be :biorth or :isometric, got $(repr(pair))"))
    vs = collect(vertices(graph(net)))
    all(v -> (v isa Tuple || v isa CartesianIndex) && length(v) == 3, vs) ||
        error("CTM3DEnvironmentCache requires a 3D grid network (vertices as (x, y, z)).")
    grid = Dict{NTuple{3, Int}, eltype(vs)}((Int(v[1]), Int(v[2]), Int(v[3])) => v for v in vs)
    length(grid) == length(vs) || error("CTM3DEnvironmentCache: two vertices share a grid position.")
    coords = Dict{eltype(vs), NTuple{3, Int}}(v => pos for (pos, v) in grid)
    all(p -> all(>=(1), p), keys(grid)) || error("CTM3DEnvironmentCache: grid positions must be ≥ 1.")
    for e in edges(graph(net))
        u = coords[src(e)]; w = coords[dst(e)]
        sum(abs.(u .- w)) == 1 || error(
            "CTM3DEnvironmentCache: bond $(src(e)) – $(dst(e)) joins non-adjacent grid positions " *
            "$u and $w. CTMRG needs an OPEN lattice whose bonds connect grid neighbours.")
    end
    ps = collect(keys(grid))
    dims = (maximum(p -> p[1], ps), maximum(p -> p[2], ps), maximum(p -> p[3], ps))
    all(>=(2), dims) || error("CTM3DEnvironmentCache: the grid must span at least 2 sites along " *
        "every axis, got $dims; a single layer is a 2D grid (use its (x, y) vertices).")
    return CTM3DEnvironmentCache(net, grid, coords, dims, Int(maxdim), nothing, opts,
                                 Int(seed_maxdim), plane_rest, pair, Ref{Any}(nothing))
end

_c3_setenv(cache::CTM3DEnvironmentCache, env) =
    CTM3DEnvironmentCache(cache.network, cache.grid, cache.coords, cache.dims, cache.maxdim, env,
                          cache.options, cache.seed_maxdim, cache.plane_rest, cache.pair, Ref{Any}(nothing))

# --- geometry --------------------------------------------------------------------------

const _C3_TR = ((2, 3), (1, 3), (1, 2))         # the transverse axes of each axis, increasing

# Is the axis spec (s, p) nonempty on 1..L, and its coordinates.
_c3_nonempty(s::Int, p::Int, L::Int) = s == -1 ? (2 <= p <= L + 1) : (s == 0 ? (1 <= p <= L) : (1 <= p <= L))
_c3_range(s::Int, p::Int, L::Int) = s == -1 ? (1:(p - 1)) : (s == 0 ? (p:p) : (p:L))

function _c3_pos(a::Int, xa::Int, b::Int, xb::Int, c::Int, xc::Int)
    v = [0, 0, 0]; v[a] = xa; v[b] = xb; v[c] = xc
    return (v[1], v[2], v[3])
end

_c3_israw(F) = F[3] == 0 && F[5] == 0
_c3_isplane(F) = F[3] != 0 && F[5] != 0

# Every block the sweep maintains: sign −1 at p ∈ 2:L, 0 at p ∈ 1:L, +1 at p ∈ 2:L per axis.
function _c3_all_blocks(L::NTuple{3, Int})
    ks = NTuple{6, Int}[]
    rng(s, La) = s == 0 ? (1:La) : (2:La)
    for s1 in -1:1, s2 in -1:1, s3 in -1:1
        (s1, s2, s3) == (0, 0, 0) && continue
        for p1 in rng(s1, L[1]), p2 in rng(s2, L[2]), p3 in rng(s3, L[3])
            push!(ks, (s1, s2, s3, p1, p2, p3))
        end
    end
    return ks
end

# The faces of block `bk`: `(F, side)`, `side` `:low` when the block lies at coordinates ≤ k along
# the face's axis and `:high` otherwise. Raw faces are included; callers skip them.
function _c3_faces(bk::NTuple{6, Int}, L::NTuple{3, Int})
    out = Tuple{NTuple{6, Int}, Symbol}[]
    for a in 1:3
        b, c = _C3_TR[a]
        (_c3_nonempty(bk[b], bk[3 + b], L[b]) && _c3_nonempty(bk[c], bk[3 + c], L[c])) || continue
        s, p = bk[a], bk[3 + a]
        tr = (bk[b], bk[3 + b], bk[c], bk[3 + c])
        if s == -1
            2 <= p <= L[a] && push!(out, ((a, p - 1, tr...), :low))
        elseif s == 1
            2 <= p <= L[a] && push!(out, ((a, p - 1, tr...), :high))
        else
            p >= 2 && push!(out, ((a, p - 1, tr...), :high))
            p <= L[a] - 1 && push!(out, ((a, p, tr...), :low))
        end
    end
    return out
end

# The block on side `sa` (−1 low, +1 high) of interface `F` with the interface's transverse extent:
# two octants for a plane interface, two quarter-planes for a line interface.
function _c3_side_block(F::NTuple{6, Int}, sa::Int)
    a, k, sb, pb, sc, pc = F
    b, c = _C3_TR[a]
    s = [0, 0, 0]; p = [0, 0, 0]
    s[a] = sa; p[a] = k + 1; s[b] = sb; p[b] = pb; s[c] = sc; p[c] = pc
    return (s[1], s[2], s[3], p[1], p[2], p[3])
end

# The PARTS of face `F` — the sub-faces (and the raw bond) its legs are made of one slab in: for
# every subset of its nonzero transverse axes collapsed onto the slab next to the face's corner,
# `((collapse b?, collapse c?), part key)`. A plane face has four parts (the old plane face, two
# line faces of the new strips, the new corner's raw bond), a line face two.
function _c3_face_parts(F::NTuple{6, Int})
    a, k, sb, pb, sc, pc = F
    step(s, p, g) = g ? (0, s == -1 ? p - 1 : p) : (s, s == 0 ? p : (s == -1 ? p - 1 : p + 1))
    out = Tuple{Tuple{Bool, Bool}, NTuple{6, Int}}[]
    for gb in (sb == 0 ? (false,) : (false, true)), gc in (sc == 0 ? (false,) : (false, true))
        sb2, pb2 = step(sb, pb, gb); sc2, pc2 = step(sc, pc, gc)
        push!(out, ((gb, gc), (a, k, sb2, pb2, sc2, pc2)))
    end
    return out
end

# The cube (upper corner position) whose two octants meet across plane interface `F`.
_c3_cube(F::NTuple{6, Int}) = (a = F[1]; (b, c) = _C3_TR[a]; _c3_pos(a, F[2] + 1, b, F[4], c, F[6]))

# Every interface key (the non-raw faces of every block), sorted for determinism.
function _c3_interfaces(L::NTuple{3, Int})
    Fs = Set{NTuple{6, Int}}()
    for bk in _c3_all_blocks(L), (F, _) in _c3_faces(bk, L)
        _c3_israw(F) || push!(Fs, F)
    end
    return sort!(collect(Fs))
end

function _c3_factor_table(cache::CTM3DEnvironmentCache)
    tbl = Dict{NTuple{3, Int}, Vector{AbstractTensor}}()
    for (pos, v) in cache.grid
        tbl[pos] = Vector{AbstractTensor}(bp_factors(network(cache), v))
    end
    return tbl
end

function _c3_links(tbl, u::NTuple{3, Int}, v::NTuple{3, Int})
    (haskey(tbl, u) && haskey(tbl, v)) || return Index[]
    is = Index[]
    for t1 in tbl[u], t2 in tbl[v]
        append!(is, commoninds(t1, t2))
    end
    return unique(is)
end

# Total bond dimension of face `F`, capped at `cap + 1` (only `min(χ, ·)` is ever needed).
function _c3_fulldim(F::NTuple{6, Int}, tbl, L::NTuple{3, Int}, cap::Int)
    a, k, sb, pb, sc, pc = F
    b, c = _C3_TR[a]
    d = 1
    for xb in _c3_range(sb, pb, L[b]), xc in _c3_range(sc, pc, L[c])
        for i in _c3_links(tbl, _c3_pos(a, k, b, xb, c, xc), _c3_pos(a, k + 1, b, xb, c, xc))
            d *= dim(i)
            d > cap && return cap + 1
        end
    end
    return d
end

# The enlarged block as a LIST: the previous state's sub-blocks plus the slab vertex's factors.
function _c3_enlarged(B, tbl, bk::NTuple{6, Int})
    s = (bk[1], bk[2], bk[3]); p = (bk[4], bk[5], bk[6])
    N = [a for a in 1:3 if s[a] != 0]
    ts = AbstractTensor[]
    for mask in 0:(2^length(N) - 1)
        s2 = [s...]; p2 = [p...]
        for (j, a) in enumerate(N)
            if (mask >> (j - 1)) & 1 == 1          # collapsed onto the slab next to the boundary
                s2[a] = 0; p2[a] = s[a] == -1 ? p[a] - 1 : p[a]
            else                                    # the old block, one slab further out
                p2[a] = s[a] == -1 ? p[a] - 1 : p[a] + 1
            end
        end
        if s2 == [0, 0, 0]
            append!(ts, get(tbl, (p2[1], p2[2], p2[3]), AbstractTensor[]))
        else
            t = get(B, (s2[1], s2[2], s2[3], p2[1], p2[2], p2[3]), nothing)
            isnothing(t) || push!(ts, t)
        end
    end
    return ts
end

# Open (uncontracted) indices of a tensor list, in order of first appearance.
function _c3_open(ts)
    seen = Index[]; cnt = Dict{Index, Int}()
    for t in ts, i in inds(t)
        n = get(cnt, i, 0)
        n == 0 && push!(seen, i)
        cnt[i] = n + 1
    end
    return Index[i for i in seen if cnt[i] == 1]
end

# The projector a block on `side` of interface `pr` consumes.
_c3_side_proj(pr, side::Symbol) = side === :low ? pr[1] : pr[2]
_c3_inputs(P, w) = [i for i in inds(P) if !(i == w || i == dag(w))]

# The pairs (from `pairs`) compressing block `bk`'s plane faces other than `skip`, where each lives
# on exactly the block's current open legs; the rest stay open.
function _c3_compressors(bk::NTuple{6, Int}, skip, pairs, openidx, L)
    extras = Any[]
    for (R, side) in _c3_faces(bk, L)
        (R == skip || !_c3_isplane(R)) && continue
        pr = get(pairs, R, nothing)
        if isnothing(pr)
            _ctm_stat!(:c3_rest_noprev); continue
        end
        P = _c3_side_proj(pr, side)
        legs = _c3_inputs(P, pr[3])
        if !isempty(legs) && all(i -> i in openidx, legs)
            push!(extras, P)
        else
            _ctm_stat!(:c3_rest_mismatch)
        end
    end
    return extras
end

# --- pairs -----------------------------------------------------------------------------

# A derived pair `(P_A, P_B, u)` onto the interface's fixed kept index `W`: zero-padded to its
# width (the padding carries no weight, `Π = P_A P_B` keeps the derived rank, as in the 2D
# cycle), aligned to the previous pair by orthogonal Procrustes, relabelled onto `W`.
function _c3_finish(pr, ins::Vector{<:Index}, W, prev, opts::CTMOptions)
    PA, PB, u = pr[1], pr[2], pr[3]
    # A non-finite pair would poison every block that consumes it: keep last sweep's pair instead
    # (it lives on the same legs) — or no pair, which leaves the face unprojected for one sweep.
    if !(isfinite(norm(PA)) && isfinite(norm(PB)))
        _ctm_stat!(:c3_pair_nonfinite)
        ok = !isnothing(prev) && issetequal(_c3_inputs(prev[1], prev[3]), ins)
        return ok ? prev : nothing
    end
    k, kt = dim(u), dim(W)
    k > kt && error("3D CTM: a pair kept $k directions on an interface of fixed width $kt")
    if k < kt
        z = pad_index(u, ins, kt)
        elt = scalartype(PA)
        rng = Xoshiro(0)
        za = adapt_like(PA, random_tensor(rng, elt, vcat(_ctm_legs_of(PA, ins), [z])) * zero(elt))
        zb = adapt_like(PB, random_tensor(rng, elt, vcat(_ctm_legs_of(PB, ins), [dag(z)])) * zero(elt))
        PA = directsum(PA => u, za => z; tags = "Link,c3")
        PB = directsum(PB => u, zb => dag(z); tags = "Link,c3")
        u = only(uniqueinds(PA, ins))
        PB = replaceind(PB, only(uniqueinds(PB, ins)), dag(u))
    end
    out = (PA, PB, u)
    if opts.gauge && !isnothing(prev)
        out = _ctm_align(out, ins, prev)
    end
    if out[3] != W
        out = (replaceind(out[1], out[3], W), replaceind(out[2], dag(out[3]), dag(W)), W)
    end
    # BALANCE the pair's scalar gauge, P_A → t·P_A, P_B → P_B/t with ‖P_A‖ = ‖P_B‖. `Π = P_A P_B`
    # and so every block and `F` are unchanged, but without it the two norms run away: the pair
    # derived from two compressed octants carries the ratio of their magnitudes, those magnitudes
    # come from the previous pairs, and the log-imbalance DOUBLED every sweep (measured, 4×4×4 at
    # χ = 4: max ‖P‖ 4 → 17 → 79 → 3e3 → 2.5e6 → … → 1e192 and overflow at sweep 13, with F
    # converged at 2e-8 throughout).
    nA, nB = norm(out[1]), norm(out[2])
    if nA > 0 && nB > 0 && isfinite(nA) && isfinite(nB)
        t = sqrt(nB / nA)
        out = (out[1] * t, out[2] / t, out[3])
    end
    return out
end

# `:cut` pair for interface `F` from the two enlarged blocks on either side, their plane rest faces
# compressed by `pairs` (the previous sweep's `:cut` pairs) and the result aligned to `pairs[F]`.
function _c3_pair_cut(F::NTuple{6, Int}, S::CTM3DEnvironments, tbl, L, opts::CTMOptions, compress::Bool,
                      pairs = S.Q; isometric::Bool = false)
    lowk = _c3_side_block(F, -1); highk = _c3_side_block(F, 1)
    Ll = _c3_enlarged(S.B, tbl, lowk); Lh = _c3_enlarged(S.B, tbl, highk)
    (isempty(Ll) || isempty(Lh)) && return nothing
    ol = _c3_open(Ll); oh = _c3_open(Lh)
    ins = Index[i for i in ol if i in oh]
    isempty(ins) && return nothing
    # normalised: the pair's subspace is scale-free, and its gauge balance then does not inherit
    # the compressors' magnitudes (see `_c3_finish`)
    Ac = _ctm_rescale(_ctm_contract(vcat(Ll, compress ? _c3_compressors(lowk, F, pairs, ol, L) : Any[]), opts))
    Bc = _ctm_rescale(_ctm_contract(vcat(Lh, compress ? _c3_compressors(highk, F, pairs, oh, L) : Any[]), opts))
    pr = isometric ? _c3_isometric(Ac, Bc, ins, dim(S.W[F]), opts) :
        _ctm_twosided_projector_qr(Ac, Bc, ins, dim(S.W[F]), opts)
    isnothing(pr) && return nothing
    return _c3_finish(pr, ins, S.W[F], get(pairs, F, nothing), opts)
end

# The ORTHOGONAL pair from both sides at once: `P` the dominant eigenvectors of `Ac†Ac + Bc†Bc` on
# the interface legs (the right singular vectors of the two triangular factors stacked), `P_A = P`,
# `P_B = P†`, so `Π = P P†`. Where the two sides are mirror images — every interface of a
# reflection-symmetric lattice at its fixed point — this IS the biorthogonal pair (`R_A = R_B` makes
# `P_A` an isometry and `P_B = P_Aᵀ`), so the fixed point is the same one; what differs is the
# iteration. A perturbation that makes the two sides differ changes `Ac†Ac + Bc†Bc` only at second
# order, while the biorthogonal pair turns it into `P_A ≠ P_Bᵀ` at first order and feeds it back.
function _c3_isometric(Ac, Bc, ins::Vector{<:Index}, maxdim::Integer, opts::CTMOptions)
    RA = _ctm_tri_factor(Ac, ins); RB = _ctm_tri_factor(Bc, ins)
    bA = only(uniqueinds(RA, ins)); bB = only(uniqueinds(RB, ins))
    Rs = directsum(RA => bA, RB => bB; tags = "Link,c3s")
    b = only(uniqueinds(Rs, ins))
    _, _, V = svd(Rs, [b]; trunc = _ctm_trunc(maxdim, opts))
    v = only(uniqueinds(V, ins))                         # V = P† in the seam's bilinear convention
    PA = scalartype(V) <: Real ? V : conj(V)
    return (PA, replaceind(V, v, dag(v)), v)
end

# `:cycle` pair for a plane interface from its two open octants `Ac`, `Bc` (other faces compressed)
# and the cube's other six octants contracted into `env`. See the header for the algebra.
function _c3_cycle_pair(Ac, Bc, env, ins::Vector{<:Index}, kt::Int, opts::CTMOptions)
    Arest = uniqueinds(Ac, ins); Brest = uniqueinds(Bc, ins)
    (isempty(Arest) || isempty(Brest)) && return nothing
    # The invariant subspaces are scale-free: normalise the three pieces so `M` sits near 1
    # whatever the blocks' and pairs' magnitudes (the pairs carry S^{-1/2} factors).
    Ac = _ctm_rescale(Ac); Bc = _ctm_rescale(Bc); env = _ctm_rescale(env)
    cA = combiner(collect(Arest)); rA = combinedind(cA)
    Ac1 = Ac * cA                                        # (ins…, rA)
    env1 = env * cA                                      # (rA, Brest…)
    BA = Bc * Ac1                                        # (Brest…, rA) — bilinear over the interface
    rA2 = sim(rA)
    M = env1 * replaceind(BA, rA, rA2)                   # (rA, rA2): M = e b aᵀ on the A rest
    m = array(M, rA, rA2)
    all(isfinite, m) || return nothing
    n = size(m, 1)
    Fr = schur(m)
    λ = Fr.values
    order = sortperm(abs.(λ); rev = true)
    amax = abs(λ[order[1]])
    amax > 0 || return nothing
    kk = min(kt, count(x -> abs(x) > 1.0e-14 * amax, λ))
    if eltype(m) <: Real && 1 <= kk < n        # never split a conjugate pair of a real Schur form
        λa, λb = λ[order[kk]], λ[order[kk + 1]]
        (!iszero(imag(λa)) && isapprox(λa, conj(λb); rtol = 1.0e-8)) && (kk -= 1)
    end
    kk >= 1 || return nothing
    sel = falses(n); sel[order[1:kk]] .= true
    Fr = ordschur(Fr, sel)
    # BOTH bases from the one Schur form, so they belong to the same eigenvalues: with
    # m = Z T Z⁻¹, T = [T11 T12; 0 T22], the right invariant subspace is Z[:, 1:k] and the left one
    # is the rows [I X] Z⁻¹ with T11 X − X T22 = T12 (`sylvester` solves AX + XB + C = 0). The two
    # are then biorthonormal by construction, and their overlap through ρ is T11 — two independent
    # Schur solves could order near-ties at the cut differently and hand the whitening a
    # near-singular overlap.
    Z, T = Fr.Z, Fr.T
    X = kk < n ? sylvester(T[1:kk, 1:kk], -T[(kk + 1):n, (kk + 1):n], -T[1:kk, (kk + 1):n]) :
        zeros(eltype(T), kk, 0)
    all(isfinite, X) || return nothing
    Y = Z[:, 1:kk]
    U = transpose(hcat(Matrix{eltype(T)}(LinearAlgebra.I, kk, kk), X) * Z')   # n × k: the left rows as columns
    kx = new_index(kk; tags = "Link,c3x"); kz = new_index(kk; tags = "Link,c3z")
    xb = Ac1 * adapt_like(Ac1, from_array(Matrix(Y), rA, kx))             # (ins…, kx): right basis aᵀY
    zb = (adapt_like(env1, from_array(Matrix(U), rA, kz)) * env1) * Bc     # (kz, ins…): left basis Uᵀ e b
    ab = try
        _ctm_biorth(zb, xb, ins, _ctm_trunc(kk, opts))              # P_A spans the left basis
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    (isnothing(ab) || !isfinite(norm(ab[1])) || !isfinite(norm(ab[2]))) && return nothing
    return ab
end

# All twelve plane interfaces of a cube under `:cycle`, each the dominant invariant subspace of its
# cube closed by `closure` — THIS sweep's `:cut` pairs of the cube's other interfaces, derived from
# the very enlarged octants they compress (so they always live on the current legs). Closing with
# the previous sweep's `:cycle` pairs instead made every pair feed back into its eleven neighbours
# through one Jacobi step and the iteration never settled (measured, 4×4×4 at χ = 4: F reached
# 2.4e-8 at sweep 1, then wandered between −3e-5 and +1e-5 for 15 sweeps); the 2D cycle has no such
# loop because it solves a plaquette's four interfaces jointly from the untruncated corners.
# A cube whose octants or closure pairs are missing falls back to its `:cut` pairs.
function _c3_cycle_cube(cube::NTuple{3, Int}, S::CTM3DEnvironments, tbl, L, opts::CTMOptions, closure)
    octs = [(s1, s2, s3) for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)]
    okey(σ) = (σ[1], σ[2], σ[3], cube[1], cube[2], cube[3])
    face(σ, a) = (a, cube[a] - 1, σ[_C3_TR[a][1]], cube[_C3_TR[a][1]], σ[_C3_TR[a][2]], cube[_C3_TR[a][2]])
    faces = unique([face(σ, a) for σ in octs for a in 1:3])
    cut() = Dict{NTuple{6, Int}, Any}(F => get(closure, F, nothing) for F in faces)
    lists = Dict(σ => _c3_enlarged(S.B, tbl, okey(σ)) for σ in octs)
    any(isempty, values(lists)) && (_ctm_stat!(:c3_cycle_cut_empty); return cut())
    opens = Dict(σ => _c3_open(lists[σ]) for σ in octs)
    proj = Dict{Tuple{NTuple{3, Int}, Int}, Any}()
    for σ in octs, a in 1:3
        pr = get(closure, face(σ, a), nothing)
        isnothing(pr) && (_ctm_stat!(:c3_cycle_cut_noclosure); return cut())
        P = _c3_side_proj(pr, σ[a] == -1 ? :low : :high)
        all(i -> i in opens[σ], _c3_inputs(P, pr[3])) || (_ctm_stat!(:c3_cycle_cut_mismatch); return cut())
        proj[(σ, a)] = P
    end
    Ct = Dict(σ => _ctm_rescale(_ctm_contract(vcat(lists[σ], [proj[(σ, a)] for a in 1:3]), opts)) for σ in octs)
    out = Dict{NTuple{6, Int}, Any}()
    for a in 1:3, sb in (-1, 1), sc in (-1, 1)
        b, c = _C3_TR[a]
        σA = [0, 0, 0]; σA[a] = -1; σA[b] = sb; σA[c] = sc; σA = Tuple(σA)
        σB = [σA...]; σB[a] = 1; σB = Tuple(σB)
        F = face(σA, a)
        ins = Index[i for i in opens[σA] if i in opens[σB]]
        pr = nothing
        if !isempty(ins)
            Ac = _ctm_rescale(_ctm_contract(vcat(lists[σA], [proj[(σA, d)] for d in (b, c)]), opts))
            Bc = _ctm_rescale(_ctm_contract(vcat(lists[σB], [proj[(σB, d)] for d in (b, c)]), opts))
            env = _ctm_contract([Ct[ρ] for ρ in octs if ρ != σA && ρ != σB], opts)
            pr = _c3_cycle_pair(Ac, Bc, env, ins, dim(S.W[F]), opts)
        end
        if isnothing(pr)                      # declined: this interface keeps its `:cut` pair
            _ctm_stat!(:c3_cycle_declined)
            out[F] = get(closure, F, nothing)
        else
            out[F] = _c3_finish(pr, ins, S.W[F], get(S.P, F, nothing), opts)
        end
    end
    return out
end

# --- the sweep -------------------------------------------------------------------------

# One Jacobi sweep: every pair from `S`'s enlarged blocks, then every block regrown and projected.
# `exact` keeps every rest face open (the seed); otherwise plane rest faces are compressed.
function _c3_sweep(cache::CTM3DEnvironmentCache, S::CTM3DEnvironments, tbl, χ::Int, exact::Bool)
    L = cache.dims; opts = cache.options
    Fs = _c3_interfaces(L)
    for F in Fs                                          # fixed kept indices, created once
        haskey(S.W, F) || (S.W[F] = new_index(min(χ, _c3_fulldim(F, tbl, L, χ)); tags = "Link,c3"))
    end
    Q = Dict{NTuple{6, Int}, Any}()                     # this sweep's `:cut` pairs
    lk = ReentrantLock()
    _ctm_foreach(eachindex(Fs)) do i
        pr = _c3_pair_cut(Fs[i], S, tbl, L, opts, !exact, S.Q; isometric = cache.pair === :isometric)
        isnothing(pr) || lock(() -> (Q[Fs[i]] = pr), lk)
    end
    P = Q
    if opts.projector === :cycle && !exact
        # plane interfaces refined against their cubes closed by `Q`; line interfaces keep `Q`
        P = copy(Q)
        cubes = sort!(unique([_c3_cube(F) for F in Fs if _c3_isplane(F)]))
        _ctm_foreach(eachindex(cubes)) do i
            res = _c3_cycle_cube(cubes[i], S, tbl, L, opts, Q)
            lock(lk) do
                for (F, pr) in res
                    isnothing(pr) || (P[F] = pr)
                end
            end
        end
    end
    bks = _c3_all_blocks(L)
    nb = Vector{Any}(nothing, length(bks))
    _ctm_foreach(eachindex(bks)) do i
        ts = _c3_enlarged(S.B, tbl, bks[i])
        isempty(ts) && return
        extras = Any[]
        for (F, side) in _c3_faces(bks[i], L)
            _c3_israw(F) && continue
            pr = get(P, F, nothing)
            isnothing(pr) || push!(extras, _c3_side_proj(pr, side))
        end
        nb[i] = _ctm_rescale(_ctm_absorb(opts, ts, extras...))
    end
    B = Dict{NTuple{6, Int}, Any}()
    for i in eachindex(bks)
        isnothing(nb[i]) || (B[bks[i]] = nb[i])
    end
    return CTM3DEnvironments(B, P, S.W, χ, Q)
end

# Re-express a converged seed on the target χ's kept indices: every kept index whose width grows is
# embedded isometrically (zero padding) in the blocks and in both sides of every pair, so every
# contraction is unchanged and the next compressed sweep finds pairs on its current legs.
function _c3_promote(S::CTM3DEnvironments, tbl, L, χ::Int)
    Wn = Dict{NTuple{6, Int}, Any}()
    emb = Dict{Index, Any}()
    for (F, w0) in S.W
        d = min(χ, _c3_fulldim(F, tbl, L, χ))
        if d <= dim(w0)
            Wn[F] = w0
        else
            w1 = new_index(d; tags = "Link,c3")
            Wn[F] = w1
            M = zeros(dim(w0), d)
            for j in 1:dim(w0)
                M[j, j] = 1
            end
            emb[w0] = from_array(M, w0, w1)
        end
    end
    lift(t) = (for i in inds(t); e = get(emb, i, nothing); isnothing(e) || (t = t * adapt_like(t, e)); end; t)
    B = Dict{NTuple{6, Int}, Any}(k => lift(t) for (k, t) in S.B)
    P = Dict{NTuple{6, Int}, Any}(F => (lift(pr[1]), lift(pr[2]), Wn[F]) for (F, pr) in S.P)
    return CTM3DEnvironments(B, P, Wn, χ)
end

# --- regions and F ---------------------------------------------------------------------

# The blocks tiling the box around region centre `c` (half-integers allowed), `nothing`s dropped.
function _c3_region_blocks(env::CTM3DEnvironments, c)
    spec(x) = isinteger(x) ? ((-1, Int(x)), (0, Int(x)), (1, Int(x) + 1)) :
                             ((-1, ceil(Int, x)), (1, ceil(Int, x)))
    ts = AbstractTensor[]
    for (s1, p1) in spec(c[1]), (s2, p2) in spec(c[2]), (s3, p3) in spec(c[3])
        (s1, s2, s3) == (0, 0, 0) && continue
        t = get(env.B, (s1, s2, s3, p1, p2, p3), nothing)
        isnothing(t) || push!(ts, t)
    end
    return ts
end

function _c3_region_lnZ(env::CTM3DEnvironments, cache::CTM3DEnvironmentCache, tbl, c)
    ts = _c3_region_blocks(env, c)
    all(isinteger, c) && append!(ts, get(tbl, (Int(c[1]), Int(c[2]), Int(c[3])), AbstractTensor[]))
    isempty(ts) && return 0.0
    return log(abs(scalar(_c3_contract_region(ts, cache.options))))
end

# Region contraction. A VERTEX region (the vertex and its 26-block shell) is the expensive one: the
# greedy order `_ctm_contract` uses above `optimal_max` tensors peaks at ~χ¹³ entries on a bulk
# shell, TreeSA at ~χ¹² (measured, see docs/ctmrg3d.md) — 8× less memory at χ = 8. So lists that
# long get a TreeSA order, found once per shape and memoised like the netcon sequences.
const _C3_REGION_SEQ = Dict{Any, Any}()
function _c3_contract_region(ts::Vector, opts::CTMOptions)
    length(ts) < 20 && return _ctm_contract(ts, opts)
    key = _ctm_seq_key(ts)
    seq = lock(CTM_GLOBAL_LOCK) do
        get(_C3_REGION_SEQ, key, nothing)
    end
    if seq === nothing
        seq = contraction_sequence(ts; alg = "omeinsum", optimizer = TreeSA(; ntrials = 4, niters = 50))
        lock(CTM_GLOBAL_LOCK) do
            _C3_REGION_SEQ[key] = seq
        end
    end
    return contract(ts; sequence = seq)
end

# Möbius sum over the half-integer region grid: (−1)^(number of half-integer coordinates).
function _c3_freenergy(env::CTM3DEnvironments, cache::CTM3DEnvironmentCache, tbl)
    L = cache.dims
    pts = [(cx, cy, cz) for cx in 1.0:0.5:L[1] for cy in 1.0:0.5:L[2] for cz in 1.0:0.5:L[3]]
    vals = Vector{Float64}(undef, length(pts))
    _ctm_foreach(eachindex(pts)) do i
        vals[i] = _c3_region_lnZ(env, cache, tbl, pts[i])
    end
    F = 0.0
    for i in eachindex(pts)
        F += (iseven(count(x -> !isinteger(x), pts[i])) ? 1 : -1) * vals[i]
    end
    return F
end

"""
    cvm_freenergy(cache::CTM3DEnvironmentCache)

The Möbius sum of region free energies over the 3D region grid (vertex +1, edge −1, face +1,
cube −1): `ln Z` at lossless `maxdim`. [`update`](@ref) the cache first.
"""
function cvm_freenergy(cache::CTM3DEnvironmentCache)
    env = environments(cache)
    isnothing(env) && error("CTM3DEnvironmentCache has not been `update`d.")
    r = cache.freenergy
    r[] === nothing && (r[] = _c3_freenergy(env, cache, _c3_factor_table(cache)))
    return r[]
end

"""
    update(cache::CTM3DEnvironmentCache; maxiter = 30, tolerance = 1e-10, verbose = false,
           seed_sweeps = maximum(dims) + 2)

Run the 3D sweep to stationarity of `F`: on a fresh cache, `seed_sweeps` uncompressed sweeps at
`seed_maxdim` first (they fill the blocks one slab per sweep), then compressed sweeps at `maxdim`
until `|ΔF| ≤ tolerance · max(1, |F|)` (after at least two). Warns if that is not reached.
"""
function update(cache::CTM3DEnvironmentCache; maxiter::Integer = 30, tolerance::Real = 1.0e-10,
                verbose::Bool = false, seed_sweeps::Integer = maximum(cache.dims) + 2)
    tbl = _c3_factor_table(cache)
    L = cache.dims
    χ = cache.maxdim
    χ0 = min(cache.seed_maxdim, χ)
    S = environments(cache)
    if isnothing(S)
        S = CTM3DEnvironments(Dict{NTuple{6, Int}, Any}(), Dict{NTuple{6, Int}, Any}(),
                              Dict{NTuple{6, Int}, Any}(), χ0)
        for it in 1:seed_sweeps
            S = _c3_sweep(cache, S, tbl, χ0, true)
            verbose && @info "3D CVM seed sweep $it (χ = $χ0)"
        end
        χ0 < χ && (S = _c3_promote(S, tbl, L, χ))
    end
    exact = cache.plane_rest === :exact
    F = _c3_freenergy(S, cache, tbl)
    converged, Δ = false, Inf
    for it in 1:maxiter
        S = _c3_sweep(cache, S, tbl, χ, exact)
        Fn = _c3_freenergy(S, cache, tbl)
        Δ = abs(Fn - F); F = Fn
        verbose && @info "3D CVM sweep $it: F = $F, |ΔF| = $Δ"
        if it >= 2 && Δ <= tolerance * max(1.0, abs(F))
            converged = true
            break
        end
    end
    converged || @warn "3D CVM sweep did not converge to tolerance $tolerance after $maxiter " *
        "sweeps (final |ΔF| = $Δ)."
    out = _c3_setenv(cache, S)
    out.freenergy[] = F
    return out
end
