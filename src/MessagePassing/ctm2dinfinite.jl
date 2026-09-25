# Infinite (translation-invariant) 2D CTMRG on a 1×1 unit cell: the 2D restriction of
# ctm3dinfinite.jl, and the contraction engine for the boundary-PEPS treatment of 3D models
# (docs/ctmrg3d.md, "Open problems"). There every quantity is a 2D network whose site is a STACK of
# layers (ket, operator, bra), so a site here is a LIST of layer tensors and a raw bond is the list
# of their legs, never fused: contractions stay lazy and a single layer's environment — the
# gradient of ln κ with respect to that layer — is one contraction away.
#
# GEOMETRY, as in 3D one dimension down. Per axis a block is `coord < p` (sign −1), `coord == p`
# (0) or `coord ≥ p` (+1), keyed `(s1, s2, p1, p2)`: a vertex's shell is 4 half-lines T (one
# nonzero sign) and 4 quadrants C. A block's legs are its FACES `(a, k, sb, pb)`, the bonds along
# axis `a` between `k` and `k + 1` over the transverse extent `(sb, pb)`: `sb ≠ 0` a LINE face (a
# half-line of bonds), `sb = 0` the raw bond(s) of a T into its vertex, never truncated. Each line
# interface's pair comes from the two enlarged quadrants on either side, rest faces open (the 2D
# engine's `:cut`, Fishman et al.'s two-sided construction); there are no plane faces in 2D.
#
# STATE. One tensor per block type (8) and one pair per line-interface type `(a, sb)` (4), on fixed
# canonical legs. An iteration places them on a virtual 5×5 box, derives the 4 pairs at
# representative faces, regrows the centre vertex's 8 blocks and relabels them back — the
# simultaneous-move iteration with every block of a type the same tensor.
#
# ln κ PER SITE, the Kikuchi (Möbius) form: ln κ = ln Z_v − ln Z_ex − ln Z_ey + ln Z_f over one
# vertex, two edge and one face region — Baxter's κ = Z(C⁴T⁴a) Z(C⁴) / (Z(C⁴T_x²) Z(C⁴T_y²)).
#
# SEED. The open-boundary vacuum at the target χ: every line leg carries e₁, every raw leg the
# `boundary` vector; the pairs' rank grows by the raw bond dimension per iteration, zero-padded
# onto the fixed kept width meanwhile.

const _I2_L = (5, 5)                                     # the virtual box …
const _I2_V = (3, 3)                                     # … and its centre vertex

_c2_israw(F) = F[3] == 0

# The faces of block `bk` on a box `L`: `(F, side)` with `side` `:low` when the block lies at
# coordinates ≤ k along the face's axis.
function _c2_faces(bk::NTuple{4, Int}, L::NTuple{2, Int})
    out = Tuple{NTuple{4, Int}, Symbol}[]
    for a in 1:2
        b = 3 - a
        _c3_nonempty(bk[b], bk[2 + b], L[b]) || continue
        s, p = bk[a], bk[2 + a]
        tr = (bk[b], bk[2 + b])
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

# The quadrant on side `sa` (−1 low, +1 high) of line interface `F`, with its transverse extent.
function _c2_side_block(F::NTuple{4, Int}, sa::Int)
    a, k, sb, pb = F
    s = [0, 0]; p = [0, 0]
    s[a] = sa; p[a] = k + 1; s[3 - a] = sb; p[3 - a] = pb
    return (s[1], s[2], p[1], p[2])
end

# The PARTS of face `F` one slab in: `(collapsed?, part)` — for a line face the old line face one
# slab further out and the raw bond next to the corner; a raw face is its own part.
function _c2_face_parts(F::NTuple{4, Int})
    a, k, sb, pb = F
    step(s, p, g) = g ? (0, s == -1 ? p - 1 : p) : (s, s == 0 ? p : (s == -1 ? p - 1 : p + 1))
    out = Tuple{Bool, NTuple{4, Int}}[]
    for g in (sb == 0 ? (false,) : (false, true))
        sb2, pb2 = step(sb, pb, g)
        push!(out, (g, (a, k, sb2, pb2)))
    end
    return out
end

# The enlarged block as a LIST: the old block one slab further out, the old half-lines of the new
# strips, and the slab's vertex (its layers).
function _c2_enlarged(B, tbl, bk::NTuple{4, Int})
    s = (bk[1], bk[2]); p = (bk[3], bk[4])
    N = [a for a in 1:2 if s[a] != 0]
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
        if s2 == [0, 0]
            append!(ts, get(tbl, (p2[1], p2[2]), AbstractTensor[]))
        else
            t = get(B, (s2[1], s2[2], p2[1], p2[2]), nothing)
            isnothing(t) || push!(ts, t)
        end
    end
    return ts
end

# The blocks of the region centred at `c` (integer or half-integer coordinates).
function _c2_region_blocks(B, c)
    spec(x) = isinteger(x) ? ((-1, Int(x)), (0, Int(x)), (1, Int(x) + 1)) :
                             ((-1, ceil(Int, x)), (1, ceil(Int, x)))
    ts = AbstractTensor[]
    for (s1, p1) in spec(c[1]), (s2, p2) in spec(c[2])
        (s1, s2) == (0, 0) && continue
        t = get(B, (s1, s2, p1, p2), nothing)
        isnothing(t) || push!(ts, t)
    end
    return ts
end

_i2_types() = [(s1, s2) for s1 in -1:1 for s2 in -1:1 if (s1, s2) != (0, 0)]
# The centre vertex's shell key of block type `s` (sign −1 and 0 at 3, +1 at 4).
_i2_centre_key(s) = (s[1], s[2], (s[a] == 1 ? 4 : 3 for a in 1:2)...)
_i2_itype(F::NTuple{4, Int}) = (F[1], F[3])
# One representative face per line-interface type, between the quadrants at column/row 4.
_i2_representatives() = [((a, sb), (a, 3, sb, 4)) for a in 1:2 for sb in (-1, 1)]

"""
    InfiniteCTM2D(site, legs, maxdim; pair = :biorth, boundary = nothing, init = nothing, kwargs...)

Infinite 2D CTMRG environment of the translation-invariant square-lattice network built from one
site, for the thermodynamic limit directly. `site` is a tensor or a LIST of layer tensors (e.g.
`[A, W, conj(A)]` for a PEPS sandwich), contracted among themselves over their internal indices;
`legs = (x⁻, x⁺, y⁻, y⁺)` gives, per direction, the site's open index or the list of its layers'
indices, in matching order between `x⁻` and `x⁺` (resp. `y⁻`, `y⁺`): every site's `x⁺` legs are
contracted with its +x neighbour's `x⁻` legs, likewise along y. Layers are never fused.

[`update`](@ref) iterates to the fixed point; [`cvm_freenergy`](@ref) returns ln κ, the log
partition function per site; [`site_ratio`](@ref) a single-site impurity ratio;
[`site_environment`](@ref) one layer's environment (the gradient of ln κ with respect to it).

`pair` is `:biorth` (default: the 2D engine's two-sided pair) or `:isometric` (an orthogonal
projector from both sides at once, see `_c3_isometric`). Both reach the same fixed point on
mirror-symmetric networks (2D Ising: Onsager to 2e-14, Yang to 3e-16 at χ = 16), but on a network
whose two sides of an interface differ, `:isometric` is UNSTABLE here: the norm of a random D = 2
PEPS read the right ln κ at iteration 9 and ln κ = −5.0 instead of 2.03 by iteration 600, while
`:biorth` converged in 18 — the reverse of the 3D engine, whose mirror-symmetric plane pairs need
`:isometric`. `boundary` is the seed's vector on every raw leg
of a half-line (default all ones; a vector of vectors gives one per layer leg). `init` — a
previously updated `InfiniteCTM2D` with the same `maxdim` and leg dimensions — warm-starts the
iteration from its state (for a slowly changing site, e.g. inside an optimisation). Other
keywords as for [`CTMOptions`](@ref); `projector = :cut` only.
"""
struct InfiniteCTM2D
    site::Vector{Any}                    # layer tensors
    legs::NTuple{4, Vector{Index}}       # x⁻, x⁺, y⁻, y⁺ — one index per layer leg
    rawdims::NTuple{2, Vector{Int}}      # per axis, the layer legs' dimensions
    maxdim::Int
    options::CTMOptions
    pair::Symbol
    boundary::Any
    c4v::Bool                            # derive one pair and two blocks per step, the rest by symmetry
    state::Any                           # `nothing`, or an `_I2State`
    lnkappa::Base.RefValue{Any}
    stats::Base.RefValue{Any}            # what the last `update` did
end

struct _I2State
    chi::Int
    B::Dict{NTuple{2, Int}, Any}         # block type → tensor on canonical legs
    P::Dict{NTuple{2, Int}, Any}         # line-interface type → (P_A, P_B) on canonical legs
    leg::Dict{Any, Any}                  # canonical legs (vectors of indices), see `_i2_legs`
end

function InfiniteCTM2D(site, legs, maxdim::Integer; pair::Symbol = :biorth, boundary = nothing,
                       init = nothing, c4v::Bool = false, kwargs...)
    opts = CTMOptions(; kwargs...)
    opts.projector === :cut || throw(ArgumentError("InfiniteCTM2D supports projector = :cut only"))
    maxdim >= 1 || throw(ArgumentError("maxdim must be ≥ 1, got $maxdim"))
    pair in (:biorth, :isometric) || throw(ArgumentError("pair must be :biorth or :isometric, got $(repr(pair))"))
    layers = _i2_layers(site)
    isempty(layers) && throw(ArgumentError("the site needs at least one layer"))
    length(legs) == 4 || throw(ArgumentError("legs must be (x⁻, x⁺, y⁻, y⁺)"))
    lg = ntuple(d -> legs[d] isa Index ? Index[legs[d]] : collect(Index, legs[d]), 4)
    for (m, p, ax) in ((1, 2, "x"), (3, 4, "y"))
        (length(lg[m]) == length(lg[p]) && dim.(lg[m]) == dim.(lg[p])) || throw(ArgumentError(
            "the $(ax)⁻ and $(ax)⁺ legs must pair up with equal dimensions, got $(dim.(lg[m])) and $(dim.(lg[p]))"))
        isempty(lg[m]) && throw(ArgumentError("the site needs at least one $(ax) leg per side"))
    end
    issetequal(_c3_open(layers), reduce(vcat, collect(lg))) || throw(ArgumentError(
        "the site's open indices (those on exactly one layer) must be exactly the given legs"))
    rawdims = (dim.(lg[1]), dim.(lg[3]))
    if c4v
        rawdims[1] == rawdims[2] || throw(ArgumentError("c4v = true needs equal x and y leg dimensions"))
        _i2_isc4v(layers, lg) || throw(ArgumentError(
            "c4v = true needs every layer invariant under the square's symmetries of its legs"))
    end
    state = nothing
    if !isnothing(init)
        init isa InfiniteCTM2D || throw(ArgumentError("init must be an InfiniteCTM2D"))
        (init.maxdim == maxdim && init.rawdims == rawdims) || throw(ArgumentError(
            "init must have the same maxdim and leg dimensions"))
        state = init.state
    end
    return InfiniteCTM2D(layers, lg, rawdims, Int(maxdim), opts, pair, boundary, c4v, state,
                         Ref{Any}(nothing), Ref{Any}(nothing))
end

# A site given as one tensor or as a collection of layer tensors. (A tensor is itself an array, so
# the test is on the tensor type, not on `AbstractVector`.)
_i2_layers(site) = site isa AbstractTensor ? Any[site] : collect(Any, site)

_i2_setstate(ic::InfiniteCTM2D, st) =
    InfiniteCTM2D(ic.site, ic.legs, ic.rawdims, ic.maxdim, ic.options, ic.pair, ic.boundary, ic.c4v, st,
                  Ref{Any}(nothing), Ref{Any}(nothing))
options(ic::InfiniteCTM2D) = ic.options

# Fresh legs for face `F`: one per layer leg for a raw face, one χ-leg for a line face.
_i2_newlegs(F, χ::Int, rawdims, tag::String) = _c2_israw(F) ?
    Index[new_index(d; tags = "Link,$tag") for d in rawdims[F[1]]] : Index[new_index(χ; tags = "Link,$tag")]

# Canonical legs at kept width χ: `(s, a, side)` per block face, `(:w, t)` per pair's kept leg,
# `(:slot, t, collapsed?)` per pair input part.
function _i2_legs(χ::Int, rawdims)
    leg = Dict{Any, Any}()
    for s in _i2_types(), (F, side) in _c2_faces(_i2_centre_key(s), _I2_L)
        leg[(s, F[1], side)] = _i2_newlegs(F, χ, rawdims, "i2")
    end
    for (t, F) in _i2_representatives()
        leg[(:w, t)] = new_index(χ; tags = "Link,i2w")
        for (g, pk) in _c2_face_parts(F)
            leg[(:slot, t, g)] = _i2_newlegs(pk, χ, rawdims, "i2s")
        end
    end
    return leg
end

# --- placement onto the virtual box, and back ----------------------------------------------

function _i2_relabel(t, old::Vector, new::Vector)
    have = Set(inds(t))
    sel = [j for j in eachindex(old) if old[j] in have]
    isempty(sel) && return t
    return replaceinds(t, old[sel], new[sel])
end

_i2_site_faces(q::NTuple{2, Int}) =
    ((1, q[1] - 1, 0, q[2]), (1, q[1], 0, q[2]), (2, q[2] - 1, 0, q[1]), (2, q[2], 0, q[1]))

function _i2_site_relabelling(legs, q::NTuple{2, Int}, VX)
    old = Index[]; new = Index[]
    for (d, F) in enumerate(_i2_site_faces(q))
        append!(old, legs[d]); append!(new, VX[F])
    end
    return old, new
end

function _i2_place_site(layers, legs, q::NTuple{2, Int}, VX)
    old, new = _i2_site_relabelling(legs, q, VX)
    return AbstractTensor[_i2_relabel(t, old, new) for t in layers]
end

function _i2_block_relabelling(s, bk::NTuple{4, Int}, VX, leg)
    can = Index[]; vir = Index[]
    for (F, side) in _c2_faces(bk, _I2_L)
        append!(can, leg[(s, F[1], side)]); append!(vir, VX[F])
    end
    return can, vir
end

function _i2_place_block(st::_I2State, bk::NTuple{4, Int}, VX)
    s = (bk[1], bk[2])
    t = get(st.B, s, nothing)
    isnothing(t) && return nothing
    can, vir = _i2_block_relabelling(s, bk, VX, st.leg)
    length(can) == length(inds(t)) || error(
        "2D iCTM: block type $s placed at $bk has $(length(can)) face legs but $(length(inds(t))) indices")
    return replaceinds(t, can, vir)
end

function _i2_pair_relabelling(F::NTuple{4, Int}, VX, leg)
    t = _i2_itype(F)
    can = Index[]; vir = Index[]
    for (g, pk) in _c2_face_parts(F)
        append!(can, leg[(:slot, t, g)]); append!(vir, VX[pk])
    end
    push!(can, leg[(:w, t)]); push!(vir, only(VX[F]))
    return can, vir
end

function _i2_place_pair(pairs, leg, F::NTuple{4, Int}, VX)
    pr = get(pairs, _i2_itype(F), nothing)
    isnothing(pr) && return nothing
    can, vir = _i2_pair_relabelling(F, VX, leg)
    return (replaceinds(pr[1], can, vir), replaceinds(pr[2], can, vir), only(VX[F]))
end

# The virtual box's view of a canonical state: face indices, sites, blocks and pairs.
function _i2_virtual(st::_I2State, ic::InfiniteCTM2D)
    VX = _C3Lazy{NTuple{4, Int}}(F -> _i2_newlegs(F, st.chi, ic.rawdims, "i2v"))
    tbl = _C3Lazy{NTuple{2, Int}}(q -> _i2_place_site(ic.site, ic.legs, q, VX))
    Bv = _C3Lazy{NTuple{4, Int}}(bk -> _i2_place_block(st, bk, VX))
    Pv = _C3Lazy{NTuple{4, Int}}(F -> _i2_place_pair(st.P, st.leg, F, VX))
    return VX, tbl, Bv, Pv
end

# --- seed and iteration --------------------------------------------------------------------

function _i2_boundary(ic::InfiniteCTM2D, j::Int, d::Int, elt)
    b = ic.boundary
    isnothing(b) && return ones(elt, d)
    v = b isa AbstractVector && !isempty(b) && first(b) isa AbstractVector ? b[j] : b
    length(v) == d || throw(ArgumentError("boundary vector has length $(length(v)) for a raw leg of dimension $d"))
    return convert(Vector{elt}, collect(v))
end

function _i2_seed_state(ic::InfiniteCTM2D, χ::Int)
    leg = _i2_legs(χ, ic.rawdims)
    elt = promote_type(map(scalartype, ic.site)...)
    ref = first(ic.site)
    B = Dict{NTuple{2, Int}, Any}()
    for s in _i2_types()
        t = nothing
        for (F, side) in _c2_faces(_i2_centre_key(s), _I2_L), (j, l) in enumerate(leg[(s, F[1], side)])
            v = _c2_israw(F) ? from_array(_i2_boundary(ic, j, dim(l), elt), l) : onehot(elt, l => 1)
            v = adapt_like(ref, v)
            t = isnothing(t) ? v : t * v
        end
        B[s] = t
    end
    return _I2State(χ, B, Dict{NTuple{2, Int}, Any}(), leg)
end

# --- the SPLIT pair: matrix-free, never forming an enlarged quadrant ----------------------------
#
# The dense pair contracts each enlarged quadrant — the old quadrant, its two half-lines and the
# site's layers — into one n×n matrix (n = χ·Π raw dims, χ·2D² for a boundary-PEPS sandwich) and
# factorises it: O(n³) = O(χ³D⁶) per pair, which is the whole cost of a step from D = 4 up
# (measured 0.48 s of the 0.56 s pair at D = 4, χ = 32). Here the pair's truncated SVD comes from
# the subspace iteration of the dense engine (`_ctm_subspace_svd`), with the quadrant applied to a
# block of k′ = χ + oversample vectors AS ITS FACTOR LIST: one netcon per application, which absorbs
# the layers into the block one at a time, so the enlarged quadrant and its layer-fused legs never
# exist. That is the "split CTMRG" of Naumann et al. (2024) and Xu, Lin & Zhang (2025) — ket and
# bra layers kept apart through the projector computation — obtained from netcon rather than hand
# written, so it applies unchanged to any number of layers.
#
#   O = L · H   over the interface legs (L the low quadrant, H the high one; rows = L's rest legs)
#   apply(X)   = L (H X)            applyadj(Q) = H̄ (L̄ Q)      (dense: the map adjoint is conj)
#   P_A = H V̄ S^{-1/2}  (consumed by the LOW side),  P_B = S^{-1/2} Ū L   — as `_ctm_whiten`
#
# Warm start from the previous step's pair (`X₀ = H̄ P̄_B`), so near the fixed point one iteration
# suffices. Dense data only; graded data, a declined gate and a subspace bail-out (flat spectrum)
# take the dense route.
_i2_conjlist(L) = AbstractTensor[scalartype(t) <: Real ? t : conj(t) for t in L]
_i2_apply(L, X, opts::CTMOptions) = _ctm_contract(vcat(AbstractTensor[t for t in L], AbstractTensor[X]), opts)

function _i2_pair_split(Ll, Lh, ins::Vector{<:Index}, prev, maxdim::Int, opts::CTMOptions, seed::UInt)
    (any(_ctm_isgraded, Ll) || any(_ctm_isgraded, Lh)) && return missing
    rows = Index[i for i in _c3_open(Ll) if !(i in ins)]
    cols = Index[i for i in _c3_open(Lh) if !(i in ins)]
    (isempty(rows) || isempty(cols)) && return missing
    k = min(maxdim, dim(rows), dim(cols), dim(ins))
    kp = min(k + opts.svd_oversample, dim(rows), dim(cols))
    _ctm_use_subspace(opts, Ll, dim(rows), dim(cols), dim(ins), kp) || return missing
    cLl = _i2_conjlist(Ll); cLh = _i2_conjlist(Lh)
    apply(X) = _i2_apply(Ll, _i2_apply(Lh, X, opts), opts)
    applyadj(Q) = _i2_apply(cLh, _i2_apply(cLl, Q, opts), opts)
    ref = first(Lh)
    elt = promote_type(map(scalartype, Ll)..., map(scalartype, Lh)...)
    X0 = nothing
    if !isnothing(prev) && length(prev) >= 3
        PBo, wo = prev[2], prev[3]
        issetequal(collect(inds(PBo)), vcat(collect(ins), [wo])) && (X0 = _i2_apply(cLh, dag(PBo, [wo]), opts))
    end
    nrand = isnothing(X0) ? kp : max(0, kp - dim(only(uniqueinds(X0, cols))))
    if nrand > 0
        R = adapt_like(ref, _ctm_random_block(Xoshiro(seed), elt, cols, nrand))
        X0 = isnothing(X0) ? R :
            directsum(X0 => only(uniqueinds(X0, cols)), R => only(uniqueinds(R, cols)); tags = "Link,blk")
    end
    F = try
        _ctm_subspace_svd(apply, applyadj, rows, cols, X0, k, _ctm_trunc(maxdim, opts), opts;
                          warm = !isnothing(prev) && nrand < kp)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    isnothing(F) && (_ctm_stat!(:i2_split_bail); return nothing)
    U, S, V = F
    isk = map_diag(x -> inv(sqrt(x)), S)
    u, v = inds(S)
    PA = _i2_apply(Lh, dag(V, [v]), opts) * dag(isk, [u])             # (ins…, u)
    PB = _i2_apply(Ll, dag(U, [u]), opts) * dag(isk, [u])             # (v, ins…)
    uA = only(uniqueinds(PA, ins))
    PB = replaceind(PB, only(uniqueinds(PB, ins)), dag(uA))
    all(isfinite, (norm(PA), norm(PB))) || (_ctm_stat!(:i2_split_bail); return nothing)
    _ctm_stat!(:i2_split)
    return PA, PB, uA
end

# The pair of line interface `F` from the enlarged quadrants on either side, aligned to `prev`.
function _i2_pair(F::NTuple{4, Int}, Bv, tbl, prev, VX, opts::CTMOptions, χ::Int, isometric::Bool)
    Ll = _c2_enlarged(Bv, tbl, _c2_side_block(F, -1))
    Lh = _c2_enlarged(Bv, tbl, _c2_side_block(F, 1))
    (isempty(Ll) || isempty(Lh)) && return nothing
    ol = _c3_open(Ll); oh = _c3_open(Lh)
    ins = Index[i for i in ol if i in oh]
    isempty(ins) && return nothing
    if !isometric && opts.svd !== :dense
        pr = _i2_pair_split(Ll, Lh, ins, prev, χ, opts, hash(F))
        pr isa Tuple && return _c3_finish(pr, ins, only(VX[F]), prev, opts)
    end
    Ac = _ctm_rescale(_ctm_contract(Ll, opts))
    Bc = _ctm_rescale(_ctm_contract(Lh, opts))
    pr = isometric ? _c3_isometric(Ac, Bc, ins, χ, opts) : _ctm_twosided_projector_qr(Ac, Bc, ins, χ, opts)
    isnothing(pr) && return nothing
    return _c3_finish(pr, ins, only(VX[F]), prev, opts)
end

# --- the square's symmetry (`c4v = true`) -----------------------------------------------------
#
# When every layer is invariant under the 8 symmetries of the square acting on its legs (and the x
# and y legs match), the whole iteration is equivariant: the 4 pairs are images of one, the 8
# blocks images of one quadrant and one half-line. A step then derives 1 pair and grows 2 blocks
# and relabels the rest — 4× less work, which on a GPU (where the 4 pairs of a step queue behind
# each other) is 4× less time. An element `g = (swap, ε₁, ε₂)` maps axis a to σ(a) (σ swaps the
# axes when `swap`) and then multiplies the sign along axis σ(a) by ε_{σ(a)}. A face keeps its kind;
# a reflection along a face's own axis exchanges its low and high side, and so exchanges the roles
# of a pair's P_A (consumed on the low side) and P_B. Raw faces map layer by layer (the legs of a
# direction are listed in the same layer order for every direction).
const _I2_C4V = [(sw, e1, e2) for sw in (false, true) for e1 in (1, -1) for e2 in (1, -1)]
_i2_sig(g, a) = g[1] ? 3 - a : a
_i2_eps(g, a) = a == 1 ? g[2] : g[3]
function _i2_act_type(g, s)
    out = [0, 0]
    for a in 1:2
        b = _i2_sig(g, a); out[b] = _i2_eps(g, b) * s[a]
    end
    return (out[1], out[2])
end
_i2_act_side(g, a, side) = _i2_eps(g, _i2_sig(g, a)) == -1 ? (side === :low ? :high : :low) : side
# direction d ∈ 1:4 = (x⁻, x⁺, y⁻, y⁺) → its image
function _i2_act_dir(g, d)
    a = cld(d, 2); sgn = isodd(d) ? -1 : 1
    b = _i2_sig(g, a); sb = _i2_eps(g, b) * sgn
    return 2b - (sb == -1 ? 1 : 0)
end

function _i2_isc4v(layers, lg; atol = 1.0e-12)
    old = reduce(vcat, collect(lg))
    for g in _I2_C4V
        new = reduce(vcat, [lg[_i2_act_dir(g, d)] for d in 1:4])
        for t in layers
            nt = norm(t)
            norm(_i2_relabel(t, old, new) - t) <= atol * max(nt, 1) || return false
        end
    end
    return true
end

_i2_find(g_of, src, dst) = something(findfirst(g -> g_of(g, src) == dst, _I2_C4V), 0)

# Block type `s0`'s tensor carried onto block type `s = g(s0)`, canonical legs to canonical legs.
function _i2_image_block(t, g, s0, leg)
    old = Index[]; new = Index[]
    for (F, side) in _c2_faces(_i2_centre_key(s0), _I2_L)
        s = _i2_act_type(g, s0)
        append!(old, leg[(s0, F[1], side)]); append!(new, leg[(s, _i2_sig(g, F[1]), _i2_act_side(g, F[1], side))])
    end
    return replaceinds(t, old, new)
end

# Pair type `t0 = (a, sb)`'s pair carried onto `t = g(t0)`; a reflection along a swaps P_A and P_B.
_i2_act_pairtype(g, t0) = (_i2_sig(g, t0[1]), _i2_eps(g, _i2_sig(g, 3 - t0[1])) * t0[2])
function _i2_image_pair(pr, g, t0, leg)
    t = _i2_act_pairtype(g, t0)
    old = Index[]; new = Index[]
    for c in (false, true)
        haskey(leg, (:slot, t0, c)) || continue
        append!(old, leg[(:slot, t0, c)]); append!(new, leg[(:slot, t, c)])
    end
    push!(old, leg[(:w, t0)]); push!(new, leg[(:w, t)])
    PA = replaceinds(pr[1], old, new); PB = replaceinds(pr[2], old, new)
    return _i2_eps(g, _i2_sig(g, t0[1])) == -1 ? (PB, PA) : (PA, PB)
end

# One iteration: every pair from the current blocks, then the centre shell regrown with them.
function _i2_step(ic::InfiniteCTM2D, st::_I2State)
    opts = ic.options
    VX, tbl, Bv, Pv = _i2_virtual(st, ic)
    reps = _i2_representatives()
    # with c4v, derive the pair of type (1, −1) only
    derive = ic.c4v ? [r for r in reps if r[1] == (1, -1)] : reps
    Pn = Dict{NTuple{4, Int}, Any}()
    lk = ReentrantLock()
    _ctm_foreach(eachindex(derive)) do i
        F = derive[i][2]
        pr = _i2_pair(F, Bv, tbl, get(Pv, F, nothing), VX, opts, st.chi, ic.pair === :isometric)
        isnothing(pr) || lock(() -> (Pn[F] = pr), lk)
    end
    all(((t, F),) -> haskey(Pn, F), derive) || error("2D iCTM: an interface type got no pair")
    newP = Dict{NTuple{2, Int}, Any}()
    for (t, F) in derive
        can, vir = _i2_pair_relabelling(F, VX, st.leg)
        newP[t] = (replaceinds(Pn[F][1], vir, can), replaceinds(Pn[F][2], vir, can))
    end
    if ic.c4v
        t0 = (1, -1)
        for (t, _) in reps
            t == t0 && continue
            newP[t] = _i2_image_pair(newP[t0], _I2_C4V[_i2_find(_i2_act_pairtype, t0, t)], t0, st.leg)
        end
    end
    Pplace = _C3Lazy{NTuple{4, Int}}(F -> _i2_place_pair(newP, st.leg, F, VX))
    types = _i2_types()
    grow = ic.c4v ? [(-1, -1), (-1, 0)] : types
    nb = Vector{Any}(nothing, length(grow))
    _ctm_foreach(eachindex(grow)) do i
        s = grow[i]; bk = _i2_centre_key(s)
        extras = Any[_c3_side_proj(Pplace[F], side) for (F, side) in _c2_faces(bk, _I2_L) if !_c2_israw(F)]
        blk = _ctm_rescale(_ctm_absorb(opts, _c2_enlarged(Bv, tbl, bk), extras...))
        can, vir = _i2_block_relabelling(s, bk, VX, st.leg)
        nb[i] = replaceinds(blk, vir, can)
    end
    B = Dict{NTuple{2, Int}, Any}(grow[i] => nb[i] for i in eachindex(grow))
    if ic.c4v
        for s in types
            haskey(B, s) && continue
            s0 = count(!=(0), s) == 2 ? (-1, -1) : (-1, 0)
            B[s] = _i2_image_block(B[s0], _I2_C4V[_i2_find(_i2_act_type, s0, s)], s0, st.leg)
        end
    end
    return _I2State(st.chi, B, newP, st.leg)
end

# Largest phase-free change of any block (blocks are norm-1 on fixed, aligned legs).
function _i2_blockdist(a::_I2State, b::_I2State)
    worst = 0.0
    for (s, ta) in a.B
        tb = b.B[s]
        na, nb = norm(ta), norm(tb)
        (na > 0 && nb > 0) || continue
        ov = dot(ta, tb)
        ph = iszero(ov) ? one(ov) : conj(ov) / abs(ov)
        worst = max(worst, norm(ta / na - tb * (ph / nb)))
    end
    return worst
end

# --- per-site quantities ----------------------------------------------------------------------

const _I2_REGIONS = [(3.0, 3.0), (3.5, 3.0), (3.0, 3.5), (3.5, 3.5)]

function _i2_lnkappa(ic::InfiniteCTM2D, st::_I2State)
    VX, tbl, Bv, _ = _i2_virtual(st, ic)
    vals = zeros(length(_I2_REGIONS))
    for i in eachindex(_I2_REGIONS)
        c = _I2_REGIONS[i]
        ts = _c2_region_blocks(Bv, c)
        all(isinteger, c) && append!(ts, tbl[_I2_V])
        vals[i] = log(abs(scalar(_ctm_contract(ts, ic.options))))
    end
    return vals[1] - vals[2] - vals[3] + vals[4]
end

"""
    cvm_freenergy(ic::InfiniteCTM2D)

ln κ, the logarithm of the partition function per site, in the Kikuchi form
ln Z_v − ln Z_ex − ln Z_ey + ln Z_f.
"""
function cvm_freenergy(ic::InfiniteCTM2D)
    isnothing(ic.state) && error("InfiniteCTM2D has not been `update`d.")
    r = ic.lnkappa
    r[] === nothing && (r[] = _i2_lnkappa(ic, ic.state))
    return r[]
end

# The centre vertex's 8-block shell contracted into one tensor on its raw legs.
function _i2_shell(ic::InfiniteCTM2D)
    isnothing(ic.state) && error("InfiniteCTM2D has not been `update`d.")
    VX, tbl, Bv, _ = _i2_virtual(ic.state, ic)
    return VX, tbl, _ctm_contract(_c2_region_blocks(Bv, Float64.(_I2_V)), ic.options)
end

"""
    site_ratio(ic::InfiniteCTM2D, impurity)

`⟨impurity⟩ / ⟨site⟩` for an impurity site — a tensor or layer list with the same legs as the
site — through the centre vertex's full environment: a single-site expectation value.
"""
function site_ratio(ic::InfiniteCTM2D, impurity)
    VX, tbl, env = _i2_shell(ic)
    imp = _i2_place_site(_i2_layers(impurity), ic.legs, _I2_V, VX)
    z0 = scalar(_ctm_contract(vcat(AbstractTensor[env], tbl[_I2_V]), ic.options))
    z1 = scalar(_ctm_contract(vcat(AbstractTensor[env], imp), ic.options))
    return z1 / z0
end

"""
    site_environment(ic::InfiniteCTM2D, k = 1) -> (E, z)

The environment of layer `k` of the site — the centre vertex's shell contracted with every other
layer — on layer `k`'s own indices, and `z = ⟨E, layer k⟩`. For a translation-invariant network in
which layer `k`'s tensor `a` appears once per site, `d ln κ = ⟨E, da⟩ / z`: the gradient of the free
energy per site needs no sum over positions.
"""
function site_environment(ic::InfiniteCTM2D, k::Integer = 1)
    1 <= k <= length(ic.site) || throw(ArgumentError("layer $k does not exist (the site has $(length(ic.site)))"))
    VX, tbl, env = _i2_shell(ic)
    placed = tbl[_I2_V]
    E = _ctm_contract(vcat(AbstractTensor[env], AbstractTensor[placed[j] for j in eachindex(placed) if j != k]), ic.options)
    z = scalar(E * placed[k])
    old, new = _i2_site_relabelling(ic.legs, _I2_V, VX)
    return _i2_relabel(E, new, old), z
end

# --- Anderson acceleration of the fixed-point iteration ----------------------------------------
#
# The step x ↦ G(x) acts on blocks with fixed, aligned legs (every pair is Procrustes-aligned to
# its predecessor), so iterates are directly comparable vectors and type-II Anderson mixing applies:
# with f_k = G(x_k) − x_k and the last m differences ΔX, ΔF,
#   γ = argmin ‖f_k − ΔF γ‖,   x_{k+1} = G(x_k) − (ΔX + ΔF) γ.
# Blocks are renormalised after mixing (their scale is a gauge). A mixed point whose residual grows
# past 10× the smallest seen restarts the history. The pairs of G(x_k) go along unchanged: they are
# only the next step's warm start and alignment reference.
mutable struct _I2Anderson
    m::Int
    X::Vector{Dict{NTuple{2, Int}, Any}}
    F::Vector{Dict{NTuple{2, Int}, Any}}
    best::Float64
end
_I2Anderson(m::Integer) = _I2Anderson(Int(m), Dict{NTuple{2, Int}, Any}[], Dict{NTuple{2, Int}, Any}[], Inf)
_i2_bdot(a, b) = sum(real(dot(a[s], b[s])) for s in keys(a))
_i2_bcomb(a, b, β) = Dict{NTuple{2, Int}, Any}(s => a[s] + β * b[s] for s in keys(a))

function _i2_anderson!(acc::_I2Anderson, x::_I2State, gx::_I2State)
    f = _i2_bcomb(gx.B, x.B, -1.0)
    r = sqrt(_i2_bdot(f, f))
    if !isfinite(r) || r > 10 * acc.best
        empty!(acc.X); empty!(acc.F); acc.best = r
        return gx
    end
    acc.best = min(acc.best, r)
    push!(acc.X, x.B); push!(acc.F, f)
    length(acc.X) > acc.m + 1 && (popfirst!(acc.X); popfirst!(acc.F))
    k = length(acc.X) - 1
    k >= 1 || return gx
    dF = [_i2_bcomb(acc.F[i + 1], acc.F[i], -1.0) for i in 1:k]
    dX = [_i2_bcomb(acc.X[i + 1], acc.X[i], -1.0) for i in 1:k]
    M = [_i2_bdot(dF[i], dF[j]) for i in 1:k, j in 1:k]
    v = [_i2_bdot(dF[i], f) for i in 1:k]
    γ = (M + 1.0e-12 * (tr(M) / k + eps()) * I) \ v
    all(isfinite, γ) || (empty!(acc.X); empty!(acc.F); return gx)
    B = copy(gx.B)
    for i in 1:k
        for s in keys(B)
            B[s] = B[s] - γ[i] * (dX[i][s] + dF[i][s])
        end
    end
    for s in keys(B)
        B[s] = _ctm_rescale(B[s])
    end
    return _I2State(gx.chi, B, gx.P, gx.leg)
end

"""
    update(ic::InfiniteCTM2D; maxiter = 1000, tolerance = 1e-10, convergence = :environment,
           miniter = 2, verbose = false)

Iterate to the fixed point from the open-boundary vacuum (or from `ic`'s state, e.g. a warm start
via `init`) until the signal falls below `tolerance`:

* `:environment` (default) — the phase-free change of the centre vertex's normalised environment
  (its 8-block shell contracted onto the site's legs). Gauge-invariant, so it measures exactly what
  observables and [`site_environment`](@ref) gradients see, and ignores kept directions that carry
  no weight: those settle slowly and made a block-change criterion take as long from a warm start
  as from the vacuum (2D Ising K = 0.5 → 0.51, χ = 16: 124 iterations warm against 118 cold).
* `:blocks` — the largest phase-free change of any block.
* `:lnkappa` — `|Δ ln κ| ≤ tolerance · max(1, |ln κ|)`. ln κ is stationary at the fixed point and
  settles long before the environment: stopping on it left Yang's magnetisation 5e-7 off and
  environment gradients 1e-7 off (K = 0.5, χ = 16), against 3e-16 and 1e-9 converged.

`anderson = m > 0` mixes the last `m` iterates (type-II Anderson acceleration, from iteration
`anderson_start`); the fixed point is unchanged. `ic.stats[]` records the iterations, the final
change and whether it converged.
"""
function update(ic::InfiniteCTM2D; maxiter::Integer = 1000, tolerance::Real = 1.0e-10,
                convergence::Symbol = :environment, miniter::Integer = 2, verbose::Bool = false,
                anderson::Integer = 0, anderson_start::Integer = 3)
    convergence in (:environment, :blocks, :lnkappa) || throw(ArgumentError(
        "convergence must be :environment, :blocks or :lnkappa, got $(repr(convergence))"))
    st = isnothing(ic.state) ? _i2_seed_state(ic, ic.maxdim) : ic.state
    signal(s) = convergence === :lnkappa ? _i2_lnkappa(ic, s) :
                convergence === :environment ? _i2_shellenv(ic, s) : nothing
    prev = signal(st)
    converged, Δ, n = false, Inf, 0
    acc = anderson > 0 ? _I2Anderson(anderson) : nothing
    x = st                                             # the point the next step is applied to
    for it in 1:maxiter
        stn = _i2_step(ic, x)
        # Anderson: the next point mixes the last iterates; the reported state stays G(x), a
        # genuine CTM step, so the fixed point and every signal are unchanged.
        x = (isnothing(acc) || it < anderson_start) ? stn : _i2_anderson!(acc, x, stn)
        cur = signal(stn)
        if convergence === :lnkappa
            Δ = abs(cur - prev)
            crit = Δ / max(1.0, abs(cur))
        elseif convergence === :environment
            Δ = _i2_phasefree(prev, cur); crit = Δ
        else
            Δ = _i2_blockdist(st, stn); crit = Δ
        end
        st = stn; prev = cur; n = it
        verbose && @info "2D iCTM iteration $it (χ = $(ic.maxdim)): $convergence change $Δ"
        if it >= miniter && crit <= tolerance
            converged = true
            break
        end
    end
    converged || @warn "2D iCTM did not converge to tolerance $tolerance after $maxiter iterations " *
        "(final $convergence change = $Δ)."
    out = _i2_setstate(ic, st)
    convergence === :lnkappa && (out.lnkappa[] = prev)
    out.stats[] = (iterations = n, change = Δ, converged = converged)
    return out
end

# The centre vertex's shell contracted onto the SITE'S legs (fixed across iterations, unlike the
# virtual box's), normalised: the gauge-invariant environment every single-site quantity reads.
function _i2_shellenv(ic::InfiniteCTM2D, st::_I2State)
    VX, _, Bv, _ = _i2_virtual(st, ic)
    env = _ctm_contract(_c2_region_blocks(Bv, Float64.(_I2_V)), ic.options)
    old, new = _i2_site_relabelling(ic.legs, _I2_V, VX)
    return _ctm_rescale(_i2_relabel(env, new, old))
end

function _i2_phasefree(a, b)
    ov = dot(a, b)
    ph = iszero(ov) ? one(ov) : conj(ov) / abs(ov)
    return norm(a - b * ph)
end

"""
    ising2d_site(β; J = (1.0, 1.0), h = 0.0) -> (site, legs, magnetisation)

The square-lattice Ising model's site tensor for [`InfiniteCTM2D`](@ref): spins on the vertices,
weight `exp(β Σ J_a σσ′ + β h Σ σ)`, each bond's Boltzmann matrix split symmetrically between its
two sites. Returns the site tensor, its legs `(x⁻, x⁺, y⁻, y⁺)` and the impurity tensor with σ
inserted.
"""
function ising2d_site(β::Real; J = (1.0, 1.0), h::Real = 0.0)
    legs = Tuple(new_index(2; tags = "i2,$n") for n in ("xm", "xp", "ym", "yp"))
    function sqrtW(K)
        K >= 0 || throw(ArgumentError("ising2d_site takes ferromagnetic couplings, got βJ = $K"))
        λ1, λ2 = cosh(K), sinh(K)
        α, ϕ = (sqrt(λ1) + sqrt(λ2)) / 2, (sqrt(λ1) - sqrt(λ2)) / 2
        return sqrt(2) * [α ϕ; ϕ α]
    end
    function tensor(sgn)
        A = zeros(2, 2, 2, 2)
        A[1, 1, 1, 1] = exp(β * h)
        A[2, 2, 2, 2] = sgn * exp(-β * h)
        t = from_array(A, legs...)
        for (n, l) in enumerate(legs)
            m = sqrtW(β * J[cld(n, 2)])
            t = replaceind(from_array(m, l, prime(l)) * t, prime(l), l)
        end
        return t
    end
    return tensor(1), legs, tensor(-1)
end
