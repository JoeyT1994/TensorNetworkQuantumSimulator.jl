# Infinite (translation-invariant) 3D CTMRG — the cubic engine of ctm3denvironmentcache.jl on a
# 1×1×1 unit cell, for the thermodynamic limit directly.
#
# STATE. Translation invariance collapses the position-resolved environment to ONE tensor per
# block type — 8 octants, 12 quarter-planes, 6 half-lines, keyed by the sign triple `s` — and ONE
# projector pair per interface type `(a, sb, sc)`: 12 plane types and 12 line types. Each carries
# fixed CANONICAL legs: a block's leg per face `(s, a, side)`, a pair's input slot per part of the
# face (`_c3_face_parts`: old face, new strips, new corner bond) and its kept leg.
#
# THE ITERATION reuses the finite engine verbatim on a small VIRTUAL box (`_I3_L`): lazy
# dictionaries hand `_c3_enlarged`, `_c3_pair_cut`, `_c3_cycle_cube` and `_c3_region_blocks`
# copies of the canonical tensors relabelled onto the box's per-face indices, on demand. One
# iteration derives the 24 pairs at representative faces around the box centre (the 12 plane
# types at the faces of one cube, so `:cycle` can close it), grows the 26 blocks of the centre
# vertex's shell with them, and relabels the results back to canonical legs. It is exactly the
# finite sweep with every block of a type replaced by the same tensor.
#
# F PER SITE. The Möbius sum per unit cell — one vertex, three edge, three face and one cube region
# per site — gives ln κ = ln Z_v − Σ ln Z_e + Σ ln Z_f − ln Z_c (the 3D form of the 2D CTMRG
# partition-function-per-site formula); every block's scale cancels as in the finite sum.
#
# SEED. Open-boundary vacuum: every plane and line leg starts one-dimensional (e₁) and a half-line's
# vertex leg carries `boundary` (ones: free spins at infinity; e₁: a fixed-spin boundary that picks
# a symmetry-broken phase). Uncompressed iterations at `seed_maxdim` grow the environment until
# ln κ settles, the state is zero-padded onto `maxdim`, and compressed iterations take over.

# A dictionary that builds values on demand (`make(key)`, `nothing` = absent) and memoises them —
# how the virtual box sees the canonical state without placing every block up front.
struct _C3Lazy{K} <: AbstractDict{K, Any}
    make::Function
    memo::Dict{K, Any}
    lk::ReentrantLock
end
_C3Lazy{K}(f) where {K} = _C3Lazy{K}(f, Dict{K, Any}(), ReentrantLock())
function Base.get(d::_C3Lazy, k, default)
    v = lock(d.lk) do
        haskey(d.memo, k) ? d.memo[k] : (d.memo[k] = d.make(k))
    end
    return isnothing(v) ? default : v
end
function Base.getindex(d::_C3Lazy, k)
    v = get(d, k, nothing)
    isnothing(v) && throw(KeyError(k))
    return v
end
Base.haskey(d::_C3Lazy, k) = !isnothing(get(d, k, nothing))
Base.length(d::_C3Lazy) = length(d.memo)
Base.iterate(d::_C3Lazy, args...) = iterate(d.memo, args...)

const _I3_L = (5, 5, 5)                                  # the virtual box …
const _I3_V = (3, 3, 3)                                  # … and its centre vertex

_i3_types() = [(s1, s2, s3) for s1 in -1:1 for s2 in -1:1 for s3 in -1:1 if (s1, s2, s3) != (0, 0, 0)]
# The centre vertex's shell key of block type `s` (sign −1 and 0 at 3, +1 at 4).
_i3_centre_key(s) = (s[1], s[2], s[3], (s[a] == 1 ? 4 : 3 for a in 1:3)...)
_i3_itype(F::NTuple{6, Int}) = (F[1], F[3], F[5])

# One representative face per interface type: the 12 plane types at the faces of cube (4,4,4) (so
# `:cycle` can close it), the 12 line types beside it.
function _i3_representatives()
    reps = Tuple{NTuple{3, Int}, NTuple{6, Int}}[]
    for a in 1:3, sb in (-1, 1), sc in (-1, 1)
        push!(reps, ((a, sb, sc), (a, 3, sb, 4, sc, 4)))
    end
    for a in 1:3, (sb, sc) in ((0, -1), (0, 1), (-1, 0), (1, 0))
        push!(reps, ((a, sb, sc), (a, 3, sb, sb == 0 ? 3 : 4, sc, sc == 0 ? 3 : 4)))
    end
    return reps
end

"""
    InfiniteCTM3D(site, legs, maxdim; projector = :cut, seed_maxdim = 2, boundary = nothing,
                  plane_rest = :compress, kwargs...)

Infinite 3D CTMRG environment of the translation-invariant cubic network built from one `site`
tensor with six legs `legs = (x⁻, x⁺, y⁻, y⁺, z⁻, z⁺)`: every site's `x⁺` leg is contracted with its
+x neighbour's `x⁻` leg, likewise along y and z. [`update`](@ref) iterates it to a fixed point;
[`cvm_freenergy`](@ref) returns ln κ, the log partition function per site, and
[`site_ratio`](@ref) a single-site impurity ratio (e.g. the magnetisation).

`boundary` is the seed's vector on a half-line's vertex leg (default all ones: free spins at
infinity); pass a fixed-spin vector (e.g. `[1, 0]` for Ising) to select a symmetry-broken phase.
Other keywords as for [`CTM3DEnvironmentCache`](@ref).
"""
struct InfiniteCTM3D
    site::Any
    legs::NTuple{6, Any}
    maxdim::Int
    options::CTMOptions
    seed_maxdim::Int
    plane_rest::Symbol
    boundary::Any
    state::Any                     # `nothing`, or an `_I3State`
    lnkappa::Base.RefValue{Any}
end

struct _I3State
    chi::Int
    B::Dict{NTuple{3, Int}, Any}   # block type → tensor on canonical legs
    P::Dict{NTuple{3, Int}, Any}   # interface type → (P_A, P_B) on canonical legs (what built B)
    Q::Dict{NTuple{3, Int}, Any}   # the same iteration's :cut pairs (=== P under :cut)
    leg::Dict{Any, Any}            # canonical legs, see `_i3_legs`
end

function InfiniteCTM3D(site, legs, maxdim::Integer; seed_maxdim::Integer = 2, plane_rest::Symbol = :compress,
                       boundary = nothing, kwargs...)
    opts = CTMOptions(; kwargs...)
    maxdim >= 1 || throw(ArgumentError("maxdim must be ≥ 1, got $maxdim"))
    seed_maxdim >= 1 || throw(ArgumentError("seed_maxdim must be ≥ 1, got $seed_maxdim"))
    plane_rest in (:compress, :exact) || throw(ArgumentError(
        "plane_rest must be :compress or :exact, got $(repr(plane_rest))"))
    (plane_rest === :exact && opts.projector === :cycle) && throw(ArgumentError(
        "projector = :cycle closes the cube with :cut pairs of compressed octants, so it needs plane_rest = :compress"))
    length(legs) == 6 || throw(ArgumentError("legs must be the six legs (x⁻, x⁺, y⁻, y⁺, z⁻, z⁺)"))
    issetequal(collect(inds(site)), collect(legs)) || throw(ArgumentError(
        "the site tensor's indices must be exactly the six given legs"))
    for a in 1:3
        dim(legs[2a - 1]) == dim(legs[2a]) || throw(ArgumentError(
            "legs along axis $a have different dimensions $(dim(legs[2a - 1])) and $(dim(legs[2a]))"))
    end
    return InfiniteCTM3D(site, Tuple(legs), Int(maxdim), opts, Int(seed_maxdim), plane_rest, boundary,
                         nothing, Ref{Any}(nothing))
end

_i3_setstate(ic::InfiniteCTM3D, st) = InfiniteCTM3D(ic.site, ic.legs, ic.maxdim, ic.options, ic.seed_maxdim,
                                                    ic.plane_rest, ic.boundary, st, Ref{Any}(nothing))
options(ic::InfiniteCTM3D) = ic.options

_i3_rawdim(legs, a::Int) = dim(legs[2a - 1])
_i3_legdim(t::NTuple{3, Int}, χ::Int, legs) = (t[2] == 0 && t[3] == 0) ? _i3_rawdim(legs, t[1]) : χ

# Canonical legs at kept width χ: `(s, a, side)` per block face, `(:w, t)` per pair's kept leg,
# `(:slot, t, G)` per pair input part.
function _i3_legs(χ::Int, legs)
    leg = Dict{Any, Any}()
    for s in _i3_types(), (F, side) in _c3_faces(_i3_centre_key(s), _I3_L)
        leg[(s, F[1], side)] = new_index(_i3_legdim(_i3_itype(F), χ, legs); tags = "Link,i3")
    end
    for (t, F) in _i3_representatives()
        leg[(:w, t)] = new_index(χ; tags = "Link,i3w")
        for (G, pk) in _c3_face_parts(F)
            leg[(:slot, t, G)] = new_index(_i3_legdim(_i3_itype(pk), χ, legs); tags = "Link,i3s")
        end
    end
    return leg
end

# --- placement onto the virtual box, and back ----------------------------------------------

_i3_vdim(F::NTuple{6, Int}, χ::Int, legs) = _c3_israw(F) ? _i3_rawdim(legs, F[1]) : χ

function _i3_place_block(st::_I3State, bk::NTuple{6, Int}, VX)
    s = (bk[1], bk[2], bk[3])
    t = get(st.B, s, nothing)
    isnothing(t) && return nothing
    faces = _c3_faces(bk, _I3_L)
    length(faces) == length(inds(t)) || error(
        "3D iCTM: block type $s placed at $bk has $(length(faces)) faces but $(length(inds(t))) legs")
    return replaceinds(t, Index[st.leg[(s, F[1], side)] for (F, side) in faces], Index[VX[F] for (F, _) in faces])
end

function _i3_place_site(site, legs, q::NTuple{3, Int}, VX)
    ks = ((1, q[1] - 1, 0, q[2], 0, q[3]), (1, q[1], 0, q[2], 0, q[3]),
          (2, q[2] - 1, 0, q[1], 0, q[3]), (2, q[2], 0, q[1], 0, q[3]),
          (3, q[3] - 1, 0, q[1], 0, q[2]), (3, q[3], 0, q[1], 0, q[2]))
    return AbstractTensor[replaceinds(site, collect(Index, legs), Index[VX[k] for k in ks])]
end

function _i3_place_pair(pairs, leg, F::NTuple{6, Int}, VX)
    t = _i3_itype(F)
    pr = get(pairs, t, nothing)
    isnothing(pr) && return nothing
    parts = _c3_face_parts(F)
    old = vcat(Index[leg[(:slot, t, G)] for (G, _) in parts], Index[leg[(:w, t)]])
    new = vcat(Index[VX[pk] for (_, pk) in parts], Index[VX[F]])
    return (replaceinds(pr[1], old, new), replaceinds(pr[2], old, new), VX[F])
end

function _i3_canon_block(t, s, bk::NTuple{6, Int}, VX, leg)
    faces = _c3_faces(bk, _I3_L)
    return replaceinds(t, Index[VX[F] for (F, _) in faces], Index[leg[(s, F[1], side)] for (F, side) in faces])
end

function _i3_canon_pair(pr, F::NTuple{6, Int}, VX, leg)
    t = _i3_itype(F)
    parts = _c3_face_parts(F)
    old = vcat(Index[VX[pk] for (_, pk) in parts], Index[VX[F]])
    new = vcat(Index[leg[(:slot, t, G)] for (G, _) in parts], Index[leg[(:w, t)]])
    return (replaceinds(pr[1], old, new), replaceinds(pr[2], old, new))
end

# The virtual box's view of a canonical state: face indices, sites, blocks, and both pair sets.
function _i3_virtual(st::_I3State, site, legs)
    VX = _C3Lazy{NTuple{6, Int}}(F -> new_index(_i3_vdim(F, st.chi, legs); tags = "Link,i3v"))
    tbl = _C3Lazy{NTuple{3, Int}}(q -> _i3_place_site(site, legs, q, VX))
    Bv = _C3Lazy{NTuple{6, Int}}(bk -> _i3_place_block(st, bk, VX))
    Pv = _C3Lazy{NTuple{6, Int}}(F -> _i3_place_pair(st.P, st.leg, F, VX))
    Qv = st.Q === st.P ? Pv : _C3Lazy{NTuple{6, Int}}(F -> _i3_place_pair(st.Q, st.leg, F, VX))
    return VX, tbl, Bv, Pv, Qv
end

# --- seed, iteration, promotion ----------------------------------------------------------------

function _i3_seed_state(ic::InfiniteCTM3D, χ::Int)
    leg = _i3_legs(χ, ic.legs)
    elt = scalartype(ic.site)
    B = Dict{NTuple{3, Int}, Any}()
    for s in _i3_types()
        t = nothing
        for (F, side) in _c3_faces(_i3_centre_key(s), _I3_L)
            l = leg[(s, F[1], side)]
            v = if _c3_israw(F)
                bnd = isnothing(ic.boundary) ? ones(elt, dim(l)) : convert(Vector{elt}, collect(ic.boundary))
                length(bnd) == dim(l) || throw(ArgumentError("boundary has length $(length(bnd)), the vertex leg $(dim(l))"))
                adapt_like(ic.site, from_array(bnd, l))
            else
                adapt_like(ic.site, onehot(elt, l => 1))
            end
            t = isnothing(t) ? v : t * v
        end
        B[s] = t
    end
    return _I3State(χ, B, Dict{NTuple{3, Int}, Any}(), Dict{NTuple{3, Int}, Any}(), leg)
end

# One iteration: every pair from the current blocks, then the centre shell regrown with them.
function _i3_step(ic::InfiniteCTM3D, st::_I3State, exact::Bool)
    opts = ic.options; Lv = _I3_L
    VX, tbl, Bv, Pv, Qv = _i3_virtual(st, ic.site, ic.legs)
    Sv = CTM3DEnvironments(Bv, Pv, VX, st.chi, Qv)
    reps = _i3_representatives()
    Qn = Dict{NTuple{6, Int}, Any}()
    lk = ReentrantLock()
    _ctm_foreach(eachindex(reps)) do i
        pr = _c3_pair_cut(reps[i][2], Sv, tbl, Lv, opts, !exact, Qv)
        isnothing(pr) || lock(() -> (Qn[reps[i][2]] = pr), lk)
    end
    Pn = Qn
    if opts.projector === :cycle && !exact
        Pn = copy(Qn)
        for (F, pr) in _c3_cycle_cube((4, 4, 4), Sv, tbl, Lv, opts, Qn)
            isnothing(pr) || (Pn[F] = pr)
        end
    end
    all(((t, F),) -> haskey(Pn, F), reps) || error("3D iCTM: an interface type got no pair")
    newP = Dict{NTuple{3, Int}, Any}(t => _i3_canon_pair(Pn[F], F, VX, st.leg) for (t, F) in reps)
    newQ = Pn === Qn ? newP :
        Dict{NTuple{3, Int}, Any}(t => _i3_canon_pair(Qn[F], F, VX, st.leg) for (t, F) in reps)
    Pplace = _C3Lazy{NTuple{6, Int}}(F -> _i3_place_pair(newP, st.leg, F, VX))
    types = _i3_types()
    nb = Vector{Any}(nothing, length(types))
    _ctm_foreach(eachindex(types)) do i
        s = types[i]; bk = _i3_centre_key(s)
        extras = Any[]
        for (F, side) in _c3_faces(bk, Lv)
            _c3_israw(F) || push!(extras, _c3_side_proj(Pplace[F], side))
        end
        blk = _ctm_rescale(_ctm_absorb(opts, _c3_enlarged(Bv, tbl, bk), extras...))
        nb[i] = _i3_canon_block(blk, s, bk, VX, st.leg)
    end
    B = Dict{NTuple{3, Int}, Any}(types[i] => nb[i] for i in eachindex(types))
    return _I3State(st.chi, B, newP, newQ, st.leg)
end

# Re-express a state on wider canonical legs (zero padding): every contraction is unchanged.
function _i3_promote(st::_I3State, χ::Int, legs)
    leg = _i3_legs(χ, legs)
    function lift(t, pairs_old_new)
        for (o, n) in pairs_old_new
            o in inds(t) || continue
            if dim(o) == dim(n)
                t = replaceind(t, o, n)
            else
                M = zeros(dim(o), dim(n))
                for j in 1:dim(o)
                    M[j, j] = 1
                end
                t = t * adapt_like(t, from_array(M, o, n))
            end
        end
        return t
    end
    on = [(st.leg[k], leg[k]) for k in keys(leg)]
    B = Dict{NTuple{3, Int}, Any}(s => lift(t, on) for (s, t) in st.B)
    P = Dict{NTuple{3, Int}, Any}(t => (lift(pr[1], on), lift(pr[2], on)) for (t, pr) in st.P)
    Q = st.Q === st.P ? P : Dict{NTuple{3, Int}, Any}(t => (lift(pr[1], on), lift(pr[2], on)) for (t, pr) in st.Q)
    return _I3State(χ, B, P, Q, leg)
end

# --- per-site quantities ----------------------------------------------------------------------

const _I3_REGIONS = [(3.0, 3.0, 3.0), (3.5, 3.0, 3.0), (3.0, 3.5, 3.0), (3.0, 3.0, 3.5),
                     (3.5, 3.5, 3.0), (3.5, 3.0, 3.5), (3.0, 3.5, 3.5), (3.5, 3.5, 3.5)]

function _i3_lnkappa(ic::InfiniteCTM3D, st::_I3State)
    VX, tbl, Bv, _, _ = _i3_virtual(st, ic.site, ic.legs)
    env = CTM3DEnvironments(Bv, Dict{NTuple{6, Int}, Any}(), VX, st.chi, Dict{NTuple{6, Int}, Any}())
    vals = zeros(length(_I3_REGIONS))
    _ctm_foreach(eachindex(_I3_REGIONS)) do i
        c = _I3_REGIONS[i]
        ts = _c3_region_blocks(env, c)
        all(isinteger, c) && append!(ts, tbl[_I3_V])
        vals[i] = log(abs(scalar(_c3_contract_region(ts, ic.options))))
    end
    return sum((iseven(count(x -> !isinteger(x), _I3_REGIONS[i])) ? 1 : -1) * vals[i] for i in eachindex(vals))
end

"""
    cvm_freenergy(ic::InfiniteCTM3D)

ln κ, the logarithm of the partition function per site, from the Möbius sum per unit cell.
"""
function cvm_freenergy(ic::InfiniteCTM3D)
    isnothing(ic.state) && error("InfiniteCTM3D has not been `update`d.")
    r = ic.lnkappa
    r[] === nothing && (r[] = _i3_lnkappa(ic, ic.state))
    return r[]
end

"""
    site_ratio(ic::InfiniteCTM3D, impurity; method = :shell)

`⟨impurity⟩ / ⟨site⟩` for a site tensor `impurity` with an observable inserted (same six legs) —
the single-site expectation value, e.g. the magnetisation for [`ising3d_site`](@ref)'s tensors.

* `:shell` — through the vertex's full 26-block environment shell. The most faithful estimator,
  but an exact shell contraction needs ~χ¹⁰–χ¹² memory (docs/ctmrg3d.md): χ ≲ 5.
* `:octant` — the site at the corner of an octant REGROWN from its eight pieces (the site, three
  half-lines, three quarter-planes, the old octant) with the current pairs on its three faces,
  closed by the cube's seven other octants. Costs one block rebuild, ~χ⁷, so it works at any χ
  the iteration does; its outer three bonds pass through the face projectors, so it sees a
  truncation the shell estimator does not.
"""
function site_ratio(ic::InfiniteCTM3D, impurity; method::Symbol = :shell)
    isnothing(ic.state) && error("InfiniteCTM3D has not been `update`d.")
    method in (:shell, :octant) || throw(ArgumentError("method must be :shell or :octant, got $(repr(method))"))
    st = ic.state
    VX, tbl, Bv, Pv, _ = _i3_virtual(st, ic.site, ic.legs)
    imp = _i3_place_site(impurity, ic.legs, _I3_V, VX)
    if method === :shell
        env = CTM3DEnvironments(Bv, Dict{NTuple{6, Int}, Any}(), VX, st.chi, Dict{NTuple{6, Int}, Any}())
        sh = _c3_region_blocks(env, Float64.(_I3_V))
        z0 = scalar(_c3_contract_region(vcat(sh, tbl[_I3_V]), ic.options))
        z1 = scalar(_c3_contract_region(vcat(sh, imp), ic.options))
        return z1 / z0
    end
    # the (−,−,−) octant of cube (4,4,4) has the centre vertex (3,3,3) at its corner
    cube = (4, 4, 4)
    σ0 = (-1, -1, -1)
    okey(σ) = (σ[1], σ[2], σ[3], cube[1], cube[2], cube[3])
    face(σ, a) = (a, cube[a] - 1, σ[_C3_TR[a][1]], cube[_C3_TR[a][1]], σ[_C3_TR[a][2]], cube[_C3_TR[a][2]])
    others = AbstractTensor[Bv[okey(σ)] for σ in ((s1, s2, s3) for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)) if σ != σ0]
    rest = _c3_contract_region(others, ic.options)                    # the seven octants, three faces open
    projs = Any[_c3_side_proj(Pv[face(σ0, a)], :low) for a in 1:3]
    list = _c3_enlarged(Bv, tbl, okey(σ0))
    site_pos = findfirst(t -> t === only(tbl[_I3_V]), list)
    isnothing(site_pos) && error("3D iCTM: the regrown octant does not contain the centre site")
    grown(site) = (l = copy(list); l[site_pos] = site; _ctm_contract(vcat(l, projs), ic.options))
    z0 = scalar(grown(only(tbl[_I3_V])) * rest)
    z1 = scalar(grown(only(imp)) * rest)
    return z1 / z0
end

# Largest change of any block between two states, immune to a block's overall phase; blocks are
# norm-1 and their kept indices fixed and Procrustes-aligned, so they compare directly. The
# convergence signal where ln κ is unaffordable (χ ≳ 5).
function _i3_blockdist(a::_I3State, b::_I3State)
    worst = 0.0
    for (s, ta) in a.B
        tb = b.B[s]
        ov = min(abs(dot(ta, tb)) / (norm(ta) * norm(tb)), 1.0)
        worst = max(worst, sqrt(max(0.0, 2 - 2ov)))
    end
    return worst
end

"""
    update(ic::InfiniteCTM3D; maxiter = 50, tolerance = 1e-10, verbose = false,
           convergence = :lnkappa, seed_iters = 40, seed_tolerance = 1e-9)

Iterate to a fixed point: on a fresh environment, uncompressed iterations at `seed_maxdim` from the
open-boundary vacuum until ln κ moves by less than `seed_tolerance` (at most `seed_iters`), then
compressed iterations at `maxdim` until the signal falls below `tolerance`:

* `:lnkappa` — `|Δ ln κ| ≤ tolerance · max(1, |ln κ|)`. Needs the vertex region every iteration,
  whose exact contraction costs ~χ¹⁰–χ¹² memory: χ ≲ 5 (docs/ctmrg3d.md).
* `:blocks` — the largest phase-free change of any block (`_i3_blockdist`) ≤ `tolerance`; costs
  nothing extra, so it is the signal for larger χ. ln κ is then not computed at all unless
  [`cvm_freenergy`](@ref) asks for it.
"""
function update(ic::InfiniteCTM3D; maxiter::Integer = 50, tolerance::Real = 1.0e-10, verbose::Bool = false,
                convergence::Symbol = :lnkappa, seed_iters::Integer = 40, seed_tolerance::Real = 1.0e-9)
    convergence in (:lnkappa, :blocks) || throw(ArgumentError("convergence must be :lnkappa or :blocks, got $(repr(convergence))"))
    χ = ic.maxdim; χ0 = min(ic.seed_maxdim, χ)
    st = ic.state
    if isnothing(st)
        st = _i3_seed_state(ic, χ0)
        f = NaN
        for it in 1:seed_iters
            st = _i3_step(ic, st, true)
            fn = _i3_lnkappa(ic, st)
            Δ = abs(fn - f); f = fn
            verbose && @info "3D iCTM seed iteration $it (χ = $χ0): ln κ = $f, |Δ| = $Δ"
            it >= 3 && Δ <= seed_tolerance && break
        end
        χ0 < χ && (st = _i3_promote(st, χ, ic.legs))
    end
    exact = ic.plane_rest === :exact
    lk = convergence === :lnkappa
    f = lk ? _i3_lnkappa(ic, st) : NaN
    converged, Δ = false, Inf
    for it in 1:maxiter
        stn = _i3_step(ic, st, exact)
        if lk
            fn = _i3_lnkappa(ic, stn)
            Δ = abs(fn - f); f = fn
            crit = Δ / max(1.0, abs(f))
        else
            Δ = _i3_blockdist(st, stn)
            crit = Δ
        end
        st = stn
        verbose && @info "3D iCTM iteration $it (χ = $χ): " * (lk ? "ln κ = $f, |Δ| = $Δ" : "block change $Δ")
        if it >= 2 && crit <= tolerance
            converged = true
            break
        end
    end
    converged || @warn "3D iCTM did not converge to tolerance $tolerance after $maxiter iterations " *
        "(final $(lk ? "|Δ ln κ|" : "block change") = $Δ)."
    out = _i3_setstate(ic, st)
    lk && (out.lnkappa[] = f)
    return out
end

"""
    ising3d_site(β; J = (1.0, 1.0, 1.0), h = 0.0) -> (site, legs, magnetisation)

The cubic-lattice Ising model's site tensor for [`InfiniteCTM3D`](@ref): spins on the vertices,
weight `exp(β Σ J_a σσ′ + β h Σ σ)`, each bond's Boltzmann matrix split symmetrically between its
two sites. `J = (Jx, Jy, Jz) ≥ 0`. Returns the site tensor, its legs `(x⁻, x⁺, y⁻, y⁺, z⁻, z⁺)`
and the impurity tensor with σ inserted (for [`site_ratio`](@ref)).
"""
function ising3d_site(β::Real; J = (1.0, 1.0, 1.0), h::Real = 0.0)
    legs = Tuple(new_index(2; tags = "i3,$n") for n in ("xm", "xp", "ym", "yp", "zm", "zp"))
    function sqrtW(K)
        K >= 0 || throw(ArgumentError("ising3d_site takes ferromagnetic couplings, got βJ = $K"))
        λ1, λ2 = cosh(K), sinh(K)
        α, ϕ = (sqrt(λ1) + sqrt(λ2)) / 2, (sqrt(λ1) - sqrt(λ2)) / 2
        return sqrt(2) * [α ϕ; ϕ α]
    end
    function tensor(sgn)
        A = zeros(ntuple(_ -> 2, 6))
        A[ones(Int, 6)...] = exp(β * h)
        A[fill(2, 6)...] = sgn * exp(-β * h)
        t = from_array(A, legs...)
        for (n, l) in enumerate(legs)
            m = sqrtW(β * J[cld(n, 2)])
            t = replaceind(from_array(m, l, prime(l)) * t, prime(l), l)
        end
        return t
    end
    return tensor(1), legs, tensor(-1)
end
