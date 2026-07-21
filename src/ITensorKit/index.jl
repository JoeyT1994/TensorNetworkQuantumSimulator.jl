# Labeled-index system for the TensorKit `TensorMap` backend.
#
# `TensorMap` legs are matched by *position*; ITensor matched them by *identity*.
# `Index` restores that identity layer: a persistent `id`, a layer discriminator
# `plev` (which copy of a physical index — ket vs bra — NOT duality), and the
# effective leg `space` (with `isdual` baked in). The contraction/set-algebra match
# key is `(id, plev)`; duality lives entirely in `space`.
#
# See dev/tensorkit_index_system_design.md for the rationale.

export Index, plev, prime, noprime, setprime, sim, dag
export commoninds, commonind, uniqueinds, uniqueind
export noncommoninds, noncommonind, unioninds, hascommoninds, hasind

"""
    Index{S<:ElementarySpace}

A labeled tensor leg: a persistent identity layer over TensorKit's *positional*
`TensorMap` legs. It carries

- `id::UInt64` — persistent identity, drawn from a process-global counter;
- `plev::Int8` — layer discriminator (which copy of a physical index, e.g. ket vs
  bra), *not* duality;
- `space::S` — the effective leg space, with `isdual` baked in.

Two indices are equal — and hence contract and match in the set algebra — iff their
match key `(id, plev)` is equal; duality lives entirely in `space`. Thus
`dag(i) == i` while `sim(i) != i`. See [`prime`](@ref), [`dag`](@ref),
[`sim`](@ref), [`commoninds`](@ref).

# Constructors

    Index(space::ElementarySpace; plev=0)
    Index(dim::Integer; plev=0)

Create a leg with a fresh `id`. A bare `dim` uses a self-dual `CartesianSpace(dim)`,
so a default dense leg never carries an arrow.
"""
struct Index{S<:ElementarySpace}
    id::UInt64    # persistent identity (atomic counter, see `_new_index_id`)
    plev::Int8    # layer discriminator (ket vs bra copy of a site) — NOT duality
    space::S      # effective leg space, with isdual baked in
end

# --- id generation ----------------------------------------------------------
# Process-global atomic counter, seeded once per session with a random UInt64.
# In-process uniqueness is guaranteed (no birthday bound) and `atomic_add!` is
# correct under task migration; the random seed only reduces accidental
# cross-session merge collisions. Index creation is rare relative to contraction,
# so atomic contention is a non-issue.
const _INDEX_ID_COUNTER = Threads.Atomic{UInt64}(rand(Random.RandomDevice(), UInt64))
_new_index_id() = Threads.atomic_add!(_INDEX_ID_COUNTER, UInt64(1))

# --- constructors (draw a fresh id) -----------------------------------------
Index(space::ElementarySpace; plev::Integer=0) = Index(_new_index_id(), Int8(plev), space)

# A bare dimension defaults to a self-dual `CartesianSpace` (ℝ^d): `isdual` is
# always `false`, so a default dense leg never carries a surprising arrow. The
# tensor element type stays free — a `TensorMap` over a `CartesianSpace` may still
# be complex-valued.
Index(dim::Integer; plev::Integer=0) = Index(CartesianSpace(dim); plev)

# --- accessors --------------------------------------------------------------
"""
    plev(i::Index) -> Int8

The prime/layer level of `i` — the discriminator (e.g. ket = 0, bra = 1) that pairs
copies of the same physical index. Distinct from duality, which lives in `space(i)`.
"""
plev(i::Index) = i.plev
TensorKit.space(i::Index) = i.space
TensorKit.dim(i::Index) = dim(i.space)
TensorKit.isdual(i::Index) = isdual(i.space)
TensorKit.spacetype(::Type{<:Index{S}}) where {S} = S
TensorKit.spacetype(i::Index) = spacetype(typeof(i))

# Internal id accessor. Deliberately *not* extending `TensorKit.id` (the identity
# morphism) to avoid overloading an unrelated meaning; the migration does not use
# an external `id(::Index)`.
_id(i::Index) = i.id

# --- equality / hashing (match key `(id, plev)` only) -----------------------
# `space` is treated as a checked invariant, not part of identity. Consequence:
# `dag(i) == i` (bond discovery & operator-sandwich pairing rely on this) while
# `sim(i) != i` (fresh id).
Base.:(==)(a::Index, b::Index) = a.id == b.id && a.plev == b.plev
Base.isequal(a::Index, b::Index) = isequal(a.id, b.id) && isequal(a.plev, b.plev)
Base.hash(i::Index, h::UInt) = hash(i.plev, hash(i.id, h))

# --- unary transforms (relabel only; share no data) -------------------------
"""
    prime(i::Index, inc::Integer=1) -> Index

A copy of `i` with its prime level raised by `inc` (default 1), keeping the same
`id` and `space`. See [`noprime`](@ref), [`setprime`](@ref).
"""
prime(i::Index, inc::Integer=1) = Index(i.id, Int8(i.plev + inc), i.space)

"""
    noprime(i::Index) -> Index

A copy of `i` with prime level reset to `0`, keeping the same `id` and `space`.
"""
noprime(i::Index) = Index(i.id, Int8(0), i.space)

"""
    setprime(i::Index, pl::Integer) -> Index

A copy of `i` with prime level set to `pl`, keeping the same `id` and `space`.
"""
setprime(i::Index, pl::Integer) = Index(i.id, Int8(pl), i.space)

"""
    sim(i::Index) -> Index

A copy of `i` with a fresh `id` (same `plev` and `space`, so `isdual` is preserved).
Use it to break index identity so a leg no longer matches/contracts with its origin.
"""
sim(i::Index) = Index(_new_index_id(), i.plev, i.space)              # fresh id; preserves plev & isdual

"""
    dag(i::Index) -> Index

The dual of `i`: same `id` and `plev`, with `space` dualized (`isdual` flipped).
Because the match key `(id, plev)` is unchanged, `dag(i) == i`, so a leg and its
`dag` contract — this is how bond endpoints and bra/ket pairs are formed.
`dag(dag(i)) == i`. Aliased by `TensorKit.dual`.
"""
dag(i::Index) = Index(i.id, i.plev, dual(i.space))                   # dualize space; keep id/plev
TensorKit.dual(i::Index) = dag(i)

# ITensor-style postfix `'` raises the prime level (it is NOT conjugation here).
Base.adjoint(i::Index) = prime(i)

# Broadcast the unary relabels over a collection of indices (ITensor accepts these
# on index vectors, e.g. `prime.(sinds)` written as `prime(sinds)`).
for f in (:prime, :noprime, :sim, :dag)
    @eval $f(is::AbstractVector{<:Index}) = map($f, is)
end

# --- index-collection set algebra -------------------------------------------
# Operate on any iterable of `Index` (Tuples or Vectors). Matching uses the
# `(id, plev)` equality above, so these are literal set operations. Returned
# indices come from the *first* argument (preserving its `space`/order). Leg
# counts are tiny (≤ ~6), so a linear `in` scan is appropriate.

"""
    hasind(inds, i::Index) -> Bool

Whether the collection `inds` contains an index matching `i` by `(id, plev)`.
"""
hasind(inds, i::Index) = any(==(i), inds)

"""
    hascommoninds(a, b) -> Bool

Whether the index collections `a` and `b` share any index (by match key).
"""
hascommoninds(a, b) = any(in(b), a)

"""
    commoninds(a, b) -> Vector{Index}

The indices of `a` whose match key `(id, plev)` also appears in `b`, taken from `a`
and in its order. `a` and `b` are any iterables of [`Index`](@ref).
"""
commoninds(a, b) = filter(in(b), collect(a))

"""
    uniqueinds(a, b) -> Vector{Index}

The indices of `a` not present in `b` (by match key), in the order of `a`.
"""
uniqueinds(a, b) = filter(!in(b), collect(a))

"""
    noncommoninds(a, b) -> Vector{Index}

The symmetric difference of `a` and `b`: indices unique to `a` followed by those
unique to `b`.
"""
noncommoninds(a, b) = vcat(uniqueinds(a, b), uniqueinds(b, a))

"""
    unioninds(a, b) -> Vector{Index}

All indices of `a` followed by those of `b` not already in `a` (deduplicated by
match key).
"""
unioninds(a, b) = vcat(collect(a), uniqueinds(b, a))

"""
    commonind(a, b) -> Union{Index,Nothing}

The first index of `a` also present in `b`, or `nothing` if they share none. Singular
counterpart of [`commoninds`](@ref).
"""
function commonind(a, b)
    for x in a
        x in b && return x
    end
    return nothing
end

"""
    uniqueind(a, b) -> Union{Index,Nothing}

The first index of `a` not present in `b`, or `nothing` if every index of `a` is in
`b`. Singular counterpart of [`uniqueinds`](@ref).
"""
function uniqueind(a, b)
    for x in a
        x in b || return x
    end
    return nothing
end

"""
    noncommonind(a, b) -> Union{Index,Nothing}

A single index not shared between `a` and `b` — unique to `a` if any, otherwise
unique to `b` — or `nothing` if every index is shared.
"""
function noncommonind(a, b)
    i = uniqueind(a, b)
    isnothing(i) || return i
    return uniqueind(b, a)
end

# --- display ----------------------------------------------------------------
function Base.show(io::IO, i::Index)
    print(io, "Index(id=", i.id, ", plev=", Int(i.plev), ", space=", i.space, ")")
end
