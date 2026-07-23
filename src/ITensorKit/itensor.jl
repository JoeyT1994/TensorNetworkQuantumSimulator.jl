# The `ITensorMap` wrapper.
#
# Pairs a TensorKit `TensorMap` with a tuple of `Index` labels (position → label,
# codomain legs first) so that ITensor-style *identity-based* legs work on top of
# TensorKit's *positional* legs. The wrapper invariant is
#
#     inds[k].space == space(data, k)   for every leg k
#
# where `space(data, k)` is the *effective* space (TensorKit dualizes domain legs).
# Relabels touch only the tuple and share storage; `dag` additionally takes a lazy
# adjoint. Contraction (`*`) dispatches to TensorOperations' dynamic `ncon`.
#
# See dev/tensorkit_index_system_design.md.

export ITensorMap, inds, replaceind, replaceinds, swapind

struct ITensorMap{S, N, T <: AbstractTensorMap{<:Any, S}}
    data::T
    inds::NTuple{N, Index{S}}

    # Checked inner constructor: enforces the wrapper invariant
    # `inds[k].space == space(data, k)` (and the leg count).
    function ITensorMap{S, N, T}(
            data::T, inds::NTuple{N, Index{S}}
        ) where {S, N, T <: AbstractTensorMap}
        numind(data) == N ||
            throw(ArgumentError("tensor has $(numind(data)) legs, but got $N indices"))
        for k in 1:N
            space(data, k) == inds[k].space || throw(
                ArgumentError(
                    "index $k has space $(inds[k].space), but tensor leg $k has space $(space(data, k))"
                )
            )
        end
        return new{S, N, T}(data, inds)
    end

    # Unchecked constructor that bypasses the invariant. For hot paths (`contract`,
    # relabels) where the invariant holds by construction. Deduces `S`/`T` from
    # `data`, so an empty `inds` (scalar result) is still well-typed.
    global function unsafe_itensormap(data::AbstractTensorMap, inds)
        tup = Tuple(inds)
        return new{spacetype(data), length(tup), typeof(data)}(data, tup)
    end
end

# Checked public constructor: deduce type parameters from `data`, then validate via
# the inner constructor.
function ITensorMap(data::AbstractTensorMap, inds)
    tup = Tuple(inds)
    length(tup) == numind(data) || throw(ArgumentError(
        "expected $(numind(data)) indices for a $(numind(data))-leg tensor, got $(length(tup))"
    ))
    return ITensorMap{spacetype(data), numind(data), typeof(data)}(data, tup)
end

# --- accessors (delegate to `data`) -----------------------------------------
"""
    inds(t::ITensorMap) -> NTuple{N,Index}

The labels of `t`'s legs, in position order (codomain legs first).
"""
function inds(t::ITensorMap; plev = nothing)
    isnothing(plev) && return t.inds
    pl = plev
    return filter(i -> i.plev == pl, t.inds)
end

TensorKit.space(t::ITensorMap, k::Int) = space(t.data, k)
TensorKit.scalartype(::Type{<:ITensorMap{S, N, T}}) where {S, N, T} = scalartype(T)
TensorKit.scalartype(t::ITensorMap) = scalartype(t.data)
Base.eltype(t::ITensorMap) = scalartype(t)
Base.size(t::ITensorMap) = map(dim, t.inds)
Base.size(t::ITensorMap, k::Integer) = dim(t.inds[k])
Base.ndims(t::ITensorMap) = numind(t)
# Sum of all tensor entries (used e.g. to normalize belief-propagation messages).
Base.sum(t::ITensorMap) = sum(array(t))
TensorKit.storagetype(t::ITensorMap) = storagetype(t.data)
TensorKit.spacetype(::Type{<:ITensorMap{S}}) where {S} = S
TensorKit.spacetype(t::ITensorMap) = spacetype(t.data)
TensorKit.numind(t::ITensorMap) = numind(t.data)
TensorKit.numout(t::ITensorMap) = numout(t.data)
TensorKit.numin(t::ITensorMap) = numin(t.data)

Base.copy(t::ITensorMap) = unsafe_itensormap(copy(t.data), t.inds)

# --- index-set algebra forwarding -------------------------------------------
# Forward to the `index.jl` helpers on `inds(t)`. Required: the generic methods
# would otherwise try to iterate the tensor itself.
# Forward to the `index.jl` helpers, treating an `ITensorMap` argument as its
# `inds`. Covers an `ITensorMap` in either position (the rest may be an index
# iterable or another tensor), mirroring ITensor's `commoninds(A, B)` etc.
for f in (
        :commoninds, :commonind, :uniqueinds, :uniqueind,
        :noncommoninds, :noncommonind, :unioninds, :hascommoninds,
    )
    @eval $f(A::ITensorMap, B::ITensorMap) = $f(inds(A), inds(B))
    @eval $f(A::ITensorMap, B) = $f(inds(A), B)
    @eval $f(A, B::ITensorMap) = $f(A, inds(B))
end
hasind(t::ITensorMap, i::Index) = hasind(inds(t), i)
# `index in tensor` means the tensor carries that leg (ITensor semantics).
Base.in(i::Index, t::ITensorMap) = hasind(t, i)

# --- relabel operations (share data unless noted) ---------------------------
"""
    prime(t::ITensorMap[, inc::Integer])
    prime(t::ITensorMap, is::Index...)

Raise the prime level of all legs (by `inc`, default 1) or only of the legs matching
the given indices `is`. Shares the underlying data.
"""
prime(t::ITensorMap) = unsafe_itensormap(t.data, map(prime, t.inds))
prime(t::ITensorMap, inc::Integer) = unsafe_itensormap(t.data, map(i -> prime(i, inc), t.inds))
function prime(t::ITensorMap, i1::Index, is::Index...)
    targets = (i1, is...)
    return unsafe_itensormap(t.data, map(j -> hasind(targets, j) ? prime(j) : j, t.inds))
end

"""
    noprime(t::ITensorMap)

Reset the prime level of every leg to 0. Shares the underlying data.
"""
noprime(t::ITensorMap) = unsafe_itensormap(t.data, map(noprime, t.inds))

"""
    setprime(t::ITensorMap, pl::Integer)

Set the prime level of every leg to `pl`. Shares the underlying data.
"""
setprime(t::ITensorMap, pl::Integer) = unsafe_itensormap(t.data, map(i -> setprime(i, pl), t.inds))

"""
    sim(t::ITensorMap)

Replace every leg label with a fresh `id` (same `plev`/`space`), breaking index
identity so `t` no longer shares legs with its origin. Shares the underlying data.
"""
sim(t::ITensorMap) = unsafe_itensormap(t.data, map(sim, t.inds))

"""
    dag(t::ITensorMap)

The dual of `t`: every leg's `space` is dualized and the data is conjugated, with
codomain and domain swapped. Implemented as a lazy `adjoint` of the data — zero-copy.
`dag(dag(t)) == t`.
"""
function dag(t::ITensorMap)
    N₁ = numout(t.data)
    daginds = map(dag, t.inds)
    # adjoint reorders legs to (domain..., codomain...); match that here
    newinds = (daginds[(N₁ + 1):end]..., daginds[1:N₁]...)
    return unsafe_itensormap(adjoint(t.data), newinds)
end

"""
    replaceind(t::ITensorMap, old::Index, new::Index)
    replaceinds(t::ITensorMap, olds, news)

Relabel the leg(s) matching `old`/`olds` with `new`/`news`, keeping the data. Each
replacement requires `space(new) == space(old)` (dimension-adapting replacement is
not yet supported). Following ITensor, indices in `olds` that are not legs of `t`
are silently skipped.
"""
function replaceind(t::ITensorMap, old::Index, new::Index)
    pos = findfirst(==(old), t.inds)
    pos === nothing && return t              # absent leg: no-op (ITensor semantics)
    space(new) == space(old) || throw(
        ArgumentError(
            "replaceind requires equal spaces (got $(space(old)) -> $(space(new)))"
        )
    )
    return unsafe_itensormap(t.data, Base.setindex(t.inds, new, pos))
end

replaceinds(t::ITensorMap, p::Pair) = replaceinds(t, first(p), last(p))
function replaceinds(t::ITensorMap, olds, news)
    length(olds) == length(news) ||
        throw(ArgumentError("replaceinds: olds and news must have equal length"))
    newinds = t.inds
    for (o, n) in zip(olds, news)
        pos = findfirst(==(o), newinds)
        pos === nothing && continue          # absent leg: skip (ITensor semantics)
        space(n) == space(o) ||
            throw(ArgumentError("replaceinds requires equal spaces (got $(space(o)) -> $(space(n)))"))
        newinds = Base.setindex(newinds, n, pos)
    end
    return unsafe_itensormap(t.data, newinds)
end

"""
    swapind(t::ITensorMap, i::Index, j::Index)

Swap the labels of the two legs matching `i` and `j`, keeping the data. Requires
`space(i) == space(j)`.
"""
function swapind(t::ITensorMap, i::Index, j::Index)
    space(i) == space(j) || throw(ArgumentError("swapind requires equal spaces"))
    pi = findfirst(==(i), t.inds)
    pj = findfirst(==(j), t.inds)
    (pi === nothing || pj === nothing) &&
        throw(ArgumentError("swapind: index not found in tensor"))
    return unsafe_itensormap(t.data, Base.setindex(Base.setindex(t.inds, j, pi), i, pj))
end

# --- scalar arithmetic / utility --------------------------------------------
Base.:*(α::Number, t::ITensorMap) = unsafe_itensormap(α * t.data, t.inds)
Base.:*(t::ITensorMap, α::Number) = α * t
LinearAlgebra.norm(t::ITensorMap, p::Real = 2) = norm(t.data, p)
LinearAlgebra.normalize(t::ITensorMap) = unsafe_itensormap(t.data / norm(t.data), t.inds)

# ITensor-style postfix `'` raises the prime level of every leg (NOT conjugation).
Base.adjoint(t::ITensorMap) = prime(t)

# --- display ----------------------------------------------------------------
function Base.show(io::IO, t::ITensorMap)
    print(io, "ITensorMap{", scalartype(t), "} with ", numind(t), " indices:")
    for i in inds(t)
        print(io, "\n  ", i)
    end
    return
end
