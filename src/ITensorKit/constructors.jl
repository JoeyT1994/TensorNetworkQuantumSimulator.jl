# Tensor construction catalogue for `ITensorMap`.
#
# These mirror the ITensor constructors the TNQS package uses. The convention is
# **all given legs become the codomain** (`V₁⊗…⊗Vₙ ← I`), which keeps the
# `array` ↔ `inds` ordering trivial and makes the per-leg space invariant
# automatic. `dag` later moves legs to the domain as needed.
#
# `ITensor` is exported as an alias of `ITensorMap` so ITensor-style call sites
# (`ITensor(elt, inds...)`, `Vector{ITensor}`, `::ITensor`) port unchanged.

export ITensor, itensor, random_itensor, onehot, delta

# `ITensor` is the public, ITensor-compatible name for the wrapper type.
const ITensor = ITensorMap

# --- space helpers ----------------------------------------------------------
# Codomain product of the leg spaces (the unit object when there are no legs).
_codomain(is) = isempty(is) ? one(_spacetype(is)) : ProductSpace(map(space, Tuple(is))...)
_spacetype(is) = spacetype(first(is))
_homspace(is) = isempty(is) ? (one(_spacetype(is)) ← one(_spacetype(is))) : (_codomain(is) ← one(spacetype(first(is))))

# --- zero / scalar tensors --------------------------------------------------
"""
    ITensor(elt::Type{<:Number}, is::Index...)
    ITensor(is::Index...)

A zero tensor with the given legs (all in the codomain). Element type defaults to
`Float64`.
"""
ITensorMap(elt::Type{<:Number}, i1::Index, is::Index...) = ITensorMap(elt, (i1, is...))
function ITensorMap(elt::Type{<:Number}, is)
    isempty(is) && return ITensorMap(zero(elt))
    return unsafe_itensormap(zeros(elt, _homspace(is)), Tuple(is))
end
ITensorMap(i1::Index, is::Index...) = ITensorMap(Float64, (i1, is...))

"""
    ITensor(x::Number)

A 0-leg tensor wrapping the scalar `x` (over a `CartesianSpace` unit object).
"""
function ITensorMap(x::Number)
    S = CartesianSpace
    data = fill!(zeros(typeof(x), one(S)), x)
    return unsafe_itensormap(data, ())
end

# --- wrap a raw Julia array -------------------------------------------------
"""
    itensor(A::AbstractArray, is)

Wrap the dense array `A` as an `ITensorMap` with legs `is` (all in the codomain).
`A` is reshaped column-major into the leg dimensions, so
`itensor(array(t), inds(t)) == t`.
"""
function itensor(A::AbstractArray, is)
    is = Tuple(is)
    isempty(is) && return ITensorMap(only(A))
    dims = map(dim, is)
    data = TensorMap(reshape(collect(A), dims), _homspace(is))
    return unsafe_itensormap(data, is)
end

# --- random -----------------------------------------------------------------
"""
    random_itensor(elt::Type{<:Number}, is)
    random_itensor(is)

A tensor with the given legs and i.i.d. random entries (`randn`). Element type
defaults to `Float64`.
"""
function random_itensor(elt::Type{<:Number}, is)
    is = Tuple(is)
    isempty(is) && return ITensorMap(randn(elt))
    return unsafe_itensormap(randn(elt, _homspace(is)), is)
end
random_itensor(is) = random_itensor(Float64, is)
random_itensor(elt::Type{<:Number}, i1::Index, is::Index...) = random_itensor(elt, (i1, is...))
random_itensor(i1::Index, is::Index...) = random_itensor(Float64, (i1, is...))

# --- onehot / basis tensor --------------------------------------------------
"""
    onehot(elt::Type{<:Number}, i => k)
    onehot(i => k)

The basis (one-hot) vector: a 1-leg tensor on `i` with a single nonzero entry `1`
at component `k`. Element type defaults to `Float64`.
"""
function onehot(elt::Type{<:Number}, p::Pair{<:Index})
    i, k = first(p), last(p)
    v = zeros(elt, dim(i))
    v[k] = one(elt)
    return itensor(v, (i,))
end
onehot(p::Pair{<:Index}) = onehot(Float64, p)

# --- delta / identity / copy tensor -----------------------------------------
"""
    delta(elt::Type{<:Number}, is::Index...)
    delta(is::Index...)
    delta(is::AbstractVector{<:Index})

The generalized Kronecker-δ (copy) tensor over `is`: entry `1` where all leg
indices coincide, `0` otherwise. For two legs this is an identity matrix; for the
2-leg dual pairing (`delta(i, dag(i)')`) it acts as the identity/trace connector.
All legs must have equal dimension. Element type defaults to `Float64`.
"""
function delta(elt::Type{<:Number}, is)
    is = Tuple(is)
    n = length(is)
    n >= 1 || throw(ArgumentError("delta requires at least one index"))
    d = dim(first(is))
    all(==(d), map(dim, is)) || throw(ArgumentError("delta requires all legs to have equal dimension"))
    A = zeros(elt, ntuple(_ -> d, n))
    for k in 1:d
        A[CartesianIndex(ntuple(_ -> k, n))] = one(elt)
    end
    return itensor(A, is)
end
delta(elt::Type{<:Number}, i1::Index, is::Index...) = delta(elt, (i1, is...))
delta(i1::Index, is::Index...) = delta(Float64, (i1, is...))
delta(is::AbstractVector{<:Index}) = delta(Float64, is)
