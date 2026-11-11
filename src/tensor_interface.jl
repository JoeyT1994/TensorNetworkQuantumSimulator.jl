#=
TensorInterface: the seam between the package and its tensor backend. Every tensor verb is a
generic function owned here; a backend (src/Backends/) adds methods for its own types, and
src/Backends/Backends.jl selects it (currently ITensorBaseBackend: ITensorBase named tensors over
dense arrays or GradedArrays block-sparse arrays, i.e. abelian symmetries and fermions).

Rules:
  1. Tensor-level code reaches the tensor library only through this module; network-level code
     calls ITensorNetworksNext directly.
  2. Functions the package also extends for its own types are seam functions too: `contract`,
     `truncate`, `inner`, `uniqueinds`, `datatype`, `scalartype`.
  3. Base/LinearAlgebra generics (`qr`, `svd`, `eigen`, `factorize`, `norm`, `normalize`, `dot`,
     `tr`, `*`, `+`, `-`, `copy`, `eltype`, `conj`, `ndims`, `exp`, `sum`, `adapt_structure`) follow
     the ITensors signatures. Factorizations return the factors first and may append more, so
     call sites destructure (`_, R = qr(t, l)`) rather than index.
  4. Indices are identified by id and carry a dimension, prime level, tags and a dual transform
     (`dag`). `A * B` contracts all `commoninds(A, B)`.
  5. Two adjoints: `dag(t)` is the bra of `t` (used site by site to build a bra layer);
     `dag(t, cod)` is the adjoint of `t` as a map with codomain legs `cod`, so that
     `dag(U, cod) * U` is the identity for an isometry `U`; passing `cod` or its complement gives
     the same tensor. They agree on dense and bosonic data but differ by parity twists for
     fermions, so use the right one. `gram(a, b, legs) = dag(a, legs) * b`.
  6. `Tensor` and `Index` (bound in Backends.jl) are used only in annotations; new indices come
     from `new_index`, new tensors from the construction verbs.
  7. Verbs have no fallbacks. Hooks (`site_op`, `site_state`) are optional: their fallback returns
     `nothing` and the caller takes the generic path.
=#
module TensorInterface

# Algorithm selection
"""
    Algorithm{Alg, Kwargs}

Lightweight dispatch token for algorithm selection: `Algorithm("bp"; maxiter = 5)` creates
`Algorithm{:bp}` carrying its keyword arguments; `Algorithm"bp"` is the type for use in
method signatures.
"""
struct Algorithm{Alg, Kwargs <: NamedTuple}
    kwargs::Kwargs
end
Algorithm{Alg}(kwargs::NamedTuple) where {Alg} = Algorithm{Alg, typeof(kwargs)}(kwargs)
Algorithm(alg::Union{String, Symbol}; kwargs...) = Algorithm{Symbol(alg)}(values(kwargs))
Algorithm(alg::Algorithm) = alg

macro Algorithm_str(s)
    return :(Algorithm{$(QuoteNode(Symbol(s)))})
end

function Base.show(io::IO, alg::Algorithm{Alg}) where {Alg}
    print(io, "Algorithm\"", Alg, "\"(", alg.kwargs, ")")
    return nothing
end

# Seam verbs, with ITensors semantics unless noted:
# - `isgraded(x)`: whether `x` carries symmetry sectors (dense: false).
# - `new_index(d; tags)`: a fresh index.
# - `from_array(A, inds)`, `from_array(A, outs, ins)`: tensor with (a copy of) data `A` on the
#   legs in order; the second form is a map `ins -> outs` (graded backends keep only the
#   symmetry-allowed blocks).
# - `dense(t)`: structured (diagonal) storage made explicit.
# - `directsum(a => ia, b => ib; tags)`: direct sum along paired legs, on a new summed bond;
#   returns the sum and that bond.
# - `contract(ts; sequence)`: `sequence = nothing` lets the backend choose. `apply(o, t)` is
#   `noprime(o * t)`.
# - `factorize_svd(t, linds; ...)`: truncated SVD split (L, R, spec); `truncated_svd(t, linds;
#   trunc)`: U, S, V truncated by a `TruncationStrategy`.
# Graded data needs a few verbs that are trivial on dense data:
# - `projector(elt, i => v)`: ⟨v| on site `i` (dense: `onehot`; graded: on the dual copy, with a
#   dim-1 "Charge" leg carrying the state's charge).
# - `charge_sectors(is)`: the dim-1 legs completing a tensor on `is` to a flux-zero one (dense: one
#   trivial leg; graded: one per reachable sector).
# - `pad_index(w, ins, t)`: a zero-padding index with the sector structure the interface fixes, or
#   `nothing` when none is needed.
# - `gram(a, b, legs)`: `dag(a, legs) * b`, the partial Hilbert inner product over `legs`.
# - `data(t)`: raw storage as a vector (a copy on block-sparse data); `scale!(t, x)` scales in place.
# - `has_closure_gauge(T)`: whether multi-vertex closures can carry a parity-gauge sign (fermions),
#   so that additive quantities from closures must be normalised by a baseline closure.
for f in [
        # index queries
        :inds, :commonind, :commoninds, :uniqueinds, :unioninds, :noncommonind,
        :noncommoninds, :hascommoninds, :dim, :plev, :tags, :isgraded,
        # index/tensor transforms
        :dag, :prime, :noprime, :sim, :replaceind, :replaceinds,
        # construction
        :new_index, :from_array, :random_tensor, :onehot, :delta, :dense,
        :combiner, :combinedind, :directsum, :projector, :charge_sectors, :pad_index,
        # contraction / evaluation
        :contract, :scalar, :apply, :inner,
        # diagonal ops
        :map_diag, :map_diag!,
        # sesquilinear partial inner product
        :gram,
        # factorizations (beyond the LinearAlgebra generics)
        :factorize_svd, :truncated_svd,
        # storage / type queries
        :datatype, :array, :data, :scale!,
    ]
    @eval function $f end
end

# A seam-owned `truncate` would clash with Base's export, so backends extend `Base.truncate`.
const truncate = Base.truncate

# Plain numbers and arrays, for type-promotion utilities.
function scalartype end
scalartype(x) = scalartype(typeof(x))
scalartype(T::Type{<:Number}) = T
scalartype(::Type{<:AbstractArray{T}}) where {T} = T
datatype(A::AbstractArray) = typeof(A)

# Some backends return full contractions as raw numbers.
scalar(x::Number) = x

# Dense data: the projector onto a basis state is the unit vector.
projector(p::Pair) = projector(Float64, p)
projector(elt::Type, p::Pair) = onehot(elt, p)

has_closure_gauge(::Type) = false
has_closure_gauge(x) = has_closure_gauge(typeof(x))

# Truncation strategies
"""
    TruncationStrategy(maxdim, rtol, degtol)

Seam-owned truncation rule for `truncated_svd` (built with [`truncation_strategy`](@ref)): keep at
most `maxdim` singular values, drop those at or below `rtol · s₁`, and back off rather than split a
multiplet whose gap is `≤ degtol · s_k`. Backends apply it to the merged spectrum of all sectors;
[`kept_count`](@ref) is the rule on one sorted spectrum.
"""
struct TruncationStrategy
    maxdim::Int
    rtol::Float64
    degtol::Float64
end
truncation_strategy(; maxdim::Integer, rtol::Real = 0.0, degtol::Real = 0.0) =
    TruncationStrategy(Int(maxdim), Float64(rtol), Float64(degtol))

"""
    kept_count(s, trunc::TruncationStrategy)

How many of the singular values `s` (sorted by decreasing magnitude) `trunc` keeps; at least one.
"""
function kept_count(s::AbstractVector{<:Real}, st::TruncationStrategy)
    n = length(s)
    k = min(st.maxdim, n)
    while k > 1 && s[k] ≤ st.rtol * s[1]
        k -= 1
    end
    while k > 1 && k < n && abs(s[k] - s[k + 1]) ≤ st.degtol * abs(s[k])
        k -= 1
    end
    return k
end

# Hooks (rule 7)
"""
    site_op(name, sites...; kwargs...)

Optional backend operator library, consulted by `op` before the package's own: the operator `name`
on `sites` (e.g. fermionic operators on fermionic sites), or `nothing`.
"""
site_op(name, sites...; kwargs...) = nothing

"""
    site_state(name, site)

Optional backend state library, consulted by `state` before the package's own, or `nothing`.
"""
site_state(name, site) = nothing

export Algorithm, @Algorithm_str

end
