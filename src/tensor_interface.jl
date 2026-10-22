#=
TensorInterface: the seam between the package and its tensor backend. Every tensor verb is a
generic function owned here; a backend (src/Backends/) adds methods for its own types, and
src/Backends/Backends.jl selects it (currently ITensorsBackend, dense storage).

Rules:
  1. Nothing outside src/Backends/ references a tensor library.
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
  7. Verbs have no fallbacks. Hooks (`fused_norm_message`, `fused_norm_scalar`)
     are optional: their fallback returns `nothing` and the caller takes the generic path.
=#
module TensorInterface

# ── Algorithm-selection machinery ───────────────────────────────────────────────────────
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

# ── Seam verbs ──────────────────────────────────────────────────────────────────────────
# Contracts (ITensors semantics unless noted):
# - `isgraded(x)`, `isdual(i)`: symmetry sectors / arrow of an index (dense: false).
# - `space(i)`: legs with equal `space` are interchangeable (dense: the dimension).
# - `new_index(d; tags)`, `new_index(ref, d; tags)`: a fresh index; `ref` lets graded backends
#   choose sectors compatible with those legs.
# - `pad_index(w, ins, kt; tags)`: index padding bond `w` to width `kt` (`nothing` if already
#   wide enough); its sectors depend only on `ins` and `kt`.
# - `from_array(A, inds)`, `from_array(A, outs, ins)`: tensor with (a copy of) data `A` on the
#   legs in order; the second form is a map `ins -> outs` (graded backends keep only the
#   symmetry-allowed blocks).
# - `dense(t)`: structured storage (diagonal, combiner) made explicit.
# - `directsum(a => ia, b => ib; tags)`: direct sum along paired legs, on a new summed bond.
# - `charge_sectors(legs)`: the dim-1 legs completing a tensor on `legs` to flux zero.
# - `contract(ts; sequence)`: `sequence = nothing` lets the backend choose. `apply(o, t)` is
#   `noprime(o * t)`.
# - `factorize_svd(t, linds; ...)`: truncated SVD split (L, R, spec); `truncated_svd(t, linds;
#   trunc)`: U, S, V truncated by a `TruncationStrategy`.
# - `data(t)`: raw storage (a copy on block-sparse data); mutate with `scale!(t, x)`.
for f in [
        # index queries
        :inds, :commonind, :commoninds, :uniqueinds, :unioninds, :noncommonind,
        :noncommoninds, :hascommoninds, :dim, :plev, :tags, :isgraded, :isdual, :space,
        # index/tensor transforms
        :dag, :prime, :noprime, :sim, :replaceind, :replaceinds,
        # construction
        :new_index, :pad_index, :from_array, :random_tensor, :onehot, :delta, :dense,
        :combiner, :combinedind, :directsum, :charge_sectors,
        # contraction / evaluation
        :contract, :scalar, :apply, :inner,
        # diagonal ops
        :map_diag, :map_diag!,
        # factorizations (beyond the LinearAlgebra generics)
        :factorize_svd, :truncated_svd, :gram,
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

# ── Truncation strategies ───────────────────────────────────────────────────────────────
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

# ── Hooks (rule 7) ─────────────────────────────────────────────────────────────────────
"""
    fused_norm_message(ψ, sinds, incoming; normalize = true)

Optional fused double-layer BP kernel: the outgoing message on the one bond of `ψ` not covered by
the `incoming` (b, b′) messages, i.e. `contract([incoming; ψ; ψ̄])` with ket and bra sharing the
site indices `sinds` and the bra's virtual legs primed — without forming the double-layer tensor.
With `normalize`, the result has unit 2-norm. `nothing` where the backend has no fused path.
"""
fused_norm_message(ψ, sinds, incoming; normalize = true) = nothing

"""
    fused_norm_scalar(ψ, sinds, incoming, ops)

Optional fused single-vertex closure: the scalar `contract([incoming; ψ; ops; ψ̄])`, where `ops`
are (s′, s) operators on site legs `s ∈ sinds` (empty for the norm). `nothing` where the backend
has no fused path.
"""
fused_norm_scalar(ψ, sinds, incoming, ops) = nothing

export Algorithm, @Algorithm_str

end
