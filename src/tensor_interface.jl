#=
TensorInterface: the seam between TensorNetworkQuantumSimulator and the tensor backend.

Every tensor-level verb the package uses is a generic function owned by this module. A backend
(src/Backends/) adds methods for its own tensor and index types; nothing outside src/Backends/
names a tensor library. The active backend is selected in src/Backends/Backends.jl, which also
binds the concrete types `Tensor` and `Index` the rest of the package annotates with. Today that is
ITensorsBackend (ITensors 0.9, dense storage only).

The rules:
  1. Tensor verbs come only from here; no backend library is referenced outside src/Backends/.
  2. Functions this package extends with its own methods (on network/cache types) are seam
     functions too: `contract`, `truncate`, `inner`, `uniqueinds`, `datatype`, `scalartype`.
  3. Base/stdlib generics are part of the interface but need no seam function: a backend
     implements `LinearAlgebra.qr/svd/eigen/factorize/norm/normalize/dot/tr`, `Base.:*`
     (pairwise contraction), `Base.:+/-`, `Base.copy/eltype/conj/ndims/exp/sum`, and
     `Adapt.adapt_structure`. Required signatures follow the ITensors conventions:
       `qr(t, linds)`        -> Q on (linds..., q), R on (q, rest...)
       `svd(t, linds)`       -> U on (linds..., u), S on (u, v), V on (rest..., v), U * S * V ≈ t
       `eigen(t, linds, rinds; ishermitian, cutoff, maxdim)`
       `factorize(t, linds; ortho, cutoff, maxdim, tags)`
     The factors come FIRST in the returned value; a backend may append more (ITensors appends
     the new bonds and the spectrum, and its `svd` result iterates but cannot be indexed), so call
     sites DESTRUCTURE (`_, R = qr(t, l)`), never `last` or `[k]`.
  4. The index model: named (id-identified) indices carrying a dimension, a prime level
     (`prime`/`noprime`/`plev`), tags (`tags`, cosmetic), and a dual transform (`dag`;
     trivial for dense data, arrow/dual-space reversal once spaces are graded). Index
     identity — not position — drives contraction: `A * B` contracts all `commoninds(A, B)`.
  5. Two adjoints. `dag(t)` is the BRA of a tensor: the conjugate that, applied site by site,
     builds the bra layer of a network (a homomorphism over contraction). `dag(t, cod)` is the
     adjoint of `t` viewed as a MAP with codomain legs `cod` (the rest is the domain): the one
     for which `dag(U, cod) * U` is the identity on the bond of an isometry `U` and `U * dag(U,
     cod)` the projector onto its range, e.g. `U = first(svd(B, ins))` → `dag(U, ins)`. On dense
     and bosonic data the two coincide; with fermionic sectors they differ by parity twists on
     the legs of mixed orientation, and using the bra where the map adjoint is meant (or vice
     versa) silently breaks projector identities. Passing either side of the split gives the
     same tensor. `gram(a, b, legs) = dag(a, legs) * b`.
  6. Types. The backend's concrete tensor and index types are bound as `Tensor` and `Index` in
     src/Backends/Backends.jl. The package uses them only in annotations and as container
     element types; it never calls their constructors — new indices come from `new_index`,
     new tensors from the construction verbs below.
  7. Hooks. Seam VERBS have no fallbacks: a missing method is a missing backend feature and
     fails loudly at the call site. Seam HOOKS (`fused_norm_message`, `fused_norm_scalar`,
     `backend_op`) are optional fast paths or extensions: their fallback returns `nothing` and
     the caller then takes its generic path, with identical results.

What a backend must implement, by group (contracts on the declarations below):

  Index construction : new_index(dim; tags), new_index(ref, dim; tags), pad_index(w, ins, kt),
                       sim, dag, prime, noprime
  Index queries      : inds, dim, plev, tags, commonind(s), uniqueinds, unioninds,
                       noncommonind(s), hascommoninds, isgraded, isdual, space
  Index replacement  : replaceind(s) (relabeling, no data movement)
  Construction       : from_array, random_tensor, onehot, delta, dense, combiner, combinedind,
                       directsum, charge_sectors
  Contraction        : contract(ts::Vector; sequence), Base.:*, scalar, apply
  Diagonal ops       : map_diag, map_diag!
  Factorizations     : the LinearAlgebra generics of rule 3, factorize_svd, truncated_svd, gram
  Storage/type       : datatype, scalartype, array, data, scale!
  Hooks (optional)   : fused_norm_message, fused_norm_scalar, backend_op
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
# Generic functions; the backend adds methods for its own types. No fallbacks (rule 7).
#
# Index queries: `inds(t)`, `dim(i)` / `dim(is)` (product), `plev(i)`, `tags(i)`,
# `commonind(s)(a, b)`, `uniqueinds(a, b...)`, `unioninds(a, b)`, `noncommonind(s)(a, b)`,
# `hascommoninds(a, b)` — the ITensors semantics, on tensors and on index collections.
#
# `isgraded(x)`: true when the tensor or index `x` carries symmetry sectors. Dense: false.
# `isdual(i)`: the arrow of a graded index (true on the dual copy). Dense: false.
# `space(i)`: the vector space of an index — two indices with equal `space` (and arrow) are
#   interchangeable as the same leg slot of a tensor. Dense: its dimension.
#
# `new_index(d; tags)`: a fresh index of total dimension `d`. `new_index(ref, d; tags)`: a fresh
#   bond of dimension `d` compatible with the legs `ref` (an index, a vector of them, or a tensor);
#   graded backends choose its sector content, dense ones ignore `ref`.
# `pad_index(w, ins, kt; tags)`: the zero-padding index that brings the retained bond `w` on the
#   interface `ins` to total width `kt`, or `nothing` when `dim(w) >= kt`. Its sector content must
#   depend only on `ins` and `kt` (so a padded bond has the same structure every sweep). Dense: a
#   fresh index of dimension `kt - dim(w)`.
# `sim(x; tags)`: a fresh copy (new identity, same space and arrow) of an index or of each index
#   of a collection, optionally retagged.
#
# `from_array(A, inds)`, `from_array(A, i1, i2, ...)`: the tensor with data `A` on legs `inds`, in
#   that order (`A` is reshaped to their dimensions and copied). `from_array(A, outs, ins)`: the
#   operator form — `A` on (outs..., ins...), viewed as a map from `ins` to `outs`; graded backends
#   project onto the symmetry-allowed blocks. This is the one verb the operator library needs.
# `random_tensor([rng], elt, inds)`: normally distributed entries of type `elt`.
# `onehot([elt], i => k)`: the basis vector k of index i. `delta([elt], inds)`: the copy tensor
#   (identity for two legs), possibly with structured (diagonal) storage — see `dense`.
# `dense(t)`: `t` with any structured storage (e.g. diagonal) made explicit, same symmetry
#   structure. Identity on tensors that are already dense.
# `combiner(inds; tags)`: the tensor fusing `inds` into one combined index, `combinedind(C)`. It
#   may be a structured object that can only fuse (or unfuse) all of `inds` at once; `dense(C)` is
#   the explicit isometry, for contractions that use it as an ordinary tensor.
# `directsum(a => ia, b => ib; tags)`: the direct sum of `a` and `b` along the paired legs `ia`,
#   `ib`, returned as a tensor carrying the newly minted summed bond. `directsum(news, a => ias,
#   b => ibs)`: several paired legs at once, the summed bonds given as `news`.
# `charge_sectors(legs)`: the dim-1 "charge" legs that complete a tensor on `legs` to a flux-zero
#   one: one trivial leg for dense data, one per reachable sector for graded data.
#
# `contract(ts::Vector; sequence)`: contract a list (nested-vector `sequence` addresses tensors by
#   position; `nothing` lets the backend choose). `scalar(t)`: the number of a 0-leg tensor.
#   `apply(o, t)`: `noprime(o * t)`.
#
# `map_diag(f, t)`, `map_diag!(f, out, t)`: apply `f` along the diagonal of a 2-leg tensor.
#
# `factorize_svd(t, linds; ortho, singular_values!, maxdim, cutoff, ...)`: the truncated SVD
#   split (L, R, spec) with `spec.truncerr`; with `ortho = "none"` the singular values are split
#   symmetrically, and a `Ref` passed as `singular_values!` receives the diagonal tensor S.
# `truncated_svd(t, linds; trunc)`: `svd(t, linds)` truncated by the seam-owned
#   `TruncationStrategy` `trunc` (below) — U, S, V on the kept bond.
# `gram(a, b, legs)`: a† b over `legs`, with a and b viewed as maps INTO `legs` (rule 5).
#
# `datatype(t)`: the storage type (Array, CuArray, ...). `scalartype(t)`: the element type.
# `array(t)`, `array(t, i1, i2, ...)`: the data as a dense Array (in that leg order).
# `data(t)`: the raw storage as a vector (a copy on block-sparse data); use `scale!` to mutate.
# `scale!(t, x)`: multiply `t`'s storage by the number `x` in place; returns `t`.
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

# `truncate` must stay the Base function (consumers get `truncate` from Base implicitly) —
# a seam-owned function would clash with Base's export at every `using` site. This package
# and backends extend `Base.truncate` with methods for their own types.
const truncate = Base.truncate

# scalartype/datatype have natural meanings for plain numbers and arrays, used by
# type-promotion utilities.
function scalartype end
scalartype(x) = scalartype(typeof(x))
scalartype(T::Type{<:Number}) = T
scalartype(::Type{<:AbstractArray{T}}) where {T} = T
datatype(A::AbstractArray) = typeof(A)

# Full contractions on some backends produce raw numbers; they flow through scalar unchanged.
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

# ── Hooks (rule 7: `nothing` = not provided, take the generic path) ───────────────────────
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

"""
    backend_op(name, sites...; kwargs...)

Optional backend operator library, consulted for operator names the package's own registry does
not define. `nothing` when the backend has none, or does not know `name` either.
"""
backend_op(name, sites...; kwargs...) = nothing

export Algorithm, @Algorithm_str

end
