#=
TensorInterface: the seam between TensorNetworkQuantumSimulator and the tensor backend.

Every tensor-level verb the package uses is a generic function owned by this module. The
KTensors module (the sole backend: named-index tensors over dense arrays, TensorOperations
contraction, MatrixAlgebraKit factorizations) implements them for its `KIndex`/`KTensor`
types; a future graded/symmetric backend (TensorKit `TensorMap` data) adds its own methods
here without touching call sites.

The rules:
  1. Tensor verbs come only from here; no backend library is referenced outside its module.
  2. Functions this package extends with its own methods (on network/cache types) are seam
     functions too: `contract`, `truncate`, `inner`, `uniqueinds`, `datatype`, `scalartype`.
  3. Base/stdlib generics are part of the interface but need no seam function: a backend
     implements `LinearAlgebra.qr/svd/eigen/factorize/norm/tr/normalize`, `Base.:*` (pairwise
     contraction), `Base.:+/-`, `Base.copy/eltype/conj`, and `Adapt.adapt_structure`.
     Required signatures follow the historical ITensors conventions:
     `factorize(t, linds; ortho, cutoff, maxdim, tags)`, `qr(t, linds)`,
     `eigen(t, linds, rinds; ishermitian)`.
  4. The index model: named (id-identified) indices carrying a dimension, a prime level
     (`prime`/`noprime`/`plev`), tags (`tags`, cosmetic), and a dual transform (`dag`;
     trivial for dense data, arrow/dual-space reversal once spaces are graded). Index
     identity — not position — drives contraction: `A * B` contracts all `commoninds(A, B)`.

What a backend must implement, by group:

  Index construction : new_index(ref, dim; tags), sim, dag, prime, noprime
  Index queries      : inds, dim, plev, tags, commonind(s), uniqueinds, unioninds,
                       noncommonind(s), hascommoninds
  Index replacement  : replaceind(s) (relabeling, no data movement)
  Construction       : from_array, random_itensor, onehot, delta, combiner (+
                       combinedind), directsum, op(name::String, siteinds...),
                       state(name::String, siteind)
  Contraction        : contract(ts::Vector; sequence), Base.:*, scalar, apply
  Diagonal ops       : map_diag, map_diag!
  Factorizations     : the LinearAlgebra generics of rule 3, factorize_svd
  Storage/type       : datatype, scalartype, array, data (raw storage vector, mutable view)
=#
module TensorInterface

# ── Algorithm-selection machinery (previously from ITensors/NDTensors) ─────────────────
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
# Generic functions; the backend adds methods for its own types. No fallbacks: a missing
# method is a missing backend feature and should fail loudly at the call site.
for f in [
        # index queries
        :inds, :commonind, :commoninds, :uniqueinds, :unioninds, :noncommonind,
        :noncommoninds, :hascommoninds, :dim, :plev, :tags,
        # index/tensor transforms
        :dag, :prime, :noprime, :sim, :replaceind, :replaceinds,
        # construction
        :onehot, :delta, :combiner, :combinedind, :random_itensor,
        :directsum, :op, :state, :new_index, :from_array,
        # contraction / evaluation
        :contract, :scalar, :apply, :inner,
        # diagonal ops
        :map_diag, :map_diag!,
        # factorizations (beyond the LinearAlgebra generics)
        :factorize_svd,
        # storage / type queries
        :datatype, :array, :data,
    ]
    @eval function $f end
end

# `truncate` must stay the Base function (consumers get `truncate` from Base implicitly) —
# a seam-owned function would clash with Base's export at every `using` site. This package
# and backends extend `Base.truncate` with methods for their own types.
const truncate = Base.truncate

# scalartype/datatype have natural meanings for plain numbers and arrays, used by
# type-promotion utilities.
scalartype(x) = scalartype(typeof(x))
scalartype(T::Type{<:Number}) = T
scalartype(::Type{<:AbstractArray{T}}) where {T} = T
datatype(A::AbstractArray) = typeof(A)

export Algorithm, @Algorithm_str

end
