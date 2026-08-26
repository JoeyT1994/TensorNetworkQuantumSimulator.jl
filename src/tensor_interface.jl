#=
TensorInterface: the single seam between TensorNetworkQuantumSimulator and the underlying
tensor library.

Every tensor-level verb the package uses is owned by this module and forwarded, generically,
to the current backend (ITensors.jl). A new backend (e.g. a named-index wrapper around a
TensorKit.jl TensorMap) is added by implementing methods of these functions for its own
tensor/index types — no call site in src/ should ever reference a tensor library directly.

The rules:
  1. Tensor verbs come only from here. `ITensors.foo(...)` must not appear outside this file,
     with one sanctioned exception: `Apply/gate_definitions.jl` is the dense-backend operator
     library and may extend `ITensors.op` / reference `ITensors.SiteType` directly.
  2. Functions this package extends with its own methods (on network/cache types) are seam
     functions too: `contract`, `truncate`, `inner`, `uniqueinds`, `datatype`, `scalartype`.
     They are brought into the package namespace with `import` so `function contract(...)`
     adds methods here.
  3. Base/stdlib generics are part of the interface but need no seam function: a backend
     implements `LinearAlgebra.qr/svd/eigen/factorize/norm/tr/diag/normalize/rmul!`,
     `Base.:*` (contraction of two tensors), `Base.:+/-`, `Base.copy/eltype/conj/isempty`,
     and `Adapt.adapt_structure` for its types. The ITensor-specific keyword conventions
     (`factorize(t, linds; ortho, cutoff, maxdim, tags)`, `qr(t, linds)`,
     `eigen(t, linds, rinds; ishermitian)`) are the required signatures.
  4. The `Index` model a backend must provide: named (id-identified) indices carrying a
     dimension, a prime level (`prime`/`noprime`/`plev`), tags (`tags`, cosmetic only), and
     a dual transform (`dag`; trivial for dense, arrow/dual-space reversal with symmetries).
     Index identity (not position) drives contraction: `A * B` contracts all
     `commoninds(A, B)`.

What a backend must implement, by group:

  Index construction : Index(dim::Integer, tags::String), sim, dag, prime, noprime
  Index queries      : inds, dim, plev, tags, commonind(s), uniqueinds, unioninds,
                       noncommonind(s), hascommoninds, hasqns
  Index replacement  : replaceind(s) (tensor-level relabeling, no data movement)
  Construction       : ITensor (scalar / from array & inds), random_itensor, onehot, delta,
                       dense, denseblocks, combiner (+ combinedind), directsum,
                       op(name::String, siteinds...), state(name::String, siteind)
  Contraction        : contract(ts::Vector; sequence), Base.:*, scalar, apply
  Diagonal ops       : map_diag, map_diag!
  Factorizations     : the LinearAlgebra generics of rule 3, factorize_svd, truncate
  Storage/type       : datatype, scalartype, array, data (raw storage vector, mutable view)
  Misc               : disable_warn_order (may be a no-op)

Allocation contract (the point of the exercise): backends should route contraction through
kernels that can (eventually) take caller-provided workspaces, fuse `dag` into the GEMM
(conjugation via BLAS flags, never a materialized conj copy), and use in-place
factorizations (e.g. MatrixAlgebraKit.jl) where the output can be preallocated.
=#
module TensorInterface

using ITensors: ITensors
using ITensors.NDTensors: NDTensors

# ── Types and dispatch machinery (re-exported as-is, for now) ──────────────────────────
# `ITensor`/`Index` are concrete aliases today; they become abstract types or unions when a
# second backend lands (struct fields and annotations in src/ will be loosened then).
const ITensor = ITensors.ITensor
const Index = ITensors.Index

using ITensors: Algorithm, @Algorithm_str, OpName, @OpName_str, SiteType, @SiteType_str

# ── Seam verbs, forwarded generically to ITensors ──────────────────────────────────────
# A second backend adds methods for its own types; these varargs fallbacks keep dense
# ITensors working untouched.
const _FORWARDED_VERBS = [
    # index queries
    :inds, :commonind, :commoninds, :uniqueinds, :unioninds, :noncommonind,
    :noncommoninds, :hascommoninds, :dim, :plev, :tags, :hasqns,
    # index/tensor transforms
    :dag, :prime, :noprime, :sim, :replaceind, :replaceinds,
    # construction
    :onehot, :delta, :dense, :denseblocks, :combiner, :combinedind, :random_itensor,
    :directsum, :op, :state,
    # contraction / evaluation
    :contract, :scalar, :apply, :inner,
    # diagonal ops
    :map_diag, :map_diag!,
    # factorizations (beyond the LinearAlgebra generics)
    :factorize_svd,
    # storage / type queries
    :datatype, :array, :data,
    # misc
    :disable_warn_order,
]

for f in _FORWARDED_VERBS
    @eval $f(args...; kwargs...) = ITensors.$f(args...; kwargs...)
end

scalartype(args...) = NDTensors.scalartype(args...)

# Backend-context construction verbs (no direct ITensors forward): `ref` supplies the
# backend — an index or tensor of the network being extended. Fallback = ITensors backend.
new_index(ref, d::Integer; tags = "") = ITensors.Index(d, string(tags))
from_array(A::AbstractArray, is...) = ITensors.ITensor(A, is...)

# `truncate` must stay the Base function (ITensors already extends `Base.truncate` for its
# tensor types, and consumers get `truncate` from Base implicitly) — a seam-owned function
# would clash with Base's export at every `using` site. Backends and this package extend
# `Base.truncate` with methods for their own types.
const truncate = Base.truncate

export ITensor, Index, Algorithm, @Algorithm_str, OpName, @OpName_str, SiteType, @SiteType_str

end
