#=
Tensors: the tensor engine over ITensorBase.

`Tensor` IS `ITensorBase.ITensor` and `Index` IS `ITensorBase.Index` — a NamedDimsArrays named
tensor over either a plain `Array` (dense) or a GradedArrays `AbelianGradedArray` (symmetric /
fermionic). Everything algebraic is delegated: contraction and permutation (including every
fermionic sign — GradedArrays twists contracted legs and `conj` carries the leg-reversal sign)
to TensorAlgebra/GradedArrays, factorizations to MatrixAlgebraKit through TensorAlgebra's named
wrappers. This file is the TensorInterface seam: label conventions, the operator/state library,
and the handful of graded helpers the generic code asks for through kernel_hooks.jl.

Conventions (mirroring the previous backend so the generic code is unchanged):
  * Index identity is (id, plev) — ITensorBase compares uuid, plev AND tags, so a tag is fixed at
    mint time and never changed on an existing index (`settags` would make a different index).
  * The per-copy dual flag lives in the index's range: `dag(i) = dual(i)`; equality ignores it,
    and a contraction requires the two copies of a bond to carry opposite orientation on graded
    data (a same-orientation pairing errors in GradedArrays; on dense data `dual` is a no-op).
  * `dag(t) = conj(t)`: on graded data this dualises every leg and applies the fermionic sign, so
    `dag(a) * b` is the Hilbert pairing on every backend — `gram` is the same thing.
  * Fully contracted results are 0-dimensional ITensors; `scalar` reads them.
=#
module Tensors

using LinearAlgebra: LinearAlgebra, Diagonal, norm, diag, I
using Random: Random
using MatrixAlgebraKit: MatrixAlgebraKit
using TensorAlgebra: TensorAlgebra
using GradedArrays: GradedArrays
using TensorKitSectors: TensorKitSectors
using ITensorBase: ITensorBase, ITensor, AbstractITensor, inds, prime, noprime, sim,
    commoninds, uniqueinds, noncommoninds, unioninds, hascommoninds, replaceinds, aligndims
import ITensorBase: Index
using VectorInterface: VectorInterface
using Adapt: Adapt, adapt
using UUIDs: UUID
using ..TensorInterface: TensorInterface

const IB = ITensorBase
const TA = TensorAlgebra
const MAK = MatrixAlgebraKit
const GA = GradedArrays
const TKS = TensorKitSectors

export Index, Tensor, AbstractTensor, register_op!, graded_space, new_fermion_index, isgraded

const Tensor = ITensor
const AbstractTensor = AbstractITensor
# The previous backend's positional constructor, `Tensor(inds, data)`, kept for the tests and any
# caller building a tensor from a dense array over dense indices.
ITensorBase.ITensor(is::AbstractVector{<:Index}, data::AbstractArray) = ITensor(data, Tuple(is))

# ── Index ───────────────────────────────────────────────────────────────────────────────

# Our tag strings ("Link,cyc", "Charge", "Site,S=1/2") become one bare ITensorBase tag each
# (key with an empty value); `tags(i)` joins them back with commas.
_totags(s::AbstractString) = Tuple(Symbol(strip(p)) => Symbol("") for p in split(String(s), ",") if !isempty(strip(p)))
_totags(s::Tuple) = s
function _tagstring(i::Index)
    return join((isempty(string(v)) ? string(k) : string(k) * "=" * string(v) for (k, v) in IB.tags(i)), ",")
end

space(i::Index) = IB.space(i)
isgraded(i::Index) = space(i) isa Union{GA.GradedOneTo, GA.SectorOneTo}
isgraded(t::AbstractTensor) = IB.unnamed(t) isa GA.AbstractGradedArray
isgraded(x) = false
isdual(i::Index) = TA.isdual(i)

Index(d::Integer, tags::AbstractString) = Index(Int(d); tags = _totags(tags))
Index(sp::Union{GA.GradedOneTo, GA.SectorOneTo}, tags::AbstractString = "") = Index(sp; tags = _totags(tags))
# Five-field constructor kept for the generic code that mints an index from another's fields:
# `Index(id, space, plev, tags, dual)`.
function Index(id::UInt64, sp, plev::Integer, tags::AbstractString, dual::Bool)
    nm = IB.IndexName(; uuid = UUID(UInt128(id)), tags = _totags(tags), plev = Int(plev))
    r = sp isa Integer ? Base.OneTo(Int(sp)) : sp
    if !(r isa Base.OneTo) && GA.isdual(r) != dual     # `sp` may itself carry an arrow: set it to `dual`
        r = GA.dual(r)
    end
    return IB.named(r, nm)
end
_id(i::Index) = UInt64(UInt128(IB.name(i).uuid) & typemax(UInt64))

TensorInterface.dim(i::Index) = length(i)
TensorInterface.dim(is::AbstractVector{<:Index}) = prod(TensorInterface.dim.(is); init = 1)
TensorInterface.plev(i::Index) = IB.plev(i)
TensorInterface.tags(i::Index) = _tagstring(i)
# `dual` of a plain range is the range itself (dense data has no orientation).
TensorInterface.dag(i::Index) = isgraded(i) ? TA.dual(i) : i
TensorInterface.prime(i::Index, n::Integer = 1) = IB.setplev(i, IB.plev(i) + n)
TensorInterface.noprime(i::Index) = IB.noprime(i)
TensorInterface.sim(i::Index) = sim(i)
# `i'` primes an index throughout the generic code (the ITensors convention). ITensorBase's
# `Index` is an `AbstractUnitRange`, whose `adjoint` would otherwise be a lazy matrix adjoint.
Base.adjoint(i::Index) = TensorInterface.prime(i)
for f in [:dag, :prime, :noprime, :sim]
    @eval TensorInterface.$f(is::AbstractVector{<:Index}, args...) = map(i -> TensorInterface.$f(i, args...), is)
end

# Fresh link of total dimension `d`. Graded: trivial-dominant split (a double layer is
# trivial-sector dominant, so a link needs at least as many trivial states as charged ones).
TensorInterface.new_index(d::Integer; tags = "") = Index(Int(d), String(tags))
function TensorInterface.new_index(ref::Union{Index, AbstractVector{<:Index}}, d::Integer; tags = "")
    r = ref isa Index ? ref : first(ref)
    isgraded(r) || return Index(Int(d), String(tags))
    d0, d1 = cld(Int(d), 2), fld(Int(d), 2)
    S = GA.sectortype(space(r))
    sp = d1 == 0 ? GA.gradedrange([GA.trivial(S) => d0]) :
        GA.gradedrange([GA.trivial(S) => d0, _unit_charge(S) => d1])
    return Index(sp, String(tags))
end
TensorInterface.new_index(t::AbstractTensor, d::Integer; tags = "") = TensorInterface.new_index(collect(inds(t)), d; tags)

# The "charge 1" sector of a sector type, for the fresh-link split above.
_unit_charge(::Type{S}) where {S <: GA.SectorRange} = _unit_charge_label(S)
_unit_charge_label(::Type{GA.SectorRange{TKS.U1Irrep}}) = GA.U1(1)
_unit_charge_label(::Type{GA.SectorRange{TKS.ZNIrrep{N}}}) where {N} = GA.Z{N}(1)
_unit_charge_label(::Type{GA.SectorRange{TKS.FermionParity}}) = GA.SectorRange(TKS.FermionParity(1))
function _unit_charge_label(::Type{GA.SectorRange{P}}) where {P <: TKS.ProductSector}
    T = fieldtype(P, :sectors)
    parts = map(fieldtypes(T)) do F
        F === TKS.FermionParity ? TKS.FermionParity(1) : F === TKS.U1Irrep ? TKS.U1Irrep(1) : F(1)
    end
    return GA.SectorRange(P(parts))
end

# ── Tensor basics ───────────────────────────────────────────────────────────────────────

Base.ndims(t::AbstractTensor) = length(inds(t))
TensorInterface.inds(t::AbstractTensor; plev = nothing) =
    plev === nothing ? collect(Index, inds(t)) : filter(i -> IB.plev(i) == plev, collect(Index, inds(t)))
TensorInterface.scalartype(t::AbstractTensor) = eltype(t)
TensorInterface.datatype(t::AbstractTensor) = typeof(IB.unnamed(t))
TensorInterface.array(t::AbstractTensor) = Array(IB.unnamed(t))
function TensorInterface.array(t::AbstractTensor, is::Index...)
    length(is) == ndims(t) || error("array: expected $(ndims(t)) indices, got $(length(is))")
    return Array(IB.unnamed(aligndims(t, is)))
end
# Raw storage as a vector: dense data flat; graded data the stored blocks concatenated.
function TensorInterface.data(t::AbstractTensor)
    A = IB.unnamed(t)
    isgraded(t) || return vec(A)
    return _storedvec(A)
end
function _storedvec(A::GA.AbstractGradedArray)
    return reduce(vcat, (vec(collect(view(A, I))) for I in GA.eachblockstoredindex(A)); init = eltype(A)[])
end
# In-place scaling of the storage (the `data` vector of a graded tensor is a copy, so `rmul!` on
# it would be lost): dense arrays directly, graded ones block by block.
function TensorInterface.scale!(t::AbstractTensor, c::Number)
    A = IB.unnamed(t)
    if isgraded(t)
        for I in GA.eachblockstoredindex(A)
            view(A, I) .*= c
        end
    else
        A .*= c
    end
    return t
end

# Fully-projected graded networks contract down to spectator dim-1 charge legs rather than a
# bare number; the entry is the amplitude (zero when the total charge is wrong).
function TensorInterface.scalar(t::AbstractTensor)
    ndims(t) == 0 && return t[]
    all(i -> length(i) == 1, inds(t)) || error("scalar: tensor with inds $(inds(t)) is not a scalar")
    return sum(Array(IB.unnamed(t)))
end

TensorInterface.prime(t::AbstractTensor, n::Integer = 1) = IB.mapinds(i -> TensorInterface.prime(i, n), t)
TensorInterface.prime(t::AbstractTensor, is::Index...) = TensorInterface.prime(t, 1, is...)
TensorInterface.prime(t::AbstractTensor, n::Integer, is::Index...) =
    IB.mapinds(i -> i ∈ is ? TensorInterface.prime(i, n) : i, t)
TensorInterface.noprime(t::AbstractTensor) = IB.mapinds(TensorInterface.noprime, t)
TensorInterface.sim(t::AbstractTensor) = sim(t)
# The bra of a ket tensor. On graded data `conj` dualises every leg and is a homomorphism over
# contraction, so a bra network built site by site is `conj` of the ket network. The one
# correction is on dangling DUAL legs (the dim-1 "Charge" legs of charged states): closing such a
# leg against its conjugate gives the supertrace θ = −1 on odd fermion parity, so the bra carries
# the compensating twist there and closed bra–ket networks evaluate to the Hilbert inner product
# (checked: `norm_sqr` of an odd-parity product state is +1, in any contraction order).
function TensorInterface.dag(t::AbstractTensor)
    c = conj(t)
    isgraded(t) || return c
    dims = Tuple(k for (k, i) in enumerate(inds(t)) if isdual(i) && occursin("Charge", _tagstring(i)))
    isempty(dims) || GA.twist!(IB.unnamed(c), dims)
    return c
end
Base.real(t::AbstractTensor) = ndims(t) == 0 ? real(t[]) : ITensor(real(IB.unnamed(t)), Tuple(inds(t)))
Base.imag(t::AbstractTensor) = ndims(t) == 0 ? imag(t[]) : ITensor(imag(IB.unnamed(t)), Tuple(inds(t)))

# The adjoint of `t` viewed as a MAP with codomain legs `cod` (every other leg is the domain).
# GradedArrays contracts in the supertrace convention, where the identity inserted on a dual leg is
# the parity-twisted identity, so the adjoint that makes `t * dag(t, cod)` (over `cod`) the
# projector onto `t`'s range — and `dag(U, cod) * U` the identity on the bond of an isometry —
# is `conj` followed by a twist on the dual legs in `cod` and the non-dual legs outside it (the two
# descriptions of the same set of legs, up to the trivial twist of a flux-zero block, so passing
# either side of the split gives the same tensor). `dag(t)` is the special case cod = the non-dual
# legs, i.e. plain `conj`, which is a homomorphism over contraction and hence the right bra for a
# network; the adjoint of an isometry is NOT that (its bond leg is dual) — hence this verb.
# Bosonic sectors: no twists, so `dag(t, cod) == dag(t)`.
function TensorInterface.dag(t::AbstractTensor, cod)
    c = conj(t)
    isgraded(t) || return c
    cv = _indvec(cod)
    dims = Tuple(k for (k, i) in enumerate(inds(t)) if (any(==(i), cv) ? isdual(i) : !isdual(i)))
    isempty(dims) || GA.twist!(IB.unnamed(c), dims)
    return c
end
TensorInterface.apply(o::AbstractTensor, t::AbstractTensor) = TensorInterface.noprime(o * t)
# a† b over `legs`, with a and b viewed as maps INTO `legs`: the map adjoint (see `dag(t, cod)`).
TensorInterface.gram(a::AbstractTensor, b::AbstractTensor, legs) = TensorInterface.dag(a, legs) * b

# Relabel by identity. The slot keeps ITS orientation: generic code passes replacement labels
# with arbitrary flags, so the new label is dualised to match the old copy on this tensor.
function TensorInterface.replaceinds(t::AbstractTensor, old, new)
    oldv, newv = _indvec(old), _indvec(new)
    length(oldv) == length(newv) || error("replaceinds: length mismatch")
    pairs = Pair{Index, Index}[]
    for (o, n) in zip(oldv, newv)
        k = findfirst(==(o), inds(t))
        k === nothing && continue
        oc = inds(t)[k]
        length(oc) == length(n) || error("replaceinds: dimension mismatch $(oc) → $(n)")
        nn = (isgraded(oc) && isdual(oc) != isdual(n)) ? TA.dual(n) : n
        push!(pairs, oc => nn)
    end
    isempty(pairs) && return t
    return replaceinds(t, pairs...)
end
TensorInterface.replaceind(t::AbstractTensor, old::Index, new::Index) = TensorInterface.replaceinds(t, [old], [new])
TensorInterface.replaceinds(t::AbstractTensor, p::Pair) = TensorInterface.replaceinds(t, first(p), last(p))

_indvec(i::Index) = Index[i]
_indvec(is::AbstractVector) = collect(Index, is)
_indvec(is::Tuple) = collect(Index, is)
_indvec(t::AbstractTensor) = collect(Index, inds(t))

const IndsLike = Union{AbstractTensor, Index, AbstractVector{<:Index}, Tuple{Index, Vararg{Index}}}
TensorInterface.commoninds(a::IndsLike, b::IndsLike) = filter(i -> i ∈ _indvec(b), _indvec(a))
function TensorInterface.commonind(a::IndsLike, b::IndsLike)
    cs = TensorInterface.commoninds(a, b)
    return isempty(cs) ? nothing : first(cs)
end
TensorInterface.uniqueinds(a::IndsLike, b::IndsLike) = filter(i -> i ∉ _indvec(b), _indvec(a))
TensorInterface.unioninds(a::IndsLike, b::IndsLike) = unique(vcat(_indvec(a), _indvec(b)))
TensorInterface.noncommoninds(a::IndsLike, b::IndsLike) =
    vcat(TensorInterface.uniqueinds(a, b), TensorInterface.uniqueinds(b, a))
function TensorInterface.noncommonind(a::IndsLike, b::IndsLike)
    ns = TensorInterface.noncommoninds(a, b)
    return isempty(ns) ? nothing : first(ns)
end
TensorInterface.hascommoninds(a::IndsLike, b::IndsLike) = !isempty(TensorInterface.commoninds(a, b))

# Present entries of `linds` on `t` (absent ones are ignored, the seam convention).
_present_inds(t::AbstractTensor, linds) = filter(i -> i ∈ inds(t), _indvec(linds))
# The copy of `i` as `t` carries it (same identity, `t`'s orientation).
_ascarried(t::AbstractTensor, i::Index) = inds(t)[findfirst(==(i), inds(t))]
_ascarried(t::AbstractTensor, is::AbstractVector) = Index[_ascarried(t, i) for i in is]

# ── Construction ────────────────────────────────────────────────────────────────────────

# Dense array into an (optionally graded) tensor: graded legs are projected onto the
# symmetry-allowed blocks — the array must live in them.
function TensorInterface.from_array(A::AbstractArray, is::Index...)
    isv = collect(Index, is)
    B = reshape(copy(A), TensorInterface.dim.(isv)...)
    any(isgraded, isv) || return ITensor(B, Tuple(isv))
    return TA.project(B, Tuple(isv), ())
end

function TensorInterface.random_tensor(rng::Random.AbstractRNG, elt::Type, is::AbstractVector{<:Index})
    isempty(is) && return ITensor(fill(randn(rng, elt)), ())
    return randn(rng, elt, Tuple(collect(Index, is))...)
end
TensorInterface.random_tensor(rng::Random.AbstractRNG, elt::Type, is::Index...) =
    TensorInterface.random_tensor(rng, elt, collect(Index, is))
TensorInterface.random_tensor(elt::Type, is::AbstractVector{<:Index}) =
    TensorInterface.random_tensor(Random.default_rng(), elt, is)
TensorInterface.random_tensor(elt::Type, is::Index...) = TensorInterface.random_tensor(elt, collect(is))
TensorInterface.random_tensor(is::AbstractVector{<:Index}) = TensorInterface.random_tensor(Float64, is)
TensorInterface.random_tensor(is::Index...) = TensorInterface.random_tensor(Float64, collect(is))

# The dim-1 legs completing a tensor on `is` to a flux-zero one: dense data has no flux
# (one trivial leg); graded data one dual charge leg per sector the legs can fuse to.
function TensorInterface.charge_sectors(is::AbstractVector{<:Index})
    any(isgraded, is) || return Index[Index(1, "Charge")]
    fused = length(is) == 1 ? GA.tensor_product(space(only(is))) :
        reduce(GA.tensor_product, (space(i) for i in is))
    return Index[Index(GA.gradedrange([GA.dual(c) => 1]), "Charge") for c in GA.sectors(fused)]
end

# Basis state / projector onto a basis state of one site. Graded sites: the vector is
# projected onto the site's grading; a charged state gets a dim-1 dangling "Charge" leg that
# absorbs its charge (`tryproject_aux`), so every tensor stays flux-zero.
function TensorInterface.onehot(elt::Type, p::Pair{<:Index, <:Integer})
    i, v = p
    data = zeros(elt, length(i))
    data[v] = one(elt)
    return _vector_tensor(data, i)
end
TensorInterface.onehot(p::Pair{<:Index, <:Integer}) = TensorInterface.onehot(Float64, p)
function TensorInterface.projector(elt::Type, p::Pair{<:Index, <:Integer})
    i, v = p
    isgraded(i) || return TensorInterface.onehot(elt, p)
    isdual(i) && error("projector: expected a non-dual (ket) site index")
    data = zeros(elt, length(i))
    data[v] = one(elt)
    return _vector_tensor(data, TA.dual(i))
end
function _vector_tensor(data::AbstractVector, i::Index)
    isgraded(i) || return ITensor(data, (i,))
    p = TA.tryproject(data, (i,), ())
    p === nothing || return p
    q = TA.tryproject_aux(data, (i,), ())
    q === nothing && error("state: vector has no consistent charge on $(i)")
    aux = last(inds(q))
    return replaceinds(q, aux => _retag(aux, "Charge"))
end
# A fresh copy of `i`'s range under a new tag (identity changes — only for indices nobody
# else holds yet).
_retag(i::Index, tags::AbstractString) = Index(space(i); tags = _totags(tags))

# Identity between paired indices. Two indices: the identity map between them (on graded data
# they must carry opposite orientations); longer lists pair up by identity.
function TensorInterface.delta(elt::Type, is::AbstractVector{<:Index})
    isempty(is) && return ITensor(fill(one(elt)), ())
    length(is) == 2 && return _delta_pair(elt, is[1], is[2])
    if all(!isgraded, is)
        data = zeros(elt, TensorInterface.dim.(is)...)
        for k in 1:minimum(TensorInterface.dim.(is))
            data[ntuple(_ -> k, length(is))...] = one(elt)
        end
        return ITensor(data, Tuple(is))
    end
    ids = unique([_id(i) for i in is])
    parts = map(ids) do id
        pair = filter(i -> _id(i) == id, is)
        length(pair) == 2 || error("delta: graded delta needs indices in same-space pairs")
        _delta_pair(elt, pair[1], pair[2])
    end
    return reduce(*, parts)
end
function _delta_pair(elt::Type, i1::Index, i2::Index)
    if !isgraded(i1) && !isgraded(i2)
        d = min(length(i1), length(i2))
        data = zeros(elt, length(i1), length(i2))
        for k in 1:d
            data[k, k] = one(elt)
        end
        return ITensor(data, (i1, i2))
    end
    isdual(i1) != isdual(i2) || error("delta: paired graded indices must have opposite orientations")
    # `id(elt, codomain, domain)` dualises the domain itself, so hand it non-dual copies and
    # the result carries the orientations asked for.
    a, b = isdual(i1) ? (i2, i1) : (i1, i2)
    t = IB.id(elt, (a,), (TA.dual(b),))
    return t
end
TensorInterface.delta(is::AbstractVector{<:Index}) = TensorInterface.delta(Float64, is)
TensorInterface.delta(elt::Type, is::Index...) = TensorInterface.delta(elt, collect(is))
TensorInterface.delta(is::Index...) = TensorInterface.delta(Float64, collect(is))

# Combiner: an explicit reshape isometry between the combined index (first) and the product
# of the combined indices. `t * C` combines, multiplying by `C` again splits.
function TensorInterface.combiner(is::AbstractVector{<:Index}; tags = "CMB,Link")
    isempty(is) && error("combiner: no indices to combine")
    if any(isgraded, is)
        # The fusion isometry: the identity on the fused space with its domain leg split back into
        # `is`. The split legs come out dualised, so `t * C` fuses the legs of `t`; the combined
        # leg (first) is non-dual. Splitting back is `(t * C) * dag(C, [c])` — the map adjoint,
        # not `conj`: with mixed-orientation legs the pair `(C, conj(C))` is not an identity
        # insertion on this backend (fermionic twists).
        all(isgraded, is) || error("combiner: cannot mix graded and dense indices")
        elt = Float64
        fused = reduce(GA.tensor_product, (space(i) for i in is))
        c = IB.named(fused, IB.IndexName(; tags = _totags(String(tags))))
        c2 = sim(c)
        I2 = IB.id(elt, (c,), (c2,))                       # (c, dual c2)
        I2 = ITensor(GA.FusedGradedMatrix(IB.unnamed(I2)), Tuple(inds(I2)))
        return TA.unmatricize(I2, c => (c,), inds(I2)[2] => Tuple(is))
    end
    D = prod(TensorInterface.dim.(is))
    c = Index(D, String(tags))
    data = reshape(Matrix{Float64}(I, D, D), (D, TensorInterface.dim.(is)...))
    return ITensor(data, (c, is...))
end
TensorInterface.combiner(is::Index...; kwargs...) = TensorInterface.combiner(collect(Index, is); kwargs...)
TensorInterface.combinedind(C::AbstractTensor) = first(inds(C))

# Direct sum along paired index axes; the minted bond comes back on the tensor.
function TensorInterface.directsum(news::AbstractVector{<:Index}, p1::Pair{<:AbstractTensor}, p2::Pair{<:AbstractTensor})
    t1, o1 = first(p1), _indvec(last(p1))
    t2, o2 = first(p2), _indvec(last(p2))
    length(news) == length(o1) == length(o2) || error("directsum: length mismatch")
    s, outs = TA.directsum(t1 => Tuple(_ascarried(t1, o1)), t2 => Tuple(_ascarried(t2, o2)))
    return TensorInterface.replaceinds(s, collect(Index, outs), news)
end
function TensorInterface.directsum(p1::Pair{<:AbstractTensor}, p2::Pair{<:AbstractTensor}; tags = "Link,sum")
    t1, o1 = first(p1), only(_indvec(last(p1)))
    t2, o2 = first(p2), only(_indvec(last(p2)))
    s, outs = TA.directsum(t1 => (_ascarried(t1, o1),), t2 => (_ascarried(t2, o2),))
    w = only(outs)
    wt = IB.named(space(w), IB.IndexName(; tags = _totags(tags)))
    return replaceinds(s, w => wt)
end

# ── Arithmetic, contraction ─────────────────────────────────────────────────────────────

Base.isapprox(a::AbstractTensor, b::AbstractTensor; atol = 0, rtol = nothing) =
    (rt = rtol === nothing ? sqrt(eps(real(promote_type(eltype(a), eltype(b))))) : rtol;
     norm(a - b) <= max(atol, rt * max(norm(a), norm(b))))

# Trace of a 2-leg (s, s′) tensor, defined to agree with sum(diag(array(t))) — the convention the
# sampler reads its diagonal in (on graded data the pairwise `tr` would apply the supertrace twist).
LinearAlgebra.tr(t::AbstractTensor) = ndims(t) == 2 ? sum(diag(TensorInterface.array(t))) : _trace_all(t)
# Contract each plev-1 index with its plev-0 partner (same id): the operator trace.
function _trace_all(t::AbstractTensor)
    lo = filter(i -> IB.plev(i) == 0, collect(Index, inds(t)))
    hi = Index[]
    for i in lo
        j = findfirst(x -> IB.plev(x) == 1 && _id(x) == _id(i), collect(Index, inds(t)))
        j === nothing && error("tr: no primed partner for index $(i)")
        push!(hi, inds(t)[j])
    end
    return LinearAlgebra.tr(t, Tuple(hi), Tuple(lo))
end

# Cached-sequence tree walk. Abstractly typed lists route to the concrete method.
function TensorInterface.contract(ts::Vector; sequence = nothing, kwargs...)
    all(t -> t isa AbstractTensor, ts) || error("contract: expected a tensor list, got $(unique(typeof.(ts)))")
    return TensorInterface.contract(collect(AbstractTensor, ts); sequence, kwargs...)
end
function TensorInterface.contract(ts::Vector{<:AbstractTensor}; sequence = nothing, dest = nothing, kwargs...)
    isnothing(sequence) && return reduce(*, ts)
    sequence isa Integer && return ts[sequence]
    # `dest`: a tensor the caller has finished with, offered as storage for the result. Honoured for
    # a two-factor contraction whose result fits it and does not alias either operand (else the
    # result goes to a pooled buffer and is copied into `dest`); longer trees ignore it.
    if length(ts) == 2 && dest isa AbstractTensor
        r = _contract_into(ts[1], ts[2], dest)
        r === nothing || return r
    end
    return _contract_seq(ts, sequence)
end
_contract_seq(ts::Vector, x::Integer) = ts[x]
_contract_seq(ts::Vector, x::Union{Vector, Tuple}) = mapreduce(y -> _contract_seq(ts, y), *, x)

# ── Consumed destinations (simple update) ───────────────────────────────────────────────
# Generic ITensorBase operations (`mul!` into a named destination), arranged so that the result of a
# step lands in storage the caller has relinquished instead of fresh memory. Dense strided data only;
# anything else takes the ordinary allocating path with identical results.

_dense_storage(t::AbstractTensor) = (A = IB.unnamed(t); (!isgraded(t) && A isa StridedArray) ? A : nothing)
_aliases(a::AbstractArray, b::AbstractArray) = Base.mightalias(a, b)

# The result indices of `a * b` (a's open legs, then b's) and their dimensions.
function _product_inds(a::AbstractTensor, b::AbstractTensor)
    ia, ib = collect(Index, inds(a)), collect(Index, inds(b))
    open_a = filter(i -> !(i in ib), ia); open_b = filter(i -> !(i in ia), ib)
    return vcat(open_a, open_b)
end

# A tensor on `is` viewing the first prod(dims) entries of `storage` (a Vector or Array).
function _view_tensor(storage::StridedArray, is::Vector{<:Index})
    n = prod(TensorInterface.dim.(is); init = 1)
    v = reshape(view(vec(storage), 1:n), TensorInterface.dim.(is)...)
    return ITensor(v, Tuple(is))
end

# `a * b` written into `dest`'s storage when it fits and aliases neither operand. Otherwise into a
# pooled buffer, then copied into `dest` (a memcpy, so the caller's storage still ends up holding
# the result). Returns a tensor over `dest`'s storage, or nothing when the types do not allow it.
function _contract_into(a::AbstractTensor, b::AbstractTensor, dest::AbstractTensor)
    A, B, Dst = _dense_storage(a), _dense_storage(b), _dense_storage(dest)
    (A === nothing || B === nothing || Dst === nothing) && return nothing
    T = promote_type(eltype(A), eltype(B))
    eltype(Dst) == T || return nothing
    out = _product_inds(a, b)
    n = prod(TensorInterface.dim.(out); init = 1)
    n <= length(Dst) || return nothing
    if _aliases(Dst, A) || _aliases(Dst, B)
        buf = first(_fused_buffers(T, n))
        tmp = _view_tensor(buf, out)
        LinearAlgebra.mul!(tmp, a, b)
        r = _view_tensor(Dst, out)
        copyto!(IB.unnamed(r), IB.unnamed(tmp))
        return r
    end
    r = _view_tensor(Dst, out)
    LinearAlgebra.mul!(r, a, b)
    return r
end

# Environment chain t · e₁ · e₂ ⋯ (shape-preserving 2-index factors), the result in `dest`'s storage
# (or fresh when `dest` is nothing). Intermediates ping-pong between one pooled buffer and `dest`'s
# storage once `t`'s data is no longer needed, so the chain costs t + one buffer live instead of one
# fresh tensor per factor.
function absorb_chain(t::AbstractTensor, envs::Vector, dest)
    isempty(envs) && return t
    A = _dense_storage(t)
    ok = A !== nothing && all(e -> _dense_storage(e) !== nothing, envs) &&
        (dest === nothing || _dense_storage(dest) !== nothing)
    ok || return reduce(*, envs; init = t)
    T = promote_type(eltype(A), (eltype(IB.unnamed(e)) for e in envs)...)
    n = length(A)
    Dst = dest === nothing ? nothing : _dense_storage(dest)
    (Dst !== nothing && (eltype(Dst) != T || length(Dst) < n)) && (Dst = nothing)
    buf1, buf2 = _fused_buffers(T, n)
    # slots: the pooled buffer, and dest's storage once the input has been read (when dest aliases
    # t, that is after the first step; otherwise immediately). With no dest, the second slot is the
    # other pooled buffer and the result is copied out fresh at the end.
    slots = Dst === nothing ? (buf1, buf2) : (buf1, Dst)
    cur = t
    k = 0
    for e in envs
        k += 1
        out_inds = _product_inds(cur, e)
        # Lay the result out with the leg the NEXT factor contracts last, so that step matricises
        # by a free reshape instead of a permuted copy of the whole intermediate (measured: a two-
        # factor chain drops from 3 F to 1 F of transient allocation).
        if k < length(envs)
            nxt = envs[k + 1]
            j = findfirst(i -> i in inds(nxt), out_inds)
            j === nothing || (out_inds = vcat(out_inds[1:(j - 1)], out_inds[(j + 1):end], out_inds[j:j]))
        end
        target = slots[mod1(k, 2)]
        if target === Dst && k == 1 && _aliases(Dst, A)
            target = buf2                                    # first step may not overwrite the input
        end
        r = _view_tensor(target, out_inds)
        LinearAlgebra.mul!(r, cur, e)
        cur = r
    end
    if Dst === nothing
        return ITensor(copy(IB.unnamed(cur)), Tuple(inds(cur)))
    end
    if !_aliases(IB.unnamed(cur), Dst)
        r = _view_tensor(Dst, collect(Index, inds(cur)))
        copyto!(IB.unnamed(r), IB.unnamed(cur))
        cur = r
    end
    return cur
end

# Left-orthogonalisation Q·R with Q on `linds`. Consuming: the matricised copy is factorised in place
# by LAPACK (geqrf + or/ungqr — Q overwrites the workspace, no second F-sized array), and when the
# input's storage is separate from that workspace, Q is placed into it. Live memory: input + one
# F-sized workspace. Non-consuming or non-BLAS data: the seam's `qr`.
function left_orthogonalize(t, linds; consume_input::Bool = false)
    A = _dense_storage(t)
    consume_input && A !== nothing && eltype(A) <: LinearAlgebra.BlasFloat || return LinearAlgebra.qr(t, linds)
    li = _ascarried(t, _present_inds(t, linds))
    ri = TensorInterface.uniqueinds(t, li)
    isempty(li) && return LinearAlgebra.qr(t, linds)
    is = collect(Index, inds(t))
    perm = [findfirst(==(i), is) for i in vcat(li, ri)]
    dl = prod(TensorInterface.dim.(li)); dr = prod(TensorInterface.dim.(ri); init = 1)
    W = perm == 1:length(is) ? copy(A) : permutedims(A, perm)      # the workspace (one F)
    M = reshape(W, dl, dr)
    k = min(dl, dr)
    R = similar(M, k, dr)
    # Q (dl × k) written straight into the consumed input's storage when that is separate from the
    # workspace and large enough; otherwise a fresh matrix. MAK's in-place QR destroys `M`.
    Q = (dl * k <= length(A) && !_aliases(A, W)) ? reshape(view(vec(A), 1:(dl * k)), dl, k) : similar(M, dl, k)
    MAK.qr_compact!(M, (Q, R))
    b = Index(k, "Link,qr")
    Qt = ITensor(reshape(Q, (TensorInterface.dim.(li)..., k)), Tuple(vcat(li, [b])))
    Rt = ITensor(reshape(R, (k, TensorInterface.dim.(ri)...)), Tuple(vcat([b], ri)))
    return Qt, Rt
end
_release_storage!(x) = nothing

# ── Diagonal operations ─────────────────────────────────────────────────────────────────

# `t` is a 2-index (diagonal) tensor; f is applied along its diagonal. Scalar indexing over the
# O(χ) diagonal — on graded data the diagonal of a sector-matched square matrix lies inside
# the allowed blocks.
# `out` is `t` or a copy of it (same off-diagonal content); only the diagonal is touched, which on
# graded data stays inside the allowed blocks of a sector-matched square matrix.
function TensorInterface.map_diag!(f::Function, out::AbstractTensor, t::AbstractTensor)
    ndims(t) == 2 || error("map_diag: expected a 2-index tensor")
    A = IB.unnamed(t)
    O = IB.unnamed(out)
    for k in 1:minimum(size(O))
        O[k, k] = f(A[k, k])
    end
    return out
end
function TensorInterface.map_diag(f::Function, t::AbstractTensor)
    out = copy(t)
    TensorInterface.map_diag!(f, out, out)
    return out
end

# ── Adapt / storage ─────────────────────────────────────────────────────────────────────

Adapt.adapt_structure(elt::Type{<:Number}, t::AbstractTensor) =
    eltype(t) === elt ? t : ITensor(convert(AbstractArray{elt}, IB.unnamed(t)), Tuple(inds(t)))
function Adapt.adapt_structure(to::Type{<:AbstractArray}, t::AbstractTensor)
    isgraded(t) && return t          # graded storage stays where it is
    A = IB.unnamed(t)
    B = adapt(to, A)
    return B === A ? t : ITensor(B, Tuple(inds(t)))
end
Adapt.adapt_structure(::Type{<:GA.AbstractGradedArray}, t::AbstractTensor) = t
Adapt.adapt_structure(to, t::AbstractTensor) = ITensor(adapt(to, IB.unnamed(t)), Tuple(inds(t)))

# ── Factorizations (MatrixAlgebraKit through ITensorBase's named wrappers) ───────────────

struct Spectrum
    truncerr::Float64
end

_tagged(tags::AbstractString) = (; tags = _totags(tags))

# Seam-convention truncation: keep the smallest set of singular values whose discarded Σs²
# fraction is ≤ cutoff (⇔ 2-norm rtol = √cutoff), capped at maxdim.
function _mak_trunc(; maxdim = nothing, cutoff = nothing)
    strategies = Any[]
    isnothing(maxdim) || push!(strategies, MAK.truncrank(Int(maxdim)))
    isnothing(cutoff) || iszero(cutoff) || push!(strategies, MAK.truncerror(; rtol = sqrt(cutoff), p = 2))
    return isempty(strategies) ? nothing : reduce(&, strategies)
end

# Rank / relative-cutoff / degeneracy truncation: keep at most `maxdim` values, drop those at or
# below `rtol·s₁`, and back off rather than split a multiplet whose gap is ≤ degtol·|s_k|. On
# graded data the rule runs over the spectrum of ALL sectors merged, so the retained bond gets
# one count per sector. Used by the CTMRG projectors.
struct RankGapTruncation <: MAK.TruncationStrategy
    maxdim::Int
    rtol::Float64
    degtol::Float64
end
TensorInterface.truncation_strategy(; maxdim::Integer, rtol::Real = 0.0, degtol::Real = 0.0) =
    RankGapTruncation(Int(maxdim), Float64(rtol), Float64(degtol))
function _rankgap_count(s::AbstractVector{<:Real}, st::RankGapTruncation)
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
function MAK.findtruncated(values::AbstractVector, st::RankGapTruncation)
    vals = Array(values)
    perm = sortperm(vals; by = abs, rev = true)
    return perm[1:_rankgap_count(abs.(vals[perm]), st)]
end
MAK.findtruncated_svd(values::AbstractVector, st::RankGapTruncation) = MAK.findtruncated(values, st)
# Graded spectra arrive as one vector per sector; decide on the merged list (each entry weighted
# once — abelian sectors have dimension 1) and hand back per-sector kept positions.
function MAK.findtruncated(v::GA.FusedGradedVector, st::RankGapTruncation)
    entries = [(abs(val), i, j) for (i, b) in enumerate(v.blocks) for (j, val) in enumerate(b)]
    sort!(entries; by = first, rev = true)
    k = _rankgap_count([e[1] for e in entries], st)
    kept = [Int[] for _ in v.blocks]
    for (_, i, j) in entries[1:k]
        push!(kept[i], j)
    end
    sort!.(kept)
    return kept
end
MAK.findtruncated_svd(v::GA.FusedGradedVector, st::RankGapTruncation) = MAK.findtruncated(v, st)

# QR: Q on (linds…, b), R on (b, rinds…); the bond is tagged "Link,qr".
function LinearAlgebra.qr(t::AbstractTensor, linds; kwargs...)
    li = _ascarried(t, _present_inds(t, linds))
    ri = TensorInterface.uniqueinds(t, li)
    Q, R = MAK.qr_compact(t, Tuple(li), Tuple(ri); name = _tagged("Link,qr"))
    return Q, R
end

function _svd_split(t::AbstractTensor, linds; maxdim = nothing, cutoff = nothing, trunc = nothing)
    li = _ascarried(t, _present_inds(t, linds))
    ri = TensorInterface.uniqueinds(t, li)
    trunc === nothing && (trunc = _mak_trunc(; maxdim, cutoff))
    if trunc === nothing
        U, S, Vh = MAK.svd_compact(t, Tuple(li), Tuple(ri))
        truncerr = 0.0
    else
        U, S, Vh, err = MAK.svd_trunc(t, Tuple(li), Tuple(ri); trunc)
        kept2 = sum(abs2, _diagvals(S))
        total = kept2 + err^2
        truncerr = total > 0 ? err^2 / total : 0.0
    end
    return U, S, Vh, truncerr
end
_diagvals(S::AbstractTensor) = (A = IB.unnamed(S); [A[k, k] for k in 1:minimum(size(A))])
# Retag the two legs of the central matrix (and the matching bond legs of U / Vh).
function _relabel_bond(t::AbstractTensor, old::Index, tags::AbstractString)
    new = IB.named(space(_ascarried(t, old)), IB.IndexName(; tags = _totags(tags), plev = IB.plev(old)))
    return replaceinds(t, _ascarried(t, old) => new), new
end

# Orientation-preserving relabel of one leg (the seam's `replaceinds`).
_rename(t::AbstractTensor, old::Index, new::Index) = TensorInterface.replaceinds(t, [old], [new])
# A fresh index over `i`'s space under `tags`, with `i`'s prime level.
_fresh_like(i::Index, tags::AbstractString) =
    IB.named(space(i), IB.IndexName(; tags = _totags(tags), plev = IB.plev(i)))

# Canonical labels for a factorization's central matrix: S on (u, v) tagged "Link,u"/"Link,v",
# U and Vh relabelled to match (orientations preserved per copy).
function _canonical_svd(U, S, Vh)
    u0, v0 = inds(S)
    u, v = _fresh_like(u0, "Link,u"), _fresh_like(v0, "Link,v")
    S = _rename(_rename(S, u0, u), v0, v)
    U = _rename(U, u0, u)
    Vh = _rename(Vh, v0, v)
    return U, S, Vh, u, v
end

# svd: U(linds…, u), S(u, v), V(rinds…, v) with `U * S * V` reconstructing. `trunc` (a
# truncation strategy, see `truncation_strategy`) overrides `maxdim`/`cutoff`.
function LinearAlgebra.svd(t::AbstractTensor, linds; maxdim = nothing, cutoff = nothing, trunc = nothing, kwargs...)
    U, S, Vh, _ = _svd_split(t, linds; maxdim, cutoff, trunc)
    U, S, Vh, _, _ = _canonical_svd(U, S, Vh)
    return U, S, Vh
end

# factorize: L isometric for ortho = "left" (L = U, R = S·V), mirrored for "right"; the
# bond carries `tags`.
function LinearAlgebra.factorize(t::AbstractTensor, linds...; ortho = "left", maxdim = nothing, cutoff = nothing, tags = "Link,fact", kwargs...)
    lv = length(linds) == 1 ? _indvec(only(linds)) : collect(Index, linds)
    U, S, Vh, _ = _svd_split(t, lv; maxdim, cutoff)
    U, S, Vh, u, v = _canonical_svd(U, S, Vh)
    if ortho == "left"
        L, R = U, S * Vh                                   # bond u
        b = _fresh_like(u, String(tags))
        return _rename(L, u, b), _rename(R, u, b)
    elseif ortho == "right"
        L, R = U * S, Vh                                   # bond v
        b = _fresh_like(v, String(tags))
        return _rename(L, v, b), _rename(R, v, b)
    end
    return error("factorize: unknown ortho = $ortho")
end

"""
factorize_svd matching the seam convention used by `simple_update`: `ortho = "none"` returns
F1 = U√S and F2 = √S·V sharing the primed bond `u′`, with the singular values reported on
unprimed `(u, v)`; `spec.truncerr` is the discarded Σs² fraction.
"""
function TensorInterface.factorize_svd(t::AbstractTensor, linds; ortho = "none", singular_values! = nothing,
                                       maxdim = nothing, cutoff = nothing, tags = nothing, kwargs...)
    U, S, Vh, truncerr = _svd_split(t, linds; maxdim, cutoff)
    U, S, Vh, u, v = _canonical_svd(U, S, Vh)
    up = TensorInterface.prime(u)
    T = eltype(U)
    if ortho == "none"
        sq = TensorInterface.map_diag(x -> T(sqrt(real(x))), S)     # (u, v)
        F1 = U * _rename(sq, v, up)                                # (li…, u′)  — u′ in the v slot's orientation
        F2 = _rename(sq, u, up) * Vh                               # (u′, ri…)  — u′ in the u slot's orientation
    elseif ortho == "left"
        F1 = _rename(U, u, up)
        F2 = _rename(S * Vh, u, up)
    elseif ortho == "right"
        F1 = _rename(U * S, v, up)
        F2 = _rename(Vh, v, up)
    else
        error("factorize_svd: unknown ortho = $ortho")
    end
    if singular_values! !== nothing
        singular_values![] = Adapt.adapt(T, S)
    end
    return F1, F2, Spectrum(truncerr)
end

# Hermitian eigendecomposition: D on (link′, link) with the eigenvalues, U on (rinds…, link),
# such that `U · D · dag(U)′`-style reconstruction holds as in the previous backend
# (symmetric_gauge relies on it).
function LinearAlgebra.eigen(t::AbstractTensor, linds, rinds; ishermitian::Bool = false, kwargs...)
    ishermitian || error("eigen: only ishermitian = true is implemented")
    lv = _ascarried(t, _indvec(linds)); rv = _ascarried(t, _indvec(rinds))
    D, V = MAK.eigh_full(t, Tuple(lv), Tuple(rv))
    d1, d2 = inds(D)                       # (fresh, fresh); V is on (domain…, d2)
    lk = _fresh_like(d2, "Link,eigen")
    Dt = _rename(_rename(D, d1, TensorInterface.prime(lk)), d2, lk)
    Vt = _rename(V, d2, lk)
    # `V` comes back on the domain legs; the seam wants the eigenvectors on `rinds`.
    return Dt, Vt
end
function LinearAlgebra.eigen(t::AbstractTensor; ishermitian::Bool = false, kwargs...)
    lv = filter(i -> IB.plev(i) == 0, collect(Index, inds(t)))
    rv = collect(Index, TensorInterface.prime.(lv))
    D, U = LinearAlgebra.eigen(t, lv, rv; ishermitian, kwargs...)
    return D, TensorInterface.replaceinds(U, rv, lv)
end

# ── S=1/2 operator & state library ──────────────────────────────────────────────────────

const _σI = ComplexF64[1 0; 0 1]
const _σx = ComplexF64[0 1; 1 0]
const _σy = ComplexF64[0 -im; im 0]
const _σz = ComplexF64[1 0; 0 -1]

_is_spinhalf(i::Index) = length(i) == 2

"""
    register_op!(name::String, f::Function; nsites::Int = 1)

Register a custom operator matrix. `f(; kwargs...)` must return the operator matrix: 2×2 for
`nsites = 1`, 4×4 for `nsites = 2` in the first-index-fastest (column-major) basis convention.
"""
function register_op!(name::String, f::Function; nsites::Int = 1)
    nsites == 1 && (OP1_REGISTRY[name] = f; return nothing)
    nsites == 2 && (OP2_REGISTRY[name] = f; return nothing)
    return error("register_op!: only 1- and 2-site operators are supported")
end

const OP1_REGISTRY = Dict{String, Function}(
    "I" => (; kwargs...) -> _σI,
    "X" => (; kwargs...) -> _σx,
    "Y" => (; kwargs...) -> _σy,
    "Z" => (; kwargs...) -> _σz,
    "H" => (; kwargs...) -> ComplexF64[1 1; 1 -1] / sqrt(2),
    "S+" => (; kwargs...) -> ComplexF64[0 1; 0 0],
    "S-" => (; kwargs...) -> ComplexF64[0 0; 1 0],
    "Sz" => (; kwargs...) -> _σz / 2,
    "Sx" => (; kwargs...) -> _σx / 2,
    "Sy" => (; kwargs...) -> _σy / 2,
    "Rx" => (; θ) -> exp(-im * θ / 2 * _σx),
    "Ry" => (; θ) -> exp(-im * θ / 2 * _σy),
    "Rz" => (; θ) -> exp(-im * θ / 2 * _σz),
    "P" => (; ϕ) -> ComplexF64[1 0; 0 exp(im * ϕ)],
)
_op1(name::String; kwargs...) = haskey(OP1_REGISTRY, name) ? OP1_REGISTRY[name](; kwargs...) : nothing

_kr(A, B) = kron(B, A)   # s1 fastest
const OP2_REGISTRY = Dict{String, Function}(
    "Rzz" => (; ϕ) -> exp(-im * ϕ * _kr(_σz, _σz)),
    "Rxx" => (; ϕ) -> exp(-im * ϕ * _kr(_σx, _σx)),
    "Ryy" => (; ϕ) -> exp(-im * ϕ * _kr(_σy, _σy)),
    "Rxxyy" => (; θ) -> exp(-im * θ * 0.5 * (_kr(_σx, _σx) + _kr(_σy, _σy))),
    "Rxxyyzz" => (; θ) -> exp(-im * θ * 0.5 * (_kr(_σx, _σx) + _kr(_σy, _σy) + _kr(_σz, _σz))),
    "xx_plus_yy" => (; θ, β) -> exp(
        -0.5 * im * θ * (
            cos(β) * 0.5 * (_kr(_σx, _σx) + _kr(_σy, _σy)) +
                sin(β) * 0.5 * (_kr(_σy, _σx) - _kr(_σx, _σy))
        )
    ),
    "CZ" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1],
    "CNOT" => (; kwargs...) -> _controlled(_σx),
    "CX" => (; kwargs...) -> _controlled(_σx),
    "CY" => (; kwargs...) -> _controlled(_σy),
    "CRx" => (; θ) -> _controlled(exp(-im * θ / 2 * _σx)),
    "CRy" => (; θ) -> _controlled(exp(-im * θ / 2 * _σy)),
    "CRz" => (; θ) -> _controlled(exp(-im * θ / 2 * _σz)),
    "CPHASE" => (; ϕ) -> ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 exp(im * ϕ)],
    "SWAP" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1],
    "iSWAP" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 0 im 0; 0 im 0 0; 0 0 0 1],
    "√SWAP" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 (1 + im)/2 (1 - im)/2 0; 0 (1 - im)/2 (1 + im)/2 0; 0 0 0 1],
    "√iSWAP" => (; kwargs...) -> ComplexF64[1 0 0 0; 0 1/sqrt(2) im/sqrt(2) 0; 0 im/sqrt(2) 1/sqrt(2) 0; 0 0 0 1],
)
_op2(name::String; kwargs...) = haskey(OP2_REGISTRY, name) ? OP2_REGISTRY[name](; kwargs...) : nothing

function _controlled(u::AbstractMatrix)
    m = Matrix{ComplexF64}(I, 4, 4)
    m[2, 2] = u[1, 1]; m[2, 4] = u[1, 2]
    m[4, 2] = u[2, 1]; m[4, 4] = u[2, 2]
    return m
end

# Operator array `data[s1', s2', …, s1, s2, …]` onto an operator tensor with codomain the primed
# sites and domain the sites. Graded sites: projected onto the grading; a parity-odd operator gets a
# dim-1 dangling charge leg instead of erroring.
function _operator_tensor(data::AbstractArray, sites::Vector{<:Index})
    outs = collect(Index, TensorInterface.prime.(sites))
    any(isgraded, sites) || return ITensor(data, Tuple(vcat(outs, sites)))
    p = TA.tryproject(data, Tuple(outs), Tuple(sites))
    p === nothing && error(
        "op: the operator has weight outside the flux-zero sector on $(sites) — operators on graded " *
            "sites must conserve the charge (e.g. \"X\" under Z2, a lone \"C\" for fermions); use " *
            "charge-conserving operators or the dense backend.")
    return p
end

function TensorInterface.op(name::String, i::Index; kwargs...)
    is_fermionic(i) && return _operator_tensor(_fermionic_op_array(name, Index[i]; kwargs...), Index[i])
    _is_spinhalf(i) || error("op: the operator library covers d=2 (S=1/2) sites only")
    m = _op1(name; kwargs...)
    m === nothing && error("op: unknown single-site operator \"$name\"")
    return _operator_tensor(copy(m), Index[i])
end
function TensorInterface.op(name::String, i1::Index, i2::Index; kwargs...)
    is_fermionic(i1) && return _operator_tensor(_fermionic_op_array(name, Index[i1, i2]; kwargs...), Index[i1, i2])
    (_is_spinhalf(i1) && _is_spinhalf(i2)) || error("op: the operator library covers d=2 (S=1/2) sites only")
    m = _op2(name; kwargs...)
    m === nothing && error("op: unknown two-site operator \"$name\"")
    return _operator_tensor(reshape(copy(m), 2, 2, 2, 2), Index[i1, i2])
end

const _STATE_VECTORS = Dict(
    "↑" => [1.0, 0.0], "0" => [1.0, 0.0], "Up" => [1.0, 0.0], "Z+" => [1.0, 0.0],
    "↓" => [0.0, 1.0], "1" => [0.0, 1.0], "Dn" => [0.0, 1.0], "Z-" => [0.0, 1.0],
    "+" => [1.0, 1.0] / sqrt(2), "X+" => [1.0, 1.0] / sqrt(2),
    "-" => [1.0, -1.0] / sqrt(2), "X-" => [1.0, -1.0] / sqrt(2),
)
function TensorInterface.state(name::String, i::Index)
    if isgraded(i)
        vec = state_vector(name, i)
        TA.tryproject(vec, (i,), ()) === nothing && error(
            "state: \"$name\" carries nonzero charge on $(i) — graded tensors carry zero total " *
                "charge. Charged product states are built per vertex by the network constructors " *
                "(which route the charge through link/charge legs); start from neutral states or " *
                "use charge-conserving gates.")
        return _vector_tensor(vec, i)
    end
    _is_spinhalf(i) || error("state: the state library covers d=2 (S=1/2) sites only")
    haskey(_STATE_VECTORS, name) || error("state: unknown state \"$name\"")
    return _vector_tensor(copy(_STATE_VECTORS[name]), i)
end

# ── Graded spaces (kernel_hooks capability surface) ─────────────────────────────────────

"""
    graded_space(symmetry::String, sectors)

A graded range from `charge => dimension` pairs. `symmetry` is one of `"Z2"`, `"U1"`, `"fZ2"`
(fermionic parity), `"fU1"` (fermions with conserved particle number) or `"fU1xU1"` (fermions with
separately conserved N↑, N↓; charges are tuples).
"""
function graded_space(symmetry::String, sectors)
    key = replace(lowercase(symmetry), " " => "")
    if key in ("fu1", "fermionnumber")
        return GA.gradedrange([(TKS.U1Irrep(q) ⊠ TKS.FermionParity(mod(q, 2))) => Int(d) for (q, d) in sectors])
    elseif key in ("fu1xu1", "fu1u1")
        return GA.gradedrange([(TKS.U1Irrep(a) ⊠ TKS.U1Irrep(b) ⊠ TKS.FermionParity(mod(a + b, 2))) => Int(d) for ((a, b), d) in sectors])
    elseif key == "z2"
        return GA.gradedrange([GA.Z2(q) => Int(d) for (q, d) in sectors])
    elseif key in ("u1", "u(1)")
        return GA.gradedrange([GA.U1(q) => Int(d) for (q, d) in sectors])
    elseif key in ("fz2", "fermion", "fermionparity")
        return GA.gradedrange([GA.SectorRange(TKS.FermionParity(q)) => Int(d) for (q, d) in sectors])
    end
    return error("graded_space: unknown symmetry \"$symmetry\" (supported: Z2, U1, fZ2, fU1, fU1xU1)")
end
const ⊠ = TKS.:⊠

is_fermionic(i::Index) = isgraded(i) && any(GA.fermionparity, GA.sectors(space(i)))
fermion_space(d0::Integer = 1, d1::Integer = 1) = graded_space("fZ2", [0 => d0, 1 => d1])
new_fermion_index(d0::Integer = 1, d1::Integer = 1; tags = "") = Index(fermion_space(d0, d1), tags)

# ── Sector bookkeeping ──────────────────────────────────────────────────────────────────

# Sectors and block ranges of an index's range IN ITS BASE (non-dual) labelling — the state and
# operator tables are written against the base space, so a dual copy is mapped back first.
_basespace(i::Index) = (sp = space(i); GA.isdual(sp) ? GA.dual(sp) : sp)
_basesectors(i::Index) = collect(GA.sectors(_basespace(i)))
function _fock_ranges(i::Index)
    lens = GA.blocklengths(_basespace(i))
    stops = cumsum(lens)
    return [(stops[k] - lens[k] + 1):stops[k] for k in eachindex(lens)]
end
# Positions of sector `c` (base label) in the dense sector-ordered layout of `i`.
function _fock_range(i::Index, c)
    secs = _basesectors(i)
    k = findfirst(==(c), secs)
    k === nothing && error("sector $c not found on $(i)")
    return _fock_ranges(i)[k]
end

_sectortype(i::Index) = GA.sectortype(space(i))
_label_type(::Type{GA.SectorRange{L}}) where {L} = L

# Mode-basis occupation labels: d = 2 → (n,), d = 4 → (n↑, n↓) over (|0⟩, |↑⟩, |↓⟩, |↑↓⟩).
_mode_occupations(d::Int) = d == 2 ? [(0,), (1,)] : [(0, 0), (1, 0), (0, 1), (1, 1)]

# The sector a mode-basis state carries under the index's grading (factor order as built by
# `graded_space`: U(1) charge(s) first, FermionParity last).
function _mode_sector(S::Type{<:GA.SectorRange}, occ::Tuple)
    L = _label_type(S)
    L === TKS.FermionParity && return GA.SectorRange(TKS.FermionParity(mod(sum(occ), 2)))
    if L <: TKS.ProductSector
        F = fieldtypes(fieldtype(L, :sectors))
        nu1 = count(==(TKS.U1Irrep), F)
        nu1 == 1 && return GA.SectorRange(L((TKS.U1Irrep(sum(occ)), TKS.FermionParity(mod(sum(occ), 2)))))
        (nu1 == 2 && length(occ) == 2) &&
            return GA.SectorRange(L((TKS.U1Irrep(occ[1]), TKS.U1Irrep(occ[2]), TKS.FermionParity(mod(sum(occ), 2)))))
    end
    return error("op/state: unsupported fermionic grading $(S) for a d = $(2^length(occ)) site")
end

# Position of each mode-basis state in the index's sector-ordered dense layout.
function _mode_perm(i::Index)
    S = _sectortype(i)
    counts = Dict{Any, Int}()
    perm = Int[]
    for occ in _mode_occupations(length(i))
        c = _mode_sector(S, occ)
        k = get(counts, c, 0)
        counts[c] = k + 1
        push!(perm, first(_fock_range(i, c)) + k)
    end
    return perm
end

fuse_sectors(a, b) = GA.tensor_product(a, b)
# The dual CHARGE as a plain (non-flagged) sector label: `GA.dual` on a `SectorRange` only sets an
# arrow flag and does not compare equal to the base label, so it cannot key a spectrum Dict.
dual_sector(c) = GA.SectorRange(TKS.dual(GA.label(c)))
trivial_sector(c) = GA.trivial(typeof(c))

# A dim-1 link index carrying charge `q` (used for routing charges through product states).
charged_link_index(q; tags = "Link") = Index(GA.gradedrange([q => 1]), String(tags))
trivial_link_index(ref::Index; tags = "Link") = charged_link_index(GA.trivial(_sectortype(ref)); tags)

# ── Graded states ───────────────────────────────────────────────────────────────────────

const F_STATES = Dict{String, Vector{Float64}}(
    "0" => [1, 0], "Emp" => [1, 0], "Empty" => [1, 0],
    "1" => [0, 1], "Occ" => [0, 1], "Occupied" => [0, 1],
)
# Spinful (d = 4) fermionic states in the MODE basis (|0⟩, |↑⟩, |↓⟩, |↑↓⟩).
const F_STATES_4 = Dict{String, Vector{Float64}}(
    "0" => [1, 0, 0, 0], "Emp" => [1, 0, 0, 0], "Empty" => [1, 0, 0, 0],
    "Up" => [0, 1, 0, 0], "↑" => [0, 1, 0, 0],
    "Dn" => [0, 0, 1, 0], "↓" => [0, 0, 1, 0],
    "UpDn" => [0, 0, 0, 1], "2" => [0, 0, 0, 1],
)

# Resolve a local state (name or raw vector) on a graded site to its dense vector in the site's
# sector-ordered layout (fermionic names for fermionic sites, the spin registry otherwise).
function state_vector(namevec, i::Index)
    fermionic = is_fermionic(i)
    d = length(i)
    vec = if namevec isa AbstractVector{<:Number}
        collect(namevec)
    elseif fermionic
        table = d == 2 ? F_STATES : d == 4 ? F_STATES_4 :
            error("state: fermionic state library covers d = 2 (spinless) and d = 4 (spinful) sites")
        get(table, String(namevec), nothing)
    else
        haskey(_STATE_VECTORS, String(namevec)) ? copy(_STATE_VECTORS[String(namevec)]) : nothing
    end
    vec === nothing && error("state: unknown state \"$namevec\" for a d = $d site")
    length(vec) == d || error("state: vector length $(length(vec)) ≠ site dimension $d")
    if fermionic
        out = zeros(eltype(vec), d)
        out[_mode_perm(i)] .= vec
        return out
    end
    return vec
end

# The (single) charge sector a state vector lives in; graded product states must have definite
# local charge.
function vector_sector(vec::AbstractVector, i::Index)
    secs = _basesectors(i); rs = _fock_ranges(i)
    live = [secs[k] for k in eachindex(secs) if any(!iszero, vec[rs[k]])]
    length(live) == 1 || error("state on a graded site must carry a definite charge; found support in sectors $(live)")
    return only(live)
end

# Product-state vertex tensor: the site's state vector with dim-1 (possibly charged, possibly
# dual) link legs attached — a single projection onto the flux-zero blocks.
function product_vertex_tensor(elt::Type, vec::AbstractVector, site::Index, links::AbstractVector{<:Index})
    all(l -> length(l) == 1, links) || error("product_vertex_tensor: links must be dim-1")
    data = reshape(elt.(vec), length(site), ntuple(_ -> 1, length(links))...)
    t = TA.tryproject(data, (site, links...), ())
    t === nothing && error("product_vertex_tensor: the link charges do not neutralize the site charge")
    return t
end

# Purification pairing state Σₛ |s⟩_kets ⊗ ⟨s|_ancillas: the identity map from the ancillas
# (dual copies) to the kets, as one tensor.
function pairing_tensor(elt::Type, kets, ancs)
    kv, av = collect(Index, kets), collect(Index, ancs)
    length(kv) == length(av) || error("pairing_tensor: need as many ancillas as kets")
    all(i -> !isdual(i), kv) && all(isdual, av) ||
        error("pairing_tensor: kets must be non-dual and ancillas dual copies")
    return IB.id(elt, Tuple(kv), Tuple(TA.dual.(av)))
end

# ── Fermionic operator library ───────────────────────────────────────────────────────────
# Fock matrices in the mode basis |0⟩, |1⟩ (site 1 slowest for 2-site ops). LOCAL matrices —
# no Jordan-Wigner strings; the graded contraction supplies them.
const _F_A = ComplexF64[0 1; 0 0]
const _F_ADAG = ComplexF64[0 0; 1 0]
const _F_N = _F_ADAG * _F_A
const _F_I2 = ComplexF64[1 0; 0 1]
const _F_HOP = ComplexF64[0 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 0]
const _F_NN = ComplexF64[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]
const _F_PAIR = ComplexF64[0 0 0 1; 0 0 0 0; 0 0 0 0; 1 0 0 0]
const _F4_AUP = ComplexF64[0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
const _F4_ADN = ComplexF64[0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0]
const _F4_NUP = _F4_AUP' * _F4_AUP
const _F4_NDN = _F4_ADN' * _F4_ADN
const _F4_I = Matrix{ComplexF64}(I, 4, 4)
const _F4_Z = ComplexF64[1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]

function _f_op1_matrix(name::String, d::Int; kwargs...)
    if d == 4
        name == "I" && return _F4_I
        name == "N" && return _F4_NUP + _F4_NDN
        name == "Nup" && return _F4_NUP
        name == "Ndn" && return _F4_NDN
        name == "NupNdn" && return _F4_NUP * _F4_NDN
        name == "Sz" && return 0.5 * (_F4_NUP - _F4_NDN)
        name == "F_int" && return exp(-im * kwargs[:θ] * _F4_NUP * _F4_NDN)
        name == "F_phase" && return exp(-im * kwargs[:θ] * (_F4_NUP + _F4_NDN))
        return nothing
    end
    name == "I" && return _F_I2
    name == "N" && return _F_N
    name == "F_phase" && return ComplexF64[1 0; 0 exp(-im * kwargs[:θ])]
    name in ("C", "A") && return _F_A
    name in ("Cdag", "Adag") && return _F_ADAG
    return nothing
end
_f4_kron(A, B) = kron(A, B)
_f4_cdagc(Cσ) = _f4_kron(Cσ' * _F4_Z, Cσ)
_f4_hop(Cσ) = _f4_cdagc(Cσ) + _f4_cdagc(Cσ)'
function _f_op2_matrix4(name::String; kwargs...)
    name == "hopping_up" && return _f4_hop(_F4_AUP)
    name == "hopping_dn" && return _f4_hop(_F4_ADN)
    name == "hopping" && return _f4_hop(_F4_AUP) + _f4_hop(_F4_ADN)
    name == "CdagC_up" && return _f4_cdagc(_F4_AUP)
    name == "CdagC_dn" && return _f4_cdagc(_F4_ADN)
    name == "F_hop_up" && return exp(-im * kwargs[:θ] * _f4_hop(_F4_AUP))
    name == "F_hop_dn" && return exp(-im * kwargs[:θ] * _f4_hop(_F4_ADN))
    name == "F_hop" && return exp(-im * kwargs[:θ] * (_f4_hop(_F4_AUP) + _f4_hop(_F4_ADN)))
    return nothing
end
function _f_op2_matrix(name::String, d::Int; kwargs...)
    if d == 4
        M = _f_op2_matrix4(name; kwargs...)
        M === nothing && error("op: unknown spinful fermionic 2-site operator \"$name\"")
        return M
    end
    name == "hopping" && return _F_HOP
    name == "NN" && return _F_NN
    name == "pairing" && return _F_PAIR
    name == "CdagC" && return ComplexF64[0 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]   # c†_v c_w
    name == "CCdag" && return ComplexF64[0 0 0 0; 0 0 -1 0; 0 0 0 0; 0 0 0 0]  # c_v c†_w
    name == "CdagCdag" && return ComplexF64[0 0 0 0; 0 0 0 0; 0 0 0 0; 1 0 0 0] # c†_v c†_w
    name == "CC" && return ComplexF64[0 0 0 -1; 0 0 0 0; 0 0 0 0; 0 0 0 0]      # c_v c_w
    name == "F_hop" && return exp(-im * kwargs[:θ] * _F_HOP)
    name == "F_nn" && return exp(-im * kwargs[:θ] * _F_NN)
    name == "F_pair" && return exp(-im * kwargs[:θ] * _F_PAIR)
    name == "F_hop_nn" && return exp(-im * (kwargs[:θ] * _F_HOP + kwargs[:ϕ] * _F_NN))
    return nothing
end
# Reorder a site-1-slowest 2-site matrix M[out, in] into the (u1, u2, s1, s2) leg array.
_two_site_array(M::AbstractMatrix, d1::Int, d2::Int) = permutedims(reshape(M, d2, d1, d2, d1), (2, 1, 4, 3))

# The dense operator array ⟨u…|O|s…⟩ with legs (u1..un, s1..sn) for a fermionic operator, each
# leg reordered from the mode basis into its site's sector layout.
function _fermionic_op_array(name::String, sites::Vector{<:Index}; kwargs...)
    d = length(first(sites))
    (d in (2, 4) && all(i -> length(i) == d, sites)) ||
        error("op: fermionic operator library covers uniform d = 2 (spinless) or d = 4 (spinful) sites")
    A = if length(sites) == 1
        _f_op1_matrix(name, d; kwargs...)
    else
        M = _f_op2_matrix(name, d; kwargs...)
        M === nothing ? nothing : _two_site_array(M, d, d)
    end
    A === nothing && error("op: unknown fermionic operator \"$name\"")
    ips = [invperm(_mode_perm(i)) for i in sites]
    return A[vcat(ips, ips)...]
end

# ── Parity gauge, spectra, link allocation (boundary MPS) ───────────────────────────────

# The BP message gauge freedom is PER SECTOR: a unit-modulus per-sector phase (the fermionic parity
# sign is one instance) gives an equally valid fixed point. Select the PSD representative by
# normalising each diagonal block's trace to positive real.
function psd_gauge(t::AbstractTensor)
    (isgraded(t) && ndims(t) == 2) || return t
    A = copy(IB.unnamed(t))
    for I in GA.eachblockstoredindex(A)
        b = view(A, I)
        n = min(size(b)...)
        n == 0 && continue
        z = sum(b[k, k] for k in 1:n)
        iszero(z) || (b .*= conj(z) / abs(z))
    end
    return ITensor(A, Tuple(inds(t)))
end

# Charge spectrum reachable by one message site: the convolution of its legs' carried sector
# spectra (dual legs contribute dual sectors), weight ∝ sector dimension.
function site_charge_spectrum(m::AbstractTensor)
    S = _sectortype(first(inds(m)))
    w = Dict{Any, Float64}(GA.trivial(S) => 1.0)
    for l in inds(m)
        sp = space(l)
        wl = Dict{Any, Float64}()
        for (c, n) in zip(GA.sectors(sp), GA.blocklengths(sp))
            q = isdual(l) ? dual_sector(c) : c          # a dual leg carries the dual charge
            wl[q] = get(wl, q, 0.0) + n / length(sp)
        end
        w = convolve_charge_spectra(w, wl)
    end
    return w
end
function convolve_charge_spectra(w1::Dict, w2::Dict)
    out = Dict{Any, Float64}()
    for (q1, a) in w1, (q2, b) in w2
        q = GA.tensor_product(q1, q2)
        out[q] = get(out, q, 0.0) + a * b
    end
    return out
end
# A link space of total dimension `d` supporting the sectors reachable from the left (spectrum `wl`)
# that can be neutralised from the right (spectrum `wr`), with dimensions ∝ joint weight.
function allocate_link_space(wl::Dict, wr::Dict, d::Integer)
    w = Dict{Any, Float64}()
    for (q, a) in wl
        b = get(wr, dual_sector(q), 0.0)
        b > 0 && (w[q] = a * b)
    end
    isempty(w) && (w = Dict{Any, Float64}(GA.trivial(typeof(first(keys(wl)))) => 1.0))
    qs = collect(keys(w))
    if length(qs) >= d
        sort!(qs; by = q -> -w[q])
        return GA.gradedrange([q => 1 for q in qs[1:Int(d)]])
    end
    tot = sum(values(w))
    alloc = Dict(q => max(1, floor(Int, d * w[q] / tot)) for q in qs)
    while sum(values(alloc)) < d
        q = argmax(q -> d * w[q] / tot - alloc[q], qs)
        alloc[q] += 1
    end
    while sum(values(alloc)) > d
        q = argmax(q -> alloc[q] - d * w[q] / tot, filter(q -> alloc[q] > 1, qs))
        alloc[q] -= 1
    end
    return GA.gradedrange([q => alloc[q] for q in qs])
end

# Fit adjoint of a boundary-MPS rail tensor: the bra rail of the fitting sweep. The rail tensor has
# crossing legs (`metric_legs`: the network bond and its primed bra copy, opposite arrows) and
# virtual MPS bonds. The adjoint that makes the one-site ALS extraction exact — the bra rail's
# QR isometries pair with the ket rail as identities on the MPS bonds — is `conj` with a parity
# twist on the NON-dual crossing legs and never on the MPS bonds (found by exhaustive search over
# the leg-class twist rules against an exact MPS fit; direction-independent, an involution, and
# the same recipe the TensorKit backend used). Trivial for bosonic sectors.
function fit_adjoint(t::AbstractTensor, metric_legs)
    c = conj(t)
    isgraded(t) || return c
    mv = _indvec(metric_legs)
    dims = Tuple(k for (k, i) in enumerate(inds(t)) if !isdual(i) && any(==(i), mv))
    isempty(dims) || GA.twist!(IB.unnamed(c), dims)
    return c
end

# A tensor anchors a parity-gauge line iff it carries a "Charge"-tagged dangling leg of odd fermion
# parity: multi-vertex closures of regions containing it pick up a gauge sign.
function TensorInterface.has_closure_gauge(t::AbstractTensor)
    return any(inds(t)) do i
        isgraded(i) && occursin("Charge", _tagstring(i)) && any(GA.fermionparity, GA.sectors(space(i)))
    end
end

# ── Fused double-layer BP kernels (dense data) ──────────────────────────────────────────
# The BP message update ψ·Πm·ψ̄ is the inner loop of every sweep. The generic sequence path
# matricises both operands of every pairwise contraction (a permuted copy of the F-sized chain
# intermediate each time), which measured ~8–9 F of allocation and 3× the time of the old fused
# kernel at D = 12. Here each incoming message is absorbed along its bond with slice GEMMs into a
# ping-pong buffer (no permutations: the bond dimension is split out of the column-major layout as
# (pre, D, post)), and the conjugate of ψ is folded into the closing GEMM. Live memory: ψ and two
# buffers = 3F, plus the D² message. Structure not recognised → `nothing` (generic path).

# ψ's data as a strided array, or nothing when the kernel does not apply.
function _fused_data(ψ::AbstractTensor)
    isgraded(ψ) && return nothing
    A = IB.unnamed(ψ)
    A isa StridedArray || return nothing
    return A
end

# For a standard doubled message on bond `b` of ψ (legs (b, b′) with b′ = prime(b)), the position
# of `b` among ψ's legs and the message matrix M[b, b′]; nothing otherwise.
function _fused_message_slot(m::AbstractTensor, ψinds::Vector{<:Index})
    ndims(m) == 2 || return nothing
    isgraded(m) && return nothing
    i1, i2 = inds(m)
    b, bp = TensorInterface.plev(i1) == 0 ? (i1, i2) : (i2, i1)
    (TensorInterface.plev(b) == 0 && TensorInterface.plev(bp) == 1 && TensorInterface.noprime(bp) == b) || return nothing
    k = findfirst(==(b), ψinds)
    k === nothing && return nothing
    M = IB.unnamed(aligndims(m, (b, bp)))
    M isa StridedArray || return nothing
    return k, M
end

# Task-local buffer pool for the chain intermediates: two F-sized buffers per element type, reused
# across messages (the kernel returns only the D² result, never a buffer). Live memory during a
# message update is therefore ψ + 2 buffers = 3F + change, and steady-state heap churn ~0.
function _fused_buffers(::Type{T}, n::Integer) where {T}
    pool = get!(task_local_storage(), :tnqs_fused_buffers) do
        Dict{DataType, Tuple{Vector, Vector}}()
    end::Dict{DataType, Tuple{Vector, Vector}}
    bufs = get(pool, T, nothing)
    if bufs === nothing || length(bufs[1]) < n
        bufs = (Vector{T}(undef, n), Vector{T}(undef, n))
        pool[T] = bufs
    end
    return bufs
end

# out[a, b′, c] = Σ_b cur[a, b, c] · M[b, b′] along dimension k, without permuting. Large `pre`:
# one GEMM per c-slice. Small `pre` (the site leg in front of the bond): fold it into a single GEMM
# with kron(Mᵀ, 𝟙_pre) — `pre`× more flops than the slices, but one BLAS call instead of `post`
# tiny ones (measured 0.63 ms → GEMM-bound at D = 12, pre = 2, post = 1728).
function _absorb_dim!(out::AbstractArray, cur::AbstractArray, M::AbstractMatrix, k::Integer)
    dims = size(cur)
    pre = prod(dims[1:(k - 1)]; init = 1); D = dims[k]; post = prod(dims[(k + 1):end]; init = 1)
    C = reshape(cur, pre, D, post); O = reshape(out, pre, D, post)
    if pre == 1
        LinearAlgebra.mul!(reshape(O, D, post), transpose(M), reshape(C, D, post))
    elseif post == 1
        LinearAlgebra.mul!(reshape(O, pre, D), reshape(C, pre, D), M)
    elseif pre <= 8 && post > 8
        K = kron(transpose(M), LinearAlgebra.I(pre))                  # (pre·D)² — small
        LinearAlgebra.mul!(reshape(O, pre * D, post), K, reshape(C, pre * D, post))
    else
        for c in 1:post
            LinearAlgebra.mul!(view(O, :, :, c), view(C, :, :, c), M)
        end
    end
    return out
end

# Absorb every incoming message (and optional site operators) into ψ's data. Returns the chain
# result (a pooled buffer; ψ's own data is never written) or nothing. `slots` are (k, M) pairs.
function _fused_chain(A::StridedArray, slots::Vector)
    T = promote_type(eltype(A), (eltype(M) for (_, M) in slots)...)
    isempty(slots) && return T == eltype(A) ? A : T.(A)
    cur = T == eltype(A) ? A : T.(A)
    b1, b2 = _fused_buffers(T, length(A))
    buf1 = reshape(view(b1, 1:length(A)), size(A)); buf2 = reshape(view(b2, 1:length(A)), size(A))
    out = buf1
    for (k, M) in slots
        _absorb_dim!(out, cur, M, k)
        cur, out = out, (out === buf1 ? buf2 : buf1)
    end
    return cur
end

# R[b′, b] = Σ_{a,c} conj(Ψ[a, b′, c]) T[a, b, c] over the split (pre, D, post) of the outgoing bond.
# Small `pre`: one (pre·D)² GEMM over c followed by a partial trace over a; large `pre`: c-slices.
function _close_bond!(R::AbstractMatrix, Ψ3::AbstractArray{<:Any, 3}, T3::AbstractArray{<:Any, 3})
    pre, D, post = size(Ψ3)
    T = eltype(R)
    if post == 1
        LinearAlgebra.mul!(R, adjoint(reshape(Ψ3, pre, D)), reshape(T3, pre, D))
    elseif pre <= 8
        G = reshape(Ψ3, pre * D, post) * adjoint(reshape(T3, pre * D, post))   # G[(a,b′),(a2,b)] = Σ_c Ψ T̄
        fill!(R, zero(T))
        G4 = reshape(G, pre, D, pre, D)
        for b in 1:D, bp in 1:D, a in 1:pre
            R[bp, b] += conj(G4[a, bp, a, b])                               # conj(Ψ) T = conj(Ψ T̄)
        end
    else
        fill!(R, zero(T))
        for c in 1:post
            LinearAlgebra.mul!(R, adjoint(view(Ψ3, :, :, c)), view(T3, :, :, c), one(T), one(T))
        end
    end
    return R
end

# Outgoing message on the one bond of ψ not covered by `incoming`: R[b′, b] = Σ conj(ψ)[…b′…] T[…b…].
function fused_norm_message(ψ::AbstractTensor, sinds::Vector{<:Index}, incoming::Vector; normalize::Bool = true)
    A = _fused_data(ψ)
    A === nothing && return nothing
    ψinds = collect(Index, inds(ψ))
    slots = Any[]
    covered = falses(length(ψinds))
    for m in incoming
        sl = _fused_message_slot(m, ψinds)
        sl === nothing && return nothing
        covered[sl[1]] && return nothing
        covered[sl[1]] = true
        push!(slots, sl)
    end
    open = [k for k in eachindex(ψinds) if !covered[k] && !(ψinds[k] in sinds)]
    length(open) == 1 || return nothing
    all(k -> covered[k] || ψinds[k] in sinds || k == only(open), eachindex(ψinds)) || return nothing
    kout = only(open)
    Tm = _fused_chain(A, slots)
    dims = size(A)
    pre = prod(dims[1:(kout - 1)]; init = 1); D = dims[kout]; post = prod(dims[(kout + 1):end]; init = 1)
    T = eltype(Tm)
    Ψ3 = reshape(A, pre, D, post); T3 = reshape(Tm, pre, D, post)
    bout = ψinds[kout]; boutp = TensorInterface.prime(bout)
    if pre == 1
        # R[b, b′] = Σ_c T[b, c] conj(ψ[b′, c])
        R = Matrix{T}(undef, D, D)
        LinearAlgebra.mul!(R, reshape(T3, D, post), adjoint(reshape(Ψ3, D, post)))
        legs = (bout, boutp)
    else
        R = Matrix{T}(undef, D, D)
        _close_bond!(R, Ψ3, T3)
        legs = (boutp, bout)
    end
    if normalize
        s = sum(R)
        iszero(s) || (R .*= inv(s))
    end
    return ITensor(R, legs)
end

# Single-vertex closure ψ·Πm·(ops)·ψ̄ → scalar; `ops` are (s′, s) operator tensors on site legs.
function fused_norm_scalar(ψ::AbstractTensor, sinds::Vector{<:Index}, incoming::Vector, ops::Vector)
    A = _fused_data(ψ)
    A === nothing && return nothing
    ψinds = collect(Index, inds(ψ))
    slots = Any[]
    covered = falses(length(ψinds))
    for m in incoming
        sl = _fused_message_slot(m, ψinds)
        sl === nothing && return nothing
        covered[sl[1]] && return nothing
        covered[sl[1]] = true
        push!(slots, sl)
    end
    for o in ops
        (ndims(o) == 2 && !isgraded(o)) || return nothing
        i1, i2 = inds(o)
        s, sp = TensorInterface.plev(i1) == 0 ? (i1, i2) : (i2, i1)
        (TensorInterface.plev(sp) == 1 && TensorInterface.noprime(sp) == s && s in sinds) || return nothing
        k = findfirst(==(s), ψinds)
        (k === nothing || covered[k]) && return nothing
        covered[k] = true
        push!(slots, (k, IB.unnamed(aligndims(o, (s, sp)))))   # M[s, s′] = O[s′, s]: T[…s′…] = Σ_s ψ[…s…] O[s′, s]
    end
    all(k -> covered[k] || ψinds[k] in sinds, eachindex(ψinds)) || return nothing
    Tm = _fused_chain(A, slots)
    return LinearAlgebra.dot(vec(A), vec(Tm))                    # Σ conj(ψ) T
end

# ── Hilbert inner products ──────────────────────────────────────────────────────────────
# NamedDimsArrays' `dot` on graded tensors is the CONTRACTION (with the fermionic twists), which
# is not positive definite on mixed-orientation fermionic tensors. KrylovKit and the CTM cycle
# deflation need the Hilbert pairing, so route both through the raw storage, aligned by name.
function _hdot(a::AbstractTensor, b::AbstractTensor)
    ia = Tuple(inds(a))
    # `aligndims` copies `b` even when the orders already agree (the common case: two sweeps of the
    # same environment); skip it then.
    bb = all(((x, y),) -> x == y, zip(ia, inds(b))) && ndims(a) == ndims(b) ? b : aligndims(b, ia)
    return LinearAlgebra.dot(IB.unnamed(a), IB.unnamed(bb))
end
LinearAlgebra.dot(a::Tensor, b::Tensor) = _hdot(a, b)
VectorInterface.inner(a::Tensor, b::Tensor) = _hdot(a, b)

end
