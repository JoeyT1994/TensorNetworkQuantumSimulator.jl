# Symmetric backend: Index-labelled wrapper around a TensorKit TensorMap over any
# graded (sector-carrying) space — bosonic Z2/U(1) irreps and fermionic parity alike.
#
# HARD RULE: nothing algebraic is implemented here. Contraction, permutation (including
# every Koszul sign for fermionic sectors — Jordan-Wigner strings emerge from the
# braiding), blockwise factorizations and truncation all delegate to TensorKit and
# MatrixAlgebraKit. This file is bookkeeping between Index labels and TensorMap slots,
# plus the operator/state quack layer.
#
# Conventions (validated against dense ground truth — Jordan-Wigner for fermions, the
# dense Tensor backend for bosonic sectors):
#   * Slot `a` of the data holds `slotspace(inds[a])`: the index's base space, dualed
#     when the per-copy `dual` flag is set. The flag is per tensor copy — the same
#     Index identity (id, plev) appears with opposite flags on the two tensors sharing
#     a bond (src = false, dst = true), and `dag` flips all flags.
#   * `dag` is the plain categorical adjoint permuted back to the codomain — no twists.
#     Site legs are non-dual on kets; with that convention closed bra-ket networks
#     evaluate to physical inner products.
#   * Operators are built by fusion-tree assignment of the dense operator array
#     ⟨u…|O|s…⟩ on a two-sided (outs ← ins) map, then permuted one-sided. For abelian
#     sectors the tree basis coincides with the sector-ordered product basis; tables
#     live in the mode basis and are permuted per site space (_mode_perm). Elements
#     outside the conserving blocks error loudly (that is the point of the symmetry).
#   * TensorMaps enforce zero total flux. Charged product states route their site
#     charges through dim-1 links (T-join, see kernel_hooks.jl); a nonzero TOTAL charge
#     rides an automatically attached dim-1 "Charge"-tagged dangling leg.
#   * Every contraction requires the two copies of a contracted index to carry opposite
#     flags; a same-flag pairing is a network-construction bug and errors loudly.

const TKSpace = TK.GradedSpace
const GradedIndex = Index{<:TKSpace}

"""
    graded_space(symmetry::String, sectors)

A TensorKit graded space from `charge => dimension` pairs. `symmetry` is one of
`"Z2"`, `"U1"`, `"fZ2"` (fermionic parity), `"fU1"` (fermions with conserved particle
number), or `"fU1xU1"` (fermions with separately conserved N↑, N↓; charges are tuples).
"""
function graded_space(symmetry::String, sectors)
    key = replace(lowercase(symmetry), " " => "")
    #fermions with conserved particle number(s): the parity factor (which carries the
    #braiding) is locked to the U(1) charge(s)
    if key in ("fu1", "fermionnumber")
        I = TK.:⊠(TK.U1Irrep, TK.FermionParity)
        return TK.Vect[I]((I(q, mod(q, 2)) => Int(d) for (q, d) in sectors)...)
    elseif key in ("fu1xu1", "fu1u1")
        I = TK.:⊠(TK.U1Irrep, TK.U1Irrep, TK.FermionParity)
        return TK.Vect[I]((I(a, b, mod(a + b, 2)) => Int(d) for ((a, b), d) in sectors)...)
    end
    I = key in ("z2",) ? TK.Z2Irrep :
        key in ("u1", "u(1)") ? TK.U1Irrep :
        key in ("fz2", "fermion", "fermionparity") ? TK.FermionParity :
        error("graded_space: unknown symmetry \"$symmetry\" (supported: Z2, U1, fZ2, fU1, fU1xU1)")
    return TK.Vect[I]((q => Int(d) for (q, d) in sectors)...)
end

#Does the grading carry fermionic statistics (a FermionParity factor)?
is_fermionic(::Type{TK.FermionParity}) = true
is_fermionic(::Type{TK.TensorKitSectors.ProductSector{T}}) where {T} = any(is_fermionic, fieldtypes(T))
is_fermionic(::Type{<:TK.Sector}) = false
is_fermionic(i::GradedIndex) = is_fermionic(TK.sectortype(space(i)))

_nfactors(::Type{<:TK.Sector}) = 1
_nfactors(::Type{TK.TensorKitSectors.ProductSector{T}}) where {T} = fieldcount(T)

#Mode-basis occupation labels: d = 2 → (n,), d = 4 → (n↑, n↓) over (|0⟩, |↑⟩, |↓⟩, |↑↓⟩)
_mode_occupations(d::Int) = d == 2 ? [(0,), (1,)] : [(0, 0), (1, 0), (0, 1), (1, 1)]

#The sector a mode-basis state carries under the index's grading (factor order as built
#by graded_space: U(1) charge(s) first, FermionParity last)
function _mode_sector(I::Type, occ::Tuple)
    I === TK.FermionParity && return TK.FermionParity(mod(sum(occ), 2))
    n = _nfactors(I)
    n == 2 && return I(sum(occ), mod(sum(occ), 2))
    (n == 3 && length(occ) == 2) && return I(occ[1], occ[2], mod(sum(occ), 2))
    return error("op/state: unsupported fermionic grading $(I) for a d = $(2^length(occ)) site")
end

#Position of each mode-basis state in the index's sector-ordered dense layout: the
#operator/state tables live in the mode basis and are permuted per site space, so any
#grading (parity, fU1, dual U(1)) with any sector order works.
function _mode_perm(i::GradedIndex)
    I = TK.sectortype(space(i))
    counts = Dict{Any, Int}()
    perm = Int[]
    for occ in _mode_occupations(dimof(i))
        c = _mode_sector(I, occ)
        k = get(counts, c, 0)
        counts[c] = k + 1
        push!(perm, first(_fock_range(i, c)) + k)
    end
    return perm
end

fermion_space(d0::Integer = 1, d1::Integer = 1) = graded_space("fZ2", [0 => d0, 1 => d1])

Index(space::TKSpace, tags::AbstractString = "") =
    Index(rand(UInt64), space, 0, String(tags), false)

"""
    new_fermion_index(d0 = 1, d1 = 1; tags = "")

A fresh fermionic (Z2-parity graded) index with `d0` even and `d1` odd states.
"""
new_fermion_index(d0::Integer = 1, d1::Integer = 1; tags = "") =
    Index(fermion_space(d0, d1), tags)

space_dim(s::TKSpace) = TK.dim(s)

#Trivial-dominant split for a fresh link of total dimension d (fermion-branch recipe:
#double-layer networks are trivial-sector dominant, so links need at least as many
#trivial-sector states as charged ones).
function TensorInterface.new_index(ref::GradedIndex, d::Integer; tags = "")
    I = TK.sectortype(space(ref))
    d0, d1 = cld(Int(d), 2), fld(Int(d), 2)
    sp = d1 == 0 ? TK.Vect[I](one(I) => d0) : TK.Vect[I](one(I) => d0, I(1) => d1)
    return Index(sp, tags)
end

slotspace(i::GradedIndex) = i.dual ? TK.dual(space(i)) : space(i)

struct GradedTensor{S <: TKSpace, TM <: TK.AbstractTensorMap} <: AbstractTensor
    inds::Vector{Index{S}}
    data::TM
    function GradedTensor(inds::AbstractVector, data::TK.AbstractTensorMap)
        isempty(inds) && error("GradedTensor: fully-contracted results are plain Numbers")
        S = typeof(space(first(inds)))
        iv = collect(Index{S}, inds)
        TK.numout(data) + TK.numin(data) == length(iv) ||
            error("GradedTensor: $(length(iv)) indices for data with $(TK.numout(data) + TK.numin(data)) legs")
        for (a, i) in enumerate(iv)
            TK.space(data, a) == slotspace(i) ||
                error("GradedTensor: slot $a is $(TK.space(data, a)), index wants $(slotspace(i))")
        end
        return new{S, typeof(data)}(iv, data)
    end
end

_like(t::GradedTensor, inds, data) = GradedTensor(inds, data)

Base.eltype(t::GradedTensor) = TK.scalartype(t.data)

#Normalization functional (BP normalizes messages by sum). Two requirements meet here:
#(a) parity-gauge insensitivity — fermionic messages appear in either parity gauge (the
#odd-sector sign is a gauge choice), so a linear component sum could vanish or flip the
#message sign: use per-tree MAGNITUDES for the size; (b) dephasing — dividing by the sum
#must remove a global phase (messages pick those up from complex scalar rescales; the
#dense backend dephases for free through its linear complex sum): carry the dominant
#tree's phase.
function Base.sum(t::GradedTensor)
    tot = zero(real(eltype(t)))
    lead = zero(eltype(t))
    for (f1, f2) in TK.fusiontrees(t.data)
        s = sum(t.data[f1, f2])
        tot += abs(s)
        abs(s) > abs(lead) && (lead = s)
    end
    return iszero(lead) ? zero(eltype(t)) : (lead / abs(lead)) * tot
end

TensorInterface.datatype(t::GradedTensor) = Vector{TK.scalartype(t.data)}
TensorInterface.data(t::GradedTensor) = t

# ── Charge-ordered basis conversion (tree basis ≡ sector-ordered product basis) ─────────

#Position range of sector `c` within an index, in the base space's own sector order.
#`c` is the sector carried by the tensor's slot: for a dual slot that is the dual
#sector, so map it back to the base space's label first.
function _fock_range(i::GradedIndex, c)
    V = space(i)
    cb = i.dual ? TK.dual(c) : c
    off = 0
    for s in TK.sectors(V)
        d = TK.dim(V, s)
        s == cb && return (off + 1):(off + d)
        off += d
    end
    return error("sector $c not found in $(V)")
end

#Normalize to the all-codomain partition (a copy; used by the basis-sensitive cold
#paths — intermediates from contraction are generally two-sided).
_onesided(t::GradedTensor) = TK.numin(t.data) == 0 ? t :
    GradedTensor(t.inds, TK.permute(t.data, (Tuple(1:ndims(t)), ())))

#NOTE: only trustworthy for all-non-dual tensors (states); dual slots involve duality
#bends whose basis convention is TensorKit-internal. Ops are built two-sided instead.
function TensorInterface.array(t::GradedTensor)
    t = _onesided(t)
    out = zeros(eltype(t), Int[dimof(i) for i in t.inds]...)
    N = ndims(t)
    for (f1, f2) in TK.fusiontrees(t.data)
        rngs = ntuple(a -> _fock_range(t.inds[a], f1.uncoupled[a]), N)
        out[rngs...] = t.data[f1, f2]
    end
    return out
end

function TensorInterface.from_array(A::AbstractArray, is::GradedIndex...)
    iv = collect(Index, is)
    all(i -> !i.dual, iv) ||
        error("from_array: graded scatter is only defined for non-dual indices")
    N = length(iv)
    A = reshape(A, Int[dimof(i) for i in iv]...)
    data = zeros(eltype(A), TK.ProductSpace(map(slotspace, iv)...))
    for (f1, f2) in TK.fusiontrees(data)
        rngs = ntuple(a -> _fock_range(iv[a], f1.uncoupled[a]), N)
        data[f1, f2] .= A[rngs...]
    end
    LinearAlgebra.norm(data) ≈ LinearAlgebra.norm(A) || error(
        "from_array: the array has weight outside the flux-zero sector — graded " *
            "tensors carry zero total charge. Charged product states (e.g. \"Occ\", " *
            "\"↓\" under U1) are not representable per-vertex; start from neutral " *
            "states and use charge-conserving gates (e.g. \"F_pair\" from the vacuum)."
    )
    return GradedTensor(iv, data)
end

# ── Index transforms (labels only; the data never moves) ───────────────────────────────

function TensorInterface.dag(t::GradedTensor)
    N = ndims(t)
    dd = TK.permute(adjoint(t.data), (Tuple(1:N), ()))
    #the adjoint swaps the codomain and domain blocks (order preserved within each), so
    #the index list reorders to match the adjoint's slot numbering
    no = TK.numout(t.data)
    order = no == N ? (1:N) : vcat((no + 1):N, 1:no)
    return GradedTensor(Index[TensorInterface.dag(t.inds[k]) for k in order], dd)
end

#Relabels by identity; the per-copy dual flag is data bookkeeping and is PRESERVED from
#the existing copy (generic code passes replacement labels with arbitrary flags).
function TensorInterface.replaceinds(t::GradedTensor, old, new)
    oldv, newv = _indvec(old), _indvec(new)
    length(oldv) == length(newv) || error("replaceinds: length mismatch")
    newinds = map(t.inds) do i
        k = findfirst(==(i), oldv)
        k === nothing && return i
        n = newv[k]
        space(n) == space(i) || error("replaceinds: space mismatch $(i) → $(n)")
        return Index(n.id, n.space, n.plev, n.tags, i.dual)
    end
    return GradedTensor(newinds, t.data)
end
# ── Arithmetic ──────────────────────────────────────────────────────────────────────────

#b's data with slots permuted into a's index order and partition (TensorKit threads
#the signs)
function _aligned_data(a::GradedTensor, b::GradedTensor)
    nout = TK.numout(a.data)
    p = map(a.inds) do i
        k = findfirst(==(i), b.inds)
        k === nothing && error("tensors do not share index $(i)")
        b.inds[k].dual == i.dual || error("aligned combine: flag mismatch on $(i)")
        k
    end
    return TK.permute(b.data, (Tuple(p[1:nout]), Tuple(p[(nout + 1):end])))
end

Base.:+(a::GradedTensor, b::GradedTensor) = GradedTensor(copy(a.inds), a.data + _aligned_data(a, b))
Base.:-(a::GradedTensor, b::GradedTensor) = GradedTensor(copy(a.inds), a.data - _aligned_data(a, b))

LinearAlgebra.dot(a::GradedTensor, b::GradedTensor) = LinearAlgebra.dot(a.data, _aligned_data(a, b))

function LinearAlgebra.rmul!(t::GradedTensor, x::Number)
    LinearAlgebra.rmul!(t.data, x)
    return t
end

function Base.isapprox(a::GradedTensor, b::GradedTensor; atol = 0, rtol = nothing)
    rt = rtol === nothing ? sqrt(eps(real(promote_type(eltype(a), eltype(b))))) : rtol
    return LinearAlgebra.norm(a - b) <= max(atol, rt * max(LinearAlgebra.norm(a), LinearAlgebra.norm(b)))
end

function Adapt.adapt_structure(elt::Type{<:Number}, t::GradedTensor)
    return GradedTensor(copy(t.inds), one(elt) * t.data)
end

# ── Contraction ─────────────────────────────────────────────────────────────────────────

function Base.:*(a::GradedTensor, b::GradedTensor)
    ca, cb = Int[], Int[]
    for (j, bj) in enumerate(b.inds)
        i = findfirst(==(bj), a.inds)
        i === nothing && continue
        ai = a.inds[i]
        space(ai) == space(bj) || error("contraction: mismatched spaces on $(bj)")
        ai.dual != bj.dual ||
            error("contraction: index $(bj) has the same orientation on both tensors")
        push!(ca, i)
        push!(cb, j)
    end
    oa = setdiff(1:ndims(a), ca)
    ob = setdiff(1:ndims(b), cb)
    la, lb = zeros(Int, ndims(a)), zeros(Int, ndims(b))
    for (x, (i, j)) in enumerate(zip(ca, cb))
        la[i] = x
        lb[j] = x
    end
    for (x, i) in enumerate(oa)
        la[i] = -x
    end
    for (x, j) in enumerate(ob)
        lb[j] = -(length(oa) + x)
    end
    if isempty(oa) && isempty(ob)
        out = TensorOperations.ncon([a.data, b.data], [la, lb])
        return out::Number
    end
    #Low-level contraction with the output allocated directly in the GEMM-natural
    #partition (A-opens ← B-opens): TensorKit then never stages a repacking copy of the
    #output (its worst-case transient footprint drops from 3F to 2F per pairwise step —
    #the repartitioned big operand plus the output). Intermediates stay two-sided; the
    #wrapper's slot numbering spans codomain and domain uniformly.
    pA = (Tuple(oa), Tuple(ca))
    pB = (Tuple(cb), Tuple(ob))
    pAB = (Tuple(1:length(oa)), Tuple((length(oa) + 1):(length(oa) + length(ob))))
    TC = promote_type(eltype(a), eltype(b))
    C = TensorOperations.tensoralloc_contract(TC, a.data, pA, false, b.data, pB, false, pAB, Val(false))
    TensorOperations.tensorcontract!(C, a.data, pA, false, b.data, pB, false, pAB)
    return GradedTensor(vcat(a.inds[oa], b.inds[ob]), C)
end

function TensorInterface.contract(ts::Vector{<:GradedTensor}; sequence = nothing, kwargs...)
    isnothing(sequence) && return reduce(*, ts)
    return _contract_seq_tk(ts, sequence)
end
_contract_seq_tk(ts, s::Integer) = ts[s]
_contract_seq_tk(ts, s::Union{Vector, Tuple}) = mapreduce(x -> _contract_seq_tk(ts, x), *, s)

#Abstractly-typed tensor lists (e.g. from Any-valued network dictionaries) route here
function TensorInterface.contract(ts::Vector; kwargs...)
    all(t -> t isa Tensor, ts) && return TensorInterface.contract(collect(Tensor, ts); kwargs...)
    all(t -> t isa GradedTensor, ts) && return TensorInterface.contract(collect(GradedTensor, ts); kwargs...)
    return error("contract: expected a homogeneous tensor list, got $(unique(typeof.(ts)))")
end


# ── Construction: onehot, delta, state, op, random ──────────────────────────────────────

function TensorInterface.onehot(elt::Type, p::Pair{<:GradedIndex, <:Integer})
    i, v = p
    data = zeros(elt, TK.ProductSpace(slotspace(i)))
    for (f1, f2) in TK.fusiontrees(data)
        r = _fock_range(i, f1.uncoupled[1])
        v ∈ r && (data[f1, f2][v - first(r) + 1] = one(elt))
    end
    return GradedTensor([i], data)
end
TensorInterface.onehot(p::Pair{<:GradedIndex, <:Integer}) = TensorInterface.onehot(Float64, p)

#Identity between a pair of same-space, opposite-orientation indices (same id — BP
#message inits from `delta(vcat(linds, prime(dag(linds))))` — or distinct ids, e.g. the
#loop-correction projectors); longer lists pair up by id.
function _delta_pair(elt::Type, i1::Index, i2::Index)
    space(i1) == space(i2) || error("delta: paired indices must share a space")
    i1.dual != i2.dual || error("delta: paired indices must have opposite orientations")
    #the one-sided bend of the identity has a handedness: building TK.id on the
    #non-dual copy's space is the twist-free insertion on fermionic bonds (the other
    #order picks up a parity twist on odd sectors; both coincide for bosonic spaces)
    a, b = i1.dual ? (i2, i1) : (i1, i2)
    return GradedTensor([a, b], TK.permute(TK.id(elt, slotspace(a)), ((1, 2), ())))
end

function _delta_tk(elt::Type, is::AbstractVector{<:Index})
    length(is) == 2 && return _delta_pair(elt, is[1], is[2])
    ids = unique([i.id for i in is])
    parts = map(ids) do id
        pair = filter(i -> i.id == id, is)
        length(pair) == 2 || error("delta: graded delta needs indices in same-space pairs")
        _delta_pair(elt, pair[1], pair[2])
    end
    return reduce(*, parts)
end
TensorInterface.delta(elt::Type, is::GradedIndex...) = _delta_tk(elt, collect(Index, is))
TensorInterface.delta(is::GradedIndex...) = _delta_tk(Float64, collect(Index, is))

#Combiner: the fuse isometry, TensorKit-native. Mirrors the dense conventions (combined
#index FIRST; `t * C` combines, `x * dag(C)` splits) — the stored copies are dag'd so
#they pair against the caller's own copies.
function TensorInterface.combiner(is::AbstractVector{<:GradedIndex}; tags = "CMB,Link")
    isempty(is) && error("combiner: no indices to combine")
    P = TK.ProductSpace(map(slotspace, is)...)
    iso = TK.isomorphism(Float64, TK.fuse(P), P)
    n = length(is)
    data = TK.permute(iso, (Tuple(1:(n + 1)), ()))
    c = Index(TK.fuse(P), String(tags))
    return GradedTensor(vcat([c], [TensorInterface.dag(i) for i in is]), data)
end
TensorInterface.combiner(is::GradedIndex...; kwargs...) = TensorInterface.combiner(collect(Index, is); kwargs...)
TensorInterface.combinedind(C::GradedTensor) = first(C.inds)

const F_STATES = Dict{String, Vector{Float64}}(
    "0" => [1, 0], "Emp" => [1, 0], "Empty" => [1, 0],
    "1" => [0, 1], "Occ" => [0, 1], "Occupied" => [0, 1],
)

#Spinful (d = 4) fermionic states in the MODE basis (|0⟩, |↑⟩, |↓⟩, |↑↓⟩); permuted to
#the site space's sector-ordered layout by _mode_perm at construction.
const F_STATES_4 = Dict{String, Vector{Float64}}(
    "0" => [1, 0, 0, 0], "Emp" => [1, 0, 0, 0], "Empty" => [1, 0, 0, 0],
    "Up" => [0, 1, 0, 0], "↑" => [0, 1, 0, 0],
    "Dn" => [0, 0, 1, 0], "↓" => [0, 0, 1, 0],
    "UpDn" => [0, 0, 0, 1], "2" => [0, 0, 0, 1],
)

#Resolve a local state (name or raw vector) on a graded site to its dense vector
#(fermionic names for parity sites, the dense registry otherwise).
function state_vector(namevec, i::GradedIndex)
    fermionic = is_fermionic(i)
    vec = if namevec isa AbstractVector{<:Number}
        collect(namevec)
    elseif fermionic
        table = dimof(i) == 2 ? F_STATES : dimof(i) == 4 ? F_STATES_4 :
            error("state: fermionic state library covers d = 2 (spinless) and d = 4 (spinful) sites")
        get(table, String(namevec), nothing)
    else
        TensorInterface.state(String(namevec), Index(dimof(i))).data
    end
    vec === nothing && error(
        "state: unknown fermionic state \"$namevec\" for a d = $(dimof(i)) site"
    )
    length(vec) == dimof(i) ||
        error("state: vector length $(length(vec)) ≠ site dimension $(dimof(i))")
    #fermionic inputs are in the mode basis; reorder into the space's sector layout
    if fermionic
        out = zeros(eltype(vec), dimof(i))
        out[_mode_perm(i)] .= vec
        return out
    end
    return vec
end

#The (single) charge sector a state vector lives in; graded product states must have
#definite local charge.
function vector_sector(vec::AbstractVector, i::GradedIndex)
    secs = [c for c in TK.sectors(space(i)) if any(!iszero, vec[_fock_range(i, c)])]
    length(secs) == 1 || error(
        "state on a graded site must carry a definite charge; found support in sectors $(secs)"
    )
    return only(secs)
end

fuse_sectors(a, b) = only(TK.otimes(a, b))
dual_sector(c) = TK.dual(c)
trivial_sector(c) = one(c)

#A dim-1 link index carrying charge `q` (used for routing charges through product states)
charged_link_index(q; tags = "Link") = Index(TK.Vect[typeof(q)](q => 1), tags)
trivial_link_index(ref::GradedIndex; tags = "Link") = charged_link_index(one(TK.sectortype(space(ref))); tags)

#States scatter the dense state vector; the from_array flux guard rejects charged
#states loudly (as SINGLE tensors — networks route charges through links instead, see
#product_vertex_tensor).
TensorInterface.state(name::String, i::GradedIndex) = TensorInterface.from_array(state_vector(name, i), i)

#Product-state vertex tensor: the site's state vector with dim-1 (possibly charged,
#possibly dual) link legs attached in a single tree assignment — charged legs cannot be
#attached by outer products, since a lone charged leg has no flux-zero trees. Any
#TensorKit-internal phase convention on dual dim-1 slots is a per-bond gauge amounting
#to at most a global phase of the state.
function product_vertex_tensor(elt::Type, vec::AbstractVector, site::GradedIndex, links::AbstractVector{<:Index})
    iv = vcat(Index[site], collect(Index, links))
    all(l -> dimof(l) == 1, links) || error("product_vertex_tensor: links must be dim-1")
    data = zeros(elt, TK.ProductSpace(map(slotspace, iv)...))
    for (f1, f2) in TK.fusiontrees(data)
        data[f1, f2][:] .= elt.(vec[_fock_range(site, f1.uncoupled[1])])
    end
    LinearAlgebra.norm(data) ≈ LinearAlgebra.norm(vec) || error(
        "product_vertex_tensor: the link charges do not neutralize the site charge"
    )
    return GradedTensor(iv, data)
end

# ── Operators: dense operator arrays tree-assigned on two-sided maps ────────────────────

#Fock matrices in the mode basis |0⟩, |1⟩ (site 1 slowest for 2-site ops). These are the
#LOCAL matrices: no Jordan-Wigner strings — the category supplies them under contraction.
const _F_A = ComplexF64[0 1; 0 0]
const _F_ADAG = ComplexF64[0 0; 1 0]
const _F_N = _F_ADAG * _F_A
const _F_I2 = ComplexF64[1 0; 0 1]
#c†₁c₂ + c†₂c₁ on adjacent modes: ⟨10|c†₁c₂|01⟩ = ⟨01|c†₂c₁|10⟩ = 1
const _F_HOP = ComplexF64[0 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 0]
const _F_NN = ComplexF64[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]
#c†₁c†₂ + c₂c₁ on adjacent modes: ⟨11|c†₁c†₂|00⟩ = ⟨00|c₂c₁|11⟩ = 1
const _F_PAIR = ComplexF64[0 0 0 1; 0 0 0 0; 0 0 0 0; 1 0 0 0]

#Spinful (d = 4) single-site matrices, from the fermionic branch: MODE basis
#(|0⟩, |↑⟩, |↓⟩, |↑↓⟩) with the ↑-before-↓ intra-site Jordan-Wigner sign carried by a↓
#(the −1 on |↑↓⟩ → |↑⟩). Permutation into each site space's sector layout happens in
#_graded_op_array via _mode_perm.
const _F4_AUP = ComplexF64[0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
const _F4_ADN = ComplexF64[0 0 1 0; 0 0 0 -1; 0 0 0 0; 0 0 0 0]
const _F4_NUP = _F4_AUP' * _F4_AUP
const _F4_NDN = _F4_ADN' * _F4_ADN
const _F4_I = Matrix{ComplexF64}(LinearAlgebra.I, 4, 4)
#site-parity operator (−1)^n, the intra-block Jordan-Wigner string for 2-site operators
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
    name in ("C", "Cdag", "A", "Adag") && error(
        "op: single-site \"$name\" is parity-odd and needs a charged auxiliary leg " *
            "(not implemented yet); parity-even observables and gates are supported"
    )
    return nothing
end

#Spinful two-site operators: 16×16 matrices over (site1 ⊗ site2) with site 1 slowest.
#The inter-site Jordan-Wigner string is the site-parity _F4_Z on site 1 (intra-site
#signs already live in the mode matrices); strings BETWEEN network sites come from the
#category as usual.
_f4_kron(A, B) = kron(A, B)   #site-1-slowest convention (matches _two_site_array)
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
    #Odd-pair two-point operators, in the two-mode ordered basis of the pair (v, w) —
    #the category threads the string over everything in between (any distance; the BP
    #expect path Steiner-completes the region). Signs from the in-block anticommutation:
    #c_v c†_w = −c†_w c_v etc.
    name == "CdagC" && return ComplexF64[0 0 0 0; 0 0 0 0; 0 1 0 0; 0 0 0 0]   # c†_v c_w
    name == "CCdag" && return ComplexF64[0 0 0 0; 0 0 -1 0; 0 0 0 0; 0 0 0 0]  # c_v c†_w
    name == "CdagCdag" && return ComplexF64[0 0 0 0; 0 0 0 0; 0 0 0 0; 1 0 0 0] # c†_v c†_w
    name == "CC" && return ComplexF64[0 0 0 -1; 0 0 0 0; 0 0 0 0; 0 0 0 0]      # c_v c_w
    name == "F_hop" && return exp(-im * kwargs[:θ] * _F_HOP)
    name == "F_nn" && return exp(-im * kwargs[:θ] * _F_NN)
    name == "F_pair" && return exp(-im * kwargs[:θ] * _F_PAIR)
    if name == "F_hop_nn"
        return exp(-im * (kwargs[:θ] * _F_HOP + kwargs[:ϕ] * _F_NN))
    end
    return nothing
end

#Reorder a site-1-slowest 2-site matrix M[out, in] into the (u1, u2, s1, s2) leg array
#(Julia reshape makes axis 1 fastest, so reshape gives (u2, u1, s2, s1) first).
_two_site_array(M::AbstractMatrix, d1::Int, d2::Int) =
    permutedims(reshape(M, d2, d1, d2, d1), (2, 1, 4, 3))

#The dense operator array ⟨u…|O|s…⟩ with legs (u1..un, s1..sn) for the named operator.
function _graded_op_array(name::String, sites::Vector{<:GradedIndex}; kwargs...)
    if is_fermionic(first(sites))
        d = dimof(first(sites))
        (d in (2, 4) && all(i -> dimof(i) == d, sites)) ||
            error("op: fermionic operator library covers uniform d = 2 (spinless) or d = 4 (spinful) sites")
        A = if length(sites) == 1
            _f_op1_matrix(name, d; kwargs...)
        else
            M = _f_op2_matrix(name, d; kwargs...)
            M === nothing ? nothing : _two_site_array(M, d, d)
        end
        A === nothing && error("op: unknown fermionic operator \"$name\"")
        #tables are in the mode basis; reorder each leg into its site's sector layout
        ips = [invperm(_mode_perm(i)) for i in sites]
        return A[vcat(ips, ips)...]
    end
    #bosonic sectors: the dense registry array is the ground truth
    dense = TensorInterface.op(name, (Index(dimof(i)) for i in sites)...; kwargs...)
    return dense.data
end

#Wrap an operator array A[u..., s...] as a GradedTensor with legs (u..., s...): u = prime(s)
#non-dual OUT legs, s dual IN legs. Built two-sided (outs ← ins) by tree assignment,
#then permuted one-sided (the construction validated against dense Jordan-Wigner).
#Weight outside the conserving blocks errors: a symmetric backend only holds symmetric
#operators (controlled violations are future charged-dummy-leg work).
function _op_from_array(A::AbstractArray, name::String, sites::Vector{<:GradedIndex})
    all(i -> !i.dual, sites) || error("op: expected non-dual site indices")
    n = length(sites)
    P = TK.ProductSpace(map(i -> space(i), sites)...)
    G = zeros(eltype(A), P, P)
    for (f1, f2) in TK.fusiontrees(G)
        rout = ntuple(a -> _fock_range(sites[a], f1.uncoupled[a]), n)
        rin = ntuple(a -> _fock_range(sites[a], f2.uncoupled[a]), n)
        G[f1, f2] .= A[rout..., rin...]
    end
    LinearAlgebra.norm(G) ≈ LinearAlgebra.norm(A) || error(
        "op: \"$name\" has weight outside the conserving blocks — it does not commute " *
            "with the grading on these sites"
    )
    Gc = TK.permute(G, (Tuple(1:(2n)), ()))
    us = [TensorInterface.prime(i) for i in sites]
    ss = [TensorInterface.dag(i) for i in sites]
    return GradedTensor(vcat(us, ss), Gc)
end

function TensorInterface.op(name::String, i::GradedIndex; kwargs...)
    return _op_from_array(_graded_op_array(name, [i]; kwargs...), name, [i])
end
function TensorInterface.op(name::String, i1::GradedIndex, i2::GradedIndex; kwargs...)
    return _op_from_array(_graded_op_array(name, [i1, i2]; kwargs...), name, [i1, i2])
end

#Random tensors: a TensorMap only populates flux-zero trees, so plain randn is already
#the symmetric random initializer.
function TensorInterface.random_itensor(elt::Type{<:Number}, is::GradedIndex...)
    iv = collect(Index, is)
    data = randn(elt, TK.ProductSpace(map(slotspace, iv)...))
    return GradedTensor(iv, data)
end
TensorInterface.random_itensor(is::GradedIndex...) = TensorInterface.random_itensor(Float64, is...)

#Fermionic BP messages carry a per-message parity gauge: m and its parity twist (odd
#sector negated) are equally valid fixed points, and update history determines which
#one BP produces (both appear in practice, always in closure-consistent pairs). For
#operations that need the message as a PSD operator (square roots for gauging), detect
#the gauge from the twist-carrying block trace and twist into the PSD representative —
#the twist cancels in any M^½ · M^{-½} sandwich, so this is exact. Bosonic sectors have
#trivial twists, so psd_gauge is the identity there.
#The gauge freedom is PER SECTOR: scaling sector c of a message by a unit-modulus α_c
#(with 1/α_c on its reverse partner) gives an equally valid fixed point, and update
#history determines which representative BP produces — the fermionic parity twist
#(α_odd = −1) is one instance, complex phases from scalar rescales another. Select the
#PSD representative by normalizing each diagonal block's trace to positive real. Any
#unit-modulus per-sector gauge cancels in the M^½ · M^{-½} sandwich, so this is exact.
function psd_gauge(t::GradedTensor)
    #block diagonals are read in the all-codomain shape; intermediates from contraction
    #may arrive in any partition (messages are small — the copy is cheap)
    t = _onesided(t)
    data = copy(t.data)
    for (f1, f2) in TK.fusiontrees(data)
        b = data[f1, f2]
        z = zero(eltype(t))
        for x in 1:minimum(size(b))
            z += b[x, x]
        end
        iszero(z) || (b .*= conj(z) / abs(z))
    end
    return GradedTensor(copy(t.inds), data)
end

# ── Diagonal operations ─────────────────────────────────────────────────────────────────

function TensorInterface.map_diag!(f::Function, out::GradedTensor, t::GradedTensor)
    ndims(t) == 2 || error("map_diag: expected a 2-index GradedTensor")
    out === t || error("map_diag!: graded backend only supports in-place (out === t)")
    for (f1, f2) in TK.fusiontrees(t.data)
        b = t.data[f1, f2]
        for x in 1:minimum(size(b))
            b[x, x] = f(b[x, x])
        end
    end
    return out
end
function TensorInterface.map_diag(f::Function, t::GradedTensor)
    out = copy(t)
    TensorInterface.map_diag!(f, out, out)
    return out
end

# ── Factorizations (MatrixAlgebraKit API through TensorKit, blockwise with signs) ───────

function _tk_split_positions(t::GradedTensor, lv::Vector{<:Index})
    lpos = Int[a for (a, i) in enumerate(t.inds) if i ∈ lv]
    rpos = Int[a for (a, i) in enumerate(t.inds) if i ∉ lv]
    return lpos, rpos, t.inds[lpos], t.inds[rpos]
end

#Wrap a TensorKit space as (base space, dual flag) for a fresh Index copy.
_wrap_slot(sp) = TK.isdual(sp) ? (TK.dual(sp), true) : (sp, false)

_with_flag(i::GradedIndex, dual::Bool) = Index(i.id, i.space, i.plev, i.tags, dual)

function _tksvd_core(t::GradedTensor, lv::Vector{<:Index}; maxdim = nothing, cutoff = nothing)
    lpos, rpos, li, ri = _tk_split_positions(t, lv)
    tp = TK.permute(t.data, (Tuple(lpos), Tuple(rpos)))
    if maxdim === nothing && cutoff === nothing
        U, S, Vh = svd_compact(tp)
        err = zero(real(eltype(t)))
    else
        strategies = Any[]
        isnothing(maxdim) || push!(strategies, truncrank(Int(maxdim)))
        isnothing(cutoff) || push!(strategies, truncerror(; rtol = sqrt(cutoff), p = 2))
        trunc = length(strategies) == 1 ? only(strategies) : reduce(&, strategies)
        U, S, Vh, err = svd_trunc(tp; trunc)
    end
    kept_s = Float64[]
    for (c, b) in TK.blocks(S)
        append!(kept_s, real.(LinearAlgebra.diag(b)))
    end
    sort!(kept_s; rev = true)
    kept_s2 = kept_s .^ 2
    truncerr = err > 0 ? err^2 / (err^2 + sum(kept_s2)) : zero(Float64)
    return lpos, rpos, li, ri, U, S, Vh, kept_s2, truncerr
end

#Rewrap map factors as GradedTensors: L carries (li..., bond dual), R carries (bond, ri...).
function _tk_wrap_left(U, li::Vector{<:Index}, b::GradedIndex)
    nl = length(li)
    Uc = TK.permute(U, (Tuple(1:(nl + 1)), ()))
    return GradedTensor(vcat(li, [_with_flag(b, true)]), Uc)
end
function _tk_wrap_right(Vh, ri::Vector{<:Index}, b::GradedIndex)
    nr = length(ri)
    Vc = TK.permute(Vh, (Tuple(1:(nr + 1)), ()))
    return GradedTensor(vcat([_with_flag(b, false)], ri), Vc)
end

function _tk_bond_index(S, tags::String)
    sp, isdual = _wrap_slot(TK.space(S, 1))
    isdual && error("factorize: unexpected dual bond space from the factorization")
    return Index(sp, tags)
end

function TensorInterface.factorize_svd(
        t::GradedTensor, linds;
        ortho = "none", singular_values! = nothing,
        maxdim = nothing, cutoff = nothing, kwargs...,
    )
    ortho == "none" || error("factorize_svd: graded backend implements ortho = \"none\"")
    lv = filter(i -> i ∈ t.inds, _indvec(linds))
    _, _, li, ri, U, S, Vh, kept_s2, truncerr = _tksvd_core(t, lv; maxdim, cutoff)
    sq = sqrt(S)
    u = _tk_bond_index(S, "Link,u")
    v = Index(u.space, "Link,v")
    up = TensorInterface.prime(u)
    F1 = _tk_wrap_left(U * sq, li, up)
    F2 = _tk_wrap_right(sq * Vh, ri, up)
    if singular_values! !== nothing
        Sc = TK.permute(S, ((1, 2), ()))
        singular_values![] = GradedTensor([_with_flag(u, false), _with_flag(v, true)], Sc)
    end
    return F1, F2, KSpectrum(kept_s2, truncerr)
end

function LinearAlgebra.factorize(
        t::GradedTensor, linds...;
        ortho = "left", maxdim = nothing, cutoff = nothing, tags = "Link,fact", kwargs...,
    )
    lv = length(linds) == 1 ? _indvec(only(linds)) : collect(Index, linds)
    lv = filter(i -> i ∈ t.inds, lv)
    _, _, li, ri, U, S, Vh, _, _ = _tksvd_core(t, lv; maxdim, cutoff)
    b = _tk_bond_index(S, String(tags))
    if ortho == "left"
        return _tk_wrap_left(U, li, b), _tk_wrap_right(S * Vh, ri, b)
    elseif ortho == "right"
        return _tk_wrap_left(U * S, li, b), _tk_wrap_right(Vh, ri, b)
    else
        error("factorize: unknown ortho = $(ortho)")
    end
end

function LinearAlgebra.svd(t::GradedTensor, linds; maxdim = nothing, cutoff = nothing, kwargs...)
    lv = filter(i -> i ∈ t.inds, _indvec(linds))
    _, _, li, ri, U, S, Vh, _, _ = _tksvd_core(t, lv; maxdim, cutoff)
    u = _tk_bond_index(S, "Link,u")
    v = Index(u.space, "Link,v")
    Ut = _tk_wrap_left(U, li, u)
    Sc = TK.permute(S, ((1, 2), ()))
    St = GradedTensor([_with_flag(u, false), _with_flag(v, true)], Sc)
    #V carries (ri..., v) with the bond last and non-dual
    nr = length(ri)
    Vc = TK.permute(Vh, (Tuple([2:(nr + 1); 1]), ()))
    Vt = GradedTensor(vcat(ri, [_with_flag(v, false)]), Vc)
    return Ut, St, Vt
end

function LinearAlgebra.qr(t::GradedTensor, linds; kwargs...)
    lv = filter(i -> i ∈ t.inds, _indvec(linds))
    lpos, rpos, li, ri = _tk_split_positions(t, lv)
    tp = TK.permute(t.data, (Tuple(lpos), Tuple(rpos)))
    Q, R = qr_compact(tp)
    sp, isdual = _wrap_slot(TK.space(R, 1))
    isdual && error("qr: unexpected dual bond space")
    b = Index(sp, "Link,qr")
    return _tk_wrap_left(Q, li, b), _tk_wrap_right(R, ri, b)
end

#Hermitian eigendecomposition, ITensors-style conventions: D on (prime(lk), lk), U on
#(rinds..., lk). Per-copy flags are read off the actual data slots, so every shared
#identity ends up with opposite orientations on its two holders.
function LinearAlgebra.eigen(t::GradedTensor, linds, rinds; ishermitian::Bool = false, kwargs...)
    ishermitian || error("eigen: only ishermitian = true is implemented for GradedTensors")
    lv, rv = _indvec(linds), _indvec(rinds)
    lpos = Int[findfirst(==(i), t.inds) for i in lv]
    rpos = Int[findfirst(==(i), t.inds) for i in rv]
    #View t as an operator on the linds space: codomain σ(l), domain dual(σ(r)) = σ(l).
    #U is labelled on rinds (the caller relabels) but its slots carry the σ(l) spaces, so
    #U D dag(U) reproduces t's own slot orientations exactly.
    tp = TK.permute(t.data, (Tuple(lpos), Tuple(rpos)))
    H = (tp + adjoint(tp)) / 2
    D, Vec = eigh_full(H)
    lk = Index(_wrap_slot(TK.space(D, 1))[1], "Link,eigen")
    nr = length(rv)
    Uc = TK.permute(Vec, (Tuple(1:(nr + 1)), ()))
    uinds = Index[
        [_with_flag(rv[k], TK.isdual(TK.space(Uc, k))) for k in 1:nr];
        _with_flag(lk, TK.isdual(TK.space(Uc, nr + 1)))
    ]
    U = GradedTensor(uinds, Uc)
    Dc = TK.permute(D, ((1, 2), ()))
    dinds = Index[
        TensorInterface.prime(_with_flag(lk, TK.isdual(TK.space(Dc, 1)))),
        _with_flag(lk, TK.isdual(TK.space(Dc, 2))),
    ]
    return GradedTensor(dinds, Dc), U
end

function LinearAlgebra.eigen(t::GradedTensor; ishermitian::Bool = false, kwargs...)
    lv = filter(i -> i.plev == 0, t.inds)
    rv = collect(Index, TensorInterface.prime.(lv))
    D, U = LinearAlgebra.eigen(t, lv, rv; ishermitian, kwargs...)
    return D, TensorInterface.replaceinds(U, rv, lv)
end

#Projector ⟨v| onto basis state v of site i, as a flux-zero tensor: the site copy is
#dualized, and when the state's sector is nontrivial its charge rides a dim-1 dangling
#"Charge"-tagged leg — paired bra-ket automatically in double-layer networks
#(norm_factors), riding as a spectator through single-layer (amplitude) contractions.
#Configurations whose total charge is wrong then contract to exactly zero.
function TensorInterface.projector(elt::Type, p::Pair{<:GradedIndex, <:Integer})
    i, v = p
    i.dual && error("projector: expected a non-dual (ket) site index")
    id = TensorInterface.dag(i)
    c = only(c for c in TK.sectors(space(i)) if v ∈ _fock_range(i, c))
    c == one(c) && return TensorInterface.onehot(elt, id => v)
    ch = charged_link_index(c; tags = "Charge")
    data = zeros(elt, TK.ProductSpace(slotspace(id), slotspace(ch)))
    for (f1, f2) in TK.fusiontrees(data)
        r = _fock_range(id, f1.uncoupled[1])
        v ∈ r && (data[f1, f2][v - first(r) + 1, 1] = one(elt))
    end
    return GradedTensor([id, ch], data)
end

#Trace of a 2-leg (s, s′) tensor, defined to agree with sum(diag(array(t))) — the
#convention the sampler's probability bookkeeping reads its diagonal in.
function LinearAlgebra.tr(t::GradedTensor)
    ndims(t) == 2 || error("tr: expected a 2-index GradedTensor")
    return sum(LinearAlgebra.diag(TensorInterface.array(t)))
end

#Fully-projected graded networks contract down to spectator dim-1 charge legs rather
#than a bare number; the entry is the amplitude (zero when the total charge is wrong,
#i.e. no flux-zero tree survives).
function TensorInterface.scalar(t::GradedTensor)
    all(i -> dimof(i) == 1, t.inds) ||
        error("scalar: GradedTensor with inds $(t.inds) is not a scalar")
    for (f1, f2) in TK.fusiontrees(t.data)
        return only(t.data[f1, f2])
    end
    return zero(eltype(t))
end

#Purification pairing state Σₛ |s⟩_kets ⊗ ⟨s|_ancillas as a one-sided tensor: the
#ancilla legs carry the DUAL representation (dag'd index copies), which is what makes
#the infinite-temperature identity state flux-zero per site — the data is TensorKit's
#identity map, nothing hand-rolled.
function pairing_tensor(elt::Type, kets, ancs)
    kv, av = collect(Index, kets), collect(Index, ancs)
    length(kv) == length(av) || error("pairing_tensor: need as many ancillas as kets")
    all(i -> !i.dual, kv) && all(i -> i.dual, av) ||
        error("pairing_tensor: kets must be non-dual and ancillas dual copies")
    all(space(kv[a]) == space(av[a]) for a in eachindex(kv)) ||
        error("pairing_tensor: ket/ancilla spaces must match pairwise")
    k = length(kv)
    P = TK.ProductSpace(map(space, kv)...)
    data = TK.permute(TK.id(elt, P), (Tuple(1:(2k)), ()))
    return GradedTensor(vcat(kv, av), data)
end

#A tensor anchors a parity-gauge line iff it carries a "Charge"-tagged dangling leg
#whose sector has a nontrivial fermionic twist (odd total fermion number): multi-vertex
#closures of regions containing such a tensor pick up a gauge sign, so additive
#closure-derived quantities (loop-correction weights) must normalize by a baseline
#closure of the same region. Bosonic charge legs (net U(1) magnetization etc.) have
#trivial twists and are exempt.
function TensorInterface.has_closure_gauge(t::GradedTensor)
    return any(t.inds) do i
        occursin("Charge", i.tags) &&
            any(c -> real(TK.twist(c)) < 0, TK.sectors(space(i)))
    end
end

#Fit adjoint for boundary-MPS bra-rail tensors: dag with the parity/supertrace twist
#applied ONLY on the given (crossing) legs — where the bra rail closes against the ket
#across a physical bond — and among those only on legs whose ORIGINAL arrow was
#outgoing (non-dual; after the dag flip they are the dual slots). Never on the virtual
#MPS bonds, which the Euclidean QR orthogonalises: a metric there, or on the wrong leg
#subset, leaves the two fitting sweep directions inconsistent and the alternating
#iteration converges to a wrong fixed point. Trivial for bosonic sectors. This is the
#fermionic-branch recipe, with TensorKit's twist supplying the metric diag((−1)^p).
function fit_adjoint(t::GradedTensor, metric_legs)
    td = TensorInterface.dag(t)
    slots = Tuple(a for (a, i) in enumerate(td.inds) if i.dual && i ∈ metric_legs)
    isempty(slots) && return td
    return GradedTensor(td.inds, TK.twist(td.data, slots))
end

# ── Boundary-MPS link-sector allocation (init recipe; used by boundarympscache.jl) ──────

#Charge spectrum reachable by one message site: the convolution of its legs' carried
#sector spectra (dual legs contribute dual sectors), weight ∝ sector dimension.
function site_charge_spectrum(m::GradedTensor)
    I = TK.sectortype(space(first(m.inds)))
    w = Dict{I, Float64}(one(I) => 1.0)
    for l in m.inds
        V = space(l)
        wl = Dict{I, Float64}()
        for c in TK.sectors(V)
            cc = l.dual ? TK.dual(c) : c
            wl[cc] = get(wl, cc, 0.0) + TK.dim(V, c) / TK.dim(V)
        end
        w = convolve_charge_spectra(w, wl)
    end
    return w
end

function convolve_charge_spectra(w1::Dict{I, Float64}, w2::Dict{I, Float64}) where {I}
    out = Dict{I, Float64}()
    for (q1, a) in w1, (q2, b) in w2
        q = only(TK.otimes(q1, q2))   #abelian fusion: a single outcome
        out[q] = get(out, q, 0.0) + a * b
    end
    return out
end

#A link space of total dimension `d` supporting the sectors reachable from the left
#(spectrum `wl`) that can be neutralized from the right (spectrum `wr`), with dimensions
#∝ joint weight (largest-remainder rounding; every supported sector keeps ≥ 1 state).
function allocate_link_space(wl::Dict{I, Float64}, wr::Dict{I, Float64}, d::Integer) where {I}
    w = Dict{I, Float64}()
    for (q, a) in wl
        b = get(wr, TK.dual(q), 0.0)
        b > 0 && (w[q] = a * b)
    end
    isempty(w) && (w = Dict(one(I) => 1.0))
    qs = collect(keys(w))
    length(qs) >= d && begin
        sort!(qs; by = q -> -w[q])
        return TK.Vect[I]((q => 1 for q in qs[1:Int(d)])...)
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
    return TK.Vect[I]((q => alloc[q] for q in qs)...)
end
