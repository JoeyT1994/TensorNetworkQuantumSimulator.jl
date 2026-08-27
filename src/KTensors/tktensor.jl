# Symmetric backend: KIndex-labelled wrapper around a TensorKit TensorMap over any
# graded (sector-carrying) space — bosonic Z2/U(1) irreps and fermionic parity alike.
#
# HARD RULE: nothing algebraic is implemented here. Contraction, permutation (including
# every Koszul sign for fermionic sectors — Jordan-Wigner strings emerge from the
# braiding), blockwise factorizations and truncation all delegate to TensorKit and
# MatrixAlgebraKit. This file is bookkeeping between KIndex labels and TensorMap slots,
# plus the operator/state quack layer.
#
# Conventions (validated against dense ground truth — Jordan-Wigner for fermions, the
# dense KTensor backend for bosonic sectors):
#   * Slot `a` of the data holds `slotspace(inds[a])`: the index's base space, dualed
#     when the per-copy `dual` flag is set. The flag is per tensor copy — the same
#     KIndex identity (id, plev) appears with opposite flags on the two tensors sharing
#     a bond (src = false, dst = true), and `dag` flips all flags.
#   * `dag` is the plain categorical adjoint permuted back to the codomain — no twists.
#     Site legs are non-dual on kets; with that convention closed bra-ket networks
#     evaluate to physical inner products.
#   * Operators are built by fusion-tree assignment of the dense operator array
#     ⟨u…|O|s…⟩ on a two-sided (outs ← ins) map, then permuted one-sided. For abelian
#     sectors the tree basis coincides with the charge-ordered product basis. Elements
#     outside the conserving blocks error loudly (that is the point of the symmetry);
#     controlled violations are future work via explicitly tracked charged dummy legs.
#   * TensorMaps enforce zero total flux. Flux-odd product states (e.g. an occupied
#     fermion site) are not representable per-vertex — start from even states and use
#     conserving gates (e.g. pair creation from the vacuum).
#   * Every contraction requires the two copies of a contracted index to carry opposite
#     flags; a same-flag pairing is a network-construction bug and errors loudly.

const TKSpace = TK.GradedSpace
const TKIndex = KIndex{<:TKSpace}

"""
    graded_space(symmetry::String, sectors)

A TensorKit graded space from `charge => dimension` pairs. `symmetry` is one of
`"Z2"`, `"U1"`, `"fZ2"` (fermionic parity; aliases `"fermion"`, `"fermionparity"`).
"""
function graded_space(symmetry::String, sectors)
    key = replace(lowercase(symmetry), " " => "")
    I = key in ("z2",) ? TK.Z2Irrep :
        key in ("u1", "u(1)") ? TK.U1Irrep :
        key in ("fz2", "fermion", "fermionparity") ? TK.FermionParity :
        error("graded_space: unknown symmetry \"$symmetry\" (supported: Z2, U1, fZ2)")
    return TK.Vect[I]((q => Int(d) for (q, d) in sectors)...)
end

fermion_space(d0::Integer = 1, d1::Integer = 1) = graded_space("fZ2", [0 => d0, 1 => d1])

KIndex(space::TKSpace, tags::AbstractString = "") =
    KIndex(rand(UInt64), space, 0, String(tags), false)

"""
    new_fermion_index(d0 = 1, d1 = 1; tags = "")

A fresh fermionic (Z2-parity graded) index with `d0` even and `d1` odd states.
"""
new_fermion_index(d0::Integer = 1, d1::Integer = 1; tags = "") =
    KIndex(fermion_space(d0, d1), tags)

space_dim(s::TKSpace) = TK.dim(s)

#Trivial-dominant split for a fresh link of total dimension d (fermion-branch recipe:
#double-layer networks are trivial-sector dominant, so links need at least as many
#trivial-sector states as charged ones).
function TensorInterface.new_index(ref::TKIndex, d::Integer; tags = "")
    I = TK.sectortype(space(ref))
    d0, d1 = cld(Int(d), 2), fld(Int(d), 2)
    sp = d1 == 0 ? TK.Vect[I](one(I) => d0) : TK.Vect[I](one(I) => d0, I(1) => d1)
    return KIndex(sp, tags)
end

slotspace(i::TKIndex) = i.dual ? TK.dual(space(i)) : space(i)

struct TKTensor{S <: TKSpace, TM <: TK.AbstractTensorMap}
    inds::Vector{KIndex{S}}
    data::TM
    function TKTensor(inds::AbstractVector, data::TK.AbstractTensorMap)
        isempty(inds) && error("TKTensor: fully-contracted results are plain Numbers")
        S = typeof(space(first(inds)))
        iv = collect(KIndex{S}, inds)
        TK.numout(data) == length(iv) && TK.numin(data) == 0 ||
            error("TKTensor: data must be one-sided with $(length(iv)) codomain legs")
        for (a, i) in enumerate(iv)
            TK.space(data, a) == slotspace(i) ||
                error("TKTensor: slot $a is $(TK.space(data, a)), index wants $(slotspace(i))")
        end
        return new{S, typeof(data)}(iv, data)
    end
end

Base.eltype(t::TKTensor) = TK.scalartype(t.data)
Base.ndims(t::TKTensor) = length(t.inds)
Base.copy(t::TKTensor) = TKTensor(copy(t.inds), copy(t.data))

#Normalization functional (BP normalizes messages by sum). Two requirements meet here:
#(a) parity-gauge insensitivity — fermionic messages appear in either parity gauge (the
#odd-sector sign is a gauge choice), so a linear component sum could vanish or flip the
#message sign: use per-tree MAGNITUDES for the size; (b) dephasing — dividing by the sum
#must remove a global phase (messages pick those up from complex scalar rescales; the
#dense backend dephases for free through its linear complex sum): carry the dominant
#tree's phase.
function Base.sum(t::TKTensor)
    tot = zero(real(eltype(t)))
    lead = zero(eltype(t))
    for (f1, f2) in TK.fusiontrees(t.data)
        s = sum(t.data[f1, f2])
        tot += abs(s)
        abs(s) > abs(lead) && (lead = s)
    end
    return iszero(lead) ? zero(eltype(t)) : (lead / abs(lead)) * tot
end

function Base.show(io::IO, t::TKTensor)
    return print(io, "TKTensor{$(eltype(t))} inds=", t.inds)
end

function TensorInterface.inds(t::TKTensor; plev = nothing)
    plev === nothing && return t.inds
    return filter(i -> i.plev == plev, t.inds)
end

TensorInterface.scalartype(t::TKTensor) = TK.scalartype(t.data)
TensorInterface.datatype(t::TKTensor) = Vector{TK.scalartype(t.data)}
TensorInterface.data(t::TKTensor) = t

# ── Charge-ordered basis conversion (tree basis ≡ sector-ordered product basis) ─────────

#Position range of sector `c` within an index, in the base space's own sector order.
#`c` is the sector carried by the tensor's slot: for a dual slot that is the dual
#sector, so map it back to the base space's label first.
function _fock_range(i::TKIndex, c)
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

#NOTE: only trustworthy for all-non-dual tensors (states); dual slots involve duality
#bends whose basis convention is TensorKit-internal. Ops are built two-sided instead.
function TensorInterface.array(t::TKTensor)
    out = zeros(eltype(t), Int[dimof(i) for i in t.inds]...)
    N = ndims(t)
    for (f1, f2) in TK.fusiontrees(t.data)
        rngs = ntuple(a -> _fock_range(t.inds[a], f1.uncoupled[a]), N)
        out[rngs...] = t.data[f1, f2]
    end
    return out
end

function TensorInterface.from_array(A::AbstractArray, is::TKIndex...)
    iv = collect(KIndex, is)
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
    return TKTensor(iv, data)
end

# ── Index transforms (labels only; the data never moves) ───────────────────────────────

_mapinds(f, t::TKTensor) = TKTensor(map(f, t.inds), t.data)

TensorInterface.prime(t::TKTensor, n::Integer = 1) = _mapinds(i -> TensorInterface.prime(i, n), t)
TensorInterface.noprime(t::TKTensor) = _mapinds(TensorInterface.noprime, t)
TensorInterface.sim(t::TKTensor) = _mapinds(TensorInterface.sim, t)

function TensorInterface.dag(t::TKTensor)
    N = ndims(t)
    dd = TK.permute(adjoint(t.data), (Tuple(1:N), ()))
    return TKTensor(KIndex[TensorInterface.dag(i) for i in t.inds], dd)
end

#Relabels by identity; the per-copy dual flag is data bookkeeping and is PRESERVED from
#the existing copy (generic code passes replacement labels with arbitrary flags).
function TensorInterface.replaceinds(t::TKTensor, old, new)
    oldv, newv = _indvec(old), _indvec(new)
    length(oldv) == length(newv) || error("replaceinds: length mismatch")
    newinds = map(t.inds) do i
        k = findfirst(==(i), oldv)
        k === nothing && return i
        n = newv[k]
        space(n) == space(i) || error("replaceinds: space mismatch $(i) → $(n)")
        return KIndex(n.id, n.space, n.plev, n.tags, i.dual)
    end
    return TKTensor(newinds, t.data)
end
TensorInterface.replaceind(t::TKTensor, old::KIndex, new::KIndex) = TensorInterface.replaceinds(t, [old], [new])
TensorInterface.replaceinds(t::TKTensor, p::Pair) = TensorInterface.replaceinds(t, first(p), last(p))

_indvec(t::TKTensor) = t.inds

for f in [:commoninds, :commonind, :uniqueinds, :unioninds, :noncommoninds, :noncommonind, :hascommoninds]
    @eval begin
        TensorInterface.$f(a::TKTensor, b) = TensorInterface.$f(a.inds, _indvec(b))
        TensorInterface.$f(a, b::TKTensor) = TensorInterface.$f(_indvec(a), b.inds)
        TensorInterface.$f(a::TKTensor, b::TKTensor) = TensorInterface.$f(a.inds, b.inds)
    end
end

# ── Arithmetic ──────────────────────────────────────────────────────────────────────────

Base.:*(t::TKTensor, x::Number) = TKTensor(copy(t.inds), t.data * x)
Base.:*(x::Number, t::TKTensor) = t * x
Base.:/(t::TKTensor, x::Number) = t * inv(x)

#b's data with slots permuted into a's index order (TensorKit threads the signs)
function _aligned_data(a::TKTensor, b::TKTensor)
    p = map(a.inds) do i
        k = findfirst(==(i), b.inds)
        k === nothing && error("tensors do not share index $(i)")
        b.inds[k].dual == i.dual || error("aligned combine: flag mismatch on $(i)")
        k
    end
    return TK.permute(b.data, (Tuple(p), ()))
end

Base.:+(a::TKTensor, b::TKTensor) = TKTensor(copy(a.inds), a.data + _aligned_data(a, b))
Base.:-(a::TKTensor, b::TKTensor) = TKTensor(copy(a.inds), a.data - _aligned_data(a, b))

LinearAlgebra.norm(t::TKTensor) = LinearAlgebra.norm(t.data)
LinearAlgebra.dot(a::TKTensor, b::TKTensor) = LinearAlgebra.dot(a.data, _aligned_data(a, b))

function LinearAlgebra.rmul!(t::TKTensor, x::Number)
    LinearAlgebra.rmul!(t.data, x)
    return t
end

function Base.isapprox(a::TKTensor, b::TKTensor; atol = 0, rtol = nothing)
    rt = rtol === nothing ? sqrt(eps(real(promote_type(eltype(a), eltype(b))))) : rtol
    return LinearAlgebra.norm(a - b) <= max(atol, rt * max(LinearAlgebra.norm(a), LinearAlgebra.norm(b)))
end

function Adapt.adapt_structure(elt::Type{<:Number}, t::TKTensor)
    return TKTensor(copy(t.inds), one(elt) * t.data)
end

# ── Contraction ─────────────────────────────────────────────────────────────────────────

function Base.:*(a::TKTensor, b::TKTensor)
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
    out = TensorOperations.ncon([a.data, b.data], [la, lb])
    out isa Number && return out
    oinds = vcat(a.inds[oa], b.inds[ob])
    return TKTensor(oinds, TK.permute(out, (Tuple(1:length(oinds)), ())))
end

function TensorInterface.contract(ts::Vector{<:TKTensor}; sequence = nothing, kwargs...)
    isnothing(sequence) && return reduce(*, ts)
    return _contract_seq_tk(ts, sequence)
end
_contract_seq_tk(ts, s::Integer) = ts[s]
_contract_seq_tk(ts, s::Union{Vector, Tuple}) = mapreduce(x -> _contract_seq_tk(ts, x), *, s)

#Abstractly-typed tensor lists (e.g. from Any-valued network dictionaries) route here
function TensorInterface.contract(ts::Vector; kwargs...)
    all(t -> t isa KTensor, ts) && return TensorInterface.contract(collect(KTensor, ts); kwargs...)
    all(t -> t isa TKTensor, ts) && return TensorInterface.contract(collect(TKTensor, ts); kwargs...)
    return error("contract: expected a homogeneous tensor list, got $(unique(typeof.(ts)))")
end

TensorInterface.apply(o::TKTensor, t::TKTensor) = TensorInterface.noprime(o * t)

# ── Construction: onehot, delta, state, op, random ──────────────────────────────────────

function TensorInterface.onehot(elt::Type, p::Pair{<:TKIndex, <:Integer})
    i, v = p
    data = zeros(elt, TK.ProductSpace(slotspace(i)))
    for (f1, f2) in TK.fusiontrees(data)
        r = _fock_range(i, f1.uncoupled[1])
        v ∈ r && (data[f1, f2][v - first(r) + 1] = one(elt))
    end
    return TKTensor([i], data)
end
TensorInterface.onehot(p::Pair{<:TKIndex, <:Integer}) = TensorInterface.onehot(Float64, p)

#Identity messages: indices come in (i, i′)-style pairs sharing an id, with opposite
#per-copy flags (e.g. from `delta(vcat(linds, prime(dag(linds))))`).
function _delta_tk(elt::Type, is::AbstractVector{<:KIndex})
    ids = unique([i.id for i in is])
    parts = map(ids) do id
        pair = filter(i -> i.id == id, is)
        length(pair) == 2 || error("delta: graded delta needs indices in (i, i′)-style pairs")
        i1, i2 = pair
        space(i1) == space(i2) || error("delta: paired indices must share a space")
        i1.dual != i2.dual || error("delta: paired indices must have opposite orientations")
        data = TK.permute(TK.id(elt, slotspace(i1)), ((1, 2), ()))
        TKTensor([i1, i2], data)
    end
    return reduce(*, parts)
end
TensorInterface.delta(elt::Type, is::TKIndex...) = _delta_tk(elt, collect(KIndex, is))
TensorInterface.delta(is::TKIndex...) = _delta_tk(Float64, collect(KIndex, is))

const F_STATES = Dict{String, Vector{Float64}}(
    "0" => [1, 0], "Emp" => [1, 0], "Empty" => [1, 0],
    "1" => [0, 1], "Occ" => [0, 1], "Occupied" => [0, 1],
)

#Resolve a local state (name or raw vector) on a graded site to its dense vector
#(fermionic names for parity sites, the dense registry otherwise).
function state_vector(namevec, i::TKIndex)
    vec = if namevec isa AbstractVector{<:Number}
        collect(namevec)
    elseif TK.sectortype(space(i)) === TK.FermionParity
        dimof(i) == 2 || error("state: fermionic state library covers d = 2 sites only")
        get(F_STATES, String(namevec), nothing)
    else
        TensorInterface.state(String(namevec), KIndex(dimof(i))).data
    end
    vec === nothing && error(
        "state: unknown fermionic state \"$namevec\" (available: $(sort(collect(keys(F_STATES)))))"
    )
    length(vec) == dimof(i) ||
        error("state: vector length $(length(vec)) ≠ site dimension $(dimof(i))")
    return vec
end

#The (single) charge sector a state vector lives in; graded product states must have
#definite local charge.
function vector_sector(vec::AbstractVector, i::TKIndex)
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
charged_link_index(q; tags = "Link") = KIndex(TK.Vect[typeof(q)](q => 1), tags)

#States scatter the dense state vector; the from_array flux guard rejects charged
#states loudly (as SINGLE tensors — networks route charges through links instead, see
#product_vertex_tensor).
TensorInterface.state(name::String, i::TKIndex) = TensorInterface.from_array(state_vector(name, i), i)

#Product-state vertex tensor: the site's state vector with dim-1 (possibly charged,
#possibly dual) link legs attached in a single tree assignment — charged legs cannot be
#attached by outer products, since a lone charged leg has no flux-zero trees. Any
#TensorKit-internal phase convention on dual dim-1 slots is a per-bond gauge amounting
#to at most a global phase of the state.
function product_vertex_tensor(elt::Type, vec::AbstractVector, site::TKIndex, links::AbstractVector{<:KIndex})
    iv = vcat(KIndex[site], collect(KIndex, links))
    all(l -> dimof(l) == 1, links) || error("product_vertex_tensor: links must be dim-1")
    data = zeros(elt, TK.ProductSpace(map(slotspace, iv)...))
    for (f1, f2) in TK.fusiontrees(data)
        data[f1, f2][:] .= elt.(vec[_fock_range(site, f1.uncoupled[1])])
    end
    LinearAlgebra.norm(data) ≈ LinearAlgebra.norm(vec) || error(
        "product_vertex_tensor: the link charges do not neutralize the site charge"
    )
    return TKTensor(iv, data)
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

function _f_op1_matrix(name::String; kwargs...)
    name == "I" && return _F_I2
    name == "N" && return _F_N
    name == "F_phase" && return ComplexF64[1 0; 0 exp(-im * kwargs[:θ])]
    name in ("C", "Cdag", "A", "Adag") && error(
        "op: single-site \"$name\" is parity-odd and needs a charged auxiliary leg " *
            "(not implemented yet); parity-even observables and gates are supported"
    )
    return nothing
end

function _f_op2_matrix(name::String; kwargs...)
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
function _graded_op_array(name::String, sites::Vector{<:TKIndex}; kwargs...)
    if TK.sectortype(space(first(sites))) === TK.FermionParity
        all(i -> dimof(i) == 2, sites) ||
            error("op: fermionic operator library covers d = 2 sites only")
        if length(sites) == 1
            M = _f_op1_matrix(name; kwargs...)
            M === nothing || return M
        else
            M = _f_op2_matrix(name; kwargs...)
            M === nothing || return _two_site_array(M, 2, 2)
        end
        error("op: unknown fermionic operator \"$name\"")
    end
    #bosonic sectors: the dense registry array is the ground truth
    dense = TensorInterface.op(name, (KIndex(dimof(i)) for i in sites)...; kwargs...)
    return dense.data
end

#Wrap an operator array A[u..., s...] as a TKTensor with legs (u..., s...): u = prime(s)
#non-dual OUT legs, s dual IN legs. Built two-sided (outs ← ins) by tree assignment,
#then permuted one-sided (the construction validated against dense Jordan-Wigner).
#Weight outside the conserving blocks errors: a symmetric backend only holds symmetric
#operators (controlled violations are future charged-dummy-leg work).
function _op_from_array(A::AbstractArray, name::String, sites::Vector{<:TKIndex})
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
    return TKTensor(vcat(us, ss), Gc)
end

function TensorInterface.op(name::String, i::TKIndex; kwargs...)
    return _op_from_array(_graded_op_array(name, [i]; kwargs...), name, [i])
end
function TensorInterface.op(name::String, i1::TKIndex, i2::TKIndex; kwargs...)
    return _op_from_array(_graded_op_array(name, [i1, i2]; kwargs...), name, [i1, i2])
end

#Random tensors: a TensorMap only populates flux-zero trees, so plain randn is already
#the symmetric random initializer.
function TensorInterface.random_itensor(elt::Type{<:Number}, is::TKIndex...)
    iv = collect(KIndex, is)
    data = randn(elt, TK.ProductSpace(map(slotspace, iv)...))
    return TKTensor(iv, data)
end
TensorInterface.random_itensor(is::TKIndex...) = TensorInterface.random_itensor(Float64, is...)

#Fermionic BP messages carry a per-message parity gauge: m and its parity twist (odd
#sector negated) are equally valid fixed points, and update history determines which
#one BP produces (both appear in practice, always in closure-consistent pairs). For
#operations that need the message as a PSD operator (square roots for gauging), detect
#the gauge from the twist-carrying block trace and twist into the PSD representative —
#the twist cancels in any M^½ · M^{-½} sandwich, so this is exact. Bosonic sectors have
#trivial twists, so psd_gauge is the identity there.
parity_twisted(t::TKTensor) = TKTensor(copy(t.inds), TK.twist(t.data, 1))

#The gauge freedom is PER SECTOR: scaling sector c of a message by a unit-modulus α_c
#(with 1/α_c on its reverse partner) gives an equally valid fixed point, and update
#history determines which representative BP produces — the fermionic parity twist
#(α_odd = −1) is one instance, complex phases from scalar rescales another. Select the
#PSD representative by normalizing each diagonal block's trace to positive real. Any
#unit-modulus per-sector gauge cancels in the M^½ · M^{-½} sandwich, so this is exact.
function psd_gauge(t::TKTensor)
    data = copy(t.data)
    for (f1, f2) in TK.fusiontrees(data)
        b = data[f1, f2]
        z = zero(eltype(t))
        for x in 1:minimum(size(b))
            z += b[x, x]
        end
        iszero(z) || (b .*= conj(z) / abs(z))
    end
    return TKTensor(copy(t.inds), data)
end

# ── Diagonal operations ─────────────────────────────────────────────────────────────────

function TensorInterface.map_diag!(f::Function, out::TKTensor, t::TKTensor)
    ndims(t) == 2 || error("map_diag: expected a 2-index TKTensor")
    out === t || error("map_diag!: graded backend only supports in-place (out === t)")
    for (f1, f2) in TK.fusiontrees(t.data)
        b = t.data[f1, f2]
        for x in 1:minimum(size(b))
            b[x, x] = f(b[x, x])
        end
    end
    return out
end
function TensorInterface.map_diag(f::Function, t::TKTensor)
    out = copy(t)
    TensorInterface.map_diag!(f, out, out)
    return out
end

# ── Factorizations (MatrixAlgebraKit API through TensorKit, blockwise with signs) ───────

function _tk_split_positions(t::TKTensor, lv::Vector{<:KIndex})
    lpos = Int[a for (a, i) in enumerate(t.inds) if i ∈ lv]
    rpos = Int[a for (a, i) in enumerate(t.inds) if i ∉ lv]
    return lpos, rpos, t.inds[lpos], t.inds[rpos]
end

#Wrap a TensorKit space as (base space, dual flag) for a fresh KIndex copy.
_wrap_slot(sp) = TK.isdual(sp) ? (TK.dual(sp), true) : (sp, false)

_with_flag(i::TKIndex, dual::Bool) = KIndex(i.id, i.space, i.plev, i.tags, dual)

function _tksvd_core(t::TKTensor, lv::Vector{<:KIndex}; maxdim = nothing, cutoff = nothing)
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

#Rewrap map factors as TKTensors: L carries (li..., bond dual), R carries (bond, ri...).
function _tk_wrap_left(U, li::Vector{<:KIndex}, b::TKIndex)
    nl = length(li)
    Uc = TK.permute(U, (Tuple(1:(nl + 1)), ()))
    return TKTensor(vcat(li, [_with_flag(b, true)]), Uc)
end
function _tk_wrap_right(Vh, ri::Vector{<:KIndex}, b::TKIndex)
    nr = length(ri)
    Vc = TK.permute(Vh, (Tuple(1:(nr + 1)), ()))
    return TKTensor(vcat([_with_flag(b, false)], ri), Vc)
end

function _tk_bond_index(S, tags::String)
    sp, isdual = _wrap_slot(TK.space(S, 1))
    isdual && error("factorize: unexpected dual bond space from the factorization")
    return KIndex(sp, tags)
end

function TensorInterface.factorize_svd(
        t::TKTensor, linds;
        ortho = "none", singular_values! = nothing,
        maxdim = nothing, cutoff = nothing, kwargs...,
    )
    ortho == "none" || error("factorize_svd: graded backend implements ortho = \"none\"")
    lv = filter(i -> i ∈ t.inds, _indvec(linds))
    _, _, li, ri, U, S, Vh, kept_s2, truncerr = _tksvd_core(t, lv; maxdim, cutoff)
    sq = sqrt(S)
    u = _tk_bond_index(S, "Link,u")
    v = KIndex(u.space, "Link,v")
    up = TensorInterface.prime(u)
    F1 = _tk_wrap_left(U * sq, li, up)
    F2 = _tk_wrap_right(sq * Vh, ri, up)
    if singular_values! !== nothing
        Sc = TK.permute(S, ((1, 2), ()))
        singular_values![] = TKTensor([_with_flag(u, false), _with_flag(v, true)], Sc)
    end
    return F1, F2, KSpectrum(kept_s2, truncerr)
end

function LinearAlgebra.factorize(
        t::TKTensor, linds...;
        ortho = "left", maxdim = nothing, cutoff = nothing, tags = "Link,fact", kwargs...,
    )
    lv = length(linds) == 1 ? _indvec(only(linds)) : collect(KIndex, linds)
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

function LinearAlgebra.svd(t::TKTensor, linds; maxdim = nothing, cutoff = nothing, kwargs...)
    lv = filter(i -> i ∈ t.inds, _indvec(linds))
    _, _, li, ri, U, S, Vh, _, _ = _tksvd_core(t, lv; maxdim, cutoff)
    u = _tk_bond_index(S, "Link,u")
    v = KIndex(u.space, "Link,v")
    Ut = _tk_wrap_left(U, li, u)
    Sc = TK.permute(S, ((1, 2), ()))
    St = TKTensor([_with_flag(u, false), _with_flag(v, true)], Sc)
    #V carries (ri..., v) with the bond last and non-dual
    nr = length(ri)
    Vc = TK.permute(Vh, (Tuple([2:(nr + 1); 1]), ()))
    Vt = TKTensor(vcat(ri, [_with_flag(v, false)]), Vc)
    return Ut, St, Vt
end

function LinearAlgebra.qr(t::TKTensor, linds; kwargs...)
    lv = filter(i -> i ∈ t.inds, _indvec(linds))
    lpos, rpos, li, ri = _tk_split_positions(t, lv)
    tp = TK.permute(t.data, (Tuple(lpos), Tuple(rpos)))
    Q, R = qr_compact(tp)
    sp, isdual = _wrap_slot(TK.space(R, 1))
    isdual && error("qr: unexpected dual bond space")
    b = KIndex(sp, "Link,qr")
    return _tk_wrap_left(Q, li, b), _tk_wrap_right(R, ri, b)
end

#Hermitian eigendecomposition, ITensors-style conventions: D on (prime(lk), lk), U on
#(rinds..., lk). Per-copy flags are read off the actual data slots, so every shared
#identity ends up with opposite orientations on its two holders.
function LinearAlgebra.eigen(t::TKTensor, linds, rinds; ishermitian::Bool = false, kwargs...)
    ishermitian || error("eigen: only ishermitian = true is implemented for TKTensors")
    lv, rv = _indvec(linds), _indvec(rinds)
    lpos = Int[findfirst(==(i), t.inds) for i in lv]
    rpos = Int[findfirst(==(i), t.inds) for i in rv]
    #View t as an operator on the linds space: codomain σ(l), domain dual(σ(r)) = σ(l).
    #U is labelled on rinds (the caller relabels) but its slots carry the σ(l) spaces, so
    #U D dag(U) reproduces t's own slot orientations exactly.
    tp = TK.permute(t.data, (Tuple(lpos), Tuple(rpos)))
    H = (tp + adjoint(tp)) / 2
    D, Vec = eigh_full(H)
    lk = KIndex(_wrap_slot(TK.space(D, 1))[1], "Link,eigen")
    nr = length(rv)
    Uc = TK.permute(Vec, (Tuple(1:(nr + 1)), ()))
    uinds = KIndex[
        [_with_flag(rv[k], TK.isdual(TK.space(Uc, k))) for k in 1:nr];
        _with_flag(lk, TK.isdual(TK.space(Uc, nr + 1)))
    ]
    U = TKTensor(uinds, Uc)
    Dc = TK.permute(D, ((1, 2), ()))
    dinds = KIndex[
        TensorInterface.prime(_with_flag(lk, TK.isdual(TK.space(Dc, 1)))),
        _with_flag(lk, TK.isdual(TK.space(Dc, 2))),
    ]
    return TKTensor(dinds, Dc), U
end

function LinearAlgebra.eigen(t::TKTensor; ishermitian::Bool = false, kwargs...)
    lv = filter(i -> i.plev == 0, t.inds)
    rv = collect(KIndex, TensorInterface.prime.(lv))
    D, U = LinearAlgebra.eigen(t, lv, rv; ishermitian, kwargs...)
    return D, TensorInterface.replaceinds(U, rv, lv)
end

#Fit adjoint for boundary-MPS bra-rail tensors: dag with the parity/supertrace twist
#applied ONLY on the given (crossing) legs — where the bra rail closes against the ket
#across a physical bond — and among those only on legs whose ORIGINAL arrow was
#outgoing (non-dual; after the dag flip they are the dual slots). Never on the virtual
#MPS bonds, which the Euclidean QR orthogonalises: a metric there, or on the wrong leg
#subset, leaves the two fitting sweep directions inconsistent and the alternating
#iteration converges to a wrong fixed point. Trivial for bosonic sectors. This is the
#fermionic-branch recipe, with TensorKit's twist supplying the metric diag((−1)^p).
function fit_adjoint(t::TKTensor, metric_legs)
    td = TensorInterface.dag(t)
    slots = Tuple(a for (a, i) in enumerate(td.inds) if i.dual && i ∈ metric_legs)
    isempty(slots) && return td
    return TKTensor(td.inds, TK.twist(td.data, slots))
end

# ── Boundary-MPS link-sector allocation (init recipe; used by boundarympscache.jl) ──────

#Charge spectrum reachable by one message site: the convolution of its legs' carried
#sector spectra (dual legs contribute dual sectors), weight ∝ sector dimension.
function site_charge_spectrum(m::TKTensor)
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
