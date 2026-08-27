# Fermionic backend: KIndex-labelled wrapper around a TensorKit fZ2 TensorMap.
#
# The data layer is a one-sided TensorMap (all legs in the codomain, trivial domain)
# over Vect[FermionParity] spaces. TensorKit's category machinery supplies every Koszul
# sign: leg permutations, non-adjacent operator strings (Jordan-Wigner strings emerge
# automatically from the braiding) and blockwise factorizations. This file only does
# bookkeeping between KIndex labels and TensorMap slots.
#
# Conventions (validated against dense Jordan-Wigner ground truth):
#   * Slot `a` of the data holds `slotspace(inds[a])`: the index's base space, dualed
#     when the per-copy `dual` flag is set. The flag is per tensor copy — the same
#     KIndex identity (id, plev) appears with opposite flags on the two tensors sharing
#     a bond (src = false, dst = true), and `dag` flips all flags.
#   * `dag` is the plain categorical adjoint permuted back to the codomain — no twists.
#     Site legs are non-dual on kets; with that convention closed bra-ket networks
#     evaluate to physical inner products (verified: factored ⟨ψ|ψ⟩ and non-adjacent
#     ⟨c†ᵢcⱼ + h.c.⟩ both match dense JW truth to machine precision).
#   * Operators are built by fusion-tree assignment of Fock matrix elements ⟨out|O|in⟩
#     on a two-sided (outs ← ins) map, then permuted one-sided. For parity sectors the
#     tree basis coincides with the mode-ordered Fock basis (site 1 slowest). Gates are
#     matrix exponentials of the LOCAL Fock matrix — no string needed.
#   * Every contraction requires the two copies of a contracted index to carry opposite
#     flags; a same-flag pairing is a network-construction bug and errors loudly.

const FermionSpace = TK.GradedSpace{TK.FermionParity, Tuple{Int, Int}}
const FIndex = KIndex{FermionSpace}

fermion_space(d0::Integer = 1, d1::Integer = 1) =
    TK.Vect[TK.FermionParity](0 => Int(d0), 1 => Int(d1))

KIndex(space::FermionSpace, tags::AbstractString = "") =
    KIndex(rand(UInt64), space, 0, String(tags), false)

"""
    new_fermion_index(d0 = 1, d1 = 1; tags = "")

A fresh fermionic (Z2-parity graded) index with `d0` even and `d1` odd states.
"""
new_fermion_index(d0::Integer = 1, d1::Integer = 1; tags = "") =
    KIndex(fermion_space(d0, d1), tags)

space_dim(s::FermionSpace) = TK.dim(s)

#Even-dominant split for a fresh link of total dimension d (fermion-branch recipe:
#double-layer networks are even-parity dominant, so links need at least as many even
#sectors as odd).
function TensorInterface.new_index(ref::FIndex, d::Integer; tags = "")
    return new_fermion_index(cld(Int(d), 2), fld(Int(d), 2); tags)
end

slotspace(i::FIndex) = i.dual ? TK.dual(space(i)) : space(i)

struct FTensor{TM <: TK.AbstractTensorMap}
    inds::Vector{FIndex}
    data::TM
    function FTensor(inds::AbstractVector, data::TK.AbstractTensorMap)
        iv = collect(FIndex, inds)
        TK.numout(data) == length(iv) && TK.numin(data) == 0 ||
            error("FTensor: data must be one-sided with $(length(iv)) codomain legs")
        for (a, i) in enumerate(iv)
            TK.space(data, a) == slotspace(i) ||
                error("FTensor: slot $a is $(TK.space(data, a)), index wants $(slotspace(i))")
        end
        return new{typeof(data)}(iv, data)
    end
end

Base.eltype(t::FTensor) = TK.scalartype(t.data)
Base.ndims(t::FTensor) = length(t.inds)
Base.copy(t::FTensor) = FTensor(copy(t.inds), copy(t.data))
#Positive, parity-gauge-insensitive normalization functional (BP normalizes messages by
#sum). Fermionic messages appear in either parity gauge (odd sector sign is a gauge
#choice), so a linear component sum could vanish or flip the message sign; summing the
#per-tree magnitudes is gauge-invariant and positive.
function Base.sum(t::FTensor)
    return sum(p -> abs(sum(t.data[p[1], p[2]])), TK.fusiontrees(t.data); init = zero(real(eltype(t))))
end

function Base.show(io::IO, t::FTensor)
    return print(io, "FTensor{$(eltype(t))} inds=", t.inds)
end

function TensorInterface.inds(t::FTensor; plev = nothing)
    plev === nothing && return t.inds
    return filter(i -> i.plev == plev, t.inds)
end

TensorInterface.scalartype(t::FTensor) = TK.scalartype(t.data)
TensorInterface.datatype(t::FTensor) = Vector{TK.scalartype(t.data)}
TensorInterface.data(t::FTensor) = t

# ── Fock-basis conversion (tree basis ≡ mode-ordered Fock basis for parity sectors) ─────

#Charge-ordered position range of sector `c` (0 = even, 1 = odd) within an index.
function _fock_range(i::FIndex, isodd::Bool)
    d0 = TK.dim(space(i), TK.FermionParity(0))
    return isodd ? (d0 + 1:d0 + TK.dim(space(i), TK.FermionParity(1))) : (1:d0)
end

#NOTE: only trustworthy for all-non-dual tensors (states); dual slots involve duality
#bends whose basis convention is TensorKit-internal. Ops are built two-sided instead.
function TensorInterface.array(t::FTensor)
    out = zeros(eltype(t), Int[dimof(i) for i in t.inds]...)
    N = ndims(t)
    for (f1, f2) in TK.fusiontrees(t.data)
        rngs = ntuple(a -> _fock_range(t.inds[a], f1.uncoupled[a].isodd), N)
        out[rngs...] = t.data[f1, f2]
    end
    return out
end

function TensorInterface.from_array(A::AbstractArray, is::FIndex...)
    iv = collect(FIndex, is)
    all(i -> !i.dual, iv) ||
        error("from_array: fermionic scatter is only defined for non-dual indices")
    N = length(iv)
    A = reshape(A, Int[dimof(i) for i in iv]...)
    data = zeros(eltype(A), TK.ProductSpace(map(slotspace, iv)...))
    for (f1, f2) in TK.fusiontrees(data)
        rngs = ntuple(a -> _fock_range(iv[a], f1.uncoupled[a].isodd), N)
        data[f1, f2] .= A[rngs...]
    end
    LinearAlgebra.norm(data) ≈ LinearAlgebra.norm(A) || error(
        "from_array: the array has weight outside the parity-even sector — fermionic " *
            "tensors carry zero total flux. Parity-odd product states (e.g. \"Occ\") are " *
            "not representable per-vertex; start from even states and create pairs with " *
            "the \"F_pair\" gate."
    )
    return FTensor(iv, data)
end

# ── Index transforms (labels only; the data never moves) ───────────────────────────────

_mapinds(f, t::FTensor) = FTensor(map(f, t.inds), t.data)

TensorInterface.prime(t::FTensor, n::Integer = 1) = _mapinds(i -> TensorInterface.prime(i, n), t)
TensorInterface.noprime(t::FTensor) = _mapinds(TensorInterface.noprime, t)
TensorInterface.sim(t::FTensor) = _mapinds(TensorInterface.sim, t)

function TensorInterface.dag(t::FTensor)
    N = ndims(t)
    dd = TK.permute(adjoint(t.data), (Tuple(1:N), ()))
    return FTensor(FIndex[TensorInterface.dag(i) for i in t.inds], dd)
end

#Relabels by identity; the per-copy dual flag is data bookkeeping and is PRESERVED from
#the existing copy (generic code passes replacement labels with arbitrary flags).
function TensorInterface.replaceinds(t::FTensor, old, new)
    oldv, newv = _indvec(old), _indvec(new)
    length(oldv) == length(newv) || error("replaceinds: length mismatch")
    newinds = map(t.inds) do i
        k = findfirst(==(i), oldv)
        k === nothing && return i
        n = newv[k]
        space(n) == space(i) || error("replaceinds: space mismatch $(i) → $(n)")
        return KIndex(n.id, n.space, n.plev, n.tags, i.dual)
    end
    return FTensor(newinds, t.data)
end
TensorInterface.replaceind(t::FTensor, old::KIndex, new::KIndex) = TensorInterface.replaceinds(t, [old], [new])
TensorInterface.replaceinds(t::FTensor, p::Pair) = TensorInterface.replaceinds(t, first(p), last(p))

_indvec(t::FTensor) = t.inds

for f in [:commoninds, :commonind, :uniqueinds, :unioninds, :noncommoninds, :noncommonind, :hascommoninds]
    @eval begin
        TensorInterface.$f(a::FTensor, b) = TensorInterface.$f(a.inds, _indvec(b))
        TensorInterface.$f(a, b::FTensor) = TensorInterface.$f(_indvec(a), b.inds)
        TensorInterface.$f(a::FTensor, b::FTensor) = TensorInterface.$f(a.inds, b.inds)
    end
end

# ── Arithmetic ──────────────────────────────────────────────────────────────────────────

Base.:*(t::FTensor, x::Number) = FTensor(copy(t.inds), t.data * x)
Base.:*(x::Number, t::FTensor) = t * x
Base.:/(t::FTensor, x::Number) = t * inv(x)

#b's data with slots permuted into a's index order (TensorKit threads the signs)
function _aligned_data(a::FTensor, b::FTensor)
    p = map(a.inds) do i
        k = findfirst(==(i), b.inds)
        k === nothing && error("tensors do not share index $(i)")
        b.inds[k].dual == i.dual || error("aligned combine: flag mismatch on $(i)")
        k
    end
    return TK.permute(b.data, (Tuple(p), ()))
end

Base.:+(a::FTensor, b::FTensor) = FTensor(copy(a.inds), a.data + _aligned_data(a, b))
Base.:-(a::FTensor, b::FTensor) = FTensor(copy(a.inds), a.data - _aligned_data(a, b))

LinearAlgebra.norm(t::FTensor) = LinearAlgebra.norm(t.data)
LinearAlgebra.dot(a::FTensor, b::FTensor) = LinearAlgebra.dot(a.data, _aligned_data(a, b))

function LinearAlgebra.rmul!(t::FTensor, x::Number)
    LinearAlgebra.rmul!(t.data, x)
    return t
end

function Base.isapprox(a::FTensor, b::FTensor; atol = 0, rtol = nothing)
    rt = rtol === nothing ? sqrt(eps(real(promote_type(eltype(a), eltype(b))))) : rtol
    return LinearAlgebra.norm(a - b) <= max(atol, rt * max(LinearAlgebra.norm(a), LinearAlgebra.norm(b)))
end

function Adapt.adapt_structure(elt::Type{<:Number}, t::FTensor)
    return FTensor(copy(t.inds), one(elt) * t.data)
end

# ── Contraction ─────────────────────────────────────────────────────────────────────────

function Base.:*(a::FTensor, b::FTensor)
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
    return FTensor(oinds, TK.permute(out, (Tuple(1:length(oinds)), ())))
end

function TensorInterface.contract(ts::Vector{<:FTensor}; sequence = nothing, kwargs...)
    isnothing(sequence) && return reduce(*, ts)
    return _contract_seq_ft(ts, sequence)
end
_contract_seq_ft(ts, s::Integer) = ts[s]
_contract_seq_ft(ts, s::Union{Vector, Tuple}) = mapreduce(x -> _contract_seq_ft(ts, x), *, s)

TensorInterface.apply(o::FTensor, t::FTensor) = TensorInterface.noprime(o * t)

# ── Construction: onehot, delta, state, op, random ──────────────────────────────────────

function TensorInterface.onehot(elt::Type, p::Pair{<:FIndex, <:Integer})
    i, v = p
    data = zeros(elt, TK.ProductSpace(slotspace(i)))
    for (f1, f2) in TK.fusiontrees(data)
        r = _fock_range(i, f1.uncoupled[1].isodd)
        v ∈ r && (data[f1, f2][v - first(r) + 1] = one(elt))
    end
    return FTensor([i], data)
end
TensorInterface.onehot(p::Pair{<:FIndex, <:Integer}) = TensorInterface.onehot(Float64, p)

#Identity messages: indices come in (i, i′)-style pairs sharing an id, with opposite
#per-copy flags (e.g. from `delta(vcat(linds, prime(dag(linds))))`).
function _delta_f(elt::Type, is::AbstractVector{<:KIndex})
    ids = unique([i.id for i in is])
    parts = map(ids) do id
        pair = filter(i -> i.id == id, is)
        length(pair) == 2 || error("delta: fermionic delta needs indices in (i, i′)-style pairs")
        i1, i2 = pair
        space(i1) == space(i2) || error("delta: paired indices must share a space")
        i1.dual != i2.dual || error("delta: paired indices must have opposite orientations")
        data = TK.permute(TK.id(elt, slotspace(i1)), ((1, 2), ()))
        FTensor([i1, i2], data)
    end
    return reduce(*, parts)
end
TensorInterface.delta(elt::Type, is::FIndex...) = _delta_f(elt, collect(KIndex, is))
TensorInterface.delta(is::FIndex...) = _delta_f(Float64, collect(KIndex, is))

function TensorInterface.state(name::String, i::FIndex)
    dimof(i) == 2 || error("state: fermionic state library covers d = 2 sites only")
    vec = get(F_STATES, name, nothing)
    vec === nothing && error(
        "state: unknown fermionic state \"$name\" (available: $(sort(collect(keys(F_STATES))))); " *
            "note fermionic states must have definite parity"
    )
    return TensorInterface.from_array(vec, i)
end

const F_STATES = Dict{String, Vector{Float64}}(
    "0" => [1, 0], "Emp" => [1, 0], "Empty" => [1, 0],
    "1" => [0, 1], "Occ" => [0, 1], "Occupied" => [0, 1],
)

# ── Operators and gates: Fock matrices tree-assigned on two-sided maps ──────────────────

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
    return error("op: unknown fermionic operator \"$name\"")
end

function _f_op2_matrix(name::String; kwargs...)
    name == "hopping" && return _F_HOP
    name == "NN" && return _F_NN
    name == "pairing" && return _F_PAIR
    name == "F_hop" && return exp(-im * kwargs[:θ] * _F_HOP)
    name == "F_nn" && return exp(-im * kwargs[:θ] * _F_NN)
    name == "F_pair" && return exp(-im * kwargs[:θ] * _F_PAIR)
    if name == "F_hop_nn"
        return exp(-im * (kwargs[:θ] * _F_HOP + kwargs[:ϕ] * _F_NN))
    end
    return error("op: unknown fermionic 2-site operator \"$name\"")
end

#Wrap a Fock matrix M[out, in] as an operator FTensor with legs (u..., s...):
#u = prime(s) non-dual OUT legs, s dual IN legs. Built two-sided (outs ← ins) by tree
#assignment, then permuted one-sided (the validated construction).
function _f_op(M::AbstractMatrix, sites::Vector{<:FIndex})
    all(i -> !i.dual && dimof(i) == 2, sites) ||
        error("op: fermionic operators expect non-dual d = 2 site indices")
    n = length(sites)
    P = TK.ProductSpace(map(i -> space(i), sites)...)
    G = zeros(eltype(M), P, P)
    for (f1, f2) in TK.fusiontrees(G)
        row = 1 + foldl((acc, c) -> 2acc + Int(c.isodd), f1.uncoupled; init = 0)
        col = 1 + foldl((acc, c) -> 2acc + Int(c.isodd), f2.uncoupled; init = 0)
        iszero(M[row, col]) || (G[f1, f2] .= M[row, col])
    end
    Gc = TK.permute(G, (Tuple(1:(2n)), ()))
    us = [TensorInterface.prime(i) for i in sites]
    ss = [TensorInterface.dag(i) for i in sites]
    return FTensor(vcat(us, ss), Gc)
end

TensorInterface.op(name::String, i::FIndex; kwargs...) = _f_op(_f_op1_matrix(name; kwargs...), [i])
function TensorInterface.op(name::String, i1::FIndex, i2::FIndex; kwargs...)
    return _f_op(_f_op2_matrix(name; kwargs...), [i1, i2])
end

#Random tensors: a TensorMap only populates charge-conserving (parity-even) trees, so
#plain randn is already the symmetric random initializer.
function TensorInterface.random_itensor(elt::Type{<:Number}, is::FIndex...)
    iv = collect(FIndex, is)
    data = randn(elt, TK.ProductSpace(map(slotspace, iv)...))
    return FTensor(iv, data)
end
TensorInterface.random_itensor(is::FIndex...) = TensorInterface.random_itensor(Float64, is...)

#Graded BP messages carry a per-message parity gauge: m and its parity twist (odd
#sector negated) are equally valid fixed points, and update history determines which
#one BP produces (both appear in practice, always in closure-consistent pairs). For
#operations that need the message as a PSD operator (square roots for gauging), detect
#the gauge from the odd-block trace and twist into the PSD representative — the twist
#cancels in any M^½ · M^{-½} sandwich, so this is exact.
parity_twisted(t::FTensor) = FTensor(copy(t.inds), TK.twist(t.data, 1))

function psd_gauge(t::FTensor)
    oddtr = zero(real(eltype(t)))
    for (f1, f2) in TK.fusiontrees(t.data)
        f1.uncoupled[1].isodd || continue
        b = t.data[f1, f2]
        for x in 1:minimum(size(b))
            oddtr += real(b[x, x])
        end
    end
    return oddtr < 0 ? parity_twisted(t) : t
end

# ── Diagonal operations ─────────────────────────────────────────────────────────────────

function TensorInterface.map_diag!(f::Function, out::FTensor, t::FTensor)
    ndims(t) == 2 || error("map_diag: expected a 2-index FTensor")
    out === t || error("map_diag!: fermionic backend only supports in-place (out === t)")
    for (f1, f2) in TK.fusiontrees(t.data)
        b = t.data[f1, f2]
        for x in 1:minimum(size(b))
            b[x, x] = f(b[x, x])
        end
    end
    return out
end
function TensorInterface.map_diag(f::Function, t::FTensor)
    out = copy(t)
    TensorInterface.map_diag!(f, out, out)
    return out
end

# ── Factorizations (MatrixAlgebraKit API through TensorKit, blockwise with signs) ───────

function _f_split_positions(t::FTensor, lv::Vector{<:KIndex})
    lpos = Int[a for (a, i) in enumerate(t.inds) if i ∈ lv]
    rpos = Int[a for (a, i) in enumerate(t.inds) if i ∉ lv]
    return lpos, rpos, t.inds[lpos], t.inds[rpos]
end

#Wrap a TensorKit space as (base space, dual flag) for a fresh KIndex copy.
_wrap_slot(sp) = TK.isdual(sp) ? (TK.dual(sp), true) : (sp, false)

_with_flag(i::FIndex, dual::Bool) = KIndex(i.id, i.space, i.plev, i.tags, dual)

function _fsvd_core(t::FTensor, lv::Vector{<:KIndex}; maxdim = nothing, cutoff = nothing)
    lpos, rpos, li, ri = _f_split_positions(t, lv)
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

#Rewrap map factors as FTensors: L carries (li..., bond dual), R carries (bond, ri...).
function _f_wrap_left(U, li::Vector{<:KIndex}, b::FIndex)
    nl = length(li)
    Uc = TK.permute(U, (Tuple(1:(nl + 1)), ()))
    return FTensor(vcat(li, [_with_flag(b, true)]), Uc)
end
function _f_wrap_right(Vh, ri::Vector{<:KIndex}, b::FIndex)
    nr = length(ri)
    Vc = TK.permute(Vh, (Tuple(1:(nr + 1)), ()))
    return FTensor(vcat([_with_flag(b, false)], ri), Vc)
end

function _f_bond_index(S, tags::String)
    sp, isdual = _wrap_slot(TK.space(S, 1))
    isdual && error("factorize: unexpected dual bond space from the factorization")
    return KIndex(sp, tags)
end

function TensorInterface.factorize_svd(
        t::FTensor, linds;
        ortho = "none", singular_values! = nothing,
        maxdim = nothing, cutoff = nothing, kwargs...,
    )
    ortho == "none" || error("factorize_svd: fermionic backend implements ortho = \"none\"")
    lv = filter(i -> i ∈ t.inds, _indvec(linds))
    _, _, li, ri, U, S, Vh, kept_s2, truncerr = _fsvd_core(t, lv; maxdim, cutoff)
    sq = sqrt(S)
    u = _f_bond_index(S, "Link,u")
    v = KIndex(u.space, "Link,v")
    up = TensorInterface.prime(u)
    F1 = _f_wrap_left(U * sq, li, up)
    F2 = _f_wrap_right(sq * Vh, ri, up)
    if singular_values! !== nothing
        Sc = TK.permute(S, ((1, 2), ()))
        singular_values![] = FTensor([_with_flag(u, false), _with_flag(v, true)], Sc)
    end
    return F1, F2, KSpectrum(kept_s2, truncerr)
end

function LinearAlgebra.factorize(
        t::FTensor, linds...;
        ortho = "left", maxdim = nothing, cutoff = nothing, tags = "Link,fact", kwargs...,
    )
    lv = length(linds) == 1 ? _indvec(only(linds)) : collect(KIndex, linds)
    lv = filter(i -> i ∈ t.inds, lv)
    _, _, li, ri, U, S, Vh, _, _ = _fsvd_core(t, lv; maxdim, cutoff)
    b = _f_bond_index(S, String(tags))
    if ortho == "left"
        return _f_wrap_left(U, li, b), _f_wrap_right(S * Vh, ri, b)
    elseif ortho == "right"
        return _f_wrap_left(U * S, li, b), _f_wrap_right(Vh, ri, b)
    else
        error("factorize: unknown ortho = $(ortho)")
    end
end

function LinearAlgebra.svd(t::FTensor, linds; maxdim = nothing, cutoff = nothing, kwargs...)
    lv = filter(i -> i ∈ t.inds, _indvec(linds))
    _, _, li, ri, U, S, Vh, _, _ = _fsvd_core(t, lv; maxdim, cutoff)
    u = _f_bond_index(S, "Link,u")
    v = KIndex(u.space, "Link,v")
    Ut = _f_wrap_left(U, li, u)
    Sc = TK.permute(S, ((1, 2), ()))
    St = FTensor([_with_flag(u, false), _with_flag(v, true)], Sc)
    #V carries (ri..., v) with the bond last and non-dual
    nr = length(ri)
    Vc = TK.permute(Vh, (Tuple([2:(nr + 1); 1]), ()))
    Vt = FTensor(vcat(ri, [_with_flag(v, false)]), Vc)
    return Ut, St, Vt
end

function LinearAlgebra.qr(t::FTensor, linds; kwargs...)
    lv = filter(i -> i ∈ t.inds, _indvec(linds))
    lpos, rpos, li, ri = _f_split_positions(t, lv)
    tp = TK.permute(t.data, (Tuple(lpos), Tuple(rpos)))
    Q, R = qr_compact(tp)
    sp, isdual = _wrap_slot(TK.space(R, 1))
    isdual && error("qr: unexpected dual bond space")
    b = KIndex(sp, "Link,qr")
    return _f_wrap_left(Q, li, b), _f_wrap_right(R, ri, b)
end

#Hermitian eigendecomposition, mirroring the BlockTensor conventions: D on
#(prime(lk), lk), U on (rinds..., lk). Per-copy flags are read off the actual data
#slots, so every shared identity ends up with opposite orientations on its two holders.
function LinearAlgebra.eigen(t::FTensor, linds, rinds; ishermitian::Bool = false, kwargs...)
    ishermitian || error("eigen: only ishermitian = true is implemented for FTensors")
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
    uinds = FIndex[
        [_with_flag(rv[k], TK.isdual(TK.space(Uc, k))) for k in 1:nr];
        _with_flag(lk, TK.isdual(TK.space(Uc, nr + 1)))
    ]
    U = FTensor(uinds, Uc)
    Dc = TK.permute(D, ((1, 2), ()))
    dinds = FIndex[
        TensorInterface.prime(_with_flag(lk, TK.isdual(TK.space(Dc, 1)))),
        _with_flag(lk, TK.isdual(TK.space(Dc, 2))),
    ]
    return FTensor(dinds, Dc), U
end

function LinearAlgebra.eigen(t::FTensor; ishermitian::Bool = false, kwargs...)
    lv = filter(i -> i.plev == 0, t.inds)
    rv = collect(KIndex, TensorInterface.prime.(lv))
    D, U = LinearAlgebra.eigen(t, lv, rv; ishermitian, kwargs...)
    return D, TensorInterface.replaceinds(U, rv, lv)
end
