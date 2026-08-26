#=
BlockTensor: abelian block-sparse tensors over graded KIndex spaces (v2 prototype).

A `BlockTensor` stores a dictionary of dense blocks keyed by the per-leg sector choice
(the position of the sector in each index's `GradedSpace`). It implements the same seam
verbs as the dense `KTensor`, so graded networks flow through every generic algorithm
unchanged; the fused kernels dispatch on `KTensor` and therefore fall back to the generic
paths automatically.

Deliberately flux-free: any block may be stored, and conservation emerges from
construction — symmetric operators scatter to few blocks (exact-zero blocks are dropped),
non-symmetric ones simply populate more blocks. Sparsity when the circuit conserves,
graceful degradation when it doesn't, never a correctness question.

Factorizations work per *charge class*: the classes are the connected components of the
bipartite (left-signature, right-signature) adjacency of the present blocks, each class is
factorized densely, and truncation ranks singular values globally across classes (the same
discarded-Σs²/total convention as the dense backend). Prototype restrictions: no combiner/
directsum, no random graded tensors, no boundary-MPS-specific paths.
=#

# ── Type ────────────────────────────────────────────────────────────────────────────────

struct BlockTensor{T, N}
    inds::Vector{KIndex}
    blocks::Dict{NTuple{N, Int}, Array{T, N}}
    function BlockTensor{T, N}(inds::AbstractVector, blocks::Dict{NTuple{N, Int}, Array{T, N}}) where {T, N}
        inds = collect(KIndex, inds)
        length(inds) == N || error("BlockTensor: $(length(inds)) indices for rank-$N blocks")
        return new{T, N}(inds, blocks)
    end
end
function BlockTensor(inds::AbstractVector, blocks::Dict{NTuple{N, Int}, Array{T, N}}) where {T, N}
    return BlockTensor{T, N}(inds, blocks)
end

const GradedIndex = KIndex{GradedSpace}

_sectordims(i::KIndex, k::Int) = space(i).dims[k]
_blockdims(is::Vector{<:KIndex}, key) = ntuple(a -> _sectordims(is[a], key[a]), length(is))

Base.eltype(::BlockTensor{T}) where {T} = T
Base.ndims(::BlockTensor{T, N}) where {T, N} = N
Base.copy(t::BlockTensor) = BlockTensor(copy(t.inds), Dict(k => copy(b) for (k, b) in t.blocks))
Base.sum(t::BlockTensor) = isempty(t.blocks) ? zero(eltype(t)) : sum(sum, values(t.blocks))

function Base.show(io::IO, t::BlockTensor{T, N}) where {T, N}
    print(io, "BlockTensor{", T, "} (", length(t.blocks), " blocks) inds: ", t.inds)
    return nothing
end

function TensorInterface.inds(t::BlockTensor; plev = nothing)
    plev === nothing && return t.inds
    return filter(i -> i.plev == plev, t.inds)
end
TensorInterface.scalartype(::BlockTensor{T}) where {T} = T
TensorInterface.datatype(::BlockTensor{T}) where {T} = Vector{T}
TensorInterface.new_index(::BlockTensor, d::Integer; tags = "") = KIndex(GradedSpace([0], [Int(d)]), tags)
TensorInterface.new_index(i::GradedIndex, d::Integer; tags = "") = KIndex(GradedSpace([0], [Int(d)]), tags)
TensorInterface.new_index(sectors::Vector{<:Pair}; tags = "") = KIndex(GradedSpace(sectors...), tags)

function TensorInterface.scalar(t::BlockTensor{T, 0}) where {T}
    return isempty(t.blocks) ? zero(T) : t.blocks[()][]
end
TensorInterface.scalar(t::BlockTensor) = error("scalar: BlockTensor with inds $(t.inds) is not a scalar")

#Densify (charge-ordered basis)
function TensorInterface.array(t::BlockTensor{T, N}) where {T, N}
    out = zeros(T, Int[dimof(i) for i in t.inds]...)
    for (k, b) in t.blocks
        out[ntuple(a -> sector_range(space(t.inds[a]), k[a]), N)...] = b
    end
    return out
end

#Scatter a dense array into blocks, dropping exact-zero blocks (that is where sparsity
#comes from: symmetric operators have structural zeros off the conserving sectors).
function TensorInterface.from_array(A::AbstractArray, is::GradedIndex...)
    isv = collect(KIndex, is)
    N = length(isv)
    T = eltype(A)
    A = reshape(A, Int[dimof(i) for i in isv]...)
    blocks = Dict{NTuple{N, Int}, Array{T, N}}()
    for key in Iterators.product((1:nsectors(space(i)) for i in isv)...)
        b = A[ntuple(a -> sector_range(space(isv[a]), key[a]), N)...]
        iszero(norm(b)) || (blocks[key] = collect(b))
    end
    return BlockTensor(isv, blocks)
end

# ── Index transforms ────────────────────────────────────────────────────────────────────

_mapinds(f, t::BlockTensor) = BlockTensor(map(f, t.inds), t.blocks)

TensorInterface.prime(t::BlockTensor, n::Integer = 1) = _mapinds(i -> TensorInterface.prime(i, n), t)
TensorInterface.noprime(t::BlockTensor) = _mapinds(TensorInterface.noprime, t)
TensorInterface.sim(t::BlockTensor) = _mapinds(TensorInterface.sim, t)
function TensorInterface.dag(t::BlockTensor)
    return BlockTensor(map(TensorInterface.dag, t.inds), Dict(k => conj(b) for (k, b) in t.blocks))
end

function TensorInterface.replaceinds(t::BlockTensor, old, new)
    oldv, newv = _indvec(old), _indvec(new)
    length(oldv) == length(newv) || error("replaceinds: length mismatch")
    newinds = map(t.inds) do i
        k = findfirst(==(i), oldv)
        if k === nothing
            i
        else
            n = newv[k]
            space(n) == space(i) || dimof(n) == dimof(i) ||
                error("replaceinds: space mismatch $(i) → $(n)")
            n
        end
    end
    return BlockTensor(newinds, t.blocks)
end
TensorInterface.replaceind(t::BlockTensor, old::KIndex, new::KIndex) = TensorInterface.replaceinds(t, [old], [new])
TensorInterface.replaceinds(t::BlockTensor, p::Pair) = TensorInterface.replaceinds(t, first(p), last(p))

_indvec(t::BlockTensor) = t.inds

#Extend the index-set queries and the hook guards to graded tensors
for f in [:commoninds, :commonind, :uniqueinds, :unioninds, :noncommoninds, :noncommonind, :hascommoninds]
    @eval begin
        TensorInterface.$f(a::BlockTensor, b) = TensorInterface.$f(a.inds, _indvec(b))
        TensorInterface.$f(a, b::BlockTensor) = TensorInterface.$f(_indvec(a), b.inds)
        TensorInterface.$f(a::BlockTensor, b::BlockTensor) = TensorInterface.$f(a.inds, b.inds)
    end
end

# ── Construction: onehot, delta, state, op ──────────────────────────────────────────────

#Locate the (sector, offset) of dense position v within a graded index
function _sector_of(s::GradedSpace, v::Integer)
    for k in 1:nsectors(s)
        r = sector_range(s, k)
        v ∈ r && return k, v - first(r) + 1
    end
    return error("position $v outside space")
end

function TensorInterface.onehot(elt::Type, p::Pair{<:GradedIndex, <:Integer})
    i, v = p
    k, off = _sector_of(space(i), v)
    b = zeros(elt, space(i).dims[k])
    b[off] = one(elt)
    return BlockTensor([i], Dict((k,) => b))
end
TensorInterface.onehot(p::Pair{<:GradedIndex, <:Integer}) = TensorInterface.onehot(Float64, p)

function _delta_graded(elt::Type, is::AbstractVector{<:KIndex})
    #Pairs of indices sharing an id (e.g. a bond and its primed partner) each get an
    #identity; independent pairs combine as an outer product.
    ids = unique([i.id for i in is])
    parts = map(ids) do id
        pair = filter(i -> i.id == id, is)
        length(pair) == 2 || error("delta: graded delta needs indices in (i, i′)-style pairs")
        i1, i2 = pair
        space(i1) == space(i2) || error("delta: paired indices must share a space")
        blocks = Dict{NTuple{2, Int}, Array{elt, 2}}()
        for k in 1:nsectors(space(i1))
            d = space(i1).dims[k]
            blocks[(k, k)] = Matrix{elt}(LinearAlgebra.I, d, d)
        end
        BlockTensor([i1, i2], blocks)
    end
    return reduce(*, parts)
end
TensorInterface.delta(elt::Type, is::GradedIndex...) = _delta_graded(elt, collect(KIndex, is))
TensorInterface.delta(is::GradedIndex...) = _delta_graded(Float64, collect(KIndex, is))

function TensorInterface.state(name::String, i::GradedIndex)
    dimof(i) == 2 || error("state: graded state library currently covers d=2 sites only")
    dense = TensorInterface.state(name, KIndex(2))
    return TensorInterface.from_array(dense.data, i)
end

function TensorInterface.op(name::String, i::GradedIndex; kwargs...)
    dense = TensorInterface.op(name, KIndex(dimof(i)); kwargs...)
    return TensorInterface.from_array(dense.data, TensorInterface.prime(i), i)
end

function TensorInterface.op(name::String, i1::GradedIndex, i2::GradedIndex; kwargs...)
    dense = TensorInterface.op(name, KIndex(dimof(i1)), KIndex(dimof(i2)); kwargs...)
    return TensorInterface.from_array(
        dense.data, TensorInterface.prime(i1), TensorInterface.prime(i2), i1, i2
    )
end

# ── Arithmetic ──────────────────────────────────────────────────────────────────────────

Base.:*(t::BlockTensor, x::Number) = BlockTensor(copy(t.inds), Dict(k => b * x for (k, b) in t.blocks))
Base.:*(x::Number, t::BlockTensor) = t * x
Base.:/(t::BlockTensor, x::Number) = t * inv(x)

#Blocks of `b` re-keyed and permuted into `a`'s index order
function _aligned_blocks(a::BlockTensor, b::BlockTensor{T, N}) where {T, N}
    perm = map(i -> findfirst(==(i), b.inds), a.inds)
    any(isnothing, perm) && error("tensors have different index sets: $(a.inds) vs $(b.inds)")
    perm = Int[perm...]
    return Dict{NTuple{N, Int}, Array{T, N}}(
        ntuple(x -> k[perm[x]], N) => permutedims(blk, perm) for (k, blk) in b.blocks
    )
end

function _combine(f, a::BlockTensor{Ta, N}, b::BlockTensor{Tb, N}) where {Ta, Tb, N}
    T = promote_type(Ta, Tb)
    bb = _aligned_blocks(a, b)
    blocks = Dict{NTuple{N, Int}, Array{T, N}}()
    for k in union(keys(a.blocks), keys(bb))
        A = get(() -> zeros(T, _blockdims(a.inds, k)), a.blocks, k)
        B = get(() -> zeros(T, _blockdims(a.inds, k)), bb, k)
        blocks[k] = f.(A, B)
    end
    return BlockTensor(copy(a.inds), blocks)
end
Base.:+(a::BlockTensor, b::BlockTensor) = _combine(+, a, b)
Base.:-(a::BlockTensor, b::BlockTensor) = _combine(-, a, b)

function Base.isapprox(a::BlockTensor, b::BlockTensor; atol = nothing, kwargs...)
    atol = isnothing(atol) ? 1.0e-10 * max(norm(a), norm(b), 1.0) : atol
    return norm(a - b) <= atol
end

LinearAlgebra.norm(t::BlockTensor) = sqrt(sum(b -> norm(b)^2, values(t.blocks); init = 0.0))
LinearAlgebra.normalize(t::BlockTensor) = t * inv(norm(t))
function LinearAlgebra.dot(a::BlockTensor, b::BlockTensor)
    bb = _aligned_blocks(a, b)
    return sum(LinearAlgebra.dot(a.blocks[k], bb[k]) for k in intersect(keys(a.blocks), keys(bb)); init = zero(promote_type(eltype(a), eltype(b))))
end
LinearAlgebra.tr(t::BlockTensor) = LinearAlgebra.tr(TensorInterface.array(t))

function LinearAlgebra.rmul!(t::BlockTensor, x::Number)
    for b in values(t.blocks)
        rmul!(b, x)
    end
    return t
end
#The raw-storage handle used for in-place normalization; scaling it scales the tensor.
TensorInterface.data(t::BlockTensor) = t

# ── Contraction ─────────────────────────────────────────────────────────────────────────

function Base.:*(a::BlockTensor, b::BlockTensor)
    ndims(a) == 0 && return b * TensorInterface.scalar(a)
    ndims(b) == 0 && return a * TensorInterface.scalar(b)

    ca, cb = Int[], Int[]
    for (j, bj) in enumerate(b.inds)
        i = findfirst(==(bj), a.inds)
        i === nothing && continue
        space(a.inds[i]) == space(bj) || error("contraction: mismatched spaces on $(bj)")
        push!(ca, i)
        push!(cb, j)
    end
    oa = setdiff(1:ndims(a), ca)
    ob = setdiff(1:ndims(b), cb)
    No = length(oa) + length(ob)
    T = promote_type(eltype(a), eltype(b))
    oinds = vcat(a.inds[oa], b.inds[ob])

    #Group b's blocks by their contracted-sector signature for O(|A| + |B| + hits) pairing
    lookup = Dict{NTuple{length(cb), Int}, Vector{Pair{NTuple{ndims(b), Int}, Array{eltype(b), ndims(b)}}}}()
    for (kb, B) in b.blocks
        push!(get!(() -> valtype(lookup)(), lookup, ntuple(x -> kb[cb[x]], length(cb))), kb => B)
    end

    pA = (Tuple(oa), Tuple(ca))
    pB = (Tuple(cb), Tuple(ob))
    pAB = (Tuple(1:No), ())
    backend = T <: BlasFloat ? TensorOperations.StridedBLAS() : TensorOperations.StridedNative()
    blocks = Dict{NTuple{No, Int}, Array{T, No}}()
    for (ka, A) in a.blocks
        sig = ntuple(x -> ka[ca[x]], length(ca))
        for (kb, B) in get(() -> valtype(lookup)(), lookup, sig)
            ko = (ntuple(x -> ka[oa[x]], length(oa))..., ntuple(x -> kb[ob[x]], length(ob))...)
            C = get!(() -> zeros(T, _blockdims(oinds, ko)), blocks, ko)
            TensorOperations.tensorcontract!(
                C, A, pA, false, B, pB, false, pAB, one(T), one(T),
                backend, TensorOperations.DefaultAllocator()
            )
        end
    end
    return BlockTensor(oinds, blocks)
end

function TensorInterface.contract(ts::Vector{<:BlockTensor}; sequence = nothing, kwargs...)
    isnothing(sequence) && return reduce(*, ts)
    return _contract_seq_bt(ts, sequence)
end
_contract_seq_bt(ts, s::Integer) = ts[s]
_contract_seq_bt(ts, s::Union{Vector, Tuple}) = mapreduce(x -> _contract_seq_bt(ts, x), *, s)

TensorInterface.apply(o::BlockTensor, t::BlockTensor) = TensorInterface.noprime(o * t)

# ── Diagonal operations ─────────────────────────────────────────────────────────────────

function TensorInterface.map_diag!(f::Function, out::BlockTensor, t::BlockTensor)
    ndims(t) == 2 || error("map_diag: expected a 2-index BlockTensor")
    out === t || error("map_diag!: graded prototype only supports in-place (out === t)")
    for (k, b) in t.blocks
        k[1] == k[2] || continue
        for x in 1:minimum(size(b))
            b[x, x] = f(b[x, x])
        end
    end
    return out
end
function TensorInterface.map_diag(f::Function, t::BlockTensor)
    out = copy(t)
    TensorInterface.map_diag!(f, out, out)
    return out
end

# ── Adapt / storage ─────────────────────────────────────────────────────────────────────

function Adapt.adapt_structure(elt::Type{<:Number}, t::BlockTensor)
    return BlockTensor(copy(t.inds), Dict(k => convert(Array{elt}, b) for (k, b) in t.blocks))
end
function Adapt.adapt_structure(to::Type{<:AbstractVector}, t::BlockTensor)
    return BlockTensor(copy(t.inds), Dict(k => reshape(adapt(to, vec(copy(b))), size(b)) for (k, b) in t.blocks))
end

# ── Factorizations (per charge class, global truncation) ────────────────────────────────

#Connected components of the bipartite (left-signature → right-signature) adjacency of the
#present blocks. Each class factorizes independently.
function _charge_classes(keys_, lpos::Vector{Int}, rpos::Vector{Int})
    sigL(k) = ntuple(x -> k[lpos[x]], length(lpos))
    sigR(k) = ntuple(x -> k[rpos[x]], length(rpos))
    #union-find over left signatures via shared right signatures
    classes = Vector{Tuple{Vector, Vector, Vector}}()  # (σLs, σRs, block keys)
    assignment = Dict{Any, Int}()
    for k in keys_
        l, r = sigL(k), sigR(k)
        cl = get(assignment, (:L, l), 0)
        cr = get(assignment, (:R, r), 0)
        if cl == 0 && cr == 0
            push!(classes, (Any[l], Any[r], Any[k]))
            assignment[(:L, l)] = assignment[(:R, r)] = length(classes)
        elseif cl == 0
            push!(classes[cr][1], l)
            push!(classes[cr][3], k)
            assignment[(:L, l)] = cr
        elseif cr == 0
            push!(classes[cl][2], r)
            push!(classes[cl][3], k)
            assignment[(:R, r)] = cl
        elseif cl == cr
            push!(classes[cl][3], k)
        else
            #merge cr into cl
            keep, drop = min(cl, cr), max(cl, cr)
            append!(classes[keep][1], classes[drop][1])
            append!(classes[keep][2], classes[drop][2])
            append!(classes[keep][3], classes[drop][3])
            for σ in classes[drop][1]
                assignment[(:L, σ)] = keep
            end
            for σ in classes[drop][2]
                assignment[(:R, σ)] = keep
            end
            classes[drop] = (Any[], Any[], Any[])
            push!(classes[keep][3], k)
        end
    end
    return [c for c in classes if !isempty(c[3])]
end

_sig_dim(is::Vector{<:KIndex}, pos::Vector{Int}, σ) = prod(Int[space(is[pos[x]]).dims[σ[x]] for x in eachindex(pos)]; init = 1)

#Assemble the dense matrix of one charge class. Returns (M, row offsets by σL, col offsets by σR).
function _class_matrix(t::BlockTensor{T}, lpos, rpos, σLs, σRs, keys_) where {T}
    rowoff = Dict{Any, Int}()
    off = 0
    for σ in σLs
        rowoff[σ] = off
        off += _sig_dim(t.inds, lpos, σ)
    end
    rows = off
    coloff = Dict{Any, Int}()
    off = 0
    for σ in σRs
        coloff[σ] = off
        off += _sig_dim(t.inds, rpos, σ)
    end
    cols = off
    M = zeros(T, rows, cols)
    perm = vcat(lpos, rpos)
    for k in unique(keys_)
        b = permutedims(t.blocks[k], perm)
        σl = ntuple(x -> k[lpos[x]], length(lpos))
        σr = ntuple(x -> k[rpos[x]], length(rpos))
        rd, cd = _sig_dim(t.inds, lpos, σl), _sig_dim(t.inds, rpos, σr)
        M[(rowoff[σl] + 1):(rowoff[σl] + rd), (coloff[σr] + 1):(coloff[σr] + cd)] = reshape(b, rd, cd)
    end
    return M, rowoff, coloff
end

#Scatter a (rows × k) class factor back into blocks keyed (σL..., class sector)
function _scatter_left!(blocks, t::BlockTensor, lpos, σLs, rowoff, U::Matrix, csector::Int)
    k = size(U, 2)
    for σ in σLs
        rd = _sig_dim(t.inds, lpos, σ)
        blk = reshape(U[(rowoff[σ] + 1):(rowoff[σ] + rd), :], (Int[space(t.inds[lpos[x]]).dims[σ[x]] for x in eachindex(lpos)]..., k))
        iszero(norm(blk)) && continue
        blocks[(σ..., csector)] = blk
    end
    return blocks
end
function _scatter_right!(blocks, t::BlockTensor, rpos, σRs, coloff, Vt::Matrix, csector::Int)
    k = size(Vt, 1)
    for σ in σRs
        cd = _sig_dim(t.inds, rpos, σ)
        blk = reshape(Vt[:, (coloff[σ] + 1):(coloff[σ] + cd)], (k, Int[space(t.inds[rpos[x]]).dims[σ[x]] for x in eachindex(rpos)]...))
        iszero(norm(blk)) && continue
        blocks[(csector, σ...)] = blk
    end
    return blocks
end

_class_charge(t::BlockTensor, lpos, σL) = sum(Int[space(t.inds[lpos[x]]).charges[σL[x]] for x in eachindex(lpos)]; init = 0)

function _split_positions(t::BlockTensor, linds::Vector{<:KIndex})
    lv = filter(i -> i ∈ t.inds, linds)
    lpos = Int[findfirst(==(i), t.inds) for i in lv]
    rpos = setdiff(1:ndims(t), lpos)
    return lpos, rpos, t.inds[lpos], t.inds[rpos]
end

function TensorInterface.factorize_svd(
        t::BlockTensor, linds;
        ortho = "none", singular_values! = nothing,
        maxdim = nothing, cutoff = nothing, kwargs...,
    )
    ortho == "none" || error("factorize_svd: graded prototype only implements ortho = \"none\"")
    T = eltype(t)
    lpos, rpos, li, ri = _split_positions(t, _indvec(linds))
    classes = _charge_classes(collect(keys(t.blocks)), lpos, rpos)

    factors = []  # (σLs, σRs, rowoff, coloff, U, S, Vt, charge)
    all_s2 = Float64[]
    for (σLs, σRs, ks) in classes
        M, rowoff, coloff = _class_matrix(t, lpos, rpos, σLs, σRs, ks)
        U, S, Vt = svd_compact(M)
        s = diag(S)
        push!(factors, (σLs, σRs, rowoff, coloff, U, s, Vt, _class_charge(t, lpos, first(σLs))))
        append!(all_s2, abs2.(s))
    end

    #Global truncation: keep the largest singular values across all classes, ITensors
    #convention for the discarded weight.
    sort!(all_s2; rev = true)
    total = sum(all_s2; init = 0.0)
    nkeep = length(all_s2)
    isnothing(maxdim) || (nkeep = min(nkeep, Int(maxdim)))
    if !isnothing(cutoff) && total > 0
        discarded = 0.0
        while nkeep > 1
            d2 = discarded + all_s2[nkeep]
            (d2 / total) > cutoff && break
            discarded = d2
            nkeep -= 1
        end
    end
    threshold2 = nkeep < length(all_s2) ? all_s2[nkeep] : 0.0
    truncerr = total > 0 ? sum(@view all_s2[(nkeep + 1):end]) / total : 0.0

    #Per-class kept counts (values strictly above the threshold always kept; ties broken by
    #class order until the global budget is exhausted)
    kept = zeros(Int, length(factors))
    budget = nkeep
    for (ci, f) in enumerate(factors)
        kept[ci] = count(x -> abs2(x) > threshold2, f[6])
        budget -= kept[ci]
    end
    for (ci, f) in enumerate(factors)
        budget <= 0 && break
        ties = count(x -> abs2(x) == threshold2 && threshold2 > 0, f[6])
        extra = min(ties, budget)
        kept[ci] += extra
        budget -= extra
    end

    #Bond space: one sector per class with kept > 0, deterministically ordered by charge
    order = sortperm([f[8] for f in factors])
    sectors = [(f = factors[ci]; (f[8], kept[ci])) for ci in order if kept[ci] > 0]
    isempty(sectors) && error("factorize_svd: everything truncated away")
    bond = GradedSpace(Int[first.(sectors)...], Int[last.(sectors)...])
    u = KIndex(bond, "Link,u")
    v = KIndex(bond, "Link,v")
    up = TensorInterface.prime(u)

    NL, NR = length(lpos) + 1, length(rpos) + 1
    f1_blocks = Dict{NTuple{NL, Int}, Array{T, NL}}()
    f2_blocks = Dict{NTuple{NR, Int}, Array{T, NR}}()
    sv_blocks = Dict{NTuple{2, Int}, Array{real(T) === T ? T : real(T), 2}}()
    csector = 0
    for ci in order
        kept[ci] > 0 || continue
        csector += 1
        σLs, σRs, rowoff, coloff, U, s, Vt, _ = factors[ci]
        kk = kept[ci]
        sq = sqrt.(s[1:kk])
        _scatter_left!(f1_blocks, t, lpos, σLs, rowoff, U[:, 1:kk] .* reshape(T.(sq), 1, kk), csector)
        _scatter_right!(f2_blocks, t, rpos, σRs, coloff, reshape(T.(sq), kk, 1) .* Vt[1:kk, :], csector)
        sv_blocks[(csector, csector)] = Matrix(Diagonal(s[1:kk]))
    end

    F1 = BlockTensor(vcat(li, [up]), f1_blocks)
    F2 = BlockTensor(vcat([up], ri), f2_blocks)
    if singular_values! !== nothing
        singular_values![] = BlockTensor([u, v], sv_blocks)
    end
    return F1, F2, KSpectrum(sort(all_s2; rev = true)[1:nkeep], truncerr)
end

function LinearAlgebra.qr(t::BlockTensor{T}, linds; kwargs...) where {T}
    lpos, rpos, li, ri = _split_positions(t, _indvec(linds))
    classes = _charge_classes(collect(keys(t.blocks)), lpos, rpos)
    factors = []
    for (σLs, σRs, ks) in classes
        M, rowoff, coloff = _class_matrix(t, lpos, rpos, σLs, σRs, ks)
        Q, R = qr_compact(M)
        push!(factors, (σLs, σRs, rowoff, coloff, Q, R, _class_charge(t, lpos, first(σLs))))
    end
    order = sortperm([f[7] for f in factors])
    bond = GradedSpace(Int[factors[ci][7] for ci in order], Int[size(factors[ci][5], 2) for ci in order])
    b = KIndex(bond, "Link,qr")
    NL, NR = length(lpos) + 1, length(rpos) + 1
    q_blocks = Dict{NTuple{NL, Int}, Array{T, NL}}()
    r_blocks = Dict{NTuple{NR, Int}, Array{T, NR}}()
    for (csector, ci) in enumerate(order)
        σLs, σRs, rowoff, coloff, Q, R, _ = factors[ci]
        _scatter_left!(q_blocks, t, lpos, σLs, rowoff, Q, csector)
        _scatter_right!(r_blocks, t, rpos, σRs, coloff, R, csector)
    end
    return BlockTensor(vcat(li, [b]), q_blocks), BlockTensor(vcat([b], ri), r_blocks)
end

function LinearAlgebra.eigen(t::BlockTensor{T}, linds, rinds; ishermitian::Bool = false, kwargs...) where {T}
    ishermitian || error("eigen: only ishermitian = true is implemented for BlockTensors")
    lpos, rpos, li, ri = _split_positions(t, _indvec(linds))
    t.inds[rpos] == _indvec(rinds) || error("eigen: rinds don't match the remaining indices")
    classes = _charge_classes(collect(keys(t.blocks)), lpos, rpos)
    factors = []
    for (σLs, σRs, ks) in classes
        M, rowoff, coloff = _class_matrix(t, lpos, rpos, σLs, σRs, ks)
        size(M, 1) == size(M, 2) || error("eigen: non-square charge class")
        D, V = eigh_full((M + M') / 2)
        push!(factors, (σLs, σRs, rowoff, coloff, V, diag(D), _class_charge(t, lpos, first(σLs))))
    end
    order = sortperm([f[7] for f in factors])
    bond = GradedSpace(Int[factors[ci][7] for ci in order], Int[length(factors[ci][6]) for ci in order])
    lk = KIndex(bond, "Link,eigen")
    ND = 2
    NU = length(rpos) + 1
    d_blocks = Dict{NTuple{2, Int}, Array{real(T), 2}}()
    u_blocks = Dict{NTuple{NU, Int}, Array{T, NU}}()
    for (csector, ci) in enumerate(order)
        σLs, σRs, rowoff, coloff, V, vals, _ = factors[ci]
        d_blocks[(csector, csector)] = Matrix(Diagonal(vals))
        #U carries the rinds: rows of V live in the left space, but for a hermitian
        #(l, l′)-paired tensor the two spaces coincide sector-by-sector.
        _scatter_right!(u_blocks, t, rpos, σRs, coloff, collect(transpose(V)), csector)
    end
    D = BlockTensor([TensorInterface.prime(lk), lk], d_blocks)
    #_scatter_right! keys (csector, σ...) with bond first; U's index order is (rinds..., lk)
    u_blocks2 = Dict{NTuple{NU, Int}, Array{T, NU}}()
    for (k, b) in u_blocks
        nk = (k[2:end]..., k[1])
        u_blocks2[nk] = permutedims(b, vcat(2:NU, 1))
    end
    U = BlockTensor(vcat(ri, [lk]), u_blocks2)
    return D, U
end

function LinearAlgebra.eigen(t::BlockTensor; ishermitian::Bool = false, kwargs...)
    lv = filter(i -> i.plev == 0, t.inds)
    rv = collect(KIndex, TensorInterface.prime.(lv))
    D, U = LinearAlgebra.eigen(t, lv, rv; ishermitian, kwargs...)
    return D, TensorInterface.replaceinds(U, rv, lv)
end

#Abstractly-typed tensor lists (e.g. from Any-valued network dictionaries) route here
function TensorInterface.contract(ts::Vector; kwargs...)
    all(t -> t isa KTensor, ts) && return TensorInterface.contract(collect(KTensor, ts); kwargs...)
    all(t -> t isa BlockTensor, ts) && return TensorInterface.contract(collect(BlockTensor, ts); kwargs...)
    return error("contract: expected a homogeneous tensor list, got $(unique(typeof.(ts)))")
end
