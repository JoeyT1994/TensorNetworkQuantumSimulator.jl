# ── Fused double-layer BP kernels (dense data) ──────────────────────────────────────────
# Each incoming message is absorbed along its bond into a pair of reused buffers, and conj(ψ) is
# folded into the closing GEMM, so the peak is ψ plus two buffers of ψ's size. On the CPU ψ's
# data is viewed as (pre, D, post) and middle bonds are absorbed slice by slice, with no
# permutations; on other devices (GPUs) the many small GEMMs are slow, so the bond is instead
# permuted to the front and absorbed by one GEMM (the "rotated" chain). Structures not recognised
# return `nothing` (generic path).

# ψ's data as a strided array in the order of `inds(ψ)` (aliasing its storage), or nothing.
function _fused_data(ψ::ITensor)
    ITensors.hasqns(ψ) && return nothing
    ITensors.storage(ψ) isa NDTensors.Dense || return nothing
    A = reshape(ITensors.data(ψ), ITensors.dims(ψ))
    A isa StridedArray || return nothing
    return A
end

# For a standard doubled message on bond `b` of ψ (legs (b, b′) with b′ = prime(b)), the position
# of `b` among ψ's legs and the message matrix M[b, b′]; nothing otherwise.
function _fused_message_slot(m::ITensor, ψinds::Vector{<:Index})
    ndims(m) == 2 || return nothing
    ITensors.hasqns(m) && return nothing
    i1, i2 = ITensors.inds(m)
    b, bp = ITensors.plev(i1) == 0 ? (i1, i2) : (i2, i1)
    (ITensors.plev(b) == 0 && ITensors.plev(bp) == 1 && ITensors.noprime(bp) == b) || return nothing
    k = findfirst(==(b), ψinds)
    k === nothing && return nothing
    M = ITensors.array(m, b, bp)
    M isa StridedArray || return nothing
    return k, M
end

# Two buffers for the chain intermediates, pooled in the kernel workspace (see
# `TI.with_kernel_workspace`) by storage family (Array, CuArray, ...) and element type. Kernels
# never return a buffer.
function _fused_buffers(ref::AbstractArray, ::Type{T}, n::Integer) where {T}
    key = (Base.typename(typeof(ref)).wrapper, T)
    pool = TI.kernel_workspace()
    bufs = get(pool, key, nothing)
    if bufs === nothing || length(bufs[1]) < n
        bufs = (similar(ref, T, n), similar(ref, T, n))
        pool[key] = bufs
    end
    return bufs
end

# out[a, b′, c] = Σ_b cur[a, b, c] · M[b, b′] along dimension k. For small `pre` a single GEMM with
# kron(Mᵀ, 𝟙_pre) beats `post` tiny per-slice GEMMs despite the extra flops.
function _absorb_dim!(out::AbstractArray, cur::AbstractArray, M::AbstractMatrix, k::Integer)
    dims = size(cur)
    pre = prod(dims[1:(k - 1)]; init = 1); D = dims[k]; post = prod(dims[(k + 1):end]; init = 1)
    C = reshape(cur, pre, D, post); O = reshape(out, pre, D, post)
    if pre == 1
        LinearAlgebra.mul!(reshape(O, D, post), transpose(M), reshape(C, D, post))
    elseif post == 1
        LinearAlgebra.mul!(reshape(O, pre, D), reshape(C, pre, D), M)
    elseif pre <= 8 && post > 8
        Id = similar(M, pre, pre)                                      # on M's device
        fill!(Id, zero(eltype(M)))
        view(Id, LinearAlgebra.diagind(Id)) .= one(eltype(M))
        K = kron(transpose(M), Id)
        LinearAlgebra.mul!(reshape(O, pre * D, post), K, reshape(C, pre * D, post))
    else
        for c in 1:post
            LinearAlgebra.mul!(view(O, :, :, c), view(C, :, :, c), M)
        end
    end
    return out
end

# Absorb the (k, M) `slots` into ψ's data (never written). Returns a pooled buffer, or ψ's data
# itself when there is nothing to absorb.
function _fused_chain(A::StridedArray, slots::Vector)
    T = promote_type(eltype(A), (eltype(M) for (_, M) in slots)...)
    cur = T == eltype(A) ? A : T.(A)
    isempty(slots) && return cur
    b1, b2 = _fused_buffers(A, T, length(A))
    buf1 = reshape(view(b1, 1:length(A)), size(A)); buf2 = reshape(view(b2, 1:length(A)), size(A))
    out = buf1
    for (k, M) in slots
        _absorb_dim!(out, cur, M, k)
        cur, out = out, (out === buf1 ? buf2 : buf1)
    end
    return cur
end

# ψ's data with its legs in `order` (positions in `inds(ψ)`) permuted to `neworder`, into `dst`.
function _reorder!(dst::AbstractVector, src::AbstractVector, dims, order, neworder)
    perm = [findfirst(==(l), order) for l in neworder]
    permutedims!(reshape(dst, dims[neworder]...), reshape(src, dims[order]...), perm)
    return dst
end

# As `_fused_chain`, with every absorption a single GEMM out[b′, rest] = Mᵀ C[b, rest] on a bond
# first permuted to the front. The leg `kout` (if any) is kept last throughout. Returns the chain
# and ψ's data, both flat with their legs in the same order (ψ permuted into the free buffer).
function _rotated_chain(A::StridedArray, slots::Vector, kout)
    T = promote_type(eltype(A), (eltype(M) for (_, M) in slots)...)
    n = length(A); dims = size(A)
    b1, b2 = _fused_buffers(A, T, n)
    bufs = (view(b1, 1:n), view(b2, 1:n))
    cur = vec(A)
    order = collect(1:ndims(A))
    free() = cur === bufs[1] ? bufs[2] : bufs[1]
    function reorder!(neworder)
        cur = _reorder!(free(), cur, dims, order, neworder)
        order = neworder
        return nothing
    end
    if kout !== nothing && last(order) != kout
        reorder!([filter(!=(kout), order); kout])
    end
    for (k, M) in slots
        first(order) == k || reorder!([k; filter(!=(k), order)])
        D = dims[k]
        out = free()
        LinearAlgebra.mul!(reshape(out, D, n ÷ D), transpose(M), reshape(cur, D, n ÷ D))
        cur = out
    end
    Ψ = order == 1:ndims(A) ? vec(A) : _reorder!(free(), vec(A), dims, collect(1:ndims(A)), order)
    return cur, Ψ
end

# R[b′, b] = Σ_{a,c} conj(Ψ[a, b′, c]) T[a, b, c] over the split (pre, D, post) of the outgoing bond.
# Small `pre`: one (pre·D)² GEMM over c followed by a partial trace over a; large `pre`: c-slices.
function _close_bond!(R::AbstractMatrix, Ψ3::AbstractArray{<:Any, 3}, T3::AbstractArray{<:Any, 3})
    pre, D, post = size(Ψ3)
    T = eltype(R)
    if post == 1
        LinearAlgebra.mul!(R, adjoint(reshape(Ψ3, pre, D)), reshape(T3, pre, D))
    elseif pre <= 8
        # G[(a, b′), (a2, b)] = Σ_c Ψ T̄, then a partial trace over a, using conj(Ψ) T = conj(Ψ T̄)
        G = reshape(Ψ3, pre * D, post) * adjoint(reshape(T3, pre * D, post))
        fill!(R, zero(T))
        G4 = reshape(G, pre, D, pre, D)
        for a in 1:pre
            R .+= conj.(view(G4, a, :, a, :))
        end
    else
        fill!(R, zero(T))
        for c in 1:post
            LinearAlgebra.mul!(R, adjoint(view(Ψ3, :, :, c)), view(T3, :, :, c), one(T), one(T))
        end
    end
    return R
end

# The (k, M) slots of `incoming` on ψ's legs and the covered positions; nothing when a message is
# not a standard doubled one or two messages share a bond.
function _fused_slots(ψinds::Vector{<:Index}, incoming::Vector)
    slots = Any[]
    covered = falses(length(ψinds))
    for m in incoming
        m isa ITensor || return nothing
        sl = _fused_message_slot(m, ψinds)
        sl === nothing && return nothing
        covered[sl[1]] && return nothing
        covered[sl[1]] = true
        push!(slots, sl)
    end
    return slots, covered
end

function TI.fused_norm_message(ψ::ITensor, sinds, incoming::Vector; normalize::Bool = true)
    return TI.with_kernel_workspace(() -> _fused_norm_message(ψ, sinds, incoming; normalize))
end

function _fused_norm_message(ψ::ITensor, sinds, incoming::Vector; normalize::Bool)
    A = _fused_data(ψ)
    A === nothing && return nothing
    ψinds = collect(Index, ITensors.inds(ψ))
    sc = _fused_slots(ψinds, incoming)
    sc === nothing && return nothing
    slots, covered = sc
    open = [k for k in eachindex(ψinds) if !covered[k] && !(ψinds[k] in sinds)]
    length(open) == 1 || return nothing
    kout = only(open)
    bout = ψinds[kout]; boutp = ITensors.prime(bout)
    D = size(A, kout)
    if !(A isa Array)
        Tm, Ψ = _rotated_chain(A, slots, kout)
        R = similar(A, eltype(Tm), D, D)
        LinearAlgebra.mul!(R, adjoint(reshape(Ψ, :, D)), reshape(Tm, :, D))   # R[b′, b]
        return _finish_message(R, (boutp, bout), normalize)
    end
    Tm = _fused_chain(A, slots)
    dims = size(A)
    pre = prod(dims[1:(kout - 1)]; init = 1); post = prod(dims[(kout + 1):end]; init = 1)
    Ψ3 = reshape(A, pre, D, post); T3 = reshape(Tm, pre, D, post)
    R = similar(A, eltype(Tm), D, D)
    if pre == 1
        # R[b, b′] = Σ_c T[b, c] conj(ψ[b′, c])
        LinearAlgebra.mul!(R, reshape(T3, D, post), adjoint(reshape(Ψ3, D, post)))
        legs = (bout, boutp)
    else
        _close_bond!(R, Ψ3, T3)
        legs = (boutp, bout)
    end
    return _finish_message(R, legs, normalize)
end

function _finish_message(R::AbstractMatrix, legs, normalize::Bool)
    if normalize
        # the 2-norm, as in the generic message update
        n = LinearAlgebra.norm(R)
        iszero(n) || rmul!(R, inv(n))
    end
    return ITensors.itensor(R, legs...)
end

# Single-vertex closure ψ·Πm·(ops)·ψ̄ → scalar; `ops` are (s′, s) operator tensors on site legs.
function TI.fused_norm_scalar(ψ::ITensor, sinds, incoming::Vector, ops::Vector)
    return TI.with_kernel_workspace(() -> _fused_norm_scalar(ψ, sinds, incoming, ops))
end

function _fused_norm_scalar(ψ::ITensor, sinds, incoming::Vector, ops::Vector)
    A = _fused_data(ψ)
    A === nothing && return nothing
    ψinds = collect(Index, ITensors.inds(ψ))
    sc = _fused_slots(ψinds, incoming)
    sc === nothing && return nothing
    slots, covered = sc
    for o in ops
        (o isa ITensor && ndims(o) == 2 && !ITensors.hasqns(o)) || return nothing
        i1, i2 = ITensors.inds(o)
        s, sp = ITensors.plev(i1) == 0 ? (i1, i2) : (i2, i1)
        (ITensors.plev(sp) == 1 && ITensors.noprime(sp) == s && s in sinds) || return nothing
        k = findfirst(==(s), ψinds)
        (k === nothing || covered[k]) && return nothing
        covered[k] = true
        push!(slots, (k, ITensors.array(o, s, sp)))   # M[s, s′] = O[s′, s]
    end
    all(k -> covered[k] || ψinds[k] in sinds, eachindex(ψinds)) || return nothing
    if !(A isa Array)
        Tm, Ψ = _rotated_chain(A, slots, nothing)
        return LinearAlgebra.dot(Ψ, Tm)
    end
    Tm = _fused_chain(A, slots)
    return LinearAlgebra.dot(vec(A), vec(Tm))   # Σ conj(ψ) T
end
