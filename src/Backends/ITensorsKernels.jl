# ── Fused double-layer BP kernels (dense data) ──────────────────────────────────────────
# Ported from FixesV2 (src/Tensors/ITensorBackend.jl, commit d41fa35 "Fused dense BP
# message/closure kernel"), on ITensors storage instead of ITensorBase's.
#
# The BP message update ψ·Πm·ψ̄ is the inner loop of every sweep. The generic sequence path
# matricises both operands of every pairwise contraction (a permuted copy of the F-sized chain
# intermediate each time). Here each incoming message is absorbed along its bond with slice GEMMs
# into a ping-pong buffer (no permutations: the bond dimension is split out of the column-major
# layout as (pre, D, post)), and the conjugate of ψ is folded into the closing GEMM. Live memory: ψ
# and two buffers = 3F, plus the D² message. Structure not recognised → `nothing` (generic path).

# ψ's data as a strided array in the order of `inds(ψ)` (aliasing its storage), or nothing when the
# kernel does not apply.
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

# Task-local buffer pool for the chain intermediates: two F-sized buffers per element type, reused
# across messages (the kernel returns only the D² result, never a buffer). Live memory during a
# message update is therefore ψ + 2 buffers = 3F + change, and steady-state heap churn ~0.
function _fused_buffers(ref::AbstractArray, ::Type{T}, n::Integer) where {T}
    # keyed by the storage family of `ref` (Array, CuArray, …) and the element type, so device data
    # gets device buffers
    key = (Base.typename(typeof(ref)).wrapper, T)
    pool = get!(task_local_storage(), :tnqs_fused_buffers) do
        Dict{Any, Tuple{AbstractVector, AbstractVector}}()
    end::Dict{Any, Tuple{AbstractVector, AbstractVector}}
    bufs = get(pool, key, nothing)
    if bufs === nothing || length(bufs[1]) < n
        bufs = (similar(ref, T, n), similar(ref, T, n))
        pool[key] = bufs
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
        Id = similar(M, pre, pre); fill!(Id, zero(eltype(M))); view(Id, LinearAlgebra.diagind(Id)) .= one(eltype(M))
        K = kron(transpose(M), Id)                                     # (pre·D)² — small, on M's device
        LinearAlgebra.mul!(reshape(O, pre * D, post), K, reshape(C, pre * D, post))
    else
        for c in 1:post
            LinearAlgebra.mul!(view(O, :, :, c), view(C, :, :, c), M)
        end
    end
    return out
end

# Absorb every incoming message (and optional site operators) into ψ's data. Returns the chain
# result (a pooled buffer; ψ's own data is never written) or ψ's data itself when there is nothing
# to absorb. `slots` are (k, M) pairs.
function _fused_chain(A::StridedArray, slots::Vector)
    T = promote_type(eltype(A), (eltype(M) for (_, M) in slots)...)
    isempty(slots) && return T == eltype(A) ? A : T.(A)
    cur = T == eltype(A) ? A : T.(A)
    b1, b2 = _fused_buffers(A, T, length(A))
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
        for a in 1:pre                                                      # partial trace over a: conj(Ψ) T = conj(Ψ T̄)
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

# The (k, M) slots of `incoming` on ψ's legs, with the covered positions; nothing when a message is
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
    A = _fused_data(ψ)
    A === nothing && return nothing
    ψinds = collect(Index, ITensors.inds(ψ))
    sc = _fused_slots(ψinds, incoming)
    sc === nothing && return nothing
    slots, covered = sc
    open = [k for k in eachindex(ψinds) if !covered[k] && !(ψinds[k] in sinds)]
    length(open) == 1 || return nothing
    kout = only(open)
    Tm = _fused_chain(A, slots)
    dims = size(A)
    pre = prod(dims[1:(kout - 1)]; init = 1); D = dims[kout]; post = prod(dims[(kout + 1):end]; init = 1)
    T = eltype(Tm)
    Ψ3 = reshape(A, pre, D, post); T3 = reshape(Tm, pre, D, post)
    bout = ψinds[kout]; boutp = ITensors.prime(bout)
    R = similar(A, T, D, D)
    if pre == 1
        # R[b, b′] = Σ_c T[b, c] conj(ψ[b′, c])
        LinearAlgebra.mul!(R, reshape(T3, D, post), adjoint(reshape(Ψ3, D, post)))
        legs = (bout, boutp)
    else
        _close_bond!(R, Ψ3, T3)
        legs = (boutp, bout)
    end
    if normalize
        # the 2-norm, as the generic message update normalises
        n = LinearAlgebra.norm(R)
        iszero(n) || rmul!(R, inv(n))
    end
    return ITensors.itensor(R, legs...)
end

# Single-vertex closure ψ·Πm·(ops)·ψ̄ → scalar; `ops` are (s′, s) operator tensors on site legs.
function TI.fused_norm_scalar(ψ::ITensor, sinds, incoming::Vector, ops::Vector)
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
        push!(slots, (k, ITensors.array(o, s, sp)))   # M[s, s′] = O[s′, s]: T[…s′…] = Σ_s ψ[…s…] O[s′, s]
    end
    all(k -> covered[k] || ψinds[k] in sinds, eachindex(ψinds)) || return nothing
    Tm = _fused_chain(A, slots)
    return LinearAlgebra.dot(vec(A), vec(Tm))                    # Σ conj(ψ) T
end
