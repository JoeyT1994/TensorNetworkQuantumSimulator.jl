# Fused double-layer BP kernels on dense data. Each incoming message is absorbed along its bond into
# a pair of reused buffers, and conj(ψ) is folded into the closing GEMM, so the peak is ψ plus two
# buffers of ψ's size. On the CPU ψ's data is viewed as (pre, D, post) and middle bonds are absorbed
# slice by slice, with no permutations. On other devices (GPUs) the many small GEMMs are slow, so the
# bond is instead permuted to the front and absorbed by one GEMM (the "rotated" chain). Structures
# not recognised return `nothing`.

# ψ's data as a strided array in the order of `inds(ψ)` (aliasing its storage), or nothing.
function fused_data(ψ::ITensor)
    ITensors.hasqns(ψ) && return nothing
    ITensors.storage(ψ) isa NDTensors.Dense || return nothing
    A = reshape(ITensors.data(ψ), ITensors.dims(ψ))
    A isa StridedArray || return nothing
    return A
end

# The legs (x, x′) of a dense two-leg tensor on x and x′ = prime(x), or nothing.
function primed_pair(t::ITensor)
    (ndims(t) == 2 && !ITensors.hasqns(t)) || return nothing
    i1, i2 = ITensors.inds(t)
    x, xp = ITensors.plev(i1) == 0 ? (i1, i2) : (i2, i1)
    (ITensors.plev(x) == 0 && xp == ITensors.prime(x)) || return nothing
    return x, xp
end

# For a doubled message on bond `b` of ψ (legs (b, b′)), the position of `b` among ψ's legs and the
# message matrix M[b, b′]; nothing otherwise.
function fused_message_slot(m::ITensor, ψinds::Vector{<:Index})
    legs = primed_pair(m)
    isnothing(legs) && return nothing
    b, bp = legs
    k = findfirst(==(b), ψinds)
    k === nothing && return nothing
    M = ITensors.array(m, b, bp)
    M isa StridedArray || return nothing
    return k, M
end

# Two buffers for the chain intermediates, pooled in the kernel workspace if one is open (see
# `TI.with_kernel_workspace`), by storage family (Array, CuArray, ...) and element type. Kernels
# never return a buffer.
function fused_buffers(ref::AbstractArray, ::Type{T}, n::Integer) where {T}
    pool = TI.kernel_workspace()
    isnothing(pool) && return (similar(ref, T, n), similar(ref, T, n))
    key = (Base.typename(typeof(ref)).wrapper, T)
    bufs = get(pool, key, nothing)
    if bufs === nothing || length(bufs[1]) < n
        bufs = (similar(ref, T, n), similar(ref, T, n))
        pool[key] = bufs
    end
    return bufs
end

# out[a, b′, c] = Σ_b cur[a, b, c] · M[b, b′] along dimension k, on the CPU. For small `pre` a
# single GEMM with kron(Mᵀ, 𝟙_pre) beats `post` tiny per-slice GEMMs despite the extra flops.
function absorb_dim!(out::AbstractArray, cur::AbstractArray, M::AbstractMatrix, k::Integer)
    dims = size(cur)
    pre = prod(dims[1:(k - 1)]; init = 1); D = dims[k]; post = prod(dims[(k + 1):end]; init = 1)
    C = reshape(cur, pre, D, post); O = reshape(out, pre, D, post)
    if pre == 1
        LinearAlgebra.mul!(reshape(O, D, post), transpose(M), reshape(C, D, post))
    elseif post == 1
        LinearAlgebra.mul!(reshape(O, pre, D), reshape(C, pre, D), M)
    elseif pre <= 8 && post > 8
        K = kron(transpose(M), Matrix{eltype(M)}(I, pre, pre))
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
function fused_chain(A::StridedArray, slots::Vector)
    T = promote_type(eltype(A), (eltype(M) for (_, M) in slots)...)
    cur = T == eltype(A) ? A : T.(A)
    isempty(slots) && return cur
    b1, b2 = fused_buffers(A, T, length(A))
    buf1 = reshape(view(b1, 1:length(A)), size(A)); buf2 = reshape(view(b2, 1:length(A)), size(A))
    out = buf1
    for (k, M) in slots
        absorb_dim!(out, cur, M, k)
        cur, out = out, (out === buf1 ? buf2 : buf1)
    end
    return cur
end

# `src`, with its legs in `order` (positions in `inds(ψ)`), permuted to `neworder` into `dst`.
function permute_into!(dst::AbstractVector, src::AbstractVector, dims, order, neworder)
    perm = [findfirst(==(l), order) for l in neworder]
    permutedims!(reshape(dst, dims[neworder]...), reshape(src, dims[order]...), perm)
    return dst
end

# As `fused_chain`, with every absorption a single GEMM out[b′, rest] = Mᵀ C[b, rest] on a bond
# first permuted to the front. The leg `kout` (if any) is kept last throughout. Returns the chain
# and ψ's data, both flat with their legs in the same order (ψ permuted into the free buffer).
function rotated_chain(A::StridedArray, slots::Vector, kout)
    T = promote_type(eltype(A), (eltype(M) for (_, M) in slots)...)
    A = T == eltype(A) ? A : T.(A)
    n = length(A); dims = size(A)
    b1, b2 = fused_buffers(A, T, n)
    bufs = (view(b1, 1:n), view(b2, 1:n))
    cur = vec(A)
    order = collect(1:ndims(A))
    free() = cur === bufs[1] ? bufs[2] : bufs[1]
    function reorder!(neworder)
        cur = permute_into!(free(), cur, dims, order, neworder)
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
        Mt = eltype(M) == T ? transpose(M) : transpose(T.(M))
        LinearAlgebra.mul!(reshape(out, D, n ÷ D), Mt, reshape(cur, D, n ÷ D))
        cur = out
    end
    Ψ = order == 1:ndims(A) ? vec(A) : permute_into!(free(), vec(A), dims, collect(1:ndims(A)), order)
    return cur, Ψ
end

# R[b′, b] = Σ_{a,c} conj(Ψ[a, b′, c]) T[a, b, c] over the split (pre, D, post) of the outgoing bond.
# Small `pre`: one (pre·D)² GEMM over c followed by a partial trace over a; large `pre`: c-slices.
function close_bond!(R::AbstractMatrix, Ψ3::AbstractArray{<:Any, 3}, T3::AbstractArray{<:Any, 3})
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
# not a doubled one or two messages share a bond.
function fused_slots(ψinds::Vector{<:Index}, incoming::Vector)
    slots = Tuple{Int, StridedArray}[]
    covered = falses(length(ψinds))
    for m in incoming
        m isa ITensor || return nothing
        sl = fused_message_slot(m, ψinds)
        sl === nothing && return nothing
        covered[sl[1]] && return nothing
        covered[sl[1]] = true
        push!(slots, sl)
    end
    return slots, covered
end

# ψ's data, its legs, and the slots and covered legs of `incoming`; nothing if either is unsupported.
function fused_setup(ψ::ITensor, incoming::Vector)
    A = fused_data(ψ)
    A === nothing && return nothing
    ψinds = collect(Index, ITensors.inds(ψ))
    sc = fused_slots(ψinds, incoming)
    sc === nothing && return nothing
    return A, ψinds, sc...
end

function TI.fused_norm_message(ψ::ITensor, sinds, incoming::Vector)
    setup = fused_setup(ψ, incoming)
    isnothing(setup) && return nothing
    A, ψinds, slots, covered = setup
    outgoing = [k for k in eachindex(ψinds) if !covered[k] && !(ψinds[k] in sinds)]
    length(outgoing) == 1 || return nothing
    kout = only(outgoing)
    D = size(A, kout)
    if A isa Array
        Tm, Ψ = fused_chain(A, slots), A
        pre, post = prod(size(A)[1:(kout - 1)]; init = 1), prod(size(A)[(kout + 1):end]; init = 1)
    else
        Tm, Ψ = rotated_chain(A, slots, kout)   # with the outgoing bond last
        pre, post = length(A) ÷ D, 1
    end
    R = close_bond!(similar(A, eltype(Tm), D, D), reshape(Ψ, pre, D, post), reshape(Tm, pre, D, post))
    bout = ψinds[kout]
    return ITensors.itensor(R, ITensors.prime(bout), bout)
end

# Single-vertex closure ψ·Πm·(ops)·ψ̄ → scalar; `ops` are (s′, s) operator tensors on site legs.
function TI.fused_norm_scalar(ψ::ITensor, sinds, incoming::Vector, ops::Vector)
    setup = fused_setup(ψ, incoming)
    isnothing(setup) && return nothing
    A, ψinds, slots, covered = setup
    for o in ops
        legs = o isa ITensor ? primed_pair(o) : nothing
        isnothing(legs) && return nothing
        s, sp = legs
        k = findfirst(==(s), ψinds)
        (s in sinds && !isnothing(k) && !covered[k]) || return nothing
        covered[k] = true
        push!(slots, (k, ITensors.array(o, s, sp)))   # M[s, s′] = O[s′, s]
    end
    all(k -> covered[k] || ψinds[k] in sinds, eachindex(ψinds)) || return nothing
    Tm, Ψ = A isa Array ? (fused_chain(A, slots), A) : rotated_chain(A, slots, nothing)
    return LinearAlgebra.dot(vec(Ψ), vec(Tm))   # Σ conj(ψ) T
end
