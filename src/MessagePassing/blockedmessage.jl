using ITensors: Algorithm, array, dim

#
# A memory-bounded message update for the degree-3 vertex of a double-layer (norm)
# network, registered as the "blocked" message-update algorithm:
#
#     update(bpc; maxiter, tolerance, message_update_alg = Algorithm("blocked"; b = 64))
#
# It specialises that one case and falls back to "contract" everywhere else, so it is safe
# to enable globally on a graph with mixed vertex degrees.
#
# The stock path holds the ket, the bra, both factor-sized intermediates and two
# permutedims scratch copies simultaneously -- roughly 6 × one factor, where a "factor" is
# the S·χ³ elements of one vertex tensor. This holds
#
#     the network's own tensor + one aligned copy + two S·χ²·b block buffers
#         = (2 + 2b/χ) × one factor
#
# for two reasons. First, the bra is never materialised: `bp_factors` builds it as
# `dag(prime(T))` with the site indices replaced back to unprimed, which is exactly
# `conj(T)` sharing the ket's site legs, so `adjoint` supplies it to the closing gemm as a
# BLAS/cuBLAS 'C' flag. Second, the contraction is blocked over the outgoing leg, so no
# factor-sized intermediate is ever formed -- only a block's worth. At χ=1024 in
# ComplexF32 that is 68 GiB rather than 384 GiB.
#
# `b` trades peak against very little: the closing gemm's arithmetic intensity is
# b flops/byte, so b = 64 is already compute-bound in fp32 and raising it only grows the
# peak. It is clamped to χ.
#
# Runs on CPU or GPU unchanged -- only mul!, permutedims!, reshape and view, all on a flat
# contiguous buffer, with no scalar indexing.
#

# One flat buffer, carved into the aligned copy, two block buffers and the output.
#
# `BeliefPropagationCacheMPI` carries it in a `scratch` field, so it is grown once and then
# reused across every edge and every sweep -- the hot path allocates nothing. Any other
# cache type has no such field and falls back to allocating per call: the *peak* is
# unchanged (same buffer, same size) but it churns the allocator, so the reusing path is
# what the MPI runs should use.
message_scratch_length(S, chi, b) = S * chi^3 + 2 * S * chi^2 * b + chi^2

message_scratch(::AbstractBeliefPropagationCache) = Base.RefValue{Any}(Bool[])

# Grow the buffer to fit. The type check also catches a change of element type or device,
# in which case the old buffer is unusable and is replaced.
function scratch_buffer!(ref::Base.RefValue{Any}, proto::AbstractVector, n::Int)
    s = ref[]
    if !(s isa typeof(proto)) || length(s) < n
        s = similar(proto, n)
        ref[] = s
    end
    return s
end

# Give the buffer back between updates: it is the largest single allocation the MPI cache
# holds, and nothing outside `updated_message` reads it, so holding it across gate
# application costs `apply_gate!` the memory its SVD needs. `scratch_buffer!` rebuilds it on
# the next message update, where the `Bool[]` sentinel fails its `isa typeof(proto)` check.
function release_message_scratch!(bp_cache::AbstractBeliefPropagationCache)
    ref = message_scratch(bp_cache)
    s = ref[]
    # `resize!` on a CuVector returns the device allocation to CUDA's pool here rather than
    # at the next finalizer run; on a Vector it does nothing that dropping the reference
    # below would not already do.
    applicable(resize!, s, 0) && resize!(s, 0)
    ref[] = Bool[]
    return bp_cache
end

# out[l_e', l_e] = Σ_{s,l_a,l_b} conj(T)[s,l_a',l_b',l_e'] ma[l_a,l_a'] mb[l_b,l_b'] T[s,l_a,l_b,l_e]
#
# `A` is the ket tensor's raw array in whatever index order the ITensor happens to have and
# `perm` takes that order to (sites…, l_a, l_b, l_e). The permutation is folded into the
# aligned copy, so an arbitrary index order costs nothing beyond the copy made anyway.
# `ma`/`mb` are χ×χ matrices oriented (l, l') and must be in stored leg order.
function blocked_message!(outT_v, buf1, buf2, Tp, A, perm, ma, mb, S, chi, b)
    permutedims!(reshape(Tp, ntuple(i -> size(A, perm[i]), length(perm))), A, perm)
    Tclose = reshape(Tp, S * chi * chi, chi)   # a matrix: l_e is trailing after the permute
    outT = reshape(outT_v, chi, chi)
    slab = S * chi * chi

    for lo in 1:b:chi
        nb = min(b, chi - lo + 1)
        n = slab * nb
        cols = lo:(lo + nb - 1)
        blk = reshape(view(Tp, ((lo - 1) * slab + 1):((lo - 1) * slab + n)), S, chi, chi, nb)

        # contract l_b, moving the contracted leg first so each step is one plain gemm
        P1 = reshape(view(buf1, 1:n), chi, S, chi, nb)
        permutedims!(P1, blk, (3, 1, 2, 4))
        G1 = reshape(view(buf2, 1:n), chi, S * chi * nb)
        mul!(G1, transpose(mb), reshape(P1, chi, S * chi * nb))

        # contract l_a
        P2 = reshape(view(buf1, 1:n), chi, chi, S, nb)
        permutedims!(P2, reshape(G1, chi, S, chi, nb), (3, 1, 2, 4))
        G2 = reshape(view(buf2, 1:n), chi, chi * S * nb)
        mul!(G2, transpose(ma), reshape(P2, chi, chi * S * nb))

        # close against conj(T): contracted legs leading and in the tensor's own order, so
        # `adjoint` conjugates the bra for free
        P3 = reshape(view(buf1, 1:n), S, chi, chi, nb)
        permutedims!(P3, reshape(G2, chi, chi, S, nb), (3, 1, 2, 4))
        mul!(view(outT, :, cols), adjoint(Tclose), reshape(P3, slab, nb))
    end
    return outT
end

default_blocked_blocksize(::Algorithm"blocked") = 64

function set_default_kwargs(alg::Algorithm"blocked", bp_cache::AbstractBeliefPropagationCache)
    normalize = get(alg.kwargs, :normalize, default_normalize(Algorithm("contract")))
    b = get(alg.kwargs, :b, default_blocked_blocksize(alg))
    return Algorithm("blocked"; normalize, b)
end

function updated_message(
        alg::Algorithm"blocked", bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge
    )
    fallback() = updated_message(
        set_default_kwargs(Algorithm("contract"; normalize = alg.kwargs.normalize), bp_cache),
        bp_cache, edge
    )

    v = src(edge)
    tn = network(bp_cache)
    ms = incoming_messages(bp_cache, v; ignore_edges = (reverse(edge),))
    length(ms) == 2 || return fallback()          # only the degree-3 vertex is specialised

    les = virtualinds(tn, edge)                   # an edge may carry several virtual inds
    length(les) == 1 || return fallback()
    le = only(les)

    T = tn[v]
    is = collect(inds(T))
    legs = [only(commoninds(m, T)) for m in ms]
    chi = dim(le)
    all(l -> dim(l) == chi, legs) || return fallback()
    nelt = prod(dim, is)
    rem(nelt, chi^3) == 0 || return fallback()
    S = nelt ÷ chi^3

    # Incoming legs in stored order: the block permutation and the closing contraction both
    # assume l_a precedes l_b in the aligned copy.
    ord = sortperm([findfirst(==(l), is) for l in legs])
    la, lb = legs[ord]
    ma, mb = ms[ord]

    b = min(alg.kwargs.b, chi)
    A = array(T)                                  # dims follow inds(T); a view when dense
    sitepos = [i for i in eachindex(is) if is[i] ∉ (la, lb, le)]
    perm = (sitepos..., findfirst(==(la), is), findfirst(==(lb), is), findfirst(==(le), is))

    s = scratch_buffer!(message_scratch(bp_cache), vec(A), message_scratch_length(S, chi, b))
    o1 = S * chi^3
    o2 = o1 + S * chi^2 * b
    o3 = o2 + S * chi^2 * b
    Tp, buf1, buf2 = view(s, 1:o1), view(s, (o1 + 1):o2), view(s, (o2 + 1):o3)
    outT = view(s, (o3 + 1):(o3 + chi^2))

    # Orient each message as (l, l') so `transpose` inside the kernel gives (l', l).
    mat(m, l) = array(m, l, only(setdiff(collect(inds(m)), [l])))

    out = blocked_message!(outT, buf1, buf2, Tp, A, perm,
        mat(ma, la), mat(mb, lb), S, chi, b)

    # `out` is [l_e', l_e] -- label it rather than transpose. Copy so the message does not
    # alias the scratch that the next edge overwrites.
    m = itensor(copy(out), prime(dag(le)), le)
    if alg.kwargs.normalize
        message_norm = sum(m)
        if !iszero(message_norm)
            m = m / message_norm
        end
    end
    # No contraction sequence is used, so `seq_changed = false` leaves the sequence cache
    # untouched.
    return m, (v => edge, nothing, false)
end
