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
message_scratch_length(S, chi, b, nlayers = 1) =
    nlayers * S * chi^3 + 2 * S * chi^2 * b + chi^2

message_scratch(::AbstractBeliefPropagationCache) = Base.RefValue{Any}(Bool[])

# Grow the buffer to fit. The type check also catches a change of element type or device,
# in which case the old buffer is unusable and is replaced.
#
# The old buffer is dropped *before* the replacement is allocated. Otherwise both are live across
# the `similar`, which at S=4, χ=1024 is 32 GiB held while 36 GiB is requested -- and since the
# buffer is regrown every time a bond dimension climbs, that doubling happens repeatedly through a
# circuit. Releasing first lets the allocator hand back the same block.
function scratch_buffer!(ref::Base.RefValue{Any}, proto::AbstractVector, n::Int)
    s = ref[]
    if !(s isa typeof(proto)) || length(s) < n
        ref[] = Bool[]
        # Returned to the allocator here, not left for a finalizer. Without this the grow path
        # leaks on a GPU while `release_message_scratch!` frees eagerly -- and growing is exactly
        # when the device is most likely to be short, since the replacement is the larger buffer.
        free_scratch_buffer!(s)
        s = similar(proto, n)
        ref[] = s
    end
    return s
end

# Hook for returning a device buffer to its allocator eagerly, rather than waiting for the
# finalizer. A no-op by default so nothing here depends on CUDA; a caller on a GPU adds
#
#     TensorNetworkQuantumSimulator.free_scratch_buffer!(x::CuArray) = CUDA.unsafe_free!(x)
#
# Note `CUDA.reclaim()` is deliberately *not* wanted here: `unsafe_free!` puts the block back in
# the pool where the next sweep reuses it, whereas `reclaim` returns it to the driver and forces
# the next allocation to be a fresh `cudaMalloc` of the same size -- the allocation most likely to
# fail when memory is tight.
free_scratch_buffer!(x) = nothing

# Called once a BP solve is done. The scratch is only needed between the first and last message
# update of a sweep sequence; holding it afterwards means a factor-sized buffer squatting while
# gate application allocates its own.
release_message_scratch!(bpc::AbstractBeliefPropagationCache) = bpc

# out[l_e', l_e] = Σ_{s,l_a,l_b} conj(T)[s,l_a',l_b',l_e'] ma[l_a,l_a'] mb[l_b,l_b'] T[s,l_a,l_b,l_e]
#
# `A` is the ket tensor's raw array in whatever index order the ITensor happens to have and
# `perm` takes that order to (sites…, l_a, l_b, l_e). The permutation is folded into the
# aligned copy, so an arbitrary index order costs nothing beyond the copy made anyway.
# `ma`/`mb` are χ×χ matrices oriented (l, l') and must be in stored leg order.
# Bumped whenever the specialised kernel actually runs. Worth having permanently: "blocked agrees
# with contract" is satisfied trivially whenever the blocked path fell back, so without a counter
# a specialisation that has silently gone inert looks exactly like one that works.
const _BLOCKED_MESSAGE_HITS = Ref(0)

function blocked_message!(
        outT_v, buf1, buf2, Tp, A, perm, ma, mb, S, chi, b, Bp = nothing, Barr = nothing,
        Bperm = nothing
    )
    _BLOCKED_MESSAGE_HITS[] += 1
    permutedims!(reshape(Tp, ntuple(i -> size(A, perm[i]), length(perm))), A, perm)
    Tclose = reshape(Tp, S * chi * chi, chi)   # a matrix: l_e is trailing after the permute

    # How the bra reaches the closing gemm is the only difference between a single network's
    # double layer and a form's. When the bra is `conj(ket)` there is nothing to store: `adjoint`
    # hands the conjugation to BLAS/cuBLAS as a 'C' flag on the ket's own aligned copy. A form's
    # bra is a different tensor, so it gets its own aligned copy and a plain `transpose` -- its
    # data is already conjugated, since the bra network is built as `dag(prime(·))`.
    closer = if isnothing(Bp)
        adjoint(Tclose)
    else
        permutedims!(reshape(Bp, ntuple(i -> size(Barr, Bperm[i]), length(Bperm))), Barr, Bperm)
        transpose(reshape(Bp, S * chi * chi, chi))
    end
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

        # close against the bra: contracted legs leading and in the ket's own order, which is the
        # order the bra was aligned to as well
        P3 = reshape(view(buf1, 1:n), S, chi, chi, nb)
        permutedims!(P3, reshape(G2, chi, chi, S, nb), (3, 1, 2, 4))
        mul!(view(outT, :, cols), closer, reshape(P3, slab, nb))
    end
    return outT
end

# The block buffers are the whole overhead above the two factor-sized arrays: 2·S·χ²·b elements,
# which is 2b/χ of a factor. So `b` has to scale with χ to hold a memory bound -- a constant 64 is
# 12.5% of a factor at χ=1024 but 200% of one at χ=64, which is how the peak drifts from 2.1× to
# 3.0×. χ/16 keeps the overhead at 12.5% everywhere, and the cap is where the closing gemm is
# already compute-bound in fp32 so raising it only grows the peak.
default_blocked_blocksize(chi::Integer) = clamp(chi ÷ 16, 1, 64)
default_normalize(::Algorithm"blocked") = true

#
# Which two layers the kernel closes together at a vertex, and which virtual index of `edge`
# belongs to each. `nothing` means "not a shape this kernel handles" and the caller falls back.
#
# `bra === nothing` is the case where the bra is `conj(ket)` on the ket's own legs, so the closing
# gemm needs no second aligned copy. A form supplies a genuinely different bra and pays for one.
#
# Reading this off the *network type* rather than off `virtualinds` is the point. A single-layer
# `TensorNetwork` -- a partition function -- has exactly one virtual index per edge and a
# degree-3 vertex whose element count divides χ³, so it passed every numeric guard the kernel used
# to have, and then died on `only(setdiff(...))` of its rank-1 message. There is no double layer
# there to close, so it must never reach the kernel at all.
#
_blocked_layers(::AbstractTensorNetwork, v, edge) = nothing

function _blocked_layers(tns::TensorNetworkState, v, edge)
    les = virtualinds(tns, edge)
    length(les) == 1 || return nothing
    le = only(les)
    return (; ket = tns[v], bra = nothing, le_ket = le, le_bra = prime(dag(le)), sites = nothing)
end

function _blocked_layers(form::AbstractForm, v, edge)
    lek, leb = virtualinds(ket(form), edge), bra_virtualinds(form, edge)
    (length(lek) == 1 && length(leb) == 1) || return nothing
    # An operator carrying its own bond would be a third layer on the edge.
    isempty(virtualinds(operator(form), edge)) || return nothing

    K, op = ket(form)[v], operator(form)[v]
    sites = collect(commoninds(K, op))
    pairing = _identity_operator_sites(op, sites)
    isnothing(pairing) && return nothing
    return (; ket = K, bra = bra_tensor(form, v), le_ket = only(lek), le_bra = only(leb),
        sites = (sites, pairing))
end

# The bra-side partner of each ket site index, or `nothing` if `op` is not the identity that
# pairs them. Restricting to the identity keeps the kernel's arithmetic exactly as it is: a
# general operator would have to be applied to the block's site axis, which is a different
# kernel. `inner`/`inner_mpi` always build the identity, so that is the case worth having.
#
# The comparison is on an S×S array -- a few elements beside the site tensor's S·χ³ -- so
# materialising it costs nothing next to one block of the contraction.
function _identity_operator_sites(op::ITensor, sites)
    isempty(sites) && return nothing
    partners = [prime(dag(s)) for s in sites]
    ndims(op) == 2 * length(sites) || return nothing
    all(i -> i ∈ inds(op), sites) && all(i -> i ∈ inds(op), partners) || return nothing
    S = prod(dim, sites; init = 1)
    # Explicitly on the host: `isapprox(A, I)` forms `A - I`, which reaches for the diagonal by
    # scalar index and would throw on a device array. S×S is a handful of elements, so the
    # transfer is free next to one block of the contraction.
    m = Array(reshape(array(op, sites..., partners...), S, S))
    isapprox(m, LinearAlgebra.I; atol = sqrt(eps(real(float(eltype(m)))))) || return nothing
    return partners
end

function set_default_kwargs(alg::Algorithm"blocked", bp_cache::AbstractBeliefPropagationCache)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    # `nothing` defers to χ, which is only known per edge in `updated_message`.
    b = get(alg.kwargs, :b, nothing)
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
    layers = _blocked_layers(tn, v, edge)
    isnothing(layers) && return fallback()
    T, Bt, le, le_bra = layers.ket, layers.bra, layers.le_ket, layers.le_bra

    ms = incoming_messages(bp_cache, v; ignore_edges = (reverse(edge),))
    length(ms) == 2 || return fallback()          # only the degree-3 vertex is specialised

    is = collect(inds(T))
    # One ket leg per incoming message. A message sharing none or several would make the
    # alignment below ambiguous, so it is checked rather than left to `only` to throw.
    all(m -> length(commoninds(m, T)) == 1, ms) || return fallback()
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

    # `clamp` rather than `min`: b must be at least 1 or the block loop gets a zero step.
    b = clamp(isnothing(alg.kwargs.b) ? default_blocked_blocksize(chi) : alg.kwargs.b, 1, chi)
    A = array(T)                                  # dims follow inds(T); a view when dense
    sitepos = [i for i in eachindex(is) if is[i] ∉ (la, lb, le)]
    perm = (sitepos..., findfirst(==(la), is), findfirst(==(lb), is), findfirst(==(le), is))

    # The bra is aligned to the *same* axis order as the ket -- site-for-site through the
    # operator's pairing, and leg-for-leg through each message -- because the closing gemm
    # contracts them as flat matrices and only their storage order relates them.
    Barr, Bperm = nothing, nothing
    if !isnothing(Bt)
        bis = collect(inds(Bt))
        all(m -> length(commoninds(m, Bt)) == 1, ms) || return fallback()
        blegs = [only(commoninds(m, Bt)) for m in ms][ord]
        bsites = last(layers.sites)[[findfirst(==(is[i]), first(layers.sites)) for i in sitepos]]
        bwanted = Index[bsites; blegs; le_bra]
        all(i -> i ∈ bis, bwanted) && length(bis) == length(bwanted) || return fallback()
        all(i -> dim(i) == chi, blegs) && dim(le_bra) == chi || return fallback()
        Barr = array(Bt)
        Bperm = ntuple(i -> findfirst(==(bwanted[i]), bis), length(bwanted))
    end

    nlayers = isnothing(Bt) ? 1 : 2
    s = scratch_buffer!(
        message_scratch(bp_cache), vec(A), message_scratch_length(S, chi, b, nlayers)
    )
    o1 = S * chi^3
    o2 = o1 + S * chi^2 * b
    o3 = o2 + S * chi^2 * b
    Tp, buf1, buf2 = view(s, 1:o1), view(s, (o1 + 1):o2), view(s, (o2 + 1):o3)
    outT = view(s, (o3 + 1):(o3 + chi^2))
    Bp = isnothing(Bt) ? nothing : view(s, (o3 + chi^2 + 1):(o3 + chi^2 + o1))

    # Orient each message as (ket leg, bra leg) so `transpose` inside the kernel gives the
    # transpose of that.
    mat(m, l) = array(m, l, only(setdiff(collect(inds(m)), [l])))

    out = blocked_message!(outT, buf1, buf2, Tp, A, perm,
        mat(ma, la), mat(mb, lb), S, chi, b, Bp, Barr, Bperm)

    # `out` is [bra, ket] -- label it rather than transpose. Copy so the message does not
    # alias the scratch that the next edge overwrites.
    m = itensor(copy(out), le_bra, le)
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
