using Random: Xoshiro
using KrylovKit: Arnoldi, schursolve

"""
    CTMRGCache(network, maxdim::Integer; positions, projector = :cut, pinv_rtol = 1e-3, seed = 1)

Corner transfer matrix environments for a tensor network on any planar graph: a single-layer
`TensorNetwork`, the norm network of a `TensorNetworkState`, or a form. Every inner face carries a
ring of corners and every directed edge a message, each the contraction of a region cut off from
the rest along rays to the outer face, with every cut truncated to `maxdim` by a projector pair.
The geometry and update rules are in ctmrg_rules.jl.

The embedding is read off `positions` (vertex => (x, y), by default the vertex names themselves;
see [`planar_embedding`](@ref)). The update rules are fixed at construction and certified exact at
lossless `maxdim` (see [`ctmrg_rules`](@ref)). `pinv_rtol` regularises the pseudo-inverses of
the corners some updates divide by. Run [`update`](@ref) before reading the free energy or
expectation values.

`projector` chooses how each interface is truncated:
- `:cut`: the optimal truncation of the product of the two enlarged corners on either side.
- `:cycle`: every face's interfaces together, from the dominant invariant subspaces of the ring of
  its enlarged corners, which makes the free energy stationary. It needs diagonal corners under
  Jacobi passes, so it is not available on graphs that need directional corners (e.g. Penrose
  tilings), and it is dense-only (the ring is formed as matrices). A face whose blocks carry legs
  beyond its interfaces is truncated by `:cut` instead.
"""
struct CTMRGCache{V, N}
    network::N
    embedding::PlanarEmbedding{V}
    rules::CTMRGRules{V}
    maxdim::Int
    projector::Symbol
    pinv_rtol::Float64
    rng::Xoshiro
    bonds::Dict{Tuple{V, V}, Vector{Index}}    # dart → the network's legs on that edge
    sides::Dict{Tuple{V, V}, Index}            # interface → its current truncated leg
    blocks::Dict{Any, Tensor}                  # (:message, dart) or (:corner, (p, f)) → tensor
    projectors::Dict{Tuple{V, V}, Any}         # interface → (P_A, P_B, its truncated leg)
end

network(cache::CTMRGCache) = cache.network
graph(cache::CTMRGCache) = graph(network(cache))

function Base.copy(cache::CTMRGCache)
    return CTMRGCache(
        cache.network, cache.embedding, cache.rules, cache.maxdim, cache.projector, cache.pinv_rtol, copy(cache.rng),
        copy(cache.bonds), copy(cache.sides), copy(cache.blocks), copy(cache.projectors)
    )
end

function CTMRGCache(
        network, maxdim::Integer;
        positions = default_positions(graph(network)),
        projector::Symbol = :cut,
        pinv_rtol::Real = 1.0e-3,
        seed::Integer = 1,
    )
    projector in (:cut, :cycle) || throw(ArgumentError("projector must be :cut or :cycle, got $(repr(projector))"))
    g = graph(network)
    V = vertextype(g)
    emb = planar_embedding(g; positions)
    rules = ctmrg_rules(emb; jacobi = projector === :cycle)
    bonds = Dict{Tuple{V, V}, Vector{Index}}()
    for e in edges(g)
        u, v = src(e), dst(e)
        bonds[(u, v)] = bonds[(v, u)] = _shared_legs(network, u, v)
    end
    sides = Dict{Tuple{V, V}, Index}(I => new_index(1; tags = "Link,ctm") for I in rules.interfaces)
    cache = CTMRGCache(
        network, emb, rules, Int(maxdim), projector, Float64(pinv_rtol), Xoshiro(seed),
        bonds, sides, Dict{Any, Tensor}(), Dict{Tuple{V, V}, Any}()
    )
    _initialize_blocks!(cache)
    return cache
end

function _shared_legs(network, u, v)
    legs(x) = unique(Iterators.flatten(inds(t) for t in bp_factors(network, x)))
    lv = legs(v)
    return Index[i for i in legs(u) if i in lv]
end

# Every block starts as a tensor of ones on its legs: the bonds it cuts and its interfaces.
function _initialize_blocks!(cache::CTMRGCache)
    emb = cache.embedding
    ref = first(bp_factors(network(cache), first(vertices(graph(cache)))))
    ones_on(legs) = adapt_like(ref, from_array(ones(scalartype(ref), dim.(legs)...), legs))
    for (u, v) in _darts(emb)
        legs = copy(cache.bonds[(u, v)])
        for h in (face_of_dart(emb, u, v), face_of_dart(emb, v, u))
            _isinner(h) && push!(legs, cache.sides[_interface(emb, h, u, _other_neighbor(emb, h, u, v))])
        end
        cache.blocks[(:message, (u, v))] = ones_on(legs)
    end
    for (p, f) in _corner_keys(emb)
        w = _walk(emb, f, p)
        cache.blocks[(:corner, (p, f))] = ones_on([cache.sides[_interface(emb, f, w[end], p)], cache.sides[_interface(emb, f, p, w[2])]])
    end
    return cache
end

# ── Evaluating recipes ──────────────────────────────────────────────────────────────────────

# The tensors of a recipe, read from the cache. `projectors` and `site` may be overridden: a pass's
# new projectors, or the factors of a vertex with an operator inserted. Enlarged corners are
# contracted on their own and kept in `memo`, when one is given, for reuse while the blocks they
# read are unchanged.
function _tensors(
        cache::CTMRGCache, parts;
        projectors = cache.projectors,
        site = v -> bp_factors(network(cache), v),
        memo = nothing,
    )
    ts = Tensor[]
    for part in parts
        kind = first(part)
        if kind === :site
            append!(ts, site(part[2]))
        elseif kind === :message || kind === :corner
            push!(ts, cache.blocks[part])
        elseif kind === :enlarged
            enlarged() = _evaluate(cache, _enlarged(cache.embedding, part[2]...); projectors, site)
            push!(ts, isnothing(memo) ? enlarged() : get!(enlarged, memo, part[2]))
        elseif kind === :inverse
            chain = _contract([cache.blocks[(:corner, k)] for k in part[2]])
            push!(ts, _pseudo_inverse(chain, cache.sides[part[3][1]], cache.sides[part[3][2]], cache.pinv_rtol))
        else
            P = projectors[part[2]]
            push!(ts, part[3] === :A ? P[1] : P[2])
        end
    end
    return ts
end

_evaluate(cache::CTMRGCache, parts; kwargs...) = _contract(_tensors(cache, parts; kwargs...))

# The same lists recur every pass, so a contraction order is searched once per structure (each leg
# labelled by first appearance, with its dimension) and reused. Exhaustive search is exponential
# in the number of tensors, so longer lists fall back to the greedy optimiser.
const _SEQUENCES = Dict{Any, Any}()
function _contract(ts::Vector)
    length(ts) == 1 && return only(ts)
    length(ts) == 2 && return ts[1] * ts[2]
    seen = Dict{Any, Int}()
    structure = Tuple(Tuple((get!(seen, i, length(seen) + 1), dim(i)) for i in inds(t)) for t in ts)
    sequence = get!(_SEQUENCES, structure) do
        length(ts) <= 12 ? contraction_sequence(ts; alg = "optimal") :
            contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod())
    end
    return contract(ts; sequence)
end

# The bilinear pseudo-inverse of a two-leg connector M on (l1, l2): the tensor N on the same legs
# with N[l1, l2] = pinv(M)[l2, l1], so that M N M = M. Singular values below `rtol · s₁` are
# dropped, which regularises the inverse.
function _pseudo_inverse(M, l1, l2, rtol)
    U, S, V = svd(M, [l1])
    cut = rtol * maximum(abs, array(S))
    return conj(U * map_diag(s -> abs(s) > cut ? inv(s) : zero(s), S) * V)
end

# ── Projectors ──────────────────────────────────────────────────────────────────────────────

# A block as one column of a matrix on the interface `ins`: its other legs fused, unit norm.
function _column(B, ins)
    B = B / norm(B)
    rest = uniqueinds(B, ins)
    if isempty(rest)
        c = new_index(1; tags = "Link,col")
        return B * adapt_like(B, onehot(scalartype(B), c => 1)), c
    end
    C = combiner(rest; tags = "Link,col")
    return B * C, combinedind(C)
end

# Blocks side by side: the direct sum of their columns.
function _stack(blocks, ins)
    acc, c = _column(first(blocks), ins)
    for B in blocks[2:end]
        col, cB = _column(B, ins)
        acc = directsum(acc => c, col => cB; tags = "Link,col")
        c = only(uniqueinds(acc, ins))
    end
    return acc
end

# A side of an interface reduced to its triangular factor R (B = Q R, R on (b, ins…)), when that
# shrinks it: the factor has the same overlap with the other side, and is at most dim(ins) wide.
function _triangular(B, ins)
    rest = uniqueinds(B, ins)
    if isempty(rest)
        b = new_index(1; tags = "Link,qr")
        return B * adapt_like(B, onehot(scalartype(B), b => 1))
    end
    !isgraded(B) && dim(rest) <= dim(ins) && return B
    _, R = qr(B, rest)
    return R
end

# The projector pair at an interface, from the blocks it truncates on either side. With A and B
# the left and right blocks stacked as columns on `ins`, Aᵀ B = U S V† truncated to `maxdim` (and
# to singular values above 1e-12 s₁) gives
#   P_A = B V S^{-1/2},   P_B = S^{-1/2} U† Aᵀ,   so   Aᵀ P_A P_B B = Aᵀ B   at full rank,
# lossless for every pairing of a left and a right block. The pairing is bilinear, not
# sesquilinear: the network contracts the two sides plainly. Each side is first reduced to its
# triangular factor, so the SVD is at most dim(ins) × dim(ins). U and V are maps (rule 5 of the
# seam), so their adjoints are `dag(·, cod)`.
function _projector_pair(left, right, ins, maxdim)
    covers(B) = all(i -> i in inds(B), ins)
    side(blocks) = _triangular(length(blocks) == 1 ? only(blocks) : _stack(filter(covers, blocks), ins), ins)
    A, B = side(left), side(right)
    U, S, V = truncated_svd(A * B, uniqueinds(A, ins); trunc = truncation_strategy(; maxdim, rtol = 1.0e-12))
    u, v = inds(S)
    whiten = dag(map_diag(x -> inv(sqrt(x)), S), [u])
    PA = (B * dag(V, [v])) * whiten
    PB = (dag(U, [u]) * A) * whiten
    w = only(uniqueinds(PA, ins))
    return PA, replaceind(PB, only(uniqueinds(PB, ins)), dag(w)), w
end

# A random isometry onto min(maxdim, dim(ins)) directions, for the first pass: at lossless χ it
# loses nothing, so the first pass is already exact.
function _random_projector(cache::CTMRGCache, ins)
    k = new_index(min(cache.maxdim, dim(ins)))
    ref = first(cache.blocks)[2]
    Q, _ = qr(adapt_like(ref, random_tensor(cache.rng, real(scalartype(ref)), [ins; k])), ins)
    return Q, dag(Q, ins), only(uniqueinds(Q, ins))
end

function _set_projectors!(cache::CTMRGCache, projectors)
    merge!(cache.projectors, projectors)
    for (I, P) in projectors
        cache.sides[I] = P[3]
    end
    return cache
end

# The legs of interface I: its bonds, and the leg of the interface its ray continues into.
_interface_legs(cache::CTMRGCache, I, side_of) =
    (J = cache.rules.next[I]; isnothing(J) ? cache.bonds[I] : Index[cache.bonds[I]; side_of(J)])

function _random_projectors!(cache::CTMRGCache)
    new = Dict{Any, Any}()
    for I in cache.rules.interfaces
        new[I] = _random_projector(cache, _interface_legs(cache, I, J -> new[J][3]))
    end
    return _set_projectors!(cache, new)
end

# One coherent projector set for a sequential pass. Interfaces are processed boundary first; the
# blocks on either side are built from the current blocks, whose legs are in the previous pass's
# coordinates, and their tail on the next interface J is first carried into J's new coordinates by
#   p side  M_I = P_B(old) · M_J · P_A(new),      q side  M_I = P_A(old) · M_J · P_B(new).
# The projector is built from the two enlarged corners, as on the grid. Where a directional corner
# is projected too, it has no enlarged corner behind it, so every block projected here — the
# directional corners and the side messages as well — is stacked in, to keep each one lossless.
function _carried_projectors!(cache::CTMRGCache)
    emb, rules = cache.embedding, cache.rules
    memo = Dict{Any, Tensor}()
    corner(p, f) = only(_tensors(cache, [(:enlarged, (p, f))]; memo))
    new = Dict{Any, Any}()
    carry = Dict{Any, Tensor}()
    for I in rules.interfaces
        p, q = I
        f = face_of_dart(emb, p, q)
        J = rules.next[I]
        Ql, Qr = _connectors(emb, rules.exits, p, q)
        moved(t, side) = isnothing(J) ? t : t * carry[(J, side)]
        on(Q, side, parts...) = moved(_contract([_tensors(cache, Q); parts...]), side)
        left = Tensor[on(Ql, :A, corner(p, f))]
        right = Tensor[on(Qr, :B, corner(q, f))]
        directional = [(c, first(_directional_core(emb, c, f, rules.modes[(c, f)]))) for c in (p, q)
            if rules.modes[(c, f)] !== :diag && last(_directional_core(emb, c, f, rules.modes[(c, f)])) == (c == p ? q : p)]
        if !isempty(directional)
            w = face_vertices(emb, f)
            k = findfirst(==(p), w)
            push!(left, on(Ql, :A, _tensors(cache, first(_message_core(emb, p, w[mod1(k - 1, length(w))])))...))
            push!(right, on(Qr, :B, _tensors(cache, first(_message_core(emb, q, w[mod1(k + 2, length(w))])))...))
            for (c, parts) in directional
                c == p ? push!(left, on(Ql, :A, _tensors(cache, parts)...)) : push!(right, on(Qr, :B, _tensors(cache, parts)...))
            end
        end
        new[I] = _projector_pair(left, right, _interface_legs(cache, I, J -> new[J][3]), cache.maxdim)
        old = cache.projectors[I]
        carry[(I, :A)] = _contract(isnothing(J) ? [old[2], new[I][1]] : [old[2], carry[(J, :A)], new[I][1]])
        carry[(I, :B)] = _contract(isnothing(J) ? [old[1], new[I][2]] : [old[1], carry[(J, :B)], new[I][2]])
    end
    return _set_projectors!(cache, new)
end

# ── Cycle projectors ────────────────────────────────────────────────────────────────────────

# The interfaces I₁, …, Iₙ of face f (I_k between its k-th and (k+1)-th vertex) cut the ring of its
# enlarged corners: Z_f = tr(K₁ ⋯ Kₙ), with K_k the enlarged corner at the k-th vertex and the
# connectors of the interfaces on either side, a map from ins(I_{k−1}) to ins(I_k). Truncating I_k
# to the dominant invariant subspaces of the ring's transfer matrix M_k = K_{k+1} ⋯ Kₙ K₁ ⋯ K_k
# keeps its maxdim largest eigenvalues, the best rank-maxdim approximation of Z_f = tr(M_k), and
# makes the free energy stationary: the grid engine's `:cycle` projector, on a face of any size.
# All M_k share their nonzero eigenvalues, so the choice is made once per face (`_ring_bases`)
# and each interface carries the spectral projector V (Wᵀ V)⁻¹ Wᵀ onto them: the right subspace V
# on the K_k side, the left one W on the K_{k+1} side, whitened as in `_projector_pair`.
# A face whose spectrum above the noise fits in maxdim is left to the `:cut` projector, which is
# lossless there: the cycle would choose nothing, and its projector would drop the ring's kernel,
# which other faces' blocks see. So at large maxdim `:cycle` is `:cut`. `nothing` also when a block
# carries legs beyond its two interfaces.
function _cycle_projectors(cache::CTMRGCache, f, ins_of, memo)
    emb, exits = cache.embedding, cache.rules.exits
    w = faces(emb)[f]
    n = length(w)
    I = [(w[k], w[mod1(k + 1, n)]) for k in 1:n]
    ins = [ins_of(I[k]) for k in 1:n]
    before(k) = ins[mod1(k - 1, n)]
    K = map(1:n) do k
        _, Qr = _connectors(emb, exits, I[mod1(k - 1, n)]...)
        Ql, _ = _connectors(emb, exits, I[k]...)
        return _evaluate(cache, [Qr; (:enlarged, (w[k], f)); Ql]; memo)
    end
    all(k -> issetequal(inds(K[k]), [before(k); ins[k]]), 1:n) || return nothing
    mats = [reshape(array(K[k], before(k)..., ins[k]...), dim(before(k)), dim(ins[k])) for k in 1:n]
    # M is a product of n blocks, so its eigenvalues are known only to ~ N ε ∏ ‖K‖
    noise = maximum(size.(mats, 2)) * eps(real(eltype(first(mats)))) * prod(norm, mats)
    bases = _ring_bases(mats, min(cache.maxdim, minimum(size.(mats, 2))), noise)
    isnothing(bases) && return nothing
    pairs = Dict{Any, Any}()
    for k in 1:n
        basis(X, tags) = (c = new_index(size(X, 2); tags); from_array(reshape(X, dim.(ins[k])..., size(X, 2)), [ins[k]; c]))
        V, W = basis(bases[1][k], "Link,right"), basis(bases[2][k], "Link,left")
        pairs[I[k]] = _projector_pair([adapt_like(K[k], W)], [adapt_like(K[k], V)], ins[k], cache.maxdim)
    end
    return pairs
end

# Right and left bases of the kept invariant subspace at every interface of a ring with blocks
# `mats` (mats[k] a matrix from ins(I_{k−1}) to ins(I_k)), or `nothing` when the face is left to the
# cut (nothing above the noise left out, or no well-defined choice). They are found at Iₙ and
# carried around the ring: K_k maps the right subspace at I_k onto the one at I_{k−1}, and K_kᵀ
# the left one at I_{k−1} onto the one at I_k. At Iₙ the ring M = K₁ ⋯ Kₙ is matrix-free when
# large: a Krylov–Schur solve for M and one for Mᵀ, each applying the n blocks in turn, never forming
# M; their eigenvalues must agree, or the dense Schur decomposition decides instead.
function _ring_bases(mats, r, noise)
    n = length(mats)
    N = size(mats[n], 2)
    ring(x) = foldr(*, mats; init = x)
    ringᵀ(y) = foldl((y, K) -> transpose(K) * y, mats; init = y)
    found = _krylov_spectral_bases(ring, ringᵀ, N, eltype(first(mats)), r, noise)
    if isnothing(found)
        found = _spectral_bases(ring(Matrix{eltype(first(mats))}(LinearAlgebra.I, N, N)), r; noise)
    end
    (isnothing(found) || !found[3]) && return nothing
    orthonormal(X) = Matrix(qr(X).Q)[:, 1:size(X, 2)]
    Vs = Vector{Matrix}(undef, n)
    Ws = Vector{Matrix}(undef, n)
    Vs[n], Ws[n] = found[1], found[2]
    for k in n:-1:2
        Vs[k - 1] = orthonormal(mats[k] * Vs[k])
    end
    for k in 1:(n - 1)
        Ws[k] = orthonormal(transpose(mats[k]) * Ws[mod1(k - 1, n)])
    end
    return Vs, Ws
end

# Which eigenvalues `λ` to keep: the (at most) r of largest modulus, never one within the rounding
# `noise`, never splitting a conjugate pair (for real matrices) or a cluster (a kept eigenvalue
# within `gap · |λ₁|` of one above the noise left out, where the subspace is ill-defined). Returns
# the kept mask and whether an eigenvalue above the noise was left out, or `nothing` if none is kept.
# Keeping noise eigenvalues makes the rank flip from pass to pass.
function _kept_eigenvalues(λ, r, noise, isreal_matrix; gap = 1.0e-8)
    order = sortperm(abs.(λ); rev = true)
    scale = abs(λ[first(order)])
    nonzero = filter(i -> abs(λ[i]) > noise, order)
    groups = Vector{Int}[]
    taken = falses(length(λ))
    for i in nonzero
        taken[i] && continue
        group = [i]
        if isreal_matrix && !isreal(λ[i])
            partner = findfirst(j -> j != i && !taken[j] && λ[j] ≈ conj(λ[i]), eachindex(λ))
            isnothing(partner) && break             # its partner lies beyond what was computed
            push!(group, partner)
        end
        taken[group] .= true
        push!(groups, group)
    end
    keep = falses(length(λ))
    for group in groups
        count(keep) + length(group) > r && break
        keep[group] .= true
    end
    separated(keep) = all(abs(λ[i] - λ[j]) > gap * scale for i in findall(keep), j in nonzero if !keep[j])
    while any(keep) && !separated(keep)
        keep[groups[findlast(g -> all(keep[g]), groups)]] .= false
    end
    any(keep) || return nothing
    return keep, count(keep) < length(nonzero)
end

# Dense: right and left bases V, W (Wᵀ V = 1) of the kept invariant subspace of M, and whether
# anything above the noise was left out, from ONE Schur decomposition, so both belong to the same
# eigenvalues: with the kept ones ordered first, M Z = Z [T₁₁ T₁₂; 0 T₂₂], V = Z[:, 1:k], and the
# spectral projector Z [1 X; 0 0] Z⁻¹ commutes with T when T₁₁ X − X T₂₂ = T₁₂, so Wᵀ = [1 X] Z⁻¹.
function _spectral_bases(M::AbstractMatrix, r::Integer; noise::Real = size(M, 1) * eps(real(eltype(M))) * norm(M))
    F = schur(M)
    kept = _kept_eigenvalues(F.values, r, noise, eltype(M) <: Real)
    isnothing(kept) && return nothing
    keep, truncated = kept
    ordschur!(F, keep)
    k = count(keep)
    T, Z = F.T, F.Z
    k == size(M, 1) && return Z, transpose(inv(Z)), truncated
    X = sylvester(T[1:k, 1:k], -T[(k + 1):end, (k + 1):end], -T[1:k, (k + 1):end])
    Wt = [Matrix{eltype(T)}(LinearAlgebra.I, k, k) X] / Z
    return Z[:, 1:k], transpose(Wt), truncated
end

# Matrix-free: the same from Krylov–Schur solves of the ring and of its transpose, asking for the
# r + 2 eigenvalues of largest modulus to see past the cut. `nothing` when the ring is too small
# for a Krylov space to pay, a solve does not converge, the kept set is not a prefix of what was
# computed, or the two solves disagree on the kept eigenvalues.
function _krylov_spectral_bases(ring, ringᵀ, N, T, r, noise; gap = 1.0e-8)
    howmany = r + 2
    krylovdim = 2howmany + 16
    krylovdim <= N ÷ 2 || return nothing
    alg = Arnoldi(; krylovdim, tol = noise, maxiter = 100, verbosity = 0)
    rng = Xoshiro(N)
    _, vR, λR, infoR = schursolve(ring, randn(rng, T, N), howmany, :LM, alg)
    _, vL, λL, infoL = schursolve(ringᵀ, randn(rng, T, N), howmany, :LM, alg)
    # Fewer converged than asked is fine when the rest are within the noise: the ring has no more
    # eigenvalues above it (its Krylov space closed), so the spectrum that matters is all there.
    for (λ, info) in ((λR, infoR), (λL, infoL))
        all(μ -> abs(μ) <= noise, λ[(info.converged + 1):end]) || return nothing
    end
    λR, vR = λR[1:infoR.converged], vR[1:infoR.converged]
    λL, vL = λL[1:infoL.converged], vL[1:infoL.converged]
    isempty(λR) && return nothing
    kept = _kept_eigenvalues(λR, r, noise, T <: Real; gap)
    isnothing(kept) && return nothing
    keep, truncated = kept
    k = count(keep)
    all(keep[1:k]) || return nothing
    # the left solve must find the same kept eigenvalues
    left = collect(λL[1:min(k, length(λL))])
    for λ in λR[1:k]
        j = findfirst(μ -> abs(μ - λ) <= gap * abs(λR[1]), left)
        isnothing(j) && return nothing
        deleteat!(left, j)
    end
    return reduce(hcat, vR[1:k]), reduce(hcat, vL[1:k]), truncated
end

# ── Passes ──────────────────────────────────────────────────────────────────────────────────

function _recompute(cache::CTMRGCache, block; kwargs...)
    rules = cache.rules
    t = _evaluate(cache, _block_recipe(cache.embedding, rules.exits, rules.modes, block); kwargs...)
    return t / norm(t)
end

# A sequential pass: one coherent projector set, then every block recomputed once in dependency
# order, each from blocks already recomputed in this pass. No leg is carried over from an earlier
# pass, so the network stays consistent whatever the projectors.
function _sequential_pass!(cache::CTMRGCache)
    isempty(cache.projectors) ? _random_projectors!(cache) : _carried_projectors!(cache)
    for block in cache.rules.order
        cache.blocks[block] = _recompute(cache, block)
    end
    return cache
end

# A Jacobi pass, for rules with no acyclic order (diagonal corners only) and for cycle projectors:
# every projector from the current blocks, then every block recomputed from the current blocks
# and the new projectors.
function _jacobi_pass!(cache::CTMRGCache)
    emb, rules = cache.embedding, cache.rules
    memo = Dict{Any, Tensor}()           # the blocks stay unchanged until the end of the pass
    ins(I) = _interface_legs(cache, I, J -> cache.sides[J])
    new = Dict{Any, Any}()
    if cache.projector === :cycle
        for f in eachindex(faces(emb))
            pairs = _cycle_projectors(cache, f, ins, memo)
            isnothing(pairs) || merge!(new, pairs)
        end
    end
    for I in rules.interfaces
        haskey(new, I) && continue
        p, q = I
        f = face_of_dart(emb, p, q)
        Ql, Qr = _connectors(emb, rules.exits, p, q)
        Bw = _evaluate(cache, [Ql; (:enlarged, (p, f))]; memo)
        Be = _evaluate(cache, [Qr; (:enlarged, (q, f))]; memo)
        new[I] = _projector_pair([Bw], [Be], ins(I), cache.maxdim)
    end
    blocks = Dict(b => _recompute(cache, b; projectors = new, memo) for b in keys(cache.blocks))
    merge!(cache.blocks, blocks)
    return _set_projectors!(cache, new)
end

"""
    update(cache::CTMRGCache; convergence = :messages, tolerance, maxiter = 30, verbose = false)

Run passes over the cache until it converges, or `maxiter` passes. What converges depends on
what the cache is for:
- `convergence = :messages` (for observables): every vertex's environment, the contraction of
  the messages and corners around it, changes by at most `tolerance` (default 1e-12) between
  passes, measured as 1 − |⟨E, E′⟩|² / (‖E‖² ‖E′‖²). Environments live on the vertex's own legs,
  so passes compare directly, whatever gauge the truncated legs are in.
- `convergence = :freenergy` (for the free energy): `freenergy` changes by at most `tolerance`
  (default 1e-10) relative to max(1, |F|). It settles before the messages do, so it does not
  certify observables.

At lossless `maxdim` two passes suffice.
"""
function update(
        cache::CTMRGCache;
        convergence::Symbol = :messages,
        tolerance::Real = convergence === :messages ? 1.0e-12 : 1.0e-10,
        maxiter::Integer = 30,
        verbose::Bool = false,
    )
    convergence in (:messages, :freenergy) || throw(ArgumentError("convergence must be :messages or :freenergy, got $(repr(convergence))"))
    cache = copy(cache)
    pass! = isnothing(cache.rules.order) ? _jacobi_pass! : _sequential_pass!
    signal(cache) = convergence === :messages ? _environments(cache) : freenergy(cache)
    change(a, b) = convergence === :messages ? maximum(message_diff.(a, b)) : abs(a - b) / max(1, abs(a))
    previous = nothing
    for i in 1:maxiter
        pass!(cache)
        current = signal(cache)
        if !isnothing(previous) && change(current, previous) <= tolerance
            verbose && println("CTMRG converged after $i passes.")
            return cache
        end
        previous = current
    end
    msg = "CTMRG did not converge to tolerance $tolerance in its $convergence after $maxiter passes."
    verbose ? println(msg) : @warn(msg)
    return cache
end

# The environment of every vertex: the messages into it and the corners around it, contracted.
function _environments(cache::CTMRGCache)
    emb = cache.embedding
    return map(collect(vertices(graph(emb)))) do v
        parts = Any[]
        for (u, f) in vertex_polygon(emb, v)
            push!(parts, (:message, (u, v)))
            append!(parts, _ring(emb, v, f))
        end
        return _evaluate(cache, parts)
    end
end

# ── Free energy and observables ─────────────────────────────────────────────────────────────

_log(z) = real(z) < 0 ? log(complex(z)) : log(z)

# Every closed region's Kikuchi counting number and log-contraction, in a fixed order: each face,
# vertex and edge, with that part of the network explicit.
function _region_logs(cache::CTMRGCache)
    emb = cache.embedding
    g = graph(emb)
    memo = Dict{Any, Tensor}()
    lnZ(parts) = _log(scalar(_evaluate(cache, parts; memo)))
    regions = Pair{Int, Number}[1 => lnZ(_face_region(emb, f)) for f in eachindex(faces(emb))]
    append!(regions, [_counting_number(emb, v) => lnZ(_vertex_region(emb, v)) for v in vertices(g)])
    append!(regions, [_counting_number(emb, src(e), dst(e)) => lnZ(_edge_region(emb, src(e), dst(e))) for e in edges(g)])
    return regions
end

"""
    freenergy(cache::CTMRGCache)

The Kikuchi estimate of the network's log-contraction, `Σ_v c_v ln Z_v + Σ_e c_e ln Z_e + Σ_f ln Z_f`,
with `Z_x` the whole network with `x` explicit, built from the cache's blocks.
"""
freenergy(cache::CTMRGCache) = sum(c * l for (c, l) in _region_logs(cache) if !iszero(c); init = 0.0)

partitionfunction(cache::CTMRGCache) = exp(freenergy(cache))

# The closed region around one vertex, or around the two ends of an edge.
function _observable_region(cache::CTMRGCache, vs)
    length(vs) == 1 && return _vertex_region(cache.embedding, only(vs))
    length(vs) == 2 && has_edge(graph(cache), vs...) && return _edge_region(cache.embedding, vs...)
    return error("alg = \"ctmrg\" supports observables on one vertex or on the two ends of an edge; got $vs.")
end
