using Random: Xoshiro
using KrylovKit: Arnoldi, schursolve

"""
    CTMRGCache(network, maxdim::Integer; positions, projector = :svd, pinv_rtol = 1e-3, seed = 1)

Corner transfer matrix (CTMRG) environments for a tensor network on any planar graph. Every inner
face carries a ring of corners and every directed edge a message, each the contraction of a region
of the network, with the cuts between them truncated to `maxdim`. Call `update` on it before
computing free energies or expectation values.

# Arguments
- `network`: A `TensorNetwork`, a `TensorNetworkState` (its norm network) or a form, on a planar graph.
- `maxdim::Integer`: The bond dimension each cut is truncated to.

# Keyword Arguments
- `positions`: Vertex positions `v => (x, y)` defining the planar embedding (default: the vertex
  names, when they are coordinates). See `planar_embedding`.
- `projector::Symbol = :svd`: How the cuts are truncated. `:svd` keeps the best approximation (a
  truncated SVD) of the two enlarged corners on either side of a cut. `:eig` (experimental) truncates
  all the cuts of a face together, keeping the dominant eigenvalues of the ring of its enlarged
  corners; it needs diagonal corners, so it is not available on every planar graph (e.g. kagome).
- `pinv_rtol::Real = 1e-3`: Relative cutoff regularising the pseudo-inverses some updates use.
- `seed::Integer = 1`: Seed for the random projectors of the first pass.

# Returns
- A `CTMRGCache`, whose update rules are fixed and certified exact at lossless `maxdim` (see `ctmrg_rules`).
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
        projector::Symbol = :svd,
        pinv_rtol::Real = 1.0e-3,
        seed::Integer = 1,
    )
    projector in (:svd, :eig) || throw(ArgumentError("projector must be :svd or :eig, got $(repr(projector))"))
    g = graph(network)
    V = vertextype(g)
    emb = planar_embedding(g; positions)
    rules = ctmrg_rules(emb; jacobi = projector === :eig)
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

#Every block starts as a tensor of ones on its legs: the bonds it cuts and its interfaces.
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


#The tensors of a recipe. `projectors` and `site` can be overridden (a pass's new projectors, a site
#with an operator inserted); enlarged corners are kept in `memo`, if given, while the blocks don't change.
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

#The same lists recur every pass, so contraction orders are cached by the lists' structure.
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

#The pseudo-inverse N of a two-leg connector M, so that M N M = M (bilinearly). Singular values
#below `rtol` (relative) are dropped.
function _pseudo_inverse(M, l1, l2, rtol)
    U, S, V = svd(M, [l1])
    cut = rtol * maximum(abs, array(S))
    return conj(U * map_diag(s -> abs(s) > cut ? inv(s) : zero(s), S) * V)
end


#A block as a column on the interface `ins`: its other legs fused, normalised.
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

#Blocks side by side, as the direct sum of their columns.
function _stack(blocks, ins)
    acc, c = _column(first(blocks), ins)
    for B in blocks[2:end]
        col, cB = _column(B, ins)
        acc = directsum(acc => c, col => cB; tags = "Link,col")
        c = only(uniqueinds(acc, ins))
    end
    return acc
end

#One side of an interface reduced to its triangular factor (B = QR) when that makes it smaller.
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

#The projector pair at an interface, from the blocks on its left (A) and right (B): with the
#truncated SVD AᵀB = U S V†, P_A = B V S^(-1/2) and P_B = S^(-1/2) U† Aᵀ, so that Aᵀ P_A P_B B = Aᵀ B
#at full rank. The pairing is bilinear because the network contracts the two sides plainly.
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

#A random isometry for the first pass: lossless at lossless maxdim, so that pass is already exact.
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

#The legs of interface I: its bonds, and the leg of the interface its ray continues into.
function _interface_legs(cache::CTMRGCache, I, side_of)
    J = cache.rules.next[I]
    return isnothing(J) ? cache.bonds[I] : Index[cache.bonds[I]; side_of(J)]
end

function _random_projectors!(cache::CTMRGCache)
    new = Dict{Any, Any}()
    for I in cache.rules.interfaces
        new[I] = _random_projector(cache, _interface_legs(cache, I, J -> new[J][3]))
    end
    return _set_projectors!(cache, new)
end

#The projectors for a sequential pass, boundary first. The blocks either side of an interface are in
#the previous pass's coordinates, so their leg on the next interface J is first carried into J's new
#ones (by P_B(old) M_J P_A(new) on the p side, P_A(old) M_J P_B(new) on the q side). Where a
#directional corner is projected, which has no enlarged corner behind it, every block projected
#there is stacked in so that each stays lossless.
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
        directional = []
        for c in (p, q)
            rules.modes[(c, f)] === :diag && continue
            parts, a = _directional_core(emb, c, f, rules.modes[(c, f)])
            a == (c == p ? q : p) && push!(directional, (c, parts))
        end
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


#Eigenvalue projectors for the interfaces of face f. They cut the ring of its enlarged corners,
#Z_f = tr(K_1 ⋯ K_n), and each keeps the dominant invariant subspaces of the ring's transfer matrix
#at that interface (the eigenvalues all the transfer matrices share). A face whose spectrum fits in
#maxdim is left to the SVD projector, which is lossless there: these would drop the ring's
#kernel, which other faces see. `nothing` also if a block has legs beyond its two interfaces.
function _eig_projectors(cache::CTMRGCache, f, ins_of, memo)
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
    #the ring's eigenproblem is small and runs on the host
    mats = [Array(reshape(array(K[k], before(k)..., ins[k]...), dim(before(k)), dim(ins[k]))) for k in 1:n]
    #The ring is a product of n blocks, so its eigenvalues are only known to about N ε ∏‖K‖
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

#Right and left bases of the kept invariant subspace at every interface of a ring, or `nothing` if
#the face is left to the SVD projector. They are found at the last interface (matrix-free when the ring is
#large, else dense) and carried around the ring by the blocks.
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

#Which eigenvalues to keep: at most r of the largest, none within the `noise`, and never splitting a
#conjugate pair or a cluster (where the subspace is ill-defined). Returns the kept mask and whether
#anything above the noise was left out, or `nothing` if nothing is kept.
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
            isnothing(partner) && break   #its partner lies beyond what was computed
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

#Dense right and left bases (Wᵀ V = 1) of the kept invariant subspace of M, both from one Schur
#decomposition so that they belong to the same eigenvalues; the left one from a Sylvester solve.
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

#The same, matrix-free, from Krylov-Schur solves for the ring and its transpose; `nothing` (so the
#dense route decides) if the ring is too small, a solve fails, or the two disagree.
function _krylov_spectral_bases(ring, ringᵀ, N, T, r, noise; gap = 1.0e-8)
    howmany = r + 2
    krylovdim = 2howmany + 16
    krylovdim <= N ÷ 2 || return nothing
    alg = Arnoldi(; krylovdim, tol = noise, maxiter = 100, verbosity = 0)
    rng = Xoshiro(N)
    _, vR, λR, infoR = schursolve(ring, randn(rng, T, N), howmany, :LM, alg)
    _, vL, λL, infoL = schursolve(ringᵀ, randn(rng, T, N), howmany, :LM, alg)
    #Fewer converged than asked is fine if the rest are noise: there is nothing more above it
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
    #the left solve must find the same kept eigenvalues
    left = collect(λL[1:min(k, length(λL))])
    for λ in λR[1:k]
        j = findfirst(μ -> abs(μ - λ) <= gap * abs(λR[1]), left)
        isnothing(j) && return nothing
        deleteat!(left, j)
    end
    return reduce(hcat, vR[1:k]), reduce(hcat, vL[1:k]), truncated
end


function _recompute(cache::CTMRGCache, block; kwargs...)
    rules = cache.rules
    t = _evaluate(cache, _block_recipe(cache.embedding, rules.exits, rules.modes, block); kwargs...)
    return t / norm(t)
end

#A sequential pass: new projectors, then every block once, in dependency order.
function _sequential_pass!(cache::CTMRGCache)
    isempty(cache.projectors) ? _random_projectors!(cache) : _carried_projectors!(cache)
    for block in cache.rules.order
        cache.blocks[block] = _recompute(cache, block)
    end
    return cache
end

#A Jacobi pass: new projectors, then every block, all from the blocks of the previous pass.
function _jacobi_pass!(cache::CTMRGCache)
    emb, rules = cache.embedding, cache.rules
    memo = Dict{Any, Tensor}()
    ins(I) = _interface_legs(cache, I, J -> cache.sides[J])
    new = Dict{Any, Any}()
    if cache.projector === :eig
        for f in eachindex(faces(emb))
            pairs = _eig_projectors(cache, f, ins, memo)
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

Update the corners and messages of a `CTMRGCache` until they converge.

# Arguments
- `cache::CTMRGCache`: The cache to update.

# Keyword Arguments
- `convergence::Symbol = :messages`: What has to converge. `:messages` (for expectation values):
  the environment of every vertex, by the same distance BP uses for its messages. `:freenergy`
  (enough for the free energy): `freenergy(cache)`, which settles first.
- `tolerance`: Convergence threshold (default `1e-12` for `:messages`, `1e-10` relative for `:freenergy`).
- `maxiter::Integer = 30`: Maximum number of passes.
- `verbose::Bool = false`: Print the outcome.

# Returns
- An updated copy of the cache.
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

#The environment of every vertex: the messages into it and the corners around it, contracted.
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


_log(z) = real(z) < 0 ? log(complex(z)) : log(z)

"""
    freenergy(cache::CTMRGCache)

The log of the network's contraction, estimated from the cache by the Kikuchi expansion over the
regions with one vertex, edge or face explicit.
"""
function freenergy(cache::CTMRGCache)
    emb = cache.embedding
    g = graph(emb)
    memo = Dict{Any, Tensor}()
    lnZ(parts) = _log(scalar(_evaluate(cache, parts; memo)))
    F = sum(lnZ(_face_region(emb, f)) for f in eachindex(faces(emb)); init = 0.0)
    for v in vertices(g)
        c = _counting_number(emb, v)
        iszero(c) || (F += c * lnZ(_vertex_region(emb, v)))
    end
    for e in edges(g)
        c = _counting_number(emb, src(e), dst(e))
        iszero(c) || (F += c * lnZ(_edge_region(emb, src(e), dst(e))))
    end
    return F
end

partitionfunction(cache::CTMRGCache) = exp(freenergy(cache))

#The closed region around one vertex, or around the two ends of an edge.
function _observable_region(cache::CTMRGCache, vs)
    length(vs) == 1 && return _vertex_region(cache.embedding, only(vs))
    length(vs) == 2 && has_edge(graph(cache), vs...) && return _edge_region(cache.embedding, vs...)
    return error("alg = \"ctmrg\" supports observables on one vertex or on the two ends of an edge; got $vs.")
end
