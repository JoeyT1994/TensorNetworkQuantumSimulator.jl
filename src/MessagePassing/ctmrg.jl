using Random: Xoshiro

"""
    CTMRGCache(network, env_maxdim::Integer; positions, projector = :svd)
    CTMRGCache(tn::TensorNetwork, messages, env_maxdim::Integer; positions, projector = :svd)

Corner transfer matrix (CTMRG) environments for a tensor network on any planar graph. Every directed
edge carries a message, as in belief propagation, and every inner face a ring of corners, each the
contraction of a region of the network, with the cuts between them truncated to `env_maxdim`. The
environment of a vertex or edge (`incoming_messages`) is its messages together with the corners
around it. Call `update` on it before computing free energies or expectation values.

# Arguments
- `network`: A `TensorNetworkState` (its norm network is contracted), a `TensorNetwork` without
  site indices (contracted as it is) or a form, on a planar graph; dense or graded (symmetric or
  fermionic).
- `env_maxdim::Integer`: The bond dimension each cut is truncated to.

# Keyword Arguments
- `positions`: Vertex positions `v => (x, y)` defining the planar embedding (default: the vertex
  names, when they are coordinates). See `planar_embedding`.
- `projector::Symbol = :svd`: How the cuts are truncated. `:svd` keeps the best approximation (a
  truncated SVD) of the two enlarged corners on either side of a cut. `:eig` (experimental) truncates
  all the cuts of a face together, keeping the dominant eigenvalues of the ring of its enlarged
  corners; it needs diagonal corners, so it is not available on every planar graph (e.g. kagome).
  On graded (symmetric or fermionic) networks every face falls back to the SVD projector.

# Returns
- A `CTMRGCache`, whose update rules are fixed and certified exact at lossless `env_maxdim` (see `ctmrg_rules`).
"""
struct CTMRGCache{V, F} <: AbstractBeliefPropagationCache{V}
    factors::F
    embedding::PlanarEmbedding{V}
    rules::CTMRGRules{V}
    env_maxdim::Int
    projector::Symbol
    bonds::Dict{Tuple{V, V}, Vector{Index}}    # dart → the network's legs on that edge
    sides::Dict{Tuple{V, V}, Index}            # interface → its current truncated leg
    messages::Dictionary{NamedEdge, Tensor}    # u => v → the message from u into v
    corners::Dict{Tuple{V, Int}, Tensor}       # (p, f) → the corner at p in face f
    projectors::Dict{Tuple{V, V}, Tuple{Tensor, Tensor, Index}}  # interface → (P_A, P_B, its truncated leg)
    sequences::Dict{Any, Any}                  # contraction orders, by the structure of the tensor list
end

graph(cache::CTMRGCache) = graph(network(cache))
messages(cache::CTMRGCache) = cache.messages
contraction_sequences(cache::CTMRGCache) = nothing

function Base.copy(cache::CTMRGCache)
    return CTMRGCache(
        copy_factors(cache.factors), cache.embedding, cache.rules, cache.env_maxdim, cache.projector,
        copy(cache.bonds), copy(cache.sides), copy(cache.messages), copy(cache.corners), copy(cache.projectors),
        cache.sequences
    )
end

function CTMRGCache(ψ::ITensorNetwork, env_maxdim::Integer; kwargs...)
    factors = all(isempty, siteinds(ψ)) ? ψ : ITNN.normnetwork(ITensorBase.prime, ψ)
    return CTMRGCache(factors, default_messages(), env_maxdim; kwargs...)
end
CTMRGCache(form::Union{ITNN.NormNetwork, BilinearForm, ITNN.QuadraticFormNetwork}, env_maxdim::Integer; kwargs...) =
    CTMRGCache(form, default_messages(), env_maxdim; kwargs...)

# Edges without a message in `messages` start from a tensor of ones (see `initialize_blocks!`).
function CTMRGCache(
        factors, messages::Dictionary, env_maxdim::Integer;
        positions = default_positions(graph(factors)),
        projector::Symbol = :svd,
    )
    projector in (:svd, :eig) || throw(ArgumentError("projector must be :svd or :eig, got $(repr(projector))"))
    g = graph(factors)
    V = vertextype(g)
    emb = planar_embedding(g; positions)
    rules = ctmrg_rules(emb; jacobi = projector === :eig)
    bonds = Dict{Tuple{V, V}, Vector{Index}}()
    for e in edges(g)
        u, v = src(e), dst(e)
        bonds[(u, v)], bonds[(v, u)] = shared_legs(factors, u, v), shared_legs(factors, v, u)
    end
    # graded like the network's legs: one trivial sector
    ref = first(bonds[first(rules.interfaces)])
    sides = Dict{Tuple{V, V}, Index}(I => new_index(ref, 1; tags = "Link,ctm") for I in rules.interfaces)
    cache = CTMRGCache(
        factors, emb, rules, Int(env_maxdim), projector,
        bonds, sides, Dictionary{NamedEdge, Tensor}(messages), Dict{Tuple{V, Int}, Tensor}(),
        Dict{Tuple{V, V}, Tuple{Tensor, Tensor, Index}}(), Dict{Any, Any}()
    )
    initialize_blocks!(cache)
    return cache
end

# A message or corner by its recipe part, (:message, (u, v)) or (:corner, (p, f)).
block_tensor(cache::CTMRGCache, (kind, key)) = kind === :message ? cache.messages[NamedEdge(key[1] => key[2])] : cache.corners[key]
function set_block!(cache::CTMRGCache, (kind, key), t)
    kind === :message ? set!(cache.messages, NamedEdge(key[1] => key[2]), t) : (cache.corners[key] = t)
    return cache
end

# The legs u shares with v, as u's tensors carry them (graded legs have an orientation, which
# index equality ignores).
function shared_legs(factors, u, v)
    legs(x) = unique(Iterators.flatten(inds(t) for t in ITNN.factor_tensors(factors, x)))
    lv = legs(v)
    return Index[i for i in legs(u) if i in lv]
end

# The leg of interface I as a block on x's side carries it: P_A's leg on p's side (I = (p, q)), its
# dual on q's.
side_leg(cache::CTMRGCache, I, x) = x == first(I) ? cache.sides[I] : dag(cache.sides[I])

# Every block starts as a tensor of ones on its legs (the bonds it cuts and its interfaces); on graded
# legs, where ones are not symmetric, as a random symmetric tensor (seeded, so every run is the same).
function initialize_blocks!(cache::CTMRGCache)
    emb = cache.embedding
    ref = first(ITNN.factor_tensors(cache.factors, first(vertices(graph(cache)))))
    rng = Xoshiro(2)
    function ones_on(legs)
        any(isgraded, legs) && return adapt_like(ref, random_tensor(rng, scalartype(ref), legs))
        return adapt_like(ref, from_array(ones(scalartype(ref), dim.(legs)...), legs))
    end
    for (u, v) in darts(emb)
        legs = copy(cache.bonds[(u, v)])
        for h in (face_of_dart(emb, u, v), face_of_dart(emb, v, u))
            isinner(h) && push!(legs, side_leg(cache, interface_dart(emb, h, u, other_neighbor(emb, h, u, v)), u))
        end
        haskey(cache.messages, NamedEdge(u => v)) || set!(cache.messages, NamedEdge(u => v), ones_on(legs))
    end
    for (p, f) in corner_keys(emb)
        w = face_walk(emb, f, p)
        cache.corners[(p, f)] = ones_on([side_leg(cache, interface_dart(emb, f, w[end], p), p), side_leg(cache, interface_dart(emb, f, p, w[2]), p)])
    end
    return cache
end

# The tensors of a recipe. `projectors` and `site` can be overridden (a pass's new projectors, a site
# with an operator inserted); enlarged corners are kept in `memo`, if given, while the blocks don't change.
function recipe_tensors(
        cache::CTMRGCache, parts;
        projectors = cache.projectors,
        site = v -> ITNN.factor_tensors(cache.factors, v),
        memo = nothing,
    )
    ts = Tensor[]
    for part in parts
        kind = first(part)
        if kind === :site
            append!(ts, site(part[2]))
        elseif kind === :message || kind === :corner
            push!(ts, block_tensor(cache, part))
        elseif kind === :enlarged
            enlarged() = evaluate_recipe(cache, enlarged_recipe(cache.embedding, part[2]...); projectors, site)
            push!(ts, isnothing(memo) ? enlarged() : get!(enlarged, memo, part[2]))
        elseif kind === :inverse
            chain = contract_blocks(cache, [cache.corners[k] for k in part[2]])
            push!(ts, pseudo_inverse(chain, cache.sides[part[3][1]], cache.sides[part[3][2]], PINV_RTOL))
        else
            P = projectors[part[2]]
            push!(ts, part[3] === :A ? P[1] : P[2])
        end
    end
    return ts
end

evaluate_recipe(cache::CTMRGCache, parts; kwargs...) = contract_blocks(cache, recipe_tensors(cache, parts; kwargs...))

# The same lists recur every pass, so contraction orders are kept by the lists' structure.
function contract_blocks(cache::CTMRGCache, ts::Vector)
    length(ts) == 1 && return only(ts)
    length(ts) == 2 && return ts[1] * ts[2]
    seen = Dict{Any, Int}()
    structure = Tuple(Tuple((get!(seen, i, length(seen) + 1), dim(i)) for i in inds(t)) for t in ts)
    sequence = get!(cache.sequences, structure) do
        length(ts) <= 12 ? contraction_sequence(ts; alg = "optimal") :
            contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod())
    end
    return contract(ts; sequence)
end

# The relative cutoff regularising the pseudo-inverses some corner updates use.
const PINV_RTOL = 1.0e-3

# The pseudo-inverse N of a two-leg connector M, so that M N M = M (bilinearly): the map adjoint of
# U S⁻¹ V, for M = U S V. Singular values below `rtol` (relative) are dropped.
function pseudo_inverse(M, l1, l2, rtol)
    U, S, V = svd(M, [l1])
    cut = rtol * maximum(abs, array(S))
    return dag(U * map_diag(s -> abs(s) > cut ? inv(s) : zero(s), S) * V, [l1])
end

# B with a new leg of dimension one, and that leg.
function with_trivial_leg(B, tags)
    c = new_index(collect(inds(B)), 1; tags)
    return B * adapt_like(B, onehot(scalartype(B), c => 1)), c
end

# A block as a column on the interface `ins`: its other legs fused, normalised.
function block_column(B, ins)
    B = B / norm(B)
    rest = uniqueinds(B, ins)
    isempty(rest) && return with_trivial_leg(B, "Link,col")
    C = combiner(rest; tags = "Link,col")
    return B * C, combinedind(C)
end

# Blocks side by side, as the direct sum of their columns.
function stack_blocks(blocks, ins)
    acc, c = block_column(first(blocks), ins)
    for B in blocks[2:end]
        col, cB = block_column(B, ins)
        acc, c = directsum(acc => c, col => cB; tags = "Link,col")
    end
    return acc
end

# One side of an interface reduced to its triangular factor (B = QR) when that makes it smaller.
function block_triangular(B, ins)
    rest = uniqueinds(B, ins)
    isempty(rest) && return first(with_trivial_leg(B, "Link,qr"))
    !isgraded(B) && dim(rest) <= dim(ins) && return B
    _, R = qr(B, rest)
    return R
end

# The projector pair at an interface, from the blocks on its left (A) and right (B): with the
# truncated SVD AᵀB = U S V†, P_A = B V S^(-1/2) and P_B = S^(-1/2) U† Aᵀ, so that Aᵀ P_A P_B B = Aᵀ B
# at full rank. The pairing is bilinear because the network contracts the two sides plainly.
function projector_pair(left, right, ins, maxdim)
    # only blocks with every leg of the interface can be stacked into columns on it
    covers(B) = all(i -> i in inds(B), ins)
    side(blocks) = block_triangular(length(blocks) == 1 ? only(blocks) : stack_blocks(filter(covers, blocks), ins), ins)
    A, B = side(left), side(right)
    U, S, V = truncated_svd(A * B, uniqueinds(A, ins); trunc = truncation_strategy(; maxdim, rtol = 1.0e-12))
    u, v = inds(S)
    whiten = dag(map_diag(x -> inv(sqrt(x)), S), [u])
    PA = (B * dag(V, [v])) * whiten
    PB = (dag(U, [u]) * A) * whiten
    w = only(uniqueinds(PA, ins))
    return PA, replaceind(PB, only(uniqueinds(PB, ins)), dag(w)), w
end

# A random isometry for the first pass: lossless at lossless env_maxdim, so that pass is already
# exact. It is the left factor of a random square map on `ins` (graded: block diagonal over the
# sectors `ins` fuse to, so every sector is kept), truncated to env_maxdim. `ins` are the legs as
# P_A carries them.
function random_projector(cache::CTMRGCache, ins, rng)
    ref = first(cache.messages)
    R = adapt_like(ref, random_tensor(rng, real(scalartype(ref)), [ins; dag(sim(ins))]))
    Q, _ = truncated_svd(R, ins; trunc = truncation_strategy(; maxdim = cache.env_maxdim))
    return Q, dag(Q, ins), only(uniqueinds(Q, ins))
end

function set_projectors!(cache::CTMRGCache, projectors)
    merge!(cache.projectors, projectors)
    for (I, P) in projectors
        cache.sides[I] = P[3]
    end
    return cache
end

# The legs of interface I = (p, q) as P_A carries them, i.e. as the blocks on q's side do: its bonds,
# and the leg of the interface J its ray continues into (whose P_A is on p's side too).
function interface_legs(cache::CTMRGCache, I, side_of)
    J = cache.rules.next[I]
    bonds = cache.bonds[reverse(I)]
    return isnothing(J) ? bonds : Index[bonds; [dag(side_of(J))]]
end

function random_projectors!(cache::CTMRGCache)
    new = empty(cache.projectors)
    rng = Xoshiro(1)    # fixed, so every run of the first pass is the same
    for I in cache.rules.interfaces
        new[I] = random_projector(cache, interface_legs(cache, I, J -> new[J][3]), rng)
    end
    return set_projectors!(cache, new)
end

# The projectors for a sequential pass, boundary first. The blocks either side of an interface are in
# the previous pass's coordinates, so their leg on the next interface J is first carried into J's new
# ones (by P_B(old) M_J P_A(new) on the p side, P_A(old) M_J P_B(new) on the q side). Where a
# directional corner is projected, which has no enlarged corner behind it, every block projected
# there is stacked in so that each stays lossless.
function carried_projectors!(cache::CTMRGCache)
    emb, rules = cache.embedding, cache.rules
    memo = Dict{Any, Tensor}()
    new = empty(cache.projectors)
    carry = Dict{Any, Tensor}()
    for I in rules.interfaces
        p, q = I
        f = face_of_dart(emb, p, q)
        J = rules.next[I]
        Ql, Qr = connectors(emb, rules.exits, p, q)
        # a block on side A (p's) or B (q's) of I with its connector, its leg on J carried over
        block(Q, side, parts) = (t = evaluate_recipe(cache, [Q; parts]; memo); isnothing(J) ? t : t * carry[(J, side)])
        left = Tensor[block(Ql, :A, [(:enlarged, (p, f))])]
        right = Tensor[block(Qr, :B, [(:enlarged, (q, f))])]
        directional = []
        for c in (p, q)
            rules.modes[(c, f)] === :diag && continue
            parts, a = directional_core(emb, c, f, rules.modes[(c, f)])
            a == (c == p ? q : p) && push!(directional, (c, parts))
        end
        if !isempty(directional)
            # the messages out of p and q to their other neighbours on f (f's walk from p is p, q, …)
            w = face_walk(emb, f, p)
            push!(left, block(Ql, :A, first(message_core(emb, p, w[end]))))
            push!(right, block(Qr, :B, first(message_core(emb, q, w[3]))))
            for (c, parts) in directional
                c == p ? push!(left, block(Ql, :A, parts)) : push!(right, block(Qr, :B, parts))
            end
        end
        new[I] = projector_pair(left, right, interface_legs(cache, I, J -> new[J][3]), cache.env_maxdim)
        old = cache.projectors[I]
        carry[(I, :A)] = contract_blocks(cache, isnothing(J) ? [old[2], new[I][1]] : [old[2], carry[(J, :A)], new[I][1]])
        carry[(I, :B)] = contract_blocks(cache, isnothing(J) ? [old[1], new[I][2]] : [old[1], carry[(J, :B)], new[I][2]])
    end
    return set_projectors!(cache, new)
end

# Eigenvalue projectors for the interfaces of face f. They cut the ring of its enlarged corners,
# Z_f = tr(K_1 ⋯ K_n), and each keeps the dominant invariant subspaces of the ring's transfer matrix
# at that interface (the eigenvalues all the transfer matrices share). A face whose spectrum fits in
# env_maxdim is left to the SVD projector, which is lossless there: these would drop the ring's
# kernel, which other faces see. `nothing` also if a block has legs beyond its two interfaces.
function eig_projectors(cache::CTMRGCache, f, ins_of, memo)
    emb, exits = cache.embedding, cache.rules.exits
    w = face_vertices(emb, f)
    n = length(w)
    I = [(w[k], w[mod1(k + 1, n)]) for k in 1:n]
    ins = [ins_of(I[k]) for k in 1:n]
    before(k) = ins[mod1(k - 1, n)]
    K = map(1:n) do k
        _, Qr = connectors(emb, exits, I[mod1(k - 1, n)]...)
        Ql, _ = connectors(emb, exits, I[k]...)
        return evaluate_recipe(cache, [Qr; (:enlarged, (w[k], f)); Ql]; memo)
    end
    all(k -> issetequal(inds(K[k]), [before(k); ins[k]]), 1:n) || return nothing
    # the dense eigenbases would mix sectors: graded faces are left to the SVD projector
    any(isgraded, K) && return nothing
    # the ring's eigenproblem is small and runs on the host
    mats = [Array(reshape(array(K[k], before(k)..., ins[k]...), dim(before(k)), dim(ins[k]))) for k in 1:n]
    # The ring is a product of n blocks, so its eigenvalues are only known to about N ε ∏‖K‖
    noise = maximum(size.(mats, 2)) * eps(real(eltype(first(mats)))) * prod(norm, mats)
    bases = ring_bases(mats, min(cache.env_maxdim, minimum(size.(mats, 2))), noise)
    isnothing(bases) && return nothing
    projectors = empty(cache.projectors)
    for k in 1:n
        function basis(X, tags)
            c = new_index(size(X, 2); tags)
            return from_array(reshape(X, dim.(ins[k])..., size(X, 2)), [ins[k]; [c]])
        end
        V, W = basis(bases[1][k], "Link,right"), basis(bases[2][k], "Link,left")
        projectors[I[k]] = projector_pair([adapt_like(K[k], W)], [adapt_like(K[k], V)], ins[k], cache.env_maxdim)
    end
    return projectors
end

# Right and left bases of the kept invariant subspace at every interface of a ring, or `nothing` if
# the face is left to the SVD projector. They are found at the last interface and carried around the
# ring by the blocks.
function ring_bases(mats, r, noise)
    n = length(mats)
    found = two_sided_bases(mats, r, noise)
    isnothing(found) && return nothing
    orthonormal(X) = Matrix(qr(X).Q)[:, 1:size(X, 2)]
    Vs = Vector{Matrix}(undef, n)
    Ws = Vector{Matrix}(undef, n)
    Vs[n], Ws[n] = found
    for k in n:-1:2
        Vs[k - 1] = orthonormal(mats[k] * Vs[k])
    end
    for k in 1:(n - 1)
        Ws[k] = orthonormal(transpose(mats[k]) * Ws[mod1(k - 1, n)])
    end
    return Vs, Ws
end

# Rings up to this size are solved on the whole space.
const DENSE_RING = 400

# Right and left bases (orthonormal) of the ring's kept invariant subspace, by a two-sided block
# Krylov-Schur: block Krylov bases for the ring and its transpose, and a Rayleigh-Ritz on the pencil
# (Wᵀ R V, Wᵀ V), so that both sides belong to the same Ritz values; restarted from the leading Schur
# directions, and on the whole space for small rings. At most r are kept, never splitting eigenvalues
# that are equal at working precision (or a conjugate pair). `nothing` if the spectrum fits in r
# above the `noise` (the SVD projector is lossless there), or if there is no gap to cut at, or no
# convergence.
function two_sided_bases(mats, r, noise)
    N = size(mats[end], 2)
    T = eltype(first(mats))
    ring(x) = foldr(*, mats; init = x)
    ringᵀ(y) = foldl((y, K) -> transpose(K) * y, mats; init = y)
    orthonormal(X) = Matrix(qr(X).Q)[:, 1:size(X, 2)]
    dense = N <= DENSE_RING
    b = min(max(r + 4, 2r), N)
    depth = min(3, max(1, N ÷ b))
    rng = Xoshiro(N)
    V, W = orthonormal(randn(rng, T, N, b)), orthonormal(randn(rng, T, N, b))
    gap = 1000 * eps(real(T))
    for _ in 1:50
        if dense
            Vk = Wk = Matrix{T}(LinearAlgebra.I, N, N)
        else
            Vk, Wk = block_krylov(ring, V, depth, N), block_krylov(ringᵀ, W, depth, N)
            d = min(size(Vk, 2), size(Wk, 2))
            Vk, Wk = Vk[:, 1:d], Wk[:, 1:d]
        end
        M = transpose(Wk) * Vk
        F = schur(complex(M \ (transpose(Wk) * ring(Vk))))
        order = sortperm(abs.(F.values); rev = true)
        λ1 = abs(F.values[order[1]])
        a = abs.(F.values[order]) ./ λ1
        k = findlast(j -> (j < length(a) ? a[j] - a[j + 1] : a[j]) >= gap, 1:min(r, length(a)))
        isnothing(k) && return nothing
        right, left = leading_pencil_bases(F, order, M, k)
        Vo, Wo = orthonormal(Vk * right), orthonormal(Wk * left)
        RVo, RᵀWo = ring(Vo), ringᵀ(Wo)
        residual = max(norm(RVo - Vo * (Vo' * RVo)), norm(RᵀWo - Wo * (Wo' * RᵀWo))) / λ1
        if dense || residual <= max(10 * noise / λ1, 100 * eps(real(T)))
            (k >= N || (k < length(a) && a[k + 1] * λ1 <= noise)) && return nothing
            return Vo, Wo
        end
        right, left = leading_pencil_bases(F, order, M, min(b, length(order)))
        V, W = orthonormal(Vk * right), orthonormal(Wk * left)
    end
    return nothing
end

# Orthonormal [X, A X, …, A^(m−1) X] for X orthonormal, without the directions already in the span
# (a breakdown), and at most `maxdim` wide.
function block_krylov(apply, X, m, maxdim)
    blocks = [X]
    total = size(X, 2)
    for _ in 2:m
        Y = apply(blocks[end])
        scale = norm(Y)
        for _ in 1:2, B in blocks
            Y -= B * (B' * Y)
        end
        F = svd(Y)
        keep = min(count(>(1000 * eps(real(eltype(Y))) * scale), F.S), maxdim - total)
        keep == 0 && break
        push!(blocks, F.U[:, 1:keep])
        total += keep
    end
    return reduce(hcat, blocks)
end

# The pencil's right and left coordinates (in the Krylov bases) of the invariant subspace of its k
# eigenvalues of largest modulus, from one Schur form F of M \ (Wᵀ R V): the right ones are its leading
# Schur vectors, the left ones the rows [I X] Z⁻¹ (X from a Sylvester solve), brought back through M.
function leading_pencil_bases(F, order, M, k)
    keep = falses(length(F.values))
    keep[order[1:k]] .= true
    G = ordschur!(LinearAlgebra.Schur(copy(F.T), copy(F.Z), copy(F.values)), keep)
    T, Z = G.T, G.Z
    n = size(T, 1)
    X = k == n ? zeros(eltype(T), k, 0) : sylvester(T[1:k, 1:k], -T[(k + 1):end, (k + 1):end], -T[1:k, (k + 1):end])
    Lt = [Matrix{eltype(T)}(LinearAlgebra.I, k, k) X] / Z
    return Z[:, 1:k], transpose(M) \ transpose(Lt)
end

function recompute_block(cache::CTMRGCache, block; kwargs...)
    rules = cache.rules
    t = evaluate_recipe(cache, block_recipe(cache.embedding, rules.exits, rules.modes, block); kwargs...)
    return t / norm(t)
end

# A sequential pass: new projectors, then every block once, in dependency order.
function sequential_pass!(cache::CTMRGCache)
    isempty(cache.projectors) ? random_projectors!(cache) : carried_projectors!(cache)
    for block in cache.rules.order
        set_block!(cache, block, recompute_block(cache, block))
    end
    return cache
end

# A Jacobi pass: new projectors, then every block, all from the blocks of the previous pass.
function jacobi_pass!(cache::CTMRGCache)
    emb, rules = cache.embedding, cache.rules
    memo = Dict{Any, Tensor}()
    ins(I) = interface_legs(cache, I, J -> cache.sides[J])
    new = empty(cache.projectors)
    if cache.projector === :eig
        for f in eachindex(faces(emb))
            pairs = eig_projectors(cache, f, ins, memo)
            isnothing(pairs) || merge!(new, pairs)
        end
    end
    for I in rules.interfaces
        haskey(new, I) && continue
        p, q = I
        f = face_of_dart(emb, p, q)
        Ql, Qr = connectors(emb, rules.exits, p, q)
        Bw = evaluate_recipe(cache, [Ql; (:enlarged, (p, f))]; memo)
        Be = evaluate_recipe(cache, [Qr; (:enlarged, (q, f))]; memo)
        new[I] = projector_pair([Bw], [Be], ins(I), cache.env_maxdim)
    end
    blocks = Dict(b => recompute_block(cache, b; projectors = new, memo) for b in block_keys(emb))
    for (b, t) in blocks
        set_block!(cache, b, t)
    end
    return set_projectors!(cache, new)
end

"""
    update(cache::CTMRGCache; convergence = :messages, tolerance, maxiter = 30, verbose = false)

Update the corners and messages of a `CTMRGCache` until they converge.

# Arguments
- `cache::CTMRGCache`: The cache to update.

# Keyword Arguments
- `convergence::Symbol = :messages`: What has to converge. `:messages` (for expectation values):
  the environment of every vertex, by the same distance BP uses for its messages, 1 − |⟨a, b⟩|² /
  (‖a‖‖b‖)², so a tolerance τ settles the environments to about √τ relative. `:freenergy` (enough for
  the free energy): `freenergy(cache)`, relative, which settles first.
- `tolerance`: Convergence threshold (default `1e-12` for `:messages` and `1e-10` for `:freenergy`,
  but never below `100 eps` of the network's precision, so `1.2e-5` in single precision).
- `maxiter::Integer = 30`: Maximum number of passes.
- `verbose::Bool = false`: Print the outcome.

# Returns
- An updated copy of the cache.
"""
function update(
        cache::CTMRGCache;
        convergence::Symbol = :messages,
        tolerance::Real = default_ctmrg_tolerance(cache, convergence),
        maxiter::Integer = 30,
        verbose::Bool = false,
    )
    convergence in (:messages, :freenergy) || throw(ArgumentError("convergence must be :messages or :freenergy, got $(repr(convergence))"))
    cache = copy(cache)
    pass! = isnothing(cache.rules.order) ? jacobi_pass! : sequential_pass!
    signal() = convergence === :messages ? vertex_environments(cache) : freenergy(cache)
    change(a, b) = convergence === :messages ? maximum(message_diff.(a, b)) : abs(a - b) / max(1, abs(a))
    previous = nothing
    function step!()
        pass!(cache)
        current = signal()
        c = isnothing(previous) ? nothing : change(current, previous)
        previous = current
        return c
    end
    iterate_to_convergence(step!, maxiter, tolerance; name = "CTMRG ($convergence)", verbose)
    return cache
end

# The default convergence threshold: what double precision resolves, and no less than the network's
# own precision does.
default_ctmrg_tolerance(cache::CTMRGCache, convergence) =
    max(convergence === :messages ? 1.0e-12 : 1.0e-10, 100 * eps(real(scalartype(network(cache)))))

# The environment of every vertex: the messages into it and the corners around it, contracted.
function vertex_environments(cache::CTMRGCache)
    return map(v -> evaluate_recipe(cache, vertex_environment(cache.embedding, v)), collect(vertices(cache)))
end

"""
    incoming_messages(cache::CTMRGCache, vertices)

The environment of one vertex, or of the two ends of an edge: the messages into it and the
corners around it.
"""
incoming_messages(cache::CTMRGCache, vs::Vector) = recipe_tensors(cache, environment_recipe(cache, vs))

"""
    freenergy(cache::CTMRGCache)

The free energy `-ln Z` of the network, `ln Z` from the cache by the standard CTM estimator,
`Σ_v ln Z_v − Σ_e ln Z_e + Σ_f ln Z_f`: `Z_v` closes vertex `v` (its site, messages and corners), `Z_e`
is the cut across edge `e` (its two messages and the corners between them) and `Z_f` the ring of face
`f`'s corners. Every message and corner enters with net count zero, so their normalisations drop out.
With `env_maxdim = 1` the corners are scalars and this is the Bethe free energy of the messages.
"""
function freenergy(cache::CTMRGCache)
    emb = cache.embedding
    g = graph(emb)
    memo = Dict{Any, Tensor}()
    lnZ(parts) = principal_log(scalar(evaluate_recipe(cache, parts; memo)))
    F = sum(lnZ(vertex_region(emb, v)) for v in vertices(g); init = 0.0)
    F -= sum(lnZ(edge_cut(emb, src(e), dst(e))) for e in edges(g); init = 0.0)
    F += sum(lnZ(face_cut(emb, f)) for f in eachindex(faces(emb)); init = 0.0)
    return -F
end

# A CTMRG cache closes regions of one vertex or of the two ends of an edge.
function observable_region(cache::CTMRGCache, vs)
    length(vs) == 1 || (length(vs) == 2 && has_edge(graph(cache), vs...)) ||
        error("alg = \"ctmrg\" supports regions of one vertex or the two ends of an edge; got $vs.")
    return vs
end

# The environment of, and the closed region around, such a region.
function environment_recipe(cache::CTMRGCache, region)
    region = observable_region(cache, region)
    return length(region) == 1 ? vertex_environment(cache.embedding, only(region)) : edge_environment(cache.embedding, region...)
end
function region_recipe(cache::CTMRGCache, region)
    region = observable_region(cache, region)
    return length(region) == 1 ? vertex_region(cache.embedding, only(region)) : edge_region(cache.embedding, region...)
end

# The region's sites, with operators from `op_strings`, contracted against its environment. The
# closed region's recipe orders the contraction through enlarged corners. A joint operator
# `(name, vertices)` keeps the site legs of its vertices primed through the recipe (inside their
# enlarged corners) and bridges them with one operator tensor at the end.
function contract_region(cache::CTMRGCache, region, op_strings::Function; joint_op = nothing)
    tn = network(cache)
    isnothing(joint_op) && return evaluate_recipe(cache, region_recipe(cache, region); site = v -> norm_factors(tn, [v]; op_strings))
    jverts = last(joint_op)
    site = v -> norm_factors(tn, [v]; op_strings = w -> w ∈ jverts ? "ρ" : op_strings(w))
    ts = recipe_tensors(cache, region_recipe(cache, region); site)
    jinds = [only(siteinds(tn, v)) for v in jverts]
    push!(ts, adapt_like(tn[first(jverts)], op(first(joint_op), jinds...)))
    return contract_blocks(cache, ts)
end

function region_scalar(cache::CTMRGCache, region, op_strings::Function; joint_op = nothing)
    isnothing(joint_op) || issubset(last(joint_op), region) ||
        error("alg = \"ctmrg\": the joint operator's vertices $(last(joint_op)) must lie in the region $region.")
    return scalar(contract_region(cache, region, op_strings; joint_op))
end

# Messages and factors as for any cache, and the corners and projectors with them.
function Adapt.adapt_structure(to, cache::CTMRGCache)
    cache = adapt_factors(to, adapt_messages(to, cache))
    for (k, c) in cache.corners
        cache.corners[k] = adapt(to)(c)
    end
    for (I, (PA, PB, k)) in cache.projectors
        cache.projectors[I] = (adapt(to)(PA), adapt(to)(PB), k)
    end
    return cache
end
