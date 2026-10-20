using Random: Xoshiro

"""
    CTMRGCache(network, maxdim::Integer; positions, pinv_rtol = 1e-3, seed = 1)

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
"""
struct CTMRGCache{V, N}
    network::N
    embedding::PlanarEmbedding{V}
    rules::CTMRGRules{V}
    maxdim::Int
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
        cache.network, cache.embedding, cache.rules, cache.maxdim, cache.pinv_rtol, copy(cache.rng),
        copy(cache.bonds), copy(cache.sides), copy(cache.blocks), copy(cache.projectors)
    )
end

function CTMRGCache(
        network, maxdim::Integer;
        positions = default_positions(graph(network)),
        pinv_rtol::Real = 1.0e-3,
        seed::Integer = 1,
    )
    g = graph(network)
    V = vertextype(g)
    emb = planar_embedding(g; positions)
    rules = ctmrg_rules(emb)
    bonds = Dict{Tuple{V, V}, Vector{Index}}()
    for e in edges(g)
        u, v = src(e), dst(e)
        bonds[(u, v)] = bonds[(v, u)] = _shared_legs(network, u, v)
    end
    sides = Dict{Tuple{V, V}, Index}(I => new_index(1; tags = "Link,ctm") for I in rules.interfaces)
    cache = CTMRGCache(
        network, emb, rules, Int(maxdim), Float64(pinv_rtol), Xoshiro(seed),
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

# A Jacobi pass, for rules with no acyclic order (diagonal corners only): every projector from the
# current blocks, then every block recomputed from the current blocks and the new projectors.
function _jacobi_pass!(cache::CTMRGCache)
    emb, rules = cache.embedding, cache.rules
    memo = Dict{Any, Tensor}()           # the blocks stay unchanged until the end of the pass
    new = Dict{Any, Any}()
    for I in rules.interfaces
        p, q = I
        f = face_of_dart(emb, p, q)
        Ql, Qr = _connectors(emb, rules.exits, p, q)
        Bw = _evaluate(cache, [Ql; (:enlarged, (p, f))]; memo)
        Be = _evaluate(cache, [Qr; (:enlarged, (q, f))]; memo)
        new[I] = _projector_pair([Bw], [Be], _interface_legs(cache, I, J -> cache.sides[J]), cache.maxdim)
    end
    blocks = Dict(b => _recompute(cache, b; projectors = new, memo) for b in keys(cache.blocks))
    merge!(cache.blocks, blocks)
    return _set_projectors!(cache, new)
end

"""
    update(cache::CTMRGCache; maxiter = 30, tolerance = 1e-10, verbose = false)

Run passes over the cache until no closed region (the network with one vertex, edge or face
explicit) changes its log-contraction by more than `tolerance` between passes, or `maxiter`
passes. Observables are ratios of these regions, so this waits for them too, not just for the
free energy, which settles first. At lossless `maxdim` two passes suffice.
"""
function update(cache::CTMRGCache; maxiter::Integer = 30, tolerance::Real = 1.0e-10, verbose::Bool = false)
    cache = copy(cache)
    pass! = isnothing(cache.rules.order) ? _jacobi_pass! : _sequential_pass!
    previous = nothing
    for i in 1:maxiter
        pass!(cache)
        regions = _region_logs(cache)
        if !isnothing(previous) && maximum(abs, last.(regions) .- last.(previous)) <= tolerance
            verbose && println("CTMRG converged after $i passes.")
            return cache
        end
        previous = regions
    end
    msg = "CTMRG did not converge to tolerance $tolerance after $maxiter passes."
    verbose ? println(msg) : @warn(msg)
    return cache
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
