# ══════════════════════════════════════════════════════════════════════════════════════════
# Linear response of the CTM environment to the Hamiltonian: the (μ, ν) tangent solve.
#
# WHAT IT REPLACES. The generating-function optimiser (`dmrg(…; alg = "ctmrg_lbfgs")`) reads its
# energy and effective operators from two CTM environments converged on the a-widened network
# ⟨ψ|G(±λ)|ψ⟩, whose site tensors carry an (r+1)-dimensional auxiliary leg on every bond (r the
# operator-Schmidt rank of the edge terms): every contraction and every seam pays powers of (r+1).
#
# THE IDEA. Split the coupling: the source factor of every edge term carries μ, the sink factor ν,
#
#     L_e(μ) = 1⊗|0⟩ + μ Σₐ Aₐ⊗|a⟩,     R_e(ν) = 1⊗|0⟩ + ν Σₐ Bₐ⊗|a⟩,     Σₐ L_a R_a = 1 + μν h_e,
#
# and an on-site term as 1 + μν h_v. Then Z(μ, ν) = Z(λ = μν) EXACTLY, and
#
#     E = dF/dλ|₀ = ∂μ∂ν F|₀ .
#
# EVERYTHING IS A POLYNOMIAL IN (μ, ν) TRUNCATED TO BIDEGREE ≤ (1, 1), and NO TENSOR CARRIES AN
# AUXILIARY LEG IT DOES NOT NEED. `PolyT` holds
#   c00  the norm-network tensor — no auxiliary legs at all (the a = 0 slot is implicit),
#   c10  a LIST of terms, each holding exactly ONE open A-half-insertion: either an auxiliary leg
#        α_e (dimension r+1, slot 0 empty) on a bond e still open on the tensor, or an excited seam
#        bond (tag "exc") into which such a leg was compressed,
#   c01  the same for B-half-insertions,
#   c11  one tensor with no open excitation: the closed insertions.
# Products (`pmul`) are the truncated polynomial product with two structural rules: a first-order
# term dies when its bond's ket leg is contracted without its partner's half-insertion (the exact
# network contracts its auxiliary leg with |0⟩ there), and a pair of half-insertions contributes to
# c11 only if it closes (no excitation left). So the (0,0) coefficient of every step is exactly a
# norm sweep, and the first-order terms cost (r+1) on one leg — linear in r, not (r+1)^4. (The first
# version padded every tensor with all its |0⟩ auxiliary legs; measured 5×5 D = 3 χ = 32: its sweep
# scaled with r exactly like the ±λ pair, 5.3× from r = 1 to r = 3, and cost 1.8× the pair.)
#
# THE TRUNCATION. The excited terms must cross the truncated seams: the O(μ) content of a corner's
# free legs is what the ring at a neighbouring vertex reads. The engine's biorthogonal (cut) pair
# cannot carry them — sandwiched between its two corners it truncates the seam operator to the top-χ
# SPECTRAL space of O⁰O⁰†, and the excited content has O(μ²) eigenvalues; forcing it in needs a 1/μ
# gain, which is not a polynomial (docs/dmrg.md, 2026-09-23). So the seams are compressed with
# constant ISOMETRIES (P_A = V, P_B = V†), one per SECTOR of the seam:
#   A   terms without an excitation on a seam bond: `χ` directions from both corners' norm content
#       plus `χ1` from their first/second-order content, onto the bond w_A;
#   B_e terms whose excitation sits on the seam bond e (legs ins ∪ α_e): `χ1` directions onto the
#       excited bond w_e.
# A block's c10/c01 lists then hold one term per link sector; c00 and c11 live on the A bonds.
# Lossless it is exact; truncated, the norm part is a corner-isometry CTMRG. After `nfree` sweeps
# the isometries are frozen and the (then linear) iteration converges geometrically.
# ══════════════════════════════════════════════════════════════════════════════════════════

_terms(x) = x === nothing ? Any[] : x isa AbstractVector ? Any[x...] : Any[x]
struct PolyT
    c00::Any
    c10::Vector{Any}
    c01::Vector{Any}
    c11::Any
    PolyT(c00, c10, c01, c11) = new(c00, _terms(c10), _terms(c01), c11)
end
PolyT(; c00 = nothing, c10 = nothing, c01 = nothing, c11 = nothing) = PolyT(c00, c10, c01, c11)
pconst(t) = PolyT(t, nothing, nothing, nothing)
_padd(a, b) = a === nothing ? b : (b === nothing ? a : a + b)
padd(p::PolyT, q::PolyT) = PolyT(_padd(p.c00, q.c00), _merge(vcat(p.c10, q.c10)), _merge(vcat(p.c01, q.c01)), _padd(p.c11, q.c11))
pmap(f, p::PolyT) = PolyT(p.c00 === nothing ? nothing : f(p.c00), Any[f(t) for t in p.c10], Any[f(t) for t in p.c01],
                          p.c11 === nothing ? nothing : f(p.c11))
_pf(a, b) = (a === nothing || b === nothing) ? nothing : a * b

# ── excitations: auxiliary legs and excited seam bonds ───────────────────────────────────
_isaux(i::Index) = occursin("aux", string(TensorInterface.tags(i)))
_isexc(i::Index) = occursin("exc", string(TensorInterface.tags(i)))
_auxlegs(t) = filter(_isaux, collect(TensorInterface.inds(t)))
_exclegs(t) = filter(i -> _isaux(i) || _isexc(i), collect(TensorInterface.inds(t)))
_hasind(t, j) = any(i -> i == j, TensorInterface.inds(t))       # `==` ignores duality
_seteq(a, b) = length(a) == length(b) && all(x -> any(y -> x == y, b), a)
# Registries, keyed by index id (an index and its dual share it):
#   auxiliary index ⇒ the ket bond index of its edge (`response_operator`);
#   excited seam bond ⇒ the norm (sector-A) bond of the same seam (`_resp_projector`).
const RESP_AUX_BONDS = Dict{UInt64, Any}()      # keyed by the index id (shared by an index and its dual)
const RESP_LINK_FAMILY = Dict{UInt64, Any}()    # reset by every `response_solve`
_register!(reg, a, b) = (reg[Tensors._id(a)] = b; nothing)
_lookup(reg, a) = get(reg, Tensors._id(a), nothing)
_bond_of(α) = _lookup(RESP_AUX_BONDS, α)
_family(ℓ) = _lookup(RESP_LINK_FAMILY, ℓ)
# A first-order term is alive iff it carries exactly one excitation and no excited seam bond on it
# coexists with the norm bond of the same seam (two sectors of one bond open on one tensor: the
# zeroth-order partner was absorbed). These two rules hold at every step of a product. The third —
# an auxiliary leg's bond (ket or bra copy) must still be open on the term, else the partner was
# absorbed at zeroth order and the exact network contracts the leg with |0⟩ — holds only for a
# FINISHED block (mid-fold the half-insertion may not have met its own bond legs yet), so it is
# applied by `_finalize` at the end of every `pcontract` and `papply`.
function _alive(t)
    ex = _exclegs(t)
    length(ex) == 1 || return false
    for i in ex
        _isaux(i) && continue
        f = _family(i)
        f === nothing || !_hasind(t, f) || return false
    end
    return true
end
function _bondopen(t)
    for α in _auxlegs(t)
        b = _bond_of(α)
        b === nothing && continue
        _hasind(t, b) || _hasind(t, TensorInterface.prime(b)) || return false
    end
    return true
end
_prune(terms) = Any[t for t in terms if _alive(t)]
_finalize(p::PolyT) = PolyT(p.c00, Any[t for t in p.c10 if _bondopen(t)], Any[t for t in p.c01 if _bondopen(t)], p.c11)
_finalize(::Nothing) = nothing
# sum terms with the same leg set
function _merge(terms)
    out = Any[]
    for t in terms
        k = findfirst(u -> _seteq(collect(TensorInterface.inds(u)), collect(TensorInterface.inds(t))), out)
        k === nothing ? push!(out, t) : (out[k] = out[k] + t)
    end
    return out
end
# a pair of half-insertions contributes to c11 only if it closes: the same excitation on both
function _pair(a, b)
    ea = _exclegs(a); eb = _exclegs(b)
    (length(ea) == 1 && length(eb) == 1 && ea[1] == eb[1]) || return nothing
    ab = a * b
    return isempty(_exclegs(ab)) ? ab : nothing
end

# A first-order term meeting a zeroth-order factor that carries the NORM bond of the term's excited
# seam is dead (the family rule) — and its product would not contract that link at all, forming an
# outer product of the two tensors (measured: 27 s per enlarged corner against 20 ms for the norm
# contraction, all of it in such products that `_prune` then discarded). Check before multiplying.
function _dead_with(t, other)
    for i in TensorInterface.inds(t)
        _isexc(i) || continue
        f = _family(i)
        f === nothing || !_hasind(other, f) || return true
    end
    return false
end
# product of two polynomials, truncated to bidegree (1,1)
function pmul(p::PolyT, q::PolyT)
    c00 = _pf(p.c00, q.c00)
    c10 = Any[]; c01 = Any[]
    if q.c00 !== nothing
        append!(c10, (t * q.c00 for t in p.c10 if !_dead_with(t, q.c00)))
        append!(c01, (t * q.c00 for t in p.c01 if !_dead_with(t, q.c00)))
    end
    if p.c00 !== nothing
        append!(c10, (p.c00 * t for t in q.c10 if !_dead_with(t, p.c00)))
        append!(c01, (p.c00 * t for t in q.c01 if !_dead_with(t, p.c00)))
    end
    c11 = _padd(_pf(p.c11, q.c00), _pf(p.c00, q.c11))
    for a in p.c10, b in q.c01
        c11 = _padd(c11, _pair(a, b))
    end
    for a in p.c01, b in q.c10
        c11 = _padd(c11, _pair(a, b))
    end
    return PolyT(c00, _merge(_prune(c10)), _merge(_prune(c01)), c11)
end
# Flatten (PolyT | Vector{PolyT} | nothing)… into one list.
function _plist(args...)
    ps = PolyT[]
    for a in args
        a === nothing && continue
        a isa PolyT ? push!(ps, a) : append!(ps, a)
    end
    return ps
end
# n-ary product: the base tensors' contraction sequence (the engine's netcon, cached on the index
# structure like `_ctm_contract`) folded pairwise with `pmul`.
function pcontract(ps::Vector{PolyT}, opts::CTMOptions)
    isempty(ps) && return nothing
    n = length(ps)
    c0 = [p.c00 for p in ps]
    any(c -> c === nothing, c0) && return PolyT()      # a zero base factor kills every term
    n == 1 && return _finalize(ps[1])
    n == 2 && return _finalize(pmul(ps[1], ps[2]))
    use_optimal = n <= opts.optimal_max
    key = (_ctm_seq_key(c0), use_optimal, :poly)
    seq = lock(CTM_GLOBAL_LOCK) do
        get(CTM_SEQ_CACHE, key, nothing)
    end
    if seq === nothing
        ts = AbstractTensor[c for c in c0]
        seq = use_optimal ? contraction_sequence(ts; alg = "optimal") :
                            contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod())
        lock(CTM_GLOBAL_LOCK) do
            CTM_SEQ_CACHE[key] = seq
        end
    end
    fold(s) = s isa Integer ? ps[s] : reduce(pmul, (fold(x) for x in s))
    return _finalize(fold(seq))
end
# the block rescaling of the sweep: all coefficients by the norm of the base (a (μ,ν)-independent
# scalar, so the fixed point is unchanged and the tangent stays linear)
function prescale(p::Union{PolyT, Nothing})
    (p === nothing || p.c00 === nothing) && return p
    n = norm(p.c00)
    (iszero(n) || !isfinite(n)) && return p
    return pmap(t -> t / n, p)
end
# strip every auxiliary leg of a padded norm-environment block (contract with the |0⟩ slot)
function _strip_all_aux(t)
    (t === nothing || !(t isa AbstractTensor)) && return t
    for i in _auxlegs(t)
        t = t * TensorInterface.onehot(real(scalartype(t)), TensorInterface.dag(i) => 1)
    end
    return t
end

# ── the (μ, ν) generating operator ───────────────────────────────────────────────────────
"""
    response_operator(H, ψ, gen) -> Dictionary(v => PolyT)

Per-vertex operator-layer polynomials of the (μ, ν) network (header): `c00` the identity (no
auxiliary legs), `c10` one A-half-insertion per edge `v` is the source of (on gen's auxiliary index
of that edge, slot 0 empty), `c01` one B-half-insertion per edge it is the sink of, `c11` the
on-site terms. Registers every auxiliary index with its ket bond for the structural rules.
"""
function response_operator(H::Vector, ψ::TensorNetworkState, gen::GeneratingOperator; cutoff::Real = 1.0e-14)
    g = graph(ψ); s = siteinds(ψ); vs = collect(vertices(g))
    edge_terms = Dict{NamedEdge, Any}(); vertex_terms = Dict{Any, Any}()
    for term in H
        t, verts = _term_tensor(term, g, s)
        if length(verts) == 1
            v = only(verts); vertex_terms[v] = haskey(vertex_terms, v) ? vertex_terms[v] + t : t
        elseif length(verts) == 2
            e = NamedEdge(verts[1] => verts[2])
            key = e ∈ edges(g) ? e : reverse(e)
            edge_terms[key] = haskey(edge_terms, key) ? edge_terms[key] + t : t
        else
            error("response_operator: only one- and two-site terms are supported, got $(term)")
        end
    end
    elt = promote_type(Float64, (eltype(t) for t in values(edge_terms))..., (eltype(t) for t in values(vertex_terms))...)
    c10 = Dict(v => Any[] for v in vs); c01 = Dict(v => Any[] for v in vs)
    for (e, h) in edge_terms
        u, w = src(e), dst(e)
        su, sw = only(s[u]), only(s[w])
        A, B, _ = factorize_svd(h, [su, TensorInterface.prime(su)]; ortho = "none", cutoff)
        k = only(TensorInterface.commoninds(A, B))
        α = only(virtualinds(gen.value, e))
        r = TensorInterface.dim(k)
        TensorInterface.dim(α) == r + 1 || error("response_operator: auxiliary dimension mismatch on $e")
        _register!(RESP_AUX_BONDS, α, only(virtualinds(ψ, e)))
        if Tensors.isgraded(su)
            # graded: the auxiliary index is trivial ⊕ k (`_graded_edge_factors`); the source holds
            # gen's copy, the sink its dual. Direct sums onto those exact indices.
            αu = only(filter(i -> i == α, collect(TensorInterface.inds(gen.value[u]))))
            αw = only(filter(i -> i == α, collect(TensorInterface.inds(gen.value[w]))))
            α0 = Tensors.trivial_link_index(su; tags = "aux")
            kA = only(filter(i -> i == k, collect(TensorInterface.inds(A))))
            kB = only(filter(i -> i == k, collect(TensorInterface.inds(B))))
            α0A = Tensors.isdual(kA) ? TensorInterface.dag(α0) : α0
            α0B = Tensors.isdual(kB) ? TensorInterface.dag(α0) : α0
            z(t) = TensorInterface.scale!(copy(t), zero(elt))
            L0w = TensorInterface.op("I", su) * TensorInterface.onehot(elt, α0A => 1)
            R0w = TensorInterface.op("I", sw) * TensorInterface.onehot(elt, α0B => 1)
            push!(c10[u], TensorInterface.directsum([αu], z(L0w) => [α0A], A => [kA]))
            push!(c01[w], TensorInterface.directsum([αw], z(R0w) => [α0B], B => [kB]))
        else
            d, dw = TensorInterface.dim(su), TensorInterface.dim(sw)
            Aarr = TensorInterface.array(A, su, TensorInterface.prime(su), k)
            Barr = TensorInterface.array(B, sw, TensorInterface.prime(sw), k)
            Lμ = zeros(elt, d, d, r + 1); Lμ[:, :, 2:end] .= Aarr
            Rν = zeros(elt, dw, dw, r + 1); Rν[:, :, 2:end] .= Barr
            push!(c10[u], TensorInterface.from_array(Lμ, su, TensorInterface.prime(su), α))
            push!(c01[w], TensorInterface.from_array(Rν, sw, TensorInterface.prime(sw), α))
        end
    end
    out = Dictionary{eltype(vs), PolyT}()
    for v in vs
        sv = only(s[v])
        set!(out, v, PolyT(TensorInterface.op("I", sv), c10[v], c01[v], get(vertex_terms, v, nothing)))
    end
    return out
end

# ── polynomial environments ──────────────────────────────────────────────────────────────
struct RespEnv
    C::Dict{Tuple{Symbol, Int, Int}, PolyT}
    T::Dict{Tuple{Symbol, Int, Int}, PolyT}
    PH::Dict{Tuple{Symbol, Int, Int}, Any}      # key => isometry bundle (see `_resp_projector`)
    PV::Dict{Tuple{Symbol, Int, Int}, Any}
    Lx::Int
    Ly::Int
end
_rnn(d, k) = get(d, k, nothing)
_rfacs(tbl, x, y) = get(tbl, (x, y), PolyT[])

# enlarged corner, mirroring `_ctm_enlarged`
function _resp_enlarged(S::RespEnv, tbl, sym::Symbol, x::Int, y::Int, opts::CTMOptions)
    grow(blocks, facs) = (l = _plist(blocks..., facs); isempty(l) ? nothing : pcontract(l, opts))
    if sym === :NW
        return grow((_rnn(S.C, (:NW, x - 1, y - 1)), _rnn(S.T, (:N, x - 1, y - 1)), _rnn(S.T, (:W, x - 1, y - 1))), _rfacs(tbl, x - 1, y - 1))
    elseif sym === :NE
        return grow((_rnn(S.C, (:NE, x + 1, y - 1)), _rnn(S.T, (:N, x, y - 1)), _rnn(S.T, (:E, x + 1, y - 1))), _rfacs(tbl, x, y - 1))
    elseif sym === :SW
        return grow((_rnn(S.C, (:SW, x - 1, y + 1)), _rnn(S.T, (:S, x - 1, y + 1)), _rnn(S.T, (:W, x - 1, y))), _rfacs(tbl, x - 1, y))
    else
        return grow((_rnn(S.C, (:SE, x + 1, y + 1)), _rnn(S.T, (:S, x, y + 1)), _rnn(S.T, (:E, x + 1, y))), _rfacs(tbl, x, y))
    end
end

# ── the seam isometries ──────────────────────────────────────────────────────────────────
_ndim(is) = isempty(is) ? 1 : prod(TensorInterface.dim(i) for i in is)
_mat(t, rows::Vector, cols::Vector) = reshape(TensorInterface.array(t, rows..., cols...), _ndim(rows), _ndim(cols))
_tens(M::AbstractMatrix, rows::Vector, cols::Vector) =
    TensorInterface.from_array(reshape(M, (TensorInterface.dim(i) for i in vcat(rows, cols))...), vcat(rows, cols)...)
_others(t, legs) = collect(uniqueinds(t, collect(Index, legs)))
# Divide-and-conquer (`gesdd`) failed to converge on a complex first-order stack (5×5 Heisenberg
# D = 3, χ = 32); fall back to QR iteration, and give up on those directions rather than the solve.
function _robust_svd(M::AbstractMatrix)
    all(isfinite, M) || return nothing
    try
        return svd(M)
    catch err
        err isa InterruptException && rethrow()
        try
            return svd(M; alg = LinearAlgebra.QRIteration())
        catch err2
            err2 isa InterruptException && rethrow()
            return nothing
        end
    end
end
# The seam-leg SIGNATURE of a tensor at an interface whose norm seam legs are `ins`: its legs that
# belong to the seam — a norm seam leg, an auxiliary leg on a seam bond, an excited bond whose
# family norm bond is on the seam. Everything else is an outer leg (an outer norm leg, an auxiliary
# leg on a free bond, the excited version of an outer link). Terms with the same signature share one
# isometry (a SECTOR of the seam); the norm blocks' signature is `ins` itself (sector A).
function _sig(t, ins)
    sig = Index[]
    for i in TensorInterface.inds(t)
        if any(j -> i == j, ins)
            push!(sig, i)
        elseif _isaux(i)
            b = _bond_of(i); b === nothing || !any(j -> b == j, ins) || push!(sig, i)
        elseif _isexc(i)
            f = _family(i); f === nothing || !any(j -> f == j, ins) || push!(sig, i)
        end
    end
    return sig
end
# the copy of index `j` carried by any tensor of the polynomial `p`, or nothing
function _copy_on(p::PolyT, j)
    for t in vcat(p.c00 === nothing ? Any[] : Any[p.c00], p.c10, p.c01, p.c11 === nothing ? Any[] : Any[p.c11])
        for i in TensorInterface.inds(t)
            i == j && return i
        end
    end
    return nothing
end
# the term's own copies of the sector's canonical legs, in canonical order
_cols(t, canon) = [only(filter(i -> i == c, collect(TensorInterface.inds(t)))) for c in canon]

# The isometry bundle of one interface: `(ins, sectors = [(sig, PA, PB, w), …], iA)`, `sig` the
# sector's seam legs as Ba's copies, `P_A` on their duals plus `w` (it contracts Ba-side blocks),
# `P_B` on the copies themselves plus `dag(w)`. Sector A (index `iA`) compresses the norm content of
# both corners (`χ` directions) plus their first/second-order content with the norm signature (`χ1`);
# every other sector — an auxiliary leg on a seam bond, or an excited link — compresses the terms
# that carry it (`χ1`), onto a bond tagged "exc" registered with its family norm bond.
function _resp_projector(Ba::PolyT, Bb::PolyT, ins::Vector{<:Index}, χ::Integer, opts::CTMOptions; χ1::Integer = χ, register::Bool = true)
    (Ba.c00 === nothing || Bb.c00 === nothing || isempty(ins)) && return nothing
    (isempty(_others(Ba.c00, ins)) || isempty(_others(Bb.c00, ins))) && return nothing
    insA = [something(_copy_on(Ba, j), j) for j in ins]           # Ba's copies
    graded = Tensors.isgraded(Ba.c00) || Tensors.isgraded(Bb.c00)
    elt = promote_type(scalartype(Ba.c00), scalartype(Bb.c00))
    # group the first-order terms of both sides by signature (canonical legs: Ba's copies when a Ba
    # term has the sector, else the duals of Bb's)
    groups = Any[]                                                # [(canon, terms)]
    for (p, isBa) in ((Ba, true), (Bb, false)), t in vcat(p.c10, p.c01)
        n = norm(t); (iszero(n) || !isfinite(n)) && continue
        s = _sig(t, insA)
        k = findfirst(g -> _seteq(g[1], s), groups)
        if k === nothing
            push!(groups, (isBa ? s : TensorInterface.dag.(s), Any[t / n]))
        else
            push!(groups[k][2], t / n)
        end
    end
    kA = findfirst(g -> _seteq(g[1], insA), groups)
    termsA = kA === nothing ? Any[] : groups[kA][2]
    for p in (Ba, Bb)
        p.c11 === nothing && continue
        n = norm(p.c11); (iszero(n) || !isfinite(n)) || push!(termsA, p.c11 / n)
    end
    sectors = Any[]
    if graded
        secA = _sector_graded(insA, Any[Ba.c00, Bb.c00], termsA, χ, χ1, opts, "Link,resp")
        secA === nothing && return nothing
        push!(sectors, secA); wA = secA[4]
        for (k, g) in enumerate(groups)
            (k == kA || χ1 <= 0) && continue
            sec = _sector_graded(g[1], Any[], g[2], 0, χ1, opts, "Link,resp,exc")
            sec === nothing && continue
            register && _register!(RESP_LINK_FAMILY, sec[4], wA)
            push!(sectors, sec)
        end
        return (ins = insA, sectors = sectors, iA = 1)
    end
    secA = _sector_dense(insA, Any[Ba.c00, Bb.c00], termsA, χ, χ1, opts, elt)
    secA === nothing && return nothing
    wA = TensorInterface.new_index(size(secA[2], 2); tags = "Link,resp")
    push!(sectors, (secA[1], _sector_tensors(secA[1], secA[2], wA)..., wA))
    # every excited sector onto ONE shared excited bond (see `_sector_tensors`)
    excs = Any[]
    for (k, g) in enumerate(groups)
        (k == kA || χ1 <= 0) && continue
        sec = _sector_dense(g[1], Any[], g[2], 0, χ1, opts, elt)
        sec === nothing || push!(excs, sec)
    end
    if !isempty(excs)
        K = sum(size(sec[2], 2) for sec in excs)
        wE = TensorInterface.new_index(K; tags = "Link,resp,exc")
        register && _register!(RESP_LINK_FAMILY, wE, wA)
        off = 0
        for (canon, V) in excs
            push!(sectors, (canon, _sector_tensors(canon, V, wE, off)..., wE))
            off += size(V, 2)
        end
    end
    return (ins = insA, sectors = sectors, iA = 1)
end
# One sector's isometry, dense: the leading eigenvectors of the base content's Gram matrix (`χ`),
# then the leading right-singular vectors of the (unit-normalised) terms projected off them (`χ1`).
function _sector_dense(canon, bases, terms, χ, χ1, opts, elt)
    isempty(bases) && isempty(terms) && return nothing
    V = nothing
    if !isempty(bases)
        ρ = nothing
        for t in bases
            M = _mat(t, _others(t, canon), _cols(t, canon))
            ρ = ρ === nothing ? M' * M : ρ + M' * M
        end
        F = eigen(Hermitian(ρ))
        s = sqrt.(max.(F.values, zero(real(elt)))); ord = sortperm(s; rev = true); smax = s[ord[1]]
        k0 = smax > 0 ? min(Int(χ), count(x -> x > opts.qr_cutoff * smax, s)) : 0
        k0 == 0 && return nothing
        V = F.vectors[:, ord[1:k0]]
    end
    if !isempty(terms) && χ1 > 0
        # Each term contributes its column space over the seam legs; its row space is irrelevant, so
        # a term with many more rows than the directions we can keep is SKETCHED to 2χ1 + 8 random
        # row combinations first (exact for a column rank ≤ 2χ1 + 8, the kept rank is ≤ χ1; measured
        # 5×5 D = 3 χ = 32: the stacked SVDs were the bulk of a 260 s sweep).
        nsk = 2 * Int(χ1) + 8
        rng = Random.MersenneTwister(0x5e7)           # FIXED sketch: a fresh draw every sweep made
        function sketch(t)                            # the isometries jitter and the solve never settled
            M = _mat(t, _others(t, canon), _cols(t, canon))
            size(M, 1) <= nsk && return M
            return (randn(rng, elt, nsk, size(M, 1)) / sqrt(size(M, 1))) * M
        end
        M1 = vcat((sketch(t) for t in terms)...)
        scale = norm(M1)                                  # BEFORE projecting: the residual may be ~0
        V === nothing || (M1 = M1 - (M1 * V) * V')
        F1 = _robust_svd(M1)
        if F1 !== nothing
            # relative to the unprojected content — relative to the residual's own top value, a
            # numerically zero residual admitted junk columns and V was no isometry (measured)
            k1 = min(Int(χ1), count(x -> x > 1.0e-10 * max(scale, eps(real(elt))), F1.S))
            k1 > 0 && (V = V === nothing ? F1.V[:, 1:k1] : hcat(V, F1.V[:, 1:k1]))
        end
    end
    V === nothing && return nothing
    return (canon, V)
end
# (P_A, P_B) of a sector from its isometry matrix `V` on the legs `canon`, onto the bond `w` (whose
# dimension may exceed size(V, 2): the sector then occupies the columns `off+1:off+size(V, 2)` of a
# bond SHARED by several sectors — every excited sector of a seam maps into one excited bond, so
# terms that left through different sectors carry the same bond and merge downstream; without this
# the sectors, and with them the terms per block and the sweep time, grew with every sweep)
function _sector_tensors(canon, V, w, off::Int = 0)
    K = TensorInterface.dim(w)
    Vp = zeros(eltype(V), size(V, 1), K); Vp[:, (off + 1):(off + size(V, 2))] .= V
    return _tens(Vp, canon, [w]), _tens(Matrix(Vp'), [w], canon)
end
# The same on a graded backend, tensor-level: Gram tensors `gram(X, X, outer) = dag(X, outer)·X`
# with the MAP adjoint of rule 5 (their legs are the sector legs and their duals, whichever side
# they come from), leading eigenvectors off a truncated `svd` (Hermitian PSD), the two budgets
# joined by `directsum` and re-orthonormalised by one more `svd`. `P_A` sits on the duals of `canon`.
function _sector_graded(canon, bases, terms, χ, χ1, opts, tags)
    isempty(bases) && isempty(terms) && return nothing
    dcanon = TensorInterface.dag.(canon)
    gr(X) = TensorInterface.gram(X, X, _others(X, canon))
    V = nothing
    if !isempty(bases)
        ρ = nothing
        for t in bases
            g = gr(t); ρ = ρ === nothing ? g : ρ + g
        end
        V, _, _ = svd(ρ, dcanon; maxdim = Int(χ), cutoff = opts.qr_cutoff^2)
        TensorInterface.dim(only(uniqueinds(V, dcanon))) == 0 && return nothing
    end
    if !isempty(terms) && χ1 > 0
        ρ1 = nothing
        for t in terms
            g = gr(t); ρ1 = ρ1 === nothing ? g : ρ1 + g
        end
        V1, _, _ = svd(ρ1, dcanon; maxdim = Int(χ1), cutoff = 1.0e-20)
        if TensorInterface.dim(only(uniqueinds(V1, dcanon))) > 0
            if V === nothing
                V = V1
            else
                W = TensorInterface.directsum(V => [only(uniqueinds(V, dcanon))], V1 => [only(uniqueinds(V1, dcanon))]; tags = "Link,resp")
                V, _, _ = svd(W, dcanon; cutoff = 1.0e-24)
            end
        end
    end
    V === nothing && return nothing
    w0 = only(uniqueinds(V, dcanon))
    w = Tensors._fresh_like(w0, tags)
    V = TensorInterface.replaceind(V, w0, w)
    return (canon, V, TensorInterface.dag(V, dcanon), w)
end

# Apply an isometry bundle to a polynomial block: `side = 1` the P_A tensors (Ba-side blocks),
# `side = 2` the P_B tensors. The norm and closed coefficients go through sector A; a first-order
# term through the sector of its signature, and is dropped if the seam kept no such sector.
function papply(p::Union{PolyT, Nothing}, P, side::Int)
    (p === nothing || P === nothing) && return p
    PA = P.sectors[P.iA][side + 1]
    function appx(t)
        s = _sig(t, P.ins)
        k = findfirst(sec -> _seteq(sec[1], s), P.sectors)
        return k === nothing ? nothing : t * P.sectors[k][side + 1]
    end
    c10 = Any[u for u in (appx(t) for t in p.c10) if u !== nothing]
    c01 = Any[u for u in (appx(t) for t in p.c01) if u !== nothing]
    return _finalize(PolyT(p.c00 === nothing ? nothing : p.c00 * PA, _merge(_prune(c10)), _merge(_prune(c01)),
                           p.c11 === nothing ? nothing : p.c11 * PA))
end

# The frozen bundle moved onto this sweep's seam legs: every link leg of a sector (tag "resp") is
# stale and is replaced through `relabel` (old bond ⇒ new bond, filled by the lower interfaces of
# the chain this sweep — `_ctm_each_interface` walks each chain from the lattice edge inward); the
# output bonds are re-minted, registered and recorded. `nothing` when a stale leg is unknown (the
# caller re-derives the bundle).
function _resp_refresh(P, Ba::PolyT, relabel)
    islink(i) = occursin("resp", string(TensorInterface.tags(i)))
    swap(t, o, nw) = (k = findfirst(i -> i == o, collect(TensorInterface.inds(t)));
                      k === nothing ? t : TensorInterface.replaceind(t, collect(TensorInterface.inds(t))[k], nw))
    newsecs = Any[]
    wA = nothing
    minted = Any[]                                        # old output bond ⇒ new (shared bonds once)
    for (k, (sig, PA, PB, w)) in enumerate(P.sectors)
        nsig = Index[]
        for i in sig
            if islink(i)
                nw = _lookup(relabel, i); nw === nothing && return nothing
                c = _copy_on(Ba, nw); c === nothing && return nothing     # Ba's copy of the new link
                PA = swap(PA, i, TensorInterface.dag(c)); PB = swap(PB, i, c)
                push!(nsig, c)
            else
                push!(nsig, i)
            end
        end
        j = findfirst(m -> m[1] == w, minted)
        wn = j === nothing ? Tensors._fresh_like(w, k == P.iA ? "Link,resp" : "Link,resp,exc") : minted[j][2]
        j === nothing && (push!(minted, (w, wn)); _register!(relabel, w, wn))
        PA = swap(PA, w, wn); PB = swap(PB, w, TensorInterface.dag(wn))
        k == P.iA && (wA = wn)
        push!(newsecs, (nsig, PA, PB, wn))
    end
    for (k, sec) in enumerate(newsecs)
        k == P.iA || _register!(RESP_LINK_FAMILY, sec[4], wA)
    end
    return (ins = newsecs[P.iA][1], sectors = newsecs, iA = P.iA)
end

# ── one polynomial sweep (the engine's sweep over PolyT) ────────────────────────────────
# `frozen = (PH, PV)`: reuse the given isometries (relabelled by `_resp_refresh`) instead of deriving
# new ones. With them fixed the sweep is a fixed linear map of the polynomial environment, so the
# response converges geometrically — re-deriving them every sweep moved the kept space with the
# coefficients it is chosen from and the iteration wandered (measured 4×4 D = 3: 30 sweeps without
# reaching 1e-8 at every χ).
const RESP_PROF = Dict{Symbol, Float64}()    # accumulated seconds per sweep phase (diagnostics)
_prof!(k, t) = (RESP_PROF[k] = get(RESP_PROF, k, 0.0) + t; nothing)
function _resp_sweep(S::RespEnv, tbl, χ::Integer, opts::CTMOptions; χ1::Integer = χ, frozen = nothing)
    Lx, Ly = S.Lx, S.Ly
    # The three independent phases are threaded like the engine's sweep (enlarged corners, corner
    # rebuilds, edge rebuilds); the projector derivation stays serial because it writes the link
    # registries, which the other phases only read. Measured 5×5 D = 3 χ = 32, one thread: the
    # enlarged corners were 192 s of a 240 s sweep.
    enl = Dict{Tuple{Symbol, Int, Int}, Any}()
    t0 = time()
    let ks = [(sym, x, y) for sym in (:NW, :NE, :SW, :SE) for x in 2:Lx for y in 2:Ly]
        vals = Vector{Any}(undef, length(ks))
        Threads.@threads for i in eachindex(ks)
            sym, x, y = ks[i]
            vals[i] = _resp_enlarged(S, tbl, sym, x, y, opts)
        end
        for i in eachindex(ks)
            enl[ks[i]] = vals[i]
        end
    end
    _prof!(:enlarged, time() - t0); t0 = time()
    E(sym, x, y) = get(enl, (sym, x, y), nothing)
    PH = Dict{Tuple{Symbol, Int, Int}, Any}(); PV = Dict{Tuple{Symbol, Int, Int}, Any}()
    relabel = Dict{UInt64, Any}()                         # old link ⇒ new link, this sweep (frozen mode)
    if frozen !== nothing
        # frozen: the chain order matters (a lower interface's relabel feeds the one above) — serial
        _ctm_each_interface(Lx, Ly) do isH, key, below, ca, cb
            Ba = E(ca...); Bb = E(cb...)
            (Ba === nothing || Bb === nothing) && return nothing
            ins = collect(commoninds(Ba.c00, Bb.c00))
            old = _rnn(isH ? frozen[1] : frozen[2], key)
            pr = old === nothing ? nothing : _resp_refresh(old, Ba, relabel)
            pr === nothing && (pr = _resp_projector(Ba, Bb, ins, χ, opts; χ1))
            pr === nothing || ((isH ? PH : PV)[key] = pr)
            return nothing
        end
    else
        # derive: interfaces are independent given the enlarged corners — threaded; the registry
        # of new excited bonds is written afterwards, serially (the derivation only reads it)
        items = Tuple{Bool, Tuple{Symbol, Int, Int}, Tuple{Symbol, Int, Int}, Tuple{Symbol, Int, Int}}[]
        _ctm_each_interface(Lx, Ly) do isH, key, below, ca, cb
            push!(items, (isH, key, ca, cb))
        end
        res = Vector{Any}(nothing, length(items))
        Threads.@threads for i in eachindex(items)
            isH, key, ca, cb = items[i]
            Ba = E(ca...); Bb = E(cb...)
            (Ba === nothing || Bb === nothing) && continue
            ins = collect(commoninds(Ba.c00, Bb.c00))
            res[i] = _resp_projector(Ba, Bb, ins, χ, opts; χ1, register = false)
        end
        for i in eachindex(items)
            pr = res[i]; pr === nothing && continue
            isH, key = items[i][1], items[i][2]
            wA = pr.sectors[pr.iA][4]
            for (k, sec) in enumerate(pr.sectors)
                k == pr.iA || _register!(RESP_LINK_FAMILY, sec[4], wA)
            end
            (isH ? PH : PV)[key] = pr
        end
    end
    _prof!(:projectors, time() - t0); t0 = time()
    apA(t, pr) = papply(t, pr, 1)
    apB(t, pr) = papply(t, pr, 2)
    C = Dict{Tuple{Symbol, Int, Int}, PolyT}(); T = Dict{Tuple{Symbol, Int, Int}, PolyT}()
    cwork = Tuple{Tuple{Symbol, Int, Int}, Function}[]
    for (sym, hfam, hA, vfam, vA) in ((:NW, :N, true, :W, true), (:NE, :N, false, :E, true),
                                      (:SW, :S, true, :W, false), (:SE, :S, false, :E, false))
        for x in 2:Lx, y in 2:Ly
            push!(cwork, ((sym, x, y), () -> begin
                t = (hA ? apA : apB)(E(sym, x, y), _rnn(PH, (hfam, x - 1, y)))
                prescale((vA ? apA : apB)(t, _rnn(PV, (vfam, x, y - 1))))
            end))
        end
    end
    cres = Vector{Any}(undef, length(cwork))
    Threads.@threads for i in eachindex(cwork)
        cres[i] = cwork[i][2]()
    end
    for i in eachindex(cwork)
        cres[i] === nothing || (C[cwork[i][1]] = cres[i])
    end
    _prof!(:corners, time() - t0); t0 = time()
    function edge(block, facs, pB, pA)
        l = _plist(block, facs); isempty(l) && return nothing
        return prescale(apA(apB(pcontract(l, opts), pB), pA))
    end
    twork = Tuple{Tuple{Symbol, Int, Int}, Function}[]
    for x in 1:Lx, y in 2:Ly
        push!(twork, ((:N, x, y), () -> edge(_rnn(S.T, (:N, x, y - 1)), _rfacs(tbl, x, y - 1), _rnn(PH, (:N, x - 1, y)), _rnn(PH, (:N, x, y)))))
        push!(twork, ((:S, x, y), () -> edge(_rnn(S.T, (:S, x, y + 1)), _rfacs(tbl, x, y), _rnn(PH, (:S, x - 1, y)), _rnn(PH, (:S, x, y)))))
    end
    for x in 2:Lx, y in 1:Ly
        push!(twork, ((:W, x, y), () -> edge(_rnn(S.T, (:W, x - 1, y)), _rfacs(tbl, x - 1, y), _rnn(PV, (:W, x, y - 1)), _rnn(PV, (:W, x, y)))))
    end
    for x in 1:(Lx - 1), y in 1:Ly
        push!(twork, ((:E, x + 1, y), () -> edge(_rnn(S.T, (:E, x + 2, y)), _rfacs(tbl, x + 1, y), _rnn(PV, (:E, x + 1, y - 1)), _rnn(PV, (:E, x + 1, y)))))
    end
    tres = Vector{Any}(undef, length(twork))
    Threads.@threads for i in eachindex(twork)
        tres[i] = twork[i][2]()
    end
    for i in eachindex(twork)
        tres[i] === nothing || (T[twork[i][1]] = tres[i])
    end
    _prof!(:edges, time() - t0)
    return RespEnv(C, T, PH, PV, Lx, Ly)
end

# ── free energy and its (1,1) coefficient ───────────────────────────────────────────────
_rfetch(S::RespEnv, d) = (kind = d[1]; _rnn(kind === :C ? S.C : S.T, (d[2], d[3], d[4])))
function _resp_region_blocks(S::RespEnv, cx::Real, cy::Real)
    ds, _, _ = _ctm_region_desc(cx, cy)
    return PolyT[p for p in (_rfetch(S, d) for d in ds) if p !== nothing]
end
# (F⁰, F^{μν}) with F^{μν} = Σ_regions w_r z^{μν}/z⁰ (a lone half-insertion never closes: z^μ = z^ν = 0)
function _resp_freeenergy(S::RespEnv, tbl, opts::CTMOptions)
    F0 = 0.0; F11 = 0.0
    for cx in 1.0:0.5:S.Lx, cy in 1.0:0.5:S.Ly
        ps = _resp_region_blocks(S, cx, cy)
        if isinteger(cx) && isinteger(cy)
            append!(ps, _rfacs(tbl, Int(cx), Int(cy)))
        end
        isempty(ps) && continue
        z = pcontract(ps, opts)
        (z === nothing || z.c00 === nothing) && continue
        sc(t) = t === nothing ? zero(ComplexF64) : ComplexF64(scalar(t))
        z0 = sc(z.c00)
        (iszero(z0) || !isfinite(abs(z0))) && continue
        w = _ctm_region_desc(cx, cy)[2]
        F0 += w * log(abs(z0))
        F11 += w * real(sc(z.c11) / z0)
    end
    return F0, F11
end

# ── the solve ────────────────────────────────────────────────────────────────────────────
struct ResponseResult
    cache::CTMEnvironmentCache        # the aux-free padded norm environment the solve started from
    env::RespEnv                      # the polynomial environment at the fixed point
    Gs::Dictionary                    # per-vertex operator polynomials
    tbl::Dict{Tuple{Int, Int}, Vector{PolyT}}
    F0::Float64                       # CVM free energy of the norm network
    energy::Float64                   # ∂μ∂ν F = the response energy
    nsweeps::Int
    history::Vector{Float64}
end

"""
    response_solve(ψ, H, χ; projector = :cut, seed = nothing, χ1 = χ, nfree = 3, maxsweeps = 40,
                   tol = 1e-9, verbose = false, ctm_kwargs...) -> ResponseResult

The response energy ∂μ∂ν F|₀ of the (μ, ν) generating network (header of this file) and the
polynomial environment it is read from: the norm environment (`generating_cache(…; aux_free = true)`,
warm-started from `seed`) converged to the isometric sweep's fixed point, then the polynomial
iteration — `nfree` sweeps deriving the seam isometries, the rest with them frozen — until the
energy is stationary to `tol`. `χ1` is the seam budget of the response directions.
"""
function response_solve(ψ::TensorNetworkState, H::Vector, χ::Integer; projector::Symbol = :cut, seed = nothing,
                        maxsweeps::Int = 40, tol::Real = 1.0e-9, verbose::Bool = false, gen = nothing,
                        Gs = nothing, χ1::Integer = χ, nfree::Int = 3, ctm_kwargs...)
    projector === :cut || error("response_solve: only projector = :cut is implemented")
    gen = gen === nothing ? generating_operator(H, ψ) : gen
    Gs = Gs === nothing ? response_operator(H, ψ, gen) : Gs
    cache = generating_cache(ψ, gen, χ; projector, aux_free = true, seed, ctm_kwargs...)
    return response_solve(cache, ψ, Gs; maxsweeps, tol, verbose, χ1, nfree)
end
function response_solve(cache::CTMEnvironmentCache, ψ::TensorNetworkState, Gs::Dictionary;
                        maxsweeps::Int = 40, tol::Real = 1.0e-9, verbose::Bool = false, χ1::Integer = cache.maxdim,
                        nfree::Int = 3)
    opts = options(cache); χ = cache.maxdim
    Lx, Ly = _ctm_dims(cache)
    empty!(RESP_LINK_FAMILY)                       # the excited bonds of a previous solve are gone
    tbl = Dict{Tuple{Int, Int}, Vector{PolyT}}()
    for ((x, y), v) in cache.grid
        t = ψ[v]
        tbl[(x, y)] = PolyT[pconst(t), Gs[v], pconst(unprime_charge_legs(dag(prime(t)), t))]
    end
    # the base: the padded norm environment with its |0⟩ auxiliary legs stripped again
    env0 = environments(cache)
    S = RespEnv(Dict(k => pconst(_strip_all_aux(t)) for (k, t) in env0.C if t !== nothing),
                Dict(k => pconst(_strip_all_aux(t)) for (k, t) in env0.T if t !== nothing),
                Dict{Tuple{Symbol, Int, Int}, Any}(), Dict{Tuple{Symbol, Int, Int}, Any}(), Lx, Ly)
    # Phase 1: the `:cut` base is not a fixed point of the ISOMETRIC sweep, and the linear response
    # converges only once its base is stationary. Converge the base with coefficient-free sweeps
    # (a plain norm sweep) until F⁰ is stationary.
    tbl0 = Dict(k => PolyT[pconst(p.c00) for p in ps] for (k, ps) in tbl)
    F0 = NaN
    for k in 1:maxsweeps
        S = _resp_sweep(S, tbl0, χ, opts; χ1 = 0)
        F0n = _resp_freeenergy(S, tbl0, opts)[1]
        verbose && (println("response base sweep $k: F⁰ = $F0n  ΔF⁰ = $(F0n - F0)"); flush(stdout))
        done = isfinite(F0) && abs(F0n - F0) <= 1.0e-12 * max(1.0, abs(F0n))
        F0 = F0n
        done && break
    end
    # Phase 2: `nfree` polynomial sweeps deriving the sector isometries from the current
    # coefficients, then frozen (linear) sweeps to `tol`.
    E = NaN; hist = Float64[]; n = 0
    for k in 1:maxsweeps
        n = k
        S = _resp_sweep(S, tbl, χ, opts; χ1, frozen = k <= nfree ? nothing : (S.PH, S.PV))
        F0n, En = _resp_freeenergy(S, tbl, opts)
        push!(hist, En)
        verbose && (println("response sweep $k$(k <= nfree ? "" : " (frozen)"): F⁰ = $F0n  E = $En  ΔE = $(En - E)"); flush(stdout))
        done = k > nfree && isfinite(E) && abs(En - E) <= tol * max(1.0, abs(En))
        F0, E = F0n, En
        done && break
    end
    return ResponseResult(cache, S, Gs, tbl, F0, E, n, hist)
end

# ── effective operators from the polynomial rings ────────────────────────────────────────
"""
    response_effective_operators(R::ResponseResult, v) -> (N_eff, H_eff)

Dense `N_eff = (ring·G)⁰` and `H_eff = (ring·G)^{μν}` at vertex `v` in the basis of `ψ[v]`'s indices,
from the polynomial 4C+4T ring around `v`. Both symmetrised; `H_eff` carries an arbitrary multiple
of `N_eff` (the block rescaling), as the finite-difference operators do.
"""
# All vertices at once, threaded (the rings are independent): the driver's per-vertex loop was
# serial and the 25 rings of a 5×5 cost 160 s per evaluation.
function response_effective_operators(R::ResponseResult)
    vs = collect(vertices(ket(network(R.cache))))
    out = Vector{Any}(undef, length(vs))
    Threads.@threads for i in eachindex(vs)
        out[i] = response_effective_operators(R, vs[i])
    end
    return Dict(vs[i] => out[i] for i in eachindex(vs))
end
function response_effective_operators(R::ResponseResult, v)
    cache = R.cache; opts = options(cache)
    x, y = cache.coords[v]
    ring = _resp_region_blocks(R.env, x, y)
    ψv = ket(network(cache))[v]
    is = collect(inds(ψv)); n = prod(TensorInterface.dim.(is))
    Ep = pcontract(vcat(ring, [R.Gs[v]]), opts)
    Ep.c00 === nothing && error("response_effective_operators: empty ring at $v")
    elt = scalartype(Ep.c00)
    # graded sites: the flattened array is not the matrix of the quadratic form (parity signs) —
    # the engine's sign-corrected readout, see `_dense_ring_operator`
    mat(t) = t === nothing ? zeros(elt, n, n) :
             any(Tensors.isgraded, is) ? _graded_operator_matrix(t, ψv, elt) :
             reshape(TensorInterface.array(t, prime.(is)..., is...), n, n)
    N = mat(Ep.c00); Hm = mat(Ep.c11)
    return (N + N') / 2, (Hm + Hm') / 2
end
