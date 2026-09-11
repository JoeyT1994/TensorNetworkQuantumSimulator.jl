"""
    TensorNetworkOperator{V} <: AbstractTensorNetwork{V}

An operator layer on the graph: one tensor per vertex acting from the unprimed site indices
(the ket's) to their primed copies (the bra's), with virtual "auxiliary" indices shared between
the operator tensors of neighbouring vertices. A product of on-site operators has no auxiliary
indices; the generating operator below has one per graph edge that carries a Hamiltonian term.
"""
struct TensorNetworkOperator{V, T, SI <: Dictionary} <: AbstractTensorNetwork{V}
    tensors::Dictionary{V, T}
    graph::NamedGraph{V}
    siteinds::SI
end

graph(tno::TensorNetworkOperator) = tno.graph
tensors(tno::TensorNetworkOperator) = tno.tensors
siteinds(tno::TensorNetworkOperator) = tno.siteinds
siteinds(tno::TensorNetworkOperator, v) = tno.siteinds[v]
Base.getindex(tno::TensorNetworkOperator, v) = tensors(tno)[v]
Base.copy(tno::TensorNetworkOperator) = TensorNetworkOperator(copy(tensors(tno)), copy(graph(tno)), copy(siteinds(tno)))
function add_tensor!(tno::TensorNetworkOperator, tensor, v)
    set!(tensors(tno), v, tensor)
    return tno
end
bp_factors(tno::TensorNetworkOperator, v) = [tno[v]]
bp_factors(tno::TensorNetworkOperator, vs::Vector) = [tno[v] for v in vs]

"""
    GeneratingOperator{V}

The generating operator `G(λ) = ∏ₑ(1 + λhₑ)∏ᵥ(1 + λhᵥ)` of a local Hamiltonian, stored as the
pair needed at first order: `value = G(0)` and `derivative = ∂λG(0) = Σ h` in the SAME
tensor-network-operator representation (identical auxiliary indices). Each edge term
`hₑ = Σₐ Aₐ ⊗ Bₐ` (operator-Schmidt decomposition) becomes a left factor `Σₐ Aₐ ⊗ |a⟩` on `src(e)`
and a right factor `1 ⊗ |0⟩ + λ Σₐ Bₐ ⊗ |a⟩` on `dst(e)` sharing an auxiliary index of dimension
`r + 1`. The auxiliary index is kept at λ = 0 (where the `a > 0` blocks of the right factor
vanish) so that the belief-propagation messages of the norm network ⟨ψ|G(0)|ψ⟩ contract with
`G(λ)` for every λ. `⟨ψ|G(λ)|ψ⟩ = ⟨ψ|ψ⟩ + λ⟨ψ|H|ψ⟩ + O(λ²)`: the energy is the λ-derivative of the
Bethe estimator at fixed messages (envelope theorem), see `bethe_energy`.
"""
struct GeneratingOperator{V, O <: TensorNetworkOperator{V}}
    value::O
    derivative::O
end

graph(gen::GeneratingOperator) = graph(gen.value)

# ── Hamiltonian terms ────────────────────────────────────────────────────────────────────
# A Hamiltonian is a `Vector` of observable-style tuples: `("ZZ", (v, w), J)`, `("X", [v], h)`,
# `(["X", "Y"], (v, w), c)`, or a joint two-site name such as `("hopping", (v, w), t)`.
function _term_tensor(term::Tuple, g::NamedGraph, s::Dictionary)
    op_strings, verts, coeff = collectobservable(term, g)
    sinds = [only(s[v]) for v in verts]
    t = op_strings isa String ? op(op_strings, sinds...) : prod(op(o, i) for (o, i) in zip(op_strings, sinds))
    return coeff * t, verts
end

# Operator product `a · b` on the site index `s` (both carry `s → s′`); auxiliary legs pass through.
function _opmul(a, b, s::Index)
    s2 = TensorInterface.prime(s, 2)
    ab = a * TensorInterface.replaceinds(b, [s, TensorInterface.prime(s)], [TensorInterface.prime(s), s2])
    return TensorInterface.replaceinds(ab, [s2], [TensorInterface.prime(s)])
end

# Product of vertex factors at λ = 0 and its λ-derivative by the product rule. Each factor is a
# pair `(f0, df)` of tensors on `(s, s′, aux...)`; `df === nothing` for λ-independent factors.
function _vertex_value_and_derivative(factors::Vector, s::Index, elt)
    isempty(factors) && error("vertex without factors")
    value = first(factors)[1]
    for (f0, _) in factors[2:end]
        value = _opmul(value, f0, s)
    end
    deriv = nothing
    for j in eachindex(factors)
        df = factors[j][2]
        df === nothing && continue
        term = j == 1 ? df : factors[1][1]
        for i in 2:length(factors)
            term = _opmul(term, i == j ? df : factors[i][1], s)
        end
        deriv = deriv === nothing ? term : deriv + term
    end
    if deriv === nothing
        deriv = TensorInterface.scale!(copy(value), zero(elt))
    end
    return value, deriv
end

"""
    generating_operator(H::Vector, ψ::TensorNetworkState; cutoff = 1e-14)
    generating_operator(H::Vector, s::Dictionary, g::NamedGraph; cutoff = 1e-14)

Build the [`GeneratingOperator`](@ref) of the Hamiltonian `H` (a vector of observable-style
terms on vertices and edges of `g`) on site indices `s`. Every graph edge carrying terms gets one
auxiliary index of dimension `r + 1`, `r` the operator-Schmidt rank of the summed edge term
(singular values below `cutoff` dropped).
"""
generating_operator(H::Vector, ψ::TensorNetworkState; kwargs...) = generating_operator(H, siteinds(ψ), graph(ψ); kwargs...)
function generating_operator(H::Vector, s::Dictionary, g::NamedGraph; cutoff::Real = 1.0e-14)
    vs = collect(vertices(g))
    all(v -> length(s[v]) == 1, vs) || error("generating_operator: one site index per vertex is required")
    edge_terms = Dict{NamedEdge, Any}()
    vertex_terms = Dict{Any, Any}()
    for term in H
        t, verts = _term_tensor(term, g, s)
        if length(verts) == 1
            v = only(verts)
            vertex_terms[v] = haskey(vertex_terms, v) ? vertex_terms[v] + t : t
        elseif length(verts) == 2
            e = NamedEdge(verts[1] => verts[2])
            has_edge(g, e) || error("generating_operator: term $(term) does not lie on an edge of the graph")
            key = e ∈ edges(g) ? e : reverse(e)
            edge_terms[key] = haskey(edge_terms, key) ? edge_terms[key] + t : t
        else
            error("generating_operator: only one- and two-site terms are supported, got $(term)")
        end
    end
    elt = promote_type(Float64, (eltype(t) for t in values(edge_terms))..., (eltype(t) for t in values(vertex_terms))...)
    # per-vertex factor lists: (value at λ = 0, λ-derivative or nothing)
    factors = Dict(v => Any[] for v in vs)
    for (e, h) in edge_terms
        u, w = src(e), dst(e)
        su, sw = only(s[u]), only(s[w])
        # operator-Schmidt decomposition h = Σₐ Aₐ ⊗ Bₐ, with the singular values split evenly
        A, B, _ = factorize_svd(h, [su, TensorInterface.prime(su)]; ortho = "none", cutoff)
        k = only(TensorInterface.commoninds(A, B))
        r = TensorInterface.dim(k)
        α = TensorInterface.new_index(r + 1; tags = "aux")
        d = TensorInterface.dim(su)
        Aarr = TensorInterface.array(A, su, TensorInterface.prime(su), k)
        Barr = TensorInterface.array(B, sw, TensorInterface.prime(sw), k)
        # left factor: Σₐ Aₐ ⊗ |a⟩ with A₀ = 1 (λ-independent)
        L = zeros(elt, d, d, r + 1)
        for i in 1:d; L[i, i, 1] = one(elt); end
        L[:, :, 2:end] .= Aarr
        # right factor: 1 ⊗ |0⟩ at λ = 0, derivative Σₐ Bₐ ⊗ |a⟩
        R0 = zeros(elt, TensorInterface.dim(sw), TensorInterface.dim(sw), r + 1)
        for i in 1:TensorInterface.dim(sw); R0[i, i, 1] = one(elt); end
        dR = zeros(elt, TensorInterface.dim(sw), TensorInterface.dim(sw), r + 1)
        dR[:, :, 2:end] .= Barr
        push!(factors[u], (TensorInterface.from_array(L, su, TensorInterface.prime(su), α), nothing))
        push!(factors[w], (TensorInterface.from_array(R0, sw, TensorInterface.prime(sw), α),
                           TensorInterface.from_array(dR, sw, TensorInterface.prime(sw), α)))
    end
    for (v, h) in vertex_terms
        sv = only(s[v])
        harr = TensorInterface.array(h, sv, TensorInterface.prime(sv))
        push!(factors[v], (TensorInterface.from_array(Matrix{elt}(LinearAlgebra.I, size(harr)), sv, TensorInterface.prime(sv)),
                           TensorInterface.from_array(Array{elt}(harr), sv, TensorInterface.prime(sv))))
    end
    vals = Any[]; ders = Any[]
    for v in vs
        sv = only(s[v])
        if isempty(factors[v])
            id = TensorInterface.from_array(Matrix{elt}(LinearAlgebra.I, TensorInterface.dim(sv), TensorInterface.dim(sv)), sv, TensorInterface.prime(sv))
            push!(factors[v], (id, nothing))
        end
        val, der = _vertex_value_and_derivative(factors[v], sv, elt)
        push!(vals, val); push!(ders, der)
    end
    T = promote_type(typeof.(vals)..., typeof.(ders)...)
    return GeneratingOperator(TensorNetworkOperator(Dictionary{eltype(vs), T}(vs, vals), copy(g), copy(s)),
                              TensorNetworkOperator(Dictionary{eltype(vs), T}(vs, ders), copy(g), copy(s)))
end
