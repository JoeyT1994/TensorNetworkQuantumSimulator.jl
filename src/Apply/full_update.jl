#=
Full update: apply a two-site gate and truncate the bond to maximise the fidelity of the result
in an environment of the bond. For a gate `o` on the bond (v₁, v₂):
  1. Split ψᵥ = Qᵥ Rᵥ with Qᵥ carrying the outer virtual legs, and contract the environment with
     the Qs, once, into a reduced environment N on the legs (r₁, r₂) of R₁, R₂, after removing its
     overall phase (it is only fixed up to a scalar).
  2. Make N Hermitian and positive semi-definite: N = W Λ W† with negative eigenvalues set to zero.
  3. Gauge fix: with ρᵢ the reduced environment of leg rᵢ (N traced over the other leg), absorb
     Lᵢ = ρᵢ^{1/2} into Rᵢ, so the environment seen by the gauged tensors is close to the identity.
  4. Start from the truncated SVD of the gate applied to the gauged tensors, then sweep: with one
     tensor fixed, the other solves a small positive-definite linear system, densely, with Tikhonov
     regularisation. Stop when the fidelity stops improving.
  5. Re-split with an SVD to balance the bond, undo the gauge, and multiply back the Qs. The bond
     keeps its index if its dimension is unchanged, so a cache built on the state stays valid.

`apply_gates_full_update` draws the environments from CTMRG. The linear algebra is on dense arrays,
so this is for dense (non-graded) backends.
=#

# ψᵥ = Qᵥ Rᵥ with Rᵥ on `keep` (the site and the bond) and a new leg rᵥ. A tensor with no other
# legs gets a trivial Qᵥ on a dimension-one rᵥ.
function split_reduced(ψv::Tensor, keep)
    outer = setdiff(inds(ψv), keep)
    if isempty(outer)
        r = new_index(1; tags = "r")
        one_r = from_array(ones(scalartype(ψv), 1), [r])
        return one_r, ψv * one_r, r
    end
    Q, R = qr(ψv, outer)
    return Q, R, commonind(Q, R)
end

# The bra of a tensor on the reduced legs: conjugated with its virtual legs primed and its site
# legs `sinds` left unprimed, since they are shared with the ket.
reduced_bra(t::Tensor, sinds) = replaceinds(dag(prime(t)), prime.(sinds), sinds)

# Hermitian square root of a positive semi-definite matrix and its pseudo-inverse.
function psd_sqrt_pinv(ρ::AbstractMatrix; rtol)
    λ, U = LinearAlgebra.eigen(LinearAlgebra.Hermitian((ρ + ρ') / 2))
    λmax = maximum(abs, λ)
    keep = λ .> rtol * λmax
    sq = map((l, k) -> k ? sqrt(l) : zero(l), λ, keep)
    isq = map((l, k) -> k ? inv(sqrt(l)) : zero(l), λ, keep)
    return U * LinearAlgebra.Diagonal(sq) * U', U * LinearAlgebra.Diagonal(isq) * U'
end

# Apply the matrix `M` to the leg `r` of `t`, in place of `r`.
function apply_to_leg(M::AbstractMatrix, t::Tensor, r)
    Mt = from_array(Matrix(transpose(M)), [r, prime(r)])
    return replaceinds(t * Mt, [prime(r)], [r])
end

# Contract an environment: in an optimal order for short lists, greedily for longer ones.
function contract_environment(ts::Vector{<:Tensor})
    length(ts) <= 12 && return contract_optimal(ts)
    return contract(ts; sequence = contraction_sequence(ts; alg = "omeinsum", optimizer = GreedyMethod()))
end

# f applied to the eigenvalues of the Hermitian matrix A.
function hermitian_map(f, A)
    λ, W = LinearAlgebra.eigen(LinearAlgebra.Hermitian((A + A') / 2))
    return W * LinearAlgebra.Diagonal(f.(λ)) * W'
end

"""
    full_update(o, ψ₁, ψ₂, envs; maxdim, cutoff = nothing, gauge_fix = true, maxsweeps = 100,
                tol = 1e-12, regularization = 1e-12, gauge_rtol = 1e-12)

Apply the two-site gate `o` to the tensors `ψ₁`, `ψ₂` of neighbouring vertices, truncating their
shared bond to at most `maxdim` (and by `cutoff`, as in `simple_update`) to maximise the fidelity in
the environment `envs` (tensors of the norm network with the bra's virtual legs primed, as returned
by `incoming_messages`).

Returns the new tensors and a named tuple of diagnostics: `fidelity` (in the environment, after the
sweeps) and `initial_fidelity` (of the starting SVD), `negativity` (the norm of the environment's
negative eigenvalues relative to all of them, before they were set to zero), `condition` (ratio of
its largest to smallest kept eigenvalue), `sweeps` and the bond's `singular_values` in the gauged
frame.
"""
function full_update(
        o::Tensor, ψ₁::Tensor, ψ₂::Tensor, envs::Vector{<:Tensor};
        maxdim::Integer, cutoff = nothing, gauge_fix = true, maxsweeps = 100, tol = 1.0e-12,
        regularization = 1.0e-12, gauge_rtol = 1.0e-12,
    )
    b = commonind(ψ₁, ψ₂)
    s₁, s₂ = commonind(ψ₁, o), commonind(ψ₂, o)
    Q₁, R₁, r₁ = split_reduced(ψ₁, [s₁, b])
    Q₂, R₂, r₂ = split_reduced(ψ₂, [s₂, b])
    d₁, d₂ = dim(r₁), dim(r₂)
    Θ = apply(o, R₁ * R₂)
    Θ̄ = reduced_bra(Θ, [s₁, s₂])

    # 1. the reduced environment, rows on the bra legs. It is only fixed up to a scalar, which may
    # be complex or negative (CTMRG corners carry arbitrary phases), so rotate it to a real,
    # positive trace.
    N = contract_environment(vcat(envs, Tensor[Q₁, Q₂, dag(prime(Q₁)), dag(prime(Q₂))]))
    Nm = reshape(array(N, prime(r₁), prime(r₂), r₁, r₂), d₁ * d₂, d₁ * d₂)
    τ = LinearAlgebra.tr(Nm)
    Nm = Nm * (conj(τ) / abs(τ))

    # 2. Hermitian, positive semi-definite, scaled to unit largest eigenvalue
    λ, W = LinearAlgebra.eigen(LinearAlgebra.Hermitian((Nm + Nm') / 2))
    negativity = LinearAlgebra.norm(min.(λ, 0)) / LinearAlgebra.norm(λ)
    λ = max.(λ, 0) ./ maximum(abs, λ)
    kept = λ[λ .> gauge_rtol]
    condition = maximum(kept) / minimum(kept)
    X = W * LinearAlgebra.Diagonal(sqrt.(λ))

    # 3. gauge fix with the square roots of the single-leg reduced environments
    if gauge_fix
        X₃ = reshape(X, d₁, d₂, :)
        ρ₁ = reshape(X₃, d₁, :) * reshape(X₃, d₁, :)'
        ρ₂ = reshape(permutedims(X₃, (2, 1, 3)), d₂, :) * reshape(permutedims(X₃, (2, 1, 3)), d₂, :)'
        L₁, L₁⁺ = psd_sqrt_pinv(ρ₁; rtol = gauge_rtol)
        L₂, L₂⁺ = psd_sqrt_pinv(ρ₂; rtol = gauge_rtol)
        X = kron(L₂⁺, L₁⁺) * X
        Θ = apply_to_leg(L₂, apply_to_leg(L₁, Θ, r₁), r₂)
        Θ̄ = reduced_bra(Θ, [s₁, s₂])
    end
    Ñ = from_array(reshape(X * X', d₁, d₂, d₁, d₂), [prime(r₁), prime(r₂), r₁, r₂])
    nΘ = real(scalar(contract([Θ̄, Ñ, Θ])))

    # fidelity of A B against the target Θ in the environment
    function fidelity(A, B)
        Ā, B̄ = reduced_bra(A, [s₁]), reduced_bra(B, [s₂])
        ov = scalar(contract([Ā, B̄, Ñ, Θ]))
        nAB = real(scalar(contract([Ā, B̄, Ñ, A, B])))
        return abs2(ov) / (nAB * nΘ)
    end

    # 4. start from the truncated SVD, then sweep
    A, B, _ = factorize_svd(Θ, [r₁, s₁]; ortho = "none", maxdim, cutoff, tags = tags(b))
    c = commonind(A, B)
    initial_fidelity = fidelity(A, B)

    # With `other` fixed, the tensor on (r, s, c) that maximises the fidelity: the solution of
    # K x = h with K the environment of x and h its overlap with the target.
    function solve(other, s_other, r, s)
        ō = reduced_bra(other, [s_other])
        n = dim(r) * dim(c)
        Km = reshape(array(contract([ō, Ñ, other]), prime(r), prime(c), r, c), n, n)
        hm = reshape(array(contract([ō, Ñ, Θ]), prime(r), prime(c), s), n, dim(s))
        Km = (Km + Km') / 2
        ε = regularization * real(LinearAlgebra.tr(Km)) / n
        # Km is positive semi-definite up to rounding; if that makes the Cholesky factorisation
        # fail, solve on its eigenvalues above ε instead
        F = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(Km + ε * LinearAlgebra.I); check = false)
        x = LinearAlgebra.issuccess(F) ? F \ hm : hermitian_map(λ -> λ > ε ? inv(λ) : zero(λ), Km) * hm
        return from_array(reshape(x, dim(r), dim(c), dim(s)), [r, c, s])
    end

    f = initial_fidelity
    sweeps = 0
    for sweep in 1:maxsweeps
        A = solve(B, s₂, r₁, s₁)
        B = solve(A, s₁, r₂, s₂)
        f_new = fidelity(A, B)
        sweeps = sweep
        converged = abs(f_new - f) <= tol * abs(f_new)
        f = f_new
        converged && break
    end

    # 5. balance the bond, undo the gauge, restore the outer legs
    singular_values! = Ref{Tensor}()
    A, B, _ = factorize_svd(A * B, [r₁, s₁]; ortho = "none", maxdim, cutoff, tags = tags(b), singular_values!)
    if gauge_fix
        A = apply_to_leg(L₁⁺, A, r₁)
        B = apply_to_leg(L₂⁺, B, r₂)
    end
    # keep the bond's index when its dimension is unchanged, so environments built on it stay valid
    c = commonind(A, B)
    if dim(c) == dim(b)
        A, B = replaceind(A, c, b), replaceind(B, c, b)
    end
    ψ₁′ = noprime(Q₁ * A)
    ψ₂′ = noprime(Q₂ * B)

    diagnostics = (; fidelity = f, initial_fidelity, negativity, condition, sweeps, singular_values = singular_values![])
    return ψ₁′, ψ₂′, diagnostics
end

# A converged CTMRG cache of ψ.
function full_update_cache(ψ; env_maxdim, positions, cache_update_kwargs)
    return update(CTMRGCache(ψ, env_maxdim; positions); cache_update_kwargs...)
end

"""
    apply_gates_full_update(circuit, ψ::TensorNetworkState; maxdim, env_maxdim, cutoff = nothing,
                            normalize_tensors = true, positions, cache_update_kwargs = (; tolerance = 1e-6))

Apply a sequence of one- and two-site gates to `ψ`, two-site gates by `full_update` in the CTMRG
environment of their bond, with the corners truncated to `env_maxdim`. The CTMRG cache is updated,
from its previous state, before any two-site gate on a vertex changed since the last update; so
gates on disjoint bonds (one colour of an edge colouring) share an environment. `circuit` is as for
`apply_gates`. The graph must be planar, with an embedding from `positions` (default: the vertex
names, when they are coordinates).

Returns the new state and the diagnostics of each two-site gate (see `full_update`).
"""
function apply_gates_full_update(circuit::Vector, ψ::TensorNetworkState; kwargs...)
    gates = eltype(circuit) <: Tensor ? [(o, vertices(o, ψ)) for o in circuit] : toitensor(circuit, graph(ψ), siteinds(ψ))
    return full_update_sequence(gates, ψ; kwargs...)
end

# Apply the `(gate, vertices)` in `gates` to `ψ` in order, two-site gates by `full_update`; see
# `apply_gates_full_update`.
function full_update_sequence(
        gates::Vector, ψ::TensorNetworkState;
        maxdim::Integer, env_maxdim::Integer, cutoff = nothing, normalize_tensors = true,
        positions = default_positions(graph(ψ)), cache_update_kwargs = (; tolerance = 1.0e-6),
    )
    g = graph(ψ)
    cache = full_update_cache(ψ; env_maxdim, positions, cache_update_kwargs)
    ψ = network(cache)
    stale = false                 # whether a bond changed dimension, so the cache must be rebuilt
    affected = Set{eltype(vertices(ψ))}()
    diagnostics = NamedTuple[]
    for (o, vs) in gates
        o = adapt_gate(o, ψ)
        if length(vs) == 1
            v = only(vs)
            t = apply(o, ψ[v])
            setindex_preserve!(cache, normalize_tensors ? normalize(t) : t, v)
            push!(affected, v)
            continue
        end
        length(vs) == 2 && has_edge(g, NamedEdge(vs[1] => vs[2])) ||
            error("full update: two-site gates must act on neighbouring vertices; got $vs.")
        if stale
            cache = full_update_cache(network(cache); env_maxdim, positions, cache_update_kwargs)
            stale = false
            empty!(affected)
        elseif any(in(affected), vs)
            cache = update(cache; cache_update_kwargs...)
            empty!(affected)
        end
        ψ = network(cache)
        u, v = vs
        bond = commonind(ψ[u], ψ[v])
        a, b, d = full_update(o, ψ[u], ψ[v], incoming_messages(cache, [u, v]); maxdim, cutoff)
        push!(diagnostics, d)
        setindex_preserve!(cache, normalize_tensors ? normalize(a) : a, u)
        setindex_preserve!(cache, normalize_tensors ? normalize(b) : b, v)
        stale |= commonind(a, b) != bond
        union!(affected, vs)
    end
    return copy(network(cache)), diagnostics
end

adapt_gate(gate::Tensor, ψ::TensorNetworkState) =
    adapt(unspecify_type_parameters(datatype(ψ)), scalartype(gate) <: Complex ? adapt(complex(scalartype(ψ)), gate) : adapt(scalartype(ψ), gate))
