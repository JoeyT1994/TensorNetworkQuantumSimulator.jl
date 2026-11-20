#=
Full update: apply a two-site gate and truncate the bond to maximise the fidelity of the result
in an environment of the bond. For a gate `o` on the bond (v₁, v₂):
  1. Split ψᵥ = Qᵥ Rᵥ with Qᵥ carrying the outer virtual legs, and contract the environment with
     the Qs into a reduced environment N on the legs (r₁, r₂) of R₁, R₂, with its overall phase
     removed (an environment is only fixed up to a scalar).
  2. Make N Hermitian and positive definite: negative eigenvalues are set to zero, then all of them
     are shifted by `regularization` (relative to the largest). An approximate environment can give
     almost no weight to directions that matter; the shift keeps them in the objective, so the
     truncation error isn't dumped there, where undoing the gauge (step 3) would amplify it.
  3. Gauge fix: absorb Lᵢ = (ρᵢ + δ)^{1/2} into Rᵢ, with ρᵢ the reduced environment of leg rᵢ, so
     the gauged tensors see an environment close to the identity. The shift δ keeps Lᵢ invertible.
  4. Start from the truncated SVD of the gated, gauged tensors, then sweep: with one tensor fixed,
     the other solves a small positive-definite linear system. Stop when the fidelity settles.
  5. Re-split with an SVD to balance the bond, undo the gauge, and multiply back the Qs.

The environment is contracted where the tensors are (on a GPU, if they are there); everything after
it is on small dense matrices on the host. This is for dense (non-graded) backends.
=#

# ψᵥ = Qᵥ Rᵥ with Rᵥ on `keep` (the site and the bond) and a new leg rᵥ. A tensor with no other
# legs gets a trivial Qᵥ on a dimension-one rᵥ.
function split_reduced(ψv::Tensor, keep)
    outer = setdiff(inds(ψv), keep)
    if isempty(outer)
        r = new_index(1; tags = "r")
        one_r = adapt_like(ψv, from_array(ones(scalartype(ψv), 1), [r]))
        return one_r, ψv * one_r, r
    end
    Q, R = qr(ψv, outer)
    return Q, R, commonind(Q, R)
end

# The new bond of A, B given the index `b` of the old one if their dimensions agree, so caches built
# on the old bond stay valid.
function keep_bond_index(A::Tensor, B::Tensor, b)
    c = commonind(A, B)
    dim(c) == dim(b) || return A, B
    return replaceind(A, c, b), replaceind(B, c, b)
end

# The bra of a tensor on the reduced legs: conjugated with its virtual legs primed and its site
# legs `sinds` left unprimed, since they are shared with the ket.
reduced_bra(t::Tensor, sinds) = replaceinds(dag(prime(t)), prime.(sinds), sinds)

# The Hermitian square root of ρ + δ λₘₐₓ, for ρ positive semi-definite with largest eigenvalue
# λₘₐₓ, and its inverse.
function regularized_sqrt(ρ::AbstractMatrix; δ)
    λ, U = LinearAlgebra.eigen(LinearAlgebra.Hermitian((ρ + ρ') / 2))
    λ = max.(λ, 0) .+ δ * maximum(λ)
    return U * LinearAlgebra.Diagonal(sqrt.(λ)) * U', U * LinearAlgebra.Diagonal(inv.(sqrt.(λ))) * U'
end

# Apply the matrix `M` to the leg `r` of `t`, in place of `r`.
function apply_to_leg(M::AbstractMatrix, t::Tensor, r)
    Mt = from_array(Matrix(transpose(M)), [r, prime(r)])
    return replaceinds(t * Mt, [prime(r)], [r])
end

# f applied to the eigenvalues of the Hermitian matrix A.
function hermitian_map(f, A)
    λ, W = LinearAlgebra.eigen(LinearAlgebra.Hermitian((A + A') / 2))
    return W * LinearAlgebra.Diagonal(f.(λ)) * W'
end

# The environment tensors on Q's side of the bond: those sharing a leg with Q, then, until none is
# added, any whose links to the other environment tensors all stay within the side.
function bond_side(envs, Q)
    side = Set(i for (i, t) in enumerate(envs) if hascommoninds(t, Q))
    links(i) = [j for j in eachindex(envs) if j != i && hascommoninds(envs[i], envs[j])]
    grown = true
    while grown
        grown = false
        for i in eachindex(envs)
            i in side && continue
            l = links(i)
            if !isempty(l) && all(in(side), l)
                push!(side, i)
                grown = true
            end
        end
    end
    return sort!(collect(side))
end

# The bond's environment contracted with the Qs of its ends and their conjugates: each end's half
# first, then the halves, so every list is short enough for the exact order search.
function bond_environment(envs, Q₁, Q₂, sequences)
    side₁, side₂ = bond_side(envs, Q₁), bond_side(envs, Q₂)
    if isempty(side₁) || isempty(side₂) || !isempty(intersect(side₁, side₂))
        return contract_memoized(sequences, vcat(envs, Tensor[Q₁, Q₂, dag(prime(Q₁)), dag(prime(Q₂))]))
    end
    half₁ = contract_memoized(sequences, vcat(envs[side₁], Tensor[Q₁, dag(prime(Q₁))]))
    half₂ = contract_memoized(sequences, vcat(envs[side₂], Tensor[Q₂, dag(prime(Q₂))]))
    rest = envs[setdiff(eachindex(envs), side₁, side₂)]
    return contract_memoized(sequences, vcat(rest, Tensor[half₁, half₂]))
end

"""
    full_update(o, ψ₁, ψ₂, envs; maxdim, cutoff = nothing, gauge_fix = true, maxsweeps = 100,
                tol = 1e-8, regularization = 1e-10, gauge_regularization = 1e-8)

Apply the two-site gate `o` to the tensors `ψ₁`, `ψ₂` of neighbouring vertices and truncate their
shared bond to maximise the fidelity of the result in the environment `envs` (tensors of the norm
network with the bra's virtual legs primed, as returned by `incoming_messages`).

# Keyword Arguments
- `maxdim::Integer`: The maximum bond dimension.
- `cutoff`: A singular value cutoff, as in `simple_update`, applied to the starting SVD.
- `gauge_fix`: Whether to gauge the tensors with the square roots of the environment of each leg
  first, which conditions the optimisation.
- `maxsweeps`, `tol`: The sweeps stop after `maxsweeps`, or once the fidelity changes by less than
  `tol` (relative).
- `regularization`: Every eigenvalue of the environment (scaled to a largest of one) is raised by
  this amount, so no direction of the bond drops out of the objective.
- `gauge_regularization`: The same shift for the environments of each leg used by the gauge.

# Returns
The new tensors and a named tuple of diagnostics: `fidelity` (in the environment, after the
sweeps), `initial_fidelity` (of the starting SVD), `negativity` (the norm of the environment's
negative eigenvalues relative to all of them), `nonhermiticity` (the norm of its anti-Hermitian part
relative to its norm), `condition` (the ratio of its largest to smallest eigenvalue, after the
shift), `sweeps` and the bond's `singular_values` in the gauged frame.
"""
function full_update(
        o::Tensor, ψ₁::Tensor, ψ₂::Tensor, envs::Vector{<:Tensor};
        maxdim::Integer, cutoff = nothing, gauge_fix = true, maxsweeps = 100, tol = 1.0e-8,
        regularization = 1.0e-10, gauge_regularization = 1.0e-8, sequences = Dict{Any, Any}(),
    )
    b = commonind(ψ₁, ψ₂)
    s₁, s₂ = commonind(ψ₁, o), commonind(ψ₂, o)
    Q₁, R₁, r₁ = split_reduced(ψ₁, [s₁, b])
    Q₂, R₂, r₂ = split_reduced(ψ₂, [s₂, b])
    d₁, d₂ = dim(r₁), dim(r₂)
    Θ = adapt(Array, apply(o, R₁ * R₂))

    # 1. the reduced environment, rows on the bra legs, rotated to a real positive trace (CTMRG
    # corners carry arbitrary phases)
    N = adapt(Array, bond_environment(envs, Q₁, Q₂, sequences))
    Nm = reshape(array(N, prime(r₁), prime(r₂), r₁, r₂), d₁ * d₂, d₁ * d₂)
    τ = LinearAlgebra.tr(Nm)
    iszero(τ) || (Nm = Nm * (conj(τ) / abs(τ)))
    nonhermiticity = LinearAlgebra.norm(Nm - Nm') / (2 * LinearAlgebra.norm(Nm))

    # 2. Hermitian, positive definite, N = X X†
    λ, W = LinearAlgebra.eigen(LinearAlgebra.Hermitian((Nm + Nm') / 2))
    negativity = LinearAlgebra.norm(min.(λ, 0)) / LinearAlgebra.norm(λ)
    λ = max.(λ, 0) ./ maximum(λ) .+ regularization
    condition = maximum(λ) / minimum(λ)
    X = W * LinearAlgebra.Diagonal(sqrt.(λ))

    # 3. gauge fix
    if gauge_fix
        X₁ = reshape(X, d₁, :)
        X₂ = reshape(permutedims(reshape(X, d₁, d₂, :), (2, 1, 3)), d₂, :)
        L₁, L₁⁻¹ = regularized_sqrt(X₁ * X₁'; δ = gauge_regularization)
        L₂, L₂⁻¹ = regularized_sqrt(X₂ * X₂'; δ = gauge_regularization)
        X = kron(L₂⁻¹, L₁⁻¹) * X
        Θ = apply_to_leg(L₂, apply_to_leg(L₁, Θ, r₁), r₂)
    end
    Θ̄ = reduced_bra(Θ, [s₁, s₂])
    Ñ = from_array(reshape(X * X', d₁, d₂, d₁, d₂), [prime(r₁), prime(r₂), r₁, r₂])
    nΘ = real(scalar(contract([Θ̄, Ñ, Θ])))

    # 4. start from the truncated SVD, then sweep
    A, B, _ = factorize_svd(Θ, [r₁, s₁]; ortho = "none", maxdim, cutoff, tags = tags(b))
    c = commonind(A, B)
    ĀB̄Ñ = contract([reduced_bra(A, [s₁]), reduced_bra(B, [s₂]), Ñ])
    initial_fidelity = abs2(scalar(ĀB̄Ñ * Θ)) / (real(scalar(contract([ĀB̄Ñ, A, B]))) * nΘ)

    # With `other` fixed, the tensor on (r, s, c) that maximises the fidelity, K x = h, and the
    # fidelity of the result
    function solve(other, s_other, r, s)
        ōÑ = reduced_bra(other, [s_other]) * Ñ
        n = dim(r) * dim(c)
        Km = reshape(array(ōÑ * other, prime(r), prime(c), r, c), n, n)
        hm = reshape(array(ōÑ * Θ, prime(r), prime(c), s), n, dim(s))
        Km = (Km + Km') / 2
        ε = regularization * real(LinearAlgebra.tr(Km)) / n
        # if rounding makes the Cholesky factorisation fail, solve on the eigenvalues above ε
        F = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(Km + ε * LinearAlgebra.I); check = false)
        x = LinearAlgebra.issuccess(F) ? F \ hm : hermitian_map(λ -> λ > ε ? inv(λ) : zero(λ), Km) * hm
        return from_array(reshape(x, dim(r), dim(c), dim(s)), [r, c, s]),
            abs2(LinearAlgebra.dot(x, hm)) / (real(LinearAlgebra.dot(x, Km * x)) * nΘ)
    end

    f = initial_fidelity
    sweeps = 0
    for sweep in 1:maxsweeps
        A, _ = solve(B, s₂, r₁, s₁)
        B, f_new = solve(A, s₁, r₂, s₂)
        sweeps = sweep
        converged = abs(f_new - f) <= tol * abs(f_new)
        f = f_new
        converged && break
    end

    # 5. balance the bond (its dimension already chosen by the starting SVD), undo the gauge
    singular_values! = Ref{Tensor}()
    A, B, _ = factorize_svd(A * B, [r₁, s₁]; ortho = "none", maxdim = dim(c), cutoff = nothing, tags = tags(b), singular_values!)
    if gauge_fix
        A = apply_to_leg(L₁⁻¹, A, r₁)
        B = apply_to_leg(L₂⁻¹, B, r₂)
    end
    A, B = keep_bond_index(A, B, b)

    diagnostics = (;
        fidelity = f, initial_fidelity, negativity, nonhermiticity, condition, sweeps,
        singular_values = singular_values![],
    )
    return Q₁ * adapt_like(Q₁, A), Q₂ * adapt_like(Q₂, B), diagnostics
end

# The two-site gate `o` applied to the neighbours ψ₁, ψ₂ with no environment, the bond split by an
# SVD truncated by `maxdim` and `cutoff`, with its singular values. With `exact`, the SVD keeps every
# singular value above rounding, or `nothing` is returned if that is more than `maxdim`.
function svd_update(o::Tensor, ψ₁::Tensor, ψ₂::Tensor; maxdim::Integer, cutoff = nothing, exact = false)
    b = commonind(ψ₁, ψ₂)
    s₁, s₂ = commonind(ψ₁, o), commonind(ψ₂, o)
    Q₁, R₁, r₁ = split_reduced(ψ₁, [s₁, b])
    Q₂, R₂, r₂ = split_reduced(ψ₂, [s₂, b])
    Θ = apply(o, R₁ * R₂)
    if exact
        σ = LinearAlgebra.svdvals(reshape(Array(array(Θ, r₁, s₁, r₂, s₂)), dim(r₁) * dim(s₁), dim(r₂) * dim(s₂)))
        kept = count(>(10 * eps(real(scalartype(Θ))) * first(σ)), σ)
        0 < kept <= maxdim || return nothing
        maxdim, cutoff = kept, nothing
    end
    singular_values! = Ref{Tensor}()
    A, B, _ = factorize_svd(Θ, [r₁, s₁]; ortho = "none", maxdim, cutoff, tags = tags(b), singular_values!)
    A, B = keep_bond_index(A, B, b)
    return Q₁ * A, Q₂ * B, singular_values![]
end

# Whether a bond's environment can be trusted: its negativity and non-Hermiticity within `health`.
healthy(d, health) = d.negativity <= health.negativity && d.nonhermiticity <= health.nonhermiticity

# The CTMRG convergence threshold for a state.
default_env_tolerance(ψ) = 1.0e-6

"""
    apply_gates_full_update(circuit, ψ::TensorNetworkState; maxdim, env_maxdim, kwargs...)
    apply_gates_full_update(circuit, cache::CTMRGCache; maxdim, kwargs...)

Apply a sequence of gates to `ψ`, two-site gates by [`full_update`](@ref) in the CTMRG environment
of their bond. The graph must be planar.

The environment is updated, from where it was, before any two-site gate that acts on a tensor changed
since the last update, so a layer of gates on disjoint bonds (one colour of an edge colouring) shares
one environment. While a bond can hold a gate's result without truncation (as while bonds grow from a
product state), the gate is applied exactly, with no environment. A gate whose environment fails the
health checks is applied by a plain truncated SVD instead.

Given a `CTMRGCache` (as returned by a previous call) instead of a state, its environment is carried
over rather than built from scratch, and the updated cache is returned: a circuit split into several
calls gets the same environments as in one call. The cache keeps its `env_maxdim`, embedding and
projector. Call `update` on it before measuring.

# Arguments
- `circuit`: A [`Circuit`](@ref), or a vector of gates as tuples `(name, vertices)` or
  `(name, vertices, parameters)`, or as gate tensors.
- `ψ::TensorNetworkState` or `cache::CTMRGCache`: The state, or a CTMRG cache of it.

# Keyword Arguments
- `maxdim::Integer`: The maximum bond dimension of the state.
- `env_maxdim::Integer`: The bond dimension of the CTMRG environment (state method only). It needs
  to grow with `maxdim` for the full update to keep improving on simple update.
- `cutoff = nothing`: A singular value cutoff, as for `apply_gates`. Gates are applied exactly
  without an environment only when it is `nothing`.
- `normalize_tensors = true`: Normalise the two tensors after each gate.
- `cache_update_kwargs`: Passed to `update` on the cache before a gate that needs it. The default,
  `(; tolerance = 1e-6, maxiter = 100)`, is
  merged with what is given.
- `health`: The largest `negativity` and `nonhermiticity` (see `full_update`) an environment may
  have before the gate falls back to a plain SVD. Default `(; negativity = 5e-2, nonhermiticity = 5e-2)`,
  merged with what is given.
- `gauge_fix = true`: Passed to `full_update`.
- `positions`: The planar embedding of the graph (state method only; default: the vertex names,
  when they are coordinates).

# Returns
- The new state (or, given a cache, the updated cache) and the diagnostics of each two-site gate:
  those of `full_update`, with `environment` (false for the gates applied exactly: fidelity 1, no
  sweeps) and `fallback` (true for the gates applied by a plain SVD because their environment
  failed the health checks: fidelity `NaN`, no sweeps).
"""
function apply_gates_full_update(
        circuit::Vector, ψ::TensorNetworkState; env_maxdim::Integer, positions = default_positions(graph(ψ)), kwargs...
    )
    cache = CTMRGCache(copy(ψ), env_maxdim; positions, projector = :svd)
    cache, diagnostics = full_update_sequence(gate_list(circuit, ψ), cache; kwargs...)
    return network(cache), diagnostics
end

apply_gates_full_update(circuit::Vector, cache::CTMRGCache; kwargs...) =
    full_update_sequence(gate_list(circuit, network(cache)), copy(cache); kwargs...)

gate_list(circuit, ψ) = eltype(circuit) <: Tensor ? [(o, vertices(o, ψ)) for o in circuit] : toitensor(circuit, graph(ψ), siteinds(ψ))

# Apply the `(gate, vertices)` in `gates` in order to the network of `cache`, which is modified, two-site
# gates by `full_update`; see `apply_gates_full_update`.
function full_update_sequence(
        gates::Vector, cache::CTMRGCache;
        maxdim::Integer, cutoff = nothing, normalize_tensors = true, cache_update_kwargs = (;),
        health = (;), gauge_fix = true,
    )
    cache_update_kwargs = merge((; tolerance = default_env_tolerance(network(cache)), maxiter = 100), cache_update_kwargs)
    health = merge((; negativity = 5.0e-2, nonhermiticity = 5.0e-2), health)
    g = graph(cache)
    # A carried cache (one with projectors) counts every vertex as changed, so it is updated at the
    # first gate that needs it; `update` rebuilds it if a bond has changed dimension.
    built = !isempty(cache.projectors)
    affected = Set{eltype(vertices(g))}(built ? vertices(g) : [])
    sequences = Dict{Any, Any}()
    diagnostics = NamedTuple[]
    for (o, vs) in gates
        o = adapt_gate(o, network(cache))
        if length(vs) == 1
            v = only(vs)
            t = apply(o, network(cache)[v])
            setindex_preserve!(cache, normalize_tensors ? normalize(t) : t, v)
            push!(affected, v)
            continue
        end
        length(vs) == 2 && has_edge(g, NamedEdge(vs[1] => vs[2])) ||
            error("full update: two-site gates must act on neighbouring vertices; got $vs.")
        u, v = vs
        ψ = network(cache)
        exact = isnothing(cutoff) ? svd_update(o, ψ[u], ψ[v]; maxdim, exact = true) : nothing
        if isnothing(exact)
            if !built || any(in(affected), vs)
                cache = update(cache; cache_update_kwargs...)
                built = true
                empty!(affected)
            end
            ψ = network(cache)
            a, b, d = full_update(o, ψ[u], ψ[v], incoming_messages(cache, [u, v]); maxdim, cutoff, sequences, gauge_fix)
            fallback = !healthy(d, health)
            if fallback
                a, b, S = svd_update(o, ψ[u], ψ[v]; maxdim, cutoff)
                d = merge(d, (; fidelity = NaN, initial_fidelity = NaN, sweeps = 0, singular_values = S))
            end
            d = merge(d, (; environment = true, fallback))
        else
            a, b, S = exact
            d = (;
                fidelity = 1.0, initial_fidelity = 1.0, negativity = 0.0, nonhermiticity = 0.0,
                condition = 1.0, sweeps = 0, singular_values = S, environment = false, fallback = false,
            )
        end
        push!(diagnostics, d)
        setindex_preserve!(cache, normalize_tensors ? normalize(a) : a, u)
        setindex_preserve!(cache, normalize_tensors ? normalize(b) : b, v)
        union!(affected, vs)
    end
    return cache, diagnostics
end
