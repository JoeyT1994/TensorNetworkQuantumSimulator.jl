#=
Full update: apply a two-site gate and truncate the bond to maximise the fidelity of the result
in an environment of the bond. For a gate `o` on the bond (v₁, v₂):
  1. Split ψᵥ = Qᵥ Rᵥ with Qᵥ carrying the outer virtual legs, and contract the environment with
     the Qs, once, into a reduced environment N on the legs (r₁, r₂) of R₁, R₂, after removing its
     overall phase (it is only fixed up to a scalar).
  2. Make N Hermitian and positive definite: N = W Λ W† with negative eigenvalues set to zero, then
     all of them shifted by `regularization` (relative to the largest). The shift keeps every
     direction of the bond in the objective: an approximate environment can give almost no weight to
     directions that matter, and with a zero cost there the truncation error would go to them, where
     undoing the gauge (step 3) amplifies it.
  3. Gauge fix: with ρᵢ the reduced environment of leg rᵢ (N traced over the other leg), absorb
     Lᵢ = (ρᵢ + δ)^{1/2} into Rᵢ, so the environment seen by the gauged tensors is close to the
     identity. The shift δ, relative to ρᵢ's largest eigenvalue, keeps Lᵢ invertible, so the gauge
     only conditions the problem and undoing it amplifies by at most δ^{-1/2}.
  4. Start from the truncated SVD of the gate applied to the gauged tensors, then sweep: with one
     tensor fixed, the other solves a small positive-definite linear system, densely, with Tikhonov
     regularisation. Stop when the fidelity stops improving.
  5. Re-split with an SVD to balance the bond, undo the gauge, and multiply back the Qs. The bond
     keeps its index if its dimension is unchanged, so a cache built on the state stays valid.

`apply_gates_full_update` draws the environments from CTMRG with SVD projectors. The environment is
contracted where the tensors are (on a GPU, if they are there); everything after it is on small dense
matrices, so it runs on the host, and the new tensors are moved back. This is for dense (non-graded)
backends.
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

"""
    full_update(o, ψ₁, ψ₂, envs; maxdim, cutoff = nothing, gauge_fix = true, maxsweeps = 100,
                tol = 1e-8, regularization = 1e-10, gauge_regularization = 1e-8, sequences)

Apply the two-site gate `o` to the tensors `ψ₁`, `ψ₂` of neighbouring vertices, truncating their
shared bond to at most `maxdim` (and by `cutoff`, as in `simple_update`, on the singular values in
the gauged frame) to maximise the fidelity in the environment `envs` (tensors of the norm network
with the bra's virtual legs primed, as returned by `incoming_messages`). `gauge_regularization` is
the shift δ of the gauge (see the notes at the top of this file); `sequences` keeps the order of
the environment's contraction between calls with environments of the same structure.
`regularization` is the relative shift of N's eigenvalues (see the notes at the top of this file).

Returns the new tensors and a named tuple of diagnostics: `fidelity` (in the environment, after the
sweeps) and `initial_fidelity` (of the starting SVD), `negativity` (the norm of the environment's
negative eigenvalues relative to all of them, before they were set to zero), `nonhermiticity` (the
norm of its anti-Hermitian part relative to its norm, before it was dropped), `condition` (ratio of
its largest to smallest eigenvalue, after the shift), `sweeps` and the bond's
`singular_values` in the gauged frame.
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

    # 1. the reduced environment, rows on the bra legs. It is only fixed up to a scalar, which may
    # be complex or negative (CTMRG corners carry arbitrary phases), so rotate it to a real,
    # positive trace.
    N = adapt(Array, contract_memoized(sequences, vcat(envs, Tensor[Q₁, Q₂, dag(prime(Q₁)), dag(prime(Q₂))])))
    Nm = reshape(array(N, prime(r₁), prime(r₂), r₁, r₂), d₁ * d₂, d₁ * d₂)
    τ = LinearAlgebra.tr(Nm)
    Nm = Nm * (conj(τ) / abs(τ))
    nonhermiticity = LinearAlgebra.norm(Nm - Nm') / (2 * LinearAlgebra.norm(Nm))

    # 2. Hermitian, positive definite: scaled to unit largest eigenvalue (positive, as the trace
    # is), every eigenvalue shifted by `regularization`, and factorised as N = X X†
    λ, W = LinearAlgebra.eigen(LinearAlgebra.Hermitian((Nm + Nm') / 2))
    negativity = LinearAlgebra.norm(min.(λ, 0)) / LinearAlgebra.norm(λ)
    λ = max.(λ, 0) ./ maximum(λ) .+ regularization
    condition = maximum(λ) / minimum(λ)
    X = W * LinearAlgebra.Diagonal(sqrt.(λ))

    # 3. gauge fix with the square roots of the single-leg reduced environments
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

    # With `other` fixed, the tensor on (r, s, c) that maximises the fidelity: the solution of
    # K x = h with K the environment of x and h its overlap with the target. Also returns the
    # fidelity of the result, x† h / x† K x up to normalisation.
    function solve(other, s_other, r, s)
        ōÑ = reduced_bra(other, [s_other]) * Ñ
        n = dim(r) * dim(c)
        Km = reshape(array(ōÑ * other, prime(r), prime(c), r, c), n, n)
        hm = reshape(array(ōÑ * Θ, prime(r), prime(c), s), n, dim(s))
        Km = (Km + Km') / 2
        ε = regularization * real(LinearAlgebra.tr(Km)) / n
        # Km is positive semi-definite up to rounding; if that makes the Cholesky factorisation
        # fail, solve on its eigenvalues above ε instead
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

    # 5. balance the bond (its dimension, and so `cutoff`, already chosen by the starting SVD), undo
    # the gauge, restore the outer legs
    singular_values! = Ref{Tensor}()
    A, B, _ = factorize_svd(A * B, [r₁, s₁]; ortho = "none", maxdim = dim(c), cutoff = nothing, tags = tags(b), singular_values!)
    if gauge_fix
        A = apply_to_leg(L₁⁻¹, A, r₁)
        B = apply_to_leg(L₂⁻¹, B, r₂)
    end
    # keep the bond's index when its dimension is unchanged, so environments built on it stay valid
    c = commonind(A, B)
    if dim(c) == dim(b)
        A, B = replaceind(A, c, b), replaceind(B, c, b)
    end

    diagnostics = (;
        fidelity = f, initial_fidelity, negativity, nonhermiticity, condition, sweeps,
        singular_values = singular_values![],
    )
    return Q₁ * adapt_like(Q₁, A), Q₂ * adapt_like(Q₂, B), diagnostics
end

# The two-site gate `o` applied exactly to the neighbours ψ₁, ψ₂, the bond split by an SVD, if it
# needs at most `maxdim` singular values (above rounding), with the bond's singular values;
# `nothing` if it would have to be truncated. Such a gate needs no environment.
function untruncated_update(o::Tensor, ψ₁::Tensor, ψ₂::Tensor; maxdim::Integer)
    b = commonind(ψ₁, ψ₂)
    s₁, s₂ = commonind(ψ₁, o), commonind(ψ₂, o)
    Q₁, R₁, r₁ = split_reduced(ψ₁, [s₁, b])
    Q₂, R₂, r₂ = split_reduced(ψ₂, [s₂, b])
    Θ = apply(o, R₁ * R₂)
    σ = LinearAlgebra.svdvals(reshape(Array(array(Θ, r₁, s₁, r₂, s₂)), dim(r₁) * dim(s₁), dim(r₂) * dim(s₂)))
    kept = count(>(10 * eps(real(scalartype(Θ))) * first(σ)), σ)
    kept > maxdim && return nothing
    singular_values! = Ref{Tensor}()
    A, B, _ = factorize_svd(Θ, [r₁, s₁]; ortho = "none", maxdim = kept, cutoff = nothing, tags = tags(b), singular_values!)
    c = commonind(A, B)
    if dim(c) == dim(b)
        A, B = replaceind(A, c, b), replaceind(B, c, b)
    end
    return Q₁ * A, Q₂ * B, singular_values![]
end

# The two-site gate `o` applied to the neighbours ψ₁, ψ₂ and the bond truncated by a plain SVD, with no
# environment (the fallback when the environment fails its health checks), with the bond's
# singular values.
function truncated_update(o::Tensor, ψ₁::Tensor, ψ₂::Tensor; maxdim::Integer, cutoff = nothing)
    b = commonind(ψ₁, ψ₂)
    s₁, s₂ = commonind(ψ₁, o), commonind(ψ₂, o)
    Q₁, R₁, r₁ = split_reduced(ψ₁, [s₁, b])
    Q₂, R₂, r₂ = split_reduced(ψ₂, [s₂, b])
    singular_values! = Ref{Tensor}()
    A, B, _ = factorize_svd(apply(o, R₁ * R₂), [r₁, s₁]; ortho = "none", maxdim, cutoff, tags = tags(b), singular_values!)
    c = commonind(A, B)
    if dim(c) == dim(b)
        A, B = replaceind(A, c, b), replaceind(B, c, b)
    end
    return Q₁ * A, Q₂ * B, singular_values![]
end

# Whether a bond's environment can be trusted, from the full update's diagnostics: its negativity and
# non-Hermiticity (before they were removed) within the thresholds of `health`.
healthy(d, health) = d.negativity <= health.negativity && d.nonhermiticity <= health.nonhermiticity

# The CTMRG convergence threshold for a state: 1e-6, or what its precision can reach (in single
# precision the change stalls at a few hundred ε, so 1000ε).
default_env_tolerance(ψ) = max(1.0e-6, 1000 * eps(real(scalartype(ψ))))

"""
    apply_gates_full_update(circuit, ψ::TensorNetworkState; maxdim, env_maxdim, cutoff = nothing,
                            normalize_tensors = true, positions, env_passes = nothing,
                            cache_update_kwargs = (; tolerance = max(1e-6, 1000ε), maxiter = 100),
                            health = (; negativity = 5e-2, nonhermiticity = 5e-2), gauge_fix = true)
    apply_gates_full_update(circuit, cache::CTMRGCache; maxdim, cutoff = nothing, …)

Apply a sequence of one- and two-site gates to `ψ`, two-site gates by `full_update` in the CTMRG
environment of their bond, with the corners truncated to `env_maxdim` by SVD projectors (the
`projector = :svd` of `CTMRGCache`). The CTMRG cache is updated before any two-site gate on a vertex
changed since the last update, so gates on disjoint bonds (one colour of an edge colouring) share an
environment. By default the cache is converged by `cache_update_kwargs` from its previous state
before every such update (with an integer `env_passes`, exactly that many CTMRG passes are run
instead), as it is when it is built, and when it is rebuilt because a bond has changed dimension;
its default tolerance is 1e-6, or 1000 times the machine epsilon ε of the state's element type if that
is larger (in single precision the change stalls at a few hundred ε), within at most 100 passes: late
in an evolution the environment can take more than `update`'s default 30, and an unconverged one
costs fidelity. A gate whose environment fails the health checks, negativity or non-Hermiticity
above the thresholds in `health` (see `full_update`), is applied instead with no environment, by a
truncated SVD of the gated tensors. `gauge_fix` is passed to `full_update`. With `cutoff = nothing`,
a two-site gate whose bond holds the result without truncation (as while bonds grow from a product
state) is applied exactly, by an SVD, with no environment: the cache isn't updated for it.
`circuit` is as for `apply_gates`. The graph must be planar, with an embedding from `positions`
(default: the vertex names, when they are coordinates).

Given a `CTMRGCache` (the one a previous call returned) instead of a state, the gates are applied to
its network and its environment is carried over: the first gate that needs one converges it from
where it was (rebuilding it if a bond has changed dimension since), rather than from scratch. Its
`env_maxdim`, embedding and projectors are kept. A sequence of gates split into several calls then
gets the same environments as in one call.

Returns the new state (or, given a cache, the updated cache) and the diagnostics of each two-site
gate (see `full_update`), with `environment` false for the gates applied exactly (fidelity 1, no
sweeps), and `fallback` true for the gates applied without their environment because it failed the
health checks.
"""
function apply_gates_full_update(
        circuit::Vector, ψ::TensorNetworkState; env_maxdim::Integer, positions = default_positions(graph(ψ)), kwargs...
    )
    cache = CTMRGCache(copy(ψ), env_maxdim; positions, projector = :svd)
    cache, diagnostics = full_update_sequence(gate_list(circuit, ψ), cache; kwargs...)
    return copy(network(cache)), diagnostics
end

apply_gates_full_update(circuit::Vector, cache::CTMRGCache; kwargs...) =
    full_update_sequence(gate_list(circuit, network(cache)), copy(cache); kwargs...)

gate_list(circuit, ψ) = eltype(circuit) <: Tensor ? [(o, vertices(o, ψ)) for o in circuit] : toitensor(circuit, graph(ψ), siteinds(ψ))

# Apply the `(gate, vertices)` in `gates` in order to the network of `cache`, which is modified, two-site
# gates by `full_update`; see `apply_gates_full_update`.
function full_update_sequence(
        gates::Vector, cache::CTMRGCache;
        maxdim::Integer, cutoff = nothing, normalize_tensors = true, env_passes::Union{Integer, Nothing} = nothing,
        cache_update_kwargs = (; tolerance = default_env_tolerance(network(cache)), maxiter = 100),
        health = (; negativity = 5.0e-2, nonhermiticity = 5.0e-2), gauge_fix = true,
    )
    g = graph(cache)
    # A fresh cache is built (converged) when the first gate that needs an environment comes; a
    # carried one (it has projectors) is converged from where it was at that gate, so every vertex
    # counts as changed. Later, a bond that changes dimension leaves only the messages and
    # projectors on that bond stale, which the environments of the other bonds don't use, so the
    # cache is rebuilt when it would be updated anyway: when such a gate acts on a vertex changed
    # since the last update.
    built = !isempty(cache.projectors)
    resized = built && stale_bonds(cache)
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
        ψ = network(cache)
        u, v = vs
        bond = commonind(ψ[u], ψ[v])
        exact = isnothing(cutoff) ? untruncated_update(o, ψ[u], ψ[v]; maxdim) : nothing
        if !isnothing(exact)
            a, b, S = exact
            push!(diagnostics, (;
                fidelity = 1.0, initial_fidelity = 1.0, negativity = 0.0, nonhermiticity = 0.0,
                condition = 1.0, sweeps = 0, singular_values = S, environment = false, fallback = false,
            ))
            setindex_preserve!(cache, normalize_tensors ? normalize(a) : a, u)
            setindex_preserve!(cache, normalize_tensors ? normalize(b) : b, v)
            resized |= commonind(a, b) != bond
            union!(affected, vs)
            continue
        end
        if !built || any(in(affected), vs)
            cache = if !built || resized
                update(CTMRGCache(network(cache), cache); cache_update_kwargs...)
            elseif isnothing(env_passes)
                update(cache; cache_update_kwargs...)
            else
                update(cache; maxiter = env_passes, tolerance = nothing)
            end
            built = true
            resized = false
            empty!(affected)
        end
        ψ = network(cache)
        a, b, d = full_update(o, ψ[u], ψ[v], incoming_messages(cache, [u, v]); maxdim, cutoff, sequences, gauge_fix)
        fallback = !healthy(d, health)
        if fallback
            a, b, S = truncated_update(o, ψ[u], ψ[v]; maxdim, cutoff)
            d = merge(d, (; singular_values = S))
        end
        push!(diagnostics, merge(d, (; environment = true, fallback)))
        setindex_preserve!(cache, normalize_tensors ? normalize(a) : a, u)
        setindex_preserve!(cache, normalize_tensors ? normalize(b) : b, v)
        resized |= commonind(a, b) != bond
        union!(affected, vs)
    end
    return cache, diagnostics
end

adapt_gate(gate::Tensor, ψ::TensorNetworkState) =
    adapt(unspecify_type_parameters(datatype(ψ)), scalartype(gate) <: Complex ? adapt(complex(scalartype(ψ)), gate) : adapt(scalartype(ψ), gate))
