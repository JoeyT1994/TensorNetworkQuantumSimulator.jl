"""
    renyi_entropy(ρ::AbstractMatrix, α::Real; normalize = true)

Compute the Rényi entropy of order `α` of a density matrix `ρ`.

The Rényi entropy is defined as

```math
S_\\alpha(\\rho) = \\frac{1}{1 - \\alpha} \\log \\operatorname{tr}(\\rho^\\alpha)
```

The limit ``\\alpha \\to 1`` recovers the von Neumann entropy ``S = -\\operatorname{tr}(\\rho \\log \\rho)``, which is handled exactly.

# Arguments
- `ρ`: Density matrix as a plain Julia matrix.
- `α`: Rényi index. Use `α = 1` for von Neumann entropy, `α = 2` for second Rényi entropy.

# Keyword Arguments
- `normalize`: If `true` (default), normalise `ρ` by its trace before computing the entropy.
"""
function renyi_entropy(ρ::AbstractMatrix, α::Real; normalize = true)
    if normalize
        ρ = ρ / tr(ρ)
    end
    λs = eigvals(Hermitian(ρ))
    filter!(λ -> abs(λ) > 10*eps(real(eltype(λs))), λs)
    α == 1 && return -sum(p -> p * log(p), λs)  # von Neumann limit
    return log(sum(λs .^ α)) / (1 - α)
end

function matricize(a::ITensor, row_inds = filter(i -> plev(i) ==0, inds(a)))
    col_inds = prime.(row_inds)
    row_name = name(ITensorBase.uniquename(first(row_inds)))
    col_name = name(ITensorBase.uniquename(first(col_inds)))
    return ITensors.array(ITensors.matricize(a, Tuple(row_inds) => row_name, Tuple(col_inds) => col_name))
end

"""
    renyi_entropy(a::ITensor, row_inds = ...; normalize = true, α = 1)

Compute the Rényi entropy of order `α` of a density matrix represented as an `ITensor`.

The tensor `a` is interpreted as a density matrix where unprimed indices are row indices
and primed indices are column indices.

# Arguments
- `a`: Density matrix as an `ITensor`.
- `row_inds`: The row (ket) indices. Defaults to all unprimed indices of `a`.

# Keyword Arguments
- `normalize`: If `true` (default), normalise by the trace.
- `α`: Rényi index (default `1`, i.e. von Neumann entropy).
"""
function renyi_entropy(a::ITensor, row_inds = filter(i -> plev(i) ==0, inds(a)); normalize = true, α = 1)
    return renyi_entropy(matricize(a, row_inds), α)
end

"""
    renyi_entropy(bp_cache::BeliefPropagationCache, e::NamedEdge; α)

Compute the Rényi entropy of order `α` across the bond `e` using the BP messages stored
in `bp_cache`.

This is an efficient single-edge computation that avoids constructing a full reduced density
matrix. It is exact on trees and approximate on loopy graphs (subject to the quality of the
BP fixed point). Requires the cache to already be updated.

# Arguments
- `bp_cache`: A converged `BeliefPropagationCache`.
- `e`: The bond edge across which to compute the entanglement entropy.

# Keyword Arguments
- `α`: Rényi index. Use `α = 1` for von Neumann entropy.
"""
function renyi_entropy(
    bp_cache::BeliefPropagationCache,
    e::NamedEdge;
    α::Real
)
    ee = 0
    m1, m2 = message(bp_cache, e), message(bp_cache, reverse(e))
    edge_ind = only(virtualinds(bp_cache, e))
    root_m2 = sqrth_safe(
        m2, (inds(m2)[1],), (inds(m2)[2],);
        atol = 10 * eps(real(scalartype(m2))), rtol = 0
    )

    edge_ind_p, edge_ind_pp = prime(edge_ind), prime(prime(edge_ind))
    ρ = (m1 * replaceind(root_m2, edge_ind_p, edge_ind_pp)) * root_m2
    ρ = replaceind(ρ, edge_ind_pp, edge_ind_p)
    return renyi_entropy(ρ; α)
end

"""
    renyi_entropy(tns::TensorNetworkState, e::NamedEdge; alg, α)

Compute the Rényi entropy of order `α` across the bond `e` of a `TensorNetworkState`.

Constructs a `BeliefPropagationCache` internally, runs BP, and computes the entropy
from the converged messages. For repeated calculations, prefer constructing and
updating the cache explicitly and calling `renyi_entropy(bp_cache, e; α)`.

# Arguments
- `tns`: The tensor network state.
- `e`: The bond edge.

# Keyword Arguments
- `alg`: Contraction algorithm. Currently only `"bp"` is supported.
- `α`: Rényi index.
"""
function renyi_entropy(tns::TensorNetworkState, e::NamedEdge; alg, α::Real)
    algorithm_check(tns, "rdm", alg)
    return renyi_entropy(Algorithm(alg), tns, e; α)
end

function renyi_entropy(alg::Algorithm"bp", tns::TensorNetworkState, e::NamedEdge; α::Real)
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = update(bp_cache)
    return renyi_entropy(bp_cache, e; α)
end

"""
    renyi_entropy(ψ, verts::Vector; alg, α, kwargs...)

Compute the Rényi entropy of order `α` of the reduced density matrix on `verts`.

Constructs the reduced density matrix on the specified vertices and computes its Rényi entropy.
Supports `BeliefPropagationCache`, `BoundaryMPSCache`, and `TensorNetworkState` inputs.
For single-bond entanglement entropy with BP, prefer the edge-based method
`renyi_entropy(bp_cache, e; α)` which avoids constructing the full RDM.

# Arguments
- `ψ`: A `TensorNetworkState`, `BeliefPropagationCache`, or `BoundaryMPSCache`.
- `verts`: Vector of vertices defining the subsystem.

# Keyword Arguments
- `alg`: Contraction algorithm (`"bp"`, `"boundarymps"`, or `"exact"`).
- `α`: Rényi index.
- Additional kwargs are forwarded to `reduced_density_matrix`.
"""
function renyi_entropy(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}, verts::Vector; alg, α::Real, kwargs...)
    algorithm_check(ψ, "rdm", alg)
    return renyi_entropy(reduced_density_matrix(ψ, verts; alg, normalize = false, kwargs...); normalize = true, α)
end

"""
    second_renyi_entanglement_entropy(args...; kwargs...)

Convenience wrapper for [`renyi_entropy`](@ref) with `α = 2`.

Accepts the same arguments as `renyi_entropy`. The second Rényi entropy is computationally
cheaper than the von Neumann entropy as it only requires ``\\operatorname{tr}(\\rho^2)``
rather than a full eigendecomposition.
"""
second_renyi_entanglement_entropy(args...; kwargs...) = renyi_entropy(args...; kwargs..., α = 2)

"""
    von_neumann_entanglement_entropy(args...; kwargs...)

Convenience wrapper for [`renyi_entropy`](@ref) with `α = 1`.

Accepts the same arguments as `renyi_entropy`. Computes the von Neumann entropy
``S = -\\operatorname{tr}(\\rho \\log \\rho)`` via the ``\\alpha \\to 1`` limit of the Rényi entropy.
"""
von_neumann_entanglement_entropy(args...; kwargs...) = renyi_entropy(args...; kwargs..., α = 1)
