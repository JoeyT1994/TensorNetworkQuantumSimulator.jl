# ρ / tr(ρ), the trace taken with deltas in ρ's storage
function normalize_rdm(ρ::Tensor)
    δ = reduce(*, (adapt(datatype(ρ))(delta(i, prime(i))) for i in inds(ρ; plev = 0)))
    return ρ / scalar(ρ * δ)
end

"""
    reduced_density_matrix(ψ, verts; alg = nothing, normalize = true, kwargs...)

Compute the reduced density matrix on the vertices `verts` of the tensor network state `ψ`.

# Arguments
- `ψ::Union{TensorNetworkState, AbstractBeliefPropagationCache}`: The tensor network state or its associated cache.
- `verts`: The vertices over which to compute the reduced density matrix. Can be a single vertex or a collection of vertices.

# Keyword Arguments
- `alg::Union{String, Nothing}`: The contraction algorithm to use. If not provided, defaults based on the type of `ψ`. Supported algorithms are `"exact"`, `"bp"`, `"boundarymps"` and `"ctmrg"` (one vertex or the two ends of an edge, requires `env_maxdim`).
- `normalize::Bool = true`: Whether to normalize the reduced density matrix so that its trace is 1.
- `kwargs...`: Additional keyword arguments specific to the chosen algorithm.

# Returns
- A `Tensor` representing the reduced density matrix on the specified vertices.
"""
function reduced_density_matrix(
        ψ::Union{TensorNetworkState, AbstractBeliefPropagationCache}, verts;
        alg::Union{String, Nothing} = default_alg(ψ), normalize::Bool = true, kwargs...
    )
    algorithm_check(ψ, "rdm", alg)
    ρ = reduced_density_matrix(Algorithm(alg), ψ, collect_vertices(verts, graph(ψ)); kwargs...)
    return normalize ? normalize_rdm(ρ) : ρ
end

function reduced_density_matrix(
        alg::Algorithm"exact",
        ψ::TensorNetworkState,
        verts::Vector;
        contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = GreedyMethod()),
    )
    ρ_tensors = norm_factors(ψ, collect(vertices(ψ)); op_strings = v -> v ∈ verts ? "ρ" : "I")
    return contract(ρ_tensors; sequence = contraction_sequence(ρ_tensors; contraction_sequence_kwargs...))
end

function reduced_density_matrix(alg::Union{Algorithm"bp", Algorithm"ctmrg"}, cache::AbstractBeliefPropagationCache, vs::Vector)
    return contract_region(cache, observable_region(cache, vs), v -> v ∈ vs ? "ρ" : "I")
end

function reduced_density_matrix(alg::Algorithm"boundarymps", cache::BoundaryMPSCache, vs::Vector; bmps_messages_up_to_date = false)
    ρ, _ = path_contract(cache, vs, v -> v ∈ vs ? "ρ" : "I"; bmps_messages_up_to_date, calculate_denom = false)
    return ρ
end

# bp and ctmrg: converge a cache over ψ (`cache_kwargs` are the cache's, e.g. `env_maxdim`).
function reduced_density_matrix(
        alg::Union{Algorithm"bp", Algorithm"ctmrg"},
        ψ::TensorNetworkState,
        verts::Vector;
        cache_update_kwargs = default_cache_update_kwargs(alg, ψ),
        cache_kwargs...,
    )
    return reduced_density_matrix(alg, converged_cache(alg, ψ; cache_kwargs..., cache_update_kwargs), verts)
end

function reduced_density_matrix(
        alg::Algorithm"boundarymps",
        ψ::TensorNetworkState,
        verts::Vector;
        cache_update_kwargs = default_bmps_update_kwargs(ψ),
        env_maxdim::Integer,
        partition_by::String = boundarymps_partitioning([verts]),
        kwargs...,
    )
    ψ_bpc = converged_cache(alg, ψ; env_maxdim, partition_by, cache_update_kwargs)
    return reduced_density_matrix(alg, ψ_bpc, verts; kwargs...)
end

const rdm = reduced_density_matrix
