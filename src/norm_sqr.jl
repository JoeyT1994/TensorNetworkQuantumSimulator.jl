function algorithm_error()
    error("Algorithm choice not supported. Currently supported: bp, boundarymps, loopcorrections and exact.")
end

function state_error(expected::String)
    error("Network type inside the cache is not a $expected.")
end

# Contract a belief-propagation / boundary-MPS cache down to its (scalar) partition function.
function cache_partitionfunction(alg::Algorithm, cache::AbstractBeliefPropagationCache; max_configuration_size = nothing)
    if alg == Algorithm("bp") || alg == Algorithm("boundarymps")
        return partitionfunction(cache)
    elseif alg == Algorithm("loopcorrections")
        return loopcorrected_partitionfunction(cache, max_configuration_size)
    else
        return algorithm_error()
    end
end

"""
    norm_sqr(ψ::Union{TensorNetworkState, AbstractBeliefPropagationCache}; alg, kwargs...)

Compute the squared norm of a `TensorNetworkState` or the state wrapped in an updated cache using the specified algorithm.

# Arguments
- `ψ::Union{TensorNetworkState, AbstractBeliefPropagationCache}`: The tensor network state or updated cache wrapping the state.

# Keyword Arguments
- `alg`: The algorithm to use. Options include:
    - `"exact"`: Exact contraction of the tensor network.
    - `"bp"`: Belief propagation approximation.
    - `"boundarymps"`: Boundary MPS approximation (requires `mps_bond_dimension`).
    - `"loopcorrections"`: Loop corrections to belief propagation (requires `max_configuration_size`).
- For `alg = "boundarymps"`:
    - `mps_bond_dimension::Integer`: The bond dimension for the boundary MPS approximation.
    - `partition_by`: How to partition the graph for boundary MPS (default is `"row"`).
    - `cache_update_kwargs`: Additional keyword arguments for updating the cache.
- For `alg = "bp"` or `"loopcorrections"`:
    - `cache_update_kwargs`: Additional keyword arguments for updating the cache.
    - `max_configuration_size`: Maximum configuration size for loop corrections (only for `"loopcorrections"`).

# Returns
- The computed squared norm as a scalar value.

# Example
```julia
s = siteinds("S=1/2", g)
ψ = random_tensornetworkstate(ComplexF32, g, s; bond_dimension = 4)
# Exact squared norm
nsq_exact = norm_sqr(ψ; alg = "exact")
# Belief propagation squared norm
nsq_bp = norm_sqr(ψ; alg = "bp")
# Boundary MPS squared norm with bond dimension 10
nsq_bmps = norm_sqr(ψ; alg = "boundarymps", mps_bond_dimension = 10)
```
"""

function norm_sqr(tns::Union{TensorNetworkState, AbstractBeliefPropagationCache}; alg, kwargs...)
    algorithm_check(tns, "norm_sqr", alg)
    return norm_sqr(Algorithm(alg), tns; kwargs...)
end

function norm_sqr(
        alg::Algorithm"exact", ψ::TensorNetworkState;
        contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = GreedyMethod())
    )
    ψIψ_tensors = norm_factors(ψ, collect(vertices(ψ)))
    denom_seq = contraction_sequence(ψIψ_tensors; contraction_sequence_kwargs...)
    return scalar(contract_network(ψIψ_tensors; sequence = denom_seq))
end

function norm_sqr(alg::Algorithm, cache::AbstractBeliefPropagationCache; max_configuration_size = nothing)
    tn = network(cache)
    z = cache_partitionfunction(alg, cache; max_configuration_size)
    tn isa TensorNetworkState && return z
    tn isa TensorNetwork && return z * z
    return state_error("TensorNetworkState")
end

function norm_sqr(alg::Union{Algorithm"bp", Algorithm"loopcorrections"}, ψ::TensorNetworkState; cache_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)
    return norm_sqr(alg, ψ_bpc; kwargs...)
end

function norm_sqr(alg::Algorithm"boundarymps", ψ::TensorNetworkState; mps_bond_dimension::Integer, partition_by = "row", cache_update_kwargs = default_bmps_update_kwargs(ψ), kwargs...)
    ψ_bmps = BoundaryMPSCache(ψ, mps_bond_dimension; partition_by)
    cache_update_kwargs = with_default_maxiter(cache_update_kwargs, ψ_bmps)
    ψ_bmps = update(ψ_bmps; cache_update_kwargs...)
    return norm_sqr(alg, ψ_bmps; kwargs...)
end

function norm_sqr(alg::Algorithm, ψ::TensorNetworkState; kwargs...)
    return algorithm_error()
end

LinearAlgebra.norm(alg::Algorithm, ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}; kwargs...) = sqrt(norm_sqr(alg, ψ; kwargs...))
LinearAlgebra.norm(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}; kwargs...) = sqrt(norm_sqr(ψ; kwargs...))
