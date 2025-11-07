function algorithm_error()
    error("Algorithm choice not supported. Currently supported: bp, boundarymps, loopcorrections and exact.")
end

function state_error()
    error("Network type inside is not a TensorNetworkState.")
end

"""
    norm_sqr(ψ::Union{TensorNetworkState, AbstractBeliefPropagationCache}; alg, kwargs...)
    Compute the squared norm of a `TensorNetworkState` or the state wrapped in an updated  `Cache` using the specified algorithm.
    # Arguments
    - `ψ::Union{TensorNetworkState, AbstractBeliefPropagationCache}`: The tensor network state or updated cache wrapping the state.
    - `alg`: The algorithm to use for the norm calculation. Options include:
        - `"exact"`: Exact contraction of the tensor network.
        - `"bp"`: Belief propagation approximation.
        - `"boundarymps"`: Boundary MPS approximation (requires `mps_bond_dimension`).
        - `"loopcorrections"`: Loop corrections to belief propagation (requires `max_configuration_size`).
    # Keyword Arguments
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
    # Exact norm
    norm_exact = LinearAlgebra.norm(ψ; alg = "exact")    
    # Belief propagation norm   
    norm_bp = LinearAlgebra.norm(ψ; alg = "bp")
    # Boundary MPS norm with bond dimension 10
    norm_bmps = LinearAlgebra.norm(ψ; alg = "boundarymps", mps_bond_dimension = 10)
    ```
"""

function norm_sqr(tns::Union{TensorNetworkState, BeliefPropagationCache}; alg, kwargs...)
    algorithm_check(tns, "norm_sqr", alg)
    return norm_sqr(Algorithm(alg), tns; kwargs...)
end

function norm_sqr(
        alg::Algorithm"exact", ψ::TensorNetworkState;
        contraction_sequence_kwargs = (; alg = "einexpr", optimizer = Greedy())
    )
    ψIψ_tensors = norm_factors(ψ, collect(vertices(ψ)))
    denom_seq = contraction_sequence(ψIψ_tensors; contraction_sequence_kwargs...)
    return contract(ψIψ_tensors; sequence = denom_seq)[]
end

function norm_sqr(alg::Algorithm, cache::AbstractBeliefPropagationCache; max_configuration_size = nothing)
    tn = network(cache)

    if alg == Algorithm("bp") || alg == Algorithm("boundarymps")
        z = partitionfunction(cache)
    elseif alg == Algorithm("loopcorrections")
        z = loopcorrected_partitionfunction(cache, max_configuration_size)
    else
        return algorithm_error()
    end

    tn isa TensorNetworkState && return z
    tn isa TensorNetwork && return z * z
    return state_error()
end

function norm_sqr(alg::Union{Algorithm"bp", Algorithm"loopcorrections"}, ψ::TensorNetworkState; cache_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)
    return norm_sqr(alg, ψ_bpc; kwargs...)
end

function norm_sqr(alg::Algorithm"boundarymps", ψ::TensorNetworkState; mps_bond_dimension::Integer, partition_by = "row", cache_update_kwargs = default_bmps_update_kwargs(ψ), kwargs...)
    ψ_bmps = BoundaryMPSCache(ψ, mps_bond_dimension; partition_by)
    maxiter = get(cache_update_kwargs, :maxiter, default_bp_maxiter(ψ_bmps))
    cache_update_kwargs = (; cache_update_kwargs..., maxiter)
    ψ_bmps = update(ψ_bmps; cache_update_kwargs...)
    return norm_sqr(alg, ψ_bmps; kwargs...)
end

function norm_sqr(alg::Algorithm, ψ::TensorNetworkState; kwargs...)
    return algorithm_error()
end

LinearAlgebra.norm(alg::Algorithm, ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}; kwargs...) = sqrt(norm_sqr(alg, ψ; kwargs...))
LinearAlgebra.norm(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}; kwargs...) = sqrt(norm_sqr(ψ; kwargs...))
