function inner_algorithm_error()
    error("Algorithm choice not supported. Currently supported: bp, boundarymps, loopcorrections and exact.")
end

function inner_state_error()
    error("Network type inside the cache is not a BilinearForm.")
end

"""
    inner(ψ::TensorNetworkState, ϕ::TensorNetworkState; alg, kwargs...)

    Compute the inner product between two different TensorNetworkStates using the specified algorithm.
    If you want the norm_squared of a tensornetwork state ψ, use `norm_sqr(ψ; alg, kwargs...)` instead.
    The two states should have the same graph structure and physical indices on each site.

    # Arguments
    - `ψ::TensorNetworkState`: The first tensor network state.
    - `ϕ::TensorNetworkState`: The second tensor network state.

    # Keyword Arguments
    - `alg`: The algorithm to use for the inner product calculation. Options include:
        - `"exact"`: Exact contraction of the tensor network.
        - `"bp"`: Belief propagation approximation.
        - `"boundarymps"`: Boundary MPS approximation (requires `mps_bond_dimension`).
        - `"loopcorrections"`: Loop corrections to belief propagation.
    - Extra kwargs for `alg = "boundarymps"`:
        - `mps_bond_dimension::Integer`: The bond dimension for the boundary MPS approximation.
        - `partition_by`: How to partition the graph for boundary MPS (default is `"row"`).
        - `cache_update_kwargs`: Additional keyword arguments for updating the cache.
    - Extra kwargs for `alg = "bp"` or `"loopcorrections"`:
        - `cache_update_kwargs`: Additional keyword arguments for updating the cache.
        - `max_configuration_size`: Maximum configuration size for loop corrections (only for `"loopcorrections"`).

    # Returns
    - The computed inner product as a scalar value.

    # Example
    ```julia
    s = siteinds("S=1/2", g)
    ψ = random_tensornetworkstate(ComplexF32, g, s; bond_dimension = 4)
    ϕ = random_tensornetworkstate(ComplexF32, g, s; bond_dimension = 4)

    # Exact inner product
    ip_exact = ITensors.inner(ψ, ϕ; alg = "exact")

    # Belief propagation inner product
    ip_bp = ITensors.inner(ψ, ϕ; alg = "bp")

    # Boundary MPS inner product with bond dimension 10
    ip_bmps = ITensors.inner(ψ, ϕ; alg = "boundarymps", mps_bond_dimension = 10)
    ```
"""
function ITensors.inner(ψ::TensorNetworkState, ϕ::TensorNetworkState; alg, kwargs...)
    algorithm_check(ψ, "inner", alg)
    algorithm_check(ϕ, "inner", alg)
    return inner(Algorithm(alg), ψ, ϕ; kwargs...)
end

function ITensors.inner(
        alg::Algorithm"exact", blf::BilinearForm;
        contraction_sequence_kwargs = (; alg = "einexpr", optimizer = Greedy())
    )
    blf_tensors = bp_factors(blf, collect(vertices(ket(blf))))
    seq = contraction_sequence(blf_tensors; contraction_sequence_kwargs...)
    return contract(blf_tensors; sequence = seq)[]
end

function ITensors.inner(alg::Algorithm, cache::AbstractBeliefPropagationCache; max_configuration_size = nothing)
    tn = network(cache)
    if alg == Algorithm("bp") || alg == Algorithm("boundarymps")
        z = partitionfunction(cache)
    elseif alg == Algorithm("loopcorrections")
        z = loopcorrected_partitionfunction(cache, max_configuration_size)
    else
        return inner_algorithm_error()
    end

    tn isa BilinearForm && return z
    return inner_state_error()
end

function ITensors.inner(alg::Union{Algorithm"bp", Algorithm"loopcorrections"}, ψ::TensorNetworkState, ϕ::TensorNetworkState; cache_update_kwargs = (;), kwargs...)
    ψϕ_bpc = BeliefPropagationCache(BilinearForm(ψ, ϕ))
    ψϕ_bpc = update(ψϕ_bpc; cache_update_kwargs...)
    return inner(alg, ψϕ_bpc; kwargs...)
end

function ITensors.inner(alg::Algorithm"boundarymps", ψ::TensorNetworkState, ϕ::TensorNetworkState; mps_bond_dimension::Integer, partition_by = "row", cache_update_kwargs = (;), kwargs...)
    ψϕ_bmps = BoundaryMPSCache(BilinearForm(ψ, ϕ), mps_bond_dimension; partition_by)
    maxiter = get(cache_update_kwargs, :maxiter, default_bp_maxiter(ψϕ_bmps))
    cache_update_kwargs = (; cache_update_kwargs..., maxiter)
    ψϕ_bmps = update(ψϕ_bmps; cache_update_kwargs...)
    return inner(alg, ψϕ_bmps; kwargs...)
end

function ITensors.inner(alg::Algorithm"exact", ψ::TensorNetworkState, ϕ::TensorNetworkState)
    return inner(alg, BilinearForm(ψ, ϕ))
end
