function LinearAlgebra.normalize(alg::Algorithm"bp", tns::TensorNetworkState; cache_update_kwargs = default_bp_update_kwargs(tns))
    tns_bpc = BeliefPropagationCache(tns)
    tns_bpc = update(tns_bpc; cache_update_kwargs...)
    rescale!(tns_bpc)
    return network(tns_bpc)
end

"""
    normalize(tns::TensorNetworkState; alg = nothing, kwargs...)
    Normalize a `TensorNetworkState` using the specified algorithm.
    The supported algorithms are:
    - `"bp"`: Normalize using Belief Propagation.
    # Arguments
    - `tns::TensorNetworkState`: The tensor network state to be normalized.
    - `alg`: The normalization algorithm to use. Default is `nothing`, so it must be specified explicitly.
    - `kwargs...`: Additional keyword arguments specific to the chosen algorithm.
    # Returns
    - The normalized `tns::TensorNetworkState` such that `norm_sqr(tns; alg = "bp) = 1`.
"""
function LinearAlgebra.normalize(tns::TensorNetworkState; alg = nothing, kwargs...)
    return normalize(Algorithm(alg), tns; kwargs...)
end