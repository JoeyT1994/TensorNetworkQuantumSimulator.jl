function LinearAlgebra.normalize(alg::Algorithm"bp", tns::TensorNetworkState; cache_update_kwargs = default_bp_update_kwargs(tns))
    tns_bpc = BeliefPropagationCache(tns)
    tns_bpc = update(tns_bpc; cache_update_kwargs...)
    rescale!(tns_bpc)
    return network(tns_bpc)
end

"""
    normalize(tns::TensorNetworkState; alg, kwargs...)

Normalize a `TensorNetworkState` using the specified algorithm.

# Arguments
- `tns::TensorNetworkState`: The tensor network state to be normalized.

# Keyword Arguments
- `alg`: The algorithm to use for normalization. Currently, only `"bp"` is supported.

# Returns
- The normalized `TensorNetworkState` such that `norm_sqr(tns; alg) ≈ 1`.
"""
function LinearAlgebra.normalize(tns::TensorNetworkState; alg, kwargs...)
    algorithm_check(tns, "normalize", alg)
    return normalize(Algorithm(alg), tns; kwargs...)
end

#TODO: Get this working for boundarymps
