function contract(alg::Algorithm"exact", tn::AbstractTensorNetwork; contraction_sequence_kwargs = default_contraction_sequence_kwargs())
    tn_tensors = [tn[v] for v in vertices(tn)]
    seq = contraction_sequence(tn_tensors; contraction_sequence_kwargs...)
    return scalar(contract(tn_tensors; sequence = seq))
end

# CTMRG on `tn` itself rather than on its norm network, which `CTMRGCache(tn, env_maxdim)` would build.
# `cache_kwargs` are those of the cache (`positions`, `projector`).
function contract(alg::Algorithm"ctmrg", tn::TensorNetwork; cache_update_kwargs = default_cache_update_kwargs(alg, tn), env_maxdim::Integer, cache_kwargs...)
    cache_update_kwargs = scalar_cache_update_kwargs(alg, cache_update_kwargs)
    return partitionfunction(update(CTMRGCache(tn, default_messages(), env_maxdim; cache_kwargs...); cache_update_kwargs...))
end

# BP on `tn` itself rather than on its norm network, which `BeliefPropagationCache(tn)` would build.
function contract(alg::Algorithm"bp", tn::TensorNetwork; cache_update_kwargs = default_cache_update_kwargs(alg, tn))
    return partitionfunction(update(BeliefPropagationCache(tn, default_messages(tn)); cache_update_kwargs...))
end

# Boundary MPS on `tn` itself rather than on its norm network, which `BoundaryMPSCache(tn, env_maxdim)` would build.
function contract(alg::Algorithm"boundarymps", tn::TensorNetwork; cache_update_kwargs = default_cache_update_kwargs(alg, tn), env_maxdim::Integer, cache_kwargs...)
    return partitionfunction(update(BoundaryMPSCache(tn, default_messages(), env_maxdim; cache_kwargs...); cache_update_kwargs...))
end

function contract(tn::AbstractTensorNetwork; alg = "exact", kwargs...)
    return contract(Algorithm(alg), tn; kwargs...)
end
