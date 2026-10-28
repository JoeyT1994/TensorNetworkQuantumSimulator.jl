function contract(alg::Algorithm"exact", tn::AbstractTensorNetwork; contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = GreedyMethod()))
    tn_tensors = [tn[v] for v in vertices(tn)]
    seq = contraction_sequence(tn_tensors; contraction_sequence_kwargs...)
    return scalar(contract(tn_tensors; sequence = seq))
end

# bp, boundarymps, ctmrg: the partition function of the converged cache. `cache_kwargs` are those
# of the cache (`env_maxdim`, ...).
function contract(alg::Union{Algorithm"bp", Algorithm"boundarymps", Algorithm"ctmrg"}, tn::TensorNetwork; cache_update_kwargs = default_cache_update_kwargs(alg, tn), cache_kwargs...)
    cache_update_kwargs = scalar_cache_update_kwargs(alg, cache_update_kwargs)
    return partitionfunction(converged_cache(alg, tn; cache_kwargs..., cache_update_kwargs))
end

function contract(tn::AbstractTensorNetwork; alg = "exact", kwargs...)
    return contract(Algorithm(alg), tn; kwargs...)
end
