function contract_network(alg::Algorithm"exact", tn::AbstractTensorNetwork; contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = GreedyMethod()))
    tn_tensors = [tn[v] for v in vertices(tn)]
    seq = contraction_sequence(tn_tensors; contraction_sequence_kwargs...)
    return scalar(contract_network(tn_tensors; sequence = seq))
end

function contract_network(alg::Algorithm"bp", tn::TensorNetwork; bp_update_kwargs = default_bp_update_kwargs(tn))
    return partitionfunction(update(BeliefPropagationCache(tn); bp_update_kwargs...))
end

function contract_network(alg::Algorithm"boundarymps", tn::TensorNetwork; mps_bond_dimension::Integer, bmps_update_kwargs = default_bmps_update_kwargs(tn))
    return partitionfunction(update(BoundaryMPSCache(tn, mps_bond_dimension); bmps_update_kwargs...))
end

function contract_network(tn::AbstractTensorNetwork; alg = "exact", kwargs...)
    return contract_network(Algorithm(alg), tn; kwargs...)
end
