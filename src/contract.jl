function ITensors.contract(alg::Algorithm"exact", tn::AbstractTensorNetwork; contraction_sequence_kwargs = (; alg = "einexpr", optimizer = Greedy()))
    tn_tensors = [tn[v] for v in vertices(tn)]
    seq = contraction_sequence(tn_tensors; contraction_sequence_kwargs...)
    return ITensors.contract(tn_tensors; sequence = seq)[]
end

function ITensors.contract(alg::Algorithm"bp", tn::TensorNetwork; bp_update_kwargs = default_bp_update_kwargs(tn))
    return partitionfunction(update(BeliefPropagationCache(tn); bp_update_kwargs...))
end

function ITensors.contract(alg::Algorithm"boundarymps", tn::TensorNetwork; mps_bond_dimension::Integer, bmps_update_kwargs = default_bmps_update_kwargs(tn))
    return partitionfunction(update(BoundaryMPSCache(tn, mps_bond_dimension); bmps_update_kwargs...))
end

function ITensors.contract(tn::AbstractTensorNetwork; alg = "exact", kwargs...)
    return ITensors.contract(Algorithm(alg), tn; kwargs...)
end
