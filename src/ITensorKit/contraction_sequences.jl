# Contraction-order finders for `ITensorMap` networks.
#
# `contraction_sequence(tensors; alg)` returns a nested-vector pairwise contraction tree
# of (1-based) tensor positions — the ITensors `sequence` format. `contract` (contract.jl)
# converts this tree into the `ncon` elimination order via `_contraction_order`.
#
# Two finders are provided: `Algorithm"optimal"` (TensorOperations' `optimaltree`) and
# `Algorithm"omeinsum"` (OMEinsumContractionOrders). Extend by adding a
# `contraction_sequence(::Algorithm"…")` method. Names (`Algorithm`, `optimaltree`, `EinCode`,
# `inds`, `dim`, `noncommoninds`, `ITensor`, `datatype`, `adapt`) resolve from the enclosing
# `ITensorKit` module scope.

export contraction_sequence

function prune_trivial_tensors(tensors::AbstractVector)
    pruned_tensors = copy(tensors)
    for (i, t) in enumerate(pruned_tensors)
        if all(d -> d == 1, dim.(inds(tensors[i])))
            pruned_tensors[i] = adapt_like(t, ITensor(1))
        end
    end
    return pruned_tensors
end

function contraction_sequence(::Algorithm"optimal", tensors::AbstractVector; prune_tensors = false)
    #Needed because tensor operations bugs on trivial tensors
    if prune_tensors
        disable_warn_order()
        tensors = prune_trivial_tensors(tensors)
    end
    network = collect.(inds.(tensors))
    #Converting dims to Float64 to minimize overflow issues
    inds_to_dims = Dict(i => Float64(dim(i)) for i in unique(Iterators.flatten(network)))
    seq, _ = TensorOperations.optimaltree(network, inds_to_dims)
    seq = typeof(seq) <: Int ? [seq] : seq
    return seq
end

function contraction_sequence(::Algorithm"omeinsum", tensors::AbstractVector; optimizer = TreeSA())
    code, size_dict = to_eincode(tensors)
    optcode = optimize_code(code, size_dict, optimizer)
    return to_contraction_sequence(optcode)
end

function contraction_sequence(tensors::AbstractVector; alg = "optimal", kwargs...)
    return contraction_sequence(Algorithm(alg), tensors; kwargs...)
end

#OMEinsumContractionOrders helpers
function to_eincode(tensors::AbstractVector)
    ixs = map(t -> collect(inds(t)), tensors)
    LT = eltype(eltype(ixs))
    iy = collect(LT, reduce(noncommoninds, tensors))
    size_dict = Dict{LT, Int}(i => dim(i) for ix in ixs for i in ix)
    return EinCode(ixs, iy), size_dict
end

function to_contraction_sequence(ne::NestedEinsum)
    OMEinsumContractionOrders.isleaf(ne) && return ne.tensorindex
    return map(to_contraction_sequence, ne.args)
end
