using OMEinsumContractionOrders: OMEinsumContractionOrders, optimize_code, EinCode, NestedEinsum, TreeSA, GreedyMethod, SABipartite, Treewidth, ExactTreewidth, HyperND, ExhaustiveSearch

# The exact "optimal" contraction order (Pfeifer 2014 netcon), from OMEinsumContractionOrders'
# `ExhaustiveSearch` optimizer.
function contraction_sequence(::Algorithm"optimal", tensors::Vector{<:Tensor})
    return contraction_sequence(Algorithm("omeinsum"), tensors; optimizer = ExhaustiveSearch())
end

function contraction_sequence(::Algorithm"omeinsum", tensors::Vector{<:Tensor}; optimizer = TreeSA())
    code, size_dict = to_eincode(tensors)
    optcode = optimize_code(code, size_dict, optimizer)
    seq = to_contraction_sequence(optcode)
    #A single-tensor network optimizes to a lone leaf; wrap it so a Vector is always returned.
    return seq isa Integer ? [seq] : seq
end

function contraction_sequence(tensors::Vector{<:Tensor}; alg = "optimal", kwargs...)
    return contraction_sequence(Algorithm(alg), tensors; kwargs...)
end

# Contract `tensors` in an optimal order (for the small lists of a vertex and its environment).
contract_optimal(tensors::Vector{<:Tensor}) = contract(tensors; sequence = contraction_sequence(tensors))

#OMEinsumContractionOrders helpers
function to_eincode(tensors::Vector{<:Tensor})
    ixs = map(t -> collect(inds(t)), tensors)
    LT = eltype(eltype(ixs))
    #`reduce` over a single tensor returns the tensor itself, not its indices; a one-tensor
    #network is trivial and its open indices are all of that tensor's indices.
    iy = length(tensors) == 1 ? collect(LT, inds(only(tensors))) : collect(LT, reduce(noncommoninds, tensors))
    size_dict = Dict{LT, Int}(i => dim(i) for ix in ixs for i in ix)
    return EinCode(ixs, iy), size_dict
end

function to_contraction_sequence(ne::NestedEinsum)
    OMEinsumContractionOrders.isleaf(ne) && return ne.tensorindex
    return map(to_contraction_sequence, ne.args)
end
