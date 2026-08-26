using OMEinsumContractionOrders: OMEinsumContractionOrders, optimize_code, EinCode, NestedEinsum, TreeSA, GreedyMethod, SABipartite, Treewidth, ExactTreewidth, HyperND, ExhaustiveSearch

# The exact "optimal" contraction order (Pfeifer 2014 netcon) is provided by
# OMEinsumContractionOrders' `ExhaustiveSearch` optimizer, which ported this routine from
# TensorOperations. It handles trivial 1-/2-tensor inputs directly, so the previous
# trivial-tensor pruning and scalar-`Int` workarounds are no longer needed.
function contraction_sequence(::Algorithm"optimal", tensors::Vector)
    return contraction_sequence(Algorithm("omeinsum"), tensors; optimizer = ExhaustiveSearch())
end

function contraction_sequence(::Algorithm"omeinsum", tensors::Vector; optimizer = TreeSA())
    code, size_dict = to_eincode(tensors)
    optcode = optimize_code(code, size_dict, optimizer)
    seq = to_contraction_sequence(optcode)
    #A single-tensor network optimizes to a lone leaf; wrap it so a Vector is always returned.
    return seq isa Integer ? [seq] : seq
end

function contraction_sequence(tensors::Vector; alg = "optimal", kwargs...)
    return contraction_sequence(Algorithm(alg), tensors; kwargs...)
end

#OMEinsumContractionOrders helpers
function to_eincode(tensors::Vector)
    #identity.() narrows the abstract-eltype index vectors to the concrete index type; the
    #omeinsum machinery needs a concrete, consistent label type
    ixs = map(t -> identity.(collect(inds(t))), tensors)
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
