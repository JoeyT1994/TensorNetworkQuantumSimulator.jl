using .ITensorsITensorBaseCompat: Index, ITensor, @Algorithm_str, inds, noncommoninds, dim
using TensorOperations: TensorOperations, optimaltree
using OMEinsumContractionOrders: OMEinsumContractionOrders, optimize_code, EinCode, NestedEinsum, TreeSA, GreedyMethod, SABipartite, Treewidth, ExactTreewidth, HyperND

# `optimaltree` (TensorOperations) bugs on fully-trivial tensors (every leg dimension 1).
# Such a tensor contributes only unit-cost factors to any contraction, so we drop its
# indices from the optimization network, leaving an empty index set in its positional slot
# so the returned sequence still indexes the original tensor list. Only the index sets feed
# `optimaltree`; the tensors themselves are never contracted here (the caller contracts the
# untouched originals), so no placeholder tensor is needed.
is_trivial_tensor(t::ITensor) = all(i -> dim(i) == 1, inds(t))

# The sequence optimizers only use indices as opaque labels (plus their dimension), so
# hand them the index names: a shared leg is stored nondual on one tensor and dual on the
# other, and comparing whole `Index` objects would see two different labels (or, for a
# space-backed index, fail to compare at all).
function contraction_network(tensors::Vector{<:ITensor}; prune_tensors = false)
    return map(tensors) do t
        is = collect(name.(inds(t)))
        (prune_tensors && is_trivial_tensor(t)) ? empty(is) : is
    end
end

function contraction_sequence(::Algorithm"optimal", tensors::Vector{<:ITensor}; prune_tensors = false)
    network = contraction_network(tensors; prune_tensors)
    #Converting dims to Float64 to minimize overflow issues
    inds_to_dims = Dict(name(i) => Float64(dim(i)) for t in tensors for i in inds(t))
    seq, _ = optimaltree(network, inds_to_dims)
    seq = typeof(seq) <: Int ? [seq] : seq
    return seq
end

function contraction_sequence(::Algorithm"omeinsum", tensors::Vector{<:ITensor}; optimizer = TreeSA())
    code, size_dict = to_eincode(tensors)
    optcode = optimize_code(code, size_dict, optimizer)
    return to_contraction_sequence(optcode)
end

function contraction_sequence(tensors::Vector{<:ITensor}; alg = "optimal", kwargs...)
    return contraction_sequence(Algorithm(alg), tensors; kwargs...)
end

#OMEinsumContractionOrders helpers
function to_eincode(tensors::Vector{<:ITensor})
    ixs = map(t -> collect(name.(inds(t))), tensors)
    LT = eltype(eltype(ixs))
    iy = collect(LT, name.(reduce(noncommoninds, tensors)))
    size_dict = Dict{LT, Int}(name(i) => dim(i) for t in tensors for i in inds(t))
    return EinCode(ixs, iy), size_dict
end

function to_contraction_sequence(ne::NestedEinsum)
    OMEinsumContractionOrders.isleaf(ne) && return ne.tensorindex
    return map(to_contraction_sequence, ne.args)
end
