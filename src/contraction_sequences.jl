using ITensors: Index, ITensor, @Algorithm_str, inds, noncommoninds, dim
using TensorOperations: TensorOperations, optimaltree
using EinExprs: EinExprs, EinExpr, einexpr, SizedEinExpr

itensors(fts::Vector{<:FermionicITensor}) = ITensors.ITensor.(fts)
itensors(ts::Vector{<:ITensor}) = ts

# A trivial (scalar, index-free) replacement of the same tensor type. Only the
# indices of the pruned tensors are used to compute the contraction sequence, so
# the numerical content is irrelevant.
trivial_tensor(t::ITensor) = adapt(datatype(t))(ITensor(1))
function trivial_tensor(t::FermionicITensor)
    return FermionicITensor(adapt(datatype(t))(ITensor(1)), Index[], Bool[], Dictionary{Index, Vector{Bool}}())
end

function prune_trivial_tensors(tensors::Vector{<:Tensor})
    pruned_tensors = copy(tensors)
    for (i, t) in enumerate(pruned_tensors)
        if all(d -> d == 1, dim.(inds(tensors[i])))
            pruned_tensors[i] = trivial_tensor(t)
        end
    end
    return pruned_tensors
end

function contraction_sequence(::Algorithm"optimal", tensors::Vector{<:Tensor}; prune_tensors = false)
    #Needed because tensor operations bugs on trivial tensors
    if prune_tensors
        ITensors.disable_warn_order()
        tensors = prune_trivial_tensors(tensors)
    end
    network = collect.(inds.(tensors))
    #Converting dims to Float64 to minimize overflow issues
    inds_to_dims = Dict(i => Float64(dim(i)) for i in unique(Iterators.flatten(network)))
    seq, _ = optimaltree(network, inds_to_dims)
    seq = typeof(seq) <: Int ? [seq] : seq
    return seq
end

function contraction_sequence(
        ::Algorithm"einexpr", tensors::Vector{<:Tensor}; optimizer = EinExprs.Exhaustive()
    )
    expr = to_einexpr(tensors)
    path = einexpr(optimizer, expr)
    return to_contraction_sequence(path, tensor_inds_to_vertex(tensors))
end

function contraction_sequence(tensors::Vector{<:Tensor}; alg = "optimal", kwargs...)
    return contraction_sequence(Algorithm(alg), tensors; kwargs...)
end

#Ein Exprs helpers
function to_einexpr(tensors::Vector{<:Tensor})
    tensors = itensors(tensors)
    IndexType = Any

    tensor_exprs = EinExpr{IndexType}[]
    inds_dims = Dict{IndexType, Int}()

    for tensor_v in tensors
        inds_v = collect(inds(tensor_v))
        push!(tensor_exprs, EinExpr{IndexType}(; head = inds_v))
        merge!(inds_dims, Dict(inds_v .=> Tuple(dim.(inds_v))))
    end

    externalinds_tn = reduce(noncommoninds, tensors)
    return SizedEinExpr(sum(tensor_exprs; skip = externalinds_tn), inds_dims)
end

function tensor_inds_to_vertex(tensors::Vector{<:Tensor})
    tensors = itensors(tensors)
    IndexType = Any
    VertexType = Int

    mapping = Dict{Set{IndexType}, VertexType}()

    for (v, tensor_v) in enumerate(tensors)
        inds_v = collect(inds(tensor_v))
        mapping[Set(inds_v)] = v
    end

    return mapping
end


function to_contraction_sequence(expr, tensor_inds_to_vertex)
    EinExprs.nargs(expr) == 0 && return tensor_inds_to_vertex[Set(expr.head)]
    return map(
        expr -> to_contraction_sequence(expr, tensor_inds_to_vertex), EinExprs.args(expr)
    )
end
