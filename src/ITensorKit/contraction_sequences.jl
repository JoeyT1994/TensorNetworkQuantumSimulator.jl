# Contraction-order finders for `ITensorMap` networks.
#
# `contraction_sequence(tensors; alg)` returns a nested-vector pairwise contraction tree
# of (1-based) tensor positions — the ITensors `sequence` format. `contract` (contract.jl)
# converts this tree into the `ncon` elimination order via `_contraction_order`.
#
# Two finders are provided: `Algorithm"optimal"` (TensorOperations' `optimaltree`) and
# `Algorithm"einexpr"` (EinExprs). Extend by adding a `contraction_sequence(::Algorithm"…")`
# method. Names (`Algorithm`, `optimaltree`, `EinExpr`, `inds`, `dim`, `noncommoninds`,
# `ITensor`, `datatype`, `adapt`) resolve from the enclosing `ITensorKit` module scope.

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

function contraction_sequence(
        ::Algorithm"einexpr", tensors::AbstractVector; optimizer = EinExprs.Exhaustive()
    )
    expr = to_einexpr(tensors)
    path = einexpr(optimizer, expr)
    return to_contraction_sequence(path, tensor_inds_to_vertex(tensors))
end

function contraction_sequence(tensors::AbstractVector; alg = "optimal", kwargs...)
    return contraction_sequence(Algorithm(alg), tensors; kwargs...)
end

#Ein Exprs helpers
function to_einexpr(tensors::AbstractVector)
    IndexType = Any

    tensor_exprs = EinExpr{IndexType}[]
    inds_dims = Dict{IndexType, Int}()

    for tensor_v in tensors
        inds_v = collect(inds(tensor_v))
        push!(tensor_exprs, EinExpr{IndexType}(; head = inds_v))
        merge!(inds_dims, Dict(inds_v .=> size(tensor_v)))
    end

    externalinds_tn = reduce(noncommoninds, tensors)
    return SizedEinExpr(sum(tensor_exprs; skip = externalinds_tn), inds_dims)
end

function tensor_inds_to_vertex(tensors::AbstractVector)
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
