# Contraction of `ITensorMap` networks.
#
# `contract` matches legs by `(id, plev)`, builds `ncon`'s integer label lists
# (contracted → positive, open → negative with magnitude fixing the output order), and
# dispatches to TensorOperations' dynamic `ncon`. The contraction order is chosen by a
# pluggable `ContractionSequenceAlg` finder (default: TensorOperations' optimizer for
# ≥3 tensors). Depends on `ITensorMap`/`unsafe_itensormap`/`inds` from itensor.jl.

export contract, ContractionSequenceAlg, TensorOperationsSequence, contraction_sequence

# --- contraction-order finders ----------------------------------------------
"""
    ContractionSequenceAlg

Abstract supertype for algorithms that determine the order in which [`contract`](@ref)
eliminates the shared legs of a network. Plug in a new finder by defining
[`contraction_sequence`](@ref) for a subtype.
"""
abstract type ContractionSequenceAlg end

"""
    TensorOperationsSequence()

Default contraction-order finder. For three or more tensors it runs TensorOperations'
cost-based optimizer (`optimaltree`) over the leg dimensions; for one or two tensors
the order is irrelevant and it defers to `ncon`'s default.
"""
struct TensorOperationsSequence <: ContractionSequenceAlg end

"""
    contraction_sequence(alg::ContractionSequenceAlg, tensors, indexlist) -> order

The `order` argument passed to `ncon` for the network of `tensors` labelled by
`indexlist` (a vector of the contracted/positive labels in elimination order, or
`nothing` for `ncon`'s default), as chosen by `alg`.
"""
function contraction_sequence(::TensorOperationsSequence, tensors, indexlist)
    length(tensors) <= 2 && return nothing
    optdata = Dict{Int, Int}()
    for (t, labels) in zip(tensors, indexlist), (k, l) in enumerate(labels)
        optdata[l] = dim(space(t, k))
    end
    tree, _ = TensorOperations.optimaltree(indexlist, optdata)
    order, _ = TensorOperations.tree2indexorder(tree, indexlist)
    return isempty(order) ? nothing : convert(Vector{Int}, order)
end

# --- contraction ------------------------------------------------------------
"""
    contract(t1::ITensorMap, ts::ITensorMap...; alg=TensorOperationsSequence(), kwargs...)

Contract a network of tensors over their shared legs (matched by `(id, plev)`),
returning an `ITensorMap` carrying the open legs (those appearing on a single tensor)
in first-appearance order. `*` is the binary alias.

A leg shared by two tensors is summed; a leg appearing once is left open.
An index shared by more than two tensors (a hyperedge) is unsupported.
Dispatches to TensorOperations' dynamic `ncon`, which handles the permutes/recoupling;
contracted legs must be mutually dual (an invariant of the bond orientation).

The contraction order is chosen by `alg` (see [`ContractionSequenceAlg`](@ref)).

The other keywords are passed on to `ncon`.
"""
function contract(
        t1::ITensorMap, ts::ITensorMap...;
        alg::ContractionSequenceAlg = TensorOperationsSequence(), kwargs...
    )::ITensorMap
    tensors = (t1, ts...)
    allinds = reduce(vcat, (collect(inds(t)) for t in tensors))
    nocc(i) = count(==(i), allinds)
    any(>(2) ∘ nocc, allinds) && throw(
        ArgumentError(
            "contract: an index is shared by more than two tensors (hyperedges unsupported)"
        )
    )

    openinds = [i for i in allinds if nocc(i) == 1]        # result legs (output order)
    contracted = unique(i for i in allinds if nocc(i) == 2)

    label(i) = let k = findfirst(==(i), contracted)
        isnothing(k) ? -findfirst(==(i), openinds) : k
    end
    indexlist = [Int[label(i) for i in inds(t)] for t in tensors]

    tensordata = map(t -> t.data, tensors)
    order = contraction_sequence(alg, tensordata, indexlist)
    data = ncon(tensordata, indexlist; order, kwargs...)

    if data isa Number # full contraction - embed as tensor again
        z = fill!(zeros(typeof(data), one(spacetype(t1))), data)
        return unsafe_itensormap(z, ())
    end
    return unsafe_itensormap(data, openinds)
end

Base.:*(A::ITensorMap, B::ITensorMap...) = contract(A, B...)

"""
    contract(tensors::AbstractVector{<:ITensorMap}; sequence=nothing, alg=TensorOperationsSequence())

Contract a list of tensors. The `sequence` keyword (an externally computed
contraction order, e.g. from ITensors/EinExprs) is accepted for source
compatibility but ignored — the order is chosen by `alg`. Returns the contracted
`ITensorMap` (a 0-leg tensor for a full contraction; extract with `scalar`/`[]`).
"""
function contract(tensors::AbstractVector; sequence = nothing, kwargs...)
    isempty(tensors) && throw(ArgumentError("contract: empty tensor list"))
    return contract(tensors...; kwargs...)
end

"""
    disable_warn_order()

No-op compatibility shim (TensorOperations issues no high-rank warning).
"""
disable_warn_order() = nothing
export disable_warn_order
