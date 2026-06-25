# Contraction of `ITensorMap` networks.
#
# `contract` matches legs by `(id, plev)`, builds `ncon`'s integer label lists
# (contracted → positive, open → negative with magnitude fixing the output order), and
# dispatches to TensorOperations' dynamic `ncon`. The contraction order is taken from a
# `sequence` (a pairwise tree of tensor positions, see contraction_sequences.jl) — either
# supplied by the caller or computed by `contraction_sequence(...; alg)` (default
# `"optimal"`) — and flattened into `ncon`'s elimination `order` by `_contraction_order`.
# Depends on `ITensorMap`/`unsafe_itensormap`/`inds` from itensor.jl and
# `contraction_sequence` from contraction_sequences.jl.

export contract

Base.@nospecialize
# --- contraction ------------------------------------------------------------
"""
    contract(t1::ITensorMap, ts::ITensorMap...; sequence=nothing, alg="optimal", kwargs...)

Contract a network of tensors over their shared legs (matched by `(id, plev)`),
returning an `ITensorMap` carrying the open legs (those appearing on a single tensor)
in first-appearance order. `*` is the binary alias.

A leg shared by two tensors is summed; a leg appearing once is left open.
An index shared by more than two tensors (a hyperedge) is unsupported.
Dispatches to TensorOperations' dynamic `ncon`, which handles the permutes/recoupling;
contracted legs must be mutually dual (an invariant of the bond orientation).

The contraction order is the `sequence` (a pairwise tree of tensor positions); if
`nothing`, it is computed via [`contraction_sequence`](@ref) with the given `alg`.
The other keywords are passed on to `ncon`.
"""
function contract(
        t1::ITensorMap, ts::ITensorMap...;
        sequence = nothing, alg = "optimal", kwargs...
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

    order = if length(tensors) <= 2
        nothing
    else
        seq = isnothing(sequence) ? contraction_sequence(collect(tensors); alg) : sequence
        _contraction_order(seq, indexlist)
    end

    tensordata = map(t -> t.data, tensors)
    data = ncon(tensordata, indexlist; order, kwargs...)

    if data isa Number # full contraction - embed as tensor again
        z = fill!(zeros(typeof(data), one(spacetype(t1))), data)
        return unsafe_itensormap(z, ())
    end
    return unsafe_itensormap(data, openinds)
end

Base.:*(A::ITensorMap, B::ITensorMap...) = contract(A, B...)

"""
    contract(tensors::AbstractVector; sequence=nothing, alg="optimal", kwargs...)

Contract a list of tensors, returning the contracted `ITensorMap` (a 0-leg tensor for a
full contraction; extract with `scalar`/`[]`). `sequence` (a pairwise tree of tensor
positions, e.g. from [`contraction_sequence`](@ref) or a cache) is honored as the
contraction order; if `nothing`, one is computed via `alg`.
"""
function contract(tensors::AbstractVector; kwargs...)
    isempty(tensors) && throw(ArgumentError("contract: empty tensor list"))
    return contract(tensors...; kwargs...)
end

"""
    _contraction_order(sequence, indexlist) -> order

Flatten a pairwise contraction tree of 1-based tensor positions into the list of
contracted (positive) `ncon` labels in elimination order (or `nothing` for `ncon`'s
default). `indexlist[v]` holds the integer labels of tensor `v`. Backend-agnostic: works
for both `optimaltree` and `omeinsum` trees.
"""
function _contraction_order(sequence, indexlist)
    order = Int[]
    walk(node::Integer) = copy(indexlist[node])            # leaf → that tensor's labels
    function walk(node)                                    # internal node → merge children
        all = reduce(vcat, map(walk, node))
        contracted = [l for l in unique(all) if l > 0 && count(==(l), all) == 2]
        append!(order, contracted)
        return [l for l in all if !(l in contracted)]      # open labels of merged node
    end
    walk(sequence)
    return isempty(order) ? nothing : order
end

"""
    disable_warn_order()

No-op compatibility shim (TensorOperations issues no high-rank warning).
"""
disable_warn_order() = nothing
export disable_warn_order

Base.@specialize
