# Direct sum of tensors along chosen legs (network `+`).
#
# `directsum(newinds, t1 => olds1, t2 => olds2)` block-embeds `t1` and `t2` into a
# tensor whose `olds` legs are jointly direct-summed (dims add): `t1` occupies the
# leading block of every summed leg, `t2` the trailing block, zeros elsewhere. The
# non-summed (shared) legs are common to both and kept from `t1`. Returns a tensor
# with legs `(shared..., newinds...)`.

export directsum

# Dense array of `t` with axes permuted to `order`.
function _array_ordered(t::ITensorMap, order)
    perm = map(o -> findfirst(==(o), inds(t)), Tuple(order))
    any(isnothing, perm) && throw(ArgumentError("directsum: an index is not a leg of the tensor"))
    return permutedims(array(t), perm)
end

function directsum(newinds, p1::Pair, p2::Pair)
    t1, olds1 = first(p1), Tuple(last(p1))
    t2, olds2 = first(p2), Tuple(last(p2))
    newinds = Tuple(newinds)
    ns = length(newinds)
    (length(olds1) == ns && length(olds2) == ns) ||
        throw(ArgumentError("directsum: olds1/olds2/newinds must have equal length"))

    keep = uniqueinds(inds(t1), olds1)          # shared (non-summed) legs
    nk = length(keep)
    # arrays ordered (keep..., olds...); t2's shared legs match t1's by identity
    A1 = _array_ordered(t1, (keep..., olds1...))
    A2 = _array_ordered(t2, (keep..., olds2...))

    kdims = map(dim, keep)
    d1 = map(dim, olds1)
    d2 = map(dim, olds2)
    newdims = map(dim, newinds)
    all(newdims .== d1 .+ d2) ||
        throw(ArgumentError("directsum: newinds dims must equal summed olds dims ($(newdims) vs $(d1 .+ d2))"))

    R = zeros(promote_type(scalartype(t1), scalartype(t2)), kdims..., newdims...)
    colons = ntuple(_ -> Colon(), nk)
    R[colons..., map(d -> 1:d, d1)...] = A1
    R[colons..., map(i -> (d1[i] + 1):(d1[i] + d2[i]), 1:ns)...] = A2

    return itensor(R, (keep..., newinds...))
end
