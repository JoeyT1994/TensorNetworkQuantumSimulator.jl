# Combiner / fusion isometry for `ITensorMap`.
#
# `combiner(is...)` fuses the legs `is` into a single leg. It returns an
# `ITensorMap` whose **first leg is the fused index** and whose remaining legs are
# `dag.(is)`, so that `t * combiner(is...)` contracts away the `is` legs (matched
# by id, mutually dual) and leaves the fused leg. `combinedind(c)` recovers the
# fused index. For dense `CartesianSpace`, `dim(fused) == prod(dim(is))`.

export combiner, combinedind

"""
    combiner(is::Index...) -> ITensorMap
    combiner(is) -> ITensorMap                # is: any collection of Index

A fusion isometry combining the legs `is` into one. The returned tensor's first leg
is the fused index (see [`combinedind`](@ref)); its other legs are `dag.(is)`, so
contracting it with a tensor carrying `is` replaces those legs with the fused one.
"""
combiner(i1::Index, is::Index...) = _combiner((i1, is...))
combiner(is) = _combiner(Tuple(is))

function _combiner(isv::Tuple)
    isempty(isv) && throw(ArgumentError("combiner requires at least one index"))
    p = ProductSpace(map(space, isv)...)
    fused = fuse(p)
    c = Index(fused)
    W = isomorphism(fused ← p)
    return unsafe_itensormap(W, (c, map(dag, isv)...))
end

"""
    combinedind(c::ITensorMap) -> Index

The fused index of a combiner `c` (its first leg).
"""
combinedind(c::ITensorMap) = first(inds(c))
