# Dense / symmetry compatibility shims.
#
# ITensor code falls back to dense storage with `dense`/`denseblocks` and branches
# on `hasqns`. For the `CartesianSpace` (dense) backend these are no-ops: there is
# no block-sparse structure to drop, and `hasqns` is false. Defined generically so
# they stay correct (and future-proof) for graded spaces.

export dense, denseblocks, hasqns

"""
    dense(t::ITensorMap)

No-op for the dense backend (already a dense `TensorMap`). Present for ITensor
source compatibility.
"""
dense(t::ITensorMap) = t

"""
    denseblocks(t::ITensorMap)

No-op for the dense backend (no block-sparse structure to densify).
"""
denseblocks(t::ITensorMap) = t

"""
    hasqns(t::ITensorMap) -> Bool
    hasqns(i::Index) -> Bool

Whether the tensor/index carries symmetry (QN) sectors. `false` for the dense
`CartesianSpace` backend.
"""
hasqns(t::ITensorMap) = !(spacetype(t) <: CartesianSpace)
hasqns(i::Index) = !(spacetype(i) <: CartesianSpace)
