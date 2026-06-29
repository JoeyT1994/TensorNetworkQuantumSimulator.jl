# Compatibility layer providing the legacy `ITensors.jl` API that TNQS was written
# against, implemented over the next-gen `ITensorBase.jl` backend. Wrapped as a
# submodule so it can later be lifted out of TNQS into a standalone package of the
# same name (the migration aid for the whole ecosystem); the export list below is the
# spec for what that package must cover.
#
# Export vs `public`: the eventual standalone package, used as
# `using ITensorBase; using ITensorsITensorBaseCompat`, would declare the names
# ITensorBase also exports (`apply`, `prime`, `noprime`, `state`) as `public` rather
# than `export`ed, so they don't shadow ITensorBase's versions — a consumer wanting
# the legacy semantics imports them explicitly. Here every name is exported instead:
# TNQS does not `using ITensorBase` broadly, so there is no collision to avoid, and
# `public` would force Julia >= 1.11 while this package still supports 1.10. The
# main module imports `contract` / `inner` / `scalartype` / `datatype` (the generics
# TNQS adds methods to) so it can extend them.
module ITensorsITensorBaseCompat

# Legacy `ITensors`/`ITensorMPS` exported `truncate`; TNQS extends it. Bind it to
# `Base.truncate` (which is always in scope, so a `using` consumer sees one
# unambiguous binding) and re-export, so `ITensors.truncate` resolves through this
# module and `function ITensors.truncate(...)` extends `Base.truncate`.
import Base: truncate

include("itensors.jl")
include("ops.jl")

export
    # Index access and set algebra
    inds, commoninds, commonind, uniqueinds, noncommonind, noncommoninds, unioninds, hascommoninds,
    # Index operations
    sim, dag, prime, noprime, replaceind, replaceinds, dim, swapind,
    # ITensor construction
    itensor, random_itensor, scalar, delta, onehot, combiner, combinedind,
    # Factorizations
    qr, svd, eigen, factorize, factorize_svd,
    # Diagonal manipulation
    map_diag, map_diag!,
    # Storage / element-type accessors
    scalartype, datatype, array, data,
    # Dense / quantum-number no-ops
    denseblocks, dense, hasqns,
    # Contraction, inner product, gate application
    contract, inner, apply,
    # Direct sum and misc legacy helpers
    directsum, disable_warn_order,
    # Algorithm dispatch tag
    Algorithm, @Algorithm_str,
    # Tags
    hastags,
    # Operator / named-state system
    state, op, OpName, SiteType, @OpName_str, @SiteType_str,
    # Bond truncation (bound to `Base.truncate`)
    truncate

end
