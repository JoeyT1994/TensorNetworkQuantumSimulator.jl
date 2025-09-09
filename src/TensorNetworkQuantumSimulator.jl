module TensorNetworkQuantumSimulator


include("imports.jl")
include("Backend/beliefpropagation.jl")
include("Backend/loopcorrection.jl")
include("Backend/boundarymps.jl")

# a helpful union types for the caches that we use
const CacheNetwork = Union{AbstractBeliefPropagationCache,BoundaryMPSCache}
const TensorNetwork = Union{AbstractITensorNetwork,CacheNetwork}


include("graph_ops.jl")
include("utils.jl")
include("constructors.jl")
include("gates.jl")
include("apply.jl")
include("apply_sqrt.jl")
include("expect.jl")
include("sample.jl")


export
    updatecache,
    build_normsqr_bp_cache,
    vertices,
    edges,
    apply,
    truncate,
    expect,
    expect_boundarymps,
    expect_loopcorrect,
    fidelity,
    fidelity_boundarymps,
    fidelity_loopcorrect,
    make_hermitian,
    build_normsqr_bp_cache,
    truncate,
    maxlinkdim,
    siteinds,
    edge_color,
    zerostate,
    getnqubits,
    named_grid,
    sample

end
