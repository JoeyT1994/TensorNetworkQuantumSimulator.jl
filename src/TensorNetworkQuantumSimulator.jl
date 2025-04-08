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
include("expect.jl")


export
    updatecache,
    build_bp_cache,
    vertices,
    edges,
    apply,
    truncate,
    get_global_bp_update_kwargs,
    set_global_bp_update_kwargs!,
    reset_global_bp_update_kwargs!,
    get_global_boundarymps_update_kwargs,
    set_global_boundarymps_update_kwargs!,
    reset_global_boundarymps_update_kwargs!,
    expect,
    expect_boundarymps,
    expect_loopcorrect,
    fidelity,
    fidelity_boundarymps,
    fidelity_loopcorrect,
    build_boundarymps_cache,
    truncate,
    maxlinkdim,
    siteinds,
    edge_color,
    zerostate,
    getnqubits,
    heavy_hexagonal_lattice_grid,
    named_grid

end
