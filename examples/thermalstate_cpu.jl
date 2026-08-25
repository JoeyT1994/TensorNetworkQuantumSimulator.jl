using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator, freenergy

using NamedGraphs
using NamedGraphs.GraphsExtensions: add_edges, add_vertices
using Graphs

using Base.Threads
using LinearAlgebra
using NPZ
using Adapt: adapt
using TOML
using ITensors
using ITensors: @OpName_str, @SiteType_str, op

BLAS.set_num_threads(nthreads())
#BLAS.set_num_threads(1)
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

# user-specified finite size cluster, position in phase diagram, bond dimension and output directory
const N_spins = 8#parse(Int, ARGS[1])
const ver = 1#parse(Int, ARGS[2])
const phase_diagram_i = 0#parse(Int, ARGS[3])
const maxdim = 16#parse(Int, ARGS[4])
const evolve_maxiter = 100#parse(Int, ARGS[5])
const io_dir = "C:/Users/Joey/.julia/dev/TensorNetworkQuantumSimulator/examples/"#ARGS[6]
const phase_diagram_n = 144 # number of equidistant points sampled inside the HB-Kitaev phase diagram

# fixed imaginary time evolution parameters
const δβ = 0.005

const monitor_interval = 25
const cutoff = 1e-12 # truncation cutoff when applying simple update gates

# fixed BP standard update kwargs
const max_msg_iters = 250
const msg_residue_tol = 1e-7
const update_kwargs = (;
    maxiter = max_msg_iters,
    tolerance = msg_residue_tol,
    verbose = false
    )

# fixed apply() kwargs
const apply_kwargs = (; maxdim, cutoff, normalize_tensors = false)

# ==========================================================
#  FUNCTIONS THAT IMPLEMENT HYPERHONEYCOMB HB-KITAEV MODEL
# ==========================================================

"""
    ITensors.op(::OpName"Rxxyyzz", ::SiteType"S=1/2"; θ::Number)

Gate for rotation by XXYYZZ at given angles
"""
function ITensors.op(::OpName"Rxxyyzz", ::SiteType"S=1/2", s1::Index, s2::Index; θxx =1, θyy =1, θzz = 1)
    h = 0.5*(θxx*op("X", s1) * op("X", s2) + θyy*op("Y",s1) * op("Y", s2) + θzz*op("Z", s1) * op("Z", s2))
    return exp( -im * h)
end

function thermal_to_rxxyyzz(sinds, pair, θxx, θyy, θzz)
    return ITensors.op("Rxxyyzz", sinds[src(pair)][1], sinds[dst(pair)][1]; θxx, θyy, θzz)
end

function thermal_honeycomb_kitaev_heisenberg_layer(sinds, J::Float64, K::Float64, δβ::Float64, ec)
    layer = ITensor[]
    append!(layer, [thermal_to_rxxyyzz(sinds, pair, -(K + J)*δβ * im, -(J)*δβ * im, -(J)*δβ * im) for pair in ec[1]])
    append!(layer, [thermal_to_rxxyyzz(sinds, pair, -(J)*δβ * im, -(K + J)*δβ * im, -(J)*δβ * im) for pair in ec[2]])
    append!(layer, [thermal_to_rxxyyzz(sinds, pair, -2*(J)*δβ * im, -2*(J)*δβ * im, -2*(K + J)*δβ * im) for pair in ec[3]])
    append!(layer, [thermal_to_rxxyyzz(sinds, pair, -(J)*δβ * im, -(K + J)*δβ * im, -(J)*δβ * im) for pair in ec[2]])
    append!(layer, [thermal_to_rxxyyzz(sinds, pair, -(K + J)*δβ * im, -(J)*δβ * im, -(J)*δβ * im) for pair in ec[1]])
    return layer
end

# read edge coloring from Kitaev TOML file
function kitaev_ec(toml)
    data = TOML.parsefile(toml)
    interactions = data["Interactions"]
    kitaev_x_interactions = [NamedEdge((x[3],) => (x[4],)) for x in filter(d -> first(d) == "KX", interactions)]
    kitaev_y_interactions = [NamedEdge((x[3],) => (x[4],)) for x in filter(d -> first(d) == "KY", interactions)]
    kitaev_z_interactions = [NamedEdge((x[3],) => (x[4],)) for x in filter(d -> first(d) == "KZ", interactions)]
    return vcat([kitaev_x_interactions], [kitaev_y_interactions], [kitaev_z_interactions])
end

#Construct a graph with edges everywhere a two-site gate appears.
function build_graph_from_interactions(list; sort_vertices = false)
    vertices = []
    edges = []
    for term in list
        vsrc, vdst = (term[3],), (term[4],)
        if vsrc ∉ vertices
            push!(vertices, vsrc)
        end
        if vdst ∉ vertices
            push!(vertices, vdst)
        end
        e = NamedEdge(vsrc => vdst)
        if e ∉ edges || reverse(e) ∉ edges
            push!(edges, e)
        end
    end
    g = NamedGraph()
    if sort_vertices
      vertices = sort(vertices; by = v -> first(v))
    end
    g = add_vertices(g, vertices)
    g = add_edges(g, edges)
    return g
end
  
function hyperhoneycomb_graph(toml; kwargs...)
    data = TOML.parsefile(toml)
    interactions = data["Interactions"]
    heisenberg_interactions = filter(d -> first(d) == "J", interactions)
    g = build_graph_from_interactions(heisenberg_interactions; kwargs...)
    return g
end




# ==========================================================
#                        MAIN FUNCTION
# ==========================================================

# main script to perform imaginary time evolution of the hyperhoneycomb Kitaev-Heisenberg model
function main()

    toml = joinpath(io_dir, "hyperhoneycomb-N-$N_spins-ver-$ver.toml")
    g = hyperhoneycomb_graph(toml; sort_vertices = true)
    ec = kitaev_ec(toml)
    
    # get system parameters
    θ = (2 * pi * phase_diagram_i) / (phase_diagram_n)
    K, J = 2*sin(θ), cos(θ)

    # initialize at β = 0 thermal state
    s = siteinds("S=1/2", g; inds_per_site = 2)
    ψ = identity_tensornetworkstate(Float64, g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))

    # define layer generating and energy calculation functions
    imag_time_layer = thermal_honeycomb_kitaev_heisenberg_layer(s, J, K, δβ, ec)
    imag_time_layer = adapt(Vector{<:Float64}).(imag_time_layer)
    
    logz = -freenergy(ψ_bpc)
    rescale!(ψ_bpc)
    for i in 1:evolve_maxiter
        # apply layer (without converging messages)
        ψ_bpc, errs = apply_gates(imag_time_layer, ψ_bpc; apply_kwargs, update_cache=false)
        # converge messages in a controlled manner (passing tolerance etc.)
        ψ_bpc = update(ψ_bpc; update_kwargs...)
        # measure observables
        logz -= freenergy(ψ_bpc)
        rescale!(ψ_bpc)
        if i % 5 == 0
            β = 2*i*δβ
            f_bp = logz  / length(vertices(g))
            println("β=$(β), f_bp = $(f_bp)")
        end
    end
end

main()