using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: add_edges, add_vertices

using Random
using TOML

using JLD2

include("utils.jl")

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

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
  
function hyperhoneycomb_graph(L; kwargs...)
      file = "/mnt/home/jtindall/ceph/Data/StructureFactors/Hyperhoneycomb/LatticeFiles/hyperhoneycomb."*string(L)*".pbc.HB.Kitaev.nosyminfo.toml"
      data = TOML.parsefile(file)
      interactions = data["Interactions"]
      heisenberg_interactions = filter(d -> first(d) == "HB", interactions)
      g = build_graph_from_interactions(heisenberg_interactions; kwargs...)
      return g
end

function main(i::Int64, maxdim::Int64)

    #Get the graph and interactions from the .tomls. Flag ensures Vertices are ordered consistent with the .toml file
    g = TN.named_biclique(3,3)

    ec = edge_color(g, 3)

    n = 144
    θ = (2 * pi * i) / (n)
    K, J = sin(θ), cos(θ)
    println("Beginning simulation with theta = $(θ), J = $(J), K = $(K) and a maxdim of $(maxdim).")

    s = ITN.siteinds("S=1/2", g; conserve_qns = false)
    Random.seed!(1234)
    ψ = ITN.ITensorNetwork(v -> "X+", s)
    #ψ = ITN.random_tensornetwork(s; link_space = 1)

    cutoff = 1e-12
    apply_kwargs = (; maxdim, cutoff, normalize = true)
    # #Parameters for BP, as the graph is not a tree (it has loops), we need to specify these
    set_global_bp_update_kwargs!(;
        maxiter = 30,
        tol = 1e-10,
        message_update_kwargs = (;
            message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))
        ),
    )

    no_eras = 8
    xx_observables, yy_observables, zz_observables = honeycomb_kitaev_heisenberg_observables(J, K, ec)
    layer_generating_function = δβ -> honeycomb_kitaev_heisenberg_layer(J, K, δβ, ec)
    obs = [xx_observables; yy_observables; zz_observables]
    energy_calculation_function = ψψ -> sum(real.(expect(ψψ, obs)))

    ψ, ψψ, energy = imaginary_time_evolution(ψ, layer_generating_function, energy_calculation_function, no_eras; apply_kwargs);

    local_zs = expect(ψψ, [("Z", [v]) for v in vertices(ψ)])

    sum_z = sum(local_zs)

    # zzs = expect(ψψ, zz_observables)
    # yys = expect(ψψ, yy_observables)
    # xxs = expect(ψψ, xx_observables)
    # @show zzs, yys, xxs

    file_name = "i"*string(i)*"maxdim"*string(maxdim)
    jldsave("/mnt/home/jtindall/ceph/Data/StructureFactors/Hyperhoneycomb/GroundStateWavefunctions/"*file_name*".jld2"; wavefunction = ψ, bp_cache = ψψ, energy = energy, Z = sum_z)

end

#i, maxdim = 0, 4
i, maxdim = parse(Int64, ARGS[1]), parse(Int64, ARGS[2])
main(i, maxdim)