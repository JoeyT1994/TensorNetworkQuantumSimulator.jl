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

include("utils.jl")

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
      file = pwd()*"/Research/Data/hyperhoneycomb."*string(L)*".pbc.HB.Kitaev.nosyminfo.toml"
      data = TOML.parsefile(file)
      interactions = data["Interactions"]
      heisenberg_interactions = filter(d -> first(d) == "HB", interactions)
      g = build_graph_from_interactions(heisenberg_interactions; kwargs...)
      return g
end

function main()

    #Get the graph and interactions from the .tomls. Flag ensures Vertices are ordered consistent with the .toml file
    L = 16
    g = hyperhoneycomb_graph(L; sort_vertices = true)

    ec = edge_color(g, 3)

    θ = pi + 0.1
    K, J = sin(θ), cos(θ)

    s = ITN.siteinds("S=1/2", g)
    #ψ = ITensorNetwork(v -> "Z+", s)
    ψ = ITN.random_tensornetwork(s; link_space = 1)

    maxdim, cutoff = 4, 1e-12
    apply_kwargs = (; maxdim, cutoff, normalize = true)
    # #Parameters for BP, as the graph is not a tree (it has loops), we need to specify these
    set_global_bp_update_kwargs!(;
        maxiter = 30,
        tol = 1e-10,
        message_update_kwargs = (;
            message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))
        ),
    )

    no_eras = 6
    xx_observables, yy_observables, zz_observables = honeycomb_kitaev_heisenberg_observables(J, K, ec)
    layer_generating_function = δβ -> honeycomb_kitaev_heisenberg_layer(J, K, δβ, ec)
    obs = [xx_observables; yy_observables; zz_observables]
    energy_calculation_function = ψψ -> sum(real.(expect(ψψ, obs)))

    ψ, ψψ = imaginary_time_evolution(ψ, layer_generating_function, energy_calculation_function, no_eras; apply_kwargs);


end

main()