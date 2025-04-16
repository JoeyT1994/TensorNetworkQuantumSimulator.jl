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

function main()

    Random.seed!(1234)

    #Get the graph and interactions from the .tomls. Flag ensures Vertices are ordered consistent with the .toml file
    g = named_grid((3,3); periodic = true)
    s = siteinds("S=1/2", g)

    ec = edge_color(g, 5)

    #ψ = ITN.random_tensornetwork(s; link_space = 1)
    ψ = ITN.ITensorNetwork(v -> "X+", s; link_space = 1)
    J, h = -1.0, -3.3

    maxdim, cutoff = 6, nothing
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
    x_observables, zz_observables = ising_observables(J, h, ec)
    layer_generating_function = δβ -> ising_layer(J, h, δβ, ec)
    obs = [x_observables; zz_observables]
    energy_calculation_function = ψψ -> sum(real.(expect(ψψ, obs)))

    ψ, ψψ = imaginary_time_evolution(ψ, layer_generating_function, energy_calculation_function, no_eras; apply_kwargs, tol = nothing);

    z_observables = [("Z", [v]) for v in vertices(ψ)]
    zs = expect(ψψ, z_observables)
    @show zs


end

main()