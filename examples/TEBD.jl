using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

function main()
    nx, ny, nz = 3, 3, 3
    #Build a qubit layout of a 3x3x3 periodic cube
    g = named_grid((nx, ny, nz); periodic = true)

    nqubits = length(vertices(g))
    s = ITN.siteinds("S=1/2", g)
    ψ = ITensorNetwork(v -> "Z+", s)

    maxdim, cutoff = 4, 1e-14
    apply_kwargs = (; maxdim, cutoff, normalize = true)
    #Parameters for BP, as the graph is not a tree (it has loops), we need to specify these
    set_global_bp_update_kwargs!(;
        maxiter = 30,
        tol = 1e-10,
        message_update_kwargs = (;
            message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))
        ),
    )

    ψψ = build_bp_cache(ψ)
    h, J = -1.0, -1.0
    no_trotter_steps = 25
    δt = 0.04

    #Do a 7-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    ec = edge_color(g, 7)
    append!(layer, ("Rz", [v], h*δt) for v in vertices(g))
    for colored_edges in ec
        append!(layer, ("Rxx", pair, 2*J*δt) for pair in colored_edges)
    end
    append!(layer, ("Rz", [v], h*δt) for v in vertices(g))

    #Vertices to measure "Z" on
    vs_measure = [first(center(g))]
    observables = [("Z", [v]) for v in vs_measure]

    #Edges to measure bond entanglement on:
    e_ent = first(edges(g))

    χinit = maxlinkdim(ψ)
    println("Initial bond dimension of the state is $χinit")

    expect_sigmaz = real.(expect(ψ, observables; (cache!) = Ref(ψψ)))
    println("Initial Sigma Z on selected sites is $expect_sigmaz")

    time = 0

    Zs = Float64[]

    # evolve! The first evaluation will take significantly longer because of compilation.
    for l = 1:no_trotter_steps
        #printing
        println("Layer $l")

        # pass BP cache manually
        t = @timed ψ, ψψ, errors =
            apply(layer, ψ, ψψ; apply_kwargs, verbose = false);

        # push expectation to list
        push!(Zs, only(real(expect(ψ, observables; (cache!) = Ref(ψψ)))))

        # printing
        println("Took time: $(t.time) [s]. Max bond dimension: $(maxlinkdim(ψ))")
        println("Maximum Gate error for layer was $(maximum(errors))")
        println("Sigma z on central site is $(last(Zs))")
    end
end

main()
