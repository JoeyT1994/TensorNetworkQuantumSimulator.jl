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

function hexagonal_kitaev_layer(Jx::Float64, Jy::Float64, Jz::Float64, δβ::Float64, ec)
    layer = []
    for (i, colored_edges) in enumerate(ec)
        if i == 1
            append!(layer, ("Rxx", pair, -2*Jx*δβ * im) for pair in colored_edges)
        elseif i == 2
            append!(layer, ("Ryy", pair, -2*Jy*δβ * im) for pair in colored_edges)
        elseif i == 3
            append!(layer, ("Rzz", pair, -2*Jz*δβ * im) for pair in colored_edges)
        end
    end
    return layer
end

function hexagonal_kitaev_observables(Jx::Float64, Jy::Float64, Jz::Float64, ec)
    xx_observables = [("XX", pair, Jx) for pair in ec[1]]
    yy_observables = [("YY", pair, Jy) for pair in ec[2]]
    zz_observables = [("ZZ", pair, Jz) for pair in ec[3]]
    return xx_observables, yy_observables, zz_observables
end

function main()
    nx, ny = 4,4
    #Build a qubit layout of a 3x3x3 periodic cube
    g =named_hexagonal_lattice_graph(nx, ny; periodic = true)

    nqubits = length(vertices(g))
    s = ITN.siteinds("S=1/2", g)
    ψ = ITensorNetwork(v -> "X+", s)

    maxdim, cutoff = 6, 1e-12
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
    Jx, Jy, Jz = 1.0, 1.0, 1.0
    δβs = [0.5, 0.1, 0.01, 0.001]
    n_steps_per_period = [50, 100, 100, 100]
    no_periods = length(δβs)

    ec = edge_color(g, 3)

    xx_observables, yy_observables, zz_observables = hexagonal_kitaev_observables(Jx, Jy, Jz, ec)

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
    for l = 1:no_periods
        #printing
        println("Period $l")
        δβ = δβs[l]


        t = @timed for step in 1:n_steps_per_period[l]

            layer = hexagonal_kitaev_layer(Jx, Jy, Jz, δβ, ec)

            # pass BP cache manually
            # only update cache every `update_every` overlapping 2-qubit gates
            ψ, ψψ, errors =
                apply(layer, ψ, ψψ; apply_kwargs, update_every = 1, verbose = false);

        end

        # printing
        println("Took time: $(t.time) [s]. Max bond dimension: $(maxlinkdim(ψ))")
        expect_sigmaz = real.(expect(ψ, observables; (cache!) = Ref(ψψ)))

        zzs = real.(expect(ψ, zz_observables; (cache!) = Ref(ψψ)))
        yys = real.(expect(ψ, yy_observables; (cache!) = Ref(ψψ)))
        xxs = real.(expect(ψ, xx_observables; (cache!) = Ref(ψψ)))
        total_energy = sum(zzs) + sum(yys) + sum(xxs)
        println("Sigma Z on selected site is $expect_sigmaz")

        println("Total energy is $(total_energy)")

        @show zzs
    end
end

main()