using TensorNetworkQuantumSimulator

using ITensors

function main()
    nx, ny, nz = 3, 3, 3
    #Build a qubit layout of a 3x3x3 periodic cube
    g = named_grid((nx, ny, nz); periodic = true)

    nqubits = length(vertices(g))
    ψ0 = tensornetworkstate(ComplexF32, v -> "↑", g, "S=1/2")

    maxdim, cutoff = 4, 1.0e-10
    apply_kwargs = (; maxdim, cutoff, normalize_tensors = true)

    ψ_bpc = BeliefPropagationCache(ψ0)
    h, J = -1.0, -1.0
    no_trotter_steps = 25
    δt = 0.04

    #Do a 7-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    ec = edge_color(g, 7)
    append!(layer, ("Rz", [v], h * δt) for v in vertices(g))
    for colored_edges in ec
        append!(layer, ("Rxx", pair, 2 * J * δt) for pair in colored_edges)
    end
    append!(layer, ("Rz", [v], h * δt) for v in vertices(g))

    #Vertices to measure "Z" on
    vs_measure = [first(center(g))]
    observables = [("Z", [v]) for v in vs_measure]

    #Edges to measure bond entanglement on:
    e_ent = first(edges(g))

    χinit = maxvirtualdim(ψ_bpc)
    println("Initial bond dimension of the state is $χinit")

    expect_sigmaz = real.(expect(ψ_bpc, observables))
    println("Initial Sigma Z on selected sites is $expect_sigmaz")

    time = 0

    Zs = []

    # evolve! The first evaluation will take significantly longer because of compilation.
    for l in 1:no_trotter_steps
        #printing
        println("Layer $l")

        # pass BP cache manually
        t = @timed ψ_bpc, errors =
            apply_gates(layer, ψ_bpc; apply_kwargs, verbose = false)

        # push BP measured expectation to list
        push!(Zs, only(real(expect(ψ_bpc, observables))))

        # printing
        println("Took time: $(t.time) [s]. Max bond dimension: $(maxvirtualdim(ψ_bpc))")
        println("Maximum Gate error for layer was $(maximum(errors))")
        println("Sigma z on central site is $(last(Zs))")
    end
    return
end

main()
