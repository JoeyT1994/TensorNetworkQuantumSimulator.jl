using TensorNetworkQuantumSimulator

using Statistics

using ITensors: ITensor, ITensors

function main()
    #Define the lattice
    g = heavy_hexagonal_lattice(5, 5)

    #Define an edge coloring
    ec = edge_color(g, 3)

    #Define the gate parametrs
    J = pi / 4
    θh = 0.4

    #Now the circuit
    Rx_layer = [("Rx", [v], θh) for v in collect(vertices(g))]
    Rzz_layer = []
    for edge_group in ec
        append!(Rzz_layer, ("Rzz", pair, 2 * J) for pair in edge_group)
    end
    layer = vcat(Rx_layer, Rzz_layer)

    #Depth of the circuit and apply parameters
    no_trotter_steps = 20
    χ = 8
    apply_kwargs = (; cutoff = 1.0e-12, maxdim = χ, normalize_tensors = true)

    #Initial state
    ψ = tensornetworkstate(ComplexF32, v -> "↑", g, "S=1/2")

    #Wrap in BP cache for the environment messages
    ψ_bpc = BeliefPropagationCache(ψ)

    #Do the evolution
    fidelities = []
    for i in 1:no_trotter_steps
        println("Applying gates for Trotter Step $(i)")
        ψ_bpc, errs = apply_gates(layer, ψ_bpc; apply_kwargs)
        fidelity = prod(1.0 .- errs)
        println("Layer fidelity was $(fidelity)")
        push!(fidelities, fidelity)
    end

    println("Total final fidelity is $(prod(fidelities))")
    ntwo_site_gates = length(edges(g)) * no_trotter_steps
    println("Avg gate fidelity is $((prod(fidelities))^((1 / ntwo_site_gates)))")

    central_site = (11, 5)

    #Use BP to get an observable, as we have the BP cache with messages already in it. We can use that.
    sz_bp = expect(ψ_bpc, [("Z", [central_site])])
    println("BP measured magnetisation on central site is $(only(sz_bp))")

    #Use boundary MPS to get same observab;e
    mps_bond_dimension = 10
    ψ = network(ψ_bpc)
    sz_bmps = expect(ψ, [("Z", [central_site])]; alg = "boundarymps", mps_bond_dimension)

    println("Boundary MPS measured magnetisation on central site with MPS rank $(mps_bond_dimension) MPSs is $(only(sz_bmps))")

    #Sample from q(x) and get p(x) / q(x) for each sample too
    nsamples = 50
    bitstrings = sample_directly_certified(ψ, nsamples; alg = "boundarymps", norm_mps_bond_dimension = mps_bond_dimension)

    st_dev = Statistics.std(first.(bitstrings))
    println("Standard deviation of p(x) / q(x) is $(st_dev)")

    #Measure observable with sample approach (use importance sampling to correct)
    sampled_sz = sum([first(b) * (-2 * last(b)[central_site] + 1) for b in bitstrings]) / Statistics.sum(first.(bitstrings))
    return println("Importance sampled value for magnetisation is $(sampled_sz)")

end

main()
