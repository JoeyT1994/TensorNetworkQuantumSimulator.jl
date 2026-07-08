using TensorNetworkQuantumSimulator

using Statistics

function z2_project(tns::TensorNetworkState)
    tns = copy(tns)
    apply_kwargs = (; normalize_tensors = false)
    gates = [("X", v) for v in vertices(tns)]
    tnsP, _ = apply_gates(gates, tns; apply_kwargs)
    return tnsP + tns
end

function ghz_state(g, s)
    ψ1, ψ2 = tensornetworkstate(Float64, v -> "↑", g, s), tensornetworkstate(Float64, v -> "↓", g, s)
    return ψ1 + ψ2
end

function main()
    n = 8
    chi = 4

    # the graph is your main friend. This will be the geometry of the TN you wull work with
    g = named_grid((n, n))
    nq = length(vertices(g))

    dt = 0.01

    hx = 3.5
    J = 1.0

    #Build a layer of the circuit. Pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
    layer = []
    append!(layer, ("Rx", [v], 2 * hx * dt) for v in vertices(g))
    #For two site gates do an edge coloring to Trotterise the circuit
    ec = edge_color(g, 4)
    for colored_edges in ec
        append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
    end

    # observables are tuples like `(pauli_string, [site_labels], optional:coefficient)`
    # it's important that the `site_labels` match the names of the vertices of the graph `g`
    n_mid = (n ÷ 2, n ÷ 2)
    n_mid_n = (n ÷ 2, n ÷ 2 + 1)
    obs = ("ZZ", [n_mid, n_mid_n])  # right in the middle

    # the number of circuit layers
    nl = 250
    s = siteinds("S=1/2", g)

    # the initial state (all up, use Float 32 precision)
    ψ0 = ghz_state(g, s)
    #ψ0 = tensornetworkstate(Float64, v-> "↓", g, s)

    println("Init bond dimension: $(maxvirtualdim(ψ0))")

    # max bond dimension for the TN
    apply_kwargs = (maxdim = chi, cutoff = 1.0e-14, normalize_tensors = true)

    # create the BP cache representing the square of the tensor network
    ψ_bpc = BeliefPropagationCache(ψ0)

    szz_bp = expect(ψ_bpc, obs)
    println("    BP Measured ZZ is $(szz_bp)")

    total_fidelity = 1
    # evolve! (First step takes long due to compilation)
    times = Float64[]
    fids = Float64[]
    szzs_bmps = ComplexF64[]
    szzs_bp = ComplexF64[]
    for l in 1:nl
        println("Layer $l")

        t1 = @timed ψ_bpc, errors =
            apply_gates(layer, ψ_bpc; apply_kwargs, verbose = false, update_cache = false)
        ψ_bpc = update(ψ_bpc)
        rescale!(ψ_bpc)
        total_fidelity *= prod(1 .- errors)

        if l % 10 == 0
            #BP expectation (already have an up-to-date BP cache)
            szz_bp = expect(ψ_bpc, obs)
            R = 2*chi
            ψ_bmps = update(BoundaryMPSCache(network(ψ_bpc), R))
            szz_bmps = expect(ψ_bmps, obs)
            sz_bp1, sz_bp2 = expect(ψ_bpc, ("Z", n_mid)), expect(ψ_bpc, ("Z", n_mid_n))
            sz_bmps1, sz_bmps2 = expect(ψ_bmps, ("Z", n_mid)), expect(ψ_bmps, ("Z", n_mid_n))

            println("    Took time: $(t1.time) [s]. Max bond dimension: $(maxvirtualdim(ψ_bpc))")
            println("    Total fidelity is $total_fidelity")
            println("    BP Measured ZZ Corr is $(szz_bp)")
            println("    BMPS Measured ZZ Corr is $(szz_bmps)")

            push!(times, l*dt)
            push!(fids, total_fidelity)
            push!(szzs_bmps, szz_bmps)
            push!(szzs_bp, szz_bp)
        end
    end
    npzwrite("")
end

main()
