using TensorNetworkQuantumSimulator

function measure_energy(ψ_bpc, Jperp, Jparr)
    g = graph(ψ_bpc)
    energy = 0
    for e in edges(ψ_bpc)
        v1, v2 = src(e), dst(e)
        J = v1[1] == v2[1] ? Jparr : Jperp
        energy += J*(expect(ψ_bpc, ("XX", [src(e), dst(e)])) + expect(ψ_bpc, ("YY", [src(e), dst(e)])) + expect(ψ_bpc, ("ZZ", [src(e), dst(e)])))
    end
    return energy / length(vertices(g))
end

function measure_energy_bmps(ψ, Jperp, Jparr, mps_bond_dimension)
    ψ_bmps_col = update(BoundaryMPSCache(ψ, mps_bond_dimension; partition_by = "col"))
    ψ_bmps_row = update(BoundaryMPSCache(ψ, mps_bond_dimension; partition_by = "row"))
    g = graph(ψ)
    energy = 0
    for e in edges(ψ)
        v1, v2 = src(e), dst(e)
        if v1[1] == v2[1]
            J = Jparr
            energy += J*(expect(ψ_bmps_row, ("XX", [src(e), dst(e)])) + expect(ψ_bmps_row, ("YY", [src(e), dst(e)])) + expect(ψ_bmps_row, ("ZZ", [src(e), dst(e)])))
        else
            J = Jperp
            energy += J*(expect(ψ_bmps_col, ("XX", [src(e), dst(e)])) + expect(ψ_bmps_col, ("YY", [src(e), dst(e)])) + expect(ψ_bmps_col, ("ZZ", [src(e), dst(e)])))
        end
    end
    return energy / length(vertices(g))
end

function main()
    L = 10
    g = named_grid((2,L); periodic = false)

    s = siteinds("S=1/2", g)

    ψ0 = tensornetworkstate(v -> isodd(v[2]) ? "Z+" : "Z-", g,s)

    @show expect(ψ0, ("Z", (1,2)); alg = "bp")

    @show TensorNetworkQuantumSimulator.maxvirtualdim(ψ0)

    Jperp, Jparr = 0.1, 1.0
    δβ = 0.02

    #Define an edge coloring
    ec = edge_color(g, 4)

    #Now the circuit
    Rxxyyzz_layer = []
    for edge_group in ec
        for pair in edge_group
            v1, v2 = src(pair), dst(pair)
            J = v1[1] == v2[1] ? Jparr : Jperp
            push!(Rxxyyzz_layer, ("Rxxyyzz", pair, -im * δβ * J))
        end
    end

    no_trotter_steps = 250
    χ = 6
    apply_kwargs = (; cutoff = 1.0e-12, maxdim = χ, normalize_tensors = true)
    ψt = copy(ψ0)
    mps_bond_dimension = 10
    ψ_bpc = update(BeliefPropagationCache(ψt))
    for i in 1:no_trotter_steps
        ψ_bpc, errs = apply_gates(Rxxyyzz_layer, ψ_bpc; apply_kwargs)
        if i % 10 == 0
            e_bp = measure_energy(ψ_bpc, Jperp, Jparr)
            e_bmps = measure_energy_bmps(network(ψ_bpc), Jperp, Jparr, mps_bond_dimension)
            println("Imaginary Time is $(i*δβ). Bond dimension is currently $(maxvirtualdim(ψ_bpc))")
            println("BP Energy density is $(e_bp)")
            println("BMPS Energy density is $(e_bmps)")
        end
    end
end

main()