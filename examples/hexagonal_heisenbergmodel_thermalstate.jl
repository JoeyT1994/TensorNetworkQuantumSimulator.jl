using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator, freenergy
using TensorNetworkQuantumSimulator: ITensors, ITensor

function main()
    χ = 32
    g = named_hexagonal_lattice_graph(2,2; periodic = true)
    s = siteinds("S=1/2", g; inds_per_site = 2)
    ψ = identity_tensornetworkstate(Float64, g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))

    println("Finite temp simulation of Hexagonal Heisenberg model in the thermodynamic limit")
    dβ = 0.01
    J = 1.0

    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14, normalize_tensors = false)
    two_site_gates =ITensor[]
    for es in ec
        append!(two_site_gates, [ITensors.op("Rxxyyzz", s[src(e)][1], s[dst(e)][1], θ = -0.5*J*dβ*im) for e in es])
    end

    nsteps = 25
    logz = -freenergy(ψ_bpc)
    rescale!(ψ_bpc)
    for i in 1:nsteps
        t1 = time()
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs)
        logz -= freenergy(ψ_bpc)
        rescale!(ψ_bpc)
        if i % 5 == 0
            #Doubled because we prepared sqrt state and measured over the norm
            β = 2*i*dβ
            f_bp = logz  / length(vertices(g))
            println("Inverse temp is $(β) and BP measured free energy density is $(f_bp)")
            f_htse_order4 = -log(2) - (9/64)*J*J*β*β - (3/128)*J*J*J*β*β*β + (27/2048)*J*J*J*J*β*β*β*β
            println("Abs diff between BP value and fourth order HTSE is $(abs(f_htse_order4 - f_bp))")
        end

    end
end

main()