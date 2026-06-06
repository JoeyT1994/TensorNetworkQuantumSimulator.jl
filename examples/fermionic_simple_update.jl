using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator
using ITensors: ITensors
Random.seed!(1234)

function main()
    ITensors.disable_warn_order()
    χ = 64
    g = named_hexagonal_lattice_graph(4,4)
    s = siteinds("spinful_fermion", g)
    ψ = fermionic_tensornetworkstate(ComplexF32, v-> isodd(sum(v)) ? "Up" : "Dn", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    dt = 0.1
    U = 5.0
    t = -1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14)
    single_site_gates = [("RInt", v, U*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    nsteps = 5
    @show sum([expect(ψ_bpc, [(["Ndn"], [v])]) for v in vertices(g)])
    for i in 1:nsteps
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)

        ψ_bpc = update(ψ_bpc)

        println("Mean two-site gate error on this step $(sum(errs)/length(errs))")
    end

    @show sum([expect(ψ_bpc, [(["Ndn"], [v])]) for v in vertices(g)])

    @show only(expect(ψ_bpc, [(["NupNdn"], [first(center(g))])]))
end

main()


