using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator
using ITensors: ITensors
Random.seed!(1234)

function main_fermions(χ)
    ITensors.disable_warn_order()
    g = named_hexagonal_lattice_graph(4,4)
    s = siteinds("fermion", g)
    ψ = fermionic_tensornetworkstate(ComplexF32, v-> isodd(sum(v)) ? "Emp" : "Occ", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    dt = 0.1
    U = 5.0
    t = -1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14)
    single_site_gates = [("RN", v, U*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    nsteps = 5
    @show sum([expect(ψ_bpc, [(["N"], [v])]) for v in vertices(g)])
    t_update =0
    t_bp = 0
    for i in 1:nsteps
        t1 = time()
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        t2 = time()

        ψ_bpc = update(ψ_bpc; niters = 10)
        t3 = time()

        t_update += (t2-t1)
        t_bp += (t3-t2)
    end

    println("Time spent applying gates: $t_update secs")
    println("Time spent updating cache: $t_bp secs")

    @show sum([expect(ψ_bpc, [(["N"], [v])]) for v in vertices(g)])
end

function main_spins(χ)
    ITensors.disable_warn_order()
    g = named_hexagonal_lattice_graph(4,4)
    s = siteinds("S=1/2", g)
    ψ = tensornetworkstate(ComplexF32, v-> isodd(sum(v)) ? "Z+" : "Z-", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    dt = 0.1
    J = 5.0
    h = 1.0
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14)
    single_site_gates = [("Rx", v, h*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("Rzz", [src(e), dst(e)], J*dt) for e in es])
    end

    nsteps = 5
    @show sum([expect(ψ_bpc, [(["Z"], [v])]) for v in vertices(g)])
    t_update =0
    t_bp = 0
    for i in 1:nsteps
        t1 = time()
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        t2 = time()

        ψ_bpc = update(ψ_bpc; niters = 10)
        t3 = time()

        t_update += (t2-t1)
        t_bp += (t3-t2)
    end

    println("Time spent applying gates: $t_update secs")
    println("Time spent updating cache: $t_bp secs")

    @show sum([expect(ψ_bpc, [(["Z"], [v])]) for v in vertices(g)])
end

χ = 16
main_spins(χ)
main_fermions(χ)


