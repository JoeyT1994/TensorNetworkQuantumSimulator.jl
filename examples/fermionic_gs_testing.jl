using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator
using ITensors: ITensors
Random.seed!(1234)

function bp_energy(ψ_bpc::BeliefPropagationCache, U, t)
    g = graph(ψ_bpc)
    e_int = 0
    for v in vertices(g)
        e_int += expect(ψ_bpc, (["NupNdn"], [v]))
    end
    e_hop = 0
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        e_hop += expect(ψ_bpc, (["Cupdag", "Cup"], [v1, v2])) + expect(ψ_bpc, (["Cupdag", "Cup"], [v2, v1]))
        e_hop += expect(ψ_bpc, (["Cdndag", "Cdn"], [v1, v2])) + expect(ψ_bpc, (["Cdndag", "Cdn"], [v2, v1]))
    end

    return t * e_hop + U * e_int
end

function main_fermions(χ)
    ITensors.disable_warn_order()
    #g = named_hexagonal_lattice_graph(4,4; periodic = true)
    g = named_grid((10,10))
    s = siteinds("spinful_fermion", g)
    ψ = fermionic_tensornetworkstate(ComplexF32, v-> isodd(sum(v)) ? "Up" : "Dn", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    println("Imaginary time Evo to find Hubbard model GS lattice of $(length(vertices(g))) sites with BP")
    dt = -0.01*im
    U = 8.0
    t = -1
    ec = edge_color(g, 4)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-12)
    single_site_gates = [("RInt", v, U*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    nsteps = 500
    t_update =0
    t_bp = 0

    e_bp = bp_energy(ψ_bpc, U, t)
    println("Initial BP energy density is $(e_bp / length(vertices(g)))")
    e_ref_U8 = -0.494
    for i in 1:nsteps
        t1 = time()
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        #ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        t2 = time()

        #ψ_bpc = update(ψ_bpc)
        t3 = time()

        t_update += (t2-t1)
        t_bp += (t3-t2)

        
        if i % 5 == 0
            ψ_bpc = update(ψ_bpc)
            e_bp = bp_energy(ψ_bpc, U, t)
            println("Imaginary time is $(i * abs(dt))")
            println("BP energy density is $(e_bp / length(vertices(g)))")

            println("Chan et al BP Ref for U = 8 is approx $e_ref_U8")

            nup_tot = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])

            println("Total Nup is $nup_tot")
        end
    end
end

χ = 12
main_fermions(χ)


