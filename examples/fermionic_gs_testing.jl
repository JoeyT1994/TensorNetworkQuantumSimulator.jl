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

    honey_comb_Us = [0.0, 1.0, 2.0, 3.0, 3.5,4.0,4.5,5.0,6.0, 7.0, 8.0]
    honey_comb_es = [-1.57, -1.59, -1.62, -1.69, -1.73, -1.78, -1.84, -1.91, -2.06, -2.24, -2.43]
    honey_comb_es = honey_comb_es + honey_comb_Us/4
    ITensors.disable_warn_order()
    g = named_hexagonal_lattice_graph(4,4; periodic = true)
    #g = named_grid((10,10))
    #g = named_grid((4,1))
    s = siteinds("spinful_fermion", g)
    ψ = fermionic_tensornetworkstate(Float64, v-> isodd(sum(v)) ? "UpDn" : "Emp", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    println("Imaginary time Evo to find Hubbard model GS lattice of $(length(vertices(g))) sites with BP")
    dt = -0.01*im
    U = 8.0

    U_index = findfirst(x -> abs(x - U) < 1e-10, honey_comb_Us)
    t = -1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14)
    single_site_gates = [("RInt", v, 0.5*U*dt) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    nsteps = 5000
    t_update =0
    t_bp = 0

    e_bp = bp_energy(ψ_bpc, U, t)
    println("Initial BP energy density is $(e_bp / length(vertices(g)))")
    #e_ref_U8 = -0.494
    for i in 1:nsteps
        t1 = time()
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs)
        rescale!(ψ_bpc)
        #ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        t2 = time()

        #ψ_bpc = update(ψ_bpc)
        t3 = time()

        t_update += (t2-t1)
        t_bp += (t3-t2)

        
        if i % 1 == 0
            #ψ_bpc = update(ψ_bpc)
            e_bp = bp_energy(ψ_bpc, U, t)
            println("Imaginary time is $(i * abs(dt))")
            println("BP energy density is $(e_bp / length(vertices(g)))")

            #println("Chan et al BP Ref for U = 8 is approx $e_ref_U8")
            println("Honeycomb QMC ref energy is $(honey_comb_es[U_index])")

            nup_tot = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])

            println("Total Nup density is $(nup_tot / length(vertices(g)))")
        end
    end
end

χ = 4
main_fermions(χ)


