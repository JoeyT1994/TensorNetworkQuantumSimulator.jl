using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator
using ITensors: ITensors
using NamedGraphs: add_edge!, NamedEdge
Random.seed!(1234)
using LinearAlgebra: eigvals, Hermitian
#using NPZ

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

function hexagonal_unit_cell()
    g = NamedGraph([(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)])
    for v in ([(1,1), (1,2), (1,3)])
        g = add_edge!(g, NamedEdge(v => (2,1)))
        g = add_edge!(g, NamedEdge(v => (2,2)))
        g = add_edge!(g, NamedEdge(v => (2,3)))
    end
    return g
end

function free_fermion_gs_energy(g, t,n_fermions)
    vs  = collect(vertices(g))
    pos = Dict(v => i for (i, v) in enumerate(vs))
    N   = length(vs)
    h = zeros(ComplexF64, N, N)
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        a, b   = pos[v1], pos[v2]
        tij    = t
        h[a, b] += tij
        h[b, a] += conj(tij)
    end
    ε    = eigvals(Hermitian(h))
    return sum(ε[1:n_fermions])
end

function main_fermions(χ)

    honey_comb_Us = [0.0, 1.0, 2.0, 3.0, 3.5,4.0,4.5,5.0,6.0, 7.0, 8.0]
    honey_comb_es = [-1.57, -1.59, -1.62, -1.69, -1.73, -1.78, -1.84, -1.91, -2.06, -2.24, -2.43]
    honey_comb_es = honey_comb_es + honey_comb_Us/4
    ITensors.disable_warn_order()
    g = named_hexagonal_lattice_graph(6,6; periodic = false)
    t = 1
    n_fermions = round(Int, length(vertices(g)) / 2)
    @show free_fermion_gs_energy(g, t, n_fermions) / (n_fermions)
    s = siteinds("spinful_fermion", g)
    ψ = fermionic_tensornetworkstate(Float64, v-> isodd(sum(v)) ? "Up" : "Dn", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    println("Imaginary time Evo to find Hubbard model GS lattice of $(length(vertices(g))) sites with BP")
    dt = -0.01*im
    U = 0.0

    U_index = findfirst(x -> abs(x - U) < 1e-10, honey_comb_Us)
    t = -1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14)
    single_site_gates = [("RInt", v, 0.5*U*dt) for v in vertices(g)]
    single_site_gates = [single_site_gates; [("RN", v, -0.25*U*dt) for v in vertices(g)]]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    nsteps = 2500
    t_update =0
    t_bp = 0

    e_bp = bp_energy(ψ_bpc, U, t)
    println("Initial BP energy density is $(e_bp / length(vertices(g)))")
    #e_ref_U8 = -0.494
    imaginary_times = Float64[]
    energies = Float64[]
    for i in 1:nsteps
        t1 = time()
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)

        
        if i % 5 == 0
            ψ_bpc = update(ψ_bpc)
            rescale!(ψ_bpc)
            e_bp = bp_energy(ψ_bpc, U, t)
            push!(energies, e_bp)
            push!(imaginary_times, i * abs(dt))
            println("Imaginary time is $(i * abs(dt))")
            println("BP energy density is $(e_bp / length(vertices(g)))")

            nup_tot = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])

            println("Total Nup density is $(nup_tot / length(vertices(g)))")
        end
    end
end

χ = 4
main_fermions(χ)


