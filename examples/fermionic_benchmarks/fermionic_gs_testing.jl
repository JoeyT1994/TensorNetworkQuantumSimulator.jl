using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator, freenergy
using ITensors: ITensors
using NamedGraphs: add_edge!, NamedEdge
Random.seed!(1234)
using NPZ
using JLD2
using Serialization

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

    return t * e_hop + U * e_int, e_int
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

function main_fermions(U, χ)
    ITensors.disable_warn_order()
    g = hexagonal_unit_cell()
    s = siteinds("spinful_fermion", g)
    ψ = fermionic_tensornetworkstate(Float64, v-> isodd(sum(v)) ? "Up" : "Dn", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    println("Imaginary time Evo to find Hubbard model GS lattice of $(length(vertices(g))) sites with BP")
    dt = -0.005*im

    t = -1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14, normalize_tensors = true)
    single_site_gates = [("RInt", v, 0.5*U*dt) for v in vertices(g)]
    single_site_gates = [single_site_gates; [("RN", v, -0.25*U*dt) for v in vertices(g)]]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    nsteps = 2500

    e_bp, Nd = bp_energy(ψ_bpc, U, t)
    println("Initial BP energy density is $(e_bp / length(vertices(g)))")
    imaginary_times = Float64[]
    energies = Float64[e_bp]
    double_occs = Float64[Nd]
    for i in 1:nsteps
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        
        if i % 2 == 0
            ψ_bpc = update(ψ_bpc)
            rescale!(ψ_bpc)
            e_bp, Nd = bp_energy(ψ_bpc, U, t)
            push!(double_occs, Nd / length(vertices(g)))
            push!(energies, e_bp / length(vertices(g)))
            push!(imaginary_times, i * abs(dt))
            println("Imaginary time is $(i * abs(dt))")
            println("BP energy density is $(e_bp / length(vertices(g)))")
            println("BP double occ is $(Nd / length(vertices(g)))")
        end
        flush(stdout)
    end

    ψ_bpc = update(ψ_bpc)
    rescale!(ψ_bpc)
    f_str = "/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).npz"
    npzwrite(f_str, energies = energies, imaginary_times = imaginary_times, double_occs = double_occs)
    #JLD2.save("/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/States/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).jld2", Dict("psi_bpc" => ψ_bpc, "sinds" => s))
    serialize("/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/States/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).ser", ψ_bpc)
end

U = 8.0
χ = 10

#U = parse(Float64, ARGS[1])
#χ = parse(Int64, ARGS[2])
main_fermions(U, χ)


