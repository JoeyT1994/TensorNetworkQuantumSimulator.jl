using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator
using ITensors: ITensors
using NamedGraphs: add_edge!, NamedEdge
Random.seed!(1234)
using LinearAlgebra: eigvals, Hermitian
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
    ITensors.disable_warn_order()
    n = 6
    g = named_hexagonal_lattice_graph(n,n; periodic = false)
    t = 1
    n_fermions = round(Int, length(vertices(g)) / 2)
    s = siteinds("spinful_fermion", g)
    ψ = fermionic_tensornetworkstate(Float64, v-> isodd(sum(v)) ? "Up" : "Dn", g, s)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    println("Imaginary time Evo to find Hubbard model GS lattice of $(length(vertices(g))) sites with BP")
    dt = -0.01*im
    U = 9.0

    t = -1
    ec = edge_color(g, 3)
    apply_kwargs= (; maxdim = χ, cutoff = 1e-14, normalize_tensors = true)
    single_site_gates = [("RInt", v, 0.5*U*dt) for v in vertices(g)]
    single_site_gates = [single_site_gates; [("RN", v, -0.25*U*dt) for v in vertices(g)]]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], t*dt) for e in es])
    end

    nsteps = 1000

    e_bp = bp_energy(ψ_bpc, U, t)
    println("Initial BP energy density is $(e_bp / length(vertices(g)))")
    imaginary_times = Float64[]
    energies = Float64[]
    for i in 1:nsteps
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
        ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)

        
        if i % 4 == 0
            ψ_bpc = update(ψ_bpc)
            rescale!(ψ_bpc)
            e_bp = bp_energy(ψ_bpc, U, t)
            push!(energies, e_bp)
            push!(imaginary_times, i * abs(dt))
            println("Imaginary time is $(i * abs(dt))")
            println("BP energy density is $(e_bp / length(vertices(g)) - U / 4)")

            nup_tot = sum([expect(ψ_bpc, (["Nup"], [v])) for v in vertices(g)])

            println("Total Nup density is $(nup_tot / length(vertices(g)))")
        end
    end

    serialize("/Users/jtindall/Files/Data/Fermions/HexagonalOBC/U$(U)maxdim$(χ).jld2", ψ_bpc)
end

χ = 16
main_fermions(χ)


