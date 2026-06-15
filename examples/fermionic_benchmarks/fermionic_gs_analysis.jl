using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator, freenergy, _fermionic_loop_weight
using ITensors: ITensors
using NamedGraphs: add_edge!, NamedEdge, NamedGraphs
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

function doublons(ψ_bpc; epsilon = 0.001)
    g = graph(ψ_bpc)
    nv = length(vertices(g))
    edge_graph6 = eg6(g)
    edge_graph10 = eg10(g)
    f1 = log(partitionfunction(ψ_bpc)) / nv
    w6 = _fermionic_loop_weight(ψ_bpc, edge_graph6)
    w10 = _fermionic_loop_weight(ψ_bpc, edge_graph10)
    f1_lc1 =f1 + 0.5 * w6
    f1_lc2 = f1 + 0.5 * w6 + 1.5*w10
    gates = [("RInt", v, im*0.5*epsilon) for v in vertices(g)]
    apply_kwargs = (;normalize_tensors = false)
    ψ_bpc, _ = apply_gates(gates, ψ_bpc; update_cache = false, apply_kwargs)
    ψ_bpc = update(ψ_bpc)
    f2 = log(partitionfunction(ψ_bpc)) / nv
    rescale!(ψ_bpc)
    w6 = _fermionic_loop_weight(ψ_bpc, edge_graph6)
    w10 = _fermionic_loop_weight(ψ_bpc, edge_graph10)
    f2_lc1 =f2 + 0.5 * w6
    f2_lc2 = f2 + 0.5 * w6 + 1.5*w10
    return (-(f1-f2) / epsilon), (-(f1_lc1 - f2_lc1)/ epsilon),  (-(f1_lc2 - f2_lc2)/ epsilon)
end

function energy(ψ_bpc, t, U; epsilon = 0.01)
    g = graph(ψ_bpc)
    edge_graph6 = eg6(g)
    edge_graph10 = eg10(g)
    nv = length(vertices(g))
    f1 = log(partitionfunction(ψ_bpc)) / nv
    w6 = _fermionic_loop_weight(ψ_bpc, edge_graph6)
    w10 = _fermionic_loop_weight(ψ_bpc, edge_graph10)
    f1_lc1 =f1 + 0.5 * w6
    f1_lc2 = f1 + 0.5 * w6 + 1.5*w10
    ec = edge_color(g, 3)
    single_site_gates = [("RInt", v, 0.25*U*im*epsilon) for v in vertices(g)]
    two_site_gates =[]
    for es in ec
        append!(two_site_gates, [("RHop", [src(e), dst(e)], 0.5*t*im*epsilon) for e in es])
    end
    apply_kwargs = (; maxdim = maxvirtualdim(ψ_bpc), cutoff = nothing, normalize_tensors = false)
    ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
    ψ_bpc, errs = apply_gates(two_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
    ψ_bpc, _ = apply_gates(single_site_gates,ψ_bpc;apply_kwargs, update_cache = false)
    ψ_bpc = update(ψ_bpc)
    f2 = log(partitionfunction(ψ_bpc)) / nv
    rescale!(ψ_bpc)
    w6 = _fermionic_loop_weight(ψ_bpc, edge_graph6)
    w10 = _fermionic_loop_weight(ψ_bpc, edge_graph10)
    f2_lc1 =f2 + 0.5 * w6
    f2_lc2 = f2 + 0.5 * w6 + 1.5*w10
    return (-(f1-f2) / epsilon) - U /4, (-(f1_lc1 - f2_lc1)/ epsilon) - U /4,  (-(f1_lc2 - f2_lc2)/ epsilon) - U /4
end

function eg6(g)
    cycs = NamedGraphs.simplecycles_limited_length(g, 6)
    cycs = filter(l -> length(l) == 6, cycs)
    cycs = filter(l -> length(unique(l)) == 6, cycs)
    cyc = cycs[4]
    cyc = NamedGraphs.cycle_to_path(cyc)
    eg = NamedGraph(vertices(g))
    eg = NamedGraphs.add_edges!(eg, cyc)
    return eg
end

function eg10(g)
    cycs = NamedGraphs.simplecycles_limited_length(g, 10)
    cycs = filter(l -> length(l) == 10, cycs)
    cycs = filter(l -> length(unique(l)) == 10, cycs)
    cyc = cycs[5]
    cyc = NamedGraphs.cycle_to_path(cyc)
    eg = NamedGraph(vertices(g))
    eg = NamedGraphs.add_edges!(eg, cyc)
    return eg
end

function hexagonal_unit_cell_V2()
    g = named_grid((2,5))
    g = add_edge!(g, (1,5) => (1,1))
    g = add_edge!(g, (2,5) => (2,1))
    @assert all([degree(g, v) == 3 for v in vertices(g)])
    return g
end

function main_fermions(U, χ)
    t = -1
    ITensors.disable_warn_order()
    g = hexagonal_unit_cell_V2()

    f_str = "/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/States/HoneyCombHubbardHalffilledU$(round(U; digits = 2))BondDimension$(χ).ser"
    ψ_bpc = deserialize(f_str)

    ψ = network(ψ_bpc)
    ψ_bpc = update(BeliefPropagationCache(ψ))
    rescale!(ψ_bpc)

    e, Nd = bp_energy(ψ_bpc, U, t)
    e, Nd = e/ length(vertices(g)), Nd / length(vertices(g))

    println("BP energy density is $(e - U/4)")
    println("BP double occ is $(Nd)")

    Nd_fd, Nd_fd_lc1, Nd_fd_lc2  = doublons(ψ_bpc)
    println("Finite Diff Double Occ $Nd_fd")
    println("LC Order 1 Finite Diff Double Occ $Nd_fd_lc1")
    println("LC Order 2 Finite Diff Double Occ $Nd_fd_lc2")

    #e_fd, e_fd_lc = energy(ψ_bpc, t, U)

    #println("Finite Diff Energy $(e_fd)")
    #println("LC Finite Diff Energy $(e_fd_lc)")

    scalars = v -> first(v) == 1 ? -1.0 : 1.0
    @show sum(expect(ψ_bpc, [(["Sz"], [v], scalars(v)) for v in vertices(g)])) / length(vertices(g))

    @show sum(expect(ψ_bpc, [(["Sz"], [v]) for v in vertices(g)]))

    @show expect(ψ_bpc, [(["NupNdn"], [v]) for v in vertices(g)])

    return real(e), real(Nd), real(Nd_fd), real(Nd_fd_lc1), real(Nd_fd_lc2)

end

Us = [3.1]
#Us = setdiff(Us, [1.1])
χ =16


es = Float64[]
Nds = Float64[]
Nd_fds = Float64[]
Nd_fd_lc1s = Float64[]
Nd_fd_lc2s = Float64[]

for U in Us
    @show U
    e, Nd, Nd_fd, Nd_fd_lc1, Nd_fd_lc2 = main_fermions(U, χ)
    push!(es, e)
    push!(Nds, Nd)
    push!(Nd_fds, Nd_fd)
    push!(Nd_fd_lc1s, Nd_fd_lc1)
    push!(Nd_fd_lc2s, Nd_fd_lc2)
end

#f_str = "/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/HoneyCombHubbardHalffilledBondDimension$(χ).npz"
#npzwrite(f_str,es = es, Nds = Nds, Nd_fds = Nd_fds, Nd_fd_lcs = Nd_fd_lcs)
