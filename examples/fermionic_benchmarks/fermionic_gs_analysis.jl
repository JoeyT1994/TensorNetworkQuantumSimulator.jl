using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator, freenergy, _fermionic_loop_weight, connected_edgeinduced_subgraphs_no_leaves, loopcorrected_free_energy
using ITensors: ITensors
using NamedGraphs: add_edge!, NamedEdge, NamedGraphs
Random.seed!(1234)
using NPZ
using JLD2
using Serialization

function iTNS_energy_doubleocc(ψ_bpc, U, t)

    ndubs = 0.5*(real(iTNS_expect(ψ_bpc, "NupNdn", :A)) + real(iTNS_expect(ψ_bpc, "NupNdn", :B)))
    e_up1, e_up2, e_up3 = 2*real(iTNS_expect(ψ_bpc, ["Cupdag", "Cup"], 1)), 2*real(iTNS_expect(ψ_bpc, ["Cupdag", "Cup"], 2)), 2*real(iTNS_expect(ψ_bpc, ["Cupdag", "Cup"], 3))
    e_dn1, e_dn2, e_dn3 = 2*real(iTNS_expect(ψ_bpc, ["Cdndag", "Cdn"], 1)), 2*real(iTNS_expect(ψ_bpc, ["Cdndag", "Cdn"], 2)), 2*real(iTNS_expect(ψ_bpc, ["Cdndag", "Cdn"], 3))
    up_hops = e_up1 + e_up2 + e_up3
    dn_hops = e_dn1 + e_dn2 + e_dn3
    hop_energy = (up_hops + dn_hops)*(-t)
    e = U*ndubs + hop_energy / 2

    return e, ndubs
end

function TNS_energy_doubleocc(ψ_bpc, U, t)
    ndubs = 0
    g = graph(ψ_bpc)
    for v in vertices(g)
        ndubs += only(expect(ψ_bpc, [(["NupNdn"], [v])]))
    end
    tot_kin_e = 0
    for e in edges(g)
        e_up = 2*real(expect(ψ_bpc, (["Cupdag", "Cup"], [src(e), dst(e)])))
        e_dn = 2*real(expect(ψ_bpc, (["Cdndag", "Cdn"], [src(e), dst(e)])))
        tot_kin_e +=  e_up
        tot_kin_e +=  e_dn
    end

    return -t  * tot_kin_e + U *ndubs, ndubs
end

function loop_corrected_doublons(ψ_bpc, C; epsilon = 1e-4)
    g = graph(ψ_bpc)
    forward_gates = [("RInt", v, 0.5*epsilon * im) for v in vertices(g)]
    backward_gates = [("RInt", v, -0.5*epsilon * im) for v in vertices(g)]

    ψ_bpc_plus, _ = apply_gates(forward_gates, ψ_bpc; apply_kwargs = (; normalize_tensors = false))
    ψ_bpc_minus, _ = apply_gates(backward_gates, ψ_bpc; apply_kwargs = (; normalize_tensors = false))
    ψ_bpc_plus, ψ_bpc_minus = update(ψ_bpc_plus), update(ψ_bpc_minus)

    f_plus, f_minus = loopcorrected_free_energy(ψ_bpc_plus, C), loopcorrected_free_energy(ψ_bpc_minus, C)
    return (f_plus - f_minus) / (2*epsilon)
end

function main_fermions(U, χ)
    t = 1
    ITensors.disable_warn_order()

    f_str = "/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/iTNS/States/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).ser"
    ψ_bpc = deserialize(f_str)

    #ψ = InfiniteTensorNetworkState(network(ψ_bpc))
    #ψ_bpc = update(BeliefPropagationCache(ψ))
    #rescale!(ψ_bpc)

    #e, Nd = iTNS_energy_doubleocc(ψ_bpc, U, t)

    #println("iTNS BP energy density is $(e - U/4)")
    #println("iTNS BP double occ is $(Nd)")

    ψ_embedded = absorb_bonds(embed(InfiniteTensorNetworkState(network(ψ_bpc)), HexagonalLattice, (12,12)))
    g = graph(ψ_embedded)
    println("Embedded onto a graph of $(length(vertices(g))) vertices")

    
    cs = connected_edgeinduced_subgraphs_no_leaves(g, 11)
    @show length(cs) / length(vertices(g))
    ψ_bpc_embedded = update(BeliefPropagationCache(ψ_embedded))
    #rescale!(ψ_bpc_embedded)

    C = 6
    Nd_lc1 = loop_corrected_doublons(ψ_bpc_embedded, C; epsilon = 1e-3)/ length(vertices(g))
    println("TNS BP 1st order LC double occ is $(Nd_lc1)")

    C = 11
    Nd_lc2 = loop_corrected_doublons(ψ_bpc_embedded, C; epsilon = 1e-3)/ length(vertices(g))
    println("TNS BP 2nd order LC double occ is $(Nd_lc2)")

    e, Nd = TNS_energy_doubleocc(ψ_bpc_embedded, U, t)
    e, Nd = e / length(vertices(g)), Nd / length(vertices(g))

    println("TNS BP energy density is $(e - U/4)")
    println("TNS BP double occ is $(Nd)")

    f_str = "/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/iTNS/LC/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).npz"
    npzwrite(f_str,e = e, Nd = Nd, Nd_lc1 = Nd_lc1, Nd_lc2 = Nd_lc2)


end

Us = [parse(Float64, ARGS[1])]
χ = parse(Int64, ARGS[2])

#Us = [0.0]
#χ = 8
for U in Us
    main_fermions(U, χ)
end


