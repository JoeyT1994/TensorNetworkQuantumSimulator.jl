using TensorNetworkQuantumSimulator
using Random
using TensorNetworkQuantumSimulator: scalar_factors_quotient, TensorNetworkQuantumSimulator, freenergy, _fermionic_loop_weight, connected_edgeinduced_subgraphs_no_leaves, loopcorrected_free_energy, gated_lc_free_energy, update
using ITensors: ITensors
using NamedGraphs: add_edge!, NamedEdge, NamedGraphs
Random.seed!(1234)
using NPZ
using JLD2
using Serialization
using ForwardDiff
using CUDA

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

function TNS_doubleocc(ψ_bpc)
    ndubs = 0
    g = graph(ψ_bpc)
    for v in vertices(g)
        ndubs += only(expect(ψ_bpc, [(["NupNdn"], [v])]))
    end
    

    return ndubs
end


function main_fermions(U, χ)
    t = 1
    ITensors.disable_warn_order()

    f_str = "/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/Cylinder/States/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).ser"
    ψ_bpc = deserialize(f_str)
    g = graph(ψ_bpc)
    e, Nd = TNS_energy_doubleocc(ψ_bpc, U, t)
    e, Nd = e / length(vertices(g)), Nd/ length(vertices(g))

    println("TNS BP energy density is $(e - U/4)")
    println("TNS BP double occ is $(Nd)")

    Rs = [1,2,4,8,16,32]
    act_Rs = Int64[]
    ND_bmpss = ComplexF64[]
    for R in Rs
        println("R is $R")
        ψ_bmps = BoundaryMPSCache(network(ψ_bpc), R; partition_by = "col")
        ψ_bmps = CUDA.cu(ψ_bmps)
        ψ_bmps = update(ψ_bmps)

        ND_bmps = TNS_doubleocc(ψ_bmps)
        ND_bmps = ND_bmps / length(vertices(g))

        println("TNS BMPS double occ is $(ND_bmps)")
        push!(ND_bmpss, ND_bmps)
        push!(act_Rs, R)

        f_str = "/mnt/home/jtindall/ceph/Data/Fermions/HexagonalHubbard/GS/Cylinder/BMPS/HoneyCombHubbardHalffilledU$(U)BondDimension$(χ).npz"
        npzwrite(f_str,e = e, Nd_bp = Nd,Rs = act_Rs, ND_bmpss = ND_bmpss)
    end




end

Us = [parse(Float64, ARGS[1])]
χ = parse(Int64, ARGS[2])

# Us = [3.0]
# χ =16
for U in Us
    main_fermions(U, χ)
end


