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

    return t * e_hop + U * e_int, e_int
end

function exact_energy(ψ, U, t)
    g = graph(ψ)
    e_int = 0
    for v in vertices(g)
        e_int += expect(ψ, (["NupNdn"], [v]); alg = "exact")
    end
    e_hop = 0
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        e_up = expect(ψ, (["Cupdag", "Cup"], [v1, v2]); alg = "exact")
        e_dn = expect(ψ, (["Cdndag", "Cdn"], [v1, v2]); alg = "exact")
        e_hop += e_up + conj(e_up)
        e_hop += e_dn + conj(e_dn)
    end

    return t * e_hop + U * e_int, e_int
end

function bmps_energy(ψ_bmps_row::BoundaryMPSCache, ψ_bmps_col::BoundaryMPSCache, g, U, t)
    e_int = 0
    for v in vertices(g)
        e_int += expect(ψ_bmps_row, (["NupNdn"], [v]))
    end
    e_hop = 0
    for e in edges(g)
        v1, v2 = src(e), dst(e)
        if v1[1] == v2[1]
            e = expect(ψ_bmps_row, (["Cupdag", "Cup"], [v1, v2])) + expect(ψ_bmps_row, (["Cdndag", "Cdn"], [v1, v2]))
            e_hop += 2*e
        else
            e = expect(ψ_bmps_col, (["Cupdag", "Cup"], [v1, v2])) + expect(ψ_bmps_col, (["Cdndag", "Cdn"], [v1, v2]))
            e_hop += 2*e
        end
    end

    return t * e_hop + U * e_int, e_int
end


function main_fermions(χ)

    U = 9.0
    t = -1

    println("U is $U")
    ITensors.disable_warn_order()
    n = 6
    #g = named_hexagonal_lattice_graph(n,n; periodic = false)
    ψ_bpc = deserialize("/Users/jtindall/Files/Data/Fermions/HexagonalOBC/U$(U)maxdim$(χ).jld2")
    g = graph(ψ_bpc)
    println("Imaginary time Evo to find Hubbard model GS lattice of $(length(vertices(g))) sites with BP")

    println("Tensor network Bond dimension is $(maxvirtualdim(ψ_bpc))")
    ec = edge_color(g, 3)
    ψ_bpc = update(ψ_bpc)
    e_bp, nd_bp = bp_energy(ψ_bpc, U, t)
    println("BP energy density is $(e_bp / length(vertices(g)) - U/4)")

    Rs = [1,2,4,8,16,32]
    ψ = network(ψ_bpc)

    for R in Rs
        println("R is $R")
        ψ_bmps_row = update(BoundaryMPSCache(ψ, R; partition_by = "row"))
        ψ_bmps_col = update(BoundaryMPSCache(ψ, R; partition_by = "col"))
        e_bmps, nd = bmps_energy(ψ_bmps_row, ψ_bmps_col, g, U, t)
        println("BMPS measured energy density is $(e_bmps / length(vertices(g)) - U/4)")
    end
end

χ = 16
main_fermions(χ)


