using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
using ITensorNetworks: IndsNetwork
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: add_edges, add_vertices

using Random
using TOML

using JLD2

include("utils.jl")

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

using Dictionaries

#Construct a graph with edges everywhere a two-site gate appears.
function build_graph_from_interactions(list; sort_vertices = false)
    vertices = []
    edges = []
    for term in list
        vsrc, vdst = (term[3],), (term[4],)
        if vsrc ∉ vertices
            push!(vertices, vsrc)
        end
        if vdst ∉ vertices
            push!(vertices, vdst)
        end
        e = NamedEdge(vsrc => vdst)
        if e ∉ edges || reverse(e) ∉ edges
            push!(edges, e)
        end
    end
    g = NamedGraph()
    if sort_vertices
      vertices = sort(vertices; by = v -> first(v))
    end
    g = add_vertices(g, vertices)
    g = add_edges(g, edges)
    return g
end
  
function hyperhoneycomb_graph(L; kwargs...)
      file = pwd()*"/Research/StructureFactors/Data/hyperhoneycomb."*string(L)*".pbc.HB.Kitaev.nosyminfo.toml"
      data = TOML.parsefile(file)
      interactions = data["Interactions"]
      heisenberg_interactions = filter(d -> first(d) == "HB", interactions)
      g = build_graph_from_interactions(heisenberg_interactions; kwargs...)
      return g
end

function reconstruct_ground_state(i::Int64, maxdim_gs::Int64, s::IndsNetwork, ec)
    file_name = "i"*string(i)*"maxdim"*string(maxdim_gs)
    f = "/mnt/home/jtindall/ceph/Data/StructureFactors/Hyperhoneycomb/GroundStateWavefunctions/"*file_name*".jld2"
    d = load(f)
    ψ_reduced = d["wavefunction"]
    g_reduced = TN.named_biclique(3,3)
    ec_reduced = edge_color(g_reduced, 3)
    #ψ_reduced, ψψ_reduced = TN.symmetric_gauge(ψ_reduced)
    #ψ_reduced, ψψ_reduced = TN.normalize(ψ_reduced, ψψ_reduced; update_cache = false)
    s_reduced = siteinds(ψ_reduced)

    Random.seed!(184)
    vA = vertices(ψ_reduced)[3]
    A = ψ_reduced[vA]
    s_indA = only(s_reduced[vA])
    vnsA = neighbors(s_reduced, vA)
    x_vnA = only(filter(vn -> (vA, vn) ∈ ec_reduced[1] || (vn, vA) ∈ ec_reduced[1], vnsA))
    y_vnA = only(filter(vn -> (vA, vn) ∈ ec_reduced[2] || (vn, vA) ∈ ec_reduced[2], vnsA))
    z_vnA = only(filter(vn -> (vA, vn) ∈ ec_reduced[3] || (vn, vA) ∈ ec_reduced[3], vnsA))
    x_indA, y_indA, z_indA = commonind(A, ψ_reduced[x_vnA]), commonind(A, ψ_reduced[y_vnA]), commonind(A, ψ_reduced[z_vnA])

    vB = vertices(ψ_reduced)[5]
    B = ψ_reduced[vB]
    s_indB = only(s_reduced[vB])
    vnsB = neighbors(s_reduced, vB)
    x_vnB = only(filter(vn -> (vB, vn) ∈ ec_reduced[1] || (vn, vB) ∈ ec_reduced[1], vnsB))
    y_vnB = only(filter(vn -> (vB, vn) ∈ ec_reduced[2] || (vn, vB) ∈ ec_reduced[2], vnsB))
    z_vnB = only(filter(vn -> (vB, vn) ∈ ec_reduced[3] || (vn, vB) ∈ ec_reduced[3], vnsB))
    x_indB, y_indB, z_indB = commonind(B, ψ_reduced[x_vnB]), commonind(B, ψ_reduced[y_vnB]), commonind(B, ψ_reduced[z_vnB])


    #B = replaceinds(B, [x_indB, y_indB, z_indB, s_indB], [x_indA, y_indA, z_indA, s_indA])
    #B = permute(B, inds(A))
    #@show norm((A / norm(A))-(B / norm(B)))

    #@show A / norm(A), B / norm(B)


    #return embed(ψ_reduced[vA], x_indA, y_indA, z_indA, s_indA, s, ec), d["energy"]
    return embed(A, x_indA, y_indA, z_indA, s_indA, B, x_indB, y_indB, z_indB, s_indB, s, ec), d["energy"]
end

function embed(A::ITensor, x_ind::Index, y_ind::Index, z_ind::Index,s_ind::Index, s::IndsNetwork, ec)
    g = ITN.underlying_graph(s)
    ψ = ITensorNetworks.random_tensornetwork(s; link_space = dim(x_ind))
    
    for v in vertices(s)
        new_x_ind, new_y_ind,new_z_ind, new_s_ind = nothing, nothing, nothing, only(s[v])
        for vn in neighbors(g, v)
            e = (v, vn)
            if e ∈ ec[1] || reverse(e) ∈ ec[1]
                new_x_ind = only(ITN.linkinds(ψ, e))
            elseif e ∈ ec[2] || reverse(e) ∈ ec[2]
                new_y_ind = only(ITN.linkinds(ψ, e))
            elseif e ∈ ec[3] || reverse(e) ∈ ec[3]
                new_z_ind = only(ITN.linkinds(ψ, e))
            end
        end
        ψ[v] = replaceinds(deepcopy(A), [x_ind, y_ind, z_ind, s_ind], [new_x_ind, new_y_ind, new_z_ind, new_s_ind])
    end
    return ψ
end

function embed(A::ITensor, x_indA::Index, y_indA::Index, z_indA::Index,s_indA::Index, B::ITensor, x_indB::Index, y_indB::Index, z_indB::Index, s_indB::Index, s::IndsNetwork, ec)
    g = ITN.underlying_graph(s)
    ψ = ITensorNetworks.random_tensornetwork(s; link_space = dim(x_indA))

    vc = vertex_color(g, 2)
    A_sublattice, B_sublattice = first(vc), last(vc)
    
    for v in vertices(s)
        if v ∈ A_sublattice
            new_x_ind, new_y_ind,new_z_ind, new_s_ind = nothing, nothing, nothing, only(s[v])
            for vn in neighbors(g, v)
                e = (v, vn)
                if e ∈ ec[1] || reverse(e) ∈ ec[1]
                    new_x_ind = only(ITN.linkinds(ψ, e))
                elseif e ∈ ec[2] || reverse(e) ∈ ec[2]
                    new_y_ind = only(ITN.linkinds(ψ, e))
                elseif e ∈ ec[3] || reverse(e) ∈ ec[3]
                    new_z_ind = only(ITN.linkinds(ψ, e))
                end
            end
            ψ[v] = replaceinds(deepcopy(A), [x_indA, y_indA, z_indA, s_indA], [new_x_ind, new_y_ind, new_z_ind, new_s_ind])
        elseif v ∈ B_sublattice
            new_x_ind, new_y_ind,new_z_ind, new_s_ind = nothing, nothing, nothing, only(s[v])
            for vn in neighbors(g, v)
                e = (v, vn)
                if e ∈ ec[1] || reverse(e) ∈ ec[1]
                    new_x_ind = only(ITN.linkinds(ψ, e))
                elseif e ∈ ec[2] || reverse(e) ∈ ec[2]
                    new_y_ind = only(ITN.linkinds(ψ, e))
                elseif e ∈ ec[3] || reverse(e) ∈ ec[3]
                    new_z_ind = only(ITN.linkinds(ψ, e))
                end
            end
            ψ[v] = replaceinds(deepcopy(B), [x_indB, y_indB, z_indB, s_indB], [new_x_ind, new_y_ind, new_z_ind, new_s_ind])
        end
    end
    return ψ
end


function main(i::Int64, maxdim_gs::Int64, maxdim::Int64)

    #Get the graph and interactions from the .tomls. Flag ensures Vertices are ordered consistent with the .toml file
    L = 16
    g = hyperhoneycomb_graph(L; sort_vertices = true)

    ec = edge_color(g, 3)

    gp = TN.named_biclique(3,3)
    @show edge_color(gp, 3)

    n = 144
    θ = (2 * pi * i) / (n)
    K, J = sin(θ), cos(θ)
    println("Beginning simulation with theta = $(θ), J = $(J), K = $(K) and a maxdim of $(maxdim).")

    s = ITN.siteinds("S=1/2", g; conserve_qns = false)
    Random.seed!(1234)
    ψ, e_gs = reconstruct_ground_state(i, maxdim_gs, s, ec)

    @show length.(ec)

    vc = vertex_color(g, 2)
    A_sublattice, B_sublattice = first(vc), last(vc)

    @show length.(vc)

    # for v in B_sublattice
    #     ψ[v] = noprime(ψ[v] * ITensors.op("Z", only(s[v])))
    #     ψ[v] = noprime(ψ[v] * ITensors.op("X", only(s[v])))
    #     #ψ[v] = noprime(ψ[v] * ITensors.op("Y", only(s[v])))
    # end

    # for v in A_sublattice
    #     ψ[v] = noprime(ψ[v] * ITensors.op("Z", only(s[v])))
    #     ψ[v] = noprime(ψ[v] * ITensors.op("X", only(s[v])))
    #     ψ[v] = noprime(ψ[v] * ITensors.op("Y", only(s[v])))
    # end

    #xx_observables, yy_observables, zz_observables = [("XX", pair) for pair in ec[1]], [("YY", pair) for pair in ec[2]], [("ZZ", pair) for pair in ec[3]]
    xx_observables, yy_observables, zz_observables = honeycomb_kitaev_heisenberg_observables(J, K, ec)
    ψψ = build_bp_cache(ψ)
    xxs, yys, zzs = expect(ψψ, xx_observables), expect(ψψ, yy_observables), expect(ψψ, zz_observables)
    e = sum(xxs) + sum(yys) + sum(zzs)

    @show e * (length(edges(TN.named_biclique(3,3))) / length(edges(g)))
    @show e_gs
    @show e

    @show sum(xxs), sum(yys), sum(zzs)
    #@show yys
    #@show zzs
end

i, maxdim_gs, maxdim = 36, 4,4
#i, maxdim_gs, maxdim = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Int64, ARGS[3])
main(i, maxdim_gs, maxdim)