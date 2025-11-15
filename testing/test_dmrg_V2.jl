using TensorNetworkQuantumSimulator

using TensorNetworkQuantumSimulator: apply_gates, update_message!, network, norm_factors, QuadraticForm, partitionfunction,
    norm_sqr
const TN = TensorNetworkQuantumSimulator

using Graphs
using Statistics
using ITensors: ITensor, Index, Algorithm, ITensors, inds, dag, prime
using ITensorMPS: ITensorMPS

using NamedGraphs: NamedGraphs, neighbors, NamedEdge, NamedGraph, incident_edges
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.PartitionedGraphs: PartitionEdge, PartitionVertex

using LinearAlgebra
using Dictionaries: Dictionary, set!

function add_itensors(A::ITensor, B::ITensor)
    A_uniqueinds = ITensors.uniqueinds(A, B)
    B_uniqueinds = ITensors.uniqueinds(B, A)
    if !isempty(B_uniqueinds)
        A = A * reduce(*, ITensors.ITensor(1.0, ui) for ui in B_uniqueinds)
    end
    if !isempty(A_uniqueinds)
        B = B * reduce(*, ITensors.ITensor(1.0, ui) for ui in A_uniqueinds)
    end
    return A + B
end

function grid_spanning_trees(nx::Int, ny::Int)
    tree_1 = named_grid((nx, ny))
    tree_2 = named_grid((nx, ny))

    for e in edges(tree_1)
        if last(src(e)) != last(dst(e))
            @assert first(src(e)) == first(dst(e))
            if first(src(e)) != 1
                tree_1 = NamedGraphs.GraphsExtensions.rem_edge(tree_1, e)
            end
        end
    end

    for e in edges(tree_2)
        if first(src(e)) != first(dst(e))
            @assert last(src(e)) == last(dst(e))
            if last(src(e)) != 1
                tree_2 = NamedGraphs.GraphsExtensions.rem_edge(tree_2, e)
            end
        end
    end

    return tree_1, tree_2
end

function build_spin_PEPO(g::NamedGraph, siteinds::Dictionary, Jx, Jy, Jz, hx, hy, hz)
    g1, g2 = grid_spanning_trees(length(unique(first.(vertices(g)))), length(unique(last.(vertices(g)))))
    row_edges, col_edges = filter(e -> first(src(e)) == first(dst(e)), edges(g)), filter(e -> last(src(e)) == last(dst(e)), edges(g))

    s1, s2 = ITensorNetworks.IndsNetwork(g1, 1, siteinds), ITensorNetworks.IndsNetwork(g2, 1, siteinds)

    M1, M2 = ITensors.OpSum(), ITensors.OpSum()
    for e in edges(g)
        if e ∈ col_edges
            if !iszero(Jx)
                M1 += Jx, "X", src(e), "X", dst(e)
            end
            if !iszero(Jy)
                M1 += Jy, "Y", src(e), "Y", dst(e)
            end
            if !iszero(Jz)
                M1 += Jz, "Z", src(e), "Z", dst(e)
            end
        else
            if !iszero(Jx)
                M2 += Jx, "X", src(e), "X", dst(e)
            end
            if !iszero(Jy)
                M2 += Jy, "Y", src(e), "Y", dst(e)
            end
            if !iszero(Jz)
                M2 += Jz, "Z", src(e), "Z", dst(e)
            end
        end
    end

    for v in vertices(g)
        if !iszero(hx)
            M1 += 0.5 * hx, "X", v
            M2 += 0.5 * hx, "X", v
        end
        if !iszero(hy)
            M1 += 0.5 * hy, "Y", v
            M2 += 0.5 * hy, "Y", v
        end
        if !iszero(hz)
            M1 += 0.5 * hz, "Z", v
            M2 += 0.5 * hz, "Z", v
        end
    end

    M1, M2 = ITensorNetworks.ttn(M1, s1; cutoff = 1.0e-16), ITensorNetworks.ttn(M2, s2; cutoff = 1.0e-16)

    M1, M2 = ITensorNetwork(M1), ITensorNetwork(M2)
    M1 = NamedGraphs.GraphsExtensions.add_edges(M1, col_edges)
    M2 = NamedGraphs.GraphsExtensions.add_edges(M2, row_edges)
    M1, M2 = ITensorNetworks.insert_linkinds(M1), ITensorNetworks.insert_linkinds(M2)
    return TN.TensorNetworkState(M1 + M2)
end

function truncate_oper(H::TensorNetworkState; mps_bond_dimension, kwargs...)
    sinds = siteinds(H)
    combiners = Dictionary(collect(vertices(H)), [ITensors.combiner(sinds[v]) for v in collect(vertices(H))])

    H = copy(H)
    for v in vertices(H)
        Hv = H[v]
        Hv = Hv * combiners[v]
        setindex!(H, Hv, v)
    end

    H_trunc = truncate(H; alg = "boundarymps", mps_bond_dimension, kwargs...)

    for v in vertices(H_trunc)
        Hv = H_trunc[v]
        Hv = Hv * dag(combiners[v])
        setindex!(H_trunc, Hv, v)
    end

    return H_trunc
end
using Random

function norm(ψ::TensorNetworkState, v, a::ITensor)
    ψ = copy(ψ)
    TN.setindex_preserve!(ψ, a, v)
    return TN.norm_sqr(ψ; alg = "boundarymps", mps_bond_dimension = 4)
end

function main()
    ITensors.disable_warn_order()
    nx, ny = 5, 5
    g = named_grid((nx, ny))
    Random.seed!(1234)

    ψ = random_tensornetworkstate(ComplexF64, g, TN.siteinds("S=1/2", g); bond_dimension = 2)
    ψ = normalize(ψ; alg = "bp")

    return @show norm(ψ, (3, 3), ψ[(3, 3)])

end
main()
