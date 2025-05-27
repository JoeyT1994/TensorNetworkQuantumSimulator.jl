using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph

using EinExprs: Greedy

using Random
Random.seed!(1634)

function main()
    nx, ny = 5, 5
    χ = 2
    ITensors.disable_warn_order()

    gs = [
        (named_grid((nx, 1)), "line"),
        (named_hexagonal_lattice_graph(nx - 2, ny - 2), "hexagonal"),
        (named_grid((nx, ny)), "square"),
    ]
    for (g, g_str) in gs
        println("Testing for $g_str lattice with $(nv(g)) vertices")
        s = siteinds("S=1/2", g)
        ψ = ITN.random_tensornetwork(ComplexF64, s; link_space = χ)
        s = ITN.siteinds(ψ)
        v_centre = first(G.center(g))

        ψ, _ = TN.symmetric_gauge(ψ)

        println("Computing single site expectation value via various means")

        sz_bp = expect(ψ, ("Z", [v_centre]); alg = "bp")
        println("BP value for Z is $sz_bp")

        boundary_mps_ranks = [1, 2, 4, 8, 16, 32]
        for r in boundary_mps_ranks
            sz_boundarymps = expect(
                ψ,
                ("Z", [v_centre]);
                alg = "boundarymps",
                cache_construction_kwargs = (; message_rank = r),
            )
            println("Boundary MPS Value for Z at Rank $r is $sz_boundarymps")
        end

        sz_exact = expect(ψ, ("Z", [v_centre]); alg = "exact")
        println("Exact value for Z is $sz_exact")

        if !is_tree(g)
            v_centre_neighbor =
                first(filter(v -> first(v) == first(v_centre), neighbors(g, v_centre)))
            println("Computing two site, neighboring, expectation value via various means")

            sz_bp = expect(ψ, ("ZZ", [v_centre, v_centre_neighbor]); alg = "bp")
            println("BP value for ZZ is $sz_bp")

            boundary_mps_ranks = [1, 2, 4, 8, 16, 32]
            for r in boundary_mps_ranks
                sz_boundarymps = expect(
                    ψ,
                    ("ZZ", [v_centre, v_centre_neighbor]);
                    alg = "boundarymps",
                    message_rank = r,
                )
                println("Boundary MPS Value for ZZ at Rank $r is $sz_boundarymps")
            end

            sz_exact = expect(ψ, ("ZZ", [v_centre, v_centre_neighbor]); alg = "exact")
            println("Exact value for ZZ is $sz_exact")
        end
    end
end

main()
