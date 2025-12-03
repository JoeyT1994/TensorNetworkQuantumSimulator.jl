using TensorNetworkQuantumSimulator

using ITensors

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
        ψ = random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = χ)
        v_centre = first(center(g))

        sz_bp = expect(ψ, ("Z", v_centre); alg = "bp")
        println("BP value for Z is $sz_bp")

        println("Computing single site expectation value via various means")

        boundary_mps_ranks = [1, 2, 4, 8, 16, 32]
        for Rmps in boundary_mps_ranks
            sz_boundarymps = expect(
                ψ,
                ("Z", v_centre);
                alg = "boundarymps",
                mps_bond_dimension = Rmps,
            )
            println("Boundary MPS Value for Z at Rank $Rmps is $sz_boundarymps")
        end

        sz_exact = expect(ψ, ("Z", v_centre); alg = "exact")
        println("Exact value for Z is $sz_exact")

        if !is_tree(g)
            v_centre_neighbor = first(neighbors(g, v_centre))
            println("Computing two site, neighboring, expectation value via various means")

            sz_bp = expect(ψ, ("ZZ", [v_centre, v_centre_neighbor]); alg = "bp")
            println("BP value for ZZ is $sz_bp")

            boundary_mps_ranks = [1, 2, 4, 8, 16, 32]
            for Rmps in boundary_mps_ranks
                sz_boundarymps = expect(
                    ψ,
                    ("ZZ", [v_centre, v_centre_neighbor]);
                    alg = "boundarymps",
                    mps_bond_dimension = Rmps,
                )
                println("Boundary MPS Value for ZZ at Rank $Rmps is $sz_boundarymps")
            end

            sz_exact = expect(ψ, ("ZZ", [v_centre, v_centre_neighbor]); alg = "exact")
            println("Exact value for ZZ is $sz_exact")
        end
    end
    return
end

main()
