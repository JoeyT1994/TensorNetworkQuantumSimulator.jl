@eval module $(gensym())
using Dictionaries: Dictionary
using ITensors: datatype
using Random
using Statistics
using TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test Expect" begin
    nx, ny = 4, 4
    χ = 2

    gs = [
        (named_grid((nx, 1)), "line"),
        (named_hexagonal_lattice_graph(nx - 2, ny - 2), "hexagonal"),
        (named_grid((nx, ny)), "square"),
    ]
    for (g, g_str) in gs
        ψ = random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = χ)
        v_centre = first(center(g))

        sz_exact = expect(ψ, ("Z", [v_centre]); alg = "exact")
        sz_bp = expect(ψ, ("Z", [v_centre]); alg = "bp")

        if is_tree(g)
            @test sz_bp ≈ sz_exact
        else
            @test sz_bp != sz_exact
        end

        Rmps = 16
        sz_boundarymps = expect(ψ, ("Z", [v_centre]); alg = "boundarymps", mps_bond_dimension = Rmps)

        @test sz_boundarymps ≈ sz_exact

        if !is_tree(g)
            v_centre_neighbor = first(neighbors(g, v_centre))
            sz_exact = expect(ψ, ("ZZ", [v_centre, v_centre_neighbor]); alg = "exact")
            sz_boundarymps = expect(ψ, ("ZZ", [v_centre, v_centre_neighbor]); alg = "boundarymps", mps_bond_dimension = Rmps)

            @test sz_boundarymps ≈ sz_exact
        end
    end
end

end
