@eval module $(gensym())
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using NamedGraphs: neighbors
using Random
using Test: @testset, @test

# Contraction side of the loop cluster expansion (Eq. 7 of arXiv:2510.05647):
# `expect_clusterexpand` assembles per-cluster ratios with the graph-side counting
# numbers. The defining checks are (i) the single-cluster limit reproduces the plain
# BP expectation exactly, and (ii) growing the cluster size moves the estimate toward
# the exact value.
@testset "Loop cluster expansion expectation" begin
    Random.seed!(1234)
    χ = 2

    # On the square lattice the smallest loop is a 4-plaquette, so C < 4 admits no
    # loops: the only region is the seeded target and the estimate must equal `bp`.
    @testset "single-cluster limit == bp ($(g_str))" for (g, g_str) in (
            (named_grid((4, 4)), "square"),
            (named_hexagonal_lattice_graph(3, 3), "hexagonal"),
        )
        ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = χ)
        ψ_bpc = update(BeliefPropagationCache(ψ))
        v = first(center(g))
        w = first(neighbors(g, v))

        for obs in (("Z", v), ("ZZ", [v, w]))
            bp = expect(ψ_bpc, obs; alg = "bp")
            # C = 2 is the bond/site target itself; loops only appear at the lattice
            # girth, so the small-C estimate collapses to the single-cluster value.
            ce = expect_clusterexpand(ψ_bpc, obs; max_configuration_size = 2)
            @test ce ≈ bp atol = 1e-10
        end
    end

    # Loop corrections should reduce the error vs. exact relative to bare BP. We use a
    # single-site observable on the square lattice and a generous cluster size; the
    # expansion need not be monotone in C, but a large-enough cluster must beat BP.
    @testset "loop corrections improve on bp" begin
        g = named_grid((4, 4))
        ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = χ)
        ψ_bpc = update(BeliefPropagationCache(ψ))
        v = first(center(g))

        exact = expect(ψ, ("Z", v); alg = "exact")
        bp = expect(ψ_bpc, ("Z", v); alg = "bp")
        ce = expect_clusterexpand(ψ_bpc, ("Z", v); max_configuration_size = 8)

        @test abs(ce - exact) < abs(bp - exact)
    end

    # Justifies the `loopy_core` reduction at the contraction level: with converged
    # BP messages on the boundary, a region's ratio O_r is unchanged by a *disconnected*
    # protect-free component (it contracts to a scalar that cancels), but a loop reached
    # through a *bridge* imposes a non-BP message across the bridge bond and shifts O_r.
    # So disconnected components may be dropped; bridge-attached loops must be kept.
    @testset "region ratio: disconnected cancels, bridge matters" begin
        g = named_grid((7, 7))
        ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = χ)
        bpc = update(BeliefPropagationCache(ψ))

        obs_vs = [(4, 4), (4, 5)]
        op_strings = ["Z", "Z"]
        A = Set([(4, 4), (4, 5), (5, 4), (5, 5)])           # plaquette holding the bond

        rA = TN._region_ratio(bpc, collect(A), obs_vs, op_strings)
        # disjoint far loop -> cancels
        r_disc = TN._region_ratio(bpc, collect(union(A, Set([(1, 1), (1, 2), (2, 1), (2, 2)]))), obs_vs, op_strings)
        # loop attached by the single bridge edge (4,3)-(4,4) -> matters
        r_bridge = TN._region_ratio(bpc, collect(union(A, Set([(4, 2), (4, 3), (3, 2), (3, 3)]))), obs_vs, op_strings)

        @test r_disc ≈ rA atol = 1e-10
        @test !isapprox(r_bridge, rA; atol = 1e-8)
    end

    # The `TensorNetworkState` overload builds + updates its own BP cache; in the
    # single-cluster limit it must reproduce `expect(ψ, obs; alg="bp")`.
    @testset "TensorNetworkState overload" begin
        g = named_grid((3, 3))
        ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = χ)
        v = first(center(g))

        bp = expect(ψ, ("Z", v); alg = "bp")
        ce = expect_clusterexpand(ψ, ("Z", v); max_configuration_size = 2)
        @test ce ≈ bp atol = 1e-10

        # vector-of-observables overload returns one entry per observable
        res = expect_clusterexpand(ψ, [("Z", v), ("X", v)]; max_configuration_size = 4)
        @test length(res) == 2
    end
end
end
