@eval module $(gensym())
using ITensors: ITensors
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test
const TNQS = TensorNetworkQuantumSimulator

@testset "Test Fermionic BoundaryMPS" begin
    ITensors.disable_warn_order()
    Random.seed!(1234)

    g = named_grid((3, 3))

    @testset "spinless fermions, χ=$χ" for χ in (2, 3)
        s = siteinds("fermion", g)
        ψ = random_fermionic_tensornetworkstate(Float64, g, s; bond_dimension = χ)
        ψ = normalize(ψ; alg = "bp")

        z_exact = norm_sqr(ψ; alg = "exact")

        # Boundary-MPS norm should converge to the exact norm with rank.
        ψ_bmps = update(BoundaryMPSCache(ψ, 16))
        @test partitionfunction(ψ_bmps) ≈ z_exact rtol = 1e-8

        # Hermitian hopping on a nearest-neighbour pair in the same row.
        v1, v2 = (2, 1), (2, 2)
        hop_exact =
            expect(ψ, (["Cdag", "C"], [v1, v2]); alg = "exact") +
            expect(ψ, (["Cdag", "C"], [v2, v1]); alg = "exact")
        hop_bmps =
            expect(ψ, (["Cdag", "C"], [v1, v2]); alg = "boundarymps", mps_bond_dimension = 16) +
            expect(ψ, (["Cdag", "C"], [v2, v1]); alg = "boundarymps", mps_bond_dimension = 16)
        @test hop_bmps ≈ hop_exact rtol = 1e-6

        # Parity-forbidden single odd operator must be exactly zero.
        @test only(expect(ψ, (["Cdag"], [v1]); alg = "boundarymps", mps_bond_dimension = 16)) == 0
    end

    @testset "spinful fermions, χ=$χ" for χ in (2, 3)
        s = siteinds("spinful_fermion", g)
        ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = χ)
        ψ = normalize(ψ; alg = "bp")

        z_exact = norm_sqr(ψ; alg = "exact")

        ψ_bmps = update(BoundaryMPSCache(ψ, 16))
        @test partitionfunction(ψ_bmps) ≈ z_exact rtol = 1e-8

        v1, v2 = (2, 1), (2, 2)
        hop_exact =
            expect(ψ, (["Cupdag", "Cup"], [v1, v2]); alg = "exact") +
            expect(ψ, (["Cupdag", "Cup"], [v2, v1]); alg = "exact")
        hop_bmps =
            expect(ψ, (["Cupdag", "Cup"], [v1, v2]); alg = "boundarymps", mps_bond_dimension = 16) +
            expect(ψ, (["Cupdag", "Cup"], [v2, v1]); alg = "boundarymps", mps_bond_dimension = 16)
        @test hop_bmps ≈ hop_exact rtol = 1e-6
    end

    # Hexagonal lattice (non-grid geometry) covers the boundary-MPS path on a
    # graph where partitions have varying lengths.
    @testset "spinless fermions on hexagonal lattice, χ=$χ" for χ in (2, 3)
        gh = named_hexagonal_lattice_graph(3, 3)
        s = siteinds("fermion", gh)
        ψ = random_fermionic_tensornetworkstate(Float64, gh, s; bond_dimension = χ)
        ψ = normalize(ψ; alg = "bp")

        z_exact = norm_sqr(ψ; alg = "exact")

        # The hexagonal geometry has longer-range correlations than the grid, so
        # use a slightly larger boundary-MPS rank to reach the convergence floor.
        ψ_bmps = update(BoundaryMPSCache(ψ, 32))
        @test partitionfunction(ψ_bmps) ≈ z_exact rtol = 1e-7

        # Nearest-neighbour pair sharing a column (same last coordinate).
        v1, v2 = (1, 1), (2, 1)
        hop_exact =
            expect(ψ, (["Cdag", "C"], [v1, v2]); alg = "exact") +
            expect(ψ, (["Cdag", "C"], [v2, v1]); alg = "exact")
        hop_bmps =
            expect(ψ, (["Cdag", "C"], [v1, v2]); alg = "boundarymps", mps_bond_dimension = 32) +
            expect(ψ, (["Cdag", "C"], [v2, v1]); alg = "boundarymps", mps_bond_dimension = 32)
        @test hop_bmps ≈ hop_exact rtol = 1e-5
    end
end

end
