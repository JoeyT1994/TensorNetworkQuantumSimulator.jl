@eval module $(gensym())
using Random
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using Test: @testset, @test

# Fully contract a tensor-network state to a dense tensor over its site indices.
contract_dense(ψ) = TNQS.contract_network([ψ[v] for v in vertices(ψ)])

@testset "Tensor-network addition (direct sum)" begin
    Random.seed!(123)
    for elt in (Float64, ComplexF64)
        g = named_grid((2, 2))
        ψ1 = random_tensornetworkstate(elt, g; bond_dimension = 2)
        s = siteinds(ψ1)
        # `ψ2` shares `ψ1`'s site indices so the two states can be added.
        ψ2 = random_tensornetworkstate(elt, g, s; bond_dimension = 3)
        ψ3 = ψ1 + ψ2
        @test graph(ψ3) == g
        @test siteinds(ψ3) == s
        # Defining property of the direct sum: contracting the sum equals the sum of the
        # contractions.
        @test contract_dense(ψ3) ≈ contract_dense(ψ1) + contract_dense(ψ2)
    end
end
end
