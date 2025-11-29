@eval module $(gensym())
using Dictionaries: Dictionary
using ITensors: ITensors, Index, dag, inds, prime
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test Constructors" begin
    Random.seed!(123)

    #TensorNetwork construction from tensors
    i, j, k, l = Index(2), Index(2), Index(2), Index(2)
    A, B, C, D = ITensors.random_itensor(i, j), ITensors.random_itensor(j, k), ITensors.random_itensor(k, l), ITensors.random_itensor(l, i)
    t = TensorNetwork([A, B, C, D])
    @test t isa TensorNetwork
    @test scalartype(t) == eltype(A)
    @test maxvirtualdim(t) == 2
    @test graph(t) isa NamedGraph
    @test graph(t) == add_edge(named_path_graph(4), 1 => 4)

    #TensorNetwork pre-defined constructor
    g = named_hexagonal_lattice_graph(3, 3)
    χ = 3
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetwork(eltype, g; bond_dimension = χ)
        @test ψ isa TensorNetwork
        @test scalartype(ψ) == eltype
        @test graph(ψ) == g
        @test maxvirtualdim(ψ) == 3
        @test all([length(inds(ψ[v])) == degree(g, v) for v in vertices(ψ)])

        ψdag = map_virtualinds(prime, map_tensors(dag, ψ))
        @test ψdag isa TensorNetwork
        @test ITensors.contract(ψdag; alg = "exact") ≈ conj(ITensors.contract(ψ; alg = "exact"))

        v = first(vertices(g))
        rem_vertex!(ψ, v)
        @test graph(ψ) == rem_vertex(g, v)
        @test !all([length(inds(ψ[v])) == degree(g, v) for v in vertices(ψ)])
    end

    #SiteInds
    s = siteinds("S=1/2", g)
    @test s isa Dictionary
    @test keys(s) == vertices(g)
    @test all([s[v] isa Vector{<:Index} for v in vertices(g)])
    @test all([length(s[v]) == 1 for v in vertices(g)])

    #TensorNetworkState
    χ = 3
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetworkstate(eltype, g, s; bond_dimension = χ)
        @test ψ isa TensorNetworkState
        @test scalartype(ψ) == eltype
        @test siteinds(ψ) == s
        @test graph(ψ) == g
        @test maxvirtualdim(ψ) == 3
        @test all([length(inds(ψ[v])) == degree(g, v) + 1 for v in vertices(ψ)])
        @test all([siteinds(ψ, v) == s[v] for v in vertices(ψ)])

        ψ = tensornetworkstate(eltype, v -> "X+", g, "S=1/2")
        @test maxvirtualdim(ψ) == 1
        @test scalartype(ψ) == eltype
        @test all([length(inds(ψ[v])) == degree(g, v) + 1 for v in vertices(ψ)])
    end

    #Test GHZ state constructor
    ψ1, ψ2 = tensornetworkstate(Float64, v -> "↑", g, s), tensornetworkstate(Float64, v -> "↓", g, s)
    ψGHZ = ψ1 + ψ2
    @test ψGHZ isa TensorNetworkState
    @test maxvirtualdim(ψGHZ) == 2
    @test entanglement(ψGHZ, first(edges(ψGHZ)); alg = "bp") ≈ log(2)

end

end
