@eval module $(gensym())
using Dictionaries: Dictionary
using ITensorBase: Index, inds
using Random
using TensorNetworkQuantumSimulator
# `contract` is TNQS-owned; reach it via the `TNQS` alias. `prime`, `inds`, and `randn`
# (over `Index`es) are ITensorBase's; `conj` is `Base`.
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: prime
using Test: @testset, @test, @test_throws


@testset "Test Constructors" begin
    Random.seed!(123)

    #TensorNetwork construction from tensors
    i, j, k, l = Index(2), Index(2), Index(2), Index(2)
    A, B, C, D = randn(i, j), randn(j, k), randn(k, l), randn(l, i)
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

        ψdag = map_virtualinds(prime, map_tensors(conj, ψ))
        @test ψdag isa TensorNetwork
        @test TNQS.contract_network(ψdag; alg = "exact") ≈ conj(TNQS.contract_network(ψ; alg = "exact"))

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
    v, vn = first(vertices(g)), first(neighbors(g, first(vertices(g))))
    @test von_neumann_entanglement_entropy(ψGHZ, first(edges(ψGHZ)); alg = "bp") ≈ log(2)

    #Test identity state constructor
    s = siteinds("S=1/2", g; inds_per_site = 2)
    I = identity_tensornetworkstate(Float64, g, s)
    @test maxvirtualdim(I) == 1
    @test all([length(siteinds(I, v)) == 2 for v in vertices(g)])

    @test_throws ErrorException identity_tensornetworkstate(Float64, g, siteinds("S=1/2", g; inds_per_site = 3))

end

end
