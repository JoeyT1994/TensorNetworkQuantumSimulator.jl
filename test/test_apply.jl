@eval module $(gensym())
using ITensors: datatype
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test


@testset "Test Apply Circuit" begin

    #Custom circuit
    circuit = [("Rx", [(1, 1)], 0.5), ("Rx", [(2, 1)], 0.2), ("CPHASE", [(1, 1), (2, 1)], -0.3)]
    g = build_graph_from_circuit(circuit)
    ψ0 = tensornetworkstate(ComplexF32, v -> "↓", g)
    apply_kwargs = (; maxdim = 2, cutoff = 1.0e-10, normalize_tensors = false)
    ψ, _ = apply_circuit(circuit, ψ0; apply_kwargs, verbose = false)

    @test ψ isa TensorNetworkState
    @test scalartype(ψ) == scalartype(ψ0)
    @test maxvirtualdim(ψ) <= 2
    @test norm_sqr(ψ; alg = "exact") ≈ 1.0

    #Ising circuit on a square grid
    Random.seed!(123)
    g = named_grid((3, 3))

    s = siteinds("S=1", g)
    ψ0 = random_tensornetworkstate(ComplexF32, g; bond_dimension = 1)
    ψ0 = normalize(ψ0; alg = "bp")

    dt = 0.25

    hx = 1.0
    hz = 0.8
    J = 0.5

    #Build a layer of the circuit. Pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
    layer = []
    append!(layer, ("Rx", [v], 2 * hx * dt) for v in vertices(g))
    append!(layer, ("Rz", [v], 2 * hz * dt) for v in vertices(g))

    #For two site gates do an edge coloring to Trotterise the circuit
    ec = edge_color(g, 4)
    for colored_edges in ec
        append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
    end

    apply_kwargs = (cutoff = 1.0e-10, normalize_tensors = false)
    ψ, errs = apply_circuit(layer, ψ0; apply_kwargs, verbose = false)

    @test ψ isa TensorNetworkState
    @test scalartype(ψ) == scalartype(ψ0)
    @test maxvirtualdim(ψ) <= 2
    @test norm_sqr(ψ; alg = "exact") ≈ 1.0
end

end
