@eval module $(gensym())
using ITensors: datatype, norm
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test
const TNQS = TensorNetworkQuantumSimulator


@testset "Test BP" begin
    Random.seed!(123)
    g = named_comb_tree((3, 3))

    #BP Cache
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetwork(eltype, g; bond_dimension = 2)
        ψ_BPC = BeliefPropagationCache(ψ)
        @test network(ψ_BPC) isa TensorNetwork
        @test ψ_BPC isa BeliefPropagationCache
        @test graph(ψ_BPC) == g
        @test isempty(messages(ψ_BPC))
        @test datatype(ψ_BPC) == datatype(ψ)
        @test scalartype(ψ_BPC) == scalartype(ψ)

        ψ_BPC = update(ψ_BPC)
        @test !isempty(messages(ψ_BPC))
        @test length(keys(messages(ψ_BPC))) == 2 * length(edges(g))
        z_bp = partitionfunction(ψ_BPC)
        @test z_bp ≈ contract(ψ; alg = "exact")
        @test z_bp ≈ contract(ψ; alg = "bp")
    end

    #BP Cache
    s = siteinds("S=1", g)
    for eltype in [Float32, Float64, ComplexF32, ComplexF64]
        ψ = random_tensornetworkstate(eltype, g; bond_dimension = 2)
        ψ_BPC = BeliefPropagationCache(ψ)
        @test ψ_BPC isa BeliefPropagationCache
        @test network(ψ_BPC) isa TensorNetworkState
        @test graph(ψ_BPC) == g
        @test isempty(messages(ψ_BPC))
        @test datatype(ψ_BPC) == datatype(ψ)
        @test scalartype(ψ_BPC) == scalartype(ψ)

        ψ_BPC = update(ψ_BPC)
        @test !isempty(messages(ψ_BPC))
        @test length(keys(messages(ψ_BPC))) == 2 * length(edges(g))
        z_bp = partitionfunction(ψ_BPC)
        @test z_bp ≈ norm_sqr(ψ; alg = "exact")
        @test z_bp ≈ norm_sqr(ψ; alg = "bp")

        vc = first(center(g))
        ρ_bp = reduced_density_matrix(ψ, vc; alg = "bp")
        ρ_exact = reduced_density_matrix(ψ, vc; alg = "exact")
        @test norm(ρ_bp - ρ_exact) <= 10 * eps(real(eltype))
    end
end

@testset "Test contraction sequence cache clearing" begin
    Random.seed!(456)
    g = named_comb_tree((3, 3))

    # Test that sequences are empty before update
    ψ = random_tensornetworkstate(Float64, g; bond_dimension = 2)
    bpc = BeliefPropagationCache(ψ)
    @test isempty(TNQS.contraction_sequences(bpc))

    # Test that sequences are cleared after update returns (only live during update)
    bpc = update(bpc)
    @test isempty(TNQS.contraction_sequences(bpc))
end

@testset "Test setting multiple messages" begin
    g = named_path_graph(2)
    tn = random_tensornetwork(Float64, g; bond_dimension = 2)
    bpc = BeliefPropagationCache(tn)
    e = first(edges(g))
    directed_edges = [e, reverse(e)]
    new_messages = [TNQS.default_message(bpc, edge) for edge in directed_edges]

    @test TNQS.setmessages!(bpc, directed_edges, new_messages) === bpc
    @test all(message(bpc, edge) == new_message for (edge, new_message) in zip(directed_edges, new_messages))
end


# The "blocked" message algorithm specialises the degree-3 vertex of a *double-layer* network and
# must fall back everywhere else. Which layers a vertex has cannot be read off `virtualinds`: a
# single-layer network and a state's norm network both show one virtual index per edge, and a form
# shows two. Getting that wrong is silent in one direction and fatal in the other, so each shape is
# pinned here -- with the hit counter asserted, because "blocked agrees with contract" holds
# trivially whenever blocked fell back.
@testset "Blocked message algorithm, per network shape" begin
    g = named_hexagonal_lattice_graph(2, 2)
    chi = 6
    deg3 = [e for e in TNQS.edges(TNQS.BeliefPropagationCache(
        random_tensornetworkstate(ComplexF64, g; bond_dimension = chi)
    )) if TNQS.degree(g, TNQS.src(e)) == 3]
    @test !isempty(deg3)

    psi = random_tensornetworkstate(ComplexF64, g; bond_dimension = chi)
    # A form is only well-formed when ket and bra share site indices, which is what `inner` builds.
    phi = random_tensornetworkstate(ComplexF64, g, siteinds(psi); bond_dimension = chi)

    shapes = [
        ("norm network", psi, true),
        ("BilinearForm", TNQS.BilinearForm(psi, phi), true),
        # Single layer: `bp_factors` returns one tensor and messages are rank 1, so there is no
        # bra to close against. This used to pass every guard and then throw on `only(setdiff(...))`.
        ("plain TensorNetwork", TNQS.random_tensornetwork(Float64, g; bond_dimension = 4), false),
    ]

    for (label, net, should_specialise) in shapes
        bpc = TNQS._seed_default_messages!(TNQS.BeliefPropagationCache(net))
        hits0 = TNQS._BLOCKED_MESSAGE_HITS[]
        worst = 0.0
        for e in TNQS.edges(bpc)
            b, _ = TNQS.updated_message(
                TNQS.set_default_kwargs(TNQS.Algorithm("blocked"), bpc), bpc, e
            )
            c, _ = TNQS.updated_message(
                TNQS.set_default_kwargs(TNQS.Algorithm("contract"), bpc), bpc, e
            )
            tb, tc = b isa Vector ? only(b) : b, c isa Vector ? only(c) : c
            # The fallback must also preserve the message's shape, not just its values.
            @test Set(collect(TNQS.inds(tb))) == Set(collect(TNQS.inds(tc)))
            worst = max(worst, norm(tb - tc) / norm(tc))
        end
        @test worst < 1.0e-12
        ran = TNQS._BLOCKED_MESSAGE_HITS[] - hits0
        if should_specialise
            @test ran == length(deg3)   # $label specialised every degree-3 vertex
        else
            @test ran == 0              # $label must never reach the kernel
        end
    end

    # A form whose operator is not the identity is a different contraction, so it has to fall back
    # rather than quietly drop the operator.
    let sinds = siteinds(psi), verts = collect(vertices(g))
        ops = [TNQS.ITensors.op("Z", only(sinds[v])) for v in verts]
        opnet = TNQS.TensorNetworkState(TNQS.Dictionary(verts, ops))
        braz = TNQS.map_tensors(t -> TNQS.ITensors.dag(TNQS.ITensors.prime(t)), phi)
        zform = TNQS.BilinearForm(psi, opnet, braz)
        bpc = TNQS._seed_default_messages!(TNQS.BeliefPropagationCache(zform))
        hits0 = TNQS._BLOCKED_MESSAGE_HITS[]
        for e in TNQS.edges(bpc)
            TNQS.updated_message(
                TNQS.set_default_kwargs(TNQS.Algorithm("blocked"), bpc), bpc, e
            )
        end
        @test TNQS._BLOCKED_MESSAGE_HITS[] == hits0
    end
end

end
