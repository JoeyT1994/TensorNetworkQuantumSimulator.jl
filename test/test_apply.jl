@eval module $(gensym())
using ITensors: ITensors, datatype, op, Index, @OpName_str, @SiteType_str
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws
const TNQS = TensorNetworkQuantumSimulator


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
    append!(layer, ("Rz", v, 2 * hz * dt) for v in vertices(g))

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

@testset "Custom Gate Registration" begin
    # Define a custom op: a Z-axis rotation under a non-built-in name.
    # (Same matrix as the built-in "Rz", under a new name, so we can verify
    # the registered gate dispatches correctly.)
    ITensors.op(::ITensors.OpName"MyZRot", ::ITensors.SiteType"S=1/2", s::Index; θ::Number) =
        exp(-im * (θ / 2) * op("Z", s))

    # Register the dispatch info: name "MyZRot" takes a single keyword `θ`.
    register_gate!("MyZRot"; paramkeys = (:θ,))

    # Apply both the built-in Rz and our newly-registered MyZRot to identical
    # initial states. They should produce the same expectation values.
    g = named_grid((2, 2))
    apply_kwargs = (; maxdim = 2, cutoff = 1.0e-12, normalize_tensors = false)
    θ = 0.7
    v = (1, 1)

    ψ_rz = tensornetworkstate(ComplexF64, w -> "↓", g)
    ψ_my = tensornetworkstate(ComplexF64, w -> "↓", g)
    ψ_rz, _ = apply_gates([("Rx", [v], 0.4), ("Rz", [v], θ)], ψ_rz; apply_kwargs)
    ψ_my, _ = apply_gates([("Rx", [v], 0.4), ("MyZRot", [v], θ)], ψ_my; apply_kwargs)

    @test expect(ψ_rz, ("X", [v]); alg = "exact") ≈ expect(ψ_my, ("X", [v]); alg = "exact")
    @test expect(ψ_rz, ("Y", [v]); alg = "exact") ≈ expect(ψ_my, ("Y", [v]); alg = "exact")
    @test expect(ψ_rz, ("Z", [v]); alg = "exact") ≈ expect(ψ_my, ("Z", [v]); alg = "exact")

    # Aliases work too.
    register_alias!("myzrot", "MyZRot")
    ψ_alias = tensornetworkstate(ComplexF64, w -> "↓", g)
    ψ_alias, _ = apply_gates([("Rx", [v], 0.4), ("myzrot", [v], θ)], ψ_alias; apply_kwargs)
    @test expect(ψ_alias, ("X", [v]); alg = "exact") ≈ expect(ψ_my, ("X", [v]); alg = "exact")

    # register_alias! requires the canonical name to exist.
    @test_throws ArgumentError register_alias!("foo", "DoesNotExist")

    # unregister_gate! removes the gate and any aliases pointing at it.
    unregister_gate!("MyZRot")
    ψ_post = tensornetworkstate(ComplexF64, w -> "↓", g)
    @test_throws ArgumentError apply_gates([("MyZRot", [v], θ)], ψ_post; apply_kwargs)
    @test_throws ArgumentError apply_gates([("myzrot", [v], θ)], ψ_post; apply_kwargs)

    # Built-in gates are locked: register_gate! and unregister_gate! both refuse
    # to operate on names from the canonical registry. Users can only add new
    # gates / aliases, never overwrite the library's own.
    @test_throws ArgumentError register_gate!("Rxx"; paramkeys = (:θ,))
    @test_throws ArgumentError unregister_gate!("Rxx")
    # The built-in still works after the failed attempts.
    ψ_check = tensornetworkstate(ComplexF64, w -> "↓", g)
    ψ_check, _ = apply_gates([("Rxx", [v, (1, 2)], 0.3)], ψ_check; apply_kwargs)
    @test ψ_check isa TensorNetworkState
end

# `apply_gate!` releases its inputs so a two-site update does not pin a full copy of each site
# tensor while the equally large QR intermediates are alive. That is a memory optimisation, so it
# has to be provably free of observable effect.
@testset "simple_update input release" begin
    g = named_hexagonal_lattice_graph(2, 2)
    ψ = random_tensornetworkstate(ComplexF64, g; bond_dimension = 6)
    bpc = update(BeliefPropagationCache(ψ); maxiter = 4, tolerance = nothing)
    apply_kwargs = (; maxdim = 6, cutoff = 1.0e-14)

    for e in TNQS.edges(bpc)
        v⃗ = [src(e), dst(e)]
        gate = TNQS.adapt_gate(
            first(TNQS.toitensor(("Rxx", v⃗, 0.41), TNQS.graph(bpc), siteinds(network(bpc)))),
            bpc
        )
        envs = TNQS.incoming_messages(bpc, v⃗)

        # The consuming path (what apply_gate! uses) must agree exactly with the plain one.
        applied, _ = TNQS.apply_gate!(gate, copy(bpc); v⃗, apply_kwargs)
        keep = ITensors.ITensor[network(bpc)[v] for v in v⃗]
        ref, _, _ = TNQS.simple_update(gate, keep; envs, consume_inputs = false, apply_kwargs...)
        a = network(applied)[v⃗[1]] * network(applied)[v⃗[2]]
        b = ref[1] * ref[2]
        @test isapprox(a, b; atol = 1.0e-12)

        # Not consuming leaves the caller's vector alone, and operating on a copy must never
        # disturb the cache it was copied from.
        @test all(!isempty(ITensors.inds(t)) for t in keep)
        @test !isempty(ITensors.inds(network(bpc)[v⃗[1]]))

        # Consuming empties it, which is the whole point.
        take = ITensors.ITensor[network(bpc)[v] for v in v⃗]
        TNQS.simple_update(gate, take; envs, consume_inputs = true, apply_kwargs...)
        @test all(isempty(ITensors.inds(t)) for t in take)
    end
end

# `blocked_gates!(true)` exists purely to bound peak memory, so it has to be numerically
# indistinguishable from the standard branch -- including when a gate truncates the bond
# (maxdim < χ) and when it grows it (maxdim > χ), which take different paths through the padding
# in `_lmul_q`.
@testset "Blocked two-site gate" begin
    g = named_hexagonal_lattice_graph(2, 2)

    for (chi, normalize_tensors, maxdim) in
        [(4, true, 4), (4, false, 4), (6, true, 3), (6, true, 12)]

        ψ = random_tensornetworkstate(ComplexF64, g; bond_dimension = chi)
        bpc = update(BeliefPropagationCache(ψ); maxiter = 4, tolerance = nothing)
        apply_kwargs = (; maxdim, cutoff = 1.0e-14, normalize_tensors)
        specialised = 0

        for e in TNQS.edges(bpc)
            v⃗ = [src(e), dst(e)]
            gate = TNQS.adapt_gate(
                first(TNQS.toitensor(("Rxx", v⃗, 0.41), TNQS.graph(bpc), siteinds(network(bpc)))),
                bpc
            )
            envs = TNQS.incoming_messages(bpc, v⃗)
            ψ⃗ = ITensors.ITensor[network(bpc)[v] for v in v⃗]

            blocked = TNQS.blocked_two_site_update(
                gate, copy(ψ⃗); envs, normalize_tensors, sqrt_cutoff = nothing,
                consume_inputs = false, apply_kwargs...
            )
            isnothing(blocked) && continue
            specialised += 1
            reference = TNQS.simple_update(gate, copy(ψ⃗); envs, apply_kwargs...)

            # Compared as the contracted pair: the two paths mint different bond Index ids.
            @test isapprox(
                blocked[1][1] * blocked[1][2], reference[1][1] * reference[1][2]; atol = 1.0e-11
            )
            @test isapprox(ITensors.norm(blocked[2]), ITensors.norm(reference[2]); atol = 1.0e-11)
            @test isapprox(blocked[3], reference[3]; atol = 1.0e-11)
        end
        # Guards against the assertions above passing because everything fell back.
        @test specialised > 0
    end

    # And the switch actually routes apply_gates through it.
    ψ = random_tensornetworkstate(ComplexF64, g; bond_dimension = 4)
    apply_kwargs = (; maxdim = 4, cutoff = 1.0e-14)
    circuit = [("Rzz", [src(e), dst(e)], 0.2) for e in edges(g)]
    plain, errs_plain = apply_gates(circuit, ψ; apply_kwargs)
    blocked_gates!(true)
    try
        fast, errs_fast = apply_gates(circuit, ψ; apply_kwargs)
        @test isapprox(errs_fast, errs_plain; atol = 1.0e-10)
        for v in vertices(g)
            @test isapprox(
                expect(plain, ("Z", [v]); alg = "bp"), expect(fast, ("Z", [v]); alg = "bp");
                atol = 1.0e-8
            )
        end
    finally
        blocked_gates!(false)
    end
end

end
