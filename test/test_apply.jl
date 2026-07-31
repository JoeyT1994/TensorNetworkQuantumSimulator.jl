@eval module $(gensym())
using ITensors: ITensors, datatype, op, Index, @OpName_str, @SiteType_str
using Random
using LinearAlgebra
using Graphs: degree
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

    # Every other test in this file uses a complex state, which hides the case that matters most
    # in practice: the state constructors default to `Float64` and every standard rotation is
    # complex, so `zerostate |> apply_gates` promotes. The specialised path builds its
    # factorization at the state's type and multiplies it into a buffer sized from the post-gate
    # SVD, and `lmul!` has no mixed-eltype method -- so this has to fall back, not throw.
    @testset "real state under a complex gate falls back" begin
        g2 = named_grid((3, 3))
        ψ_real = zerostate(g2)
        @test scalartype(ψ_real) == Float64          # the premise, in case a default changes
        circuit = [("Rxx", [(1, 1), (2, 1)], 0.3), ("Rz", [(1, 1)], 0.2)]
        apply_kwargs = (; maxdim = 4, cutoff = 1.0e-14)

        reference, errs_ref = apply_gates(circuit, ψ_real; apply_kwargs)
        TNQS.blocked_gates!(true)
        try
            got, errs = apply_gates(circuit, ψ_real; apply_kwargs)
            @test scalartype(got) == ComplexF64
            @test isapprox(errs, errs_ref; atol = 1.0e-12)
            for v in vertices(g2)
                @test isapprox(
                    expect(got, ("Z", [v]); alg = "exact"),
                    expect(reference, ("Z", [v]); alg = "exact"); atol = 1.0e-10
                )
            end
        finally
            TNQS.blocked_gates!(false)
        end
    end

    # The environments are absorbed by gemms that rotate each gauged leg from one end of the
    # storage order to the other, rather than by `contract`. The lattice tests above cover it
    # end-to-end, but only ever with an environment on *every* row leg, because every bond leaving
    # `v⃗` carries a message. The cases below drive it directly, against an explicit `contract` as
    # the reference.
    @testset "gauging by rotation" begin
        i, j, k, sph = Index(3, "i"), Index(4, "j"), Index(5, "k"), Index(2, "s")
        t = ITensors.random_itensor(ComplexF64, i, j, k, sph)
        allrows = [i, j, k]
        mats = [ITensors.random_itensor(ComplexF64, x, ITensors.prime(x)) for x in allrows]
        matof(x) = mats[findfirst(isequal(x), allrows)]

        # One environment per row leg, two row legs, one ungauged row leg, and none at all: the
        # last two are the paths a lattice cannot produce.
        for gauged in ([i, j, k], [i, k], [j], Index[])
            legs = [
                TNQS.GaugeLeg(ITensors.array(matof(x), x, ITensors.prime(x)), x, ITensors.prime(x)) for x in gauged
            ]
            colinds = [sph]

            M, newrows = TNQS._gauge_matrixize(Base.RefValue{Any}(t), allrows, colinds, legs)
            # Gauged legs rotate to the front, in order; ungauged row legs follow.
            @test newrows == vcat([ITensors.prime(x) for x in gauged], setdiff(allrows, gauged))

            want = t
            for x in gauged
                want = want * matof(x)
            end
            @test isapprox(
                ITensors.itensor(
                    reshape(M, ITensors.dim.(vcat(newrows, colinds))...),
                    vcat(newrows, colinds)...
                ),
                want; atol = 1.0e-11
            )
        end

        # An environment on a column leg is rotated out of the row space by its own gemm, so the
        # caller must fall back rather than hand the QR a matrix that is not the gauged tensor.
        @test isnothing(
            TNQS.blocked_two_site_update(
                ITensors.random_itensor(ComplexF64, sph, ITensors.prime(sph)),
                ITensors.ITensor[t, ITensors.random_itensor(ComplexF64, k, sph)];
                envs = ITensors.ITensor[], normalize_tensors = true, sqrt_cutoff = nothing,
                consume_inputs = false, maxdim = 4, cutoff = 0.0
            )
        )
    end

    # The QR is split into row blocks once it exceeds `qr_block_limit()` -- on a GPU that limit is
    # cuSOLVER's 32-bit dense API, which a χ²xS·χ matrix passes at χ > 812 (S = 4). Lowering the
    # limit here drives the same code path on the host, so the block arithmetic is verified even
    # though the vendor limit itself cannot be.
    @testset "tall-skinny QR" begin
        for (m, n, nb) in [(64, 8, 2), (64, 8, 4), (100, 5, 3), (33, 4, 3), (2048, 16, 7)]
            M = randn(ComplexF64, m, n)
            F = TNQS._tall_skinny_qr!(copy(M), nb)
            R = TNQS._qr_r(F, M)
            Q = zeros(ComplexF64, m, n)
            Q[1:size(R, 1), :] = Matrix{ComplexF64}(LinearAlgebra.I, size(R, 1), n)
            TNQS._apply_q!(F, Q)
            @test LinearAlgebra.norm(Q' * Q - LinearAlgebra.I) < 1.0e-10   # orthonormal
            @test LinearAlgebra.norm(Q * R - M) / LinearAlgebra.norm(M) < 1.0e-10   # reconstructs
        end

        # A wide matrix (a degree-2 vertex) must never be split: its blocks would be rank
        # deficient and the two-level product would not be a QR.
        TNQS.qr_block_limit!(16)
        try
            @test !(TNQS._qr_tall!(randn(ComplexF64, 4, 8)) isa TNQS.TallSkinnyQR)
            @test TNQS._qr_tall!(randn(ComplexF64, 64, 8)) isa TNQS.TallSkinnyQR
        finally
            TNQS.qr_block_limit!(typemax(Int32))
        end
    end

    # Splitting the QR must not change the answer, at any block count.
    @testset "blocked gate is block-count invariant" begin
        ψ = random_tensornetworkstate(ComplexF64, g; bond_dimension = 6)
        bpc = update(BeliefPropagationCache(ψ); maxiter = 4, tolerance = nothing)
        apply_kwargs = (; maxdim = 6, cutoff = 1.0e-14, normalize_tensors = true)
        e = first(x for x in TNQS.edges(bpc) if degree(TNQS.graph(bpc), src(x)) == 3)
        v⃗ = [src(e), dst(e)]
        gate = TNQS.adapt_gate(
            first(TNQS.toitensor(("Rxx", v⃗, 0.41), TNQS.graph(bpc), siteinds(network(bpc)))), bpc
        )
        envs = TNQS.incoming_messages(bpc, v⃗)
        ψ⃗ = ITensors.ITensor[network(bpc)[v] for v in v⃗]
        reference = TNQS.simple_update(gate, copy(ψ⃗); envs, apply_kwargs...)

        split = 0
        try
            for limit in (typemax(Int32), 4096, 512, 64)
                TNQS.qr_block_limit!(limit)
                TNQS._qr_tall!(randn(ComplexF64, 36, 12)) isa TNQS.TallSkinnyQR && (split += 1)
                blocked = TNQS.blocked_two_site_update(
                    gate, copy(ψ⃗); envs, normalize_tensors = true, sqrt_cutoff = nothing,
                    consume_inputs = false, apply_kwargs...
                )
                @test !isnothing(blocked)
                @test isapprox(
                    blocked[1][1] * blocked[1][2], reference[1][1] * reference[1][2]; atol = 1.0e-11
                )
            end
        finally
            TNQS.qr_block_limit!(typemax(Int32))
        end
        @test split > 0    # the blocked path was actually taken, not just the single-block one
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
