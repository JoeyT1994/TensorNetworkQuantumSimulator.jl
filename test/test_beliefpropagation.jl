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


# The kernel must fire on a double layer and fall back everywhere else, and that cannot be read off
# `virtualinds`: a single-layer network and a norm network both show one virtual index per edge. The
# hit counter is asserted because "blocked agrees with contract" holds trivially on a fallback, and
# it must fire on *every* edge, not only where the source has degree 3.
@testset "Blocked message algorithm, per network shape" begin
    g = named_hexagonal_lattice_graph(2, 2)
    chi = 6
    psi = random_tensornetworkstate(ComplexF64, g; bond_dimension = chi)
    nedges = length(collect(TNQS.edges(TNQS.BeliefPropagationCache(psi))))
    @test nedges > 0
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
            @test ran == nedges         # $label specialised every edge, whatever the degree
        else
            @test ran == 0              # $label must never reach the kernel
        end
    end

    # A non-identity operator is one more tensor in the network, not a fallback. The identity is
    # still recognised and dropped.
    let sinds = siteinds(psi), verts = collect(vertices(g))
        ops = [TNQS.ITensors.op("Z", only(sinds[v])) for v in verts]
        opnet = TNQS.TensorNetworkState(TNQS.Dictionary(verts, ops))
        braz = TNQS.map_tensors(t -> TNQS.ITensors.dag(TNQS.ITensors.prime(t)), phi)
        forms = [
            ("BilinearForm with Z", TNQS.BilinearForm(psi, opnet, braz)),
            ("QuadraticForm with Z", TNQS.QuadraticForm(psi, opnet)),
        ]
        for (label, form) in forms
            bpc = TNQS._seed_default_messages!(TNQS.BeliefPropagationCache(form))
            hits0 = TNQS._BLOCKED_MESSAGE_HITS[]
            worst = 0.0
            for e in TNQS.edges(bpc)
                b, _ = TNQS.updated_message(
                    TNQS.set_default_kwargs(TNQS.Algorithm("blocked"), bpc), bpc, e
                )
                c, _ = TNQS.updated_message(
                    TNQS.set_default_kwargs(TNQS.Algorithm("contract"), bpc), bpc, e
                )
                worst = max(worst, norm(b - c) / norm(c))
            end
            @test worst < 1.0e-12
            @test TNQS._BLOCKED_MESSAGE_HITS[] - hits0 == nedges   # $label
        end
    end
end


# A degree-4 vertex -- what a square lattice is made of -- has three incoming messages, which the
# hand-written kernel could not express at all.
@testset "Blocked message on a degree-4 vertex" begin
    g = named_grid((3, 3))
    @test any(TNQS.degree(g, v) == 4 for v in vertices(g))
    chi = 4
    psi = random_tensornetworkstate(ComplexF64, g; bond_dimension = chi)
    phi = random_tensornetworkstate(ComplexF64, g, siteinds(psi); bond_dimension = chi)

    for (label, net) in (("norm network", psi), ("BilinearForm", TNQS.BilinearForm(psi, phi)))
        bpc = TNQS._seed_default_messages!(TNQS.BeliefPropagationCache(net))
        hits0 = TNQS._BLOCKED_MESSAGE_HITS[]
        worst = 0.0
        nedges = 0
        for e in TNQS.edges(bpc)
            nedges += 1
            b, _ = TNQS.updated_message(
                TNQS.set_default_kwargs(TNQS.Algorithm("blocked"), bpc), bpc, e
            )
            c, _ = TNQS.updated_message(
                TNQS.set_default_kwargs(TNQS.Algorithm("contract"), bpc), bpc, e
            )
            @test Set(collect(TNQS.inds(b))) == Set(collect(TNQS.inds(c)))
            worst = max(worst, norm(b - c) / norm(c))
        end
        @test worst < 1.0e-12
        @test TNQS._BLOCKED_MESSAGE_HITS[] - hits0 == nedges   # $label

        # `vertex_scalar` takes this path unconditionally, so it has to agree here too.
        for v in vertices(g)
            fast = TNQS.blocked_vertex_scalar(bpc, v)
            @test !isnothing(fast)
            cl = [TNQS.bp_factors(bpc, v); TNQS.incoming_messages(bpc, v)]
            slow = TNQS.scalar(
                TNQS.contract(cl; sequence = TNQS.contraction_sequence(cl; alg = "optimal"))
            )
            @test abs(fast - slow) / abs(slow) < 1.0e-12
        end
    end
end


# The form path streams the ket one `l_e` slab at a time rather than aligning it whole, and writes
# each slab through a view with one axis per index. A state with a *single* site index happens to
# have exactly as many axes as the collapsed `(S, ka, kb, nb)` block, which hid an ndims mismatch;
# the superket shape this exists for has two. Both layer counts are checked, since only the form
# streams.
@testset "Blocked message with multiple site indices" begin
    g = named_hexagonal_lattice_graph(2, 2)
    verts = collect(vertices(g))
    # The element type must be `Vector{<:Index}`, not `Vector{Index}` -- `TensorNetworkState`'s
    # constructor is written against the covariant form and will not match otherwise.
    sinds = TNQS.Dictionary{eltype(verts), Vector{<:TNQS.ITensors.Index}}(
        verts,
        [TNQS.ITensors.Index[
            TNQS.ITensors.Index(2, "p$v"), TNQS.ITensors.Index(2, "a$v")
        ] for v in verts]
    )
    psi = random_tensornetworkstate(ComplexF64, g, sinds; bond_dimension = 5)
    phi = random_tensornetworkstate(ComplexF64, g, sinds; bond_dimension = 5)

    for (label, net) in (("norm network", psi), ("BilinearForm", TNQS.BilinearForm(psi, phi)))
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
            @test Set(collect(TNQS.inds(tb))) == Set(collect(TNQS.inds(tc)))
            worst = max(worst, norm(tb - tc) / norm(tc))
        end
        @test worst < 1.0e-12
        @test TNQS._BLOCKED_MESSAGE_HITS[] > hits0    # the kernel really ran for $label
    end
end


# cuTENSOR dispatches on the exact `(eltype(A), eltype(B), eltype(C))` triple and defines no mixed
# real/complex contraction, so the kernel has to promote every operand before handing it over. This
# is invisible without a deliberate test: Strided and Base promote quietly, so a network that is
# real in one place and complex everywhere else contracts fine on CPU and dies on a GPU as
# `KeyError: (ComplexF32, Float32, ComplexF32)` from inside the backend, naming no tensor.
#
# The mix is reachable in practice. `datatype(tn)` reduces with `promote_type`, which has no rule
# for storage types, so `promote_type(Vector{ComplexF32}, Vector{Float64})` falls back to `typejoin`
# and yields a bare `Vector` -- and `adapt` to a free eltype is a no-op, so `default_message` seeds
# a real `delta` into an otherwise complex network. A real on-site operator does the same.
@testset "Blocked message promotes mixed eltypes" begin
    g = named_hexagonal_lattice_graph(2, 2)
    verts = collect(vertices(g))
    sinds = TNQS.Dictionary{eltype(verts), Vector{<:TNQS.ITensors.Index}}(
        verts,
        [TNQS.ITensors.Index[
            TNQS.ITensors.Index(2, "p$v"), TNQS.ITensors.Index(2, "a$v")
        ] for v in verts]
    )
    psi = random_tensornetworkstate(ComplexF32, g, sinds; bond_dimension = 4)

    # A real identity operator layer against a complex ket -- what a `QuadraticForm` holds on any
    # vertex the perturbation does not touch.
    identnet = TNQS.TensorNetworkState(
        TNQS.Dictionary(
            verts,
            [reduce(
                    *,
                    TNQS.ITensors.ITensor[
                        TNQS.ITensors.denseblocks(TNQS.ITensors.delta(s, TNQS.prime(s)))
                            for s in sinds[v]
                    ]
                ) for v in verts]
        )
    )
    psir = random_tensornetworkstate(Float32, g, sinds; bond_dimension = 4)

    cases = [
        ("complex net, real messages", psi, true),
        ("real operator, complex ket", TNQS.QuadraticForm(psi, identnet), false),
        ("real op + real messages", TNQS.QuadraticForm(psi, identnet), true),
        # The promotion falling the other way: a real ket under a real operator stays real, so the
        # kernel must not gratuitously widen either.
        ("real ket throughout", TNQS.QuadraticForm(psir, identnet), false),
    ]

    for (label, net, realise_messages) in cases
        bpc = TNQS._seed_default_messages!(TNQS.BeliefPropagationCache(net))
        if realise_messages
            for e in TNQS.edges(bpc)
                m = message(bpc, e)
                m isa TNQS.ITensors.ITensor || continue
                TNQS.setmessages!(
                    bpc, [e],
                    [TNQS.ITensors.itensor(real.(TNQS.array(m)), TNQS.inds(m)...)]
                )
            end
        end
        hits0 = TNQS._BLOCKED_MESSAGE_HITS[]
        worst, nedges = 0.0, 0
        for e in TNQS.edges(bpc)
            nedges += 1
            # Without the promotion this throws the kernel's own mixed-eltype assertion, which is
            # the CPU-visible stand-in for the backend's `KeyError`.
            b, _ = TNQS.updated_message(
                TNQS.set_default_kwargs(TNQS.Algorithm("blocked"), bpc), bpc, e
            )
            c, _ = TNQS.updated_message(
                TNQS.set_default_kwargs(TNQS.Algorithm("contract"), bpc), bpc, e
            )
            tb, tc = b isa Vector ? only(b) : b, c isa Vector ? only(c) : c
            worst = max(worst, norm(tb - tc) / norm(tc))
        end
        @test TNQS._BLOCKED_MESSAGE_HITS[] - hits0 == nedges    # $label never fell back
        @test worst < 1.0e-5                                    # $label agrees with contract
    end
end

end
