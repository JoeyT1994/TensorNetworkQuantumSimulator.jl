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

end

@eval module $(gensym())
using LinearAlgebra: Diagonal, I, istriu, norm, svd, transpose
using Random: Random
using TensorNetworkQuantumSimulator: absorb_matrices, absorb_matrices_mul, absorb_matrices_qr,
    gate_split, simple_update_dense!, truncation_strategy
using Test: @test, @testset, @test_throws

# Built from `mapslices` so it shares no machinery with the code under test.
function absorb_reference(tensor, matrices; transposed = false)
    t = copy(tensor)
    for (k, matrix) in enumerate(matrices)
        M = transposed ? matrix : transpose(matrix)
        t = mapslices(v -> M * v, t; dims = k)
    end
    return t
end

# Contract the two vertex tensors over their shared (last) axis, then apply `gate` to the two site
# axes. With invertible environments and their inverses the gauging cancels, so an untruncated
# simple update has to reproduce this exactly.
function pair_reference(t1, t2, gate, n1, n2, d1, d2, b)
    M1, M2 = prod(size(t1)[1:(end - 1)]), prod(size(t2)[1:(end - 1)])
    T = reshape(
        reshape(t1, M1, b) * transpose(reshape(t2, M2, b)),
        size(t1)[1:(end - 1)]..., size(t2)[1:(end - 1)]...
    )
    perm = (n1 + 1, n1 + n2 + 2, ntuple(identity, n1)..., ntuple(i -> n1 + 1 + i, n2)...)
    P = permutedims(T, perm)
    P = reshape(gate * reshape(P, d1 * d2, :), size(P)...)
    return permutedims(P, invperm(perm))
end

# A `middle!` for `simple_update_dense!`: contract the shared bond, apply the gate to the site
# axes, and split with an SVD keeping `maxdim` values. Splits the singular values evenly between
# the two factors, matching `factorize_svd`'s `ortho = "none"`.
function gate_middle(gate, d1, d2, maxdim)
    return function (R1, R2)
        q1, b, q2 = size(R1, 1), size(R1, 3), size(R2, 1)
        M = reshape(R1, q1 * d1, b) * transpose(reshape(R2, q2 * d2, b))
        P = permutedims(reshape(M, q1, d1, q2, d2), (2, 4, 1, 3))
        P = reshape(gate * reshape(P, d1 * d2, q1 * q2), d1, d2, q1, q2)
        M = reshape(permutedims(P, (3, 1, 4, 2)), q1 * d1, q2 * d2)
        F = svd(M)
        k = isnothing(maxdim) ? length(F.S) : min(maxdim, length(F.S))
        w = abs2.(F.S)
        rs = sqrt.(F.S[1:k])
        return reshape(F.U[:, 1:k] * Diagonal(rs), q1, d1, k),
            reshape(transpose(Diagonal(rs) * F.Vt[1:k, :]), q2, d2, k),
            F.S[1:k], sum(w[(k + 1):end]) / sum(w)
    end
end

# Runs one two-site update and compares against `pair_reference`. Environment legs all have
# dimension `a`, so the QR side lengths are a^n vs d*b -- keep a^n >= d*b or the thin QR has no
# tall matrix to work with.
function update_vs_reference(; a1, a2, n1, n2, d1 = 2, d2 = 2, b = 3, maxdim = nothing, normalize_tensors = false)
    t1 = randn(ComplexF64, ntuple(_ -> a1, n1)..., d1, b)
    t2 = randn(ComplexF64, ntuple(_ -> a2, n2)..., d2, b)
    envs1 = ntuple(_ -> randn(ComplexF64, a1, a1), n1)
    envs2 = ntuple(_ -> randn(ComplexF64, a2, a2), n2)
    gate = randn(ComplexF64, d1 * d2, d1 * d2)
    want = pair_reference(t1, t2, gate, n1, n2, d1, d2, b)

    (u1, u2), svals, err = simple_update_dense!(
        (t1, t2), (envs1, envs2),
        (map(m -> transpose(inv(m)), envs1), map(m -> transpose(inv(m)), envs2)),
        gate_middle(gate, d1, d2, maxdim); normalize_tensors,
    )

    k = size(u1)[end]
    got = reshape(
        reshape(u1, prod(size(u1)[1:(end - 1)]), k) * transpose(reshape(u2, prod(size(u2)[1:(end - 1)]), k)),
        size(u1)[1:(end - 1)]..., size(u2)[1:(end - 1)]...
    )
    return (; got, want, u1, u2, svals, err, k)
end

@testset "Dense simple update" begin
    Random.seed!(1234)

    @testset "absorb_matrices absorbs axis by axis, $k matrices" for k in 0:3
        A = randn(ComplexF64, ntuple(_ -> 3, k)..., 2, 4)
        matrices = ntuple(_ -> randn(ComplexF64, 3, 3), k)
        for transposed in (false, true)
            want = absorb_reference(A, matrices; transposed)
            @test absorb_matrices(A, matrices; transposed) ≈ want
        end
    end

    @testset "absorb_matrices leaves its input alone" begin
        A = randn(ComplexF64, 3, 3, 2)
        keep = copy(A)
        absorb_matrices(A, (randn(ComplexF64, 3, 3),))
        @test A == keep
    end

    @testset "absorb_matrices_qr: $elt, transposed = $transposed" for
            elt in (Float64, ComplexF64), transposed in (false, true)

        dims, qrdims = (4, 3), (2, 3)          # deliberately unequal leading dims
        A = randn(elt, dims..., qrdims...)
        matrices = (randn(elt, 4, 4), randn(elt, 3, 3))
        m, n = prod(dims), prod(qrdims)

        want = absorb_reference(A, matrices; transposed)
        Q, R = absorb_matrices_qr(A, matrices; transposed)

        @test size(Q) == (dims..., n)
        @test size(R) == (n, qrdims...)
        Qm, Rm = reshape(Q, m, n), reshape(R, n, n)
        @test Qm * Rm ≈ reshape(want, m, n)
        @test Qm' * Qm ≈ I                     # the QR's Q is isometric
        @test istriu(Rm)
    end

    @testset "absorb_matrices_qr splits at the matrix count" begin
        A = randn(4, 3, 2, 5)
        matrices = (randn(4, 4), randn(3, 3))
        want = absorb_reference(A, matrices)
        Q, R = absorb_matrices_qr(A, matrices)
        @test size(Q) == (4, 3, 10)
        @test size(R) == (10, 2, 5)
        @test reshape(Q, 12, 10) * reshape(R, 10, 10) ≈ reshape(want, 12, 10)
    end

    # A degree-1 vertex gives a wide row block, where Q comes back square.
    @testset "absorb_matrices_qr carries a wide row block, $b bond" for b in (3, 5)
        a, d = 2, 2
        A = randn(ComplexF64, a, d, b)
        envs = (randn(ComplexF64, a, a),)
        inv_envs = map(m -> transpose(inv(m)), envs)
        @assert a < d * b

        Q, R = absorb_matrices_qr(A, envs)
        @test size(Q) == (a, a)
        @test size(R) == (a, d, b)
        @test reshape(Q, a, a) * reshape(R, a, d * b) ≈ reshape(absorb_reference(A, envs), a, d * b)
        @test absorb_matrices_mul(Q, inv_envs, R; transposed = true) ≈ A
    end

    # Absorbing the inverse environments and multiplying `R` back must undo the first half. The
    # inverse direction contracts each matrix's second index, which is what `transposed` does.
    @testset "absorb_matrices_mul inverts absorb_matrices_qr, $k env legs" for k in 1:3
        a = 8
        A = randn(ComplexF64, ntuple(_ -> a, k)..., 2, 3)
        envs = ntuple(_ -> randn(ComplexF64, a, a), k)
        inv_envs = map(m -> transpose(inv(m)), envs)

        Q, R = absorb_matrices_qr(A, envs)
        @test absorb_matrices_mul(Q, inv_envs, R; transposed = true) ≈ A
    end

    # A truncating SVD shrinks `R`'s trailing extent, a grown bond enlarges it.
    @testset "absorb_matrices_mul handles R narrower and wider than chi" begin
        Q = randn(ComplexF64, 8, 8, 2, 6)
        envs = (randn(ComplexF64, 8, 8), randn(ComplexF64, 8, 8))
        for cols in (2, 6, 9)
            R = randn(ComplexF64, 6, 2, cols)
            absorbed = absorb_reference(Q, envs; transposed = true)
            want = reshape(reshape(absorbed, 128, 6) * reshape(R, 6, 2cols), 8, 8, 2, 2, cols)
            u = absorb_matrices_mul(Q, envs, R; transposed = true)
            @test u ≈ want
            @test size(u) == (8, 8, 2, 2, cols)
        end
    end

    # The rank `cutoff` asks for: the fewest values whose discarded weight, relative to the total,
    # stays within it. That is what the sqrt in `truncation_strategy` has to get right.
    @testset "gate_split truncates on the discarded weight" begin
        q1, q2, d, b = 6, 6, 2, 5
        R1 = randn(ComplexF64, q1, d, b)
        R2 = randn(ComplexF64, q2, d, b)
        gate = reshape(randn(ComplexF64, d * d, d * d), d, d, d, d)

        T = reshape(reshape(R1, q1 * d, b) * transpose(reshape(R2, q2 * d, b)), q1, d, q2, d)
        P = permutedims(T, (2, 4, 1, 3))
        P = reshape(reshape(gate, d * d, d * d) * reshape(P, d * d, q1 * q2), d, d, q1, q2)
        M = reshape(permutedims(P, (3, 1, 4, 2)), q1 * d, q2 * d)
        full = svd(M).S
        w = abs2.(full)
        keep(cutoff) = findfirst(k -> sum(w[(k + 1):end]) <= cutoff * sum(w), eachindex(w))

        for kw in ((;), (; maxdim = 3), (; cutoff = 0.2), (; maxdim = 3, cutoff = 0.2))
            L, Rr, svals, err = gate_split(gate, R1, R2; kw...)
            k = min(get(kw, :maxdim, length(full)), keep(get(kw, :cutoff, 0.0)))
            @test length(svals) == k
            @test svals ≈ full[1:k]
            @test err ≈ sum(w[(k + 1):end]) / sum(w)
            @test reshape(L, q1 * d, k) * transpose(reshape(Rr, q2 * d, k)) ≈
                svd(M).U[:, 1:k] * Diagonal(full[1:k]) * svd(M).Vt[1:k, :]
        end

        @test keep(0.2) < length(full)          # the cutoff case must actually drop something
    end

    # Sides differ in size and in environment-leg count, so each side's matrices and axis counts have
    # to travel together or these fail on shape.
    shapes = [
        ("side 1 larger, 1 v 1 legs", (a1 = 16, a2 = 8, n1 = 1, n2 = 1)),
        ("side 2 larger, 1 v 1 legs", (a1 = 8, a2 = 16, n1 = 1, n2 = 1)),
        ("equal sides,   1 v 1 legs", (a1 = 8, a2 = 8, n1 = 1, n2 = 1)),
        ("side 1 larger, 2 v 2 legs", (a1 = 6, a2 = 4, n1 = 2, n2 = 2)),
        ("side 2 larger, 2 v 2 legs", (a1 = 4, a2 = 6, n1 = 2, n2 = 2)),
        ("side 1 larger, 2 v 1 legs", (a1 = 6, a2 = 8, n1 = 2, n2 = 1)),
        ("side 2 larger, 1 v 2 legs", (a1 = 8, a2 = 6, n1 = 1, n2 = 2)),
    ]

    @testset "simple_update_dense! is exact untruncated: $label" for (label, kw) in shapes
        r = update_vs_reference(; kw...)
        @test r.got ≈ r.want
        @test r.err ≈ 0 atol = 1.0e-12
        @test size(r.u1)[1:(end - 1)] == (ntuple(_ -> kw.a1, kw.n1)..., 2)
        @test size(r.u2)[1:(end - 1)] == (ntuple(_ -> kw.a2, kw.n2)..., 2)
    end

    @testset "simple_update_dense! truncates: $label" for (label, kw) in shapes[1:4]
        for maxdim in (3, 2)
            r = update_vs_reference(; kw..., maxdim)
            @test r.k == maxdim
            @test 0 < r.err < 1
        end
    end

    @testset "simple_update_dense! normalizes: $label" for (label, kw) in shapes[1:4]
        r = update_vs_reference(; kw..., normalize_tensors = true)
        @test norm(r.u1) ≈ 1
        @test norm(r.u2) ≈ 1
        @test norm(r.svals) ≈ 1
        # Normalizing rescales the pair by a scalar, so directions still agree.
        s = r.got[1] / r.want[1]
        @test r.got ≈ s .* r.want
    end
end

end
