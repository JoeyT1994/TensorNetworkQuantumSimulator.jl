@eval module $(gensym())
using ITensors: ITensors, datatype, op, Index, @OpName_str, @SiteType_str
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws


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

end

@eval module $(gensym())
using LinearAlgebra: Diagonal, I, istriu, norm, svd, transpose
using Random: Random
using TensorNetworkQuantumSimulator: absorb_boundary_in!, absorb_boundary_out!, absorb_chain!,
    absorb_matrices!, absorb_matrices_mul!, absorb_matrices_qr!, mul_strided_batched!,
    simple_update_dense!
using Test: @test, @testset, @test_throws

# Contract `matrices[k]`'s first index with axis `inds[k]`, then permute to (inds..., qrinds...).
# Built from `mapslices` so it shares no machinery with the code under test.
function absorb_reference(tensor, matrices, inds, qrinds; op = identity)
    t = copy(tensor)
    for (k, matrix) in enumerate(matrices)
        M = transpose(op(matrix))
        t = mapslices(v -> M * v, t; dims = inds[k])
    end
    return permutedims(t, (inds..., qrinds...))
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
        (copy(t1), copy(t2)), (envs1, envs2),
        (map(m -> transpose(inv(m)), envs1), map(m -> transpose(inv(m)), envs2)),
        ((n1 + 1, n1 + 2), (n2 + 1, n2 + 2)), gate_middle(gate, d1, d2, maxdim);
        normalize_tensors,
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

    @testset "mul_strided_batched! covers every slice, trail = $trail" for trail in (1, 2, 5)
        lead, chi = 3, 4
        A = randn(lead, chi, trail)
        B = randn(chi, chi)
        C = similar(A)
        mul_strided_batched!(C, A, B)
        @test all(C[:, :, t] ≈ A[:, :, t] * B for t in 1:trail)
        Ct = similar(A)
        mul_strided_batched!(Ct, A, transpose(B))
        @test all(Ct[:, :, t] ≈ A[:, :, t] * transpose(B) for t in 1:trail)
    end

    # `absorb_chain!` hands back the buffer holding the result and the one left free, whichever way
    # the alternation happened to land.
    @testset "absorb_chain! returns (result, free) for $k matrices" for k in 0:3
        A = randn(3, 3, 3, 2)
        matrices = ntuple(_ -> randn(3, 3), k)
        want = absorb_reference(A, matrices, ntuple(identity, k), Tuple((k + 1):4))
        live, free = absorb_chain!(copy(A), similar(A), matrices)
        @test live ≈ want
        @test pointer(free) != pointer(live)
        @test length(free) == length(live)
    end

    # A caller that lends out part of a larger buffer needs all of it back, so a supplied `scratch`
    # always comes back free with the result in the input's own storage -- including for an even
    # matrix count, where the bare alternation would end the other way round.
    @testset "absorb_matrices! honours a supplied scratch, $k matrices" for k in 1:3
        tensor = randn(ComplexF64, ntuple(_ -> 8, k)..., 2, 3)
        matrices = ntuple(_ -> randn(ComplexF64, 8, 8), k)
        inds = ntuple(identity, k)
        qrinds = (k + 1, k + 2)
        want = absorb_reference(tensor, matrices, inds, qrinds)

        input = copy(tensor)
        base = pointer(input)
        scratch = Vector{ComplexF64}(undef, length(tensor))
        live, free = absorb_matrices!(input, matrices, inds, qrinds; scratch)
        @test live ≈ want
        @test pointer(live) == base
        @test pointer(free) == pointer(scratch)
    end

    @testset "absorb_matrices_qr!: $elt, op = $op" for elt in (Float64, ComplexF64),
            op in (identity, transpose)

        dims, qrdims = (4, 3), (2, 3)          # deliberately unequal leading dims
        A = randn(elt, dims..., qrdims...)
        matrices = (randn(elt, 4, 4), randn(elt, 3, 3))
        m, n = prod(dims), prod(qrdims)

        want = absorb_reference(A, matrices, (1, 2), (3, 4); op)
        Q, R, free = absorb_matrices_qr!(copy(A), matrices, (1, 2), (3, 4); op)

        @test size(Q) == (dims..., n)
        @test size(R) == (n, qrdims...)
        Qm, Rm = reshape(Q, m, n), reshape(R, n, n)
        @test Qm * Rm ≈ reshape(want, m, n)
        @test Qm' * Qm ≈ I                     # the QR's Q is isometric
        @test istriu(Rm)
        # `free` is meant to be handed straight to `absorb_matrices_mul!` as its scratch.
        @test length(free) == length(Q)
        @test pointer(free) != pointer(Q)
    end

    @testset "absorb_matrices_qr! index handling" begin
        A = randn(4, 3, 2, 3)
        matrices = (randn(4, 4), randn(3, 3))
        Q1, R1, _ = absorb_matrices_qr!(copy(A), matrices, (1, 2))
        Q2, R2, _ = absorb_matrices_qr!(copy(A), matrices, (1, 2), (3, 4))
        @test Q1 ≈ Q2                          # default qrinds is the complement of inds
        @test R1 ≈ R2

        # A non-ascending `qrinds` is honoured rather than sorted.
        B = randn(4, 3, 2, 5)
        want = absorb_reference(B, matrices, (1, 2), (4, 3))
        Q, R, _ = absorb_matrices_qr!(copy(B), matrices, (1, 2), (4, 3))
        @test size(R) == (10, 5, 2)
        @test reshape(Q, 12, 10) * reshape(R, 10, 10) ≈ reshape(want, 12, 10)

        @test_throws DimensionMismatch absorb_matrices_qr!(
            copy(A), matrices, (1, 2), (3, 4); scratch = Vector{Float64}(undef, 5)
        )
    end

    # Absorbing the inverse environments and multiplying `R` back must undo the first half. The
    # inverse direction contracts each matrix's second index, which is what `op = transpose` does.
    @testset "absorb_matrices_mul! inverts absorb_matrices_qr!, $k env legs" for k in 1:3
        a = 8
        A = randn(ComplexF64, ntuple(_ -> a, k)..., 2, 3)
        envs = ntuple(_ -> randn(ComplexF64, a, a), k)
        inv_envs = map(m -> transpose(inv(m)), envs)
        inds, qrinds = ntuple(identity, k), (k + 1, k + 2)

        Q, R, free = absorb_matrices_qr!(copy(A), envs, inds, qrinds)
        u, spare = absorb_matrices_mul!(Q, inv_envs, R; op = transpose, scratch = free)
        @test u ≈ A
        @test length(spare) == length(Q)
        @test pointer(spare) != pointer(u)
    end

    # `R`'s trailing extent is not `chi`: a truncating SVD makes it smaller and one that grows the
    # bond makes it larger, and only the first case fits in the buffer left over.
    @testset "absorb_matrices_mul! handles R narrower and wider than chi" begin
        Q = randn(ComplexF64, 8, 8, 6)
        envs = (randn(ComplexF64, 8, 8), randn(ComplexF64, 8, 8))
        for cols in (2, 6, 9)
            R = randn(ComplexF64, 6, 2, cols)
            absorbed = absorb_chain!(copy(Q), similar(Q), envs; op = transpose)[1]
            want = reshape(reshape(absorbed, 64, 6) * reshape(R, 6, 2cols), 8, 8, 2, cols)
            u, spare = absorb_matrices_mul!(copy(Q), envs, R; op = transpose)
            @test u ≈ want
            @test size(u) == (8, 8, 2, cols)
            @test length(spare) == length(Q)
        end
    end

    # Across a cut edge there is no partner tensor to fall back on, so the wide case must be carried.
    @testset "absorb_boundary_in! delegates when the matrix is tall, $k env legs" for k in 1:3
        a = 8
        A = randn(ComplexF64, ntuple(_ -> a, k)..., 2, 3)
        envs = ntuple(_ -> randn(ComplexF64, a, a), k)
        inds, qrinds = ntuple(identity, k), (k + 1, k + 2)
        @assert a^k >= 6                       # tall, so this must take the QR branch

        Qw, Rw, _ = absorb_matrices_qr!(copy(A), envs, inds, qrinds)
        Q, R, free = absorb_boundary_in!(copy(A), envs, inds, qrinds)
        @test !isnothing(Q)
        @test Q ≈ Qw
        @test R ≈ Rw

        # `absorb_boundary_out!` is then the same round trip as the pair it wraps.
        inv_envs = map(m -> transpose(inv(m)), envs)
        @test absorb_boundary_out!(Q, inv_envs, R, free; op = transpose) ≈ A
    end

    # One environment leg against a site and a bond is wide, so the whole absorbed tensor stands in
    # for `R` and comes back with the environment legs still attached.
    @testset "absorb_boundary_in! carries the wide case, $b bond" for b in (3, 5)
        a, d = 2, 2
        A = randn(ComplexF64, a, d, b)
        envs = (randn(ComplexF64, a, a),)
        inv_envs = map(m -> transpose(inv(m)), envs)
        @assert a < d * b                      # wide, so this must take the no-QR branch

        Q, R, free = absorb_boundary_in!(copy(A), envs, (1,), (2, 3))
        @test isnothing(Q)
        @test R ≈ absorb_reference(A, envs, (1,), (2, 3))
        @test size(R) == (a, d, b)

        # A truncation that shrank the bond, and one that grew it past the buffer left behind.
        for newb in (2, b, 3b)
            Rp = randn(ComplexF64, a, d, newb)
            want = absorb_reference(Rp, inv_envs, (1,), (2, 3); op = transpose)
            @test absorb_boundary_out!(nothing, inv_envs, copy(Rp), free; op = transpose) ≈ want
        end
    end

    @testset "absorb_boundary_in! with no environment legs" begin
        A = randn(ComplexF64, 2, 3)
        Q, R, free = absorb_boundary_in!(copy(A), (), (), (1, 2))
        @test isnothing(Q)
        @test R ≈ A
        @test absorb_boundary_out!(nothing, (), copy(A), free) ≈ A
    end

    # Sides differ in size and in environment-leg count, so the larger-side-first reordering has to
    # carry each side's matrices and axis labels with it or these fail on shape.
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
