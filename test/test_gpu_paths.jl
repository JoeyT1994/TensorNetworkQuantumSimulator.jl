# Exercises the device code paths without a GPU.
#
# `JLArray` is GPUArrays' reference backend: it stores its data in a plain host array but behaves
# like device memory in the ways that matter here -- scalar indexing is an error, and every
# operation dispatches through GPUArrays rather than hitting a Base fallback. NDTensors ships a
# JLArrays extension, so ITensors' factorisations work on it too.
#
# Every check below stands for a bug that previously only surfaced on real hardware: a `copyto!`
# between memory spaces silently falling back to an elementwise loop, buffers being allocated on
# the wrong side, and the blocked message kernel reaching for a scalar index.
#
# Two things it deliberately does NOT cover, so don't read a pass here as "the GPU is fine":
#   * `datatype` comes back concrete for JLArray (`JLArray{ComplexF32,1}`) but is a UnionAll for
#     CuArray (`CuArray{T,1,DeviceMemory} where T`), so type-construction bugs hide here.
#   * CUDA-aware MPI, device pointers reaching UCX, and CUDA stream ordering have no analogue.
@eval module $(gensym())
using Test: @test, @testset
using JLArrays: JLArray
using TensorNetworkQuantumSimulator
# Everything else is reached through the package, so this file needs no dependency beyond
# JLArrays -- Adapt, Graphs and ITensors are already in scope inside it.
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: ITensors, Algorithm, adapt, degree, norm

# Minimal stand-in whose buffer constructor reports what the scratch Ref held at allocation time.
mutable struct SpyVec{T} <: AbstractVector{T}
    n::Int
end
Base.size(v::SpyVec) = (v.n,)
Base.getindex(::SpyVec, ::Int) = error("scalar indexing is disallowed")
const SPY_SEEN = Ref{Any}(nothing)
const SPY_REF = Ref{Any}(nothing)
# The storage type comes from `similar(_, UInt8, 0)`, the buffer from `Storage(undef, n)`.
Base.similar(::SpyVec, ::Type{S}, dims::Dims{1}) where {S} = SpyVec{S}(dims[1])
SpyVec{T}(::UndefInitializer, n::Integer) where {T} = (SPY_SEEN[] = SPY_REF[][]; SpyVec{T}(n))

@testset "Device code paths (JLArray)" begin
    g = named_hexagonal_lattice_graph(2, 2)
    chi = 4
    ψ_host = random_tensornetworkstate(ComplexF32, g; bond_dimension = chi)
    ψ = adapt(JLArray, ψ_host)
    bpc_host = update(BeliefPropagationCache(ψ_host); maxiter = 4, tolerance = nothing)
    bpc = adapt(JLArray, bpc_host)

    @testset "device detection" begin
        @test !(ITensors.data(ψ[first(vertices(g))]) isa Array)
        # Drives whether MPI payloads are staged through host memory. A false negative here puts
        # device pointers into MPI, which segfaults rather than erroring.
        @test TNQS._is_device_backed(bpc)
        @test !TNQS._is_device_backed(bpc_host)
    end

    # The arena lives in the network's memory space, is reused while it still matches, and on a
    # rebuild drops the old buffer *before* allocating the replacement. The stand-in's constructor
    # reads the Ref at the moment of allocation, so that last one is observed, not inferred.
    @testset "message arena follows the device and is reused" begin
        ref = Base.RefValue{Any}(nothing)
        SPY_REF[] = ref

        device = TNQS.message_allocator!(ref, adapt(JLArray, zeros(ComplexF32, 4)))
        @test device isa TNQS.TensorOperations.BufferAllocator
        @test !(device.buffer isa Array)                 # device prototype -> device arena
        @test TNQS.message_allocator!(ref, adapt(JLArray, zeros(ComplexF32, 8))) === device

        host = TNQS.message_allocator!(ref, zeros(ComplexF32, 4))
        @test host !== device
        @test host.buffer isa Vector{UInt8}

        # A third storage family forces one more rebuild, and that is the one the spy watches.
        SPY_SEEN[] = :never_called
        spied = TNQS.message_allocator!(ref, SpyVec{ComplexF32}(0))
        @test spied.buffer isa SpyVec{UInt8}
        @test SPY_SEEN[] !== :never_called               # the rebuild really did allocate
        @test SPY_SEEN[] === nothing                     # ...with the old arena already dropped
    end

    # The default hook is a no-op, so nothing in the package depends on CUDA.
    @testset "free_scratch_buffer! default" begin
        @test TNQS.free_scratch_buffer!(zeros(ComplexF32, 4)) === nothing
    end

    # Pre-permuting the closing layer is a property of the backend, not of the network.
    @testset "closer alignment follows the backend" begin
        TO = TNQS.TensorOperations
        @test TNQS._needs_aligned_closer(TO.StridedBLAS())
        @test TNQS._needs_aligned_closer(TO.StridedNative())
        @test !TNQS._needs_aligned_closer(TO.cuTENSORBackend())
        @test TO.select_backend(
            TO.tensorcontract!, zeros(ComplexF32, 2, 2), zeros(ComplexF32, 2, 2),
            zeros(ComplexF32, 2, 2)
        ) isa TO.StridedBackend
    end

    @testset "exchange buffers" begin
        @test !(TNQS._alloc_buffer(bpc, ComplexF32, 16) isa Array)
        @test TNQS._alloc_buffer(bpc_host, ComplexF32, 16) isa Array

        # The five-argument copyto! is the only form with methods for every host/device pairing;
        # the two-argument form on views falls back to scalar indexing and throws.
        want = ComplexF32.(collect(1:8))
        to_host = Vector{ComplexF32}(undef, 8)
        TNQS._copy_range!(to_host, 1, adapt(JLArray, want), 1, 8)
        @test to_host == want

        to_device = adapt(JLArray, zeros(ComplexF32, 8))
        TNQS._copy_range!(to_device, 5, ComplexF32.(collect(1:4)), 1, 4)
        @test Array(to_device) == ComplexF32[0, 0, 0, 0, 1, 2, 3, 4]

        both_device = adapt(JLArray, zeros(ComplexF32, 8))
        TNQS._copy_range!(both_device, 1, adapt(JLArray, want), 1, 8)
        @test Array(both_device) == want
    end

    @testset "blocked message update" begin
        e = first(x for x in TNQS.edges(bpc) if degree(g, TNQS.src(x)) == 3)
        ref, _ = TNQS.updated_message(TNQS.set_default_kwargs(Algorithm("contract"), bpc), bpc, e)
        for b in (1, 2, 64)
            got, _ = TNQS.updated_message(
                TNQS.set_default_kwargs(Algorithm("blocked"; b), bpc), bpc, e
            )
            @test norm(got - ref) / norm(ref) < 1.0f-4
        end

        blocked = update(
            bpc; maxiter = 3, tolerance = nothing,
            message_update_alg = Algorithm("blocked"; b = 2)
        )
        plain = update(bpc; maxiter = 3, tolerance = nothing)
        @test maximum(
            TNQS.message_diff(TNQS.message(blocked, x), TNQS.message(plain, x))
                for x in TNQS.edges(blocked)
        ) < 1.0f-4
    end

    # Covers qr / factorize_svd / the env gauging on a device array type.
    @testset "gate application" begin
        e = first(x for x in TNQS.edges(bpc) if degree(g, TNQS.src(x)) == 3)
        v⃗ = [TNQS.src(e), TNQS.dst(e)]
        out, errs = TNQS.apply_gates(
            [("Rzz", v⃗, 0.3)], bpc;
            apply_kwargs = (; maxdim = chi, cutoff = 0.0),
            bp_update_kwargs = (; maxiter = 2, tolerance = nothing)
        )
        @test out isa TNQS.BeliefPropagationCache
        @test !(ITensors.data(TNQS.network(out)[first(v⃗)]) isa Array)
        @test all(isfinite, errs)
    end

    # The buffer `_apply_q!` hands to `lmul!` has to be a *concrete* device matrix. CUDA declares
    # its Q-multiply on `CuVecOrMat`, so a `SubArray` misses it and falls through to host LAPACK,
    # which throws on a device pointer.
    #
    # JLArray cannot catch that by running the code -- it has no GPU `qr!`/`lmul!` override, so
    # everything below it is host LAPACK and a `SubArray` works fine. The only thing that
    # generalises is the TYPE, so that is what is asserted. The negative cases are the two
    # spellings that look right and are not: a row range of a matrix is non-contiguous and stays
    # a `SubArray` in both.
    @testset "block buffer stays a concrete device array" begin
        buf = adapt(JLArray, zeros(ComplexF32, 12 * 5))
        for rows in (12, 7, 1)                       # full length and two partial prefixes
            b = TNQS._block_buffer(buf, rows, 5)
            @test size(b) == (rows, 5)
            @test !(b isa SubArray)
            @test !(b isa Array)
        end

        # The hazard being avoided, stated as a test so it cannot silently stop being true.
        m = adapt(JLArray, zeros(ComplexF32, 12, 5))
        @test view(m, 2:6, :) isa SubArray            # row range: not contiguous
        @test !(view(m, :, 2:4) isa SubArray)         # column range: contiguous, collapses
    end

    # TSQR on a device array type. This is the check that matters for GPU friendliness: every
    # block copy in it moves data between strided views, and `copyto!` has no specialised method
    # for that -- it walks elementwise, which is scalar indexing and throws here. Running it under
    # JLArray is what proves those are broadcasts.
    #
    # Note what this canNOT prove: JLArray has no GPU `qr!`/`lmul!`, so the factorization itself
    # runs on host LAPACK. Numerical agreement here says nothing about CUSOLVER dispatch.
    @testset "tall-skinny QR, device array" begin
        M = adapt(JLArray, randn(ComplexF32, 256, 16))
        F = TNQS._tall_skinny_qr!(copy(M), 4)
        R = TNQS._qr_r(F, M)
        @test !(R isa Array)

        Q = adapt(JLArray, zeros(ComplexF32, 256, 16))
        Q[1:16, :] .= adapt(JLArray, Matrix{ComplexF32}(one(ComplexF32) * TNQS.LinearAlgebra.I, 16, 16))
        TNQS._apply_q!(F, Q)
        @test !(Q isa Array)
        Qh, Rh, Mh = Array(Q), Array(R), Array(M)
        @test norm(Qh' * Qh - one(ComplexF32) * TNQS.LinearAlgebra.I) < 1.0f-3
        @test norm(Qh * Rh - Mh) / norm(Mh) < 1.0f-3
    end

    # The whole gate, with the QR split, on a device array type.
    @testset "blocked gate with split QR, device array" begin
        e = first(x for x in TNQS.edges(bpc) if degree(g, TNQS.src(x)) == 3)
        v⃗ = [TNQS.src(e), TNQS.dst(e)]
        gate = TNQS.adapt_gate(
            first(TNQS.toitensor(("Rxx", v⃗, 0.41), TNQS.graph(bpc), TNQS.siteinds(TNQS.network(bpc)))),
            bpc
        )
        envs = TNQS.incoming_messages(bpc, v⃗)
        ψ⃗ = [TNQS.network(bpc)[v] for v in v⃗]
        apply_kwargs = (; maxdim = chi, cutoff = 0.0f0, normalize_tensors = true)
        reference = TNQS.simple_update(gate, copy(ψ⃗); envs, apply_kwargs...)

        split = 0
        try
            for limit in (typemax(Int32), 64)
                TNQS.qr_block_limit!(limit)
                TNQS._qr_tall!(adapt(JLArray, randn(ComplexF32, chi^2, 2chi))) isa
                    TNQS.TallSkinnyQR && (split += 1)
                blocked = TNQS.blocked_two_site_update(
                    gate, copy(ψ⃗); envs, normalize_tensors = true, sqrt_cutoff = nothing,
                    consume_inputs = false, apply_kwargs...
                )
                @test !isnothing(blocked)
                @test !(ITensors.data(blocked[1][1]) isa Array)
                @test norm(blocked[1][1] * blocked[1][2] - reference[1][1] * reference[1][2]) /
                    norm(reference[1][1] * reference[1][2]) < 1.0f-3
            end
        finally
            TNQS.qr_block_limit!(TNQS.default_qr_block_limit())
        end
        @test split > 0
    end

    # The memory-bounded gate path drives `qr!` and `lmul!(F.Q, C)` directly rather than going
    # through ITensors' `qr`. What this proves is that the dispatch resolves on a non-Array type,
    # that no step reaches for a scalar index, and that the result stays device-resident. What it
    # cannot prove is that CUDA reaches CUSOLVER's geqrf/ormqr rather than a generic fallback --
    # JLArray is host-backed, so LAPACK succeeds on it for the wrong reason.
    @testset "blocked two-site gate" begin
        e = first(x for x in TNQS.edges(bpc) if degree(g, TNQS.src(x)) == 3)
        v⃗ = [TNQS.src(e), TNQS.dst(e)]
        gate = TNQS.adapt_gate(
            first(TNQS.toitensor(("Rxx", v⃗, 0.41), TNQS.graph(bpc), TNQS.siteinds(TNQS.network(bpc)))),
            bpc
        )
        envs = TNQS.incoming_messages(bpc, v⃗)
        ψ⃗ = [TNQS.network(bpc)[v] for v in v⃗]
        apply_kwargs = (; maxdim = chi, cutoff = 0.0f0, normalize_tensors = true)

        blocked = TNQS.blocked_two_site_update(
            gate, copy(ψ⃗); envs, normalize_tensors = true, sqrt_cutoff = nothing,
            consume_inputs = false, apply_kwargs...
        )
        @test !isnothing(blocked)
        @test !(ITensors.data(blocked[1][1]) isa Array)
        reference = TNQS.simple_update(gate, copy(ψ⃗); envs, apply_kwargs...)
        @test norm(blocked[1][1] * blocked[1][2] - reference[1][1] * reference[1][2]) /
            norm(reference[1][1] * reference[1][2]) < 1.0f-4

        TNQS.blocked_gates!(true)
        try
            out, errs = TNQS.apply_gates(
                [("Rzz", v⃗, 0.3)], bpc;
                apply_kwargs = (; maxdim = chi, cutoff = 0.0f0),
                bp_update_kwargs = (; maxiter = 2, tolerance = nothing)
            )
            @test !(ITensors.data(TNQS.network(out)[first(v⃗)]) isa Array)
            @test all(isfinite, errs)
        finally
            TNQS.blocked_gates!(false)
        end
    end
end
end
