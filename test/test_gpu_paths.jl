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
