# Exercises the device code paths without a GPU.
#
# `JLArray` is GPUArrays' reference backend: it stores its data in a plain host array but behaves
# like device memory in the ways that matter here -- scalar indexing is an error, and every
# operation dispatches through GPUArrays rather than hitting a Base fallback. NDTensors ships a
# JLArrays extension, so ITensors' factorisations work on it too.
#
# Every check below stands for a bug that previously only surfaced on real hardware: the message
# arena being built in the wrong memory space, and the blocked message kernel reaching for a
# scalar index.
#
# Two things it deliberately does NOT cover, so don't read a pass here as "the GPU is fine":
#   * `datatype` comes back concrete for JLArray (`JLArray{ComplexF32,1}`) but is a UnionAll for
#     CuArray (`CuArray{T,1,DeviceMemory} where T`), so type-construction bugs hide here.
#   * CUDA-aware MPI, device pointers reaching UCX, and CUDA stream ordering have no analogue.
@eval module $(gensym())
using Test: @test, @test_broken, @testset
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
    #
    # Broken upstream: NDTensorsJLArraysExt's `Expose.qr` passes the bare JLArray to
    # `LinearAlgebra.qr`, which has no GPU method and reaches host `geqrt!` -- "Illegal conversion
    # of a JLArray to a Ptr". Its neighbours (`qr_positive`, `ql`, `eigen`) call `cpu` first.
    # CUDA.jl supplies a cuSOLVER `qr` for CuArray, so this is a JLArray gap, not a GPU one.
    @testset "gate application" begin
        e = first(x for x in TNQS.edges(bpc) if degree(g, TNQS.src(x)) == 3)
        v⃗ = [TNQS.src(e), TNQS.dst(e)]
        @test_broken begin
            out, errs = TNQS.apply_gates(
                [("Rzz", v⃗, 0.3)], bpc;
                apply_kwargs = (; maxdim = chi, cutoff = 0.0),
                bp_update_kwargs = (; maxiter = 2, tolerance = nothing)
            )
            out isa TNQS.BeliefPropagationCache &&
                !(ITensors.data(TNQS.network(out)[first(v⃗)]) isa Array) &&
                all(isfinite, errs)
        end
    end

end
end
