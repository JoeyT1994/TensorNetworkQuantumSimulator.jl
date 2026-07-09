# Multi-GPU CUDA equivalence tests.
#
# Run on a node with >=2 functional CUDA devices:
#   TNQS_TEST_MULTIGPU=true julia --project=. test/runtests.jl

using Test
using Random
using CUDA
using Adapt: adapt
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using NamedGraphs.NamedGraphGenerators: named_hexagonal_lattice_graph
using NamedGraphs: NamedEdge
using Graphs: vertices, edges, src, dst

to_host(t) = adapt(Array, t)

@testset "MultiGPU CUDA equivalence" begin
    Random.seed!(1234)

    g = named_hexagonal_lattice_graph(3, 3)
    s = TN.siteinds("S=1/2", g)
    χ = 2
    ψ = TN.random_tensornetworkstate(ComplexF64, g, s; bond_dimension = χ)

    maxiter = 400
    tolerance = 1.0e-12
    α = 1.0

    # Reference: single host/owner BP cache.
    ref = TN.BeliefPropagationCache(ψ)
    ref = TN.update(ref; alg = "bp", maxiter, tolerance)

    # Distributed: partitioned across all available devices (Float64 to match ref).
    n_devices = length(CUDA.devices())
    mgpu = TN.MultiGPUBeliefPropagationCache(TN.BeliefPropagationCache(ψ); n_devices, complex_type = ComplexF64)
    mgpu = TN.update(mgpu; maxiter, tolerance, α)

    @testset "reference is converged" begin
        # A genuinely converged BP is a fixed point
        ref_deep = TN.update(TN.BeliefPropagationCache(ψ); alg = "bp", maxiter = 2 * maxiter, tolerance)
        for v in vertices(g)
            @test isapprox(TN.expect(ref, ("Z", v); alg = "bp"),
                           TN.expect(ref_deep, ("Z", v); alg = "bp"); atol = 1.0e-8)
        end
    end

    @testset "converged messages agree" begin
        for e in edges(g)
            for de in (NamedEdge(src(e) => dst(e)), NamedEdge(dst(e) => src(e)))
                d = TN.message_diff(TN.message(ref, de), to_host(TN.message(mgpu, de)))
                @test d < 1.0e-10
            end
        end
    end

    @testset "single-site observables agree" begin
        host = TN.collect_to_cpu!(mgpu)
        for v in vertices(g)
            ev_ref = TN.expect(ref, ("Z", v); alg = "bp")
            ev_mgpu = TN.expect(host, ("Z", v); alg = "bp")
            @test isapprox(ev_ref, ev_mgpu; atol = 1.0e-6)
        end
    end

    @testset "two-site observables agree" begin
        host = TN.collect_to_cpu!(mgpu)
        for e in collect(edges(g))[1:min(end, 8)]
            obs = ("ZZ", [src(e), dst(e)])
            ev_ref = TN.expect(ref, obs; alg = "bp")
            ev_mgpu = TN.expect(host, obs; alg = "bp")
            @test isapprox(ev_ref, ev_mgpu; atol = 1.0e-6)
        end
    end

    @testset "circuit equivalence via apply_gates" begin
        Random.seed!(1234)
        ψ0 = TN.random_tensornetworkstate(ComplexF32, g, s; bond_dimension = 1)

        dt = 0.2
        layer = []
        append!(layer, ("Rx", [v], 2 * 1.0 * dt) for v in vertices(g))
        for colored_edges in TN.colored_edge_groups(g)
            append!(layer, ("Rzz", pair, 2 * 0.5 * dt) for pair in colored_edges)
        end
        apply_kwargs = (; maxdim = 4, cutoff = 1.0e-12, normalize_tensors = false)

        ref_c = TN.BeliefPropagationCache(copy(ψ0))
        ref_c, _ = TN.apply_gates(layer, ref_c; apply_kwargs)
        ref_c = TN.update(ref_c; alg = "bp", maxiter, tolerance)

        mgpu_c = TN.MultiGPUBeliefPropagationCache(TN.BeliefPropagationCache(copy(ψ0)); n_devices)
        mgpu_c, _ = TN.apply_gates(layer, mgpu_c; apply_kwargs)
        mgpu_c = TN.update(mgpu_c; maxiter, tolerance, α)
        host_c = TN.collect_to_cpu!(mgpu_c)

        for v in vertices(g)
            ev_ref = TN.expect(ref_c, ("Z", v); alg = "bp")
            ev_mgpu = TN.expect(host_c, ("Z", v); alg = "bp")
            @test isapprox(ev_ref, ev_mgpu; atol = 1.0e-4)
        end
    end
end
