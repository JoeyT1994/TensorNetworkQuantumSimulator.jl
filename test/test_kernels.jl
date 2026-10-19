@eval module $(gensym())
using Graphs: neighbors
using LinearAlgebra: norm
using NamedGraphs: NamedEdge
using Random
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: bp_factors, contraction_sequence, incoming_messages, network,
    norm_factors, norm_message_kernel, norm_scalar_kernel, normalize_message, op, vertex_scalar
using TensorNetworkQuantumSimulator.TensorInterface: fused_norm_message, fused_norm_scalar, inds, scalar
using Test: @testset, @test

# The fused double-layer BP kernels (src/kernel_hooks.jl, src/Backends/ITensorsKernels.jl) must
# agree with the generic contraction they replace, and must actually be taken on dense states.

same_legs(a, b) = issetequal(collect(inds(a)), collect(inds(b)))

function generic_message(ψ, v, incoming)
    ts = [incoming; bp_factors(ψ, v)]
    m = contract(ts; sequence = contraction_sequence(ts; alg = "optimal"))
    return normalize_message(m)
end

function generic_scalar(ψ, v, incoming, name)
    ts = [norm_factors(ψ, [v]; op_strings = _ -> name); incoming]
    return scalar(contract(ts; sequence = contraction_sequence(ts; alg = "optimal")))
end

@testset "Fused BP kernels match the generic contraction" begin
    Random.seed!(1234)
    geometries = [
        ("square 4x4", named_grid((4, 4)), 3, 1),
        ("hexagonal 3x3", named_hexagonal_lattice_graph(3, 3), 2, 1),
        ("heavy-hex 2x2", heavy_hexagonal_lattice(2, 2), 2, 1),
        ("square 3x3, two site legs", named_grid((3, 3)), 2, 2),
    ]
    for (label, g, D, nsite) in geometries, elt in (Float64, ComplexF64)
        s = siteinds("S=1/2", g; inds_per_site = nsite)
        ψ = random_tensornetworkstate(elt, g, s; bond_dimension = D)
        bpc = update(BeliefPropagationCache(ψ))
        nmsg = 0
        worst_msg = 0.0
        for v in vertices(g), vn in neighbors(g, v)
            e = NamedEdge(v => vn)
            incoming = incoming_messages(bpc, v; ignore_edges = (reverse(e),))
            raw = norm_message_kernel(ψ, v, incoming; normalize = false)
            @test !isnothing(raw)                                    # the kernel is taken
            fast = normalize_message(raw)
            # the kernel's own normalisation: unit 2-norm, the same message up to phase
            own = norm_message_kernel(ψ, v, incoming; normalize = true)
            @test norm(own) ≈ 1
            @test norm(normalize_message(own) - fast) < 1.0e-12
            ref = generic_message(ψ, v, incoming)
            @test same_legs(fast, ref)
            worst_msg = max(worst_msg, norm(fast - ref))
            nmsg += 1
        end
        @test nmsg > 0
        @test worst_msg < 1.0e-12
        worst_scalar = 0.0
        for v in vertices(g)
            incoming = incoming_messages(bpc, v)
            names = nsite == 1 ? ("I", "Z", "X") : ("I",)       # ops need a single site leg
            for name in names
                fast = norm_scalar_kernel(ψ, [v], incoming; op_strings = _ -> name)
                @test !isnothing(fast)
                ref = generic_scalar(ψ, v, incoming, name)
                worst_scalar = max(worst_scalar, abs(fast - ref) / max(abs(ref), 1.0e-300))
            end
        end
        @test worst_scalar < 1.0e-12
        # end to end: a BP observable through the kernel equals the generic region contraction
        if nsite == 1
            v = first(vertices(g))
            incoming = incoming_messages(bpc, v)
            ref = generic_scalar(ψ, v, incoming, "Z") / generic_scalar(ψ, v, incoming, "I")
            @test expect(bpc, ("Z", [v])) ≈ ref atol = 1.0e-12
        end
    end
end

@testset "Fused BP kernels decline what they do not recognise" begin
    Random.seed!(7)
    g = named_grid((3, 3))
    ψ = random_tensornetworkstate(Float64, g, siteinds("S=1/2", g); bond_dimension = 2)
    bpc = update(BeliefPropagationCache(ψ))
    v = (2, 2)
    incoming = incoming_messages(bpc, v)
    # every bond covered: no single outgoing bond, so no message
    @test isnothing(fused_norm_message(ψ[v], collect(siteinds(ψ, v)), incoming; normalize = true))
    # multi-vertex regions and ρ insertions take the generic path
    @test isnothing(norm_scalar_kernel(ψ, [(1, 1), (1, 2)], incoming_messages(bpc, [(1, 1), (1, 2)]); op_strings = _ -> "I"))
    @test isnothing(norm_scalar_kernel(ψ, [v], incoming; op_strings = _ -> "ρ"))
    # a non-tensor (boundary-MPS style) message
    @test isnothing(norm_message_kernel(ψ, v, Any[[first(incoming)]]; normalize = true))
    # and the seam's fallback is `nothing` for objects no backend claims
    @test isnothing(fused_norm_message(1.0, [], []; normalize = true))
    @test isnothing(fused_norm_scalar(1.0, [], [], []))
    # vertex_scalar goes through the kernel and agrees with the generic contraction
    @test vertex_scalar(bpc, v) ≈ generic_scalar(ψ, v, incoming, "I") rtol = 1.0e-12
end

end
