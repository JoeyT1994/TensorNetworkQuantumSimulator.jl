using Test
using CUDA
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using NamedGraphs
using NamedGraphs.NamedGraphGenerators: named_hexagonal_lattice_graph
using Graphs: degree, vertices, edges, src, dst

@testset "MultiGPU BP" begin

    @testset "partition_graph (memory_balanced)" begin
        g = named_hexagonal_lattice_graph(3, 3)
        n = length(collect(vertices(g)))
        for n_parts in (1, 2, 4)
            pmap = TN.partition_graph(g, n_parts; alg = "memory_balanced")

            # every vertex assigned to a valid 0-indexed partition
            @test all(haskey(pmap, v) for v in vertices(g))
            @test all(0 <= pmap[v] < n_parts for v in vertices(g))
        end
    end

    @testset "partition_graph balances degree load vs min_cut" begin
        g = named_hexagonal_lattice_graph(3, 3)
        n_parts = 4

        degree_load(pmap) = begin
            dl = zeros(Int, n_parts)
            for v in vertices(g)
                dl[pmap[v] + 1] += degree(g, v)
            end
            dl
        end
        spread(dl) = maximum(dl) - minimum(dl)

        pmap_bal = TN.partition_graph(g, n_parts; alg = "memory_balanced")
        pmap_cut = TN.partition_graph(g, n_parts; alg = "min_cut")

        # valid 0-indexed assignment of every vertex (both modes)
        @test all(haskey(pmap_bal, v) for v in vertices(g))
        @test all(0 <= pmap_bal[v] < n_parts for v in vertices(g))
        @test all(haskey(pmap_cut, v) for v in vertices(g))
        @test all(0 <= pmap_cut[v] < n_parts for v in vertices(g))

        # memory_balanced balances degree load at least as well as min_cut
        @test spread(degree_load(pmap_bal)) <= spread(degree_load(pmap_cut))
        @test spread(degree_load(pmap_bal)) <= 1

        # an unknown algorithm is rejected
        @test_throws ErrorException TN.partition_graph(g, n_parts; alg = "bogus")
    end

    @testset "colored_edge_groups" begin
        g = named_hexagonal_lattice_graph(3, 3)
        groups = TN.colored_edge_groups(g)
        # proper edge coloring: no two edges in a group share a vertex
        for grp in groups
            touched = Set()
            for e in grp
                @test !(src(e) in touched) && !(dst(e) in touched)
                push!(touched, src(e)); push!(touched, dst(e))
            end
        end
        # every edge appears exactly once across all groups
        @test sum(length, groups) == length(collect(edges(g)))
    end

    # Gated on a functional multi-GPU CUDA backend
    if get(ENV, "TNQS_TEST_MULTIGPU", "false") == "true"
        if !CUDA.functional()
            @warn "CUDA.functional() == false (driver/runtime issue?); skipping GPU tests"
        elseif length(CUDA.devices()) < 2
            @warn "CUDA is functional but only $(length(CUDA.devices())) device(s) visible; need >=2"
        else
            include("test_multigpu_cuda.jl")
        end
    end
end
