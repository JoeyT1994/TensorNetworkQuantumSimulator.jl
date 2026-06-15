@eval module $(gensym())
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws
using NamedGraphs: edges, vertices, edgeinduced_subgraphs_no_leaves
using NamedGraphs.GraphsExtensions: src, dst, degree, subgraph, is_connected
using ITensors: ITensors
using Random: Random

ITensors.disable_warn_order()

# canonical, vertex-type-agnostic signature of an edge-induced subgraph
_ekey(e) = (a = src(e); b = dst(e); a <= b ? (a, b) : (b, a))
_sig(h) = sort([_ekey(e) for e in edges(h)])
_sigset(hs) = Set(Any[_sig(h) for h in hs])
# every vertex touched by the subgraph has induced degree >= 2 (no leaves)
function _is_no_leaf(h)
    deg = Dict{Any, Int}()
    for e in edges(h)
        deg[src(e)] = get(deg, src(e), 0) + 1
        deg[dst(e)] = get(deg, dst(e), 0) + 1
    end
    return !isempty(deg) && all(>=(2), values(deg))
end

# brute force: ALL connected no-leaf edge subsets up to `maxe` edges (bitmask over edges)
function _brute_connected_noleaf(g, maxe)
    E = collect(edges(g))
    m = length(E)
    out = Set()
    for mask in 1:(2^m - 1)
        S = [i for i in 1:m if (mask >> (i - 1)) & 1 == 1]
        (length(S) < 3 || length(S) > maxe) && continue
        es = E[S]
        deg = Dict{Any, Int}()
        for e in es
            deg[src(e)] = get(deg, src(e), 0) + 1
            deg[dst(e)] = get(deg, dst(e), 0) + 1
        end
        all(>=(2), values(deg)) || continue
        is_connected(TN.edge_subgraph(g, es)) || continue
        push!(out, sort([_ekey(e) for e in es]))
    end
    return out
end

@testset "Free-energy loop cluster expansion" begin

    # =======================================================================
    # 1. Building the generalized loops: connected_edgeinduced_subgraphs_no_leaves
    #    is connected-by-construction, leaf-free, exhaustive (== brute force),
    #    and anchorable.
    # =======================================================================
    @testset "connected_edgeinduced_subgraphs_no_leaves" begin
        gg = named_grid((3, 3))
        for C in (4, 6, 8)
            egs = connected_edgeinduced_subgraphs_no_leaves(gg, C)
            @test _sigset(egs) == _brute_connected_noleaf(gg, C)   # exhaustive & exact
            for h in egs
                @test is_connected(h)
                @test _is_no_leaf(h)
                @test length(collect(edges(h))) <= C
            end
        end
        # below girth (smallest grid loop is a 4-cycle) nothing fits
        @test isempty(connected_edgeinduced_subgraphs_no_leaves(gg, 3))

        # anchoring restricts to subgraphs containing the anchor, nothing more/less
        v = first(vertices(gg))
        anchored = _sigset(connected_edgeinduced_subgraphs_no_leaves(gg, 8; anchor = v))
        containing = _sigset(filter(h -> v in vertices(h),
                                    connected_edgeinduced_subgraphs_no_leaves(gg, 8)))
        @test anchored == containing
    end

    # =======================================================================
    # 2. Comparison to the existing loop-correction enumerator.
    #    (a) bridge-completeness: the new grower catches no-leaf subgraphs joined
    #        by a bridge edge that the cycle-union enumerator drops;
    #    (b) where no bridge diagrams fit (a small grid) the two enumerators agree
    #        exactly, and the new enumerator + the existing `weight` machinery
    #        reproduce both `loopcorrected_partitionfunction` and the exact Z.
    # =======================================================================
    @testset "vs edgeinduced_subgraphs_no_leaves: bridge completeness" begin
        # two triangles joined by a single bridge edge. The bridge lies on no cycle, so
        # the cycle-union enumerator drops the whole dumbbell; the connected grower keeps
        # it. The old result must be a strict subset of the new.
        g = NamedGraph(collect(1:6))
        for p in (1 => 2, 2 => 3, 1 => 3, 4 => 5, 5 => 6, 4 => 6, 3 => 4)
            g = TN.add_edge(g, p)
        end
        dumbbell = sort([(min(a, b), max(a, b))
                         for (a, b) in ((1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6), (3, 4))])
        new_sigs = _sigset(connected_edgeinduced_subgraphs_no_leaves(g, 7))
        old_sigs = _sigset(edgeinduced_subgraphs_no_leaves(g, 7))
        @test dumbbell in new_sigs
        @test !(dumbbell in old_sigs)
        @test issubset(old_sigs, new_sigs)
        # exactly the two triangles (old) plus the full dumbbell (new)
        @test length(new_sigs) == 3
        @test length(old_sigs) == 2
    end

    @testset "vs loopcorrected_partitionfunction: agreement & exact Z" begin
        Random.seed!(42)
        g = named_grid((3, 3))   # dense enough that no bridge diagram fits => enumerators agree
        ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 2)
        bpc = update(TN.BeliefPropagationCache(ψ); TN.default_bp_update_kwargs(ψ)...)
        zbp = TN.partitionfunction(bpc)
        rbpc = TN.rescale(bpc)

        C = 12   # covers every no-leaf subgraph of the 3x3 grid (all 12 edges)
        egs_new = connected_edgeinduced_subgraphs_no_leaves(g, C)
        egs_old = edgeinduced_subgraphs_no_leaves(g, C)
        @test _sigset(egs_new) == _sigset(egs_old)             # no bridges fit => identical

        # new enumerator + existing weight machinery == existing loop-corrected Z
        Z_new = zbp * (1 + sum(TN.weights(rbpc, egs_new)))
        Z_old = norm_sqr(bpc; alg = "loopcorrections", max_configuration_size = C)
        @test Z_new ≈ Z_old
        # ... and both reconstruct the exact partition function (no disconnected loops on 3x3)
        @test Z_new ≈ norm_sqr(ψ; alg = "exact") rtol = 1e-6
    end

    # =======================================================================
    # 3. loopcorrected_free_energy: additive linked-cluster form. Below girth it
    #    reduces to ln Z_BP (and to log of the loop-corrected partition function);
    #    in general it is exactly ln Z_BP + Σ_C w_C over the connected clusters,
    #    differing from log(loopcorrected_partitionfunction) by Σw − log(1+Σw).
    # =======================================================================
    @testset "loopcorrected_free_energy structure" begin
        Random.seed!(1234)
        g = named_grid((4, 4))
        ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 3)
        bpc = update(TN.BeliefPropagationCache(ψ); TN.default_bp_update_kwargs(ψ)...)
        zbp = TN.partitionfunction(bpc)

        # below girth: no loops fit => F == ln Z_BP == log(loop-corrected Z)
        @test connected_edgeinduced_subgraphs_no_leaves(g, 3) |> isempty
        @test loopcorrected_free_energy(bpc, 3) ≈ log(complex(zbp))
        @test loopcorrected_free_energy(bpc, 3) ≈
            log(complex(norm_sqr(bpc; alg = "loopcorrections", max_configuration_size = 3)))

        # general C: F == ln Z_BP + Σ_C w_C, and F − log(pf_connected) == Σw − log(1+Σw)
        rbpc = TN.rescale(bpc)
        for C in (4, 6, 8)
            egs = connected_edgeinduced_subgraphs_no_leaves(TN.graph(rbpc), C)
            Σw = sum(TN.weights(rbpc, egs))
            F = loopcorrected_free_energy(bpc, C)
            @test F ≈ log(complex(zbp)) + Σw
            pf_connected = zbp * (1 + Σw)         # same connected clusters, product form
            @test (F - log(complex(pf_connected))) ≈ (Σw - log(1 + Σw))
        end
    end

    # =======================================================================
    # 4. The expect(...; alg = "loopcorrections") interface is the single public entry
    #    point for the free-energy estimator (single-site Hermitian observables only):
    #    it reproduces BP below girth, beats BP once loops are included, agrees across
    #    the state & cache entry points, handles a vector of observables elementwise,
    #    and rejects multi-site observables.
    # =======================================================================
    @testset "expect alg=loopcorrections interface" begin
        Random.seed!(1234)
        g = named_grid((4, 4))
        ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 3)
        bpc = update(TN.BeliefPropagationCache(ψ); TN.default_bp_update_kwargs(ψ)...)
        v = first(center(g))
        obs = ("Z", v)

        exact = real(expect(ψ, obs; alg = "exact"))
        bp = real(expect(bpc, obs; alg = "bp"))

        # below girth (no loops fit) the estimator reduces to BP
        f3 = real(expect(ψ, obs; alg = "loopcorrections", max_configuration_size = 3))
        @test isapprox(f3, bp; atol = 1e-3)

        # with loops, the estimator improves substantially on BP
        f12 = real(expect(ψ, obs; alg = "loopcorrections", max_configuration_size = 12))
        @test abs(f12 - exact) < abs(bp - exact)

        # cache entry point routes too and agrees with the state entry point
        @test expect(bpc, obs; alg = "loopcorrections", max_configuration_size = 6) ≈
            expect(ψ, obs; alg = "loopcorrections", max_configuration_size = 6)

        # a vector of observables is handled elementwise
        w = first(setdiff(collect(vertices(g)), [v]))
        @test expect(ψ, [("Z", v), ("Z", w)]; alg = "loopcorrections", max_configuration_size = 6) ≈
            [expect(ψ, ("Z", v); alg = "loopcorrections", max_configuration_size = 6),
             expect(ψ, ("Z", w); alg = "loopcorrections", max_configuration_size = 6)]

        # multi-site observables error through the expect interface
        @test_throws ErrorException expect(ψ, ("ZZ", [v, w]); alg = "loopcorrections", max_configuration_size = 6)
    end
end
end
