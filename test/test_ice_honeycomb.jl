@eval module $(gensym())
# The honeycomb ice formulation (examples/ice/honeycomb.jl, docs/ice.md): BP simple update in the Vidal
# gauge against the library's BP gauge, the honeycomb networks against the library's cell-PEPS
# evaluation (`ice_site`), the paired networks against the layered ones, the exact arrow-reversal
# overlap, CTM resumption, and the U(1) variant's charge conservation. ~2 minutes after compilation.
using Test
using TensorNetworkQuantumSimulator
using LinearAlgebra
using Logging: NullLogger, with_logger
const TNQS = TensorNetworkQuantumSimulator
include(joinpath(pkgdir(TNQS), "examples", "ice", "honeycomb.jl"))
include(joinpath(pkgdir(TNQS), "examples", "ice", "honeycomb_u1.jl"))

# the enlarged state (M applied, bonds 2D) on a periodic 3×3 honeycomb torus, for the library's BP
function torus_state(Xe, Ye; L = 3)
    V = vec([(i, j, s) for i in 1:L, j in 1:L, s in 1:2])
    g = TNQS.NamedGraph(V)
    a(i, j) = (mod1(i, L), mod1(j, L), 1); b(i, j) = (mod1(i, L), mod1(j, L), 2)
    ed = Dict{Any, Any}(); ty = Dict{Any, Int}()
    for i in 1:L, j in 1:L, (k, bb) in ((1, b(i, j)), (2, b(i - 1, j)), (3, b(i, j - 1)))
        TNQS.add_edge!(g, a(i, j) => bb)
        ed[(a(i, j), k)] = ed[(bb, k)] = TNQS.new_index(size(Xe, k); tags = "e$k")
        ty[TNQS.NamedEdge(a(i, j) => bb)] = k
    end
    ts = TNQS.Dictionary{Tuple{Int, Int, Int}, Any}()
    for v in V
        TNQS.set!(ts, v, v[3] == 1 ? TNQS.from_array(Xe, (ed[(v, k)] for k in 1:3)...) :
                         TNQS.from_array(Ye, (ed[(v, k)] for k in 1:3)..., TNQS.new_index(2; tags = "q")))
    end
    return TNQS.TensorNetworkState(ts, g), ty
end

@testset "ice on the honeycomb" begin
    X2, Y2, w2, info2 = hexstate(2)
    X3, Y3, w3, info3 = hexstate(3)
    @test info2.δ < 1.0e-12 && info3.δ < 1.0e-12
    # the fixed point is symmetric under the three bond types (C3 of the bilayer) up to the sequential
    # truncation, which breaks it at the discarded weight's level (1e-7 at D = 2, 3)
    @test maximum(norm(w3[k] - w3[1]) for k in 2:3) < 1.0e-6

    @testset "Vidal gauge = the library's BP gauge" begin
        Xp, Yp = apply_bilayer(X3, Y3)
        wp = [vcat(w3[k], w3[k]) for k in 1:3]
        _, _, wv, _ = regauge(Xp, Yp, wp, typemax(Int))
        Xh, Yh = Xp, Yp
        for k in 1:3
            Xh = scaleleg(Xh, sqrt.(wp[k]), k); Yh = scaleleg(Yh, sqrt.(wp[k]), k)
        end
        tns, ty = torus_state(Xh, Yh)
        bpc = TNQS.symmetric_gauge(TNQS.update(TNQS.BeliefPropagationCache(tns); maxiter = 2000, tolerance = 1.0e-15))
        for k in 1:3
            e = first(e for (e, kk) in ty if kk == k)
            m = TNQS.message(bpc, e)
            s = sort(abs.(diag(Array(TNQS.array(m, TNQS.inds(m)...)))); rev = true)
            s /= norm(s)
            mine = sort(wv[k]; rev = true)
            n = min(length(s), length(mine))
            @test maximum(abs.(s[1:n] - mine[1:n])) < 1.0e-7
        end
    end

    @testset "networks" begin
        χ = 9
        n, sn, _ = lnk(X3, Y3, w3, χ; kind = :norm)
        r, sr, _ = lnk(X3, Y3, w3, χ; kind = :rev)
        i, si, _ = lnk(X3, Y3, w3, χ; kind = :inv)
        s, ss, _ = lnk(X3, Y3, w3, χ; kind = :sand)
        @test sn.converged && sr.converged && si.converged && ss.converged
        @test abs(r - n) < 1.0e-11                                  # ⟨Rψ|ψ⟩ = ⟨ψ|ψ⟩ exactly
        @test -1.0e-5 < i - n < 0                                   # ln F_I = −7.06e-6 at D = 3
        @test abs((s - n) - libf(X3, Y3, w3, χ)) < 1.0e-10          # the library's cell-PEPS evaluation
        @test 1.5072 < exp((s - n) / 2) < 1.50747                   # w_h(D = 3) = 1.5072772, below the exact
        # the paired 2-layer networks are the same networks
        for (kind, ref) in ((:norm, n), (:inv, i), (:sand, s))
            @test abs(lnkp(X3, Y3, w3, χ; kind)[1] - ref) < 1.0e-10
        end
        # resumption: a CTM continued from a checkpointed environment in chunks of one iteration
        site, legs, sv = paired(X3, Y3, w3; kind = :norm)
        ic = with_logger(NullLogger()) do
            TNQS.update(TNQS.InfiniteCTM2D(site, legs, χ; boundary = sv); tolerance = 1.0e-11, maxiter = 5)
        end
        its = 5
        while !ic.stats[].converged && its < 200
            site, legs, sv = paired(X3, Y3, w3; kind = :norm)         # a rebuilt network, as on restart
            ic = with_logger(NullLogger()) do                        # one-iteration chunks warn by design
                TNQS.update(TNQS.InfiniteCTM2D(site, legs, χ; boundary = sv, init = ic); tolerance = 1.0e-11,
                            maxiter = 1, miniter = 1)
            end
            its += 1
        end
        @test ic.stats[].converged
        @test abs(TNQS.cvm_freenergy(ic) - n) < 1.0e-10
    end

    @testset "U(1) simple update conserves the flux" begin
        X, Y, w, qs = u1start()
        for _ in 1:4
            X, Y, w, qs, _ = bilayer_u1(X, Y, w, qs, 12)
        end
        @test maximum(u1violation(X, Y, qs)) < 1.0e-12
        @test sort(qs[1]) == sort(-qs[1])                            # ±q pairs (arrow reversal)
    end
end
end
