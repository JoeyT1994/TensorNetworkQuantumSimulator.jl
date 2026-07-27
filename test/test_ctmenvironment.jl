@eval module $(gensym())
using Dictionaries: Dictionary
using ITensors: commoninds, delta, dim, inds, scalar
using LinearAlgebra: norm
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws, @test_logs
const TNQS = TensorNetworkQuantumSimulator

@testset "CVM free energy: Ising, anisotropic and non-square" begin
    # Isotropic, anisotropic, square and non-square finite Ising partition functions vs the
    # library's exact contraction. χ large enough to be lossless.
    for (Lx, Ly, Kx, Ky) in [(5, 5, 0.3, 0.3), (5, 5, 0.3, 0.6), (4, 5, 0.3, 0.6)]
        g = named_grid((Lx, Ly))
        es = collect(edges(g))
        Js = Dictionary(es, [(src(e)[2] == dst(e)[2]) ? Kx : Ky for e in es])   # Kx horiz, Ky vert
        tn = ising_partitionfunction(g, 1.0; Js)
        lnZ = log(real(contract(tn; alg = "exact")))
        # maxiter must let the STATE converge, not just F: F ~ sd^2, so it lands first.
        @test cvm_freenergy(update(CTMEnvironmentCache(tn, 40); maxiter = 30)) ≈ lnZ atol = 1.0e-8
    end

    # χ-convergence: near-critical and too big to be lossless → error shrinks with χ.
    g = named_grid((6, 6))
    es = collect(edges(g))
    Js = Dictionary(es, [0.44 for _ in es])
    tn = ising_partitionfunction(g, 1.0; Js)
    lnZ = log(real(contract(tn; alg = "exact")))
    errs = [abs(cvm_freenergy(update(CTMEnvironmentCache(tn, χ))) - lnZ) for χ in (2, 8)]
    @test errs[2] < errs[1]
    @test errs[2] < 1.0e-6
end

@testset "CVM per-vertex environments" begin
    Random.seed!(123)
    # Non-square as well as square: the `:S`/`:E` block families are keyed by their first
    # included row/column, so an off-by-one there leaves regions unclosable at the boundary.
    for (Lx, Ly) in [(3, 3), (4, 3)]
        tn = random_tensornetwork(Float64, named_grid((Lx, Ly)); bond_dimension = 2)
        lnZ = log(abs(real(contract(tn; alg = "exact"))))
        cache = update(CTMEnvironmentCache(tn, 100); maxiter = 30)
        @test !isnothing(environments(cache))       # `update` stores them on the cache

        # 1. EVERY region type must close — interior, edge and corner vertices, boundary edge
        # strips, corner plaquettes. Blocks are renormalized as they are built, so an individual
        # `region_lnZ` carries an arbitrary scale and is NOT comparable to ln Z; what this
        # catches is the failure mode that actually occurs, an unclosable region (a projector
        # applied on one side only leaves dangling indices and `region_lnZ` throws).
        for cx in 1:0.5:Lx, cy in 1:0.5:Ly
            @test isfinite(region_lnZ(cache, cx, cy))
        end
        # 2. Mobius identity: V − E + P = 1, so the weighted SUM returns ln Z — and the
        # per-block rescaling cancels out of it exactly.
        @test cvm_freenergy(cache) ≈ lnZ atol = 1.0e-8
    end

    # Double layer, kept lazy: the sweep is generic over `bp_factors`, so ⟨ψ|ψ⟩ and the
    # equivalent QuadraticForm must both close their regions and agree.
    ψ = random_tensornetworkstate(Float64, named_grid((3, 3)); bond_dimension = 2)
    lnN = log(abs(real(norm_sqr(ψ; alg = "exact"))))
    for net in (ψ, QuadraticForm(ψ))
        cache = update(CTMEnvironmentCache(net, 200); maxiter = 30)
        @test cvm_freenergy(cache) ≈ lnN atol = 1.0e-8
        @test isfinite(region_lnZ(cache, 2, 2))                # interior vertex closes
        @test isfinite(region_lnZ(cache, 1.5, 1.5))            # corner plaquette closes
    end

    # The two-sided sweep must improve on the greedy one-sided pass and be monotone in χ.
    #
    # Compare only in the CONVERGENT regime. At χ too small for the problem both methods sit
    # at O(1) error and either can win by luck — a measured scan over 4×4 D=3 found the sweep
    # ahead in 29/32 cases, the exceptions being χ where both had already reached machine
    # precision. Racing them at a fixed small χ on an unseeded network is what made an earlier
    # version of this test flaky.
    tn = random_tensornetwork(Float64, named_grid((4, 4)); bond_dimension = 3)
    lnZ = log(abs(real(contract(tn; alg = "exact"))))
    swept = Float64[]
    for χ in (6, 8)
        fresh = CTMEnvironmentCache(tn, χ)
        cache = update(fresh; maxiter = 30, tolerance = 1.0e-11)
        F = cvm_freenergy(cache)
        push!(swept, abs(F - lnZ))
        # Stationary: sweeping the converged cache again barely moves F. Uses the sweep directly
        # rather than `update(...; maxiter = 2)` — the point is "one more sweep changes nothing",
        # and a 2-sweep `update` cannot certify convergence (it needs a state distance, which is
        # unavailable on the first sweep) so it would warn about something this test does not care
        # about.
        @test cvm_freenergy(TNQS.sweep_vertex_environments(cache, TNQS.environments(cache)),
                            cache) ≈ F atol = 1.0e-7
    end
    @test swept[2] < swept[1]                                  # monotone in χ
    @test swept[2] < 1.0e-3                                    # actually converging
    # The rho-route projector is kept as the reference path but `qr` defaults on, so nothing
    # else exercises it. One test keeps it honest: the two routes must agree, since they are the
    # same truncation reached by different arithmetic. Per-cache options, so a failure here
    # cannot leak the non-default route into any later testset.
    for χ in (6, 12)
        r = map((true, false)) do use_qr
            cache = update(CTMEnvironmentCache(tn, χ; qr = use_qr);
                           maxiter = 20, tolerance = 1.0e-11)
            @test TNQS.options(cache).qr == use_qr     # options survive `update`
            cvm_freenergy(cache)
        end
        @test r[1] ≈ r[2] atol = 1.0e-10
    end

    # Options are carried BY the cache, so two caches with different numerical strategies
    # coexist — no global state to save and restore.
    @test TNQS.options(CTMEnvironmentCache(tn, 6)).qr                  # default route
    let c = CTMEnvironmentCache(tn, 6; qr = false, degtol = 1.0e-9)
        @test !TNQS.options(c).qr
        @test TNQS.options(c).degtol == 1.0e-9
        @test TNQS.options(c).gauge                                    # untouched fields default
    end
    # A mistyped option is an error, not a silently ignored keyword.
    @test_throws MethodError CTMEnvironmentCache(tn, 6; qr_cuttoff = 1.0e-9)

    # Beats greedy where greedy is still visibly wrong. The greedy pass is asked for EXPLICITLY,
    # via its environments — `cvm_freenergy(fresh8)` would return the same number but warn, since
    # an implicit fallback is almost always a forgotten `update`.
    fresh8 = CTMEnvironmentCache(tn, 8)
    @test swept[2] < abs(cvm_freenergy(vertex_environments(fresh8), fresh8) - lnZ)
    # And the implicit fallback does warn, rather than quietly returning the greedy number.
    @test_logs (:warn, r"has not been `update`d") cvm_freenergy(CTMEnvironmentCache(tn, 4))
end

@testset "CVM with complex tensors" begin
    # REGRESSION. The interface projector used to be derived from CONJUGATED blocks (`ρ = A†A`,
    # `W = R_A R_B†`) while the sweep contracts the enlarged corners BILINEARLY (`Bw * Be`
    # conjugates nothing). For real tensors `ᵀ ≡ †` and it was exact; for complex ones the pair was
    # 11% off its own full-rank identity, so every truncation sat in the wrong subspace and no χ
    # could repair it. Symptom: 4×4 complex double layer stuck 3.7e-3 from the exact norm at χ=16
    # AND χ=64, while boundary MPS was exact — the saturation in χ is the tell.
    #
    # Small lattices hid it (2×2 and 3×3 were exact), so test at 4×4 and check saturation.
    Random.seed!(31)
    g = named_grid((4, 4)); si = siteinds("S=1/2", g)
    for elt in (Float64, ComplexF64)
        ψ = random_tensornetworkstate(elt, g, si; bond_dimension = 2)
        lnN = log(abs(real(norm_sqr(ψ; alg = "exact"))))
        for χ in (16, 64)                       # both lossless: the value must not drift with χ
            cache = update(CTMEnvironmentCache(ψ, χ); maxiter = 40, tolerance = 1.0e-11)
            @test cvm_freenergy(cache) ≈ lnN atol = 1.0e-10
        end
    end

    # The projector pair must be EXACT at full rank — the invariant the bug violated. Checked
    # directly, since an end-to-end free energy can mask it (small lattices did).
    ψc = random_tensornetworkstate(ComplexF64, g, si; bond_dimension = 2)
    let cache = CTMEnvironmentCache(ψc, 64), opts = TNQS.options(cache)
        S = TNQS.vertex_environments(cache)
        tbl = TNQS._ctm_factor_table(cache)
        worst, nchecked = 0.0, 0
        for x in 1:3, y in 2:4
            Bw = TNQS._ctm_enlarged(S, tbl, :NW, x + 1, y, opts)
            Be = TNQS._ctm_enlarged(S, tbl, :NE, x + 1, y, opts)
            (isnothing(Bw) || isnothing(Be)) && continue
            ins = collect(commoninds(Bw, Be)); isempty(ins) && continue
            pr = TNQS._ctm_interface_proj2(Bw, Be, ins, prod(dim.(ins)), opts)  # full rank
            isnothing(pr) && continue
            exact = Bw * Be
            worst = max(worst, norm(exact - (Bw * pr[1]) * (pr[2] * Be)) / norm(exact))
            nchecked += 1
        end
        @test nchecked > 0
        @test worst < 1.0e-12
    end

    # A genuinely complex single-layer Z: `F` must be log|Z|. `region_lnZ` used `abs(real(·))`,
    # which telescoped to log|Re Z| instead — a no-op for real tensors and for the double-layer
    # norm (both real positive), wrong for anything with a phase.
    Random.seed!(5)
    tnc = random_tensornetwork(ComplexF64, named_grid((4, 4)); bond_dimension = 3)
    Z = contract(tnc; alg = "exact")
    @test abs(imag(Z)) > 0.1 * abs(real(Z))         # the test is vacuous without a real phase
    Fc = cvm_freenergy(update(CTMEnvironmentCache(tnc, 64); maxiter = 40, tolerance = 1.0e-11))
    @test Fc ≈ log(abs(Z)) atol = 1.0e-10
    @test !isapprox(Fc, log(abs(real(Z))); atol = 1.0e-3)      # and not the old quantity

    # The ρ route is sesquilinear by construction and cannot be made bilinear without replacing
    # its Hermitian-PSD machinery, so it REFUSES complex input rather than returning a plausible
    # wrong number. Real input still reaches it (asserted in the two-route test above).
    @test_throws ErrorException cvm_freenergy(
        update(CTMEnvironmentCache(ψc, 8; qr = false); maxiter = 2))
end

@testset "CVM convergence cannot be certified from one sweep" begin
    # REGRESSION, and it hid behind the same Möbius cancellation as the projector bug above.
    #
    # `update` used to accept convergence on the FIRST sweep, where `_ctm_statedist` returns
    # `nothing` (the interface bases are still bootstrapping) so the criterion degenerates to
    # `|ΔF|` alone. `F` is a signed Möbius sum whose cancellation is worth ~4000×, so at some χ it
    # already sits at its final value while the state is still the one-sided GREEDY seed — which is
    # 3–4 orders worse and non-monotone in χ.
    #
    # Measured, complex hex 4×4 D=2 at χ=64: sweep 1 reported `|ΔF| = 2.2e-16`, `update` returned
    # after ONE sweep, and the norm was still exact to 1.3e-15 (all cancellation) while `⟨Z⟩` was
    # 7.0e-4 wrong and `marginal_inconsistency` 2.9e-6 against 8.7e-10 at χ=32 and χ=128. Nothing
    # was special about χ=64 — `Δ` just got unlucky, which is exactly the point: a single `Δ`
    # carries no information about the state. The observable is the sensitive probe because a
    # single-region ratio gets no cancellation; the norm cannot see this class of bug at all.
    Random.seed!(1234)
    g = named_hexagonal_lattice_graph(4, 4)
    s = siteinds("S=1/2", g)
    ψ = gauge_and_scale(random_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2))
    obs = ("Z", (2, 2))
    O_exact = expect(ψ, obs; alg = "exact")

    # One sweep can never certify, even at a lossless χ where `|ΔF|` is at the roundoff floor.
    @test_logs (:warn, r"did not converge") update(CTMEnvironmentCache(ψ, 64); maxiter = 1)

    # End to end: the observable must be at machine precision at every lossless χ, with no
    # anomalous value. χ=64 is the one that used to fail; 32 brackets it as a control.
    for χ in (32, 64)
        cache = update(CTMEnvironmentCache(ψ, χ); maxiter = 100, tolerance = 1.0e-14)
        @test abs(O_exact - expect(cache, obs)) < 1.0e-12
    end
end

@testset "CVM on sparse (x,y) grids: hexagonal and heavy-hexagonal" begin
    Random.seed!(2024)
    # Hex and heavy-hex are laid out on an (x,y) grid with vertices AND edges missing. The 4C+4T
    # tiling survives holes because the quadrant/strip definitions partition by COMPARISON, not
    # occupancy, and the Möbius identity is a telescoping one on the BOUNDING BOX:
    #   Lx·Ly − (Lx−1)Ly − Lx(Ly−1) + (Lx−1)(Ly−1) = 1
    # independent of which slots are filled. Unit squares need NOT be faces of the graph — the
    # plaquette region is the four-quadrant overlap of a cut, not a lattice face.
    for (lbl, g) in (("hex 2x2", named_hexagonal_lattice_graph(2, 2)),
                     ("hex 3x3", named_hexagonal_lattice_graph(3, 3)),
                     ("heavy-hex 2x2", heavy_hexagonal_lattice(2, 2)))
        vs = collect(vertices(g))
        Lx, Ly = maximum(first.(vs)), maximum(last.(vs))
        @test length(vs) < Lx * Ly                      # genuinely sparse
        tn = random_tensornetwork(Float64, g; bond_dimension = 2)
        lnZ = log(abs(real(contract(tn; alg = "exact"))))
        @test cvm_freenergy(update(CTMEnvironmentCache(tn, 40); maxiter = 30)) ≈ lnZ atol = 1.0e-8
    end

    # Observables and the diagnostic must work on a sparse grid too. Hex vertices have degree
    # 2-3, so their rings are 3-5 blocks rather than 8 — the `nothing` paths must absorb that.
    Random.seed!(21)
    g = named_hexagonal_lattice_graph(2, 2)
    si = siteinds("S=1/2", g)
    ψh = random_tensornetworkstate(Float64, g, si; bond_dimension = 2)
    ch = update(CTMEnvironmentCache(ψh, 40); maxiter = 30)
    @test marginal_inconsistency(ch) < 1.0e-10          # lossless χ: marginals parallel
    for v in collect(vertices(g))[1:6]
        @test 0 < length(vertex_ring(ch, v)) <= 8
        @test expect(ch, ("Z", [v])) ≈ expect(ψh, ("Z", [v]); alg = "exact") atol = 1.0e-7
    end
    @test rdm(ch, [collect(vertices(g))[3]]) ≈
          rdm(ψh, [collect(vertices(g))[3]]; alg = "exact") atol = 1.0e-8

    # And it must beat BP, i.e. the corner tier carries real information on hex even though no
    # unit square is a face there.
    tn = random_tensornetwork(Float64, named_hexagonal_lattice_graph(3, 3); bond_dimension = 3)
    lnZ = log(abs(real(contract(tn; alg = "exact"))))
    ebp = abs(log(abs(real(contract(tn; alg = "bp")))) - lnZ)
    # maxiter = 20: this case needs ~14 sweeps for the state distance to clear the default
    # tolerance. At 12 it converged in `F` but warned, which reads as a solver fault and is not.
    ecvm = abs(cvm_freenergy(update(CTMEnvironmentCache(tn, 8); maxiter = 20)) - lnZ)
    @test ecvm < ebp / 10
end

@testset "CVM single-site observables" begin
    Random.seed!(456)
    L, D = 4, 2
    g = named_grid((L, L))
    s = siteinds("S=1/2", g)
    ψ = random_tensornetworkstate(Float64, g, s; bond_dimension = D)
    # interior, edge, corner and far-corner vertices — boundary rings are where it breaks
    vs = [(1, 1), (2, 1), (2, 2), (L, L)]
    ex = Dict(v => expect(ψ, ("Z", [v]); alg = "exact") for v in vs)

    # At lossless χ the ring is the exact environment, so ⟨Z⟩ is exact everywhere.
    cache = update(CTMEnvironmentCache(ψ, 16))
    for v in vs
        @test expect(cache, ("Z", [v])) ≈ ex[v] atol = 1.0e-8       # alg defaults to "ctmrg"
        @test expect(cache, ("Z", [v]); alg = "ctmrg") ≈ ex[v] atol = 1.0e-8
    end
    # vector-of-observables form, and the state-level entry point that builds its own cache
    @test all(isapprox.(expect(cache, [("Z", [v]) for v in vs]), [ex[v] for v in vs]; atol = 1.0e-8))
    @test expect(ψ, ("Z", [(2, 2)]); alg = "ctmrg", maxdim = 16) ≈ ex[(2, 2)] atol = 1.0e-8

    # `vertex_ring` is the documented primitive: 4C+4T in the bulk, fewer at the boundary,
    # and contracting it by hand with the site factors must reproduce `expect`.
    @test length(vertex_ring(cache, (2, 2))) == 8
    @test length(vertex_ring(cache, (1, 1))) < 8            # blocks fall off the lattice
    ring = vertex_ring(cache, (2, 2))
    num = scalar(contract([TNQS.norm_factors(ψ, [(2, 2)]; op_strings = _ -> "Z"); ring]))
    den = scalar(contract([TNQS.norm_factors(ψ, [(2, 2)]; op_strings = _ -> "I"); ring]))
    @test num / den ≈ ex[(2, 2)] atol = 1.0e-8
    # The ring carries no factor from its own vertex.
    @test all(t -> isempty(commoninds(t, only(siteinds(ψ, (2, 2))))), ring)

    # rdm through the same ring: trace-normalized and equal to the exact one.
    ρ = rdm(cache, [(2, 2)])
    ρx = rdm(ψ, [(2, 2)]; alg = "exact")
    @test ρ ≈ ρx atol = 1.0e-8
    @test real(scalar(ρ * delta(inds(ρ)...))) ≈ 1.0

    # Accuracy must improve with χ, and beat BP at lossless χ.
    errs = [abs(expect(update(CTMEnvironmentCache(ψ, χ)), ("Z", [(2, 2)])) - ex[(2, 2)])
            for χ in (2, 16)]
    @test errs[2] < errs[1]
    @test errs[2] < abs(expect(ψ, ("Z", [(2, 2)]); alg = "bp") - ex[(2, 2)])

    # `marginal_inconsistency`: the only lnZ-free quality measure. Must be exactly 0 where the
    # contraction is lossless (the marginals are then genuinely parallel) and must shrink with χ.
    Random.seed!(99)
    tn2 = random_tensornetwork(Float64, named_grid((4, 4)); bond_dimension = 3)
    mi = [marginal_inconsistency(update(CTMEnvironmentCache(tn2, χ); maxiter = 30, tolerance = 1.0e-11))
          for χ in (4, 8, 16)]
    @test all(>=(0.0), mi)                  # it is a distance
    @test mi[3] < 1.0e-10                   # lossless χ: marginals exactly parallel
    @test mi[1] > mi[3]                     # and it shrinks with χ

    # `vertex_window`: a bigger window keeps more of the lattice EXACT around the site, which is
    # the lever for observable accuracy at fixed χ. w=0 must reproduce the ring exactly, any w
    # must stay exact at lossless χ, and on a lattice big enough for w=1 to be genuinely partial
    # it must beat w=0.
    @test length(vertex_window(cache, (2, 2), 0)) == length(vertex_ring(cache, (2, 2)))
    @test expect(cache, ("Z", [(2, 2)]); window = 1) ≈ ex[(2, 2)] atol = 1.0e-8
    @test rdm(cache, [(2, 2)]; window = 1) ≈ rdm(ψ, [(2, 2)]; alg = "exact") atol = 1.0e-8

    Random.seed!(456)
    g6 = named_grid((6, 6)); s6 = siteinds("S=1/2", g6)
    ψ6 = random_tensornetworkstate(Float64, g6, s6; bond_dimension = 2)
    c6 = update(CTMEnvironmentCache(ψ6, 6); maxiter = 20, tolerance = 1.0e-11)
    for v in [(4, 3), (2, 2)]
        exv = expect(ψ6, ("Z", [v]); alg = "exact")
        e0 = abs(expect(c6, ("Z", [v])) - exv)
        e1 = abs(expect(c6, ("Z", [v]); window = 1) - exv)
        @test e1 < e0                                  # more exact context wins
        @test length(vertex_window(c6, v, 1)) > length(vertex_window(c6, v, 0))
    end

    # Multi-site is not supported: the ring encloses exactly one vertex.
    @test_throws ErrorException expect(cache, ("ZZ", [(1, 1), (2, 1)]))
    @test_throws ErrorException rdm(cache, [(1, 1), (2, 1)])
end
end
