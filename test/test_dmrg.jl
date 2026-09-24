@eval module $(gensym())
using Test: @test, @testset
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
const TI = TNQS.TensorInterface
using LinearAlgebra: norm, eigvals, Hermitian, I, kron
using Random: Random

# Heisenberg exchange plus a transverse field on every edge/vertex of `g`
function heisenberg_terms(g)
    H = Any[]
    for e in edges(g), P in ("XX", "YY", "ZZ")
        push!(H, (P, (src(e), dst(e)), 0.25))
    end
    for v in vertices(g)
        push!(H, ("X", [v], 0.3))
    end
    return H
end
_obs(t) = (t[1], t[2] isa Tuple ? collect(t[2]) : t[2], t[3])
energy(ψ, H; alg) = sum(real(only(expect(ψ, _obs(t); alg))) for t in H)
function energy(bpc::BeliefPropagationCache, H)
    return sum(real(only(expect(bpc, _obs(t)))) for t in H)
end
function ed_ground_energy(g, H)
    vs = collect(vertices(g)); idx = Dict(v => i for (i, v) in enumerate(vs)); n = length(vs)
    P = Dict('X' => ComplexF64[0 1; 1 0], 'Y' => ComplexF64[0 -im; im 0], 'Z' => ComplexF64[1 0; 0 -1])
    I2 = Matrix{ComplexF64}(I, 2, 2)
    Hm = zeros(ComplexF64, 2^n, 2^n)
    for t in H
        verts = t[2] isa Tuple ? collect(t[2]) : t[2]
        ops = Dict(idx[verts[k]] => P[t[1][k]] for k in eachindex(verts))
        Hm += t[3] * reduce(kron, [get(ops, i, I2) for i in 1:n])
    end
    return minimum(real(eigvals(Hermitian(Hm))))
end

@testset "generating-function BP-DMRG" begin
    Random.seed!(0x5eed_d3a6)

    @testset "generating operator and Bethe energy" begin
        for (g, exact_on) in ((named_comb_tree((2, 3)), true), (named_grid((3, 3)), false))
            H = heisenberg_terms(g)
            ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 3)
            gen = generating_operator(H, ψ)
            @test gen isa GeneratingOperator
            # every edge carries an auxiliary index of dimension r + 1 = 4 (XX + YY + ZZ has rank 3)
            for e in edges(g)
                @test TI.dim(only(virtualinds(gen.value, e))) == 4
                @test virtualinds(gen.value, e) == virtualinds(gen.derivative, e)
            end
            bpc = generating_cache(ψ, gen; maxiter = 60, tolerance = 1.0e-13)
            # the norm sector of the generating network is the norm network
            @test partitionfunction(bpc) ≈ norm_sqr(ψ; alg = "bp") rtol = 1.0e-9
            # envelope theorem: the energy at fixed messages is the BP expectation value of H
            E_B = bethe_energy(bpc, gen)
            @test abs(imag(E_B)) < 1.0e-7      # zero up to BP convergence of the fixed point
            nbpc = update(BeliefPropagationCache(ψ); maxiter = 60, tolerance = 1.0e-13)
            @test real(E_B) ≈ energy(nbpc, H) rtol = 1.0e-6   # two BP fixed points, each converged to its tolerance
            exact_on && @test real(E_B) ≈ energy(ψ, H; alg = "exact") atol = 1.0e-10
            @test real(bethe_energy(ψ, H; maxiter = 60, tolerance = 1.0e-13)) ≈ real(E_B) rtol = 1.0e-6
        end
    end

    @testset "effective operators: Hermitian, and the gradient of the BP energy" begin
        for g in (named_comb_tree((2, 3)), named_grid((3, 3)))
            H = heisenberg_terms(g)
            ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 2)
            gen = generating_operator(H, ψ)
            bpc = generating_cache(ψ, gen; maxiter = 80, tolerance = 1.0e-14)
            dX = message_response(bpc, gen; tol = 1.0e-13)
            v = first(vertices(g))
            N, Hf = effective_operators(bpc, gen, dX, v)
            ψv = ψ[v]
            φ = TI.random_tensor(ComplexF64, collect(TI.inds(ψv))...)
            @test abs(TI.scalar(TI.dag(φ) * Hf(ψv)) - conj(TI.scalar(TI.dag(ψv) * Hf(φ)))) < 1.0e-7
            @test abs(TI.scalar(TI.dag(φ) * N(ψv)) - conj(TI.scalar(TI.dag(ψv) * N(φ)))) < 1.0e-7
            # dE_B/dψᵥ* = (H_eff − Eᵥ N_eff)ψᵥ / Zᵥ against a central finite difference of the
            # re-converged BP energy: this checks the environment-response term
            Zv = real(TI.scalar(TI.dag(ψv) * N(ψv)))
            Ev = real(TI.scalar(TI.dag(ψv) * Hf(ψv))) / Zv
            grad = Hf(ψv) - Ev * N(ψv)
            δ = TI.random_tensor(ComplexF64, collect(TI.inds(ψv))...); δ = δ / norm(δ)
            h = 1.0e-5
            ψp = copy(ψ); ψp[v] = ψv + h * δ
            ψm = copy(ψ); ψm[v] = ψv - h * δ
            ebp(ϕ) = energy(update(BeliefPropagationCache(ϕ); maxiter = 80, tolerance = 1.0e-14), H)
            fd = (ebp(ψp) - ebp(ψm)) / (2h)
            @test 2 * real(TI.scalar(TI.dag(δ) * grad)) / Zv ≈ fd rtol = 1.0e-5
        end
    end

    @testset "DMRG: exact on a tree, monotone on a loopy graph" begin
        g = named_comb_tree((2, 3)); H = heisenberg_terms(g)
        ψ0 = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 8)   # lossless at 6 sites
        ψ, Es = dmrg(ψ0, H; nsweeps = 2, verbose = false)
        @test last(Es) ≈ ed_ground_energy(g, H) atol = 1.0e-9
        @test energy(ψ, H; alg = "exact") ≈ last(Es) atol = 1.0e-9

        g = named_grid((3, 3)); H = heisenberg_terms(g)
        ψ0 = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 2)
        ψ, Es = dmrg(ψ0, H; nsweeps = 2, verbose = false, bp_kwargs = (; maxiter = 50, tolerance = 1.0e-11), response_kwargs = (; tol = 1.0e-9))
        @test last(Es) < first(Es)
        @test last(Es) > ed_ground_energy(g, H) - 0.5      # the Bethe energy of a D = 2 state is not below ED by much
        @test energy(ψ, H; alg = "exact") > ed_ground_energy(g, H) - 1.0e-8   # variational in the exact energy
    end

    @testset "CTM (MP-BP) environments: ring energy, finite-difference gradient, one-site sweep" begin
        # 3×3 TFIM, real D = 2 state. χ = 16 is lossless here (two D²·2 = 8-wide interfaces per
        # corner), so the ring energy of the generating network must be the exact energy and the
        # finite-difference H_eff must give the exact energy gradient.
        g = named_grid((3, 3))
        H = vcat(Any[("ZZ", (src(e), dst(e)), -1.0) for e in edges(g)], Any[("X", [v], -2.0) for v in vertices(g)])
        ψ = random_tensornetworkstate(Float64, g, "S=1/2"; bond_dimension = 2)
        ψ = gauge_and_scale(ψ)
        gen = generating_operator(H, ψ)
        @test TI.dim(only(virtualinds(gen.value, first(edges(g))))) == 2     # ZZ has operator-Schmidt rank 1
        Eex = energy(ψ, H; alg = "exact")
        for projector in (:cut, :cycle)
            cache = generating_cache(ψ, gen, 16; projector)
            @test cvm_freenergy(cache) ≈ log(norm_sqr(ψ; alg = "exact")) atol = 1.0e-9
            @test bethe_energy(cache, gen) ≈ Eex atol = 1.0e-8
            # gradient of the energy along a random direction at the centre vertex
            λ = projector === :cycle ? 1.0e-6 : 1.0e-7     # the measured windows, see dmrg(::Algorithm"ctmrg")
            cp = generating_cache(ψ, gen, 16; λ, seed = cache, projector)
            cm = generating_cache(ψ, gen, 16; λ = -λ, seed = cache, projector)
            v = (2, 2)
            N, Hf = effective_operators(cache, cp, cm, gen, v, λ)
            is = collect(TI.inds(ψ[v]))
            x = vec(TI.array(ψ[v], is...))
            δ = TI.random_tensor(Float64, is...); δ = δ / norm(δ); d = vec(TI.array(δ, is...))
            Z = x' * N * x; Ev = (x' * Hf * x) / Z
            grad = 2 * (d' * (Hf * x - Ev * N * x)) / Z
            h = 1.0e-4
            ψp = copy(ψ); ψp[v] = ψ[v] + h * δ
            ψm = copy(ψ); ψm[v] = ψ[v] - h * δ
            fd = (energy(ψp, H; alg = "exact") - energy(ψm, H; alg = "exact")) / (2h)
            @test grad ≈ fd rtol = 1.0e-4
        end
        # lever 3: the aux-free λ = 0 cache (norm network converged, blocks padded with
        # onehot(aux ⇒ 1)) has the same F and N_eff as the true λ = 0 cache, and the ±λ pair seeded
        # from it gives the exact energy by FD of F
        let
            c_full = generating_cache(ψ, gen, 16; projector = :cut)
            c_free = generating_cache(ψ, gen, 16; projector = :cut, aux_free = true)
            @test cvm_freenergy(c_free) ≈ cvm_freenergy(c_full) atol = 1.0e-12
            v = (2, 2); opts = TNQS.options(c_full)
            Nfull = TNQS._dense_ring_operator(TNQS.vertex_ring(c_full, v), gen.value[v], ψ[v], opts)
            Nfree = TNQS._dense_ring_operator(TNQS.vertex_ring(c_free, v), gen.value[v], ψ[v], opts)
            # proportional, not equal: every block is unit-normalised and the norm-network blocks
            # carry no half-insertion weight, so the padded ring sits on its own overall scale
            # (measured ~2×). This is why `effective_operators` takes N_eff from the ±λ rings.
            @test norm(Nfree / norm(Nfree) - Nfull / norm(Nfull)) < 1.0e-10
            λ = 1.0e-7
            cp = generating_cache(ψ, gen, 16; λ, seed = c_free, projector = :cut)
            cm = generating_cache(ψ, gen, 16; λ = -λ, seed = c_free, projector = :cut)
            @test (cvm_freenergy(cp) - cvm_freenergy(cm)) / (2λ) ≈ Eex atol = 5.0e-8
            # a second aux-free cache seeded from the first (the optimiser's warm path)
            c_free2 = generating_cache(ψ, gen, 16; projector = :cut, aux_free = true, seed = c_free)
            @test cvm_freenergy(c_free2) ≈ cvm_freenergy(c_full) atol = 1.0e-12
        end
        # one sweep descends monotonically and stays variational; ring energy = exact energy
        ψ1, Es = dmrg(ψ, H; alg = "ctmrg", maxdim = 16, nsweeps = 1, projector = :cut, verbose = false)
        @test all(diff(Es) .< 1.0e-8)
        # 5e-8: the FD of F at λ = 1e-7 carries ~1e-15 / 1e-7 of roundoff (measured ±2e-8 here)
        @test last(Es) ≈ energy(ψ1, H; alg = "exact") atol = 5.0e-8
        @test last(Es) < Eex
        @test last(Es) > ed_ground_energy(g, H) - 1.0e-8
        # global L-BFGS with one environment set per step: every step descends (Armijo), the
        # recorded FD-of-F energy is the exact energy (χ = 16 lossless), and stays variational
        ψ2, E2 = dmrg(ψ, H; alg = "ctmrg_lbfgs", maxdim = 16, maxiter = 4, verbose = false)
        @test length(E2) == 5
        @test all(diff(E2) .< 0)
        # (5e-8: the FD of F at λ = 1e-7 carries ~1e-15 / 1e-7 of roundoff; measured 1.6e-8 here)
        @test last(E2) ≈ energy(ψ2, H; alg = "exact") atol = 5.0e-8
        @test last(E2) > ed_ground_energy(g, H) - 1.0e-8
        # the (μ,ν) response solve (src/response.jl): at lossless budgets the energy ∂μ∂νF and the
        # ring gradient at an OFF-CENTRE vertex are exact (the cut pair got the centre only)
        let
            R = TNQS.response_solve(ψ, H, 16; χ1 = 128, gen, maxsweeps = 5, tol = 1.0e-12)
            @test R.energy ≈ Eex atol = 1.0e-10
            v = (1, 1)
            N, Hf = TNQS.response_effective_operators(R, v)
            is = collect(TI.inds(ψ[v]))
            x = vec(TI.array(ψ[v], is...))
            δ = TI.random_tensor(Float64, is...); δ = δ / norm(δ); d = vec(TI.array(δ, is...))
            Z = x' * N * x; Ev = (x' * Hf * x) / Z
            grad = 2 * (d' * (Hf * x - Ev * N * x)) / Z
            h = 1.0e-4
            ψp = copy(ψ); ψp[v] = ψ[v] + h * δ
            ψm = copy(ψ); ψm[v] = ψ[v] - h * δ
            fd = (energy(ψp, H; alg = "exact") - energy(ψm, H; alg = "exact")) / (2h)
            @test grad ≈ fd rtol = 1.0e-5
            # and it drives the optimiser: same descent, energy exact (no finite-difference noise)
            ψ3, E3 = dmrg(ψ, H; alg = "ctmrg_lbfgs", maxdim = 16, maxiter = 2, verbose = false, response = true, χ1 = 32)
            @test length(E3) == 3 && all(diff(E3) .< 0)
            @test last(E3) ≈ energy(ψ3, H; alg = "exact") atol = 1.0e-9
        end
    end

    @testset "Z2-symmetric (graded) generating operator and sweep" begin
        # Rotated TFIM H = −Σ XX − g Σ Z conserves ∏Z. A joint two-site XX is needed on graded sites
        # (a per-character "XX" would build charge-odd single-site X's). Graded and dense
        # representations of the same imaginary-time state must agree on everything.
        register_op!("XXjoint", (; kwargs...) -> kron([0.0 1; 1 0], [0.0 1; 1 0]); nsites = 2)
        g = named_grid((3, 3)); gx = 3.0
        H = vcat(Any[("XXjoint", (src(e), dst(e)), -1.0) for e in edges(g)], Any[("Z", [v], -gx) for v in vertices(g)])
        function build(sym)
            s = sym === nothing ? siteinds("S=1/2", g) : siteinds("S=1/2", g; sectors = [0 => 1, 1 => 1], symmetry = "Z2")
            bpc = BeliefPropagationCache(tensornetworkstate(ComplexF64, v -> "↑", g, s))
            layer = Any[("Rz", [v], im * gx * 0.1) for v in vertices(g)]
            for ce in edge_color(g, 4); append!(layer, ("Rxx", pair, 2im * 0.1) for pair in ce); end
            for _ in 1:15
                bpc, _ = apply_gates(layer, bpc; apply_kwargs = (maxdim = 2, cutoff = 1.0e-12, normalize_tensors = true), verbose = false)
            end
            return gauge_and_scale(TNQS.network(bpc))
        end
        ψd, ψg = build(nothing), build("Z2")
        @test TNQS.Tensors.isgraded(ψg[(1, 1)])
        Ed = energy(ψd, H; alg = "exact"); Eg = energy(ψg, H; alg = "exact")
        @test Ed ≈ Eg atol = 1.0e-9
        gen_g = generating_operator(H, ψg)
        α = only(virtualinds(gen_g.value, first(edges(g))))
        @test TNQS.Tensors.isgraded(α) && TI.dim(α) == 2          # trivial ⊕ odd
        for projector in (:cut, :cycle)
            cache = generating_cache(ψg, gen_g, 16; projector)
            @test cvm_freenergy(cache) ≈ log(real(norm_sqr(ψg; alg = "exact"))) atol = 1.0e-9
            @test bethe_energy(cache, gen_g) ≈ Eg atol = 1.0e-9
        end
        ψd1, Esd = dmrg(ψd, H; alg = "ctmrg", maxdim = 16, nsweeps = 1, verbose = false)
        ψg1, Esg = dmrg(ψg, H; alg = "ctmrg", maxdim = 16, nsweeps = 1, verbose = false)
        @test last(Esd) ≈ last(Esg) atol = 1.0e-7                # same sweep in both representations
        @test energy(ψg1, H; alg = "exact") ≈ last(Esg) atol = 5.0e-8   # FD-of-F roundoff, see the dense sweep
        @test TNQS.Tensors.isgraded(ψg1[(2, 2)])
    end

    @testset "fermionic (spinful fZ2) BP DMRG: exact on a tree" begin
        # Spinful Hubbard on a 3-site path. A tree, so the response is exact after one sweep and the
        # one-site update must reach the exact ground energy at lossless D. Guards the three graded
        # defects fixed 2026-09-17 (docs/dmrg.md): the aux-slice arrow in `_norm_roots`, the flattened
        # local solve, and the layout-dependent message-normalisation functional — with the middle
        # vertex's outgoing responses (the propagated term) exercised.
        t, U, μ = 1.0, 4.0, 2.0
        g = named_grid((3, 1)); vs = collect(vertices(g))
        s = siteinds("Electron", g; symmetry = "fZ2")
        H = Any[]
        for e in edges(g); push!(H, ("hopping", (src(e), dst(e)), -t)); end
        for v in vs; push!(H, ("NupNdn", [v], U)); push!(H, ("N", [v], -μ)); end
        # exact reference from the backend's own local Fock matrices (mode basis |0⟩,|↑⟩,|↓⟩,|↑↓⟩,
        # site 1 slowest; the two-site hopping matrix carries the intra-site parity string), in the
        # N = 3 sector the half-filled start lives in
        FB = TNQS.Tensors
        hop = FB._f4_hop(FB._F4_AUP) + FB._f4_hop(FB._F4_ADN)
        nn = FB._F4_NUP * FB._F4_NDN; ntot = FB._F4_NUP + FB._F4_NDN; I4 = Matrix{ComplexF64}(I, 4, 4)
        Hm = -t * (kron(hop, I4) + kron(I4, hop)) + sum(U * kron([k == j ? nn : I4 for k in 1:3]...) - μ * kron([k == j ? ntot : I4 for k in 1:3]...) for j in 1:3)
        nsite = [0, 1, 1, 2]
        keep = [i for i in 1:64 if sum(nsite[d + 1] for d in digits(i - 1; base = 4, pad = 3)) == 3]
        E_ed = minimum(real(eigvals(Hermitian(Hm[keep, keep]))))
        # imaginary-time simple-update start from the alternating ↑↓ product state
        ψ = tensornetworkstate(ComplexF64, v -> isodd(findfirst(==(v), vs)) ? "Up" : "Dn", g, s)
        dτ = 0.05
        layer = Any[("F_hop", (src(e), dst(e)), im * t * dτ) for e in edges(g)]
        append!(layer, ("F_int", [v], -im * U * dτ) for v in vs)
        append!(layer, ("F_phase", [v], im * μ * dτ) for v in vs)
        bpc = BeliefPropagationCache(ψ)
        for _ in 1:10
            bpc, _ = apply_gates(layer, bpc; apply_kwargs = (; maxdim = 4, cutoff = 1.0e-14))
        end
        ψ = TNQS.network(bpc)
        gen = generating_operator(H, ψ)
        gbpc = generating_cache(ψ, gen; maxiter = 100)
        @test real(bethe_energy(gbpc, gen)) ≈ energy(ψ, H; alg = "exact") atol = 1.0e-9   # tree: Bethe = exact
        hist = Float64[]
        TNQS.message_response(gbpc, gen; maxiter = 5, history = hist)
        @test hist[2] < 1.0e-12                                                           # exact after one sweep
        ψ1, Es = dmrg(ψ, H; alg = "bp", nsweeps = 2, verbose = false)
        @test all(diff(Es) .< 1.0e-9)                                                       # every update accepted, monotone
        @test last(Es) ≈ E_ed atol = 1.0e-8
        @test energy(ψ1, H; alg = "exact") ≈ E_ed atol = 1.0e-8
    end
end
end
