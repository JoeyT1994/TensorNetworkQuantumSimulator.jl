@eval module $(gensym())
using Test: @test, @testset, @test_throws
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator.Tensors: Tensors, Index, Tensor
using Graphs: Graphs
using NamedGraphs: NamedGraph
const TI = TensorNetworkQuantumSimulator.TensorInterface
using LinearAlgebra: LinearAlgebra, norm, qr, factorize

# Cross-checks against the historical ITensors implementation run when ITensors is
# available (the Pkg.test target includes it); plain `julia --project=.` runs skip them.
const HAS_ITENSORS = !isnothing(Base.find_package("ITensors"))
HAS_ITENSORS || @info "ITensors not available: skipping cross-backend conformance checks"

# Array of a Tensor in a requested index order
function tarray(t::Tensor, is...)
    perm = map(i -> findfirst(==(i), t.inds), collect(is))
    return permutedims(t.data, perm)
end

@testset "Tensors backend" begin
    @testset "index algebra" begin
        i = Index(3, "i")
        @test TI.dim(i) == 3
        @test TI.plev(TI.prime(i)) == 1
        @test TI.prime(i) != i
        @test TI.noprime(TI.prime(i)) == i
        @test TI.dag(i) == i           # dual flip preserves identity
        @test TI.sim(i) != i
        @test i' == TI.prime(i)
        @test eltype(TI.random_itensor(i, TI.sim(i))) == Float64   # eltype defaults to Float64
    end

    @testset "contraction" begin
        ki, kj, kk = Index(3, "i"), Index(4, "j"), Index(2, "k")
        A = rand(3, 4)
        B = rand(4, 2)
        Ak = Tensor([ki, kj], copy(A))
        Bk = Tensor([kj, kk], copy(B))
        @test tarray(Ak * Bk, ki, kk) ≈ A * B
        # outer product
        Ck = Tensor([kk], rand(2))
        @test length(vec((Ak * Ck).data)) == 24
        # scalar contraction and sequence execution agree
        @test TI.scalar(Ak * Tensor([ki, kj], copy(A))) ≈ TI.scalar(TI.contract([Ak, Ak]; sequence = [1, 2]))
    end

    @testset "combiner and directsum" begin
        ki, kj, kk = Index(2, "i"), Index(3, "j"), Index(2, "k")
        T = Tensor([ki, kj, kk], rand(2, 3, 2))
        C = TI.combiner([ki, kj])
        c = TI.combinedind(C)
        Tc = T * C
        @test sort(TI.dim.(TI.inds(Tc))) == [2, 6]
        @test tarray(Tc * C, ki, kj, kk) ≈ T.data     # combine then split is the identity
        # directsum block-embeds along the paired axes
        l1, l2, ln = Index(2, "l1"), Index(3, "l2"), Index(5, "ln")
        A = Tensor([ki, l1], rand(2, 2))
        B = Tensor([ki, l2], rand(2, 3))
        D = TI.directsum([ln], A => (l1,), B => (l2,))
        @test tarray(D, ki, ln)[:, 1:2] ≈ A.data
        @test tarray(D, ki, ln)[:, 3:5] ≈ B.data
    end

    @testset "factorizations" begin
        ki, kj, kk, kl = Index(3, "i"), Index(4, "j"), Index(3, "k"), Index(2, "l")
        A = rand(ComplexF64, 3, 4, 3, 2)
        Tk = Tensor([ki, kj, kk, kl], copy(A))

        Qk, Rk = qr(Tk, [ki, kj])
        @test tarray(Qk * Rk, ki, kj, kk, kl) ≈ A
        bq = TI.commonind(Qk, Rk)
        qmat = reshape(tarray(Qk, ki, kj, bq), 12, TI.dim(bq))
        @test qmat' * qmat ≈ one(qmat' * qmat)

        for maxdim in (6, 3)
            svk = Ref{Any}(nothing)
            F1k, F2k, speck = TI.factorize_svd(Tk, [ki, kj]; ortho = "none", singular_values! = svk, maxdim)
            @test size(svk[].data, 1) <= maxdim
            maxdim == 6 && @test tarray(F1k * F2k, ki, kj, kk, kl) ≈ A
            maxdim == 6 && @test speck.truncerr < 1e-12
            maxdim == 3 && @test speck.truncerr > 0
        end

        Lk, Rk2 = factorize(Tk, ki, kj; ortho = "left", tags = "Link,l=1")
        @test tarray(Lk * Rk2, ki, kj, kk, kl) ≈ A
        lmat = reshape(tarray(Lk, ki, kj, TI.commonind(Lk, Rk2)), 12, :)
        @test lmat' * lmat ≈ one(lmat' * lmat)

        # hermitian eigen path used by simple update
        m = rand(ComplexF64, 5, 5)
        M = m + m' + 20.0 * one(m)     # comfortably PSD
        hi1, hi2 = Index(5, "h"), Index(5, "h2")
        Ms, Mis = TNQS.pseudo_sqrt_inv_sqrt(Tensor([hi1, hi2], copy(M)))
        @test tarray(Ms, hi1, hi2) * tarray(Ms, hi1, hi2) ≈ M
        @test tarray(Ms, hi1, hi2) * tarray(Mis, hi1, hi2) ≈ one(M)
    end

    if HAS_ITENSORS
        @eval using ITensors: ITensors
        @testset "ops and states match ITensors" begin
            s1i, s2i = ITensors.Index(2, "S=1/2,Site"), ITensors.Index(2, "S=1/2,Site")
            k1, k2 = Index(2, "S=1/2,Site"), Index(2, "S=1/2,Site")
            for name in ["X", "Y", "Z", "H", "I", "S+", "S-"]
                @test tarray(TI.op(name, k1), k1', k1) ≈ Array(ITensors.op(name, s1i), s1i', s1i)
            end
            for (name, kw) in [("Rx", (θ = 0.37,)), ("Ry", (θ = 0.37,)), ("Rz", (θ = 0.37,)), ("P", (ϕ = 0.37,))]
                @test tarray(TI.op(name, k1; kw...), k1', k1) ≈ Array(ITensors.op(name, s1i; kw...), s1i', s1i)
            end
            # Rxxyy / Rxxyyzz / xx_plus_yy were this package's own ITensors extensions
            # (now retired), so vanilla ITensors can't cross-check them; their matrices
            # were validated against those definitions before removal and are covered by
            # the frozen end-to-end digests below.
            for (name, kw) in [
                    ("Rzz", (ϕ = 0.37,)), ("Rxx", (ϕ = 0.37,)), ("Ryy", (ϕ = 0.37,)),
                    ("CZ", (;)), ("CNOT", (;)), ("CY", (;)), ("CPHASE", (ϕ = 0.37,)),
                    ("SWAP", (;)), ("iSWAP", (;)), ("√SWAP", (;)), ("√iSWAP", (;)),
                    ("CRx", (θ = 0.37,)), ("CRy", (θ = 0.37,)), ("CRz", (θ = 0.37,)),
                ]
                oit = isempty(kw) ? ITensors.op(name, s1i, s2i) : ITensors.op(name, s1i, s2i; kw...)
                okt = isempty(kw) ? TI.op(name, k1, k2) : TI.op(name, k1, k2; kw...)
                @test tarray(okt, k1', k2', k1, k2) ≈ Array(oit, s1i', s2i', s1i, s2i)
            end
            for st in ["↑", "↓", "+", "-"]
                @test tarray(TI.state(st, k1), k1) ≈ Array(ITensors.state(st, s1i), s1i)
            end
        end

        @testset "factorization conventions match ITensors" begin
            i, j, k, l = ITensors.Index(3, "i"), ITensors.Index(4, "j"), ITensors.Index(3, "k"), ITensors.Index(2, "l")
            ki, kj, kk, kl = Index(3, "i"), Index(4, "j"), Index(3, "k"), Index(2, "l")
            A = rand(ComplexF64, 3, 4, 3, 2)
            Ti = ITensors.ITensor(A, i, j, k, l)
            Tk = Tensor([ki, kj, kk, kl], copy(A))
            for maxdim in (6, 3)
                svi = Ref{Any}(nothing)
                F1i, F2i, speci = ITensors.factorize_svd(Ti, (i, j); ortho = "none", singular_values! = svi, maxdim)
                svk = Ref{Any}(nothing)
                F1k, F2k, speck = TI.factorize_svd(Tk, [ki, kj]; ortho = "none", singular_values! = svk, maxdim)
                @test speck.truncerr ≈ speci.truncerr atol = 1e-13
                @test tarray(F1k * F2k, ki, kj, kk, kl) ≈ Array(F1i * F2i, i, j, k, l)
                @test sort(collect(ITensors.diag(svi[])); rev = true) ≈
                    sort(real.(LinearAlgebra.diag(svk[].data)); rev = true)
            end
        end
    end

    @testset "graded (TensorKit Z2) backend" begin
        using TensorNetworkQuantumSimulator.Tensors: GradedTensor
        #Z2-symmetric workload on a CHARGED product state: the ↓ sites' charges are routed
        #through dim-1 links along a spanning tree (T-join) so every vertex tensor is
        #flux-zero; the checkerboard has an even number of ↓s, so the total vanishes.
        function graded_digest(sectors)
            g = named_grid((3, 3))
            s = siteinds("S=1/2", g; sectors, symmetry = "Z2")
            ψ = tensornetworkstate(ComplexF64, v -> iseven(sum(v)) ? "↑" : "↓", g, s)
            layer = Any[("Rz", [v], 0.4) for v in vertices(g)]
            for ces in edge_color(g, 4)
                append!(layer, ("Rxx", pair, 0.7) for pair in ces)
                append!(layer, ("xx_plus_yy", pair, (0.7, 0.3)) for pair in ces)
                append!(layer, ("Rzz", pair, 0.2) for pair in ces)
            end
            circuit = reduce(vcat, [layer for _ in 1:2])
            bp_update_kwargs = (; maxiter = 30, tolerance = 1.0e-12)
            ψ, errs = apply_gates(circuit, ψ; apply_kwargs = (; maxdim = 4, cutoff = 1.0e-14), bp_update_kwargs)
            zs = expect(ψ, [("Z", [v]) for v in vertices(g)]; alg = "bp", cache_update_kwargs = bp_update_kwargs)
            zx = expect(ψ, ("Z", [(2, 2)]); alg = "exact")
            return ψ, real(sum(zs)), real(zx), sum(errs)
        end
        ψd, zd, zxd, ed = graded_digest(nothing)
        ψg, zg, zxg, eg = graded_digest([0 => 1, 1 => 1])
        @test ψg[(1, 1)] isa GradedTensor
        @test zd ≈ zg atol = 1e-10
        @test zxd ≈ zxg atol = 1e-10
        @test ed ≈ eg atol = 1e-12
        #the conserving circuit leaves the state genuinely block-sparse: the flux-zero
        #constraint keeps exactly half the product basis for balanced parity sectors
        nstored = sum(
            sum(p -> length(ψg[v].data[p[1], p[2]]), Tensors.TK.fusiontrees(ψg[v].data); init = 0)
                for v in vertices(ψg)
        )
        nfull = sum(prod(Int[TI.dim(i) for i in TI.inds(ψg[v])]) for v in vertices(ψg))
        @test nstored <= 0.5 * nfull

        #graded combiner: fuse isometry with dense conventions (combine with C, split
        #with dag(C)); round trip is exact
        vg = (2, 2)
        cis = collect(TI.inds(ψg[vg]))[1:2]
        Cg = TI.combiner(cis)
        tc = ψg[vg] * Cg
        @test TI.dim(TI.combinedind(Cg)) == prod(TI.dim.(cis))
        trt = tc * TI.dag(Cg)
        @test trt ≈ ψg[vg] atol = 1e-13

        #non-conserving pieces must fail loudly: that is the point of the symmetry
        sg = only(TNQS.siteinds(TNQS.tensornetwork(ψg))[(1, 1)])
        @test_throws Exception TI.op("X", sg)
        @test_throws Exception TI.state("↓", sg)   #charged SINGLE tensor: flux-odd
        #an odd number of charged sites: the nonzero total rides an automatic Charge leg
        gz = named_grid((3, 3))
        sz = siteinds("S=1/2", gz; sectors = [0 => 1, 1 => 1], symmetry = "Z2")
        ψz = tensornetworkstate(ComplexF64, v -> v == (1, 1) ? "↓" : "↑", gz, sz)
        @test real(norm_sqr(ψz; alg = "exact")) ≈ 1.0

        #graded boundary MPS: random conserving init over convolved charged link spectra
        #(fermion-branch recipe; conservation itself is native — the init only picks link
        #sectors). Fixed per-sector allocation makes this variational at ~1e-4. Certified
        #sampling needs single-layer (amplitude) messages, which carry net flux — that
        #waits for charged dummy legs.
        ne = real(norm_sqr(ψg; alg = "exact"))
        nb = real(norm_sqr(ψg; alg = "boundarymps", mps_bond_dimension = 20))
        @test abs(nb / ne - 1) < 5e-3

        #sampling: projected amplitudes agree with the dense twin exactly (deterministic),
        #and certified boundary-MPS sampling runs with positive finite certificates
        function amp2(ψt, bits)
            sd = TNQS.siteinds(ψt)
            tnp = copy(TNQS.tensornetwork(ψt))
            for v in vertices(tnp)
                P = TNQS.adapt_like(tnp[v], TI.projector(ComplexF64, only(sd[v]) => bits(v)))
                TNQS.setindex_preserve!(tnp, tnp[v] * P, v)
            end
            ts = [tnp[v] for v in vertices(tnp)]
            amp = TI.scalar(TI.contract(ts; sequence = TNQS.contraction_sequence(ts; alg = "optimal")))
            return abs2(amp) / real(norm_sqr(ψt; alg = "exact"))
        end
        bits = v -> iseven(v[1]) ? 1 : 2
        @test amp2(ψg, bits) ≈ amp2(ψd, bits) atol = 1.0e-10
        gsamples = sample_directly_certified(
            ψg, 2; alg = "boundarymps",
            norm_mps_bond_dimension = 12, projected_mps_bond_dimension = 12
        )
        @test all(x -> abs(real(x.poverq) - 1) < 0.1, gsamples)

        #graded purification: the infinite-T identity state pairs ket sites with
        #DUAL-representation ancillas (flux-zero per site); U(1) Heisenberg imaginary
        #time matches the dense twin exactly when truncation has rank headroom (bound
        #truncation differs only by degenerate tie-breaking)
        function thermal_logz(symmetry)
            gt = named_grid((2, 2))
            st = symmetry === nothing ? siteinds("S=1/2", gt; inds_per_site = 2) :
                siteinds("S=1/2", gt; inds_per_site = 2, symmetry)
            ψth = identity_tensornetworkstate(ComplexF64, gt, st)
            ψ_bpc = update(BeliefPropagationCache(ψth))
            gates = []
            for ces in edge_color(gt, 4)
                append!(gates, [op("Rxxyyzz", st[src(e)][1], st[dst(e)][1], θ = -0.01im) for e in ces])
            end
            logz = -TNQS.freenergy(ψ_bpc)
            rescale!(ψ_bpc)
            for _ in 1:2
                ψ_bpc, _ = apply_gates(gates, ψ_bpc; apply_kwargs = (; maxdim = 32, cutoff = 1.0e-14))
                logz -= TNQS.freenergy(ψ_bpc)
                rescale!(ψ_bpc)
            end
            return logz
        end
        @test thermal_logz("U1") ≈ thermal_logz(nothing) atol = 1.0e-11

        #graded factorization round-trip on a generic conserving 4-leg tensor
        si = Tensors.Index(Tensors.graded_space("Z2", [0 => 1, 1 => 2]), "a")
        sj = Tensors.Index(Tensors.graded_space("Z2", [0 => 2, 1 => 1]), "b")
        T4 = TI.random_itensor(ComplexF64, si', sj', TI.dag(si), TI.dag(sj))
        svr = Ref{Any}(nothing)
        F1, F2, spec = TI.factorize_svd(T4, [si', sj']; ortho = "none", singular_values! = svr)
        @test norm(F1 * F2 - T4) < 1e-10
        @test spec.truncerr < 1e-13
        Q, R = qr(T4, [si', sj'])
        @test norm(Q * R - T4) < 1e-10
    end

    @testset "end-to-end BP digests" begin
        function digest(g)
            ψ = tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2")
            layer = Any[("Rx", [v], 0.6) for v in vertices(g)]
            for ces in edge_color(g, 4)
                append!(layer, ("Rzz", pair, 0.4) for pair in ces)
            end
            circuit = reduce(vcat, [layer for _ in 1:2])
            bp_update_kwargs = (; maxiter = 30, tolerance = 1.0e-12)
            ψ, errs = apply_gates(circuit, ψ; apply_kwargs = (; maxdim = 4, cutoff = 1.0e-14), bp_update_kwargs)
            zs = expect(ψ, [("Z", [v]) for v in vertices(g)]; alg = "bp", cache_update_kwargs = bp_update_kwargs)
            zzs = expect(ψ, [("ZZ", [src(e), dst(e)]) for e in edges(g)]; alg = "bp", cache_update_kwargs = bp_update_kwargs)
            return ψ, real(sum(zs)), real(sum(zzs))
        end

        # On a tree, BP is exact: compare against exact contraction as ground truth.
        ψt, zt, zzt = digest(named_comb_tree((3, 3)))
        z_ex = sum(real, expect(ψt, [("Z", [v]) for v in vertices(ψt)]; alg = "exact"))
        zz_ex = sum(real, expect(ψt, [("ZZ", [src(e), dst(e)]) for e in edges(ψt)]; alg = "exact"))
        @test zt ≈ z_ex atol = 1e-9
        @test zzt ≈ zz_ex atol = 1e-9

        # Frozen regression digest on a loopy 3x3 grid (values validated against the
        # historical ITensors backend to ~1e-13 before its removal).
        _, z33, zz33 = digest(named_grid((3, 3)))
        @test z33 ≈ 4.482750429812405 atol = 1e-8
        @test zz33 ≈ 3.930327161012339 atol = 1e-8
    end

    @testset "fermionic (TensorKit fZ2) backend" begin
        #Ground truth throughout: dense Jordan-Wigner evolution. The TN applies LOCAL
        #gate matrices only — all strings come from TensorKit's graded category.
        a = ComplexF64[0 1; 0 0]
        Zm = ComplexF64[1 0; 0 -1]
        id2 = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
        jw_ops(n) = [reduce(kron, [k < j ? Zm : (k == j ? a : id2) for k in 1:n]) for j in 1:n]
        function jw_evolve(layer, cs, mode, n; occupied = ())
            ψv = zeros(ComplexF64, 2^n)
            ψv[1 + sum(v -> 2^(n - mode[v]), occupied; init = 0)] = 1.0
            return foldl(layer; init = ψv) do ϕv, gate
                if gate[1] == "F_phase"
                    j = mode[only(gate[2])]
                    exp(-im * gate[3] * (cs[j]' * cs[j])) * ϕv
                else
                    j, k = mode[gate[2][1]], mode[gate[2][2]]
                    H = gate[1] == "F_hop" ? (cs[j]' * cs[k] + cs[k]' * cs[j]) :
                        (cs[j]' * cs[k]' + cs[k] * cs[j])
                    exp(-im * gate[3] * H) * ϕv
                end
            end
        end

        @testset "unit checks" begin
            s1 = Tensors.new_fermion_index(1, 1; tags = "s1")
            s2 = Tensors.new_fermion_index(1, 1; tags = "s2")
            t = TI.random_itensor(ComplexF64, s1, s2)
            @test real(TI.dag(t) * t) ≈ norm(t)^2
            F1, F2, _ = TI.factorize_svd(t, [s1])
            @test norm(TI.array(F1 * F2) - TI.array(t)) < 1.0e-13
            Q, R = qr(t, [s1])
            @test norm(TI.array(Q * R) - TI.array(t)) < 1.0e-13
            #occupied product states are flux-odd and must fail loudly
            @test_throws Exception TI.state("Occ", s1)
        end

        @testset "Kitaev-chain quench ≡ dense JW (tree ⇒ BP exact)" begin
            #charged initial state: two occupied sites (even total parity), routed
            #through dim-1 odd links along the chain (T-join)
            n = 6
            g = NamedGraph(Graphs.path_graph(n))
            s = TNQS.siteinds("Fermion", g)
            ψ = tensornetworkstate(ComplexF64, v -> v in (2, 5) ? "Occ" : "Emp", g, s)
            @test real(norm_sqr(ψ; alg = "exact")) ≈ 1.0
            #odd total parity rides an automatically attached "Charge" dummy leg
            ψodd = tensornetworkstate(ComplexF64, v -> v == 1 ? "Occ" : "Emp", g, s)
            @test real(norm_sqr(ψodd; alg = "exact")) ≈ 1.0
            layer = Any[]
            for _ in 1:3
                append!(layer, ("F_pair", (v, v + 1), 0.29) for v in 1:2:(n - 1))
                append!(layer, ("F_hop", (v, v + 1), 0.37) for v in 2:2:(n - 1))
                append!(layer, ("F_hop", (v, v + 1), 0.37) for v in 1:2:(n - 1))
                append!(layer, ("F_phase", [v], 0.21) for v in 1:n)
            end
            ψt, errs = apply_gates(layer, ψ; apply_kwargs = (; maxdim = 32, cutoff = 1.0e-14))
            @test maximum(errs) < 1.0e-12
            cs = jw_ops(n)
            ψv = jw_evolve(layer, cs, Dict(v => v for v in 1:n), n; occupied = (2, 5))
            nrm = real(ψv' * ψv)
            for v in 1:n
                occ = real(only(expect(ψt, ("N", [v]); alg = "bp")))
                @test occ ≈ real(ψv' * (cs[v]' * cs[v]) * ψv) / nrm atol = 1.0e-12
            end
            #joint two-site observables (odd⊗odd operators spanning a region)
            hop = real(only(expect(ψt, ("hopping", (3, 4)); alg = "bp")))
            @test hop ≈ real(ψv' * (cs[3]' * cs[4] + cs[4]' * cs[3]) * ψv) / nrm atol = 1.0e-12
            pr = real(only(expect(ψt, ("pairing", (3, 4)); alg = "bp")))
            @test pr ≈ real(ψv' * (cs[3]' * cs[4]' + cs[4] * cs[3]) * ψv) / nrm atol = 1.0e-12
            #odd-pair two-point functions at ANY distance: the operator has legs only on
            #(v, w) — the category threads the string; BP Steiner-completes the region
            for (name, mat, (v, w)) in (
                    ("CdagC", (j, k) -> cs[j]' * cs[k], (1, 6)),
                    ("CCdag", (j, k) -> cs[j] * cs[k]', (2, 4)),
                    ("CdagCdag", (j, k) -> cs[j]' * cs[k]', (3, 5)),
                    ("CC", (j, k) -> cs[j] * cs[k], (1, 4)),
                )
                tn = only(expect(ψt, (name, (v, w)); alg = "bp"))
                @test tn ≈ (ψv' * mat(v, w) * ψv) / nrm atol = 1.0e-12
            end
            #nonzero TOTAL charge carried by a dangling "Charge"-tagged dummy leg
            ψc = tensornetworkstate(ComplexF64, v -> v in (2, 4, 5) ? "Occ" : "Emp", g, s)
            ψct, _ = apply_gates(layer, ψc; apply_kwargs = (; maxdim = 32, cutoff = 1.0e-14))
            ψcv = jw_evolve(layer, cs, Dict(v => v for v in 1:n), n; occupied = (2, 4, 5))
            nrmc = real(ψcv' * ψcv)
            occ3 = real(only(expect(ψct, ("N", [3]); alg = "bp")))
            @test occ3 ≈ real(ψcv' * (cs[3]' * cs[3]) * ψcv) / nrmc atol = 1.0e-12
            tnc = only(expect(ψct, ("CdagC", (1, 5)); alg = "bp"))
            @test tnc ≈ (ψcv' * (cs[1]' * cs[5]) * ψcv) / nrmc atol = 1.0e-12
            #projected amplitudes (the sampling primitive): full projection onto a
            #basis configuration — projector charge legs + the root Charge leg all
            #fuse; configurations in the wrong total sector give exactly zero
            for x in ((0, 1, 0, 1, 1, 0), (1, 1, 1, 1, 1, 0))
                p_jw = abs2(ψcv[1 + sum(v -> x[v] * 2^(n - v), 1:n)]) / nrmc
                sd = TNQS.siteinds(ψct)
                tnp = copy(TNQS.tensornetwork(ψct))
                for v in vertices(tnp)
                    P = TNQS.adapt_like(tnp[v], TI.projector(ComplexF64, only(sd[v]) => x[v] + 1))
                    TNQS.setindex_preserve!(tnp, tnp[v] * P, v)
                end
                ts = [tnp[v] for v in vertices(tnp)]
                amp = TI.scalar(TI.contract(ts; sequence = TNQS.contraction_sequence(ts; alg = "optimal")))
                @test abs2(amp) / real(norm_sqr(ψct; alg = "exact")) ≈ p_jw atol = 1.0e-12
            end
        end

        @testset "fU1: number conservation is structural" begin
            n = 4
            g = NamedGraph(Graphs.path_graph(n))
            s = TNQS.siteinds("Fermion", g; symmetry = "fU1")
            #nonzero total N rides a dangling Charge leg
            ψ = tensornetworkstate(ComplexF64, v -> v == 2 ? "Occ" : "Emp", g, s)
            @test real(norm_sqr(ψ; alg = "exact")) ≈ 1.0
            layer = Any[("F_hop", (v, v + 1), 0.37) for v in 1:(n - 1)]
            ψt, _ = apply_gates(layer, ψ; apply_kwargs = (; maxdim = 8, cutoff = 1.0e-14))
            cs = jw_ops(n)
            ψv = jw_evolve(layer, cs, Dict(v => v for v in 1:n), n; occupied = (2,))
            nrm = real(ψv' * ψv)
            for v in 1:n
                occ = real(only(expect(ψt, ("N", [v]); alg = "bp")))
                @test occ ≈ real(ψv' * (cs[v]' * cs[v]) * ψv) / nrm atol = 1.0e-12
            end
            c_tn = only(expect(ψt, ("CdagC", (1, 4)); alg = "bp"))
            @test c_tn ≈ (ψv' * (cs[1]' * cs[4]) * ψv) / nrm atol = 1.0e-12
            #pair creation does not commute with particle number: must error
            @test_throws Exception TI.op("F_pair", only(s[1]), only(s[2]); θ = 0.1)
        end

        @testset "spinful (d = 4) Hubbard chain ≡ dense JW (2 modes/site)" for symm in ("fZ2", "fU1xU1")
            n = 4
            nm = 2n
            g = NamedGraph(Graphs.path_graph(n))
            s = TNQS.siteinds("SpinfulFermion", g; symmetry = symm)
            cs = jw_ops(nm)
            up(v) = 2v - 1
            dn(v) = 2v
            init = Dict(1 => "Up", 2 => "Dn", 3 => "UpDn", 4 => "Emp")
            #the nonzero total (N↑, N↓) under fU1xU1 rides an automatic Charge leg
            ψ = tensornetworkstate(ComplexF64, v -> init[v], g, s)
            @test real(norm_sqr(ψ; alg = "exact")) ≈ 1.0
            layer = Any[]
            for _ in 1:2
                append!(layer, ("F_hop", (v, v + 1), 0.31) for v in 1:2:(n - 1))
                append!(layer, ("F_hop", (v, v + 1), 0.31) for v in 2:2:(n - 1))
                append!(layer, ("F_int", [v], 0.23) for v in 1:n)
                append!(layer, ("F_hop_up", (v, v + 1), 0.155) for v in 1:2:(n - 1))
            end
            ψt, errs = apply_gates(layer, ψ; apply_kwargs = (; maxdim = 32, cutoff = 1.0e-14))
            @test maximum(errs) < 1.0e-12
            hop_m(j, k) = cs[j]' * cs[k] + cs[k]' * cs[j]
            ψv = zeros(ComplexF64, 2^nm)
            ψv[1 + sum(m -> 2^(nm - m), [up(1), dn(2), up(3), dn(3)]; init = 0)] = 1.0
            ψv = foldl(layer; init = ψv) do ϕv, gate
                if gate[1] == "F_int"
                    v = only(gate[2])
                    exp(-im * gate[3] * (cs[up(v)]' * cs[up(v)] * cs[dn(v)]' * cs[dn(v)])) * ϕv
                else
                    v, w = gate[2]
                    H = gate[1] == "F_hop" ? hop_m(up(v), up(w)) + hop_m(dn(v), dn(w)) :
                        hop_m(up(v), up(w))
                    exp(-im * gate[3] * H) * ϕv
                end
            end
            nrm = real(ψv' * ψv)
            for (name, mop) in (
                    ("Nup", v -> cs[up(v)]' * cs[up(v)]),
                    ("Ndn", v -> cs[dn(v)]' * cs[dn(v)]),
                    ("NupNdn", v -> cs[up(v)]' * cs[up(v)] * cs[dn(v)]' * cs[dn(v)]),
                )
                for v in 1:n
                    tn = only(expect(ψt, (name, [v]); alg = "bp"))
                    @test tn ≈ (ψv' * mop(v) * ψv) / nrm atol = 1.0e-12
                end
            end
            c_tn = only(expect(ψt, ("CdagC_up", (1, 3)); alg = "bp"))
            @test c_tn ≈ (ψv' * (cs[up(1)]' * cs[up(3)]) * ψv) / nrm atol = 1.0e-12
            h_tn = only(expect(ψt, ("hopping_dn", (2, 3)); alg = "bp"))
            @test h_tn ≈ (ψv' * hop_m(dn(2), dn(3)) * ψv) / nrm atol = 1.0e-12
        end

        @testset "2D grid ≡ dense JW (strings on vertical bonds)" begin
            g = named_grid((2, 3))
            vs = sort(collect(vertices(g)))
            n = length(vs)
            mode = Dict(v => i for (i, v) in enumerate(vs))
            s = TNQS.siteinds("Fermion", g)
            ψ = tensornetworkstate(ComplexF64, v -> "Emp", g, s)
            es = collect(edges(g))
            layer = Any[]
            append!(layer, ("F_pair", (src(e), dst(e)), 0.29) for e in es)
            append!(layer, ("F_hop", (src(e), dst(e)), 0.37) for e in es)
            append!(layer, ("F_phase", [v], 0.21) for v in vs)
            ψt, errs = apply_gates(layer, ψ; apply_kwargs = (; maxdim = 64))
            @test maximum(errs) < 1.0e-12
            cs = jw_ops(n)
            ψv = jw_evolve(layer, cs, mode, n)
            nrm = real(ψv' * ψv)
            occ_tn = [real(only(expect(ψt, ("N", [v]); alg = "exact"))) for v in vs]
            occ_jw = [real(ψv' * (cs[mode[v]]' * cs[mode[v]]) * ψv) / nrm for v in vs]
            @test maximum(abs.(occ_tn .- occ_jw)) < 1.0e-12
            #vertical edge: non-adjacent JW modes — the string must come out of the category
            e_v = first(filter(e -> src(e)[1] != dst(e)[1], es))
            v1, v2 = src(e_v), dst(e_v)
            j, k = mode[v1], mode[v2]
            hop = real(only(expect(ψt, ("hopping", (v1, v2)); alg = "exact")))
            @test hop ≈ real(ψv' * (cs[j]' * cs[k] + cs[k]' * cs[j]) * ψv) / nrm atol = 1.0e-12
            pr = real(only(expect(ψt, ("pairing", (v1, v2)); alg = "exact")))
            @test pr ≈ real(ψv' * (cs[j]' * cs[k]' + cs[k] * cs[j]) * ψv) / nrm atol = 1.0e-12
            #certified sampling on the fermionic state through the full gauged pipeline
            #(symmetric gauge + psd_gauge'd ρ closures): certificates ≈ p/q ≈ 1, and
            #every sampled configuration lands in the physical charge sector
            samples = sample_directly_certified(
                ψt, 2; alg = "boundarymps",
                norm_mps_bond_dimension = 8, projected_mps_bond_dimension = 8
            )
            @test all(x -> abs(real(x.poverq) - 1) < 0.1, samples)
            @test all(x -> iseven(sum(values(x.bitstring))), samples)
            #fermionic boundary MPS: fitted seam messages (fit-adjoint supertrace metric
            #on out-arrow crossing legs) + a joint odd-pair operator through the walk.
            #Residual ~1e-4 is the known fixed-link-sector fitting allocation.
            bmps = update(BoundaryMPSCache(ψt, 8))
            e_h = first(filter(e -> src(e)[1] == dst(e)[1], es))
            w1, w2 = src(e_h), dst(e_h)
            n_bmps = real(only(expect(bmps, ("N", [w1]); alg = "boundarymps")))
            @test n_bmps ≈ real(ψv' * (cs[mode[w1]]' * cs[mode[w1]]) * ψv) / nrm atol = 1.0e-3
            c_bmps = only(expect(bmps, ("CdagC", (w1, w2)); alg = "boundarymps"))
            @test c_bmps ≈ (ψv' * (cs[mode[w1]]' * cs[mode[w2]]) * ψv) / nrm atol = 1.0e-3
        end

        @testset "loop corrections at odd total parity (charge-leg gauge line)" begin
            #3 fermions on a 2x3 grid: odd total parity hangs a fermion-odd Charge leg on
            #the T-join root. Loop weights of regions threaded by that gauge line need
            #the baseline-closure normalization; the mcs=8 series is complete on this
            #graph, so the corrected norm must converge to exact.
            g = named_grid((2, 3))
            s = TNQS.siteinds("Fermion", g)
            ψ = tensornetworkstate(ComplexF64, v -> isodd(sum(v)) ? "Occ" : "Emp", g, s)
            half = Any[]
            for ces in edge_color(g, 4)
                append!(half, ("F_hop", pair, -0.05) for pair in ces)
            end
            ψt, _ = apply_gates(vcat(half, reverse(half)), ψ; apply_kwargs = (; maxdim = 8, cutoff = 1.0e-12))
            ne = real(norm_sqr(ψt; alg = "exact"))
            err(mcs) = abs(real(norm_sqr(ψt; alg = "loopcorrections", max_configuration_size = mcs)) - ne) / ne
            e0, e4, e8 = err(0), err(4), err(8)
            @test e4 < 0.01 * e0    #squares captured
            @test e8 < 1.0e-12      #complete series ⇒ exact
        end
    end
end
end
