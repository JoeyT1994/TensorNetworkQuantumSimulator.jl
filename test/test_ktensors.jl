@eval module $(gensym())
using Test: @test, @testset, @test_throws
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator.KTensors: KTensors, KIndex, KTensor
using Graphs: Graphs
using NamedGraphs: NamedGraph
const TI = TensorNetworkQuantumSimulator.TensorInterface
using LinearAlgebra: LinearAlgebra, norm, qr, factorize

# Cross-checks against the historical ITensors implementation run when ITensors is
# available (the Pkg.test target includes it); plain `julia --project=.` runs skip them.
const HAS_ITENSORS = !isnothing(Base.find_package("ITensors"))
HAS_ITENSORS || @info "ITensors not available: skipping cross-backend conformance checks"

# Array of a KTensor in a requested index order
function karray(t::KTensor, is...)
    perm = map(i -> findfirst(==(i), t.inds), collect(is))
    return permutedims(t.data, perm)
end

@testset "KTensors backend" begin
    @testset "index algebra" begin
        i = KIndex(3, "i")
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
        ki, kj, kk = KIndex(3, "i"), KIndex(4, "j"), KIndex(2, "k")
        A = rand(3, 4)
        B = rand(4, 2)
        Ak = KTensor([ki, kj], copy(A))
        Bk = KTensor([kj, kk], copy(B))
        @test karray(Ak * Bk, ki, kk) ≈ A * B
        # outer product
        Ck = KTensor([kk], rand(2))
        @test length(vec((Ak * Ck).data)) == 24
        # scalar contraction and sequence execution agree
        @test TI.scalar(Ak * KTensor([ki, kj], copy(A))) ≈ TI.scalar(TI.contract([Ak, Ak]; sequence = [1, 2]))
    end

    @testset "combiner and directsum" begin
        ki, kj, kk = KIndex(2, "i"), KIndex(3, "j"), KIndex(2, "k")
        T = KTensor([ki, kj, kk], rand(2, 3, 2))
        C = TI.combiner([ki, kj])
        c = TI.combinedind(C)
        Tc = T * C
        @test sort(TI.dim.(TI.inds(Tc))) == [2, 6]
        @test karray(Tc * C, ki, kj, kk) ≈ T.data     # combine then split is the identity
        # directsum block-embeds along the paired axes
        l1, l2, ln = KIndex(2, "l1"), KIndex(3, "l2"), KIndex(5, "ln")
        A = KTensor([ki, l1], rand(2, 2))
        B = KTensor([ki, l2], rand(2, 3))
        D = TI.directsum([ln], A => (l1,), B => (l2,))
        @test karray(D, ki, ln)[:, 1:2] ≈ A.data
        @test karray(D, ki, ln)[:, 3:5] ≈ B.data
    end

    @testset "factorizations" begin
        ki, kj, kk, kl = KIndex(3, "i"), KIndex(4, "j"), KIndex(3, "k"), KIndex(2, "l")
        A = rand(ComplexF64, 3, 4, 3, 2)
        Tk = KTensor([ki, kj, kk, kl], copy(A))

        Qk, Rk = qr(Tk, [ki, kj])
        @test karray(Qk * Rk, ki, kj, kk, kl) ≈ A
        bq = TI.commonind(Qk, Rk)
        qmat = reshape(karray(Qk, ki, kj, bq), 12, TI.dim(bq))
        @test qmat' * qmat ≈ one(qmat' * qmat)

        for maxdim in (6, 3)
            svk = Ref{Any}(nothing)
            F1k, F2k, speck = TI.factorize_svd(Tk, [ki, kj]; ortho = "none", singular_values! = svk, maxdim)
            @test size(svk[].data, 1) <= maxdim
            maxdim == 6 && @test karray(F1k * F2k, ki, kj, kk, kl) ≈ A
            maxdim == 6 && @test speck.truncerr < 1e-12
            maxdim == 3 && @test speck.truncerr > 0
        end

        Lk, Rk2 = factorize(Tk, ki, kj; ortho = "left", tags = "Link,l=1")
        @test karray(Lk * Rk2, ki, kj, kk, kl) ≈ A
        lmat = reshape(karray(Lk, ki, kj, TI.commonind(Lk, Rk2)), 12, :)
        @test lmat' * lmat ≈ one(lmat' * lmat)

        # hermitian eigen path used by simple update
        m = rand(ComplexF64, 5, 5)
        M = m + m' + 20.0 * one(m)     # comfortably PSD
        hi1, hi2 = KIndex(5, "h"), KIndex(5, "h2")
        Ms, Mis = TNQS.pseudo_sqrt_inv_sqrt(KTensor([hi1, hi2], copy(M)))
        @test karray(Ms, hi1, hi2) * karray(Ms, hi1, hi2) ≈ M
        @test karray(Ms, hi1, hi2) * karray(Mis, hi1, hi2) ≈ one(M)
    end

    if HAS_ITENSORS
        @eval using ITensors: ITensors, Index, random_itensor
        @testset "ops and states match ITensors" begin
            s1i, s2i = Index(2, "S=1/2,Site"), Index(2, "S=1/2,Site")
            k1, k2 = KIndex(2, "S=1/2,Site"), KIndex(2, "S=1/2,Site")
            for name in ["X", "Y", "Z", "H", "I", "S+", "S-"]
                @test karray(TI.op(name, k1), k1', k1) ≈ Array(ITensors.op(name, s1i), s1i', s1i)
            end
            for (name, kw) in [("Rx", (θ = 0.37,)), ("Ry", (θ = 0.37,)), ("Rz", (θ = 0.37,)), ("P", (ϕ = 0.37,))]
                @test karray(TI.op(name, k1; kw...), k1', k1) ≈ Array(ITensors.op(name, s1i; kw...), s1i', s1i)
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
                @test karray(okt, k1', k2', k1, k2) ≈ Array(oit, s1i', s2i', s1i, s2i)
            end
            for st in ["↑", "↓", "+", "-"]
                @test karray(TI.state(st, k1), k1) ≈ Array(ITensors.state(st, s1i), s1i)
            end
        end

        @testset "factorization conventions match ITensors" begin
            i, j, k, l = Index(3, "i"), Index(4, "j"), Index(3, "k"), Index(2, "l")
            ki, kj, kk, kl = KIndex(3, "i"), KIndex(4, "j"), KIndex(3, "k"), KIndex(2, "l")
            A = rand(ComplexF64, 3, 4, 3, 2)
            Ti = ITensors.ITensor(A, i, j, k, l)
            Tk = KTensor([ki, kj, kk, kl], copy(A))
            for maxdim in (6, 3)
                svi = Ref{Any}(nothing)
                F1i, F2i, speci = ITensors.factorize_svd(Ti, (i, j); ortho = "none", singular_values! = svi, maxdim)
                svk = Ref{Any}(nothing)
                F1k, F2k, speck = TI.factorize_svd(Tk, [ki, kj]; ortho = "none", singular_values! = svk, maxdim)
                @test speck.truncerr ≈ speci.truncerr atol = 1e-13
                @test karray(F1k * F2k, ki, kj, kk, kl) ≈ Array(F1i * F2i, i, j, k, l)
                @test sort(collect(ITensors.diag(svi[])); rev = true) ≈
                    sort(real.(LinearAlgebra.diag(svk[].data)); rev = true)
            end
        end
    end

    @testset "graded (TensorKit Z2) backend" begin
        using TensorNetworkQuantumSimulator.KTensors: TKTensor
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
        @test ψg[(1, 1)] isa TKTensor
        @test zd ≈ zg atol = 1e-10
        @test zxd ≈ zxg atol = 1e-10
        @test ed ≈ eg atol = 1e-12
        #the conserving circuit leaves the state genuinely block-sparse: the flux-zero
        #constraint keeps exactly half the product basis for balanced parity sectors
        nstored = sum(
            sum(p -> length(ψg[v].data[p[1], p[2]]), KTensors.TK.fusiontrees(ψg[v].data); init = 0)
                for v in vertices(ψg)
        )
        nfull = sum(prod(Int[TI.dim(i) for i in TI.inds(ψg[v])]) for v in vertices(ψg))
        @test nstored <= 0.5 * nfull

        #non-conserving pieces must fail loudly: that is the point of the symmetry
        sg = only(TNQS.siteinds(TNQS.tensornetwork(ψg))[(1, 1)])
        @test_throws Exception TI.op("X", sg)
        @test_throws Exception TI.state("↓", sg)   #charged SINGLE tensor: flux-odd
        #an odd number of charged sites has nonzero total charge: not representable
        gz = named_grid((3, 3))
        sz = siteinds("S=1/2", gz; sectors = [0 => 1, 1 => 1], symmetry = "Z2")
        @test_throws Exception tensornetworkstate(ComplexF64, v -> v == (1, 1) ? "↓" : "↑", gz, sz)

        #graded boundary MPS: random conserving init over convolved charged link spectra
        #(fermion-branch recipe; conservation itself is native — the init only picks link
        #sectors). Fixed per-sector allocation makes this variational at ~1e-4. Certified
        #sampling needs single-layer (amplitude) messages, which carry net flux — that
        #waits for charged dummy legs.
        ne = real(norm_sqr(ψg; alg = "exact"))
        nb = real(norm_sqr(ψg; alg = "boundarymps", mps_bond_dimension = 20))
        @test abs(nb / ne - 1) < 5e-3

        #graded factorization round-trip on a generic conserving 4-leg tensor
        si = KTensors.KIndex(KTensors.graded_space("Z2", [0 => 1, 1 => 2]), "a")
        sj = KTensors.KIndex(KTensors.graded_space("Z2", [0 => 2, 1 => 1]), "b")
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
            s1 = KTensors.new_fermion_index(1, 1; tags = "s1")
            s2 = KTensors.new_fermion_index(1, 1; tags = "s2")
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
            #odd total parity is a nonzero total charge: not representable
            @test_throws Exception tensornetworkstate(ComplexF64, v -> v == 1 ? "Occ" : "Emp", g, s)
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
            ψc = tensornetworkstate(ComplexF64, v -> v in (2, 4, 5) ? "Occ" : "Emp", g, s; charge_leg = true)
            ψct, _ = apply_gates(layer, ψc; apply_kwargs = (; maxdim = 32, cutoff = 1.0e-14))
            ψcv = jw_evolve(layer, cs, Dict(v => v for v in 1:n), n; occupied = (2, 4, 5))
            nrmc = real(ψcv' * ψcv)
            occ3 = real(only(expect(ψct, ("N", [3]); alg = "bp")))
            @test occ3 ≈ real(ψcv' * (cs[3]' * cs[3]) * ψcv) / nrmc atol = 1.0e-12
            tnc = only(expect(ψct, ("CdagC", (1, 5)); alg = "bp"))
            @test tnc ≈ (ψcv' * (cs[1]' * cs[5]) * ψcv) / nrmc atol = 1.0e-12
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
    end
end
end
