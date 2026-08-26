@eval module $(gensym())
using Test: @test, @testset
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator.KTensors: KIndex, KTensor
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
end
end
