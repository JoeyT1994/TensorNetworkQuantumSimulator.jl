@eval module $(gensym())
using Test: @test, @testset
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator.KTensors: KIndex, KTensor
const TI = TensorNetworkQuantumSimulator.TensorInterface
using ITensors: ITensors, Index, random_itensor
using LinearAlgebra: LinearAlgebra, norm, qr, factorize

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
    end

    @testset "contraction against ITensors" begin
        i, j, k = Index(3, "i"), Index(4, "j"), Index(2, "k")
        A = random_itensor(Float64, i, j)
        B = random_itensor(Float64, j, k)
        ki, kj, kk = KIndex(3, "i"), KIndex(4, "j"), KIndex(2, "k")
        Ak = KTensor([ki, kj], Array(A, i, j))
        Bk = KTensor([kj, kk], Array(B, j, k))
        @test karray(Ak * Bk, ki, kk) ≈ Array(A * B, i, k)
        # outer product
        Ck = KTensor([kk], rand(2))
        @test size((Ak * Ck).data |> vec, 1) == 24
        # scalar contraction
        @test TI.scalar(Ak * KTensor([ki, kj], Array(A, i, j))) ≈ TI.scalar(TI.contract([Ak, Ak]; sequence = [1, 2]))
    end

    @testset "ops and states match ITensors" begin
        s1i, s2i = Index(2, "S=1/2,Site"), Index(2, "S=1/2,Site")
        k1, k2 = KIndex(2, "S=1/2,Site"), KIndex(2, "S=1/2,Site")
        for name in ["X", "Y", "Z", "H", "I", "S+", "S-"]
            oit = ITensors.op(name, s1i)
            okt = TI.op(name, k1)
            @test karray(okt, k1', k1) ≈ Array(oit, s1i', s1i)
        end
        for (name, kw) in [("Rx", (θ = 0.37,)), ("Ry", (θ = 0.37,)), ("Rz", (θ = 0.37,)), ("P", (ϕ = 0.37,))]
            oit = ITensors.op(name, s1i; kw...)
            okt = TI.op(name, k1; kw...)
            @test karray(okt, k1', k1) ≈ Array(oit, s1i', s1i)
        end
        for (name, kw) in [("Rzz", (ϕ = 0.37,)), ("Rxx", (ϕ = 0.37,)), ("Ryy", (ϕ = 0.37,)), ("CZ", (;)), ("CNOT", (;)), ("CPHASE", (ϕ = 0.37,)), ("SWAP", (;))]
            oit = isempty(kw) ? ITensors.op(name, s1i, s2i) : ITensors.op(name, s1i, s2i; kw...)
            okt = isempty(kw) ? TI.op(name, k1, k2) : TI.op(name, k1, k2; kw...)
            @test karray(okt, k1', k2', k1, k2) ≈ Array(oit, s1i', s2i', s1i, s2i)
        end
        for st in ["↑", "↓", "+", "-"]
            @test karray(TI.state(st, k1), k1) ≈ Array(ITensors.state(st, s1i), s1i)
        end
    end

    @testset "factorizations match ITensors" begin
        i, j, k, l = Index(3, "i"), Index(4, "j"), Index(3, "k"), Index(2, "l")
        ki, kj, kk, kl = KIndex(3, "i"), KIndex(4, "j"), KIndex(3, "k"), KIndex(2, "l")
        A = rand(ComplexF64, 3, 4, 3, 2)
        Ti = ITensors.ITensor(A, i, j, k, l)
        Tk = KTensor([ki, kj, kk, kl], copy(A))

        # qr: reconstruction + isometry
        Qk, Rk = qr(Tk, [ki, kj])
        @test karray(Qk * Rk, ki, kj, kk, kl) ≈ A
        bq = TI.commonind(Qk, Rk)
        qmat = reshape(karray(Qk, ki, kj, bq), 12, TI.dim(bq))
        @test qmat' * qmat ≈ one(qmat' * qmat)

        # factorize_svd ortho = "none": F1 = U√S, F2 = √S·V, matching truncerr with ITensors
        for maxdim in (6, 3)
            svi = Ref{Any}(nothing)
            F1i, F2i, speci = TI.factorize_svd(Ti, (i, j); ortho = "none", singular_values! = svi, maxdim)
            svk = Ref{Any}(nothing)
            F1k, F2k, speck = TI.factorize_svd(Tk, [ki, kj]; ortho = "none", singular_values! = svk, maxdim)
            @test speck.truncerr ≈ speci.truncerr atol = 1e-13
            @test karray(F1k * F2k, ki, kj, kk, kl) ≈ Array(F1i * F2i, i, j, k, l)
            @test sort(collect(ITensors.diag(svi[])); rev = true) ≈ sort(real.(LinearAlgebra.diag(svk[].data)); rev = true)
        end

        # factorize ortho = "left": isometric L, reconstruction
        Lk, Rk2 = factorize(Tk, ki, kj; ortho = "left", tags = "Link,l=1", maxdim = 5)
        Li, Ri2 = factorize(Ti, i, j; ortho = "left", tags = "Link,l=1", maxdim = 5)
        @test karray(Lk * Rk2, ki, kj, kk, kl) ≈ Array(Li * Ri2, i, j, k, l)

        # hermitian eigen path used by simple update
        m = rand(ComplexF64, 5, 5)
        H = m + m'
        hi1, hi2 = KIndex(5, "h"), KIndex(5, "h2")
        Hk = KTensor([hi1, hi2], copy(H))
        Ms, Mis = TNQS.pseudo_sqrt_inv_sqrt(Hk + KTensor([hi1, hi2], 20.0 * one(H)))  # shift to PSD
        M = H + 20.0 * one(H)
        @test karray(Ms, hi1, hi2) * karray(Ms, hi1, hi2) ≈ M
    end

    @testset "end-to-end BP digest matches ITensors" begin
        function digest(backend)
            g = named_grid((3, 3))
            ψ = tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2"; backend)
            layer = Any[("Rx", [v], 0.6) for v in vertices(g)]
            for ces in edge_color(g, 4)
                append!(layer, ("Rzz", pair, 0.4) for pair in ces)
            end
            circuit = reduce(vcat, [layer for _ in 1:2])
            bp_update_kwargs = (; maxiter = 30, tolerance = 1.0e-12)
            ψ, errs = apply_gates(circuit, ψ; apply_kwargs = (; maxdim = 3, cutoff = 1.0e-14), bp_update_kwargs)
            zs = expect(ψ, [("Z", [v]) for v in vertices(g)]; alg = "bp", cache_update_kwargs = bp_update_kwargs)
            zzs = expect(ψ, [("ZZ", [src(e), dst(e)]) for e in edges(g)]; alg = "bp", cache_update_kwargs = bp_update_kwargs)
            return real(sum(zs)), real(sum(zzs)), sum(errs)
        end
        zi, zzi, ti = digest("itensors")
        zk, zzk, tk = digest("ktensors")
        @test zi ≈ zk atol = 1e-10
        @test zzi ≈ zzk atol = 1e-10
        @test ti ≈ tk atol = 1e-12
    end
end
end
