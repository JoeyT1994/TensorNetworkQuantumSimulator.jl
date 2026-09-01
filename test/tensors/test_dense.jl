@testset "dense tensor algebra" begin
    Random.seed!(0x5eed_0001)

    @testset "index algebra" begin
        i = Index(3, "i")
        @test TI.dim(i) == 3
        @test TI.plev(TI.prime(i)) == 1
        @test TI.prime(i) != i
        @test TI.noprime(TI.prime(i)) == i
        @test TI.dag(i) == i
        @test TI.sim(i) != i
        @test i' == TI.prime(i)
        @test eltype(TI.random_tensor(i, TI.sim(i))) == Float64
    end

    @testset "contraction" begin
        ki, kj, kk = Index(3, "i"), Index(4, "j"), Index(2, "k")
        A = rand(3, 4)
        B = rand(4, 2)
        Ak = Tensor([ki, kj], copy(A))
        Bk = Tensor([kj, kk], copy(B))
        @test tarray(Ak * Bk, ki, kk) ≈ A * B
        Ck = Tensor([kk], rand(2))
        @test length(vec((Ak * Ck).data)) == 24
        @test TI.scalar(Ak * Tensor([ki, kj], copy(A))) ≈
            TI.scalar(TI.contract([Ak, Ak]; sequence = [1, 2]))
    end

    @testset "combiner and directsum" begin
        ki, kj, kk = Index(2, "i"), Index(3, "j"), Index(2, "k")
        T = Tensor([ki, kj, kk], rand(2, 3, 2))
        C = TI.combiner([ki, kj])
        c = TI.combinedind(C)
        Tc = T * C
        @test sort(TI.dim.(TI.inds(Tc))) == [2, 6]
        @test tarray(Tc * C, ki, kj, kk) ≈ T.data
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
            F1k, F2k, speck = TI.factorize_svd(
                Tk, [ki, kj]; ortho = "none", singular_values! = svk, maxdim
            )
            @test size(svk[].data, 1) <= maxdim
            maxdim == 6 && @test tarray(F1k * F2k, ki, kj, kk, kl) ≈ A
            maxdim == 6 && @test speck.truncerr < 1e-12
            maxdim == 3 && @test speck.truncerr > 0
        end

        Lk, Rk2 = factorize(Tk, ki, kj; ortho = "left", tags = "Link,l=1")
        @test tarray(Lk * Rk2, ki, kj, kk, kl) ≈ A
        lmat = reshape(tarray(Lk, ki, kj, TI.commonind(Lk, Rk2)), 12, :)
        @test lmat' * lmat ≈ one(lmat' * lmat)

        m = rand(ComplexF64, 5, 5)
        M = m + m' + 20.0 * one(m)
        hi1, hi2 = Index(5, "h"), Index(5, "h2")
        Ms, Mis = TNQS.pseudo_sqrt_inv_sqrt(Tensor([hi1, hi2], copy(M)))
        @test tarray(Ms, hi1, hi2) * tarray(Ms, hi1, hi2) ≈ M
        @test tarray(Ms, hi1, hi2) * tarray(Mis, hi1, hi2) ≈ one(M)

        #A PSD eigensolve may acquire a small negative eigenvalue from roundoff.
        #cutoff = 0 retains every strictly positive mode but must still project that
        #negative mode out, without changing the complex storage type.
        P = ComplexF64[-1.0e-12 0 0; 0 1.0e-20 0; 0 0 4]
        pi, pj = Index(3, "psd-i"), Index(3, "psd-j")
        Ps, Pis = TNQS.pseudo_sqrt_inv_sqrt(Tensor([pi, pj], P); cutoff = 0.0)
        @test tarray(Ps, pi, pj) ≈ ComplexF64[0 0 0; 0 1.0e-10 0; 0 0 2]
        @test tarray(Pis, pi, pj) ≈ ComplexF64[0 0 0; 0 1.0e10 0; 0 0 0.5]
        @test all(isfinite, tarray(Pis, pi, pj))

        Ps_default, Pis_default = TNQS.pseudo_sqrt_inv_sqrt(Tensor([pi, pj], P))
        @test tarray(Ps_default, pi, pj) ≈ ComplexF64[0 0 0; 0 0 0; 0 0 2]
        @test tarray(Pis_default, pi, pj) ≈ ComplexF64[0 0 0; 0 0 0; 0 0 0.5]

        Ps_cut, Pis_cut = TNQS.pseudo_sqrt_inv_sqrt(
            Tensor([pi, pj], P); cutoff = 1.0e-10
        )
        @test tarray(Ps_cut, pi, pj) ≈ ComplexF64[0 0 0; 0 0 0; 0 0 2]
        @test tarray(Pis_cut, pi, pj) ≈ ComplexF64[0 0 0; 0 0 0; 0 0 0.5]
        @test_throws ArgumentError TNQS.pseudo_sqrt_inv_sqrt(
            Tensor([pi, pj], P); cutoff = -eps()
        )
    end
end
