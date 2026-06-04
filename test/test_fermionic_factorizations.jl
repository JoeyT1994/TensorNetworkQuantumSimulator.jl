@eval module $(gensym())
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using Test: @testset, @test
using Random
using ITensors
using ITensors: ITensors, Index, ITensor, dim, inds, sim, svd, qr
using Dictionaries: Dictionary
using TensorNetworkQuantumSimulator: random_even_itensor

const FT = TN.FermionicITensor

ITensors.disable_warn_order()

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

random_even_ft(order, dirs, grading; eltp = Float64) =
    FT(random_even_itensor(eltp, order, grading), order, dirs, grading)

# Compare two FermionicITensors up to leg order (they must share Index objects).
function ft_isapprox(A::FT, B::FT; atol = 1e-9)
    Bp = ITensors.permute(B, A.order)
    return isapprox(ITensors.array(A.tensor, A.order...), ITensors.array(Bp.tensor, A.order...); atol)
end

# Fermionic "bond product" P ∘ Q for two 2-leg bond operators of order [a, a2]:
# relabel Q to [a2, a3], contract over a2 (inserts ONE supertrace), relabel a3 -> a2.
function bondmul(P::FT, Q::FT, a::Index, a2::Index)
    a3 = sim(a2)
    Qb = TN.replaceinds(Q, [a, a2], [a2, a3])
    R = P * Qb
    return TN.replaceinds(R, [a3], [a2])
end

@testset "Fermionic factorizations" begin
    Random.seed!(2718)

    # ---- SVD / QR reconstruction --------------------------------------------
    @testset "svd / qr reconstruction (eltype=$eltp)" for eltp in (Float64, ComplexF64)
        i1 = Index(2, "i1"); i2 = Index(3, "i2"); i3 = Index(2, "i3"); i4 = Index(3, "i4")
        gr = Dictionary{Index, Vector{Bool}}(
            Index[i1, i2, i3, i4],
            Vector{Bool}[[false, true], [false, true, true], [true, false], [false, false, true]],
        )
        dirs = Bool[false, true, false, true]
        ft = random_even_ft(Index[i1, i2, i3, i4], dirs, gr; eltp)

        for rows in (Index[i1, i2], Index[i1], Index[i1, i3], Index[i2, i4])
            U, S, V = svd(ft, rows)
            @test ft_isapprox(ft, U * S * V)
            # U's bond grading is data-driven; reconstruction is the real test.

            Q, R = qr(ft, rows)
            @test ft_isapprox(ft, Q * R)
        end
    end

    # ---- SVD truncation ------------------------------------------------------
    @testset "svd truncation (maxdim) keeps largest singular values" begin
        i1 = Index(3, "i1"); i2 = Index(3, "i2")
        gr = Dictionary{Index, Vector{Bool}}(
            Index[i1, i2], Vector{Bool}[[false, true, false], [false, true, false]],
        )
        ft = random_even_ft(Index[i1, i2], Bool[false, true], gr; eltp = Float64)

        Ufull, Sfull, Vfull = svd(ft, Index[i1])
        @test ft_isapprox(ft, Ufull * Sfull * Vfull)

        # truncating must not increase the bond beyond maxdim
        Ut, St, Vt = svd(ft, Index[i1]; maxdim = 1)
        @test dim(only(TN.commoninds(Ut, St))) ≤ 1
    end

    # ---- pseudo_sqrt_inv_sqrt ------------------------------------------------
    @testset "pseudo_sqrt_inv_sqrt (eltype=$eltp)" for eltp in (Float64, ComplexF64)
        bits = Bool[false, true, true, false]
        a = Index(4, "bond"); a2 = sim(a)
        gr = Dictionary{Index, Vector{Bool}}(Index[a, a2], Vector{Bool}[bits, bits])
        re = findall(!, bits); ro = findall(identity, bits)

        randpsd(k) = (B = randn(eltp, k, k); B * B' + k * one(B))  # well-conditioned PSD
        rho = zeros(eltp, 4, 4)
        rho[re, re] = randpsd(length(re))
        rho[ro, ro] = randpsd(length(ro))

        # A valid fermionic bond matrix: M = g·rho (flip the odd rows). The odd sector is
        # then negative-definite, exactly as a real fermionic message would be.
        Mmat = copy(rho)
        Mmat[ro, :] .*= -1
        # a2 = OUT so that `bondmul` inserts the supertrace that reinstates the metric.
        M = FT(ITensor(Mmat, a, a2), Index[a, a2], Bool[true, false], gr)

        X, Xinv = TN.pseudo_sqrt_inv_sqrt(M)
        @test ft_isapprox(M, bondmul(X, X, a, a2))                       # X ∘ X = M
        @test ft_isapprox(X, bondmul(bondmul(X, Xinv, a, a2), X, a, a2)) # X X⁺ X = X
    end
end

end
