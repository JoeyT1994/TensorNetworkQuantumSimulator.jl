@eval module $(gensym())
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_throws
const TNQS = TensorNetworkQuantumSimulator

@testset "Ice: bilayer transfer operator and boundary PEPS" begin
    # ICE: the bilayer cell against a brute-force count of the ice rules on the 2 × 3 cell torus (the
    # transfer matrix M between vertical bonds), Mᵀ = I M I with I the in-plane inversion, and M not
    # symmetric (the stackings differ).
    let
        W, wl = ice_site()
        Wa = TNQS.array(W, wl...)
        @test sum(Wa) == 18 && maximum(Wa) == 1           # 3 × 3 per value of the intra-cell bond
        @test TNQS.norm(TNQS.replaceinds(W, collect(wl[1:4]), collect(wl[[3, 4, 1, 2]])) - W) == 0        # diagonal mirror
        @test TNQS.norm(TNQS.replaceinds(W, collect(wl), collect(wl[[2, 1, 4, 3, 6, 5]])) - W) == 0   # inversion × z-flip
        L1, L2 = 2, 3
        N = L1 * L2
        cell(i, j) = mod(i, L1) + L1 * mod(j, L2) + 1
        xb = [TNQS.new_index(2) for _ in 1:N]; yb = [TNQS.new_index(2) for _ in 1:N]
        zi = [TNQS.new_index(2) for _ in 1:N]; zo = [TNQS.new_index(2) for _ in 1:N]
        ts = [TNQS.replaceinds(W, collect(wl), [xb[cell(i - 1, j)], xb[cell(i, j)], yb[cell(i, j - 1)], yb[cell(i, j)],
                                                zi[cell(i, j)], zo[cell(i, j)]]) for j in 0:(L2 - 1) for i in 0:(L1 - 1)]
        M = reshape(Array(TNQS.array(reduce(*, ts), zo..., zi...)), 2^N, 2^N)
        # brute force: h = 1 when an in-plane bond's H is near its a end; a(c) meets the bonds 1–3 of
        # cell c, b(c) bond 1 of c, 2 of c + x̂, 3 of c + ŷ
        Mb = zeros(2^N, 2^N)
        for h in 0:(2^(3N) - 1)
            bit(e) = (h >> (e - 1)) & 1
            zin = 0; zout = 0; ok = true
            for i in 0:(L1 - 1), j in 0:(L2 - 1)
                c = cell(i, j)
                na = bit(3c - 2) + bit(3c - 1) + bit(3c)
                nb = 3 - bit(3c - 2) - bit(3cell(i + 1, j) - 1) - bit(3cell(i, j + 1))
                (1 <= na <= 2 && 1 <= nb <= 2) || (ok = false; break)
                zin |= (2 - na) << (c - 1); zout |= (nb - 1) << (c - 1)
            end
            ok && (Mb[zout + 1, zin + 1] += 1)
        end
        @test M == Mb
        iv = [cell(-i, -j) for j in 0:(L2 - 1) for i in 0:(L1 - 1)]
        P = [sum(((s >> (c - 1)) & 1) << (iv[c] - 1) for c in 1:N) + 1 for s in 0:(2^N - 1)]
        @test M' == M[P, P] && M' != M
    end

    # … and its boundary PEPS, D = 2, χ = 16, diagonal-mirror symmetry: the hexagonal estimator
    # (bra_perm = inversion, maximised) by L-BFGS and Newton–Krylov, the cubic one (both perms,
    # stationary) by Newton–Krylov from it; the permuted-bra gradient against finite differences.
    let INV = (2, 1, 4, 3)
        W, wl = ice_site()
        al = Tuple(TNQS.new_index(2) for _ in 1:5); bl = Tuple(TNQS.new_index(2) for _ in 1:4)
        G = TNQS._bp_group(:diagonal)
        rng = Xoshiro(2)
        A = TNQS._bp_symmetrize(TNQS._bp_initial(W, wl, al, 2, nothing, 0.0, rng) +
                                0.02 * TNQS.random_tensor(rng, Float64, collect(al)), al[1:4], G)
        A = A / TNQS.norm(A)
        dA = TNQS._bp_symmetrize(TNQS.random_tensor(rng, Float64, collect(al)), al[1:4], G); dA = dA / TNQS.norm(dA)
        # The environment gradient is the derivative of the finite-χ ln κ up to the truncation: near
        # the product state 7e-9 at χ = 16 and 1e-12 at χ = 32 (measured 2026-09-27; 2e-5 and 1e-6
        # from a scrambled A with 0.2 noise — the ice networks need χ).
        for (sp, np) in ((INV, INV), (INV, (1, 2, 3, 4)))
            ctx = (; al, bl, site = W, legs = Tuple(wl), maxdim = 32, group = G, bra_perm = sp, norm_perm = np,
                   ctm_tolerance = 1.0e-12, ctm_maxiter = 3000, ctm_kwargs = NamedTuple())
            f0, g0, ln, ls = TNQS._bp_evaluate(A, ctx, nothing, nothing)
            h = 1.0e-4
            fs = Dict(k => TNQS._bp_evaluate(A + k * h * dA, ctx, ln, ls)[1] for k in (-2, -1, 1, 2))
            @test abs(real(TNQS.dot(g0, dA)) - (-fs[2] + 8fs[1] - 8fs[-1] + fs[-2]) / (12h)) < 1.0e-9
            @test abs(real(TNQS.dot(g0, A))) < 1.0e-12
        end

        # The ice networks' gradient has a noise floor ~2e-6 at χ = 16 (Ising's is ~1e-9): solve to it.
        bh = boundary_peps(W, wl, 2; maxdim = 16, symmetry = :diagonal, bra_perm = INV, maxiter = 40, gtol = 1.0e-5)
        @test bh.bra_perm == INV && bh.norm_perm == (1, 2, 3, 4) && issorted(bh.history)
        bhk, ih = boundary_peps_krylov(W, wl, bh; symmetry = :diagonal, norm_perm = (1, 2, 3, 4), merit = :f,
                                       tol = 1.0e-6, noise_tol = 1.0e-5, maxiter = 25)
        @test ih.converged && cvm_freenergy(bhk) >= cvm_freenergy(bh) - 1.0e-12
        bck, ic = boundary_peps_krylov(W, wl, bhk; symmetry = :diagonal, norm_perm = INV, tol = 1.0e-6,
                                       noise_tol = 1.0e-5, maxiter = 25)
        @test ic.converged && bck.norm_perm == INV
        # both near Pauling's 3/2 per molecule, and 3D-ice-like (Xu, Lin & Zhang: W ≈ 1.50745)
        @test all(abs(exp(cvm_freenergy(b) / 2) - 1.5074) < 2.0e-3 for b in (bhk, bck))

        @test_throws ArgumentError boundary_peps(W, wl, 2; maxdim = 8)                       # not C4v
        @test_throws ArgumentError boundary_peps(W, wl, 2; maxdim = 8, symmetry = :nonsense)
        @test_throws ArgumentError boundary_peps(W, wl, 2; maxdim = 8, symmetry = :diagonal, bra_perm = (2, 1, 3, 4))
        @test_throws ArgumentError boundary_peps_krylov(W, wl, bh; symmetry = :diagonal, norm_perm = INV, merit = :f)
    end

end
end
