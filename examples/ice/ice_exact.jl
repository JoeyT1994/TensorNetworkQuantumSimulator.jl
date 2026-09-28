# Exact bilayer transfer matrices of ice (ice rules, all configurations weight 1) on small periodic
# cross-sections: L1 × L2 cells of the puckered honeycomb bilayer, cell R = (a(R), b(R) = a(R) + δ₁).
# M maps the vertical bonds below the a sites to those above the b sites (variable: H near the upper
# O), so Z(Ic) = Tr M^n and Z(Ih) = Tr (M Mᵀ)^{n/2} (Onsager; Xu, Lin & Zhang 2025).
#
# Checks Mᵀ = I M I, I the in-plane inversion R → −R, and then compares, per molecule,
#   S_c = ln λ(M),  S_h = ln σ_max(M) = ln λ(I M)  (I M is symmetric),  S_sym = ln λ((M + Mᵀ)/2)
# (the maximum of the real Rayleigh quotient, which a single real PEPS optimises), and the overlap of
# the hexagonal boundary state (top eigenvector of I M) with its inversion image.
using LinearAlgebra, Printf

function transfer(L1, L2)
    N = L1 * L2
    cell(i, j) = mod(i, L1) + L1 * mod(j, L2) + 1
    # edges of cell c: 3(c−1)+1: a(c)–b(c); +2: a(c)–b(c − e1); +3: a(c)–b(c − e2). h = 1: H near a.
    ea = [3(c - 1) + k for c in 1:N, k in 1:3]
    eb = zeros(Int, N, 3)                              # b(c) meets e0(c), e1(c + e1), e2(c + e2)
    for i in 0:(L1 - 1), j in 0:(L2 - 1)
        c = cell(i, j)
        eb[c, :] = [3(c - 1) + 1, 3(cell(i + 1, j) - 1) + 2, 3(cell(i, j + 1) - 1) + 3]
    end
    M = zeros(2^N, 2^N)
    for h in 0:(2^(3N) - 1)
        inn = 0; out = 0; ok = true
        for c in 1:N
            da = ((h >> (ea[c, 1] - 1)) & 1) + ((h >> (ea[c, 2] - 1)) & 1) + ((h >> (ea[c, 3] - 1)) & 1)
            mb = 3 - (((h >> (eb[c, 1] - 1)) & 1) + ((h >> (eb[c, 2] - 1)) & 1) + ((h >> (eb[c, 3] - 1)) & 1))
            (1 <= da <= 2 && 1 <= mb <= 2) || (ok = false; break)
            inn |= (2 - da) << (c - 1)                 # H near a on the bond below: 2 − (in-plane H near a)
            out |= (mb - 1) << (c - 1)                 # H near the O above b: 1 − (2 − in-plane H near b)
        end
        ok && (M[out + 1, inn + 1] += 1)
    end
    iv = [cell(-i, -j) for j in 0:(L2 - 1) for i in 0:(L1 - 1)]   # cell index → its inversion image
    P = [sum(((s >> (c - 1)) & 1) << (iv[c] - 1) for c in 1:N) + 1 for s in 0:(2^N - 1)]
    return M, P
end

function main()
    @printf("%-6s %6s  %9s  %10s %10s %10s  %9s %9s  %9s\n", "L1×L2", "dim", "|Mᵀ−IMI|", "w_c", "w_h", "w_sym",
            "S_h−S_c", "S_sym−S_c", "F_I/site")
    for (L1, L2) in ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
        t = @elapsed M, P = transfer(L1, L2)
        N = L1 * L2; n = 2N                                 # molecules
        IMI = M[P, P]
        dev = norm(M' - IMI)
        IM = M[P, :]
        λc = maximum(real, eigvals(M))
        σ = opnorm(M)
        λs = eigmax(Symmetric((M + M') / 2))
        E = eigen(Symmetric((IM + IM') / 2))
        R = E.vectors[:, end]
        F = abs(dot(R[P], R)) / dot(R, R)
        @printf("%-6s %6d  %9.1e  %10.7f %10.7f %10.7f  %9.2e %9.2e  %9.6f   (%.0f s, |IM−IMᵀ| = %.0e, λ(IM) − σ = %.0e)\n",
                "$(L1)×$(L2)", 2^N, dev, λc^(1 / n), σ^(1 / n), λs^(1 / n), (log(σ) - log(λc)) / n, (log(λs) - log(λc)) / n,
                F^(1 / N), t, norm(IM - IM'), E.values[end] - σ)
        flush(stdout)
    end
    println("3D references: w ≈ 1.507458 (Ih, Xu–Lin–Zhang D = 7), Pauling 1.5")
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
