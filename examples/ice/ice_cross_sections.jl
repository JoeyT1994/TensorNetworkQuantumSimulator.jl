# S_c, S_h and the Rayleigh maximum S_sym on larger finite cross-sections (L1 × L2 cell tori), per
# molecule, from matrix-free products: M v by contracting the torus of `ice_site` cells with v, Mᵀ v =
# I M I v (the identity, spot-checked here), top eigenvalues by Krylov in the dominant FLUX sector.
# The ice rules conserve the vertical flux k = Σ_c in_c = Σ_c out_c, so M is block diagonal; a torus
# with an odd number of cells has no zero-flux sector and is forced to carry flux ±1.
using TensorNetworkQuantumSimulator, LinearAlgebra, Printf, Random
using KrylovKit: eigsolve
BLAS.set_num_threads(4)
const T = TensorNetworkQuantumSimulator
include(joinpath(@__DIR__, "ice_exact.jl"))          # brute-force transfer(L1, L2) for the check

struct Torus
    L1::Int; L2::Int; N::Int
    ins::Vector; outs::Vector
    cells::Vector{Any}                                # cell tensors in row-major order
    P::Vector{Int}                                    # the inversion on configurations (1-based)
end
function Torus(L1, L2)
    W, wl = ice_site()
    N = L1 * L2
    cell(i, j) = mod(i, L1) + L1 * mod(j, L2) + 1
    xb = [T.new_index(2; tags = "xb") for _ in 1:N]; yb = [T.new_index(2; tags = "yb") for _ in 1:N]
    ins = [T.new_index(2; tags = "in") for _ in 1:N]; outs = [T.new_index(2; tags = "out") for _ in 1:N]
    cells = Any[T.replaceinds(W, collect(wl), [xb[cell(i - 1, j)], xb[cell(i, j)], yb[cell(i, j - 1)], yb[cell(i, j)],
                                               ins[cell(i, j)], outs[cell(i, j)]]) for j in 0:(L2 - 1) for i in 0:(L1 - 1)]
    iv = [cell(-i, -j) for j in 0:(L2 - 1) for i in 0:(L1 - 1)]
    P = [sum(((s >> (c - 1)) & 1) << (iv[c] - 1) for c in 1:N) + 1 for s in 0:(2^N - 1)]
    return Torus(L1, L2, N, ins, outs, cells, P)
end
function mul(t::Torus, v::AbstractVector)
    acc = T.from_array(reshape(Vector{Float64}(v), ntuple(_ -> 2, t.N)), t.ins...)
    for c in t.cells
        acc = acc * c
    end
    return vec(Array(T.array(acc, t.outs...)))
end
mulT(t::Torus, v) = mul(t, v[t.P])[t.P]              # Mᵀ v = I M I v

function sector(N, k)
    idx = [s + 1 for s in 0:(2^N - 1) if count_ones(s) == k]
    return idx
end

function entropies(L1, L2; tol = 1.0e-12)
    t = Torus(L1, L2)
    N = t.N; n = 2N
    k = N ÷ 2                                          # the dominant sector (odd N: flux −1; +1 is its mirror)
    sec = sector(N, k)
    proj(v) = (w = zeros(2^N); w[sec] = v[sec]; w)
    x0 = proj(ones(2^N))
    # spot-check Mᵀ = I M I on random vectors: ⟨u, M v⟩ = ⟨Mᵀ u, v⟩
    rng = Xoshiro(1); u = randn(rng, 2^N); v = randn(rng, 2^N)
    chk = abs(dot(u, mul(t, v)) - dot(mulT(t, u), v)) / abs(dot(u, mul(t, v)))
    vals_c, vecs_c, _ = eigsolve(v -> proj(mul(t, v)), x0, 1, :LR; ishermitian = false, tol, krylovdim = 30)
    vals_h, vecs_h, _ = eigsolve(v -> proj(mul(t, v)[t.P]), x0, 1, :LR; ishermitian = true, tol, krylovdim = 30)
    vals_s, _, _ = eigsolve(v -> proj((mul(t, v) + mulT(t, v)) / 2), x0, 1, :LR; ishermitian = true, tol, krylovdim = 30)
    λc, σ, λs = real(vals_c[1]), real(vals_h[1]), real(vals_s[1])
    R = real(vecs_h[1])
    F = abs(dot(R[t.P], R)) / dot(R, R)
    return (; L1, L2, N, k, dim = length(sec), chk, wc = λc^(1 / n), wh = σ^(1 / n), ws = λs^(1 / n),
            dhc = (log(σ) - log(λc)) / n, dsc = (log(λs) - log(λc)) / n, FI = F^(1 / N))
end

# the matrix-free products against brute force (3 × 3, all sectors) first
let t = Torus(3, 3)
    Me, _ = transfer(3, 3)
    v = randn(Xoshiro(2), 2^9)
    @printf("3×3 check: |M v − M_ED v| / |M v| = %.1e\n", norm(mul(t, v) - Me * v) / norm(Me * v))
end
@printf("%-6s %4s %2s %7s  %9s  %12s %12s %12s  %10s %10s  %10s\n", "L1×L2", "N", "k", "sector", "|Mᵀ−IMI|",
        "w_c", "w_h", "w_sym", "S_h−S_c", "S_sym−S_c", "1−F_I/cell")
t0 = time()
for (L1, L2) in ((3, 3), (2, 5), (2, 6), (3, 4), (2, 7), (2, 8), (3, 5), (2, 9), (4, 4))   # L1 ≤ L2: the mirror gives L2 × L1
    time() - t0 > 480 && (println("(time: stop before $(L1)×$(L2))"); break)
    ts = time()
    r = entropies(L1, L2)
    @printf("%-6s %4d %2d %7d  %9.1e  %.10f %.10f %.10f  %10.2e %10.2e  %10.2e  (%.0f s)\n", "$(L1)×$(L2)", r.N, r.k, r.dim,
            r.chk, r.wc, r.wh, r.ws, r.dhc, r.dsc, 1 - r.FI, time() - ts)
    flush(stdout)
end
