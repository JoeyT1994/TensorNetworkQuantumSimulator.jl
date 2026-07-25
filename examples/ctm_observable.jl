# Local observable from the finite CTMRG environment.
#
# The payoff of per-vertex environments: LOCAL quantities. Here the nearest-neighbour
# correlation ⟨s_i s_j⟩ on a horizontal bond, computed by sandwiching the target row
# between the top environment (rows above, CTM-absorbed) and the bottom environment
# (rows below), with spin-weighted site tensors inserted at the two sites. Validated
# against a brute-force spin sum.
#
# Run: julia --project=. --startup-file=no examples/ctm_observable.jl

include("ctm_finite_aniso.jl")   # sqrtW, site_array, absorb_and_truncate (main() guarded)

# Spin-weighted site tensor: Σ_s s · W[u,s]W[r,s]W[d,s]W[l,s]  (measures ⟨s⟩ at the site)
function spin_site_array(Kx, Ky, x, y, Lx, Ly)
    Wx, Wy = sqrtW(Kx), sqrtW(Ky); bx, by = Wx \ ones(2), Wy \ ones(2)
    hN, hE, hS, hW = y > 1, x < Lx, y < Ly, x > 1
    du, dr, dd, dl = hN ? 2 : 1, hE ? 2 : 1, hS ? 2 : 1, hW ? 2 : 1
    sv = (1.0, -1.0)
    t = zeros(Float64, du, dr, dd, dl)
    for u in 1:du, r in 1:dr, dn in 1:dd, l in 1:dl
        acc = 0.0
        for s in 1:2
            wu = hN ? Wy[u, s] : by' * Wy[:, s]
            wr = hE ? Wx[r, s] : bx' * Wx[:, s]
            wd = hS ? Wy[dn, s] : by' * Wy[:, s]
            wl = hW ? Wx[l, s] : bx' * Wx[:, s]
            acc += sv[s] * wu * wr * wd * wl
        end
        t[u, r, dn, l] = acc
    end
    return t
end

# bond indices shared across the whole lattice
function bond_indices(Lx, Ly)
    hb = Dict{Tuple{Int,Int},Index}(); vb = Dict{Tuple{Int,Int},Index}()
    for x in 1:Lx, y in 1:Ly
        x < Lx && (hb[(x, y)] = Index(2, "h$(x)_$(y)"))
        y < Ly && (vb[(x, y)] = Index(2, "v$(x)_$(y)"))
    end
    return hb, vb
end

# build row y as ITensors sharing hb/vb; sites in `spins` are spin-weighted
function make_row(Kx, Ky, y, Lx, Ly, hb, vb; spins = Set{Tuple{Int,Int}}())
    row = ITensor[]
    for x in 1:Lx
        legs = Index[]
        y > 1  && push!(legs, vb[(x, y - 1)]); x < Lx && push!(legs, hb[(x, y)])
        y < Ly && push!(legs, vb[(x, y)]);     x > 1  && push!(legs, hb[(x - 1, y)])
        arr = (x, y) in spins ? spin_site_array(Kx, Ky, x, y, Lx, Ly) :
                                site_array(Kx, Ky, x, y, Lx, Ly)
        push!(row, ITensor(reshape(arr, filter(!=(1), size(arr))...), legs...))
    end
    return row
end

top_env(rows, y, χ) = y == 1 ? nothing :
    foldl((cur, yy) -> absorb_and_truncate(cur, rows[yy], χ), 2:(y - 1); init = rows[1])
bot_env(rows, y, χ) = y == length(rows) ? nothing :
    foldl((cur, yy) -> absorb_and_truncate(cur, rows[yy], χ), (length(rows) - 1):-1:(y + 1); init = rows[end])

# sandwich contraction: (top env) · (row) · (bottom env) → scalar
function contract_three(U, rowy, D)
    Lx = length(rowy)
    merged = ITensor[]
    for x in 1:Lx
        t = rowy[x]
        U !== nothing && (t = t * U[x])
        D !== nothing && (t = t * D[x])
        push!(merged, t)
    end
    Z = ITensor(1.0)
    for x in 1:Lx
        Z *= merged[x]
    end
    return scalar(Z)
end

# ⟨s_(x0,y0) s_(x0+1,y0)⟩ from the CTM environment
function ctm_bond_corr(Kx, Ky, Lx, Ly, x0, y0; χ = 24)
    hb, vb = bond_indices(Lx, Ly)
    rows = [make_row(Kx, Ky, y, Lx, Ly, hb, vb) for y in 1:Ly]
    U, D = top_env(rows, y0, χ), bot_env(rows, y0, χ)
    rowspin = make_row(Kx, Ky, y0, Lx, Ly, hb, vb; spins = Set([(x0, y0), (x0 + 1, y0)]))
    return contract_three(U, rowspin, D) / contract_three(U, rows[y0], D)
end

function brute_bond_corr(Lx, Ly, Kx, Ky, x0, y0)
    N = Lx * Ly; idx(x, y) = (y - 1) * Lx + x
    num = 0.0; den = 0.0; s = zeros(Int, N)
    for c in 0:(2^N - 1)
        for i in 1:N; s[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1; end
        E = 0.0
        for y in 1:Ly, x in 1:Lx
            x < Lx && (E += Kx * s[idx(x, y)] * s[idx(x + 1, y)])
            y < Ly && (E += Ky * s[idx(x, y)] * s[idx(x, y + 1)])
        end
        w = exp(E); den += w; num += w * s[idx(x0, y0)] * s[idx(x0 + 1, y0)]
    end
    return num / den
end

function main()
    @printf("%-10s %-10s %-8s %-18s %-18s %-10s\n",
        "grid", "(Kx,Ky)", "bond", "⟨sᵢsⱼ⟩ (CTM)", "⟨sᵢsⱼ⟩ (brute)", "|err|")
    println("-"^76)
    for (Lx, Ly, Kx, Ky, x0, y0) in
        [(5, 5, 0.4, 0.4, 2, 3), (5, 5, 0.3, 0.6, 3, 2), (5, 5, 0.3, 0.6, 1, 1)]
        c = ctm_bond_corr(Kx, Ky, Lx, Ly, x0, y0; χ = 24)
        b = brute_bond_corr(Lx, Ly, Kx, Ky, x0, y0)
        @printf("%-10s (%.1f,%.1f)  %-8s %-18.12f %-18.12f %-10.2e\n",
            "$(Lx)x$(Ly)", Kx, Ky, "($x0,$y0)", c, b, abs(c - b))
    end
end

main()
