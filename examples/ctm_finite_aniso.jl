# Full finite anisotropic CTMRG.
#
# Contract a finite Lx×Ly anisotropic Ising lattice (Kx horizontal, Ky vertical, free
# boundaries) by directional absorption with EIGENVALUE full projectors — the corner
# density matrix ρ = M M† diagonalised, keep χ. Validate ln Z against brute force.
#
# Directional move (absorb the top row of the remaining block into the top boundary):
#   1. grow each top tensor by the site below it (horizontal bonds → χ·d)
#   2. for every grown horizontal bond, build the reduced density matrix from the
#      accumulated LEFT half, eig, keep χ  → isometry P  (full projector, both halves)
#   3. renormalise with P.  The two ends of the boundary are the corners.
# Rotating the block reuses one move for all four directions; absorb to a single tensor.
#
# Run: julia --project=. --startup-file=no examples/ctm_finite_aniso.jl

using ITensors
using LinearAlgebra
using Printf

sqrtW(K) = sqrt(Symmetric([exp(K * s1 * s2) for s1 in (1.0, -1.0), s2 in (1.0, -1.0)]))

# Site (x,y): 4-leg (u,r,d,l); off-lattice legs capped by b=W⁻¹·1 (→ dim 1).
# Vertical legs use Wy, horizontal use Wx.
function site_array(Kx, Ky, x, y, Lx, Ly)
    Wx, Wy = sqrtW(Kx), sqrtW(Ky); bx, by = Wx \ ones(2), Wy \ ones(2)
    hN, hE, hS, hW = y > 1, x < Lx, y < Ly, x > 1
    du, dr, dd, dl = hN ? 2 : 1, hE ? 2 : 1, hS ? 2 : 1, hW ? 2 : 1
    t = zeros(Float64, du, dr, dd, dl)
    for u in 1:du, r in 1:dr, dn in 1:dd, l in 1:dl
        acc = 0.0
        for s in 1:2
            wu = hN ? Wy[u, s] : by' * Wy[:, s]
            wr = hE ? Wx[r, s] : bx' * Wx[:, s]
            wd = hS ? Wy[dn, s] : by' * Wy[:, s]
            wl = hW ? Wx[l, s] : bx' * Wx[:, s]
            acc += wu * wr * wd * wl
        end
        t[u, r, dn, l] = acc
    end
    return t
end

# eig projector from a density matrix ρ(bnd, bnd'): keep χ dominant eigenvectors.
function eig_projector(ρ::ITensor, bnd::Index, χ::Int)
    ρm = Array(ρ, bnd, bnd')
    F = eigen(Hermitian((ρm + ρm') / 2))
    keep = sortperm(F.values; rev = true)[1:min(χ, length(F.values))]
    w = Index(length(keep), "χ")
    return ITensor(F.vectors[:, keep], bnd, w), w, F.values[keep]
end

# Build the row of ITensors for a given y; horizontal bonds shared with neighbours.
function build_rows(Kx, Ky, Lx, Ly)
    hb = Dict{Tuple{Int,Int},Index}()                 # hbond[(x,y)] between (x,y),(x+1,y)
    vb = Dict{Tuple{Int,Int},Index}()                 # vbond[(x,y)] between (x,y),(x,y+1)
    for x in 1:Lx, y in 1:Ly
        x < Lx && (hb[(x, y)] = Index(2, "h$(x)_$(y)"))
        y < Ly && (vb[(x, y)] = Index(2, "v$(x)_$(y)"))
    end
    rows = Vector{Vector{ITensor}}(undef, Ly)
    for y in 1:Ly
        row = ITensor[]
        for x in 1:Lx
            legs = Index[]
            y > 1  && push!(legs, vb[(x, y - 1)]); x < Lx && push!(legs, hb[(x, y)])
            y < Ly && push!(legs, vb[(x, y)]);     x > 1  && push!(legs, hb[(x - 1, y)])
            arr = site_array(Kx, Ky, x, y, Lx, Ly)
            push!(row, ITensor(reshape(arr, filter(!=(1), size(arr))...), legs...))
        end
        rows[y] = row
    end
    return rows, hb, vb
end

# Absorb `top` row into `next` row (contract vertical bonds), then truncate the
# horizontal bonds to χ using the ACCUMULATED left-block (corner) density matrix, eig.
# The accumulated left block IS the corner; ρL grows as we sweep left→right.
function absorb_and_truncate(top::Vector{ITensor}, next::Vector{ITensor}, χ::Int)
    Lx = length(top)
    merged = ITensor[top[x] * next[x] for x in 1:Lx]         # contract shared vbond
    # merge the doubled horizontal bonds (top's + next's) between adjacent tensors
    for x in 1:(Lx - 1)
        shared = commoninds(merged[x], merged[x + 1])
        if length(shared) > 1
            C = combiner(shared...)
            merged[x]     = merged[x] * C
            merged[x + 1] = merged[x + 1] * C
        end
    end
    ρL = ITensor(1.0)                                        # left-block density matrix
    for x in 1:(Lx - 1)
        bnd = commonind(merged[x], merged[x + 1])            # bond to truncate
        M = merged[x]
        lb = commonind(M, ρL)                                # left bond (nothing at x=1)
        toprime = isnothing(lb) ? (bnd,) : (bnd, lb)
        Mp = prime(dag(M), toprime...)                       # prime bra: bnd and left bond
        ρext = ρL * M * Mp                                   # trace left bond + down-bulk → (bnd,bnd')
        P, w, λ = eig_projector(ρext, bnd, χ)
        merged[x]     = M * P
        merged[x + 1] = merged[x + 1] * dag(P)
        ρL = ITensor(diagm(λ), w, w')                        # truncated corner density mat
    end
    return merged
end

function ctmrg_lnZ(Kx, Ky, Lx, Ly; χ = 32)
    rows, hb, vb = build_rows(Kx, Ky, Lx, Ly)
    cur = rows[1]
    for y in 2:Ly
        cur = absorb_and_truncate(cur, rows[y], χ)
    end
    Z = ITensor(1.0)
    for x in 1:Lx
        Z *= cur[x]
    end
    return log(scalar(Z))
end

function brute_lnZ(Lx, Ly, Kx, Ky)
    N = Lx * Ly; idx(x, y) = (y - 1) * Lx + x; tot = 0.0; s = zeros(Int, N)
    for c in 0:(2^N - 1)
        for i in 1:N; s[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1; end
        E = 0
        for y in 1:Ly, x in 1:Lx
            x < Lx && (E += Kx * s[idx(x, y)] * s[idx(x + 1, y)])
            y < Ly && (E += Ky * s[idx(x, y)] * s[idx(x, y + 1)])
        end
        tot += exp(E)
    end
    return log(tot)
end

function main()
    @printf("%-8s %-10s %-18s %-18s %-10s\n", "grid", "(Kx,Ky)", "lnZ (CTMRG)", "lnZ (brute)", "|err|")
    println("-"^68)
    for (Lx, Ly, Kx, Ky) in [(3, 3, 0.4, 0.4), (3, 3, 0.3, 0.6), (4, 3, 0.3, 0.6)]
        c = ctmrg_lnZ(Kx, Ky, Lx, Ly; χ = 32)
        b = brute_lnZ(Lx, Ly, Kx, Ky)
        @printf("%-8s (%.1f,%.1f)  %-18.12f %-18.12f %-10.2e\n",
            "$(Lx)x$(Ly)", Kx, Ky, c, b, abs(c - b))
    end
end

import TensorNetworkQuantumSimulator as TNQS
using Dictionaries: Dictionary

# χ-convergence vs the library's exact contraction, near-critical so truncation bites.
function main_convergence()
    for (Lx, Ly, Kx, Ky) in [(10, 10, 0.44, 0.44), (10, 10, 0.30, 0.62)]
        g = TNQS.named_grid((Lx, Ly))
        es = collect(TNQS.edges(g))
        Js = Dictionary(es, [(TNQS.src(e)[2] == TNQS.dst(e)[2]) ? Kx : Ky for e in es])
        tn = TNQS.ising_partitionfunction(g, 1.0; Js)
        lz_exact = log(TNQS.contract(tn; alg = "exact"))
        @printf("\nχ-convergence: %dx%d (Kx=%.2f, Ky=%.2f, near-crit), exact lnZ = %.12f\n",
            Lx, Ly, Kx, Ky, real(lz_exact))
        @printf("%-6s %-18s %-10s\n", "χ", "lnZ (CTMRG)", "|err|")
        println("-"^36)
        for χ in (2, 4, 8, 16, 24)
            c = ctmrg_lnZ(Kx, Ky, Lx, Ly; χ)
            @printf("%-6d %-18.12f %-10.2e\n", χ, c, abs(c - real(lz_exact)))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
    main_convergence()
end
