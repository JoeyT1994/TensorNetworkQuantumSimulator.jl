# Genuine 4-directional finite CTMRG.
#
# Shrink the bulk by ALTERNATING up/down/left/right absorption moves, so the four
# corners accumulate 2D quadrants (not a 1D strip). Each move truncates the growing
# frame edge to χ using the accumulated corner density matrix (EIGENDECOMPOSITION).
# This is the corner-transfer truncation that makes Z_v stationary — the corner
# spectrum, which decays faster than a boundary-MPS Schmidt cut. Test vs exact + bMPS.
#
# Run: julia --project=. --startup-file=no examples/ctm_4directional.jl

using TensorNetworkQuantumSimulator
using ITensors
using Dictionaries: Dictionary
using LinearAlgebra
using Printf
const TNQS = TensorNetworkQuantumSimulator

function eig_projector(ρ::ITensor, bnd::Index, χ::Int)
    bp = prime(bnd); ρm = Array(ρ, bnd, bp)
    F = eigen(Hermitian((ρm + ρm') / 2))
    order = sortperm(F.values; rev = true)
    k = min(χ, length(F.values)); idx = order[1:k]; w = Index(k)
    return ITensor(F.vectors[:, idx], bnd, w), w, F.values[idx]
end

# Truncate a frame-edge chain (2D corners at the ends) to bond dim χ using the
# accumulated CORNER density matrix, eig. Keeping the corner spectrum (not the boundary
# Schmidt spectrum, which is what SVD/bMPS keeps) is the corner-transfer truncation.
function truncate_chain!(chain::Vector{ITensor}, χ::Int)
    n = length(chain); n <= 1 && return chain
    for i in 1:(n - 1)                                    # combine doubled bonds
        sh = commoninds(chain[i], chain[i + 1])
        if length(sh) > 1
            C = combiner(sh...); chain[i] = chain[i] * C; chain[i + 1] = chain[i + 1] * C
        end
    end
    ρL = ITensor(1.0)                                     # accumulated corner density matrix
    for i in 1:(n - 1)
        bnd = commonind(chain[i], chain[i + 1])
        M = chain[i]; lb = commonind(M, ρL)
        Mp = isnothing(lb) ? prime(dag(M), bnd) : prime(dag(M), bnd, lb)
        ρext = ρL * M * Mp
        P, w, λ = eig_projector(ρext, bnd, χ)
        chain[i] = M * P; chain[i + 1] = chain[i + 1] * dag(P)
        ρL = ITensor(diagm(λ), w, prime(w))
    end
    return chain
end

mutable struct CTMFrame
    A::Dict{Tuple{Int,Int},ITensor}
    t::Int; b::Int; l::Int; r::Int
    ctl::ITensor; ctr::ITensor; cbl::ITensor; cbr::ITensor
    tt::Dict{Int,ITensor}; tb::Dict{Int,ITensor}
    tl::Dict{Int,ITensor}; tr::Dict{Int,ITensor}
end

function init_frame(A, Lx, Ly)
    CTMFrame(A, 2, Ly - 1, 2, Lx - 1,
        A[(1, 1)], A[(Lx, 1)], A[(1, Ly)], A[(Lx, Ly)],
        Dict(x => A[(x, 1)] for x in 2:(Lx - 1)), Dict(x => A[(x, Ly)] for x in 2:(Lx - 1)),
        Dict(y => A[(1, y)] for y in 2:(Ly - 1)), Dict(y => A[(Lx, y)] for y in 2:(Ly - 1)))
end

function absorb_up!(F, χ)
    t = F.t
    for x in F.l:F.r; F.tt[x] = F.tt[x] * F.A[(x, t)]; end
    F.ctl = F.ctl * F.tl[t]; F.ctr = F.ctr * F.tr[t]
    chain = ITensor[F.ctl; [F.tt[x] for x in F.l:F.r]; F.ctr]
    truncate_chain!(chain, χ)
    F.ctl = chain[1]; F.ctr = chain[end]
    for (i, x) in enumerate(F.l:F.r); F.tt[x] = chain[i + 1]; end
    delete!(F.tl, t); delete!(F.tr, t); F.t += 1
end

function absorb_down!(F, χ)
    b = F.b
    for x in F.l:F.r; F.tb[x] = F.tb[x] * F.A[(x, b)]; end
    F.cbl = F.cbl * F.tl[b]; F.cbr = F.cbr * F.tr[b]
    chain = ITensor[F.cbl; [F.tb[x] for x in F.l:F.r]; F.cbr]
    truncate_chain!(chain, χ)
    F.cbl = chain[1]; F.cbr = chain[end]
    for (i, x) in enumerate(F.l:F.r); F.tb[x] = chain[i + 1]; end
    delete!(F.tl, b); delete!(F.tr, b); F.b -= 1
end

function absorb_left!(F, χ)
    l = F.l
    for y in F.t:F.b; F.tl[y] = F.tl[y] * F.A[(l, y)]; end
    F.ctl = F.ctl * F.tt[l]; F.cbl = F.cbl * F.tb[l]
    chain = ITensor[F.ctl; [F.tl[y] for y in F.t:F.b]; F.cbl]
    truncate_chain!(chain, χ)
    F.ctl = chain[1]; F.cbl = chain[end]
    for (i, y) in enumerate(F.t:F.b); F.tl[y] = chain[i + 1]; end
    delete!(F.tt, l); delete!(F.tb, l); F.l += 1
end

function absorb_right!(F, χ)
    r = F.r
    for y in F.t:F.b; F.tr[y] = F.tr[y] * F.A[(r, y)]; end
    F.ctr = F.ctr * F.tt[r]; F.cbr = F.cbr * F.tb[r]
    chain = ITensor[F.ctr; [F.tr[y] for y in F.t:F.b]; F.cbr]
    truncate_chain!(chain, χ)
    F.ctr = chain[1]; F.cbr = chain[end]
    for (i, y) in enumerate(F.t:F.b); F.tr[y] = chain[i + 1]; end
    delete!(F.tt, r); delete!(F.tb, r); F.r -= 1
end

function ctm_lnZ(A, Lx, Ly, χ)
    F = init_frame(A, Lx, Ly)
    while F.t < F.b || F.l < F.r
        F.t < F.b && absorb_up!(F, χ)
        F.t < F.b && absorb_down!(F, χ)
        F.l < F.r && absorb_left!(F, χ)
        F.l < F.r && absorb_right!(F, χ)
    end
    # bulk is 1×1 at (F.l, F.t): contract the ring + centre
    ts = ITensor[F.ctl, F.tt[F.l], F.ctr, F.tr[F.t], F.cbr, F.tb[F.l], F.cbl, F.tl[F.t], F.A[(F.l, F.t)]]
    return log(real(scalar(reduce(*, ts))))
end

function main()
    L, K = 12, 0.44068679350977147   # exact critical point
    g = named_grid((L, L)); es = collect(edges(g))
    tn = ising_partitionfunction(g, 1.0; Js = Dictionary(es, [K for _ in es]))
    A = Dict((x, y) => tn[(x, y)] for x in 1:L, y in 1:L)
    lz_exact = log(real(contract(tn; alg = "exact")))
    @printf("Critical Ising %dx%d  (ln Z_exact = %.8f)\n", L, L, lz_exact)
    @printf("%-6s %-18s %-18s\n", "χ", "4-dir CTMRG", "boundary MPS")
    for χ in (2, 4, 6, 8, 12, 16)
        lz_ctm = ctm_lnZ(A, L, L, χ)
        lz_bmps = log(real(partitionfunction(update(BoundaryMPSCache(tn, χ)))))
        @printf("%-6d %-18.3e %-18.3e\n", χ, abs(lz_ctm - lz_exact), abs(lz_bmps - lz_exact))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
