# CVM (region-graph) free energy of a finite 2D tensor network, benchmarked on a random
# (non-symmetric) network against boundary MPS at matched bond dimension χ.
#
#   F = Σ_v ln Z_v  −  Σ_e ln Z_e  +  Σ_p ln Z_p          (Möbius numbers +1/−1/+1)
#
#   Z_v : vertex ring     = 4 corners + 4 edges + centre a
#   Z_e : edge strip      = 4 corners + 2 edges
#   Z_p : plaquette loop  = 4 corners
#
# Each region is keyed by a centre point (cx,cy): integer ⇒ vertex, half-integer in one
# axis ⇒ edge, both ⇒ plaquette. Corner blocks are truncated to χ by the corner-growth
# EIGENDECOMPOSITION move: eig of the corner block's reduced density matrix on each of its
# two interfaces gives the projectors, which also truncate the adjoining edges. Because
# V−E+P = 1 for a disk, every region contracts to the exact Z as χ→∞, so F → ln Z; at
# finite χ the region errors cancel in the Möbius sum (this is the CVM advantage).
#
# Run: julia --project=. --startup-file=no examples/ctm_cvm_random.jl

using TensorNetworkQuantumSimulator
using ITensors, LinearAlgebra, Printf, Random
using Dictionaries: Dictionary, set!
const TNQS = TensorNetworkQuantumSimulator

# top-χ eigenvector isometry of a Hermitian ρ on index `bnd`
function eig_isometry(ρ::ITensor, bnd::Index, χ::Int)
    ρm = Array(ρ, bnd, prime(bnd)); F = eigen(Hermitian((ρm + ρm') / 2))
    ord = sortperm(F.values; rev = true)
    k = min(χ, length(F.values)); w = Index(k)
    return ITensor(F.vectors[:, ord[1:k]], bnd, w), w
end

# projector for the `own` interface of a corner block, tracing its `oth` interface
function iface_proj(B, own, oth, χ)
    (isempty(own) || isnothing(B)) && return nothing
    co = combiner(own...); io = combinedind(co); Bc = B * co
    ρ = Bc * prime(dag(Bc), io)
    P, _ = eig_isometry(ρ, io, χ)
    return P * co        # legs: own..., w
end

# ln of the region centred at (cx,cy) with corner bond dim χ
function region_lnZ(A, Lx, Ly, cx, cy, χ)
    blk(pred) = (s = [A[(x, y)] for x in 1:Lx, y in 1:Ly if pred(x, y)];
                 isempty(s) ? nothing : reduce(*, s))
    els = Dict(
        :NW => blk((x, y) -> x < cx && y < cy), :NE => blk((x, y) -> x > cx && y < cy),
        :SW => blk((x, y) -> x < cx && y > cy), :SE => blk((x, y) -> x > cx && y > cy),
        :N  => blk((x, y) -> x == cx && y < cy), :S => blk((x, y) -> x == cx && y > cy),
        :W  => blk((x, y) -> x < cx && y == cy), :E => blk((x, y) -> x > cx && y == cy),
        :C  => blk((x, y) -> x == cx && y == cy))
    order = [:NW, :N, :NE, :E, :SE, :S, :SW, :W]
    ring = [(t, els[t]) for t in order if !isnothing(els[t])]
    n = length(ring); corners = (:NW, :NE, :SE, :SW)
    trunc = Dict{Symbol,ITensor}(t => T for (t, T) in ring)
    for i in 1:n                                    # truncate each ring interface
        (ti, Ti) = ring[i]; (tj, Tj) = ring[mod1(i + 1, n)]
        s = commoninds(Ti, Tj); isempty(s) && continue
        if ti in corners; src, srcT, othT = ti, Ti, ring[mod1(i - 1, n)][2]
        else;             src, srcT, othT = tj, Tj, ring[mod1(i + 2, n)][2]; end
        P = iface_proj(srcT, s, commoninds(srcT, othT), χ); isnothing(P) && continue
        trunc[ti] *= (src == ti ? P : dag(P)); trunc[tj] *= (src == tj ? P : dag(P))
    end
    ts = ITensor[trunc[t] for (t, _) in ring]
    !isnothing(els[:C]) && push!(ts, els[:C])
    return log(real(scalar(reduce(*, ts))))
end

function cvm_lnZ(A, Lx, Ly, χ)
    F = 0.0
    for x in 1:Lx,   y in 1:Ly;     F += region_lnZ(A, Lx, Ly, x, y, χ);         end  # +1
    for x in 1:Lx-1, y in 1:Ly;     F -= region_lnZ(A, Lx, Ly, x + 0.5, y, χ);   end  # −1
    for x in 1:Lx,   y in 1:Ly-1;   F -= region_lnZ(A, Lx, Ly, x, y + 0.5, χ);   end  # −1
    for x in 1:Lx-1, y in 1:Ly-1;   F += region_lnZ(A, Lx, Ly, x + 0.5, y + 0.5, χ); end  # +1
    return F
end

# random POSITIVE network (region free energies real) sharing tensors with the library
function positive_network(g, D; seed)
    Random.seed!(seed)
    ψ0 = random_tensornetwork(g; bond_dimension = D)
    tensors = Dictionary{vertextype(g),ITensor}()
    for v in vertices(g); set!(tensors, v, abs.(ψ0[v])); end
    return TensorNetwork(tensors, g)
end

function main()
    L = 4
    for D in (2, 3)
        g = named_grid((L, L))
        ψ = positive_network(g, D; seed = 7)
        A = Dict((x, y) => ψ[(x, y)] for x in 1:L, y in 1:L)
        zex = log(real(contract(ψ; alg = "exact")))
        @printf("\nrandom positive %dx%d, D=%d  (ln Z_exact = %.8f)\n", L, L, D, zex)
        @printf("%-5s  %-16s  %-16s\n", "χ", "CVM |Δln Z|", "bMPS |Δln Z|")
        for χ in (2, 3, 4, 6, 8)
            ec = abs(cvm_lnZ(A, L, L, χ) - zex)
            eb = abs(log(real(contract(ψ; alg = "boundarymps", mps_bond_dimension = χ))) - zex)
            @printf("%-5d  %-16.3e  %-16.3e\n", χ, ec, eb)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
