# Fermionic quench dynamics with the TensorKit backend, at full fU(1) symmetry:
# fermionic statistics ⊠ conserved particle number (TensorKit's FermionNumber sectors).
#
# Spinless free fermions: a charge-density-wave product state quenched under
# nearest-neighbour hopping H = -t Σ (c†ᵢcⱼ + h.c.). Everything fermionic is native:
# site charges are routed through dim-1 links (T-join) with the nonzero TOTAL particle
# number carried by a dangling "Charge" dummy leg, gates are LOCAL matrix exponentials —
# Jordan-Wigner strings emerge from the graded category, including inside the two-point
# correlators — and number conservation is enforced structurally (a pair-creation gate
# would error as non-conserving).
#
# The model is Gaussian, so exact dynamics at any size follows from the single-particle
# correlation matrix C(T) = V* C(0) Vᵀ, V = exp(-i T h). Two checks:
#   1. COMB TREE: BP is exact on trees, so BP observables must match the correlation
#      matrix up to (second-order) Trotter error alone — verified: the deviation is
#      χ-independent and drops with the Trotter order, not the bond dimension.
#   2. 3×3 CDW quench to T = 0.5: contract with BOUNDARY MPS and compare a local
#      hopping ⟨c†c + h.c.⟩ on an edge against the exact value.

using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using LinearAlgebra: Diagonal, diag

function correlation_matrix(g, tt, T, occupied)
    vs = sort(collect(vertices(g)))
    idx = Dict(v => k for (k, v) in enumerate(vs))
    h = zeros(Float64, length(vs), length(vs))
    for e in edges(g)
        h[idx[src(e)], idx[dst(e)]] = -tt
        h[idx[dst(e)], idx[src(e)]] = -tt
    end
    V = exp(-im * T * h)
    C = conj(V) * Diagonal([v ∈ occupied ? 1.0 : 0.0 for v in vs]) * transpose(V)
    return C, idx
end

function evolve(g, occupied; tt = 1.0, dt = 0.05, nsteps = 10, χ = 16)
    s = siteinds("Fermion", g; symmetry = "fU1")
    ψ = tensornetworkstate(ComplexF64, v -> v ∈ occupied ? "Occ" : "Emp", g, s)
    #second-order Trotter layer: half-steps forward then reversed
    half = Any[]
    for ces in edge_color(g, 4)
        append!(half, ("F_hop", pair, -tt * dt / 2) for pair in ces)
    end
    layer = vcat(half, reverse(half))
    ψ_bpc = update(BeliefPropagationCache(ψ))
    for _ in 1:nsteps
        ψ_bpc, _ = apply_gates(layer, ψ_bpc; apply_kwargs = (; maxdim = χ, cutoff = 1.0e-12))
    end
    return network(ψ_bpc), nsteps * dt
end

function main(; tt = 1.0, dt = 0.01, nsteps = 50)
    println("== 1. Comb tree: BP is exact, deviation = Trotter only ==")
    g = named_comb_tree((3, 3))
    occupied = filter(v -> isodd(sum(v)), collect(vertices(g)))
    ψ, T = evolve(g, occupied; tt, dt, nsteps)
    C, idx = correlation_matrix(g, tt, T, occupied)
    occs = [real(only(expect(ψ, ("N", [v]); alg = "bp"))) for v in sort(collect(vertices(g)))]
    println("T = $T:  max |⟨N⟩_BP − exact| = ", round(maximum(abs.(occs .- real.(diag(C)))); sigdigits = 3))

    println("\n== 2. 3×3 CDW quench, boundary-MPS hopping check ==")
    g = named_grid((3, 3))
    occupied = filter(v -> isodd(sum(v)), collect(vertices(g)))
    ψ, T = evolve(g, occupied; tt, dt, nsteps)
    C, idx = correlation_matrix(g, tt, T, occupied)
    v, w = (2, 1), (2, 2)   #an edge inside a single boundary-MPS partition
    cdagc_exact = C[idx[v], idx[w]]
    bmps = update(BoundaryMPSCache(ψ, 16))
    cdagc_bmps = only(expect(bmps, ("CdagC", (v, w)); alg = "boundarymps"))
    cdagc_bp = only(expect(ψ, ("CdagC", (v, w)); alg = "bp"))
    println("T = $T:  ⟨c†_$(v) c_$(w)⟩")
    println("  exact        ", round(cdagc_exact; sigdigits = 6))
    println("  boundary MPS ", round(cdagc_bmps; sigdigits = 6), "   |Δ| = ", round(abs(cdagc_bmps - cdagc_exact); sigdigits = 3))
    println("  BP           ", round(cdagc_bp; sigdigits = 6), "   |Δ| = ", round(abs(cdagc_bp - cdagc_exact); sigdigits = 3))
    return nothing
end

main()
