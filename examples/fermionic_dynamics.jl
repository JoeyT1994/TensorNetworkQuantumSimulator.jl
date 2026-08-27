# Fermionic quench dynamics on a square lattice with the TensorKit (fZ2) backend.
#
# Spinless free fermions: a charge-density-wave product state |1010...⟩ quenched under
# nearest-neighbour hopping H = -t Σ (c†ᵢcⱼ + h.c.). Everything fermionic is native:
# sites are Vect[FermionParity] spaces, the initial state's odd site charges are routed
# through dim-1 links (T-join), gates are LOCAL matrix exponentials — Jordan-Wigner
# strings emerge from TensorKit's graded category, including inside the two-point
# correlators ⟨c†_v c_w⟩ measured at the end.
#
# Since the model is Gaussian, the exact dynamics at ANY size follows from the
# single-particle correlation matrix: C(T) = V* C(0) Vᵀ with V = exp(-i T h). The BP
# columns should track it up to Trotter + BP-loop error.

using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using LinearAlgebra: Diagonal, diag, norm

function exact_correlation_matrix(g, tt, T, occupied)
    vs = sort(collect(vertices(g)))
    idx = Dict(v => k for (k, v) in enumerate(vs))
    h = zeros(Float64, length(vs), length(vs))
    for e in edges(g)
        h[idx[src(e)], idx[dst(e)]] = -tt
        h[idx[dst(e)], idx[src(e)]] = -tt
    end
    C0 = Diagonal([v ∈ occupied ? 1.0 : 0.0 for v in vs])
    V = exp(-im * T * h)
    return conj(V) * C0 * transpose(V), vs, idx
end

function main(; L = 4, tt = 1.0, dt = 0.05, nsteps = 10, χ = 8)
    g = named_grid((L, L))
    s = siteinds("Fermion", g)
    occupied = filter(v -> isodd(sum(v)), collect(vertices(g)))
    ψ = tensornetworkstate(ComplexF64, v -> v ∈ occupied ? "Occ" : "Emp", g, s; charge_leg = true)

    layer = Any[]
    for ces in edge_color(g, 4)
        #exp(-i dt (-t) (c†c + h.c.)) per edge — the LOCAL matrix; no strings anywhere
        append!(layer, ("F_hop", pair, -tt * dt) for pair in ces)
    end

    apply_kwargs = (; maxdim = χ, cutoff = 1.0e-12)
    for step in 1:nsteps
        ψ, errs = apply_gates(layer, ψ; apply_kwargs)
        step % 5 == 0 && println("step $step: max gate err $(round(maximum(errs); sigdigits = 3))")
    end

    T = nsteps * dt
    C, vs, idx = exact_correlation_matrix(g, tt, T, occupied)

    #site occupations: CDW order parameter
    occs = [real(only(expect(ψ, ("N", [v]); alg = "bp"))) for v in vs]
    occs_exact = real.(diag(C))
    cdw = sum(v -> (isodd(sum(v)) ? 1 : -1) * occs[idx[v]], vs) / length(vs)
    cdw_exact = sum(v -> (isodd(sum(v)) ? 1 : -1) * occs_exact[idx[v]], vs) / length(vs)
    println("\nT = $T")
    println("CDW order:   BP $(round(cdw; sigdigits = 6))   exact $(round(cdw_exact; sigdigits = 6))")
    println("max |Δ⟨N⟩|:  ", round(maximum(abs.(occs .- occs_exact)); sigdigits = 3))

    #two-point functions ⟨c†_v c_w⟩ along a row (odd-pair joint operators at distance)
    v0 = (1, 1)
    println("\n⟨c†_{(1,1)} c_{(1,j)}⟩:")
    for j in 2:L
        w = (1, j)
        tn = only(expect(ψ, ("CdagC", (v0, w)); alg = "bp"))
        ex = C[idx[v0], idx[w]]
        println("  j = $j:  BP $(round(tn; sigdigits = 5))   exact $(round(ex; sigdigits = 5))   |Δ| $(round(abs(tn - ex); sigdigits = 3))")
    end
    return ψ
end

main()
