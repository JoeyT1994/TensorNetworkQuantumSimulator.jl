# Demo of the finite CTMRG engine, `CTMEnvironmentCache`.
#
# There is ONE engine (src/MessagePassing/ctmenvironmentcache.jl); everything here is a
# call into it, so there is no duplicated truncation machinery to drift out of sync.
# It handles single-layer partition functions (isotropic or anisotropic, square or not),
# double-layer state norms ⟨ψ|ψ⟩ and forms ⟨ψ|O|ψ⟩, exposes row environments for row-local
# observables, and builds the per-vertex CVM regions whose Möbius sum gives the free energy.
#
# Run: julia --project=. --startup-file=no examples/ctm_environment.jl

using TensorNetworkQuantumSimulator
using ITensors
using Dictionaries: Dictionary
using Printf, Random
const TNQS = TensorNetworkQuantumSimulator

lnerr(z, ref) = abs(log(abs(real(z))) - log(abs(real(ref))))

# --- 1. classical partition function, isotropic and anisotropic ------------------
function ising_free_energy(; L = 6, K = 0.44, aniso = false)
    g = named_grid((L, L))
    Js = Dictionary(collect(edges(g)),
                    [aniso && src(e)[2] == dst(e)[2] ? 0.3 : K for e in edges(g)])
    tn = ising_partitionfunction(g, 1.0; Js)
    ref = contract(tn; alg = "exact")
    @printf("\nIsing %dx%d %s (ln Z_exact = %.10f)\n", L, L,
            aniso ? "anisotropic" : "isotropic", log(real(ref)))
    for χ in (4, 8, 16)
        @printf("   χ=%-3d |Δ ln Z| = %.3e\n", χ,
                lnerr(partitionfunction(CTMEnvironmentCache(tn, χ)), ref))
    end
end

# --- 2. double-layer state norm ⟨ψ|ψ⟩ (kept lazy: never folds ket⊗bra) -----------
function peps_norm(; L = 4, D = 2, seed = 3)
    Random.seed!(seed)
    ψ = random_tensornetworkstate(Float64, named_grid((L, L)); bond_dimension = D)
    ref = norm_sqr(ψ; alg = "exact")
    @printf("\nPEPS norm %dx%d D=%d (ln⟨ψ|ψ⟩_exact = %.10f)\n", L, L, D, log(abs(real(ref))))
    for χ in (4, 9, 16)
        @printf("   χ=%-3d |Δ ln N| = %.3e\n", χ,
                lnerr(partitionfunction(CTMEnvironmentCache(ψ, χ)), ref))
    end
end

# --- 3. row environments: sandwiching row y must reproduce Z ---------------------
function row_environment_check(; L = 5, K = 0.4, χ = 16)
    g = named_grid((L, L))
    Js = Dictionary(collect(edges(g)), [K for _ in edges(g)])
    tn = ising_partitionfunction(g, 1.0; Js)
    ref = contract(tn; alg = "exact")
    cache = CTMEnvironmentCache(tn, χ)
    @printf("\nRow-environment sandwich (Ising %dx%d, χ=%d)\n", L, L, χ)
    for y in 1:L
        top, bot = row_environments(cache, y)
        rowts = ITensor[tn[v] for v in cache.rows[y]]
        @printf("   row %d: |Δ ln Z| = %.3e\n", y, lnerr(contract_row(top, rowts, bot), ref))
    end
end

# --- 4. per-vertex CVM regions: F = Σ_v ln Z_v − Σ_e ln Z_e + Σ_p ln Z_p ---------
# A 4C+4T ring on every vertex, with each interface truncated by a two-sided
# (biorthogonal) projector. The projector needs the complement environment, so the build is a
# fixed-point iteration: `update(cache)` sweeps it to stationarity and returns a cache you
# read `cvm_freenergy` off. An UN-updated cache falls back to the greedy one-sided pass —
# 3-4 orders worse and non-monotone in χ — which is the "greedy" column here. Compared
# against boundary MPS at matched χ on a random NON-SYMMETRIC network, since a symmetric
# Ising model can be passed by accident via symmetry crutches.
function cvm_free_energy(; L = 4, D = 3, seed = 5, χs = (4, 6, 8, 12))
    Random.seed!(seed)
    tn = random_tensornetwork(Float64, named_grid((L, L)); bond_dimension = D)
    ref = contract(tn; alg = "exact")
    lnZ = log(abs(real(ref)))
    @printf("\nCVM regions vs boundary MPS (random %dx%d D=%d, ln|Z| = %.10f)\n", L, L, D, lnZ)
    @printf("   %-4s %-11s %-11s %-11s\n", "χ", "greedy", "swept", "boundaryMPS")
    for χ in χs
        cache = CTMEnvironmentCache(tn, χ)
        greedy = abs(cvm_freenergy(cache) - lnZ)                  # un-updated: greedy pass
        swept = abs(cvm_freenergy(update(cache)) - lnZ)           # two-sided, to stationarity
        bmps = lnerr(contract(tn; alg = "boundarymps", mps_bond_dimension = χ), ref)
        @printf("   %-4d %-11.3e %-11.3e %-11.3e\n", χ, greedy, swept, bmps)
    end
end

function main()
    ising_free_energy()
    ising_free_energy(; aniso = true)
    peps_norm()
    peps_norm(; D = 3)
    row_environment_check()
    cvm_free_energy()
end

main()
