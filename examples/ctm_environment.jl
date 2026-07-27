# Demo of the finite CTMRG engine, `CTMEnvironmentCache`.
#
# There is ONE engine (src/MessagePassing/ctmenvironmentcache.jl); everything here is a call
# into it. It is position-resolved CTMRG framed as a region-graph (CVM) free energy — a 4C+4T
# ring on EVERY vertex, grown and projected by local corner moves. No row absorption, no
# whole-lattice chain: this is meant to supersede boundary MPS, not reuse it.
#
#   cache = update(CTMEnvironmentCache(net, χ))   # two-sided sweep to stationarity
#   cvm_freenergy(cache)                          # F = Σ_v lnZ_v − Σ_e lnZ_e + Σ_p lnZ_p
#   expect(cache, ("Z", v))                       # single-site observable from v's own ring
#
# Handles single-layer partition functions (isotropic or anisotropic, square or not),
# double-layer state norms ⟨ψ|ψ⟩ and forms ⟨ψ|O|ψ⟩.
#
# Run: julia --project=. --startup-file=no examples/ctm_environment.jl

using TensorNetworkQuantumSimulator
using ITensors
using Dictionaries: Dictionary
using Printf, Random
const TNQS = TensorNetworkQuantumSimulator

# --- 1. classical partition function, isotropic and anisotropic ------------------
function ising_free_energy(; L = 6, K = 0.44, aniso = false)
    g = named_grid((L, L))
    Js = Dictionary(collect(edges(g)),
                    [aniso && src(e)[2] == dst(e)[2] ? 0.3 : K for e in edges(g)])
    tn = ising_partitionfunction(g, 1.0; Js)
    lnZ = log(real(contract(tn; alg = "exact")))
    @printf("\nIsing %dx%d %s (ln Z_exact = %.10f)\n", L, L,
            aniso ? "anisotropic" : "isotropic", lnZ)
    for χ in (4, 8, 16)
        F = cvm_freenergy(update(CTMEnvironmentCache(tn, χ)))
        @printf("   χ=%-3d |Δ ln Z| = %.3e\n", χ, abs(F - lnZ))
    end
end

# --- 2. double-layer state norm ⟨ψ|ψ⟩ (kept lazy: never folds ket⊗bra) -----------
function peps_norm(; L = 4, D = 2, seed = 3)
    Random.seed!(seed)
    ψ = random_tensornetworkstate(Float64, named_grid((L, L)); bond_dimension = D)
    lnN = log(abs(real(norm_sqr(ψ; alg = "exact"))))
    @printf("\nPEPS norm %dx%d D=%d (ln⟨ψ|ψ⟩_exact = %.10f)\n", L, L, D, lnN)
    for χ in (4, 9, 16)
        F = cvm_freenergy(update(CTMEnvironmentCache(ψ, χ)))
        @printf("   χ=%-3d |Δ ln N| = %.3e\n", χ, abs(F - lnN))
    end
end

# --- 3. single-site observables from each vertex's own ring ----------------------
# `update` gives every vertex a 4C+4T ring; `expect` contracts `vertex_ring(cache, v)` with
# v's ket/op/bra factors. Position-resolved, so boundary and corner sites work too.
function single_site_observables(; L = 4, D = 2, seed = 42, χs = (2, 4, 8, 16))
    Random.seed!(seed)
    g = named_grid((L, L))
    s = siteinds("S=1/2", g)
    ψ = random_tensornetworkstate(Float64, g, s; bond_dimension = D)
    vs = [(1, 1), (2, 1), (2, 2), (L, L)]
    ex = Dict(v => expect(ψ, ("Z", [v]); alg = "exact") for v in vs)

    @printf("\n⟨Z_v⟩ error vs exact (random %dx%d D=%d)\n", L, L, D)
    @printf("   %-8s %-12s %-10s", "vertex", "exact", "bp")
    for χ in χs; @printf(" %-10s", "ctm χ=$χ"); end
    println()
    caches = Dict(χ => update(CTMEnvironmentCache(ψ, χ)) for χ in χs)
    for v in vs
        bp = expect(ψ, ("Z", [v]); alg = "bp")
        @printf("   %-8s %-12.8f %-10.2e", string(v), ex[v], abs(bp - ex[v]))
        for χ in χs
            @printf(" %-10.2e", abs(expect(caches[χ], ("Z", [v])) - ex[v]))
        end
        println()
    end
end

# --- 4. CVM vs boundary MPS at matched χ ----------------------------------------
# A random NON-SYMMETRIC network: a symmetric Ising model can be passed by accident via
# symmetry crutches. `greedy` is an un-updated cache, i.e. the one-sided single pass.
function cvm_vs_boundarymps(; L = 4, D = 3, seed = 5, χs = (4, 6, 8, 12))
    Random.seed!(seed)
    tn = random_tensornetwork(Float64, named_grid((L, L)); bond_dimension = D)
    lnZ = log(abs(real(contract(tn; alg = "exact"))))
    @printf("\nCVM regions vs boundary MPS (random %dx%d D=%d, ln|Z| = %.10f)\n", L, L, D, lnZ)
    @printf("   %-4s %-11s %-11s %-11s\n", "χ", "greedy", "swept", "boundaryMPS")
    for χ in χs
        cache = CTMEnvironmentCache(tn, χ)
        # Greedy asked for explicitly. `cvm_freenergy(cache)` on an un-updated cache returns the
        # same number but warns — the implicit fallback is almost always a forgotten `update`.
        greedy = abs(cvm_freenergy(vertex_environments(cache), cache) - lnZ)
        swept = abs(cvm_freenergy(update(cache)) - lnZ)       # two-sided, to stationarity
        bmps = abs(log(abs(real(contract(tn; alg = "boundarymps",
                                         mps_bond_dimension = χ)))) - lnZ)
        @printf("   %-4d %-11.3e %-11.3e %-11.3e\n", χ, greedy, swept, bmps)
    end
end

function main()
    ising_free_energy()
    ising_free_energy(; aniso = true)
    peps_norm()
    single_site_observables()
    cvm_vs_boundarymps()
end

main()
