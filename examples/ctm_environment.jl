# Demo of the finite CTMRG engine, `CTMEnvironmentCache`.
#
# There is ONE engine (src/MessagePassing/ctmenvironmentcache.jl); everything here is a
# call into it, so there is no duplicated truncation machinery to drift out of sync.
# It handles single-layer partition functions (isotropic or anisotropic, square or not),
# double-layer state norms ⟨ψ|ψ⟩ and forms ⟨ψ|O|ψ⟩, and exposes row environments for
# row-local observables.
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

function main()
    ising_free_energy()
    ising_free_energy(; aniso = true)
    peps_norm()
    peps_norm(; D = 3)
    row_environment_check()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
