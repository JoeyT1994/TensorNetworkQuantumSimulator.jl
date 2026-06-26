# Microbenchmark for the fermionic contraction `*(ft1, ft2)`.
#
# Goal: isolate the OVERHEAD of the locally-ordered sign machinery (the two
# `_apply_reorder_sign` passes + per-bond parity `g`) relative to the bare BLAS
# contraction it wraps. For each shape we time:
#
#   * `A * B`               — full FermionicITensor contraction (signs + BLAS), and
#   * `A.tensor * B.tensor` — the SAME contraction on the raw ITensors (BLAS only,
#                             no sign work; the floor the fermionic path builds on).
#
# `overhead×` = ferm / plain is the multiplicative cost of the signs. The arrays
# are identical, so the difference is purely sign-mask construction + application
# and the metadata bookkeeping in `contract`.

using TensorNetworkQuantumSimulator
const TNS = TensorNetworkQuantumSimulator

using ITensors: ITensors, ITensor, Index, dim
using Dictionaries: Dictionary
using Printf
using Random

# Half-even / half-odd Z2 grading for a leg of dimension d (matches the package default).
bond_grading(d) = Bool[falses(cld(d, 2)); trues(fld(d, 2))]

# Random parity-even FermionicITensor over indices `is` with arrows `dirs`
# (true = in/−, false = out/+).
function rand_ft(eltype, is::Vector{<:Index}, dirs::Vector{Bool})
    gr = Dictionary{Index, Vector{Bool}}(is, [bond_grading(dim(i)) for i in is])
    t = TNS.random_even_itensor(eltype, is, gr)
    return TNS.FermionicITensor(t, copy(is), copy(dirs), gr)
end

# Minimum wall-time (s) and last-trial allocation (bytes) over `nrep` trials, after a
# warmup call. Minimum is the standard robust estimator for a deterministic kernel.
function bestof(f; nrep = 50)
    f()                                   # warmup / compile
    tbest = Inf
    bytes = 0
    for _ in 1:nrep
        r = @timed f()
        tbest = min(tbest, r.time)
        bytes = r.bytes
    end
    return tbest, bytes
end

# Build a contraction pair sharing `n_shared` bonds, each tensor also carrying
# `n_open` open legs. Every leg has dimension χ.
#   A = [a_1 … a_o , k_1 … k_s]   (all out)
#   B = [k_1 … k_s , b_1 … b_o]   (shared in, open out)  → arrow points A → B,
# so `contract` inserts the bond-parity g on every shared bond (the realistic case).
function make_pair(eltype, n_open::Int, χ::Int, n_shared::Int)
    ks = [Index(χ, "k$i") for i in 1:n_shared]
    as = [Index(χ, "a$i") for i in 1:n_open]
    bs = [Index(χ, "b$i") for i in 1:n_open]
    A = rand_ft(eltype, Index[as; ks], Bool[fill(false, n_open); fill(false, n_shared)])
    B = rand_ft(eltype, Index[ks; bs], Bool[fill(true, n_shared); fill(false, n_open)])
    return A, B
end

function sweep(; eltype = ComplexF64, n_open = 2, n_shared = 1, χs = [4, 8, 16, 32, 48])
    Random.seed!(1)
    ITensors.disable_warn_order()
    println("\nshape: $(n_open) open + $(n_shared) shared leg(s) per tensor, eltype=$(eltype)")
    @printf("%-5s %-12s %-12s %-10s %-11s %-11s\n",
        "χ", "ferm(s)", "plain(s)", "overhead×", "ferm(MB)", "plain(MB)")
    for χ in χs
        A, B = make_pair(eltype, n_open, χ, n_shared)
        # Auto-scale trial count by the result-array size (rank 2·n_open): many reps for
        # cheap small tensors, few for the big ones so the sweep stays fast and bounded.
        result_elts = float(χ)^(2 * n_open)
        nrep = clamp(round(Int, 2.0e6 / result_elts), 5, 200)
        tf, bf = bestof(() -> A * B; nrep)
        tp, bp = bestof(() -> A.tensor * B.tensor; nrep)
        @printf("%-5d %-12.3e %-12.3e %-10.2f %-11.2f %-11.2f\n",
            χ, tf, tp, tf / tp, bf / 1e6, bp / 1e6)
    end
end

function main()
    # Typical bond contraction (rank-3 ⊗ rank-3 over one bond → rank-4 result, χ^4).
    sweep(; n_open = 2, n_shared = 1, χs = [4, 8, 16, 32, 48])
    # Heavier vertex tensors (rank-4 ⊗ rank-4 over one bond → rank-6 result, χ^6 —
    # grows fast, so cap χ low).
    sweep(; n_open = 3, n_shared = 1, χs = [2, 4, 8, 12])
    # Multi-bond contraction — exercises the reversed-fused-block sign path (rank-4 result).
    sweep(; n_open = 2, n_shared = 2, χs = [4, 8, 16, 32])
    return
end

main()
