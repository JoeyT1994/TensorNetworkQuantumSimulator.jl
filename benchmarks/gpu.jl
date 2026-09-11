# GPU vs CPU speed on the dominant operations, quickly: one BP iteration and ONE two-site gate on
# the centre bond of a `named_comb_tree((3, 3))` (centre tensor 2·D³ = F), for a few D and both
# precisions, best of two after a warm-up at D = 32. ~2 min on an RTX A6000 (mostly compilation).
# Usage: julia --project=. benchmarks/gpu.jl [D1,D2,...]
using TensorNetworkQuantumSimulator, LinearAlgebra, Random, Printf, CUDA, Adapt
const TNQS = TensorNetworkQuantumSimulator
CUDA.allowscalar(false)
Ds = isempty(ARGS) ? (60, 120, 200) : parse.(Int, split(ARGS[1], ","))
function setup(elt, D)
    Random.seed!(1)
    g = named_comb_tree((3, 3))
    ψ = random_tensornetworkstate(elt, g, "S=1/2"; bond_dimension = D)
    c = first(center(g)); nb = first(neighbors(g, c))
    gate = Any[("Rxx", (c, nb), 0.1)]
    F = sizeof(elt) * prod(TNQS.dim.(TNQS.inds(ψ[c])))
    return ψ, gate, F
end
sync(dev) = dev == "gpu" && CUDA.synchronize()
function timeit(ψ, gate, D, dev)
    bpc = TNQS.update(BeliefPropagationCache(ψ); maxiter = 1)
    tb = tg = Inf
    for _ in 1:2                                    # best of two: the first call can carry library set-up
        sync(dev); tb = min(tb, @elapsed (TNQS.update(bpc; maxiter = 1); sync(dev)))
        bpc2 = BeliefPropagationCache(deepcopy(ψ))
        sync(dev); tg = min(tg, @elapsed (apply_gates!(gate, bpc2; apply_kwargs = (; maxdim = D, cutoff = nothing), update_cache = false, verbose = false); sync(dev)))
    end
    return tb, tg
end
for elt in (ComplexF32, ComplexF64)
    ψw, gw, _ = setup(elt, 32)                      # warm-up (compilation, library set-up) on both devices
    timeit(ψw, gw, 32, "cpu"); timeit(adapt(CuArray, ψw), gw, 32, "gpu")
    @printf("\n%s  (CPU BLAS threads = %d)\n", elt, BLAS.get_num_threads())
    @printf("  %5s %8s | %-22s | %-22s\n", "D", "F", "BP iteration cpu/gpu", "one gate cpu/gpu")
    for D in Ds
        ψ, gate, F = setup(elt, D)
        cb, cg = timeit(ψ, gate, D, "cpu")
        gb, gg = timeit(adapt(CuArray, ψ), gate, D, "gpu")
        @printf("  %5d %6.0f MB | %6.3f s / %6.3f s %4.1fx | %6.3f s / %6.3f s %4.1fx\n", D, F / 1e6, cb, gb, cb / gb, cg, gg, cg / gg)
    end
end
