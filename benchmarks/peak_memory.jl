# Peak-memory / timing harness for BP and simple update.
# resident-set high-water mark over the operation reads in units of F = bytes of the centre tensor.
# Usage: julia --project=<branch> bench_peak.jl MODE D   (MODE ∈ su, bp)
using TensorNetworkQuantumSimulator, LinearAlgebra, Random, Printf
const TNQS = TensorNetworkQuantumSimulator
BLAS.set_num_threads(1)
mode = ARGS[1]; D = parse(Int, ARGS[2])
rss_now() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096   # resident bytes
function measure(f)
    GC.gc(); GC.gc()
    base = rss_now(); m0 = Sys.maxrss()
    st = @timed f()
    m1 = Sys.maxrss()
    peak = m1 > m0 ? m1 - base : NaN                 # NaN: the run stayed below the earlier high-water mark
    return st.time, st.bytes, peak
end
function build(D)
    g = named_comb_tree((3, 3))
    ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = D)
    c = first(center(g))
    F = 16 * prod(TNQS.dim.(TNQS.inds(ψ[c])))
    layer = Any[]
    for ces in edge_color(g, 3); append!(layer, ("Rxx", pair, 0.1) for pair in ces); end
    append!(layer, ("Rz", [v], 0.2) for v in vertices(g))
    return ψ, layer, F, c
end
# warm-up at a small size (compilation), then the measurement
for (Dk, tag) in ((6, "warm"), (D, "run"))
    ψ, layer, F, c = build(Dk)
    apply_kwargs = (; maxdim = Dk, cutoff = nothing, normalize_tensors = true)
    if mode == "su"
        bpc = BeliefPropagationCache(ψ)
        # gates only (update_cache = false): the simple update itself
        ag = isdefined(TNQS, :apply_gates!) ? TNQS.apply_gates! : TNQS.apply_gates
        t, bytes, peak = measure(() -> ag(layer, bpc; apply_kwargs, update_cache = false, verbose = false))
        tag == "run" && @printf("SU  D=%d  F=%.1f MB  layer: %.3f s  alloc %.1f F  peak-over-baseline %.2f F\n", Dk, F / 1e6, t, bytes / F, peak / F)
    elseif mode == "bp"
        bpc = BeliefPropagationCache(ψ)
        update(bpc; maxiter = 1)
        t, bytes, peak = measure(() -> update(bpc; maxiter = 3))
        tag == "run" && @printf("BP  D=%d  F=%.1f MB  3 iters: %.3f s  alloc %.1f F  peak-over-baseline %.2f F\n", Dk, F / 1e6, t, bytes / F, peak / F)
    end
end
