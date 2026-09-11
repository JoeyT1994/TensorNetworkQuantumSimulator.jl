# Boundary-MPS timing / peak harness. Usage: julia --project=<branch> bench_bmps.jl L D chi
using TensorNetworkQuantumSimulator, LinearAlgebra, Random, Printf
const TNQS = TensorNetworkQuantumSimulator
BLAS.set_num_threads(1)
L = parse(Int, ARGS[1]); D = parse(Int, ARGS[2]); χ = parse(Int, ARGS[3])
rss_now() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096
tensor_bytes(t) = sizeof(eltype(t)) * prod(TNQS.dim.(TNQS.inds(t)))
function run(L, D, χ)
    Random.seed!(7)
    g = named_grid((L, L))
    ψ = random_tensornetworkstate(ComplexF64, g, siteinds("S=1/2", g); bond_dimension = D)
    c = BoundaryMPSCache(ψ, χ)
    c = update(c; maxiter = 1)                                   # warm-up (compile)
    c = BoundaryMPSCache(ψ, χ)
    GC.gc(); GC.gc(); base = rss_now(); m0 = Sys.maxrss()
    st = @timed update(c; maxiter = 3)
    m1 = Sys.maxrss()
    F = maximum(tensor_bytes(m) for (_, m) in pairs(TNQS.messages(st.value)))
    z = real(expect(st.value, ("Z", [(L ÷ 2, L ÷ 2)])))
    @printf("BMPS L=%d D=%d chi=%d  largest message %.1f MB  3 iters: %.2f s  alloc %.0f MB (%.1f F)  peak-over-baseline %s   Z=%.6f\n",
        L, D, χ, F / 1e6, st.time, st.bytes / 1e6, st.bytes / F, m1 > m0 ? @sprintf("%.1f MB (%.1f F)", (m1 - base) / 1e6, (m1 - base) / F) : "below high-water mark", z)
end
run(L, D, χ)
