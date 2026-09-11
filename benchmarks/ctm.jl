# CTM timing / peak-memory harness. Usage: julia --project=<branch> bench_ctm.jl PROJECTOR L D chi
using TensorNetworkQuantumSimulator, LinearAlgebra, Random, Printf
const TNQS = TensorNetworkQuantumSimulator
BLAS.set_num_threads(1)
proj = Symbol(ARGS[1]); L = parse(Int, ARGS[2]); D = parse(Int, ARGS[3]); χ = parse(Int, ARGS[4])
rss_now() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096
tensor_bytes(t) = sizeof(eltype(t)) * prod(TNQS.dim.(TNQS.inds(t)))
function envbytes(cache)
    env = TNQS.environments(cache)
    best = 0
    for fld in (:C, :T, :PH, :PV)
        isdefined(env, fld) || continue
        for (_, v) in getfield(env, fld)
            v === nothing && continue
            for t in (v isa Tuple ? v : (v,))
                t isa TNQS.Tensor || continue
                best = max(best, tensor_bytes(t))
            end
        end
    end
    return best
end
function run(L, D, χ, proj)
    Random.seed!(7)
    g = named_grid((L, L))
    ψ = random_tensornetworkstate(ComplexF64, g, siteinds("S=1/2", g); bond_dimension = D)
    kw = proj == :cycle ? (; projector = :cycle) : (;)
    ckw = proj == :cycle ? (; convergence = :marginal) : (;)
    c = CTMEnvironmentCache(ψ, χ; kw...)
    c = update(c; maxiter = 1, ckw...)                       # warm-up sweep (compile)
    GC.gc(); GC.gc(); base = rss_now(); m0 = Sys.maxrss()
    st = @timed update(c; maxiter = 3, tolerance = 0.0, ckw...)
    m1 = Sys.maxrss()
    F = envbytes(st.value)
    @printf("CTM %s L=%d D=%d chi=%d  largest env tensor %.1f MB  3 sweeps: %.2f s  alloc %.0f MB (%.1f F)  peak-over-baseline %s\n",
        proj, L, D, χ, F / 1e6, st.time, st.bytes / 1e6, st.bytes / F, m1 > m0 ? @sprintf("%.1f MB (%.1f F)", (m1 - base) / 1e6, (m1 - base) / F) : "below earlier high-water mark")
end
run(L, D, χ, proj)
