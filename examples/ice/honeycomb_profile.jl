# Profile the honeycomb ice CTM on the current device — calibrates a campaign before it runs. Per
# (D, kind): seconds per WARM CTM iteration (after NW warm-up iterations from the seed, which include
# compilation), the pair route taken (split = matrix-free subspace SVD, never forming the enlarged
# quadrant; dense = the quadrant formed and factorised — n = χ·r, O(n³)), and with PROFILE=1 a device
# kernel breakdown of one iteration (CUDA.@profile). Peak device memory: poll nvidia-smi alongside.
#
# ENV: DS ("4,5,6"), KINDS ("norm,sand"), CHIF (χ = CHIF·D², 1), NW (6), NIT (4), DEVICE (cpu | gpu),
# ELT (Float64 | Float32 — Float32 only to probe the FLOP-rich regime; not for physics), BLASN (4),
# SUDIR (a directory with su_D<D>.jls to reuse, optional), PROFILE (0), NET (paired | layered),
# SVD_OVERSAMPLE / SVD_MAXITER / SVD_TOL (projector options). Working set: JULIA_CUDA_HARD_MEMORY_LIMIT.

include(joinpath(@__DIR__, "honeycomb.jl"))
using Adapt, Serialization
using Logging: NullLogger, with_logger
const DEVICE = get(ENV, "DEVICE", "cpu")
if DEVICE == "gpu"
    @eval using CUDA
    CUDA.allowscalar(false)
end
const CONV = Symbol(get(ENV, "CONV", "environment"))     # the convergence signal (see `update`)
const ELT = get(ENV, "ELT", "Float64") == "Float32" ? Float32 : Float64
BLAS.set_num_threads(parse(Int, get(ENV, "BLASN", "4")))
sync() = DEVICE == "gpu" ? CUDA.synchronize() : nothing
function todev(t)
    a = ELT.(Array(T.array(t, T.inds(t)...)))
    return T.from_array(DEVICE == "gpu" ? adapt(CUDA.CuArray, a) : a, T.inds(t)...)
end
quiet(f) = with_logger(f, NullLogger())
# projector options (CTMOptions): SVD_OVERSAMPLE, SVD_MAXITER, SVD_TOL
const CTMKW = (; (Symbol(lowercase(k)) => (k == "SVD_TOL" ? parse(Float64, ENV[k]) : parse(Int, ENV[k]))
                  for k in ("SVD_OVERSAMPLE", "SVD_MAXITER", "SVD_TOL") if haskey(ENV, k))...)

function profile_one(D, kind; χ, nw, nit, prof)
    f = joinpath(get(ENV, "SUDIR", ""), "su_D$(D).jls")
    X, Y, w = isfile(f) ? deserialize(f)[1:3] : hexstate(D)[1:3]
    if get(ENV, "NET", "paired") == "layered"            # legs never fused (split layers)
        site, legs = network(X, Y, w; kind); sv = seed(X, kind)
    else
        site, legs, sv = paired(X, Y, w; kind)
    end
    site = Any[todev(t) for t in site]
    tw = @elapsed begin
        ic = quiet(() -> T.update(T.InfiniteCTM2D(site, legs, χ; boundary = sv, CTMKW...); tolerance = 0.0, maxiter = nw, convergence = CONV))
        sync()
    end
    empty!(T.CTM_SVD_STATS)
    t = @elapsed begin
        ic = quiet(() -> T.update(T.InfiniteCTM2D(site, legs, χ; boundary = sv, init = ic, CTMKW...); tolerance = 0.0, maxiter = nit, convergence = CONV,
                                  miniter = 1))
        sync()
    end
    nsplit = get(T.CTM_SVD_STATS, :i2_split, 0); bail = get(T.CTM_SVD_STATS, :i2_split_bail, 0)
    r = prod(T.dim.(legs[1]))
    @printf("D = %d %-4s %s χ = %d (raw bond %d, n = χr = %d, %s, %s): %.2f s per warm iteration  [warm-up %d its %.0f s]  pairs: %d split, %d dense, %d bail-outs  ln κ = %.10f\n",
            D, kind, get(ENV, "NET", "paired"), χ, r, χ * r, DEVICE, ELT, t / nit, nw, tw, nsplit, 4nit - nsplit, bail, T.cvm_freenergy(ic))
    flush(stdout)
    if prof && DEVICE == "gpu"
        rep = CUDA.@profile quiet(() -> T.update(T.InfiniteCTM2D(site, legs, χ; boundary = sv, init = ic, CTMKW...); tolerance = 0.0,
                                                maxiter = 1, miniter = 1))
        io = IOBuffer(); show(IOContext(io, :displaysize => (60, 220)), rep)
        println(join(first(split_lines(String(take!(io))), 60), "\n"))
        flush(stdout)
    end
end
split_lines(s) = split(s, '\n')

function main()
    DS = parse.(Int, split(get(ENV, "DS", "4,5,6"), ","))
    KINDS = Symbol.(split(get(ENV, "KINDS", "norm,sand"), ","))
    chif = parse(Float64, get(ENV, "CHIF", "1"))
    nw = parse(Int, get(ENV, "NW", "6")); nit = parse(Int, get(ENV, "NIT", "4"))
    prof = get(ENV, "PROFILE", "0") == "1"
    println("device $DEVICE, $ELT, $(Threads.nthreads()) threads, BLAS $(BLAS.get_num_threads()), projector options $CTMKW")
    DEVICE == "gpu" && println(CUDA.name(CUDA.device()))
    for D in DS, kind in KINDS
        profile_one(D, kind; χ = round(Int, chif * D^2), nw, nit, prof)
    end
end
main()
