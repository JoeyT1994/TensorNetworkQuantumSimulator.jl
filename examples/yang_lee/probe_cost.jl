# Cost of one 2D CTMRG step (norm ⟨A|A⟩, bond D², and sandwich ⟨A|T|A⟩, bond 2D²) against D and χ, on
# the β = 0.20 imaginary-field boundary state near the fold (D = 3 embedded at larger D), and where the
# time goes (flat profile at one mid-size point). Sets the D-scaling for a GPU-cluster campaign.
using TensorNetworkQuantumSimulator, Printf, LinearAlgebra, Serialization, Random, Profile
BLAS.set_num_threads(1)
const T = TensorNetworkQuantumSimulator
const β = 0.2
S = deserialize("ylk3_beta0.2_D3_chi24.jls")
legs = S.legs; bp = S.cur; θ = S.θc
x = ising3d_site(β; h = im * θ / β)
site = T.replaceinds(x[1], collect(x[2]), collect(legs))
println("β = $β, θ = $θ, D = 3 state |g| = $(bp.gnorm); $(Threads.nthreads()) threads, BLAS 1")
const CONFIGS = [(3, 24), (4, 32), (4, 48), (5, 50), (5, 72)]
const WARM, TIMED, BUDGET = 3, 3, 420.0
t0 = time()
function envstep(layers, lg, χ)
    e = T.update(T.InfiniteCTM2D(layers, lg, χ; c4v = true); tolerance = 0.0, maxiter = WARM)
    t = @elapsed e2 = T.update(T.InfiniteCTM2D(layers, lg, χ; init = e, c4v = true); tolerance = 0.0, maxiter = TIMED)
    return t / TIMED, e
end
println("  D   χ   Ds(norm/sand)  s/step norm  s/step sandwich")
res = []
for (D, χ) in CONFIGS
    time() - t0 > BUDGET && (println("  (budget: stop before D = $D, χ = $χ)"); break)
    al = (T.new_index(D; tags = "bp,xm"), T.new_index(D; tags = "bp,xp"), T.new_index(D; tags = "bp,ym"),
          T.new_index(D; tags = "bp,yp"), T.new_index(T.dim(legs[5]); tags = "bp,p"))
    bl = Tuple(T.new_index(D; tags = "bp,bra") for _ in 1:4)
    A = T._bp_symmetrize(T._bp_embed(bp.A, bp.Alegs, al, 1.0e-3, Xoshiro(1)), al[1:4])
    nl, nlg = T._bp_norm_layers(A, al, bl; bilinear = true)
    sl, slg = T._bp_sandwich_layers(A, al, bl, site, legs; bilinear = true)
    tn, _ = envstep(nl, nlg, χ)
    ts, es = envstep(sl, slg, χ)
    @printf("  %d  %3d   %3d / %3d      %8.3f     %8.3f\n", D, χ, D^2, 2D^2, tn, ts)
    flush(stdout)
    push!(res, (D, χ, tn, ts))
    if (D, χ) == (4, 48)                             # where the sandwich step's time goes
        Profile.clear()
        Profile.init(; n = 10^7, delay = 0.005)
        @profile T.update(T.InfiniteCTM2D(sl, slg, χ; init = es, c4v = true); tolerance = 0.0, maxiter = 2)
        io = IOBuffer()
        Profile.print(IOContext(io, :displaysize => (2000, 250)); format = :flat, sortedby = :count, noisefloor = 2)
        lines = split(String(take!(io)), '\n')
        tot = something(tryparse(Int, something(match(r"Total snapshots:\s*(\d+)", join(lines, "\n")), (captures = ["0"],)).captures[1]), 0)
        println("  profile (D = 4, χ = 48, sandwich; $tot snapshots) — frames of interest:")
        for l in lines
            occursin(r"LAPACK|lapack|geev|gees|gesdd|gesvd|heev|syev|eigen|svd|gemm|_contract|contract!|permutedims|qr", l) &&
                println("    ", strip(l)[1:min(end, 200)])
        end
    end
end
if length(res) >= 3                                  # power law in the sandwich bond, χ ∝ Ds
    ok = [r for r in res if r[2] >= 1.3 * 2 * r[1]^2]
    if length(ok) >= 2
        X = [ones(length(ok)) log.(2 .* first.(ok) .^ 2)]; y = log.([r[4] for r in ok])
        c = X \ y
        @printf("sandwich step ∝ Ds^%.2f at χ ≈ 1.3–1.5 Ds (from %d points)\n", c[2], length(ok))
    end
end
@printf("total %.0f s\n", time() - t0)
