# The first-order transition of the 3D three-state Potts model, H = −J Σ δ(s_i, s_j) on the cubic
# lattice, in the thermodynamic limit with the boundary PEPS (docs/boundary_peps.md).
#
# One run follows ONE branch through the transition region:
#   PT_BRANCH=ordered     from high β downwards, seeded by a fixed-spin boundary (state 1),
#   PT_BRANCH=disordered  from low β upwards, seeded by the uniform (Z₃-symmetric) boundary.
# Each point warm-starts from the previous one, so a branch follows its own metastable
# continuation until it becomes unstable (m collapses, or grows, and the run stops there). With both
# branches' CSVs present the script also reports the crossing of their free energies — the
# transition point β_t(D) — and, interpolated there, the latent heat Q = e_dis − e_ord and the
# order-parameter jump.
#
# References (J = 1, per site): Monte Carlo, Janke & Villanova, Nucl. Phys. B 489, 679 (1997):
# β_t = 0.550565(10), Q = 0.16160(47). Tensor product variational approach, Gendiar & Nishino,
# cond-mat/0102425: β_t = 0.5496, Q = 0.228.
#
# Readouts: m = (q⟨δ(s, 1)⟩ − 1)/(q − 1) and e = −∂ ln κ/∂β (the exact ∂site/∂β as an impurity,
# Hellmann–Feynman at the variational optimum), f = ln κ per site.
#
# Environment: PT_BRANCH, PT_D, PT_CHI, PT_GPU=1, PT_BETAS (comma-separated, in scan order),
# PT_MAXITER, PT_TAG (output prefix).

using TensorNetworkQuantumSimulator
using Printf
const GPU = get(ENV, "PT_GPU", "0") == "1"
if GPU
    using CUDA, Adapt
end

const Q = 3
const BRANCH = get(ENV, "PT_BRANCH", "ordered")
const D = parse(Int, get(ENV, "PT_D", "3"))
const CHI = parse(Int, get(ENV, "PT_CHI", string(3 * D^2)))
const TAG = get(ENV, "PT_TAG", "potts3d")
const DEFAULT_BETAS = BRANCH == "ordered" ?
    [0.565, 0.56, 0.5575, 0.555, 0.5525, 0.551, 0.55, 0.549, 0.5475, 0.545, 0.5425, 0.54] :
    [0.535, 0.54, 0.545, 0.5475, 0.549, 0.55, 0.551, 0.5525, 0.555, 0.5575, 0.56]
const BETAS = haskey(ENV, "PT_BETAS") ? parse.(Float64, split(ENV["PT_BETAS"], ",")) : DEFAULT_BETAS
const MAXITER = parse(Int, get(ENV, "PT_MAXITER", "1000"))
const GTOL = 1.0e-6
const CTM_TOL = 1.0e-10
# A branch has left its phase once |m| crosses this. m reads (q⟨δ(s, 1)⟩ − 1)/(q − 1): ordering into
# another state gives m = −m₀/(q − 1), so the disordered branch is tested on |m| (measured: at D = 3 it
# ordered into state 2 or 3 at β = 0.555, m = −0.277 = −0.554/2, and a test on m alone missed it).
const MJUMP = 0.1
outfile(branch) = "$(TAG)_D$(D)_chi$(CHI)_$(branch).csv"

device(x) = GPU ? adapt(CuArray, x) : x
chi_for(d) = max(8, ceil(Int, CHI * d^2 / D^2))
order_parameter(bp, o) = (Q * real(site_ratio(bp, o)) - 1) / (Q - 1)

redirect_stderr(stdout)
println("3D q=$Q Potts, $BRANCH branch: D = $D, χ = $CHI, maxiter = $MAXITER, threads = $(Threads.nthreads()), ",
        GPU ? "GPU" : "CPU")
open(outfile(BRANCH), "w") do io
    println(io, "beta,f,m,e,gnorm,iters,seconds")
end
@printf("%8s %14s %12s %12s %10s %6s %8s\n", "β", "f = ln κ", "m", "e", "|g|", "iters", "s")
flush(stdout)

boundary = BRANCH == "ordered" ? [1.0; zeros(Q - 1)] : ones(Q)
global prev = nothing
for β in BETAS
    site0, legs, o0, e0 = potts3d_site(β; q = Q)
    site = device(site0); o = device(o0); en = device(e0)
    t = @elapsed begin
        if isnothing(prev)
            # climb D = q, …, D at this β (from D = q the seed T|b⟩ embeds without truncation)
            d0 = min(Q, D)
            global bp = boundary_peps(site, legs, d0; maxdim = chi_for(d0), boundary, maxiter = MAXITER,
                                      gtol = GTOL, ctm_tolerance = CTM_TOL)
            for d in (d0 + 1):D
                println("  first β: D = $(d - 1) done (f = $(bp.lnkappa), m = $(order_parameter(bp, o))), embedding into D = $d")
                flush(stdout)
                global bp = boundary_peps(site, legs, d; maxdim = chi_for(d), init = bp, maxiter = MAXITER,
                                          gtol = GTOL, ctm_tolerance = CTM_TOL)
            end
        else
            global bp = boundary_peps(site, legs, D; maxdim = CHI, init = prev, maxiter = MAXITER,
                                      gtol = GTOL, ctm_tolerance = CTM_TOL)
        end
    end
    f = cvm_freenergy(bp); m = order_parameter(bp, o); e = -real(site_ratio(bp, en))
    @printf("%8.4f %14.10f %12.8f %12.8f %10.2e %6d %8.1f\n", β, f, m, e, bp.gnorm, length(bp.history) - 1, t)
    flush(stdout)
    open(outfile(BRANCH), "a") do io
        println(io, join((β, f, m, e, bp.gnorm, length(bp.history) - 1, t), ","))
    end
    left = BRANCH == "ordered" ? m < MJUMP : abs(m) > MJUMP
    left && (println("  the $BRANCH branch has left its phase at β = $β (m = $m): stopping"); break)
    global prev = bp
end

# The crossing, once both branches are on disk.
function readbranch(branch)
    isfile(outfile(branch)) || return nothing
    rows = [parse.(Float64, split(l, ",")) for l in readlines(outfile(branch))[2:end]]
    keep = [r for r in rows if (branch == "ordered" ? r[3] >= MJUMP : abs(r[3]) <= MJUMP)]
    isempty(keep) && return nothing
    sort!(keep; by = first)
    return keep
end
function lerp(xs, ys, x)
    i = findlast(<=(x), xs)
    isnothing(i) && return NaN
    xs[i] == x && return ys[i]
    i == length(xs) && return NaN
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
end
# (a function, so the loop's assignments are not lost to top-level soft scope)
function report_crossing()
    ord, dis = readbranch("ordered"), readbranch("disordered")
    (isnothing(ord) || isnothing(dis)) && return nothing
    bo = first.(ord); bd = first.(dis)
    grid = sort(unique(vcat(bo, bd)))
    common = [b for b in grid if bo[1] <= b <= bo[end] && bd[1] <= b <= bd[end]]
    Δ(b) = lerp(bo, getindex.(ord, 2), b) - lerp(bd, getindex.(dis, 2), b)
    bt = NaN
    for k in 1:(length(common) - 1)
        a, b = common[k], common[k + 1]
        if Δ(a) <= 0 <= Δ(b) || Δ(a) >= 0 >= Δ(b)
            bt = a + (b - a) * Δ(a) / (Δ(a) - Δ(b))
            break
        end
    end
    if isnan(bt)
        println("\nno crossing of the two branches' free energies inside their common range $(common)")
    else
        eo = lerp(bo, getindex.(ord, 4), bt); ed = lerp(bd, getindex.(dis, 4), bt)
        mo = lerp(bo, getindex.(ord, 3), bt)
        @printf("\nD = %d, χ = %d: β_t = %.6f (MC 0.550565), latent heat Q = %.5f (MC 0.16160, TPVA 0.228), m jump %.5f\n",
                D, CHI, bt, ed - eo, mo)
    end
    return bt
end
report_crossing()
