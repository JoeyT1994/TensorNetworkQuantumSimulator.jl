# 3D classical Ising in the thermodynamic limit with the boundary PEPS (docs/boundary_peps.md):
# magnetisation and free energy per site against β, compared with Monte Carlo (the Talapov–Blöte
# fit, J. Phys. A 29, 5727 (1996), valid for β_c < β ≲ 0.30). Beyond the fit's range there is no
# reference of comparable quality here: the low-temperature series through u¹² still oscillates at
# the 1e-3 level at β = 0.35 (its u¹³ term), so those points are printed without a reference.
#
# The scan runs from high β downwards so that every point warm-starts from the ordered solution
# of the previous one (A and both 2D environments). Below β_c the finite-D optimum keeps a
# residual m: that pseudo-transition is the D-dependent error of the method.
#
# The first β climbs to D through D = 2, 3, … (each optimised and embedded into the next, χ scaled
# as D²), since a cold start at large D is slow and can land in a poor local optimum.
#
# Output: one row per β on stdout and in the CSV `OUT` (β, m, m_ref, f, |g|, L-BFGS iterations, s).
# Override D, χ, the output file, the β list (comma-separated, scanned in the given order), the
# iteration cap and the device with BP_D, BP_CHI, BP_OUT, BP_BETAS, BP_MAXITER and BP_GPU=1 (the
# GPU pays from D = 4; at D = 3 it is no faster than 8 CPU threads).

using TensorNetworkQuantumSimulator
using Printf
const GPU = get(ENV, "BP_GPU", "0") == "1"
if GPU
    using CUDA, Adapt
end

const D = parse(Int, get(ENV, "BP_D", "2"))
const CHI = parse(Int, get(ENV, "BP_CHI", "16"))
const OUT = get(ENV, "BP_OUT", "ising3d_bp_D$(D)_chi$(CHI).csv")
const BETAS = haskey(ENV, "BP_BETAS") ? parse.(Float64, split(ENV["BP_BETAS"], ",")) :
              [0.40, 0.35, 0.30, 0.28, 0.26, 0.25, 0.24, 0.235, 0.23, 0.2275, 0.225, 0.2235,
               0.2225, 0.2217, 0.221, 0.22, 0.218, 0.215, 0.21, 0.20]
const MAXITER = parse(Int, get(ENV, "BP_MAXITER", "1000"))  # L-BFGS iterations per β
const GTOL = 1.0e-6
const CTM_TOL = 1.0e-10     # the floor of the adaptive CTMRG tolerance
const BETA_C = 0.2216544

# Monte Carlo reference for m(β)
function m_reference(β)
    β <= BETA_C && return 0.0
    β > 0.305 && return NaN  # outside the fit's validated range 0.0005 < t < 0.26
    t = 1 - BETA_C / β       # Talapov–Blöte fit to Monte Carlo
    return t^0.32694109 * (1.6919045 - 0.34357731 * t^0.50842026 - 0.42572366 * t)
end

device(x) = GPU ? adapt(CuArray, x) : x
chi_for(d) = max(8, ceil(Int, CHI * d^2 / D^2))

redirect_stderr(stdout)
println("3D Ising boundary PEPS scan: D = $D, χ = $CHI, maxiter = $MAXITER, threads = $(Threads.nthreads()), ",
        GPU ? "GPU" : "CPU")
open(OUT, "w") do io
    println(io, "beta,m,m_ref,f,gnorm,iters,seconds")
end
@printf("%8s %12s %12s %10s %12s %10s %6s %8s\n", "β", "m", "m_MC", "m − m_MC", "f = ln κ", "|g|", "iters", "s")
flush(stdout)

global prev = nothing
for β in BETAS
    site0, legs, mag = ising3d_site(β)
    site = device(site0)
    t = @elapsed begin
        if isnothing(prev)
            # climb D = 2, 3, …, D at this β
            global bp = boundary_peps(site, legs, 2; maxdim = chi_for(2), boundary = [1.0, 0.0],
                                      maxiter = MAXITER, gtol = GTOL, ctm_tolerance = CTM_TOL)
            for d in 3:D
                println("  first β: D = $(d - 1) done (f = $(bp.lnkappa)), embedding into D = $d"); flush(stdout)
                global bp = boundary_peps(site, legs, d; maxdim = chi_for(d), init = bp, maxiter = MAXITER,
                                          gtol = GTOL, ctm_tolerance = CTM_TOL)
            end
        else
            global bp = boundary_peps(site, legs, D; maxdim = CHI, init = prev, maxiter = MAXITER,
                                      gtol = GTOL, ctm_tolerance = CTM_TOL)
        end
    end
    m = abs(real(site_ratio(bp, mag)))
    f = cvm_freenergy(bp)
    mref = m_reference(β)
    @printf("%8.4f %12.8f %12.8f %+10.2e %12.8f %10.2e %6d %8.1f\n", β, m, mref, m - mref, f, bp.gnorm,
            length(bp.history) - 1, t)
    flush(stdout)
    open(OUT, "a") do io
        println(io, join((β, m, mref, f, bp.gnorm, length(bp.history) - 1, t), ","))
    end
    global prev = bp
end
