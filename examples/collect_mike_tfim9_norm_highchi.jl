"""Extend the 9x9 D=3 TFIM norm and lattice-average X scan to high chi.

CTMRG uses strict environment convergence because X is measured from the
unperturbed double-layer environment. Methods can be split across processes.
"""

include("ctm_ising9x9_benchmark.jl")

using LinearAlgebra: BLAS
using Logging: SimpleLogger, Warn, with_logger
using Statistics: mean

mike_norm_ints(value) = parse.(Int, strip.(split(value, ',')))

function mike_norm_value(state, sites, chi, method)
    warning_buffer = IOBuffer()
    values = with_logger(SimpleLogger(warning_buffer, Warn)) do
        if method === :bmps
            F, X = bmps_values(state, chi, sites)
            return F, X, NaN
        end
        cache = update(CTMEnvironmentCache(state, chi; projector = method);
                       convergence = :environment, tolerance = 1e-10, maxiter = 80)
        observables = [("X", [site]) for site in sites]
        return cvm_freenergy(cache), mean(real.(expect(cache, observables))),
               marginal_inconsistency(cache)
    end
    warnings = String(take!(warning_buffer))
    status = isempty(strip(warnings)) ? "ok" :
             (occursin("did not converge", warnings) ? "not_converged" : "warning")
    if method === :cycle && status == "ok" && isfinite(values[3]) && values[3] > 1e-10
        status = "inconsistent"
    end
    return values..., status
end

function main_mike_norm_highchi()
    BLAS.set_num_threads(parse(Int, get(ENV, "MIKE_NORM_BLAS_THREADS", "1")))
    state = build_state(load_peps())
    sites = collect(vertices(graph(state)))
    chis = mike_norm_ints(get(ENV, "MIKE_NORM_CHIS", join(33:64, ',')))
    methods = Symbol.(strip.(split(get(ENV, "MIKE_NORM_METHODS", "cut"), ',')))
    all(method -> method in (:cut, :cycle, :bmps), methods) || error("invalid method")
    output = get(ENV, "MIKE_NORM_OUTPUT", "")
    io = isempty(output) ? stdout : open(output, "w")
    try
        println(io, "linear_size,bond_dimension,chi,method,lnZ_value,meanX_value," *
                    "lnZ_ground_truth,meanX_ground_truth,marginal_inconsistency," *
                    "seconds,status")
        for chi in chis, method in methods
            result = nothing
            seconds = @elapsed result = mike_norm_value(state, sites, chi, method)
            F, X, marginal, status = result
            println(io, join((9, 3, chi, method, F, X,
                              ISING9_DEFAULT_F_REFERENCE, ISING9_DEFAULT_X_REFERENCE,
                              marginal, seconds, status), ','))
            io === stdout || println(stdout,
                "chi=$chi method=$method F=$F X=$X status=$status seconds=$seconds")
            flush(io)
        end
    finally
        io === stdout || close(io)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_mike_norm_highchi()
