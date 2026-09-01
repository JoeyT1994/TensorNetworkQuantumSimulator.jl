"""Collect actual RBIM values (not only errors) for the Mike handoff."""

include("ctm_rbim20_jjprime.jl")

using Logging: SimpleLogger, Warn, with_logger

function mike_rbim_capture(f)
    warning_buffer = IOBuffer()
    value = with_logger(f, SimpleLogger(warning_buffer, Warn))
    warnings = String(take!(warning_buffer))
    status = isempty(strip(warnings)) ? "ok" :
             (occursin("did not converge", warnings) ? "not_converged" : "warning")
    return value, status, warnings
end

function main_mike_rbim20()
    K, Kp, probability = 0.88, 0.17, 0.5
    L, h, site = 20, 0.01, (10, 10)
    chis = parseints("MIKE_CHIS", join(1:16, ','))
    seeds = parseints("MIKE_RBIM_SEEDS", "401,402,403,404,405,406")
    methods = Symbol.(strip.(split(get(ENV, "MIKE_METHODS", "cut,cycle,bmps"), ',')))
    reference_chi = parse(Int, get(ENV, "MIKE_REFERENCE_CHI", "24"))
    reference_maxiter = parse(Int, get(ENV, "MIKE_REFERENCE_MAXITER", "120"))
    maxiter = parse(Int, get(ENV, "MIKE_MAXITER", "60"))
    tolerance = parse(Float64, get(ENV, "MIKE_TOL", "1e-10"))
    output = get(ENV, "MIKE_OUTPUT", "")
    io = isempty(output) ? stdout : open(output, "w")
    try
        println(io, "linear_size,seed,chi,method,lnZ_value,local_magnetization_value," *
                    "marginal_inconsistency,seconds,status,lnZ_ground_truth," *
                    "local_magnetization_ground_truth,magnetization_site,betaJ,betaJprime,h")
        for seed in seeds
            Random.seed!(seed)
            g = named_grid((L, L))
            couplings = Dictionary(collect(edges(g)),
                                   [rand() < probability ? K : Kp for _ in edges(g)])
            tn, mk, _ = rbim_setup(g, 1.0, h, couplings)
            tspin, tplain = mk(site; spin = true), mk(site)
            reference_result, reference_status, reference_warnings = mike_rbim_capture() do
                update(CTMEnvironmentCache(tn, reference_chi; projector = :cut);
                       maxiter = reference_maxiter, tolerance,
                       convergence = :environment)
            end
            reference_status == "ok" || error(
                "RBIM reference failed for seed $seed: $reference_status\n$reference_warnings")
            reference = reference_result
            Fref = cvm_freenergy(reference)
            mref = local_magnetisation(reference, tspin, tplain, site)
            for chi in chis, method in methods
                values = nothing
                seconds = @elapsed begin
                    if method === :bmps
                        F, m = bmps_freeenergy_and_local(tn, tspin, tplain, site, chi)
                        values = (F, m, NaN, "ok")
                    else
                        cache_result, cache_status, _ = mike_rbim_capture() do
                            update(CTMEnvironmentCache(
                                tn, chi; projector = method, cycle_gapcut = 0.0);
                                maxiter, tolerance, convergence = :environment)
                        end
                        cache = cache_result
                        values = (cvm_freenergy(cache),
                                  local_magnetisation(cache, tspin, tplain, site),
                                  marginal_inconsistency(cache), cache_status)
                    end
                end
                F, m, marginal, status = values
                println(io, join((L, seed, chi, method, F, m, marginal, seconds, status,
                                  Fref, mref, "10;10", K, Kp, h), ','))
                flush(io)
            end
        end
    finally
        io === stdout || close(io)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_mike_rbim20()
