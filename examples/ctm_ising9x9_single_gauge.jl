"""Collect one reproducible gauge attack without contracting unused variants.

This focused driver is intended for matched-error scaling studies. SVD-CTMRG
sees the raw attacked tensors, while eig-CTMRG sees the same physical state after
the library's Vidal/BP `symmetric_gauge` preconditioner.
"""

include("ctm_ising9x9_gauge.jl")

function main_single_gauge()
    log_condition = parse(Float64, get(ENV, "ISING9_GAUGE_LOG10_KAPPA", "1"))
    seed = parse(Int, get(ENV, "ISING9_GAUGE_SEED", "1234"))
    chis = parse_ints(get(ENV, "ISING9_GAUGE_CHIS", "4,8,12,16,20,24,28,32"))
    methods = parse_methods(get(ENV, "ISING9_GAUGE_METHODS", "cut,cycle"))
    references = (env_reference("ISING9_REFERENCE_F", ISING9_DEFAULT_F_REFERENCE),
                  env_reference("ISING9_REFERENCE_X", ISING9_DEFAULT_X_REFERENCE))
    vidal_regularization = parse(Float64,
        get(ENV, "ISING9_GAUGE_VIDAL_REGULARIZATION", string(10eps(Float64))))

    check_gauge_identity([log_condition])
    state = build_state(load_peps())
    sites = collect(vertices(graph(state)))
    attacked, pair_residual, max_kappa = apply_gauge_attack(
        state, log_condition; seed)

    prepared = attacked
    preparation_seconds = 0.0
    preparation_status = "ok"
    if :cycle in methods
        prepared, preparation_seconds, preparation_status = safe_prepare_variant(
            attacked, :vidal; bp_maxiter = 80, bp_tolerance = 1.0e-10,
            vidal_regularization)
    end

    output = get(ENV, "ISING9_GAUGE_OUTPUT", "")
    io = isempty(output) ? stdout : open(output, "w")
    try
        println(io, "chi,method,gauge_seed,requested_kappa,measured_max_condition," *
                    "gauge_pair_residual,preconditioner,lnZ_value,lnZ_ground_truth," *
                    "lnZ_relative_error,mean_X_value,mean_X_ground_truth," *
                    "mean_X_relative_error,marginal_inconsistency,preparation_seconds," *
                    "contraction_seconds,status")
        for chi in chis, method in methods
            active_state = method === :cycle ? prepared : attacked
            values, elapsed, method_status = safe_method_values(
                active_state, chi, method, sites)
            lnz, average_x, marginal = values
            status = preparation_status == "ok" ? method_status : preparation_status
            preconditioner = method === :cycle ? "symmetric/BP" : "raw"
            lnz_error = abs(expm1(lnz - references[1]))
            x_error = abs(average_x - references[2]) / abs(references[2])
            row = (chi, method, seed, 10.0^log_condition, max_kappa,
                   pair_residual, preconditioner, lnz, references[1], lnz_error,
                   average_x, references[2], x_error, marginal,
                   method === :cycle ? preparation_seconds : 0.0,
                   elapsed, status)
            println(io, join(row, ','))
            io === stdout || println(stdout, join(row, ','))
            flush(io)
        end
    finally
        io === stdout || close(io)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_single_gauge()
