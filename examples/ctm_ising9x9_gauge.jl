"""Random-gauge attack for the saved 9x9 D=3 Ising PEPS.

Every virtual bond receives a reproducible pair ``G`` and ``G^-1``. The physical
PEPS is therefore unchanged, while non-gauge-invariant truncations see a
different representation. The benchmark emits the original PEPS, two independent
moderate random gauges, and the first random gauge after `symmetric_gauge` (the
library's Vidal/BP gauge).

The output reports drift from the identity network with the *same*
preconditioner, plus error from the high-chi bMPS reference. For example:

    ISING9_GAUGE_CHIS=8,16 \
    ISING9_GAUGE_LOG10_KAPPA=1 \
    ISING9_GAUGE_SEEDS=1234,5678 \
    julia --project=. --startup-file=no examples/ctm_ising9x9_gauge.jl
"""

include("ctm_ising9x9_benchmark.jl")

using Graphs: src, dst
using LinearAlgebra: Diagonal, I, cond, inv, opnorm, qr
using Logging: SimpleLogger, Warn, with_logger
using Random: Xoshiro
using Statistics: mean

parse_floats(s) = parse.(Float64, strip.(split(s, ',')))

function random_bond_gauge(rng, dimension, log10_condition)
    left = Matrix(qr(randn(rng, dimension, dimension)).Q)
    right = Matrix(qr(randn(rng, dimension, dimension)).Q)
    exponents = range(-log10_condition / 2, log10_condition / 2; length = dimension)
    return left * Diagonal(10.0 .^ exponents) * right'
end

"""Apply paired non-unitary gauges and return `(state, max_pair_residual, max_kappa)`."""
function apply_gauge_attack(state, log10_condition; seed = 1234)
    attacked = copy(state)
    rng = Xoshiro(seed)
    max_pair_residual = 0.0
    max_kappa = 1.0
    for edge in edges(attacked)
        u, v = src(edge), dst(edge)
        old = only(commoninds(attacked[u], attacked[v]))
        fresh = sim(old)
        matrix = random_bond_gauge(rng, dim(old), log10_condition)
        inverse_matrix = inv(matrix)

        # A'_a = sum_i A_i G_ia and B'_a = sum_j B_j (G^-1)_aj, so the
        # shared fresh index contracts to G * G^-1 = I.
        left = attacked[u] * ITensor(matrix, old, fresh)
        right = attacked[v] * ITensor(transpose(inverse_matrix), old, fresh)
        TNQS.setindex_preserve!(attacked, left, u)
        TNQS.setindex_preserve!(attacked, right, v)

        identity = Matrix{Float64}(I, dim(old), dim(old))
        max_pair_residual = max(max_pair_residual,
                                opnorm(matrix * inverse_matrix - identity, Inf))
        max_kappa = max(max_kappa, cond(matrix))
    end
    return attacked, max_pair_residual, max_kappa
end

"""Catch an index-orientation bug before launching the expensive 9x9 scan."""
function check_gauge_identity(log_conditions)
    graph = named_grid((2, 2))
    state = random_tensornetworkstate(Float64, graph, siteinds("S=1/2", graph);
                                      bond_dimension = 3)
    before = norm_sqr(state; alg = "exact")
    for logk in log_conditions
        attacked, pair_residual, _ = apply_gauge_attack(state, logk)
        after = norm_sqr(attacked; alg = "exact")
        relative_drift = abs(after - before) / abs(before)
        pair_residual <= 1.0e-10 || error(
            "gauge pair failed at log10(kappa)=$logk: residual=$pair_residual")
        relative_drift <= 1.0e-8 || error(
            "gauge attack changed an exact 2x2 contraction at log10(kappa)=$logk: " *
            "relative drift=$relative_drift")
    end
end

function method_values(state, chi, method, sites)
    warning_buffer = IOBuffer()
    values = with_logger(SimpleLogger(warning_buffer, Warn)) do
        measure_observable = parse_bool(get(ENV, "ISING9_GAUGE_OBSERVABLE", "true"))
        observables = measure_observable ? [("X", [site]) for site in sites] : Tuple[]
        if method === :bmps
            cache = update(BoundaryMPSCache(state, chi; partition_by = "row",
                                            gauge_state = false))
            lnN = log(abs(real(norm_sqr(cache; alg = "boundarymps"))))
            measure_observable || return lnN, NaN, NaN
            cache = TNQS.update_partitions(cache, sites)
            average_x = mean(real.(expect(cache, observables; alg = "boundarymps",
                                          bmps_messages_up_to_date = true)))
            return lnN, average_x, NaN
        end
        cycle_subspace = parse_bool(get(ENV, "ISING9_CYCLE_SUBSPACE", "false"))
        cycle_iters = parse(Int, get(ENV, "ISING9_CYCLE_ITERS", "20"))
        cycle_warmstart = parse_bool(get(ENV, "ISING9_CYCLE_WARMSTART", "true"))
        cache = update(CTMEnvironmentCache(state, chi; projector = method,
                       cycle_subspace = method === :cycle && cycle_subspace,
                       cycle_iters, cycle_warmstart);
                       convergence = :environment, tolerance = 1.0e-10, maxiter = 80)
        average_x = measure_observable ? mean(real.(expect(cache, observables))) : NaN
        return cvm_freenergy(cache), average_x,
               marginal_inconsistency(cache)
    end
    warnings = String(take!(warning_buffer))
    status = isempty(strip(warnings)) ? "ok" :
             (occursin("did not converge", warnings) ? "not_converged" : "warning")
    if method === :cycle && status == "ok" && isfinite(values[3]) && values[3] > 1.0e-10
        status = "inconsistent"
    end
    return values, status
end

function env_reference(name, default)
    value = get(ENV, name, "")
    return isempty(value) ? default : parse(Float64, value)
end

function prepare_variant(state, preconditioner; bp_maxiter, bp_tolerance,
                         vidal_regularization)
    preconditioner === :raw && return state, 0.0, "ok"
    prepared = nothing
    elapsed = @elapsed prepared = symmetric_gauge(
        state; cache_update_kwargs = (; maxiter = bp_maxiter,
                                       tolerance = bp_tolerance,
                                       verbose = false),
        regularization = vidal_regularization)
    return prepared, elapsed, "ok"
end

function safe_prepare_variant(state, preconditioner; kwargs...)
    try
        return prepare_variant(state, preconditioner; kwargs...)
    catch error
        return nothing, NaN, replace(sprint(showerror, error), ',' => ';')
    end
end

function safe_method_values(state, chi, method, sites)
    state === nothing && return (NaN, NaN, NaN), NaN, "preparation failed"
    try
        result = nothing
        elapsed = @elapsed result = method_values(state, chi, method, sites)
        values, status = result
        return values, elapsed, status
    catch error
        return (NaN, NaN, NaN), NaN,
               replace(sprint(showerror, error), ',' => ';')
    end
end

function emit_row(io; chi, method, preconditioner, attack, logk, max_kappa,
                  pair_residual, values, baseline, references,
                  vidal_regularization, preparation_seconds, contraction_seconds, status)
    F, X, marginal = values
    F0, X0, _ = baseline
    Fref, Xref = references
    row = (chi, method, preconditioner, attack, logk, max_kappa, pair_residual,
           F, abs(F - F0), abs(F - Fref), X, abs(X - X0), abs(X - Xref),
           marginal, vidal_regularization, preparation_seconds, contraction_seconds, status)
    println(io, join(row, ','))
    io === stdout || println(stdout, join(row, ','))
    flush(io)
end

function main_gauge()
    log_condition = parse(Float64, get(ENV, "ISING9_GAUGE_LOG10_KAPPA", "1"))
    gauge_seeds = parse_ints(get(ENV, "ISING9_GAUGE_SEEDS", "1234,5678"))
    length(gauge_seeds) == 2 || error("ISING9_GAUGE_SEEDS must contain exactly two seeds")
    check_gauge_identity([log_condition])

    state = build_state(load_peps())
    sites = collect(vertices(graph(state))) # Mike's observable is the 81-site mean X.
    chis = parse_ints(get(ENV, "ISING9_GAUGE_CHIS", "4,8,12,16,20,24,28,32"))
    methods = parse_methods(get(ENV, "ISING9_GAUGE_METHODS", "cut,cycle,bmps"))
    references = (env_reference("ISING9_REFERENCE_F", ISING9_DEFAULT_F_REFERENCE),
                  env_reference("ISING9_REFERENCE_X", ISING9_DEFAULT_X_REFERENCE))
    bp_maxiter = parse(Int, get(ENV, "ISING9_GAUGE_BP_MAXITER", "80"))
    bp_tolerance = parse(Float64, get(ENV, "ISING9_GAUGE_BP_TOLERANCE", "1e-10"))
    vidal_regularization = parse(Float64,
        get(ENV, "ISING9_GAUGE_VIDAL_REGULARIZATION", string(10eps(Float64))))

    # Prepare every representation once; method/chi scans must see identical tensors. `symmetric`
    # is the Vidal/BP gauge of the original tensors, matching the convention used in the paper.
    identity = (attack = "identity", logk = NaN, state,
                pair_residual = 0.0, max_kappa = 1.0)
    random_variants = []
    for (number, seed) in enumerate(gauge_seeds)
        attacked, pair_residual, max_kappa = apply_gauge_attack(
            state, log_condition; seed)
        push!(random_variants, (attack = "random_$(number == 1 ? "a" : "b")",
                                logk = log_condition, state = attacked,
                                pair_residual, max_kappa))
    end
    vidal_identity_state, _, _ =
        safe_prepare_variant(identity.state, :vidal; bp_maxiter, bp_tolerance,
                             vidal_regularization)
    symmetric_source = identity
    symmetric_state = vidal_identity_state
    symmetric_seconds = 0.0
    symmetric_status = "ok"

    output = get(ENV, "ISING9_GAUGE_OUTPUT", "")
    io = isempty(output) ? stdout : open(output, "w")
    try
        println(io, "chi,method,preconditioner,attack,log10_condition,max_condition," *
                    "gauge_pair_residual,F,delta_F,F_abs_error,X,delta_X,X_abs_error," *
                    "marginal_inconsistency,vidal_regularization,preparation_seconds," *
                    "contraction_seconds,status")
        for chi in chis, method in methods
            baseline, baseline_seconds, baseline_status = safe_method_values(
                identity.state, chi, method, sites)
            emit_row(io; chi, method, preconditioner = :raw, attack = identity.attack,
                     logk = identity.logk, max_kappa = identity.max_kappa,
                     pair_residual = identity.pair_residual, values = baseline,
                     baseline, references, preparation_seconds = 0.0,
                     vidal_regularization, contraction_seconds = baseline_seconds,
                     status = baseline_status)

            for variant in random_variants
                values, elapsed, method_status = safe_method_values(
                    variant.state, chi, method, sites)
                emit_row(io; chi, method, preconditioner = :raw, attack = variant.attack,
                         logk = variant.logk, max_kappa = variant.max_kappa,
                         pair_residual = variant.pair_residual, values, baseline, references,
                         preparation_seconds = 0.0,
                         vidal_regularization, contraction_seconds = elapsed,
                         status = method_status)
            end

            vidal_baseline, _, vidal_baseline_status = safe_method_values(
                vidal_identity_state, chi, method, sites)
            values, elapsed, symmetric_method_status = safe_method_values(
                symmetric_state, chi, method, sites)
            status = symmetric_status != "ok" ? symmetric_status :
                     (vidal_baseline_status != "ok" ? vidal_baseline_status :
                      symmetric_method_status)
            source = symmetric_source
            emit_row(io; chi, method, preconditioner = :vidal,
                     attack = "symmetric", logk = source.logk,
                     max_kappa = source.max_kappa, pair_residual = source.pair_residual,
                     values, baseline = vidal_baseline, references,
                     preparation_seconds = symmetric_seconds,
                     vidal_regularization, contraction_seconds = elapsed, status)
        end
    finally
        io === stdout || close(io)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_gauge()
