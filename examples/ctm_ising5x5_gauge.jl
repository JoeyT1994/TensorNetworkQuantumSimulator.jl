"""Gauge benchmark matching the paper's exactly contractible 5x5 D=3 TFIM PEPS.

The four representations are the original tensors, Vidal/BP (`symmetric_gauge`),
and two reproducible random bond gauges with condition number kappa=5 by default.
Both ln<psi|psi> and the lattice-average <X> use exact references.
"""

include("ctm_ising5x5_benchmark.jl")

using Graphs: src, dst
using LinearAlgebra: BLAS, Diagonal, I, cond, inv, opnorm, qr
using Logging: SimpleLogger, Warn, with_logger
using Random: Xoshiro
using Statistics: mean

BLAS.set_num_threads(parse(Int, get(ENV, "ISING5_BLAS_THREADS", "1")))

parse_ints(s) = parse.(Int, strip.(split(s, ',')))
parse_methods(s) = Symbol.(strip.(split(s, ',')))
parse_bool(s) = lowercase(strip(s)) in ("1", "true", "yes", "on")

function random_bond_gauge(rng, dimension, condition)
    left = Matrix(qr(randn(rng, dimension, dimension)).Q)
    right = Matrix(qr(randn(rng, dimension, dimension)).Q)
    exponents = range(-log10(condition) / 2, log10(condition) / 2; length = dimension)
    return left * Diagonal(10.0 .^ exponents) * right'
end

function apply_gauge_attack(state, condition; seed)
    attacked = copy(state)
    rng = Xoshiro(seed)
    max_pair_residual = 0.0
    max_condition = 1.0
    for edge in edges(attacked)
        u, v = src(edge), dst(edge)
        old = only(commoninds(attacked[u], attacked[v]))
        fresh = sim(old)
        G = random_bond_gauge(rng, dim(old), condition)
        Ginv = inv(G)
        TNQS.setindex_preserve!(attacked, attacked[u] * ITensor(G, old, fresh), u)
        TNQS.setindex_preserve!(attacked,
                                attacked[v] * ITensor(transpose(Ginv), old, fresh), v)
        eye = Matrix{Float64}(I, dim(old), dim(old))
        max_pair_residual = max(max_pair_residual, opnorm(G * Ginv - eye, Inf))
        max_condition = max(max_condition, cond(G))
    end
    return attacked, max_pair_residual, max_condition
end

function method_values(state, chi, method, sites)
    warnings = IOBuffer()
    values = with_logger(SimpleLogger(warnings, Warn)) do
        measure_observable = lowercase(get(ENV, "ISING5_GAUGE_OBSERVABLE", "true")) in
                             ("1", "true", "yes", "on")
        observables = measure_observable ? [("X", [site]) for site in sites] : Tuple[]
        if method === :bmps
            cache = update(BoundaryMPSCache(state, chi; partition_by = "row",
                                            gauge_state = false))
            lnZ = log(abs(real(norm_sqr(cache; alg = "boundarymps"))))
            measure_observable || return lnZ, NaN, NaN
            cache = TNQS.update_partitions(cache, sites)
            xmean = mean(real.(expect(cache, observables; alg = "boundarymps",
                                      bmps_messages_up_to_date = true)))
            return lnZ, xmean, NaN
        end
        cycle_subspace = method === :cycle &&
                         parse_bool(get(ENV, "ISING5_CYCLE_SUBSPACE", "false"))
        cache = update(CTMEnvironmentCache(
                           state, chi; projector = method,
                           cycle_subspace,
                           cycle_iters = parse(Int, get(ENV, "ISING5_CYCLE_ITERS", "20")),
                           cycle_warmstart = parse_bool(get(
                               ENV, "ISING5_CYCLE_WARMSTART", "true")));
                       convergence = :environment, tolerance = 1e-10, maxiter = 80)
        xmean = measure_observable ? mean(real.(expect(cache, observables))) : NaN
        return cvm_freenergy(cache), xmean,
               marginal_inconsistency(cache)
    end
    warning_text = String(take!(warnings))
    status = isempty(strip(warning_text)) ? "ok" :
             (occursin("did not converge", warning_text) ? "not_converged" : "warning")
    return values, status
end

function lossless_references(state, sites)
    # A five-site double-layer boundary has maximal central Schmidt rank 9^2 = 81.
    # Reuse that one lossless boundary for all 25 observables instead of launching 25
    # independent exact 2D contractions.
    cache = update(BoundaryMPSCache(state, 81; partition_by = "row", gauge_state = false))
    lnZ = log(abs(real(norm_sqr(cache; alg = "boundarymps"))))
    abs(lnZ - (-6.217866847854575)) < 1e-12 ||
        error("chi=81 boundary reference is not lossless: lnZ=$lnZ")
    cache = TNQS.update_partitions(cache, sites)
    observables = [("X", [site]) for site in sites]
    xmean = mean(real.(expect(cache, observables; alg = "boundarymps",
                              bmps_messages_up_to_date = true)))
    return lnZ, xmean
end

function main_gauge()
    state = build_state(load_peps())
    sites = collect(vertices(graph(state)))
    condition = parse(Float64, get(ENV, "ISING5_GAUGE_KAPPA", "5"))
    seeds = parse_ints(get(ENV, "ISING5_GAUGE_SEEDS", "1234,5678"))
    length(seeds) == 2 || error("ISING5_GAUGE_SEEDS must contain exactly two seeds")
    measure_observable = lowercase(get(ENV, "ISING5_GAUGE_OBSERVABLE", "true")) in
                         ("1", "true", "yes", "on")
    Fref, Xref = measure_observable ? lossless_references(state, sites) :
                                      (-6.217866847854575, NaN)

    variants = [(name = "original", state, condition = 1.0, pair_residual = 0.0)]
    vidal = symmetric_gauge(state; cache_update_kwargs = (; maxiter = 80,
                                                            tolerance = 1e-10,
                                                            verbose = false))
    push!(variants, (name = "vidal_bp", state = vidal, condition = 1.0,
                     pair_residual = 0.0))
    for (i, seed) in enumerate(seeds)
        gauged, residual, actual_condition = apply_gauge_attack(state, condition; seed)
        residual <= 1e-12 || error("paired gauge residual $residual")
        push!(variants, (name = "random_g$i", state = gauged,
                         condition = actual_condition, pair_residual = residual))
    end
    if parse_bool(get(ENV, "ISING5_CYCLE_PRECONDITION_SYMMETRIC", "false"))
        precondition_maxiter = parse(Int, get(
            ENV, "ISING5_CYCLE_PRECONDITION_MAXITER", "80"))
        precondition_tolerance = parse(Float64, get(
            ENV, "ISING5_CYCLE_PRECONDITION_TOLERANCE", "1e-10"))
        variants = map(variants) do variant
            gauged_state = symmetric_gauge(
                variant.state; cache_update_kwargs = (; maxiter = precondition_maxiter,
                                                        tolerance = precondition_tolerance,
                                                        verbose = false))
            merge(variant, (; state = gauged_state))
        end
    end

    output = get(ENV, "ISING5_GAUGE_OUTPUT", "")
    io = isempty(output) ? stdout : open(output, "w")
    try
        println(io, "chi,method,gauge,max_condition,gauge_pair_residual,F,F_abs_error," *
                    "Xmean,Xmean_abs_error,marginal_inconsistency,status")
        for chi in parse_ints(get(ENV, "ISING5_GAUGE_CHIS", "4,8,12,16,20,24,28,32")),
            method in parse_methods(get(ENV, "ISING5_GAUGE_METHODS", "cut,cycle,bmps")),
            variant in variants
            (F, X, marginal), status = method_values(variant.state, chi, method, sites)
            println(io, join((chi, method, variant.name, variant.condition,
                              variant.pair_residual, F, abs(F - Fref), X,
                              abs(X - Xref), marginal, status), ','))
            io === stdout || println(stdout, join((chi, method, variant.name, F,
                                                   X, marginal, status), ','))
            flush(io)
        end
    finally
        io === stdout || close(io)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_gauge()
