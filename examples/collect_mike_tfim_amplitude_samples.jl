"""Collect five-seed `<s|psi>` sample scans separately from TFIM norm data.

Each integer seed defines one deterministic Xoshiro computational-basis sample.
The bitstring is deliberately not emitted: the seed is the public sample identifier.
For CTMRG the actual scalar functional is `log|<s|psi>|`; the CSV also stores its
exponentiated magnitude. References are converged bMPS values checked at chi=81,96.
"""

using TensorNetworkQuantumSimulator
using ITensors
using LinearAlgebra: BLAS
using Logging: SimpleLogger, Warn, with_logger
using Random: Xoshiro
const TNQS_MIKE_AMPLITUDE = TensorNetworkQuantumSimulator
const MIKE_AMPLITUDE_BMPS_KW = (;
    message_update_alg = TNQS_MIKE_AMPLITUDE.Algorithm("zipup"; cutoff = 0.0))

const MIKE_AMPLITUDE_SIZE = parse(Int, get(ENV, "MIKE_AMPLITUDE_SIZE", "9"))
if MIKE_AMPLITUDE_SIZE == 5
    include("ctm_ising5x5_gauge.jl")
elseif MIKE_AMPLITUDE_SIZE == 9
    include("ctm_ising9x9_benchmark.jl")
else
    error("MIKE_AMPLITUDE_SIZE must be 5 or 9")
end

mike_amplitude_ints(value) = parse.(Int, strip.(split(value, ',')))

function mike_seeded_configuration(state, seed)
    sites = collect(vertices(graph(state)))
    rng = Xoshiro(seed)
    return Dict(site => rand(rng, 0:1) for site in sites)
end

function mike_projected_network(state, configuration)
    projected = copy(TNQS_MIKE_AMPLITUDE.tensornetwork(state))
    physical = siteinds(state)
    for site in vertices(graph(state))
        basis = onehot(only(physical[site]) => configuration[site] + 1)
        TNQS_MIKE_AMPLITUDE.setindex_preserve!(
            projected, projected[site] * basis, site)
    end
    return projected
end

function mike_bmps_amplitude(network, chi)
    cache = update(BoundaryMPSCache(
        network, chi; partition_by = "row", gauge_state = false);
        MIKE_AMPLITUDE_BMPS_KW...)
    return ComplexF64(partitionfunction(cache))
end

function mike_amplitude_reference(network, ref_chis)
    values = [mike_bmps_amplitude(network, chi) for chi in ref_chis]
    reference = last(values)
    reference == 0 && error("zero reference amplitude")
    drift = abs(first(values) / reference - 1)
    drift <= 5e-14 || error(
        "amplitude references did not converge: values=$values drift=$drift")
    return abs(reference), drift
end

function mike_amplitude_value(network, chi, method)
    warning_buffer = IOBuffer()
    values = with_logger(SimpleLogger(warning_buffer, Warn)) do
        if method === :bmps
            magnitude = abs(mike_bmps_amplitude(network, chi))
            return log(magnitude), magnitude
        end
        cache = update(CTMEnvironmentCache(network, chi; projector = method);
                       convergence = :free_energy, tolerance = 1e-12, maxiter = 100)
        logabs = cvm_freenergy(cache)
        return logabs, exp(logabs)
    end
    warnings = String(take!(warning_buffer))
    status = isempty(strip(warnings)) ? "ok" :
             (occursin("did not converge", warnings) ? "not_converged" : "warning")
    return values..., status
end

function main_mike_amplitude_samples()
    BLAS.set_num_threads(parse(Int, get(ENV, "MIKE_AMPLITUDE_BLAS_THREADS", "1")))
    size = MIKE_AMPLITUDE_SIZE
    state = build_state(load_peps())
    seeds = mike_amplitude_ints(get(ENV, "MIKE_AMPLITUDE_SEEDS", "1,2,3,4,5"))
    length(unique(seeds)) == length(seeds) || error("sample seeds must be unique")
    chis = mike_amplitude_ints(get(ENV, "MIKE_AMPLITUDE_CHIS", join(1:32, ',')))
    methods = Symbol.(strip.(split(
        get(ENV, "MIKE_AMPLITUDE_METHODS", "cut,cycle,bmps"), ',')))
    all(method -> method in (:cut, :cycle, :bmps), methods) || error("invalid method")
    ref_chis = mike_amplitude_ints(get(
        ENV, "MIKE_AMPLITUDE_REFERENCE_CHIS", "81,96"))
    output = get(ENV, "MIKE_AMPLITUDE_OUTPUT", "")
    io = isempty(output) ? stdout : open(output, "w")
    try
        println(io, "linear_size,bond_dimension,sample_seed,chi,method," *
                    "logabs_amplitude_value,abs_amplitude_value," *
                    "logabs_amplitude_ground_truth,abs_amplitude_ground_truth," *
                    "logabs_amplitude_absolute_error,logabs_amplitude_relative_error," *
                    "abs_amplitude_absolute_error,abs_amplitude_relative_error," *
                    "reference_relative_drift,seconds,status")
        for seed in seeds
            configuration = mike_seeded_configuration(state, seed)
            network = mike_projected_network(state, configuration)
            reference, reference_drift = mike_amplitude_reference(network, ref_chis)
            log_reference = log(reference)
            for chi in chis, method in methods
                result = nothing
                seconds = @elapsed result = mike_amplitude_value(network, chi, method)
                logabs, magnitude, status = result
                log_absolute = abs(logabs - log_reference)
                log_relative = log_absolute / abs(log_reference)
                absolute = abs(magnitude - reference)
                relative = absolute / reference
                println(io, join((size, 3, seed, chi, method, logabs, magnitude,
                                  log_reference, reference, log_absolute, log_relative,
                                  absolute, relative, reference_drift, seconds, status), ','))
                io === stdout || println(stdout,
                    "size=$size seed=$seed chi=$chi method=$method relerr=$relative status=$status")
                flush(io)
            end
        end
    finally
        io === stdout || close(io)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_mike_amplitude_samples()
