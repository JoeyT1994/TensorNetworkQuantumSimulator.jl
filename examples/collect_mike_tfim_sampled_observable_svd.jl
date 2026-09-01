"""Collect SVD-CTMRG data for the sampled TFIM local estimator.

For each supplied configuration ``s``, converge SVD-CTMRG on the single-layer
network ``<s|psi>`` and evaluate

    (1 / N) * sum_i <s|X_i|psi> / <s|psi>

from the local CTM environment ratios. The first samples in Yantao's ``ps.npy``
archive are used in their original order. Lossless boundary-MPS contractions give
the per-sample references.
"""

using TensorNetworkQuantumSimulator
using ITensors
using LinearAlgebra: BLAS
using Logging: SimpleLogger, Warn, with_logger
using Statistics: mean

const TNQS_SAMPLED_X = TensorNetworkQuantumSimulator
const SAMPLED_X_SIZE = parse(Int, get(ENV, "MIKE_SAMPLED_X_SIZE", "9"))

if SAMPLED_X_SIZE == 5
    include("ctm_ising5x5_benchmark.jl")
elseif SAMPLED_X_SIZE == 9
    include("ctm_ising9x9_benchmark.jl")
else
    error("MIKE_SAMPLED_X_SIZE must be 5 or 9")
end

const SAMPLED_X_BMPS_KW = (;
    message_update_alg = TNQS_SAMPLED_X.Algorithm("zipup"; cutoff = 0.0))

sampled_x_ints(value) = parse.(Int, strip.(split(value, ',')))

"""Read the C-ordered ``int64`` NumPy array used for the supplied samples."""
function read_npy_samples(path, expected_size)
    open(path, "r") do io
        read(io, 6) == UInt8[0x93, codeunits("NUMPY")...] ||
            error("$path is not an NPY file")
        major, minor = read(io, UInt8), read(io, UInt8)
        header_length = major == 1 ? Int(read(io, UInt16)) :
                        major in (2, 3) ? Int(read(io, UInt32)) :
                        error("unsupported NPY version $major.$minor")
        header = String(read(io, header_length))
        occursin(r"['\"]descr['\"]\s*:\s*['\"]<i8['\"]", header) ||
            error("expected little-endian int64 samples; header=$header")
        occursin(r"['\"]fortran_order['\"]\s*:\s*False", header) ||
            error("expected C-ordered samples; header=$header")
        shape_match = match(r"['\"]shape['\"]\s*:\s*\(([^)]*)\)", header)
        isnothing(shape_match) && error("could not parse NPY shape from $header")
        dims = parse.(Int, filter(!isempty, strip.(split(shape_match.captures[1], ','))))
        length(dims) == 3 || error("expected a rank-3 sample array; got $dims")
        dims[1:2] == [expected_size, expected_size] ||
            error("sample lattice $(dims[1:2]) does not match $expected_size x $expected_size")
        data = read!(io, Vector{Int64}(undef, prod(dims)))
        eof(io) || error("unexpected trailing bytes in $path")
        # NumPy advances the final index first. Reverse once for Julia's column-major
        # reshape, then restore the requested (x, y, sample) axis order.
        samples = permutedims(reshape(data, Tuple(reverse(dims))), (3, 2, 1))
        all(value -> value in (0, 1), samples) || error("samples must be binary")
        return samples
    end
end

function projected_sample(state, samples, sample_index)
    projected = copy(TNQS_SAMPLED_X.tensornetwork(state))
    flipped = Dict{Any, ITensor}()
    physical = siteinds(state)
    for site in vertices(graph(state))
        x, y = site
        spin = samples[x, y, sample_index]
        index = only(physical[site])
        tensor = state[site]
        TNQS_SAMPLED_X.setindex_preserve!(
            projected, tensor * onehot(index => spin + 1), site)
        flipped[site] = tensor * onehot(index => 2 - spin)
    end
    return projected, flipped
end

function local_scalar(tensors)
    sequence = TNQS_SAMPLED_X.contraction_sequence(tensors; alg = "optimal")
    return scalar(TNQS_SAMPLED_X.contract(tensors; sequence))
end

function bmps_values(network, flipped, chi, sites)
    cache = update(BoundaryMPSCache(
        network, chi; partition_by = "row", gauge_state = false);
        SAMPLED_X_BMPS_KW...)
    cache = TNQS_SAMPLED_X.update_partitions(cache, sites)
    ratios = ComplexF64[]
    for site in sites
        incoming = TNQS_SAMPLED_X.incoming_messages(cache, site)
        denominator = local_scalar(ITensor[network[site]; incoming])
        numerator = local_scalar(ITensor[flipped[site]; incoming])
        push!(ratios, numerator / denominator)
    end
    amplitude = ComplexF64(partitionfunction(cache))
    return amplitude, mean(ratios), maximum(abs ∘ imag, ratios)
end

function reference_values(network, flipped, sites, chis)
    values = [bmps_values(network, flipped, chi, sites) for chi in chis]
    amplitude, mean_x, imag_residual = last(values)
    amplitude == 0 && error("zero reference amplitude")
    amplitude_drift = abs(first(values)[1] / amplitude - 1)
    mean_x_drift = abs(first(values)[2] - mean_x)
    amplitude_drift <= 5e-13 || error(
        "amplitude references did not converge: values=$values drift=$amplitude_drift")
    mean_x_drift <= 5e-12 || error(
        "local-estimator references did not converge: values=$values drift=$mean_x_drift")
    return amplitude, mean_x, max(imag_residual, first(values)[3]),
           amplitude_drift, mean_x_drift
end

function ctm_local_estimator(cache, network, flipped, sites)
    ratios = ComplexF64[]
    for site in sites
        ring = vertex_ring(cache, site)
        denominator = scalar(TNQS_SAMPLED_X._ctm_contract(
            ITensor[network[site]; ring], cache.options))
        numerator = scalar(TNQS_SAMPLED_X._ctm_contract(
            ITensor[flipped[site]; ring], cache.options))
        push!(ratios, numerator / denominator)
    end
    return mean(ratios), maximum(abs ∘ imag, ratios)
end

function svd_ctm_values(network, flipped, chi, sites; tolerance, maxiter)
    warning_buffer = IOBuffer()
    values = with_logger(SimpleLogger(warning_buffer, Warn)) do
        cache = update(CTMEnvironmentCache(
            network, chi; projector = :cut, gauge_state = false);
            convergence = :environment, tolerance, maxiter)
        mean_x, imag_residual = ctm_local_estimator(
            cache, network, flipped, sites)
        return cvm_freenergy(cache), mean_x, imag_residual,
               marginal_inconsistency(cache)
    end
    warnings = String(take!(warning_buffer))
    status = isempty(strip(warnings)) ? "ok" :
             (occursin("did not converge", warnings) ? "not_converged" : "warning")
    return values..., status
end

function main_sampled_x()
    BLAS.set_num_threads(parse(Int, get(ENV, "MIKE_SAMPLED_X_BLAS_THREADS", "1")))
    linear_size = SAMPLED_X_SIZE
    state = build_state(load_peps())
    sites = collect(vertices(graph(state)))
    default_sample_path = joinpath(
        @__DIR__, "data", "peps", "data_ising_$(linear_size)x$(linear_size)", "ps.npy")
    sample_path = get(ENV, "MIKE_SAMPLED_X_SAMPLE_PATH", default_sample_path)
    samples = read_npy_samples(sample_path, linear_size)
    nsamples = parse(Int, get(ENV, "MIKE_SAMPLED_X_NSAMPLES", "15"))
    1 <= nsamples <= size(samples, 3) || error(
        "requested $nsamples samples but archive contains $(size(samples, 3))")
    chis = sampled_x_ints(get(ENV, "MIKE_SAMPLED_X_CHIS", join(1:10, ',')))
    reference_chis = sampled_x_ints(get(
        ENV, "MIKE_SAMPLED_X_REFERENCE_CHIS", "81,96"))
    tolerance = parse(Float64, get(ENV, "MIKE_SAMPLED_X_TOLERANCE", "1e-12"))
    maxiter = parse(Int, get(ENV, "MIKE_SAMPLED_X_MAXITER", "100"))
    output = get(ENV, "MIKE_SAMPLED_X_OUTPUT", "")
    io = isempty(output) ? stdout : open(output, "w")
    source_note = "Yantao ps.npy first $nsamples of $(size(samples, 3)) samples"
    try
        println(io, "problem,method,sample_index,archive_sample_index,chi," *
                    "logabs_amplitude_value,logabs_amplitude_ground_truth," *
                    "logabs_amplitude_absolute_error,logabs_amplitude_relative_error," *
                    "meanX_local_estimator_value,meanX_local_estimator_ground_truth," *
                    "meanX_absolute_error,meanX_relative_error,max_imaginary_residual," *
                    "reference_logabs_relative_drift,reference_meanX_absolute_drift," *
                    "marginal_inconsistency,seconds,status,convergence_criterion," *
                    "tolerance,maxiter,reference_method,reference_chi,sample_source")
        for sample_index in 1:nsamples
            network, flipped = projected_sample(state, samples, sample_index)
            reference, mean_x_reference, reference_imaginary,
                reference_amplitude_drift, reference_mean_x_drift =
                reference_values(network, flipped, sites, reference_chis)
            log_reference = log(abs(reference))
            for chi in chis
                result = nothing
                seconds = @elapsed result = svd_ctm_values(
                    network, flipped, chi, sites; tolerance, maxiter)
                logabs, mean_x, imag_residual, marginal, status = result
                mean_x_real = real(mean_x)
                mean_x_reference_real = real(mean_x_reference)
                log_absolute_error = abs(logabs - log_reference)
                log_relative_error = log_absolute_error / abs(log_reference)
                mean_x_absolute_error = abs(mean_x_real - mean_x_reference_real)
                mean_x_relative_error = mean_x_absolute_error / abs(mean_x_reference_real)
                row = ("D=3 $(linear_size)x$(linear_size) TFIM sampled X local estimator",
                       "svd-CTMRG", sample_index, sample_index - 1, chi,
                       logabs, log_reference, log_absolute_error, log_relative_error,
                       mean_x_real, mean_x_reference_real, mean_x_absolute_error,
                       mean_x_relative_error, max(imag_residual, reference_imaginary),
                       reference_amplitude_drift, reference_mean_x_drift,
                       marginal, seconds, status, "environment", tolerance, maxiter,
                       "lossless bMPS", join(reference_chis, '|'), source_note)
                println(io, join(row, ','))
                io === stdout || println(stdout,
                    "size=$linear_size sample=$sample_index chi=$chi " *
                    "X=$(mean_x_real) Xerr=$mean_x_absolute_error status=$status")
                flush(io)
            end
        end
    finally
        io === stdout || close(io)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_sampled_x()
