"""SVD-CTMRG convergence of random 9x9 TFIM PEPS amplitudes `<s|psi>`.

Five reproducible uniform computational-basis configurations are used by default. A
single-layer 9-site boundary has maximum Schmidt rank `3^4 = 81`, so bMPS chi=81 is
lossless; chi=96 is also evaluated and required to agree before the CTMRG scan starts.
"""

include("ctm_ising9x9_benchmark.jl")

using LinearAlgebra: BLAS
using Random: Xoshiro

const BMPS_KW = (; message_update_alg = TNQS.Algorithm("zipup"; cutoff = 0.0))

function random_configurations(state, nsamples, seed)
    sites = collect(vertices(graph(state)))
    rng = Xoshiro(seed)
    return sites, [Dict(site => rand(rng, 0:1) for site in sites) for _ in 1:nsamples]
end

function projected_network(state, configuration)
    projected = copy(TNQS.tensornetwork(state))
    physical = siteinds(state)
    for site in vertices(graph(state))
        basis = onehot(only(physical[site]) => configuration[site] + 1)
        TNQS.setindex_preserve!(projected, projected[site] * basis, site)
    end
    return projected
end

function bmps_amplitude(network, chi)
    cache = BoundaryMPSCache(network, chi; partition_by = "row", gauge_state = false)
    cache = update(cache; BMPS_KW...)
    return partitionfunction(cache)
end

bitstring(configuration, sites) = join(configuration[site] for site in sites)
parse_ints(s) = parse.(Int, strip.(split(s, ',')))

function main_amplitudes()
    BLAS.set_num_threads(parse(Int, get(ENV, "ISING9_BLAS_THREADS", "1")))
    state = build_state(load_peps())
    nsamples = parse(Int, get(ENV, "ISING9_AMPLITUDE_SAMPLES", "5"))
    seed = parse(Int, get(ENV, "ISING9_AMPLITUDE_SEED", "9142026"))
    chis = parse_ints(get(ENV, "ISING9_AMPLITUDE_CHIS", "2,4,6,8,10,12,16,20,24,32,40,48"))
    ref_chis = parse_ints(get(ENV, "ISING9_AMPLITUDE_REFERENCE_CHIS", "81,96"))
    sites, configurations = random_configurations(state, nsamples, seed)

    output = get(ENV, "ISING9_AMPLITUDE_OUTPUT", "")
    io = isempty(output) ? stdout : open(output, "w")
    try
        println(io, "sample,bitstring,chi,method,logabs_amplitude,relative_error," *
                    "reference_logabs,reference_amplitude,seconds,status")
        for (sample, configuration) in enumerate(configurations)
            network = projected_network(state, configuration)
            references = ComplexF64[]
            for chi in ref_chis
                amplitude = nothing
                seconds = @elapsed amplitude = bmps_amplitude(network, chi)
                push!(references, ComplexF64(amplitude))
                println(io, join((sample, bitstring(configuration, sites), chi, "bmps_reference",
                                  log(abs(amplitude)), NaN, NaN, amplitude, seconds, "ok"), ','))
                io === stdout || println(stdout, "sample=$sample reference chi=$chi " *
                                                "log|amp|=$(log(abs(amplitude)))")
                flush(io)
            end
            reference = last(references)
            reference == 0 && error("zero reference amplitude for sample $sample")
            rel_reference_drift = abs(first(references) / reference - 1)
            rel_reference_drift <= 5e-14 || error(
                "bMPS references did not converge for sample $sample: drift=$rel_reference_drift")

            for chi in chis
                cache = nothing
                seconds = @elapsed cache = update(
                    CTMEnvironmentCache(network, chi; projector = :cut);
                    convergence = :free_energy, tolerance = 1e-12, maxiter = 100)
                logabs = cvm_freenergy(cache)
                relative_error = abs(expm1(logabs - log(abs(reference))))
                println(io, join((sample, bitstring(configuration, sites), chi, "cut", logabs,
                                  relative_error, log(abs(reference)), reference, seconds, "ok"), ','))
                io === stdout || println(stdout, "sample=$sample chi=$chi relerr=$relative_error")
                flush(io)
            end
        end
    finally
        io === stdout || close(io)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_amplitudes()
