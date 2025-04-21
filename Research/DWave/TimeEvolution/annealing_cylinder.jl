using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

include("../utils.jl")

using Random
using Serialization

using JLD2

using Base.Threads
using MKL
using LinearAlgebra
using NPZ

using ITensors
const IT = ITensors

using Dictionaries: Dictionary, set!

BLAS.set_num_threads(min(12, Sys.CPU_THREADS))
println("Julia is using "*string(nthreads()))
println("BLAS is using "*string(BLAS.get_num_threads()))
@show BLAS.get_config()

IT.disable_warn_order()

function main()
    # radius = parse(Int64, ARGS[1])
    # disorder_no = parse(Int64, ARGS[2])
    # annealing_time = parse(Int64, ARGS[3])
    # χ, cutoff, χ_trunc = parse(Int64, ARGS[4]), 1e-12, parse(Int64, ARGS[5])

    radius = 4
    disorder_no = 1
    annealing_time = 1
    χ, cutoff, χ_trunc = 4, 1e-12, 4

    height = radius
    g = named_cylinder(radius, height)
    Random.seed!(1234)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(v -> "X-", s)
    dβs = [(10, 0.25), (100, 0.1), (200, 0.01), (200, 0.005), (200, 0.001)]

    precision = 256

    δt = 0.01
    nsteps = Int64(annealing_time / δt) + 1

    χ_GS = 4

    set_global_bp_update_kwargs!(maxiter = 15, tol = 1e-10, message_update_kwargs = (; message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))))
    apply_kwargs = (maxdim = χ, cutoff, normalize = true)
    gs_apply_kwargs = (maxdim = χ_GS, cutoff, normalize = true)
    ψIψ = build_bp_cache(ψ)

    disorder_no_str = string(disorder_no - 1, pad = 2)
    instance_file = "/mnt/home/jtindall/ceph/Data/DWave/Instances/2d_($(radius), $(radius))_precision$(precision)/seed"*disorder_no_str*".npz"
    J_dict = couplings_to_edge_dict(g, radius, npzread(instance_file))
    h_dict = Dictionary([v for v in vertices(g)], [1 for v in vertices(g)])
    k = maximum([degree(g, v) for v in vertices(g)])
    ecs = edge_color(g, k)
    ordered_edges = reduce(vcat, ecs)
    t = 0


    println("Performing imaginary time evo.")
    Gamma_s_init, J_s_init = annealing_schedule(t, annealing_time)
    for (nGSsteps, dβ) in dβs
        layer = []
        append!(layer, ("Rx", [v], -1.0 * im * dβ * Gamma_s_init * h_dict[v]) for v in vertices(g))
        append!(layer, ("Rzz", pair, -2.0 * J_s_init * im * dβ * J_dict[NamedEdge(first(pair) => last(pair))]) for pair in ordered_edges)
        append!(layer, ("Rx", [v], -1.0 * im * dβ * Gamma_s_init * h_dict[v]) for v in vertices(g))
        for i in 1:nGSsteps
            ψ, ψIψ, errs = apply(layer, ψ, ψIψ; apply_kwargs = gs_apply_kwargs, update_cache = false)
            ψIψ = updatecache(ψIψ)
        end
    end
    s = siteinds(ψ)
    ψIψ = build_bp_cache(ψ)
    ψ, ψIψ  = normalize(ψ, ψIψ; update_cache = false)

    sf = 1.0 * pi

    t_start = time()

    s_cutoff = 0.6

    flush(stdout)
    for i in 1:round(Int, s_cutoff*nsteps)
        sr = t / annealing_time
        println("Time is $t (ns), s is $sr")
        Gamma_s, J_s = annealing_schedule(t + 0.5*δt, annealing_time)
        @show (Gamma_s, J_s)

        Gamma_s, J_s = sf * Gamma_s, sf * J_s

        layer = []
        append!(layer, ("Rx", [v], δt * Gamma_s * h_dict[v]) for v in vertices(g))
        append!(layer, ("Rzz", pair, 2 * J_s * δt * J_dict[NamedEdge(first(pair) => last(pair))]) for pair in ordered_edges)
        append!(layer, ("Rx", [v], δt * Gamma_s * h_dict[v]) for v in vertices(g))

        ψ, ψIψ, errs = apply(layer, ψ, ψIψ; apply_kwargs, update_cache = false)
        ψIψ = updatecache(ψIψ)
        
        max_linkdim=  maxlinkdim(ψ)
        println("Maximum bond dimension is $max_linkdim")
        t += δt

        flush(stdout)
    end

    set_global_bp_update_kwargs!(maxiter = 30, tol = 1e-14, message_update_kwargs = (; message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))))

    ψ_trunc = ITN.truncate(ψ; maxdim = χ_trunc)
    ψIψ_trunc = build_bp_cache(ψ_trunc)
    ψ_trunc, ψIψ_trunc  = normalize(ψ_trunc, ψIψ_trunc ; update_cache = false)
    ψ_trunc = ITN.VidalITensorNetwork(ψ_trunc; cache! = Ref(ψIψ_trunc), cache_update_kwargs = (; maxiter = 0))
    ψ_trunc = ITensorNetwork(ψ_trunc)
    total_time = time() - t_start
    save("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/TruncatedWavefunctions/wfRadius$(radius)Chi$(χ)ChiTrunc$(χ_trunc)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2", Dict("Wavefunction" => ψ_trunc, "TimeTaken" => total_time))
end

main()