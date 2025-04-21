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
    nx, ny, nz = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Int64, ARGS[3])
    disorder_no = parse(Int64, ARGS[4])
    annealing_time = parse(Int64, ARGS[5])
    χ, cutoff = parse(Int64, ARGS[6]), 1e-12

    # nx, ny, nz = 3,3,8
    # disorder_no = 1
    # annealing_time = 1
    # χ, cutoff = 4, 1e-12

    disorder_no_str = string(disorder_no - 1, pad = 2)
    instance_file = "/mnt/home/jtindall/ceph/Data/DWave/Instances/diamond_($(nx), $(ny), $(nz))_precision256/seed"*disorder_no_str*".npz"
    g, J_dict = graph_couplings_from_instance(instance_file)
    h_dict = Dictionary([v for v in vertices(g)], [1 for v in vertices(g)])
    k = maximum([degree(g, v) for v in vertices(g)])
    ecs = edge_color(g, k)
    ordered_edges = reduce(vcat, ecs)

    Random.seed!(1234)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(v -> "X-", s)
    dβs = [(10, 0.25), (100, 0.1), (200, 0.01), (200, 0.005), (200, 0.001)]


    δt = 0.01
    nsteps = Int64(annealing_time / δt) + 1

    χ_GS = 4

    set_global_bp_update_kwargs!(maxiter = 15, tol = 1e-10, message_update_kwargs = (; message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))))
    apply_kwargs = (maxdim = χ, cutoff, normalize = true)
    gs_apply_kwargs = (maxdim = χ_GS, cutoff, normalize = true)
    ψIψ = build_bp_cache(ψ)

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

    ψ, ψIψ  = normalize(ψ, ψIψ ; update_cache = false)
    ψ = ITN.VidalITensorNetwork(ψ; cache! = Ref(ψIψ), cache_update_kwargs = (; maxiter = 0))
    ψ = ITensorNetwork(ψ)
    total_time = time() - t_start
    save("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Diamond/nx$(nx)ny$(ny)nz$(nz)Chi$(χ)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2", Dict("Wavefunction" => ψ, "TimeTaken" => total_time))
end

main()