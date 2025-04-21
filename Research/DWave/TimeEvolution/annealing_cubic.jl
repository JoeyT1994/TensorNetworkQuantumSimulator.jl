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
    nx = parse(Int64, ARGS[1])
    ny = parse(Int64, ARGS[2])
    nz = parse(Int64, ARGS[3])
    disorder_no = parse(Int64, ARGS[4])
    annealing_time = parse(Int64, ARGS[5])
    χ, cutoff = parse(Int64, ARGS[6]), 1e-12
    # nx, ny, nz = 3,2,2
    # disorder_no = 1
    # annealing_time = 0.1
    # χ, cutoff = 2, 1e-14
    

    disorder_no_str = string(disorder_no - 1, pad = 2)
    instance_file = "/mnt/home/jtindall/ceph/Data/DWave/Instances/3ddimer_($(nx), $(ny), $(nz))_precision256/seed"*disorder_no_str*".npz"
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

    dimer_edges = filter(e -> abs(J_dict[e] + 2) <= 1e-10, keys(J_dict))
    operator_inds = ITN.union_all_inds(s, prime(s))
    I = ITensorNetwork(IT.Op("I"), operator_inds; link_space = nothing)

    for e in dimer_edges
        ψ = ITN.contract(ψ, e; merged_vertex = (src(e), dst(e)))
        I = ITN.contract(I, e; merged_vertex = (src(e), dst(e)))
    end

    ψ = ITN.combine_linkinds(ψ)

    ψIψ = ITN.QuadraticFormNetwork(I, ψ)

    ψIψ_bpc = BeliefPropagationCache(ψIψ)
    ms = ITN.messages(ψIψ_bpc)
    for e in ITN.partitionpairs(ψIψ_bpc)
        me = ITN.contract(ITN.message(ψIψ_bpc, e))
        mer = ITN.contract(ITN.message(ψIψ_bpc, reverse(e)))
        set!(ms, e, ITensor[me])
        set!(ms, reverse(e), ITensor[mer])
    end
    ψIψ = updatecache(ψIψ_bpc)

    t = 0


    println("Performing imaginary time evo.")
    Gamma_s_init, J_s_init = annealing_schedule(t, annealing_time)
    for (nGSsteps, dβ) in dβs
        layer = []
        append!(layer, ITensors.op("Rx", only(s[v]); θ = -1.0 * im * dβ * Gamma_s_init * h_dict[v]) for v in vertices(g))
        append!(layer, ITensors.op("Rzz", only(s[first(pair)]), only(s[last(pair)]); ϕ = - 0.5 * 2.0 * J_s_init * im * dβ * J_dict[NamedEdge(first(pair) => last(pair))]) for pair in ordered_edges)
        append!(layer, ITensors.op("Rx", only(s[v]); θ = -1.0 * im * dβ * Gamma_s_init * h_dict[v]) for v in vertices(g))
        for i in 1:nGSsteps
            ψ, ψIψ, errs = apply(layer, ψ, ψIψ; apply_kwargs = gs_apply_kwargs, update_cache = false)
            ψIψ = updatecache(ψIψ)
        end
    end
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
        append!(layer, ITensors.op("Rx", only(s[v]); θ = δt * Gamma_s * h_dict[v]) for v in vertices(g))
        append!(layer, ITensors.op("Rzz", only(s[first(pair)]), only(s[last(pair)]); ϕ = 0.5 * 2.0 * J_s * δt * J_dict[NamedEdge(first(pair) => last(pair))]) for pair in ordered_edges)
        append!(layer, ITensors.op("Rx", only(s[v]); θ = δt * Gamma_s * h_dict[v]) for v in vertices(g))

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
    save("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Cubic/Dimerized/nx$(nx)ny$(ny)nz$(nz)Chi$(χ)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2", Dict("Wavefunction" => ψ, "TimeTaken" => total_time))
end

main()