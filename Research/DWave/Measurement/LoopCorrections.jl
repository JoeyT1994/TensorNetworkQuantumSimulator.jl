using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics
using NamedGraphs: AbstractNamedGraph

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

function get_job_corrs(g::AbstractNamedGraph, job_no::Int64, no_corrs_per_job::Int64)
    job_v1_v2s = []
    vs = collect(vertices(g))
    nvs = length(vs)
    count = 0
    for (i, v1) in enumerate(1:nvs)
        for (j, v2) in enumerate((i+1):nvs)
            if count >= (job_no - 1)*no_corrs_per_job
                push!(job_v1_v2s, (v1, v2))
            end
            count += 1
            length(job_v1_v2s) == no_corrs_per_job && return job_v1_v2s
        end
    end
    return job_v1_v2s
end

function main_cylinder()
    # radius = parse(Int64, ARGS[1])
    # max_loop_length = parse(Int64, ARGS[2])
    # disorder_no = parse(Int64, ARGS[3])
    # annealing_time = parse(Int64, ARGS[4])
    # χ_state = parse(Int64, ARGS[5])
    # χ_state_truncate = parse(Int64, ARGS[6])
    # v1col, v1row = parse(Int64, ARGS[7]), parse(Int64, ARGS[8])

    radius = 4
    max_loop_length = 7
    disorder_no = 1
    annealing_time = 20
    χ_state = 32
    χ_state_truncate = 10
    v1col, v1row = 1, 1
    v1 = (v1col, v1row)

    f = load("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/TruncatedWavefunctions/wfRadius$(radius)Chi$(χ_state)ChiTrunc$(χ_state_truncate)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2")
    ψ = f["Wavefunction"]


    set_global_bp_update_kwargs!(; maxiter =50, tol = 1e-14, message_update_kwargs = (; message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))))
    ψIψ = build_bp_cache(ψ)
    ψ, ψIψ = normalize(ψ, ψIψ; update_cache = false)
    ψ = ITN.VidalITensorNetwork(ψ; cache! = Ref(ψIψ), cache_update_kwargs = (; maxiter = 0))
    ψ = ITensorNetwork(ψ)

    vs = collect(vertices(ψ))
    v1_pos = findfirst(v -> v == v1, vs)
    v1v2s = [(v1, vs[v2_ind]) for v2_ind in (v1_pos+1):length(vs)]
    
    egs = NG.edgeinduced_subgraphs_no_leaves(ITN.partitioned_graph(ψIψ), max_loop_length)
    circuit_lengths = vcat([0], sort(unique(length.(edges.(egs)))))
    corrs = zeros(ComplexF64, (length(v1v2s), length(circuit_lengths)))

    @show length(egs)
    @show length.(edges.(egs))
    @show v1v2s

    for (i, (v1, v2)) in enumerate(v1v2s)
        println("Computing Correlations No. $i")
        corrs[i, :] = zz_correlation_bp_loopcorrectfull(ψ, v1, v2, egs)
        flush(stdout)
    end

    for (i, cl) in enumerate(circuit_lengths)
        cs = corrs[:, i]
        corr_dict=  Dictionary()
        for (j, (v1, v2)) in enumerate(v1v2s)
            set!(corr_dict, NamedEdge(v1 => v2), cs[j])
        end
        @show corr_dict
        save("/mnt/home/jtindall/ceph/Data/DWave/Corrs/BPCorrected/Cylinder/LoopLength$(cl)CorrsRadius$(radius)AnnealingTime$(annealing_time)Chi$(χ_state)DisorderNo$(disorder_no)/vc$(v1col)vr$(v1row).jld2",
        "corrs", corr_dict)
    end
end

    
function main_diamond()
    println("Diamond Regular analysis")
    nx, ny, nz = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Int64, ARGS[3])
    max_loop_length = parse(Int64, ARGS[4])
    disorder_no = parse(Int64, ARGS[5])
    annealing_time = parse(Int64, ARGS[6])
    χ_state = parse(Int64, ARGS[7])
    job_no = parse(Int64, ARGS[8])
    no_corrs_per_job = parse(Int64, ARGS[9])

    # nx, ny, nz = 4,4,8
    # max_loop_length = 6
    # disorder_no = 1
    # annealing_time = 20
    # χ_state = 16
    # job_no = 13

    # no_corrs_per_job = 10


    f = nothing
    try
        f= load("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Diamond/nx$(nx)ny$(ny)nz$(nz)Chi$(χ_state)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2")
    catch
        f= load("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Diamond/nx$(nx)ny$(ny)nz$(nz)Chi$(χ_state)DisorderNo$(disorder_no)AnnealingTime$(annealing_time)Cutoff.jld2")
    end
    ψ = f["Wavefunction"]

    set_global_bp_update_kwargs!(; maxiter =50, tol = 1e-14, message_update_kwargs = (; message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))))
    ψIψ = build_bp_cache(ψ)
    ψ, ψIψ = normalize(ψ, ψIψ; update_cache = false)
    ψ = ITN.VidalITensorNetwork(ψ; cache! = Ref(ψIψ), cache_update_kwargs = (; maxiter = 0))
    ψ = ITensorNetwork(ψ)

    v1v2s = get_job_corrs(ITN.underlying_graph(ψ), job_no, no_corrs_per_job)
    
    egs = NG.edgeinduced_subgraphs_no_leaves(ITN.partitioned_graph(ψIψ), max_loop_length)
    circuit_lengths = vcat([0], sort(unique(length.(edges.(egs)))))
    corrs = zeros(ComplexF64, (length(v1v2s), length(circuit_lengths)))

    @show length(egs)
    @show length.(edges.(egs))
    @show v1v2s

    for (i, (v1, v2)) in enumerate(v1v2s)
        println("Computing Correlations No. $i")
        corrs[i, :] = zz_correlation_bp_loopcorrectfull(ψ, v1, v2, egs)
        flush(stdout)
    end

    for (i, cl) in enumerate(circuit_lengths)
        cs = corrs[:, i]
        corr_dict=  Dictionary()
        for (j, (v1, v2)) in enumerate(v1v2s)
            set!(corr_dict, NamedEdge(v1 => v2), cs[j])
        end
        save("/mnt/home/jtindall/ceph/Data/DWave/Corrs/BPCorrected/DiamondErrorAnalysis/LoopLength$(cl)Corrsnx$(nx)ny$(ny)nz$(nz)AnnealingTime$(annealing_time)Chi$(χ_state)DisorderNo$(disorder_no)/JobNo$(job_no)NCorrs$(no_corrs_per_job).jld2",
        "corrs", corr_dict)
    end
end

function main_cubic()
    nx, ny, nz = parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Int64, ARGS[3])
    max_loop_length = parse(Int64, ARGS[4])
    disorder_no = parse(Int64, ARGS[5])
    annealing_time = parse(Int64, ARGS[6])
    χ_state = parse(Int64, ARGS[7])
    v1 = parse(Int64, ARGS[8])

    # nx, ny, nz = 3,2,2
    # max_loop_length = 4
    # disorder_no =1
    # annealing_time =20
    # χ_state = 8
    # v1 = 16

    f = nothing
    try
        f= load("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Cubic/Dimerized/nx$(nx)ny$(ny)nz$(nz)Chi$(χ_state)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2")
    catch
        f= load("/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Cubic/Dimerized/nx$(nx)ny$(ny)nz$(nz)Chi$(χ_state)DisorderNo$(disorder_no)AnnealingTime$(annealing_time)Cutoff.jld2")
    end
    ψ = f["Wavefunction"]
    old_sinds = f["OldSinds"]

    set_global_bp_update_kwargs!(; maxiter =50, tol = 1e-14, message_update_kwargs = (; message_update_function = ms -> make_eigs_real.(ITN.default_message_update(ms))))

    ψIψ = BeliefPropagationCache(_quadraticformnetwork(ψ))
    ψIψ = updatecache(ψIψ)
    ψ, ψIψ = normalize(ψ, ψIψ; update_cache = false)

    no_vs = nv(old_sinds) - v1 + 1

    egs = NG.edgeinduced_subgraphs_no_leaves(ITN.partitioned_graph(ψIψ), max_loop_length)
    circuit_lengths = vcat([0], sort(unique(length.(edges.(egs)))))
    corrs = zeros(ComplexF64, (no_vs, length(circuit_lengths)))

    for (i, v2) in enumerate((v1+1):(nv(old_sinds)))
        corrs[i, :] = zz_correlation_bp_loopcorrectfull_dimerized(old_sinds, ψ, v1, v2, egs)
        flush(stdout)
    end

    for (i, cl) in enumerate(circuit_lengths)
        cs = corrs[:, i]
        corr_dict=  Dictionary()
        for (j, v2) in enumerate((v1+1):(nv(old_sinds)))
            set!(corr_dict, NamedEdge(v1 => v2), cs[j])
        end
        save("/mnt/home/jtindall/ceph/Data/DWave/Corrs/BPCorrected/Cubic/Dimerized/LoopLength$(cl)Corrsnx$(nx)ny$(ny)nz$(nz)AnnealingTime$(annealing_time)Chi$(χ_state)DisorderNo$(disorder_no)/v$(v1).jld2",
        "corrs", corr_dict)
    end
end

main_diamond()
#main_cylinder()
#main_cubic()