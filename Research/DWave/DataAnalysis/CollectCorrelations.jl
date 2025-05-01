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

function correlation_error(tns_correlations, groundtruth_correlations)
    num, den = 0, 0
    for i in 1:length(tns_correlations)
        num += (tns_correlations[i] - groundtruth_correlations[i])*(tns_correlations[i] - groundtruth_correlations[i])
        den += groundtruth_correlations[i]*groundtruth_correlations[i]
    end
    return sqrt(num / den)
end

function non_duplicate_merge(dict1::Dictionary, dict2::Dictionary)
    dict1 = copy(dict1)
    dict1_keys = copy(keys(dict1))
    for e in dict1_keys
        if e ∈ keys(dict2) || reverse(e) ∈ keys(dict2)
            delete!(dict1, e)
        end
    end
    return merge(dict1, dict2)
end

function files_to_corrs(g::AbstractGraph, annealing_time, tns_bond_dimension, disorder_no, lattice::String, var_name::String; tns_trunc_bond_dimension = nothing, max_loop_length = 0)
    file_prefix = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/BPCorrected/"*lattice*"/LoopLength$(max_loop_length)"
    file_save_prefix = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/BPCorrected/"*lattice*"FullCorrs/LoopLength$(max_loop_length)"
    if isnothing(tns_trunc_bond_dimension)
        file_suffix = "Corrs"*var_name*"AnnealingTime$(annealing_time)Chi$(tns_bond_dimension)DisorderNo$(disorder_no)"
    else
        file_suffix = "Corrs"*var_name*"AnnealingTime$(annealing_time)Chi$(tns_bond_dimension)ChiTrunc$(tns_trunc_bond_dimension)DisorderNo$(disorder_no)"
    end
    
    time_taken = 0
    corrs = Dictionary()
    vs = collect(vertices(g))
    for v in setdiff(vs, [nv(g)])
        file_name = file_prefix *file_suffix * "/v$(v).jld2"
        f = load(file_name)
        corrs = non_duplicate_merge(f["corrs"], corrs)
    end

    L = nv(g)
    if length(keys(corrs)) == L * (L-1) / 2
        wf = nothing
        wf_file_prefix = lattice != "Cylinder" ? "/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/"*lattice*"/" : "/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/wf"
        try
            wf_file_suffix = var_name*"Chi$(tns_bond_dimension)DisorderNo$(disorder_no)AnnealingTime$(annealing_time)Cutoff.jld2"
            wf = load(wf_file_prefix * wf_file_suffix)
        catch
            wf_file_suffix = var_name*"Chi$(tns_bond_dimension)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2"
            wf = load(wf_file_prefix * wf_file_suffix)
        end
        time_taken_evo = wf["TimeTaken"]
        corrs_vector, _ = convert_dict_to_correlations(g, corrs)
        isdir(file_save_prefix *file_suffix) || mkdir(file_save_prefix *file_suffix)
        npzwrite(file_save_prefix *file_suffix * "/FullCorrs.npz", corrs = corrs_vector, TimeTakenBoundaryMPS = time_taken, TimeTakenBP = time_taken_evo)
    end
    return corrs
end

function files_to_corrs_V2(nv::Int64, annealing_time, tns_bond_dimension, disorder_no, no_jobs::Int64, corrs_per_job::Int64, var_name::String; max_loop_length = 0)
    file_prefix = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/BPCorrected/DiamondErrorAnalysis/LoopLength$(max_loop_length)"
    file_save_prefix = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/BPCorrected/DiamondErrorAnalysis/FullCorrs/LoopLength$(max_loop_length)"
    file_suffix = "Corrs"*var_name*"AnnealingTime$(annealing_time)Chi$(tns_bond_dimension)DisorderNo$(disorder_no)"
    
    time_taken = 0
    corrs = Dictionary()
    missed_jobs = []
    for i in 1:no_jobs
        try
            file_name = file_prefix *file_suffix * "/JobNo" * string(i)*"NCorrs"*string(corrs_per_job)*".jld2"
            f = load(file_name)
            corrs = non_duplicate_merge(f["corrs"], corrs)
        catch
            push!(missed_jobs, i)
        end
    end

    corrs_vector = []
    @show missed_jobs
    if length(keys(corrs)) == nv * (nv-1) / 2
        wf = nothing
        wf_file_prefix = "/mnt/home/jtindall/ceph/Data/DWave/Wavefunctions/Diamond/"
        try
            wf_file_suffix = var_name*"Chi$(tns_bond_dimension)DisorderNo$(disorder_no)AnnealingTime$(annealing_time)Cutoff.jld2"
            wf = load(wf_file_prefix * wf_file_suffix)
        catch
            wf_file_suffix = var_name*"Chi$(tns_bond_dimension)DisorderNo$(disorder_no)AnnealingTime$(annealing_time).jld2"
            wf = load(wf_file_prefix * wf_file_suffix)
        end
        time_taken_evo = wf["TimeTaken"]
        corrs_vector, _ = convert_dict_to_correlations(nv, corrs)
        dir = file_save_prefix *file_suffix
        isdir(dir) || mkdir(dir)
        npzwrite(file_save_prefix *file_suffix * "/FullCorrs.npz", corrs = corrs_vector, TimeTakenBoundaryMPS = time_taken, TimeTakenBP = time_taken_evo)
    end
    return corrs_vector
end

function files_to_corrs_cylinder(g::AbstractGraph, radius::Int64, annealing_time, tns_bond_dimension, disorder_no, lattice::String, var_name::String; tns_trunc_bond_dimension = nothing, max_loop_length = 0)
    file_prefix = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/BPCorrected/"*lattice*"/LoopLength$(max_loop_length)"
    if isnothing(tns_trunc_bond_dimension)
        file_suffix = "Corrs"*var_name*"AnnealingTime$(annealing_time)Chi$(tns_bond_dimension)DisorderNo$(disorder_no)"
    else
        file_suffix = "Corrs"*var_name*"AnnealingTime$(annealing_time)Chi$(tns_bond_dimension)ChiTrunc$(tns_trunc_bond_dimension)DisorderNo$(disorder_no)"
    end
    
    time_taken = 0
    corrs = Dictionary()
    vs = collect(vertices(g))
    for vrow in 1:radius
        for vcol in 1:radius
            file_name = file_prefix *file_suffix * "/vc$(vcol)vr$(vrow).jld2"
            f = load(file_name)
            corrs = non_duplicate_merge(f["corrs"], corrs)
            #time_taken += f["TimeTaken"]
        end
    end

    L = nv(g)
    if length(keys(corrs)) == L * (L-1) / 2
        corrs_vector = convert_dict_to_correlations(g, radius, corrs)
        npzwrite(file_prefix *file_suffix * "/FullCorrs.npz", corrs = corrs_vector, TimeTakenBoundaryMPS = time_taken)
        return corrs_vector
    end
    return nothing
end

function main_diamond()
    disorder_nos = [i for i in 2:2]
    ns =[(8,8,8)]
    annealing_times =[7]
    tns_bond_dimension =48
    max_loop_lengths =[0,6]
    no_corrs_per_job = 25
    no_jobs = 326
    mps_bond_dimension = 512

    for (nx,ny,nz) in ns
        println("Nx is $nx")
        nv = round(Int64, nx * ny * nz * 0.25)
        for annealing_time in annealing_times
            println("Annealing Time is $annealing_time")
            for max_loop_length in max_loop_lengths
                println("Loop length is $max_loop_length")
                errs = []
                for disorder_no in disorder_nos
                    println("Analysing Error for disorder $disorder_no")
                    boundarympsratio_corrs = files_to_corrs_V2(nv, annealing_time, tns_bond_dimension, disorder_no, no_jobs, no_corrs_per_job, "nx$(nx)ny$(ny)nz$(nz)"; max_loop_length)
                    q_sq = mean([x*x for x in boundarympsratio_corrs])
                    @show q_sq

                    #mps_file_name = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/MPS/diamond_($(nx)_$(ny)_$(nz))_precision256/$(annealing_time)ns/chi$(mps_bond_dimension)/correlations_uppertriangular_20_seeds.npz"
                    #mps_file = npzread(mps_file_name)
                    #mps_corrs = mps_file["corrs"][disorder_no, :]
                    #err = correlation_error(mps_corrs, boundarympsratio_corrs)
                    #push!(errs, err)

                    #sorted_corrs = reverse(sort([(i, abs(corr - mps_corrs[i])) for (i, corr) in enumerate(boundarympsratio_corrs)]; by = x -> last(x)))

                    #@show first(sorted_corrs)
                    #@show mps_corrs[first(first(sorted_corrs))]
                end
                #@show mean(errs)
                #@show Statistics.std(errs)
            end
        end
    end
end

function main_cylinder()
    disorder_nos = [i for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    radii =[4]
    annealing_times =[20]
    tns_bond_dimension = 32
    tns_trunc_bond_dimension = nothing
    mps_bond_dimension =64
    max_loop_lengths = [7]


    for radius in radii
        println("Radius is $radius")
        for annealing_time in annealing_times
            println("Annealing Time is $annealing_time")
            errs = []
            for disorder_no in disorder_nos
                for max_loop_length in max_loop_lengths
                    try
                        g = named_cylinder(radius, radius)
                        println("Analysing Error for disorder $disorder_no")
                        boundarympsratio_corrs = files_to_corrs_cylinder(g, radius, annealing_time, tns_bond_dimension, disorder_no, "Cylinder", "Radius$(radius)"; max_loop_length, tns_trunc_bond_dimension)
                        mps_file_name = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/MPS/2d_($(radius)_$(radius))_precision256/$(annealing_time)ns/chi$(mps_bond_dimension)/correlations_uppertriangular_20_seeds.npz"
                        mps_file = npzread(mps_file_name)
                        mps_corrs = mps_file["corrs"][disorder_no, :]
                        #boundarympsratio_corrs = convert_dict_to_correlations(g, boundarympsratio_corrs)
                        err = correlation_error(boundarympsratio_corrs, mps_corrs)
                        @show err
                        push!(errs, err)
                    catch
                        continue
                    end
                end
            end
            @show mean(errs)
            @show Statistics.std(errs)
        end
    end
end

function main_cubic()
    disorder_nos = [i for i in 1:20]
    ns = [(3,2,2)]
    annealing_times =[20]
    tns_bond_dimension = 8
    mps_bond_dimension = 64
    max_loop_lengths =[0, 4, 7]

    for (nx,ny,nz) in ns
        println("Nx is $nx")
        for annealing_time in annealing_times
            println("Annealing Time is $annealing_time")

            for max_loop_length in max_loop_lengths
                errs = []
                for disorder_no in disorder_nos
                    disorder_no_str = string(disorder_no - 1, pad = 2)
                    instance_file = "/mnt/home/jtindall/ceph/Data/DWave/Instances/3ddimer_($(nx), $(ny), $(nz))_precision256/seed"*disorder_no_str*".npz"
                    g, J_dict = graph_couplings_from_instance(instance_file)
                    println("Analysing Error for disorder $disorder_no")
                    boundarympsratio_corrs = files_to_corrs(g, annealing_time, tns_bond_dimension, disorder_no, "Cubic/Dimerized", "nx$(nx)ny$(ny)nz$(nz)"; max_loop_length)
                    mps_file_name = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/MPS/3d_($(nx)_$(ny)_$(nz))_precision256/$(annealing_time)ns/chi$(mps_bond_dimension)/correlations_uppertriangular_20_seeds.npz"
                    mps_file = npzread(mps_file_name)
                    mps_corrs = mps_file["corrs"][disorder_no, :]

                    boundarympsratio_corrs, es = convert_dict_to_correlations(g, boundarympsratio_corrs)
                    err = correlation_error(boundarympsratio_corrs, mps_corrs)
                    @show err
                    push!(errs, err)

                end
                @show mean(errs)
                @show Statistics.std(errs)


            end

        end
    end
end




main_diamond()
#main_cylinder()
#main_cubic()