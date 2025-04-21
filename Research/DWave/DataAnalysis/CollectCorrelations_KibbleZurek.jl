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

function main()
    radius = 10
    annealing_times = [3 + (i-1) for i in 1:13]
    disorder_nos = [i for i in 1:20]
    χ_state = 32
    χ_state_truncate = 10

    ss = Float64[0.2 + 0.01*(i-1) for i in 1:30]
    g = cylinder_graph(radius, radius)
    precision = 256

    for annealing_time in annealing_times
        Gammas, Js = [1.0 * pi * first(annealing_schedule(s * annealing_time + 0.5*0.01, annealing_time)) for s in ss], [1.0 * pi * last(annealing_schedule(s * annealing_time + 0.5*0.01, annealing_time)) for s in ss]
        for disorder_no in disorder_nos
            disorder_no_str = string(disorder_no - 1, pad = 2)
            instance_file = "/mnt/home/jtindall/ceph/Data/DWave/Instances/2d_($(radius), $(radius))_precision$(precision)/seed"*disorder_no_str*".npz"
            J_dict = couplings_to_edge_dict(g, radius, npzread(instance_file))
            mags = zeros(ComplexF64, (length(ss), radius, radius))
            zzs = ones(ComplexF64, (length(ss), radius, radius, radius))
            nn_couplings = zeros(Float64, (radius, radius, radius))
            for (i, sr) in enumerate(ss)
                try
                    file_name = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/BoundaryMPS/KibbleZurek/InfiniteRandomness/Radius$(radius)Chi$(χ_state)ChiTrunc$(χ_state_truncate)DisorderNo$(disorder_no)AnnealingTime$(annealing_time)s$(round(sr, digits =2)).jld2"
                    d = load(file_name)
                    ms = d["x_mags"]
                    for (column, row) in keys(ms)
                        mags[i, column, row] = ms[(column, row)]
                    end

                    
                    cs =d["zzcorrs"]
                    for k in keys(cs)
                        column, row1, row2 = first(src(k)), last(src(k)), last(dst(k))
                        zzs[i, column, row1, row2] = cs[k]
                        zzs[i, column, row2, row1] = cs[k]
                        if haskey(J_dict, k)
                            nn_couplings[column, row1, row2] = J_dict[k]
                        end
                    end
                catch
                    continue
                end
            end
            file_name = "/mnt/home/jtindall/ceph/Data/DWave/Corrs/BoundaryMPS/KibbleZurek/InfiniteRandomnessComposedData/L$(radius)Chi$(χ_state)ChiTrunc$(χ_state_truncate)AnnealingTime$(annealing_time)DisorderNo$(disorder_no).npz"
            npzwrite(file_name, renormalized_times = ss, xs = mags, zzs = zzs, Js = Js, Gammas = Gammas, nn_couplings = nn_couplings)
        end
    end
end

main()