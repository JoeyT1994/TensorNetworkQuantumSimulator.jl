using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionEdge
using Statistics
ENV["JULIA_CUDA_MEMORY_POOL"] = "none"


using ITensors: Algorithm

using ITensors

ITensors.set_warn_order(20)

using LinearAlgebra

BLAS.set_num_threads(6)


function main(n::Int, hx::Number, χ::Int, χ_trunc::Int, RMPS::Int)
    nx = n
    ny = n

    half_n = round(Int, n / 2)
    # the graph is your main friend. This will be the geometry of the TN you wull work with
    g = named_grid((nx, ny))

    dt = 0.01

    J = 1.0
    hx = 3.5
    hz = 8.0

    trise = 1.5
    tsweep = 1.5
    tfall = 3.0

    # Parameters
    a = trise
    b = tsweep
    c = tfall
    times = 0:dt:(a + b + c)

    # Define hx pulse
    hx_list = Float32[
        t <= a ? (hx / a) * t :
            t <= a + b ? hx :
            t <= a + b + c ? hx * (1 - (t - a - b) / c) :
            0.0 for t in times
    ]

    # Define hz pulse
    hz_list = Float32[
        t <= a ? hz :
            t <= a + b ? hz * (1 - 1.0 * (t - a) / b) :
            t <= a + b + c ? 0 :
            0.0 for t in times
    ]


    nl = length(times) - 1
    measure_steps = 5


    # observables are tuples like `(pauli_string, [site_labels], optional:coefficient)`
    # it's important that the `site_labels` match the names of the vertices of the graph `g`
    sz_obs = [("Z", [(half_n, half_n + i)]) for i in 0:half_n]  # right in the middle
    szz_obs = [("ZZ", [(half_n, half_n), (half_n, half_n + i)]) for i in 1:half_n]
    # the number of circuit layers

    #The inds network
    s = siteinds("S=1/2", g)
    # the initial state (all up, use Float 32 precision)
    ψ0 = tensornetworkstate(ComplexF32, v -> "↓", g, s)

    # max bond dimension for the TN
    apply_kwargs = (maxdim = χ, cutoff = 1.0e-10, normalize_tensors = true)

    # create the BP cache representing the square of the tensor network
    ψ_bpc = TN.BeliefPropagationCache(ψ0)
    ψ_bpc = TN.update(ψ_bpc)

    sz_expectations_boundarymps = zeros(Float64, (round(Int, nl / measure_steps) + 1, length(sz_obs)))
    szz_expectations_boundarymps = zeros(Float64, (round(Int, nl / measure_steps) + 1, length(szz_obs) + 1))
    correlators_boundarymps = zeros(Float64, (round(Int, nl / measure_steps) + 1, length(szz_obs) + 1))
    szz_expectations_boundarymps[1, :] = [1.0 for i in 1:(length(szz_obs) + 1)]
    sz_expectations_boundarymps[1, :] = [-1.0 for i in 1:length(sz_obs)]

    times = [0.0]
    errs = Float64[]
    t = 0
    i = 2

    simulation_time = time()
    # evolve! (First step takes long due to compilation)
    for l in 1:nl
        println("Layer $l")

        #Build a layer of the circuit. Pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
        layer = []

        append!(layer, ("Rx", [v], hx_list[l] * dt) for v in vertices(g))
        append!(layer, ("Rz", [v], hz_list[l] * dt) for v in vertices(g))

        #For two site gates do an edge coloring to Trotterise the circuit
        ec = edge_color(g, 4)
        for colored_edges in ec
            append!(layer, ("Rzz", pair, 2 * J * dt) for pair in colored_edges)
        end

        append!(layer, ("Rx", [v], hx_list[l] * dt) for v in vertices(g))
        append!(layer, ("Rz", [v], hz_list[l] * dt) for v in vertices(g))


        t1 = @timed ψ_bpc, errors =
            apply_gates(layer, ψ_bpc; apply_kwargs, verbose = false)
        errs = append!(errs, errors)
        t += dt

        if l % measure_steps == 0

            ψ_bpc = TN.update(ψ_bpc; maxiter = 50, tolerance = 1.0e-14)
            ψ_bpc = TN.symmetrize_and_normalize(ψ_bpc)
            if χ_trunc < TN.maxvirtualdim(ψ_bpc)
                println("Truncating!")
                ψ_trunc = network(TN.truncate(ψ_bpc; maxdim = χ_trunc))
                @assert TN.maxvirtualdim(ψ_trunc) == χ_trunc
            else
                ψ_trunc = network(ψ_bpc)
            end

            println("    Took time: $(t1.time) [s]. Max bond dimension: $(TN.maxvirtualdim(ψ_bpc))")
            println("    Maximum Gate error for layer was $(maximum(errors))")
            println("Running BMPS")
            ψ_trunc_bmps = TN.BoundaryMPSCache(ψ_trunc, RMPS; partition_by = "row")


            #As we only measure things in one row, we only need to get all messages pointing towards it (half of all possible MPS messages)
            edge_seq = PartitionEdge.(vcat([n - i + 1 => n - i for i in 1:(half_n)], [i => i + 1 for i in 1:(half_n - 1)]))
            ψ_trunc_bmps = TN.update(ψ_trunc_bmps; alg = "bp", edge_sequence = edge_seq)
            println("Ran BMPS")
            sz_boundarymps = TensorNetworkQuantumSimulator.expect(ψ_trunc_bmps, sz_obs)
            szzs_boundarymps = TensorNetworkQuantumSimulator.expect(ψ_trunc_bmps, szz_obs)
            println("    Boundary MPS Measured Sigmaz is $(first(sz_boundarymps))")
            println("    Boundary MPS Measured Sigmazz is $(first((szzs_boundarymps)))")
            sz_expectations_boundarymps[i, :] = real.(sz_boundarymps)
            szz_expectations_boundarymps[i, :] = vcat([1.0], real.(szzs_boundarymps))
            correlators_boundarymps[i, :] = [szz_expectations_boundarymps[i, j] - sz_expectations_boundarymps[i, 1] * sz_expectations_boundarymps[i, j] for j in 1:length(szz_expectations_boundarymps[i, :])]
            push!(times, t)
            i += 1
            GC.gc()
        end
        flush(stdout)
    end

    #Simulation time (secs)
    simulation_time = time() - simulation_time
    #Wavefunction memory (GB)
    wf_mem = Base.summarysize(network(ψ_bpc)) / 1.0e9

    println("Simulation took $(simulation_time) secs")
    println("PEPS took up $(wf_mem) GB in memory")

    @show sz_expectations_boundarymps
    @show szz_expectations_boundarymps
    return @show correlators_boundarymps

    #save_file = "/work/sjfarre/BP_paper/results/2DIsingnx$(nx)ny$(ny)Annealing1h$(hx)Chi$(χ)ChiTrunc$(χ_trunc)RMPS$(RMPS)Expects.npz"
    #npzwrite(save_file, wf_mem = wf_mem, simulation_time = simulation_time, sz_expectations_boundarymps = sz_expectations_boundarymps, szz_expectations_boundarymps = szz_expectations_boundarymps, correlators_boundarymps = correlators_boundarymps, errs = errs, times = times)


end


# n =parse(Int64, ARGS[1])
# hx =parse(Float64, ARGS[2])
# χ =parse(Int64, ARGS[3])
# χ_trunc =parse(Int64, ARGS[4])
# RMPS =parse(Int64, ARGS[5])

n = 20
hx = 2.0
χ = 40
χ_trunc = 20
RMPS = 32

main(n, hx, χ, χ_trunc, RMPS)
