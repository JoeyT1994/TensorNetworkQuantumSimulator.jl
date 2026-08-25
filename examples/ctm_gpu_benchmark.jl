"""Synchronized CPU/GPU throughput benchmark for finite CTMRG.

Environment variables:
  CTM_GPU_L=6 CTM_GPU_D=3 CTM_GPU_CHIS=8,16 CTM_GPU_SWEEPS=4
  CTM_GPU_METHODS=cut,cycle,cycle_block CTM_GPU_REPEATS=2
  CTM_GPU_COMPONENTS=false CTM_GPU_PROFILE=false CTM_GPU_CPU_SEED=false
"""

using Adapt: adapt
using CUDA
using Logging: NullLogger, with_logger
using Printf
using Random
using Statistics: median
using TensorNetworkQuantumSimulator
import TensorNetworkQuantumSimulator: update
import TensorNetworkQuantumSimulator: _ctm_env, _ctm_factor_table, _ctm_setenv, _ctm_statedist
import TensorNetworkQuantumSimulator: sweep_vertex_environments

const METHODS = (:cut, :cycle, :cycle_block)

parse_ints(name, default) = parse.(Int, split(get(ENV, name, default), ','))
parse_methods() = Symbol.(strip.(split(get(ENV, "CTM_GPU_METHODS", join(METHODS, ',')), ',')))

function solve(tn, χ, method, sweeps, convergence, gauge; cpu_seed = false)
    projector = method === :cut ? :cut : :cycle
    block = method === :cycle_block
    cache = CTMEnvironmentCache(tn, χ; projector, gauge, cycle_subspace = block,
                                cycle_warmstart = block)
    CUDA.synchronize()
    start = time_ns()
    if cpu_seed
        cpu_network = adapt(Array, tn)
        seed_cache = CTMEnvironmentCache(cpu_network, χ; projector, gauge,
                                         cycle_subspace = block, cycle_warmstart = block)
        cache = _ctm_setenv(cache, adapt(CuArray, vertex_environments(seed_cache)))
    end
    cache = with_logger(NullLogger()) do
        update(cache; maxiter = sweeps, tolerance = 0.0, convergence)
    end
    CUDA.synchronize()
    return cache, (time_ns() - start) / 1.0e9
end

function gpu_elapsed(f)
    CUDA.synchronize()
    start = time_ns()
    result = f()
    CUDA.synchronize()
    return result, (time_ns() - start) / 1.0e9
end

function main()
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    length = parse(Int, get(ENV, "CTM_GPU_L", "6"))
    bonddim = parse(Int, get(ENV, "CTM_GPU_D", "3"))
    sweeps = parse(Int, get(ENV, "CTM_GPU_SWEEPS", "4"))
    repeats = parse(Int, get(ENV, "CTM_GPU_REPEATS", "2"))
    convergence = Symbol(get(ENV, "CTM_GPU_CONVERGENCE", "free_energy"))
    gauge = lowercase(get(ENV, "CTM_GPU_GAUGE", "true")) in ("1", "true", "yes", "on")
    chis = parse_ints("CTM_GPU_CHIS", "8,16")
    methods = parse_methods()
    cpu_seed = lowercase(get(ENV, "CTM_GPU_CPU_SEED", "false")) in
               ("1", "true", "yes", "on")
    all(in(METHODS), methods) || error("CTM_GPU_METHODS must use $(join(METHODS, ','))")

    Random.seed!(1234)
    graph = named_grid((length, length))
    sites = siteinds("S=1/2", graph)
    cpu = gauge_and_scale(random_tensornetworkstate(
        ComplexF32, graph, sites; bond_dimension = bonddim))
    gpu = adapt(CuArray, cpu)

    if lowercase(get(ENV, "CTM_GPU_COMPONENTS", "false")) in ("1", "true", "yes", "on")
        χ, method = first(chis), first(methods)
        cache, _ = solve(gpu, χ, method, 2, convergence, gauge; cpu_seed)
        env = _ctm_env(cache)
        table = _ctm_factor_table(cache)
        next, sweep_time = gpu_elapsed() do
            sweep_vertex_environments(cache, env, table)
        end
        _, free_energy_time = gpu_elapsed() do
            cvm_freenergy(next, cache)
        end
        _, state_time = gpu_elapsed() do
            _ctm_statedist(next, env)
        end
        @printf("component,seconds\nsweep,%.6f\nfree_energy,%.6f\nstate_distance,%.6f\n",
                sweep_time, free_energy_time, state_time)
        return
    end

    if lowercase(get(ENV, "CTM_GPU_PROFILE", "false")) in ("1", "true", "yes", "on")
        χ, method = first(chis), first(methods)
        solve(gpu, χ, method, 2, convergence, gauge; cpu_seed)
        profile = CUDA.@profile solve(gpu, χ, method, sweeps, convergence, gauge; cpu_seed)
        show(stdout, MIME("text/plain"), profile)
        println()
        return
    end

    @printf("# %dx%d random ComplexF32 PEPS, D=%d, fixed sweeps=%d, repeats=%d\n",
            length, length, bonddim, sweeps, repeats)
    println("chi,method,cpu_seconds,gpu_seconds,speedup,abs_lnZ_difference")
    for χ in chis, method in methods
        # Compile kernels and populate the structural contraction-sequence cache.
        solve(cpu, χ, method, 2, convergence, gauge)
        solve(gpu, χ, method, 2, convergence, gauge; cpu_seed)
        cpu_times = Float64[]
        gpu_times = Float64[]
        cpu_cache = gpu_cache = nothing
        for _ in 1:repeats
            cpu_cache, elapsed = solve(cpu, χ, method, sweeps, convergence, gauge)
            push!(cpu_times, elapsed)
            gpu_cache, elapsed = solve(gpu, χ, method, sweeps, convergence, gauge; cpu_seed)
            push!(gpu_times, elapsed)
        end
        tcpu, tgpu = median(cpu_times), median(gpu_times)
        difference = abs(cvm_freenergy(cpu_cache) - cvm_freenergy(gpu_cache))
        @printf("%d,%s,%.6f,%.6f,%.3f,%.6e\n",
                χ, method, tcpu, tgpu, tcpu / tgpu, difference)
        flush(stdout)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
