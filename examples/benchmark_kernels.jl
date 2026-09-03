# Walltime and allocation volume for gate application and BP expectation values, on CPU
# and (if CUDA is loaded) GPU.
#
# BP closures use the fused two-slot path; gate application uses generic buffered
# contractions with consumed destinations and backend-native factorizations. Use this to
# track runtime/allocation-volume regressions and to compare machines. For exact CUDA
# pool high-water measurements, use `benchmark_gpu_hotpaths.jl`.
#
#   julia --project=. examples/benchmark_kernels.jl              # CPU only
#   julia --project=. -e 'using CUDA; include("examples/benchmark_kernels.jl")'   # + GPU
#
# Run it on an idle machine: a loaded box swings these numbers by tens of percent.

using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using Printf

const L = 6              # grid side
const CHI = 24           # bond dimension
const LAYERS = 3
const REPS = 3           # timings are the minimum over this many runs

function workload(device)
    g = named_grid((L, L))
    ψ = tensornetworkstate(ComplexF32, v -> iseven(sum(v)) ? "↑" : "↓", g, siteinds("S=1/2", g))
    layer = Any[]
    for ces in edge_color(g, 4)
        append!(layer, ("Rxx", pair, 0.7) for pair in ces)
        append!(layer, ("Rzz", pair, 0.2) for pair in ces)
    end
    circuit = reduce(vcat, [layer for _ in 1:LAYERS])
    if device === :gpu
        CUDA = Base.loaded_modules[Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")]
        ψ = Base.invokelatest(CUDA.cu, ψ)
    end
    return g, ψ, circuit
end

function run_case(device)
    g, ψ, circuit = workload(device)
    apply_kwargs = (; maxdim = CHI, cutoff = 1.0f-10)
    obs = [("Z", [v]) for v in vertices(g)]
    #warm up (compilation, arena sizing)
    ψt, _ = apply_gates(circuit, ψ; apply_kwargs)
    expect(ψt, obs; alg = "bp")

    t_apply = minimum(@elapsed(apply_gates(circuit, ψ; apply_kwargs)) for _ in 1:REPS)
    a_apply = @allocated apply_gates(circuit, ψ; apply_kwargs)
    t_expect = minimum(@elapsed(expect(ψt, obs; alg = "bp")) for _ in 1:REPS)
    a_expect = @allocated expect(ψt, obs; alg = "bp")
    z = real(first(expect(ψt, obs; alg = "bp")))

    return (; t_apply, a_apply, t_expect, a_expect, z)
end

function report(device)
    println("\n", uppercase(string(device)), "   (L=$L, χ=$CHI, $LAYERS layers, ComplexF32)")
    println("  apply_gates            expect")
    r = run_case(device)
    @printf("  %6.2f s / %7.3f GiB   %6.2f s / %7.3f GiB   ⟨Z⟩ = %+.10f\n",
        r.t_apply, r.a_apply / 2^30, r.t_expect, r.a_expect / 2^30, r.z)
    return nothing
end

report(:cpu)

#GPU section runs only when CUDA is loaded by the caller
if any(k -> k.name == "CUDA", keys(Base.loaded_modules))
    report(:gpu)
else
    println("\nGPU: skipped (load CUDA before including this file to enable)")
end
