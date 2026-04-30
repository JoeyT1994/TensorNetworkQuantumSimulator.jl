# Advanced Topics

## GPU Support

Almost all operations support GPU acceleration. Load the relevant Julia GPU package (e.g. CUDA.jl or Metal.jl) and transfer the state or cache from which point subsequent operations on that object will be done on GPU:

```julia
using TensorNetworkQuantumSimulator
using CUDA

g = named_grid((8, 8))
ψ_cpu = random_tensornetworkstate(ComplexF32, g; bond_dimension = 8)
ψ_gpu = CUDA.cu(ψ_cpu)

@time expect(ψ_cpu, ("Z", (1, 1)); alg = "boundarymps", mps_bond_dimension = 16)
@time expect(ψ_gpu, ("Z", (1, 1)); alg = "boundarymps", mps_bond_dimension = 16)
```

Caches can also be transferred to the GPU:

```julia
ψ_bpc_gpu = CUDA.cu(BeliefPropagationCache(ψ))
```

Significant speedups are seen on NVIDIA GPUs for many operations (BP, BoundaryMPS, Gate Application) at moderate to large bond dimensions. Use `ComplexF32` element types for best GPU performance. We highly recommend CUDA.jl (NVidia GPUs) as speedups are well documented in this case [[Rudolph2025]](index.md#references). Experience using Metal.jl is very limited and so proceed with caution.

## Loop Corrections

On loopy graphs, belief propagation provides approximate results. Loop corrections can be used to systematically improve the BP estimate of the norm by accounting for the loops up to size `max_configuration_size` in the graph [[Evenbly2026]](index.md#references):

```julia
norm_bp = norm_sqr(ψ; alg = "bp")
norm_lc = norm_sqr(ψ; alg = "loopcorrections", max_configuration_size = 4)
```

See `examples/loopcorrections.jl` for a benchmark implementation across different lattice types.

## Element Types and Precision

The package supports arbitrary element types. Use the first argument of constructors to set the precision:

```julia
ψ_f32 = tensornetworkstate(ComplexF32, v -> "↑", g, "S=1/2")   # single precision
ψ_f64 = tensornetworkstate(ComplexF64, v -> "↑", g, "S=1/2")   # double precision
ψ_real = tensornetworkstate(Float64, v -> "↑", g, "S=1/2")     # real-valued
```

Use `ComplexF32` or `Float32` for GPU workloads where single precision suffices. Use `ComplexF64` or `Float64` (or omit the type argument) for higher precision. Imaginary time simulations can all be done without `Complex` arithmetic. Real time simulations will require it (although the conversion will happen automatically if needed).
