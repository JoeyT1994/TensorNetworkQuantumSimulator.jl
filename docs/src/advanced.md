# Advanced Topics

## GPU Support

!!! warning "In flux"
    GPU execution is being re-established on the current tensor engine and is not yet
    validated. The GPU results reported in [[Rudolph2025]](index.md#references) were
    obtained with an earlier ITensors-based backend (available on the `main` branch
    history). The building blocks for the new path exist — TensorOperations provides
    cuTENSOR-backed contraction and CUDA buffer allocators, and MatrixAlgebraKit provides
    CUSOLVER/ROCSOLVER factorizations — but wiring and benchmarking them here is ongoing
    work. States and caches can be transferred with `CUDA.cu`/`adapt` as before; the fused
    CPU kernels detect non-CPU storage and fall back to the generic contraction path.

Use `ComplexF32` element types for best GPU performance once available; imaginary-time
simulations can be run without `Complex` arithmetic entirely.

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
