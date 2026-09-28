# Advanced Topics

## GPU Support

!!! warning "Validated; benchmarking in progress"
    GPU execution on the current tensor engine is validated by `test/test_gpu_paths.jl` —
    belief propagation, exact contraction, gate application, finite CTM (`:cut`, `:cycle`),
    boundary MPS, `InfiniteCTM2D` and the 3D boundary-PEPS solvers agree with the host to
    roundoff with scalar indexing disallowed (passed on an RTX 3070, 2026-09-27). Performance
    benchmarking is ongoing (docs/status_3d.md, "GPU readiness"). The GPU results reported in
    [[Rudolph2025]](index.md#references) were obtained with an earlier ITensors-based backend
    (available on the `main` branch history). States and caches are transferred with
    `CUDA.cu`/`adapt`; the fused CPU kernels detect non-CPU storage and fall back to the
    generic contraction path. For large `InfiniteCTM2D` networks use `convergence = :lnkappa`
    when only free energies are needed (the default `:environment` signal forms the site
    environment with every raw leg open) and raise `svd_oversample` if pairs bail out of the
    subspace SVD (`CTM_SVD_STATS[:i2_split_bail]`).

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
