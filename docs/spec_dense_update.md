# Dense two-site update: memory peak and in-place QR

Notes moved out of the source comments in `src/Apply/simple_update_dense.jl`. All figures were
measured on an RTX PRO 6000 unless stated otherwise.

## The alternating-buffer chain

The large side of a two-site update is a chain of equally sized `d·χ^deg` objects:

```
T --(× √env)--> ψᵥ --(QR)--> Q --(× env^-1/2)--> Q' --(× R)--> out
```

With `contract`, each link allocates its own output plus its own permuted temporaries, so several
coexist. `absorb_chain!` instead alternates two preallocated buffers, so at most two of these are
live (three transiently, while the input is still being read).

Everything in the chain is `mul!`, `permutedims!`, `reshape` and `view` on flat buffers — no scalar
indexing — so it runs on CPU or GPU. The working layout is `(env legs…, site…, bond)`: the env legs
lead, so the QR's row block is contiguous and needs no permute of its own. `mul_strided_batched!`
is the one operation with a device-specific method (`ext/TensorNetworkQuantumSimulatorCUDAExt.jl`),
which replaces the CPU loop over trailing slices with a single `gemm_strided_batched!` launch.

Contiguous slices are safe to pass into the chain as scratch: `GPUArrays` routes those through
`derive`, which returns another `CuArray`, so cuBLAS still dispatches.

## Axis conventions and footprint of `simple_update_dense!`

Per side, `qrinds` names the axes to the right of the QR — (site, shared bond) — and the environment
legs are the remaining axes in ascending order. Each output is laid out
`(environment legs…, trailing axes of the R that middle! returned)`.

`simple_update_dense!` allocates one buffer, sized to the larger side, and carves every later
scratch out of it, so the footprint is 2 × the larger vertex tensor plus 1 × the smaller. It orders
the two sides largest-first for that reason, which is why index 1 inside it is not necessarily the
caller's side 1.

## Reusing the input's storage (`absorb_matrices!`)

`permutedims` cannot work in place for a non-trivial permutation, so the input and the permute
destination must coexist once. That pair is the 2 × floor; the point of the chain is to need no
third buffer beyond it. `absorb_matrices!` permutes into a fresh (or supplied) buffer and then
repurposes the input's own storage as the second scratch.

Dropping the reference is not enough. Clearing the caller's binding and then allocating a second
buffer gives 3 ×, not 2 ×: on CPU the input is still resident after the assignment *and* a full
`GC.gc(true)`, because the stack slot that held it is only overwritten by the next call. On the GPU
that is a hard failure — at χ = 512 three 4 GiB buffers OOM'd under a 12 GiB cap. Taking the
input's buffer over as scratch makes the 2 × peak deterministic rather than dependent on when the
collector runs.

This scribbles on the input's memory, a stronger claim than merely releasing it. It is sound only
because `apply_gate!` clears the network's entry first, leaving the argument the last reference —
which is why `simple_update_dense` documents itself as consuming.

For the same reason both `simple_update_dense` and `simple_update_dense_boundary` rewrap results
with `itensor` rather than `ITensor`: the capitalised constructor defaults to `NeverAlias` and
copies its input, a third factor-sized buffer. At χ = 1024 that copy is the 32 GiB allocation that
OOM'd on a 95 GiB card. (`Dense` vecs the reshaped array, which also shares memory, so the
aliasing path really is zero-copy.)

Peak with the dense path is 2 × one vertex tensor against 3.25 × for the `contract`/`qr` route,
both measured on device at χ = 512. That is what makes χ = 1024 reachable: a degree-3 vertex is
then 32 GiB, so 2 × fits a 95 GiB card and 3.25 × does not.

## Consuming inputs is a contract on the caller

A shallow `copy` of a network or cache shares the `ITensor` objects and hence their data buffers —
a write through one is visible through the other — so anything the caller still holds that came
from the same network is overwritten. `apply_gates` copies the cache on entry, and that copy does
not protect the caller's tensors from this path. Callers that need the input afterwards must
duplicate its *data* first: rebuild each tensor with the copying `ITensor(array(t), inds(t)...)`.

## `thin_qr_matrix!`: `geqrf!`/`orgqr!` rather than `qr!`

`geqrf!` leaves R in the upper triangle and the Householder reflectors below; `orgqr!` then
overwrites the whole thing with Q, so R must be copied out in between.

`LinearAlgebra.qr!` followed by `lmul!(F.Q, ...)` — and equivalently `CuMatrix(F.Q)`, which
CUDA.jl implements via `lmul!` — routes through cuSOLVER `ormqr`, which fails with
`CUSOLVER_STATUS_INVALID_VALUE` once the matrix exceeds `typemax(Int32)` elements. Both are fine
at χ = 512 (5.4e8 elements) and both fail at χ = 1024 (4.3e9), while `orgqr` accepts the same
dimensions. `permutedims`, gemm and broadcast are all fine at that size.

No device-specific method is needed: cuSOLVER.jl itself adds `LAPACK.geqrf!` and `LAPACK.orgqr!`
methods for `StridedCuMatrix` (its `orgqr!` covers complex too, via `cusolverDnCungqr` — there is
no separate `ungqr!`). Hence the `AbstractMatrix` signature plus an `applicable` check, rather
than `StridedMatrix`, which would exclude `CuArray`.

`absorb_matrices_qr!` probes `applicable(LAPACK.geqrf!, ...)` and the tall/wide shape *before* the
permute consumes the input, because bailing out afterwards would leave the caller's fallback with
an empty tensor — which is how this first failed on the GPU.

## Against `ITensors.qr`

`ITensors.qr` allocates several factor-sized temporaries; the in-place QR allocates only the small
R, because Q lands in the permuted copy of the input. For the matrix a degree-3 vertex produces
(χ² × dχ, d = 4, `ComplexF32`) at χ = 128: 0.04 × one vertex tensor against 7.13 × for
`ITensors.qr`.

The permute into matrix form is the one copy still paid; `geqrf!` then consumes it. Building ψᵥ
with the row indices already leading would make the view a plain reshape and remove that too.

A thin QR requires m >= n so that Q fills the input's storage exactly. A degree-2 vertex gives a
wide matrix; `simple_update_dense` falls back to `simple_update` for that, and
`absorb_boundary_in!` carries it by letting the whole absorbed tensor stand in for `R`. Those
tensors are O(χ²) and irrelevant to the peak.

## SVD convergence

`factorize_svd` returns `nothing` when the SVD fails to converge — it prints an explanation and
hands back nothing rather than throwing, so destructuring it raises `MethodError: no method
matching iterate(::Nothing)`, which reads like a bug in the caller. On CUDA the default algorithm
(`"qr_algorithm"`, cuSOLVER `gesvd`) does fail on the matrices a large-χ update produces;
`"jacobi_algorithm"` is the robust GPU choice. On CPU, `"qr_iteration"` and `"recursive"` are the
reliable fallbacks.
