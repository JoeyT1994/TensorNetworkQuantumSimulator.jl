# Sylvester solvers

This folder contains a pure-JAX dense periodic Bartels--Stewart solver, the
native single-period real-Schur solver, and the real and complex periodic
compressed solvers. `compressed.py` owns the public single-period and periodic
contracts, including the JAX dense-GMRES implementation used for
single-period complex triangular problems. The extension exports typed CPU
XLA FFI handlers for float32, float64, complex64, and complex128.

The public entry points are

```python
sylvester_compressed(..., method="dense_gmres")
sylvester_compressed(..., method="schur_gmres")

sylvester_compressed_periodic(..., method="dense_gmres")
sylvester_compressed_periodic(..., method="periodic_schur_gmres")
sylvester_compressed_periodic(..., method="periodic_schur_galerkin")
```

Arithmetic-specific names are private details of `jax_ffi.py` and the native
ABI. Public callers dispatch on the dtype of `H`.

## Compressed problem

Let `H` have shape `n x n` and let the real Schur form `w` have shape `d x d`.
Block Arnoldi gives

```text
R   = Q beta,
A Q = Q H + Q_F residual_r E_tail,
```

where

```text
beta[:d, :] = v,
G[:, -d:]   = residual_r.
```

Writing the physical correction as `dX = Q Y` reduces the residual
minimization to

```text
min_Y ||H Y - Y w - beta||_F^2 + ||G Y||_F^2.                (1)
```

The routine returns `x = Y.T`, so the physical-space correction is
`Q @ x.T`.

## Real-Schur block substitution

For upper quasi-triangular `w`, diagonal blocks are visited from left to right.
Lower quasi-triangular problems are reversed and use the same sweep. For a
diagonal block `S = w[J,J]` of size `s`, with `s` equal to one or two, the
already solved blocks give

```text
rhs_J = beta[:, J] + Y @ w[:, J].
```

The coupled local problem is

```text
min_U ||H U - U S - rhs_J||_F^2 + ||G U||_F^2.              (2)
```

A boolean array `block_2x2_start` marks the first index of every 2x2 block.
The two columns of a 2x2 block must be solved together.

## Schur reduction of H

The solver forms one real Schur decomposition

```text
H = Z T Z.T,
Y = Z U,
beta_hat = Z.T beta,
C = G Z.
```

Because `Z` is orthogonal, equation (1) becomes

```text
min_U ||T U - U w - beta_hat||_F^2 + ||C U||_F^2.           (3)
```

For a block `S` of size `s`, column-major vectorization gives the local top
operator

```text
I_s kron T - S.T kron I.
```

The implementation stores this operator in row-interleaved order. A Schur
block of `T` of size `r` and a block of `w` of size `s` produce a dense local
diagonal block of size at most `r*s <= 4`. Givens rotations triangularize only
these small diagonal blocks.

The remaining least-squares matrix has a triangular top and the dense residual
panel `C kron I_s` underneath. LAPACK `DTPQRT` factors this
triangular-plus-dense matrix, `DTPMQRT` applies its orthogonal factor to the
right-hand side, and `DTRTRS` performs the triangular solve.

The compressed residual is reconstructed once after the sweep to return a
separate residual norm for every column. This is necessary because a coupled
2x2 solve directly produces only one joint residual norm.

## Why not general augmented QR?

A direct implementation can vectorize (2) and apply `DGEQRF` to the complete
augmented matrix. For a 2x2 block this doubles both the row and column
dimensions of the general QR. Benchmarks with `n = 3d` and all 2x2 blocks found
that approach about four to five times slower than the Schur-`DTPQRT` method,
so only the structured method is retained for the single-period kernel.

## Periodic compressed problem

The periodic entry points solve the right-oriented Arnoldi problem

```text
min_{Y_k} sum_k ||H_k Y_{k+1} + Y_k w_k - beta_k||_F^2
                  + ||R_k E_tail Y_{k+1}||_F^2,
```

with cyclic site indices, `beta_k[:d, :] = v_k`, and `R_k` of shape `d x d`.
`E_tail` selects the final `d` Arnoldi coordinates. Following the `MB03WD`
convention, `w[0]` is quasi-upper triangular and the other `w[k]` are upper
triangular. The boolean block mask therefore describes the 1x1 and 2x2 blocks
of `w[0]`.

The mathematical backends are dense GMRES, periodic-Schur GMRES, and
periodic-Schur Galerkin. QR is used internally to solve the GMRES projected
least-squares problems; it is not part of the backend names. Both real and
complex Galerkin first reduce the active factors to periodic Schur form.

The Galerkin route accepts the temporary
`krylov_cfg["galerkin_block_solver"]` comparison flag. `"dgesv"` selects LAPACK
for the small cyclic Schur-coordinate systems (`DGESV` for real input and
`ZGESV` for complex input), while `"mb03ke"` uses SLICOT's specialized real
small-periodic-Sylvester routine.
Real cycles of length at least two default to `"mb03ke"`; complex and
single-period problems default to `"dgesv"`.

The runtime scalar `rank` removes the exact right-space padding, while
`active_cols[k]` selects the live Arnoldi coordinates at each site. If `E_k`
is the corresponding column selector, the kernel packs

```text
Hhat_k = E_k.T H_k E_{k+1},   Yhat_k = E_k.T Y_k[:, :rank],
what_k = w_k[:rank, :rank],   Ghat_k = R_k E_tail E_{k+1}.
```

The active dimensions may differ between sites. For a Schur block of width
`s`, let `N = s*sum_k active_cols[k].sum()`. The reference implementation assembles
the dense augmented matrix `[A b]` with shape `(N + p*d*s) x (N + 1)`, applies
LAPACK `DGEQRF`, and solves its leading triangular system with `DTRTRS`. It
does not reduce `H_k` to periodic Schur form. This intentionally dumb dense QR
is retained as the baseline for the periodic-Schur implementation. The result is
scattered into `x[k] = Y[k].T`; inactive coordinates and columns `rank:` remain
exactly zero.

The structured `D/Z` implementations use the unified periodic-Schur NRed
pipeline. After `H_k Z_{k+1} = Z_k T_k`, they transform `beta_k` and
`R_k E_tail`, then process the 1x1/2x2 blocks of `w[0]`. Ordering each local
operator by Schur rows exposes cyclic diagonal blocks of dimension at most
`4*p`. Dense QR triangularizes only those blocks; `DTPQRT/ZTPQRT` then adds
the `p*d*s` Arnoldi-tail rows before the final triangular solve.

The public periodic interface also accepts an optional length-`p`
`scale_tol`. For the periodic-Schur backends, rows of the reduced Hessenberg
factor `k` whose Euclidean norm is below `scale_tol[k]` are set to exact zero
before the Schur iteration. `None` is exact-mode shorthand for a zero cutoff.
The dense backend accepts the same option but has no Hessenberg stage and
therefore ignores it, as does Schur-GMRES when unequal active ranks select its
dense fallback. Returned residuals are always evaluated against the original
active `H_k`, including any coupling removed by this deflation. If the
deflated periodic Schur iteration does not converge, the compressed solver
retries once with the original undeflated factors; other native failures are
returned directly.

Both structured solvers are XLA CPU FFI handlers. They call the GIL-free raw
periodic-Schur API directly inside the native solve; no Python host callback
or intermediate Schur-factor copy occurs. The Python-visible
`periodic_schur_D/Z` functions allocate arrays around the same raw API.

The Galerkin coordinate solve uses the structured route for equal and unequal
active dimensions. Unequal dimensions are zero padded to the largest active
dimension; the nonsingular right factors force the artificial solution
coordinates to zero. Schur-GMRES retains its exact dense rectangular fallback
because its Euclidean least-squares norm is not invariant under the
rank-deficient map returned at a smaller active site.

For complex Schur form every `w[k]` is triangular. Schur-GMRES processes
columns one at a time with `ZGEQRF` and `ZTRTRS`; Schur-Galerkin traverses the
triangular `T[k]` rows backward and solves each scalar cyclic diagonal block
with `ZGESV`. Complex64 inputs are promoted internally to complex128, just as
float32 inputs are promoted to float64.

## Dense gauge-reduced Bartels--Stewart solver

`dense.py` contains the pure-JAX four-factor periodic solver used by dCTMRG.
It cycles the periodic equation to one dense Sylvester equation, restricts
that equation to the complement of the retained left Ritz space, and
propagates the remaining three solutions backward. Both ordinary dense
pseudoinverses and padded real/complex Schur factors are supported.

## Build and test

Build from this folder so generated artifacts remain local:

```bash
python linalg/periodic_schur/build_slicot.py
python linalg/periodic_schur/setup_periodic_schur.py build_ext --inplace
python linalg/sylvester_solvers/setup.py build_ext --inplace
```

Then run the focused regression from the repository root:

```bash
python -m pytest -q tests/test_krylov_schur.py -k 'sylvester_compressed_schur_gmres or real_schur_block_uses_ffi_under_jit'
python -m pytest -q tests/test_periodic_schur.py -k 'periodic_sylvester_compressed or periodic_krylov_sylvester'
```

The extension computes internally in `float64` or `complex128` and runs on the
CPU. The typed FFI entry points can be traced through an enclosing `jax.jit`.
The FFI kernels intentionally have no JVP or transpose rule.
