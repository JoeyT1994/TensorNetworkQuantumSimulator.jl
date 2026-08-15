# Local SLICOT backend

This package uses a small f2py extension, `_slicot_periodic`, for periodic and
generalized periodic Schur operations. The extension is a generated local
artifact: edit the `.pyf` interface or Python adapters, then rebuild it. Do not
edit generated C, object, or shared-library files.

## Routine paths

| Operation | SLICOT path | Python adapter |
|---|---|---|
| Real periodic Schur | `MB03VD -> MB03VY -> MB03WD` | `periodic_schur.pyx`, `slicot_interface.py` |
| Real periodic Schur eigenvalues | `MB03WX` | `periodic_schur.pyx` |
| Complex periodic Schur eigenvalues | diagonal products | `periodic_schur.pyx` |
| Small real periodic Sylvester blocks | `MB03KE` | native Cython C API |
| Real periodic reordering | `MB03KD` | `periodic_schur.pyx`, `slicot_interface.py` |
| Complex periodic reordering | custom 1-by-1 exchanges | `periodic_schur.pyx` |
| Real generalized periodic Schur | `MB03BD` | `generalized_periodic_schur.py` |
| Complex generalized periodic Schur | `MB03BZ` | `generalized_periodic_schur.py` |

`MB04PY` is a helper required by the ordinary periodic decomposition. The full
source closure is listed once in `build_slicot.py`.

The legacy ordinary adapter in `slicot_interface.py` uses

```text
A[l] @ Z[l] = Z[(l + 1) % period] @ T[l].
```

The unified Cython driver instead uses the convention documented below. Keep
each boundary's period packing internal; do not expose SLICOT's trailing-period
storage convention to callers.

## Active rectangular preprocessing

### Disabled legacy cyclic reducers

The older `_cyclic_rank_reduce` and `signed_cyclic_reduce` paths are disabled
with `NotImplementedError`. They determined a leading rank by stopping at the
first subthreshold diagonal of an unpivoted QR or RQ factor. This is a
correctness bug for the block-Arnoldi pattern: inactive columns can be
interspersed, so a null column in the middle can be followed by independent
live columns. For example, a triangular diagonal pattern `[1, 0, 1]` has rank
two, but the legacy non-pivoted reducer returned rank one.

Do not repair this by merely counting the later nonzero diagonals and slicing
the first `rank` columns of the unpivoted Q; that basis still need not span the
later live coordinates.  The valid choices are explicit `active_cols`
compaction (the intended Arnoldi path) or a genuinely rank-revealing pivoted
factorization with its permutation propagated consistently.  The existing
`schur_rank_pivoting=True` option avoids the simple example, but it does not
encode the known Arnoldi active-column map and is not the preferred new path.

periodic_schur.pyx exposes two complete production entry points:

    periodic_schur_D(H, active_cols=None, reduction="NRed", rank_tol=None,
                     schur_deflation_tol=10.0)
    periodic_schur_Z(H, active_cols=None, reduction="NRed", rank_tol=None)

H has leading-period shape (period, m, m) and is already ordered as the formal
product H[0] H[1] ... H[p-1]. The driver has only the R-oriented relation
shown below; Arnoldi callers own any L-to-R relabeling. When active_cols is
None, every coordinate is active. A scalar integer active_cols=r selects the
prefix [:r] at every periodic cut; a (period, m) mask retains the explicit
possibly-scattered selection. The reduction argument accepts NRed or CRed.
NRed_D/Z retain the maximum active rank. CRed_D/Z rotate to the minimum active
rank and cyclically reduce to a square eig-only problem.

The public outputs are final, site-ordered periodic Schur data:

    D: T, Z, wr, wi
    Z: T, Z, alpha, beta, scale

T has shape (period, n, n), Z has shape (period, m, n), and

    H[k] @ Z[(k + 1) % period] = Z[k] @ T[k].

The real eigenvalues are wr + 1j*wi. The complex routine retains SLICOT's
scaled representation alpha / beta * 2**scale rather than forming it inside
the kernel, because the decoded values can overflow. Callers do not receive
active-compaction buffers, periodic Hessenberg factors, bases, q, or a cyclic
cut offset.

The same extension exposes standalone eigenvalue-readout and reorder boundaries

    periodic_schur_eigenvalues(T, rank=None)
    reorder_periodic_schur_D(T, Z, select, tol=100.0)
    reorder_periodic_schur_Z(T, Z, select, tol=100.0)

The eigenvalue readout accepts public leading-period Schur factors and returns
one ordered `complex128` array. For real factors it calls `MB03WX`, preserving
real 2-by-2 blocks; SLICOT has no dedicated complex triangular readout, so the
complex route multiplies corresponding diagonal entries. A supplied `rank`
uses only `T[:, :rank, :rank]` and returns length `rank`.

For reordering, `T` has shape `(period, n, n)`, `Z` may have the lifted rectangular
shape `(period, m, n)`, and selected blocks are moved to the leading sector.
The real routine treats conjugate-pair blocks atomically. Every block in the
complex Schur form is 1-by-1, so the complex routine uses stable adjacent
exchanges. The returned `T_ord, Z_ord` preserve

    H[k] @ Z_ord[k+1] = Z_ord[k] @ T_ord[k].

Internally, the wrapper reverses the factor sequence and cyclic basis indexing
for `MB03KD`, whose formal positive product is stored as `T_K ... T_1`, then
maps the updated factors and basis rotations back to the public order.

For each complex adjacent exchange, the local factors are

    A[k] = [[a[k], d[k]], [0, b[k]]].

The kernel row-scales and solves the scalar periodic Sylvester recurrence

    a[k] x[k+1] - b[k] x[k] = -d[k],

then uses `ZLARTG` to map `[x[k], 1]` to the first coordinate. The resulting
sitewise rotations swap the two periodic eigenvalues. As in the production
real wrapper, the complex path performs the weak stability test and rejects
an ill-conditioned exchange before changing the caller-visible copies.

## Internal stage-state mathematics

The overall Cython routine owns the following three internal stages.

### Stage 1: exact active compaction

Let r[k] be the number of active Arnoldi coordinates at cut k, and let

    E[k] : C**r[k] -> C**m

contain the corresponding coordinate columns. E[k] is an inclusion, while
E[k]^H is the projection into active coordinates:

    E[k]^H E[k] = I,
    E[k] E[k]^H = P_active[k].

The logical compact factor is

    C[k] = E[k]^H H[k] E[k+1],

with shape (r[k], r[k+1]). For an exact active mask,

    H[k] E[k+1] = E[k] C[k].

Stage 1 preserves the incoming cyclic order unconditionally. It never chooses
a minimum-rank cut and never produces cut_offset. Internally, C[k] occupies the
top-left rectangle of a trailing-period Fortran buffer with capacity

    n_max = max(ranks),

and bases[:, :r[k], k] stores E[k]. The unused columns and factor entries are
exact zeros. Thus the active-coordinate removal is represented by the
rectangular E[k] maps, even though fixed-capacity storage is used.

### Stage 2: periodic Hessenberg preparation

NRed retains n = n_max. Define the padded inclusion

    B[k] = [E[k]  0],

and let Cbar[k] be C[k] padded to n by n. Then

    H[k] B[k+1] = B[k] Cbar[k].

The Hessenberg reduction constructs square unitary Q_H[k] such that

    Cbar[k] Q_H[k+1] = Q_H[k] K[k],

where K[0] is upper Hessenberg and later K[k] are upper triangular. NRed_D
uses SLICOT MB03VD followed by MB03VY. NRed_Z uses the custom square complex
Householder sweep because SLICOT has no complex MB03VD/VY analogue.

The internal bases buffer remains B[k]; the square internal q buffer contains
Q_H[k]. When r[k] < n, B[k] and B[k] Q_H[k] have rank r[k], so they are
intertwining maps rather than orthonormal n-column bases. This rank-deficient
rectangular structure is how NRed retains the padded zero-product sector
without inventing additional Arnoldi directions.

CRed instead uses n = min(ranks). It chooses and rotates the first minimum-rank
cut to zero internally, then a reverse thin-QR sweep constructs rectangular
isometries

    W[k] : C**n -> C**r[k],

and folds them into bases as E[k] W[k]. CRed alone owns cut_offset. After the
reverse thin-QR sweep, it rotates the square factors and folded bases back to
incoming period order before the ordinary Hessenberg reduction. Consequently
the real public form always has its quasi-triangular factor at position zero,
as required by `MB03WX` eigenvalue extraction and the fixed-`KSCHUR` reorder
boundary. No cut offset crosses the stage-2 or public boundaries.

TODO: fuse CRed and periodic Hessenberg preparation so the rotated triangular
structure can be consumed without the separate canonical-order array pass.

When CRed receives `rank_tol`, it additionally scans every square factor with
pivoted QR and retains pivot `i` when

    abs(R[i, i]) > rank_tol * eps * incoming_column_norm[piv[i]].

The smallest revealed range is folded into its node basis and propagated
around the cycle by another reverse thin-QR sweep. This scan and contraction
repeat until the square cycle is numerically full rank. The option is confined
to eig-only CRed; NRed and periodic Sylvester retain their padded dimensions.

### Stage 3: SLICOT periodic Schur

For real data, MB03WD reduces the prepared periodic Hessenberg product, then
MB03WX recomputes the eigenvalues from the final Schur blocks so their order
matches the returned factors. For complex data, all-positive MB03BZ performs
the periodic Schur step. If S[k] denotes the additional Schur transformations,
SLICOT accumulates them into the existing square workspace:

    q[k] = Q_H[k] S[k],
    K[k] S[k+1] = S[k] T[k].

Immediately before MB03WD, the real path applies the same implicit-product
subdiagonal test with its machine-precision term multiplied by
`schur_deflation_tol`, whose default is 10. A split is committed only when the
deleted subdiagonal of the distinguished Hessenberg factor is also within its
`O(n eps)` backward scale. This guard prevents an exact-zero diagonal in a
later triangular factor from triggering deletion of an order-one entry. The
complex path is unchanged.

The overall driver then forms the only basis map returned publicly:

    Z[k] = bases[:, :n, k] @ q[k].

Consequently H[k] Z[k+1] = Z[k] T[k]. The periodic Hessenberg factors, bases,
and q cease to be observable once this composition is complete.

The NRed production path calls the linked Fortran routines from GIL-free
Cython entry points. Its private workspaces are Fortran contiguous, while the
Python and C API outputs are ordinary leading-period C-order arrays. The
descriptive Cython API names are

    compute_periodic_schur_active_D
    compute_periodic_schur_active_Z

and accept an output width independently of the compact active capacity.
Other Cython extensions use compact output width `n`. The XLA FFI handlers use
the static input width `m`; the common composition stage writes the leading
compact blocks and initializes the remaining entries to zero. The FFI returns
`n` explicitly. For complex data its inactive eigenvalue representation is
`alpha=0`, `beta=1`, and `scale=0`.

The Python-visible stage hooks and all three production adapters call the same
buffer-level compaction, Hessenberg, Schur, and composition functions. The
f2py module remains available only for direct routine experiments.

The JAX wrappers live in `linalg/periodic_schur/jax_ffi.py`:

    linalg.periodic_schur.jax_ffi.periodic_schur_eigenvalues
    linalg.periodic_schur.jax_ffi.periodic_schur_D
    linalg.periodic_schur.jax_ffi.periodic_schur_Z
    linalg.periodic_schur.jax_ffi.reorder_periodic_schur_D
    linalg.periodic_schur.jax_ffi.reorder_periodic_schur_Z

They use static outputs, work under `jax.jit`, and declare sequential
`jax.vmap` support. Eigenvalue readout always returns the full static Schur
width. Decomposition returns full input-width zero-padded arrays
and the live Schur rank `n`. The default NRed/CRed routes use the native FFI;
the optional data-dependent QRP CRed route currently calls the host driver
through `jax.pure_callback` and pads its dynamic result. Reordering accepts
the live rank as `schur_size`, packs only the leading live `T` and `Z` sectors,
and zero-repads the reordered result. All four preserve the same R-oriented
public relation.

## Archived CRed kernel experiments

Alternative CRed kernels are quarantined in
make_periodic_hessenberg_experiments.pyx and are excluded from the production
build. That source preserves full and panel economic-QR sweeps, the Givens
chase, fused Householder D/Z kernels, and the num_rows specialization.

Early benchmarks found no useful end-to-end advantage from panel QR or
num_rows because fill-in rapidly destroys the initial block-Hessenberg
sparsity. The fused Householder path was likewise not meaningfully faster than
the modular QR-plus-Hessenberg route. The production CRed path therefore uses
the direct reverse thin-QR sweep followed by the ordinary square
Hessenberg/Schur stages.


## Build

From the repository root, using the intended Python/Conda environment:

```bash
python linalg/periodic_schur/build_slicot.py
python linalg/periodic_schur/setup_periodic_schur.py build_ext --inplace
```

The first command builds both the Fortran/f2py SLICOT ABI module and
`generated/build_slicot/libslicot_periodic.a`. The second compiles
`periodic_schur.pyx` together with `ffi.cc` and links the static archive into
the single production extension. NRed_D/Z and eig-only CRed_D/Z work end to
end. The experimental Cython source is not part of this build.

`setup_periodic_schur.py` declares the static archive as an extension build
dependency. Preserve that dependency: without it, rebuilding
`libslicot_periodic.a` can leave the previously linked, potentially
unoptimized Fortran objects inside `_periodic_schur`. For an older checkout
that lacks the dependency, force the second command with `--force`.

The complete pristine SLICOT 5.9.1 release is bundled as
`linalg/periodic_schur/SLICOT/SLICOT-Reference-5.9.1.tar.gz`. The build helper
verifies the pinned SHA-256, extracts the source into its temporary build
directory, smoke-tests the exported routines, and installs the ABI-tagged
extension beside `slicot_interface.py` plus the static archive under
`generated/`. An alternate expanded source checkout can be selected explicitly:

```bash
python linalg/periodic_schur/build_slicot.py --source /path/to/SLICOT-Reference
```

The active environment needs NumPy/f2py, Meson, Ninja, and gfortran. Rebuild
after changing Python or NumPy environments; do not copy an ABI-tagged `.so`
between environments.

The important Fortran flags are:

```text
-O3 -fPIC -frecursive -fallow-argument-mismatch -mcpu=native -mtune=native
```

SLICOT's own release configuration uses `-O3`. The first local extension was
compiled manually with only `-fPIC`, leaving gfortran at `-O0`; this made
`MB03WD` about four times slower. `-Ofast`, explicit loop unrolling, and LTO did
not materially improve the optimized result, so the build keeps standard
floating-point semantics.

On macOS, the extension must link directly to Accelerate. The build helper
adds `-framework Accelerate` and verifies the resulting dependency. Linking
through Conda's BLAS re-export adds measurable overhead to these small,
BLAS-call-heavy routines.

Check the installed binary with:

```bash
otool -L linalg/periodic_schur/_slicot_periodic*.so
```

The output should contain `Accelerate.framework`, not `libblas_reexport` or
`libvecLibFort-ng`.

## Correctness validation

Run the focused decomposition and reordering coverage:

```bash
python -m pytest -q tests/test_periodic_schur.py -k 'slicot or generalized'
```

For an ordinary periodic decomposition, validate the sitewise relation above,
orthogonality of every `Z[l]`, the Hessenberg/triangular structure, and the
eigenvalues. Do not validate only the eigenvalues of the explicit product; that
would miss period-order and basis-index errors.

## Performance validation

Run the fixed FP64 comparison used to diagnose the build:

```bash
python linalg/periodic_schur/benchmark_slicot.py
```

The benchmark uses `period=4`, `n=27`, warm calls, and reports median and
minimum time. A representative Apple Accelerate result from 2026-07-21 was:

| Operation | Original `-O0` build | Optimized build |
|---|---:|---:|
| Full periodic SLICOT | 0.837 ms | 0.257 ms |
| `MB03WD`, Schur form and vectors | 0.781 ms | 0.208 ms |
| JAX brute periodic Schur | 0.202 ms | 0.212 ms |
| SciPy real Schur of explicit product | 0.066 ms | 0.068 ms |

Treat timings as exploratory data rather than test thresholds. Record the
period, matrix size, dtype, compiler flags, linked BLAS/LAPACK, warmup, and
residual whenever reporting new numbers.

SciPy's ordinary Schur timing is not the same operation: it starts from the
explicitly formed one-cycle product and does not return the full set of
periodic factors and site bases. After the corrected build, the remaining gap
to product-based JAX/SciPy paths is algorithmic rather than an f2py compilation
problem.

## Failure checklist

- Runtime near the old 0.8 ms result: inspect the compile output for `-O3` and
  rebuild with `build_slicot.py`.
- Wrong macOS dependency: inspect `LDFLAGS` and confirm direct Accelerate
  linkage with `otool -L`.
- Import or architecture error: rebuild inside the active Python environment.
- Correct eigenvalues but wrong sitewise residual: inspect factor packing,
  basis indexing, and the `MB03KD` period roll before changing tolerances.
