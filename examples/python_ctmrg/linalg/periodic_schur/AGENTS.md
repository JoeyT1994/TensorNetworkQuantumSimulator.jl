# Repository Guidelines for `periodic_schur`

## Scope
- Read this folder when working explicitly on generalized periodic Schur, signed periodic products, `generalized_phessenberg`, SLICOT periodic Schur/reorder adapters, or the Cython Givens chase.
- Do not read this folder as background for ordinary CTMRG, MPS BP, DMRG, Pauli helpers, or unrelated Krylov experiments unless an import or failure points here.

## Layout
- `__init__.py` contains the public low-level periodic Schur wrappers and the
  host callbacks used by `linalg/periodic_krylov_schur.py`.
- `generalized_periodic_schur.py` contains the signed generalized Hessenberg
  orchestration and SLICOT `MB03BD/MB03BZ` adapters. The signed square Givens
  chase itself is implemented in `periodic_schur.pyx`.
- `julia_version.py` contains the dormant Julia reference implementation and is never a production dependency.
- `periodic_schur.pyx` exposes the complete `periodic_schur_D/Z(H,
  active_cols=None, reduction="NRed"|"CRed")` boundary. A scalar integer
  `active_cols=r` is Python shorthand for the same prefix rank at every cut;
  the native ABI remains mask-only. It owns active compaction, Hessenberg
  preparation, SLICOT Schur, basis composition, and any rotation restoration.
  It uses the R-oriented relation only. NRed retains the maximum active rank;
  eig-only CRed rotates to a minimum-rank cut and applies a reverse thin-QR
  sweep, then restores incoming period order before the usual square
  Hessenberg/Schur stages. Callers must not orchestrate the private stages.
- `_periodic_schur.pxd` declares the descriptive Cython APIs
  `compute_periodic_schur_active_D/Z` and
  `compute_reordered_periodic_schur_D/Z` used by other extensions and FFI.
- `ffi.cc` contains only the XLA FFI ABI handlers. `jax_ffi.py` registers those
  handlers and exposes static full-size JAX wrappers; neither file implements
  a numerical stage.
- `make_periodic_hessenberg_experiments.pyx` archives the unbuilt CRed
  experiments: full/panel QR, Givens, fused Householder, and num_rows kernels.
  Do not expose those experimental symbols through the production extension.
- `setup_periodic_schur.py` builds `periodic_schur.pyx` and `ffi.cc` into the
  unified `linalg.periodic_schur._periodic_schur` extension.
- `slicot_periodic.pyf` is the f2py interface for the local SLICOT extension.
- `SLICOT/SLICOT-Reference-5.9.1.tar.gz` is the complete pristine upstream
  source release. Keep local wrappers and experiments outside the archive.
- `SLICOT.md` is the source of truth for routine roles, optimized build and
  linkage requirements, correctness checks, and performance methodology. Read
  it before rebuilding, changing, or benchmarking the SLICOT extension.
- `build_slicot.py` verifies and extracts the bundled release into its temporary
  build directory by default, or accepts an alternate source tree, then builds
  the extension with release optimization and the platform BLAS/LAPACK
  implementation.
- `benchmark_slicot.py` reproduces the fixed FP64 build-performance comparison.
- `generated/`, `*.c`, `*.html`, object files, and build directories are generated artifacts. Do not read them during normal work; inspect them only when debugging Cython/f2py build output or generated-code behavior.

## Local Rules
- Keep imports of `_periodic_schur` and `_slicot_periodic` lazy, so plain module import does not require compiled extensions.
- Keep Julia reference code isolated in `julia_version.py`; production modules must not import it.
- Preserve the signed-product convention: `True` means `A[l]`, and `False` means `A[l]^{-1}`.
