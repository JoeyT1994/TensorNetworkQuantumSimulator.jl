"""Internal JAX CPU FFI bindings for compressed Sylvester solvers."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np


@cache
def _register_ffi_targets():
    """Load the native extension and register all compressed solver targets."""
    import ctypes

    from . import _sylvester_solvers as extension

    library = ctypes.CDLL(extension.__file__)
    targets = {
        "sylvester_compressed_schur_gmres_real_f64":
            library.SylvesterCompressedSchurGmresRealF64,
        "sylvester_compressed_periodic_dense_gmres_real_f64":
            library.SylvesterCompressedPeriodicDenseGmresRealF64,
        "sylvester_compressed_periodic_dense_gmres_complex_c128":
            library.SylvesterCompressedPeriodicDenseGmresComplexC128,
        "sylvester_compressed_periodic_schur_galerkin_complex_c128":
            library.SylvesterCompressedPeriodicSchurGalerkinComplexC128,
        "sylvester_compressed_periodic_schur_gmres_real_f64":
            library.SylvesterCompressedPeriodicSchurGmresRealF64,
        "sylvester_compressed_periodic_schur_galerkin_real_f64":
            library.SylvesterCompressedPeriodicSchurGalerkinRealF64,
        "sylvester_compressed_periodic_schur_gmres_complex_c128":
            library.SylvesterCompressedPeriodicSchurGmresComplexC128,
    }
    for name, target in targets.items():
        jax.ffi.register_ffi_target(
            name,
            jax.ffi.pycapsule(target),
            platform="cpu",
        )
    return library


def _sylvester_compressed_schur_gmres_real(
    H,
    w,
    v,
    residual_r,
    block_2x2_start,
    w_triangular,
    tpqrt_block_size=32,
):
    """Solve one real compressed Sylvester problem through XLA FFI."""
    _register_ffi_targets()
    dtype = H.dtype
    result_types = (
        jax.ShapeDtypeStruct((w.shape[0], H.shape[0]), jnp.float64),
        jax.ShapeDtypeStruct((w.shape[0],), jnp.float64),
    )
    call = jax.ffi.ffi_call(
        "sylvester_compressed_schur_gmres_real_f64",
        result_types,
        vmap_method="sequential",
    )
    x, error = call(
        H.astype(jnp.float64),
        w.astype(jnp.float64),
        v.astype(jnp.float64),
        residual_r.astype(jnp.float64),
        block_2x2_start,
        upper=np.bool_(w_triangular == "upper"),
        tpqrt_block_size=np.int64(tpqrt_block_size),
    )
    return x.astype(dtype), error.astype(dtype)


def _sylvester_compressed_periodic_dense_gmres_real(
    H,
    w,
    v,
    residual_r,
    block_2x2_start,
    active_cols,
    rank,
):
    r"""Solve real periodic compressed GMRES with a dense projected problem."""
    _register_ffi_targets()
    dtype = H.dtype
    period, n_krylov = H.shape[:2]
    d_block = w.shape[1]
    result_types = (
        jax.ShapeDtypeStruct((period, d_block, n_krylov), jnp.float64),
        jax.ShapeDtypeStruct((period, d_block), jnp.float64),
    )
    call = jax.ffi.ffi_call(
        "sylvester_compressed_periodic_dense_gmres_real_f64",
        result_types,
        vmap_method="sequential",
    )
    x, error = call(
        H.astype(jnp.float64),
        w.astype(jnp.float64),
        v.astype(jnp.float64),
        residual_r.astype(jnp.float64),
        block_2x2_start,
        active_cols,
        jnp.asarray(rank, dtype=jnp.int32),
    )
    return x.astype(dtype), error.astype(dtype)


def _sylvester_compressed_periodic_dense_gmres_complex(
    H,
    w,
    v,
    residual_r,
    active_cols,
    rank,
):
    """Solve complex periodic compressed GMRES with a dense projected problem."""
    _register_ffi_targets()
    dtype = H.dtype
    real_dtype = {
        jnp.dtype(jnp.complex64): jnp.dtype(jnp.float32),
        jnp.dtype(jnp.complex128): jnp.dtype(jnp.float64),
    }[dtype]
    period, n_krylov = H.shape[:2]
    d_block = w.shape[1]
    result_types = (
        jax.ShapeDtypeStruct((period, d_block, n_krylov), jnp.complex128),
        jax.ShapeDtypeStruct((period, d_block), jnp.float64),
    )
    call = jax.ffi.ffi_call(
        "sylvester_compressed_periodic_dense_gmres_complex_c128",
        result_types,
        vmap_method="sequential",
    )
    x, error = call(
        H.astype(jnp.complex128),
        w.astype(jnp.complex128),
        v.astype(jnp.complex128),
        residual_r.astype(jnp.complex128),
        active_cols,
        jnp.asarray(rank, dtype=jnp.int32),
    )
    return x.astype(dtype), error.astype(real_dtype)


def _sylvester_compressed_periodic_schur_galerkin_complex(
    H,
    w,
    v,
    residual_r,
    scale_tol,
    active_cols,
    rank,
):
    """Solve complex periodic Galerkin equations using periodic Schur."""
    _register_ffi_targets()
    dtype = H.dtype
    real_dtype = {
        jnp.dtype(jnp.complex64): jnp.dtype(jnp.float32),
        jnp.dtype(jnp.complex128): jnp.dtype(jnp.float64),
    }[dtype]
    period, n_krylov = H.shape[:2]
    d_block = w.shape[1]
    result_types = (
        jax.ShapeDtypeStruct((period, d_block, n_krylov), jnp.complex128),
        jax.ShapeDtypeStruct((period, d_block), jnp.float64),
    )
    call = jax.ffi.ffi_call(
        "sylvester_compressed_periodic_schur_galerkin_complex_c128",
        result_types,
        vmap_method="sequential",
    )
    x, error = call(
        H.astype(jnp.complex128),
        w.astype(jnp.complex128),
        v.astype(jnp.complex128),
        residual_r.astype(jnp.complex128),
        scale_tol.astype(jnp.float64),
        active_cols,
        jnp.asarray(rank, dtype=jnp.int32),
    )
    return x.astype(dtype), error.astype(real_dtype)


def _sylvester_compressed_periodic_schur_gmres_real(
    H,
    w,
    v,
    residual_r,
    scale_tol,
    block_2x2_start,
    active_cols,
    rank,
):
    """Solve real periodic compressed GMRES using periodic Schur structure."""
    _register_ffi_targets()
    dtype = H.dtype
    period, n_krylov = H.shape[:2]
    d_block = w.shape[1]
    result_types = (
        jax.ShapeDtypeStruct((period, d_block, n_krylov), jnp.float64),
        jax.ShapeDtypeStruct((period, d_block), jnp.float64),
    )
    call = jax.ffi.ffi_call(
        "sylvester_compressed_periodic_schur_gmres_real_f64",
        result_types,
        vmap_method="sequential",
    )
    x, error = call(
        H.astype(jnp.float64),
        w.astype(jnp.float64),
        v.astype(jnp.float64),
        residual_r.astype(jnp.float64),
        scale_tol.astype(jnp.float64),
        block_2x2_start,
        active_cols,
        jnp.asarray(rank, dtype=jnp.int32),
    )
    return x.astype(dtype), error.astype(dtype)


def _sylvester_compressed_periodic_schur_galerkin_real(
    H,
    w,
    v,
    residual_r,
    scale_tol,
    block_2x2_start,
    active_cols,
    rank,
    *,
    galerkin_block_solver="dgesv",
):
    r"""Solve the real periodic projected equation using Schur structure."""
    _register_ffi_targets()
    if galerkin_block_solver not in ("dgesv", "mb03ke"):
        raise ValueError(
            f"unknown Galerkin block solver {galerkin_block_solver!r}"
        )
    dtype = H.dtype
    period, n_krylov = H.shape[:2]
    if galerkin_block_solver == "mb03ke" and period < 2:
        raise ValueError(
            "MB03KE requires a periodic sequence of length at least two"
        )
    d_block = w.shape[1]
    result_types = (
        jax.ShapeDtypeStruct((period, d_block, n_krylov), jnp.float64),
        jax.ShapeDtypeStruct((period, d_block), jnp.float64),
    )
    call = jax.ffi.ffi_call(
        "sylvester_compressed_periodic_schur_galerkin_real_f64",
        result_types,
        vmap_method="sequential",
    )
    x, error = call(
        H.astype(jnp.float64),
        w.astype(jnp.float64),
        v.astype(jnp.float64),
        residual_r.astype(jnp.float64),
        scale_tol.astype(jnp.float64),
        block_2x2_start,
        active_cols,
        jnp.asarray(rank, dtype=jnp.int32),
        use_mb03ke=np.bool_(galerkin_block_solver == "mb03ke"),
    )
    return x.astype(dtype), error.astype(dtype)


def _sylvester_compressed_periodic_schur_gmres_complex(
    H,
    w,
    v,
    residual_r,
    scale_tol,
    active_cols,
    rank,
):
    """Solve complex periodic compressed GMRES using Schur structure."""
    _register_ffi_targets()
    dtype = H.dtype
    real_dtype = {
        jnp.dtype(jnp.complex64): jnp.dtype(jnp.float32),
        jnp.dtype(jnp.complex128): jnp.dtype(jnp.float64),
    }[dtype]
    period, n_krylov = H.shape[:2]
    d_block = w.shape[1]
    result_types = (
        jax.ShapeDtypeStruct((period, d_block, n_krylov), jnp.complex128),
        jax.ShapeDtypeStruct((period, d_block), jnp.float64),
    )
    call = jax.ffi.ffi_call(
        "sylvester_compressed_periodic_schur_gmres_complex_c128",
        result_types,
        vmap_method="sequential",
    )
    x, error = call(
        H.astype(jnp.complex128),
        w.astype(jnp.complex128),
        v.astype(jnp.complex128),
        residual_r.astype(jnp.complex128),
        scale_tol.astype(jnp.float64),
        active_cols,
        jnp.asarray(rank, dtype=jnp.int32),
    )
    return x.astype(dtype), error.astype(real_dtype)
