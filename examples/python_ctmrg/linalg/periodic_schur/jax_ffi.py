"""JAX FFI adapters for the R-oriented active periodic-Schur drivers."""

from functools import cache

import jax
import jax.numpy as jnp


@cache
def _register_periodic_schur_real_ffi():
    """Register and retain the CPU handler for real periodic Schur."""
    import ctypes

    from . import _periodic_schur as extension

    library = ctypes.CDLL(extension.__file__)
    jax.ffi.register_ffi_target(
        "periodic_schur_active_real_f64",
        jax.ffi.pycapsule(library.PeriodicSchurActiveRealF64),
        platform="cpu",
    )
    return library


@cache
def _register_periodic_schur_real_cred_ffi():
    """Register and retain the CPU handler for real eig-only CRed Schur."""
    import ctypes

    from . import _periodic_schur as extension

    library = ctypes.CDLL(extension.__file__)
    jax.ffi.register_ffi_target(
        "periodic_schur_active_cred_real_f64",
        jax.ffi.pycapsule(library.PeriodicSchurActiveRealCRedF64),
        platform="cpu",
    )
    return library


@cache
def _register_periodic_schur_complex_ffi():
    """Register and retain the CPU handler for complex periodic Schur."""
    import ctypes

    from . import _periodic_schur as extension

    library = ctypes.CDLL(extension.__file__)
    jax.ffi.register_ffi_target(
        "periodic_schur_active_complex_c128",
        jax.ffi.pycapsule(library.PeriodicSchurActiveComplexC128),
        platform="cpu",
    )
    return library


@cache
def _register_periodic_schur_complex_cred_ffi():
    """Register and retain the CPU handler for complex eig-only CRed Schur."""
    import ctypes

    from . import _periodic_schur as extension

    library = ctypes.CDLL(extension.__file__)
    jax.ffi.register_ffi_target(
        "periodic_schur_active_cred_complex_c128",
        jax.ffi.pycapsule(library.PeriodicSchurActiveComplexCRedC128),
        platform="cpu",
    )
    return library


@cache
def _register_periodic_schur_reorder_real_ffi():
    """Register and retain the CPU handler for real periodic reordering."""
    import ctypes

    from . import _periodic_schur as extension

    library = ctypes.CDLL(extension.__file__)
    jax.ffi.register_ffi_target(
        "periodic_schur_reorder_real_f64",
        jax.ffi.pycapsule(library.PeriodicSchurReorderRealF64),
        platform="cpu",
    )
    return library


@cache
def _register_periodic_schur_reorder_complex_ffi():
    """Register and retain the CPU handler for complex periodic reordering."""
    import ctypes

    from . import _periodic_schur as extension

    library = ctypes.CDLL(extension.__file__)
    jax.ffi.register_ffi_target(
        "periodic_schur_reorder_complex_c128",
        jax.ffi.pycapsule(library.PeriodicSchurReorderComplexC128),
        platform="cpu",
    )
    return library


@cache
def _register_periodic_schur_eigenvalues_real_ffi():
    """Register and retain the CPU handler for real Schur eigenvalue readout."""
    import ctypes

    from . import _periodic_schur as extension

    library = ctypes.CDLL(extension.__file__)
    jax.ffi.register_ffi_target(
        "periodic_schur_eigenvalues_real_f64",
        jax.ffi.pycapsule(library.PeriodicSchurEigenvaluesRealF64),
        platform="cpu",
    )
    return library


@cache
def _register_periodic_schur_eigenvalues_complex_ffi():
    """Register and retain the CPU handler for complex Schur eigenvalue readout."""
    import ctypes

    from . import _periodic_schur as extension

    library = ctypes.CDLL(extension.__file__)
    jax.ffi.register_ffi_target(
        "periodic_schur_eigenvalues_complex_c128",
        jax.ffi.pycapsule(library.PeriodicSchurEigenvaluesComplexC128),
        platform="cpu",
    )
    return library


def _periodic_schur_inputs(H, active_cols, dtype):
    """Return exact-precision square factors and their static active mask."""
    H = jnp.asarray(H)
    if H.dtype != dtype:
        raise TypeError(f"H must have dtype {dtype}")
    if H.ndim != 3 or H.shape[1] != H.shape[2]:
        raise ValueError("H must have shape (period, m, m)")
    if H.shape[0] < 1:
        raise ValueError("H must contain at least one factor")
    if active_cols is None:
        active_cols = jnp.ones(H.shape[:2], dtype=jnp.bool_)
    else:
        active_cols = jnp.asarray(active_cols)
        if active_cols.ndim == 0:
            if not jnp.issubdtype(active_cols.dtype, jnp.integer):
                raise TypeError("scalar active_cols must be an integer rank")
            active_cols = jnp.broadcast_to(
                jnp.arange(H.shape[1]) < active_cols,
                H.shape[:2],
            )
        else:
            active_cols = active_cols.astype(jnp.bool_)
    if active_cols.shape != H.shape[:2]:
        raise ValueError("active_cols must have shape (period, m)")
    return H, active_cols


def _periodic_schur_reorder_inputs(T, Z, select, schur_size, dtype):
    """Return static Schur carriers, their selection, and live prefix size."""
    T = jnp.asarray(T)
    Z = jnp.asarray(Z)
    if T.dtype != dtype or Z.dtype != dtype:
        raise TypeError(f"T and Z must have dtype {dtype}")
    if T.ndim != 3 or T.shape[1] != T.shape[2]:
        raise ValueError("T must have shape (period, n, n)")
    period, n, _ = T.shape
    if period < 1:
        raise ValueError("T must contain at least one factor")
    if Z.ndim != 3 or Z.shape[0] != period or Z.shape[2] != n:
        raise ValueError("Z must have shape (period, m, n)")
    select = jnp.asarray(select, dtype=jnp.bool_)
    if select.shape != (n,):
        raise ValueError("select must have shape (n,)")
    schur_size = jnp.asarray(schur_size, dtype=jnp.int32)
    if schur_size.shape != ():
        raise ValueError("schur_size must be a scalar")
    return T, Z, select, schur_size


def periodic_schur_eigenvalues(T):
    r"""Return all ordered eigenvalues of static periodic Schur factors.

    ``T`` must have shape ``(period, n, n)`` and dtype ``float64`` or
    ``complex128``. The returned ``complex128`` array always has length ``n``.
    ``jax.vmap`` is supported through sequential native calls.
    """
    T = jnp.asarray(T)
    if T.ndim != 3 or T.shape[1] != T.shape[2]:
        raise ValueError("T must have shape (period, n, n)")
    if T.shape[0] < 1:
        raise ValueError("T must contain at least one factor")
    if T.dtype == jnp.float64:
        _register_periodic_schur_eigenvalues_real_ffi()
        target = "periodic_schur_eigenvalues_real_f64"
    elif T.dtype == jnp.complex128:
        _register_periodic_schur_eigenvalues_complex_ffi()
        target = "periodic_schur_eigenvalues_complex_c128"
    else:
        raise TypeError("T must have dtype float64 or complex128")

    result_type = jax.ShapeDtypeStruct((T.shape[1],), jnp.complex128)
    call = jax.ffi.ffi_call(
        target,
        result_type,
        vmap_method="sequential",
    )
    return call(T)


def periodic_schur_D(
    H,
    active_cols=None,
    reduction="NRed",
    rank_tol=None,
    schur_deflation_tol=10.0,
):
    r"""Return a static-padded real NRed or eig-only CRed decomposition.

    The R-oriented relation is
    ``H[k] @ Z[k+1] = Z[k] @ T[k]``. If the compact active capacity is ``n``,
    the leading ``n`` rows and columns of ``T`` and leading ``n`` columns of
    ``Z`` contain the decomposition. Remaining entries are exact zeros. A
    scalar integer ``active_cols=r`` selects the prefix ``[:r]`` at every cut.

    ``rank_tol`` enables iterative pivoted-QR deflation for CRed. Each pivot is
    compared to its incoming column norm in machine-epsilon units.
    ``schur_deflation_tol`` multiplies the real post-Hessenberg split test.

    Returns ``(T, Z, wr, wi, n)`` with static shapes based on the input width.
    ``jax.vmap`` is supported through sequential host calls.
    """
    if reduction == "NRed":
        if rank_tol is not None:
            raise ValueError("rank_tol is available only for CRed")
        _register_periodic_schur_real_ffi()
        target = "periodic_schur_active_real_f64"
    elif reduction == "CRed":
        if rank_tol is None:
            _register_periodic_schur_real_cred_ffi()
            target = "periodic_schur_active_cred_real_f64"
        else:
            target = None
    else:
        raise ValueError("reduction must be 'NRed' or 'CRed'")
    H, active_cols = _periodic_schur_inputs(H, active_cols, jnp.float64)
    period, m, _ = H.shape
    result_types = (
        jax.ShapeDtypeStruct((period, m, m), jnp.float64),
        jax.ShapeDtypeStruct((period, m, m), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    if target is None:
        from . import _periodic_schur_cred_qrp_D_callback

        return jax.pure_callback(
            _periodic_schur_cred_qrp_D_callback,
            result_types,
            H,
            active_cols,
            jnp.asarray(rank_tol, dtype=jnp.float64),
            jnp.asarray(schur_deflation_tol, dtype=jnp.float64),
            vmap_method="sequential",
        )
    call = jax.ffi.ffi_call(
        target,
        result_types,
        vmap_method="sequential",
    )
    return call(
        H,
        active_cols,
        jnp.asarray(schur_deflation_tol, dtype=jnp.float64),
    )


def periodic_schur_Z(
    H,
    active_cols=None,
    reduction="NRed",
    rank_tol=None,
):
    r"""Return a static-padded complex NRed or eig-only CRed decomposition.

    The R-oriented relation and padding convention match
    :func:`periodic_schur_D`. Returns
    ``(T, Z, alpha, beta, scale, n)``. Inactive eigenvalue entries use
    ``alpha=0``, ``beta=1``, and ``scale=0``. Scalar ``active_cols`` follows
    the same prefix-rank convention. ``rank_tol`` enables the same iterative
    pivoted-QR contraction as the real driver.
    """
    if reduction == "NRed":
        if rank_tol is not None:
            raise ValueError("rank_tol is available only for CRed")
        _register_periodic_schur_complex_ffi()
        target = "periodic_schur_active_complex_c128"
    elif reduction == "CRed":
        if rank_tol is None:
            _register_periodic_schur_complex_cred_ffi()
            target = "periodic_schur_active_cred_complex_c128"
        else:
            target = None
    else:
        raise ValueError("reduction must be 'NRed' or 'CRed'")
    H, active_cols = _periodic_schur_inputs(H, active_cols, jnp.complex128)
    period, m, _ = H.shape
    result_types = (
        jax.ShapeDtypeStruct((period, m, m), jnp.complex128),
        jax.ShapeDtypeStruct((period, m, m), jnp.complex128),
        jax.ShapeDtypeStruct((m,), jnp.complex128),
        jax.ShapeDtypeStruct((m,), jnp.complex128),
        jax.ShapeDtypeStruct((m,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    if target is None:
        from . import _periodic_schur_cred_qrp_Z_callback

        return jax.pure_callback(
            _periodic_schur_cred_qrp_Z_callback,
            result_types,
            H,
            active_cols,
            jnp.asarray(rank_tol, dtype=jnp.float64),
            vmap_method="sequential",
        )
    call = jax.ffi.ffi_call(
        target,
        result_types,
        vmap_method="sequential",
    )
    return call(H, active_cols)


def reorder_periodic_schur_D(T, Z, select, schur_size, tol=100.0):
    r"""Move selected real periodic Schur blocks to the leading sector.

    ``T`` and ``Z`` satisfy the R-oriented relation
    ``H[k] @ Z[k+1] = Z[k] @ T[k]``. Real 2-by-2 blocks are selected
    atomically. ``schur_size`` is the live leading dimension within the static
    carriers. The returned arrays preserve the input shapes, zero padding, and
    relation.
    """
    _register_periodic_schur_reorder_real_ffi()
    T, Z, select, schur_size = _periodic_schur_reorder_inputs(
        T,
        Z,
        select,
        schur_size,
        jnp.float64,
    )
    result_types = (
        jax.ShapeDtypeStruct(T.shape, jnp.float64),
        jax.ShapeDtypeStruct(Z.shape, jnp.float64),
    )
    call = jax.ffi.ffi_call(
        "periodic_schur_reorder_real_f64",
        result_types,
        vmap_method="sequential",
    )
    return call(
        T,
        Z,
        select,
        schur_size,
        jnp.asarray(tol, dtype=jnp.float64),
    )


def reorder_periodic_schur_Z(T, Z, select, schur_size, tol=100.0):
    r"""Move selected complex periodic Schur entries to the leading sector.

    The R-oriented relation, static output shapes, and sequential ``vmap``
    behavior match :func:`reorder_periodic_schur_D`. Only the leading
    ``schur_size`` columns participate in adjacent exchanges.
    """
    _register_periodic_schur_reorder_complex_ffi()
    T, Z, select, schur_size = _periodic_schur_reorder_inputs(
        T,
        Z,
        select,
        schur_size,
        jnp.complex128,
    )
    result_types = (
        jax.ShapeDtypeStruct(T.shape, jnp.complex128),
        jax.ShapeDtypeStruct(Z.shape, jnp.complex128),
    )
    call = jax.ffi.ffi_call(
        "periodic_schur_reorder_complex_c128",
        result_types,
        vmap_method="sequential",
    )
    return call(
        T,
        Z,
        select,
        schur_size,
        jnp.asarray(tol, dtype=jnp.float64),
    )
