"""Isolated Julia reference backend for ordinary periodic Schur operations.

The production periodic Krylov--Schur path does not import this module.  It is
kept as a reference implementation while the active path uses LAPACK for one
factor and SLICOT for real multi-factor problems.
"""

from pathlib import Path
import threading

import jax.numpy as jnp
import numpy as np


_JULIA_PERIODIC_SCHUR = None
_JULIA_CALLBACK_LOCK = threading.Lock()


def _import_juliacall_main():
    """Import ``juliacall.Main`` without Julia's process-exit hook."""
    import atexit

    original_register = atexit.register
    juliacall_exit_hooks = []

    def capture_register(func):
        """Record Julia's exit hook while preserving normal registration."""
        result = original_register(func)
        if getattr(func, "__name__", None) == "at_jl_exit":
            juliacall_exit_hooks.append(func)
        return result

    try:
        atexit.register = capture_register
        from juliacall import Main as jl
    finally:
        atexit.register = original_register

    for hook in juliacall_exit_hooks:
        atexit.unregister(hook)
    return jl


def _julia_periodic_schur():
    """Return the lazily initialized Julia periodic Schur bridge."""
    global _JULIA_PERIODIC_SCHUR
    if _JULIA_PERIODIC_SCHUR is None:
        jl = _import_juliacall_main()
        jl.seval(
            r"""
            using LinearAlgebra
            using Logging
            using PythonCall
            using PeriodicSchurDecompositions

            function _pks_pack(P)
                p = P.period
                m = size(P.T1, 1)
                T = Array{eltype(P.T1)}(undef, m, m, p)
                for l in 1:p
                    if l == P.schurindex
                        T[:, :, l] = P.T1
                    else
                        idx = l < P.schurindex ? l : l - 1
                        T[:, :, l] = P.T[idx]
                    end
                end
                Z = Array{eltype(P.Z[1])}(undef, m, m, p)
                for l in 1:p
                    Z[:, :, l] = P.Z[l]
                end
                return T, Z, Vector{ComplexF64}(P.values)
            end

            function _pks_from_arrays(T_py, Z_py, eigvals_py)
                T = pyconvert(Array, T_py)
                Z = pyconvert(Array, Z_py)
                eigvals = Vector{ComplexF64}(pyconvert(Array, eigvals_py))
                p = size(T, 3)
                T1 = @view T[:, :, p]
                Ts = [@view T[:, :, l] for l in 1:(p - 1)]
                Zs = [@view Z[:, :, l] for l in 1:p]
                return PeriodicSchur(T1, Ts, Zs, eigvals, 'L', p)
            end

            function _pks_pschur(H_py)
                old_gc = GC.enable(false)
                try
                    H = pyconvert(Array, H_py)
                    p = size(H, 3)
                    As = [@view H[:, :, l] for l in 1:p]
                    P = pschur!(As, :L; wantZ=true)
                    return _pks_pack(P)
                finally
                    GC.enable(old_gc)
                end
            end

            function _pks_ordschur(T_py, Z_py, eigvals_py, select_py)
                old_gc = GC.enable(false)
                try
                    P = _pks_from_arrays(T_py, Z_py, eigvals_py)
                    select = Bool.(vec(pyconvert(Array, select_py)))
                    ordschur!(P, select; wantZ=true)
                    return _pks_pack(P)
                finally
                    GC.enable(old_gc)
                end
            end
            """
        )
        jl.seval("GC.enable(false); Logging.disable_logging(Logging.Error)")
        _JULIA_PERIODIC_SCHUR = jl
    return _JULIA_PERIODIC_SCHUR


def _period_axis_to_julia(x):
    """Move the leading period axis to Julia's trailing position."""
    return np.moveaxis(np.asarray(x), 0, -1)


def _period_axis_from_julia(x):
    """Move Julia's trailing period axis back to the leading position."""
    return np.moveaxis(np.asarray(x), -1, 0)


def _compact_periodic_schur_inputs(H, active_cols):
    """Pack active site-local coordinates into a square periodic problem."""
    H = np.asarray(H)
    active_cols = np.asarray(active_cols, dtype=bool)
    if active_cols.shape != (H.shape[0], H.shape[1]):
        raise ValueError("active_cols must have shape (period, m)")

    period, m, _ = H.shape
    orders = np.empty((period, m), dtype=np.int64)
    for l in range(period):
        active = np.flatnonzero(active_cols[l])
        inactive = np.flatnonzero(~active_cols[l])
        orders[l] = np.concatenate([active, inactive])

    k = int(np.max(np.sum(active_cols, axis=1), initial=0))
    H_compact = np.zeros((k, k, period), dtype=H.dtype, order="F")
    for l in range(period):
        src = orders[l, :k]
        dst = orders[(l + 1) % period, :k]
        active_src = active_cols[l, src]
        active_dst = active_cols[(l + 1) % period, dst]
        H_compact[:, :, l] = H[l][np.ix_(dst, src)]*active_dst[:, None]*active_src[None, :]
    return H_compact, orders, k


def _pad_periodic_schur_outputs(T_compact, Z_compact, eigvals_compact, orders, m, dtype):
    """Embed compact periodic Schur data back into fixed-width coordinates."""
    period = orders.shape[0]
    k = T_compact.shape[1]
    T = np.zeros((period, m, m), dtype=dtype)
    Z = np.zeros((period, m, m), dtype=dtype)
    eigvals = np.zeros((m,), dtype=np.complex128)
    T[:, :k, :k] = T_compact
    eigvals[:k] = eigvals_compact
    for l in range(period):
        compact_rows = orders[l, :k]
        tail_rows = orders[l, k:]
        Z[l][np.ix_(compact_rows, np.arange(k))] = Z_compact[l]
        Z[l][tail_rows, np.arange(k, m)] = 1
    return T, Z, eigvals


def periodic_schur_callback(H):
    """Compute a real or complex periodic Schur decomposition through Julia."""
    H = np.asarray(H)
    with _JULIA_CALLBACK_LOCK:
        jl = _julia_periodic_schur()
        T, Z, eigvals = jl._pks_pschur(_period_axis_to_julia(H))
    return _period_axis_from_julia(T), _period_axis_from_julia(Z), np.asarray(eigvals)


def periodic_schur_compressed_callback(H, active_cols):
    """Compute Julia periodic Schur data after active-column compaction."""
    H = np.asarray(H)
    H_compact, orders, k = _compact_periodic_schur_inputs(H, active_cols)
    if k == 0:
        T_compact = np.zeros((H.shape[0], 0, 0), dtype=H.dtype)
        Z_compact = np.zeros((H.shape[0], 0, 0), dtype=H.dtype)
        eigvals_compact = np.zeros((0,), dtype=np.complex128)
    else:
        with _JULIA_CALLBACK_LOCK:
            jl = _julia_periodic_schur()
            T_compact, Z_compact, eigvals_compact = jl._pks_pschur(H_compact)
        T_compact = _period_axis_from_julia(T_compact)
        Z_compact = _period_axis_from_julia(Z_compact)
        eigvals_compact = np.asarray(eigvals_compact)
    return _pad_periodic_schur_outputs(
        T_compact,
        Z_compact,
        eigvals_compact,
        orders,
        H.shape[1],
        H.dtype,
    )


def reorder_periodic_schur_callback(T, Z, eigvals, select_mask):
    """Reorder an existing real or complex periodic Schur form through Julia."""
    T = np.asarray(T)
    Z = np.asarray(Z)
    with _JULIA_CALLBACK_LOCK:
        jl = _julia_periodic_schur()
        T_ord, Z_ord, eigvals_ord = jl._pks_ordschur(
            _period_axis_to_julia(T),
            _period_axis_to_julia(Z),
            np.asarray(eigvals),
            np.asarray(select_mask),
        )
    return (
        _period_axis_from_julia(T_ord),
        _period_axis_from_julia(Z_ord),
        np.asarray(eigvals_ord),
    )


def print_unexpected_periodic_schur_subdiagonal(T):
    """Print strict-factor subdiagonal leakage before Julia reordering."""
    T = np.asarray(T)
    period = T.shape[0]
    if period <= 1:
        return
    lower = np.tril(np.ones(T.shape[-2:], dtype=bool), k=-1)
    strict = T[:period - 1]
    values = np.abs(strict[:, lower])
    if not np.any(values > 0):
        return
    flat = int(np.argmax(values))
    factor = flat // np.count_nonzero(lower)
    lower_index = flat % np.count_nonzero(lower)
    rows, cols = np.nonzero(lower)
    i = int(rows[lower_index])
    j = int(cols[lower_index])
    path = Path("/private/tmp/ctmrg_periodic_context/unreordered_T_before_ordschur.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, T)
    print(
        "[periodic_ks] unexpected strict Schur subdiagonal before Julia ordschur: "
        f"factor={factor} i={i} j={j} value={T[factor, i, j]!r}; saved {path}",
        flush=True,
    )


def clean_periodic_schur_subdiagonals(T, eps_factor=100.0):
    """Zero roundoff-scale lower-triangular leakage before Julia reordering."""
    period = T.shape[0]
    scale = jnp.abs(T[:, 0, 0])[:, None, None]
    cutoff = eps_factor*jnp.finfo(T.dtype).eps*scale
    lower = jnp.tril(jnp.ones(T.shape[-2:], dtype=jnp.bool_), k=-1)
    strict_factor = (jnp.arange(period) < period - 1)[:, None, None]
    small_lower = strict_factor & lower[None, :, :] & (jnp.abs(T) < cutoff)
    return jnp.where(small_lower, jnp.zeros((), dtype=T.dtype), T)
