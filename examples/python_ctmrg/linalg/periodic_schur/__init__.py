"""Public periodic Schur wrappers and host-side callback helpers.

Static-shape JAX entry points live in ``linalg.periodic_schur.jax_ffi`` so ordinary
NumPy imports do not initialize JAX.
"""

from __future__ import annotations

import numpy as np

from .generalized_periodic_schur import (
    GeneralizedHessenbergResult,
    PeriodicGeneralizedSchurResult,
    generalized_phessenberg,
    periodic_generalized_schur,
    signed_cyclic_reduce,
)


def periodic_schur_D(
    H,
    active_cols=None,
    reduction="NRed",
    rank_tol=None,
    schur_deflation_tol=10.0,
):
    """Return the real form with optional QRP and Schur deflation controls."""
    from ._periodic_schur import periodic_schur_D as driver

    return driver(
        H,
        active_cols,
        reduction=reduction,
        rank_tol=rank_tol,
        schur_deflation_tol=schur_deflation_tol,
    )


def periodic_schur_Z(
    H,
    active_cols=None,
    reduction="NRed",
    rank_tol=None,
):
    """Return the complex form, optionally applying QRP deflation in CRed."""
    from ._periodic_schur import periodic_schur_Z as driver

    return driver(
        H,
        active_cols,
        reduction=reduction,
        rank_tol=rank_tol,
    )


def periodic_schur_eigenvalues(T, rank=None):
    """Return ordered eigenvalues from real or complex periodic Schur factors."""
    from ._periodic_schur import periodic_schur_eigenvalues as driver

    return driver(T, rank=rank)


def _periodic_schur_cred_qrp_D_callback(
    H,
    active_cols,
    rank_tol,
    schur_deflation_tol,
):
    """Return static-padded real QRP-CRed outputs for ``jax.pure_callback``."""
    H = np.asarray(H)
    period, m, _ = H.shape
    T, Z, wr, wi = periodic_schur_D(
        H,
        np.asarray(active_cols),
        reduction="CRed",
        rank_tol=float(np.asarray(rank_tol)),
        schur_deflation_tol=float(np.asarray(schur_deflation_tol)),
    )
    n = T.shape[1]
    T_pad = np.zeros((period, m, m), dtype=np.float64)
    Z_pad = np.zeros((period, m, m), dtype=np.float64)
    wr_pad = np.zeros(m, dtype=np.float64)
    wi_pad = np.zeros(m, dtype=np.float64)
    T_pad[:, :n, :n] = T
    Z_pad[:, :, :n] = Z
    wr_pad[:n] = wr
    wi_pad[:n] = wi
    return T_pad, Z_pad, wr_pad, wi_pad, np.asarray(n, dtype=np.int32)


def _periodic_schur_cred_qrp_Z_callback(H, active_cols, rank_tol):
    """Return static-padded complex QRP-CRed outputs for ``jax.pure_callback``."""
    H = np.asarray(H)
    period, m, _ = H.shape
    T, Z, alpha, beta, scale = periodic_schur_Z(
        H,
        np.asarray(active_cols),
        reduction="CRed",
        rank_tol=float(np.asarray(rank_tol)),
    )
    n = T.shape[1]
    T_pad = np.zeros((period, m, m), dtype=np.complex128)
    Z_pad = np.zeros((period, m, m), dtype=np.complex128)
    alpha_pad = np.zeros(m, dtype=np.complex128)
    beta_pad = np.ones(m, dtype=np.complex128)
    scale_pad = np.zeros(m, dtype=np.int32)
    T_pad[:, :n, :n] = T
    Z_pad[:, :, :n] = Z
    alpha_pad[:n] = alpha
    beta_pad[:n] = beta
    scale_pad[:n] = scale
    return (
        T_pad,
        Z_pad,
        alpha_pad,
        beta_pad,
        scale_pad,
        np.asarray(n, dtype=np.int32),
    )


def reorder_periodic_schur_D(T, Z, select, tol=100.0):
    """Return a real periodic Schur form with selected blocks leading."""
    from ._periodic_schur import reorder_periodic_schur_D as driver

    return driver(T, Z, select, tol=tol)


def reorder_periodic_schur_Z(T, Z, select, tol=100.0):
    """Return a complex periodic Schur form with selected entries leading."""
    from ._periodic_schur import reorder_periodic_schur_Z as driver

    return driver(T, Z, select, tol=tol)


def _lapack_reorder_schur_callback(T, eigvals, select_mask):
    """Host-side p=1 Schur reorder using LAPACK ``trsen``."""
    import scipy.linalg.lapack

    T = np.asarray(T)
    T0 = np.array(T[0], order="F", copy=True)
    U0 = np.eye(T0.shape[0], dtype=T0.dtype, order="F")
    select_mask = np.asarray(select_mask, dtype=np.int32)
    trsen = scipy.linalg.lapack.get_lapack_funcs("trsen", (T0,))
    result = trsen(
        select_mask,
        T0,
        U0,
        job="N",
        wantq=1,
        overwrite_t=1,
        overwrite_q=1,
    )
    if result[-1] != 0:
        raise np.linalg.LinAlgError(f"trsen failed with info={result[-1]}")
    T_ord, U_ord = result[0], result[1]
    selected = select_mask.astype(bool)
    eigvals_ord = np.concatenate([eigvals[selected], eigvals[~selected]])
    return T_ord, U_ord, eigvals_ord


def _normalize_columns_np(X):
    """Normalize dense host columns, leaving numerically zero columns unchanged."""
    norms = np.linalg.norm(X, axis=0)
    return X / np.where(norms == 0, 1.0, norms)[None, :]


def _match_eigenvector_order(values, vectors, target_values):
    """Greedily order host eigenvectors to match a target eigenvalue list."""
    values = np.asarray(values)
    target_values = np.asarray(target_values)
    remaining = list(range(values.shape[0]))
    order = []
    for target in target_values:
        distances = np.abs(values[remaining] - target)
        picked = remaining.pop(int(np.argmin(distances)))
        order.append(picked)
    return vectors[:, order]


def _diagonalize_periodic_schur_callback(T, Z, eigvals, schur_size):
    """Diagonalize the live prefix of a static R-oriented Schur carrier."""
    import scipy.linalg

    T = np.asarray(T)
    Z = np.asarray(Z)
    eigvals = np.asarray(eigvals)
    n = int(np.asarray(schur_size))
    period = T.shape[0]
    X_pad = np.zeros(
        Z.shape,
        dtype=np.result_type(Z, np.complex128),
    )
    if n == 0:
        return X_pad

    T = T[:, :n, :n]
    Z = Z[:, :, :n]
    product = np.eye(n, dtype=np.result_type(T, np.complex128))
    for l in range(period):
        product = product @ np.asarray(T[l], dtype=product.dtype)

    values, Y = scipy.linalg.eig(product)
    Y = _match_eigenvector_order(values, Y, eigvals[:n])
    Y = _normalize_columns_np(Y)

    Ys = np.empty(
        (period, n, n), dtype=np.result_type(Y, Z, np.complex128)
    )
    Ys[0] = Y
    for l in range(period - 1, 0, -1):
        Y = np.asarray(T[l], dtype=Ys.dtype) @ Y
        Y = _normalize_columns_np(Y)
        Ys[l] = Y

    X = np.einsum("pnm,pmd->pnd", Z, Ys)
    for l in range(period):
        X[l] = _normalize_columns_np(X[l])
    X_pad[:, :, :n] = X
    return X_pad
