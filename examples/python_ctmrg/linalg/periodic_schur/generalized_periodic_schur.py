"""Generalized periodic Schur reduction for signed periodic products.

This module owns the Python orchestration for signed Hessenberg preprocessing
and the SLICOT ``MB03BD/MB03BZ`` drivers. The square signed Givens chase itself
is implemented by ``_periodic_schur.givens_chase`` in Cython.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeneralizedHessenbergResult:
    """Container returned by generalized periodic Hessenberg reduction."""

    factors: np.ndarray
    bases: list[np.ndarray]
    signs: np.ndarray
    trim_ranks: np.ndarray
    trim_converged: bool
    trim_sweeps: int


@dataclass
class PeriodicGeneralizedSchurResult:
    """Container returned by the generalized periodic Schur driver."""

    factors: np.ndarray
    bases: list[np.ndarray]
    eigvals: np.ndarray
    signs: np.ndarray
    hessenberg: GeneralizedHessenbergResult
    alpha: np.ndarray
    beta: np.ndarray
    scale: np.ndarray
    iwarn: int = 0


def _as_factor_list(factors):
    """Copy input factors into a mutable complex- or real-valued list."""
    dtype = np.result_type(*[np.asarray(a).dtype for a in factors])
    return [np.array(a, dtype=dtype, copy=True) for a in factors]


def _factor_rank(diag, scale, rank_tol):
    """Return the legacy retained leading rank from triangular diagonal data.

    Known bug: this stops at the first rejected diagonal. It is only valid when
    the live triangular sector is known to be leading and contiguous, not when
    null Arnoldi columns are interspersed with later live columns.
    """
    diag_abs = np.abs(np.asarray(diag))
    if diag_abs.size == 0:
        return 0
    if not rank_tol:
        return diag_abs.size
    eps = np.finfo(diag_abs.dtype).eps
    scale = np.asarray(scale, dtype=diag_abs.dtype)
    cutoff = rank_tol * eps * np.maximum(scale, 1.0)
    keep = diag_abs > cutoff
    dropped = np.flatnonzero(~keep)
    return int(dropped[0]) if dropped.size else diag.size


def _economic_qr(A, rank_tol):
    """Return ``A ~= Q @ R`` with optional leading-rank truncation."""
    import scipy.linalg

    Q, R = scipy.linalg.qr(A, mode="economic")
    diag = np.diag(R)
    scale = np.linalg.norm(A, axis=0)[: diag.size]
    rank = _factor_rank(diag, scale, rank_tol)
    return Q[:, :rank], R[:rank, :], rank


def _rq_diag(R, q_rows):
    """Return the active RQ diagonal used for rank truncation."""
    core = R[-q_rows:, :] if R.shape[0] >= q_rows else R[:, -q_rows:]
    return np.diag(core)


def _economic_rq(A, rank_tol):
    """Return ``A ~= R @ Q`` with optional trailing-rank truncation."""
    import scipy.linalg

    R, Q = scipy.linalg.rq(A, mode="economic")
    diag = _rq_diag(R, Q.shape[0])
    scale = np.linalg.norm(A, axis=1)
    rank = _factor_rank(diag, scale[-diag.size :], rank_tol)
    if rank == 0:
        return R[:, :0], Q[:0, :], rank
    if rank == Q.shape[0]:
        return R, Q, rank
    return R[:, -rank:], Q[-rank:, :], rank


def _right_apply_positive(factor, Q, sign):
    """Push a positive-factor QR basis into the previous signed factor."""
    return factor @ Q if sign else Q.conj().T @ factor


def _right_apply_negative(factor, Q, sign):
    """Push a negative-factor RQ basis into the previous signed factor."""
    return factor @ Q.conj().T if sign else Q @ factor


def _basis_after_positive(basis, Q):
    """Update a node basis after positive-factor QR compression."""
    return basis @ Q


def _basis_after_negative(basis, Q):
    """Update a node basis after negative-factor RQ compression."""
    return basis @ Q.conj().T


def _signed_shapes_are_square(factors):
    """Return whether all reduced factors are square with one common size."""
    shapes = [a.shape for a in factors]
    if not all(m == n for m, n in shapes):
        return False
    return len({m for m, _ in shapes}) == 1


def _basis_dim(factor, sign):
    """Return the node dimension acted on by a factor's local basis."""
    return factor.shape[0] if sign else factor.shape[1]


def signed_cyclic_reduce(factors, signs, rank_tol=1e-8, max_sweeps=None):
    """Reject the signed cyclic reducer until rank propagation is corrected."""
    del factors, signs, rank_tol, max_sweeps
    raise NotImplementedError(
        "signed_cyclic_reduce is disabled: its leading-rank inference discards "
        "live coordinates after interspersed null columns"
    )


def _square_givens_chase(factors, signs, bases):
    """Run the Cython signed Givens chase and compose its square bases."""
    from ._periodic_schur import givens_chase

    factors_cy, chase_bases = givens_chase(factors, signs)
    return factors_cy, [
        bases[l] @ chase_bases[l] for l in range(len(bases))
    ]


def generalized_phessenberg(factors, signs, rank_tol=1e-8):
    """Reduce a signed periodic product to Hessenberg/triangular form.

    Inputs may be rectangular. They are first trimmed by
    :func:`signed_cyclic_reduce`; the resulting live square product is then
    reduced so factor ``0`` is upper Hessenberg and all later factors are upper
    triangular. ``signs[0]`` must be positive for the generalized SLICOT
    convention.
    """
    signs = np.asarray(signs).copy()
    trim_factors, trim_bases, trim_ranks, trim_converged, trim_sweeps = (
        signed_cyclic_reduce(
            factors,
            signs,
            rank_tol=rank_tol,
        )
    )
    reduced, bases = _square_givens_chase(trim_factors, signs, trim_bases)
    return GeneralizedHessenbergResult(
        reduced,
        bases,
        signs.copy(),
        trim_ranks,
        trim_converged,
        trim_sweeps,
    )


def _import_slicot_periodic():
    """Import the local f2py SLICOT periodic extension."""
    from . import _slicot_periodic

    return _slicot_periodic


def _slicot_signs(signs):
    """Return strict SLICOT integer signs from the boolean convention."""
    return np.where(signs, 1, -1).astype(np.intc)


def _pack_slicot_factors(factors, dtype):
    """Pack leading-period factors as Fortran ``(n, n, period)`` arrays."""
    return np.asfortranarray(
        np.moveaxis(factors, 0, -1).astype(dtype, copy=False)
    )


def _unpack_slicot_factors(factors):
    """Unpack Fortran ``(n, n, period)`` factors to leading-period order."""
    return np.moveaxis(np.asarray(factors), -1, 0).copy()


def _identity_slicot_stack(n, period, dtype):
    """Return an identity stack in SLICOT's trailing-period layout."""
    eye = np.eye(n, dtype=dtype)
    return np.broadcast_to(eye[:, :, None], (n, n, period)).copy(order="F")


def _scaled_eigvals(alpha, beta, scale):
    """Decode SLICOT scaled generalized eigenvalues."""
    return (alpha / beta) * np.exp2(scale)


def _mb03bz_schur(factors, signs):
    """Run complex SLICOT ``MB03BZ`` on Hessenberg-triangular factors."""
    mb03 = _import_slicot_periodic()
    period, n, _ = factors.shape
    a = _pack_slicot_factors(factors, np.complex128)
    q = _identity_slicot_stack(n, period, np.complex128)
    alpha = np.empty(n, dtype=np.complex128)
    beta = np.empty(n, dtype=np.complex128)
    scale = np.empty(n, dtype=np.intc)
    dwork = np.empty(max(1, n), dtype=np.float64)
    zwork = np.empty(max(1, n), dtype=np.complex128)

    a, q, alpha, beta, scale, _, _, info = mb03.mb03bz(
        "S",
        "I",
        1,
        n,
        _slicot_signs(signs),
        a,
        q,
        alpha,
        beta,
        scale,
        dwork,
        zwork,
    )
    if info != 0:
        raise np.linalg.LinAlgError(f"MB03BZ failed with info={info}")
    return (
        _unpack_slicot_factors(a),
        _unpack_slicot_factors(q),
        _scaled_eigvals(alpha, beta, scale),
        alpha,
        beta,
        scale,
        0,
    )


def _mb03bd_schur(factors, signs):
    """Run real SLICOT ``MB03BD`` on Hessenberg-triangular factors."""
    mb03 = _import_slicot_periodic()
    period, n, _ = factors.shape
    a = _pack_slicot_factors(factors, np.float64)
    q = _identity_slicot_stack(n, period, np.float64)
    qind = np.arange(1, period + 1, dtype=np.intc)
    alphar = np.empty(n, dtype=np.float64)
    alphai = np.empty(n, dtype=np.float64)
    beta = np.empty(n, dtype=np.float64)
    scale = np.empty(n, dtype=np.intc)
    iwork = np.empty(2 * period + n, dtype=np.intc)
    dwork = np.empty(period + max(2 * n, 8 * period), dtype=np.float64)

    _, a, q, alphar, alphai, beta, scale, _, _, iwarn, info = mb03.mb03bd(
        "S",
        "C",
        "I",
        qind,
        1,
        1,
        n,
        _slicot_signs(signs),
        a,
        q,
        alphar,
        alphai,
        beta,
        scale,
        iwork,
        dwork,
    )
    if info != 0:
        raise np.linalg.LinAlgError(f"MB03BD failed with info={info}")
    alpha = alphar + 1j * alphai
    return (
        _unpack_slicot_factors(a),
        _unpack_slicot_factors(q),
        _scaled_eigvals(alpha, beta, scale),
        alpha,
        beta,
        scale,
        int(iwarn),
    )


def _slicot_generalized_schur(factors, signs):
    """Run the real or complex generalized periodic Schur SLICOT driver."""
    if np.iscomplexobj(factors):
        return _mb03bz_schur(factors, signs)
    return _mb03bd_schur(factors, signs)


def periodic_generalized_schur(factors, signs, rank_tol=1e-8):
    """Reduce a signed periodic product to generalized periodic Schur form.

    The represented product is ``A[0]^S[0] A[1]^S[1] ... A[p-1]^S[p-1]`` with
    strict signs ``True`` for ``A[l]`` and ``False`` for ``A[l]^{-1}``. Factor
    ``0`` is the SLICOT Hessenberg factor, so ``signs[0]`` follows the SLICOT
    convention and is positive.

    The returned factors ``T`` and bases ``Z`` satisfy, for positive factors,
    ``A[l] @ Z[l+1] = Z[l] @ T[l]``; for negative factors,
    ``A[l] @ Z[l] = Z[l+1] @ T[l]``. Indices are modulo the period.
    """
    hessenberg = generalized_phessenberg(
        factors, signs, rank_tol=rank_tol
    )
    (
        schur_factors,
        schur_bases,
        eigvals,
        alpha,
        beta,
        scale,
        iwarn,
    ) = _slicot_generalized_schur(
        hessenberg.factors,
        hessenberg.signs,
    )
    bases = [
        hessenberg.bases[l] @ schur_bases[l]
        for l in range(len(hessenberg.bases))
    ]
    return PeriodicGeneralizedSchurResult(
        schur_factors,
        bases,
        eigvals,
        hessenberg.signs.copy(),
        hessenberg,
        alpha,
        beta,
        scale,
        iwarn,
    )
