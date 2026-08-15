# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

r"""Periodic Schur decomposition and reordering pipeline.

Public decomposition contract
=============================

The production entry points are

``periodic_schur_D(H, active_cols=None, reduction="NRed"|"CRed",
rank_tol=None, schur_deflation_tol=10.0)``
    Real ``float64`` periodic Schur decomposition.

``periodic_schur_Z(H, active_cols=None, reduction="NRed"|"CRed",
rank_tol=None)``
    Complex ``complex128`` periodic Schur decomposition.

These are top-level orchestrators, not merely wrappers around one constituent
stage.  For input ``H`` with shape ``(period, m, m)``, each call runs the
complete pipeline

``active compaction -> periodic Hessenberg reduction -> periodic Schur -> lift``

and returns final site-ordered factors and basis maps satisfying

``H[k] @ Z[(k + 1) % period] = Z[k] @ T[k]``.

The real return value is ``(T, Z, wr, wi)``.  The complex return value is
``(T, Z, alpha, beta, scale)``; the eigenvalues retain SLICOT's scaled
representation ``alpha / beta * 2**scale`` to avoid overflow.

Pipeline stages
===============

1. Active-coordinate compaction

   ``active_cols[k]`` selects known live input coordinates at cyclic cut
   ``k``.  If ``E[k]`` is the corresponding coordinate inclusion, the compact
   factor is

   ``C[k] = E[k]^H @ H[k] @ E[k+1]``.

   ``active_cols=None`` means that every coordinate is active. A scalar
   ``active_cols=r`` selects the prefix ``[:r]`` at every cut.
   ``reduction="NRed"`` keeps

   ``n = max_k number_of_active_columns[k]``

   and zero-pads smaller compact factors to ``(n, n)``.
   ``reduction="CRed"`` instead rotates a minimum-rank cut to zero and uses a
   reverse thin-QR sweep to obtain a square cycle with

   ``n = min_k number_of_active_columns[k]``.

   When ``rank_tol`` is supplied, CRed then applies pivoted QR to every square
   factor and numerically contracts the smallest revealed range. A pivot is
   retained when

   ``abs(R[i, i]) > rank_tol * eps * incoming_column_norm[piv[i]]``.

   The contraction and reverse QR sweep repeat until no factor reveals a
   smaller rank. This option is eig-only and is not used by NRed/Sylvester.

2. Periodic Hessenberg reduction

   CRed restores its square cycle and folded bases to the incoming period order
   before Hessenberg reduction. The square cycle is then transformed so factor
   ``0`` is upper Hessenberg and factors ``1:`` are upper triangular. The real
   route uses SLICOT ``MB03VD/MB03VY``. The complex route uses the native
   Householder implementation in this file. Sitewise transformations are
   accumulated in an internal square basis ``q``.

3. Periodic Schur reduction

   The real route calls SLICOT ``MB03WD`` followed by ``MB03WX``; the latter
   refreshes the eigenvalues from the final Schur blocks. The complex route
   calls SLICOT ``MB03BZ``. They transform the periodic-Hessenberg cycle into
   final Schur factors ``T[k]`` while updating ``q[k]``. In the real case
   ``T[0]`` is quasi-upper-triangular and the remaining factors are upper
   triangular. In the complex case every ``T[k]`` is upper triangular.

4. Lift to the incoming coordinates

   The active-coordinate inclusions and Hessenberg/Schur transformations are
   composed into the only basis returned to callers:

   ``Z[k] = basis[k] @ q[k]``.

   For NRed, ``basis`` is the padded active-coordinate inclusion. For CRed it
   also contains the thin QR maps. Thus callers receive final ``T`` and ``Z``
   directly; they do not need to call the compaction, Hessenberg, or Schur
   stage helpers themselves.

Reordering is a separate top-level operation
============================================

``periodic_schur_D`` and ``periodic_schur_Z`` do not impose a requested
eigenvalue order.  After decomposition, callers may use

``reorder_periodic_schur_D(T, Z, select)``
    Real periodic reordering through SLICOT ``MB03KD``.  A real 2-by-2 Schur
    block is selected atomically.

``reorder_periodic_schur_Z(T, Z, select, tol=100.0)``
    Complex periodic reordering through native adjacent 1-by-1 block swaps.

Both return new ``T, Z`` arrays, move the selected periodic eigenvalues to the
leading sector, and preserve the same public sitewise relation.  Reordering
does not rerun active compaction or the periodic Schur decomposition.

Internal and auxiliary entry points
===================================

The ``_compact_*``, ``_make_periodic_hessenberg_*``, and
``_periodic_hessenberg_to_schur_*`` functions expose constituent stages for
focused testing. Production callers should use ``periodic_schur_D/Z`` instead.
``compute_periodic_schur_active_D/Z`` and
``compute_periodic_schur_active_CRed_D/Z`` are the C-level adapters used by
the Python entry points, other Cython extensions, and the XLA FFI handlers.
All adapters call the same named buffer-level stages in this file.
``compute_reordered_periodic_schur_D/Z`` similarly provide the shared
buffer-level implementation used by the Python and XLA FFI reorder APIs.
``slicot_mb03ke_D`` is an auxiliary periodic Sylvester block solver; it is not
a stage of the Schur decomposition.
"""

import numpy as np
import scipy.linalg

cimport numpy as cnp
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
from libc.float cimport DBL_EPSILON, DBL_MIN
from libc.math cimport fabs, hypot
from libc.stdlib cimport calloc, free, malloc


ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t ZTYPE_t


cdef extern from "slicot_periodic_c_api.h":
    void mb03vd_(
        int* n, int* p, int* ilo, int* ihi,
        double* a, int* lda1, int* lda2,
        double* tau, int* ldtau, double* dwork, int* info,
    ) noexcept nogil
    void mb03vy_(
        int* n, int* p, int* ilo, int* ihi,
        double* a, int* lda1, int* lda2,
        double* tau, int* ldtau,
        double* dwork, int* ldwork, int* info,
    ) noexcept nogil
    void mb03wd_(
        char* job, char* compz,
        int* n, int* p, int* ilo, int* ihi, int* iloz, int* ihiz,
        double* h, int* ldh1, int* ldh2,
        double* z, int* ldz1, int* ldz2,
        double* wr, double* wi,
        double* dwork, int* ldwork, int* info,
    ) noexcept nogil
    void mb03wx_(
        int* n, int* p,
        double* t, int* ldt1, int* ldt2,
        double* wr, double* wi, int* info,
    ) noexcept nogil
    void mb03kd_(
        char* compq, int* whichq, char* strong,
        int* k, int* nc, int* kschur,
        int* n, int* ni, int* signs, int* select,
        double* t, int* ldt, int* ixt,
        double* q, int* ldq, int* ixq,
        int* m, double* tol,
        int* iwork, double* dwork, int* ldwork, int* info,
    ) noexcept nogil
    void mb03ke_(
        int* trana, int* tranb, int* isgn,
        int* k, int* m, int* n,
        double* prec, double* smin, int* signs,
        double* a, double* b, double* c, double* scale,
        double* dwork, int* ldwork, int* info,
    ) noexcept nogil
    void mb03bz_(
        char* job, char* compq,
        int* k, int* n, int* ilo, int* ihi, int* signs,
        double complex* a, int* lda1, int* lda2,
        double complex* q, int* ldq1, int* ldq2,
        double complex* alpha, double complex* beta, int* scale,
        double* dwork, int* ldwork,
        double complex* zwork, int* lzwork, int* info,
    ) noexcept nogil


def periodic_schur_D(
    H,
    active_cols=None,
    reduction="NRed",
    rank_tol=None,
    schur_deflation_tol=10.0,
):
    r"""Run active compaction through real periodic Schur in one call.

    ``H`` has leading-period shape ``(period, m, m)`` and is already ordered as
    the formal product ``H[0] H[1] ... H[period-1]``. ``active_cols[k]``
    selects the live Arnoldi coordinates at cut ``k``; ``None`` selects every
    coordinate, while a scalar integer ``r`` selects ``active_cols[:, :r]``.
    ``reduction='NRed'`` retains the zero-padded dimension ``max(ranks)``;
    ``'CRed'`` returns the minimum-rank eig-only cycle after a reverse thin-QR
    sweep starting at a minimum-rank cut. If ``rank_tol`` is supplied, pivoted
    QR repeatedly contracts numerical null directions before Hessenberg
    reduction, comparing each pivot against its incoming column norm.
    After Hessenberg reduction, ``schur_deflation_tol`` multiplies MB03WD's
    machine-precision test for negligible implicit-product subdiagonals.

    Returns ``(T, Z, wr, wi)`` in the incoming leading-period order, where
    ``H[k] @ Z[k+1] = Z[k] @ T[k]``. ``T`` has shape ``(period, n, n)`` and
    ``Z`` has shape ``(period, m, n)``. The real eigenvalues are
    ``wr + 1j*wi``. The default reductions call the shared buffer-level cores
    used by the Cython and XLA FFI adapters; numerical QRP CRed composes the
    same public stage helpers around its data-dependent rank contraction.
    """
    cdef tuple inputs = _compaction_inputs(H, active_cols, np.float64)
    cdef cnp.ndarray H_arr = inputs[0]
    cdef cnp.ndarray active_arr = inputs[1]
    cdef int period = H_arr.shape[0]
    cdef int m = H_arr.shape[1]
    cdef int n
    cdef int capacity
    cdef int info
    cdef cnp.ndarray T_arr
    cdef cnp.ndarray Z_arr
    cdef cnp.ndarray wr_arr
    cdef cnp.ndarray wi_arr
    cdef double deflation_value = float(schur_deflation_tol)
    cdef tuple compact
    cdef tuple prepared
    cdef tuple schur
    cdef tuple output
    if deflation_value < 1.0:
        raise ValueError("schur_deflation_tol must be at least 1")
    if reduction == "NRed":
        if rank_tol is not None:
            raise ValueError("rank_tol is available only for CRed")
        n = periodic_schur_active_size(
            <const unsigned char*>active_arr.data, period, m
        )
        T_arr = np.empty((period, n, n), dtype=np.float64)
        Z_arr = np.empty((period, m, n), dtype=np.float64)
        wr_arr = np.empty(n, dtype=np.float64)
        wi_arr = np.empty(n, dtype=np.float64)
        with nogil:
            info = compute_periodic_schur_active_D(
                <const double*>H_arr.data,
                <const unsigned char*>active_arr.data,
                deflation_value, period, m, n, n,
                <double*>T_arr.data,
                <double*>Z_arr.data,
                <double*>wr_arr.data,
                <double*>wi_arr.data,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"real periodic Schur failed with info={info}"
            )
        return T_arr, Z_arr, wr_arr, wi_arr
    elif reduction == "CRed":
        if rank_tol is not None:
            compact = _compact_active_slicot_d(H_arr, active_arr)
            prepared = _make_periodic_hessenberg_CRed_D(
                compact[0],
                compact[1],
                compact[2],
                rank_tol=rank_tol,
            )
            schur = _periodic_hessenberg_to_schur_D(
                prepared[0],
                prepared[2],
                schur_deflation_tol=deflation_value,
            )
            output = _compose_periodic_schur_output(
                schur[0],
                prepared[1],
                schur[1],
            )
            return output[0], output[1], schur[2], schur[3]
        capacity = periodic_schur_active_size(
            <const unsigned char*>active_arr.data, period, m
        )
        n = periodic_schur_active_min_size(
            <const unsigned char*>active_arr.data, period, m
        )
        T_arr = np.empty((period, n, n), dtype=np.float64)
        Z_arr = np.empty((period, m, n), dtype=np.float64)
        wr_arr = np.empty(n, dtype=np.float64)
        wi_arr = np.empty(n, dtype=np.float64)
        with nogil:
            info = compute_periodic_schur_active_CRed_D(
                <const double*>H_arr.data,
                <const unsigned char*>active_arr.data,
                deflation_value, period, m, capacity, n, n,
                <double*>T_arr.data,
                <double*>Z_arr.data,
                <double*>wr_arr.data,
                <double*>wi_arr.data,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"real periodic CRed Schur failed with info={info}"
            )
        return T_arr, Z_arr, wr_arr, wi_arr
    else:
        raise ValueError("reduction must be 'NRed' or 'CRed'")


def periodic_schur_Z(
    H,
    active_cols=None,
    reduction="NRed",
    rank_tol=None,
):
    r"""Run active compaction through complex periodic Schur in one call.

    Input ordering, reduction selection, output shapes, and the sitewise Schur
    relation match :func:`periodic_schur_D`. Returns
    ``(T, Z, alpha, beta, scale)`` with the native scaled eigenvalue
    representation from all-positive SLICOT ``MB03BZ``; it deliberately does
    not form ``alpha / beta * 2**scale``.
    """
    cdef tuple inputs = _compaction_inputs(H, active_cols, np.complex128)
    cdef cnp.ndarray H_arr = inputs[0]
    cdef cnp.ndarray active_arr = inputs[1]
    cdef int period = H_arr.shape[0]
    cdef int m = H_arr.shape[1]
    cdef int n
    cdef int capacity
    cdef int info
    cdef cnp.ndarray T_arr
    cdef cnp.ndarray Z_arr
    cdef cnp.ndarray alpha_arr
    cdef cnp.ndarray beta_arr
    cdef cnp.ndarray scale_arr
    cdef tuple compact
    cdef tuple prepared
    cdef tuple schur
    cdef tuple output
    if reduction == "NRed":
        if rank_tol is not None:
            raise ValueError("rank_tol is available only for CRed")
        n = periodic_schur_active_size(
            <const unsigned char*>active_arr.data, period, m
        )
        T_arr = np.empty((period, n, n), dtype=np.complex128)
        Z_arr = np.empty((period, m, n), dtype=np.complex128)
        alpha_arr = np.empty(n, dtype=np.complex128)
        beta_arr = np.empty(n, dtype=np.complex128)
        scale_arr = np.empty(n, dtype=np.int32)
        with nogil:
            info = compute_periodic_schur_active_Z(
                <const void*>H_arr.data,
                <const unsigned char*>active_arr.data,
                period, m, n, n,
                <void*>T_arr.data,
                <void*>Z_arr.data,
                <void*>alpha_arr.data,
                <void*>beta_arr.data,
                <int*>scale_arr.data,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"complex periodic Schur failed with info={info}"
            )
        return T_arr, Z_arr, alpha_arr, beta_arr, scale_arr
    elif reduction == "CRed":
        if rank_tol is not None:
            compact = _compact_active_slicot_z(H_arr, active_arr)
            prepared = _make_periodic_hessenberg_CRed_Z(
                compact[0],
                compact[1],
                compact[2],
                rank_tol=rank_tol,
            )
            schur = _periodic_hessenberg_to_schur_Z(
                prepared[0],
                prepared[2],
            )
            output = _compose_periodic_schur_output(
                schur[0],
                prepared[1],
                schur[1],
            )
            return output[0], output[1], schur[2], schur[3], schur[4]
        capacity = periodic_schur_active_size(
            <const unsigned char*>active_arr.data, period, m
        )
        n = periodic_schur_active_min_size(
            <const unsigned char*>active_arr.data, period, m
        )
        T_arr = np.empty((period, n, n), dtype=np.complex128)
        Z_arr = np.empty((period, m, n), dtype=np.complex128)
        alpha_arr = np.empty(n, dtype=np.complex128)
        beta_arr = np.empty(n, dtype=np.complex128)
        scale_arr = np.empty(n, dtype=np.int32)
        with nogil:
            info = compute_periodic_schur_active_CRed_Z(
                <const void*>H_arr.data,
                <const unsigned char*>active_arr.data,
                period, m, capacity, n, n,
                <void*>T_arr.data,
                <void*>Z_arr.data,
                <void*>alpha_arr.data,
                <void*>beta_arr.data,
                <int*>scale_arr.data,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"complex periodic CRed Schur failed with info={info}"
            )
        return T_arr, Z_arr, alpha_arr, beta_arr, scale_arr
    else:
        raise ValueError("reduction must be 'NRed' or 'CRed'")


def periodic_schur_eigenvalues(T, rank=None):
    r"""Return the ordered eigenvalues of real or complex periodic Schur factors.

    ``T`` has shape ``(period, n, n)`` in the public leading-period order.
    Real ``float64`` factors are read with SLICOT ``MB03WX`` so a leading
    quasi-triangular factor may contain real 2-by-2 blocks. Complex
    ``complex128`` factors are upper triangular, so their eigenvalues are the
    products of corresponding diagonal entries.

    If ``rank`` is supplied, only ``T[:, :rank, :rank]`` participates and the
    returned ``complex128`` array has length ``rank``.
    """
    cdef cnp.ndarray T_arr = np.ascontiguousarray(T)
    cdef cnp.ndarray eigenvalues_arr
    cdef int period
    cdef int width
    cdef int n
    cdef int info

    if T_arr.ndim != 3 or T_arr.shape[1] != T_arr.shape[2]:
        raise ValueError("T must have shape (period, n, n)")
    if T_arr.shape[0] < 1:
        raise ValueError("T must contain at least one factor")
    if (T_arr.dtype != np.dtype(np.float64) and
            T_arr.dtype != np.dtype(np.complex128)):
        raise TypeError("T must have dtype float64 or complex128")

    period = T_arr.shape[0]
    width = T_arr.shape[1]
    n = width if rank is None else rank
    if n < 0 or n > width:
        raise ValueError("rank must satisfy 0 <= rank <= T.shape[1]")

    eigenvalues_arr = np.empty(n, dtype=np.complex128)
    if T_arr.dtype == np.dtype(np.float64):
        with nogil:
            info = compute_periodic_schur_eigenvalues_D(
                <const double*>T_arr.data,
                period,
                width,
                n,
                <void*>eigenvalues_arr.data,
            )
    else:
        with nogil:
            info = compute_periodic_schur_eigenvalues_Z(
                <const void*>T_arr.data,
                period,
                width,
                n,
                <void*>eigenvalues_arr.data,
            )
    if info != 0:
        raise np.linalg.LinAlgError(
            f"periodic Schur eigenvalue readout failed with info={info}"
        )
    return eigenvalues_arr


def reorder_periodic_schur_D(T, Z, select, tol=100.0):
    r"""Move selected real periodic eigenvalue blocks to the leading sector.

    ``T`` and ``Z`` use the public convention

    ``H[k] @ Z[k+1] = Z[k] @ T[k]``.

    ``T`` has shape ``(period, n, n)`` and ``Z`` has shape
    ``(period, m, n)``.  SLICOT ``MB03KD`` reorders the selected real or
    conjugate-pair blocks and returns new arrays satisfying the same relation.
    Selecting either member of a real-Schur 2-by-2 block selects the full
    conjugate pair.
    """
    cdef cnp.ndarray T_arr = np.ascontiguousarray(T)
    cdef cnp.ndarray Z_arr = np.ascontiguousarray(Z)
    cdef cnp.ndarray select_arr
    cdef cnp.ndarray T_out
    cdef cnp.ndarray Z_out
    cdef int period
    cdef int m
    cdef int n
    cdef int info
    cdef double tol_value = tol

    if T_arr.dtype != np.dtype(np.float64):
        raise TypeError("T must have dtype float64")
    if Z_arr.dtype != np.dtype(np.float64):
        raise TypeError("Z must have dtype float64")
    if T_arr.ndim != 3 or T_arr.shape[1] != T_arr.shape[2]:
        raise ValueError("T must have shape (period, n, n)")
    period = T_arr.shape[0]
    n = T_arr.shape[1]
    if period < 1:
        raise ValueError("T must contain at least one factor")
    if Z_arr.ndim != 3 or Z_arr.shape[0] != period or Z_arr.shape[2] != n:
        raise ValueError("Z must have shape (period, m, n)")
    m = Z_arr.shape[1]

    select_arr = np.ascontiguousarray(select, dtype=np.uint8)
    if select_arr.ndim != 1 or select_arr.shape[0] != n:
        raise ValueError("select must have shape (n,)")
    T_out = np.empty_like(T_arr)
    Z_out = np.empty_like(Z_arr)
    with nogil:
        info = compute_reordered_periodic_schur_D(
            <const double*>T_arr.data,
            <const double*>Z_arr.data,
            <const unsigned char*>select_arr.data,
            period,
            m,
            n,
            tol_value,
            <double*>T_out.data,
            <double*>Z_out.data,
        )
    if info != 0:
        raise np.linalg.LinAlgError(
            f"real periodic Schur reordering failed with info={info}"
        )
    return T_out, Z_out


def reorder_periodic_schur_Z(T, Z, select, tol=100.0):
    r"""Move selected complex periodic eigenvalues to the leading sector.

    ``T`` and ``Z`` use the public convention

    ``H[k] @ Z[k+1] = Z[k] @ T[k]``.

    Every diagonal block of the complex Schur form is 1-by-1. Selected
    entries are therefore collected by stable adjacent exchanges, each using
    a scalar periodic Sylvester solve and one complex Givens rotation per
    site.
    """
    cdef cnp.ndarray T_arr = np.ascontiguousarray(T)
    cdef cnp.ndarray Z_arr = np.ascontiguousarray(Z)
    cdef cnp.ndarray select_arr
    cdef cnp.ndarray T_out
    cdef cnp.ndarray Z_out
    cdef int period
    cdef int m
    cdef int n
    cdef int info
    cdef double tol_value = tol

    if T_arr.dtype != np.dtype(np.complex128):
        raise TypeError("T must have dtype complex128")
    if Z_arr.dtype != np.dtype(np.complex128):
        raise TypeError("Z must have dtype complex128")
    if T_arr.ndim != 3 or T_arr.shape[1] != T_arr.shape[2]:
        raise ValueError("T must have shape (period, n, n)")
    period = T_arr.shape[0]
    n = T_arr.shape[1]
    if period < 1:
        raise ValueError("T must contain at least one factor")
    if Z_arr.ndim != 3 or Z_arr.shape[0] != period or Z_arr.shape[2] != n:
        raise ValueError("Z must have shape (period, m, n)")
    m = Z_arr.shape[1]

    select_arr = np.ascontiguousarray(select, dtype=np.uint8)
    if select_arr.ndim != 1 or select_arr.shape[0] != n:
        raise ValueError("select must have shape (n,)")
    T_out = np.empty_like(T_arr)
    Z_out = np.empty_like(Z_arr)
    with nogil:
        info = compute_reordered_periodic_schur_Z(
            <const void*>T_arr.data,
            <const void*>Z_arr.data,
            <const unsigned char*>select_arr.data,
            period,
            m,
            n,
            tol_value,
            <void*>T_out.data,
            <void*>Z_out.data,
        )
    if info != 0:
        raise np.linalg.LinAlgError(
            f"complex periodic Schur reordering failed with info={info}"
        )
    return T_out, Z_out


cdef public api int periodic_schur_active_size(
    const unsigned char* active_cols,
    int period,
    int m,
) noexcept nogil:
    """Return the largest sitewise active-coordinate count."""
    cdef int k, i, count, n = 0
    for k in range(period):
        count = 0
        for i in range(m):
            count += 1 if active_cols[k*m + i] else 0
        if count > n:
            n = count
    return n


cdef public api int periodic_schur_active_min_size(
    const unsigned char* active_cols,
    int period,
    int m,
) noexcept nogil:
    """Return the smallest sitewise active-coordinate count."""
    cdef int k, i, count, n = m
    for k in range(period):
        count = 0
        for i in range(m):
            count += 1 if active_cols[k*m + i] else 0
        if count < n:
            n = count
    return n


cdef public api int compute_periodic_schur_eigenvalues_D(
    const double* T,
    int period,
    int input_width,
    int n,
    void* eigenvalues_buffer,
) noexcept nogil:
    """Read a C-order real periodic Schur prefix with SLICOT ``MB03WX``."""
    cdef double complex* eigenvalues = <double complex*>eigenvalues_buffer
    cdef double* factors
    cdef double* wr
    cdef double* wi
    cdef int k, i, j, info = 0

    if n < 0 or n > input_width:
        return -3
    if n == 0:
        return 0

    factors = <double*>malloc(period*n*n*sizeof(double))
    wr = <double*>malloc(n*sizeof(double))
    wi = <double*>malloc(n*sizeof(double))
    if factors == NULL or wr == NULL or wi == NULL:
        free(factors)
        free(wr)
        free(wi)
        return -1

    for k in range(period):
        for i in range(n):
            for j in range(n):
                factors[i + n*j + n*n*k] = (
                    T[(k*input_width + i)*input_width + j]
                )
    mb03wx_(&n, &period, factors, &n, &n, wr, wi, &info)
    if info == 0:
        for i in range(n):
            eigenvalues[i] = wr[i] + wi[i]*1j

    free(factors)
    free(wr)
    free(wi)
    return 3100 + info if info != 0 else 0


cdef public api int compute_periodic_schur_eigenvalues_Z(
    const void* T_buffer,
    int period,
    int input_width,
    int n,
    void* eigenvalues_buffer,
) noexcept nogil:
    """Multiply corresponding diagonals of complex triangular factors."""
    cdef const double complex* T = <const double complex*>T_buffer
    cdef double complex* eigenvalues = <double complex*>eigenvalues_buffer
    cdef int k, i

    if n < 0 or n > input_width:
        return -3
    for i in range(n):
        eigenvalues[i] = 1.0
        for k in range(period):
            eigenvalues[i] *= T[(k*input_width + i)*input_width + i]
    return 0


cdef api int slicot_mb03ke_D(
    int trana,
    int tranb,
    int isgn,
    int period,
    int m,
    int n,
    const int* signs,
    const double* A,
    const double* B,
    double* C,
    double* scale,
    double* work,
    int lwork,
) noexcept nogil:
    r"""Solve one small real periodic Sylvester-like system with MB03KE.

    ``A``, ``B``, and ``C`` use SLICOT's packed Fortran sequence storage.
    Machine precision and safe-minimum parameters are supplied from LAPACK.
    """
    cdef int trana_value = trana
    cdef int tranb_value = tranb
    cdef int isgn_value = isgn
    cdef int info = 0
    cdef char precision_code = 80
    cdef char safe_minimum_code = 83
    cdef double precision = lapack.dlamch(&precision_code)
    cdef double safe_minimum = lapack.dlamch(&safe_minimum_code)
    cdef double smin = safe_minimum / precision

    mb03ke_(
        &trana_value, &tranb_value, &isgn_value,
        &period, &m, &n, &precision, &smin, <int*>signs,
        <double*>A, <double*>B, C, scale, work, &lwork, &info,
    )
    return info


cdef void _square_periodic_hessenberg_Z_buffers(
    double complex* factors,
    double complex* q,
    int n,
    int period,
    double complex* work,
) noexcept nogil:
    """Reduce one square complex cycle in Fortran-order buffers."""
    cdef char side_l = 76
    cdef char side_r = 82
    cdef int i, k, row, tail_row
    cdef int reflector_len, trail_cols, one = 1
    cdef double complex tau, tau_h, beta

    for i in range(n - 1):
        trail_cols = n - i - 1
        for k in range(period - 1, 0, -1):
            reflector_len = n - i
            lapack.zlarfg(
                &reflector_len,
                &factors[i + n*i + n*n*k],
                &factors[i + 1 + n*i + n*n*k],
                &one,
                &tau,
            )
            beta = factors[i + n*i + n*n*k]
            factors[i + n*i + n*n*k] = 1.0
            tau_h = tau.conjugate()
            lapack.zlarf(
                &side_r, &n, &reflector_len,
                &factors[i + n*i + n*n*k], &one, &tau,
                &factors[n*i + n*n*(k - 1)], &n, work,
            )
            lapack.zlarf(
                &side_l, &reflector_len, &trail_cols,
                &factors[i + n*i + n*n*k], &one, &tau_h,
                &factors[i + n*(i + 1) + n*n*k], &n, work,
            )
            lapack.zlarf(
                &side_r, &n, &reflector_len,
                &factors[i + n*i + n*n*k], &one, &tau,
                &q[n*i + n*n*k], &n, work,
            )
            factors[i + n*i + n*n*k] = beta
            for row in range(i + 1, n):
                factors[row + n*i + n*n*k] = 0.0

        reflector_len = n - i - 1
        tail_row = i + 2 if i + 2 < n else i + 1
        lapack.zlarfg(
            &reflector_len,
            &factors[i + 1 + n*i],
            &factors[tail_row + n*i],
            &one,
            &tau,
        )
        beta = factors[i + 1 + n*i]
        factors[i + 1 + n*i] = 1.0
        tau_h = tau.conjugate()
        lapack.zlarf(
            &side_r, &n, &reflector_len,
            &factors[i + 1 + n*i], &one, &tau,
            &factors[n*(i + 1) + n*n*(period - 1)], &n, work,
        )
        lapack.zlarf(
            &side_r, &n, &reflector_len,
            &factors[i + 1 + n*i], &one, &tau,
            &q[n*(i + 1)], &n, work,
        )
        lapack.zlarf(
            &side_l, &reflector_len, &trail_cols,
            &factors[i + 1 + n*i], &one, &tau_h,
            &factors[i + 1 + n*(i + 1)], &n, work,
        )
        factors[i + 1 + n*i] = beta
        for row in range(i + 2, n):
            factors[row + n*i] = 0.0


cdef int _collect_active_metadata_buffers(
    const unsigned char* active_cols,
    int period,
    int m,
    int n,
    int* indices,
    int* ranks,
) noexcept nogil:
    """Collect active indices and ranks for the compaction stage."""
    cdef int k, i, count

    for k in range(period):
        count = 0
        for i in range(m):
            if active_cols[k*m + i]:
                indices[k*m + count] = i
                count += 1
        ranks[k] = count
        if count > n:
            return -2
    return 0


cdef void _compact_active_factors_D_buffers(
    const double* H,
    const int* indices,
    const int* ranks,
    int period,
    int m,
    int n,
    double* factors,
) noexcept nogil:
    """Pack active real factor rectangles into square Fortran storage."""
    cdef int k, kp, i, j, row, col

    for k in range(period):
        kp = (k + 1) % period
        for i in range(ranks[k]):
            row = indices[k*m + i]
            for j in range(ranks[kp]):
                col = indices[kp*m + j]
                factors[i + n*j + n*n*k] = H[(k*m + row)*m + col]


cdef void _compact_active_factors_Z_buffers(
    const double complex* H,
    const int* indices,
    const int* ranks,
    int period,
    int m,
    int n,
    double complex* factors,
) noexcept nogil:
    """Pack active complex factor rectangles into square Fortran storage."""
    cdef int k, kp, i, j, row, col

    for k in range(period):
        kp = (k + 1) % period
        for i in range(ranks[k]):
            row = indices[k*m + i]
            for j in range(ranks[kp]):
                col = indices[kp*m + j]
                factors[i + n*j + n*n*k] = H[(k*m + row)*m + col]


cdef void _compact_active_factors_rotated_D_buffers(
    const double* H,
    const int* indices,
    const int* ranks,
    int cut,
    int period,
    int m,
    int capacity,
    double* factors,
) noexcept nogil:
    """Pack real active rectangles after moving ``cut`` to cycle position zero."""
    cdef int k, kp, source, source_p, i, j, row, col

    for k in range(period):
        kp = (k + 1) % period
        source = (k + cut) % period
        source_p = (source + 1) % period
        for i in range(ranks[source]):
            row = indices[source*m + i]
            for j in range(ranks[source_p]):
                col = indices[source_p*m + j]
                factors[i + capacity*j + capacity*capacity*k] = (
                    H[(source*m + row)*m + col]
                )


cdef void _compact_active_factors_rotated_Z_buffers(
    const double complex* H,
    const int* indices,
    const int* ranks,
    int cut,
    int period,
    int m,
    int capacity,
    double complex* factors,
) noexcept nogil:
    """Pack complex active rectangles after moving ``cut`` to cycle position zero."""
    cdef int k, kp, source, source_p, i, j, row, col

    for k in range(period):
        kp = (k + 1) % period
        source = (k + cut) % period
        source_p = (source + 1) % period
        for i in range(ranks[source]):
            row = indices[source*m + i]
            for j in range(ranks[source_p]):
                col = indices[source_p*m + j]
                factors[i + capacity*j + capacity*capacity*k] = (
                    H[(source*m + row)*m + col]
                )


cdef int _cyclic_qr_reduce_D_buffers(
    double* factors,
    const int* ranks,
    int cut,
    double* bases,
    double* tau,
    double* work,
    int lwork,
    int period,
    int capacity,
    int n,
) noexcept nogil:
    """Reduce rotated real rectangles to an ``n``-dimensional square cycle."""
    cdef char side = 82
    cdef char trans = 78
    cdef int k, source, source_m, rows, prev_rows, i, j, info = 0
    cdef double* factor
    cdef double* predecessor
    cdef double* basis

    for i in range(n):
        bases[i + capacity*i] = 1.0

    for k in range(period - 1, 0, -1):
        source = (k + cut) % period
        source_m = (source - 1 + period) % period
        rows = ranks[source]
        prev_rows = ranks[source_m]
        factor = &factors[capacity*capacity*k]
        predecessor = &factors[capacity*capacity*(k - 1)]
        basis = &bases[capacity*n*k]

        lapack.dgeqrf(
            &rows, &n, factor, &capacity, tau, work, &lwork, &info,
        )
        if info != 0:
            return 5000 + info

        for j in range(n):
            for i in range(rows):
                basis[i + capacity*j] = factor[i + capacity*j]

        lapack.dormqr(
            &side, &trans, &prev_rows, &rows, &n,
            factor, &capacity, tau,
            predecessor, &capacity, work, &lwork, &info,
        )
        if info != 0:
            return 5100 + info

        lapack.dorgqr(
            &rows, &n, &n, basis, &capacity, tau, work, &lwork, &info,
        )
        if info != 0:
            return 5200 + info

        for j in range(n):
            for i in range(j + 1, rows):
                factor[i + capacity*j] = 0.0
    return 0


cdef int _cyclic_qr_reduce_Z_buffers(
    double complex* factors,
    const int* ranks,
    int cut,
    double complex* bases,
    double complex* tau,
    double complex* work,
    int lwork,
    int period,
    int capacity,
    int n,
) noexcept nogil:
    """Reduce rotated complex rectangles to an ``n``-dimensional square cycle."""
    cdef char side = 82
    cdef char trans = 78
    cdef int k, source, source_m, rows, prev_rows, i, j, info = 0
    cdef double complex* factor
    cdef double complex* predecessor
    cdef double complex* basis

    for i in range(n):
        bases[i + capacity*i] = 1.0

    for k in range(period - 1, 0, -1):
        source = (k + cut) % period
        source_m = (source - 1 + period) % period
        rows = ranks[source]
        prev_rows = ranks[source_m]
        factor = &factors[capacity*capacity*k]
        predecessor = &factors[capacity*capacity*(k - 1)]
        basis = &bases[capacity*n*k]

        lapack.zgeqrf(
            &rows, &n, factor, &capacity, tau, work, &lwork, &info,
        )
        if info != 0:
            return 5000 + info

        for j in range(n):
            for i in range(rows):
                basis[i + capacity*j] = factor[i + capacity*j]

        lapack.zunmqr(
            &side, &trans, &prev_rows, &rows, &n,
            factor, &capacity, tau,
            predecessor, &capacity, work, &lwork, &info,
        )
        if info != 0:
            return 5100 + info

        lapack.zungqr(
            &rows, &n, &n, basis, &capacity, tau, work, &lwork, &info,
        )
        if info != 0:
            return 5200 + info

        for j in range(n):
            for i in range(j + 1, rows):
                factor[i + capacity*j] = 0.0
    return 0


cdef void _restore_CRed_order_D_buffers(
    const double* rotated_factors,
    const double* rotated_bases,
    int cut,
    int period,
    int capacity,
    int n,
    double* factors,
    double* bases,
) noexcept nogil:
    """Restore real CRed factors and folded bases to physical period order."""
    cdef int k, source, i, j

    for k in range(period):
        source = (k + cut) % period
        for j in range(n):
            for i in range(n):
                factors[i + n*j + n*n*source] = (
                    rotated_factors[
                        i + capacity*j + capacity*capacity*k
                    ]
                )
            for i in range(capacity):
                bases[i + capacity*j + capacity*n*source] = (
                    rotated_bases[i + capacity*j + capacity*n*k]
                )


cdef void _restore_CRed_order_Z_buffers(
    const double complex* rotated_factors,
    const double complex* rotated_bases,
    int cut,
    int period,
    int capacity,
    int n,
    double complex* factors,
    double complex* bases,
) noexcept nogil:
    """Restore complex CRed factors and folded bases to physical period order."""
    cdef int k, source, i, j

    for k in range(period):
        source = (k + cut) % period
        for j in range(n):
            for i in range(n):
                factors[i + n*j + n*n*source] = (
                    rotated_factors[
                        i + capacity*j + capacity*capacity*k
                    ]
                )
            for i in range(capacity):
                bases[i + capacity*j + capacity*n*source] = (
                    rotated_bases[i + capacity*j + capacity*n*k]
                )


cdef int _make_periodic_hessenberg_NRed_D_buffers(
    double* factors,
    double* q,
    double* tau,
    double* work,
    int period,
    int n,
    int ldtau,
    int lwork,
) noexcept nogil:
    """Reduce real compact factors to periodic Hessenberg form and form Q."""
    cdef int i, j, k, info = 0
    cdef int ilo = 1
    cdef int ihi = n
    cdef int lda = n

    mb03vd_(
        &n, &period, &ilo, &ihi,
        factors, &lda, &lda, tau, &ldtau, work, &info,
    )
    if info != 0:
        return 1000 + info

    for i in range(n*n*period):
        q[i] = factors[i]
    mb03vy_(
        &n, &period, &ilo, &ihi,
        q, &lda, &lda, tau, &ldtau, work, &lwork, &info,
    )
    if info != 0:
        return 2000 + info

    for k in range(period):
        for j in range(n):
            for i in range(j + (2 if k == 0 else 1), n):
                factors[i + n*j + n*n*k] = 0.0
    return 0


cdef void _make_periodic_hessenberg_NRed_Z_buffers(
    double complex* factors,
    double complex* q,
    double complex* work,
    int period,
    int n,
) noexcept nogil:
    """Reduce complex compact factors to periodic Hessenberg form and form Q."""
    cdef int k, i

    for k in range(period):
        for i in range(n):
            q[i + n*i + n*n*k] = 1.0
    _square_periodic_hessenberg_Z_buffers(factors, q, n, period, work)


cdef int _deflate_small_periodic_hessenberg_subdiagonals_D_buffers(
    double* factors,
    double schur_deflation_tol,
    int period,
    int n,
    int factor_ld,
) noexcept nogil:
    """Zero negligible implicit-product subdiagonals before MB03WD."""
    cdef double* t_diag = <double*>malloc(n*sizeof(double))
    cdef double* t_super = <double*>calloc(
        n - 1 if n > 1 else 1,
        sizeof(double),
    )
    cdef double* product_diag = <double*>malloc(n*sizeof(double))
    cdef int factor_stride = factor_ld*factor_ld
    cdef int k, i, count = 0
    cdef double factor_scale = 0.0
    cdef double product_subdiag
    cdef double raw_threshold
    cdef double threshold
    cdef double smlnum = DBL_MIN*(n/DBL_EPSILON)

    if t_diag == NULL or t_super == NULL or product_diag == NULL:
        free(t_diag)
        free(t_super)
        free(product_diag)
        return -1

    for i in range(n):
        t_diag[i] = 1.0

    for i in range(n*n):
        if fabs(factors[i]) > factor_scale:
            factor_scale = fabs(factors[i])
    raw_threshold = (
        schur_deflation_tol*DBL_EPSILON*n*factor_scale
    )
    if raw_threshold < smlnum:
        raw_threshold = smlnum

    # T = H_2 ... H_p is upper triangular. Its diagonal and first
    # superdiagonal determine the diagonal and subdiagonal of H_1 T.
    for k in range(1, period):
        for i in range(n - 1):
            t_super[i] = (
                t_diag[i]
                * factors[i + factor_ld*(i + 1) + factor_stride*k]
                + t_super[i]
                * factors[i + 1 + factor_ld*(i + 1) + factor_stride*k]
            )
        for i in range(n):
            t_diag[i] *= factors[
                i + factor_ld*i + factor_stride*k
            ]

    product_diag[0] = factors[0]*t_diag[0]
    for i in range(1, n):
        product_diag[i] = (
            factors[i + factor_ld*(i - 1)]*t_super[i - 1]
            + factors[i + factor_ld*i]*t_diag[i]
        )

    for i in range(n - 1, 0, -1):
        product_subdiag = (
            factors[i + factor_ld*(i - 1)]*t_diag[i - 1]
        )
        threshold = (
            schur_deflation_tol*DBL_EPSILON
            * (fabs(product_diag[i - 1]) + fabs(product_diag[i]))
        )
        if threshold < smlnum:
            threshold = smlnum
        # An implicit split can also arise from an exact-zero diagonal in a
        # later factor. Only delete H_1(i,i-1) when that deletion is itself
        # within the O(n*eps) backward scale of the distinguished factor.
        if (fabs(product_subdiag) <= threshold and
                fabs(factors[i + factor_ld*(i - 1)]) <= raw_threshold):
            factors[i + factor_ld*(i - 1)] = 0.0
            count += 1

    free(t_diag)
    free(t_super)
    free(product_diag)
    return count


cdef int _periodic_hessenberg_to_schur_D_buffers(
    double* factors,
    double* q,
    double* wr,
    double* wi,
    double* work,
    double schur_deflation_tol,
    int period,
    int n,
    int factor_ld,
    int q_ld,
    int lwork,
) noexcept nogil:
    """Reduce real periodic-Hessenberg buffers to periodic Schur form."""
    cdef int info = 0
    cdef int ilo = 1
    cdef int ihi = n
    cdef char job = 83
    cdef char compz = 86

    info = _deflate_small_periodic_hessenberg_subdiagonals_D_buffers(
        factors, schur_deflation_tol, period, n, factor_ld
    )
    if info < 0:
        return info

    mb03wd_(
        &job, &compz, &n, &period, &ilo, &ihi, &ilo, &ihi,
        factors, &factor_ld, &factor_ld, q, &q_ld, &q_ld,
        wr, wi, work, &lwork, &info,
    )
    if info != 0:
        return 3000 + info

    mb03wx_(
        &n, &period, factors, &factor_ld, &factor_ld, wr, wi, &info,
    )
    return 3100 + info if info != 0 else 0


cdef void _zero_small_hessenberg_rows_D_buffers(
    double* factors,
    const double* scale_tol,
    int period,
    int n,
) noexcept nogil:
    """Zero post-Hessenberg real rows below their per-factor cutoff."""
    cdef int k, i, j
    cdef int inc = n
    cdef double row_norm

    for k in range(period):
        for i in range(n):
            row_norm = blas.dnrm2(
                &n,
                &factors[i + n*n*k],
                &inc,
            )
            if row_norm < scale_tol[k]:
                for j in range(n):
                    factors[i + n*j + n*n*k] = 0.0


cdef void _zero_small_hessenberg_rows_Z_buffers(
    double complex* factors,
    const double* scale_tol,
    int period,
    int n,
) noexcept nogil:
    """Zero post-Hessenberg complex rows below their per-factor cutoff."""
    cdef int k, i, j
    cdef int inc = n
    cdef double row_norm

    for k in range(period):
        for i in range(n):
            row_norm = blas.dznrm2(
                &n,
                &factors[i + n*n*k],
                &inc,
            )
            if row_norm < scale_tol[k]:
                for j in range(n):
                    factors[i + n*j + n*n*k] = 0.0


def _deflate_small_periodic_hessenberg_subdiagonals_D(
    factors,
    schur_deflation_tol=10.0,
):
    """Zero negligible real implicit-product subdiagonals in place."""
    cdef cnp.ndarray factors_arr = np.asarray(factors)
    cdef int n
    cdef int period
    cdef int count
    cdef double tol = float(schur_deflation_tol)

    if factors_arr.dtype != np.dtype(np.float64):
        raise TypeError("factors must have dtype float64")
    if (factors_arr.ndim != 3 or
            factors_arr.shape[0] != factors_arr.shape[1] or
            factors_arr.shape[2] < 1):
        raise ValueError("factors must have shape (n, n, period)")
    if not factors_arr.flags.f_contiguous:
        raise ValueError("factors must use Fortran storage")
    if tol < 1.0:
        raise ValueError("schur_deflation_tol must be at least 1")

    n = factors_arr.shape[0]
    period = factors_arr.shape[2]
    if n < 2:
        return 0
    with nogil:
        count = _deflate_small_periodic_hessenberg_subdiagonals_D_buffers(
            <double*>factors_arr.data,
            tol,
            period,
            n,
            n,
        )
    if count < 0:
        raise MemoryError("post-Hessenberg real deflation allocation failed")
    return count


cdef int _periodic_hessenberg_to_schur_Z_buffers(
    double complex* factors,
    double complex* q,
    double complex* alpha,
    double complex* beta,
    int* scale,
    int* signs,
    double* dwork,
    double complex* zwork,
    int period,
    int n,
    int factor_ld,
    int q_ld,
    int lwork,
) noexcept nogil:
    """Reduce complex periodic-Hessenberg buffers to periodic Schur form."""
    cdef int k, info = 0
    cdef int ilo = 1
    cdef int ihi = n
    cdef char job = 83
    cdef char compq = 86

    for k in range(period):
        signs[k] = 1
    mb03bz_(
        &job, &compq, &period, &n, &ilo, &ihi, signs,
        factors, &factor_ld, &factor_ld, q, &q_ld, &q_ld,
        alpha, beta, scale, dwork, &lwork, zwork, &lwork, &info,
    )
    return 4000 + info if info != 0 else 0


cdef void _compose_periodic_schur_D_buffers(
    const double* factors,
    const double* q,
    const int* indices,
    const int* ranks,
    int period,
    int m,
    int n,
    int output_width,
    double* T_out,
    double* Z_out,
) noexcept nogil:
    """Lift real Schur vectors and write leading compact output blocks."""
    cdef int k, i, j, row

    for k in range(period):
        for i in range(n):
            for j in range(n):
                T_out[(k*output_width + i)*output_width + j] = (
                    factors[i + n*j + n*n*k]
                )
        for i in range(ranks[k]):
            row = indices[k*m + i]
            for j in range(n):
                Z_out[(k*m + row)*output_width + j] = q[i + n*j + n*n*k]


cdef void _compose_periodic_schur_Z_buffers(
    const double complex* factors,
    const double complex* q,
    const int* indices,
    const int* ranks,
    int period,
    int m,
    int n,
    int output_width,
    double complex* T_out,
    double complex* Z_out,
) noexcept nogil:
    """Lift complex Schur vectors and write leading compact output blocks."""
    cdef int k, i, j, row

    for k in range(period):
        for i in range(n):
            for j in range(n):
                T_out[(k*output_width + i)*output_width + j] = (
                    factors[i + n*j + n*n*k]
                )
        for i in range(ranks[k]):
            row = indices[k*m + i]
            for j in range(n):
                Z_out[(k*m + row)*output_width + j] = q[i + n*j + n*n*k]


cdef void _compose_periodic_schur_CRed_D_buffers(
    const double* factors,
    const double* q,
    const double* bases,
    const int* indices,
    const int* ranks,
    int period,
    int m,
    int capacity,
    int n,
    int output_width,
    double* T_out,
    double* Z_out,
) noexcept nogil:
    """Lift a real minimum-rank core already in physical period order."""
    cdef int k, rank, i, j, a, row
    cdef double value

    for k in range(period):
        rank = ranks[k]
        for i in range(n):
            for j in range(n):
                T_out[(k*output_width + i)*output_width + j] = (
                    factors[i + n*j + n*n*k]
                )
        for i in range(rank):
            row = indices[k*m + i]
            for j in range(n):
                value = 0.0
                for a in range(n):
                    value += (
                        bases[i + capacity*a + capacity*n*k]
                        * q[a + n*j + n*n*k]
                    )
                Z_out[(k*m + row)*output_width + j] = value


cdef void _compose_periodic_schur_CRed_Z_buffers(
    const double complex* factors,
    const double complex* q,
    const double complex* bases,
    const int* indices,
    const int* ranks,
    int period,
    int m,
    int capacity,
    int n,
    int output_width,
    double complex* T_out,
    double complex* Z_out,
) noexcept nogil:
    """Lift a complex minimum-rank core already in physical period order."""
    cdef int k, rank, i, j, a, row
    cdef double complex value

    for k in range(period):
        rank = ranks[k]
        for i in range(n):
            for j in range(n):
                T_out[(k*output_width + i)*output_width + j] = (
                    factors[i + n*j + n*n*k]
                )
        for i in range(rank):
            row = indices[k*m + i]
            for j in range(n):
                value = 0.0
                for a in range(n):
                    value += (
                        bases[i + capacity*a + capacity*n*k]
                        * q[a + n*j + n*n*k]
                    )
                Z_out[(k*m + row)*output_width + j] = value


cdef int _run_periodic_schur_stages_D(
    const double* H,
    const unsigned char* active_cols,
    const double* scale_tol,
    double schur_deflation_tol,
    int period,
    int m,
    int n,
    int output_width,
    double* T_out,
    double* Z_out,
    double* wr,
    double* wi,
) noexcept nogil:
    """Run the canonical staged real NRed periodic-Schur algorithm."""
    cdef int* indices = <int*>malloc(period*m*sizeof(int))
    cdef int* ranks = <int*>calloc(period, sizeof(int))
    cdef int ldtau = n - 1 if n > 1 else 1
    cdef double* factors = <double*>calloc(n*n*period, sizeof(double))
    cdef double* q = <double*>calloc(n*n*period, sizeof(double))
    cdef double* tau = <double*>calloc(ldtau*period, sizeof(double))
    cdef int lwork = 8*n if 8*n > n + period else n + period
    cdef double* work = <double*>malloc(lwork*sizeof(double))
    cdef int info = 0

    if (indices == NULL or ranks == NULL or factors == NULL or q == NULL or
            tau == NULL or work == NULL):
        free(indices)
        free(ranks)
        free(factors)
        free(q)
        free(tau)
        free(work)
        return -1

    info = _collect_active_metadata_buffers(
        active_cols, period, m, n, indices, ranks
    )
    if info == 0:
        _compact_active_factors_D_buffers(
            H, indices, ranks, period, m, n, factors
        )
    if info == 0:
        info = _make_periodic_hessenberg_NRed_D_buffers(
            factors, q, tau, work, period, n, ldtau, lwork
        )
    if info == 0 and scale_tol != NULL:
        _zero_small_hessenberg_rows_D_buffers(
            factors, scale_tol, period, n
        )
    if info == 0:
        info = _periodic_hessenberg_to_schur_D_buffers(
            factors, q, wr, wi, work, schur_deflation_tol,
            period, n, n, n, lwork
        )
    if info == 0:
        _compose_periodic_schur_D_buffers(
            factors, q, indices, ranks, period, m, n, output_width,
            T_out, Z_out,
        )

    free(indices)
    free(ranks)
    free(factors)
    free(q)
    free(tau)
    free(work)
    return info


cdef int _run_periodic_schur_stages_Z(
    const double complex* H,
    const unsigned char* active_cols,
    const double* scale_tol,
    int period,
    int m,
    int n,
    int output_width,
    double complex* T_out,
    double complex* Z_out,
    double complex* alpha,
    double complex* beta,
    int* scale,
) noexcept nogil:
    """Run the canonical staged complex NRed periodic-Schur algorithm."""
    cdef int* indices = <int*>malloc(period*m*sizeof(int))
    cdef int* ranks = <int*>calloc(period, sizeof(int))
    cdef int* signs = <int*>malloc(period*sizeof(int))
    cdef double complex* factors = <double complex*>calloc(
        n*n*period, sizeof(double complex)
    )
    cdef double complex* q = <double complex*>calloc(
        n*n*period, sizeof(double complex)
    )
    cdef double complex* zwork = <double complex*>malloc(
        n*sizeof(double complex)
    )
    cdef double* dwork = <double*>malloc(n*sizeof(double))
    cdef int info = 0
    cdef int lwork = n

    if (indices == NULL or ranks == NULL or signs == NULL or factors == NULL or
            q == NULL or zwork == NULL or dwork == NULL):
        free(indices)
        free(ranks)
        free(signs)
        free(factors)
        free(q)
        free(zwork)
        free(dwork)
        return -1

    info = _collect_active_metadata_buffers(
        active_cols, period, m, n, indices, ranks
    )
    if info == 0:
        _compact_active_factors_Z_buffers(
            H, indices, ranks, period, m, n, factors
        )
    if info == 0:
        _make_periodic_hessenberg_NRed_Z_buffers(
            factors, q, zwork, period, n
        )
    if info == 0 and scale_tol != NULL:
        _zero_small_hessenberg_rows_Z_buffers(
            factors, scale_tol, period, n
        )
    if info == 0:
        info = _periodic_hessenberg_to_schur_Z_buffers(
            factors, q, alpha, beta, scale, signs, dwork, zwork,
            period, n, n, n, lwork,
        )
    if info == 0:
        _compose_periodic_schur_Z_buffers(
            factors, q, indices, ranks, period, m, n, output_width,
            T_out, Z_out,
        )

    free(indices)
    free(ranks)
    free(signs)
    free(factors)
    free(q)
    free(zwork)
    free(dwork)
    return info


cdef int _run_periodic_schur_CRed_D(
    const double* H,
    const unsigned char* active_cols,
    double schur_deflation_tol,
    int period,
    int m,
    int capacity,
    int n,
    int output_width,
    double* T_out,
    double* Z_out,
    double* wr,
    double* wi,
) noexcept nogil:
    """Run real eig-only minimum-rank QR reduction followed by SLICOT Schur."""
    cdef int* indices = <int*>malloc(period*m*sizeof(int))
    cdef int* ranks = <int*>calloc(period, sizeof(int))
    cdef double* compact = <double*>calloc(
        capacity*capacity*period, sizeof(double)
    )
    cdef double* bases = <double*>calloc(
        capacity*n*period, sizeof(double)
    )
    cdef double* restored_bases = <double*>calloc(
        capacity*n*period, sizeof(double)
    )
    cdef double* qr_tau = <double*>malloc(n*sizeof(double))
    cdef double* factors = <double*>calloc(n*n*period, sizeof(double))
    cdef double* q = <double*>calloc(n*n*period, sizeof(double))
    cdef int ldtau = n - 1 if n > 1 else 1
    cdef double* hess_tau = <double*>calloc(
        ldtau*period, sizeof(double)
    )
    cdef int lwork = 8*n if 8*n > n + period else n + period
    cdef double* work
    cdef int cut = -1
    cdef int k, info = 0

    if capacity > lwork:
        lwork = capacity
    work = <double*>malloc(lwork*sizeof(double))
    if (indices == NULL or ranks == NULL or compact == NULL or
            bases == NULL or restored_bases == NULL or qr_tau == NULL or
            factors == NULL or q == NULL or hess_tau == NULL or work == NULL):
        info = -1
    if info == 0:
        info = _collect_active_metadata_buffers(
            active_cols, period, m, capacity, indices, ranks
        )
    if info == 0:
        for k in range(period):
            if cut < 0 and ranks[k] == n:
                cut = k
        if cut < 0:
            info = -4
    if info == 0:
        _compact_active_factors_rotated_D_buffers(
            H, indices, ranks, cut, period, m, capacity, compact
        )
        info = _cyclic_qr_reduce_D_buffers(
            compact, ranks, cut, bases, qr_tau, work, lwork,
            period, capacity, n,
        )
    if info == 0:
        # TODO: Fuse CRed and periodic Hessenberg to avoid this intervening
        # canonical-order copy while retaining factor zero as the public
        # Hessenberg/quasi-triangular factor.
        _restore_CRed_order_D_buffers(
            compact, bases, cut, period, capacity, n,
            factors, restored_bases,
        )
        info = _make_periodic_hessenberg_NRed_D_buffers(
            factors, q, hess_tau, work, period, n, ldtau, lwork
        )
    if info == 0:
        info = _periodic_hessenberg_to_schur_D_buffers(
            factors, q, wr, wi, work, schur_deflation_tol,
            period, n, n, n, lwork
        )
    if info == 0:
        _compose_periodic_schur_CRed_D_buffers(
            factors, q, restored_bases, indices, ranks,
            period, m, capacity, n, output_width, T_out, Z_out,
        )

    free(indices)
    free(ranks)
    free(compact)
    free(bases)
    free(restored_bases)
    free(qr_tau)
    free(factors)
    free(q)
    free(hess_tau)
    free(work)
    return info


def _debug_periodic_schur_CRed_D_stages(H, active_cols=None):
    r"""Return every real CRed factor checkpoint from the production kernels.

    This diagnostic entry point follows the same buffer-level path as
    :func:`periodic_schur_D` with ``reduction="CRed"``. It returns
    ``(H_compact, H_cred, H_hessenberg, T, Z, wr, wi, ranks, cut)``.
    All factor checkpoints use the incoming cyclic order. ``cut`` records the
    minimum-rank cut used internally by the reverse thin-QR sweep.
    """
    cdef tuple inputs = _compaction_inputs(H, active_cols, np.float64)
    cdef cnp.ndarray H_arr = inputs[0]
    cdef cnp.ndarray active_arr = inputs[1]
    cdef int period = H_arr.shape[0]
    cdef int m = H_arr.shape[1]
    cdef int capacity = periodic_schur_active_size(
        <const unsigned char*>active_arr.data, period, m
    )
    cdef int n = periodic_schur_active_min_size(
        <const unsigned char*>active_arr.data, period, m
    )
    cdef cnp.ndarray indices_arr = np.empty(
        (period, m), dtype=np.intc
    )
    cdef cnp.ndarray ranks_c_arr = np.empty(period, dtype=np.intc)
    cdef cnp.ndarray compact_before_arr = np.zeros(
        (capacity, capacity, period), dtype=np.float64, order="F"
    )
    cdef cnp.ndarray compact_arr = np.zeros(
        (capacity, capacity, period), dtype=np.float64, order="F"
    )
    cdef cnp.ndarray bases_arr = np.zeros(
        (capacity, n, period), dtype=np.float64, order="F"
    )
    cdef cnp.ndarray restored_bases_arr = np.zeros(
        (capacity, n, period), dtype=np.float64, order="F"
    )
    cdef cnp.ndarray qr_tau_arr = np.empty(max(1, n), dtype=np.float64)
    cdef cnp.ndarray factors_arr = np.zeros(
        (n, n, period), dtype=np.float64, order="F"
    )
    cdef cnp.ndarray q_arr = np.zeros(
        (n, n, period), dtype=np.float64, order="F"
    )
    cdef int ldtau = n - 1 if n > 1 else 1
    cdef cnp.ndarray hess_tau_arr = np.zeros(
        (ldtau, period), dtype=np.float64, order="F"
    )
    cdef int lwork = max(capacity, max(8*n, n + period))
    cdef cnp.ndarray work_arr = np.empty(max(1, lwork), dtype=np.float64)
    cdef cnp.ndarray after_cred_arr
    cdef cnp.ndarray after_hessenberg_arr
    cdef cnp.ndarray T_arr = np.zeros(
        (period, n, n), dtype=np.float64
    )
    cdef cnp.ndarray Z_arr = np.zeros(
        (period, m, n), dtype=np.float64
    )
    cdef cnp.ndarray wr_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray wi_arr = np.empty(n, dtype=np.float64)
    cdef int cut = -1
    cdef int k, info = 0

    if n == 0:
        raise ValueError("the CRed diagnostic requires a nonempty live core")

    with nogil:
        info = _collect_active_metadata_buffers(
            <const unsigned char*>active_arr.data,
            period, m, capacity,
            <int*>indices_arr.data,
            <int*>ranks_c_arr.data,
        )
        if info == 0:
            _compact_active_factors_D_buffers(
                <const double*>H_arr.data,
                <const int*>indices_arr.data,
                <const int*>ranks_c_arr.data,
                period, m, capacity,
                <double*>compact_before_arr.data,
            )
            for k in range(period):
                if cut < 0 and (<int*>ranks_c_arr.data)[k] == n:
                    cut = k
            if cut < 0:
                info = -4
        if info == 0:
            _compact_active_factors_rotated_D_buffers(
                <const double*>H_arr.data,
                <const int*>indices_arr.data,
                <const int*>ranks_c_arr.data,
                cut, period, m, capacity,
                <double*>compact_arr.data,
            )
            info = _cyclic_qr_reduce_D_buffers(
                <double*>compact_arr.data,
                <const int*>ranks_c_arr.data,
                cut,
                <double*>bases_arr.data,
                <double*>qr_tau_arr.data,
                <double*>work_arr.data,
                lwork, period, capacity, n,
            )
        if info == 0:
            _restore_CRed_order_D_buffers(
                <const double*>compact_arr.data,
                <const double*>bases_arr.data,
                cut, period, capacity, n,
                <double*>factors_arr.data,
                <double*>restored_bases_arr.data,
            )

    if info != 0:
        raise np.linalg.LinAlgError(
            f"real periodic CRed preprocessing failed with info={info}"
        )

    after_cred_arr = np.array(factors_arr, order="F", copy=True)
    with nogil:
        info = _make_periodic_hessenberg_NRed_D_buffers(
            <double*>factors_arr.data,
            <double*>q_arr.data,
            <double*>hess_tau_arr.data,
            <double*>work_arr.data,
            period, n, ldtau, lwork,
        )
    if info != 0:
        raise np.linalg.LinAlgError(
            f"real periodic Hessenberg reduction failed with info={info}"
        )

    after_hessenberg_arr = np.array(factors_arr, order="F", copy=True)
    with nogil:
        info = _periodic_hessenberg_to_schur_D_buffers(
            <double*>factors_arr.data,
            <double*>q_arr.data,
            <double*>wr_arr.data,
            <double*>wi_arr.data,
            <double*>work_arr.data,
            10.0, period, n, n, n, lwork,
        )
        if info == 0:
            _compose_periodic_schur_CRed_D_buffers(
                <const double*>factors_arr.data,
                <const double*>q_arr.data,
                <const double*>restored_bases_arr.data,
                <const int*>indices_arr.data,
                <const int*>ranks_c_arr.data,
                period, m, capacity, n, n,
                <double*>T_arr.data,
                <double*>Z_arr.data,
            )
    if info != 0:
        raise np.linalg.LinAlgError(f"MB03WD failed with info={info}")

    return (
        np.moveaxis(compact_before_arr, -1, 0).copy(),
        np.moveaxis(after_cred_arr, -1, 0).copy(),
        np.moveaxis(after_hessenberg_arr, -1, 0).copy(),
        T_arr,
        Z_arr,
        wr_arr,
        wi_arr,
        np.asarray(ranks_c_arr, dtype=np.intp),
        cut,
    )


cdef int _run_periodic_schur_CRed_Z(
    const double complex* H,
    const unsigned char* active_cols,
    int period,
    int m,
    int capacity,
    int n,
    int output_width,
    double complex* T_out,
    double complex* Z_out,
    double complex* alpha,
    double complex* beta,
    int* scale,
) noexcept nogil:
    """Run complex eig-only minimum-rank QR reduction and periodic Schur."""
    cdef int* indices = <int*>malloc(period*m*sizeof(int))
    cdef int* ranks = <int*>calloc(period, sizeof(int))
    cdef int* signs = <int*>malloc(period*sizeof(int))
    cdef double complex* compact = <double complex*>calloc(
        capacity*capacity*period, sizeof(double complex)
    )
    cdef double complex* bases = <double complex*>calloc(
        capacity*n*period, sizeof(double complex)
    )
    cdef double complex* restored_bases = <double complex*>calloc(
        capacity*n*period, sizeof(double complex)
    )
    cdef double complex* qr_tau = <double complex*>malloc(
        n*sizeof(double complex)
    )
    cdef double complex* factors = <double complex*>calloc(
        n*n*period, sizeof(double complex)
    )
    cdef double complex* q = <double complex*>calloc(
        n*n*period, sizeof(double complex)
    )
    cdef int lwork = capacity if capacity > n else n
    cdef double complex* zwork = <double complex*>malloc(
        lwork*sizeof(double complex)
    )
    cdef double* dwork = <double*>malloc(lwork*sizeof(double))
    cdef int cut = -1
    cdef int k, info = 0

    if (indices == NULL or ranks == NULL or signs == NULL or compact == NULL or
            bases == NULL or restored_bases == NULL or qr_tau == NULL or
            factors == NULL or q == NULL or zwork == NULL or dwork == NULL):
        info = -1
    if info == 0:
        info = _collect_active_metadata_buffers(
            active_cols, period, m, capacity, indices, ranks
        )
    if info == 0:
        for k in range(period):
            if cut < 0 and ranks[k] == n:
                cut = k
        if cut < 0:
            info = -4
    if info == 0:
        _compact_active_factors_rotated_Z_buffers(
            H, indices, ranks, cut, period, m, capacity, compact
        )
        info = _cyclic_qr_reduce_Z_buffers(
            compact, ranks, cut, bases, qr_tau, zwork, lwork,
            period, capacity, n,
        )
    if info == 0:
        _restore_CRed_order_Z_buffers(
            compact, bases, cut, period, capacity, n,
            factors, restored_bases,
        )
        _make_periodic_hessenberg_NRed_Z_buffers(
            factors, q, zwork, period, n
        )
        info = _periodic_hessenberg_to_schur_Z_buffers(
            factors, q, alpha, beta, scale, signs, dwork, zwork,
            period, n, n, n, lwork,
        )
    if info == 0:
        _compose_periodic_schur_CRed_Z_buffers(
            factors, q, restored_bases, indices, ranks,
            period, m, capacity, n, output_width, T_out, Z_out,
        )

    free(indices)
    free(ranks)
    free(signs)
    free(compact)
    free(bases)
    free(restored_bases)
    free(qr_tau)
    free(factors)
    free(q)
    free(zwork)
    free(dwork)
    return info


cdef public api int compute_periodic_schur_active_D(
    const double* H,
    const unsigned char* active_cols,
    double schur_deflation_tol,
    int period,
    int m,
    int n,
    int output_width,
    double* T_out,
    double* Z_out,
    double* wr,
    double* wi,
) noexcept nogil:
    r"""Compute the active real NRed Schur form into C-order output buffers."""
    cdef int i

    if output_width < n or schur_deflation_tol < 1.0:
        return -3
    for i in range(period*output_width*output_width):
        T_out[i] = 0.0
    for i in range(period*m*output_width):
        Z_out[i] = 0.0
    for i in range(output_width):
        wr[i] = 0.0
        wi[i] = 0.0
    if n == 0:
        return 0
    return _run_periodic_schur_stages_D(
        H, active_cols, NULL, schur_deflation_tol,
        period, m, n, output_width,
        T_out, Z_out, wr, wi,
    )


cdef public api int compute_periodic_schur_active_CRed_D(
    const double* H,
    const unsigned char* active_cols,
    double schur_deflation_tol,
    int period,
    int m,
    int capacity,
    int n,
    int output_width,
    double* T_out,
    double* Z_out,
    double* wr,
    double* wi,
) noexcept nogil:
    """Compute the real eig-only minimum-rank periodic Schur form."""
    cdef int i

    if (output_width < n or capacity < n or
            schur_deflation_tol < 1.0):
        return -3
    for i in range(period*output_width*output_width):
        T_out[i] = 0.0
    for i in range(period*m*output_width):
        Z_out[i] = 0.0
    for i in range(output_width):
        wr[i] = 0.0
        wi[i] = 0.0
    if n == 0:
        return 0
    return _run_periodic_schur_CRed_D(
        H, active_cols, schur_deflation_tol,
        period, m, capacity, n, output_width,
        T_out, Z_out, wr, wi,
    )


cdef public api int compute_periodic_schur_active_scaled_D(
    const double* H,
    const unsigned char* active_cols,
    const double* scale_tol,
    double schur_deflation_tol,
    int period,
    int m,
    int n,
    int output_width,
    double* T_out,
    double* Z_out,
    double* wr,
    double* wi,
) noexcept nogil:
    """Compute active real Schur form after post-Hessenberg row deflation."""
    cdef int i

    if output_width < n or schur_deflation_tol < 1.0:
        return -3
    for i in range(period*output_width*output_width):
        T_out[i] = 0.0
    for i in range(period*m*output_width):
        Z_out[i] = 0.0
    for i in range(output_width):
        wr[i] = 0.0
        wi[i] = 0.0
    if n == 0:
        return 0
    return _run_periodic_schur_stages_D(
        H, active_cols, scale_tol, schur_deflation_tol,
        period, m, n, output_width,
        T_out, Z_out, wr, wi,
    )


cdef public api int compute_periodic_schur_active_Z(
    const void* H_buffer,
    const unsigned char* active_cols,
    int period,
    int m,
    int n,
    int output_width,
    void* T_buffer,
    void* Z_buffer,
    void* alpha_buffer,
    void* beta_buffer,
    int* scale,
) noexcept nogil:
    r"""Compute the active complex NRed Schur form into C-order buffers."""
    cdef const double complex* H = <const double complex*>H_buffer
    cdef double complex* T_out = <double complex*>T_buffer
    cdef double complex* Z_out = <double complex*>Z_buffer
    cdef double complex* alpha = <double complex*>alpha_buffer
    cdef double complex* beta = <double complex*>beta_buffer
    cdef int i

    if output_width < n:
        return -3
    for i in range(period*output_width*output_width):
        T_out[i] = 0.0
    for i in range(period*m*output_width):
        Z_out[i] = 0.0
    for i in range(output_width):
        alpha[i] = 0.0
        beta[i] = 1.0
        scale[i] = 0
    if n == 0:
        return 0
    return _run_periodic_schur_stages_Z(
        H, active_cols, NULL, period, m, n, output_width,
        T_out, Z_out, alpha, beta, scale,
    )


cdef public api int compute_periodic_schur_active_CRed_Z(
    const void* H_buffer,
    const unsigned char* active_cols,
    int period,
    int m,
    int capacity,
    int n,
    int output_width,
    void* T_buffer,
    void* Z_buffer,
    void* alpha_buffer,
    void* beta_buffer,
    int* scale,
) noexcept nogil:
    """Compute the complex eig-only minimum-rank periodic Schur form."""
    cdef const double complex* H = <const double complex*>H_buffer
    cdef double complex* T_out = <double complex*>T_buffer
    cdef double complex* Z_out = <double complex*>Z_buffer
    cdef double complex* alpha = <double complex*>alpha_buffer
    cdef double complex* beta = <double complex*>beta_buffer
    cdef int i

    if output_width < n or capacity < n:
        return -3
    for i in range(period*output_width*output_width):
        T_out[i] = 0.0
    for i in range(period*m*output_width):
        Z_out[i] = 0.0
    for i in range(output_width):
        alpha[i] = 0.0
        beta[i] = 1.0
        scale[i] = 0
    if n == 0:
        return 0
    return _run_periodic_schur_CRed_Z(
        H, active_cols, period, m, capacity, n, output_width,
        T_out, Z_out, alpha, beta, scale,
    )


cdef public api int compute_periodic_schur_active_scaled_Z(
    const void* H_buffer,
    const unsigned char* active_cols,
    const double* scale_tol,
    int period,
    int m,
    int n,
    int output_width,
    void* T_buffer,
    void* Z_buffer,
    void* alpha_buffer,
    void* beta_buffer,
    int* scale,
) noexcept nogil:
    """Compute active complex Schur form after Hessenberg row deflation."""
    cdef const double complex* H = <const double complex*>H_buffer
    cdef double complex* T_out = <double complex*>T_buffer
    cdef double complex* Z_out = <double complex*>Z_buffer
    cdef double complex* alpha = <double complex*>alpha_buffer
    cdef double complex* beta = <double complex*>beta_buffer
    cdef int i

    if output_width < n:
        return -3
    for i in range(period*output_width*output_width):
        T_out[i] = 0.0
    for i in range(period*m*output_width):
        Z_out[i] = 0.0
    for i in range(output_width):
        alpha[i] = 0.0
        beta[i] = 1.0
        scale[i] = 0
    if n == 0:
        return 0
    return _run_periodic_schur_stages_Z(
        H, active_cols, scale_tol, period, m, n, output_width,
        T_out, Z_out, alpha, beta, scale,
    )


cdef int _reorder_period_one_D(
    const double* T,
    const double* Z,
    const unsigned char* select,
    int m,
    int n,
    double* T_out,
    double* Z_out,
) noexcept nogil:
    """Reorder one real Schur factor with LAPACK DTRSEN."""
    cdef double* T_f = <double*>malloc(n*n*sizeof(double))
    cdef double* Q_f = <double*>malloc(n*n*sizeof(double))
    cdef double* wr = <double*>malloc(n*sizeof(double))
    cdef double* wi = <double*>malloc(n*sizeof(double))
    cdef double* work = <double*>malloc((n if n > 1 else 1)*sizeof(double))
    cdef int* select_i = <int*>malloc(n*sizeof(int))
    cdef int* iwork = <int*>malloc((n if n > 1 else 1)*sizeof(int))
    cdef int i, j
    cdef int selected_count = 0
    cdef int lwork = n if n > 1 else 1
    cdef int liwork = n if n > 1 else 1
    cdef int info = 0
    cdef double s = 0.0
    cdef double sep = 0.0
    cdef double alpha = 1.0
    cdef double beta = 0.0
    cdef char job = 78
    cdef char compq = 86
    cdef char trans = 84
    cdef char no_trans = 78

    if (T_f == NULL or Q_f == NULL or wr == NULL or wi == NULL or
            work == NULL or select_i == NULL or iwork == NULL):
        free(T_f)
        free(Q_f)
        free(wr)
        free(wi)
        free(work)
        free(select_i)
        free(iwork)
        return -100

    for j in range(n):
        select_i[j] = 1 if select[j] else 0
        for i in range(n):
            T_f[i + n*j] = T[i*n + j]
            Q_f[i + n*j] = 1.0 if i == j else 0.0

    lapack.dtrsen(
        &job, &compq, <bint*>select_i, &n,
        T_f, &n, Q_f, &n, wr, wi,
        &selected_count, &s, &sep,
        work, &lwork, iwork, &liwork, &info,
    )
    if info == 0:
        for i in range(n):
            for j in range(n):
                T_out[i*n + j] = T_f[i + n*j]
        # Row-major Z_out = Z @ Q is column-major Z_out.T = Q.T @ Z.T.
        blas.dgemm(
            &trans, &no_trans, &n, &m, &n,
            &alpha, Q_f, &n, <double*>Z, &n,
            &beta, Z_out, &n,
        )

    free(T_f)
    free(Q_f)
    free(wr)
    free(wi)
    free(work)
    free(select_i)
    free(iwork)
    return info


cdef int _reorder_period_one_Z(
    const double complex* T,
    const double complex* Z,
    const unsigned char* select,
    int m,
    int n,
    double complex* T_out,
    double complex* Z_out,
) noexcept nogil:
    """Reorder one complex Schur factor with LAPACK ZTRSEN."""
    cdef double complex* T_f = <double complex*>malloc(
        n*n*sizeof(double complex)
    )
    cdef double complex* Q_f = <double complex*>malloc(
        n*n*sizeof(double complex)
    )
    cdef double complex* w = <double complex*>malloc(
        n*sizeof(double complex)
    )
    cdef double complex* work = <double complex*>malloc(
        (n if n > 1 else 1)*sizeof(double complex)
    )
    cdef int* select_i = <int*>malloc(n*sizeof(int))
    cdef int i, j
    cdef int selected_count = 0
    cdef int lwork = n if n > 1 else 1
    cdef int info = 0
    cdef double s = 0.0
    cdef double sep = 0.0
    cdef double complex alpha = 1.0
    cdef double complex beta = 0.0
    cdef char job = 78
    cdef char compq = 86
    cdef char trans = 84
    cdef char no_trans = 78

    if (T_f == NULL or Q_f == NULL or w == NULL or
            work == NULL or select_i == NULL):
        free(T_f)
        free(Q_f)
        free(w)
        free(work)
        free(select_i)
        return -100

    for j in range(n):
        select_i[j] = 1 if select[j] else 0
        for i in range(n):
            T_f[i + n*j] = T[i*n + j]
            Q_f[i + n*j] = 1.0 if i == j else 0.0

    lapack.ztrsen(
        &job, &compq, <bint*>select_i, &n,
        T_f, &n, Q_f, &n, w,
        &selected_count, &s, &sep,
        work, &lwork, &info,
    )
    if info == 0:
        for i in range(n):
            for j in range(n):
                T_out[i*n + j] = T_f[i + n*j]
        # Use transpose, not conjugate transpose: Z_out = Z @ Q.
        blas.zgemm(
            &trans, &no_trans, &n, &m, &n,
            &alpha, Q_f, &n, <double complex*>Z, &n,
            &beta, Z_out, &n,
        )

    free(T_f)
    free(Q_f)
    free(w)
    free(work)
    free(select_i)
    return info


cdef public api int compute_reordered_periodic_schur_D(
    const double* T,
    const double* Z,
    const unsigned char* select,
    int period,
    int m,
    int n,
    double tol,
    double* T_out,
    double* Z_out,
) noexcept nogil:
    """Reorder an R-oriented real periodic Schur form into C-order outputs."""
    cdef double* T_slicot
    cdef double* Q_slicot
    cdef int* whichq
    cdef int* dims
    cdef int* induced_dims
    cdef int* signs
    cdef int* leading_dims
    cdef int* offsets
    cdef int* select_i
    cdef int* iwork
    cdef double* dwork
    cdef int k, qk, i, j
    cdef int selected_count = 0
    cdef int ldwork
    cdef int info = 0
    cdef double alpha = 1.0
    cdef double beta = 0.0
    cdef char compq = 73
    cdef char strong = 78
    cdef char trans = 84
    cdef char no_trans = 78

    for i in range(period*n*n):
        T_out[i] = T[i]
    for i in range(period*m*n):
        Z_out[i] = Z[i]
    if n == 0:
        return 0
    if period == 1:
        return _reorder_period_one_D(T, Z, select, m, n, T_out, Z_out)

    ldwork = 42*period + (n if n > 10 else 0)
    if 80*period - 48 > ldwork:
        ldwork = 80*period - 48
    if ldwork < 1:
        ldwork = 1

    T_slicot = <double*>malloc(period*n*n*sizeof(double))
    Q_slicot = <double*>malloc(period*n*n*sizeof(double))
    whichq = <int*>calloc(period, sizeof(int))
    dims = <int*>malloc(period*sizeof(int))
    induced_dims = <int*>calloc(period, sizeof(int))
    signs = <int*>malloc(period*sizeof(int))
    leading_dims = <int*>malloc(period*sizeof(int))
    offsets = <int*>malloc(period*sizeof(int))
    select_i = <int*>malloc(n*sizeof(int))
    iwork = <int*>malloc(4*period*sizeof(int))
    dwork = <double*>malloc(ldwork*sizeof(double))
    if (T_slicot == NULL or Q_slicot == NULL or whichq == NULL or
            dims == NULL or induced_dims == NULL or signs == NULL or
            leading_dims == NULL or offsets == NULL or select_i == NULL or
            iwork == NULL or dwork == NULL):
        info = -100
    else:
        for k in range(period):
            dims[k] = n
            signs[k] = 1
            leading_dims[k] = n
            offsets[k] = 1 + k*n*n
            for i in range(n):
                for j in range(n):
                    T_slicot[i + n*j + n*n*k] = (
                        T[((period - 1 - k)*n + i)*n + j]
                    )
        for i in range(n):
            select_i[i] = 1 if select[i] else 0

        mb03kd_(
            &compq, whichq, &strong,
            &period, &n, &period,
            dims, induced_dims, signs, select_i,
            T_slicot, leading_dims, offsets,
            Q_slicot, leading_dims, offsets,
            &selected_count, &tol,
            iwork, dwork, &ldwork, &info,
        )
        if info == 0:
            for k in range(period):
                for i in range(n):
                    for j in range(n):
                        T_out[(k*n + i)*n + j] = (
                            T_slicot[i + n*j + n*n*(period - 1 - k)]
                        )
                qk = (period - k) % period
                blas.dgemm(
                    &trans, &no_trans, &n, &m, &n,
                    &alpha, &Q_slicot[n*n*qk], &n,
                    <double*>&Z[k*m*n], &n,
                    &beta, &Z_out[k*m*n], &n,
                )

    free(T_slicot)
    free(Q_slicot)
    free(whichq)
    free(dims)
    free(induced_dims)
    free(signs)
    free(leading_dims)
    free(offsets)
    free(select_i)
    free(iwork)
    free(dwork)
    return info


cdef public api int compute_reordered_periodic_schur_Z(
    const void* T_buffer,
    const void* Z_buffer,
    const unsigned char* select,
    int period,
    int m,
    int n,
    double tol,
    void* T_out_buffer,
    void* Z_out_buffer,
) noexcept nogil:
    """Reorder an R-oriented complex periodic Schur form into C-order outputs."""
    cdef const double complex* T = <const double complex*>T_buffer
    cdef const double complex* Z = <const double complex*>Z_buffer
    cdef double complex* T_out = <double complex*>T_out_buffer
    cdef double complex* Z_out = <double complex*>Z_out_buffer
    cdef double complex* sylvester
    cdef double complex* rhs
    cdef int* piv
    cdef double* gc
    cdef double complex* gs
    cdef int i, position, leading = 0, j, info = 0

    for i in range(period*n*n):
        T_out[i] = T[i]
    for i in range(period*m*n):
        Z_out[i] = Z[i]
    if n == 0:
        return 0
    if period == 1:
        return _reorder_period_one_Z(T, Z, select, m, n, T_out, Z_out)

    sylvester = <double complex*>malloc(
        period*period*sizeof(double complex)
    )
    rhs = <double complex*>malloc(period*sizeof(double complex))
    piv = <int*>malloc(period*sizeof(int))
    gc = <double*>malloc(period*sizeof(double))
    gs = <double complex*>malloc(period*sizeof(double complex))
    if (sylvester == NULL or rhs == NULL or piv == NULL or
            gc == NULL or gs == NULL):
        info = -100
    else:
        for position in range(n):
            if select[position]:
                for j in range(position - 1, leading - 1, -1):
                    info = _adjacent_swap_complex_buffers(
                        T_out, Z_out, period, m, n, j, tol,
                        sylvester, rhs, piv, gc, gs,
                    )
                    if info != 0:
                        break
                if info != 0:
                    break
                leading += 1

    free(sylvester)
    free(rhs)
    free(piv)
    free(gc)
    free(gs)
    return info


cdef tuple _compaction_inputs(object H, object active_cols, object dtype):
    """Return exact-dtype C-order factors and an expanded active mask."""
    cdef cnp.ndarray H_arr = np.asarray(H)
    cdef cnp.ndarray active_arr
    cdef Py_ssize_t rank

    if H_arr.dtype != np.dtype(dtype):
        raise TypeError(f"H must have dtype {np.dtype(dtype)}")
    if H_arr.ndim != 3 or H_arr.shape[1] != H_arr.shape[2]:
        raise ValueError("H must have shape (period, m, m)")
    if H_arr.shape[0] < 1:
        raise ValueError("H must contain at least one factor")

    if active_cols is None:
        active_arr = np.ones(
            (H_arr.shape[0], H_arr.shape[1]), dtype=np.uint8
        )
    elif np.ndim(active_cols) == 0:
        rank = active_cols.__index__()
        if rank < 0 or rank > H_arr.shape[1]:
            raise ValueError("active_cols rank must satisfy 0 <= r <= m")
        active_arr = np.zeros(
            (H_arr.shape[0], H_arr.shape[1]), dtype=np.uint8
        )
        active_arr[:, :rank] = 1
    else:
        active_arr = np.ascontiguousarray(active_cols, dtype=np.uint8)
    if (active_arr.ndim != 2 or
            active_arr.shape[0] != H_arr.shape[0] or
            active_arr.shape[1] != H_arr.shape[1]):
        raise ValueError("active_cols must have shape (period, m)")
    return np.ascontiguousarray(H_arr), active_arr


cdef tuple _compact_active_slicot_d(const DTYPE_t[:, :, ::1] H,
                                    unsigned char[:, ::1] active):
    """Pack real active factors and selectors in trailing-period storage."""
    cdef int period = H.shape[0]
    cdef int m = H.shape[1]
    cdef int max_rank = periodic_schur_active_size(
        &active[0, 0], period, m
    )
    cdef cnp.ndarray ranks_c_arr = np.empty(period, dtype=np.intc)
    cdef cnp.ndarray indices_arr = np.empty((period, m), dtype=np.intc)
    cdef cnp.ndarray ranks_arr
    cdef cnp.ndarray factors_arr = np.zeros(
        (max_rank, max_rank, period), dtype=np.float64, order="F"
    )
    cdef cnp.ndarray bases_arr = np.zeros(
        (m, max_rank, period), dtype=np.float64, order="F"
    )
    cdef int[::1] ranks = ranks_c_arr
    cdef int[:, ::1] indices = indices_arr
    cdef DTYPE_t[::1, :, :] factors = factors_arr
    cdef DTYPE_t[::1, :, :] bases = bases_arr
    cdef int k, i
    cdef int info

    with nogil:
        info = _collect_active_metadata_buffers(
            &active[0, 0], period, m, max_rank,
            <int*>indices_arr.data, <int*>ranks_c_arr.data,
        )
        if info == 0 and max_rank:
            _compact_active_factors_D_buffers(
                &H[0, 0, 0],
                <int*>indices_arr.data,
                <int*>ranks_c_arr.data,
                period, m, max_rank,
                <double*>factors_arr.data,
            )
        for k in range(period):
            for i in range(ranks[k]):
                bases[indices[k, i], i, k] = 1.0

    if info != 0:
        raise ValueError(f"active compaction failed with info={info}")
    ranks_arr = np.asarray(ranks_c_arr, dtype=np.intp)
    return factors_arr, bases_arr, ranks_arr


cdef tuple _compact_active_slicot_z(const ZTYPE_t[:, :, ::1] H,
                                    unsigned char[:, ::1] active):
    """Pack complex active factors and selectors in trailing-period storage."""
    cdef int period = H.shape[0]
    cdef int m = H.shape[1]
    cdef int max_rank = periodic_schur_active_size(
        &active[0, 0], period, m
    )
    cdef cnp.ndarray ranks_c_arr = np.empty(period, dtype=np.intc)
    cdef cnp.ndarray indices_arr = np.empty((period, m), dtype=np.intc)
    cdef cnp.ndarray ranks_arr
    cdef cnp.ndarray factors_arr = np.zeros(
        (max_rank, max_rank, period), dtype=np.complex128, order="F"
    )
    cdef cnp.ndarray bases_arr = np.zeros(
        (m, max_rank, period), dtype=np.complex128, order="F"
    )
    cdef int[::1] ranks = ranks_c_arr
    cdef int[:, ::1] indices = indices_arr
    cdef ZTYPE_t[::1, :, :] factors = factors_arr
    cdef ZTYPE_t[::1, :, :] bases = bases_arr
    cdef int k, i
    cdef int info

    with nogil:
        info = _collect_active_metadata_buffers(
            &active[0, 0], period, m, max_rank,
            <int*>indices_arr.data, <int*>ranks_c_arr.data,
        )
        if info == 0 and max_rank:
            _compact_active_factors_Z_buffers(
                &H[0, 0, 0],
                <int*>indices_arr.data,
                <int*>ranks_c_arr.data,
                period, m, max_rank,
                <double complex*>factors_arr.data,
            )
        for k in range(period):
            for i in range(ranks[k]):
                bases[indices[k, i], i, k] = 1.0

    if info != 0:
        raise ValueError(f"active compaction failed with info={info}")
    ranks_arr = np.asarray(ranks_c_arr, dtype=np.intp)
    return factors_arr, bases_arr, ranks_arr


def _compact_active_slicot_stage1_d(H, active_cols=None):
    r"""Expose FP64 Stage 1 state for focused internal tests.

    Input plane ``k`` is already factor ``C[k]`` in SLICOT's formal product
    ``C[0] C[1] ... C[period-1]``.  Its compact form has logical shape
    ``(ranks[k], ranks[k+1])``.  Active columns retain their order within every
    cut.

    Stage 1 never rotates the cycle. A future CRed stage owns the minimum-rank
    rotation and its ``cut_offset``. Returns
    ``(factors, bases, ranks)``. ``factors`` has
    Fortran shape ``(max(ranks), max(ranks), period)``; only each logical
    top-left rectangle is live.  ``bases[:, :ranks[k], k]`` is the active
    selector for node ``k`` in the incoming cyclic order.
    """
    H_arr, active_arr = _compaction_inputs(H, active_cols, np.float64)
    return _compact_active_slicot_d(H_arr, active_arr)


def _compact_active_slicot_stage1_z(H, active_cols=None):
    r"""Expose complex128 Stage 1 state for focused internal tests.

    The storage, node ordering, logical rectangular dimensions, and returned
    metadata are identical to :func:`_compact_active_slicot_stage1_d`.
    """
    H_arr, active_arr = _compaction_inputs(H, active_cols, np.complex128)
    return _compact_active_slicot_z(H_arr, active_arr)


def _make_periodic_hessenberg_slicot_d(factors):
    r"""Reduce a zero-padded FP64 periodic product with ``MB03VD/VY``.

    ``factors`` has trailing-period Fortran shape ``(n, n, period)``.  The
    logical rectangular factors produced by active compaction are embedded in
    this smallest common square size by exact zero padding.  ``MB03VD`` mutates
    the buffer to periodic Hessenberg-triangular form, while ``MB03VY`` forms
    the square node transforms ``Q`` satisfying

    ``H[k] @ Q[k+1] = Q[k] @ factors[k]``.

    The returned ``Q`` stays separate from the rectangular compaction basis;
    their product is the map from Schur coordinates to the original padded
    Arnoldi coordinates.
    """
    cdef cnp.ndarray factors_arr = np.asarray(factors)
    cdef int n
    cdef int period
    cdef int ldtau
    cdef int lwork
    cdef cnp.ndarray tau_arr
    cdef cnp.ndarray q_arr
    cdef cnp.ndarray work_arr
    cdef int info

    if factors_arr.dtype != np.dtype(np.float64):
        raise TypeError("factors must have dtype float64")
    if (factors_arr.ndim != 3 or
            factors_arr.shape[0] != factors_arr.shape[1] or
            factors_arr.shape[2] < 1):
        raise ValueError("factors must have shape (n, n, period)")
    if not factors_arr.flags.f_contiguous:
        raise ValueError("factors must use Fortran storage")

    n = factors_arr.shape[0]
    period = factors_arr.shape[2]
    if n == 0:
        return factors_arr, np.empty((0, 0, period), dtype=np.float64, order="F")

    ldtau = max(1, n - 1)
    tau_arr = np.zeros((ldtau, period), dtype=np.float64, order="F")
    q_arr = np.empty((n, n, period), dtype=np.float64, order="F")
    lwork = max(8*n, n + period)
    work_arr = np.empty(lwork, dtype=np.float64)
    with nogil:
        info = _make_periodic_hessenberg_NRed_D_buffers(
            <double*>factors_arr.data,
            <double*>q_arr.data,
            <double*>tau_arr.data,
            <double*>work_arr.data,
            period, n, ldtau, lwork,
        )
    if info != 0:
        raise np.linalg.LinAlgError(
            f"real periodic Hessenberg reduction failed with info={info}"
        )
    return factors_arr, q_arr


cdef tuple _periodic_hessenberg_inputs(
    object factors,
    object bases,
    object ranks,
    object dtype,
):
    """Validate the common stage-1 state consumed by stage 2."""
    cdef cnp.ndarray factors_arr = np.asarray(factors)
    cdef cnp.ndarray bases_arr = np.asarray(bases)
    cdef cnp.ndarray ranks_arr = np.asarray(ranks)
    cdef Py_ssize_t period

    if factors_arr.dtype != np.dtype(dtype) or bases_arr.dtype != np.dtype(dtype):
        raise TypeError(f"factors and bases must have dtype {np.dtype(dtype)}")
    if (factors_arr.ndim != 3 or
            factors_arr.shape[0] != factors_arr.shape[1] or
            factors_arr.shape[2] < 1):
        raise ValueError("factors must have shape (capacity, capacity, period)")
    if not factors_arr.flags.f_contiguous:
        raise ValueError("factors must use Fortran storage")
    period = factors_arr.shape[2]
    if (bases_arr.ndim != 3 or
            bases_arr.shape[1] != factors_arr.shape[0] or
            bases_arr.shape[2] != period):
        raise ValueError("bases must have shape (m, capacity, period)")
    if not bases_arr.flags.f_contiguous:
        raise ValueError("bases must use Fortran storage")
    if (ranks_arr.dtype != np.dtype(np.intp) or
            ranks_arr.ndim != 1 or
            ranks_arr.shape[0] != period or
            not ranks_arr.flags.c_contiguous):
        raise ValueError("ranks must be a contiguous intp vector of length period")
    if np.any(ranks_arr < 0) or np.any(ranks_arr > factors_arr.shape[0]):
        raise ValueError("ranks must lie within the factor capacity")
    return (
        factors_arr,
        bases_arr,
        ranks_arr,
    )


def _make_periodic_hessenberg_NRed_D(
    factors,
    bases,
    ranks,
):
    r"""Make the zero-padded FP64 product periodic Hessenberg without reduction.

    The three inputs are exactly the stage-1 output in incoming cyclic order.
    The factor capacity ``max(ranks)`` is retained, including its exact padded
    zero sector.  ``MB03VD/MB03VY`` reduce the factors and form square node
    transforms ``q``.

    Returns ``(factors, bases, q, ranks)``. The input arrays and metadata are
    returned by identity and are not repacked. With
    ``n = q.shape[0]`` and ``Z_H[k] = bases[:, :n, k] @ q[:, :, k]``, the
    stage-2 relation is ``H[k] Z_H[k+1] = Z_H[k] factors[k]`` in packed cyclic
    order.  Stage 3 may therefore accumulate Schur transforms directly into
    ``q``.
    """
    cdef tuple state = _periodic_hessenberg_inputs(
        factors, bases, ranks, np.float64
    )
    cdef cnp.ndarray q
    factors, q = _make_periodic_hessenberg_slicot_d(state[0])
    return factors, state[1], q, state[2]


def _make_periodic_hessenberg_NRed_Z(
    factors,
    bases,
    ranks,
):
    r"""Make the zero-padded complex128 product periodic Hessenberg.

    The input and output contract is identical to
    :func:`_make_periodic_hessenberg_NRed_D`. SLICOT has no complex analogue of
    ``MB03VD/MB03VY``, so a square cyclic Householder sweep reduces the full
    padded capacity and accumulates its transformations into ``q``.
    """
    cdef tuple state = _periodic_hessenberg_inputs(
        factors, bases, ranks, np.complex128
    )
    cdef cnp.ndarray factors_arr = state[0]
    cdef int n = factors_arr.shape[0]
    cdef int period = factors_arr.shape[2]
    cdef cnp.ndarray q_arr = np.zeros(
        (n, n, period), dtype=np.complex128, order="F"
    )
    cdef cnp.ndarray work_arr = np.empty(max(1, n), dtype=np.complex128)

    if n:
        with nogil:
            _make_periodic_hessenberg_NRed_Z_buffers(
                <double complex*>factors_arr.data,
                <double complex*>q_arr.data,
                <double complex*>work_arr.data,
                period, n,
            )
    return factors_arr, state[1], q_arr, state[2]


cdef tuple _minimum_cut_qr_CRed(
    object factors,
    object bases,
    object ranks,
    object dtype,
):
    """Return a minimum-cut square cycle and its folded rectangular bases."""
    cdef tuple state = _periodic_hessenberg_inputs(
        factors, bases, ranks, dtype
    )
    cdef cnp.ndarray factors_arr = state[0]
    cdef cnp.ndarray bases_arr = state[1]
    cdef cnp.ndarray ranks_arr = state[2]
    cdef int period = factors_arr.shape[2]
    cdef int cut = int(np.argmin(ranks_arr))
    cdef int n = int(ranks_arr[cut])
    cdef cnp.ndarray rotated_ranks = np.roll(ranks_arr, -cut)
    cdef cnp.ndarray rotated_factors = np.asfortranarray(
        np.roll(factors_arr, -cut, axis=2)
    )
    cdef cnp.ndarray rotated_bases = np.asfortranarray(
        np.roll(bases_arr, -cut, axis=2)
    )
    cdef list factor_list = []
    cdef list basis_list = []
    cdef int k
    cdef object q_k
    cdef object r_k

    for k in range(period):
        factor_list.append(
            np.array(
                rotated_factors[
                    :rotated_ranks[k],
                    :rotated_ranks[(k + 1) % period],
                    k,
                ],
                order="F",
                copy=True,
            )
        )
        basis_list.append(
            np.array(
                rotated_bases[:, :rotated_ranks[k], k],
                order="F",
                copy=True,
            )
        )

    for k in range(period - 1, 0, -1):
        q_k, r_k = np.linalg.qr(factor_list[k], mode="reduced")
        factor_list[k] = r_k
        factor_list[k - 1] = factor_list[k - 1] @ q_k
        basis_list[k] = basis_list[k] @ q_k

    return (
        np.asfortranarray(np.stack(factor_list, axis=2)),
        np.asfortranarray(np.stack(basis_list, axis=2)),
        np.full(period, n, dtype=np.intp),
        ranks_arr,
        cut,
    )


def _qrp_deflate_CRed(
    factors,
    bases,
    ranks,
    rank_tol,
):
    r"""Contract numerical null directions revealed by cyclic pivoted QR.

    Every square factor is factorized as ``A[:, piv] = Q R``. Pivot ``i`` is
    retained when

    ``abs(R[i, i]) > rank_tol * eps * norm(A[:, piv[i]])``.

    The smallest rank found around the cycle selects the contraction cut.
    Its retained ``Q`` range is folded into that node basis and propagated
    backward through a reverse thin-QR sweep. The scan and contraction repeat
    until every factor has full numerical rank at the current dimension.

    Returns ``(factors, bases, ranks, cut_offset)`` in the final rotated order.
    ``cut_offset`` maps that order back to the order received here.
    """
    cdef tuple state = _periodic_hessenberg_inputs(
        factors, bases, ranks, np.asarray(factors).dtype
    )
    cdef cnp.ndarray factors_arr = state[0]
    cdef cnp.ndarray bases_arr = state[1]
    cdef int period = factors_arr.shape[2]
    cdef int n
    cdef int cut
    cdef int new_n
    cdef int total_cut = 0
    cdef int k
    cdef double tol = float(rank_tol)
    cdef double eps = np.finfo(np.float64).eps
    cdef list candidate_ranks
    cdef list range_bases
    cdef list factor_list
    cdef list basis_list
    cdef object factor
    cdef object column_norms
    cdef object q_k
    cdef object r_k
    cdef object piv
    cdef object keep
    cdef object U

    if tol < 0:
        raise ValueError("rank_tol must be nonnegative")
    if tol == 0:
        return factors_arr, bases_arr, state[2], total_cut

    while True:
        n = factors_arr.shape[0]
        if n == 0:
            return factors_arr, bases_arr, state[2], total_cut

        candidate_ranks = []
        range_bases = []
        for k in range(period):
            factor = factors_arr[:, :, k]
            column_norms = np.linalg.norm(factor, axis=0)
            q_k, r_k, piv = scipy.linalg.qr(
                factor,
                mode="economic",
                pivoting=True,
            )
            keep = (
                np.abs(np.diag(r_k))
                > tol * eps * column_norms[piv]
            )
            candidate_ranks.append(int(np.sum(keep)))
            range_bases.append(np.asfortranarray(q_k[:, keep]))

        cut = int(np.argmin(candidate_ranks))
        new_n = candidate_ranks[cut]
        if new_n == n:
            return (
                factors_arr,
                bases_arr,
                np.full(period, n, dtype=np.intp),
                total_cut,
            )

        factor_list = [
            np.array(factor, order="F", copy=True)
            for factor in np.roll(factors_arr, -cut, axis=2).transpose(2, 0, 1)
        ]
        basis_list = [
            np.array(basis, order="F", copy=True)
            for basis in np.roll(bases_arr, -cut, axis=2).transpose(2, 0, 1)
        ]
        U = range_bases[cut]

        # H[0] maps node 1 into the contracted node 0. H[-1] maps node 0 out.
        if period == 1:
            factor_list[0] = U.conj().T @ factor_list[0] @ U
        else:
            factor_list[0] = U.conj().T @ factor_list[0]
            factor_list[period - 1] = factor_list[period - 1] @ U
        basis_list[0] = basis_list[0] @ U

        for k in range(period - 1, 0, -1):
            q_k, r_k = np.linalg.qr(factor_list[k], mode="reduced")
            factor_list[k] = r_k
            factor_list[k - 1] = factor_list[k - 1] @ q_k
            basis_list[k] = basis_list[k] @ q_k

        factors_arr = np.asfortranarray(np.stack(factor_list, axis=2))
        bases_arr = np.asfortranarray(np.stack(basis_list, axis=2))
        state = (
            factors_arr,
            bases_arr,
            np.full(period, new_n, dtype=np.intp),
        )
        total_cut = (total_cut + cut) % period


cdef tuple _restore_CRed_order(object factors, object bases, int cut):
    """Restore rotated CRed factors and folded bases to physical period order."""
    # TODO: Fuse CRed and periodic Hessenberg so this canonicalization does
    # not require a separate array pass.
    if cut == 0:
        return factors, bases
    return (
        np.asfortranarray(np.roll(factors, cut, axis=2)),
        np.asfortranarray(np.roll(bases, cut, axis=2)),
    )


def _make_periodic_hessenberg_CRed_D(
    factors,
    bases,
    ranks,
    rank_tol=None,
):
    r"""Reduce the real compact cycle and restore it before MB03VD/VY.

    If ``rank_tol`` is supplied, iterative QRP contraction runs after the
    structural minimum-rank sweep. The reduced factors and folded bases return
    to incoming period order before periodic Hessenberg reduction.
    """
    cdef tuple reduced = _minimum_cut_qr_CRed(
        factors, bases, ranks, np.float64
    )
    cdef tuple deflated
    cdef tuple restored
    cdef tuple prepared
    cdef int cut = reduced[4]
    if rank_tol is not None:
        deflated = _qrp_deflate_CRed(
            reduced[0],
            reduced[1],
            reduced[2],
            rank_tol,
        )
        reduced = (
            deflated[0],
            deflated[1],
            deflated[2],
            reduced[3],
            (cut + deflated[3]) % np.asarray(ranks).shape[0],
        )
    restored = _restore_CRed_order(reduced[0], reduced[1], reduced[4])
    prepared = _make_periodic_hessenberg_NRed_D(
        restored[0], restored[1], reduced[2]
    )
    return prepared


def _make_periodic_hessenberg_CRed_Z(
    factors,
    bases,
    ranks,
    rank_tol=None,
):
    r"""Reduce and restore the complex cycle before the Hessenberg sweep.

    If ``rank_tol`` is supplied, the real driver's iterative QRP contraction
    is applied with conjugate-transpose range projections. The reduced factors
    and folded bases return to incoming period order before stage 2.
    """
    cdef tuple reduced = _minimum_cut_qr_CRed(
        factors, bases, ranks, np.complex128
    )
    cdef tuple deflated
    cdef tuple restored
    cdef tuple prepared
    cdef int cut = reduced[4]
    if rank_tol is not None:
        deflated = _qrp_deflate_CRed(
            reduced[0],
            reduced[1],
            reduced[2],
            rank_tol,
        )
        reduced = (
            deflated[0],
            deflated[1],
            deflated[2],
            reduced[3],
            (cut + deflated[3]) % np.asarray(ranks).shape[0],
        )
    restored = _restore_CRed_order(reduced[0], reduced[1], reduced[4])
    prepared = _make_periodic_hessenberg_NRed_Z(
        restored[0], restored[1], reduced[2]
    )
    return prepared


def _periodic_hessenberg_to_schur_D(
    factors,
    q,
    schur_deflation_tol=10.0,
):
    r"""Run ``MB03WD`` and refresh its FP64 eigenvalues with ``MB03WX``.

    ``q`` contains any transformations accumulated while making periodic
    Hessenberg form and is updated in place by ``COMPZ='V'``.  Its square
    dimension defines the live SLICOT problem; ``factors`` may have a larger
    leading dimension for a minimum-rank preprocessing mode. ``MB03WX``
    recomputes ``wr, wi`` from the final Schur blocks so their order matches
    the returned factors. Returns ``(factors, q, wr, wi)`` in formal SLICOT
    order without decoding, rotating, or composing the rectangular Arnoldi
    basis.
    """
    cdef cnp.ndarray factors_arr = np.asarray(factors)
    cdef cnp.ndarray q_arr = np.asarray(q)
    cdef int capacity
    cdef int n
    cdef int period
    cdef int lwork
    cdef cnp.ndarray wr_arr
    cdef cnp.ndarray wi_arr
    cdef cnp.ndarray work_arr
    cdef int info
    cdef double deflation_value = float(schur_deflation_tol)

    if factors_arr.dtype != np.dtype(np.float64) or q_arr.dtype != np.dtype(np.float64):
        raise TypeError("factors and q must have dtype float64")
    if (factors_arr.ndim != 3 or
            factors_arr.shape[0] != factors_arr.shape[1] or
            factors_arr.shape[2] < 1):
        raise ValueError("factors must have shape (capacity, capacity, period)")
    if (q_arr.ndim != 3 or
            q_arr.shape[0] != q_arr.shape[1] or
            q_arr.shape[2] != factors_arr.shape[2]):
        raise ValueError("q must have shape (n, n, period)")
    if not factors_arr.flags.f_contiguous or not q_arr.flags.f_contiguous:
        raise ValueError("factors and q must use Fortran storage")
    if deflation_value < 1.0:
        raise ValueError("schur_deflation_tol must be at least 1")

    capacity = factors_arr.shape[0]
    n = q_arr.shape[0]
    period = factors_arr.shape[2]
    if n > capacity:
        raise ValueError("q dimension cannot exceed factor capacity")
    if n == 0:
        return (
            factors_arr,
            q_arr,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    wr_arr = np.empty(n, dtype=np.float64)
    wi_arr = np.empty(n, dtype=np.float64)
    lwork = max(8*n, n + period)
    work_arr = np.empty(lwork, dtype=np.float64)
    with nogil:
        info = _periodic_hessenberg_to_schur_D_buffers(
            <double*>factors_arr.data,
            <double*>q_arr.data,
            <double*>wr_arr.data,
            <double*>wi_arr.data,
            <double*>work_arr.data,
            deflation_value, period, n, capacity, n, lwork,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"MB03WD failed with info={info}")
    return factors_arr, q_arr, wr_arr, wi_arr


def _periodic_hessenberg_to_schur_Z(factors, q):
    r"""Run all-positive ``MB03BZ`` on a complex periodic Hessenberg product.

    Storage, in-place transform accumulation, and formal ordering match
    :func:`_periodic_hessenberg_to_schur_D`. Returns
    ``(factors, q, alpha, beta, scale)`` with SLICOT's scaled eigenvalue
    representation left intact for native reordering or Sylvester consumers.
    """
    cdef cnp.ndarray factors_arr = np.asarray(factors)
    cdef cnp.ndarray q_arr = np.asarray(q)
    cdef int capacity
    cdef int n
    cdef int period
    cdef int lwork
    cdef cnp.ndarray signs_arr
    cdef cnp.ndarray alpha_arr
    cdef cnp.ndarray beta_arr
    cdef cnp.ndarray scale_arr
    cdef cnp.ndarray dwork_arr
    cdef cnp.ndarray zwork_arr
    cdef int info

    if (factors_arr.dtype != np.dtype(np.complex128) or
            q_arr.dtype != np.dtype(np.complex128)):
        raise TypeError("factors and q must have dtype complex128")
    if (factors_arr.ndim != 3 or
            factors_arr.shape[0] != factors_arr.shape[1] or
            factors_arr.shape[2] < 1):
        raise ValueError("factors must have shape (capacity, capacity, period)")
    if (q_arr.ndim != 3 or
            q_arr.shape[0] != q_arr.shape[1] or
            q_arr.shape[2] != factors_arr.shape[2]):
        raise ValueError("q must have shape (n, n, period)")
    if not factors_arr.flags.f_contiguous or not q_arr.flags.f_contiguous:
        raise ValueError("factors and q must use Fortran storage")

    capacity = factors_arr.shape[0]
    n = q_arr.shape[0]
    period = factors_arr.shape[2]
    if n > capacity:
        raise ValueError("q dimension cannot exceed factor capacity")
    if n == 0:
        return (
            factors_arr,
            q_arr,
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.intc),
        )

    signs_arr = np.ones(period, dtype=np.intc)
    alpha_arr = np.empty(n, dtype=np.complex128)
    beta_arr = np.empty(n, dtype=np.complex128)
    scale_arr = np.empty(n, dtype=np.intc)
    lwork = max(1, n)
    dwork_arr = np.empty(lwork, dtype=np.float64)
    zwork_arr = np.empty(lwork, dtype=np.complex128)
    with nogil:
        info = _periodic_hessenberg_to_schur_Z_buffers(
            <double complex*>factors_arr.data,
            <double complex*>q_arr.data,
            <double complex*>alpha_arr.data,
            <double complex*>beta_arr.data,
            <int*>scale_arr.data,
            <int*>signs_arr.data,
            <double*>dwork_arr.data,
            <double complex*>zwork_arr.data,
            period, n, capacity, n, lwork,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"MB03BZ failed with info={info}")
    return factors_arr, q_arr, alpha_arr, beta_arr, scale_arr


cdef tuple _compose_periodic_schur_output(
    object factors,
    object bases,
    object q,
):
    r"""Compose the physical Schur map from canonical site-ordered factors."""
    cdef cnp.ndarray factors_arr = np.asarray(factors)
    cdef cnp.ndarray bases_arr = np.asarray(bases)
    cdef cnp.ndarray q_arr = np.asarray(q)
    cdef Py_ssize_t n = q_arr.shape[0]
    cdef Py_ssize_t period = q_arr.shape[2]
    cdef Py_ssize_t k
    cdef cnp.ndarray z_arr = np.empty(
        (bases_arr.shape[0], n, period), dtype=bases_arr.dtype, order="F"
    )
    cdef cnp.ndarray t_out
    cdef cnp.ndarray z_out

    for k in range(period):
        z_arr[:, :, k] = bases_arr[:, :n, k] @ q_arr[:, :, k]
    t_out = np.moveaxis(factors_arr[:n, :n, :], -1, 0).copy()
    z_out = np.moveaxis(z_arr, -1, 0).copy()
    return t_out, z_out


cdef inline void _left_real(DTYPE_t[:, ::1] A, Py_ssize_t p, Py_ssize_t q,
                           double c, double s, Py_ssize_t col_start) noexcept nogil:
    """Apply a real Givens rotation to two rows."""
    cdef Py_ssize_t j
    cdef double xp, xq
    for j in range(col_start, A.shape[1]):
        xp = A[p, j]
        xq = A[q, j]
        A[p, j] = c * xp + s * xq
        A[q, j] = -s * xp + c * xq


cdef inline void _right_adj_real(DTYPE_t[:, ::1] A, Py_ssize_t p, Py_ssize_t q,
                                double c, double s, Py_ssize_t row_start,
                                Py_ssize_t row_stop) noexcept nogil:
    """Apply a real adjoint Givens rotation to two columns."""
    cdef Py_ssize_t i
    cdef double xp, xq
    for i in range(row_start, row_stop):
        xp = A[i, p]
        xq = A[i, q]
        A[i, p] = c * xp + s * xq
        A[i, q] = -s * xp + c * xq


cdef inline void _left_complex(ZTYPE_t[:, ::1] A, Py_ssize_t p, Py_ssize_t q,
                              double c, double complex s,
                              Py_ssize_t col_start) noexcept nogil:
    """Apply a complex Givens rotation to two rows."""
    cdef Py_ssize_t j
    cdef double complex xp, xq
    for j in range(col_start, A.shape[1]):
        xp = A[p, j]
        xq = A[q, j]
        A[p, j] = c * xp + s * xq
        A[q, j] = -s.conjugate() * xp + c * xq


cdef inline void _right_adj_complex(ZTYPE_t[:, ::1] A, Py_ssize_t p, Py_ssize_t q,
                                   double c, double complex s,
                                   Py_ssize_t row_start,
                                   Py_ssize_t row_stop) noexcept nogil:
    """Apply a complex adjoint Givens rotation to two columns."""
    cdef Py_ssize_t i
    cdef double complex xp, xq
    for i in range(row_start, row_stop):
        xp = A[i, p]
        xq = A[i, q]
        A[i, p] = c * xp + s.conjugate() * xq
        A[i, q] = -s * xp + c * xq


cdef int _adjacent_swap_complex_buffers(
    double complex* T,
    double complex* Z,
    int period,
    int m,
    int n,
    int j,
    double tol,
    double complex* sylvester,
    double complex* rhs,
    int* piv,
    double* gc,
    double complex* gs,
) noexcept nogil:
    r"""Swap adjacent 1-by-1 blocks in a complex periodic Schur form.

    For the local factors

    ``A[k] = [[a[k], d[k]], [0, b[k]]]``,

    solve ``a[k] x[k+1] - b[k] x[k] = -d[k]`` and construct the sitewise
    Givens rotations that map ``[x[k], 1]`` to the first coordinate.
    """
    cdef int nrhs = 1
    cdef int info = 0
    cdef int k, kp, row, col, base
    cdef double complex one = 1.0
    cdef double complex r
    cdef double complex a, b, d, v0, v1, fill
    cdef double complex xp, xq
    cdef double local_norm = 0.0
    cdef double factor_norm
    cdef double threshold
    cdef double eps
    cdef double small
    cdef char precision_code = 80
    cdef char safe_minimum_code = 83

    eps = lapack.dlamch(&precision_code)
    small = lapack.dlamch(&safe_minimum_code) / eps

    for col in range(period):
        for row in range(period):
            sylvester[row + period*col] = 0.0
    for k in range(period):
        kp = (k + 1) % period
        base = k*n*n
        a = T[base + j*n + j]
        b = T[base + (j + 1)*n + j + 1]
        d = T[base + j*n + j + 1]
        sylvester[k + period*k] = -b
        sylvester[k + period*kp] += a
        rhs[k] = -d
        local_norm = hypot(local_norm, hypot(a.real, a.imag))
        local_norm = hypot(local_norm, hypot(b.real, b.imag))
        local_norm = hypot(local_norm, hypot(d.real, d.imag))
        factor_norm = hypot(
            hypot(a.real, a.imag),
            hypot(hypot(b.real, b.imag), hypot(d.real, d.imag)),
        )
        if factor_norm < small:
            factor_norm = small
        sylvester[k + period*k] /= factor_norm
        sylvester[k + period*kp] /= factor_norm
        rhs[k] /= factor_norm

    lapack.zgesv(
        &period, &nrhs, sylvester, &period, piv, rhs, &period, &info
    )
    if info != 0:
        return info

    for k in range(period):
        lapack.zlartg(&rhs[k], &one, &gc[k], &gs[k], &r)

    threshold = tol * eps * local_norm
    if threshold < small:
        threshold = small

    # Weak stability test: every transformed local block must remain upper
    # triangular before any caller-visible factor or basis is changed.
    for k in range(period):
        kp = (k + 1) % period
        base = k*n*n
        a = T[base + j*n + j]
        b = T[base + (j + 1)*n + j + 1]
        d = T[base + j*n + j + 1]
        v0 = a*gc[kp] + d*gs[kp].conjugate()
        v1 = b*gs[kp].conjugate()
        fill = -gs[k].conjugate()*v0 + gc[k]*v1
        if hypot(fill.real, fill.imag) > threshold:
            return period + 1

    for k in range(period):
        kp = (k + 1) % period
        base = k*n*n
        for col in range(n):
            xp = T[base + j*n + col]
            xq = T[base + (j + 1)*n + col]
            T[base + j*n + col] = gc[k]*xp + gs[k]*xq
            T[base + (j + 1)*n + col] = (
                -gs[k].conjugate()*xp + gc[k]*xq
            )
        for row in range(n):
            xp = T[base + row*n + j]
            xq = T[base + row*n + j + 1]
            T[base + row*n + j] = (
                gc[kp]*xp + gs[kp].conjugate()*xq
            )
            T[base + row*n + j + 1] = -gs[kp]*xp + gc[kp]*xq
        T[base + (j + 1)*n + j] = 0.0

        base = k*m*n
        for row in range(m):
            xp = Z[base + row*n + j]
            xq = Z[base + row*n + j + 1]
            Z[base + row*n + j] = gc[k]*xp + gs[k].conjugate()*xq
            Z[base + row*n + j + 1] = -gs[k]*xp + gc[k]*xq
    return 0


cdef void _chase_real_views(DTYPE_t[:, :, ::1] factors, unsigned char[::1] signs,
                            DTYPE_t[:, :, ::1] bases) noexcept:
    """Run the real signed Givens chase on a contiguous factor stack."""
    cdef Py_ssize_t p = factors.shape[0]
    cdef Py_ssize_t n = factors.shape[1]
    cdef Py_ssize_t j, i, l
    cdef DTYPE_t[:, ::1] A0 = factors[0]
    cdef DTYPE_t[:, ::1] Al
    cdef DTYPE_t[:, ::1] Bl
    cdef double c, s, r
    cdef Py_ssize_t[:] gp = np.empty(n, dtype=np.intp)
    cdef Py_ssize_t[:] gq = np.empty(n, dtype=np.intp)
    cdef DTYPE_t[:] gc = np.empty(n, dtype=np.float64)
    cdef DTYPE_t[:] gs = np.empty(n, dtype=np.float64)

    with nogil:
        for j in range(n - 2):
            for i in range(n - 1, j + 1, -1):
                lapack.dlartg(&A0[i - 1, j], &A0[i, j], &c, &s, &r)
                A0[i - 1, j] = r
                A0[i, j] = 0.0
                _left_real(A0, i - 1, i, c, s, j + 1)
                Bl = bases[0]
                _right_adj_real(Bl, i - 1, i, c, s, 0, n)
                gp[i] = i - 1
                gq[i] = i
                gc[i] = c
                gs[i] = s

            for l in range(p - 1, 0, -1):
                Al = factors[l]
                if signs[l]:
                    for i in range(n - 1, j + 1, -1):
                        _right_adj_real(Al, gp[i], gq[i], gc[i], gs[i], 0, i + 1)
                        lapack.dlartg(&Al[i - 1, i - 1], &Al[i, i - 1], &c, &s, &r)
                        Al[i - 1, i - 1] = r
                        Al[i, i - 1] = 0.0
                        _left_real(Al, i - 1, i, c, s, i)
                        gp[i] = i - 1
                        gq[i] = i
                        gc[i] = c
                        gs[i] = s
                else:
                    for i in range(n - 1, j + 1, -1):
                        _left_real(Al, gp[i], gq[i], gc[i], gs[i], i - 1)
                        lapack.dlartg(&Al[i, i], &Al[i, i - 1], &c, &s, &r)
                        Al[i, i] = r
                        Al[i, i - 1] = 0.0
                        _right_adj_real(Al, i, i - 1, c, s, 0, i)
                        gp[i] = i - 1
                        gq[i] = i
                        gc[i] = c
                        gs[i] = -s

                Bl = bases[l]
                for i in range(n - 1, j + 1, -1):
                    _right_adj_real(Bl, gp[i], gq[i], gc[i], gs[i], 0, n)

            for i in range(n - 1, j + 1, -1):
                _right_adj_real(A0, gp[i], gq[i], gc[i], gs[i], 0, n)

cdef void _chase_complex_views(ZTYPE_t[:, :, ::1] factors, unsigned char[::1] signs,
                               ZTYPE_t[:, :, ::1] bases) noexcept:
    """Run the complex signed Givens chase on a contiguous factor stack."""
    cdef Py_ssize_t p = factors.shape[0]
    cdef Py_ssize_t n = factors.shape[1]
    cdef Py_ssize_t j, i, l
    cdef ZTYPE_t[:, ::1] A0 = factors[0]
    cdef ZTYPE_t[:, ::1] Al
    cdef ZTYPE_t[:, ::1] Bl
    cdef double complex s, r
    cdef double c
    cdef Py_ssize_t[:] gp = np.empty(n, dtype=np.intp)
    cdef Py_ssize_t[:] gq = np.empty(n, dtype=np.intp)
    cdef DTYPE_t[:] gc = np.empty(n, dtype=np.float64)
    cdef ZTYPE_t[:] gs = np.empty(n, dtype=np.complex128)

    with nogil:
        for j in range(n - 2):
            for i in range(n - 1, j + 1, -1):
                lapack.zlartg(&A0[i - 1, j], &A0[i, j], &c, &s, &r)
                A0[i - 1, j] = r
                A0[i, j] = 0.0
                _left_complex(A0, i - 1, i, c, s, j + 1)
                Bl = bases[0]
                _right_adj_complex(Bl, i - 1, i, c, s, 0, n)
                gp[i] = i - 1
                gq[i] = i
                gc[i] = c
                gs[i] = s

            for l in range(p - 1, 0, -1):
                Al = factors[l]
                if signs[l]:
                    for i in range(n - 1, j + 1, -1):
                        _right_adj_complex(Al, gp[i], gq[i], gc[i], gs[i], 0, i + 1)
                        lapack.zlartg(&Al[i - 1, i - 1], &Al[i, i - 1], &c, &s, &r)
                        Al[i - 1, i - 1] = r
                        Al[i, i - 1] = 0.0
                        _left_complex(Al, i - 1, i, c, s, i)
                        gp[i] = i - 1
                        gq[i] = i
                        gc[i] = c
                        gs[i] = s
                else:
                    for i in range(n - 1, j + 1, -1):
                        _left_complex(Al, gp[i], gq[i], gc[i], gs[i], i - 1)
                        lapack.zlartg(&Al[i, i], &Al[i, i - 1], &c, &s, &r)
                        Al[i, i] = r
                        Al[i, i - 1] = 0.0
                        _right_adj_complex(Al, i, i - 1, c, s.conjugate(), 0, i)
                        gp[i] = i - 1
                        gq[i] = i
                        gc[i] = c
                        gs[i] = -s

                Bl = bases[l]
                for i in range(n - 1, j + 1, -1):
                    _right_adj_complex(Bl, gp[i], gq[i], gc[i], gs[i], 0, n)

            for i in range(n - 1, j + 1, -1):
                _right_adj_complex(A0, gp[i], gq[i], gc[i], gs[i], 0, n)

def givens_chase(factors, signs):
    """Return factor and square basis stacks after the signed Givens chase."""
    dtype = np.asarray(factors).dtype
    signs_arr = np.ascontiguousarray(np.asarray(signs, dtype=np.uint8))
    if np.issubdtype(dtype, np.complexfloating):
        F = np.ascontiguousarray(factors, dtype=np.complex128)
        B = np.ascontiguousarray(np.broadcast_to(np.eye(F.shape[1], dtype=np.complex128), F.shape).copy())
        _chase_complex_views(F, signs_arr, B)
        return F, B

    F = np.ascontiguousarray(factors, dtype=np.float64)
    B = np.ascontiguousarray(np.broadcast_to(np.eye(F.shape[1], dtype=np.float64), F.shape).copy())
    _chase_real_views(F, signs_arr, B)
    return F, B
