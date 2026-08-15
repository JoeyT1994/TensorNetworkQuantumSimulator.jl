# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

"""Archived experiments toward the future CRed_D/Z stage-2 workflows.

This source is intentionally excluded from setup_periodic_schur.py. It
preserves the partially working minimum-rank cyclic reductions developed before
the public API was reduced to the four semantic NRed_D/Z and CRed_D/Z paths.

The experiments below contain:

* a full-factor reverse economic-QR sweep followed by a Givens chase;
* a block-panel QR variant intended to exploit Arnoldi block-Hessenberg input;
* a fused MB03VD-style Householder reduction combining cyclic QR with periodic
  Hessenberg reduction; and
* a structural num_rows specialization of that fused reduction.

Benchmarks found no useful end-to-end advantage from panel QR or num_rows:
fill-in during the cyclic sweep rapidly destroys the input sparsity. The fused
Householder route was likewise not meaningfully faster than the modular
economic-QR-plus-Hessenberg route. None of these symbols are production API.

CRed_D/Z are nevertheless unfinished, not abandoned. A future implementation
must reduce to min(ranks), preserve the common stage-2 basis relation, and
return exactly (factors, bases, q, ranks, block_ranks, cut_offset) as specified
by the NotImplemented production entry points. This file is a source archive
and may need light reconciliation with the production validators before being
added to a dedicated experimental build.

"""

import numpy as np

cimport numpy as cnp
cimport scipy.linalg.cython_lapack as lapack


ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t ZTYPE_t


cdef tuple _full_qr_inputs(object factors, object bases, object ranks,
                           object dtype):
    """Validate mutable Fortran workspaces for the in-place full QR sweep."""
    cdef cnp.ndarray factors_arr = np.asarray(factors)
    cdef cnp.ndarray bases_arr = np.asarray(bases)
    cdef cnp.ndarray ranks_arr = np.asarray(ranks)

    if factors_arr.dtype != np.dtype(dtype) or bases_arr.dtype != np.dtype(dtype):
        raise TypeError(f"factors and bases must have dtype {np.dtype(dtype)}")
    if (factors_arr.ndim != 3 or
            factors_arr.shape[0] != factors_arr.shape[1] or
            factors_arr.shape[2] < 1):
        raise ValueError("factors must have shape (capacity, capacity, period)")
    if not factors_arr.flags.f_contiguous:
        raise ValueError("factors must use Fortran storage")
    if (bases_arr.ndim != 3 or
            bases_arr.shape[1] != factors_arr.shape[0] or
            bases_arr.shape[2] != factors_arr.shape[2]):
        raise ValueError("bases must have shape (m, capacity, period)")
    if not bases_arr.flags.f_contiguous:
        raise ValueError("bases must use Fortran storage")
    if (ranks_arr.dtype != np.dtype(np.intp) or
            ranks_arr.ndim != 1 or
            ranks_arr.shape[0] != factors_arr.shape[2] or
            not ranks_arr.flags.c_contiguous):
        raise ValueError("ranks must be a contiguous intp vector of length period")
    if np.any(ranks_arr < 0) or np.any(ranks_arr > factors_arr.shape[0]):
        raise ValueError("ranks must lie within the factor capacity")
    if np.any(ranks_arr < ranks_arr[0]):
        raise ValueError("cut zero must have minimum rank")
    return factors_arr, bases_arr, ranks_arr


def _full_qr_sweep_d(factors, bases, ranks):
    r"""Reduce packed FP64 factors to one dense and later triangular factors.

    The operation is in place.  Cut zero must already have minimum rank ``n``.
    For ``k = period-1, ..., 1``, this computes ``C[k] = Q[k] R[k]``, applies
    ``Q[k]`` from the right to ``C[k-1]`` and ``bases[k]``, and retains their
    first ``n`` columns.  On return all logical ranks equal ``n``, factor zero
    is dense, and factors one onward are upper triangular.
    """
    cdef tuple arrays = _full_qr_inputs(factors, bases, ranks, np.float64)
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef DTYPE_t[::1, :, :] F = factors_arr
    cdef DTYPE_t[::1, :, :] U = bases_arr
    cdef Py_ssize_t[::1] r = ranks_arr
    cdef int capacity = factors_arr.shape[0]
    cdef int period = factors_arr.shape[2]
    cdef int basis_rows = bases_arr.shape[0]
    cdef int n = r[0]
    cdef cnp.ndarray tau_arr
    cdef cnp.ndarray query_arr
    cdef cnp.ndarray work_arr
    cdef DTYPE_t[::1] tau
    cdef DTYPE_t[::1] query
    cdef DTYPE_t[::1] work
    cdef char side = 'R'
    cdef char trans = 'N'
    cdef int lwork = -1
    cdef int info = 0
    cdef int query_m, query_n, query_k
    cdef int qr_work, factor_work, basis_work
    cdef int k, km, rows, prev_rows, i, j, failure = 0

    if n == 0:
        ranks_arr[:] = 0
        return factors_arr, bases_arr, ranks_arr
    if period == 1:
        return factors_arr, bases_arr, ranks_arr

    tau_arr = np.empty(n, dtype=np.float64)
    query_arr = np.empty(1, dtype=np.float64)
    tau = tau_arr
    query = query_arr
    query_m = capacity
    query_n = n
    query_k = n

    with nogil:
        lapack.dgeqrf(
            &query_m, &query_n, &F[0, 0, 0], &capacity, &tau[0],
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"DGEQRF workspace query failed with info={info}")
    qr_work = max(1, <int>query[0])

    query_m = capacity
    query_n = capacity
    with nogil:
        lapack.dormqr(
            &side, &trans, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &F[0, 0, 0], &capacity,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"DORMQR factor workspace query failed with info={info}")
    factor_work = max(1, <int>query[0])

    query_m = basis_rows
    query_n = capacity
    with nogil:
        lapack.dormqr(
            &side, &trans, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &U[0, 0, 0], &basis_rows,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"DORMQR basis workspace query failed with info={info}")
    basis_work = max(1, <int>query[0])

    lwork = max(qr_work, factor_work, basis_work)
    work_arr = np.empty(lwork, dtype=np.float64)
    work = work_arr

    with nogil:
        for k in range(period - 1, 0, -1):
            km = k - 1
            rows = <int>r[k]
            prev_rows = <int>r[km]
            info = 0
            lapack.dgeqrf(
                &rows, &n, &F[0, 0, k], &capacity, &tau[0],
                &work[0], &lwork, &info,
            )
            if info != 0:
                failure = 1
                break
            lapack.dormqr(
                &side, &trans, &prev_rows, &rows, &n,
                &F[0, 0, k], &capacity, &tau[0],
                &F[0, 0, km], &capacity, &work[0], &lwork, &info,
            )
            if info != 0:
                failure = 2
                break
            lapack.dormqr(
                &side, &trans, &basis_rows, &rows, &n,
                &F[0, 0, k], &capacity, &tau[0],
                &U[0, 0, k], &basis_rows, &work[0], &lwork, &info,
            )
            if info != 0:
                failure = 3
                break

            for j in range(n):
                for i in range(j + 1, rows):
                    F[i, j, k] = 0.0
            for j in range(n, rows):
                for i in range(prev_rows):
                    F[i, j, km] = 0.0
                for i in range(basis_rows):
                    U[i, j, k] = 0.0
            r[k] = n

    if failure:
        raise np.linalg.LinAlgError(
            f"full QR sweep failed at factor {k}, stage {failure}, info={info}"
        )
    return factors_arr, bases_arr, ranks_arr


def _full_qr_sweep_z(factors, bases, ranks):
    r"""Apply :func:`_full_qr_sweep_d` to complex128 packed factors in place."""
    cdef tuple arrays = _full_qr_inputs(factors, bases, ranks, np.complex128)
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef ZTYPE_t[::1, :, :] F = factors_arr
    cdef ZTYPE_t[::1, :, :] U = bases_arr
    cdef Py_ssize_t[::1] r = ranks_arr
    cdef int capacity = factors_arr.shape[0]
    cdef int period = factors_arr.shape[2]
    cdef int basis_rows = bases_arr.shape[0]
    cdef int n = r[0]
    cdef cnp.ndarray tau_arr
    cdef cnp.ndarray query_arr
    cdef cnp.ndarray work_arr
    cdef ZTYPE_t[::1] tau
    cdef ZTYPE_t[::1] query
    cdef ZTYPE_t[::1] work
    cdef char side = 'R'
    cdef char trans = 'N'
    cdef int lwork = -1
    cdef int info = 0
    cdef int query_m, query_n, query_k
    cdef int qr_work, factor_work, basis_work
    cdef int k, km, rows, prev_rows, i, j, failure = 0

    if n == 0:
        ranks_arr[:] = 0
        return factors_arr, bases_arr, ranks_arr
    if period == 1:
        return factors_arr, bases_arr, ranks_arr

    tau_arr = np.empty(n, dtype=np.complex128)
    query_arr = np.empty(1, dtype=np.complex128)
    tau = tau_arr
    query = query_arr
    query_m = capacity
    query_n = n
    query_k = n

    with nogil:
        lapack.zgeqrf(
            &query_m, &query_n, &F[0, 0, 0], &capacity, &tau[0],
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"ZGEQRF workspace query failed with info={info}")
    qr_work = max(1, <int>query[0].real)

    query_m = capacity
    query_n = capacity
    with nogil:
        lapack.zunmqr(
            &side, &trans, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &F[0, 0, 0], &capacity,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"ZUNMQR factor workspace query failed with info={info}")
    factor_work = max(1, <int>query[0].real)

    query_m = basis_rows
    query_n = capacity
    with nogil:
        lapack.zunmqr(
            &side, &trans, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &U[0, 0, 0], &basis_rows,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"ZUNMQR basis workspace query failed with info={info}")
    basis_work = max(1, <int>query[0].real)

    lwork = max(qr_work, factor_work, basis_work)
    work_arr = np.empty(lwork, dtype=np.complex128)
    work = work_arr

    with nogil:
        for k in range(period - 1, 0, -1):
            km = k - 1
            rows = <int>r[k]
            prev_rows = <int>r[km]
            info = 0
            lapack.zgeqrf(
                &rows, &n, &F[0, 0, k], &capacity, &tau[0],
                &work[0], &lwork, &info,
            )
            if info != 0:
                failure = 1
                break
            lapack.zunmqr(
                &side, &trans, &prev_rows, &rows, &n,
                &F[0, 0, k], &capacity, &tau[0],
                &F[0, 0, km], &capacity, &work[0], &lwork, &info,
            )
            if info != 0:
                failure = 2
                break
            lapack.zunmqr(
                &side, &trans, &basis_rows, &rows, &n,
                &F[0, 0, k], &capacity, &tau[0],
                &U[0, 0, k], &basis_rows, &work[0], &lwork, &info,
            )
            if info != 0:
                failure = 3
                break

            for j in range(n):
                for i in range(j + 1, rows):
                    F[i, j, k] = 0.0
            for j in range(n, rows):
                for i in range(prev_rows):
                    F[i, j, km] = 0.0
                for i in range(basis_rows):
                    U[i, j, k] = 0.0
            r[k] = n

    if failure:
        raise np.linalg.LinAlgError(
            f"full QR sweep failed at factor {k}, stage {failure}, info={info}"
        )
    return factors_arr, bases_arr, ranks_arr


cdef inline void _left_real_f(DTYPE_t[::1, :] A, Py_ssize_t p, Py_ssize_t q,
                              double c, double s, Py_ssize_t col_start,
                              Py_ssize_t col_stop) noexcept nogil:
    """Apply a real Givens rotation to two rows of a Fortran matrix."""
    cdef Py_ssize_t j
    cdef double xp, xq
    for j in range(col_start, col_stop):
        xp = A[p, j]
        xq = A[q, j]
        A[p, j] = c * xp + s * xq
        A[q, j] = -s * xp + c * xq


cdef inline void _right_adj_real_f(DTYPE_t[::1, :] A, Py_ssize_t p,
                                   Py_ssize_t q, double c, double s,
                                   Py_ssize_t row_stop) noexcept nogil:
    """Apply a real adjoint Givens rotation to two Fortran-matrix columns."""
    cdef Py_ssize_t i
    cdef double xp, xq
    for i in range(row_stop):
        xp = A[i, p]
        xq = A[i, q]
        A[i, p] = c * xp + s * xq
        A[i, q] = -s * xp + c * xq


cdef inline void _left_complex_f(ZTYPE_t[::1, :] A, Py_ssize_t p,
                                 Py_ssize_t q, double c, double complex s,
                                 Py_ssize_t col_start,
                                 Py_ssize_t col_stop) noexcept nogil:
    """Apply a complex Givens rotation to two rows of a Fortran matrix."""
    cdef Py_ssize_t j
    cdef double complex xp, xq
    for j in range(col_start, col_stop):
        xp = A[p, j]
        xq = A[q, j]
        A[p, j] = c * xp + s * xq
        A[q, j] = -s.conjugate() * xp + c * xq


cdef inline void _right_adj_complex_f(ZTYPE_t[::1, :] A, Py_ssize_t p,
                                      Py_ssize_t q, double c,
                                      double complex s,
                                      Py_ssize_t row_stop) noexcept nogil:
    """Apply a complex adjoint Givens rotation to Fortran-matrix columns."""
    cdef Py_ssize_t i
    cdef double complex xp, xq
    for i in range(row_stop):
        xp = A[i, p]
        xq = A[i, q]
        A[i, p] = c * xp + s.conjugate() * xq
        A[i, q] = -s * xp + c * xq


cdef void _hessenberg_chase_d_views(DTYPE_t[::1, :, :] factors,
                                    DTYPE_t[::1, :, :] bases,
                                    Py_ssize_t n) noexcept:
    """Reduce factor zero to Hessenberg while restoring later triangles."""
    cdef Py_ssize_t period = factors.shape[2]
    cdef Py_ssize_t basis_rows = bases.shape[0]
    cdef Py_ssize_t j, i, k
    cdef DTYPE_t[::1, :] A0 = factors[:, :, 0]
    cdef DTYPE_t[::1, :] Ak
    cdef DTYPE_t[::1, :] Uk
    cdef double c, s, value
    cdef cnp.ndarray gp_arr = np.empty(n, dtype=np.intp)
    cdef cnp.ndarray gq_arr = np.empty(n, dtype=np.intp)
    cdef cnp.ndarray gc_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray gs_arr = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t[::1] gp = gp_arr
    cdef Py_ssize_t[::1] gq = gq_arr
    cdef DTYPE_t[::1] gc = gc_arr
    cdef DTYPE_t[::1] gs = gs_arr

    with nogil:
        for j in range(n - 2):
            for i in range(n - 1, j + 1, -1):
                lapack.dlartg(&A0[i - 1, j], &A0[i, j], &c, &s, &value)
                A0[i - 1, j] = value
                A0[i, j] = 0.0
                _left_real_f(A0, i - 1, i, c, s, j + 1, n)
                Uk = bases[:, :, 0]
                _right_adj_real_f(Uk, i - 1, i, c, s, basis_rows)
                gp[i] = i - 1
                gq[i] = i
                gc[i] = c
                gs[i] = s

            for k in range(period - 1, 0, -1):
                Ak = factors[:, :, k]
                for i in range(n - 1, j + 1, -1):
                    _right_adj_real_f(
                        Ak, gp[i], gq[i], gc[i], gs[i], i + 1
                    )
                    lapack.dlartg(
                        &Ak[i - 1, i - 1], &Ak[i, i - 1], &c, &s, &value
                    )
                    Ak[i - 1, i - 1] = value
                    Ak[i, i - 1] = 0.0
                    _left_real_f(Ak, i - 1, i, c, s, i, n)
                    gp[i] = i - 1
                    gq[i] = i
                    gc[i] = c
                    gs[i] = s

                Uk = bases[:, :, k]
                for i in range(n - 1, j + 1, -1):
                    _right_adj_real_f(
                        Uk, gp[i], gq[i], gc[i], gs[i], basis_rows
                    )

            for i in range(n - 1, j + 1, -1):
                _right_adj_real_f(A0, gp[i], gq[i], gc[i], gs[i], n)


cdef void _hessenberg_chase_z_views(ZTYPE_t[::1, :, :] factors,
                                    ZTYPE_t[::1, :, :] bases,
                                    Py_ssize_t n) noexcept:
    """Apply the complex periodic Hessenberg-triangular Givens chase."""
    cdef Py_ssize_t period = factors.shape[2]
    cdef Py_ssize_t basis_rows = bases.shape[0]
    cdef Py_ssize_t j, i, k
    cdef ZTYPE_t[::1, :] A0 = factors[:, :, 0]
    cdef ZTYPE_t[::1, :] Ak
    cdef ZTYPE_t[::1, :] Uk
    cdef double c
    cdef double complex s, value
    cdef cnp.ndarray gp_arr = np.empty(n, dtype=np.intp)
    cdef cnp.ndarray gq_arr = np.empty(n, dtype=np.intp)
    cdef cnp.ndarray gc_arr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray gs_arr = np.empty(n, dtype=np.complex128)
    cdef Py_ssize_t[::1] gp = gp_arr
    cdef Py_ssize_t[::1] gq = gq_arr
    cdef DTYPE_t[::1] gc = gc_arr
    cdef ZTYPE_t[::1] gs = gs_arr

    with nogil:
        for j in range(n - 2):
            for i in range(n - 1, j + 1, -1):
                lapack.zlartg(&A0[i - 1, j], &A0[i, j], &c, &s, &value)
                A0[i - 1, j] = value
                A0[i, j] = 0.0
                _left_complex_f(A0, i - 1, i, c, s, j + 1, n)
                Uk = bases[:, :, 0]
                _right_adj_complex_f(Uk, i - 1, i, c, s, basis_rows)
                gp[i] = i - 1
                gq[i] = i
                gc[i] = c
                gs[i] = s

            for k in range(period - 1, 0, -1):
                Ak = factors[:, :, k]
                for i in range(n - 1, j + 1, -1):
                    _right_adj_complex_f(
                        Ak, gp[i], gq[i], gc[i], gs[i], i + 1
                    )
                    lapack.zlartg(
                        &Ak[i - 1, i - 1], &Ak[i, i - 1], &c, &s, &value
                    )
                    Ak[i - 1, i - 1] = value
                    Ak[i, i - 1] = 0.0
                    _left_complex_f(Ak, i - 1, i, c, s, i, n)
                    gp[i] = i - 1
                    gq[i] = i
                    gc[i] = c
                    gs[i] = s

                Uk = bases[:, :, k]
                for i in range(n - 1, j + 1, -1):
                    _right_adj_complex_f(
                        Uk, gp[i], gq[i], gc[i], gs[i], basis_rows
                    )

            for i in range(n - 1, j + 1, -1):
                _right_adj_complex_f(A0, gp[i], gq[i], gc[i], gs[i], n)


def _hessenberg_chase_d(factors, bases, ranks):
    r"""Reduce packed FP64 square factors to periodic Hessenberg-triangular form."""
    cdef tuple arrays = _full_qr_inputs(factors, bases, ranks, np.float64)
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef Py_ssize_t n = ranks_arr[0]
    cdef DTYPE_t[::1, :, :] F = factors_arr
    cdef DTYPE_t[::1, :, :] U = bases_arr

    if np.any(ranks_arr != n):
        raise ValueError("all ranks must be equal before the Hessenberg chase")
    if n > 2:
        _hessenberg_chase_d_views(F, U, n)
    return factors_arr, bases_arr, ranks_arr


def _hessenberg_chase_z(factors, bases, ranks):
    r"""Reduce packed complex128 factors to periodic Hessenberg-triangular form."""
    cdef tuple arrays = _full_qr_inputs(factors, bases, ranks, np.complex128)
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef Py_ssize_t n = ranks_arr[0]
    cdef ZTYPE_t[::1, :, :] F = factors_arr
    cdef ZTYPE_t[::1, :, :] U = bases_arr

    if np.any(ranks_arr != n):
        raise ValueError("all ranks must be equal before the Hessenberg chase")
    if n > 2:
        _hessenberg_chase_z_views(F, U, n)
    return factors_arr, bases_arr, ranks_arr




"""Block-panel QR implementation section for ``periodic_schur.pyx``."""


cdef tuple _panel_qr_inputs(object factors, object bases, object ranks,
                            object block_ranks, object dtype):
    """Validate mutable Fortran workspaces and block-cut metadata."""
    cdef cnp.ndarray factors_arr = np.asarray(factors)
    cdef cnp.ndarray bases_arr = np.asarray(bases)
    cdef cnp.ndarray ranks_arr = np.asarray(ranks)
    cdef cnp.ndarray block_ranks_arr = np.asarray(block_ranks)

    if factors_arr.dtype != np.dtype(dtype) or bases_arr.dtype != np.dtype(dtype):
        raise TypeError(f"factors and bases must have dtype {np.dtype(dtype)}")
    if (factors_arr.ndim != 3 or
            factors_arr.shape[0] != factors_arr.shape[1] or
            factors_arr.shape[2] < 1):
        raise ValueError("factors must have shape (capacity, capacity, period)")
    if not factors_arr.flags.f_contiguous:
        raise ValueError("factors must use Fortran storage")
    if (bases_arr.ndim != 3 or
            bases_arr.shape[1] != factors_arr.shape[0] or
            bases_arr.shape[2] != factors_arr.shape[2]):
        raise ValueError("bases must have shape (m, capacity, period)")
    if not bases_arr.flags.f_contiguous:
        raise ValueError("bases must use Fortran storage")
    if (ranks_arr.dtype != np.dtype(np.intp) or
            ranks_arr.ndim != 1 or
            ranks_arr.shape[0] != factors_arr.shape[2] or
            not ranks_arr.flags.c_contiguous):
        raise ValueError("ranks must be a contiguous intp vector of length period")
    if (block_ranks_arr.dtype != np.dtype(np.intp) or
            block_ranks_arr.ndim != 2 or
            block_ranks_arr.shape[0] != factors_arr.shape[2] or
            not block_ranks_arr.flags.c_contiguous):
        raise ValueError("block_ranks must be a contiguous (period, depth) intp array")
    if np.any(ranks_arr < 0) or np.any(ranks_arr > factors_arr.shape[0]):
        raise ValueError("ranks must lie within the factor capacity")
    if np.any(ranks_arr < ranks_arr[0]):
        raise ValueError("cut zero must have minimum rank")
    if np.any(block_ranks_arr < 0):
        raise ValueError("block_ranks must be nonnegative")
    if np.any(np.sum(block_ranks_arr, axis=1) != ranks_arr):
        raise ValueError("each block_ranks row must sum to its cut rank")
    return factors_arr, bases_arr, ranks_arr, block_ranks_arr


def _panel_qr_sweep_d(factors, bases, ranks, block_ranks):
    r"""Reduce FP64 block-Hessenberg factors by a reverse panel QR sweep.

    The operation is in place and has the same factor/basis/rank contract as
    ``_full_qr_sweep_d``.  Input cut zero must have minimum rank.  Factor ``k``
    initially has block-Hessenberg lower block bandwidth one.  Absorbing the
    QR of factor ``k+1`` increases the bandwidth of factor ``k`` by one, so
    factor ``k`` is reduced with panels whose active rows extend through block
    ``j + period - k``.

    On return all cuts have rank ``ranks[0]``.  Rows of ``block_ranks`` are
    overwritten by the original cut-zero row: retained QR coordinates inherit
    the column partition of each factor, recursively propagating the cut-zero
    partition left through the cycle.  Thus output ``block_ranks[k, j]`` is the
    size of reduced Arnoldi panel ``j`` at cut ``k``; it is no longer the count
    of input active vectors at that cut.
    """
    cdef tuple arrays = _panel_qr_inputs(
        factors, bases, ranks, block_ranks, np.float64
    )
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef cnp.ndarray block_ranks_arr = arrays[3]
    cdef DTYPE_t[::1, :, :] F = factors_arr
    cdef DTYPE_t[::1, :, :] U = bases_arr
    cdef Py_ssize_t[::1] r = ranks_arr
    cdef Py_ssize_t[:, ::1] br = block_ranks_arr
    cdef int capacity = factors_arr.shape[0]
    cdef int period = factors_arr.shape[2]
    cdef int depth = block_ranks_arr.shape[1]
    cdef int basis_rows = bases_arr.shape[0]
    cdef int n = r[0]
    cdef int max_panel = 0
    cdef cnp.ndarray tau_arr
    cdef cnp.ndarray query_arr
    cdef cnp.ndarray work_arr
    cdef DTYPE_t[::1] tau
    cdef DTYPE_t[::1] query
    cdef DTYPE_t[::1] work
    cdef char side_l = 'L'
    cdef char side_r = 'R'
    cdef char trans_t = 'T'
    cdef char trans_n = 'N'
    cdef int lwork = -1
    cdef int info = 0
    cdef int qr_work, left_work, factor_work, basis_work
    cdef int query_m, query_n, query_k
    cdef int k, km, j, t, i, col, rows, prev_rows, bandwidth
    cdef int col_start, col_stop, row_stop, structural_stop
    cdef int panel_rows, panel_cols, trail_cols, failure = 0

    if n == 0:
        ranks_arr[:] = 0
        block_ranks_arr[:] = 0
        return factors_arr, bases_arr, ranks_arr, block_ranks_arr
    if period == 1:
        return factors_arr, bases_arr, ranks_arr, block_ranks_arr

    for j in range(depth):
        if br[0, j] > max_panel:
            max_panel = <int>br[0, j]
    if max_panel == 0:
        return factors_arr, bases_arr, ranks_arr, block_ranks_arr

    tau_arr = np.empty(max_panel, dtype=np.float64)
    query_arr = np.empty(1, dtype=np.float64)
    tau = tau_arr
    query = query_arr
    query_m = capacity
    query_n = max_panel
    query_k = max_panel

    with nogil:
        lapack.dgeqrf(
            &query_m, &query_n, &F[0, 0, 0], &capacity, &tau[0],
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"DGEQRF workspace query failed with info={info}")
    qr_work = max(1, <int>query[0])

    query_m = capacity
    query_n = capacity
    with nogil:
        lapack.dormqr(
            &side_l, &trans_t, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &F[0, 0, 0], &capacity,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"DORMQR left workspace query failed with info={info}")
    left_work = max(1, <int>query[0])

    with nogil:
        lapack.dormqr(
            &side_r, &trans_n, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &F[0, 0, 0], &capacity,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"DORMQR factor workspace query failed with info={info}")
    factor_work = max(1, <int>query[0])

    query_m = basis_rows
    query_n = capacity
    with nogil:
        lapack.dormqr(
            &side_r, &trans_n, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &U[0, 0, 0], &basis_rows,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"DORMQR basis workspace query failed with info={info}")
    basis_work = max(1, <int>query[0])

    lwork = max(qr_work, left_work, factor_work, basis_work)
    work_arr = np.empty(lwork, dtype=np.float64)
    work = work_arr

    with nogil:
        for k in range(period - 1, 0, -1):
            km = k - 1
            rows = <int>r[k]
            prev_rows = <int>r[km]
            bandwidth = period - k
            col_start = 0
            structural_stop = 0

            for j in range(depth):
                panel_cols = <int>br[0, j]
                col_stop = col_start + panel_cols

                # The untransformed row blocks through j+bandwidth support
                # this panel.  Earlier panel Qs occupy rows below col_start.
                structural_stop = 0
                for t in range(depth):
                    if t <= j + bandwidth:
                        structural_stop += <int>br[k, t]
                row_stop = structural_stop
                if row_stop < col_stop:
                    row_stop = col_stop
                if row_stop > rows:
                    row_stop = rows

                if panel_cols == 0:
                    col_start = col_stop
                    continue

                panel_rows = row_stop - col_start
                trail_cols = n - col_stop
                info = 0
                lapack.dgeqrf(
                    &panel_rows, &panel_cols, &F[col_start, col_start, k],
                    &capacity, &tau[0], &work[0], &lwork, &info,
                )
                if info != 0:
                    failure = 1
                    break
                if trail_cols:
                    lapack.dormqr(
                        &side_l, &trans_t, &panel_rows, &trail_cols, &panel_cols,
                        &F[col_start, col_start, k], &capacity, &tau[0],
                        &F[col_start, col_stop, k], &capacity,
                        &work[0], &lwork, &info,
                    )
                    if info != 0:
                        failure = 2
                        break
                lapack.dormqr(
                    &side_r, &trans_n, &prev_rows, &panel_rows, &panel_cols,
                    &F[col_start, col_start, k], &capacity, &tau[0],
                    &F[0, col_start, km], &capacity,
                    &work[0], &lwork, &info,
                )
                if info != 0:
                    failure = 3
                    break
                lapack.dormqr(
                    &side_r, &trans_n, &basis_rows, &panel_rows, &panel_cols,
                    &F[col_start, col_start, k], &capacity, &tau[0],
                    &U[0, col_start, k], &basis_rows,
                    &work[0], &lwork, &info,
                )
                if info != 0:
                    failure = 4
                    break

                for col in range(col_start, col_stop):
                    for i in range(col + 1, row_stop):
                        F[i, col, k] = 0.0
                col_start = col_stop

            if failure:
                break

            for col in range(n):
                for i in range(col + 1, rows):
                    F[i, col, k] = 0.0
            for col in range(n, rows):
                for i in range(prev_rows):
                    F[i, col, km] = 0.0
                for i in range(basis_rows):
                    U[i, col, k] = 0.0
            r[k] = n
            for j in range(depth):
                br[k, j] = br[0, j]

    if failure:
        raise np.linalg.LinAlgError(
            f"panel QR sweep failed at factor {k}, panel {j}, "
            f"stage {failure}, info={info}"
        )
    return factors_arr, bases_arr, ranks_arr, block_ranks_arr


def _panel_qr_sweep_z(factors, bases, ranks, block_ranks):
    r"""Apply :func:`_panel_qr_sweep_d` to complex128 factors in place."""
    cdef tuple arrays = _panel_qr_inputs(
        factors, bases, ranks, block_ranks, np.complex128
    )
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef cnp.ndarray block_ranks_arr = arrays[3]
    cdef ZTYPE_t[::1, :, :] F = factors_arr
    cdef ZTYPE_t[::1, :, :] U = bases_arr
    cdef Py_ssize_t[::1] r = ranks_arr
    cdef Py_ssize_t[:, ::1] br = block_ranks_arr
    cdef int capacity = factors_arr.shape[0]
    cdef int period = factors_arr.shape[2]
    cdef int depth = block_ranks_arr.shape[1]
    cdef int basis_rows = bases_arr.shape[0]
    cdef int n = r[0]
    cdef int max_panel = 0
    cdef cnp.ndarray tau_arr
    cdef cnp.ndarray query_arr
    cdef cnp.ndarray work_arr
    cdef ZTYPE_t[::1] tau
    cdef ZTYPE_t[::1] query
    cdef ZTYPE_t[::1] work
    cdef char side_l = 'L'
    cdef char side_r = 'R'
    cdef char trans_c = 'C'
    cdef char trans_n = 'N'
    cdef int lwork = -1
    cdef int info = 0
    cdef int qr_work, left_work, factor_work, basis_work
    cdef int query_m, query_n, query_k
    cdef int k, km, j, t, i, col, rows, prev_rows, bandwidth
    cdef int col_start, col_stop, row_stop, structural_stop
    cdef int panel_rows, panel_cols, trail_cols, failure = 0

    if n == 0:
        ranks_arr[:] = 0
        block_ranks_arr[:] = 0
        return factors_arr, bases_arr, ranks_arr, block_ranks_arr
    if period == 1:
        return factors_arr, bases_arr, ranks_arr, block_ranks_arr

    for j in range(depth):
        if br[0, j] > max_panel:
            max_panel = <int>br[0, j]
    if max_panel == 0:
        return factors_arr, bases_arr, ranks_arr, block_ranks_arr

    tau_arr = np.empty(max_panel, dtype=np.complex128)
    query_arr = np.empty(1, dtype=np.complex128)
    tau = tau_arr
    query = query_arr
    query_m = capacity
    query_n = max_panel
    query_k = max_panel

    with nogil:
        lapack.zgeqrf(
            &query_m, &query_n, &F[0, 0, 0], &capacity, &tau[0],
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"ZGEQRF workspace query failed with info={info}")
    qr_work = max(1, <int>query[0].real)

    query_m = capacity
    query_n = capacity
    with nogil:
        lapack.zunmqr(
            &side_l, &trans_c, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &F[0, 0, 0], &capacity,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"ZUNMQR left workspace query failed with info={info}")
    left_work = max(1, <int>query[0].real)

    with nogil:
        lapack.zunmqr(
            &side_r, &trans_n, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &F[0, 0, 0], &capacity,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"ZUNMQR factor workspace query failed with info={info}")
    factor_work = max(1, <int>query[0].real)

    query_m = basis_rows
    query_n = capacity
    with nogil:
        lapack.zunmqr(
            &side_r, &trans_n, &query_m, &query_n, &query_k,
            &F[0, 0, 0], &capacity, &tau[0], &U[0, 0, 0], &basis_rows,
            &query[0], &lwork, &info,
        )
    if info != 0:
        raise np.linalg.LinAlgError(f"ZUNMQR basis workspace query failed with info={info}")
    basis_work = max(1, <int>query[0].real)

    lwork = max(qr_work, left_work, factor_work, basis_work)
    work_arr = np.empty(lwork, dtype=np.complex128)
    work = work_arr

    with nogil:
        for k in range(period - 1, 0, -1):
            km = k - 1
            rows = <int>r[k]
            prev_rows = <int>r[km]
            bandwidth = period - k
            col_start = 0

            for j in range(depth):
                panel_cols = <int>br[0, j]
                col_stop = col_start + panel_cols
                structural_stop = 0
                for t in range(depth):
                    if t <= j + bandwidth:
                        structural_stop += <int>br[k, t]
                row_stop = structural_stop
                if row_stop < col_stop:
                    row_stop = col_stop
                if row_stop > rows:
                    row_stop = rows

                if panel_cols == 0:
                    col_start = col_stop
                    continue

                panel_rows = row_stop - col_start
                trail_cols = n - col_stop
                info = 0
                lapack.zgeqrf(
                    &panel_rows, &panel_cols, &F[col_start, col_start, k],
                    &capacity, &tau[0], &work[0], &lwork, &info,
                )
                if info != 0:
                    failure = 1
                    break
                if trail_cols:
                    lapack.zunmqr(
                        &side_l, &trans_c, &panel_rows, &trail_cols, &panel_cols,
                        &F[col_start, col_start, k], &capacity, &tau[0],
                        &F[col_start, col_stop, k], &capacity,
                        &work[0], &lwork, &info,
                    )
                    if info != 0:
                        failure = 2
                        break
                lapack.zunmqr(
                    &side_r, &trans_n, &prev_rows, &panel_rows, &panel_cols,
                    &F[col_start, col_start, k], &capacity, &tau[0],
                    &F[0, col_start, km], &capacity,
                    &work[0], &lwork, &info,
                )
                if info != 0:
                    failure = 3
                    break
                lapack.zunmqr(
                    &side_r, &trans_n, &basis_rows, &panel_rows, &panel_cols,
                    &F[col_start, col_start, k], &capacity, &tau[0],
                    &U[0, col_start, k], &basis_rows,
                    &work[0], &lwork, &info,
                )
                if info != 0:
                    failure = 4
                    break

                for col in range(col_start, col_stop):
                    for i in range(col + 1, row_stop):
                        F[i, col, k] = 0.0
                col_start = col_stop

            if failure:
                break

            for col in range(n):
                for i in range(col + 1, rows):
                    F[i, col, k] = 0.0
            for col in range(n, rows):
                for i in range(prev_rows):
                    F[i, col, km] = 0.0
                for i in range(basis_rows):
                    U[i, col, k] = 0.0
            r[k] = n
            for j in range(depth):
                br[k, j] = br[0, j]

    if failure:
        raise np.linalg.LinAlgError(
            f"panel QR sweep failed at factor {k}, panel {j}, "
            f"stage {failure}, info={info}"
        )
    return factors_arr, bases_arr, ranks_arr, block_ranks_arr


"""Fused Householder implementation section for ``periodic_schur.pyx``."""

# TODO: Retain reflector panels and apply them to the physical bases in a
# blocked Level-3 pass instead of the per-reflector xLARF updates below.


cdef tuple _householder_inputs(object factors, object bases, object ranks,
                               object block_ranks, object dtype):
    """Validate compact mutable trailing-period Fortran workspaces."""
    cdef cnp.ndarray factors_arr = np.asarray(factors)
    cdef cnp.ndarray bases_arr = np.asarray(bases)
    cdef cnp.ndarray ranks_arr = np.asarray(ranks)
    cdef cnp.ndarray block_ranks_arr = np.asarray(block_ranks)

    if factors_arr.dtype != np.dtype(dtype) or bases_arr.dtype != np.dtype(dtype):
        raise TypeError(f"factors and bases must have dtype {np.dtype(dtype)}")
    if (factors_arr.ndim != 3 or
            factors_arr.shape[0] != factors_arr.shape[1] or
            factors_arr.shape[2] < 1):
        raise ValueError("factors must have shape (capacity, capacity, period)")
    if not factors_arr.flags.f_contiguous:
        raise ValueError("factors must use Fortran storage")
    if (bases_arr.ndim != 3 or
            bases_arr.shape[1] != factors_arr.shape[0] or
            bases_arr.shape[2] != factors_arr.shape[2]):
        raise ValueError("bases must have shape (m, capacity, period)")
    if not bases_arr.flags.f_contiguous:
        raise ValueError("bases must use Fortran storage")
    if (ranks_arr.dtype != np.dtype(np.intp) or
            ranks_arr.ndim != 1 or
            ranks_arr.shape[0] != factors_arr.shape[2] or
            not ranks_arr.flags.c_contiguous):
        raise ValueError("ranks must be a contiguous intp vector of length period")
    if (block_ranks_arr.dtype != np.dtype(np.intp) or
            block_ranks_arr.ndim != 2 or
            block_ranks_arr.shape[0] != factors_arr.shape[2] or
            not block_ranks_arr.flags.c_contiguous):
        raise ValueError("block_ranks must be a contiguous (period, depth) intp array")
    if np.any(ranks_arr < 0) or np.any(ranks_arr > factors_arr.shape[0]):
        raise ValueError("ranks must lie within the factor capacity")
    if np.any(ranks_arr < ranks_arr[0]):
        raise ValueError("cut zero must have minimum rank")
    if np.any(block_ranks_arr < 0):
        raise ValueError("block_ranks must be nonnegative")
    if np.any(np.sum(block_ranks_arr, axis=1) != ranks_arr):
        raise ValueError("each block_ranks row must sum to its cut rank")
    return factors_arr, bases_arr, ranks_arr, block_ranks_arr


cdef cnp.ndarray _num_rows_input(object num_rows, cnp.ndarray factors,
                                 cnp.ndarray ranks):
    """Validate exclusive structural row bounds for every stored column."""
    cdef cnp.ndarray num_rows_arr = np.asarray(num_rows)
    cdef Py_ssize_t[:, ::1] nr
    cdef Py_ssize_t[::1] r = ranks
    cdef Py_ssize_t capacity = factors.shape[0]
    cdef Py_ssize_t period = factors.shape[2]
    cdef Py_ssize_t k, kp, col

    if (num_rows_arr.dtype != np.dtype(np.intp) or
            num_rows_arr.ndim != 2 or
            num_rows_arr.shape[0] != capacity or
            num_rows_arr.shape[1] != period or
            not num_rows_arr.flags.c_contiguous):
        raise ValueError(
            "num_rows must be a contiguous (capacity, period) intp array"
        )
    nr = num_rows_arr
    for k in range(period):
        kp = (k + 1) % period
        for col in range(r[kp]):
            if nr[col, k] < 0 or nr[col, k] > r[k]:
                raise ValueError("num_rows bounds must lie within each logical factor")
    return num_rows_arr


cdef inline void _right_num_rows(Py_ssize_t[:, ::1] nr, Py_ssize_t k,
                                 Py_ssize_t start, Py_ssize_t stop) noexcept nogil:
    """Propagate support when a dense reflector mixes a column interval."""
    cdef Py_ssize_t col, support = 0
    for col in range(start, stop):
        if nr[col, k] > support:
            support = nr[col, k]
    for col in range(start, stop):
        nr[col, k] = support


cdef inline void _left_num_rows(Py_ssize_t[:, ::1] nr, Py_ssize_t k,
                                Py_ssize_t start, Py_ssize_t stop,
                                Py_ssize_t col_start,
                                Py_ssize_t n) noexcept nogil:
    """Propagate support when a reflector mixes a bounded row interval."""
    cdef Py_ssize_t col
    for col in range(col_start, n):
        if nr[col, k] > start and nr[col, k] < stop:
            nr[col, k] = stop


def make_periodic_hessenberg_HOUSEHOLDER_D(factors, bases, ranks, block_ranks):
    r"""Prepare FP64 compact factors by a fused cyclic Householder sweep.

    Factor ``k`` initially has logical shape ``(ranks[k], ranks[k+1])``
    in formal SLICOT product order.  Cut zero must have minimum rank ``n``.
    A reverse thin-QR pass is performed only at cuts whose rank exceeds
    ``n``.  The following MB03VD-style sweep makes factors ``1:p`` upper
    triangular and factor zero upper Hessenberg while preserving

    ``C[k] @ bases[k+1] = bases[k] @ factors[k]``.

    ``block_ranks`` is validated and retained as input active-coordinate
    provenance; this first kernel treats every resulting live square factor
    as structurally general.
    """
    cdef tuple arrays = _householder_inputs(
        factors, bases, ranks, block_ranks, np.float64
    )
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef cnp.ndarray block_ranks_arr = arrays[3]
    cdef DTYPE_t[::1, :, :] F = factors_arr
    cdef DTYPE_t[::1, :, :] U = bases_arr
    cdef Py_ssize_t[::1] r = ranks_arr
    cdef int capacity = factors_arr.shape[0]
    cdef int period = factors_arr.shape[2]
    cdef int basis_rows = bases_arr.shape[0]
    cdef int n = r[0]
    cdef bint needs_qr = bool(np.any(ranks_arr > n))
    cdef cnp.ndarray tau_qr_arr
    cdef cnp.ndarray query_arr
    cdef cnp.ndarray work_qr_arr
    cdef cnp.ndarray work_h_arr
    cdef DTYPE_t[::1] tau_qr
    cdef DTYPE_t[::1] query
    cdef DTYPE_t[::1] work_qr
    cdef DTYPE_t[::1] work_h
    cdef char side_l = 'L'
    cdef char side_r = 'R'
    cdef char trans_n = 'N'
    cdef int lwork = -1
    cdef int info = 0
    cdef int query_m, query_n, query_k
    cdef int qr_work, factor_work, basis_work
    cdef int k, km, i, row, rows, prev_rows, tail_row, failure = 0
    cdef int reflector_len, trail_cols, one = 1
    cdef double tau, beta

    if n == 0:
        ranks_arr[:] = 0
        return factors_arr, bases_arr, ranks_arr, block_ranks_arr

    if needs_qr:
        tau_qr_arr = np.empty(n, dtype=np.float64)
        query_arr = np.empty(1, dtype=np.float64)
        tau_qr = tau_qr_arr
        query = query_arr
        query_m = capacity
        query_n = n
        query_k = n

        with nogil:
            lapack.dgeqrf(
                &query_m, &query_n, &F[0, 0, 0], &capacity, &tau_qr[0],
                &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"DGEQRF workspace query failed with info={info}"
            )
        qr_work = max(1, <int>query[0])

        query_m = capacity
        query_n = capacity
        with nogil:
            lapack.dormqr(
                &side_r, &trans_n, &query_m, &query_n, &query_k,
                &F[0, 0, 0], &capacity, &tau_qr[0],
                &F[0, 0, 0], &capacity, &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"DORMQR factor workspace query failed with info={info}"
            )
        factor_work = max(1, <int>query[0])

        query_m = basis_rows
        with nogil:
            lapack.dormqr(
                &side_r, &trans_n, &query_m, &query_n, &query_k,
                &F[0, 0, 0], &capacity, &tau_qr[0],
                &U[0, 0, 0], &basis_rows, &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"DORMQR basis workspace query failed with info={info}"
            )
        basis_work = max(1, <int>query[0])
        lwork = max(qr_work, factor_work, basis_work)
        work_qr_arr = np.empty(lwork, dtype=np.float64)
        work_qr = work_qr_arr

        with nogil:
            for k in range(period - 1, 0, -1):
                rows = <int>r[k]
                if rows == n:
                    continue
                km = k - 1
                prev_rows = <int>r[km]
                info = 0
                lapack.dgeqrf(
                    &rows, &n, &F[0, 0, k], &capacity, &tau_qr[0],
                    &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 1
                    break
                lapack.dormqr(
                    &side_r, &trans_n, &prev_rows, &rows, &n,
                    &F[0, 0, k], &capacity, &tau_qr[0],
                    &F[0, 0, km], &capacity, &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 2
                    break
                lapack.dormqr(
                    &side_r, &trans_n, &basis_rows, &rows, &n,
                    &F[0, 0, k], &capacity, &tau_qr[0],
                    &U[0, 0, k], &basis_rows, &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 3
                    break

                for i in range(n):
                    for row in range(i + 1, rows):
                        F[row, i, k] = 0.0
                for i in range(n, rows):
                    for row in range(prev_rows):
                        F[row, i, km] = 0.0
                    for row in range(basis_rows):
                        U[row, i, k] = 0.0
                r[k] = n

        if failure:
            raise np.linalg.LinAlgError(
                f"selective QR failed at factor {k}, stage {failure}, info={info}"
            )

    work_h_arr = np.empty(max(n, basis_rows), dtype=np.float64)
    work_h = work_h_arr
    with nogil:
        for i in range(n - 1):
            trail_cols = n - i - 1
            for k in range(period - 1, 0, -1):
                reflector_len = n - i
                lapack.dlarfg(
                    &reflector_len, &F[i, i, k], &F[i + 1, i, k],
                    &one, &tau,
                )
                beta = F[i, i, k]
                F[i, i, k] = 1.0

                # F[k-1] H and U[k] H accompany H' F[k].
                lapack.dlarf(
                    &side_r, &n, &reflector_len, &F[i, i, k], &one,
                    &tau, &F[0, i, k - 1], &capacity, &work_h[0],
                )
                lapack.dlarf(
                    &side_l, &reflector_len, &trail_cols, &F[i, i, k],
                    &one, &tau, &F[i, i + 1, k], &capacity, &work_h[0],
                )
                lapack.dlarf(
                    &side_r, &basis_rows, &reflector_len, &F[i, i, k],
                    &one, &tau, &U[0, i, k], &basis_rows, &work_h[0],
                )
                F[i, i, k] = beta
                for row in range(i + 1, n):
                    F[row, i, k] = 0.0

            reflector_len = n - i - 1
            tail_row = i + 2 if i + 2 < n else i + 1
            lapack.dlarfg(
                &reflector_len, &F[i + 1, i, 0], &F[tail_row, i, 0],
                &one, &tau,
            )
            beta = F[i + 1, i, 0]
            F[i + 1, i, 0] = 1.0

            # Close the cycle with F[p-1] H and U[0] H before H' F[0].
            lapack.dlarf(
                &side_r, &n, &reflector_len, &F[i + 1, i, 0], &one,
                &tau, &F[0, i + 1, period - 1], &capacity, &work_h[0],
            )
            lapack.dlarf(
                &side_r, &basis_rows, &reflector_len, &F[i + 1, i, 0],
                &one, &tau, &U[0, i + 1, 0], &basis_rows, &work_h[0],
            )
            lapack.dlarf(
                &side_l, &reflector_len, &trail_cols, &F[i + 1, i, 0],
                &one, &tau, &F[i + 1, i + 1, 0], &capacity, &work_h[0],
            )
            F[i + 1, i, 0] = beta
            for row in range(i + 2, n):
                F[row, i, 0] = 0.0

    return factors_arr, bases_arr, ranks_arr, block_ranks_arr


def make_periodic_hessenberg_HOUSEHOLDER_Z(factors, bases, ranks, block_ranks):
    r"""Apply the fused periodic Householder preparation to complex128 data.

    ``ZLARFG`` defines ``H`` so that the column annihilation is ``H^H``.
    Therefore this kernel applies ``conj(tau)`` from the left to the current
    factor, but applies ``tau`` from the right to the preceding factor and
    the current physical basis.  Storage and rank behavior otherwise match
    :func:`make_periodic_hessenberg_HOUSEHOLDER_D`.
    """
    cdef tuple arrays = _householder_inputs(
        factors, bases, ranks, block_ranks, np.complex128
    )
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef cnp.ndarray block_ranks_arr = arrays[3]
    cdef ZTYPE_t[::1, :, :] F = factors_arr
    cdef ZTYPE_t[::1, :, :] U = bases_arr
    cdef Py_ssize_t[::1] r = ranks_arr
    cdef int capacity = factors_arr.shape[0]
    cdef int period = factors_arr.shape[2]
    cdef int basis_rows = bases_arr.shape[0]
    cdef int n = r[0]
    cdef bint needs_qr = bool(np.any(ranks_arr > n))
    cdef cnp.ndarray tau_qr_arr
    cdef cnp.ndarray query_arr
    cdef cnp.ndarray work_qr_arr
    cdef cnp.ndarray work_h_arr
    cdef ZTYPE_t[::1] tau_qr
    cdef ZTYPE_t[::1] query
    cdef ZTYPE_t[::1] work_qr
    cdef ZTYPE_t[::1] work_h
    cdef char side_l = 'L'
    cdef char side_r = 'R'
    cdef char trans_n = 'N'
    cdef int lwork = -1
    cdef int info = 0
    cdef int query_m, query_n, query_k
    cdef int qr_work, factor_work, basis_work
    cdef int k, km, i, row, rows, prev_rows, tail_row, failure = 0
    cdef int reflector_len, trail_cols, one = 1
    cdef double complex tau, tau_h, beta

    if n == 0:
        ranks_arr[:] = 0
        return factors_arr, bases_arr, ranks_arr, block_ranks_arr

    if needs_qr:
        tau_qr_arr = np.empty(n, dtype=np.complex128)
        query_arr = np.empty(1, dtype=np.complex128)
        tau_qr = tau_qr_arr
        query = query_arr
        query_m = capacity
        query_n = n
        query_k = n

        with nogil:
            lapack.zgeqrf(
                &query_m, &query_n, &F[0, 0, 0], &capacity, &tau_qr[0],
                &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"ZGEQRF workspace query failed with info={info}"
            )
        qr_work = max(1, <int>query[0].real)

        query_m = capacity
        query_n = capacity
        with nogil:
            lapack.zunmqr(
                &side_r, &trans_n, &query_m, &query_n, &query_k,
                &F[0, 0, 0], &capacity, &tau_qr[0],
                &F[0, 0, 0], &capacity, &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"ZUNMQR factor workspace query failed with info={info}"
            )
        factor_work = max(1, <int>query[0].real)

        query_m = basis_rows
        with nogil:
            lapack.zunmqr(
                &side_r, &trans_n, &query_m, &query_n, &query_k,
                &F[0, 0, 0], &capacity, &tau_qr[0],
                &U[0, 0, 0], &basis_rows, &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"ZUNMQR basis workspace query failed with info={info}"
            )
        basis_work = max(1, <int>query[0].real)
        lwork = max(qr_work, factor_work, basis_work)
        work_qr_arr = np.empty(lwork, dtype=np.complex128)
        work_qr = work_qr_arr

        with nogil:
            for k in range(period - 1, 0, -1):
                rows = <int>r[k]
                if rows == n:
                    continue
                km = k - 1
                prev_rows = <int>r[km]
                info = 0
                lapack.zgeqrf(
                    &rows, &n, &F[0, 0, k], &capacity, &tau_qr[0],
                    &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 1
                    break
                lapack.zunmqr(
                    &side_r, &trans_n, &prev_rows, &rows, &n,
                    &F[0, 0, k], &capacity, &tau_qr[0],
                    &F[0, 0, km], &capacity, &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 2
                    break
                lapack.zunmqr(
                    &side_r, &trans_n, &basis_rows, &rows, &n,
                    &F[0, 0, k], &capacity, &tau_qr[0],
                    &U[0, 0, k], &basis_rows, &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 3
                    break

                for i in range(n):
                    for row in range(i + 1, rows):
                        F[row, i, k] = 0.0
                for i in range(n, rows):
                    for row in range(prev_rows):
                        F[row, i, km] = 0.0
                    for row in range(basis_rows):
                        U[row, i, k] = 0.0
                r[k] = n

        if failure:
            raise np.linalg.LinAlgError(
                f"selective QR failed at factor {k}, stage {failure}, info={info}"
            )

    work_h_arr = np.empty(max(n, basis_rows), dtype=np.complex128)
    work_h = work_h_arr
    with nogil:
        for i in range(n - 1):
            trail_cols = n - i - 1
            for k in range(period - 1, 0, -1):
                reflector_len = n - i
                lapack.zlarfg(
                    &reflector_len, &F[i, i, k], &F[i + 1, i, k],
                    &one, &tau,
                )
                beta = F[i, i, k]
                F[i, i, k] = 1.0
                tau_h = tau.conjugate()

                # ZLARFG annihilates with H'; the adjacent map and basis get H.
                lapack.zlarf(
                    &side_r, &n, &reflector_len, &F[i, i, k], &one,
                    &tau, &F[0, i, k - 1], &capacity, &work_h[0],
                )
                lapack.zlarf(
                    &side_l, &reflector_len, &trail_cols, &F[i, i, k],
                    &one, &tau_h, &F[i, i + 1, k], &capacity, &work_h[0],
                )
                lapack.zlarf(
                    &side_r, &basis_rows, &reflector_len, &F[i, i, k],
                    &one, &tau, &U[0, i, k], &basis_rows, &work_h[0],
                )
                F[i, i, k] = beta
                for row in range(i + 1, n):
                    F[row, i, k] = 0.0

            reflector_len = n - i - 1
            tail_row = i + 2 if i + 2 < n else i + 1
            lapack.zlarfg(
                &reflector_len, &F[i + 1, i, 0], &F[tail_row, i, 0],
                &one, &tau,
            )
            beta = F[i + 1, i, 0]
            F[i + 1, i, 0] = 1.0
            tau_h = tau.conjugate()

            lapack.zlarf(
                &side_r, &n, &reflector_len, &F[i + 1, i, 0], &one,
                &tau, &F[0, i + 1, period - 1], &capacity, &work_h[0],
            )
            lapack.zlarf(
                &side_r, &basis_rows, &reflector_len, &F[i + 1, i, 0],
                &one, &tau, &U[0, i + 1, 0], &basis_rows, &work_h[0],
            )
            lapack.zlarf(
                &side_l, &reflector_len, &trail_cols, &F[i + 1, i, 0],
                &one, &tau_h, &F[i + 1, i + 1, 0], &capacity, &work_h[0],
            )
            F[i + 1, i, 0] = beta
            for row in range(i + 2, n):
                F[row, i, 0] = 0.0

    return factors_arr, bases_arr, ranks_arr, block_ranks_arr


def make_periodic_hessenberg_HOUSEHOLDER_NUM_ROWS_D(
    factors, bases, ranks, block_ranks, num_rows
):
    r"""Apply the FP64 fused reduction using exclusive column row bounds.

    ``num_rows[y, k]`` states that ``factors[num_rows[y, k]:, y, k]``
    is structurally zero on entry.  The bounds are updated in place as QR
    transformations and cyclic Householders mix columns and bounded row
    intervals.  The factor, basis, rank, and block metadata contract matches
    :func:`make_periodic_hessenberg_HOUSEHOLDER_D`.
    """
    cdef tuple arrays = _householder_inputs(
        factors, bases, ranks, block_ranks, np.float64
    )
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef cnp.ndarray block_ranks_arr = arrays[3]
    cdef cnp.ndarray num_rows_arr = _num_rows_input(
        num_rows, factors_arr, ranks_arr
    )
    cdef DTYPE_t[::1, :, :] F = factors_arr
    cdef DTYPE_t[::1, :, :] U = bases_arr
    cdef Py_ssize_t[::1] r = ranks_arr
    cdef Py_ssize_t[:, ::1] nr = num_rows_arr
    cdef int capacity = factors_arr.shape[0]
    cdef int period = factors_arr.shape[2]
    cdef int basis_rows = bases_arr.shape[0]
    cdef int n = r[0]
    cdef bint needs_qr = bool(np.any(ranks_arr > n))
    cdef cnp.ndarray tau_qr_arr
    cdef cnp.ndarray query_arr
    cdef cnp.ndarray work_qr_arr
    cdef cnp.ndarray work_h_arr
    cdef DTYPE_t[::1] tau_qr
    cdef DTYPE_t[::1] query
    cdef DTYPE_t[::1] work_qr
    cdef DTYPE_t[::1] work_h
    cdef char side_l = 'L'
    cdef char side_r = 'R'
    cdef char trans_n = 'N'
    cdef int lwork = -1
    cdef int info = 0
    cdef int query_m, query_n, query_k
    cdef int qr_work, factor_work, basis_work
    cdef int k, km, i, row, col, rows, prev_rows, tail_row
    cdef int start, stop, support, failure = 0
    cdef int reflector_len, trail_cols, one = 1
    cdef double tau, beta

    if n == 0:
        ranks_arr[:] = 0
        num_rows_arr[:] = 0
        return (
            factors_arr, bases_arr, ranks_arr, block_ranks_arr, num_rows_arr
        )

    if needs_qr:
        tau_qr_arr = np.empty(n, dtype=np.float64)
        query_arr = np.empty(1, dtype=np.float64)
        tau_qr = tau_qr_arr
        query = query_arr
        query_m = capacity
        query_n = n
        query_k = n

        with nogil:
            lapack.dgeqrf(
                &query_m, &query_n, &F[0, 0, 0], &capacity, &tau_qr[0],
                &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"DGEQRF workspace query failed with info={info}"
            )
        qr_work = max(1, <int>query[0])

        query_m = capacity
        query_n = capacity
        with nogil:
            lapack.dormqr(
                &side_r, &trans_n, &query_m, &query_n, &query_k,
                &F[0, 0, 0], &capacity, &tau_qr[0],
                &F[0, 0, 0], &capacity, &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"DORMQR factor workspace query failed with info={info}"
            )
        factor_work = max(1, <int>query[0])

        query_m = basis_rows
        with nogil:
            lapack.dormqr(
                &side_r, &trans_n, &query_m, &query_n, &query_k,
                &F[0, 0, 0], &capacity, &tau_qr[0],
                &U[0, 0, 0], &basis_rows, &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"DORMQR basis workspace query failed with info={info}"
            )
        basis_work = max(1, <int>query[0])
        lwork = max(qr_work, factor_work, basis_work)
        work_qr_arr = np.empty(lwork, dtype=np.float64)
        work_qr = work_qr_arr

        with nogil:
            for k in range(period - 1, 0, -1):
                rows = <int>r[k]
                if rows == n:
                    continue
                km = k - 1
                prev_rows = <int>r[km]
                support = 0
                for col in range(rows):
                    if nr[col, km] > support:
                        support = <int>nr[col, km]
                if support > prev_rows:
                    support = prev_rows

                info = 0
                lapack.dgeqrf(
                    &rows, &n, &F[0, 0, k], &capacity, &tau_qr[0],
                    &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 1
                    break
                lapack.dormqr(
                    &side_r, &trans_n, &prev_rows, &rows, &n,
                    &F[0, 0, k], &capacity, &tau_qr[0],
                    &F[0, 0, km], &capacity, &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 2
                    break
                lapack.dormqr(
                    &side_r, &trans_n, &basis_rows, &rows, &n,
                    &F[0, 0, k], &capacity, &tau_qr[0],
                    &U[0, 0, k], &basis_rows, &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 3
                    break

                for col in range(n):
                    nr[col, k] = col + 1
                    nr[col, km] = support
                    for row in range(col + 1, rows):
                        F[row, col, k] = 0.0
                for col in range(n, capacity):
                    nr[col, k] = 0
                    nr[col, km] = 0
                for col in range(n, rows):
                    for row in range(prev_rows):
                        F[row, col, km] = 0.0
                    for row in range(basis_rows):
                        U[row, col, k] = 0.0
                r[k] = n

        if failure:
            raise np.linalg.LinAlgError(
                f"selective QR failed at factor {k}, stage {failure}, info={info}"
            )

    with nogil:
        for k in range(period):
            for col in range(n, capacity):
                nr[col, k] = 0

    work_h_arr = np.empty(max(n, basis_rows), dtype=np.float64)
    work_h = work_h_arr
    with nogil:
        for i in range(n - 1):
            trail_cols = n - i - 1
            for k in range(period - 1, 0, -1):
                start = i
                stop = <int>nr[i, k]
                if stop < start + 1:
                    stop = start + 1
                elif stop > n:
                    stop = n
                reflector_len = stop - start
                lapack.dlarfg(
                    &reflector_len, &F[start, i, k], &F[start + 1, i, k],
                    &one, &tau,
                )
                beta = F[start, i, k]
                F[start, i, k] = 1.0

                lapack.dlarf(
                    &side_r, &n, &reflector_len, &F[start, i, k], &one,
                    &tau, &F[0, start, k - 1], &capacity, &work_h[0],
                )
                lapack.dlarf(
                    &side_l, &reflector_len, &trail_cols, &F[start, i, k],
                    &one, &tau, &F[start, i + 1, k], &capacity, &work_h[0],
                )
                lapack.dlarf(
                    &side_r, &basis_rows, &reflector_len,
                    &F[start, i, k], &one, &tau, &U[0, start, k],
                    &basis_rows, &work_h[0],
                )
                if tau != 0.0:
                    _right_num_rows(nr, k - 1, start, stop)
                    _left_num_rows(nr, k, start, stop, i + 1, n)
                F[start, i, k] = beta
                for row in range(i + 1, n):
                    F[row, i, k] = 0.0
                nr[i, k] = i + 1

            start = i + 1
            stop = <int>nr[i, 0]
            if stop < start + 1:
                stop = start + 1
            elif stop > n:
                stop = n
            reflector_len = stop - start
            tail_row = start + 1 if start + 1 < n else start
            lapack.dlarfg(
                &reflector_len, &F[start, i, 0], &F[tail_row, i, 0],
                &one, &tau,
            )
            beta = F[start, i, 0]
            F[start, i, 0] = 1.0

            lapack.dlarf(
                &side_r, &n, &reflector_len, &F[start, i, 0], &one,
                &tau, &F[0, start, period - 1], &capacity, &work_h[0],
            )
            lapack.dlarf(
                &side_r, &basis_rows, &reflector_len, &F[start, i, 0],
                &one, &tau, &U[0, start, 0], &basis_rows, &work_h[0],
            )
            lapack.dlarf(
                &side_l, &reflector_len, &trail_cols, &F[start, i, 0],
                &one, &tau, &F[start, start, 0], &capacity, &work_h[0],
            )
            if tau != 0.0:
                _right_num_rows(nr, period - 1, start, stop)
                _left_num_rows(nr, 0, start, stop, start, n)
            F[start, i, 0] = beta
            for row in range(i + 2, n):
                F[row, i, 0] = 0.0
            nr[i, 0] = i + 2

    return factors_arr, bases_arr, ranks_arr, block_ranks_arr, num_rows_arr


def make_periodic_hessenberg_HOUSEHOLDER_NUM_ROWS_Z(
    factors, bases, ranks, block_ranks, num_rows
):
    r"""Apply the complex128 fused reduction using ``num_rows`` bounds.

    The structural-bound contract and in-place metadata propagation match
    :func:`make_periodic_hessenberg_HOUSEHOLDER_NUM_ROWS_D`.  ``ZLARFG`` reflectors apply
    ``H^H`` to the current factor and ``H`` to its predecessor and basis.
    """
    cdef tuple arrays = _householder_inputs(
        factors, bases, ranks, block_ranks, np.complex128
    )
    cdef cnp.ndarray factors_arr = arrays[0]
    cdef cnp.ndarray bases_arr = arrays[1]
    cdef cnp.ndarray ranks_arr = arrays[2]
    cdef cnp.ndarray block_ranks_arr = arrays[3]
    cdef cnp.ndarray num_rows_arr = _num_rows_input(
        num_rows, factors_arr, ranks_arr
    )
    cdef ZTYPE_t[::1, :, :] F = factors_arr
    cdef ZTYPE_t[::1, :, :] U = bases_arr
    cdef Py_ssize_t[::1] r = ranks_arr
    cdef Py_ssize_t[:, ::1] nr = num_rows_arr
    cdef int capacity = factors_arr.shape[0]
    cdef int period = factors_arr.shape[2]
    cdef int basis_rows = bases_arr.shape[0]
    cdef int n = r[0]
    cdef bint needs_qr = bool(np.any(ranks_arr > n))
    cdef cnp.ndarray tau_qr_arr
    cdef cnp.ndarray query_arr
    cdef cnp.ndarray work_qr_arr
    cdef cnp.ndarray work_h_arr
    cdef ZTYPE_t[::1] tau_qr
    cdef ZTYPE_t[::1] query
    cdef ZTYPE_t[::1] work_qr
    cdef ZTYPE_t[::1] work_h
    cdef char side_l = 'L'
    cdef char side_r = 'R'
    cdef char trans_n = 'N'
    cdef int lwork = -1
    cdef int info = 0
    cdef int query_m, query_n, query_k
    cdef int qr_work, factor_work, basis_work
    cdef int k, km, i, row, col, rows, prev_rows, tail_row
    cdef int start, stop, support, failure = 0
    cdef int reflector_len, trail_cols, one = 1
    cdef double complex tau, tau_h, beta

    if n == 0:
        ranks_arr[:] = 0
        num_rows_arr[:] = 0
        return (
            factors_arr, bases_arr, ranks_arr, block_ranks_arr, num_rows_arr
        )

    if needs_qr:
        tau_qr_arr = np.empty(n, dtype=np.complex128)
        query_arr = np.empty(1, dtype=np.complex128)
        tau_qr = tau_qr_arr
        query = query_arr
        query_m = capacity
        query_n = n
        query_k = n

        with nogil:
            lapack.zgeqrf(
                &query_m, &query_n, &F[0, 0, 0], &capacity, &tau_qr[0],
                &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"ZGEQRF workspace query failed with info={info}"
            )
        qr_work = max(1, <int>query[0].real)

        query_m = capacity
        query_n = capacity
        with nogil:
            lapack.zunmqr(
                &side_r, &trans_n, &query_m, &query_n, &query_k,
                &F[0, 0, 0], &capacity, &tau_qr[0],
                &F[0, 0, 0], &capacity, &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"ZUNMQR factor workspace query failed with info={info}"
            )
        factor_work = max(1, <int>query[0].real)

        query_m = basis_rows
        with nogil:
            lapack.zunmqr(
                &side_r, &trans_n, &query_m, &query_n, &query_k,
                &F[0, 0, 0], &capacity, &tau_qr[0],
                &U[0, 0, 0], &basis_rows, &query[0], &lwork, &info,
            )
        if info != 0:
            raise np.linalg.LinAlgError(
                f"ZUNMQR basis workspace query failed with info={info}"
            )
        basis_work = max(1, <int>query[0].real)
        lwork = max(qr_work, factor_work, basis_work)
        work_qr_arr = np.empty(lwork, dtype=np.complex128)
        work_qr = work_qr_arr

        with nogil:
            for k in range(period - 1, 0, -1):
                rows = <int>r[k]
                if rows == n:
                    continue
                km = k - 1
                prev_rows = <int>r[km]
                support = 0
                for col in range(rows):
                    if nr[col, km] > support:
                        support = <int>nr[col, km]
                if support > prev_rows:
                    support = prev_rows

                info = 0
                lapack.zgeqrf(
                    &rows, &n, &F[0, 0, k], &capacity, &tau_qr[0],
                    &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 1
                    break
                lapack.zunmqr(
                    &side_r, &trans_n, &prev_rows, &rows, &n,
                    &F[0, 0, k], &capacity, &tau_qr[0],
                    &F[0, 0, km], &capacity, &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 2
                    break
                lapack.zunmqr(
                    &side_r, &trans_n, &basis_rows, &rows, &n,
                    &F[0, 0, k], &capacity, &tau_qr[0],
                    &U[0, 0, k], &basis_rows, &work_qr[0], &lwork, &info,
                )
                if info != 0:
                    failure = 3
                    break

                for col in range(n):
                    nr[col, k] = col + 1
                    nr[col, km] = support
                    for row in range(col + 1, rows):
                        F[row, col, k] = 0.0
                for col in range(n, capacity):
                    nr[col, k] = 0
                    nr[col, km] = 0
                for col in range(n, rows):
                    for row in range(prev_rows):
                        F[row, col, km] = 0.0
                    for row in range(basis_rows):
                        U[row, col, k] = 0.0
                r[k] = n

        if failure:
            raise np.linalg.LinAlgError(
                f"selective QR failed at factor {k}, stage {failure}, info={info}"
            )

    with nogil:
        for k in range(period):
            for col in range(n, capacity):
                nr[col, k] = 0

    work_h_arr = np.empty(max(n, basis_rows), dtype=np.complex128)
    work_h = work_h_arr
    with nogil:
        for i in range(n - 1):
            trail_cols = n - i - 1
            for k in range(period - 1, 0, -1):
                start = i
                stop = <int>nr[i, k]
                if stop < start + 1:
                    stop = start + 1
                elif stop > n:
                    stop = n
                reflector_len = stop - start
                lapack.zlarfg(
                    &reflector_len, &F[start, i, k], &F[start + 1, i, k],
                    &one, &tau,
                )
                beta = F[start, i, k]
                F[start, i, k] = 1.0
                tau_h = tau.conjugate()

                lapack.zlarf(
                    &side_r, &n, &reflector_len, &F[start, i, k], &one,
                    &tau, &F[0, start, k - 1], &capacity, &work_h[0],
                )
                lapack.zlarf(
                    &side_l, &reflector_len, &trail_cols, &F[start, i, k],
                    &one, &tau_h, &F[start, i + 1, k], &capacity, &work_h[0],
                )
                lapack.zlarf(
                    &side_r, &basis_rows, &reflector_len,
                    &F[start, i, k], &one, &tau, &U[0, start, k],
                    &basis_rows, &work_h[0],
                )
                if tau != 0.0:
                    _right_num_rows(nr, k - 1, start, stop)
                    _left_num_rows(nr, k, start, stop, i + 1, n)
                F[start, i, k] = beta
                for row in range(i + 1, n):
                    F[row, i, k] = 0.0
                nr[i, k] = i + 1

            start = i + 1
            stop = <int>nr[i, 0]
            if stop < start + 1:
                stop = start + 1
            elif stop > n:
                stop = n
            reflector_len = stop - start
            tail_row = start + 1 if start + 1 < n else start
            lapack.zlarfg(
                &reflector_len, &F[start, i, 0], &F[tail_row, i, 0],
                &one, &tau,
            )
            beta = F[start, i, 0]
            F[start, i, 0] = 1.0
            tau_h = tau.conjugate()

            lapack.zlarf(
                &side_r, &n, &reflector_len, &F[start, i, 0], &one,
                &tau, &F[0, start, period - 1], &capacity, &work_h[0],
            )
            lapack.zlarf(
                &side_r, &basis_rows, &reflector_len, &F[start, i, 0],
                &one, &tau, &U[0, start, 0], &basis_rows, &work_h[0],
            )
            lapack.zlarf(
                &side_l, &reflector_len, &trail_cols, &F[start, i, 0],
                &one, &tau_h, &F[start, start, 0], &capacity, &work_h[0],
            )
            if tau != 0.0:
                _right_num_rows(nr, period - 1, start, stop)
                _left_num_rows(nr, 0, start, stop, start, n)
            F[start, i, 0] = beta
            for row in range(i + 2, n):
                F[row, i, 0] = 0.0
            nr[i, 0] = i + 2

    return factors_arr, bases_arr, ranks_arr, block_ranks_arr, num_rows_arr


def make_periodic_hessenberg_GIVENS_D(factors, bases, ranks):
    r"""Run the archived FP64 full-QR-plus-Givens CRed experiment."""
    _full_qr_sweep_d(factors, bases, ranks)
    return _hessenberg_chase_d(factors, bases, ranks)


def make_periodic_hessenberg_GIVENS_Z(factors, bases, ranks):
    r"""Run the archived complex128 full-QR-plus-Givens CRed experiment."""
    _full_qr_sweep_z(factors, bases, ranks)
    return _hessenberg_chase_z(factors, bases, ranks)


def make_periodic_hessenberg_PANEL_GIVENS_D(
    factors, bases, ranks, block_ranks
):
    r"""Run the archived FP64 panel-QR-plus-Givens CRed experiment."""
    _panel_qr_sweep_d(factors, bases, ranks, block_ranks)
    _hessenberg_chase_d(factors, bases, ranks)
    return factors, bases, ranks, block_ranks


def make_periodic_hessenberg_PANEL_GIVENS_Z(
    factors, bases, ranks, block_ranks
):
    r"""Run the archived complex panel-QR-plus-Givens CRed experiment."""
    _panel_qr_sweep_z(factors, bases, ranks, block_ranks)
    _hessenberg_chase_z(factors, bases, ranks)
    return factors, bases, ranks, block_ranks
