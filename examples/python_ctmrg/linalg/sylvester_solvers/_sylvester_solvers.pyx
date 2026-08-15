# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

r"""Native least-squares kernels for compressed Sylvester equations.

The periodic entry points solve the following reduced problem.  All site
indices are modulo the period ``p``.  Let

``H[k]``             have shape ``(m, m)``,
``w[k]``             have shape ``(d, d)``,
``v[k]``             have shape ``(d, d)``, and
``R[k] = residual_r[k]`` have shape ``(d, d)``.

Let ``E_tail: C^m -> C^d`` select the final ``d`` Arnoldi coordinates.  The
compressed Arnoldi tail operator is

``G[k] = R[k] E_tail``.

Only the leading structural-rank block

``W[k] = w[k, :rank, :rank]``

is part of the problem; the remaining physical columns are exact zero
padding.  If ``J: C^d -> C^m`` embeds a vector into the first ``d`` Arnoldi
coordinates, define

``beta[k] = J v[k, :, :rank]``.

For each site, let ``P[k] = diag(active_cols[k])`` be the coordinate
projection selected by periodic Arnoldi.  The unknowns are matrices
``Y[k]`` of shape ``(m, rank)`` constrained by

``Y[k] = P[k] Y[k]``.

The dense-QR and least-squares periodic-Schur algorithms have the same
mathematical contract: they minimize

``sum_k ( ||H[k] Y[k+1] + Y[k] W[k] - beta[k]||_F^2``
``      + ||R[k] E_tail Y[k+1]||_F^2 )``.

There is no need to place ``P[k]`` explicitly around the first residual in
this statement.  Periodic Arnoldi makes the inactive coordinates structural:

``(I - P[k]) H[k] P[k+1] = 0`` and ``(I - P[k]) beta[k] = 0``.

Consequently the first residual already has support in ``P[k]`` whenever
``Y[k] = P[k] Y[k]``.  The implementations compact these active coordinates
solely to avoid arithmetic on known zeros.

The Schur-Galerkin variants instead impose the projected core equations

``P[k] (H[k] Y[k+1] + Y[k] W[k] - beta[k]) = 0``.

Thus its solution is independent of ``residual_r``.  The Arnoldi tail is
evaluated afterward only so that this variant returns the same full residual
diagnostic as the least-squares solvers.

The returned array has shape ``(p, d, m)`` and stores

``x[k, :rank, :] = Y[k].T``,  ``x[k, rank:, :] = 0``.

For comparison between algorithms, both report the same untransformed
site-and-column residual norms

``error[k, j]^2 = ||(H[k] Y[k+1] + Y[k] W[k] - beta[k])[:, j]||_2^2``
``                + ||(R[k] E_tail Y[k+1])[:, j]||_2^2``.

For real input, ``w[0]`` is quasi-upper-triangular and ``w[1:]`` are upper
triangular.  For complex input every ``w[k]`` is upper triangular.

Ownership boundary
------------------
Periodic Arnoldi and the final lift into the physical space remain in JAX.
This module receives only the compressed Arnoldi data and returns ``x = Y.T``;
it does not receive the physical Arnoldi bases ``Q[k]``.  The caller forms

``dX[k] = Q[k] Y[k]``

after the compressed solve.  Thus the native solver remains independent of
the large physical dimension and both compressed algorithms expose exactly
the same small input/output contract.

Naming convention
-----------------
The LAPACK suffix ``D`` denotes real float64 arithmetic and ``Z`` denotes
complex128 arithmetic. The C ABI solver names are

``sylvester_compressed_periodic_dense_gmres_D``
``sylvester_compressed_periodic_dense_gmres_Z``
    Dense-GMRES entry points that assemble the full active cyclic projected
    least-squares problem and solve it by QR.

``sylvester_compressed_periodic_schur_gmres_D``
``sylvester_compressed_periodic_schur_gmres_Z``
    Periodic-Schur-GMRES entry points that first reduce ``H[k]`` to periodic
    Schur form and then exploit its triangular or quasi-triangular structure.

``sylvester_compressed_periodic_schur_galerkin_D``
``sylvester_compressed_periodic_schur_galerkin_Z``
    Real or complex periodic-Schur block substitution for the projected core
    equations; the Arnoldi tail contributes only to the reported residual.

Within each arithmetic type, the dense-GMRES and periodic-Schur-GMRES
functions have identical arguments, outputs, support constraints, and
residual definitions; only their algorithms differ.  The Galerkin function
keeps that interface but changes the equation used to choose ``Y`` as stated
above.  The non-periodic specialization

``sylvester_compressed_schur_gmres_D``

is reserved for the existing ``p = 1`` real solver.  Private implementation
helpers carry a leading underscore and describe the narrower operation they
perform, such as solving one Schur block.
"""

import numpy as np

cimport numpy as cnp
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
from linalg.periodic_schur._periodic_schur cimport (
    compute_periodic_schur_active_D,
    compute_periodic_schur_active_scaled_D,
    compute_periodic_schur_active_Z,
    compute_periodic_schur_active_scaled_Z,
    periodic_schur_active_size,
    slicot_mb03ke_D,
)
from libc.math cimport sqrt
from libc.stdlib cimport calloc, free, malloc


cdef extern from "fenv.h":
    int FE_ALL_EXCEPT
    int feclearexcept(int excepts) noexcept nogil


# Single-period real structured solver.
cdef int _solve_schur_block_D(
    const double* T,
    const double* C,
    const double* S,
    const double* rhs,
    int n,
    int d_residual,
    int s,
    int tpqrt_block_size,
    double* solution,
) noexcept nogil:
    r"""Compute ``U`` minimizing

    ``||T U - U S - rhs||_F^2 + ||C U||_F^2``.

    ``T`` is ``(n, n)`` real quasi-upper-triangular, ``C`` is
    ``(d_residual, n)``, and ``S`` is one ``(s, s)`` real-Schur block with
    ``s`` equal to one or two.  The inputs and the ``(n, s)`` output
    ``solution`` use LAPACK column-major storage.

    In vector form the augmented least-squares operator is

    ``[I_s kron T - S.T kron I_n; C kron I_s]``.

    The implementation interleaves the Schur-block rows, triangularizes the
    at-most-4-by-4 diagonal blocks with Givens rotations, and factors the
    remaining triangular-plus-dense matrix with ``DTPQRT``.

    Return zero on success, ``-1`` on allocation failure, or ``2000 + info``,
    ``3000 + info``, and ``4000 + info`` for failures in ``DTPQRT``,
    ``DTPMQRT``, and ``DTRTRS``, respectively.
    """
    cdef int N = s*n
    cdef int M = s*d_residual
    cdef int L = 0
    cdef int NB = tpqrt_block_size if tpqrt_block_size < N else N
    cdef int nrhs = 1
    cdef int lda = N
    cdef int ldb = M if M > 1 else 1
    cdef int ldt = NB
    cdef int info = 0
    cdef int i, k, a, p, col, row
    cdef int h_size, local_size, local_col, local_row, rotate_length
    cdef int row_stride = N
    cdef double cs, sn, r, x0, x1
    cdef char side = 76
    cdef char trans_t = 84
    cdef char upper = 85
    cdef char no_trans = 78
    cdef char non_unit = 78
    cdef size_t total = N*N + M*N + 2*NB*N + N + M + NB
    cdef double* memory = <double*>calloc(total, sizeof(double))
    cdef double* A
    cdef double* B
    cdef double* reflectors
    cdef double* work
    cdef double* rhs_top
    cdef double* rhs_bottom
    cdef double* apply_work

    if memory == NULL:
        return -1
    A = memory
    B = A + N*N
    reflectors = B + M*N
    work = reflectors + NB*N
    rhs_top = work + NB*N
    rhs_bottom = rhs_top + N
    apply_work = rhs_bottom + M

    # Assemble I_s kron T - S.T kron I_n.  These are Kronecker placements,
    # not matrix products.
    for i in range(n):
        for k in range(n):
            for a in range(s):
                A[s*i + a + N*(s*k + a)] = T[i + n*k]
        for a in range(s):
            for p in range(s):
                A[s*i + a + N*(s*i + p)] -= S[p + s*a]
        for a in range(s):
            rhs_top[s*i + a] = rhs[i + n*a]

    i = 0
    while i < n:
        h_size = 2 if i + 1 < n and T[i + 1 + n*i] != 0.0 else 1
        local_size = h_size*s
        p = s*i
        for local_col in range(local_size):
            col = p + local_col
            for local_row in range(local_size - 1, local_col, -1):
                row = p + local_row
                lapack.dlartg(
                    &A[row - 1 + N*col], &A[row + N*col], &cs, &sn, &r
                )
                rotate_length = N - col
                blas.drot(
                    &rotate_length,
                    &A[row - 1 + N*col], &row_stride,
                    &A[row + N*col], &row_stride,
                    &cs, &sn,
                )
                A[row - 1 + N*col] = r
                A[row + N*col] = 0.0
                x0 = rhs_top[row - 1]
                x1 = rhs_top[row]
                rhs_top[row - 1] = cs*x0 + sn*x1
                rhs_top[row] = -sn*x0 + cs*x1
        i += h_size

    # Assemble C kron I_s in row-interleaved ordering.  This is another
    # Kronecker placement rather than a GEMM.
    for p in range(d_residual):
        for i in range(n):
            for a in range(s):
                B[s*p + a + M*(s*i + a)] = C[p + d_residual*i]

    lapack.dtpqrt(
        &M, &N, &L, &NB,
        A, &lda,
        B, &ldb,
        reflectors, &ldt,
        work, &info,
    )
    if info != 0:
        free(memory)
        return 2000 + info

    lapack.dtpmqrt(
        &side, &trans_t, &M, &nrhs, &N, &L, &NB,
        B, &ldb,
        reflectors, &ldt,
        rhs_top, &lda,
        rhs_bottom, &ldb,
        apply_work, &info,
    )
    if info != 0:
        free(memory)
        return 3000 + info

    lapack.dtrtrs(
        &upper, &no_trans, &non_unit,
        &N, &nrhs,
        A, &lda,
        rhs_top, &lda,
        &info,
    )
    if info != 0:
        free(memory)
        return 4000 + info
    feclearexcept(FE_ALL_EXCEPT)

    for i in range(n):
        for a in range(s):
            solution[i + n*a] = rhs_top[s*i + a]
    free(memory)
    return 0


cdef int _sylvester_compressed_schur_gmres_D(
    const double* H,
    const double* w,
    const double* v,
    const double* residual_r,
    const unsigned char* block_2x2_start,
    int n,
    int d,
    int upper_problem,
    int tpqrt_block_size,
    double* x,
    double* error,
) noexcept nogil:
    r"""Compute ``Y`` minimizing

    ``||H Y - Y w - beta||_F^2 + ||G Y||_F^2``,

    where ``beta[:d, :] = v`` and ``G[:, -d:] = residual_r``.  The routine
    returns ``x = Y.T`` and

    ``error[j] = ||[H Y - Y w - beta; G Y][:, j]||_2``.

    The C-order inputs have shapes ``H: (n, n)``, ``w: (d, d)``,
    ``v: (d, d)``, ``residual_r: (d, d)``, and
    ``block_2x2_start: (d,)``; outputs have shapes ``x: (d, n)`` and
    ``error: (d,)``.  A true mask entry at ``j`` marks the real-Schur block
    ``j:j+2``.  ``upper_problem`` selects ascending upper-quasi-triangular
    block substitution; otherwise reversing the Schur problem implements a
    descending lower-quasi-triangular sweep.

    The routine forms ``H = Z T Z.T`` once and calls
    :func:`_solve_schur_block_D` for each block
    of ``w``.  Return zero
    on success, ``-1`` on allocation failure, ``1000 + info`` on ``DGEES``
    failure, or the inner block solver's status.
    """
    cdef size_t total = 2*n*n + 6*n + 2*d*d + 4*n*d + 4
    cdef double* memory = <double*>calloc(total, sizeof(double))
    cdef unsigned char* blocks = <unsigned char*>calloc(d, sizeof(unsigned char))
    cdef bint* bwork = <bint*>calloc(n, sizeof(bint))
    cdef double* work = NULL
    cdef double* T
    cdef double* Z
    cdef double* wr
    cdef double* wi
    cdef double* w_work
    cdef double* v_work
    cdef double* beta
    cdef double* C
    cdef double* U
    cdef double* Y
    cdef double* rhs
    cdef double* S
    cdef double* solution
    cdef double work_query
    cdef double value, residual_value, error_squared
    cdef int i, j, k, p, a, s, old_start, original_j
    cdef int info = 0
    cdef int sdim = 0
    cdef int lwork = -1
    cdef int lda = n
    cdef int ldvs = n
    cdef int inc_one = 1
    cdef char jobvs = 86
    cdef char sort = 78
    cdef char no_trans = 78
    cdef char trans = 84
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef double minus_one = -1.0

    if memory == NULL or blocks == NULL or bwork == NULL:
        free(memory)
        free(blocks)
        free(bwork)
        return -1

    T = memory
    Z = T + n*n
    wr = Z + n*n
    wi = wr + n
    w_work = wi + n
    v_work = w_work + d*d
    beta = v_work + d*d
    C = beta + n*d
    U = C + d*n
    Y = U + n*d
    rhs = Y + n*d
    S = rhs + 2*n
    solution = S + 4

    # FFI buffers use row-major layout; LAPACK and the internal contractions
    # below use column-major storage.
    for i in range(n):
        for k in range(n):
            T[i + n*k] = H[i*n + k]
    for i in range(d):
        for k in range(d):
            if upper_problem:
                w_work[i + d*k] = w[i*d + k]
                v_work[i + d*k] = v[i*d + k]
            else:
                w_work[i + d*k] = w[(d - 1 - i)*d + (d - 1 - k)]
                v_work[i + d*k] = v[i*d + (d - 1 - k)]
    if upper_problem:
        for j in range(d):
            blocks[j] = block_2x2_start[j]
    else:
        for old_start in range(d - 1):
            if block_2x2_start[old_start]:
                blocks[d - old_start - 2] = 1

    lapack.dgees(
        &jobvs, &sort, <lapack.dselect2*>NULL,
        &n, T, &lda, &sdim, wr, wi, Z, &ldvs,
        &work_query, &lwork, bwork, &info,
    )
    if info != 0:
        free(memory)
        free(blocks)
        free(bwork)
        return 1000 + info
    lwork = <int>work_query
    if lwork < 3*n:
        lwork = 3*n
    work = <double*>malloc(lwork*sizeof(double))
    if work == NULL:
        free(memory)
        free(blocks)
        free(bwork)
        return -1
    lapack.dgees(
        &jobvs, &sort, <lapack.dselect2*>NULL,
        &n, T, &lda, &sdim, wr, wi, Z, &ldvs,
        work, &lwork, bwork, &info,
    )
    free(work)
    if info != 0:
        free(memory)
        free(blocks)
        free(bwork)
        return 1000 + info
    feclearexcept(FE_ALL_EXCEPT)

    # beta = Z.T[:, :d] v.
    blas.dgemm(
        &trans, &no_trans, &n, &d, &d,
        &one, Z, &n, v_work, &d, &zero, beta, &n,
    )
    # C = residual_r Z[-d:, :].  residual_r is a C-order FFI input, hence
    # its column-major view is transposed.
    blas.dgemm(
        &trans, &no_trans, &d, &n, &d,
        &one, <double*>residual_r, &d, &Z[n - d], &n, &zero, C, &d,
    )

    j = 0
    while j < d:
        s = 2 if blocks[j] else 1
        for a in range(s):
            blas.dcopy(
                &n, &beta[n*(j + a)], &inc_one, &rhs[n*a], &inc_one,
            )
            for p in range(s):
                S[a + s*p] = w_work[j + a + d*(j + p)]
        # rhs[:, :s] = beta[:, j:j+s] + U w[:, j:j+s].
        blas.dgemm(
            &no_trans, &no_trans, &n, &s, &d,
            &one, U, &n, &w_work[d*j], &d, &one, rhs, &n,
        )
        info = _solve_schur_block_D(
            T, C, S, rhs, n, d, s, tpqrt_block_size, solution
        )
        if info != 0:
            free(memory)
            free(blocks)
            free(bwork)
            return info
        for a in range(s):
            for i in range(n):
                U[i + n*(j + a)] = solution[i + n*a]
        j += s

    # Y_work = Z U restores H's Schur vectors.
    blas.dgemm(
        &no_trans, &no_trans, &n, &d, &n,
        &one, Z, &n, U, &n, &zero, beta, &n,
    )
    # Undo the lower-problem column reversal into Y.
    for j in range(d):
        original_j = j if upper_problem else d - 1 - j
        blas.dcopy(
            &n, &beta[n*j], &inc_one, &Y[n*original_j], &inc_one,
        )
    for j in range(d):
        for i in range(n):
            x[j*n + i] = Y[i + n*j]

    # beta = H Y - Y w - [v; 0].  H and w are C-order FFI inputs, so their
    # column-major views are transposed.
    blas.dgemm(
        &trans, &no_trans, &n, &d, &n,
        &one, <double*>H, &n, Y, &n, &zero, beta, &n,
    )
    blas.dgemm(
        &no_trans, &trans, &n, &d, &d,
        &minus_one, Y, &n, <double*>w, &d, &one, beta, &n,
    )
    for j in range(d):
        for i in range(d):
            beta[i + n*j] -= v[i*d + j]

    # C[:, :d] = residual_r Y[-d:, :].
    blas.dgemm(
        &trans, &no_trans, &d, &d, &d,
        &one, <double*>residual_r, &d, &Y[n - d], &n, &zero, C, &d,
    )

    # Return one norm per physical column.
    for j in range(d):
        error_squared = 0.0
        for i in range(n):
            residual_value = beta[i + n*j]
            error_squared += residual_value*residual_value
        for p in range(d):
            residual_value = C[p + d*j]
            error_squared += residual_value*residual_value
        error[j] = sqrt(error_squared)

    free(memory)
    free(blocks)
    free(bwork)
    return 0


cdef public int sylvester_compressed_schur_gmres_D(
    const double* H,
    const double* w,
    const double* v,
    const double* residual_r,
    const unsigned char* block_2x2_start,
    int n_krylov,
    int d_block,
    int upper,
    int tpqrt_block_size,
    double* x,
    double* error,
) noexcept nogil:
    r"""Solve ``min_Y ||H Y-Y w-beta||_F^2 + ||G Y||_F^2`` in float64.

    Here ``beta[:d, :] = v`` and ``G[:, -d:] = residual_r``.  This C-order
    XLA FFI entry point returns ``x = Y.T`` and one stacked-residual norm per
    column. See :func:`_sylvester_compressed_schur_gmres_D` for buffer
    shapes, Schur
    block conventions, and status codes.
    """
    return _sylvester_compressed_schur_gmres_D(
        H, w, v, residual_r, block_2x2_start,
        n_krylov, d_block, upper, tpqrt_block_size, x, error,
    )


# Periodic active-coordinate dense-QR solvers.
cdef int _solve_periodic_real_active_dense_qr(
    const double* H,
    const double* w,
    const double* v,
    const double* residual_r,
    const unsigned char* block_2x2_start,
    const unsigned char* active_cols,
    int period,
    int m,
    int d,
    int rank,
    double* x,
    double* error,
) noexcept nogil:
    r"""Minimize the rank-restricted real periodic compressed residual.

    Let ``E[k]`` select the coordinates marked by ``active_cols[k]``, let
    ``Y[k] = E[k] Yhat[k]``, set ``beta[k, :d, :rank]`` from
    ``v[k, :, :rank]``, and write ``W[k] = w[k, :rank, :rank]``.  With cyclic
    site indices, this routine computes ``Yhat`` minimizing

    ``sum_k ||E[k].T (H[k] Y[k+1] + Y[k] W[k] - beta[k])||_F^2``
    ``      + ||residual_r[k] E_tail Y[k+1]||_F^2``.

    Thus inactive Krylov coordinates are removed from both the variables and
    the Galerkin rows, while every row of the Arnoldi residual tail remains in
    the objective.  The C-order inputs have shapes ``H: (p, m, m)``,
    ``w: (p, d, d)``, ``v: (p, d, d)``,
    ``residual_r: (p, d, d)``, ``active_cols: (p, m)``, and
    ``block_2x2_start: (d,)``.

    ``w[0]`` is real quasi-upper triangular and the other ``w[k]`` are upper
    triangular.  A true block-mask entry at ``j`` couples columns ``j:j+2``.
    For each block of size ``s``, the implementation appends the right-hand
    side to the full cyclic augmented matrix, factors it with ``DGEQRF``, and
    solves the leading triangular system with ``DTRTRS``.

    The output ``x[k, j, :] = Y[k, :, j]`` has shape ``(p, d, m)`` and is zero
    outside active coordinates or in columns ``rank:``.  ``error[k, j]`` is the
    corresponding site-and-column norm of the two residual terms above.
    Return zero on success, ``-1`` on allocation failure, ``5000 + info`` or
    ``6000 + info`` on ``DGEQRF`` query/factorization failure, and
    ``6500 + info`` on ``DTRTRS`` failure.
    """
    cdef size_t y_size = period*m*d
    cdef double* Y = <double*>calloc(y_size, sizeof(double))
    cdef double* A = NULL
    cdef double* tau = NULL
    cdef double* work = NULL
    cdef double work_query
    cdef double value, core, tail, error_squared
    cdef int* counts = <int*>malloc(period*sizeof(int))
    cdef int* offsets = <int*>malloc((period + 1)*sizeof(int))
    cdef int* indices = <int*>malloc(period*m*sizeof(int))
    cdef int nactive = 0
    cdef int j = 0
    cdef int s, nvar, nrow, naug, nrhs, lda, ldb, lwork, info
    cdef int k, kp, i, ell, a, b, q, row, col, u, z
    cdef char upper = 85
    cdef char no_trans = 78
    cdef char non_unit = 78

    if Y == NULL or counts == NULL or offsets == NULL or indices == NULL:
        free(Y)
        free(counts)
        free(offsets)
        free(indices)
        return -1

    for z in range(period*d*m):
        x[z] = 0.0
    for z in range(period*d):
        error[z] = 0.0
    for k in range(period):
        offsets[k] = nactive
        counts[k] = 0
        for i in range(m):
            if active_cols[k*m + i]:
                indices[nactive] = i
                nactive += 1
                counts[k] += 1
    offsets[period] = nactive
    if nactive == 0:
        free(Y)
        free(counts)
        free(offsets)
        free(indices)
        return 0

    while j < rank:
        s = 2 if block_2x2_start[j] else 1
        nvar = nactive*s
        nrow = nvar + period*d*s
        naug = nvar + 1
        nrhs = 1
        lda = nrow
        ldb = nvar
        A = <double*>calloc(nrow*naug, sizeof(double))
        tau = <double*>malloc(naug*sizeof(double))
        if A == NULL or tau == NULL:
            free(A)
            free(tau)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return -1

        for k in range(period):
            kp = (k + 1) % period
            for a in range(s):
                for u in range(counts[k]):
                    i = indices[offsets[k] + u]
                    row = s*offsets[k] + a*counts[k] + u

                    # beta[k] = [v[k]; 0], with previously solved Schur
                    # columns moved to the right-hand side.
                    value = v[(k*d + i)*d + j + a] if i < d else 0.0
                    for q in range(rank):
                        value -= Y[(k*m + i)*d + q]*w[(k*d + q)*d + j + a]
                    A[row + nrow*nvar] = value

                    # U[k] S[k] contributes S[k]^T kron I_m.
                    for b in range(s):
                        col = s*offsets[k] + b*counts[k] + u
                        A[row + nrow*col] += w[(k*d + j + b)*d + j + a]

                    # Copy H[k][active[k], active[k+1]] into its cyclic block
                    # of I_s kron H[k].  This assembles the operator; it does
                    # not evaluate H[k] U[k+1].
                    for z in range(counts[kp]):
                        ell = indices[offsets[kp] + z]
                        col = s*offsets[kp] + a*counts[kp] + z
                        A[row + nrow*col] += H[(k*m + i)*m + ell]

                # Copy residual_r[k] E_tail into the GMRES tail panel.  This
                # is also operator assembly, not a GEMM.
                for i in range(d):
                    row = nvar + (k*s + a)*d + i
                    for z in range(counts[kp]):
                        ell = indices[offsets[kp] + z]
                        if ell >= m - d:
                            col = s*offsets[kp] + a*counts[kp] + z
                            A[row + nrow*col] += residual_r[(k*d + i)*d + ell - (m - d)]

        lwork = -1
        lapack.dgeqrf(&nrow, &naug, A, &lda, tau, &work_query, &lwork, &info)
        if info != 0:
            free(A)
            free(tau)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return 5000 + info
        lwork = <int>work_query
        if lwork < 1:
            lwork = 1
        work = <double*>malloc(lwork*sizeof(double))
        if work == NULL:
            free(A)
            free(tau)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return -1
        lapack.dgeqrf(&nrow, &naug, A, &lda, tau, work, &lwork, &info)
        free(work)
        work = NULL
        free(tau)
        tau = NULL
        if info != 0:
            free(A)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return 6000 + info
        lapack.dtrtrs(
            &upper, &no_trans, &non_unit, &nvar, &nrhs,
            A, &lda, &A[nrow*nvar], &ldb, &info,
        )
        if info != 0:
            free(A)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return 6500 + info
        feclearexcept(FE_ALL_EXCEPT)

        for k in range(period):
            for a in range(s):
                for u in range(counts[k]):
                    i = indices[offsets[k] + u]
                    col = s*offsets[k] + a*counts[k] + u
                    Y[(k*m + i)*d + j + a] = A[nrow*nvar + col]
        free(A)
        A = NULL
        j += s

    # Return x[k] = Y[k].T and one residual norm per site and column.
    for k in range(period):
        kp = (k + 1) % period
        for j in range(rank):
            error_squared = 0.0
            for u in range(counts[k]):
                i = indices[offsets[k] + u]
                core = -v[(k*d + i)*d + j] if i < d else 0.0
                for z in range(counts[kp]):
                    ell = indices[offsets[kp] + z]
                    core += H[(k*m + i)*m + ell]*Y[(kp*m + ell)*d + j]
                for q in range(rank):
                    core += Y[(k*m + i)*d + q]*w[(k*d + q)*d + j]
                error_squared += core*core
                x[(k*d + j)*m + i] = Y[(k*m + i)*d + j]
            for i in range(d):
                tail = 0.0
                for z in range(counts[kp]):
                    ell = indices[offsets[kp] + z]
                    if ell >= m - d:
                        tail += residual_r[(k*d + i)*d + ell - (m - d)]*Y[(kp*m + ell)*d + j]
                error_squared += tail*tail
            error[k*d + j] = sqrt(error_squared)

    free(Y)
    free(counts)
    free(offsets)
    free(indices)
    return 0


cdef public int sylvester_compressed_periodic_dense_gmres_D(
    const double* H,
    const double* w,
    const double* v,
    const double* residual_r,
    const unsigned char* block_2x2_start,
    const unsigned char* active_cols,
    int period,
    int n_krylov,
    int d_block,
    int rank,
    double* x,
    double* error,
) noexcept nogil:
    r"""Solve the active real periodic GMRES equation in float64.

    In selection-matrix notation the minimized objective is

    ``sum_k ||E[k].T(H[k]Y[k+1]+Y[k]W[k]-beta[k])||_F^2``
    ``      + ||residual_r[k]Y[k+1]||_F^2``.

    This is the C-order XLA FFI entry point.  See
    :func:`_solve_periodic_real_active_dense_qr` for the definitions of
    ``E``, ``Y``, ``W``, and ``beta``, as well as shapes and status codes.
    """
    return _solve_periodic_real_active_dense_qr(
        H, w, v, residual_r, block_2x2_start, active_cols,
        period, n_krylov, d_block, rank, x, error,
    )


cdef int _solve_periodic_complex_active_dense_qr(
    const double complex* H,
    const double complex* w,
    const double complex* v,
    const double complex* residual_r,
    const unsigned char* active_cols,
    int period,
    int m,
    int d,
    int rank,
    double complex* x,
    double* error,
) noexcept nogil:
    r"""Minimize the rank-restricted complex periodic compressed residual.

    Let ``E[k]`` select the coordinates marked by ``active_cols[k]``, let
    ``Y[k] = E[k] Yhat[k]``, set ``beta[k, :d, :rank]`` from
    ``v[k, :, :rank]``, and write ``W[k] = w[k, :rank, :rank]``.  With cyclic
    site indices, this routine computes ``Yhat`` minimizing

    ``sum_k ||E[k].H (H[k] Y[k+1] + Y[k] W[k] - beta[k])||_F^2``
    ``      + ||residual_r[k] E_tail Y[k+1]||_F^2``.

    Here each ``E[k]`` is a real coordinate-selection matrix, so ``E[k].H``
    simply selects the active Galerkin rows.  The C-order inputs have shapes
    ``H: (p, m, m)``, ``w: (p, d, d)``, ``v: (p, d, d)``,
    ``residual_r: (p, d, d)``, and ``active_cols: (p, m)``.

    Every ``w[k]`` is assumed upper triangular.  Already solved columns are
    moved to the right-hand side, and each remaining scalar column is solved
    from a full cyclic augmented matrix using ``ZGEQRF`` and ``ZTRTRS``.
    ``x[k, j, :] = Y[k, :, j]`` has shape ``(p, d, m)`` and is zero outside
    active coordinates or in columns ``rank:``.  ``error[k, j]`` is the
    site-and-column norm of the two displayed residual terms.

    Return zero on success, ``-1`` on allocation failure, ``7000 + info`` or
    ``8000 + info`` on ``ZGEQRF`` query/factorization failure, and
    ``8500 + info`` on ``ZTRTRS`` failure.
    """
    cdef size_t y_size = period*m*d
    cdef double complex* Y = <double complex*>calloc(y_size, sizeof(double complex))
    cdef double complex* A = NULL
    cdef double complex* tau = NULL
    cdef double complex* work = NULL
    cdef double complex work_query
    cdef double complex value, core, tail
    cdef double error_squared
    cdef int* counts = <int*>malloc(period*sizeof(int))
    cdef int* offsets = <int*>malloc((period + 1)*sizeof(int))
    cdef int* indices = <int*>malloc(period*m*sizeof(int))
    cdef int nactive = 0
    cdef int j, nvar, nrow, naug, nrhs, lda, ldb, lwork, info
    cdef int k, kp, i, ell, q, row, col, u, z
    cdef char upper = 85
    cdef char no_trans = 78
    cdef char non_unit = 78

    if Y == NULL or counts == NULL or offsets == NULL or indices == NULL:
        free(Y)
        free(counts)
        free(offsets)
        free(indices)
        return -1

    for z in range(period*d*m):
        x[z] = 0.0
    for z in range(period*d):
        error[z] = 0.0
    for k in range(period):
        offsets[k] = nactive
        counts[k] = 0
        for i in range(m):
            if active_cols[k*m + i]:
                indices[nactive] = i
                nactive += 1
                counts[k] += 1
    offsets[period] = nactive
    if nactive == 0:
        free(Y)
        free(counts)
        free(offsets)
        free(indices)
        return 0

    nvar = nactive
    nrow = nvar + period*d
    naug = nvar + 1
    nrhs = 1
    lda = nrow
    ldb = nvar
    for j in range(rank):
        A = <double complex*>calloc(nrow*naug, sizeof(double complex))
        tau = <double complex*>malloc(naug*sizeof(double complex))
        if A == NULL or tau == NULL:
            free(A)
            free(tau)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return -1

        for k in range(period):
            kp = (k + 1) % period
            for u in range(counts[k]):
                i = indices[offsets[k] + u]
                row = offsets[k] + u
                value = v[(k*d + i)*d + j] if i < d else 0.0
                for q in range(rank):
                    value -= Y[(k*m + i)*d + q]*w[(k*d + q)*d + j]
                A[row + nrow*nvar] = value

                # w[k,j,j] I_m acts at site k.
                col = offsets[k] + u
                A[row + nrow*col] += w[(k*d + j)*d + j]

                # Copy H[k][active[k], active[k+1]] into the next-site block
                # of the cyclic operator; no H[k]Y[k+1] product is evaluated.
                for z in range(counts[kp]):
                    ell = indices[offsets[kp] + z]
                    col = offsets[kp] + z
                    A[row + nrow*col] += H[(k*m + i)*m + ell]

            # Copy residual_r[k] E_tail into the GMRES tail panel.
            for i in range(d):
                row = nvar + k*d + i
                for z in range(counts[kp]):
                    ell = indices[offsets[kp] + z]
                    if ell >= m - d:
                        col = offsets[kp] + z
                        A[row + nrow*col] += residual_r[(k*d + i)*d + ell - (m - d)]

        lwork = -1
        lapack.zgeqrf(&nrow, &naug, A, &lda, tau, &work_query, &lwork, &info)
        if info != 0:
            free(A)
            free(tau)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return 7000 + info
        lwork = <int>work_query.real
        if lwork < 1:
            lwork = 1
        work = <double complex*>malloc(lwork*sizeof(double complex))
        if work == NULL:
            free(A)
            free(tau)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return -1
        lapack.zgeqrf(&nrow, &naug, A, &lda, tau, work, &lwork, &info)
        free(work)
        work = NULL
        free(tau)
        tau = NULL
        if info != 0:
            free(A)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return 8000 + info
        lapack.ztrtrs(
            &upper, &no_trans, &non_unit, &nvar, &nrhs,
            A, &lda, &A[nrow*nvar], &ldb, &info,
        )
        if info != 0:
            free(A)
            free(Y)
            free(counts)
            free(offsets)
            free(indices)
            return 8500 + info
        feclearexcept(FE_ALL_EXCEPT)

        for k in range(period):
            for u in range(counts[k]):
                i = indices[offsets[k] + u]
                Y[(k*m + i)*d + j] = A[nrow*nvar + offsets[k] + u]
        free(A)
        A = NULL

    # Return x[k] = Y[k].T and one residual norm per site and column.
    for k in range(period):
        kp = (k + 1) % period
        for j in range(rank):
            error_squared = 0.0
            for u in range(counts[k]):
                i = indices[offsets[k] + u]
                core = -v[(k*d + i)*d + j] if i < d else 0.0
                for z in range(counts[kp]):
                    ell = indices[offsets[kp] + z]
                    core += H[(k*m + i)*m + ell]*Y[(kp*m + ell)*d + j]
                for q in range(rank):
                    core += Y[(k*m + i)*d + q]*w[(k*d + q)*d + j]
                error_squared += core.real*core.real + core.imag*core.imag
                x[(k*d + j)*m + i] = Y[(k*m + i)*d + j]
            for i in range(d):
                tail = 0.0
                for z in range(counts[kp]):
                    ell = indices[offsets[kp] + z]
                    if ell >= m - d:
                        tail += residual_r[(k*d + i)*d + ell - (m - d)]*Y[(kp*m + ell)*d + j]
                error_squared += tail.real*tail.real + tail.imag*tail.imag
            error[k*d + j] = sqrt(error_squared)

    free(Y)
    free(counts)
    free(offsets)
    free(indices)
    return 0


cdef int _solve_periodic_schur_galerkin_coordinates_Z(
    const double complex* T,
    const double complex* w,
    const double complex* beta,
    const double complex* C,
    int period,
    int n,
    int d,
    int rank,
    double complex* U,
    double* error,
) noexcept nogil:
    r"""Solve the complex periodic-Schur Galerkin equation.

    ``T[k] U[k+1] + U[k] w[k] = beta[k]`` is solved by traversing
    upper-triangular columns of ``w`` and rows of ``T``. Each diagonal
    subproblem is one dense cyclic system of order ``period`` solved by
    ``ZGESV``. ``C[k] U[k+1]`` enters only the returned residual norm.
    """
    cdef double complex* local_matrix = <double complex*>malloc(
        period*period*sizeof(double complex),
    )
    cdef double complex* local_rhs = <double complex*>malloc(
        period*sizeof(double complex),
    )
    cdef int* ipiv = <int*>malloc(period*sizeof(int))
    cdef double complex value, core, tail
    cdef double error_squared
    cdef int nrhs = 1, info = 0
    cdef int j, k, kp, i, ell, q, row, col, z

    if local_matrix == NULL or local_rhs == NULL or ipiv == NULL:
        free(local_matrix)
        free(local_rhs)
        free(ipiv)
        return -1

    for z in range(period*n*rank):
        U[z] = 0.0
    for z in range(period*d):
        error[z] = 0.0

    for j in range(rank):
        for i in range(n - 1, -1, -1):
            for z in range(period*period):
                local_matrix[z] = 0.0
            for k in range(period):
                kp = (k + 1) % period
                row = k
                value = beta[(k*n + i)*rank + j]
                for q in range(j):
                    value -= U[(k*n + i)*rank + q]*w[(k*d + q)*d + j]
                for ell in range(i + 1, n):
                    value -= T[(k*n + i)*n + ell]*U[
                        (kp*n + ell)*rank + j
                    ]
                local_rhs[row] = value
                col = kp
                local_matrix[row + period*col] += T[
                    (k*n + i)*n + i
                ]
                col = k
                local_matrix[row + period*col] += w[
                    (k*d + j)*d + j
                ]

            lapack.zgesv(
                &period, &nrhs, local_matrix, &period,
                ipiv, local_rhs, &period, &info,
            )
            if info != 0:
                info = 12000 + info
                break
            feclearexcept(FE_ALL_EXCEPT)
            for k in range(period):
                U[(k*n + i)*rank + j] = local_rhs[k]
        if info != 0:
            break

    if info == 0:
        for k in range(period):
            kp = (k + 1) % period
            for j in range(rank):
                error_squared = 0.0
                for i in range(n):
                    core = -beta[(k*n + i)*rank + j]
                    for ell in range(n):
                        core += T[(k*n + i)*n + ell]*U[
                            (kp*n + ell)*rank + j
                        ]
                    for q in range(rank):
                        core += U[(k*n + i)*rank + q]*w[
                            (k*d + q)*d + j
                        ]
                    error_squared += (
                        core.real*core.real + core.imag*core.imag
                    )
                for i in range(d):
                    tail = 0.0
                    for ell in range(n):
                        tail += C[(k*d + i)*n + ell]*U[
                            (kp*n + ell)*rank + j
                        ]
                    error_squared += (
                        tail.real*tail.real + tail.imag*tail.imag
                    )
                error[k*d + j] = sqrt(error_squared)

    free(local_matrix)
    free(local_rhs)
    free(ipiv)
    return info


cdef public int sylvester_compressed_periodic_dense_gmres_Z(
    const void* H_raw,
    const void* w_raw,
    const void* v_raw,
    const void* residual_r_raw,
    const unsigned char* active_cols,
    int period,
    int n_krylov,
    int d_block,
    int rank,
    void* x_raw,
    double* error,
) noexcept nogil:
    r"""Solve the active complex periodic GMRES equation in complex128.

    In selection-matrix notation the minimized objective is

    ``sum_k ||E[k].H(H[k]Y[k+1]+Y[k]W[k]-beta[k])||_F^2``
    ``      + ||residual_r[k]Y[k+1]||_F^2``.

    This is the C-order XLA FFI entry point.  ``x_raw`` receives the embedded
    coefficients ``x[k] = Y[k].T`` and ``error`` receives real float64 norms.
    See :func:`_solve_periodic_complex_active_dense_qr` for definitions,
    shapes, triangular assumptions, and status codes.  ``void`` pointers avoid
    nonportable exported C declarations for complex pointer types.
    """
    return _solve_periodic_complex_active_dense_qr(
        <const double complex*>H_raw,
        <const double complex*>w_raw,
        <const double complex*>v_raw,
        <const double complex*>residual_r_raw,
        active_cols, period, n_krylov, d_block, rank,
        <double complex*>x_raw, error,
    )


cdef int _solve_periodic_schur_block_D(
    const double* T,
    const double* w,
    const double* beta,
    const double* C,
    int period,
    int n,
    int d,
    int rank,
    int j,
    int s,
    double* U,
) noexcept nogil:
    r"""Solve one real ``w[0]`` block in periodic Schur coordinates.

    ``T``, ``w``, ``beta``, ``C``, and ``U`` use C storage with shapes
    ``(p,n,n)``, ``(p,d,d)``, ``(p,n,rank)``,
    ``(p,d,n)``, and ``(p,n,rank)``.  The square cyclic core is made upper
    triangular by QR factorizing its at-most-``4*p`` Schur-row diagonal
    blocks.  ``DTPQRT`` then adds the dense Arnoldi-tail rows.
    """
    cdef int N = period*n*s
    cdef int M = period*d*s
    cdef int NB = 32 if N > 32 else N
    cdef int L_zero = 0
    cdef int nrhs = 1
    cdef int lda = N
    cdef int ldb = M
    cdef int ldt = NB
    cdef int lwork = 64*N if N > 0 else 1
    cdef int info = 0
    cdef int i, ell, k, kp, a, b, q, row, col
    cdef int h, local_n, start, right, local_col, local_row
    cdef double value
    cdef char left = 76
    cdef char trans = 84
    cdef char upper = 85
    cdef char no_trans = 78
    cdef char non_unit = 78
    cdef double* K = <double*>calloc(N*N, sizeof(double))
    cdef double* B = <double*>calloc(M*N, sizeof(double))
    cdef double* rhs_top = <double*>calloc(N, sizeof(double))
    cdef double* rhs_bottom = <double*>calloc(M, sizeof(double))
    cdef double* tau = <double*>malloc((4*period if period else 1)*sizeof(double))
    cdef double* work = <double*>malloc(lwork*sizeof(double))
    cdef double* reflectors = <double*>calloc(NB*N, sizeof(double))
    cdef double* tp_work = <double*>calloc(NB*N, sizeof(double))
    cdef double* apply_work = <double*>calloc(NB, sizeof(double))

    if (K == NULL or B == NULL or rhs_top == NULL or rhs_bottom == NULL or
            tau == NULL or work == NULL or reflectors == NULL or
            tp_work == NULL or apply_work == NULL):
        free(K)
        free(B)
        free(rhs_top)
        free(rhs_bottom)
        free(tau)
        free(work)
        free(reflectors)
        free(tp_work)
        free(apply_work)
        return -1

    # K maps the site-stacked vec(U[:, j:j+s]) to the periodic core residual.
    for i in range(n):
        for k in range(period):
            kp = (k + 1) % period
            for a in range(s):
                row = (i*period + k)*s + a
                value = beta[(k*n + i)*rank + j + a]
                for q in range(j):
                    value -= U[(k*n + i)*rank + q]*w[(k*d + q)*d + j + a]
                rhs_top[row] = value
                # T[0] may contain the one real-Schur subdiagonal belonging
                # to this row's 2-by-2 block; all earlier entries are exact
                # zero, so copying the full row keeps that coupling explicit.
                for ell in range(n):
                    col = (ell*period + kp)*s + a
                    K[row + N*col] += T[(k*n + i)*n + ell]
                for b in range(s):
                    col = (i*period + k)*s + b
                    K[row + N*col] += w[(k*d + j + b)*d + j + a]

    # The Arnoldi tail is block diagonal in the site index after the cyclic
    # next-site permutation.
    for k in range(period):
        kp = (k + 1) % period
        for a in range(s):
            for q in range(d):
                row = (k*s + a)*d + q
                for ell in range(n):
                    col = (ell*period + kp)*s + a
                    B[row + M*col] = C[(k*d + q)*n + ell]

    # Group the real 2-by-2 blocks of T[0].  In this row-block ordering the
    # cyclic core is block upper triangular, with diagonal blocks of size
    # period*h*s.
    i = 0
    while i < n:
        h = 2 if i + 1 < n and T[(i + 1)*n + i] != 0.0 else 1
        local_n = period*h*s
        start = period*i*s
        lapack.dgeqrf(
            &local_n, &local_n, &K[start + N*start], &lda,
            tau, work, &lwork, &info,
        )
        if info != 0:
            info = 9000 + info
            break
        right = N - start - local_n
        if right:
            lapack.dormqr(
                &left, &trans, &local_n, &right, &local_n,
                &K[start + N*start], &lda, tau,
                &K[start + N*(start + local_n)], &lda,
                work, &lwork, &info,
            )
            if info != 0:
                info = 9100 + info
                break
        lapack.dormqr(
            &left, &trans, &local_n, &nrhs, &local_n,
            &K[start + N*start], &lda, tau,
            &rhs_top[start], &lda, work, &lwork, &info,
        )
        if info != 0:
            info = 9200 + info
            break
        for local_col in range(local_n):
            for local_row in range(local_col + 1, local_n):
                K[start + local_row + N*(start + local_col)] = 0.0
        i += h

    if info == 0:
        lapack.dtpqrt(
            &M, &N, &L_zero, &NB,
            K, &lda, B, &ldb, reflectors, &ldt, tp_work, &info,
        )
        if info != 0:
            info = 9300 + info
    if info == 0:
        lapack.dtpmqrt(
            &left, &trans, &M, &nrhs, &N, &L_zero, &NB,
            B, &ldb, reflectors, &ldt,
            rhs_top, &lda, rhs_bottom, &ldb, apply_work, &info,
        )
        if info != 0:
            info = 9400 + info
    if info == 0:
        lapack.dtrtrs(
            &upper, &no_trans, &non_unit, &N, &nrhs,
            K, &lda, rhs_top, &lda, &info,
        )
        if info != 0:
            info = 9500 + info
    if info == 0:
        feclearexcept(FE_ALL_EXCEPT)
        for i in range(n):
            for k in range(period):
                for a in range(s):
                    row = (i*period + k)*s + a
                    U[(k*n + i)*rank + j + a] = rhs_top[row]

    free(K)
    free(B)
    free(rhs_top)
    free(rhs_bottom)
    free(tau)
    free(work)
    free(reflectors)
    free(tp_work)
    free(apply_work)
    return info


cdef int _solve_periodic_schur_coordinates_D(
    const double* T,
    const double* w,
    const double* beta,
    const double* C,
    const unsigned char* block_2x2_start,
    int period,
    int n,
    int d,
    int rank,
    double* U,
    double* error,
) noexcept nogil:
    r"""Sweep real ``w[0]`` blocks through a periodic-Schur problem."""
    cdef int j = 0
    cdef int s, info, i, ell, k, kp, q
    cdef double core, tail, error_squared

    for i in range(period*n*rank):
        U[i] = 0.0
    for i in range(period*d):
        error[i] = 0.0
    while j < rank:
        s = 2 if block_2x2_start[j] else 1
        info = _solve_periodic_schur_block_D(
            T, w, beta, C, period, n, d, rank, j, s, U,
        )
        if info != 0:
            return info
        j += s

    for k in range(period):
        kp = (k + 1) % period
        for j in range(rank):
            error_squared = 0.0
            for i in range(n):
                core = -beta[(k*n + i)*rank + j]
                for ell in range(n):
                    core += T[(k*n + i)*n + ell]*U[(kp*n + ell)*rank + j]
                for q in range(rank):
                    core += U[(k*n + i)*rank + q]*w[(k*d + q)*d + j]
                error_squared += core*core
            for i in range(d):
                tail = 0.0
                for ell in range(n):
                    tail += C[(k*d + i)*n + ell]*U[(kp*n + ell)*rank + j]
                error_squared += tail*tail
            error[k*d + j] = sqrt(error_squared)
    return 0


cdef int _solve_periodic_schur_galerkin_coordinates_D(
    const double* T,
    const double* w,
    const double* beta,
    const double* C,
    const unsigned char* block_2x2_start,
    int period,
    int n,
    int d,
    int rank,
    int use_mb03ke,
    double* U,
    double* error,
) noexcept nogil:
    r"""Solve the periodic-Schur Galerkin equation by block substitution.

    ``T[k] U[k+1] + U[k] w[k] = beta[k]`` is swept first through the
    quasi-triangular blocks of ``w[0]`` and then backward through those of
    ``T[0]``.  Each diagonal subproblem is a dense cyclic system of order at
    most ``4*period``. ``C[k] U[k+1]`` is evaluated only for the returned true
    residual norms.
    """
    cdef int max_local = 4*period if period else 1
    cdef int mb03ke_lwork = (
        (4*period - 3)*16 + 4*period
        if period else 1
    )
    cdef double* local_matrix = NULL
    cdef double* local_rhs = NULL
    cdef int* ipiv = NULL
    cdef double* mb03ke_A = NULL
    cdef double* mb03ke_B = NULL
    cdef double* mb03ke_C = NULL
    cdef double* mb03ke_work = NULL
    cdef int* mb03ke_signs = NULL
    cdef int j = 0
    cdef int i_end, i0, h, s, L, nrhs = 1, info = 0
    cdef int k, kp, r, ell_local, a, b, q, ell, row, col, z
    cdef double value, core, tail, error_squared, scale

    if use_mb03ke:
        mb03ke_A = <double*>malloc(4*period*sizeof(double))
        mb03ke_B = <double*>malloc(4*period*sizeof(double))
        mb03ke_C = <double*>malloc(4*period*sizeof(double))
        mb03ke_work = <double*>malloc(mb03ke_lwork*sizeof(double))
        mb03ke_signs = <int*>malloc(period*sizeof(int))
    else:
        local_matrix = <double*>malloc(max_local*max_local*sizeof(double))
        local_rhs = <double*>malloc(max_local*sizeof(double))
        ipiv = <int*>malloc(max_local*sizeof(int))

    if (
        (use_mb03ke and (
            mb03ke_A == NULL or mb03ke_B == NULL or mb03ke_C == NULL or
            mb03ke_work == NULL or mb03ke_signs == NULL
        ))
        or
        (not use_mb03ke and (
            local_matrix == NULL or local_rhs == NULL or ipiv == NULL
        ))
    ):
        free(local_matrix)
        free(local_rhs)
        free(ipiv)
        free(mb03ke_A)
        free(mb03ke_B)
        free(mb03ke_C)
        free(mb03ke_work)
        free(mb03ke_signs)
        return -1
    for k in range(period):
        if use_mb03ke:
            mb03ke_signs[k] = 1
    for z in range(period*n*rank):
        U[z] = 0.0
    for z in range(period*d):
        error[z] = 0.0

    while j < rank:
        s = 2 if block_2x2_start[j] else 1
        i_end = n
        while i_end > 0:
            if i_end >= 2 and T[(i_end - 1)*n + i_end - 2] != 0.0:
                i0 = i_end - 2
                h = 2
            else:
                i0 = i_end - 1
                h = 1
            L = period*h*s
            if not use_mb03ke:
                for z in range(L*L):
                    local_matrix[z] = 0.0
            for k in range(period):
                kp = (k + 1) % period
                if use_mb03ke:
                    for a in range(s):
                        for b in range(s):
                            # A[k] = w[k, J, J] in Fortran storage.
                            mb03ke_A[k*s*s + a + s*b] = w[
                                (k*d + j + a)*d + j + b
                            ]
                    for r in range(h):
                        for ell_local in range(h):
                            # B[k] = T[k, I, I] in Fortran storage.
                            mb03ke_B[k*h*h + r + h*ell_local] = T[
                                (k*n + i0 + r)*n + i0 + ell_local
                            ]
                for r in range(h):
                    for a in range(s):
                        row = (k*h + r)*s + a
                        value = beta[(k*n + i0 + r)*rank + j + a]
                        for q in range(j):
                            value -= U[(k*n + i0 + r)*rank + q]*w[(k*d + q)*d + j + a]
                        for ell in range(i0 + h, n):
                            value -= T[(k*n + i0 + r)*n + ell]*U[(kp*n + ell)*rank + j + a]
                        if use_mb03ke:
                            # C[k] = -D[k].T, with M=s and N=h.
                            mb03ke_C[k*s*h + a + s*r] = -value
                        else:
                            local_rhs[row] = value
                            for ell_local in range(h):
                                col = (kp*h + ell_local)*s + a
                                local_matrix[row + L*col] += T[
                                    (k*n + i0 + r)*n + i0 + ell_local
                                ]
                            for b in range(s):
                                col = (k*h + r)*s + b
                                local_matrix[row + L*col] += w[
                                    (k*d + j + b)*d + j + a
                                ]
            if use_mb03ke:
                scale = 1.0
                info = slicot_mb03ke_D(
                    1, 1, 1, period, s, h, mb03ke_signs,
                    mb03ke_A, mb03ke_B, mb03ke_C, &scale,
                    mb03ke_work, mb03ke_lwork,
                )
                if info == 0 and scale != 1.0:
                    info = 1
                if info != 0:
                    info = 12500 + info
                    break
                for k in range(period):
                    for r in range(h):
                        for a in range(s):
                            U[(k*n + i0 + r)*rank + j + a] = (
                                mb03ke_C[k*s*h + a + s*r]
                            )
            else:
                lapack.dgesv(
                    &L, &nrhs, local_matrix, &L, ipiv, local_rhs, &L, &info,
                )
                if info != 0:
                    info = 12000 + info
                    break
                for k in range(period):
                    for r in range(h):
                        for a in range(s):
                            row = (k*h + r)*s + a
                            U[(k*n + i0 + r)*rank + j + a] = local_rhs[row]
            i_end = i0
        if info != 0:
            break
        j += s

    if info == 0:
        for k in range(period):
            kp = (k + 1) % period
            for j in range(rank):
                error_squared = 0.0
                for r in range(n):
                    core = -beta[(k*n + r)*rank + j]
                    for ell in range(n):
                        core += T[(k*n + r)*n + ell]*U[(kp*n + ell)*rank + j]
                    for q in range(rank):
                        core += U[(k*n + r)*rank + q]*w[(k*d + q)*d + j]
                    error_squared += core*core
                for r in range(d):
                    tail = 0.0
                    for ell in range(n):
                        tail += C[(k*d + r)*n + ell]*U[(kp*n + ell)*rank + j]
                    error_squared += tail*tail
                error[k*d + j] = sqrt(error_squared)

    free(local_matrix)
    free(local_rhs)
    free(ipiv)
    free(mb03ke_A)
    free(mb03ke_B)
    free(mb03ke_C)
    free(mb03ke_work)
    free(mb03ke_signs)
    return info


cdef int _solve_periodic_schur_block_Z(
    const double complex* T,
    const double complex* w,
    const double complex* beta,
    const double complex* C,
    int period,
    int n,
    int d,
    int rank,
    int j,
    double complex* U,
) noexcept nogil:
    r"""Solve one complex column in periodic Schur coordinates."""
    cdef int N = period*n
    cdef int M = period*d
    cdef int NB = 32 if N > 32 else N
    cdef int L_zero = 0
    cdef int nrhs = 1
    cdef int lda = N
    cdef int ldb = M
    cdef int ldt = NB
    cdef int lwork = 64*N if N > 0 else 1
    cdef int info = 0
    cdef int i, ell, k, kp, q, row, col
    cdef int local_n, start, right, local_col, local_row
    cdef double complex value
    cdef char left = 76
    cdef char adjoint = 67
    cdef char upper = 85
    cdef char no_trans = 78
    cdef char non_unit = 78
    cdef double complex* K = <double complex*>calloc(N*N, sizeof(double complex))
    cdef double complex* B = <double complex*>calloc(M*N, sizeof(double complex))
    cdef double complex* rhs_top = <double complex*>calloc(N, sizeof(double complex))
    cdef double complex* rhs_bottom = <double complex*>calloc(M, sizeof(double complex))
    cdef double complex* tau = <double complex*>malloc((period if period else 1)*sizeof(double complex))
    cdef double complex* work = <double complex*>malloc(lwork*sizeof(double complex))
    cdef double complex* reflectors = <double complex*>calloc(NB*N, sizeof(double complex))
    cdef double complex* tp_work = <double complex*>calloc(NB*N, sizeof(double complex))
    cdef double complex* apply_work = <double complex*>calloc(NB, sizeof(double complex))

    if (K == NULL or B == NULL or rhs_top == NULL or rhs_bottom == NULL or
            tau == NULL or work == NULL or reflectors == NULL or
            tp_work == NULL or apply_work == NULL):
        free(K)
        free(B)
        free(rhs_top)
        free(rhs_bottom)
        free(tau)
        free(work)
        free(reflectors)
        free(tp_work)
        free(apply_work)
        return -1

    for i in range(n):
        for k in range(period):
            kp = (k + 1) % period
            row = i*period + k
            value = beta[(k*n + i)*rank + j]
            for q in range(j):
                value -= U[(k*n + i)*rank + q]*w[(k*d + q)*d + j]
            rhs_top[row] = value
            for ell in range(i, n):
                col = ell*period + kp
                K[row + N*col] += T[(k*n + i)*n + ell]
            col = i*period + k
            K[row + N*col] += w[(k*d + j)*d + j]

    for k in range(period):
        kp = (k + 1) % period
        for q in range(d):
            row = k*d + q
            for ell in range(n):
                col = ell*period + kp
                B[row + M*col] = C[(k*d + q)*n + ell]

    for i in range(n):
        local_n = period
        start = period*i
        lapack.zgeqrf(
            &local_n, &local_n, &K[start + N*start], &lda,
            tau, work, &lwork, &info,
        )
        if info != 0:
            info = 10000 + info
            break
        right = N - start - local_n
        if right:
            lapack.zunmqr(
                &left, &adjoint, &local_n, &right, &local_n,
                &K[start + N*start], &lda, tau,
                &K[start + N*(start + local_n)], &lda,
                work, &lwork, &info,
            )
            if info != 0:
                info = 10100 + info
                break
        lapack.zunmqr(
            &left, &adjoint, &local_n, &nrhs, &local_n,
            &K[start + N*start], &lda, tau,
            &rhs_top[start], &lda, work, &lwork, &info,
        )
        if info != 0:
            info = 10200 + info
            break
        for local_col in range(local_n):
            for local_row in range(local_col + 1, local_n):
                K[start + local_row + N*(start + local_col)] = 0.0

    if info == 0:
        lapack.ztpqrt(
            &M, &N, &L_zero, &NB,
            K, &lda, B, &ldb, reflectors, &ldt, tp_work, &info,
        )
        if info != 0:
            info = 10300 + info
    if info == 0:
        lapack.ztpmqrt(
            &left, &adjoint, &M, &nrhs, &N, &L_zero, &NB,
            B, &ldb, reflectors, &ldt,
            rhs_top, &lda, rhs_bottom, &ldb, apply_work, &info,
        )
        if info != 0:
            info = 10400 + info
    if info == 0:
        lapack.ztrtrs(
            &upper, &no_trans, &non_unit, &N, &nrhs,
            K, &lda, rhs_top, &lda, &info,
        )
        if info != 0:
            info = 10500 + info
    if info == 0:
        feclearexcept(FE_ALL_EXCEPT)
        for i in range(n):
            for k in range(period):
                row = i*period + k
                U[(k*n + i)*rank + j] = rhs_top[row]

    free(K)
    free(B)
    free(rhs_top)
    free(rhs_bottom)
    free(tau)
    free(work)
    free(reflectors)
    free(tp_work)
    free(apply_work)
    return info


cdef int _solve_periodic_schur_coordinates_Z(
    const double complex* T,
    const double complex* w,
    const double complex* beta,
    const double complex* C,
    int period,
    int n,
    int d,
    int rank,
    double complex* U,
    double* error,
) noexcept nogil:
    r"""Sweep complex columns through a periodic-Schur problem."""
    cdef int j, info, i, ell, k, kp, q
    cdef double complex core, tail
    cdef double error_squared

    for i in range(period*n*rank):
        U[i] = 0.0
    for i in range(period*d):
        error[i] = 0.0
    for j in range(rank):
        info = _solve_periodic_schur_block_Z(
            T, w, beta, C, period, n, d, rank, j, U,
        )
        if info != 0:
            return info

    for k in range(period):
        kp = (k + 1) % period
        for j in range(rank):
            error_squared = 0.0
            for i in range(n):
                core = -beta[(k*n + i)*rank + j]
                for ell in range(n):
                    core += T[(k*n + i)*n + ell]*U[(kp*n + ell)*rank + j]
                for q in range(rank):
                    core += U[(k*n + i)*rank + q]*w[(k*d + q)*d + j]
                error_squared += core.real*core.real + core.imag*core.imag
            for i in range(d):
                tail = 0.0
                for ell in range(n):
                    tail += C[(k*d + i)*n + ell]*U[(kp*n + ell)*rank + j]
                error_squared += tail.real*tail.real + tail.imag*tail.imag
            error[k*d + j] = sqrt(error_squared)
    return 0


cdef void _periodic_compressed_error_D(
    const double* H,
    const double* w,
    const double* v,
    const double* residual_r,
    const unsigned char* active_cols,
    int period,
    int m,
    int d,
    int rank,
    const double* x,
    double* error,
) noexcept nogil:
    r"""Evaluate the real compressed residual against the original factors."""
    cdef int k, kp, i, ell, j, q
    cdef double core, tail, error_squared

    for i in range(period*d):
        error[i] = 0.0
    for k in range(period):
        kp = (k + 1) % period
        for j in range(rank):
            error_squared = 0.0
            for i in range(m):
                if active_cols[k*m + i]:
                    core = -v[(k*d + i)*d + j] if i < d else 0.0
                    for ell in range(m):
                        core += (
                            H[(k*m + i)*m + ell]
                            * x[(kp*d + j)*m + ell]
                        )
                    for q in range(rank):
                        core += (
                            x[(k*d + q)*m + i]
                            * w[(k*d + q)*d + j]
                        )
                    error_squared += core*core
            for i in range(d):
                tail = 0.0
                for ell in range(d):
                    tail += (
                        residual_r[(k*d + i)*d + ell]
                        * x[(kp*d + j)*m + m - d + ell]
                    )
                error_squared += tail*tail
            error[k*d + j] = sqrt(error_squared)


cdef void _periodic_compressed_error_Z(
    const double complex* H,
    const double complex* w,
    const double complex* v,
    const double complex* residual_r,
    const unsigned char* active_cols,
    int period,
    int m,
    int d,
    int rank,
    const double complex* x,
    double* error,
) noexcept nogil:
    r"""Evaluate the complex compressed residual against original factors."""
    cdef int k, kp, i, ell, j, q
    cdef double complex core, tail
    cdef double error_squared

    for i in range(period*d):
        error[i] = 0.0
    for k in range(period):
        kp = (k + 1) % period
        for j in range(rank):
            error_squared = 0.0
            for i in range(m):
                if active_cols[k*m + i]:
                    core = -v[(k*d + i)*d + j] if i < d else 0.0
                    for ell in range(m):
                        core += (
                            H[(k*m + i)*m + ell]
                            * x[(kp*d + j)*m + ell]
                        )
                    for q in range(rank):
                        core += (
                            x[(k*d + q)*m + i]
                            * w[(k*d + q)*d + j]
                        )
                    error_squared += (
                        core.real*core.real + core.imag*core.imag
                    )
            for i in range(d):
                tail = 0.0
                for ell in range(d):
                    tail += (
                        residual_r[(k*d + i)*d + ell]
                        * x[(kp*d + j)*m + m - d + ell]
                    )
                error_squared += (
                    tail.real*tail.real + tail.imag*tail.imag
                )
            error[k*d + j] = sqrt(error_squared)


cdef int _periodic_schur_scaled_with_exact_retry_D(
    const double* H,
    const unsigned char* active_cols,
    const double* scale_tol,
    int period,
    int m,
    int n,
    double* T,
    double* Z,
    double* wr,
    double* wi,
) noexcept nogil:
    """Retry the original real Hessenberg problem after a scaled Schur stall."""
    cdef int info = compute_periodic_schur_active_scaled_D(
        H, active_cols, scale_tol, 10.0,
        period, m, n, n, T, Z, wr, wi,
    )

    if 3000 <= info < 4000:
        info = compute_periodic_schur_active_D(
            H, active_cols, 10.0,
            period, m, n, n, T, Z, wr, wi,
        )
    return info


cdef int _periodic_schur_scaled_with_exact_retry_Z(
    const double complex* H,
    const unsigned char* active_cols,
    const double* scale_tol,
    int period,
    int m,
    int n,
    double complex* T,
    double complex* Z,
    double complex* alpha,
    double complex* beta,
    int* scale,
) noexcept nogil:
    """Retry the original complex Hessenberg problem after a scaled stall."""
    cdef int info = compute_periodic_schur_active_scaled_Z(
        H, active_cols, scale_tol, period, m, n, n,
        T, Z, alpha, beta, scale,
    )

    if 4000 <= info < 5000:
        info = compute_periodic_schur_active_Z(
            H, active_cols, period, m, n, n,
            T, Z, alpha, beta, scale,
        )
    return info


cdef public int sylvester_compressed_periodic_schur_gmres_D(
    const double* H,
    const double* w,
    const double* v,
    const double* residual_r,
    const double* scale_tol,
    const unsigned char* block_2x2_start,
    const unsigned char* active_cols,
    int period,
    int m,
    int d,
    int rank,
    double* x,
    double* error,
) noexcept nogil:
    r"""Solve the real periodic compressed problem through periodic Schur.

    This C-order, GIL-free entry point has the same contract as dense GMRES.
    Equal active dimensions use the NRed periodic Schur factorization; unequal
    dimensions retain the exact dense active-coordinate fallback.
    """
    cdef int n = periodic_schur_active_size(active_cols, period, m)
    cdef int i, j, a, k, kp, count, info = 0
    cdef char no_trans = 78
    cdef char trans = 84
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef double* T = NULL
    cdef double* Z = NULL
    cdef double* wr = NULL
    cdef double* wi = NULL
    cdef double* beta = NULL
    cdef double* C = NULL
    cdef double* U = NULL

    for i in range(period*d*m):
        x[i] = 0.0
    for i in range(period*d):
        error[i] = 0.0
    if rank == 0 or n == 0:
        return 0
    for k in range(period):
        count = 0
        for i in range(m):
            count += 1 if active_cols[k*m + i] else 0
        if count != n:
            return _solve_periodic_real_active_dense_qr(
                H, w, v, residual_r, block_2x2_start, active_cols,
                period, m, d, rank, x, error,
            )

    T = <double*>malloc(period*n*n*sizeof(double))
    Z = <double*>calloc(period*m*n, sizeof(double))
    wr = <double*>malloc(n*sizeof(double))
    wi = <double*>malloc(n*sizeof(double))
    beta = <double*>malloc(period*n*rank*sizeof(double))
    C = <double*>malloc(period*d*n*sizeof(double))
    U = <double*>malloc(period*n*rank*sizeof(double))
    if (T == NULL or Z == NULL or wr == NULL or wi == NULL or beta == NULL or
            C == NULL or U == NULL):
        info = -1
    if info == 0:
        info = _periodic_schur_scaled_with_exact_retry_D(
            H, active_cols, scale_tol, period, m, n, T, Z, wr, wi,
        )
    if info == 0:
        for k in range(period):
            kp = (k + 1) % period
            # C-order beta is the column-major view beta.T = v.T Z[:d].
            blas.dgemm(
                &no_trans, &trans, &rank, &n, &d,
                &one, &(<double*>v)[k*d*d], &d, &Z[k*m*n], &n,
                &zero, &beta[k*n*rank], &rank,
            )
            # C-order C is the column-major view C.T = Z_tail.T residual.T.
            blas.dgemm(
                &no_trans, &no_trans, &n, &d, &d,
                &one, &Z[(kp*m + m - d)*n], &n,
                &(<double*>residual_r)[k*d*d], &d,
                &zero, &C[k*d*n], &n,
            )
        info = _solve_periodic_schur_coordinates_D(
            T, w, beta, C, block_2x2_start,
            period, n, d, rank, U, error,
        )
    if info == 0:
        for k in range(period):
            # The column-major output view is x.T = Z U.
            blas.dgemm(
                &trans, &trans, &m, &rank, &n,
                &one, &Z[k*m*n], &n, &U[k*n*rank], &rank,
                &zero, &x[k*d*m], &m,
            )
        _periodic_compressed_error_D(
            H, w, v, residual_r, active_cols,
            period, m, d, rank, x, error,
        )

    free(T)
    free(Z)
    free(wr)
    free(wi)
    free(beta)
    free(C)
    free(U)
    return info


cdef public int sylvester_compressed_periodic_schur_galerkin_D(
    const double* H,
    const double* w,
    const double* v,
    const double* residual_r,
    const double* scale_tol,
    const unsigned char* block_2x2_start,
    const unsigned char* active_cols,
    int period,
    int m,
    int d,
    int rank,
    int use_mb03ke,
    double* x,
    double* error,
) noexcept nogil:
    r"""Solve the real periodic compressed Galerkin equation via Schur.

    The active projected equation is solved exactly; unlike the least-squares
    Schur-GMRES solver, the Arnoldi tail ``residual_r`` enters only the returned
    true residual norm. Unequal active dimensions use the same square NRed
    route after zero padding to the largest active dimension. The nonsingular
    right factors force the artificial solution coordinates to zero.
    """
    cdef int n = periodic_schur_active_size(active_cols, period, m)
    cdef int i, k, kp, info = 0
    cdef char no_trans = 78
    cdef char trans = 84
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef double* T = NULL
    cdef double* Z = NULL
    cdef double* wr = NULL
    cdef double* wi = NULL
    cdef double* beta = NULL
    cdef double* C = NULL
    cdef double* U = NULL

    for i in range(period*d*m):
        x[i] = 0.0
    for i in range(period*d):
        error[i] = 0.0
    if rank == 0 or n == 0:
        return 0

    # TODO: If allocation profiling warrants it, consolidate these buffers into
    # one per-call slab or a caller-managed reusable native workspace.
    T = <double*>malloc(period*n*n*sizeof(double))
    Z = <double*>calloc(period*m*n, sizeof(double))
    wr = <double*>malloc(n*sizeof(double))
    wi = <double*>malloc(n*sizeof(double))
    beta = <double*>malloc(period*n*rank*sizeof(double))
    C = <double*>malloc(period*d*n*sizeof(double))
    U = <double*>malloc(period*n*rank*sizeof(double))
    if (T == NULL or Z == NULL or wr == NULL or wi == NULL or beta == NULL or
            C == NULL or U == NULL):
        info = -1
    if info == 0:
        info = _periodic_schur_scaled_with_exact_retry_D(
            H, active_cols, scale_tol, period, m, n, T, Z, wr, wi,
        )
    if info == 0:
        for k in range(period):
            kp = (k + 1) % period
            # beta[k] = Z[k].T beta_original[k].
            blas.dgemm(
                &no_trans, &trans, &rank, &n, &d,
                &one, &(<double*>v)[k*d*d], &d, &Z[k*m*n], &n,
                &zero, &beta[k*n*rank], &rank,
            )
            # C[k] = residual_r[k] E_tail Z[k+1].
            blas.dgemm(
                &no_trans, &no_trans, &n, &d, &d,
                &one, &Z[(kp*m + m - d)*n], &n,
                &(<double*>residual_r)[k*d*d], &d,
                &zero, &C[k*d*n], &n,
            )
        info = _solve_periodic_schur_galerkin_coordinates_D(
            T, w, beta, C, block_2x2_start,
            period, n, d, rank, use_mb03ke, U, error,
        )
    if info == 0:
        for k in range(period):
            # x[k].T = Z[k] U[k].
            blas.dgemm(
                &trans, &trans, &m, &rank, &n,
                &one, &Z[k*m*n], &n, &U[k*n*rank], &rank,
                &zero, &x[k*d*m], &m,
            )
        _periodic_compressed_error_D(
            H, w, v, residual_r, active_cols,
            period, m, d, rank, x, error,
        )

    free(T)
    free(Z)
    free(wr)
    free(wi)
    free(beta)
    free(C)
    free(U)
    return info


cdef public int sylvester_compressed_periodic_schur_galerkin_Z(
    const void* H_raw,
    const void* w_raw,
    const void* v_raw,
    const void* residual_r_raw,
    const double* scale_tol,
    const unsigned char* active_cols,
    int period,
    int m,
    int d,
    int rank,
    void* x_raw,
    double* error,
) noexcept nogil:
    r"""Solve the complex periodic compressed Galerkin equation via Schur.

    This is the complex analogue of
    ``sylvester_compressed_periodic_schur_galerkin_D``. Complex Schur factors
    are triangular throughout, so the coordinate solve needs only scalar
    Schur blocks and dense cyclic ``ZGESV`` solves of order ``period``.
    """
    cdef const double complex* H = <const double complex*>H_raw
    cdef const double complex* w = <const double complex*>w_raw
    cdef const double complex* v = <const double complex*>v_raw
    cdef const double complex* residual_r = <const double complex*>residual_r_raw
    cdef double complex* x = <double complex*>x_raw
    cdef int n = periodic_schur_active_size(active_cols, period, m)
    cdef int i, k, kp, info = 0
    cdef char no_trans = 78
    cdef char trans = 84
    cdef char adjoint = 67
    cdef double complex one = 1.0
    cdef double complex zero = 0.0
    cdef double complex* T = NULL
    cdef double complex* Z = NULL
    cdef double complex* alpha = NULL
    cdef double complex* eig_beta = NULL
    cdef int* scale = NULL
    cdef double complex* beta = NULL
    cdef double complex* C = NULL
    cdef double complex* U = NULL

    for i in range(period*d*m):
        x[i] = 0.0
    for i in range(period*d):
        error[i] = 0.0
    if rank == 0 or n == 0:
        return 0

    T = <double complex*>malloc(period*n*n*sizeof(double complex))
    Z = <double complex*>calloc(period*m*n, sizeof(double complex))
    alpha = <double complex*>malloc(n*sizeof(double complex))
    eig_beta = <double complex*>malloc(n*sizeof(double complex))
    scale = <int*>malloc(n*sizeof(int))
    beta = <double complex*>malloc(period*n*rank*sizeof(double complex))
    C = <double complex*>malloc(period*d*n*sizeof(double complex))
    U = <double complex*>malloc(period*n*rank*sizeof(double complex))
    if (T == NULL or Z == NULL or alpha == NULL or eig_beta == NULL or
            scale == NULL or beta == NULL or C == NULL or U == NULL):
        info = -1
    if info == 0:
        info = _periodic_schur_scaled_with_exact_retry_Z(
            H, active_cols, scale_tol, period, m, n,
            T, Z, alpha, eig_beta, scale,
        )
    if info == 0:
        for k in range(period):
            kp = (k + 1) % period
            # beta[k] = Z[k].H beta_original[k].
            blas.zgemm(
                &no_trans, &adjoint, &rank, &n, &d,
                &one, &(<double complex*>v)[k*d*d], &d, &Z[k*m*n], &n,
                &zero, &beta[k*n*rank], &rank,
            )
            # C[k] = residual_r[k] E_tail Z[k+1].
            blas.zgemm(
                &no_trans, &no_trans, &n, &d, &d,
                &one, &Z[(kp*m + m - d)*n], &n,
                &(<double complex*>residual_r)[k*d*d], &d,
                &zero, &C[k*d*n], &n,
            )
        info = _solve_periodic_schur_galerkin_coordinates_Z(
            T, w, beta, C, period, n, d, rank, U, error,
        )
    if info == 0:
        for k in range(period):
            # x[k].T = Z[k] U[k].
            blas.zgemm(
                &trans, &trans, &m, &rank, &n,
                &one, &Z[k*m*n], &n, &U[k*n*rank], &rank,
                &zero, &x[k*d*m], &m,
            )
        _periodic_compressed_error_Z(
            H, w, v, residual_r, active_cols,
            period, m, d, rank, x, error,
        )

    free(T)
    free(Z)
    free(alpha)
    free(eig_beta)
    free(scale)
    free(beta)
    free(C)
    free(U)
    return info


cdef public int sylvester_compressed_periodic_schur_gmres_Z(
    const void* H_raw,
    const void* w_raw,
    const void* v_raw,
    const void* residual_r_raw,
    const double* scale_tol,
    const unsigned char* active_cols,
    int period,
    int m,
    int d,
    int rank,
    void* x_raw,
    double* error,
) noexcept nogil:
    r"""Solve complex periodic compressed GMRES through periodic Schur."""
    cdef const double complex* H = <const double complex*>H_raw
    cdef const double complex* w = <const double complex*>w_raw
    cdef const double complex* v = <const double complex*>v_raw
    cdef const double complex* residual_r = <const double complex*>residual_r_raw
    cdef double complex* x = <double complex*>x_raw
    cdef int n = periodic_schur_active_size(active_cols, period, m)
    cdef int i, j, a, k, kp, count, info = 0
    cdef char no_trans = 78
    cdef char trans = 84
    cdef char adjoint = 67
    cdef double complex one = 1.0
    cdef double complex zero = 0.0
    cdef double complex* T = NULL
    cdef double complex* Z = NULL
    cdef double complex* alpha = NULL
    cdef double complex* eig_beta = NULL
    cdef int* scale = NULL
    cdef double complex* beta = NULL
    cdef double complex* C = NULL
    cdef double complex* U = NULL

    for i in range(period*d*m):
        x[i] = 0.0
    for i in range(period*d):
        error[i] = 0.0
    if rank == 0 or n == 0:
        return 0
    for k in range(period):
        count = 0
        for i in range(m):
            count += 1 if active_cols[k*m + i] else 0
        if count != n:
            return _solve_periodic_complex_active_dense_qr(
                H, w, v, residual_r, active_cols,
                period, m, d, rank, x, error,
            )

    T = <double complex*>malloc(period*n*n*sizeof(double complex))
    Z = <double complex*>calloc(period*m*n, sizeof(double complex))
    alpha = <double complex*>malloc(n*sizeof(double complex))
    eig_beta = <double complex*>malloc(n*sizeof(double complex))
    scale = <int*>malloc(n*sizeof(int))
    beta = <double complex*>malloc(period*n*rank*sizeof(double complex))
    C = <double complex*>malloc(period*d*n*sizeof(double complex))
    U = <double complex*>malloc(period*n*rank*sizeof(double complex))
    if (T == NULL or Z == NULL or alpha == NULL or eig_beta == NULL or
            scale == NULL or beta == NULL or C == NULL or U == NULL):
        info = -1
    if info == 0:
        info = _periodic_schur_scaled_with_exact_retry_Z(
            H, active_cols, scale_tol, period, m, n,
            T, Z, alpha, eig_beta, scale,
        )
    if info == 0:
        for k in range(period):
            kp = (k + 1) % period
            # beta.T = v.T conj(Z[:d]) in the column-major buffer views.
            blas.zgemm(
                &no_trans, &adjoint, &rank, &n, &d,
                &one, &(<double complex*>v)[k*d*d], &d, &Z[k*m*n], &n,
                &zero, &beta[k*n*rank], &rank,
            )
            # C.T = Z_tail.T residual.T, with no conjugation.
            blas.zgemm(
                &no_trans, &no_trans, &n, &d, &d,
                &one, &Z[(kp*m + m - d)*n], &n,
                &(<double complex*>residual_r)[k*d*d], &d,
                &zero, &C[k*d*n], &n,
            )
        info = _solve_periodic_schur_coordinates_Z(
            T, w, beta, C, period, n, d, rank, U, error,
        )
    if info == 0:
        for k in range(period):
            # The column-major output view is x.T = Z U.
            blas.zgemm(
                &trans, &trans, &m, &rank, &n,
                &one, &Z[k*m*n], &n, &U[k*n*rank], &rank,
                &zero, &x[k*d*m], &m,
            )
        _periodic_compressed_error_Z(
            H, w, v, residual_r, active_cols,
            period, m, d, rank, x, error,
        )

    free(T)
    free(Z)
    free(alpha)
    free(eig_beta)
    free(scale)
    free(beta)
    free(C)
    free(U)
    return info
