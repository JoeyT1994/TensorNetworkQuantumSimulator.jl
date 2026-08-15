"""Shared JAX linear-algebra helpers."""

import functools

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import scipy.linalg as sp_linalg


def schur(A):
    r"""Return ``Z, T, block_2x2_start`` with ``A = Z T Z^H``.

    Complex input uses complex Schur form, so ``T`` is upper triangular and
    ``block_2x2_start`` is all false.  Real input uses real Schur form, where a
    true entry at ``j`` marks a quasi-triangular block spanning ``j:j + 2``.
    """
    is_complex = jnp.issubdtype(A.dtype, jnp.complexfloating)
    output = "complex" if is_complex else "real"
    T, Z = jsp_linalg.schur(A, output=output)

    block_2x2_start = jnp.zeros((A.shape[0],), dtype=jnp.bool_)
    if not is_complex:
        block_2x2_start = block_2x2_start.at[:-1].set(
            jnp.diag(T, k=-1) != 0
        )
    return Z, T, block_2x2_start


def check_schur(T):
    """Return whether ``T`` has exact periodic upper-Schur structure.

    Complex factors must all be upper triangular.  For real factors, ``T[0]``
    may be quasi-upper-triangular with disjoint 2 x 2 diagonal blocks, while
    ``T[1:]`` must be upper triangular.
    """
    if jnp.issubdtype(T.dtype, jnp.complexfloating):
        return jnp.all(jnp.tril(T, k=-1) == 0)

    block_2x2_start = jnp.diag(T[0], k=-1) != 0
    T0_is_quasi_triangular = (
        jnp.all(jnp.tril(T[0], k=-2) == 0)
        & jnp.all(~(block_2x2_start[:-1] & block_2x2_start[1:]))
    )
    return T0_is_quasi_triangular & jnp.all(jnp.tril(T[1:], k=-1) == 0)


def _zero_last_schur_basis(A, zero_tol):
    """Return a Schur basis with numerically nonzero eigenvalues first."""
    real_dtype = np.asarray(A).real.dtype
    cutoff = float(zero_tol) * np.finfo(real_dtype).eps * max(
        float(np.linalg.norm(A, ord=np.inf)),
        1.0,
    )
    if np.iscomplexobj(A):
        _, Z, _ = sp_linalg.schur(
            A,
            output="complex",
            sort=lambda w: np.abs(w) > cutoff,
        )
    else:
        _, Z, _ = sp_linalg.schur(
            A,
            output="real",
            sort=lambda wr, wi: np.hypot(wr, wi) > cutoff,
        )
    return np.asarray(Z, dtype=A.dtype)


def periodic_schur_bruteforce(A, zero_tol=100.0):
    r"""Return stacked ``Z, T`` in the SLICOT periodic-Schur convention.

    For ``P = A[0] A[1] ... A[period - 1]``, the factors satisfy

    ``Z[j]^H A[j] Z[(j + 1) % period] = T[j]``.

    In other words: A[j] Z[j+1] = Z[j] T[j]

    For nonsingular real factors, ``T[0]`` is quasi-upper-triangular and the
    remaining factors are upper triangular.  Complex factors give triangular
    ``T[j]`` throughout.  Numerically zero product eigenvalues, using
    ``zero_tol`` in machine-epsilon units, are ordered after all nonzero ones.
    This reference implementation forms ``P`` explicitly.
    """
    if isinstance(A, (list, tuple)):
        A = jnp.stack(A)
    else:
        A = jnp.asarray(A)

    P = A[0]
    for j in range(1, A.shape[0]):
        P = jnp.dot(P, A[j])

    real_dtype = jnp.real(P).dtype
    Z0 = jax.pure_callback(
        _zero_last_schur_basis,
        jax.ShapeDtypeStruct(P.shape, P.dtype),
        P,
        jnp.asarray(zero_tol, dtype=real_dtype),
        vmap_method="sequential",
    )
    Z = [None]*A.shape[0]
    T = [None]*A.shape[0]
    Z[0] = Z0

    for j in range(A.shape[0] - 1, 0, -1):
        Z[j], T[j] = jnp.linalg.qr(
            jnp.dot(A[j], Z[(j + 1) % A.shape[0]])
        )

    T[0] = jnp.dot(Z[0].conj().T, jnp.dot(A[0], Z[1 % A.shape[0]]))
    return jnp.stack(Z), jnp.stack(T)


def _split_tall_qrp(A):
    """Return ``A[:, p] = Q R`` via ``A = Q0 R0`` and QRP of ``R0``."""
    Q, R = jax.lax.linalg.qr(A, pivoting=False, full_matrices=False)
    # R0[:, p] = q R, hence A[:, p] = (Q q) R.
    q, R, p = jax.lax.linalg.qr(
        R,
        pivoting=True,
        full_matrices=False,
        use_magma=False,
    )
    Q = jnp.dot(Q, q)
    return Q, R, p


def split_qrp(A):
    """Return thin ``Q, R, p`` satisfying ``A[:, p] = Q R`` and ``Q^H Q = I``."""
    m, n = A.shape
    if m <= n:
        return jax.lax.linalg.qr(
            A,
            pivoting=True,
            full_matrices=False,
            use_magma=False,
        )

    return jax.lax.platform_dependent(
        A,
        cpu=lambda X: jax.lax.linalg.qr(
            X,
            pivoting=True,
            full_matrices=False,
            use_magma=False,
        ),
        cuda=_split_tall_qrp,
        rocm=_split_tall_qrp,
    )


def ql(A):
    """Return reduced ``Q, L`` satisfying ``A = Q L`` with lower-triangular ``L``."""
    Q, L = jnp.linalg.qr(jnp.flip(A, axis=(0, 1)), mode="reduced")
    Q = jnp.flip(Q, axis=(0, 1))
    L = jnp.flip(L, axis=(0, 1))
    diag = jnp.diag(L)
    diag_abs = jnp.abs(diag)
    phase = jnp.where(diag_abs > 0, diag/diag_abs, jnp.ones_like(diag))
    Q = Q*phase[None, :]
    L = jnp.conj(phase)[:, None]*L
    return Q, L


def _triangular_pinv_solve_impl(
    a,
    b,
    rank,
    *,
    left_side=False,
    lower=False,
    transpose_a=False,
    conjugate_a=False,
    unit_diagonal=False,
):
    """Apply the primal retained-sector solve without its custom JVP."""
    active = jnp.arange(a.shape[-1]) < rank
    active_block = active[:, None] & active[None, :]
    a_solve = jnp.where(active_block, a, 0)
    a_solve = a_solve + jnp.diag((~active).astype(a.dtype))
    if unit_diagonal:
        diagonal = jnp.eye(a.shape[-1], dtype=jnp.bool_)
        a_solve = jnp.where(diagonal, 1, a_solve)
    op_a = jnp.conj(a_solve) if conjugate_a else a_solve
    op_a = jnp.swapaxes(op_a, -1, -2) if transpose_a else op_a
    rhs_active = active[:, None] if left_side else active[None, :]
    b_solve = jnp.where(rhs_active, b, 0)
    del lower
    x = (
        jnp.linalg.solve(op_a, b_solve)
        if left_side
        else jnp.swapaxes(
            jnp.linalg.solve(
                jnp.swapaxes(op_a, -1, -2),
                jnp.swapaxes(b_solve, -1, -2),
            ),
            -1,
            -2,
        )
    )
    return jnp.where(rhs_active, x, 0)


@functools.partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5, 6, 7))
def _triangular_pinv_solve_jvp(
    a,
    b,
    rank,
    left_side,
    lower,
    transpose_a,
    conjugate_a,
    unit_diagonal,
):
    """Apply a padded triangular solve with the full fixed-rank inverse JVP."""
    return _triangular_pinv_solve_impl(
        a,
        b,
        rank,
        left_side=left_side,
        lower=lower,
        transpose_a=transpose_a,
        conjugate_a=conjugate_a,
        unit_diagonal=unit_diagonal,
    )


@_triangular_pinv_solve_jvp.defjvp
def _triangular_pinv_solve_jvp_rule(
    left_side,
    lower,
    transpose_a,
    conjugate_a,
    unit_diagonal,
    primals,
    tangents,
):
    """Differentiate the active inverse, including non-triangular ``da``."""
    a, b, rank = primals
    da, db, _ = tangents
    x = _triangular_pinv_solve_impl(
        a,
        b,
        rank,
        left_side=left_side,
        lower=lower,
        transpose_a=transpose_a,
        conjugate_a=conjugate_a,
        unit_diagonal=unit_diagonal,
    )

    op_da = jnp.conj(da) if conjugate_a else da
    op_da = jnp.swapaxes(op_da, -1, -2) if transpose_a else op_da
    if unit_diagonal:
        diagonal = jnp.eye(op_da.shape[-1], dtype=jnp.bool_)
        op_da = jnp.where(diagonal, 0, op_da)
    rhs = db - (op_da @ x if left_side else x @ op_da)
    dx = _triangular_pinv_solve_impl(
        a,
        rhs,
        rank,
        left_side=left_side,
        lower=lower,
        transpose_a=transpose_a,
        conjugate_a=conjugate_a,
        unit_diagonal=unit_diagonal,
    )
    return x, dx


def triangular_pinv_solve(
    a,
    b,
    rank,
    *,
    left_side=False,
    lower=False,
    transpose_a=False,
    conjugate_a=False,
    unit_diagonal=False,
):
    r"""Solve with the structurally padded pseudoinverse of ``a``.

    ``a_active = a[:rank, :rank]`` must be nonsingular.  The
    trailing rows and columns are ignored, so this applies the retained-sector
    inverse ``P a_active^-1 P``; exact structural zero-padding is a special
    case, but inactive Schur blocks may also be present.  A small dense solve
    handles both triangular factors and real quasi-triangular Schur factors
    containing 2 x 2 diagonal blocks.  ``lower`` records the intended Schur
    orientation but does not change the dense active-block solve.
    ``rank`` is the known retained rank; this routine performs no numerical
    rank detection.  ``a`` has shape ``(..., m, m)`` and ``rank`` is a scalar
    shared by any batch dimensions.  As in the LAX primitive, ``b`` has shape
    ``(..., m, n)`` for a left solve or ``(..., n, m)`` for a right solve.

    The inactive diagonal is replaced by ones only for the padded solve.
    Inactive rows or columns of ``b`` and the result are exact zeros.  Its JVP
    differentiates the full fixed-rank active inverse, so ``P da P`` need not
    itself be triangular: ``d(a^+ b) = a^+ (db - da x)`` and
    ``d(b a^+) = (db - x da) a^+``.  The remaining keyword arguments have the
    same meaning as in
    ``jax.lax.linalg.triangular_solve``: solve ``op(a) @ x = b`` when
    ``left_side=True`` and ``x @ op(a) = b`` otherwise, with optional transpose,
    conjugation, and implicit unit diagonal of the active block.
    """
    return _triangular_pinv_solve_jvp(
        a,
        b,
        rank,
        left_side,
        lower,
        transpose_a,
        conjugate_a,
        unit_diagonal,
    )


def balanced_triangular_split(R):
    r"""Split nonsingular upper-triangular ``R`` into upper-triangular ``R1, R2`` with
    ``R=R1 R2`` and
    ``R1^H R1=R2 R2^H``.

    The SVD gives balanced dense factors ``C=U sqrt(s)`` and
    ``D=sqrt(s) V^H``.  Factoring ``D=Q R2`` moves the remaining unitary
    gauge into ``R1=C Q`` while preserving balance.
    """
    U, s, Vh = jnp.linalg.svd(R, full_matrices=False)
    sqrt_s = jnp.sqrt(s)
    C = U*sqrt_s[None, :]
    D = sqrt_s[:, None]*Vh

    Q, R2 = jnp.linalg.qr(D, mode="reduced")
    R1 = jnp.dot(C, Q)
    return jnp.triu(R1), jnp.triu(R2)


def balanced_triangular_inv_split(S):
    r"""Split ``S^-1`` into upper-triangular ``SR, SL`` with
    ``SR^H SR=SL SL^H`` without forming the inverse.

    For ``S=U diag(s) V^H``, balanced dense inverse factors are
    ``C=V diag(s^-1/2)`` and ``D=diag(s^-1/2) U^H``.  Factoring
    ``D=Q SL`` moves the remaining unitary gauge into ``SR=C Q``.
    """
    U, s, Vh = jnp.linalg.svd(S, full_matrices=False)
    sqrt_s_inv = jax.lax.rsqrt(s)
    C = Vh.T.conj()*sqrt_s_inv[None, :]
    D = sqrt_s_inv[:, None]*U.T.conj()

    Q, SL = jnp.linalg.qr(D, mode="reduced")
    SR = jnp.dot(C, Q)
    return jnp.triu(SR), jnp.triu(SL)


def eig_diagonal_normalize(XL, XR):
    """Scale pairs to ``XL_i^H XR_i=1`` when ``|XL_i^H XR_i|>1e-16``.

    Zero rejected pairs and return the phase-corrected overlaps before scaling.
    """
    d = jnp.einsum("ni,ni->i", jnp.conj(XL), XR)
    d_abs = jnp.abs(d)
    phase = jnp.where(d_abs > 1e-16, jnp.conj(d)/d_abs, 1.0)
    XR = XR*phase[None, :]
    d = jnp.einsum("ni,ni->i", jnp.conj(XL), XR)
    d_abs = jnp.abs(d)
    scale = jnp.where(d_abs > 1e-16, 1.0/jnp.sqrt(d_abs), 0.0)
    XL = XL*scale[None, :]
    XR = XR*scale[None, :]
    return XL, XR, d


def eig_biorth_error(XL, XR):
    """Return the biorthogonality error ``||XL^H XR - I||_F``."""
    B = jnp.dot(XL.conj().T, XR)
    return jnp.linalg.norm(B - jnp.eye(B.shape[0], dtype=B.dtype))


def qr_rank_from_r(R, rank_tol):
    """Count rows with ``||R_i||_2 > rank_tol*eps*max(max_j ||R_j||_2, 1)``."""
    row_nrm = jnp.linalg.norm(R, axis=1)
    cutoff = rank_tol*jnp.finfo(row_nrm.dtype).eps*jnp.maximum(jnp.max(row_nrm), 1.0)
    return jnp.sum(row_nrm > cutoff)


def qrp_basis_and_rank(X, tol):
    """For ``X[:,p]=Q R``, return ``Q, R, keep, sum(keep)``.

    Here ``keep_i`` means ``|R_ii| > tol*||X||_F``; ``tol`` is Frobenius-relative.
    """
    Q, R, _ = split_qrp(X)
    diag_nrm = jnp.abs(jnp.diag(R))
    scale = jnp.linalg.norm(X)
    keep = diag_nrm > tol*scale
    rank = jnp.sum(keep)
    return Q, R, keep, rank


def safe_inv_sqrt(x, cutoff):
    """Return ``x_i^-1/2`` for ``x_i>cutoff``, zero otherwise, and retained count.

    ``cutoff`` is an absolute scalar in the same units as ``x``.
    """
    keep = x > cutoff
    x_safe = jnp.where(keep, x, jnp.ones_like(x))
    sqrt_x_inv = jnp.where(keep, jax.lax.rsqrt(x_safe), jnp.zeros_like(x))
    rank = jnp.sum(keep)
    return sqrt_x_inv, rank


def biorthogonalize_bases(Q_R, Q_L, tol=10.0):
    """Return balanced biorthogonal bases in the input right Schur flag. The goal is the following:

    We look for gauge transform QR --> QR GR, QL ---> GL QL s.t.:

    QL QR = Id (bi-orthogonal). If rank deficient, QL QR = projector 

    QR^D QR = QL QL^D  (balanced)

    The above two determine the pair modulo a unitary QR ---> QR U, QL --> U^D QL

    To fix U, we further require that the applied transformation is triangular:
    
        QR_out = QR_in R

    This will have the nice property of preserving Schur forms.     
    
    Algorithm:

    For ``Q_L Q_R=U diag(s) V^H`` and ``D=diag(s^-1/2)``, factor

    ``D V^H = Z L``

    by QL, then apply the total gauges ``G_R=L^H`` and ``G_L=Z^H D U^H``.
    These are algebraically equivalent to balanced SVD whitening followed by
    the unitary rotation that restores the input right Schur flag.  Thus the
    bases satisfy ``Q_L Q_R=I`` and, for orthonormal input row/column bases,
    ``Q_R^H Q_R=Q_L Q_L^H``.  ``G_R`` is upper triangular, so an existing
    upper-triangular projected action stays upper triangular.

    Singular values below ``tol*eps*max(max(s),1)`` are zeroed.  The returned
    gauges are identity when the input overlap is identity.  Returns the
    transformed bases, total gauges, original overlap, singular values, and
    numerical rank.
    """
    M = jnp.dot(Q_L, Q_R)
    U, s, Vh = jnp.linalg.svd(M)
    cutoff = tol*jnp.finfo(s.dtype).eps*jnp.maximum(jnp.max(s), 1.0)
    sqrt_s_inv, rank = safe_inv_sqrt(s, cutoff)
    Z, L = ql(sqrt_s_inv[:, None]*Vh)
    G_R = L.T.conj()
    G_L = jnp.dot(Z.T.conj(), sqrt_s_inv[:, None]*U.T.conj())
    Q_R = jnp.dot(Q_R, G_R)
    Q_L = jnp.dot(G_L, Q_L)
    return Q_R, Q_L, G_R, G_L, M, s, rank


def svd_pinv(E, tol):
    """Return ``E^+=V diag(s_i^-1) U^H`` for ``s_i>tol*max_j(s_j)``, zero below, plus ``s``."""
    U, s, Vh = jnp.linalg.svd(E, full_matrices=False)
    cutoff = tol*jnp.max(s, axis=-1, keepdims=True)
    keep = s > cutoff
    s_safe = jnp.where(keep, s, 1.0)
    s_inv = jnp.where(keep, 1.0/s_safe, 0.0)
    # Vh_{kn}^* s_k^{-1} U_{mk}^* -> E^+_{nm}
    E_inv = jnp.einsum("...kn,...k,...mk->...nm", Vh.conj(), s_inv, U.conj())
    return E_inv, s
