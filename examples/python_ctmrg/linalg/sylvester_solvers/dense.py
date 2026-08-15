"""Pure-JAX dense Sylvester solvers."""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_sylvester

from linalg.jax_linalg import triangular_pinv_solve


def _padded_pinv_solve(
    a,
    b,
    rank,
    *,
    schur_form=False,
    lower=False,
    rcond=1e-14,
):
    """Apply the padded right pseudoinverse of ``a`` to ``b``."""
    if schur_form:
        return triangular_pinv_solve(
            a,
            b,
            rank,
            lower=lower,
        )
    return b @ jnp.linalg.pinv(a, rtol=rcond)


def _pack_matrices(X):
    """Flatten a tuple of matrices into one vector in cyclic order."""
    return jnp.concatenate(tuple(x.reshape(-1) for x in X))


def dense_periodic_sylvester_bartels_stewart(
    C,
    V,
    c,
    Bhat,
    rank,
    rcond=1e-14,
    schur_form=False,
    lower=False,
):
    r"""Solve four periodic equations through one gauge-reduced Sylvester solve.

    Cycling the equation based at bond zero gives

    ``(A0 X0 - X0 s0) h0 = R0``.

    We solve directly for ``Y0 = X0 h0``. Since ``s0 h0 = h0 s3``, this
    obeys ``A0 Y0 - Y0 s3 = R0``. Restricting to ``ker(VL[0])`` removes
    the retained Ritz block, whose exact spectral resonance would otherwise
    make the full-space Sylvester equation singular. The remaining three
    ``X[k]`` are propagated backward through the original periodic equations.
    With ``schur_form=True``, solves use the leading rank-dimensional sector
    of the triangular ``c`` factors; inactive Schur blocks are ignored.
    ``lower=True`` selects the transposed reversed-cycle form.
    """
    VL, _ = V
    active = jnp.arange(c[0].shape[0]) < rank
    Bhat = tuple(Bk * active[None, :] for Bk in Bhat)
    C01 = C[0] @ C[1]
    C012 = C01 @ C[2]
    A_cycle = C012 @ C[3]
    c12 = c[1] @ c[2]
    h = c[0] @ c12
    s3 = c[3] @ h
    active_block = active[:, None] & active[None, :]
    s3_active = jnp.where(active_block, s3, jnp.zeros_like(s3))
    s3 = s3_active + jnp.diag((~active).astype(s3.dtype))
    R = (
        Bhat[0] @ h
        + C[0] @ Bhat[1] @ c12
        + C01 @ Bhat[2] @ c[2]
        + C012 @ Bhat[3]
    )

    # Complete QR puts the retained row space first. The ordinary full-rank
    # path uses the compact N-chi transverse basis. At deficient rank, padded
    # zero rows of VL generate extra QR completion vectors which belong to
    # ker(VL) and must be retained; use a static masked N x N representation
    # only for that branch.
    Q, _ = jnp.linalg.qr(VL[0].conj().T, mode="complete")

    def solve_full_rank(_):
        """Solve in the compact kernel basis when no Ritz vectors are padded."""
        U = Q[:, c[0].shape[0]:]
        A_transverse = U.conj().T @ A_cycle @ U
        R_transverse = U.conj().T @ R
        scale = jnp.maximum(
            jnp.linalg.norm(A_transverse),
            jnp.linalg.norm(s3_active),
        )
        return U @ solve_sylvester(
            A_transverse,
            -s3,
            R_transverse,
            method="schur",
            tol=rcond * scale,
        )

    def solve_rank_deficient(_):
        """Retain zero-row QR completions with static masked coordinates."""
        transverse = jnp.arange(Q.shape[1]) >= rank
        U = Q * transverse[None, :]
        A_transverse = U.conj().T @ A_cycle @ U
        scale = jnp.maximum(
            jnp.linalg.norm(A_transverse),
            jnp.linalg.norm(s3_active),
        )
        A_transverse = A_transverse + jnp.diag(
            2.0 * (~transverse).astype(A_transverse.dtype)
        )
        R_transverse = (U.conj().T @ R) * active[None, :]
        return U @ solve_sylvester(
            A_transverse,
            -s3,
            R_transverse,
            method="schur",
            tol=rcond * scale,
        )

    Y = jax.lax.cond(
        rank == c[0].shape[0],
        solve_full_rank,
        solve_rank_deficient,
        operand=None,
    )
    Y = Y * active[None, :]

    # Recover X0 from X0 h = Y, then propagate X3, X2, X1 backward.
    X = [None] * 4
    X[0] = _padded_pinv_solve(
        h,
        Y,
        rank,
        schur_form=schur_form,
        lower=lower,
        rcond=rcond,
    ) * active[None, :]
    for k in range(3, 0, -1):
        p = c[(k + 1) % 4] @ c[(k + 2) % 4] @ c[(k + 3) % 4]
        q = c[k] @ p
        rhs = C[k] @ X[(k + 1) % 4] @ p - Bhat[k]
        X[k] = _padded_pinv_solve(
            q,
            rhs,
            rank,
            schur_form=schur_form,
            lower=lower,
            rcond=rcond,
        ) * active[None, :]

    X = tuple(X)
    sylvester = []
    gauge = []
    for k in range(4):
        p = c[(k + 1) % 4] @ c[(k + 2) % 4] @ c[(k + 3) % 4]
        q = c[k] @ p
        sylvester.append(C[k] @ X[(k + 1) % 4] @ p - X[k] @ q - Bhat[k])
        gauge.append(VL[k] @ X[k])
    residual = jnp.linalg.norm(
        jnp.concatenate((_pack_matrices(sylvester), _pack_matrices(gauge)))
    )
    return X, residual
