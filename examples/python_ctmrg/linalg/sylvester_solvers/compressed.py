"""Common JAX contracts for compressed Sylvester solvers."""

from jax_config import configure_jax

configure_jax()
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from . import jax_ffi


def _sylvester_compressed_dense_gmres(
    H,
    w,
    v,
    residual_r,
    *,
    w_triangular,
):
    r"""Solve the single-period compressed objective by dense shifted QR.

    For ``beta[:d, :] = v`` and ``G[:, -d:] = residual_r``, sweep through the
    triangular columns of ``w`` and solve

    ``min_Y ||H Y - Y w - beta||_F^2 + ||G Y||_F^2``.

    The returned array is ``x = Y.T`` and ``error[j]`` is the absolute norm of
    column ``j`` of the stacked compressed residual.
    """
    d_block = v.shape[0]
    n_krylov = H.shape[0]

    beta = jnp.zeros((n_krylov, d_block), dtype=H.dtype)
    beta = beta.at[:d_block].set(v)
    residual_term = jnp.zeros((d_block, n_krylov), dtype=H.dtype)
    residual_term = residual_term.at[:, -d_block:].set(residual_r)
    identity = jnp.eye(n_krylov, dtype=H.dtype)

    def solve_shift(shift, rhs):
        """Solve one augmented reduced GMRES least-squares problem."""
        reduced_operator = jnp.vstack([H - shift*identity, residual_term])
        reduced_rhs = jnp.concatenate([
            rhs,
            jnp.zeros((d_block,), dtype=H.dtype),
        ])
        augmented = jnp.column_stack([reduced_operator, reduced_rhs])
        augmented_r = jnp.linalg.qr(augmented, mode="r")
        operator_r = augmented_r[:n_krylov, :n_krylov]
        rhs_q = augmented_r[:n_krylov, n_krylov]
        x = jsp_linalg.solve_triangular(operator_r, rhs_q, lower=False)
        return x, jnp.abs(augmented_r[n_krylov, n_krylov])

    if w_triangular == "upper":
        order = jnp.arange(d_block)
    elif w_triangular == "lower":
        order = jnp.arange(d_block - 1, -1, -1)
    else:
        raise ValueError(
            "dense_gmres requires w_triangular='upper' or 'lower'"
        )

    x0 = jnp.zeros((d_block, n_krylov), dtype=H.dtype)
    error0 = jnp.zeros((d_block,), dtype=jnp.real(H).dtype)

    def triangular_step(carry, j):
        """Solve one column after inserting known triangular couplings."""
        x, error = carry
        rhs = beta[:, j] + jnp.dot(x.T, w[:, j])
        solution, error_j = solve_shift(w[j, j], rhs)
        x = x.at[j].set(solution)
        error = error.at[j].set(error_j)
        return (x, error), None

    (x, error), _ = jax.lax.scan(
        triangular_step,
        (x0, error0),
        order,
    )
    return x, error


def sylvester_compressed(
    H,
    w,
    v,
    residual_r,
    *,
    method,
    w_triangular,
    block_2x2_start=None,
    tpqrt_block_size=32,
):
    r"""Solve one compressed block-Arnoldi Sylvester problem.

    Both methods use ``beta[:d, :] = v``, ``G[:, -d:] = residual_r``, return
    ``x = Y.T``, and report one absolute residual norm per right-hand-side
    column for

    ``||H Y - Y w - beta||_F^2 + ||G Y||_F^2``.

    ``method="dense_gmres"`` uses JAX dense shifted QR and requires scalar
    upper- or lower-triangular ``w``. ``method="schur_gmres"`` uses the native
    real-Schur structured solver and supports 1x1 and 2x2 diagonal blocks.
    """
    if method == "dense_gmres":
        return _sylvester_compressed_dense_gmres(
            H,
            w,
            v,
            residual_r,
            w_triangular=w_triangular,
        )
    if method == "schur_gmres":
        if jnp.issubdtype(H.dtype, jnp.complexfloating):
            raise ValueError("schur_gmres is currently real-only")
        if block_2x2_start is None:
            block_2x2_start = jnp.zeros((w.shape[0],), dtype=jnp.bool_)
        return jax_ffi._sylvester_compressed_schur_gmres_real(
            H,
            w,
            v,
            residual_r,
            block_2x2_start,
            w_triangular,
            tpqrt_block_size,
        )
    raise ValueError(f"unknown compressed Sylvester method {method!r}")


def sylvester_compressed_periodic(
    H,
    w,
    v,
    residual_r,
    active_cols,
    rank,
    *,
    method,
    block_2x2_start=None,
    galerkin_block_solver=None,
    scale_tol=None,
):
    r"""Solve one periodic compressed block-Arnoldi Sylvester problem.

    ``method="dense_gmres"`` and ``method="periodic_schur_gmres"`` minimize

    ``sum_k ||H[k]Y[k+1] + Y[k]w[k] - beta[k]||_F^2``
    ``      + ||residual_r[k] E_tail Y[k+1]||_F^2``

    on the coordinates selected by ``active_cols`` and leading physical
    columns selected by ``rank``. They differ only in whether the active
    projected problem is assembled densely or reduced to periodic Schur form.

    ``method="periodic_schur_galerkin"`` instead solves the projected core
    equations exactly; the Arnoldi tail contributes only to the returned
    residual diagnostic. Its default block solver is SLICOT ``MB03KE`` for
    real cycles of length at least two and LAPACK ``GESV`` otherwise. Both
    arithmetic types first reduce the active ``H[k]`` factors to periodic
    Schur form. All methods return ``x[k] = Y[k].T`` with inactive Arnoldi
    coordinates and physical columns ``rank:`` exactly zero.

    For periodic-Schur methods, ``scale_tol[k]`` is an absolute row-norm
    cutoff applied to factor ``k`` after periodic Hessenberg reduction and
    before the Schur iteration. ``None`` disables this deflation. Dense GMRES
    accepts the option for a common interface but does not use it. Neither
    does the unequal-active-rank dense fallback in periodic-Schur GMRES. If
    row deflation itself stalls periodic Schur, the structured solver retries
    once with the original undeflated factors.
    """
    methods = (
        "dense_gmres",
        "periodic_schur_gmres",
        "periodic_schur_galerkin",
    )
    if method not in methods:
        raise ValueError(f"unknown periodic compressed Sylvester method {method!r}")

    is_complex = jnp.issubdtype(H.dtype, jnp.complexfloating)
    if scale_tol is None:
        scale_tol = jnp.zeros((H.shape[0],), dtype=jnp.real(H).dtype)
    else:
        scale_tol = jnp.asarray(scale_tol, dtype=jnp.real(H).dtype)
    if galerkin_block_solver is None:
        galerkin_block_solver = (
            "mb03ke" if H.shape[0] >= 2 and not is_complex else "dgesv"
        )
    if is_complex:
        if method == "dense_gmres":
            return jax_ffi._sylvester_compressed_periodic_dense_gmres_complex(
                H,
                w,
                v,
                residual_r,
                active_cols,
                rank,
            )
        if method == "periodic_schur_gmres":
            return jax_ffi._sylvester_compressed_periodic_schur_gmres_complex(
                H,
                w,
                v,
                residual_r,
                scale_tol,
                active_cols,
                rank,
            )
        if galerkin_block_solver != "dgesv":
            raise ValueError("complex Galerkin supports only the GESV block solver")
        return jax_ffi._sylvester_compressed_periodic_schur_galerkin_complex(
            H,
            w,
            v,
            residual_r,
            scale_tol,
            active_cols,
            rank,
        )

    if block_2x2_start is None:
        block_2x2_start = jnp.zeros((w.shape[1],), dtype=jnp.bool_)
    block_2x2_start = jnp.asarray(block_2x2_start, dtype=jnp.bool_)
    if method == "dense_gmres":
        return jax_ffi._sylvester_compressed_periodic_dense_gmres_real(
            H,
            w,
            v,
            residual_r,
            block_2x2_start,
            active_cols,
            rank,
        )
    if method == "periodic_schur_gmres":
        return jax_ffi._sylvester_compressed_periodic_schur_gmres_real(
            H,
            w,
            v,
            residual_r,
            scale_tol,
            block_2x2_start,
            active_cols,
            rank,
        )
    return jax_ffi._sylvester_compressed_periodic_schur_galerkin_real(
        H,
        w,
        v,
        residual_r,
        scale_tol,
        block_2x2_start,
        active_cols,
        rank,
        galerkin_block_solver=galerkin_block_solver,
    )
