import functools
import os
from typing import NamedTuple

from jax_config import configure_jax

configure_jax()
import jax
import jax.numpy as jnp
import numpy as np

from linalg.jax_linalg import (
    biorthogonalize_bases,
    eig_biorth_error,
    eig_diagonal_normalize,
    qr_rank_from_r,
    qrp_basis_and_rank,
    schur as jax_schur,
    split_qrp,
    safe_inv_sqrt,
    svd_pinv,
)
from linalg.sylvester_solvers.compressed import sylvester_compressed

_ENABLE_DEBUG_PRINT = False


def _debug_print_cond(pred, true_fun, operand=None):
    if not _ENABLE_DEBUG_PRINT:
        return None
    return jax.lax.cond(pred, true_fun, lambda _: None, operand=operand)


class _StaticDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class KrylovDebug(NamedTuple):
    """Carry dynamic debug selection and static location labels together."""

    enabled: bool = False
    x: int = -1
    y: int = -1
    sweep: str = ""


def _static_dict(cfg):
    if isinstance(cfg, _StaticDict):
        return cfg
    return _StaticDict(cfg or {})


def dump_krylov_schur_context(path, matvec, vecmat, V_R, V_L, metadata=None):
    """Materialize and write a Krylov-Schur operator context to ``path``."""
    A_matvec, A_vecmat = _materialize_krylov_actions(matvec, vecmat, V_R, V_L)
    jax.block_until_ready((A_matvec, A_vecmat, V_R, V_L))
    payload = {
        "A_matvec": np.asarray(A_matvec),
        "A_vecmat": np.asarray(A_vecmat),
        "V_R": np.asarray(V_R),
        "V_L": np.asarray(V_L),
    }
    if metadata:
        payload.update({key: np.asarray(value) for key, value in metadata.items()})
    _write_npz(path, payload)
    return path


def _materialize_krylov_actions(matvec, vecmat, V_R, V_L):
    """Return dense matrices for the matvec and vecmat block actions."""
    A_matvec = _materialize_block_action(matvec, V_R.shape[0], V_R.shape[1], V_R.dtype)
    A_vecmat = _materialize_block_action(vecmat, V_L.shape[1], V_L.shape[0], V_L.dtype)
    return A_matvec, A_vecmat


def _materialize_rectangular_block_action(action, N, d_block, dtype):
    """Materialize a block-column action with possibly different output size."""
    num_blocks = (N + d_block - 1)//d_block
    eye = jnp.eye(N, dtype=dtype)
    eye = jnp.pad(eye, ((0, 0), (0, num_blocks*d_block - N)))
    eye_blocks = jnp.reshape(eye, (N, num_blocks, d_block))
    eye_blocks = jnp.transpose(eye_blocks, (1, 0, 2))
    action_blocks = jax.vmap(action)(eye_blocks)
    A = jnp.transpose(action_blocks, (1, 0, 2))
    return jnp.reshape(A, (A.shape[0], num_blocks*d_block))[:, :N]


def _materialize_block_action(action, N, d_block, dtype):
    """Materialize a block-column action ``X -> A X`` as a dense matrix."""
    num_blocks = (N + d_block - 1)//d_block
    eye = jnp.eye(N, dtype=dtype)
    eye = jnp.pad(eye, ((0, 0), (0, num_blocks*d_block - N)))
    eye_blocks = jnp.reshape(eye, (N, num_blocks, d_block))
    eye_blocks = jnp.transpose(eye_blocks, (1, 0, 2))
    action_blocks = jax.vmap(action)(eye_blocks)
    A = jnp.transpose(action_blocks, (1, 0, 2))
    return jnp.reshape(A, (N, num_blocks*d_block))[:, :N]


def _write_npz(path, payload):
    """Write a NumPy payload, creating the parent directory when needed."""
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(path, **payload)


def _debug_print_one_sided_ritz_mismatch(
    debug_sweep,
    debug_x,
    debug_y,
    rank_R,
    rank_L,
    dropped_edge_R,
    dropped_edge_L,
    ritz_R,
    ritz_L,
):
    """Print a compact side-by-side table for a one-sided Ritz-rank mismatch."""
    def fmt_complex(z):
        z = complex(z)
        if abs(z.imag) < 5e-16:
            return f"{z.real: .6e}"
        return f"{z.real: .6e}{z.imag:+.1e}j"

    ritz_R = np.asarray(ritz_R)
    ritz_L = np.asarray(ritz_L)
    print(
        f"{debug_sweep} x={int(debug_x)} y={int(debug_y)} one-sided Ritz rank mismatch: "
        f"R={int(rank_R)} L={int(rank_L)} "
        f"dropped_edge_R={bool(dropped_edge_R)} dropped_edge_L={bool(dropped_edge_L)}"
    )
    print("  i  lambda_R                |lambda_R|  lambda_L                |lambda_L|")
    for i, (lam_R, lam_L) in enumerate(zip(ritz_R, ritz_L)):
        print(
            f"  {i:2d} {fmt_complex(lam_R):>22} {abs(lam_R):.1e}  "
            f"{fmt_complex(lam_L):>22} {abs(lam_L):.1e}"
        )


def krylov_eig(
    matvec,
    vecmat,
    V_R,
    V_L,
    chi_max=None,
    info_level=0,
    debug=None,
    krylov_cfg=None,
):
    """ Given a linear operator specified via A.v = matvec(v)   u.A = vecmat(u),

        obtains rank-chi approximate invariant subspaces QL, QR of A s.t.:

            QL.QR = Id

            A.QR = QR.h + error
            QL.A = h.QL + error

        where "error" satisfy oblique Petrov-Galerkin condition coming from the chosen block Krylov space

        Input:
            matvec: action of N x N  matrix  v -->  A v
            vecmat: action of N x N matrix   u -->  A.T u
            V_R = (N, d_block) initial guess
            V_L = (d_block, N) initial guess
            chi_max: requested output width; defaults to ``V_R.shape[1]``.

        krylov_cfg:
            Optional dict with keys such as ``num_krylov_iter``,
            ``basis_method``, ``CTM_eig_cutoff``, ``rank_tol``, and
            ``rank_tol_seed`` and ``pivot``.
            ``basis_method="arnoldi"`` uses independent two-sided Arnoldi
            bases followed by SVD biorthogonalization. ``"bilanczos"`` first
            applies the local block Lanczos recurrence, then does one full
            reorthogonalization pass.

        Returns:
            Q_R, Q_L in the same order as V_R, V_L, the inverse
            sqrt(abs(Ritz)) scale, the Ritz phase/sign data, and an info dict.
            Level 1 contains the CTMRG convergence payload: projected Ritz
            data, initial orthogonality, seed overlaps, ranks, and final
            biorthogonality. Level 2 adds selected-subspace validation, which
            may reapply the physical operators.
    """
    if chi_max is None:
        chi_max = V_R.shape[1]
    if info_level is None:
        info_level = 0
    info_level = int(info_level)
    debug = KrylovDebug() if debug is None else debug
    debug_print, debug_x, debug_y, debug_sweep = debug
    krylov_cfg = _static_dict(krylov_cfg)
    num_iter = krylov_cfg.get('num_krylov_iter', 2)
    basis_method = krylov_cfg.get('basis_method', "arnoldi")
    CTM_eig_cutoff = krylov_cfg.get('CTM_eig_cutoff', 1e-15)
    rank_tol = krylov_cfg.get('rank_tol', None)
    ritz_q0_lock_tol = krylov_cfg.get('ritz_q0_lock_tol', 0.0)
    ritz_selection = krylov_cfg.get('ritz_selection', "q0_lock")
    ritz_error_floor = krylov_cfg.get('ritz_error_floor', 1e-13)
    eig_biorth_tol = krylov_cfg.get('eig_biorth_tol', 0.0)
    eig_clean_max_iter = krylov_cfg.get('eig_clean_max_iter', 1)
    pivot = krylov_cfg.get('pivot', True)

    krylov_info = {}
    d_block = V_R.shape[1]  
    arnoldi_residual_context = None
    basis_rank = V_R.shape[1]

    if basis_method == "arnoldi":
        rank_tol_arnoldi = 1e-8 if rank_tol is None else rank_tol
        rank_tol_seed = krylov_cfg.get('rank_tol_seed', rank_tol_arnoldi)
        Q_R, H_R, residual_R, _, active_R = arnoldi_basis(
            matvec,
            V_R,
            num_iter,
            rank_tol=rank_tol_arnoldi,
            rank_tol_seed=rank_tol_seed,
            full_res=True,
            pivot=pivot,
        )
        Q_L, H_L, residual_L, _, active_L = arnoldi_basis(
            vecmat,
            V_L.T,
            num_iter,
            rank_tol=rank_tol_arnoldi,
            rank_tol_seed=rank_tol_seed,
            full_res=True,
            pivot=pivot,
        )
        Q_L = Q_L.T
        if info_level >= 2:
            krylov_info["active_R_rank"] = jnp.sum(active_R)
            krylov_info["active_L_rank"] = jnp.sum(active_L)

        M = jnp.dot(Q_L, Q_R)
        U, s, Vh = jnp.linalg.svd(M)
        s_fmt = f"{debug_sweep} x={{}} y={{}} s(M) {{}}"
        _debug_print_cond(
            debug_print,
            lambda _: jax.debug.print(s_fmt, debug_x, debug_y, s, ordered=True),
        )
        ortho_err_R = jnp.linalg.norm(
            jnp.dot(Q_R.T.conj(), Q_R) - jnp.eye(Q_R.shape[1], dtype=Q_R.dtype)
        )
        ortho_fmt_R = f"{debug_sweep} x={{}} y={{}} ortho-error-R {{}}"
        _debug_print_cond(
            debug_print,
            lambda _: jax.debug.print(ortho_fmt_R, debug_x, debug_y, ortho_err_R, ordered=True),
        )
        if info_level > 0:
            # Let P0 = QR0 (QL0 QR0)^-1 QL0 be initial guess for projector 
            # Compute || (1 - P0) A P0|| / ||P0|| where || = matrix 2-norm
            # Define Mij = QLi QRj
            # Hij = QRj^D A QRj. So long as num_iter > 1,
            # || (1 - QR0 M00^{-1} QL0) A QR0 M00^{-1} QL0 ||
            # || (1 - QR0 M00^{-1} QL0) QR1 H10  M00^{-1} QL0 ||
            # || (1 - QR0 M00^{-1} QL0) QR1 H10  M00^{-1} ||
            # || (QR0 - QR1 M00^{-1} M01 ) H10  M00^{-1} ||
            # Since the QR/L are orthogonal, the 2-norm can be computed
            # via SVD in smaller space
            # || vstack ( H10 M00^{-1}, -M00^{-1} M01 H10  M00^{-1}  )||

            M00 = M[:d_block, :d_block]
            M01 = M[:d_block, d_block:2*d_block]

            M00_inv, s00 = svd_pinv(M00, 1e-16)
            initial_M00_cond = s00[-1]

            H10 = H_R[d_block:2*d_block, :d_block]
            trace_abs = jnp.abs(jnp.trace(H_R))
            H10 = H10 / trace_abs

            x = jnp.dot(H10, M00_inv)
            s_num = jnp.linalg.svdvals(jnp.vstack([x,  -jnp.dot(M00_inv, M01).dot(x)]))
            
            initial_oblique_res_R = s_num[0]*s00[-1]
            initial_ortho_res_R = jnp.linalg.norm(H10)

            H10 = H_L[d_block:2*d_block, :d_block]
            trace_abs = jnp.abs(jnp.trace(H_L))
            H10 = H10 / trace_abs
            initial_ortho_res_L = jnp.linalg.norm(H10)
            
            krylov_info['initial_M00_cond'] = initial_M00_cond
            krylov_info['initial_oblique_res_R'] = initial_oblique_res_R
            krylov_info['initial_ortho_res_R'] = initial_ortho_res_R
            krylov_info['initial_ortho_res_L'] = initial_ortho_res_L

        # Arnoldi gives A Q_R = Q_R H plus either a final-block residual or a
        # full-width residual containing intermediate deflation terms.
        A = jnp.dot(M, H_R)
        if residual_R.shape[1] == H_R.shape[1]:
            A = A + jnp.dot(Q_L, residual_R)
        else:
            A = A.at[:, -d_block:].add(jnp.dot(Q_L, residual_R))

        cutoff = 100.0*jnp.finfo(s.dtype).eps*jnp.maximum(jnp.max(s), 1.0)
        sqrt_s_inv, basis_rank = safe_inv_sqrt(s, cutoff)
        # 2-norm of the active Krylov-space projector.
        active_s_idx = jnp.maximum(basis_rank, 1) - 1
        krylov_info['M_krylov_cond'] = s[active_s_idx]
        C_R = Vh.T.conj()*sqrt_s_inv[None, :]
        C_L = U.T.conj()*sqrt_s_inv[:, None]
        Q_R = jnp.dot(Q_R, C_R)
        Q_L = jnp.dot(C_L, Q_L)
        A = jnp.dot(jnp.dot(U.T.conj(), A), Vh.T.conj())
        A = A*sqrt_s_inv[:, None]*sqrt_s_inv[None, :]
        arnoldi_residual_context = (
            residual_R,
            residual_L,
            C_R,
            C_L,
            residual_R.shape[1] == H_R.shape[1],
            residual_L.shape[1] == H_L.shape[1],
        )
    elif basis_method in ("bilanczos", "bilanczos_v2"):
        bilanczos_basis_fn = bilanczos_basis_v2 if basis_method == "bilanczos_v2" else bilanczos_basis
        Q_R, Q_L, A, residual_R, residual_L, bilanczos_info = bilanczos_basis_fn(
            matvec,
            vecmat,
            V_R,
            V_L.T,
            num_iter,
        )
        Q_L = Q_L.T
        krylov_info.update(bilanczos_info)

    else:
        raise ValueError(f"Unknown Krylov basis method: {basis_method!r}")

    if ritz_q0_lock_tol or ritz_selection == "interval_q0":
        Q_seed_R, _ = jnp.linalg.qr(V_R, mode="reduced")
        Q_seed_L, _ = jnp.linalg.qr(V_L.T, mode="reduced")
        q0_proj_R = jnp.dot(Q_seed_R.T.conj(), Q_R)
        q0_proj_L = jnp.dot(Q_seed_L.T.conj(), Q_L.T)
        q0_metric_R = jnp.dot(Q_R.T.conj(), Q_R)
        q0_metric_L = jnp.dot(Q_L.conj(), Q_L.T)
    else:
        q0_proj_R = None
        q0_proj_L = None
        q0_metric_R = None
        q0_metric_L = None

    XR, XL, ritz_sqrt_abs, ritz_phase, eigenspace_info = dominant_eigenspace(
        A,
        chi_max,
        info_level=info_level,
        CTM_eig_cutoff=CTM_eig_cutoff,
        ritz_q0_lock_tol=ritz_q0_lock_tol,
        q0_proj_R=q0_proj_R,
        q0_proj_L=q0_proj_L,
        q0_metric_R=q0_metric_R,
        q0_metric_L=q0_metric_L,
        ritz_selection=ritz_selection,
        ritz_error_floor=ritz_error_floor,
        residual_Q_R=Q_R,
        residual_Q_L=Q_L,
        residual_R=residual_R if arnoldi_residual_context is not None else None,
        residual_L=residual_L if arnoldi_residual_context is not None else None,
        residual_C_R=C_R if arnoldi_residual_context is not None else None,
        residual_C_L=C_L if arnoldi_residual_context is not None else None,
        d_block=d_block,
        residual_R_full_width=(
            arnoldi_residual_context[-2] if arnoldi_residual_context is not None else True
        ),
        residual_L_full_width=(
            arnoldi_residual_context[-1] if arnoldi_residual_context is not None else True
        ),
        eig_biorth_tol=eig_biorth_tol,
        eig_clean_max_iter=eig_clean_max_iter,
        return_ritz_vectors=arnoldi_residual_context is not None and info_level >= 1,
    )
    ritz_vectors_R_full = eigenspace_info.pop("_ritz_vectors_R_full", None)
    ritz_vectors_L_full = eigenspace_info.pop("_ritz_vectors_L_full", None)
    if arnoldi_residual_context is not None and ritz_vectors_R_full is not None:
        (
            residual_R,
            residual_L,
            C_R,
            C_L,
            residual_R_full_width,
            residual_L_full_width,
        ) = arnoldi_residual_context
        (
            ritz_residual_R_full,
            ritz_residual_L_full,
            ritz_condition_estimate_full,
        ) = _arnoldi_ritz_residual_diagnostics(
            A,
            eigenspace_info["ritz_values"],
            Q_R,
            Q_L,
            residual_R,
            residual_L,
            C_R,
            C_L,
            ritz_vectors_R_full,
            ritz_vectors_L_full,
            d_block,
            residual_R_full_width,
            residual_L_full_width,
        )
        eigenspace_info.update({
            "ritz_residual_R_full": ritz_residual_R_full,
            "ritz_residual_L_full": ritz_residual_L_full,
            "ritz_condition_estimate_full": ritz_condition_estimate_full,
            "ritz_error_estimate_max_full": (
                ritz_condition_estimate_full
                * jnp.maximum(ritz_residual_R_full, ritz_residual_L_full)
            ),
            "ritz_error_estimate_geo_full": (
                ritz_condition_estimate_full
                * jnp.sqrt(ritz_residual_R_full*ritz_residual_L_full)
            ),
        })
    krylov_info.update(eigenspace_info)
    if info_level >= 1:
        krylov_info["basis_rank"] = eigenspace_info["ritz_rank"]
    ritz_values_full = eigenspace_info.get(
        "ritz_values",
        jnp.zeros((A.shape[0],), dtype=jnp.result_type(A.dtype, jnp.complex64)),
    )
    ritz_fmt = f"{debug_sweep} x={{}} y={{}} ritz spectrum {{}}"
    _debug_print_cond(
        debug_print,
        lambda _: jax.debug.print(ritz_fmt, debug_x, debug_y, ritz_values_full, ordered=True),
    )
    exact_fmt = f"{debug_sweep} x={{}} y={{}} exact spectrum {{}}"
    ratio_fmt = f"{debug_sweep} x={{}} y={{}} ritz/exact {{}}"

    def print_ritz_exact_ratio(_):
        N = V_R.shape[0]
        num_blocks = (N + d_block - 1)//d_block
        eye = jnp.eye(N, dtype=V_R.dtype)
        eye = jnp.pad(eye, ((0, 0), (0, num_blocks*d_block - N)))
        eye_blocks = jnp.reshape(eye, (N, num_blocks, d_block))
        eye_blocks = jnp.transpose(eye_blocks, (1, 0, 2))
        A_blocks = jax.vmap(matvec)(eye_blocks)
        A_dense = jnp.transpose(A_blocks, (1, 0, 2))
        A_dense = jnp.reshape(A_dense, (N, num_blocks*d_block))[:, :N]
        exact_eigs = jnp.linalg.eigvals(A_dense)
        p = jnp.lexsort((-jnp.imag(exact_eigs), -jnp.real(exact_eigs), -jnp.abs(exact_eigs)))
        exact_eigs = exact_eigs[p]
        exact_prefix = exact_eigs[:ritz_values_full.shape[0]]
        ratio = ritz_values_full / exact_prefix
        jax.debug.print(exact_fmt, debug_x, debug_y, exact_prefix, ordered=True)
        jax.debug.print(ratio_fmt, debug_x, debug_y, ratio, ordered=True)
        return None

    _debug_print_cond(
        debug_print,
        print_ritz_exact_ratio,
    )

    ritz_sqrt_abs_pinv = jnp.where(ritz_sqrt_abs > 1e-16, 1.0 / ritz_sqrt_abs, 0.0)
    
    if info_level >= 2:
        basis_biorth = jnp.dot(Q_L, Q_R)
        basis_idx = jnp.arange(basis_biorth.shape[0])
        Id_basis = (
            jnp.eye(basis_biorth.shape[0], dtype=basis_biorth.dtype)
            * (basis_idx < basis_rank)[:, None]
        )
        basis_biorth_delta = basis_biorth - Id_basis
        coeff_biorth = jnp.dot(XL, XR)
        ritz_rank = eigenspace_info["ritz_rank"]
        ritz_idx = jnp.arange(coeff_biorth.shape[0])
        Id_ritz = (
            jnp.eye(coeff_biorth.shape[0], dtype=coeff_biorth.dtype)
            * (ritz_idx < ritz_rank)[:, None]
        )
        krylov_info["coeff_biorth_matrix"] = coeff_biorth
        krylov_info["basis_biorth_err"] = jnp.linalg.norm(basis_biorth_delta)
        krylov_info["coeff_biorth_err"] = jnp.linalg.norm(
            coeff_biorth - Id_ritz
        )
        krylov_info["basis_biorth_selected_err"] = jnp.linalg.norm(
            jnp.dot(XL, jnp.dot(basis_biorth_delta, XR))
        )
        krylov_info["basis_biorth_delta_2norm"] = jnp.linalg.norm(basis_biorth_delta, ord=2)
        krylov_info["coeff_XL_2norm"] = jnp.linalg.norm(XL, ord=2)
        krylov_info["coeff_XR_2norm"] = jnp.linalg.norm(XR, ord=2)

    Q_R = jnp.dot(Q_R, XR)
    Q_L = jnp.dot(XL, Q_L)
    if info_level >= 1:
        biorth = jnp.dot(Q_L, Q_R)
        out_idx = jnp.arange(biorth.shape[0])
        Id_out = (
            jnp.eye(biorth.shape[0], dtype=biorth.dtype)
            * (out_idx < eigenspace_info["ritz_rank"])[:, None]
        )
        krylov_info["final_biorth_err"] = jnp.linalg.norm(
            biorth - Id_out
        )
        Q_seed_R, R_seed_R = jnp.linalg.qr(V_R, mode="reduced")
        Q_out_R, _ = jnp.linalg.qr(Q_R, mode="reduced")
        Q_seed_L, R_seed_L = jnp.linalg.qr(V_L.T, mode="reduced")
        Q_out_L, _ = jnp.linalg.qr(Q_L.T, mode="reduced")
        rank_tol_seed = krylov_cfg.get(
            'rank_tol_seed',
            1e-8 if rank_tol is None else rank_tol,
        )
        seed_R_rank = qr_rank_from_r(R_seed_R, rank_tol_seed)
        seed_L_rank = qr_rank_from_r(R_seed_L, rank_tol_seed)
        krylov_info["seed_R_rank"] = seed_R_rank
        krylov_info["seed_L_rank"] = seed_L_rank
        krylov_info["seed_R_svals"] = _masked_seed_overlap_svals(
            Q_seed_R,
            Q_out_R,
            jnp.arange(Q_seed_R.shape[1]) < seed_R_rank,
            eigenspace_info["ritz_rank"],
        )
        krylov_info["seed_L_svals"] = _masked_seed_overlap_svals(
            Q_seed_L,
            Q_out_L,
            jnp.arange(Q_seed_L.shape[1]) < seed_L_rank,
            eigenspace_info["ritz_rank"],
        )
    if info_level >= 2:
        ritz_op = _selected_ritz_operator(eigenspace_info["ritz_values_kept"], Q_R.dtype)
        ritz_res_R = matvec(Q_R) - jnp.dot(Q_R, ritz_op)
        ritz_res_L = vecmat(Q_L.T).T - jnp.dot(ritz_op, Q_L)
        krylov_info["selected_ritz_res_R"] = (
            jnp.linalg.norm(ritz_res_R, axis=0)
            / jnp.maximum(jnp.linalg.norm(Q_R, axis=0), 1e-300)
        )
        krylov_info["selected_ritz_res_L"] = (
            jnp.linalg.norm(ritz_res_L, axis=1)
            / jnp.maximum(jnp.linalg.norm(Q_L, axis=1), 1e-300)
        )
    return (
        Q_R,
        Q_L,
        ritz_sqrt_abs_pinv,
        ritz_phase,
        krylov_info,
    )




@functools.partial(jax.jit, static_argnames=("biorth_tol", "max_iter"))
def eig_cleaned(A, biorth_tol=0.0, max_iter=1):
    """Return eig vectors, optionally cleaned by whitening their biorthogonal overlap."""
    w, XL, XR = jax.lax.linalg.eig(
        A,
        compute_left_eigenvectors=True,
        compute_right_eigenvectors=True,
    )
    XL, XR, d = eig_diagonal_normalize(XL, XR)
    biorth_err_initial = eig_biorth_error(XL, XR)
    clean_count = jnp.array(0, dtype=jnp.int32)

    def clean_once(carry):
        w, XL, XR, d, clean_count = carry
        L = XL.conj().T
        R = XR
        B = jnp.dot(L, R)
        U, s, Vh = jnp.linalg.svd(B)
        cutoff = 100.0*jnp.finfo(s.dtype).eps*jnp.maximum(jnp.max(s), 1.0)
        sqrt_s_inv, _ = safe_inv_sqrt(s, cutoff)
        C_R = Vh.T.conj()*sqrt_s_inv[None, :]
        C_L = U.T.conj()*sqrt_s_inv[:, None]
        Rw = jnp.dot(R, C_R)
        Lw = jnp.dot(C_L, L)
        Aw = jnp.dot(Lw, jnp.dot(A, Rw))
        w, XLw, XRw = jax.lax.linalg.eig(
            Aw,
            compute_left_eigenvectors=True,
            compute_right_eigenvectors=True,
        )
        XR = jnp.dot(Rw, XRw)
        XL = jnp.dot(Lw.conj().T, XLw)
        XL, XR, d = eig_diagonal_normalize(XL, XR)
        return w, XL, XR, d, clean_count + 1

    if biorth_tol:
        for _ in range(max_iter):
            do_clean = eig_biorth_error(XL, XR) > biorth_tol
            w, XL, XR, d, clean_count = jax.lax.cond(
                do_clean,
                clean_once,
                lambda carry: carry,
                (w, XL, XR, d, clean_count),
            )

    biorth_err_final = eig_biorth_error(XL, XR)
    info = {
        "biorth_err_initial": biorth_err_initial,
        "biorth_err_final": biorth_err_final,
        "clean_count": clean_count,
    }
    return w, XL, XR, d, info


@functools.partial(
    jax.jit,
    static_argnames=(
        "chi",
        "info_level",
        "CTM_eig_cutoff",
        "ritz_q0_lock_tol",
        "ritz_selection",
        "d_block",
        "residual_R_full_width",
        "residual_L_full_width",
        "eig_biorth_tol",
        "eig_clean_max_iter",
        "return_ritz_vectors",
    ),
)
def dominant_eigenspace(
    A,
    chi,
    info_level=0,
    CTM_eig_cutoff=1e-14,
    ritz_q0_lock_tol=0.0,
    q0_proj_R=None,
    q0_proj_L=None,
    q0_metric_R=None,
    q0_metric_L=None,
    ritz_selection="q0_lock",
    ritz_error_floor=1e-13,
    residual_Q_R=None,
    residual_Q_L=None,
    residual_R=None,
    residual_L=None,
    residual_C_R=None,
    residual_C_L=None,
    d_block=1,
    residual_R_full_width=True,
    residual_L_full_width=True,
    eig_biorth_tol=0.0,
    eig_clean_max_iter=1,
    return_ritz_vectors=False,
):
    """Return dominant right and left invariant subspaces of ``A``.

    Eigenpairs are ordered by decreasing ``abs(eigval)``, with deterministic
    tie-breaks by real then imaginary part.  For complex ``A`` this returns the
    first ``chi`` right eigenvectors and left covectors.  For real ``A``,
    complex conjugate eigenvectors are converted to real bases ``Re(v), Im(v)``
    for the invariant two-plane.

    For real matrices, conjugate pairs are never split: a pair is kept only
    when both ``Re(v)`` and ``Im(v)`` fit, otherwise selection keeps scanning
    for lower-ranked real modes or complete conjugate pairs.

    Always returns selected-basis Ritz values, sqrt(abs(Ritz)), the block
    phase/sign matrix for the selected basis, and an info dict.  With
    ``info_level >= 1``, the dict also contains ``{"ritz": w}`` for the full
    ordered spectrum.
    """
    info_level = int(info_level)
    #TODO add comment which clarifies LACPACK convention for XL in complex case (A^T XL = XL w or A^D XL = XL w*?)
    w, XL, XR, eig_d, eig_clean_info = eig_cleaned(
        A,
        biorth_tol=eig_biorth_tol,
        max_iter=eig_clean_max_iter,
    )
    if ritz_q0_lock_tol or ritz_selection == "interval_q0":
        q0_num_R = jnp.linalg.norm(jnp.dot(q0_proj_R, XR), axis=0)
        q0_den_R = jnp.sqrt(jnp.abs(jnp.einsum(
            "ni,nm,mi->i",
            jnp.conj(XR),
            q0_metric_R,
            XR,
        )))
        XL_conj = jnp.conj(XL)
        q0_num_L = jnp.linalg.norm(jnp.dot(q0_proj_L, XL_conj), axis=0)
        q0_den_L = jnp.sqrt(jnp.abs(jnp.einsum(
            "ni,nm,mi->i",
            jnp.conj(XL_conj),
            q0_metric_L,
            XL_conj,
        )))
        q0_weight_R = q0_num_R / jnp.maximum(q0_den_R, 1e-300)
        q0_weight_L = q0_num_L / jnp.maximum(q0_den_L, 1e-300)
        q0_weight_R = jnp.minimum(q0_weight_R, 1.0)
        q0_weight_L = jnp.minimum(q0_weight_L, 1.0)
        ritz_q0_defect_unsorted = 1.0 - jnp.minimum(q0_weight_R, q0_weight_L)
        ritz_q0_locked_unsorted = ritz_q0_defect_unsorted < ritz_q0_lock_tol
    else:
        q0_weight_R = jnp.zeros_like(jnp.real(w))
        q0_weight_L = jnp.zeros_like(jnp.real(w))
        ritz_q0_defect_unsorted = jnp.ones_like(jnp.real(w))
        ritz_q0_locked_unsorted = jnp.zeros_like(jnp.real(w), dtype=jnp.bool_)
    p_q0 = jnp.lexsort((
        -jnp.imag(w),
        -jnp.real(w),
        -jnp.abs(w),
        (~ritz_q0_locked_unsorted).astype(jnp.int32),
    ))
    keep_ritz_unsorted = jnp.abs(w) >= CTM_eig_cutoff*jnp.abs(jnp.sum(w))

    if ritz_selection == "interval_q0":
        (
            ritz_residual_R_unsorted,
            ritz_residual_L_unsorted,
            ritz_condition_estimate_unsorted,
        ) = _arnoldi_ritz_residual_diagnostics(
            A,
            w,
            residual_Q_R,
            residual_Q_L,
            residual_R,
            residual_L,
            residual_C_R,
            residual_C_L,
            XR,
            XL,
            d_block,
            residual_R_full_width,
            residual_L_full_width,
        )
        ritz_error_unsorted = (
            ritz_condition_estimate_unsorted
            * jnp.maximum(ritz_residual_R_unsorted, ritz_residual_L_unsorted)
        )
        ritz_error_unsorted = jnp.maximum(ritz_error_unsorted, ritz_error_floor)
        q0_overlap_unsorted = jnp.minimum(q0_weight_R, q0_weight_L)
        p_selected = ritz_interval_q0_order(
            w,
            ritz_error_unsorted,
            q0_overlap_unsorted,
            keep_ritz_unsorted,
            chi,
            pair_complex=not jnp.issubdtype(A.dtype, jnp.complexfloating),
        )
        mode_idx = jnp.arange(w.shape[0])
        rank_idx = jnp.arange(w.shape[0], dtype=jnp.int32)
        selected = jnp.any(p_selected[:, None] == mode_idx[None, :], axis=0)
        q0_rank = jnp.zeros((w.shape[0],), dtype=jnp.int32).at[p_q0].set(rank_idx)
        selected_pos = jnp.min(
            jnp.where(
                p_selected[:, None] == mode_idx[None, :],
                jnp.arange(chi, dtype=jnp.int32)[:, None],
                chi,
            ),
            axis=0,
        )
        p = jnp.lexsort((
            q0_rank,
            selected_pos,
            (~selected).astype(jnp.int32),
        ))
    else:
        ritz_residual_R_unsorted = None
        ritz_residual_L_unsorted = None
        ritz_condition_estimate_unsorted = None
        ritz_error_unsorted = None
        p = p_q0
    w = w[p]
    XR = XR[:, p]
    XL = XL[:, p]
    if info_level >= 1:
        XR_abs_max = jnp.max(jnp.abs(XR), axis=0)
        XL_abs_max = jnp.max(jnp.abs(XL), axis=0)
        info_eigvec_imag_R_full = (
            jnp.max(jnp.abs(jnp.imag(XR)), axis=0)
            / jnp.maximum(XR_abs_max, 1e-300)
        )
        info_eigvec_imag_L_full = (
            jnp.max(jnp.abs(jnp.imag(XL)), axis=0)
            / jnp.maximum(XL_abs_max, 1e-300)
        )
        phase_R = jnp.sum(XR*XR, axis=0)
        phase_L = jnp.sum(XL*XL, axis=0)
        phase_R = jnp.where(jnp.abs(phase_R) > 1e-300, jnp.exp(-0.5j*jnp.angle(phase_R)), 1.0)
        phase_L = jnp.where(jnp.abs(phase_L) > 1e-300, jnp.exp(-0.5j*jnp.angle(phase_L)), 1.0)
        XR_phase = XR*phase_R[None, :]
        XL_phase = XL*phase_L[None, :]
        info_eigvec_imag_R_phase_full = (
            jnp.linalg.norm(jnp.imag(XR_phase), axis=0)
            / jnp.maximum(jnp.linalg.norm(XR_phase, axis=0), 1e-300)
        )
        info_eigvec_imag_L_phase_full = (
            jnp.linalg.norm(jnp.imag(XL_phase), axis=0)
            / jnp.maximum(jnp.linalg.norm(XL_phase, axis=0), 1e-300)
        )
    ritz_q0_defect_full = ritz_q0_defect_unsorted[p]
    ritz_q0_locked_full = ritz_q0_locked_unsorted[p]
    ritz_cutoff_value = CTM_eig_cutoff*jnp.abs(jnp.sum(w))
    keep_ritz = keep_ritz_unsorted[p]
    if info_level >= 1:
        info = {
            "ritz_values": w,
            "ritz_cutoff_value": ritz_cutoff_value,
            "eigvec_imag_R_full": info_eigvec_imag_R_full,
            "eigvec_imag_L_full": info_eigvec_imag_L_full,
            "eigvec_imag_R_phase_full": info_eigvec_imag_R_phase_full,
            "eigvec_imag_L_phase_full": info_eigvec_imag_L_phase_full,
        }
        if eig_biorth_tol:
            info.update({
                "eig_biorth_err_initial": eig_clean_info["biorth_err_initial"],
                "eig_biorth_err_final": eig_clean_info["biorth_err_final"],
                "eig_clean_count": eig_clean_info["clean_count"],
                "eig_biorth_diag_full": eig_d[p],
            })
        if ritz_q0_lock_tol or ritz_selection == "interval_q0":
            info.update({
                "ritz_q0_defect_full": ritz_q0_defect_full,
                "ritz_q0_defect_R_full": (1.0 - q0_weight_R)[p],
                "ritz_q0_defect_L_full": (1.0 - q0_weight_L)[p],
                "ritz_q0_locked_full": ritz_q0_locked_full,
                "ritz_q0_lock_tol": jnp.asarray(ritz_q0_lock_tol),
            })
        if ritz_selection == "interval_q0":
            ritz_residual_R_full = ritz_residual_R_unsorted[p]
            ritz_residual_L_full = ritz_residual_L_unsorted[p]
            ritz_condition_estimate_full = ritz_condition_estimate_unsorted[p]
            ritz_error_estimate_max_full = ritz_error_unsorted[p]
            info.update({
                "ritz_selection": jnp.asarray(1),
                "ritz_residual_R_full": ritz_residual_R_full,
                "ritz_residual_L_full": ritz_residual_L_full,
                "ritz_condition_estimate_full": ritz_condition_estimate_full,
                "ritz_error_estimate_max_full": ritz_error_estimate_max_full,
                "ritz_error_estimate_geo_full": (
                    ritz_condition_estimate_full
                    * jnp.sqrt(ritz_residual_R_full*ritz_residual_L_full)
                ),
            })
    else:
        info = {}
    if return_ritz_vectors:
        info.update({
            "_ritz_vectors_R_full": XR,
            "_ritz_vectors_L_full": XL,
        })


    #jax.debug.print("dominant_eigenspace eig spec:\n{}", w)

    if jnp.issubdtype(A.dtype, jnp.complexfloating):
        keep_slots = keep_ritz[:chi]
        XR = XR[:, :chi]
        XL = XL[:, :chi].conj().T
        XR = XR*keep_slots[None, :]
        XL = XL*keep_slots[:, None]
        ritz_values = jnp.where(keep_slots, w[:chi], 0.0)
        ritz_abs = jnp.abs(ritz_values)
        ritz_sqrt_abs = jnp.sqrt(ritz_abs)
        ritz_phase = jnp.where(ritz_abs > 1e-16, ritz_values / ritz_abs, 0.0)
        ritz_rank = jnp.sum(keep_slots)
    else:
        tol = 100*jnp.finfo(A.dtype).eps*jnp.maximum(1, jnp.abs(w))
        is_complex = jnp.abs(jnp.imag(w)) > tol
        is_pair_first = is_complex & (jnp.imag(w) > 0)

        alpha = jnp.einsum("ni,ni->i", jnp.conj(XL), XR)
        alpha_abs = jnp.abs(alpha)
        alpha_phase_conj = jnp.where(alpha_abs > 1e-16, jnp.conj(alpha) / alpha_abs, 1.0)
        XR = XR * alpha_phase_conj[None, :]
        if info_level >= 2:
            XL_prereal = XL[:, :chi].conj().T
            XR_prereal = XR[:, :chi]
            d_prereal = jnp.einsum("ij,ji->i", XL_prereal, XR_prereal)
            d_prereal_abs = jnp.abs(d_prereal)
            d_prereal_abs_sqrt = jnp.sqrt(d_prereal_abs)
            d_prereal_abs_sqrt_pinv = jnp.where(
                d_prereal_abs > 1e-16,
                1.0 / d_prereal_abs_sqrt,
                0.0,
            )
            d_prereal_phase_conj = jnp.where(
                d_prereal_abs > 1e-16,
                jnp.conj(d_prereal) / d_prereal_abs,
                0.0,
            )
            XL_prereal = XL_prereal * d_prereal_abs_sqrt_pinv[:, None]
            XR_prereal = XR_prereal * (d_prereal_phase_conj * d_prereal_abs_sqrt_pinv)[None, :]
            coeff_biorth_complex_prereal = jnp.dot(XL_prereal, XR_prereal)
            info["coeff_biorth_complex_prereal_matrix"] = coeff_biorth_complex_prereal
            info["coeff_biorth_complex_prereal_err"] = jnp.linalg.norm(
                coeff_biorth_complex_prereal
                - jnp.eye(coeff_biorth_complex_prereal.shape[0], dtype=coeff_biorth_complex_prereal.dtype)
            )

        XL, XR, ritz_values, ritz_sqrt_abs, ritz_phase = _dominant_real_basis_and_ritz(
            XL,
            XR,
            w,
            keep_ritz,
            is_complex,
            is_pair_first,
            chi,
            A.dtype,
        )
        ritz_rank = jnp.sum(ritz_values != 0)
        XR = XR.T

    d = jnp.einsum("ij,ji->i", XL, XR)
    d_abs = jnp.abs(d)
    if info_level >= 1:
        info["ritz_condition_number"] = jnp.abs(d)
        info["ritz_values_kept"] = ritz_values
        info["ritz_rank"] = ritz_rank

    d_abs_sqrt = jnp.sqrt(d_abs)
    d_abs_sqrt_pinv = jnp.where(d_abs > 1e-16, 1.0 / d_abs_sqrt, 0.0)
    d_phase_conj = jnp.where(d_abs > 1e-16, jnp.conj(d) / d_abs, 0.0)
    XL = XL * d_abs_sqrt_pinv[:, None]
    XR = XR * (d_phase_conj * d_abs_sqrt_pinv)[None, :]
    return XR, XL, ritz_sqrt_abs, ritz_phase, info




@functools.partial(jax.jit, static_argnames=("chi", "pair_complex"))
def ritz_interval_q0_order(
    ritz_values,
    ritz_error,
    q0_overlap,
    keep_ritz,
    chi,
    pair_complex=True,
):
    """Order Ritz modes by certified value intervals, using q0 on frontiers."""
    n = ritz_values.shape[0]
    idx = jnp.arange(n)
    dtype = jnp.real(ritz_values).dtype
    if pair_complex:
        tol = 100*jnp.finfo(dtype).eps*jnp.maximum(1, jnp.abs(ritz_values))
        is_complex = jnp.abs(jnp.imag(ritz_values)) > tol
        is_pair_first = is_complex & (jnp.imag(ritz_values) > 0)
        is_pair_second = is_complex & (jnp.imag(ritz_values) < 0)
    else:
        is_pair_first = jnp.zeros((n,), dtype=jnp.bool_)
        is_pair_second = jnp.zeros((n,), dtype=jnp.bool_)

    next_idx = jnp.minimum(idx + 1, n - 1)
    block_size = jnp.where(is_pair_first, 2, 1)
    block_start = ~is_pair_second
    block_keep = keep_ritz & block_start
    block_keep = block_keep & jnp.where(is_pair_first, keep_ritz[next_idx], True)

    rho = jnp.abs(ritz_values)
    block_rho = jnp.where(is_pair_first, jnp.maximum(rho, rho[next_idx]), rho)
    block_err = jnp.where(is_pair_first, jnp.maximum(ritz_error, ritz_error[next_idx]), ritz_error)
    block_q0 = jnp.where(is_pair_first, jnp.minimum(q0_overlap, q0_overlap[next_idx]), q0_overlap)
    block_lo = jnp.maximum(block_rho - block_err, 0.0)
    block_hi = block_rho + block_err

    selected = -jnp.ones((chi,), dtype=jnp.int32)

    def cond(carry):
        remaining, selected, count = carry
        slots = chi - count
        feasible = remaining & (block_size <= slots)
        return (count < chi) & jnp.any(feasible)

    def body(carry):
        remaining, selected, count = carry
        slots = chi - count
        feasible = remaining & (block_size <= slots)
        dominates = (block_lo[:, None] > block_hi[None, :]) & feasible[:, None] & feasible[None, :]
        frontier = feasible & ~jnp.any(dominates, axis=0)
        order = jnp.lexsort((
            idx,
            block_err,
            -block_rho,
            -block_q0,
            (~frontier).astype(jnp.int32),
        ))
        best = order[0]
        selected = selected.at[count].set(best.astype(jnp.int32))

        def put_pair(selected):
            return selected.at[count + 1].set((best + 1).astype(jnp.int32))

        selected = jax.lax.cond(block_size[best] == 2, put_pair, lambda x: x, selected)
        remaining = remaining.at[best].set(False)
        return remaining, selected, count + block_size[best]

    _, selected, _ = jax.lax.while_loop(
        cond,
        body,
        (block_keep, selected, jnp.array(0, dtype=jnp.int32)),
    )
    return selected


def _arnoldi_ritz_residual_diagnostics(
    A_projected,
    ritz_values,
    Q_R,
    Q_L,
    residual_R,
    residual_L,
    C_R,
    C_L,
    XR,
    XL,
    d_block,
    residual_R_full_width,
    residual_L_full_width,
):
    """Reconstruct relative physical residuals of two-sided Arnoldi Ritz pairs.

    Let ``Q_R`` (N x m) and ``Q_L`` (m x N) be the post-whitening
    biorthogonal bases, ``A_projected = Q_L A Q_R``, and let ``x_i`` and
    ``z_i^H`` be column ``i`` of ``XR`` and row ``i`` of ``XL^H``.  The
    physical Ritz vectors are

        r_i = Q_R x_i,                 l_i^H = z_i^H Q_L.

    ``C_R`` and ``C_L`` map coefficients in the whitened bases back to the
    original right and left Arnoldi bases.  If ``F_R`` and ``F_L`` denote the
    supplied Arnoldi residual arrays, this routine evaluates

        A r_i - lambda_i r_i
          = Q_R (A_projected x_i - lambda_i x_i)
            + (I - Q_R Q_L) F_R C_R x_i,

        l_i^H A - lambda_i l_i^H
          = (z_i^H A_projected - lambda_i z_i^H) Q_L
            + z_i^H C_L F_L^T (I - Q_R Q_L).

    For a compact block-Arnoldi residual, ``F_R`` or ``F_L`` has only
    ``d_block`` columns and therefore acts on the final coefficient block;
    a full-width residual acts on every coefficient.  The first two returns
    are the 2-norms of these residuals divided by ``||r_i||_2`` and
    ``||l_i^H||_2``.  The third is

        kappa_i = ||r_i||_2 ||l_i^H||_2 / |l_i^H r_i|,

    evaluated as ``|z_i^H x_i|`` under the assumed ``Q_L Q_R = I``.
    """
    ZL = XL.conj().T
    Y_R = jnp.dot(C_R, XR)
    Y_R_tail = Y_R if residual_R_full_width else Y_R[-d_block:]
    tail_R = jnp.dot(residual_R, Y_R_tail)
    projected_residual_R = jnp.dot(A_projected, XR) - XR*ritz_values[None, :]
    residual_vec_R = (
        jnp.dot(Q_R, projected_residual_R)
        + tail_R
        - jnp.dot(Q_R, jnp.dot(Q_L, tail_R))
    )

    Y_L = jnp.dot(ZL, C_L)
    Y_L_tail = Y_L if residual_L_full_width else Y_L[:, -d_block:]
    tail_L = jnp.dot(Y_L_tail, residual_L.T)
    projected_residual_L = jnp.dot(ZL, A_projected) - ritz_values[:, None]*ZL
    residual_row_L = (
        jnp.dot(projected_residual_L, Q_L)
        + tail_L
        - jnp.dot(jnp.dot(tail_L, Q_R), Q_L)
    )

    R = jnp.dot(Q_R, XR)
    L = jnp.dot(ZL, Q_L)
    R_norm = jnp.linalg.norm(R, axis=0)
    L_norm = jnp.linalg.norm(L, axis=1)
    overlap = jnp.einsum("ij,ji->i", ZL, XR)
    res_R = (
        jnp.linalg.norm(residual_vec_R, axis=0)
        / jnp.maximum(R_norm, 1e-300)
    )
    res_L = (
        jnp.linalg.norm(residual_row_L, axis=1)
        / jnp.maximum(L_norm, 1e-300)
    )
    condition_estimate = (
        R_norm
        * L_norm
        / jnp.maximum(jnp.abs(overlap), 1e-300)
    )
    return res_R, res_L, condition_estimate

def krylov_block_sylvester_reduced(
    matvec,
    w,
    B,
    X0=None,
    krylov_cfg=None,
    w_triangular=None,
    block_2x2_start=None,
):
    r"""Take one block-GMRES cycle for ``A X - X w = B``.

    Here ``A`` is ``D x D``, while ``X`` and ``B`` are ``D x d_block`` with
    ``D >> d_block``.  ``w`` is a two-dimensional scalar upper or lower
    triangular matrix, selected by ``w_triangular="upper"`` or
    ``w_triangular="lower"``.  Scalar eigenvalues are represented as ``1 x 1``
    diagonal blocks.  For real Schur form, ``block_2x2_start[j]`` marks a
    quasi-triangular block spanning ``j:j + 2``.  Supplying this mask selects
    the real structured-QR FFI kernel; ``None`` selects the JAX scalar-
    triangular kernel used for complex Schur form.

    Write the solution as ``X = X0 + dX`` and move the initial-guess error to
    the right-hand side:

        A dX - dX w = R,
        R = B - A X0 + X0 w.

    Block Arnoldi, seeded by all columns of ``R``, constructs an orthonormal
    basis ``Q`` and the compact Arnoldi relation

        A Q = Q H + F E_tail,

    where ``F`` is the final residual block and ``E_tail`` selects the last
    ``d_block`` rows of a Krylov coefficient vector.  Arnoldi also returns
    ``R = Q[:, :d_block] v``; padding ``v`` with zero rows gives ``R = Q beta``.

    Now restrict the correction to ``dX = Q Y``.  Its equation residual is

        A Q Y - Q Y w - R
          = Q (H Y - Y w - beta) + F E_tail Y.

    Factor ``F = Q_F R_F``.  Arnoldi gives ``Q^H Q_F = 0``, so the physical
    residual norm is exactly the norm of the stacked reduced residual.  Column
    ``j`` of the triangular reduced equation is

        (H - w[j,j] I) Y_j
          = beta_j + sum_{k != j} Y_k w[k,j].

    The triangular structure determines which coefficients on the right are
    already known.  For upper triangular ``w``, only ``k < j`` contributes, so
    the columns are scanned in ascending order:

        rhs_j = beta_j + sum_{k < j} Y_k w[k,j].

    For lower triangular ``w``, only ``k > j`` contributes, so the columns are
    scanned in descending order:

        rhs_j = beta_j + sum_{k > j} Y_k w[k,j].

    In both cases the current and unsolved columns of ``Y`` are zero.  The
    implementation stores ``x = Y.T``, allowing one scan body to form
    ``rhs_j = beta_j + x.T @ w[:,j]`` and call the same shifted least-squares
    kernel used by the diagonal case.  All columns and shifts share one
    block-Arnoldi basis.

    ``error[i]`` is the absolute residual norm of the completed column solve.
    With finite-depth shifted solves, the triangular path is one substitution
    sweep rather than a global minimum-residual solve over every column at
    once.  Passing the returned ``X`` back as ``X0`` recomputes

        R = B - A X0 + X0 w

    and performs a full coupled restart.  This routine deliberately performs
    only one such cycle.

    ``sylvester_compressed`` solves the resulting small problem and returns the
    transpose of its coefficient matrix, so this routine forms the physical
    correction as ``Q @ x.T``. A scalar-triangular problem selects
    ``method="dense_gmres"``; supplying a real-Schur block mask selects
    ``method="schur_gmres"``.
    """
    krylov_cfg = _static_dict(krylov_cfg)
    num_iter = krylov_cfg.get("num_krylov_iter", 3)
    rank_tol = krylov_cfg.get("rank_tol", None)
    rank_tol_arnoldi = 1e-8 if rank_tol is None else rank_tol
    rank_tol_seed = krylov_cfg.get("rank_tol_seed", rank_tol_arnoldi)
    pivot = krylov_cfg.get("pivot", True)
    tpqrt_block_size = krylov_cfg.get("sylvester_tpqrt_block_size", 32)

    X0 = jnp.zeros_like(B) if X0 is None else X0
    R = B - matvec(X0) + jnp.dot(X0, w)

    Q, H, residual, v, _ = arnoldi_basis(
        matvec,
        R,
        num_iter,
        rank_tol=rank_tol_arnoldi,
        rank_tol_seed=rank_tol_seed,
        full_res=False,
        pivot=pivot,
    )

    # Compress the final physical residual block before entering the entirely
    # reduced-space solve.
    residual_r = jnp.linalg.qr(residual, mode="r")
    method = "dense_gmres" if block_2x2_start is None else "schur_gmres"
    x, error = sylvester_compressed(
        H,
        w,
        v,
        residual_r,
        method=method,
        w_triangular=w_triangular,
        block_2x2_start=block_2x2_start,
        tpqrt_block_size=tpqrt_block_size,
    )

    X = X0 + jnp.dot(Q, x.T)
    return X, error


def krylov_deig_one_sided(
    matvec,
    vecmat,
    V_R,
    V_L_T,
    w,
    dmatvec,
    dvecmat,
    dV_R=None,
    dV_L_T=None,
    krylov_cfg=None,
    w_upper_triangular=False,
):
    r"""Differentiate a one-sided invariant subspace with Sylvester Eqn + block GMRES.

    ``V_R`` is a ``D x d`` column basis and ``V_L_T`` is the corresponding
    ``d x D`` row basis, matching the convention of
    ``krylov_eig_one_sided``.  They satisfy

        A V_R = V_R w,       V_L_T A = w V_L_T,
        V_L_T V_R = I.

    Assumes on input that 
        
        V_L_T V_R = Id
        A V_R = V_R w; V_L_T A = w V_L_T

    I'm not sure what happens if this is not the case (TODO)

    ``w`` may be ``None`` or a two-dimensional reduced operator.  ``None``
    recomputes ``V_L_T A V_R`` at the expense of one extra matvec.  Dense
    operators are Schur reduced by default, using real Schur for real input and
    complex Schur otherwise.  Set ``w_upper_triangular=True`` when a supplied
    ``w`` is already in upper Schur form.  Scalar eigenvalues are represented
    by ``1 x 1`` diagonal blocks; there is no one-dimensional diagonal API.
    
    ``dV_R`` and ``dV_L_T`` use the same column/row conventions and are
    optional initial guesses; ``None`` means zero.

    Algorithm:

    Let ``P0 = V_R V_L_T`` be the invariant-subspace projector and
    ``P_perp = I - P0``.  Differentiating the invariant equations in the gauge

        V_L_T dV_R = 0,      V_R^T dV_L_T^T = 0

    gives the projected Sylvester equations

        P_perp A dV_R - dV_R w
            = -P_perp dA V_R,

        P_perp^T A^T dV_L_T^T - dV_L_T^T w^T
            = -P_perp^T dA^T V_L_T^T.

    If ``w`` is dense, take its Schur decomposition

        w = Z T Z^H,

    and transform ``V_R -> V_R Z`` and ``V_L -> V_L Z^*``.  The right
    Sylvester equation then contains upper triangular ``T``, while the left
    equation contains lower triangular ``T^T``.  Both share the block-GMRES
    implementation in ``krylov_block_sylvester_reduced``.  Complex triangular
    forms use ``sylvester_compressed`` in JAX.  Real quasi-triangular forms use
    the CPU FFI kernel, with the 2x2 block mask returned by ``jax_linalg.schur``.

    At convergence this is the JVP of the invariant subspace returned by
    ``krylov_eig_one_sided`` in the gauge-invariant sense

        dP0 = dV_R V_L_T + V_R dV_L_T.

    Individual eigenvector derivatives may differ by an internal basis gauge,
    but ``dP0`` is the differential of ``P0 = V_R V_L_T``.  The returned left
    tangent is ``dV_L_T`` with shape ``d x D``, again matching
    ``krylov_eig_one_sided``.  At finite Krylov depth the returned vectors
    approximate this JVP.  The third return is ``(res_R, res_L)``, the right
    and left per-mode Sylvester residual norms.
    """
    V_L = V_L_T.T
    dV_R = jnp.zeros_like(V_R) if dV_R is None else dV_R
    dV_L = jnp.zeros_like(V_L) if dV_L_T is None else dV_L_T.T

    schur_reduce_w = w is None or not w_upper_triangular
    if w is None:
        w = jnp.dot(V_L.T, matvec(V_R))

    if schur_reduce_w:
        Z, T, block_2x2_start = jax_schur(w)
        V_R = jnp.dot(V_R, Z)
        dV_R = jnp.dot(dV_R, Z)
        V_L = jnp.dot(V_L, Z.conj())
        dV_L = jnp.dot(dV_L, Z.conj())
        w_R = T
        w_L = T.T
    else:
        w_R = w
        w_L = w.T
        block_2x2_start = jnp.zeros((w.shape[0],), dtype=jnp.bool_)

    if jnp.issubdtype(w.dtype, jnp.complexfloating):
        block_2x2_start = None

    w_triangular_R = "upper"
    w_triangular_L = "lower"
    
    def P(x): #Oblique projector
        return x - jnp.dot(V_R, jnp.dot(V_L.T, x))
    def P_matvec(x):
        return P(matvec(x))
    
    dV_R = P(dV_R)
    B = -P(dmatvec(V_R))

    dV_R, res_R =  krylov_block_sylvester_reduced(
        P_matvec,
        w_R,
        B,
        X0=P(dV_R),
        krylov_cfg=krylov_cfg,
        w_triangular=w_triangular_R,
        block_2x2_start=block_2x2_start,
    )

    def PT(x): #Oblique projector
        return x - jnp.dot(V_L, jnp.dot(V_R.T, x))
    def PT_vecmat(x):
        return PT(vecmat(x))
    
    B = -PT(dvecmat(V_L))
    dV_L = PT(dV_L)
    dV_L, res_L =  krylov_block_sylvester_reduced(
        PT_vecmat,
        w_L,
        B,
        X0=dV_L,
        krylov_cfg=krylov_cfg,
        w_triangular=w_triangular_L,
        block_2x2_start=block_2x2_start,
    )

    if schur_reduce_w:
        dV_R = jnp.dot(dV_R, Z.conj().T)
        dV_L = jnp.dot(dV_L, Z.T)

    return dV_R, dV_L.T, (res_R, res_L)

def krylov_eig_one_sided(
    matvec,
    vecmat,
    V_R,
    V_L,
    chi_max=None,
    info_level=0,
    debug=None,
    krylov_cfg=None,
):
    """Run one-sided Arnoldi extraction and return diagnostics by level.

    ``chi_max`` defaults to the right seed width ``V_R.shape[1]``.
    Level 1 supplies convergence spectra, residuals, seed overlaps, and ranks.
    Level 2 adds extended projected diagnostics.  Level 3 additionally
    reapplies the physical operators to check the full Arnoldi structure.
    """
    if chi_max is None:
        chi_max = V_R.shape[1]
    if info_level is None:
        info_level = 0
    info_level = int(info_level)
    debug = KrylovDebug() if debug is None else debug
    _, debug_x, debug_y, debug_sweep = debug
    krylov_cfg = _static_dict(krylov_cfg)
    num_iter = krylov_cfg.get('num_krylov_iter', 3)
    basis_method = krylov_cfg.get('basis_method', "arnoldi")
    ritz_q0_lock_tol = krylov_cfg.get('ritz_q0_lock_tol', 1e-7)
    CTM_eig_cutoff = krylov_cfg.get('CTM_eig_cutoff', 2e-14)
    ctm_rank_tol = krylov_cfg.get('ctm_rank_tol', 1e-14)
    paired_edge_policy = krylov_cfg.get('paired_edge_policy', "fill_real")
    rank_tol = krylov_cfg.get('rank_tol', None)
    pivot = krylov_cfg.get('pivot', True)

    krylov_info = {}
    d_block = V_R.shape[1]
    rank_tol_arnoldi = 1e-8 if rank_tol is None else rank_tol
    rank_tol_seed = krylov_cfg.get('rank_tol_seed', rank_tol_arnoldi)
    seed_rank_tol = rank_tol_seed

    if basis_method != "arnoldi":
        raise ValueError(
            "krylov_eig_one_sided supports only basis_method='arnoldi', "
            f"got {basis_method!r}"
        )
    Q_R, H_R, residual_R, _, active_R = arnoldi_basis(
        matvec,
        V_R,
        num_iter,
        rank_tol=rank_tol_arnoldi,
        rank_tol_seed=rank_tol_seed,
        full_res=True,
        pivot=pivot,
    )
    Q_L, H_L, residual_L, _, active_L = arnoldi_basis(
        vecmat,
        V_L.T,
        num_iter,
        rank_tol=rank_tol_arnoldi,
        rank_tol_seed=rank_tol_seed,
        full_res=True,
        pivot=pivot,
    )
    if info_level >= 2:
        krylov_info["active_R_rank"] = jnp.sum(active_R)
        krylov_info["active_L_rank"] = jnp.sum(active_L)

    def arnoldi_structure_info():
        """Reapply both operators and return full Arnoldi-structure errors."""
        (
            arnoldi_err_R,
            arnoldi_err_L,
            ortho_err_R,
            ortho_err_L,
            inactive_H_R_count,
            inactive_H_L_count,
            rank_R,
            rank_L,
        ) = check_arnoldi_structure(
            matvec,
            vecmat,
            Q_R,
            Q_L,
            H_R,
            H_L,
            residual_R,
            residual_L,
            active_R,
            active_L,
            d_block=d_block,
        )
        return {
            "arnoldi_structure_err_R": arnoldi_err_R,
            "arnoldi_structure_err_L": arnoldi_err_L,
            "arnoldi_structure_ortho_err_R": ortho_err_R,
            "arnoldi_structure_ortho_err_L": ortho_err_L,
            "arnoldi_structure_inactive_H_R_count": inactive_H_R_count,
            "arnoldi_structure_inactive_H_L_count": inactive_H_L_count,
            "arnoldi_structure_rank_R": rank_R,
            "arnoldi_structure_rank_L": rank_L,
        }

    if info_level >= 3:
        krylov_info.update(arnoldi_structure_info())

    XR, ritz_sqrt_abs_R, ritz_phase_R, info_R = dominant_eigenspace_one_sided(
        H_R,
        chi_max,
        info_level=info_level,
        residual=residual_R,
        ritz_q0_lock_tol=ritz_q0_lock_tol,
        CTM_eig_cutoff=CTM_eig_cutoff,
        paired_edge_policy=paired_edge_policy,
    )

    XL_col, ritz_sqrt_abs_L, ritz_phase_L, info_L = dominant_eigenspace_one_sided(
        H_L,
        chi_max,
        info_level=info_level,
        residual=residual_L,
        ritz_q0_lock_tol=ritz_q0_lock_tol,
        CTM_eig_cutoff=CTM_eig_cutoff,
        paired_edge_policy=paired_edge_policy,
    )
    krylov_info["ritz_rank_R"] = info_R["ritz_rank"]
    krylov_info["ritz_rank_L"] = info_L["ritz_rank"]
    krylov_info["dropped_paired_edge_R"] = info_R["dropped_paired_edge"]
    krylov_info["dropped_paired_edge_L"] = info_L["dropped_paired_edge"]
    krylov_info["ritz_rank"] = info_R["ritz_rank"]
    ctm_rank_R = jnp.sum(jnp.abs(info_R["ritz_values"]) > ctm_rank_tol)
    ctm_rank_L = jnp.sum(jnp.abs(info_L["ritz_values"]) > ctm_rank_tol)
    krylov_info["ctm_rank_R"] = ctm_rank_R
    krylov_info["ctm_rank_L"] = ctm_rank_L
    krylov_info["ctm_rank_tol"] = jnp.asarray(ctm_rank_tol)
    krylov_info["ctm_rank"] = jnp.minimum(ctm_rank_R, ctm_rank_L)
    ritz_rank_mismatch = info_R["ritz_rank"] != info_L["ritz_rank"]
    krylov_info["ritz_rank_mismatch"] = ritz_rank_mismatch
    mismatch_ritz_R = info_R["ritz_values"][:chi_max+2]
    mismatch_ritz_L = info_L["ritz_values"][:chi_max+2]
    jax.lax.cond(
        ritz_rank_mismatch,
        lambda _: jax.debug.callback(
            functools.partial(_debug_print_one_sided_ritz_mismatch, debug_sweep),
            debug_x,
            debug_y,
            info_R["ritz_rank"],
            info_L["ritz_rank"],
            info_R["dropped_paired_edge"],
            info_L["dropped_paired_edge"],
            mismatch_ritz_R,
            mismatch_ritz_L,
            ordered=True,
        ),
        lambda _: None,
        operand=None,
    )
    if info_level >= 1:
        krylov_info.update({
            "ritz_values_R": info_R["ritz_values"],
            "ritz_values_kept_R": info_R["ritz_values_kept"],
            "ritz_q0_defect_full_R": info_R["ritz_q0_defect_full"],
            "ritz_cutoff_value_R": info_R["ritz_cutoff_value"],
            "ritz_residual_R": info_R["ritz_residual"],
            "ritz_residual_kept_R": info_R["ritz_residual_kept"],
            "ritz_values_L": info_L["ritz_values"],
            "ritz_values_kept_L": info_L["ritz_values_kept"],
            "ritz_q0_defect_full_L": info_L["ritz_q0_defect_full"],
            "ritz_cutoff_value_L": info_L["ritz_cutoff_value"],
            "ritz_residual_L": info_L["ritz_residual"],
            "ritz_residual_kept_L": info_L["ritz_residual_kept"],
            "ritz_values": info_R["ritz_values"],
            "ritz_values_kept": info_R["ritz_values_kept"],
            "ritz_cutoff_value": info_R["ritz_cutoff_value"],
            "ritz_residual": info_R["ritz_residual"],
            "ritz_residual_kept": info_R["ritz_residual_kept"],
        })

    M00 = jnp.dot(Q_L[:, :d_block].T, Q_R[:, :d_block])
    s00 = jnp.linalg.svdvals(M00)
    krylov_info["initial_M00_cond"] = s00[-1]

    H10 = H_R[d_block:2*d_block, :d_block]
    H_scale = jnp.abs(jnp.trace(H_R))
    seed_R_mask = active_R[:d_block]
    seed_R_rank = jnp.sum(seed_R_mask)
    H10 = H10 * seed_R_mask[None, :]
    H10 = H10 / H_scale
    krylov_info["initial_ortho_res_R"] = jnp.linalg.norm(H10)

    H10 = H_L[d_block:2*d_block, :d_block]
    H_scale = jnp.abs(jnp.trace(H_L))
    seed_L_mask = active_L[:d_block]
    seed_L_rank = jnp.sum(seed_L_mask)
    H10 = H10 * seed_L_mask[None, :]
    H10 = H10 / H_scale
    krylov_info["initial_ortho_res_L"] =  jnp.linalg.norm(H10)

    Q_R_arnoldi = Q_R
    Q_L_arnoldi = Q_L
    Q_R = jnp.dot(Q_R_arnoldi, XR)
    Q_L = jnp.dot(XL_col.T, Q_L_arnoldi.T)

    if info_level >= 2:
        ritz_op_R = _selected_ritz_operator(info_R["ritz_values_kept"], H_R.dtype)
        ritz_op_L = _selected_ritz_operator(info_L["ritz_values_kept"], H_L.dtype)
        residual_XR = (
            jnp.dot(residual_R, XR)
            if residual_R.shape[1] == XR.shape[0]
            else jnp.dot(residual_R, XR[-d_block:])
        )
        residual_XL = (
            jnp.dot(residual_L, XL_col)
            if residual_L.shape[1] == XL_col.shape[0]
            else jnp.dot(residual_L, XL_col[-d_block:])
        )
        ritz_res_R = (
            jnp.dot(Q_R_arnoldi, jnp.dot(H_R, XR) - jnp.dot(XR, ritz_op_R))
            + residual_XR
        )
        Q_L_col_dbg = Q_L.T
        ritz_res_L = (
            jnp.dot(Q_L_arnoldi, jnp.dot(H_L, XL_col) - jnp.dot(XL_col, ritz_op_L))
            + residual_XL
        )
        ritz_res_R_col = (
            jnp.linalg.norm(ritz_res_R, axis=0)
            / jnp.maximum(jnp.linalg.norm(Q_R, axis=0), 1e-300)
        )
        ritz_res_L_col = (
            jnp.linalg.norm(ritz_res_L, axis=0)
            / jnp.maximum(jnp.linalg.norm(Q_L_col_dbg, axis=0), 1e-300)
        )
        krylov_info["selected_ritz_res_R"] = ritz_res_R_col
        krylov_info["selected_ritz_res_L"] = ritz_res_L_col

    Q_R, Q_L, _, _, _, s, basis_rank = biorthogonalize_bases(
        Q_R,
        Q_L,
        tol=10.,
    )
    krylov_info["basis_overlap_svals"] = s
    krylov_info["basis_rank"] = basis_rank
    Q_out_R, _, _, _ = qrp_basis_and_rank(Q_R, seed_rank_tol)
    Q_out_L, _, _, _ = qrp_basis_and_rank(Q_L.T, seed_rank_tol)
    krylov_info["seed_R_svals"] = _masked_seed_overlap_svals(
        Q_R_arnoldi[:, :d_block],
        Q_out_R,
        seed_R_mask,
        basis_rank,
    )
    krylov_info["seed_L_svals"] = _masked_seed_overlap_svals(
        Q_L_arnoldi[:, :d_block],
        Q_out_L,
        seed_L_mask,
        basis_rank,
    )
    krylov_info["seed_R_rank"] = seed_R_rank
    krylov_info["seed_L_rank"] = seed_L_rank
    if info_level >= 1:
        biorth = jnp.dot(Q_L, Q_R)
        idx = jnp.arange(biorth.shape[0])
        Id_rank = jnp.eye(biorth.shape[0], dtype=biorth.dtype)*(idx < basis_rank)[:, None]
        krylov_info["final_biorth_err"] = jnp.linalg.norm(
            biorth - Id_rank
        )
        active_s_idx = jnp.maximum(basis_rank, 1) - 1
        krylov_info["M_krylov_cond"] = s[active_s_idx]
        krylov_info["residual_R"] = jnp.linalg.norm(residual_R)
        krylov_info["residual_L"] = jnp.linalg.norm(residual_L)

    ritz_sqrt_abs_pinv = jnp.where(ritz_sqrt_abs_R > 1e-16, 1.0 / ritz_sqrt_abs_R, 0.0)
    return (
        Q_R,
        Q_L,
        ritz_sqrt_abs_pinv,
        ritz_phase_R,
        krylov_info,
    )


def krylov_eigh(matvec, V, chi_max, info_level=0, krylov_cfg=None):
    """Hermitian one-sided Arnoldi extraction using dense projected ``eigh``."""
    if info_level is None:
        info_level = 0
    info_level = int(info_level)
    krylov_cfg = _static_dict(krylov_cfg)
    num_iter = krylov_cfg.get('num_krylov_iter', 3)
    CTM_eig_cutoff = krylov_cfg.get('CTM_eig_cutoff', 1e-15)
    cutoff_relative_to_one = krylov_cfg.get('cutoff_relative_to_one', False)
    rank_tol = krylov_cfg.get('rank_tol', None)
    ritz_q0_lock_tol = krylov_cfg.get('ritz_q0_lock_tol', 0.0)
    pivot = krylov_cfg.get('pivot', True)

    d_block = V.shape[1]
    rank_tol_arnoldi = 1e-8 if rank_tol is None else rank_tol
    rank_tol_seed = krylov_cfg.get('rank_tol_seed', rank_tol_arnoldi)
    seed_rank_tol = rank_tol_seed
    Q, H, residual, _, active = arnoldi_basis(
        matvec,
        V,
        num_iter,
        rank_tol=rank_tol_arnoldi,
        rank_tol_seed=rank_tol_seed,
        full_res=True,
        pivot=pivot,
    )
    H10 = H[d_block:2*d_block, :d_block]
    H_scale = jnp.abs(jnp.trace(H))
    seed_mask = active[:d_block]
    seed_rank = jnp.sum(seed_mask)
    H10 = H10 * seed_mask[None, :]
    initial_ortho_res = jnp.linalg.norm(H10 / H_scale)
    X, ritz_kept, info = dominant_eigenspace_eigh(
        H,
        chi_max,
        info_level=info_level,
        residual=residual,
        ritz_q0_lock_tol=ritz_q0_lock_tol,
        CTM_eig_cutoff=CTM_eig_cutoff,
        cutoff_relative_to_one=cutoff_relative_to_one,
    )
    Q_arnoldi = Q
    Q = jnp.dot(Q_arnoldi, X)
    krylov_info = {
        "ritz_rank": info["ritz_rank"],
        "basis_rank": info["ritz_rank"],
        "ritz_values": info["ritz_values"],
        "ritz_cutoff_value": info["ritz_cutoff_value"],
        "initial_ortho_res": initial_ortho_res,
        "seed_rank": seed_rank,
    }
    if info_level >= 1:
        krylov_info.update(info)
        krylov_info["active_rank"] = jnp.sum(active)
    if info_level >= 2:
        ritz_op = _selected_ritz_operator(info["ritz_values_kept"], H.dtype)
        ritz_res = matvec(Q) - jnp.dot(Q, ritz_op)
        krylov_info["selected_ritz_res"] = (
            jnp.linalg.norm(ritz_res, axis=0)
            / jnp.maximum(jnp.linalg.norm(Q, axis=0), 1e-300)
        )
        Q_out, _, _, _ = qrp_basis_and_rank(Q, seed_rank_tol)
        krylov_info["seed_svals"] = _masked_seed_overlap_svals(
            Q_arnoldi[:, :d_block],
            Q_out,
            seed_mask,
            krylov_info["basis_rank"],
        )
    return Q, ritz_kept, krylov_info


@functools.partial(
    jax.jit,
    static_argnames=(
        "chi",
        "info_level",
        "CTM_eig_cutoff",
        "cutoff_relative_to_one",
        "ritz_q0_lock_tol",
    ),
)
def dominant_eigenspace_eigh(
    A,
    chi,
    info_level=0,
    residual=None,
    ritz_q0_lock_tol=0.0,
    CTM_eig_cutoff=1e-14,
    cutoff_relative_to_one=False,
):
    """Return dominant Hermitian Ritz vectors and selected Ritz values."""
    info_level = int(info_level)
    w, X = jnp.linalg.eigh(A)
    if residual is None:
        residual = jnp.zeros_like(A)
    ritz_abs_unsorted = jnp.abs(w)
    ritz_residual_unsorted = jnp.linalg.norm(jnp.dot(residual, X), axis=0)
    ritz_q0_weight_unsorted = jnp.linalg.norm(X[:chi], axis=0)
    ritz_q0_defect_unsorted = 1.0 - ritz_q0_weight_unsorted
    ritz_q0_locked_unsorted = ritz_q0_defect_unsorted < ritz_q0_lock_tol
    p = jnp.lexsort((
        -w,
        -ritz_abs_unsorted,
        (~ritz_q0_locked_unsorted).astype(jnp.int32),
    ))
    w = w[p]
    X = X[:, p]
    if cutoff_relative_to_one:
        ritz_cutoff_value = jnp.asarray(CTM_eig_cutoff, dtype=ritz_abs_unsorted.dtype)
    else:
        ritz_cutoff_value = CTM_eig_cutoff*jnp.abs(jnp.sum(w))
    keep_ritz = jnp.abs(w) >= ritz_cutoff_value
    keep_slots = keep_ritz[:chi]
    X = X[:, :chi]
    X = X*keep_slots[None, :]
    ritz_kept = jnp.where(keep_slots, w[:chi], 0.0)
    ritz_residual_ordered = ritz_residual_unsorted[p]
    ritz_residual_kept = jnp.where(keep_slots, ritz_residual_ordered[:chi], 0.0)
    ritz_rank = jnp.sum(keep_slots)
    info = {
        "ritz_values": w,
        "ritz_rank": ritz_rank,
        "ritz_cutoff_value": ritz_cutoff_value,
    }
    if info_level >= 1:
        info.update({
            "ritz_values_kept": ritz_kept,
            "ritz_q0_defect_full": ritz_q0_defect_unsorted[p],
            "ritz_q0_locked_full": ritz_q0_locked_unsorted[p],
            "ritz_q0_lock_tol": jnp.asarray(ritz_q0_lock_tol),
            "ritz_residual": ritz_residual_ordered,
            "ritz_residual_kept": ritz_residual_kept,
        })
    return X, ritz_kept, info


@functools.partial(
    jax.jit,
    static_argnames=(
        "chi",
        "info_level",
        "CTM_eig_cutoff",
        "paired_edge_policy",
    ),
)
def dominant_eigenspace_one_sided(
    A,
    chi,
    info_level=0,
    residual=None,
    ritz_q0_lock_tol=1e-6,
    CTM_eig_cutoff=1e-14,
    paired_edge_policy="fill_real",
):
    """Return dominant right eigenvectors of ``A`` without left-vector scaling."""
    info_level = int(info_level)
    if paired_edge_policy not in ("fill_real", "drop_paired_edge"):
        raise ValueError(f"Unknown paired_edge_policy {paired_edge_policy!r}")
    w, X = jax.lax.linalg.eig(
        A,
        compute_left_eigenvectors=False,
        compute_right_eigenvectors=True,
    )
    X = X / jnp.linalg.norm(X, axis=0)[None, :]
    ritz_abs_unsorted = jnp.abs(w)
    ritz_residual_unsorted = jnp.linalg.norm(jnp.dot(residual, X), axis=0)
    ritz_q0_weight_unsorted = jnp.linalg.norm(X[:chi], axis=0)
    ritz_q0_defect_unsorted = 1.0 - ritz_q0_weight_unsorted
    ritz_q0_locked_unsorted = ritz_q0_defect_unsorted < ritz_q0_lock_tol
    p = jnp.lexsort((
        -jnp.imag(w),
        -jnp.real(w),
        -ritz_abs_unsorted,
        (~ritz_q0_locked_unsorted).astype(jnp.int32),
    ))
    w = w[p]
    X = X[:, p]
    ritz_cutoff_value = CTM_eig_cutoff*jnp.abs(jnp.sum(w))
    keep_ritz = jnp.abs(w) >= ritz_cutoff_value
    ritz_q0_defect_full = ritz_q0_defect_unsorted[p]
    ritz_residual_ordered = ritz_residual_unsorted[p]
    if jnp.issubdtype(A.dtype, jnp.complexfloating):
        keep_slots = keep_ritz[:chi]
        X = X[:, :chi]
        X = X*keep_slots[None, :]
        ritz_values = jnp.where(keep_slots, w[:chi], 0.0)
        ritz_abs = jnp.abs(ritz_values)
        ritz_sqrt_abs = jnp.sqrt(ritz_abs)
        ritz_phase = jnp.where(ritz_abs > 1e-16, ritz_values / ritz_abs, 0.0)
        ritz_residual_kept = jnp.where(keep_slots, ritz_residual_ordered[:chi], 0.0)
        ritz_rank = jnp.sum(keep_slots)
        dropped_paired_edge = jnp.array(False)
    else:
        tol = 100*jnp.finfo(A.dtype).eps*jnp.abs(w)
        is_complex = jnp.abs(jnp.imag(w)) > tol
        is_pair_first = is_complex & (jnp.imag(w) > 0)
        (
            X,
            ritz_values,
            ritz_sqrt_abs,
            ritz_phase,
            ritz_residual_kept,
            ritz_rank,
            dropped_paired_edge,
        ) = _dominant_real_basis_and_ritz_one_sided(
            X,
            w,
            keep_ritz,
            is_complex,
            is_pair_first,
            chi,
            A.dtype,
            ritz_residual_ordered,
            paired_edge_policy == "drop_paired_edge",
        )
    info = {
        "ritz_values": w,
        "ritz_rank": ritz_rank,
        "dropped_paired_edge": dropped_paired_edge,
    }
    if info_level >= 1:
        info.update({
            "ritz_values_kept": ritz_values,
            "ritz_q0_defect_full": ritz_q0_defect_full,
            "ritz_cutoff_value": ritz_cutoff_value,
            "ritz_residual": ritz_residual_ordered,
            "ritz_residual_kept": ritz_residual_kept,
        })

    return X, ritz_sqrt_abs, ritz_phase, info


def block_mgs(Q, V, j):
    """Apply ``h_i=Q_i^H V_i``, ``V_{i+1}=V_i-Q_i h_i`` for ``i<=j``.

    Return ``V_{j+1}`` and all ``h_i``, with ``h_i=0`` for ``i>j``.
    """
    num_iter = Q.shape[0]
    d_block = V.shape[1]
    h0 = jnp.zeros((d_block, d_block), dtype=V.dtype)

    def step(V, i):
        def active(_):
            h = jnp.dot(Q[i].conj().T, V)
            return V - jnp.dot(Q[i], h), h

        return jax.lax.cond(i <= j, active, lambda _: (V, h0), operand=None)

    return jax.lax.scan(step, V, jnp.arange(num_iter))


def block_cgs(Q, V, j):
    """Return ``h_i=Q_i^H V`` and ``V'=V-sum_i Q_i h_i``, assuming ``Q_i=0`` for ``i>j``."""
    del j
    # Q[i,n,a]^* V[n,b] -> h[i,a,b]
    h = jnp.einsum("ina,nb->iab", Q.conj(), V)
    # Q[i,n,a] h[i,a,b] -> projection[n,b]
    V = V - jnp.einsum("ina,iab->nb", Q, h)
    return V, h


@functools.partial(
    jax.jit,
    static_argnums=(0, 2, 3, 5, 6),
    static_argnames=("rank_tol", "rank_tol_seed", "full_res", "pivot"),
)
def arnoldi_basis(
    matvec,
    V,
    num_iter,
    max_reortho=3,
    eta=1.0/np.sqrt(2.0),
    rank_tol=1e-8,
    ortho_method="MGS",
    rank_tol_seed=None,
    full_res=False,
    pivot=True,
):
    """Block Arnoldi with operator and seed-coordinate factorizations.

        By default only the final residual block is returned. ``full_res=True``
        also retains residuals from intermediate rank deflation and returns the
        complete fixed-shape Arnoldi relation.

        The returned ``v`` has shape ``(d_block, d_block)`` and satisfies
        ``V = Q[:, :d_block] @ v`` up to seed-rank truncation and roundoff.
        The return order is ``Q, H, residual, v, active_cols``.

        ``pivot=False`` compiles a separate kernel using only unpivoted QR;
        it contains no column-permutation gathers or scatters.

        0 < rank_tol << 1  (e.g. eps_M < rank_tol < 1) will allow the arnoldi to "invent"
        new orthogonal directions not present in the seed Krylov space K = {A^m V}

        rank_tol = O(1) should force deflation if K is rank deficient

        rank_tol = 0 is not recommended because we can't gaurantee the resulting basis is orthogonal
    """
    rank_tol_seed = rank_tol if rank_tol_seed is None else rank_tol_seed
    d_block = V.shape[1]
    N = V.shape[0]
    num_iter = min(num_iter, N // d_block)
    Q = jnp.zeros((num_iter, N, d_block), dtype=V.dtype)
    active_cols = jnp.zeros((num_iter, d_block), dtype=jnp.bool_)

    def ortho(Q, V, j):
        if ortho_method == "MGS":
            return block_mgs(Q, V, j)
        elif ortho_method == "CGS":
            return block_cgs(Q, V, j)
        else:
            raise ValueError(f"Unknown ortho_method: {ortho_method!r}")

    def reortho(Q, V, j):
        hj = jnp.zeros((num_iter, d_block, d_block), dtype=V.dtype)
        accepted = jnp.array(False)
        k = jnp.array(0, dtype=jnp.int32)

        def cond(carry):
            _, _, k, accepted = carry
            return (k < max_reortho) & (~accepted)

        def step(carry):
            V, hj, k, _ = carry
            old_nrm = jnp.linalg.norm(V, axis=0)
            V, dh = ortho(Q, V, j)
            new_nrm = jnp.linalg.norm(V, axis=0)
            accepted = jnp.all(new_nrm >= eta*old_nrm)
            return V, hj + dh, k + 1, accepted

        V, hj, _, _ = jax.lax.while_loop(cond, step, (V, hj, k, accepted))
        return V, hj

    def mask_rank(X, r, old_nrm):
        if pivot:
            q, r_piv, p = split_qrp(r)
            X = jnp.dot(X, q)
            scale = jnp.take(old_nrm, p)
        else:
            r_piv = r
            p = None
            scale = old_nrm

        diag_nrm = jnp.abs(jnp.diag(r_piv))
        cutoff = rank_tol*jnp.finfo(diag_nrm.dtype).eps*scale

        if rank_tol:
            keep = diag_nrm > cutoff
            dropped = ~keep
            dropped_residual = (
                jnp.dot(X, r_piv*dropped[:, None])
                if full_res
                else None
            )
            X = X*keep[None, :]
            r_piv = r_piv*keep[:, None]
        else:
            keep = jnp.ones((d_block,), dtype=jnp.bool_)
            dropped_residual = jnp.zeros_like(X) if full_res else None

        return X, r_piv, keep, p, dropped_residual

    def mask_seed(V):
        seed_nrm = jnp.linalg.norm(V)
        if pivot:
            V, r_piv, p = split_qrp(V)
        else:
            V, r_piv = jnp.linalg.qr(V, mode="reduced")
        row_nrm = jnp.linalg.norm(r_piv, axis=1)
        cutoff = rank_tol_seed*jnp.finfo(row_nrm.dtype).eps*seed_nrm
        if rank_tol_seed:
            keep = row_nrm > cutoff
            V = V*keep[None, :]
            r_piv = r_piv*keep[:, None]
        else:
            keep = jnp.ones((d_block,), dtype=jnp.bool_)
        if pivot:
            # V_input[:, p] = V r_piv, so V_input = V v with v[:, p] = r_piv.
            v = jnp.zeros_like(r_piv).at[:, p].set(r_piv)
        else:
            v = r_piv
        return V, v, keep

    V, v, seed_active = mask_seed(V)
    Q = Q.at[0].set(V)
    active_cols = active_cols.at[0].set(seed_active)
    seed_order = jnp.arange(d_block, dtype=jnp.int32) if pivot else None

    def project_and_normalize(Q, V, j):
        # Maintain V0 = sum_i Q_i H_i + X R while reorthogonalizing X.
        old_nrm = jnp.linalg.norm(V, axis=0)
        X, hj = ortho(Q, V, j)
        X, r = jnp.linalg.qr(X, mode="reduced")
        accepted = jnp.all(jnp.abs(jnp.diag(r)) >= eta*old_nrm)
        k = jnp.array(1, dtype=jnp.int32)

        def cond(carry):
            _, _, _, k, accepted = carry
            return (k < max_reortho) & (~accepted)

        def step(carry):
            X, hj, r, k, _ = carry
            X, dh = ortho(Q, X, j)
            X, r2 = jnp.linalg.qr(X, mode="reduced")
            hj = hj + jnp.einsum("iab,bc->iac", dh, r)
            r = jnp.dot(r2, r)
            accepted = jnp.all(jnp.abs(jnp.diag(r2)) >= eta)
            return X, hj, r, k + 1, accepted

        X, hj, r, _, _ = jax.lax.while_loop(cond, step, (X, hj, r, k, accepted))
        X, r, active_next, p, dropped_residual = mask_rank(X, r, old_nrm)
        return X, hj, r, active_next, p, dropped_residual

    def arnoldi_step(carry, j):
        Q, H_cols, residual_blocks, seed_order, active_cols, active = carry
        hj0 = jnp.zeros((num_iter, d_block, d_block), dtype=V.dtype)
        def active_step(args):
            Q, H_cols, residual_blocks, seed_order, active_cols = args
            # W = A Q_j, W <- W - sum_{i <= j} Q_i H_{ij}, W = Q_{j+1} R.
            Vj = matvec(Q[j])
            Vj, hj, r, active_next, p, dropped_residual = project_and_normalize(Q, Vj, j)

            if pivot:
                Q = Q.at[j].set(jnp.take(Q[j], p, axis=1))
                seed_order = jax.lax.cond(
                    j == 0,
                    lambda _: p,
                    lambda x: x,
                    seed_order,
                )
                active_cols = active_cols.at[j].set(jnp.take(active_cols[j], p))
                H_cols = H_cols.at[j - 1, j].set(
                    jnp.take(H_cols[j - 1, j], p, axis=0)
                )
                hj = jnp.take(hj, p, axis=2)
                hj = hj.at[j].set(jnp.take(hj[j], p, axis=0))
            active_cols = active_cols.at[j + 1].set(active_next)
            hj = hj.at[j + 1].set(r)
            Q = Q.at[j + 1].set(Vj)
            H_cols = H_cols.at[j].set(hj)
            if full_res:
                residual_blocks = residual_blocks.at[j].set(dropped_residual)
            return Q, H_cols, residual_blocks, seed_order, active_cols, jnp.any(active_next)

        def inactive_step(args):
            Q, H_cols, residual_blocks, seed_order, active_cols = args
            H_cols = H_cols.at[j].set(hj0)
            return Q, H_cols, residual_blocks, seed_order, active_cols, jnp.array(False)

        Q, H_cols, residual_blocks, seed_order, active_cols, active_next = jax.lax.cond(
            active,
            active_step,
            inactive_step,
            (Q, H_cols, residual_blocks, seed_order, active_cols),
        )
        return (Q, H_cols, residual_blocks, seed_order, active_cols, active_next), None

    H_cols_work = jnp.zeros((num_iter, num_iter, d_block, d_block), dtype=V.dtype)
    residual_blocks = (
        jnp.zeros((num_iter, N, d_block), dtype=V.dtype)
        if full_res
        else None
    )
    (Q, H_cols_work, residual_blocks, seed_order, active_cols, active), _ = jax.lax.scan(
        arnoldi_step,
        (Q, H_cols_work, residual_blocks, seed_order, active_cols, jnp.any(seed_active)),
        jnp.arange(num_iter - 1),
    )
    H_cols = H_cols_work[:num_iter - 1]
    if pivot:
        v = jnp.take(v, seed_order, axis=0)

    def final_active(Q):
        V_last = matvec(Q[num_iter - 1])
        return reortho(Q, V_last, num_iter - 1)

    def final_inactive(Q):
        residual = jnp.zeros((N, d_block), dtype=V.dtype)
        h_last = jnp.zeros((num_iter, d_block, d_block), dtype=V.dtype)
        return residual, h_last

    residual, h_last = jax.lax.cond(active, final_active, final_inactive, Q)
    if full_res:
        residual_blocks = residual_blocks.at[num_iter - 1].add(residual)
    H = jnp.concatenate([H_cols, h_last[None]], axis=0)

    Q = jnp.transpose(Q, [1, 0, 2])
    Q = jnp.reshape(Q, (N, num_iter*d_block))
    H = jnp.transpose(H, [1, 2, 0, 3])
    H = jnp.reshape(H, (num_iter*d_block, num_iter*d_block))
    active_cols = jnp.reshape(active_cols, (num_iter*d_block,))
    if full_res:
        residual = jnp.transpose(residual_blocks, [1, 0, 2])
        residual = jnp.reshape(residual, (N, num_iter*d_block))
        residual_coeff = jnp.dot(Q.T.conj(), residual)
        residual_coeff = residual_coeff*active_cols[:, None]*active_cols[None, :]
        H = H + residual_coeff
        residual = residual - jnp.dot(Q, residual_coeff)
    return Q, H, residual, v, active_cols


@jax.jit
def reduce_arnoldi_rank(Q, H, residual, active, rank):
    """Mask an Arnoldi relation down to the first `rank` active columns.

    Given A Q = Q H + residual, return same-shape Q_red, H_red, residual_red
    satisfying A Q_red = Q_red H_red + residual_red.  ``residual`` may be a
    full-width matrix or a single final block.
    """
    active_count = jnp.cumsum(active.astype(jnp.int32))
    retained = active & (active_count <= rank)
    dropped = ~retained
    Q_red = Q*retained[None, :]
    H_red = H*retained[:, None]*retained[None, :]

    dropped_weights = H*dropped[:, None]*retained[None, :]
    residual_red = jnp.dot(Q, dropped_weights)

    if residual.shape[1] == H.shape[0]:
        residual_red = residual_red + residual*retained[None, :]
    else:
        d_block = residual.shape[1]
        last_start = H.shape[0] - d_block
        residual_red = residual_red.at[:, last_start:].add(
            residual*retained[last_start:][None, :]
        )
    return Q_red, H_red, residual_red




@functools.partial(jax.jit, static_argnums=(0, 1, 4))
def bilanczos_basis(matvec, vecmat, Q_R, Q_L, num_iter):
    """Block bi-Lanczos using the local recurrence plus one full reorthogonalization."""
    d_block = Q_R.shape[1]
    N = Q_R.shape[0]
    num_iter = min(num_iter, N // d_block)
    QR = jnp.zeros((num_iter, N, d_block), dtype=Q_R.dtype)
    QL = jnp.zeros((num_iter, N, d_block), dtype=Q_L.dtype)

    def biorthogonalize(R, L):
        M = jnp.dot(L.T, R)
        U, s, Vh = jnp.linalg.svd(M)
        cutoff = 2.0*jnp.finfo(s.dtype).eps*jnp.maximum(s[0], 1.0)
        s_keep = s > cutoff
        rank = jnp.sum(s_keep)
        s_safe = jnp.where(s_keep, s, 1.0)
        s_inv_sqrt = jnp.where(s_keep, 1.0/jnp.sqrt(s_safe), 0.0)
        s = jnp.where(s_keep, s, 0.0)
        R = jnp.dot(R, Vh.T.conj())*s_inv_sqrt[None, :]
        L = jnp.dot(L, U.conj())*s_inv_sqrt[None, :]
        s_sqrt = jnp.sqrt(s)
        hR = s_sqrt[:, None]*Vh
        hL = s_sqrt[:, None]*U.T
        return R, L, hR, hL, rank

    Q_R, Q_L, _, _, rank = biorthogonalize(Q_R, Q_L)
    #jax.debug.print("rank in {}", rank)
    QR = QR.at[0].set(Q_R)
    QL = QL.at[0].set(Q_L)
    T_blocks = jnp.zeros((num_iter, num_iter, d_block, d_block), dtype=Q_R.dtype)

    def lanczos_ortho(QR, QL, T_blocks, R, L, j):
        hR = jnp.zeros((num_iter, d_block, d_block), dtype=R.dtype)
        hL = jnp.zeros((num_iter, d_block, d_block), dtype=L.dtype)

        def subtract_previous(args):
            R, L, hR, hL = args
            hR_prev = T_blocks[j - 1, j]
            hL_prev = T_blocks[j, j - 1].T
            R = R - jnp.einsum("na,ab->nb", QR[j - 1], hR_prev)
            L = L - jnp.einsum("na,ab->nb", QL[j - 1], hL_prev)
            hR = hR.at[j - 1].set(hR_prev)
            hL = hL.at[j - 1].set(hL_prev)
            return R, L, hR, hL

        R, L, hR, hL = jax.lax.cond(
            j > 0,
            subtract_previous,
            lambda args: args,
            (R, L, hR, hL),
        )
        hR_j = jnp.einsum("na,nb->ab", QL[j], R)
        hL_j = jnp.einsum("na,nb->ab", QR[j], L)
        R = R - jnp.einsum("na,ab->nb", QR[j], hR_j)
        L = L - jnp.einsum("na,ab->nb", QL[j], hL_j)
        hR = hR.at[j].set(hR_j)
        hL = hL.at[j].set(hL_j)
        return R, L, hR, hL

    def full_reortho(QR, QL, R, L, j):
        hR0 = jnp.zeros((d_block, d_block), dtype=R.dtype)
        hL0 = jnp.zeros((d_block, d_block), dtype=L.dtype)

        def step(carry, i):
            def active(carry):
                R, L = carry
                # QL[i,n,a] R[n,b] -> hR[a,b]
                hR = jnp.einsum("na,nb->ab", QL[i], R)
                # QR[i,n,a] L[n,b] -> hL[a,b]
                hL = jnp.einsum("na,nb->ab", QR[i], L)
                R = R - jnp.einsum("na,ab->nb", QR[i], hR)
                L = L - jnp.einsum("na,ab->nb", QL[i], hL)
                return (R, L), (hR, hL)

            return jax.lax.cond(i <= j, active, lambda carry: (carry, (hR0, hL0)), operand=carry)

        (R, L), (hR, hL) = jax.lax.scan(step, (R, L), jnp.arange(num_iter))
        return R, L, hR, hL

    def project(QR, QL, T_blocks, R, L, j):
        R, L, hR, hL = lanczos_ortho(QR, QL, T_blocks, R, L, j)
        R, L, dhR, dhL = full_reortho(QR, QL, R, L, j)
        return R, L, hR + dhR, hL + dhL

    def set_projected_row_col(T_blocks, j, hR, hL):
        T_blocks = T_blocks.at[:, j].set(hR)
        T_blocks = T_blocks.at[j].set(jnp.transpose(hL, [0, 2, 1]))
        return T_blocks

    def bilanczos_step(carry):
        j, QR, QL, T_blocks, residual_R, residual_L, _ = carry
        R = matvec(QR[j])
        L = vecmat(QL[j])
        R, L, hR, hL = project(QR, QL, T_blocks, R, L, j)
        T_blocks = set_projected_row_col(T_blocks, j, hR, hL)
        R_next, L_next, hR_next, hL_next, rank_next = biorthogonalize(R, L)
        #jax.debug.print("  rank {}", rank_next)
        active_next = rank_next > 0
        T_blocks = T_blocks.at[j + 1, j].set(hR_next)
        T_blocks = T_blocks.at[j, j + 1].set(hL_next.T)
        QR = QR.at[j + 1].set(R_next)
        QL = QL.at[j + 1].set(L_next)
        residual_R = jnp.where(active_next, residual_R, R)
        residual_L = jnp.where(active_next, residual_L, L)
        return j + 1, QR, QL, T_blocks, residual_R, residual_L, active_next

    def cond(carry):
        j, _, _, _, _, _, active = carry
        return (j < num_iter - 1) & active

    active = rank > 0
    residual_R = jnp.zeros((N, d_block), dtype=Q_R.dtype)
    residual_L = jnp.zeros((N, d_block), dtype=Q_L.dtype)
    j, QR, QL, T_blocks, residual_R, residual_L, active = jax.lax.while_loop(
        cond,
        bilanczos_step,
        (
            jnp.array(0, dtype=jnp.int32),
            QR,
            QL,
            T_blocks,
            residual_R,
            residual_L,
            active,
        )
    )

    def final_project(args):
        QR, QL, T_blocks, residual_R, residual_L = args
        R = matvec(QR[j])
        L = vecmat(QL[j])
        residual_R, residual_L, hR, hL = project(QR, QL, T_blocks, R, L, j)
        T_blocks = set_projected_row_col(T_blocks, j, hR, hL)
        return QR, QL, T_blocks, residual_R, residual_L

    QR, QL, T_blocks, residual_R, residual_L = jax.lax.cond(
        active,
        final_project,
        lambda args: args,
        (QR, QL, T_blocks, residual_R, residual_L),
    )

    QR_mat = jnp.reshape(jnp.transpose(QR, [1, 0, 2]), (N, num_iter*d_block))
    QL_mat = jnp.reshape(jnp.transpose(QL, [1, 0, 2]), (N, num_iter*d_block))
    T = jnp.reshape(jnp.transpose(T_blocks, [0, 2, 1, 3]), (num_iter*d_block, num_iter*d_block))
    actual_num_iter = jnp.where(active, j + 1, j)
    if num_iter > 1:
        T10_norm = jnp.linalg.norm(T_blocks[1, 0])
        T01_norm = jnp.linalg.norm(T_blocks[0, 1])
    else:
        zero_norm = jnp.zeros((), dtype=jnp.real(T_blocks[0, 0, 0, 0]).dtype)
        T10_norm = zero_norm
        T01_norm = zero_norm
    info = {
        "bilanczos_num_iter": actual_num_iter,
        "bilanczos_T10_norm": T10_norm,
        "bilanczos_T01_norm": T01_norm,
    }
    return QR_mat, QL_mat, T, residual_R, residual_L, info


@functools.partial(jax.jit, static_argnums=(0, 1, 4))
def bilanczos_basis_v2(matvec, vecmat, Q_R, Q_L, num_iter):
    """Block bi-Lanczos with Netlib-style ordering and tridiagonal T blocks."""
    d_block = Q_R.shape[1]
    N = Q_R.shape[0]
    num_iter = min(num_iter, N // d_block)
    QR = jnp.zeros((num_iter, N, d_block), dtype=Q_R.dtype)
    QL = jnp.zeros((num_iter, N, d_block), dtype=Q_L.dtype)

    def biorthogonalize(R, L):
        M = jnp.dot(L.T, R)
        U, s, Vh = jnp.linalg.svd(M)
        s_sqrt = jnp.sqrt(s)
        cutoff = 0.0*jnp.finfo(s.dtype).eps*jnp.maximum(s_sqrt[0], 1.0)
        s_keep = s_sqrt > cutoff
        rank = jnp.sum(s_keep)
        s_sqrt_safe = jnp.where(s_keep, s_sqrt, 1.0)
        s_inv_sqrt = jnp.where(s_keep, 1.0/s_sqrt_safe, 0.0)
        s_sqrt = jnp.where(s_keep, s_sqrt, 0.0)
        R = jnp.dot(R, Vh.T.conj())*s_inv_sqrt[None, :]
        L = jnp.dot(L, U.conj())*s_inv_sqrt[None, :]
        hR = s_sqrt[:, None]*Vh
        hL = s_sqrt[:, None]*U.T
        return R, L, hR, hL, rank

    Q_R, Q_L, _, _, rank = biorthogonalize(Q_R, Q_L)
    QR = QR.at[0].set(Q_R)
    QL = QL.at[0].set(Q_L)
    T_blocks = jnp.zeros((num_iter, num_iter, d_block, d_block), dtype=Q_R.dtype)

    def lanczos_ortho(QR, QL, T_blocks, R, L, j):
        """Apply the local three-term block bi-Lanczos projection.

        With ``R = A_op QR[j]`` and ``L = A_op.T QL[j]``, subtract the previous
        off-diagonal recurrence blocks ``QR[j - 1] B_j`` and ``QL[j - 1] C_j.T``.
        Then compute the diagonal block from the two equivalent projections,
        average them in the right-action orientation, and remove
        ``QR[j] A_j`` / ``QL[j] A_j.T``.  The returned residuals are the
        unnormalized candidates for the next right/left Lanczos blocks.
        """
        def subtract_previous(args):
            R, L = args
            B = T_blocks[j - 1, j]
            C_T = T_blocks[j, j - 1].T
            R = R - jnp.einsum("na,ab->nb", QR[j - 1], B)
            L = L - jnp.einsum("na,ab->nb", QL[j - 1], C_T)
            return R, L

        R, L = jax.lax.cond(
            j > 0,
            subtract_previous,
            lambda args: args,
            (R, L),
        )
        A_from_R = jnp.einsum("na,nb->ab", QL[j], R)
        A_from_L = jnp.einsum("na,nb->ab", QR[j], L)
        A = 0.5*(A_from_R + A_from_L.T)
        R = R - jnp.einsum("na,ab->nb", QR[j], A)
        L = L - jnp.einsum("na,ab->nb", QL[j], A.T)
        return R, L, A

    def rebiorthogonalize(QR, QL, R, L, j):
        """Clean up a newly normalized block against the existing bi-basis.

        After SVD biorthogonalization gives candidate blocks ``R`` and ``L``
        with ``L.T R = I``, remove the small overlaps ``QL[i].T R`` and
        ``QR[i].T L`` for all existing blocks ``i <= j``.  Following the
        Netlib ordering, these detected cleanup coefficients are not folded
        back into the block-tridiagonal recurrence matrix; only their norms are
        returned as diagnostics.
        """
        zero_norm = jnp.zeros((), dtype=jnp.real(R[0, 0]).dtype)

        def step(carry, i):
            def active(carry):
                R, L = carry
                # Netlib TSMGS: keep these cleanup coefficients out of T.
                C = jnp.einsum("na,nb->ab", QL[i], R)
                B_T = jnp.einsum("na,nb->ab", QR[i], L)
                R = R - jnp.einsum("na,ab->nb", QR[i], C)
                L = L - jnp.einsum("na,ab->nb", QL[i], B_T)
                return (R, L), (jnp.linalg.norm(B_T), jnp.linalg.norm(C))

            return jax.lax.cond(
                i <= j,
                active,
                lambda carry: (carry, (zero_norm, zero_norm)),
                operand=carry,
            )

        (R, L), (dB_norms, dC_norms) = jax.lax.scan(step, (R, L), jnp.arange(num_iter))
        return R, L, dB_norms, dC_norms

    def bilanczos_step(carry):
        j, QR, QL, T_blocks, residual_R, residual_L, _ = carry
        R = matvec(QR[j])
        L = vecmat(QL[j])
        residual_R, residual_L, A = lanczos_ortho(QR, QL, T_blocks, R, L, j)
        T_blocks = T_blocks.at[j, j].set(A)

        def build_next(args):
            QR, QL, T_blocks = args
            R_next, L_next, C_next, B_next_T, rank_next = biorthogonalize(residual_R, residual_L)
            R_next, L_next, dB_norms, dC_norms = rebiorthogonalize(QR, QL, R_next, L_next, j)
            QR = QR.at[j + 1].set(R_next)
            QL = QL.at[j + 1].set(L_next)
            T_blocks = T_blocks.at[j + 1, j].set(C_next)
            T_blocks = T_blocks.at[j, j + 1].set(B_next_T.T)
            return QR, QL, T_blocks, rank_next > 0

        QR, QL, T_blocks, active_next = jax.lax.cond(
            j < num_iter - 1,
            build_next,
            lambda args: (*args, jnp.array(False)),
            (QR, QL, T_blocks),
        )
        return j + 1, QR, QL, T_blocks, residual_R, residual_L, active_next

    def cond(carry):
        _, _, _, _, _, _, active = carry
        return active

    active = rank > 0
    residual_R = jnp.zeros((N, d_block), dtype=Q_R.dtype)
    residual_L = jnp.zeros((N, d_block), dtype=Q_L.dtype)
    actual_num_iter, QR, QL, T_blocks, residual_R, residual_L, active = jax.lax.while_loop(
        cond,
        bilanczos_step,
        (
            jnp.array(0, dtype=jnp.int32),
            QR,
            QL,
            T_blocks,
            residual_R,
            residual_L,
            active,
        )
    )
    QR_mat = jnp.reshape(jnp.transpose(QR, [1, 0, 2]), (N, num_iter*d_block))
    QL_mat = jnp.reshape(jnp.transpose(QL, [1, 0, 2]), (N, num_iter*d_block))
    T = jnp.reshape(jnp.transpose(T_blocks, [0, 2, 1, 3]), (num_iter*d_block, num_iter*d_block))
    if num_iter > 1:
        T10_norm = jnp.where(actual_num_iter > 1, jnp.linalg.norm(T_blocks[1, 0]), 0.0)
        T01_norm = jnp.where(actual_num_iter > 1, jnp.linalg.norm(T_blocks[0, 1]), 0.0)
    else:
        zero_norm = jnp.zeros((), dtype=jnp.real(T_blocks[0, 0, 0, 0]).dtype)
        T10_norm = zero_norm
        T01_norm = zero_norm
    info = {
        "bilanczos_num_iter": actual_num_iter,
        "bilanczos_T10_norm": T10_norm,
        "bilanczos_T01_norm": T01_norm,
    }
    return QR_mat, QL_mat, T, residual_R, residual_L, info


def _selected_ritz_operator(ritz_values, dtype):
    """Build the selected Ritz operator in the basis returned by eigenspace selection.

    For complex arithmetic this is just diag(lambda).  For real arithmetic,
    complex-conjugate Ritz pairs are represented by the real 2x2 block
    [[Re(lambda), Im(lambda)], [-Im(lambda), Re(lambda)]], so the projected
    relation can still be checked as H X = X J without leaving real dtype.
    """
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return jnp.diag(ritz_values.astype(dtype))
    lam_real = jnp.real(ritz_values).astype(dtype)
    lam_imag = jnp.imag(ritz_values).astype(dtype)
    ritz_op = jnp.diag(lam_real)
    idx = jnp.arange(ritz_values.shape[0] - 1)
    tol = 100*jnp.finfo(dtype).eps*jnp.maximum(1, jnp.abs(ritz_values[:-1]))
    pair_first = jnp.abs(lam_imag[:-1]) > tol
    pair_first = pair_first & (lam_imag[:-1] > 0)
    ritz_op = ritz_op.at[idx, idx + 1].set(jnp.where(pair_first, lam_imag[:-1], 0.0))
    ritz_op = ritz_op.at[idx + 1, idx].set(jnp.where(pair_first, -lam_imag[:-1], 0.0))
    return ritz_op

def check_arnoldi_structure(
    matvec,
    vecmat,
    Q_R,
    Q_L,
    H_R,
    H_L,
    residual_R,
    residual_L,
    active_R=None,
    active_L=None,
    d_block=None,
):
    """Return Arnoldi-relation and orthogonality errors for block bases.

    ``Q_R`` and ``Q_L`` are flattened block Arnoldi bases with columns ordered
    as ``[Q_0, Q_1, ...]`` where each block has width ``d_block``.  ``matvec``
    and ``vecmat`` act on one ``(N, d_block)`` block at a time, so this helper
    maps them over the block axis before comparing ``A Q = Q H + residual``.
    The residual may be full-width, or it may be a single final block from the
    older no-intermediate-deflation relation.
    If ``active_R`` / ``active_L`` are supplied, they mark nontrivial columns
    of the flattened bases, and the orthogonality checks compare ``Q^D Q`` to
    ``diag(active)``.  The inactive rows/columns of ``H_R`` and ``H_L`` are
    also checked for exact zeros.  Otherwise all columns are treated as active.
    It returns ``(arnoldi_err_R, arnoldi_err_L, ortho_err_R, ortho_err_L,
    inactive_H_R_count, inactive_H_L_count, rank_R, rank_L)``.
    """
    if d_block is None:
        d_block = residual_R.shape[1]

    def apply_block_mat(mat, Q):
        num_blocks = Q.shape[1] // d_block
        Q_blocks = jnp.reshape(Q, (Q.shape[0], num_blocks, d_block))
        Q_blocks = jnp.transpose(Q_blocks, (1, 0, 2))
        AQ_blocks = jax.vmap(mat)(Q_blocks)
        AQ = jnp.transpose(AQ_blocks, (1, 0, 2))
        return jnp.reshape(AQ, Q.shape)

    if residual_R.shape[1] == Q_R.shape[1]:
        residual_R_full = residual_R
    else:
        residual_R_full = jnp.zeros_like(Q_R).at[:, -d_block:].set(residual_R)
    if residual_L.shape[1] == Q_L.shape[1]:
        residual_L_full = residual_L
    else:
        residual_L_full = jnp.zeros_like(Q_L).at[:, -d_block:].set(residual_L)
    arnoldi_err_R = jnp.linalg.norm(
        apply_block_mat(matvec, Q_R) - (jnp.dot(Q_R, H_R) + residual_R_full)
    )
    arnoldi_err_L = jnp.linalg.norm(
        apply_block_mat(vecmat, Q_L) - (jnp.dot(Q_L, H_L) + residual_L_full)
    )
    if active_R is None:
        active_R = jnp.ones((Q_R.shape[1],), dtype=jnp.bool_)
    if active_L is None:
        active_L = jnp.ones((Q_L.shape[1],), dtype=jnp.bool_)
    active_diag_R = jnp.diag(active_R.astype(Q_R.dtype))
    active_diag_L = jnp.diag(active_L.astype(Q_L.dtype))
    ortho_err_R = jnp.linalg.norm(jnp.dot(Q_R.T.conj(), Q_R) - active_diag_R)
    ortho_err_L = jnp.linalg.norm(jnp.dot(Q_L.T.conj(), Q_L) - active_diag_L)
    H_inactive_R = (H_R != 0) & ((~active_R)[:, None] | (~active_R)[None, :])
    H_inactive_L = (H_L != 0) & ((~active_L)[:, None] | (~active_L)[None, :])
    inactive_H_R_count = jnp.sum(H_inactive_R)
    inactive_H_L_count = jnp.sum(H_inactive_L)
    rank_R = jnp.sum(active_R)
    rank_L = jnp.sum(active_L)
    return (
        arnoldi_err_R,
        arnoldi_err_L,
        ortho_err_R,
        ortho_err_L,
        inactive_H_R_count,
        inactive_H_L_count,
        rank_R,
        rank_L,
    )


def _masked_seed_overlap_svals(Q_seed, Q_out, seed_keep, out_rank):
    """Return seed/output overlap singular values after fixed-shape masking."""
    overlap = jnp.dot(Q_seed.T.conj(), Q_out)
    col_keep = jnp.arange(overlap.shape[1]) < out_rank
    overlap = jnp.where(seed_keep[:, None] & col_keep[None, :], overlap, 0.0)
    return jnp.linalg.svd(overlap, compute_uv=False)


def _dominant_real_basis_and_ritz(XL, XR, w, keep_ritz, is_complex, is_pair_first, chi, dtype):
    def put_col(out_L, out_R, count, col_L, col_R):
        def put(args):
            out_L, out_R, count, col_L, col_R = args
            return out_L.at[count].set(col_L), out_R.at[count].set(col_R), count + 1
        return jax.lax.cond(
            count < chi,
            put,
            lambda args: (args[0], args[1], args[2]),
            (out_L, out_R, count, col_L, col_R),
        )

    def put_real_ritz(ritz_values, ritz_sqrt_abs, ritz_phase, count, lam):
        lam_abs = jnp.abs(lam)
        lam_real = jnp.real(lam).astype(dtype)
        lam_sign = jnp.where(lam_abs > 1e-16, lam_real / lam_abs, 0.0)
        ritz_values = ritz_values.at[count].set(lam)
        ritz_sqrt_abs = ritz_sqrt_abs.at[count].set(jnp.sqrt(lam_abs))
        ritz_phase = ritz_phase.at[count, count].set(lam_sign)
        return ritz_values, ritz_sqrt_abs, ritz_phase

    def put_pair_ritz(ritz_values, ritz_sqrt_abs, ritz_phase, count, lam):
        lam_abs = jnp.abs(lam)
        lam_sqrt_abs = jnp.sqrt(lam_abs)
        c = jnp.where(lam_abs > 1e-16, jnp.real(lam) / lam_abs, 0.0).astype(dtype)
        s = jnp.where(lam_abs > 1e-16, jnp.imag(lam) / lam_abs, 0.0).astype(dtype)
        ritz_values = ritz_values.at[count].set(lam)
        ritz_values = ritz_values.at[count + 1].set(jnp.conj(lam))
        ritz_sqrt_abs = ritz_sqrt_abs.at[count].set(lam_sqrt_abs)
        ritz_sqrt_abs = ritz_sqrt_abs.at[count + 1].set(lam_sqrt_abs)
        ritz_phase = ritz_phase.at[count, count].set(c)
        ritz_phase = ritz_phase.at[count, count + 1].set(s)
        ritz_phase = ritz_phase.at[count + 1, count].set(-s)
        ritz_phase = ritz_phase.at[count + 1, count + 1].set(c)
        return ritz_values, ritz_sqrt_abs, ritz_phase

    def step(carry, args):
        out_L, out_R, ritz_values, ritz_sqrt_abs, ritz_phase, count = carry
        XL_i, XR_i, lam_i, keep_i, complex_i, pair_first_i = args
        real_col_L = jnp.real(XL_i)
        imag_col_L = jnp.imag(XL_i)
        real_col_R = jnp.real(XR_i)
        imag_col_R = jnp.imag(XR_i)

        def take_real(_):
            out_L1, out_R1, count1 = put_col(out_L, out_R, count, real_col_L, real_col_R)
            ritz_values1, ritz_sqrt_abs1, ritz_phase1 = put_real_ritz(
                ritz_values,
                ritz_sqrt_abs,
                ritz_phase,
                count,
                lam_i,
            )
            return out_L1, out_R1, ritz_values1, ritz_sqrt_abs1, ritz_phase1, count1

        def take_pair(_):
            enough_room = count + 1 < chi

            def put_pair(_):
                out_L1, out_R1, count1 = put_col(out_L, out_R, count, real_col_L, real_col_R)
                out_L2, out_R2, count2 = put_col(out_L1, out_R1, count1, imag_col_L, imag_col_R)
                ritz_values1, ritz_sqrt_abs1, ritz_phase1 = put_pair_ritz(
                    ritz_values,
                    ritz_sqrt_abs,
                    ritz_phase,
                    count,
                    lam_i,
                )
                return out_L2, out_R2, ritz_values1, ritz_sqrt_abs1, ritz_phase1, count2

            return jax.lax.cond(
                enough_room,
                put_pair,
                lambda _: (out_L, out_R, ritz_values, ritz_sqrt_abs, ritz_phase, count),
                operand=None,
            )

        def complex_case(_):
            return jax.lax.cond(
                pair_first_i,
                take_pair,
                lambda _: (out_L, out_R, ritz_values, ritz_sqrt_abs, ritz_phase, count),
                operand=None,
            )

        return jax.lax.cond(
            keep_i,
            lambda _: jax.lax.cond(complex_i, complex_case, take_real, operand=None),
            lambda _: carry,
            operand=None,
        ), None

    out_L = jnp.zeros((chi, XL.shape[0]), dtype=dtype)
    out_R = jnp.zeros((chi, XR.shape[0]), dtype=dtype)
    ritz_values = jnp.zeros((chi,), dtype=w.dtype)
    ritz_sqrt_abs = jnp.zeros((chi,), dtype=jnp.real(w).dtype)
    ritz_phase = jnp.zeros((chi, chi), dtype=dtype)
    (out_L, out_R, ritz_values, ritz_sqrt_abs, ritz_phase, count), _ = jax.lax.scan(
        step,
        (out_L, out_R, ritz_values, ritz_sqrt_abs, ritz_phase, jnp.array(0, dtype=jnp.int32)),
        (XL.T, XR.T, w, keep_ritz, is_complex, is_pair_first),
    )
    return out_L, out_R, ritz_values, ritz_sqrt_abs, ritz_phase


def _dominant_real_basis_and_ritz_one_sided(
    X,
    w,
    keep_ritz,
    is_complex,
    is_pair_first,
    chi,
    dtype,
    ritz_residual_in,
    drop_paired_edge_policy,
):
    """Select real Ritz-basis columns from a one-sided complex eigensystem."""
    def put_col(out_X, count, col):
        def put(args):
            out_X, count, col = args
            return out_X.at[count].set(col), count + 1
        return jax.lax.cond(
            count < chi,
            put,
            lambda args: (args[0], args[1]),
            (out_X, count, col),
        )

    def put_real_ritz(
        ritz_values,
        ritz_sqrt_abs,
        ritz_phase,
        ritz_residual,
        count,
        lam,
        ritz_res,
    ):
        lam_abs = jnp.abs(lam)
        lam_real = jnp.real(lam).astype(dtype)
        lam_sign = jnp.where(lam_abs > 1e-16, lam_real / lam_abs, 0.0)
        ritz_values = ritz_values.at[count].set(lam)
        ritz_sqrt_abs = ritz_sqrt_abs.at[count].set(jnp.sqrt(lam_abs))
        ritz_phase = ritz_phase.at[count, count].set(lam_sign)
        ritz_residual = ritz_residual.at[count].set(ritz_res)
        return (
            ritz_values,
            ritz_sqrt_abs,
            ritz_phase,
            ritz_residual,
        )

    def put_pair_ritz(
        ritz_values,
        ritz_sqrt_abs,
        ritz_phase,
        ritz_residual,
        count,
        lam,
        ritz_res,
    ):
        lam_abs = jnp.abs(lam)
        lam_sqrt_abs = jnp.sqrt(lam_abs)
        c = jnp.where(lam_abs > 1e-16, jnp.real(lam) / lam_abs, 0.0).astype(dtype)
        s = jnp.where(lam_abs > 1e-16, jnp.imag(lam) / lam_abs, 0.0).astype(dtype)
        ritz_values = ritz_values.at[count].set(lam)
        ritz_values = ritz_values.at[count + 1].set(jnp.conj(lam))
        ritz_sqrt_abs = ritz_sqrt_abs.at[count].set(lam_sqrt_abs)
        ritz_sqrt_abs = ritz_sqrt_abs.at[count + 1].set(lam_sqrt_abs)
        ritz_phase = ritz_phase.at[count, count].set(c)
        ritz_phase = ritz_phase.at[count, count + 1].set(s)
        ritz_phase = ritz_phase.at[count + 1, count].set(-s)
        ritz_phase = ritz_phase.at[count + 1, count + 1].set(c)
        ritz_residual = ritz_residual.at[count].set(ritz_res)
        ritz_residual = ritz_residual.at[count + 1].set(ritz_res)
        return (
            ritz_values,
            ritz_sqrt_abs,
            ritz_phase,
            ritz_residual,
        )

    def step(carry, args):
        (
            out_X,
            ritz_values,
            ritz_sqrt_abs,
            ritz_phase,
            ritz_residual,
            count,
            dropped_paired_edge,
        ) = carry
        X_i, lam_i, keep_i, complex_i, pair_first_i, ritz_res_i = args
        real_col = jnp.real(X_i)
        imag_col = jnp.imag(X_i)

        def take_real(_):
            out_X1, count1 = put_col(out_X, count, real_col)
            (
                ritz_values1,
                ritz_sqrt_abs1,
                ritz_phase1,
                ritz_residual1,
            ) = put_real_ritz(
                ritz_values,
                ritz_sqrt_abs,
                ritz_phase,
                ritz_residual,
                count,
                lam_i,
                ritz_res_i,
            )
            return (
                out_X1,
                ritz_values1,
                ritz_sqrt_abs1,
                ritz_phase1,
                ritz_residual1,
                count1,
                dropped_paired_edge,
            )

        def take_pair(_):
            enough_room = count + 1 < chi

            def put_pair(_):
                out_X1, count1 = put_col(out_X, count, real_col)
                out_X2, count2 = put_col(out_X1, count1, imag_col)
                (
                    ritz_values1,
                    ritz_sqrt_abs1,
                    ritz_phase1,
                    ritz_residual1,
                ) = put_pair_ritz(
                    ritz_values,
                    ritz_sqrt_abs,
                    ritz_phase,
                    ritz_residual,
                    count,
                    lam_i,
                    ritz_res_i,
                )
                return (
                    out_X2,
                    ritz_values1,
                    ritz_sqrt_abs1,
                    ritz_phase1,
                    ritz_residual1,
                    count2,
                    dropped_paired_edge,
                )

            def drop_or_skip_pair(_):
                return (
                    out_X,
                    ritz_values,
                    ritz_sqrt_abs,
                    ritz_phase,
                    ritz_residual,
                    count,
                    jnp.asarray(drop_paired_edge_policy),
                )

            return jax.lax.cond(
                enough_room,
                put_pair,
                drop_or_skip_pair,
                operand=None,
            )

        def complex_case(_):
            return jax.lax.cond(
                pair_first_i,
                take_pair,
                lambda _: (
                    out_X,
                    ritz_values,
                    ritz_sqrt_abs,
                    ritz_phase,
                    ritz_residual,
                    count,
                    dropped_paired_edge,
                ),
                operand=None,
            )

        def select_candidate(_):
            return jax.lax.cond(
                keep_i,
                lambda _: jax.lax.cond(complex_i, complex_case, take_real, operand=None),
                lambda _: carry,
                operand=None,
            )

        return jax.lax.cond(
            dropped_paired_edge,
            lambda _: carry,
            select_candidate,
            operand=None,
        ), None

    out_X = jnp.zeros((chi, X.shape[0]), dtype=dtype)
    ritz_values = jnp.zeros((chi,), dtype=w.dtype)
    ritz_sqrt_abs = jnp.zeros((chi,), dtype=jnp.real(w).dtype)
    ritz_phase = jnp.zeros((chi, chi), dtype=dtype)
    ritz_residual = jnp.zeros((chi,), dtype=ritz_residual_in.dtype)
    (
        out_X,
        ritz_values,
        ritz_sqrt_abs,
        ritz_phase,
        ritz_residual,
        count,
        dropped_paired_edge,
    ), _ = jax.lax.scan(
        step,
        (
            out_X,
            ritz_values,
            ritz_sqrt_abs,
            ritz_phase,
            ritz_residual,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
        ),
        (
            X.T,
            w,
            keep_ritz,
            is_complex,
            is_pair_first,
            ritz_residual_in,
        ),
    )
    return (
        out_X.T,
        ritz_values,
        ritz_sqrt_abs,
        ritz_phase,
        ritz_residual,
        count,
        dropped_paired_edge,
    )




__all__ = [
    "KrylovDebug",
    "dump_krylov_schur_context",
    "krylov_eig",
    "krylov_eig_one_sided",
    "krylov_eigh",
    "check_arnoldi_structure",
    "eig_cleaned",
    "dominant_eigenspace",
    "dominant_eigenspace_one_sided",
    "dominant_eigenspace_eigh",
    "ritz_interval_q0_order",
    "arnoldi_basis",
    "reduce_arnoldi_rank",
    "bilanczos_basis",
    "bilanczos_basis_v2",
    "_materialize_krylov_actions",
    "_materialize_rectangular_block_action",
    "_materialize_block_action",
]
