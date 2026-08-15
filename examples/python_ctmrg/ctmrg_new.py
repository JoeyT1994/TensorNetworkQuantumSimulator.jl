"""Local corner-transfer-matrix updates.

The local ``Z22`` network is

    c1[x, y+1] ---- A1[x, y+1] ---- A1[x+1, y+1] ---- c0[x+1, y+1]
         |               |                 |                 |
    A2[x, y+1] ---- T[x, y+1] ----- T[x+1, y+1] ---- A0[x+1, y+1]
         |               |                 |                 |
    A2[x, y] ------ T[x, y] ------- T[x+1, y] ------ A0[x+1, y]
         |               |                 |                 |
    c2[x, y] ------ A3[x, y] ------ A3[x+1, y] ------ c3[x+1, y]

with the boundary ring ordered counterclockwise as in ``CTM/ctm_state.py``.

The high-level local-update contract is

    Y_new = local_update(X, Y_guess),

where ``X`` is the read-only ``Z22`` network drawn above. The updated tensors
are

    Y = (
        A0[x, y:y+2],
        A1[x:x+2, y],
        A2[x+1, y:y+2],
        A3[x:x+2, y+1],
        c0[x, y],
        c1[x+1, y],
        c2[x+1, y+1],
        c3[x, y+1],
    ).
"""

import functools
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ctm_primitives import construct_CTM_k
from linalg.jax_linalg import (
    biorthogonalize_bases,
    qrp_basis_and_rank,
    triangular_pinv_solve,
)
from linalg.krylov import _static_dict, krylov_eig_one_sided
from linalg.periodic_krylov_schur import (
    periodic_krylov_schur_projectors,
    periodic_power_projectors,
)
from linalg.periodic_schur.jax_ffi import (
    periodic_schur_D,
    periodic_schur_Z,
    periodic_schur_eigenvalues,
)


NUM_KRYLOV_ITER = 3
PERIODIC_NUM_KRYLOV_ITER = 6
DEFAULT_PROJECTOR_METHOD = "eig one sided"


# Spatial offset of quadrant C_k within a local 2 x 2 stencil.
offset_k = ((1, 1), (0, 1), (0, 0), (1, 0))


class Z22(NamedTuple):
    """A 2 x 2 site patch enclosed by two edge tensors per side and four corners."""

    T: jax.Array | tuple[jax.Array, jax.Array]
    A: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    c: tuple[jax.Array, jax.Array, jax.Array, jax.Array]


class CTMRGInfo(NamedTuple):
    """Local, sweep, or iteration-stacked CTMRG diagnostics."""

    rank: jax.Array
    dVL: jax.Array
    dVR: jax.Array
    dVL_rank_in: jax.Array
    dVL_rank_out: jax.Array
    dVR_rank_in: jax.Array
    dVR_rank_out: jax.Array
    c_cycle_eigvals_in: jax.Array
    c_cycle_eigvals_out: jax.Array
    c_svals_in: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    c_svals_out: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    c_lstsq_rank_in: jax.Array
    bond_biorth_svals: jax.Array
    cycle_ritz_rank_R: jax.Array
    cycle_ritz_rank_L: jax.Array
    ortho_res_R: jax.Array
    ortho_res_L: jax.Array

    @property
    def num_iterations(self):
        """Return the iteration count for info returned by ``ctmrg``."""
        return self.dVL.shape[0]


def _c_pinv_solve(
    c,
    b,
    rank=None,
    *,
    schur_form=False,
    left_side=False,
    lower=False,
    transpose_a=False,
    rcond=1e-14,
):
    """Apply a dense or triangular padded pseudoinverse of ``c`` to ``b``."""
    if schur_form:
        return triangular_pinv_solve(
            c,
            b,
            rank,
            left_side=left_side,
            lower=lower,
            transpose_a=transpose_a,
        )

    op_c = jnp.swapaxes(c, -1, -2) if transpose_a else c
    c_pinv = jnp.linalg.pinv(op_c, rtol=rcond)
    return c_pinv @ b if left_side else b @ c_pinv


def V_from_Ac(A, c, rcond=1e-14, *, rank=None, schur_form=False):
    """Convert stencils into ``(VL0,...,VL3)`` and ``(VR0,...,VR3)``.

    The cyclic convention is ``VL[j] @ C[j] = c[j] @ VL[j+1]`` and
    ``C[j] @ VR[j+1] = VR[j] @ c[j]``, with indices modulo four. Each
    ``VL[j]`` has shape ``(chi, N[j])`` and each ``VR[j]`` has shape
    ``(N[j], chi)``.

    Writing ``s[j] = offset_k[j]`` and taking all labels modulo four, the
    defining equations are

    ``c[j-1][s[j+1]] @ VL[j] = A[j-1][s[j]] @ c[j-1][s[j]]``,

    ``VR[j] @ c[j][s[j+2]] = c[j][s[j+3]] @ A[j+1][s[j+3]]``.

    The physical A leg and the adjacent boundary leg are fused into ``N[j]``.
    For example, ``j=0`` gives
    ``c3[0,1] @ VL0 = A3[1,1] @ c3[1,1]`` and
    ``VR0 @ c0[0,0] = c0[1,0] @ A1[1,0]``.

    With ``schur_form=True``, ``rank`` selects the common leading active sector
    of the periodic-Schur corner cycle.  Individual corner factors may have
    additional inactive Schur blocks; the coordinate solves ignore them.
    """
    VL = []
    for j in range(4):
        edge = (j - 1) % 4
        xy = offset_k[j]
        xy_inner = offset_k[(j + 1) % 4]
        # VL_j = c_{j-1,int}^{-1} A_{j-1} c_{j-1,ext}.
        c_inner = c[edge][xy_inner]
        rhs = A[edge][xy] @ c[edge][xy]
        rhs = jnp.moveaxis(rhs, -2, 0).reshape((c_inner.shape[-1], -1))
        V = _c_pinv_solve(
            c_inner,
            rhs,
            rank,
            schur_form=schur_form,
            left_side=True,
            rcond=rcond,
        )
        VL.append(V)

    VR = []
    for j in range(4):
        edge = (j + 1) % 4
        xy = offset_k[(j + 3) % 4]
        xy_inner = offset_k[(j + 2) % 4]
        # VR_j = c_{j,ext} A_{j+1} c_{j,int}^{-1}.
        c_inner = c[j][xy_inner]
        rhs = c[j][xy] @ A[edge][xy]
        rhs = jnp.swapaxes(rhs, -1, -2)
        rhs = jnp.moveaxis(rhs, -2, 0).reshape((c_inner.shape[-1], -1))
        V = _c_pinv_solve(
            c_inner,
            rhs,
            rank,
            schur_form=schur_form,
            left_side=True,
            transpose_a=True,
            rcond=rcond,
        )
        VR.append(V.T)

    # Both tuples are stored in the cyclic order 0, 1, 2, 3.
    return tuple(VL), tuple(VR)


def A_from_Vc(
    V,
    c,
    rcond=1e-14,
    *,
    exterior_ranks=None,
    schur_form=False,
):
    """Reconstruct the two inner A tensors for each edge from cyclic bases.

    With ``s[j] = offset_k[j]``, this solves the defining identities for A:

    ``A[j-1][s[j]] @ c[j-1][s[j]] = c[j-1][s[j+1]] @ VL[j]``,

    ``c[j][s[j+3]] @ A[j+1][s[j+3]] = VR[j] @ c[j][s[j+2]]``.

    Thus, for ``j=0``, the reconstructed tensors obey
    ``A3[1,1] @ c3[1,1] = c3[0,1] @ VL0`` and
    ``c0[1,0] @ A1[1,0] = VR0 @ c0[0,0]``.

    With ``schur_form=True``, each exterior corner inverse is restricted to
    the neighboring cycle's leading active Schur sector.
    ``exterior_ranks[j]`` is the rank of the neighboring plaquette in direction
    east, north, west, or south for ``j = 0, 1, 2, 3``. The two exterior
    corners used at fixed ``j`` share that active-sector rank even when their
    individual full-matrix ranks differ.
    """
    VL, VR = V
    A_inner = [[None, None] for _ in range(4)]

    for j in range(4):
        edge = (j - 1) % 4
        xy = offset_k[j]
        xy_inner = offset_k[(j + 1) % 4]
        c_inner = c[edge][xy_inner]
        chi = c_inner.shape[-1]
        Vj = jnp.moveaxis(VL[j].reshape((chi, -1, chi)), 0, -2)
        # A_{j-1} = c_{j-1,int} VL_j c_{j-1,ext}^{-1}.
        rhs = c_inner @ Vj
        rhs = jnp.swapaxes(rhs, -1, -2)
        rhs = jnp.moveaxis(rhs, -2, 0).reshape((chi, -1))
        c_exterior = c[edge][xy]
        rank = None if exterior_ranks is None else exterior_ranks[j]
        Aj = _c_pinv_solve(
            c_exterior,
            rhs,
            rank,
            schur_form=schur_form,
            left_side=True,
            transpose_a=True,
            rcond=rcond,
        )
        A_inner[edge][0] = Aj.T.reshape((-1, chi, chi))

    for j in range(4):
        edge = (j + 1) % 4
        xy = offset_k[(j + 3) % 4]
        xy_inner = offset_k[(j + 2) % 4]
        c_inner = c[j][xy_inner]
        chi = c_inner.shape[-1]
        Vj = VR[j].reshape((-1, chi, chi))
        # A_{j+1} = c_{j,ext}^{-1} VR_j c_{j,int}.
        rhs = Vj @ c_inner
        rhs = jnp.moveaxis(rhs, -2, 0).reshape((chi, -1))
        c_exterior = c[j][xy]
        rank = None if exterior_ranks is None else exterior_ranks[j]
        Aj = _c_pinv_solve(
            c_exterior,
            rhs,
            rank,
            schur_form=schur_form,
            left_side=True,
            rcond=rcond,
        )
        A_inner[edge][1] = jnp.moveaxis(
            Aj.reshape((chi, -1, chi)),
            0,
            -2,
        )

    return tuple(tuple(A_edge) for A_edge in A_inner)


def write_A_stencils(A, A_inner):
    """Return A stencils with their eight inner tensors replaced."""
    A_new = []
    for edge in range(4):
        xy0 = offset_k[(edge + 1) % 4]
        xy1 = offset_k[(edge + 2) % 4]
        A_edge = A[edge].at[xy0].set(A_inner[edge][0].reshape(A[edge][xy0].shape))
        A_edge = A_edge.at[xy1].set(A_inner[edge][1].reshape(A[edge][xy1].shape))
        A_new.append(A_edge)
    return tuple(A_new)


def write_c_stencils(c, c_inner):
    """Return corner stencils with their four inner tensors replaced."""
    c_new = []
    for k in range(4):
        xy = offset_k[(k + 2) % 4]
        c_new.append(c[k].at[xy].set(c_inner[k].reshape(c[k][xy].shape)))
    return tuple(c_new)


def _c_cycle_spectrum(c, *, schur_gauge=False, rank=None):
    """Return the cycle eigenvalues and four corner singular-value fields."""
    if schur_gauge:
        eigvals = schur_cycle_eigvals(c, rank)
    else:
        c_cycle = c[0] @ c[1] @ c[2] @ c[3]
        eigvals = jnp.linalg.eigvals(c_cycle)
    svals = tuple(jnp.linalg.svdvals(ck) for ck in c)
    return eigvals, svals


def schur_cycle_eigvals(c, rank):
    """Read the leading physical eigenvalues of a Schur-gauge corner cycle."""
    eigvals = periodic_schur_eigenvalues(jnp.stack(c))
    return jnp.where(jnp.arange(eigvals.size) < rank, eigvals, 0)


def _check_schur_gauge_assertion(ok, defect, tolerance, rank):
    """Raise when a periodic projector returns a non-Schur reduced cycle."""
    if not bool(ok):
        raise AssertionError(
            "projector returned c outside padded Schur gauge at "
            f"rank {int(rank)}: defect={float(defect):.6e}, "
            f"tolerance={float(tolerance):.6e}"
        )


def _assert_periodic_schur_gauge(c, rank):
    """Assert that ``c`` has Schur structure and zero padding beyond ``rank``."""
    factors = jnp.stack(c)
    if jnp.issubdtype(factors.dtype, jnp.complexfloating):
        lower_defect = jnp.max(jnp.abs(jnp.tril(factors, k=-1)))
    else:
        lower_defect = jnp.maximum(
            jnp.max(jnp.abs(jnp.tril(factors[0], k=-2))),
            jnp.max(jnp.abs(jnp.tril(factors[1:], k=-1))),
        )
    active = jnp.arange(factors.shape[-1]) < rank
    active_block = active[:, None] & active[None, :]
    padding_defect = jnp.max(
        jnp.abs(jnp.where(active_block[None, :, :], 0, factors))
    )
    defect = jnp.maximum(lower_defect, padding_defect)
    real_dtype = jnp.real(factors).dtype
    scale = jnp.max(jnp.abs(factors))
    tolerance = (
        100
        * jnp.finfo(real_dtype).eps
        * jnp.maximum(jnp.asarray(1.0, dtype=real_dtype), scale)
    )
    jax.debug.callback(
        _check_schur_gauge_assertion,
        defect <= tolerance,
        defect,
        tolerance,
        rank,
        ordered=True,
    )


def random_vectors(key, shape, dtype):
    """Return uniform random columns with expected squared norm one."""
    d = shape[0]
    real_dtype = jnp.real(jnp.zeros((), dtype=dtype)).dtype
    if jnp.issubdtype(dtype, jnp.complexfloating):
        key_real, key_imag = jax.random.split(key)
        real = jax.random.uniform(
            key_real, shape, real_dtype, minval=-1.0, maxval=1.0
        )
        imag = jax.random.uniform(
            key_imag, shape, real_dtype, minval=-1.0, maxval=1.0
        )
        V = real.astype(dtype) + jnp.asarray(1j, dtype=dtype) * imag.astype(dtype)
        return V * jnp.sqrt(jnp.asarray(3.0 / (2 * d), dtype=real_dtype))
    V = jax.random.uniform(key, shape, dtype, minval=-1.0, maxval=1.0)
    return V * jnp.sqrt(jnp.asarray(3.0 / d, dtype=real_dtype))


def _stochastic_expand_range(V, dV, rank_tol):
    """QR-complete ``range(V)`` using random replacements for null columns."""
    X, R = jnp.linalg.qr(V, mode="reduced")
    rdiag = jnp.abs(jnp.diag(R))
    keep = rdiag > rank_tol * jnp.max(rdiag)
    X = jnp.where(keep[None, :], X, dV)
    return jnp.linalg.qr(X, mode="reduced")[0]


def _rank_masked_qr(X, rank):
    """Return ordered QR with columns and rows beyond known ``rank`` zeroed."""
    Q, R = jnp.linalg.qr(X, mode="reduced")
    keep = jnp.arange(R.shape[0]) < rank
    Q = Q * keep[None, :]
    R = R * keep[:, None]
    return Q, R


def _krylov_cycle_projectors(
    C,
    VR0,
    VL0,
    *,
    chi_max,
    schur_form=False,
    krylov_cfg=None,
    info_level=0,
):
    """Extract bond-zero cycle modes, then propagate and whiten every bond.

    The CTM convention is ``C[k] VR[k+1] = VR[k] c[k]`` and
    ``VL[k] C[k] = c[k] VL[k+1]``. Thus the right cycle based at bond zero is
    ``C0 C1 C2 C3``, while its transpose action applies ``C0.T`` through
    ``C3.T`` in ascending order. With ``schur_form=True``, the known active
    reduced cycle is put in R-oriented periodic Schur form without re-ranking.
    """

    def cycle_matvec(V):
        """Apply ``C0 C1 C2 C3`` to columns based at bond zero."""

        for k in range(3, -1, -1):
            V = C[k] @ V
        return V

    def cycle_vecmat(V):
        """Apply the transposed bond-zero cycle to left-basis columns."""

        for k in range(4):
            V = C[k].T @ V
        return V

    VR0, VL0, _, _, info = krylov_eig_one_sided(
        cycle_matvec,
        cycle_vecmat,
        VR0,
        VL0,
        chi_max=chi_max,
        info_level=info_level,
        krylov_cfg=krylov_cfg,
    )
    basis_rank = info["basis_rank"]
    basis_keep = jnp.arange(VR0.shape[1]) < basis_rank
    VR0 = VR0 * basis_keep[None, :]
    VL0 = VL0 * basis_keep[:, None]

    VR = [None] * 4
    RR = [None] * 4
    VR[0] = VR0
    for k in range(3, 0, -1):
        # Zero QR's padded tail before the next C factor can activate arbitrary
        # completion columns.
        VR[k], RR[k] = _rank_masked_qr(
            C[k] @ VR[(k + 1) % 4],
            basis_rank,
        )

    VL = [None] * 4
    RL = [None] * 4
    VL[0] = VL0
    for k in range(3):
        # The transposed path applies the matching rank-aware row propagation.
        VL_column, RL_column_T = _rank_masked_qr(
            (VL[k] @ C[k]).T,
            basis_rank,
        )
        VL[k + 1] = VL_column.T
        RL[k] = RL_column_T.T

    # Bond zero was already biorthogonalized by the cycle eigensolve. Preserve
    # that original overlap spectrum, then whiten the three propagated bonds.
    biorth_svals = [info["basis_overlap_svals"]]
    overlap = [VL[0] @ VR[0]]
    identity = jnp.eye(VR[0].shape[1], dtype=VR[0].dtype)
    G_R = [identity]
    G_L = [identity]
    for k in range(1, 4):
        VR[k], VL[k], G_R_k, G_L_k, overlap_k, s, _ = biorthogonalize_bases(
            VR[k],
            VL[k],
        )
        G_R.append(G_R_k)
        G_L.append(G_L_k)
        overlap.append(overlap_k)
        biorth_svals.append(s)

    VR = tuple(VR)
    VL = tuple(VL)
    # Before whitening, the saved QR factors give the projected actions
    #
    #   VL[k] C[k] VR[k+1] = (VL[k] VR[k]) RR[k],  k = 1,2,3,
    #   VL[0] C[0] VR[1]   = RL[0] (VL[1] VR[1]).
    #
    # Carry these small matrices through the whitening gauges, avoiding four
    # repeated full-space CTM actions.
    c_unwhitened = (RL[0] @ overlap[1],) + tuple(
        overlap[k] @ RR[k] for k in range(1, 4)
    )
    c = tuple(
        G_L[k] @ c_unwhitened[k] @ G_R[(k + 1) % 4]
        for k in range(4)
    )
    if schur_form:
        c_stacked = jnp.stack(c)
        if jnp.issubdtype(c_stacked.dtype, jnp.complexfloating):
            c_stacked, Z, _, _, _, schur_size = periodic_schur_Z(
                c_stacked,
                active_cols=basis_rank,
                reduction="NRed",
            )
        else:
            c_stacked, Z, _, _, schur_size = periodic_schur_D(
                c_stacked,
                active_cols=basis_rank,
                reduction="NRed",
            )
        VR = tuple(VR[k] @ Z[k] for k in range(4))
        VL = tuple(Z[k].T.conj() @ VL[k] for k in range(4))
        c = tuple(c_stacked)
        info["schur_size"] = schur_size
    info["bond_biorth_svals"] = jnp.stack(biorth_svals)
    return VR, VL, c, info


def _periodic_krylov_cycle_projectors(
    C,
    VR,
    VL,
    *,
    chi_max,
    seed_ritz_values=None,
    seed_schur_size=None,
    krylov_cfg=None,
    info_level=0,
):
    """Run periodic Krylov-Schur directly in the CTM/SLICOT R ordering."""
    matvecs = tuple(lambda V, Ck=Ck: Ck @ V for Ck in C)
    vecmats = tuple(lambda V, Ck=Ck: Ck.T @ V for Ck in C)

    VR, VL_T, c, info = periodic_krylov_schur_projectors(
        matvecs,
        vecmats,
        tuple(VR),
        tuple(V.T for V in VL),
        chi_max=chi_max,
        info_level=info_level,
        cfg=krylov_cfg,
        seed_ritz_values=seed_ritz_values,
        seed_schur_size=seed_schur_size,
    )
    VR = tuple(VR)
    VL = tuple(V.T for V in VL_T)
    c = tuple(c)
    info = dict(info)
    info["basis_rank"] = info["biorthogonal_rank"]
    info["bond_biorth_svals"] = info["overlap_singular_values"]
    if info_level >= 1:
        info["initial_ortho_res_R"] = info["right"]["initial_ortho_res"]
        info["initial_ortho_res_L"] = info["left"]["initial_ortho_res"]
    return VR, VL, c, info


def _projector_subspace_change(
    VL_in,
    VR_in,
    VL_out,
    VR_out,
    rank_tol,
):
    r"""Measure the numerical left and right projector subspace changes.

    For each ``k`` modulo four, define

    ``X_L^in[k] = VL_in[k].T``,
    ``X_L^out[k] = VL_out[k].T``,

    ``X_R^in[k] = VR_in[k]``, and
    ``X_R^out[k] = VR_out[k]``.

    For ``S`` equal to ``L`` or ``R``, pivoted QR gives numerical-range bases
    ``Q_S^in[k]`` and ``Q_S^out[k]`` after retaining pivots
    ``|R_ii| > rank_tol ||X||_F``. Define
    ``r_S[k] = max(r_S^in[k], r_S^out[k])``. The corresponding change is

    ``dV_S[k] = 1 - clip(sigma_{r_S[k]}((Q_S^out[k])^H Q_S^in[k]), 0, 1)``

    when ``r_S[k] > 0``, and zero otherwise, with singular values in descending
    order. A rank mismatch supplies zero singular values beyond the smaller
    rank and therefore gives ``dV_S[k] = 1``. Thus, up to the roundoff clip,
    ``dV_S[k] = 1 - cos(theta_S,max)`` under the extended principal-angle
    convention. Here ``T`` is transpose and ``H`` is conjugate transpose. The
    three returned arrays are ``dV``, ``rank_in``, and ``rank_out``, each
    ordered ``(L, R)`` along its first axis.
    """
    X_in = (
        tuple(V.T for V in VL_in),
        tuple(V for V in VR_in),
    )
    X_out = (
        tuple(V.T for V in VL_out),
        tuple(V for V in VR_out),
    )

    dV = []
    rank_in = []
    rank_out = []
    for X_in_side, X_out_side in zip(X_in, X_out):
        dV_side = []
        rank_in_side = []
        rank_out_side = []
        for X_in_k, X_out_k in zip(X_in_side, X_out_side):
            Q_in, _, keep_in, rank_in_k = qrp_basis_and_rank(
                X_in_k,
                rank_tol,
            )
            Q_out, _, keep_out, rank_out_k = qrp_basis_and_rank(
                X_out_k,
                rank_tol,
            )
            Q_in = Q_in * keep_in[None, :]
            Q_out = Q_out * keep_out[None, :]
            # Rank mismatch adds zero singular values, hence a pi/2 angle.
            s = jnp.linalg.svdvals(Q_out.T.conj() @ Q_in)
            rank_k = jnp.maximum(rank_in_k, rank_out_k)
            smin = s[jnp.maximum(rank_k - 1, 0)]
            dV_side.append(
                jnp.where(
                    rank_k > 0,
                    1.0 - jnp.clip(smin, 0.0, 1.0),
                    0.0,
                )
            )
            rank_in_side.append(rank_in_k)
            rank_out_side.append(rank_out_k)
        dV.append(jnp.stack(dV_side))
        rank_in.append(jnp.stack(rank_in_side))
        rank_out.append(jnp.stack(rank_out_side))

    return (
        jnp.stack(dV),
        jnp.stack(rank_in),
        jnp.stack(rank_out),
    )


def local_update(
    T,
    A,
    c,
    *,
    rank=None,
    exterior_ranks=None,
    krylov_cfg=None,
    pinv_rtol=1e-14,
    schur_gauge=False,
    krylov_diagnostics=0,
    return_info=False,
):
    """Update inner A and c tensors with the configured cycle projector method.

    With ``schur_gauge=True``, the incoming corner cycles are assumed to have
    leading Schur sectors of size ``rank``. Corner inverse actions then use the
    padded retained-sector solve, and ``exterior_ranks`` supplies the east,
    north, west, and south ranks needed when reconstructing the owned edges.
    ``krylov_diagnostics >= 2`` additionally records the incoming and outgoing
    corner spectra; lower levels return zero placeholders for those fields.
    """
    cfg = {} if krylov_cfg is None else krylov_cfg
    method = cfg.get("method", DEFAULT_PROJECTOR_METHOD)
    if schur_gauge and rank is None:
        raise ValueError("schur_gauge=True requires the incoming plaquette rank")
    if schur_gauge and exterior_ranks is None:
        raise ValueError("schur_gauge=True requires the four exterior ranks")
    if schur_gauge and method not in (
        "eig one sided",
        "periodic krylov schur one sided",
    ):
        raise NotImplementedError(
            f"schur_gauge=True is not implemented for projector method {method!r}"
        )

    VL_guess, VR_guess = V_from_Ac(
        A,
        c,
        rank=rank,
        schur_form=schur_gauge,
        rcond=pinv_rtol,
    )
    VL_in, VR_in = VL_guess, VR_guess

    # Rank expansion if the initial projectors are rank deficient
    # We detect rank deficiency of the V via QR, and then stochastically complete columns
    stochastic_expand = cfg.get("V_guess_stochastic", False)
    if stochastic_expand:
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        dVR = tuple(
            random_vectors(key, VR.shape, VR.dtype)
            for VR, key in zip(VR_guess, keys)
        )
        VR_guess = tuple(
            _stochastic_expand_range(VR, dV, pinv_rtol)
            for VR, dV in zip(VR_guess, dVR)
        )
        VL_guess = tuple(
            _stochastic_expand_range(VL.T, jnp.conj(dV), pinv_rtol).T
            for VL, dV in zip(VL_guess, dVR)
        )

    C = []
    for k in range(4):
        xy = offset_k[k]
        Tk = T[xy]
        Ak = tuple(a[xy] for a in A)
        ck = tuple(corner[xy] for corner in c)
        C.append(construct_CTM_k(Tk, Ak, ck, k))
    C = tuple(C)

    c_inner_in = tuple(c[k][offset_k[(k + 2) % 4]] for k in range(4))
    rank_in = rank
    seed_ritz_values = None
    seed_schur_size = None
    if (
        schur_gauge
        and method == "periodic krylov schur one sided"
        and cfg.get("lock_policy", "ritz_difference") == "ritz_difference"
    ):
        seed_ritz_values = schur_cycle_eigvals(c_inner_in, rank)
        seed_schur_size = rank

    if method == "power":
        matvec = tuple(lambda V, Ck=Ck: Ck @ V for Ck in C)
        vecmat = tuple(lambda V, Ck=Ck: Ck.T @ V for Ck in C)
        VR, VL, c_inner, projector_info = periodic_power_projectors(
            matvec,
            vecmat,
            VR_guess,
            VL_guess,
            num_iter=cfg.get("num_power_iter", 1),
            return_info=True,
        )
    elif method == "eig one sided":
        VR, VL, c_inner, projector_info = _krylov_cycle_projectors(
            C,
            VR_guess[0],
            VL_guess[0],
            chi_max=VR_guess[0].shape[1],
            schur_form=schur_gauge,
            krylov_cfg=cfg,
            info_level=krylov_diagnostics,
        )
    elif method == "periodic krylov schur one sided":
        VR, VL, c_inner, projector_info = _periodic_krylov_cycle_projectors(
            C,
            VR_guess,
            VL_guess,
            chi_max=VR_guess[0].shape[1],
            seed_ritz_values=seed_ritz_values,
            seed_schur_size=seed_schur_size,
            krylov_cfg=cfg,
            info_level=krylov_diagnostics,
        )
    else:
        raise ValueError(f"unknown CTM projector method {method!r}")
    if schur_gauge:
        _assert_periodic_schur_gauge(c_inner, projector_info["basis_rank"])
    rank = (
        projector_info["basis_rank"]
        if method in ("eig one sided", "periodic krylov schur one sided")
        else jnp.min(projector_info["bond_biorth_rank"])
    )
    dV, dV_rank_in, dV_rank_out = _projector_subspace_change(
        VL_in,
        VR_in,
        VL,
        VR,
        pinv_rtol,
    )
    dVL, dVR = dV
    dVL_rank_in, dVR_rank_in = dV_rank_in
    dVL_rank_out, dVR_rank_out = dV_rank_out
    chi = c_inner[0].shape[-2]
    if krylov_diagnostics >= 2:
        c_cycle_eigvals_in, c_svals_in = _c_cycle_spectrum(
            c_inner_in,
            schur_gauge=schur_gauge,
            rank=rank_in,
        )
        c_cycle_eigvals_out, c_svals_out = _c_cycle_spectrum(
            c_inner,
            schur_gauge=schur_gauge,
            rank=rank,
        )
        c_lstsq_rank_in = jnp.stack(
            [jnp.sum(s > pinv_rtol * jnp.max(s)) for s in c_svals_in]
        )
    else:
        eig_dtype = jnp.result_type(c_inner[0].dtype, jnp.complex64)
        c_cycle_eigvals_in = jnp.zeros((chi,), dtype=eig_dtype)
        c_cycle_eigvals_out = jnp.zeros_like(c_cycle_eigvals_in)
        c_svals_in = tuple(
            jnp.zeros((min(ck.shape[-2:]),), dtype=jnp.real(ck).dtype)
            for ck in c_inner_in
        )
        c_svals_out = tuple(jnp.zeros_like(s) for s in c_svals_in)
        c_lstsq_rank_in = jnp.zeros((4,), dtype=rank.dtype)
    bond_biorth_svals = projector_info.get(
        "bond_biorth_svals",
        jnp.full((4, chi), jnp.nan, dtype=jnp.real(c_inner[0]).dtype),
    )
    cycle_ritz_rank_R = projector_info.get("ritz_rank_R", jnp.asarray(-1))
    cycle_ritz_rank_L = projector_info.get("ritz_rank_L", jnp.asarray(-1))
    no_arnoldi = jnp.asarray(jnp.nan, dtype=jnp.real(c_inner[0]).dtype)
    ortho_res_R = projector_info.get("initial_ortho_res_R", no_arnoldi)
    ortho_res_L = projector_info.get("initial_ortho_res_L", no_arnoldi)
    c = write_c_stencils(c, c_inner)
    A_inner = A_from_Vc(
        (VL, VR),
        c,
        exterior_ranks=exterior_ranks,
        schur_form=schur_gauge,
        rcond=pinv_rtol,
    )
    A = write_A_stencils(A, A_inner)
    info = CTMRGInfo(
        rank=rank,
        dVL=dVL,
        dVR=dVR,
        dVL_rank_in=dVL_rank_in,
        dVL_rank_out=dVL_rank_out,
        dVR_rank_in=dVR_rank_in,
        dVR_rank_out=dVR_rank_out,
        c_cycle_eigvals_in=c_cycle_eigvals_in,
        c_cycle_eigvals_out=c_cycle_eigvals_out,
        c_svals_in=c_svals_in,
        c_svals_out=c_svals_out,
        c_lstsq_rank_in=c_lstsq_rank_in,
        bond_biorth_svals=bond_biorth_svals,
        cycle_ritz_rank_R=cycle_ritz_rank_R,
        cycle_ritz_rank_L=cycle_ritz_rank_L,
        ortho_res_R=ortho_res_R,
        ortho_res_L=ortho_res_L,
    )
    if return_info:
        return A, c, info
    return A, c


def _gather_2x2(X, x, y):
    """Gather a two-by-two spatial stencil, wrapping either lattice axis."""
    xs = jnp.mod(x + jnp.arange(2), X.shape[0])
    ys = jnp.mod(y + jnp.arange(2), X.shape[1])
    return X[xs[:, None], ys[None, :]]


def _gather_local_stencils(T, A, c, x, y):
    """Gather all T, A, and c stencils for the plaquette at ``(x, y)``."""
    T_patch = (
        tuple(_gather_2x2(t, x, y) for t in T)
        if isinstance(T, tuple)
        else _gather_2x2(T, x, y)
    )
    A_patch = tuple(_gather_2x2(a, x, y) for a in A)
    c_patch = tuple(_gather_2x2(corner, x, y) for corner in c)
    return T_patch, A_patch, c_patch


def _gather_exterior_ranks(rank, x, y, finite):
    """Gather east, north, west, and south neighboring plaquette ranks.

    Missing finite-boundary neighbors correspond to the fixed one-hot corner
    tensors and therefore have structural rank one. Periodic axes wrap.
    """
    exterior_ranks = []
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        xn = x + dx
        yn = y + dy
        valid_x = (
            (xn >= 0) & (xn < rank.shape[0])
            if finite[0]
            else jnp.asarray(True)
        )
        valid_y = (
            (yn >= 0) & (yn < rank.shape[1])
            if finite[1]
            else jnp.asarray(True)
        )
        neighbor_rank = rank[
            jnp.mod(xn, rank.shape[0]),
            jnp.mod(yn, rank.shape[1]),
        ]
        exterior_ranks.append(jnp.where(valid_x & valid_y, neighbor_rank, 1))
    return tuple(exterior_ranks)


def _write_local_Y(A, c, A_patch, c_patch, x, y):
    """Scatter only the eight A and four c tensors owned by one plaquette."""
    A_new = []
    for k in range(4):
        Ak = A[k]
        for xy in (offset_k[(k + 1) % 4], offset_k[(k + 2) % 4]):
            gx = jnp.mod(x + xy[0], Ak.shape[0])
            gy = jnp.mod(y + xy[1], Ak.shape[1])
            Ak = Ak.at[gx, gy].set(A_patch[k][xy])
        A_new.append(Ak)

    c_new = []
    for k in range(4):
        xy = offset_k[(k + 2) % 4]
        gx = jnp.mod(x + xy[0], c[k].shape[0])
        gy = jnp.mod(y + xy[1], c[k].shape[1])
        c_new.append(c[k].at[gx, gy].set(c_patch[k][xy]))
    return tuple(A_new), tuple(c_new)


def nested_palindrome(Nx, Ny, finite=(1, 1)):
    """Return the nested forward/backward ``(x, y)`` update schedule."""
    num_x = Nx - int(finite[0])
    num_y = Ny - int(finite[1])

    x_order = jnp.concatenate(
        (
            jnp.arange(num_x, dtype=jnp.int32),
            jnp.arange(num_x - 2, 0, -1, dtype=jnp.int32),
        )
    )
    y_order = jnp.concatenate(
        (
            jnp.arange(num_y, dtype=jnp.int32),
            jnp.arange(num_y - 2, -1, -1, dtype=jnp.int32),
        )
    )
    return jnp.stack(
        (
            jnp.repeat(x_order, y_order.size),
            jnp.tile(y_order, x_order.size),
        ),
        axis=1,
    )


def lex(Nx, Ny, finite=(1, 1)):
    """Return the lexicographic schedule followed by its reverse path."""
    #TODO handle the infinite cases
    if finite != (1, 1):
        raise NotImplementedError
    num_x = Nx - finite[0]
    num_y = Ny - finite[1]
    x = jnp.arange(num_x, dtype=jnp.int32)
    y = jnp.arange(num_y, dtype=jnp.int32)
    forward = jnp.stack(
        (jnp.repeat(x, num_y), jnp.tile(y, num_x)),
        axis=1,
    )
    return jnp.concatenate((forward[:-1], forward[::-1]))


def snake(Nx, Ny, finite=(1, 1)):
    """Return alternating up/down columns followed by the exact reverse."""
    num_x = Nx - int(finite[0])
    num_y = Ny - int(finite[1])
    x = jnp.arange(num_x, dtype=jnp.int32)
    y = jnp.arange(num_y, dtype=jnp.int32)
    y_snake = jnp.where(
        x[:, None] % 2 == 0,
        y[None, :],
        y[::-1][None, :],
    )
    forward = jnp.stack(
        (jnp.repeat(x, num_y), y_snake.reshape(-1)),
        axis=1,
    )
    return jnp.concatenate((forward[:-1], forward[::-1]))


def lex_twice(Nx, Ny, finite=(1, 1)):
    """Return the bottom-left to top-right lexicographic path twice."""

    num_x = Nx - int(finite[0])
    num_y = Ny - int(finite[1])
    x = jnp.arange(num_x, dtype=jnp.int32)
    y = jnp.arange(num_y, dtype=jnp.int32)
    forward = jnp.stack(
        (jnp.repeat(x, num_y), jnp.tile(y, num_x)),
        axis=1,
    )
    return jnp.concatenate((forward, forward))


def column_up_down(Nx, Ny, finite=(1, 1)):
    """Sweep up then down each column while moving only left to right."""

    num_x = Nx - int(finite[0])
    num_y = Ny - int(finite[1])
    x = jnp.arange(num_x, dtype=jnp.int32)
    y = jnp.arange(num_y, dtype=jnp.int32)
    y_up_down = jnp.concatenate((y, y[::-1]))
    return jnp.stack(
        (
            jnp.repeat(x, y_up_down.size),
            jnp.tile(y_up_down, num_x),
        ),
        axis=1,
    )


def ctmrg_sweep(
    state,
    *,
    schedule=None,
    krylov_cfg=None,
    pinv_rtol=1e-14,
    schur_gauge=False,
    krylov_diagnostics=0,
    return_info=False,
):
    """Run a scheduled plaquette sweep and report final local diagnostics.

    ``schur_gauge=True`` assumes every corner cycle has the leading Schur rank
    stored in ``state.rank`` and uses retained-sector corner solves.
    """
    method = ({} if krylov_cfg is None else krylov_cfg).get(
        "method",
        DEFAULT_PROJECTOR_METHOD,
    )
    if schur_gauge and method not in (
        "eig one sided",
        "periodic krylov schur one sided",
    ):
        raise NotImplementedError(
            f"schur_gauge=True is not implemented for projector method {method!r}"
        )
    T0 = state.T[0] if isinstance(state.T, tuple) else state.T
    if schedule is None:
        schedule = snake(*T0.shape[:2], finite=state.finite)
    else:
        schedule = jnp.asarray(schedule, dtype=jnp.int32)
    A, c, info = _ctmrg_sweep_jit(
        state.T,
        state.A,
        state.c,
        state.rank,
        schedule,
        finite=state.finite,
        krylov_cfg=_static_dict(krylov_cfg),
        pinv_rtol=pinv_rtol,
        schur_gauge=schur_gauge,
        krylov_diagnostics=krylov_diagnostics,
    )
    state = state._replace(A=A, c=c, rank=info.rank)
    if return_info:
        return state, info
    return state


def update_info(info, local_info, x, y):
    """Write one plaquette's diagnostics into the sweep-wide info arrays."""
    return info._replace(
        rank=info.rank.at[x, y].set(local_info.rank),
        dVL=info.dVL.at[:, x, y].set(local_info.dVL),
        dVR=info.dVR.at[:, x, y].set(local_info.dVR),
        dVL_rank_in=info.dVL_rank_in.at[:, x, y].set(
            local_info.dVL_rank_in
        ),
        dVL_rank_out=info.dVL_rank_out.at[:, x, y].set(
            local_info.dVL_rank_out
        ),
        dVR_rank_in=info.dVR_rank_in.at[:, x, y].set(
            local_info.dVR_rank_in
        ),
        dVR_rank_out=info.dVR_rank_out.at[:, x, y].set(
            local_info.dVR_rank_out
        ),
        c_cycle_eigvals_in=info.c_cycle_eigvals_in.at[x, y].set(
            local_info.c_cycle_eigvals_in
        ),
        c_cycle_eigvals_out=info.c_cycle_eigvals_out.at[x, y].set(
            local_info.c_cycle_eigvals_out
        ),
        c_svals_in=tuple(
            field.at[x, y].set(svals)
            for field, svals in zip(info.c_svals_in, local_info.c_svals_in)
        ),
        c_svals_out=tuple(
            field.at[x, y].set(svals)
            for field, svals in zip(info.c_svals_out, local_info.c_svals_out)
        ),
        c_lstsq_rank_in=info.c_lstsq_rank_in.at[:, x, y].set(
            local_info.c_lstsq_rank_in
        ),
        bond_biorth_svals=info.bond_biorth_svals.at[:, x, y].set(
            local_info.bond_biorth_svals
        ),
        cycle_ritz_rank_R=info.cycle_ritz_rank_R.at[x, y].set(
            local_info.cycle_ritz_rank_R
        ),
        cycle_ritz_rank_L=info.cycle_ritz_rank_L.at[x, y].set(
            local_info.cycle_ritz_rank_L
        ),
        ortho_res_R=info.ortho_res_R.at[x, y].set(local_info.ortho_res_R),
        ortho_res_L=info.ortho_res_L.at[x, y].set(local_info.ortho_res_L),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "finite",
        "krylov_cfg",
        "pinv_rtol",
        "schur_gauge",
        "krylov_diagnostics",
    ),
)
def _ctmrg_sweep_jit(
    T,
    A,
    c,
    rank,
    schedule,
    *,
    finite=(True, True),
    krylov_cfg=None,
    pinv_rtol=1e-14,
    schur_gauge=False,
    krylov_diagnostics=0,
):
    """JIT core for a Gauss-Seidel sweep over overlapping Z22 stencils."""
    num_x = T[0].shape[0] if isinstance(T, tuple) else T.shape[0]
    num_y = T[0].shape[1] if isinstance(T, tuple) else T.shape[1]
    num_x -= int(finite[0])
    num_y -= int(finite[1])

    def update(carry, xy):
        """Apply one local update and scatter only its owned inner ring."""
        A, c, sweep_info = carry
        x, y = xy
        T_patch, A_patch, c_patch = _gather_local_stencils(T, A, c, x, y)
        exterior_ranks = _gather_exterior_ranks(
            sweep_info.rank,
            x,
            y,
            finite,
        )
        A_patch, c_patch, local_info = local_update(
            T_patch,
            A_patch,
            c_patch,
            rank=sweep_info.rank[x, y],
            exterior_ranks=exterior_ranks,
            krylov_cfg=krylov_cfg,
            pinv_rtol=pinv_rtol,
            schur_gauge=schur_gauge,
            krylov_diagnostics=krylov_diagnostics,
            return_info=True,
        )
        A, c = _write_local_Y(A, c, A_patch, c_patch, x, y)
        sweep_info = update_info(sweep_info, local_info, x, y)
        return (A, c, sweep_info), None

    T0 = T[0] if isinstance(T, tuple) else T
    dVL = jnp.zeros((4, num_x, num_y), dtype=jnp.real(T0).dtype)
    dVR = jnp.zeros_like(dVL)
    rank_dtype = jnp.asarray(0).dtype
    rank = jnp.asarray(rank, dtype=rank_dtype)
    dVL_rank_in = jnp.zeros((4, num_x, num_y), dtype=rank_dtype)
    dVL_rank_out = jnp.zeros((4, num_x, num_y), dtype=rank_dtype)
    dVR_rank_in = jnp.zeros_like(dVL_rank_in)
    dVR_rank_out = jnp.zeros_like(dVL_rank_out)
    eig_dtype = jnp.result_type(c[0].dtype, jnp.complex64)
    c_cycle_eigvals_in = jnp.zeros(
        (num_x, num_y, c[0].shape[-2]), dtype=eig_dtype
    )
    c_cycle_eigvals_out = jnp.zeros_like(c_cycle_eigvals_in)
    c_svals_in = tuple(
        jnp.zeros(
            (num_x, num_y, min(ck.shape[-2:])),
            dtype=jnp.real(ck).dtype,
        )
        for ck in c
    )
    c_svals_out = tuple(jnp.zeros_like(svals) for svals in c_svals_in)
    c_lstsq_rank_in = jnp.zeros((4, num_x, num_y), dtype=rank_dtype)
    chi = c[0].shape[-2]
    bond_biorth_svals = jnp.zeros(
        (4, num_x, num_y, chi), dtype=jnp.real(c[0]).dtype
    )
    cycle_ritz_rank_R = jnp.zeros((num_x, num_y), dtype=rank_dtype)
    cycle_ritz_rank_L = jnp.zeros((num_x, num_y), dtype=rank_dtype)
    ortho_res_R = jnp.full(
        (num_x, num_y), jnp.nan, dtype=jnp.real(c[0]).dtype
    )
    ortho_res_L = jnp.full_like(ortho_res_R, jnp.nan)
    info = CTMRGInfo(
        rank=rank,
        dVL=dVL,
        dVR=dVR,
        dVL_rank_in=dVL_rank_in,
        dVL_rank_out=dVL_rank_out,
        dVR_rank_in=dVR_rank_in,
        dVR_rank_out=dVR_rank_out,
        c_cycle_eigvals_in=c_cycle_eigvals_in,
        c_cycle_eigvals_out=c_cycle_eigvals_out,
        c_svals_in=c_svals_in,
        c_svals_out=c_svals_out,
        c_lstsq_rank_in=c_lstsq_rank_in,
        bond_biorth_svals=bond_biorth_svals,
        cycle_ritz_rank_R=cycle_ritz_rank_R,
        cycle_ritz_rank_L=cycle_ritz_rank_L,
        ortho_res_R=ortho_res_R,
        ortho_res_L=ortho_res_L,
    )
    (A, c, info), _ = jax.lax.scan(
        update,
        (A, c, info),
        schedule,
    )
    return A, c, info


def _ctmrg_iteration_summary(info):
    """Return the minimum nonzero bond overlap and maximum projector motion."""
    s = info.bond_biorth_svals
    nonzero = s != 0
    sigma_min = jnp.where(
        jnp.any(nonzero),
        jnp.min(jnp.where(nonzero, s, jnp.inf)),
        0.0,
    )
    return sigma_min, jnp.max(jnp.stack((info.dVL, info.dVR)))


def _ctmrg_spatial_max(values):
    """Return the global maximum and its trailing Cartesian ``(x, y)`` site."""
    index = jnp.unravel_index(jnp.argmax(values), values.shape)
    return values[index], index[-2:]


def _print_ctmrg_header(*, report_ortho=False):
    """Print the CTMRG iteration-table header."""
    header = f"  {'iter':>4}  {'min sigma(M)':>14}"
    if report_ortho:
        header += f"  {'max ortho':>22}"
    print(header + f"  {'max dV':>22}")


def _print_ctmrg_iteration(iteration, info, *, report_ortho=False):
    """Print one row of aggregate CTMRG diagnostics."""
    sigma_min, dV_max = _ctmrg_iteration_summary(info)
    row = f"  {iteration:4d}  {float(sigma_min):14.6e}"
    if report_ortho:
        ortho_max, (ortho_x, ortho_y) = _ctmrg_spatial_max(
            jnp.stack((info.ortho_res_L, info.ortho_res_R))
        )
        ortho_field = (
            f"{float(ortho_max):14.6e} @({int(ortho_x)},{int(ortho_y)})"
        )
        row += f"  {ortho_field:>22}"
    _, (dV_x, dV_y) = _ctmrg_spatial_max(
        jnp.stack((info.dVL, info.dVR))
    )
    dV_field = f"{float(dV_max):14.6e} @({int(dV_x)},{int(dV_y)})"
    print(row + f"  {dV_field:>22}")


def ctmrg(state, ctmrg_cfg_opt=None):
    """Run CTMRG until ``dV_tol`` or the requested sweep limit is reached.

    ``dV_tol`` defaults to ``1e-12`` and compares against the maximum left or
    right projector-range motion over the full completed sweep. Set it to
    ``None`` to run exactly ``num_ctmrg_iter`` sweeps. The periodic
    Krylov-Schur projector defaults to six single-edge Krylov slots,
    ``cred_rank_tol=4.0``, and skips a non-fitting real Schur pair to fill the
    requested rank with later scalar modes. Cycle eig retains its depth-three
    default.

    Periodic Krylov-Schur extraction automatically uses Schur-gauge corner
    solves. Set ``ctmrg_cfg_opt["schur_gauge"]=True`` to opt ordinary cycle eig
    into the same coordinate convention. The input state is converted once
    with the configured ``CTM_eig_cutoff`` so its retained corner sector is
    explicitly zero-padded.
    """
    cfg = {} if ctmrg_cfg_opt is None else ctmrg_cfg_opt
    num_ctmrg_iter = cfg.get("num_ctmrg_iter", 10)
    dV_tol = cfg.get("dV_tol", 1e-12)
    krylov_cfg = dict(cfg.get("krylov_cfg", {}) or {})
    stochastic_num_iter = krylov_cfg.pop("V_guess_stochastic_num_iter", 2)
    method = krylov_cfg.get("method", DEFAULT_PROJECTOR_METHOD)
    krylov_cfg.setdefault(
        "num_krylov_iter",
        PERIODIC_NUM_KRYLOV_ITER
        if method == "periodic krylov schur one sided"
        else NUM_KRYLOV_ITER,
    )
    krylov_cfg.setdefault("cred_rank_tol", 4.0)
    pinv_rtol = cfg.get("pinv_rtol", 1e-14)
    krylov_diagnostics = cfg.get("krylov_diagnostics", 0)
    if method == "periodic krylov schur one sided":
        krylov_cfg.setdefault("split_pair_policy", "replace_split_pair")
    schur_gauge = (
        method == "periodic krylov schur one sided"
        or cfg.get("schur_gauge", False)
    )
    if schur_gauge:
        default_schur_cutoff = (
            1e-15
            if method == "periodic krylov schur one sided"
            else 2e-14
        )
        state = state.schur_gauge(
            CTM_eig_cutoff=krylov_cfg.get(
                "CTM_eig_cutoff",
                default_schur_cutoff,
            )
        )
        state = jax.block_until_ready(state)
    sweep = cfg.get("sweep", "snake")
    schedule_fn = {
        "nested palindrome": nested_palindrome,
        "lex": lex,
        "snake": snake,
        "lex twice": lex_twice,
        "column up down": column_up_down,
    }[sweep]
    T0 = state.T[0] if isinstance(state.T, tuple) else state.T
    schedule = schedule_fn(*T0.shape[:2], finite=state.finite)
    sweep_infos = []
    sweep_times = []
    report_ortho = method == "eig one sided" or (
        method == "periodic krylov schur one sided"
        and krylov_diagnostics >= 1
    )
    _print_ctmrg_header(report_ortho=report_ortho)

    for iteration in range(num_ctmrg_iter):
        iteration_krylov_cfg = {
            **krylov_cfg,
            "V_guess_stochastic": iteration < stochastic_num_iter,
        }
        t0 = perf_counter()
        state, sweep_info = ctmrg_sweep(
            state,
            schedule=schedule,
            krylov_cfg=iteration_krylov_cfg,
            pinv_rtol=pinv_rtol,
            schur_gauge=schur_gauge,
            krylov_diagnostics=krylov_diagnostics,
            return_info=True,
        )
        jax.block_until_ready((state, sweep_info))
        sweep_times.append(perf_counter() - t0)
        sweep_infos.append(sweep_info)
        _print_ctmrg_iteration(
            iteration + 1,
            sweep_info,
            report_ortho=report_ortho,
        )
        _, dV_max = _ctmrg_iteration_summary(sweep_info)
        if dV_tol is not None and float(dV_max) < dV_tol:
            print(
                f"CTMRG converged at iter {iteration + 1}: "
                f"max dV = {float(dV_max):.6e} < {dV_tol:g}"
            )
            break

    if len(sweep_times) > 1:
        warm_time = sum(sweep_times[1:]) / (len(sweep_times) - 1)
        print(f"CTMRG time / sweep, excluding first: {warm_time:.6g} s")

    info = jax.tree.map(lambda *fields: jnp.stack(fields), *sweep_infos)
    state = state._replace(rank=info.rank[-1])
    return state, info
