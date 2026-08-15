"""Block periodic Krylov-Schur notes.

The periodic algorithms live together in this file and share only the generic
``split_qrp`` helper with the rest of the repository.  The main public entry
points are:

* ``periodic_arnoldi_basis``: build the fixed-shape block periodic Arnoldi
  relation.
* ``periodic_power``: warm up cyclic right or left bases with QR or QL sweeps.
* ``periodic_power_projectors``: independently warm up right and left cyclic
  projector bases.
* ``periodic_sylvester_schur``: transform dense right Sylvester factors to
  periodic Schur form.
* ``periodic_krylov_sylvester``: solve a Sylvester correction equation with a
  right-oriented periodic Arnoldi compression.
* ``periodic_krylov_schur_onesided``: return a selected one-sided periodic invariant
  subspace.
* ``krylov_schur_twosided``: experimental two-sided Krylov-Schur scaffold.
* ``periodic_krylov_schur_projectors``: run the one-sided solver on right and left
  periodic problems, then biorthogonalize the two returned bases.

Mathematical convention
-----------------------

There are ``period`` vector spaces arranged cyclically.  The one-sided solver
uses the R convention: ``matvecs[l]`` sends a block at site ``l + 1`` to site
``l``.  If the full-space factors are ``A_l``, the one-cycle operator starting
at site 0 is

    A_0 A_1 ... A_{period-1}.

The code never forms that large product.  Instead it builds one Arnoldi basis
per site and one small projected factor per edge.  With ``Q[l]`` containing
orthonormal Arnoldi columns for site ``l``, the periodic Arnoldi relation is

    A_l Q_{l+1} = Q_l H_l + R_l,

where site indices are modulo ``period``.  In array notation this is

    matvecs[l](Q[(l + 1) % period]) = Q[l] @ H[l] + residual[l].

``Q`` has shape ``(period, N, m)`` and ``H`` has shape
``(period, m, m)``, where ``m = num_krylov_iter * d_block``.  ``d_block`` is the
incoming seed width.  The Arnoldi construction is block Arnoldi: every Krylov
slot has width ``d_block``.  If a slot loses numerical rank, the arrays do not
shrink.  Instead, inactive columns are zeroed and tracked in ``active_cols`` so
JAX sees static shapes.

The Arnoldi factors pass directly to the R-oriented native periodic Schur
package:

    H[l] @ Z[l + 1] = Z[l] @ T[l].

After selecting and reordering columns of the small Schur factors, the selected
full-space basis is lifted as

    X[l] = Q[l] @ Z_selected[l].

For ``period == 1`` this reduces to ordinary block Arnoldi followed by a Schur
decomposition and Schur-vector selection of the projected Hessenberg matrix.

Index conventions for inputs and returned bases
-----------------------------------------------

All public tuples are indexed by the same physical period-site label
``p = 0, ..., period - 1``.  Indices wrap modulo ``period``.

For a one-sided R-oriented solve:

* ``Vs[p]`` is the seed block at physical site ``p``.
* ``matvecs[p]`` maps a block at site ``p + 1`` to site ``p``.
* ``periodic_krylov_schur_onesided(matvecs, Vs)`` returns ``X[p]`` and
  ``T_selected[p]`` at physical site/edge ``p``.

For the two-sided projector helper:

* ``VRs[p]`` is the right seed block at physical site ``p``.
* ``VLs[p]`` is the left seed block at the same physical site ``p``.  The input
  seed projector is therefore ``VRs[p] @ VLs[p].T``.
* ``matvecs[p]`` is the R-oriented edge ``A_p: p + 1 -> p``.
* ``vecmats[p]`` applies ``A_p.T`` from site ``p`` to site ``p + 1``.  The
  left space is bilinear-dual, not conjugate-adjoint: use transpose, not
  conjugate transpose.

``periodic_krylov_schur_projectors`` returns ``XR[p]`` and ``XL[p]`` at the
same physical projector site ``p`` and the residual-dropped Schur-gauge factor
``c[p]``.  Callers should use the same-site bilinear convention

    XL[p].T @ XR[p]

and the same-site oblique projector

    XR[p] @ XL[p].T.

The returned reduced factor has the R-oriented labels

    c[p] ~= XL[p].T @ A_p @ XR[p + 1].

It keeps only the periodic-Schur part of this projected action and deliberately
drops the lifted Arnoldi residual.  Internally the right problem passes
directly to the one-sided R solver, while the transpose ``vecmats`` and left
seeds are reversed into R order and restored afterward.

Fixed-shape rank handling
-------------------------

The Arnoldi code is written for JAX transformations, so it avoids dynamically
changing the width of any block.  By default, ``mask_rank`` uses QR with column
pivoting on the small block factor; ``pivot=False`` selects an unpivoted static
kernel.  Columns whose diagonal factor is too small are marked inactive,
zeroed in the returned basis, and represented in ``dropped_residual`` when
``full_res=True``.  Full mode therefore keeps the Arnoldi relation
algebraically honest while preserving static array shapes.

The pivoted rank cutoff is

    abs(diag(R_pivoted)) > rank_tol * eps * original_column_norm.

Periodic Sylvester solves may also supply an absolute ``scale_tol[k]`` for
operator-generated blocks.  Their cutoff is the maximum of the relative and
absolute thresholds.  The Sylvester scale also has the roundoff floor
``4 * eps * ||c[k]||_2``.  Setting all tolerances to zero disables masking.

Orthogonalization and residuals
-------------------------------

``ortho_method="MGS"`` uses sequential block modified Gram-Schmidt against the
existing block prefix.  ``ortho_method="CGS"`` uses one classical projection
with GEMM/einsum style contractions.  ``max_reortho`` and ``eta`` implement a
DGKS-style reorthogonalization test: a newly orthogonalized block is accepted
when its column norms did not drop by more than the ``eta`` factor.  The
default ``eta=1/sqrt(2)`` is the usual conservative test.

By default, ``periodic_arnoldi_basis`` returns only the final residual block.
With ``full_res=True`` it also retains residuals from intermediate rank
deflation, folds their components in the destination Arnoldi basis back into
``H``, and returns the complete fixed-shape Arnoldi relation.

Schur implementation
--------------------

The projected Krylov-Schur factors use the native R convention

    H[l] @ Z[l + 1] = Z[l] @ T[l].

Arnoldi, decomposition, reordering, locking, and lifting all retain these site
labels. Real inputs use the staged SLICOT ``MB03VD/VY/WD`` decomposition and
``MB03KD`` reordering. Complex inputs use the native Hessenberg reducer with
SLICOT ``MB03BZ`` and the custom 1-by-1 complex exchange kernel.
The old Julia implementation is isolated in
``linalg/periodic_schur/julia_version.py``
and is not imported by this module.

Selection model
---------------

Selection happens in Python/JAX around the small Schur problem:

1. Compute scores from the returned Ritz values.
2. Identify inseparable real-Schur complex pairs.
3. Mark policy-specific locks.
4. Build a pair-safe ``select_mask``.
5. Reorder the existing Schur form so selected columns move
   to the front.
6. Lift the selected Schur vectors back through the Arnoldi bases.

``_candidate_scores`` scores each scalar Schur column by ``abs(ritz_value)``.
For real Schur forms, complex conjugate eigenvalues appear as adjacent 2-by-2
blocks; both columns in such a block are assigned the larger magnitude and are
selected or dropped together.  The pair detector deliberately does not assume
that the Schur routine returns ``a + ib`` before ``a - ib``.

Locking policies
----------------

Locking is only a priority mechanism.  A locked block is sorted before
unlocked blocks, but it can still be rejected if it is below
``CTM_eig_cutoff`` or cannot fit because of pair-size constraints.

``lock_policy="none"``
    No mode is locked.  Selection is by descending Ritz magnitude, subject to
    cutoff and pair rules.

``lock_policy="q0_eig"``
    Diagonalize the small periodic Schur product and ask whether each resulting
    approximate eigenvector still lives in the initial Arnoldi seed block
    ``Q[:, :, :d_block]``.  The helper intentionally forms the dense small
    product; it is a scoring heuristic, not a stable periodic triangular
    eigensolver.

``lock_policy="ritz_difference"``
    Compare the full projected Ritz values with supplied incoming cycle
    eigenvalues. CTMRG supplies the rank-truncated physical corner-cycle
    spectrum; standalone calls without that spectrum fall back to the leading
    seed block ``H[:, :d_block, :d_block]``. Distances are measured as

        abs(log(lambda_full / lambda_seed)).

    A seed Ritz value locks its nearest full Ritz value when the nearest match
    is less than ``ritz_difference_tau_ratio`` times the second-nearest match.
    Values below ``max(CTM_eig_cutoff * scale,
    ritz_difference_log_floor)`` are ignored so tiny values do not create
    meaningless log ratios.  This is the default because it is cheap and often
    captures the CTMRG-style desire to keep modes already present in the seed.

Split complex-pair policy
-------------------------

Real 2-by-2 Schur pairs occupy two output columns.  If the next best block is a
pair and only one output slot remains:

``split_pair_policy="drop_split_pair"``
    Stop selection at the previous column.  The trailing output slot remains
    inactive.  This avoids replacing a stronger pair with a weaker scalar just
    to fill the requested width.

``split_pair_policy="replace_split_pair"``
    Skip the non-fitting pair and continue scanning later candidates.  This may
    fill the width with weaker scalar blocks.

Configuration reference
-----------------------

The top-level projector API is

    XR, XL, c, info = periodic_krylov_schur_projectors(
        matvecs,
        vecmats,
        VRs,
        VLs,
        chi_max=chi_max,
        info_level=info_level,
        cfg={
            "num_krylov_iter": 5,
            "lock_policy": "ritz_difference",
            "split_pair_policy": "drop_split_pair",
            "CTM_eig_cutoff": 1e-15,
        },
    )

``chi_max`` and ``info_level`` are direct keyword arguments.  All algorithmic
options below go in the single ``cfg`` dictionary.  ``periodic_krylov_schur_projectors``
freezes that dictionary for JAX and passes the same options to both internal
one-sided solves.  The right problem is already R-oriented and passes directly
to ``periodic_krylov_schur_onesided``.  The left transpose problem is reversed
into R order before its one-sided solve and mapped back to physical sites
afterward.  There is no separate left/right cfg path unless one is added
explicitly.

``chi_max``
    Public argument, not a cfg key.  Requested output width.  Defaults to the
    input seed width.  Returned arrays keep this width, with inactive columns
    zeroed.

``info_level``
    Public argument, not a cfg key.  Must be nonnegative.  Level 0 returns the
    cheap always-present info.  Level 1 adds extra diagnostics.

``num_krylov_iter``
    Number of block Arnoldi slots, including the seed slot.  The projected
    dimension is ``m = num_krylov_iter * d_block`` after clipping to
    ``N // d_block``.
    Default in ``periodic_krylov_schur_onesided`` is 3.

``max_reortho``
    Maximum reorthogonalization passes per block.  Default 3.

``eta``
    DGKS reorthogonalization acceptance threshold.  Default ``1/sqrt(2)``.

``rank_tol``
    QRCP rank-mask scale in machine-epsilon units.  Default ``4``.  Use 0 to
    disable relative rank dropping.

``cred_rank_tol``
    Optional eig-only CRed QRP rank scale in machine-epsilon units. Each
    pivoted diagonal is compared to the norm of its incoming factor column.
    If omitted, CRed performs only structural active-rank compaction.

``sylvester_deflation_tol``
    Dimensionless absolute Arnoldi deflation scale relative to the smallest
    active eigenvalue magnitude of the corresponding Sylvester Schur factor.
    Default ``1e-6``.  The absolute scale is at least
    ``4 * eps * ||c[k]||_2``. The same absolute cutoff is applied to rows
    revealed by the compressed solver's periodic Hessenberg reduction.

``ortho_method``
    ``"MGS"`` or ``"CGS"``.  Default ``"MGS"``.

``lock_policy``
    ``"ritz_difference"``, ``"q0_eig"``, or ``"none"``.  Default
    ``"ritz_difference"``.

``ritz_q0_lock_tol``
    Threshold used by ``q0_eig``.  Default 0.0, which only locks exact
    zero-defect candidates.

``ritz_difference_tau_ratio``
    Ambiguity threshold for ``ritz_difference`` locking.  Default 0.05.

``ritz_difference_log_floor``
    Floor used before logarithmic Ritz-ratio distances.  Default ``1e-300``.

``CTM_eig_cutoff``
    Relative cutoff against the largest candidate Ritz magnitude.  Candidates
    below this threshold are not selected or locked.  Default ``1e-15``.

``split_pair_policy``
    ``"drop_split_pair"`` or ``"replace_split_pair"``.  Default
    ``"drop_split_pair"``.

``biorthogonalize_tol``
    SVD overlap rank tolerance in units of machine epsilon, used when
    biorthogonalizing selected right/left projectors.  Default ``10.0``.

``return_diagnostics``
    Adds selection internals when calling ``select_periodic_schur_subspace``
    directly.  ``periodic_krylov_schur_onesided`` does not currently copy those fields
    into its public ``info`` dictionary.

Return info
-----------

``periodic_krylov_schur_onesided`` returns ``(X, T_selected, info)``:

``X``
    Shape ``(period, N, chi_max)``.  Selected full-space Schur vectors.  Columns
    beyond the retained rank are zero.

``T_selected``
    Shape ``(period, chi_max, chi_max)``.  Selected R-oriented periodic-Schur
    factors satisfying ``H[p] Z[p + 1] = Z[p] T_selected[p]`` on the retained
    projected subspace.

``info["ritz_rank"]``
    Number of active selected output columns.

``info["schur_size"]``
    Live leading dimension returned by the full projected CRed decomposition.

``info["ritz_values"]``
    Static-width Ritz carrier returned before reordering/selection. Only the
    leading ``schur_size`` entries are live.

``info["ritz_values_kept"]``
    Length ``chi_max``.  Reordered kept Ritz values in selected-column order,
    with inactive columns zeroed.

``info["seed_schur_size"]``
    Live leading dimension of ``seed_ritz_values`` when seed locking is used,
    or zero for lock policies that do not compute the seed spectrum.

For ``info_level >= 1``:

``info["initial_ortho_res"]``
    Norm of the block immediately below the seed block,
    ``H[:, d_block:2*d_block, :d_block]``.  This is a cheap signal for how much
    the first Krylov expansion moved outside the original seed subspace.

``info["q0_X_svals"]``
    Per-period singular values between the initial seed block and the selected
    lifted subspace.

``info["q0_rank"]``
    Per-period numerical ranks of the initial seed blocks.

``periodic_krylov_schur_projectors`` returns ``(XR, XL, c, info)``:

``XR`` and ``XL``
    Shape ``(period, N, chi_max)``.  Public convention is same-site bilinear
    duality: after biorthogonalization, callers should use
    ``XL[p].T @ XR[p]`` and projectors ``XR[p] @ XL[p].T``.  There is no
    conjugation in this pairing because the left problem is built from
    transpose maps, not adjoint maps.

``c``
    Shape ``(period, chi_max, chi_max)``.  Residual-dropped Schur-gauge factors
    with R labels, ``c[p] ~= XL[p].T @ matvecs[p](XR[p + 1])``.  They are
    formed as the retained-rank triangular solve
    ``pinv(G_R[p]) @ T_selected[p] @ G_R[p + 1]``.

``info["right"]`` and ``info["left"]``
    The one-sided ``info`` dictionaries for the right and left solves.

``info["ritz_rank_R"]`` and ``info["ritz_rank_L"]``
    Retained ranks before biorthogonal truncation.  A mismatch triggers a
    ``jax.debug.print`` warning with the kept Ritz values.

``info["overlap_singular_values"]``
    Singular values of the same-site bilinear overlap ``XL[p].T @ XR[p]``
    before whitening.

``info["overlap_rank_by_site"]``
    Numerical rank of each same-site bilinear overlap, using
    ``biorthogonalize_tol``.

``info["biorthogonal_rank"]``
    ``min(ritz_rank_R, ritz_rank_L, overlap_rank_by_site.min())``.  At full
    overlap rank, ``XL[p].T @ XR[p]`` is the identity.  At deficient rank it is
    the rank-``r`` projector returned by ``biorthogonalize_bases``.

JAX and callback notes
----------------------

The top-level projector path is jitted.  ``cfg`` is frozen into a hashable
static tuple before it reaches ``jax.jit``. Periodic Schur decomposition and
reordering use XLA FFI.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from linalg.jax_linalg import (
    biorthogonalize_bases,
    periodic_schur_bruteforce,
    ql,
    split_qrp,
    triangular_pinv_solve,
)
from linalg.periodic_schur import _diagonalize_periodic_schur_callback
from linalg.periodic_schur import jax_ffi as periodic_schur_ffi
from linalg.sylvester_solvers.compressed import sylvester_compressed_periodic


DEFAULT_PERIODIC_KRYLOV_ITER = 6


Array = jax.Array
Matvec = Callable[[Array], Array]


def periodic_sylvester_schur(B, C):
    r"""Transform the dense right factors of a periodic Sylvester equation.

    Given

        A[k] X[k + 1] + X[k] B[k] = C[k],

    compute periodic Schur factors

        B[k] Z[k + 1] = Z[k] T[k].

    With ``X_schur[k] = X[k] Z[k]``, the transformed equation is

        A[k] X_schur[k + 1] + X_schur[k] T[k] = C_schur[k],

    where ``C_schur[k] = C[k] Z[k + 1]``.  Site indices are cyclic.
    """
    # TODO: Wire in the SLICOT periodic-Schur routine.
    Z, T = periodic_schur_bruteforce(B)
    C = jnp.stack(C) if isinstance(C, (list, tuple)) else jnp.asarray(C)
    Z_next = jnp.roll(Z, -1, axis=0)
    # C[k]_{mi} Z[k+1]_{ij} -> C_schur[k]_{mj}.
    C_schur = jnp.einsum("kmi,kij->kmj", C, Z_next)
    return T, Z, C_schur


@partial(jax.jit, static_argnames=("matvecs", "num_iter", "which"))
def periodic_power(
    matvecs: tuple[Matvec, ...],
    Vs: tuple[Array, ...],
    num_iter: int,
    which: str = "QR",
):
    """Warm up periodic column bases using descending QR or QL actions.

    Both factorizations use the CTM/SLICOT-oriented relation

    ``matvecs[k](Q[k + 1]) = Q[k] @ F[k]``,

    where ``F[k]`` is upper triangular for ``which="QR"`` and lower triangular
    for ``which="QL"``. ``num_iter`` complete descending sweeps are followed
    by one single-bond action at the start of the next sweep, for
    ``num_iter*period + 1`` total matvecs. After that final action the relation
    is exact on edges ``0,...,period-3,period-1``; edge ``period-2`` becomes
    exact at convergence.
    """
    period = len(matvecs)
    Q = list(Vs)
    R = [None] * period

    for _ in range(num_iter):
        for k in range(period - 1, -1, -1):
            X = matvecs[k](Q[(k + 1) % period])
            if which == "QR":
                Q[k], R[k] = jnp.linalg.qr(
                    X,
                    mode="reduced",
                )
            elif which == "QL":
                Q[k], R[k] = ql(X)
            else:
                raise ValueError(f"unknown periodic power factorization {which!r}")

    # Start the next descending sweep so a converged periodic Schur flag is
    # tested and restored across the cyclic seam without another full sweep.
    k = period - 1
    X = matvecs[k](Q[0])
    if which == "QR":
        Q[k], R[k] = jnp.linalg.qr(X, mode="reduced")
    elif which == "QL":
        Q[k], R[k] = ql(X)
    else:
        raise ValueError(f"unknown periodic power factorization {which!r}")

    return tuple(Q), tuple(R)


def left_projectors_to_reversed_cycle(
    vecmat: tuple[Matvec, ...],
    VL: tuple[Array, ...],
):
    """Convert physical row-left projectors to reversed column-solver order."""
    vecmat_reverse = vecmat[::-1]
    VL_columns_reverse = (VL[0].T,) + tuple(V.T for V in VL[:0:-1])
    return vecmat_reverse, VL_columns_reverse


def left_projectors_from_reversed_cycle(
    VL_columns_reverse: tuple[Array, ...],
    RL_reverse: tuple[Array, ...],
):
    """Restore physical row-left projectors and factors from reversed order."""
    VL_columns = (VL_columns_reverse[0],) + VL_columns_reverse[:0:-1]
    VL = tuple(V.T for V in VL_columns)
    RL = tuple(R.T for R in RL_reverse[::-1])
    return VL, RL


@partial(
    jax.jit,
    static_argnames=("matvec", "vecmat", "num_iter", "return_info"),
)
def periodic_power_projectors(
    matvec: tuple[Matvec, ...],
    vecmat: tuple[Matvec, ...],
    VR: tuple[Array, ...],
    VL: tuple[Array, ...],
    num_iter: int,
    return_info: bool = False,
):
    """Independently warm up right and left periodic projector bases.

    ``VR[k]`` is column-stored and ``VL[k]`` is row-stored. ``matvec[k]``
    applies ``C[k]`` and ``vecmat[k]`` applies ``C[k].T``.  The returned bases
    are whitened so ``VL_new[k] @ VR_new[k] = I`` at full overlap rank.  Both
    returned factor tuples contain the common projected action

    ``c[k] = VL_new[k] @ C[k] @ VR_new[k + 1]``.

    At convergence these factors additionally satisfy

    ``C[k] @ VR_new[k + 1] = VR_new[k] @ c[k]`` and
    ``VL_new[k] @ C[k] = c[k] @ VL_new[k + 1]``.

    The left solve reverses the transpose maps and physical cuts around a QL
    power sweep. Complementary exact QR/QL edges construct every ``c[k]``
    without another full-space action, except at periods one and three where
    the two lagged edges coincide and require one direct projected action.
    """
    VR_new, RR = periodic_power(matvec, VR, num_iter, which="QR")

    vecmat_reverse, VL_columns_reverse = left_projectors_to_reversed_cycle(
        vecmat,
        VL,
    )
    VL_columns_power, RL_reverse = periodic_power(
        vecmat_reverse,
        VL_columns_reverse,
        num_iter,
        which="QL",
    )
    VL_new, RL = left_projectors_from_reversed_cycle(
        VL_columns_power,
        RL_reverse,
    )

    VR_biorthogonal = []
    VL_biorthogonal = []
    G_R = []
    G_L = []
    overlap = []
    overlap_svals = []
    overlap_rank = []
    for k in range(len(VR_new)):
        VR_k, VL_k, G_R_k, G_L_k, overlap_k, s_k, rank_k = biorthogonalize_bases(
            VR_new[k],
            VL_new[k],
        )
        VR_biorthogonal.append(VR_k)
        VL_biorthogonal.append(VL_k)
        G_R.append(G_R_k)
        G_L.append(G_L_k)
        overlap.append(overlap_k)
        overlap_svals.append(s_k)
        overlap_rank.append(rank_k)
    VR_new = tuple(VR_biorthogonal)
    VL_new = tuple(VL_biorthogonal)

    period = len(VR_new)
    if period == 1:
        c = (jnp.dot(VL_new[0], matvec[0](VR_new[0])),)
    else:
        # After the extra cyclic action, the right sweep is exact except on
        # edge p-2, while the reversed left sweep is exact there. Splice that
        # one complementary edge. At p=3 both sweeps lag on edge one, so form
        # that projected action directly.
        c = []
        for k in range(period):
            if period == 3 and k == 1:
                c.append(jnp.dot(VL_new[k], matvec[k](VR_new[k + 1])))
                continue
            c_unwhitened = (
                jnp.dot(RL[k], overlap[(k + 1) % period])
                if k == period - 2
                else jnp.dot(overlap[k], RR[k])
            )
            c.append(
                jnp.dot(
                    G_L[k],
                    jnp.dot(c_unwhitened, G_R[(k + 1) % period]),
                )
            )
        c = tuple(c)
    if return_info:
        info = {
            "bond_biorth_svals": jnp.stack(overlap_svals),
            "bond_biorth_rank": jnp.stack(overlap_rank),
        }
        return VR_new, VL_new, c, info
    return VR_new, VL_new, c, c


def _freeze_static_cfg(cfg: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Return a hashable cfg representation suitable for static jit args."""

    def freeze_value(value):
        if isinstance(value, dict):
            return tuple((key, freeze_value(subvalue)) for key, subvalue in sorted(value.items()))
        if isinstance(value, list):
            return tuple(freeze_value(item) for item in value)
        if isinstance(value, tuple):
            return tuple(freeze_value(item) for item in value)
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"cfg value {value!r} is not hashable/static") from exc
        return value

    return tuple((key, freeze_value(value)) for key, value in sorted(cfg.items()))


def MGS(Q, V, j):
    """Sequentially orthogonalize V against block prefix Q[:j + 1]."""
    num_krylov_iter = Q.shape[0]
    d_block = V.shape[1]
    h0 = jnp.zeros((d_block, d_block), dtype=V.dtype)

    def step(V, i):
        def active(_):
            h = jnp.dot(Q[i].conj().T, V)
            return V - jnp.dot(Q[i], h), h

        return jax.lax.cond(i <= j, active, lambda _: (V, h0), operand=None)

    return jax.lax.scan(step, V, jnp.arange(num_krylov_iter))


def CGS(Q, V, j):
    """Classically orthogonalize V against block prefix Q[:j + 1] with GEMMs."""
    # Q[i,n,a]^* V[n,b] -> h[i,a,b]
    h = jnp.einsum("ina,nb->iab", Q.conj(), V)
    # Q[i,n,a] h[i,a,b] -> projection[n,b]
    V = V - jnp.einsum("ina,iab->nb", Q, h)
    return V, h


def _ortho(Q, V, j, ortho_method):
    """Orthogonalize V against a block Arnoldi prefix using the selected method."""
    if ortho_method == "MGS":
        return MGS(Q, V, j)
    elif ortho_method == "CGS":
        return CGS(Q, V, j)
    else:
        raise ValueError(f"Unknown ortho_method: {ortho_method!r}")
    
def reortho(Q, V, j, ortho_method, max_reortho, eta):
    """Reorthogonalize V against a block Arnoldi prefix.

    This is used for the final Arnoldi block, where there is no next basis block
    to normalize into.  It repeatedly projects ``V`` away from ``Q[:j + 1]`` and
    accumulates the corresponding projected coefficients in ``hj`` until the
    Daniel-Gragg-Kaufman-Stewart norm test accepts the residual or
    ``max_reortho`` passes have been used.

    Returns:
      ``(residual, hj)`` with ``residual`` orthogonalized against the prefix and
      ``hj[i]`` holding the block coefficient for basis block ``Q[i]``.
    """
    num_krylov_iter = Q.shape[0]
    d_block = V.shape[1]
    hj = jnp.zeros((num_krylov_iter, d_block, d_block), dtype=V.dtype)
    accepted = jnp.array(False)
    k = jnp.array(0, dtype=jnp.int32)

    def cond(carry):
        _, _, k, accepted = carry
        return (k < max_reortho) & (~accepted)

    def step(carry):
        V, hj, k, _ = carry
        old_nrm = jnp.linalg.norm(V, axis=0)
        V, dh = _ortho(Q, V, j, ortho_method)
        new_nrm = jnp.linalg.norm(V, axis=0)
        accepted = jnp.all(new_nrm >= eta * old_nrm)
        return V, hj + dh, k + 1, accepted

    V, hj, _, _ = jax.lax.while_loop(cond, step, (V, hj, k, accepted))
    return V, hj


def mask_rank(X, r, old_nrm, rank_tol, scale_tol, full_res, pivot):
    """Apply fixed-shape rank masking to a block factorization.

    The Arnoldi code keeps all arrays at their original block width so the
    result remains compatible with JAX transformations.  Instead of returning
    a shorter basis when a column is numerically dependent, this helper zeros
    rejected columns and returns a boolean mask identifying which columns are
    still active.

    Args:
      X: Optional left factor in ``V = X @ r``.  If ``None``, ``r`` itself is
        interpreted as the factor to orthonormalize.
      r: Seed block or small block factor to mask.
      old_nrm: Original column norms used to scale the rank cutoff after
        pivoting.
      rank_tol: Cutoff multiplier in units of machine epsilon.  A zero value
        disables the relative cutoff.
      scale_tol: Absolute cutoff for the diagonal pivots.  Setting both
        ``rank_tol`` and ``scale_tol`` to zero disables rank dropping.
      full_res: Whether to retain the residual removed by rank masking.
      pivot: Whether to use QR with column pivoting.

    Returns:
      ``(X, r_piv, keep, p, dropped_residual)``.  In pivoted mode ``p`` is the
      QRCP column permutation; otherwise it is ``None``.  ``keep`` marks
      numerically independent columns, and ``dropped_residual`` contains the
      part removed by masking in full mode.
    """
    if pivot:
        q, r_piv, p = split_qrp(r)
        if X is None:
            X = q
        else:
            X = jnp.dot(X, q)
        scale = jnp.take(old_nrm, p)
    else:
        if X is None:
            X, r_piv = jnp.linalg.qr(r, mode="reduced")
        else:
            r_piv = r
        p = None
        scale = old_nrm

    diag_nrm = jnp.abs(jnp.diag(r_piv))
    relative_cutoff = rank_tol * jnp.finfo(diag_nrm.dtype).eps * scale
    scale_tol = jnp.asarray(scale_tol, dtype=diag_nrm.dtype)
    cutoff = jnp.maximum(relative_cutoff, scale_tol)
    masking = (rank_tol != 0) | (scale_tol != 0)
    keep = jnp.where(masking, diag_nrm > cutoff, True)
    dropped = ~keep
    dropped_residual = (
        jnp.dot(X, r_piv * dropped[:, None])
        if full_res
        else None
    )
    X = X * keep[None, :]
    r_piv = r_piv * keep[:, None]

    return X, r_piv, keep, p, dropped_residual



def project_and_normalize(
    Q,
    V,
    j,
    ortho_method,
    max_reortho,
    eta,
    rank_tol,
    scale_tol,
    full_res,
    pivot,
):
    """Project V into Q plus a normalized fixed-shape residual block.

    For an incoming block ``V0`` the pivoted path constructs
    ``V0[:, p] = sum_{i <= j} Q[i] @ hj[i] + X @ hj[j + 1] + dropped_residual``.
    The unpivoted path has the same relation in the original column order.
    The residual factor ``X`` is QR-normalized, then passed through
    ``mask_rank`` so rank-deficient or absolutely negligible columns are
    represented by zeroed fixed-shape columns rather than by changing the
    array shape.

    Returns:
      ``(X, hj, active_next, p, dropped_residual)``.  ``X`` is the next
      Arnoldi block, ``hj[j + 1]`` is its subdiagonal block coefficient,
      ``active_next`` marks active columns of ``X``, and ``p`` is the column
      permutation that must also be applied to the source block and affected
      rows of ``H``; it is ``None`` in the unpivoted path.
    """
    old_nrm = jnp.linalg.norm(V, axis=0)
    X, hj = _ortho(Q, V, j, ortho_method)
    X, r = jnp.linalg.qr(X, mode="reduced")
    accepted = jnp.all(jnp.abs(jnp.diag(r)) >= eta * old_nrm)
    k = jnp.array(1, dtype=jnp.int32)

    def cond(carry):
        _, _, _, k, accepted = carry
        return (k < max_reortho) & (~accepted)

    def step(carry):
        X, hj, r, k, _ = carry
        X, dh = _ortho(Q, X, j, ortho_method)
        X, r2 = jnp.linalg.qr(X, mode="reduced")
        hj = hj + jnp.einsum("iab,bc->iac", dh, r)
        r = jnp.dot(r2, r)
        accepted = jnp.all(jnp.abs(jnp.diag(r2)) >= eta)
        return X, hj, r, k + 1, accepted

    X, hj, r, _, _ = jax.lax.while_loop(cond, step, (X, hj, r, k, accepted))
    X, r, active_next, p, dropped_residual = mask_rank(
        X, r, old_nrm, rank_tol, scale_tol, full_res, pivot
    )
    if pivot:
        hj = jnp.take(hj, p, axis=2)
    hj = hj.at[j + 1].set(r)
    return X, hj, active_next, p, dropped_residual


@partial(
    jax.jit,
    static_argnames=(
        "matvecs",
        "num_krylov_iter",
        "max_reortho",
        "eta",
        "rank_tol",
        "ortho_method",
        "full_res",
        "pivot",
        "orientation",
    ),
)
def periodic_arnoldi_basis(
    matvecs: tuple[Matvec, ...],
    Vs: tuple[Array, ...],
    num_krylov_iter,
    max_reortho=3,
    eta=1.0 / np.sqrt(2.0),
    rank_tol=4.0,
    scale_tol=None,
    ortho_method="MGS",
    full_res=False,
    pivot=True,
    orientation="L",
):
    """Build a block periodic Arnoldi relation.

    With ``orientation="L"``, ``matvecs[l]`` maps vectors at site ``l`` into
    the space at site ``(l + 1) % period`` and the projected relation is

    ``matvecs[l](Q[l]) = Q[(l + 1) % period] @ H[l] + residual[l]``.

    With ``orientation="R"``, it instead returns the opposite cyclic relation

    ``matvecs[l](Q[(l + 1) % period]) = Q[l] @ H[l] + residual[l]``.

    In compact mode, ``residual[l]`` is the final block in this relation;
    intermediate deflation residuals are intentionally omitted.

    The right-oriented path only reverses the cyclic input labels before the
    existing left-oriented Arnoldi construction and restores physical labels
    on return.

    Each Arnoldi slot has width ``d_block``.  Rank loss is handled with active
    masks and zeroed columns so the returned arrays keep static shape
    ``num_krylov_iter * d_block`` even if some directions deflate.

    Args:
      matvecs: Periodic linear maps, one per site.
      Vs: Seed blocks, one per site, each with shape ``(N, d_block)``.
      num_krylov_iter: Number of block Arnoldi slots, including the seed slot.
      max_reortho: Maximum reorthogonalization passes per block.
      eta: DGKS norm threshold for accepting an orthogonalized block.
      rank_tol: QR rank mask scale in units of machine epsilon.
      scale_tol: Optional length-period absolute pivot cutoffs for
        operator-generated Arnoldi blocks.  The seed blocks use only
        ``rank_tol``.
      ortho_method: ``"MGS"`` for sequential block MGS or ``"CGS"`` for one
        classical projection.
      full_res: If true, retain intermediate deflation residuals and return the
        complete Arnoldi relation.  Otherwise return only the final residual
        block.
      pivot: Whether seed and block-rank factorizations use QR with column
        pivoting.  False selects a separate unpivoted kernel without
        permutation gathers or scatters.
      orientation: ``"L"`` for the forward relation or ``"R"`` for the
        opposite cyclic relation.

    Returns:
      ``(Q, H, residual, v, active_cols)``.  ``Q``, ``H``, ``v``, and
      ``active_cols`` have shapes ``(period, N, m)``, ``(period, m, m)``,
      ``(period, d_block, d_block)``, and ``(period, m)``, where
      ``m = num_krylov_iter * d_block``.  ``residual`` has shape
      ``(period, N, m)`` in full mode and ``(period, N, d_block)`` otherwise.
      The seed coordinates satisfy ``Vs[k] = Q[k, :, :d_block] @ v[k]`` up to
      seed-rank truncation.  ``active_cols`` tells which flattened Arnoldi
      columns should be interpreted as live Krylov directions.
    """
    if len(matvecs) != len(Vs):
        raise ValueError("matvecs and Vs must have the same period")

    period = len(Vs)
    if scale_tol is None:
        scale_tol = jnp.zeros(
            (period,), dtype=jnp.real(jnp.asarray(Vs[0])).dtype
        )
    else:
        scale_tol = jnp.asarray(scale_tol)
    if orientation == "R":
        site_order = tuple((-l) % period for l in range(period))
        edge_order = tuple((-l - 1) % period for l in range(period))
        matvecs = tuple(matvecs[k] for k in edge_order)
        Vs = tuple(Vs[k] for k in site_order)
        scale_tol = jnp.take(scale_tol, jnp.asarray(edge_order))
    elif orientation != "L":
        raise ValueError(f"unknown periodic Arnoldi orientation {orientation!r}")

    Vs = jnp.stack(Vs)
    N, d_block = Vs.shape[1:]

    num_krylov_iter = min(num_krylov_iter, N // d_block)
    Q = jnp.zeros((period, num_krylov_iter, N, d_block), dtype=Vs.dtype)
    active_cols = jnp.zeros((period, num_krylov_iter, d_block), dtype=jnp.bool_)

    def initialize_seed(V):
        V, r, seed_active, p, _ = mask_rank(
            None,
            V,
            jnp.linalg.norm(V, axis=0),
            rank_tol,
            0.0,
            False,
            pivot,
        )
        if pivot:
            # V_input[:, p] = V @ r, hence V_input = V @ v with v[:, p] = r.
            v = jnp.zeros_like(r).at[:, p].set(r)
        else:
            v = r
        return V, v, seed_active

    seed_Vs, v, seed_active = jax.vmap(initialize_seed)(Vs)
    Q = Q.at[:, 0].set(seed_Vs)
    active_cols = active_cols.at[:, 0].set(seed_active)

    def periodic_step(carry, j):
        Q, H_blocks, residual_blocks, v, active_cols = carry
        for l in range(period):
            active = jnp.any(active_cols[l, j])

            def active_step(args):
                Q, H_blocks, residual_blocks, v, active_cols = args
                lprev = (l - 1) % period
                lp = (l + 1) % period

                V = matvecs[l](Q[l, j])
                V, hj, active_next, p, dropped_residual = project_and_normalize(
                    Q[lp],
                    V,
                    j,
                    ortho_method,
                    max_reortho,
                    eta,
                    rank_tol,
                    scale_tol[l],
                    full_res,
                    pivot,
                )

                if pivot:
                    # QRCP may permute the source block columns.  Apply the
                    # same permutation to the block and its coefficient rows.
                    Q = Q.at[l, j].set(jnp.take(Q[l, j], p, axis=1))
                    active_cols = active_cols.at[l, j].set(
                        jnp.take(active_cols[l, j], p)
                    )
                    v_l = jax.lax.cond(
                        j == 0,
                        lambda v_l: jnp.take(v_l, p, axis=0),
                        lambda v_l: v_l,
                        v[l],
                    )
                    v = v.at[l].set(v_l)

                Q = Q.at[lp, j + 1].set(V)
                active_cols = active_cols.at[lp, j + 1].set(active_next)
                H_blocks = H_blocks.at[l, :, j].set(hj)

                if pivot:
                    H_blocks = H_blocks.at[lprev, j, j].set(
                        jnp.take(H_blocks[lprev, j, j], p, axis=0)
                    )

                    def pivot_previous_step_rows(H_blocks):
                        return H_blocks.at[lprev, j, j - 1].set(
                            jnp.take(H_blocks[lprev, j, j - 1], p, axis=0)
                        )

                    H_blocks = jax.lax.cond(
                        j > 0,
                        pivot_previous_step_rows,
                        lambda H_blocks: H_blocks,
                        H_blocks,
                    )
                if full_res:
                    residual_blocks = residual_blocks.at[l, j].set(dropped_residual)
                return Q, H_blocks, residual_blocks, v, active_cols

            def inactive_step(args):
                return args

            Q, H_blocks, residual_blocks, v, active_cols = jax.lax.cond(
                active,
                active_step,
                inactive_step,
                (Q, H_blocks, residual_blocks, v, active_cols),
            )
        return (Q, H_blocks, residual_blocks, v, active_cols), None

    # H_blocks_work[l, row, col] stores the block from source column col
    # to destination row row for factor l.
    H_blocks_work = jnp.zeros((period, num_krylov_iter, num_krylov_iter, d_block, d_block), dtype=Vs.dtype)
    residual_blocks = (
        jnp.zeros((period, num_krylov_iter, N, d_block), dtype=Vs.dtype)
        if full_res
        else None
    )
    if num_krylov_iter > 1:
        (Q, H_blocks_work, residual_blocks, v, active_cols), _ = jax.lax.scan(
            periodic_step,
            (Q, H_blocks_work, residual_blocks, v, active_cols),
            jnp.arange(num_krylov_iter - 1),
        )

    def final_site(l, H_blocks):
        active = jnp.any(active_cols[l, num_krylov_iter - 1])

        def active_step(H_blocks):
            lp = (l + 1) % period
            V = matvecs[l](Q[l, num_krylov_iter - 1])
            residual, h_last = reortho(Q[lp], V, num_krylov_iter - 1, ortho_method, max_reortho, eta)
            H_blocks = H_blocks.at[l, :, num_krylov_iter - 1].set(h_last)
            return H_blocks, residual

        def inactive_step(H_blocks):
            residual = jnp.zeros((N, d_block), dtype=Vs.dtype)
            return H_blocks, residual

        return jax.lax.cond(active, active_step, inactive_step, H_blocks)

    final_residuals = []
    for l in range(period):
        H_blocks_work, residual_l = final_site(l, H_blocks_work)
        final_residuals.append(residual_l)
    residual = jnp.stack(final_residuals)

    if full_res:
        residual_blocks = residual_blocks.at[:, num_krylov_iter - 1].add(residual)
        for l in range(period):
            lp = (l + 1) % period
            # Fold any residual component lying back in the destination Arnoldi
            # basis into H.  This keeps the final residual orthogonal to Q[lp]
            # without changing the fixed array shapes used by JAX.
            residual_coeff = jnp.einsum("ina,jnb->iajb", Q[lp].conj(), residual_blocks[l])
            residual_coeff = residual_coeff * active_cols[lp, :, :, None, None] * active_cols[l, None, None, :, :]
            H_blocks_work = H_blocks_work.at[l].add(jnp.transpose(residual_coeff, [0, 2, 1, 3]))
            residual_blocks = residual_blocks.at[l].add(-jnp.einsum("ina,iajb->jnb", Q[lp], residual_coeff))

        residual = jnp.transpose(residual_blocks, [0, 2, 1, 3])
        residual = jnp.reshape(residual, (period, N, num_krylov_iter * d_block))

    Q_flat = jnp.transpose(Q, [0, 2, 1, 3])
    Q_flat = jnp.reshape(Q_flat, (period, N, num_krylov_iter * d_block))
    H = jnp.transpose(H_blocks_work, [0, 1, 3, 2, 4])
    H = jnp.reshape(H, (period, num_krylov_iter * d_block, num_krylov_iter * d_block))
    active_cols = jnp.reshape(active_cols, (period, num_krylov_iter * d_block))

    if orientation == "R":
        Q_flat = jnp.take(Q_flat, jnp.asarray(site_order), axis=0)
        H = jnp.take(H, jnp.asarray(edge_order), axis=0)
        residual = jnp.take(residual, jnp.asarray(edge_order), axis=0)
        v = jnp.take(v, jnp.asarray(site_order), axis=0)
        active_cols = jnp.take(active_cols, jnp.asarray(site_order), axis=0)

    return Q_flat, H, residual, v, active_cols


def _sylvester_min_abs_eig(w, rank, block_2x2_start):
    """Return the smallest active eigenvalue magnitude of each Schur factor."""
    eig_abs = jnp.abs(jnp.diagonal(w, axis1=1, axis2=2))
    if not jnp.issubdtype(w.dtype, jnp.complexfloating):
        starts = block_2x2_start[:-1]
        w0 = w[0]
        pair_abs = jnp.sqrt(jnp.abs(
            jnp.diag(w0)[:-1] * jnp.diag(w0)[1:]
            - jnp.diag(w0, k=1) * jnp.diag(w0, k=-1)
        ))
        eig_abs = eig_abs.at[0, :-1].set(
            jnp.where(starts, pair_abs, eig_abs[0, :-1])
        )
        eig_abs = eig_abs.at[0, 1:].set(
            jnp.where(starts, pair_abs, eig_abs[0, 1:])
        )
    active = jnp.arange(w.shape[1]) < rank
    min_abs_eig = jnp.min(
        jnp.where(active[None, :], eig_abs, jnp.inf),
        axis=1,
    )
    return jnp.where(rank > 0, min_abs_eig, 0.0)


def periodic_krylov_sylvester(
    matvecs,
    w,
    B,
    rank,
    X0=None,
    krylov_cfg=None,
    block_2x2_start=None,
):
    r"""Periodic block-GMRES cycle for a Sylvester equation.

    The physical equation and cyclic Arnoldi orientation are

        matvecs[k](X[k + 1]) + X[k] w[k] = B[k],
        matvecs[k](Q[k + 1]) = Q[k] H[k] + residual[k].

    The dense factors ``w[k]`` are assumed to be in upper periodic Schur form.
    Following SLICOT ``MB03WD``, ``w[0]`` is quasi-upper-triangular for real
    input and ``w[1:]`` are upper triangular.  ``block_2x2_start`` therefore
    describes diagonal blocks of ``w[0]``.  Complex factors are triangular
    throughout. ``rank`` is the known structural rank of the right factors;
    columns ``rank:`` of ``w``, ``B``, and the returned correction are exact
    zero padding.

    Writing ``X = X0 + dX`` gives the correction equation

        matvecs[k](dX[k + 1]) + dX[k] w[k] = R[k],
        R[k] = B[k] - matvecs[k](X0[k + 1]) - X0[k] w[k].

    Factoring the final Arnoldi residual block as

        residual[k] = Q_residual[k] residual_r[k],

    with ``residual_r[k]`` of shape ``(d_block, d_block)``.

    GMRES solvers proceed from the least-squares objective for ``dX[k] = Q[k] Y[k]``:

        sum_k ||H[k] Y[k+1] + Y[k] w[k] - beta[k]||_F^2
            + ||residual_r[k] Y[k+1, -d_block:, :]||_F^2,

    where ``beta[k] = [v[k]; 0]``.

    Galerkin instead solves H[k] Y[k+1] + Y[k] w[k] - beta[k] = 0

    Galerkin is significantly faster.

    The solver choices are:

    * ``"periodic_schur_galerkin"`` (default): real periodic Schur followed
      by an exact solve of the projected core equation.  The Arnoldi tail
      affects only the returned residual norm.
    * ``"dense_gmres"``: GMRES with a dense projected solve.
    * ``"periodic_schur_gmres"``: GMRES with periodic Schur followed by
      structured least-squares solves including the Arnoldi-tail rows.

    For the Galerkin route, ``krylov_cfg["galerkin_block_solver"]`` selects
    the dense ``"dgesv"`` route or SLICOT ``"mb03ke"``.  The default uses
    ``MB03KE`` for real cycles of length at least two and LAPACK ``GESV``
    otherwise; the complex periodic-Schur coordinate solve calls ``ZGESV``.

    If ``E[k]`` selects ``active_cols[k]`` from periodic Arnoldi, the native
    problem uses ``E[k].T H[k] E[k+1]`` and solves only
    ``E[k].T Y[k, :, :rank]``. Active dimensions may differ between sites.
    The solution is scattered back into the fixed JAX shapes with inactive
    Arnoldi coordinates and physical columns ``rank:`` exactly zero.

    The native solver returns ``x[k] = Y[k].T`` and a residual norm for each
    site and right-hand-side column. The physical answer is reconstructed as
    ``X[k] = X0[k] + Q[k] Y[k]``. This route assumes the upper/quasi-upper
    periodic Schur convention described above.
    """
    krylov_cfg = {} if krylov_cfg is None else krylov_cfg
    num_iter = krylov_cfg.get("num_krylov_iter", 3)
    sylvester_solver = krylov_cfg.get(
        "sylvester_solver",
        "periodic_schur_galerkin",
    )
    rank_tol = krylov_cfg.get("rank_tol", None)
    rank_tol_arnoldi = 4.0 if rank_tol is None else rank_tol
    sylvester_deflation_tol = krylov_cfg.get("sylvester_deflation_tol", 1e-6)

    w = jnp.stack(w) if isinstance(w, (list, tuple)) else jnp.asarray(w)
    B = jnp.stack(B) if isinstance(B, (list, tuple)) else jnp.asarray(B)
    period = B.shape[0]
    if block_2x2_start is None:
        block_2x2_start = jnp.zeros((w.shape[1],), dtype=jnp.bool_)
    block_2x2_start = jnp.asarray(block_2x2_start, dtype=jnp.bool_)
    c_min = _sylvester_min_abs_eig(w, rank, block_2x2_start)
    active = jnp.arange(w.shape[1]) < rank
    c_active = w * active[None, :, None] * active[None, None, :]
    c_max = jnp.linalg.norm(c_active, ord=2, axis=(-2, -1))
    scale_tol = jnp.maximum(
        sylvester_deflation_tol
        * c_min,
        4.0 * jnp.finfo(c_min.dtype).eps * c_max,
    )
    galerkin_block_solver = krylov_cfg.get(
        "galerkin_block_solver",
        (
            "mb03ke"
            if period >= 2 and not jnp.issubdtype(w.dtype, jnp.complexfloating)
            else "dgesv"
        ),
    )
    if X0 is None:
        X0 = jnp.zeros_like(B)
        R = B
    else:
        X0 = jnp.stack(X0) if isinstance(X0, (list, tuple)) else jnp.asarray(X0)
        # B[k] - A[k] X0[k+1] - X0[k] w[k] -> R[k].
        R = jnp.stack(tuple(
            B[k]
            - matvecs[k](X0[(k + 1) % period])
            - jnp.dot(X0[k], w[k])
            for k in range(period)
        ))

    Q, H, residual, v, active_cols = periodic_arnoldi_basis(
        matvecs,
        tuple(R[k] for k in range(period)),
        num_iter,
        max_reortho=krylov_cfg.get("max_reortho", 3),
        eta=krylov_cfg.get("eta", 1.0 / np.sqrt(2.0)),
        rank_tol=rank_tol_arnoldi,
        scale_tol=scale_tol,
        ortho_method=krylov_cfg.get("ortho_method", "MGS"),
        full_res=False,
        pivot=krylov_cfg.get("pivot", True),
        orientation="R",
    )

    # Only the final d-column Arnoldi residual is retained, so its thin QR
    # gives the d-by-d factor multiplying the trailing Arnoldi block.
    # TODO: Galerkin does not use this factor in its solution. Design a useful
    # inexpensive diagnostic proxy before making its residual path optional.
    residual_r = jax.vmap(lambda x: jnp.linalg.qr(x, mode="r"))(residual)
    x, error = sylvester_compressed_periodic(
        H,
        w,
        v,
        residual_r,
        active_cols,
        rank,
        method=sylvester_solver,
        block_2x2_start=block_2x2_start,
        galerkin_block_solver=galerkin_block_solver,
        scale_tol=scale_tol,
    )

    # Q[k] Y[k], with x[k] = Y[k].T.
    X = X0 + jnp.einsum("kmi,kji->kmj", Q, x)
    return X, error


def diagonalize_periodic_schur(T, Z, eigenvalues, schur_size):
    """Return live eigenvectors of a static R-oriented Schur carrier.

    This intentionally forms the dense product in Schur coordinates, computes
    right eigenvectors of that product, and propagates them around the period
    with the local ``T[l]`` factors. Only the leading ``schur_size`` sector is
    diagonalized; remaining output columns are zero. It is a temporary, simple
    scoring helper, not a stable periodic triangular backsolve.
    """
    T = jnp.asarray(T)
    Z = jnp.asarray(Z)
    eigenvalues = jnp.asarray(eigenvalues)
    return jax.pure_callback(
        _diagonalize_periodic_schur_callback,
        jax.ShapeDtypeStruct(Z.shape, jnp.complex128),
        T,
        Z,
        eigenvalues,
        jnp.asarray(schur_size, dtype=jnp.int32),
    )


def _candidate_scores(T, eigvals, schur_size):
    """Compute selection scores for Schur columns or real 2-by-2 blocks.

    ``T`` and ``eigvals`` come from the native periodic Schur decomposition.
    The dominant-size score is ``ritz_abs = abs(eigvals)``. For real 2-by-2
    blocks both columns of a pair receive the larger returned magnitude.

    Returns:
      ``(ritz_abs, pair_start)``.  ``pair_start[i]`` is true when column ``i``
      starts a real Schur 2-by-2 block; the following column belongs to the same
      inseparable pair.
    """
    T = jnp.asarray(T)
    T0 = T[0]
    eigvals = jnp.asarray(eigvals)
    live = jnp.arange(eigvals.shape[0]) < schur_size
    ritz_abs = jnp.where(live, jnp.abs(eigvals), 0.0)
    m = T0.shape[0]
    real_dtype = jnp.real(T0).dtype

    pair_start = jnp.zeros((m,), dtype=jnp.bool_)
    if not jnp.iscomplexobj(T0) and m > 1:
        # Real Schur complex pairs are represented by an inseparable 2-by-2
        # block.  Selection must score and move both columns together.  Backends
        # may return either conjugate sign first, so group adjacent complex
        # eigenvalues by position within each conjugate pair.
        imag_abs = jnp.abs(jnp.imag(eigvals))
        imag_scale = 100 * jnp.finfo(real_dtype).eps * jnp.maximum(1.0, jnp.abs(jnp.real(eigvals)))
        complex_ritz = live & (imag_abs > imag_scale)
        complex_count = jnp.cumsum(complex_ritz.astype(jnp.int32))
        followed_by_complex = jnp.concatenate([complex_ritz[1:], jnp.zeros((1,), dtype=jnp.bool_)])
        pair_start = complex_ritz & followed_by_complex & ((complex_count % 2) == 1)

    pair_second = jnp.concatenate([jnp.zeros((1,), dtype=jnp.bool_), pair_start[:-1]])
    if m > 1:
        pair_abs = jnp.maximum(ritz_abs[:-1], ritz_abs[1:])
        pair_start_abs = jnp.concatenate([pair_abs, jnp.zeros((1,), dtype=real_dtype)])
        pair_second_abs = jnp.concatenate([jnp.zeros((1,), dtype=real_dtype), pair_abs])
        ritz_abs = jnp.where(pair_start, pair_start_abs, ritz_abs)
        ritz_abs = jnp.where(pair_second, pair_second_abs, ritz_abs)

    return ritz_abs, pair_start


def compute_locks(
    T,
    Z,
    eigvals,
    is_complex_pair,
    schur_size,
    d_block,
    cfg,
    seed_ritz_values=None,
    seed_schur_size=None,
):
    """Compute policy-specific candidate locks for Schur selection."""
    cfg = {} if cfg is None else cfg
    lock_policy = cfg.get("lock_policy", "ritz_difference")
    eigvals = jnp.asarray(eigvals)
    is_complex_pair = jnp.asarray(is_complex_pair, dtype=jnp.bool_)
    m = eigvals.shape[0]
    full_live = jnp.arange(m) < schur_size

    if lock_policy == "none":
        locked = jnp.zeros((m,), dtype=jnp.bool_)
        return locked, {"lock_policy": lock_policy}

    if lock_policy == "q0_eig":
        q0_columns = diagonalize_periodic_schur(
            T,
            Z,
            eigvals,
            schur_size,
        )
        real_dtype = jnp.real(q0_columns).dtype
        pair_second = jnp.concatenate([jnp.zeros((1,), dtype=jnp.bool_), is_complex_pair[:-1]])
        col_numer = jnp.einsum(
            "pnc,pnc->pc",
            q0_columns[:, :d_block, :].conj(),
            q0_columns[:, :d_block, :],
        ).real
        col_denom = jnp.einsum("pnc,pnc->pc", q0_columns.conj(), q0_columns).real
        col_weight = jnp.where(col_denom == 0, 0.0, col_numer / col_denom)
        q0_defect = 1.0 - jnp.min(col_weight, axis=0)
        if m > 1:
            pair_numer = col_numer[:, :-1] + col_numer[:, 1:]
            pair_denom = col_denom[:, :-1] + col_denom[:, 1:]
            pair_weight = jnp.where(pair_denom == 0, 0.0, pair_numer / pair_denom)
            pair_defect = 1.0 - jnp.min(pair_weight, axis=0)
            pair_start_defect = jnp.concatenate([pair_defect, jnp.ones((1,), dtype=real_dtype)])
            pair_second_defect = jnp.concatenate([jnp.ones((1,), dtype=real_dtype), pair_defect])
            q0_defect = jnp.where(is_complex_pair, pair_start_defect, q0_defect)
            q0_defect = jnp.where(pair_second, pair_second_defect, q0_defect)
        locked = full_live & (
            q0_defect < cfg.get("ritz_q0_lock_tol", 0.0)
        )
        return locked, {"lock_policy": lock_policy, "q0_defect": q0_defect}

    if lock_policy == "ritz_difference":
        if seed_ritz_values is None:
            raise ValueError("seed_ritz_values are required for lock_policy='ritz_difference'")
        if seed_schur_size is None:
            raise ValueError("seed_schur_size is required for lock_policy='ritz_difference'")
        seed_ritz_values = jnp.asarray(seed_ritz_values)
        seed_live = (
            jnp.arange(seed_ritz_values.shape[0]) < seed_schur_size
        )
        log_floor = cfg.get("ritz_difference_log_floor", 1e-300)
        tau_ratio = cfg.get("ritz_difference_tau_ratio", 0.05)
        eig_cutoff = cfg.get("CTM_eig_cutoff", 1e-15)

        full_abs = jnp.abs(eigvals)
        seed_abs = jnp.abs(seed_ritz_values)
        full_scale = jnp.max(jnp.where(full_live, full_abs, 0.0))
        seed_scale = jnp.max(jnp.where(seed_live, seed_abs, 0.0))
        full_threshold = jnp.maximum(eig_cutoff * full_scale, log_floor)
        seed_threshold = jnp.maximum(eig_cutoff * seed_scale, log_floor)
        full_valid = full_live & (full_abs >= full_threshold)
        seed_valid = seed_live & (seed_abs >= seed_threshold)
        full_safe = jnp.where(full_abs > log_floor, eigvals, log_floor + 0j)
        seed_safe = jnp.where(seed_abs > log_floor, seed_ritz_values, log_floor + 0j)
        distance = jnp.abs(jnp.log(full_safe[None, :] / seed_safe[:, None]))
        distance = jnp.where(seed_valid[:, None] & full_valid[None, :], distance, jnp.inf)
        order = jnp.argsort(distance, axis=1)
        best_idx = order[:, 0]
        best = jnp.take_along_axis(distance, best_idx[:, None], axis=1)[:, 0]
        if m > 1:
            second_idx = order[:, 1]
            second = jnp.take_along_axis(distance, second_idx[:, None], axis=1)[:, 0]
        else:
            second_idx = best_idx
            second = jnp.full_like(best, jnp.inf)
        accepted = seed_valid & jnp.isfinite(second) & (best < tau_ratio * second)

        pair_second = jnp.concatenate([jnp.zeros((1,), dtype=jnp.bool_), is_complex_pair[:-1]])
        best_is_pair_second = jnp.take(pair_second, best_idx)
        best_start = jnp.where(best_is_pair_second, best_idx - 1, best_idx)
        idx = jnp.arange(m)
        locked = jnp.any((idx[None, :] == best_start[:, None]) & accepted[:, None], axis=0)
        return locked, {
            "lock_policy": lock_policy,
            "seed_ritz_values": seed_ritz_values,
            "ritz_difference_full_valid": full_valid,
            "ritz_difference_seed_valid": seed_valid,
            "ritz_difference_distance": distance,
            "ritz_difference_best_index": best_idx,
            "ritz_difference_second_index": second_idx,
            "ritz_difference_best": best,
            "ritz_difference_second": second,
        }

    else:
        raise ValueError(f"unknown lock_policy {lock_policy!r}")


def _q0_X_svals(Q, X, active_cols, ritz_rank, d_block):
    """Return per-site Q0-vs-selected-subspace singular values."""
    Q0 = Q[:, :, :d_block]
    overlap = jnp.einsum("pna,pnb->pab", Q0.conj(), X)
    q0_rank = jnp.sum(active_cols[:, :d_block], axis=1)
    row_keep = jnp.arange(d_block) < q0_rank[:, None]
    col_keep = jnp.arange(X.shape[2]) < ritz_rank
    overlap = jnp.where(row_keep[:, :, None] & col_keep[None, None, :], overlap, 0.0)
    return jnp.linalg.svd(overlap, compute_uv=False)


def selector_policy(
    ritz_abs,
    is_complex_pair,
    locked,
    schur_size,
    d,
    cfg,
):
    """Choose a pair-safe Schur reorder mask from scores.

    The selector ranks only block starts: scalar columns and the first column of
    each real 2-by-2 pair.  Locked blocks are ordered before unlocked blocks;
    ties are then broken by descending ``ritz_abs`` and finally by original
    column index.  Blocks below ``CTM_eig_cutoff`` relative to the largest
    candidate are skipped.

    The output has two masks.  ``select_mask`` has length ``m`` and marks the
    original Schur columns that should be reordered to the front.  ``active`` has
    length ``d`` and marks which output columns are actually filled.  ``active``
    can contain false entries when too few valid modes are available or when a
    real pair would overrun the requested width.

    cfg keys:
      CTM_eig_cutoff: reject modes with |lambda|/|lambda_max| below this.
      split_pair_policy: use "drop_split_pair" or "replace_split_pair".
    """
    ritz_abs = jnp.asarray(ritz_abs)
    is_complex_pair = jnp.asarray(is_complex_pair, dtype=jnp.bool_)
    locked = jnp.asarray(locked, dtype=jnp.bool_)
    m = ritz_abs.shape[0]
    CTM_eig_cutoff = cfg.get("CTM_eig_cutoff", 1e-15)
    split_pair_policy = cfg.get("split_pair_policy", "drop_split_pair")
    if split_pair_policy not in ("drop_split_pair", "replace_split_pair"):
        raise ValueError(f"unknown split_pair_policy {split_pair_policy!r}")

    idx = jnp.arange(m)
    pair_second = jnp.concatenate([jnp.zeros((1,), dtype=jnp.bool_), is_complex_pair[:-1]])
    live = idx < schur_size
    block_start = live & ~pair_second
    max_ritz_abs = jnp.max(jnp.where(block_start, ritz_abs, 0.0))
    eig_valid = live & (
        ritz_abs >= CTM_eig_cutoff * max_ritz_abs
    )
    candidate = block_start & eig_valid
    locked = candidate & locked
    sort_ritz_abs = jnp.where(candidate, ritz_abs, -1.0)
    order = jnp.lexsort((idx, -sort_ritz_abs, -locked.astype(jnp.int32)))
    block_size = jnp.where(is_complex_pair, 2, 1)
    cols = jnp.arange(m)
    out_cols = jnp.arange(d)
    drop_split_pair = split_pair_policy == "drop_split_pair"

    def step(carry, start):
        used, stopped, select_mask, active = carry
        size = block_size[start]
        valid_block = candidate[start] & (used < d) & ~stopped
        fits = used + size <= d
        split_tail_pair = valid_block & (size == 2) & (used + 1 == d)
        take = valid_block & fits
        stopped = stopped | (drop_split_pair & split_tail_pair)
        select_mask = select_mask | (take & (cols >= start) & (cols < start + size))
        active = active | (take & (out_cols >= used) & (out_cols < used + size))
        used = jnp.where(take, used + size, used)
        return (used, stopped, select_mask, active), None

    init = (
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.zeros((m,), dtype=jnp.bool_),
        jnp.zeros((d,), dtype=jnp.bool_),
    )
    (_, _, select_mask, active), _ = jax.lax.scan(step, init, order)
    info = {"order": order}
    return select_mask, active, info


def select_periodic_schur_subspace(
    T,
    Z,
    eigvals,
    schur_size,
    d,
    d_block,
    cfg,
    seed_ritz_values=None,
    seed_schur_size=None,
):
    """Select and reorder a dominant R-oriented periodic Schur subspace.

    Args:
      T: R-oriented periodic Schur triangular factors with shape
        ``(period, m, m)``.
      Z: R-oriented periodic Schur vectors with shape ``(period, m, m)``.
      eigvals: Schur eigenvalues in the same block order as
        ``T`` and ``Z``.
      schur_size: Live leading size returned by the CRed decomposition.
      d: Requested number of Schur-vector columns to keep.
      d_block: Width of the original seed block, used when computing
        lock-policy quantities.
      cfg: Selection options passed to ``selector_policy``.
      seed_ritz_values: Optional Ritz spectrum of the leading seed block,
        required for ``lock_policy="ritz_difference"``.

    Returns:
      A dictionary containing reordered ``T`` and ``Z``, the fixed-width Schur
      vector basis ``X`` with shape ``(period, m, d)``, ``T_selected`` with
      shape ``(period, d, d)``, the output-column ``active`` mask, and the
      original-column ``select_mask``.  With ``return_diagnostics=True`` the
      score arrays and policy-specific lock diagnostics are also included.
    """
    cfg = {} if cfg is None else cfg
    ritz_abs, is_complex_pair = _candidate_scores(
        T,
        eigvals,
        schur_size,
    )
    locked, lock_info = compute_locks(
        T,
        Z,
        eigvals,
        is_complex_pair,
        schur_size,
        d_block,
        cfg,
        seed_ritz_values=seed_ritz_values,
        seed_schur_size=seed_schur_size,
    )
    select_mask, active, _ = selector_policy(
        ritz_abs,
        is_complex_pair,
        locked,
        schur_size,
        d,
        cfg,
    )
    reorder = (
        periodic_schur_ffi.reorder_periodic_schur_Z
        if jnp.iscomplexobj(T)
        else periodic_schur_ffi.reorder_periodic_schur_D
    )
    T_ord, Z_ord = reorder(T, Z, select_mask, schur_size)
    # selector_policy expands real 2-by-2 blocks, so this stable partition
    # matches the block order produced by both package reorder drivers.
    eig_order = jnp.argsort(~select_mask, stable=True)
    eigvals_ord = eigvals[eig_order]
    X = Z_ord[:, :, :d] * active[None, None, :]
    T_selected = T_ord[:, :d, :d] * active[None, :, None] * active[None, None, :]
    selection = {
        "T": T_ord,
        "Z": Z_ord,
        "eigvals": eigvals_ord,
        "X": X,
        "T_selected": T_selected,
        "active": active,
        "select_mask": select_mask,
        "schur_size": schur_size,
    }
    if cfg.get("return_diagnostics", False):
        selection["ritz_abs"] = ritz_abs
        selection["locked"] = locked
        selection.update(lock_info)
    return selection


def periodic_krylov_schur_onesided(
    matvecs: tuple[Matvec, ...],
    Vs: tuple[Array, ...],
    chi_max: int | None = None,
    info_level: int = 0,
    cfg: dict[str, Any] | None = None,
    seed_ritz_values=None,
    seed_schur_size=None,
):
    """Return selected one-sided periodic Krylov-Schur vectors and factors.

    The workflow is:

    1. ``periodic_arnoldi_basis(..., orientation="R")`` builds the large basis
       ``Q`` and small projected factors ``H`` in the relation
       ``matvecs[l](Q[l + 1]) = Q[l] @ H[l] + residual[l]``.
    2. ``linalg.periodic_schur.jax_ffi`` decomposes the small ``H`` factors.
    3. ``select_periodic_schur_subspace`` chooses Schur vectors in the
       projected space from ``T``, ``Z``, and the returned eigenvalues.
    4. The selected vectors are lifted to the original space as
       ``X[p] = Q[p] @ X_projected[p]``.

    Eigenvalue and subspace selection is controlled by ``selector_policy`` and
    its cfg options.

    Args:
      matvecs: R-oriented periodic maps; ``matvecs[l]`` sends site ``l + 1`` to
        site ``l``.
      Vs: Seed blocks, one per site.
      chi_max: Maximum number of Schur-vector columns to select and lift.  If
        ``None``, defaults to the incoming block size ``Vs[0].shape[1]``.
      info_level: Nonnegative info level.  Level 0 returns only the
        always-present info fields; higher levels will add more fields.
      cfg: Optional algorithm and selector options.
      seed_ritz_values: Optional physical incoming cycle eigenvalues for
        ``lock_policy="ritz_difference"``. When supplied, these replace the
        projected H00 seed spectrum.
      seed_schur_size: Leading live size of ``seed_ritz_values``.

    cfg keys:
      num_krylov_iter: number of single-edge block Arnoldi slots including the
        seed block. Defaults to 6 because each periodic slot advances only one
        edge rather than applying a complete cycle.
      max_reortho: maximum reorthogonalization passes.
      eta: Daniel-Gragg-Kaufman-Stewart reorthogonalization threshold.
      rank_tol: QR rank mask scale in units of machine epsilon.
      cred_rank_tol: optional CRed QRP rank scale in machine-epsilon units.
      ortho_method: "MGS" or "CGS" for block orthogonalization.
      pivot: whether Arnoldi rank masking uses QR with column pivoting.
      lock_policy: "ritz_difference", "q0_eig", or "none".
      ritz_q0_lock_tol: for q0_eig, lock modes with q0_defect below this threshold.
      ritz_difference_tau_ratio: for "ritz_difference", lock a full Ritz value
        when the nearest log-distance match is below this ratio times the
        second-nearest log-distance.
      CTM_eig_cutoff: reject modes with |lambda|/|lambda_max| below this.
      split_pair_policy: use "drop_split_pair" or "replace_split_pair".

    Returns:
      ``(X, T_selected, info)``.  ``X`` has shape
      ``(period, N, chi_max)`` and is zero in inactive columns.
      ``T_selected`` has shape ``(period, chi_max, chi_max)`` and contains the
      matching R-oriented periodic-Schur factors. ``info`` always contains
      ``ritz_rank``, ``schur_size``, ``ritz_values``, ``ritz_values_kept``,
      ``seed_ritz_values``, and ``seed_schur_size``. ``ritz_values_kept`` has
      length ``chi_max`` and is zero in inactive columns. ``seed_ritz_values``
      is the supplied incoming spectrum, or the H00 Ritz spectrum when no
      physical spectrum was supplied. It is zero for policies that do not use
      incoming seed Ritz values. For
      ``info_level >= 1``, ``info`` also contains ``initial_ortho_res``,
      ``q0_rank``, and ``q0_X_svals``.
    """
    if info_level < 0:
        raise ValueError("info_level must be nonnegative")
    cfg = {} if cfg is None else cfg
    if chi_max is None:
        chi_max = Vs[0].shape[1]
    Q, H, residual, _, active_cols = periodic_arnoldi_basis(
        matvecs,
        Vs,
        cfg.get("num_krylov_iter", DEFAULT_PERIODIC_KRYLOV_ITER),
        max_reortho=cfg.get("max_reortho", 3),
        eta=cfg.get("eta", 1.0 / np.sqrt(2.0)),
        rank_tol=cfg.get("rank_tol", 4.0),
        ortho_method=cfg.get("ortho_method", "MGS"),
        pivot=cfg.get("pivot", True),
        orientation="R",
    )

    d_block = Vs[0].shape[1]
    cred_kwargs = {}
    if cfg.get("cred_rank_tol", None) is not None:
        cred_kwargs["rank_tol"] = cfg["cred_rank_tol"]
    supplied_seed_ritz_values = seed_ritz_values
    supplied_seed_schur_size = seed_schur_size
    seed_ritz_values = jnp.zeros((d_block,), dtype=jnp.complex128)
    seed_schur_size = jnp.array(0, dtype=jnp.int32)
    if jnp.iscomplexobj(H):
        T, Z, alpha, beta, scale, schur_size = (
            periodic_schur_ffi.periodic_schur_Z(
                H,
                active_cols,
                reduction="CRed",
                **cred_kwargs,
            )
        )
        ritz_values = alpha / beta * jnp.exp2(scale)
    else:
        T, Z, wr, wi, schur_size = (
            periodic_schur_ffi.periodic_schur_D(
                H,
                active_cols,
                reduction="CRed",
                schur_deflation_tol=10.0,
                **cred_kwargs,
            )
        )
        ritz_values = wr + 1j * wi

    seed_ritz_values_for_selection = None
    if cfg.get("lock_policy", "ritz_difference") == "ritz_difference":
        if supplied_seed_ritz_values is not None:
            if supplied_seed_schur_size is None:
                raise ValueError(
                    "seed_schur_size is required with seed_ritz_values"
                )
            seed_ritz_values_for_selection = jnp.asarray(
                supplied_seed_ritz_values
            )
            seed_schur_size = jnp.asarray(
                supplied_seed_schur_size,
                dtype=jnp.int32,
            )
        else:
            H_seed = H[:, :d_block, :d_block]
            active_seed = active_cols[:, :d_block]
            if jnp.iscomplexobj(H_seed):
                _, _, alpha, beta, scale, seed_schur_size = (
                    periodic_schur_ffi.periodic_schur_Z(
                        H_seed,
                        active_seed,
                        reduction="CRed",
                        **cred_kwargs,
                    )
                )
                seed_ritz_values_for_selection = (
                    alpha / beta * jnp.exp2(scale)
                )
            else:
                _, _, wr, wi, seed_schur_size = (
                    periodic_schur_ffi.periodic_schur_D(
                        H_seed,
                        active_seed,
                        reduction="CRed",
                        schur_deflation_tol=10.0,
                        **cred_kwargs,
                    )
                )
                seed_ritz_values_for_selection = wr + 1j * wi
        seed_ritz_values = seed_ritz_values_for_selection
    selection = select_periodic_schur_subspace(
        T,
        Z,
        ritz_values,
        schur_size,
        chi_max,
        d_block,
        cfg,
        seed_ritz_values=seed_ritz_values_for_selection,
        seed_schur_size=seed_schur_size,
    )
    X_projected = selection["X"]
    T_selected = selection["T_selected"]
    ritz_values_kept = selection["eigvals"][:chi_max]
    active = selection["active"]
    ritz_values_kept = ritz_values_kept * active.astype(ritz_values_kept.dtype)
    X = jnp.einsum("pnm,pmd->pnd", Q, X_projected)
    info = {
        "ritz_rank": jnp.sum(active),
        "ritz_values": ritz_values,
        "ritz_values_kept": ritz_values_kept,
        "seed_ritz_values": seed_ritz_values,
        "schur_size": schur_size,
        "seed_schur_size": seed_schur_size,
    }
    if info_level >= 1:
        q0_rank = jnp.sum(active_cols[:, :d_block], axis=1)
        q0_X_svals = _q0_X_svals(Q, X, active_cols, info["ritz_rank"], d_block)
        info["initial_ortho_res"] = jnp.linalg.norm(H[:, d_block : 2 * d_block, :d_block])
        info["q0_rank"] = q0_rank
        info["q0_X_svals"] = q0_X_svals
    return X, T_selected, info


def krylov_schur_twosided(
    matvecs,
    vecmats,
    VRs,
    VLs,
    chi_max=None,
    info_level=0,
    cfg=None,
):
    """Build the initial right and left periodic Arnoldi data.

    This is the first-stage scaffold for a two-sided periodic Krylov-Schur
    routine.  It independently runs ``periodic_arnoldi_basis`` on the supplied
    right problem ``(matvecs, VRs)`` and left problem ``(vecmats, VLs)``. 

    
    """
    if info_level < 0:
        raise ValueError("info_level must be nonnegative")
    cfg = {} if cfg is None else cfg
    if chi_max is None:
        chi_max = VRs[0].shape[1]

    arnoldi_kwargs = {
        "num_krylov_iter": cfg.get(
            "num_krylov_iter",
            DEFAULT_PERIODIC_KRYLOV_ITER,
        ),
        "max_reortho": cfg.get("max_reortho", 3),
        "eta": cfg.get("eta", 1.0 / np.sqrt(2.0)),
        "rank_tol": cfg.get("rank_tol", 4.0),
        "ortho_method": cfg.get("ortho_method", "MGS"),
        "full_res": True,
        "pivot": cfg.get("pivot", True),
    }
    QR, HR, resR, _, activeR = periodic_arnoldi_basis(
        matvecs,
        VRs,
        **arnoldi_kwargs,
    )

    vecmats_forward = vecmats[::-1]
    VLs_forward = (VLs[0],) + VLs[:0:-1]

    QL, HL, resL, _, activeL = periodic_arnoldi_basis(
        vecmats_forward,
        VLs_forward,
        **arnoldi_kwargs,
    )

    #We want VL[j+1]^T A[j] = VL[j]^T
    # A[j]^T VL[j+1] = VL[j]

    #At this  A0^T  A1^T  A_{p-1}^T  


    #        A_{p-1}^T QL[0] = QL[1].HL[0] + resL[0]
    #        A_{p-2}^T QL[1] = QL[2].HL[1] + resL[1]...
    #        A_{p-1-j}^T QL[j] =  QL[j+1].HL[j] + resL[j]

    # Now reverse
    QL = jnp.roll(QL[::-1], shift=1, axis=0)
    activeL = jnp.roll(activeL[::-1], shift=1, axis=0)
    HL = HL[::-1, :, :]
    resL = jnp.roll(resL[::-1], shift=1, axis=0)
    
    A, B, QR, QL = two_sided_projected_system(QR, HR, resR, activeR , QL, activeL)


    # At this point we have cyclic product
    # Tr(... A[1] B[1]^-1 A[0] B[0]^-1).  The signed periodic Schur path wants
    # factors ordered as [A[p-1], B[p-1], A[p-2], B[p-2], ...] with signs
    # [1, -1, 1, -1, ...].
    schur_factors = []
    schur_signs = []
    for j in range(len(A) - 1, -1, -1):
        schur_factors.extend([A[j], B[j]])
        schur_signs.extend([1, -1])

    info = {
        "chi_max": chi_max,
        "HR": HR,
        "HL": HL,
        "resR": resR,
        "resL": resL,
        "active_cols_R": activeR,
        "active_cols_L": activeL,
        "projected_A": A,
        "projected_B": B,
        "schur_factors": schur_factors,
        "schur_signs": np.asarray(schur_signs, dtype=np.int32),
    }
    return QR, QL, info

def two_sided_projected_system(QR, HR, resR, activeR , QL, activeL):
    """Given left/ right arnoldi systems, form the projected system
    
        A[j] <--- QL^T[j+1] matvec[j] QR[j]
        B[j] <--- QL^T[j] QR[j]

        so that 

         Tr(prod[j] matvec[j]) approx Tr(... A[1] B[1]^-1   A[0] B[0]^-1 )

    """
    p = QR.shape[0]
    activeR = np.asarray(activeR, dtype=bool)
    activeL = np.asarray(activeL, dtype=bool)
    QR = [np.asarray(QR[j])[:, activeR[j]] for j in range(p)]
    resR = [np.asarray(resR[j])[:, activeR[j]] for j in range(p)]
    HR = [np.asarray(HR[j])[:, activeR[j]] for j in range(p)]
    HR = [HR[j][activeR[(j+1)%p], :] for j in range(p)]
    QL = [np.asarray(QL[j])[:, activeL[j]] for j in range(p)]

    B = [np.dot(QL[j].T, QR[j]) for j in range(p)]

    #A[j] ---> QL^T[j+1] A[j] QR[j] = QL^T[j+1] ( QR[j+1] HR[j] + resR[j])
    #                               = B[j+1] HR[j] + QL^T[j+1] resR[j]

    A = [np.dot(B[(j+1)%p], HR[j]) + np.dot(QL[(j+1)%p].T, resR[j]) for j in range(p)]

    return A, B, QR, QL



def periodic_krylov_schur_projectors(
    matvecs,
    vecmats,
    VRs,
    VLs,
    chi_max=None,
    info_level=0,
    cfg=None,
    seed_ritz_values=None,
    seed_schur_size=None,
):
    """Build R-oriented Krylov-Schur projectors and reduced Schur factors.

    Input site labels follow the SLICOT R convention. ``matvecs[j]`` is
    ``A_j: j + 1 -> j``. ``vecmats[j]`` applies ``A_j.T`` from site ``j`` to
    site ``j + 1``. ``VRs[j]`` and ``VLs[j]`` both live at physical site
    ``j``, so the input seed projector is ``VRs[j] @ VLs[j].T``.

    ``periodic_krylov_schur_onesided`` uses the same R convention, so the right
    problem passes directly to it.  The transpose problem has the opposite
    orientation and is reversed before its one-sided solve, then restored to
    physical site labels.

    Returned ``XR[p]`` and ``XL[p]`` both live at physical site ``p``.  Each pair
    is passed to ``biorthogonalize_bases``, which whitens the bilinear overlap
    ``XL[p].T @ XR[p]`` and uses a QL step so the applied right gauge remains
    upper triangular.  This preserves the right Schur form.  At full overlap
    rank the pairing is the identity; at deficient rank it is the retained-rank
    projector.  The resulting projector at site ``p`` is
    ``P[p] = XR[p] @ XL[p].T``.

    If ``T[p]`` is the selected right periodic-Schur factor and ``G_R[p]`` is
    the upper-triangular right whitening gauge, the returned reduced factor is

    ``c[p] = pinv(G_R[p]) @ T[p] @ G_R[p + 1]``.

    ``triangular_pinv_solve`` applies the inverse only on the common retained
    leading rank.  The lifted Arnoldi residual is deliberately omitted, so
    ``c[p]`` is the Schur-gauge approximation to
    ``XL[p].T @ matvecs[p](XR[p + 1])`` rather than its exact finite-Krylov
    projection.

    If the retained right and left Ritz ranks differ, a warning is issued and
    the kept right/left Ritz values are printed.

    ``seed_ritz_values`` and ``seed_schur_size`` optionally supply the physical
    incoming cycle spectrum used by ``lock_policy="ritz_difference"``. The same
    spectrum is used for the right and reversed-transpose left problems.

    All algorithmic options are supplied through the single ``cfg`` dictionary.
    ``chi_max`` and ``info_level`` are direct keyword arguments; ``cfg`` is
    frozen for JIT and forwarded unchanged to both the right and left
    ``periodic_krylov_schur_onesided`` calls.  Returns ``(XR, XL, c, info)``.
    """
    if info_level < 0:
        raise ValueError("info_level must be nonnegative")
    cfg = {} if cfg is None else cfg
    if chi_max is None:
        chi_max = VRs[0].shape[1]
    return _periodic_krylov_schur_projectors_jit(
        tuple(matvecs),
        tuple(vecmats),
        tuple(VRs),
        tuple(VLs),
        seed_ritz_values,
        seed_schur_size,
        chi_max,
        info_level,
        _freeze_static_cfg(cfg),
    )


@partial(
    jax.jit,
    static_argnames=("matvecs", "vecmats", "chi_max", "info_level", "cfg_items"),
)
def _periodic_krylov_schur_projectors_jit(
    matvecs: tuple[Matvec, ...],
    vecmats: tuple[Matvec, ...],
    VRs: tuple[Array, ...],
    VLs: tuple[Array, ...],
    seed_ritz_values,
    seed_schur_size,
    chi_max: int,
    info_level: int,
    cfg_items: tuple[tuple[str, Any], ...],
):
    """Compiled implementation for ``periodic_krylov_schur_projectors``."""
    cfg = dict(cfg_items)
    period = len(VRs)

    # The public right problem already uses the one-sided/SLICOT convention
    # A[p](Q[p+1]) -> Q[p].  Its transpose maps Q[p] -> Q[p+1], so reverse only
    # the left problem into R order and restore physical labels afterward.
    site_order = tuple((-r) % period for r in range(period))
    edge_order = tuple((-r - 1) % period for r in range(period))
    vecmats_R = tuple(vecmats[p] for p in edge_order)
    VLs_R = tuple(VLs[p] for p in site_order)

    XR, T_R, info_R = periodic_krylov_schur_onesided(
        matvecs,
        VRs,
        chi_max=chi_max,
        info_level=info_level,
        cfg=cfg,
        seed_ritz_values=seed_ritz_values,
        seed_schur_size=seed_schur_size,
    )
    XL_R, _, info_L = periodic_krylov_schur_onesided(
        vecmats_R,
        VLs_R,
        chi_max=chi_max,
        info_level=info_level,
        cfg=cfg,
        seed_ritz_values=seed_ritz_values,
        seed_schur_size=seed_schur_size,
    )
    XL = jnp.take(XL_R, jnp.asarray(site_order), axis=0)
    if info_level >= 1:
        info_L = dict(info_L)
        info_L["q0_X_svals"] = jnp.take(
            info_L["q0_X_svals"],
            jnp.asarray(site_order),
            axis=0,
        )
        info_L["q0_rank"] = jnp.take(
            info_L["q0_rank"],
            jnp.asarray(site_order),
            axis=0,
        )
    rank_R = info_R["ritz_rank"]
    rank_L = info_L["ritz_rank"]

    def print_rank_mismatch(args):
        rank_R, rank_L, ritz_R, ritz_L = args
        jax.debug.print(
            "right and left Krylov-Schur ranks differ: right={} left={}",
            rank_R,
            rank_L,
        )
        jax.debug.print("right ritz_values_kept: {}", ritz_R)
        jax.debug.print("left ritz_values_kept: {}", ritz_L)
        return ()

    def skip_rank_mismatch(args):
        return ()

    _ = jax.lax.cond(
        rank_R != rank_L,
        print_rank_mismatch,
        skip_rank_mismatch,
        (rank_R, rank_L, info_R["ritz_values_kept"], info_L["ritz_values_kept"]),
    )

    # Remove any unpaired trailing Ritz directions before whitening.  In a
    # non-normal problem, allowing the overlap SVD to see the larger flag can
    # rotate its unmatched direction into the otherwise paired subspace.
    selected_rank = jnp.minimum(rank_R, rank_L)
    selected = jnp.arange(chi_max) < selected_rank
    XR = jnp.where(selected[None, None, :], XR, 0)
    XL = jnp.where(selected[None, None, :], XL, 0)

    # vecmats are A.T maps, so pass XL[p].T with no conjugation.  The helper
    # balances the pair while keeping the right Schur gauge upper triangular.
    XR, XL_T, G_R, _, _, s, overlap_rank_by_site = jax.vmap(
        biorthogonalize_bases,
        in_axes=(0, 0, None),
    )(
        XR,
        jnp.swapaxes(XL, -1, -2),
        cfg.get("biorthogonalize_tol", 10.0),
    )
    XL = jnp.swapaxes(XL_T, -1, -2)
    overlap_rank = jnp.min(overlap_rank_by_site).astype(selected_rank.dtype)
    rank = jnp.minimum(selected_rank, overlap_rank)
    active = jnp.arange(chi_max) < rank
    active_block = active[:, None] & active[None, :]
    T_R = jnp.where(active_block[None, :, :], T_R, 0)
    G_R_next = jnp.roll(G_R, -1, axis=0)
    G_R_next = jnp.where(active_block[None, :, :], G_R_next, 0)
    c = triangular_pinv_solve(
        G_R,
        T_R @ G_R_next,
        rank,
        left_side=True,
    )
    # T_R[0] is the fixed quasi-triangular factor; the upper-triangular G_R
    # gauges preserve that placement in c.

    if info_level >= 1:
        def print_biorthogonal_rank_drop(args):
            """Print overlap singular values when biorthogonal rank drops."""
            selected_rank, overlap_rank_by_site, rank, s = args
            jax.debug.print(
                "periodic biorthogonal overlap rank drops: selected={} "
                "overlap ranks={} final={} singular values={}",
                selected_rank,
                overlap_rank_by_site,
                rank,
                s,
                ordered=True,
            )
            return ()

        def skip_biorthogonal_rank_drop(args):
            """Do nothing when the overlap rank matches the selected rank."""
            return ()

        _ = jax.lax.cond(
            rank < selected_rank,
            print_biorthogonal_rank_drop,
            skip_biorthogonal_rank_drop,
            (selected_rank, overlap_rank_by_site, rank, s),
        )

    info = {
        "right": info_R,
        "left": info_L,
        "ritz_rank_R": rank_R,
        "ritz_rank_L": rank_L,
        "overlap_singular_values": s,
        "overlap_rank_by_site": overlap_rank_by_site,
        "biorthogonal_rank": rank.astype(info_R["ritz_rank"].dtype),
    }
    if info_level >= 1:
        info.update({
            "seed_R_svals": info_R["q0_X_svals"],
            "seed_L_svals": info_L["q0_X_svals"],
            "seed_R_rank": info_R["q0_rank"],
            "seed_L_rank": info_L["q0_rank"],
        })
    return XR, XL, c, info
