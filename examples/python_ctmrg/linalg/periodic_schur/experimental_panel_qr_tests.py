"""Archived checks for the unbuilt block-panel periodic QR experiment."""

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _block_hessenberg_factors(rng, period, depth, d_block, dtype):
    """Return dense storage whose block entries vanish below one subdiagonal."""
    m = depth * d_block
    H = np.zeros((period, m, m), dtype=dtype)
    for k in range(period):
        for j in range(depth):
            row_stop = min(depth, j + 2) * d_block
            values = rng.standard_normal((row_stop, d_block))
            if dtype == np.complex128:
                values = values + 1j * rng.standard_normal(values.shape)
            H[k, :row_stop, j * d_block:(j + 1) * d_block] = values
    return H


@pytest.mark.parametrize(
    ("dtype", "compact_name", "full_name", "panel_name"),
    [
        (np.float64, "compact_active_slicot_d", "_full_qr_sweep_d", "_panel_qr_sweep_d"),
        (np.complex128, "compact_active_slicot_z", "_full_qr_sweep_z", "_panel_qr_sweep_z"),
    ],
)
def test_panel_qr_matches_full_sweep_invariants(
    dtype,
    compact_name,
    full_name,
    panel_name,
):
    """Panel and full QR preserve the same local maps and triangular structure."""
    from linalg.periodic_schur import _periodic_schur

    rng = np.random.default_rng(184)
    period = 4
    depth = 3
    d_block = 3
    m = depth * d_block
    H = _block_hessenberg_factors(rng, period, depth, d_block, dtype)
    active = np.asarray(
        [
            [1, 1, 0, 0, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=bool,
    )
    packed = getattr(_periodic_schur, compact_name)(H, active, d_block)
    old_F, old_U, old_ranks, old_block_ranks, cut_offset = packed
    assert cut_offset == 0
    old_factors = [
        old_F[:old_ranks[k], :old_ranks[(k + 1) % period], k].copy()
        for k in range(period)
    ]

    full_F = old_F.copy(order="F")
    full_U = old_U.copy(order="F")
    full_ranks = old_ranks.copy()
    getattr(_periodic_schur, full_name)(full_F, full_U, full_ranks)

    F = old_F.copy(order="F")
    U = old_U.copy(order="F")
    ranks = old_ranks.copy()
    block_ranks = old_block_ranks.copy()
    result = getattr(_periodic_schur, panel_name)(F, U, ranks, block_ranks)

    assert result[0] is F
    assert result[1] is U
    assert result[2] is ranks
    assert result[3] is block_ranks
    n = old_ranks[0]
    np.testing.assert_array_equal(ranks, np.full(period, n))
    np.testing.assert_array_equal(
        block_ranks,
        np.repeat(old_block_ranks[0:1], period, axis=0),
    )
    np.testing.assert_allclose(F[n:, :, :], 0, atol=0)
    np.testing.assert_allclose(F[:, n:, :], 0, atol=0)
    np.testing.assert_allclose(U[:, n:, :], 0, atol=0)

    gauges = []
    for k in range(period):
        old_basis = old_U[:, :old_ranks[k], k]
        new_basis = U[:, :n, k]
        gauges.append(old_basis.conj().T @ new_basis)
        np.testing.assert_allclose(new_basis.conj().T @ new_basis, np.eye(n), atol=3e-12)
        if k:
            np.testing.assert_allclose(np.tril(F[:n, :n, k], -1), 0, atol=3e-12)

    for k in range(period):
        kp = (k + 1) % period
        np.testing.assert_allclose(
            old_factors[k] @ gauges[kp],
            gauges[k] @ F[:n, :n, k],
            atol=8e-12,
        )

    panel_product = F[:n, :n, 0]
    full_product = full_F[:n, :n, 0]
    for k in range(1, period):
        panel_product = panel_product @ F[:n, :n, k]
        full_product = full_product @ full_F[:n, :n, k]
    np.testing.assert_allclose(panel_product, full_product, atol=1e-11)


def test_panel_qr_rejects_block_metadata_inconsistent_with_ranks():
    """Panel dimensions must describe every current cut exactly."""
    from linalg.periodic_schur import _periodic_schur

    factors = np.zeros((4, 4, 2), dtype=np.float64, order="F")
    bases = np.zeros((4, 4, 2), dtype=np.float64, order="F")
    ranks = np.asarray([3, 4], dtype=np.intp)
    block_ranks = np.asarray([[2, 1], [2, 1]], dtype=np.intp)
    with pytest.raises(ValueError, match="sum to its cut rank"):
        _periodic_schur._panel_qr_sweep_d(factors, bases, ranks, block_ranks)


def test_panel_qr_bandwidth_growth_with_random_compacted_blocks():
    """Growing panel support remains exact across periods and active patterns."""
    from linalg.periodic_schur import _periodic_schur

    rng = np.random.default_rng(608)
    for period, depth in ((2, 4), (3, 3), (5, 4)):
        d_block = 3
        m = depth * d_block
        for _ in range(6):
            H = _block_hessenberg_factors(
                rng, period, depth, d_block, np.float64
            )
            active = rng.random((period, m)) > 0.28
            active[:, 0] = True
            F, U, ranks, block_ranks, _ = (
                _periodic_schur.compact_active_slicot_d(H, active, d_block)
            )
            old_F = F.copy(order="F")
            old_U = U.copy(order="F")
            old_ranks = ranks.copy()

            _periodic_schur._panel_qr_sweep_d(F, U, ranks, block_ranks)
            n = ranks[0]
            gauges = [
                old_U[:, :old_ranks[k], k].T @ U[:, :n, k]
                for k in range(period)
            ]
            for k in range(period):
                kp = (k + 1) % period
                old_factor = old_F[
                    :old_ranks[k], :old_ranks[kp], k
                ]
                np.testing.assert_allclose(
                    old_factor @ gauges[kp],
                    gauges[k] @ F[:n, :n, k],
                    atol=2e-11,
                )


@pytest.mark.parametrize(
    ("dtype", "schur_name"),
    [
        (np.float64, "_slicot_active_schur_d"),
        (np.complex128, "_slicot_active_schur_z"),
    ],
)
def test_active_slicot_pipeline_swaps_full_and_panel_qr(dtype, schur_name):
    """Both QR implementations feed the common chase and direct SLICOT call."""
    from linalg.periodic_schur import slicot_interface

    rng = np.random.default_rng(923)
    period = 3
    depth = 3
    d_block = 2
    H = _block_hessenberg_factors(rng, period, depth, d_block, dtype)
    active = np.asarray(
        [
            [True, True, True, False, True, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )
    schur = getattr(slicot_interface, schur_name)
    full_T, full_Z, full_eigvals = schur(H, active, d_block, qr_sweep="full")
    panel_T, panel_Z, panel_eigvals = schur(H, active, d_block, qr_sweep="panel")

    np.testing.assert_allclose(
        np.sort_complex(panel_eigvals),
        np.sort_complex(full_eigvals),
        atol=3e-9,
        rtol=3e-9,
    )
    n = full_eigvals.size
    assert full_T.shape == panel_T.shape == (period, n, n)
    assert full_Z.shape == panel_Z.shape == (period, H.shape[1], n)
    for k in range(period):
        kp = (k + 1) % period
        rows = np.flatnonzero(active[k])
        cols = np.flatnonzero(active[kp])
        np.testing.assert_allclose(
            H[k][np.ix_(rows, cols)] @ panel_Z[kp, cols, :],
            panel_Z[k, rows, :] @ panel_T[k],
            atol=3e-10,
            rtol=3e-10,
        )
