"""Archived checks for the unbuilt fused Householder experiments."""

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _random_factors(rng, period, m, dtype):
    """Return one well-scaled dense factor stack in formal SLICOT order."""
    factors = rng.standard_normal((period, m, m))
    if dtype == np.complex128:
        factors = factors + 1j * rng.standard_normal(factors.shape)
    return factors.astype(dtype)


def _triangular_block_hessenberg_factors(rng, period, depth, d_block, dtype):
    """Return block-Hessenberg factors with triangular subdiagonal blocks."""
    m = depth * d_block
    factors = np.zeros((period, m, m), dtype=dtype)
    for k in range(period):
        for block in range(depth):
            col = slice(block * d_block, (block + 1) * d_block)
            top = (block + 1) * d_block
            values = rng.standard_normal((top, d_block))
            if dtype == np.complex128:
                values = values + 1j * rng.standard_normal(values.shape)
            factors[k, :top, col] = values
            if block + 1 < depth:
                values = rng.standard_normal((d_block, d_block))
                if dtype == np.complex128:
                    values = values + 1j * rng.standard_normal(values.shape)
                row = slice(top, top + d_block)
                factors[k, row, col] = np.triu(values)
    return factors


def _active_pattern(equal):
    """Return equal- or unequal-rank active cuts with cut zero minimal."""
    if equal:
        return np.asarray(
            [
                [1, 1, 0, 1, 1, 0, 1, 1],
                [1, 0, 1, 1, 0, 1, 1, 1],
                [0, 1, 1, 0, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
    return np.asarray(
        [
            [1, 1, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
        ],
        dtype=bool,
    )


def _factor_num_rows(factors, ranks):
    """Return exact exclusive row bounds for compact logical columns."""
    capacity, _, period = factors.shape
    num_rows = np.zeros((capacity, period), dtype=np.intp)
    for k in range(period):
        kp = (k + 1) % period
        for col in range(ranks[kp]):
            nonzero = np.flatnonzero(factors[: ranks[k], col, k] != 0)
            num_rows[col, k] = nonzero[-1] + 1 if nonzero.size else 0
    return num_rows


def _logical_factors(factors, ranks):
    """Copy the live rectangular factors from trailing-period storage."""
    period = len(ranks)
    return [
        factors[: ranks[k], : ranks[(k + 1) % period], k].copy()
        for k in range(period)
    ]


def _formal_product(factors):
    """Form the formal SLICOT-order product ``C[0] ... C[p-1]``."""
    product = factors[0]
    for factor in factors[1:]:
        product = product @ factor
    return product


def _sorted_eigvals(factors):
    """Return deterministically sorted product eigenvalues."""
    return np.sort_complex(np.linalg.eigvals(_formal_product(factors)))


def _assert_preparation(old_F, old_U, old_ranks, F, U, ranks, atol):
    """Check structure, orthonormal bases, and every compact local relation."""
    period = len(ranks)
    n = int(ranks[0])
    old_factors = _logical_factors(old_F, old_ranks)
    np.testing.assert_array_equal(ranks, np.full(period, n))
    np.testing.assert_allclose(np.tril(F[:n, :n, 0], -2), 0.0, atol=atol)
    for k in range(1, period):
        np.testing.assert_allclose(np.tril(F[:n, :n, k], -1), 0.0, atol=atol)

    gauges = []
    for k in range(period):
        old_basis = old_U[:, : old_ranks[k], k]
        new_basis = U[:, :n, k]
        gauges.append(old_basis.conj().T @ new_basis)
        np.testing.assert_allclose(
            new_basis.conj().T @ new_basis,
            np.eye(n),
            atol=atol,
        )

    for k in range(period):
        kp = (k + 1) % period
        np.testing.assert_allclose(
            old_factors[k] @ gauges[kp],
            gauges[k] @ F[:n, :n, k],
            atol=4 * atol,
            rtol=4 * atol,
        )


@pytest.mark.parametrize(
    ("dtype", "compact_name", "householder_name", "qr_name", "chase_name"),
    [
        (
            np.float64,
            "compact_active_slicot_d",
            "make_periodic_hessenberg_HOUSEHOLDER_D",
            "_full_qr_sweep_d",
            "_hessenberg_chase_d",
        ),
        (
            np.complex128,
            "compact_active_slicot_z",
            "make_periodic_hessenberg_HOUSEHOLDER_Z",
            "_full_qr_sweep_z",
            "_hessenberg_chase_z",
        ),
    ],
)
@pytest.mark.parametrize("equal", [True, False], ids=["equal-ranks", "unequal-ranks"])
def test_fused_householder_matches_givens_invariants(
    dtype,
    compact_name,
    householder_name,
    qr_name,
    chase_name,
    equal,
):
    """D/Z fused and full-QR-plus-Givens paths preserve the same live problem."""
    from linalg.periodic_schur import _periodic_schur

    rng = np.random.default_rng(2401 + int(equal))
    H = _random_factors(rng, period=3, m=8, dtype=dtype)
    active = _active_pattern(equal)
    packed = getattr(_periodic_schur, compact_name)(H, active, 2)
    F0, U0, ranks0, block_ranks0, cut_offset = packed
    assert cut_offset == 0

    F = F0.copy(order="F")
    U = U0.copy(order="F")
    ranks = ranks0.copy()
    block_ranks = block_ranks0.copy()
    result = getattr(_periodic_schur, householder_name)(
        F,
        U,
        ranks,
        block_ranks,
    )

    assert result[0] is F
    assert result[1] is U
    assert result[2] is ranks
    assert result[3] is block_ranks
    np.testing.assert_array_equal(block_ranks, block_ranks0)
    atol = 4e-12 if dtype == np.float64 else 7e-12
    _assert_preparation(F0, U0, ranks0, F, U, ranks, atol)

    F_givens = F0.copy(order="F")
    U_givens = U0.copy(order="F")
    ranks_givens = ranks0.copy()
    getattr(_periodic_schur, qr_name)(F_givens, U_givens, ranks_givens)
    getattr(_periodic_schur, chase_name)(F_givens, U_givens, ranks_givens)
    _assert_preparation(
        F0,
        U0,
        ranks0,
        F_givens,
        U_givens,
        ranks_givens,
        atol,
    )

    n = int(ranks[0])
    householder_factors = [F[:n, :n, k] for k in range(len(ranks))]
    givens_factors = [F_givens[:n, :n, k] for k in range(len(ranks))]
    original_factors = _logical_factors(F0, ranks0)
    np.testing.assert_allclose(
        _sorted_eigvals(householder_factors),
        _sorted_eigvals(original_factors),
        atol=2e-9,
        rtol=2e-9,
    )
    np.testing.assert_allclose(
        _sorted_eigvals(householder_factors),
        _sorted_eigvals(givens_factors),
        atol=2e-9,
        rtol=2e-9,
    )


def test_real_fused_householder_matches_mb03vd_mb03vy():
    """The equal-rank real sweep agrees directly with MB03VD and MB03VY."""
    from linalg.periodic_schur import _periodic_schur
    from linalg.periodic_schur.slicot_interface import _import_slicot_periodic

    rng = np.random.default_rng(2403)
    period = 4
    n = 7
    H = _random_factors(rng, period, n, np.float64)
    active = np.ones((period, n), dtype=bool)
    F0, U0, ranks, block_ranks, _ = _periodic_schur.compact_active_slicot_d(
        H,
        active,
        1,
    )

    F = F0.copy(order="F")
    U = U0.copy(order="F")
    _periodic_schur.make_periodic_hessenberg_HOUSEHOLDER_D(
        F,
        U,
        ranks.copy(),
        block_ranks.copy(),
    )

    mb03 = _import_slicot_periodic()
    ldtau = max(1, n - 1)
    tau = np.zeros((ldtau, period), dtype=np.float64, order="F")
    A, tau, _, info = mb03.mb03vd(
        1,
        n,
        F0.copy(order="F"),
        tau,
        np.empty(max(1, n), dtype=np.float64),
        n=n,
        p=period,
        lda1=n,
        lda2=n,
        ldtau=ldtau,
    )
    assert info == 0
    Q, _, info = mb03.mb03vy(
        n,
        1,
        n,
        np.array(A, copy=True, order="F"),
        tau,
        np.empty(max(1, 8 * n), dtype=np.float64),
        p=period,
        lda1=n,
        lda2=n,
        ldtau=ldtau,
        ldwork=max(1, 8 * n),
    )
    assert info == 0
    A[:, :, 0] = np.triu(A[:, :, 0], -1)
    for k in range(1, period):
        A[:, :, k] = np.triu(A[:, :, k])

    np.testing.assert_allclose(F, A, atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(U, Q, atol=5e-13, rtol=5e-13)


@pytest.mark.parametrize(
    ("dtype", "compact_name", "general_name", "num_rows_name"),
    [
        (
            np.float64,
            "compact_active_slicot_d",
            "make_periodic_hessenberg_HOUSEHOLDER_D",
            "make_periodic_hessenberg_HOUSEHOLDER_NUM_ROWS_D",
        ),
        (
            np.complex128,
            "compact_active_slicot_z",
            "make_periodic_hessenberg_HOUSEHOLDER_Z",
            "make_periodic_hessenberg_HOUSEHOLDER_NUM_ROWS_Z",
        ),
    ],
)
@pytest.mark.parametrize("equal", [True, False], ids=["equal-ranks", "unequal-ranks"])
def test_num_rows_householder_matches_general_fused_reduction(
    dtype,
    compact_name,
    general_name,
    num_rows_name,
    equal,
):
    """Structural D/Z sweeps track fill and preserve the general-kernel result."""
    from linalg.periodic_schur import _periodic_schur

    rng = np.random.default_rng(2411 + int(equal))
    H = _triangular_block_hessenberg_factors(
        rng,
        period=3,
        depth=4,
        d_block=2,
        dtype=dtype,
    )
    packed = getattr(_periodic_schur, compact_name)(
        H,
        _active_pattern(equal),
        2,
    )
    F0, U0, ranks0, block_ranks0, cut_offset = packed
    assert cut_offset == 0
    num_rows0 = _factor_num_rows(F0, ranks0)

    F_general = F0.copy(order="F")
    U_general = U0.copy(order="F")
    ranks_general = ranks0.copy()
    getattr(_periodic_schur, general_name)(
        F_general,
        U_general,
        ranks_general,
        block_ranks0.copy(),
    )

    F = F0.copy(order="F")
    U = U0.copy(order="F")
    ranks = ranks0.copy()
    block_ranks = block_ranks0.copy()
    num_rows = num_rows0.copy()
    result = getattr(_periodic_schur, num_rows_name)(
        F,
        U,
        ranks,
        block_ranks,
        num_rows,
    )

    assert result[0] is F
    assert result[1] is U
    assert result[2] is ranks
    assert result[3] is block_ranks
    assert result[4] is num_rows
    np.testing.assert_array_equal(block_ranks, block_ranks0)
    atol = 8e-12 if dtype == np.float64 else 1.2e-11
    _assert_preparation(F0, U0, ranks0, F, U, ranks, atol)
    np.testing.assert_allclose(F, F_general, atol=atol, rtol=atol)
    np.testing.assert_allclose(U, U_general, atol=atol, rtol=atol)

    n = int(ranks[0])
    for k in range(len(ranks)):
        for col in range(n):
            np.testing.assert_allclose(
                F[num_rows[col, k]:n, col, k],
                0.0,
                atol=0.0,
            )
        np.testing.assert_array_equal(num_rows[n:, k], 0)
