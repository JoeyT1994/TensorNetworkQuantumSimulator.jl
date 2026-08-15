"""Archived benchmark for the unbuilt num_rows Householder experiment."""

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from linalg.periodic_schur.experimental_benchmark_householder import _median_ms


def _triangular_block_hessenberg(rng, period, depth, d_block, dtype):
    """Return block-Hessenberg factors with triangular subdiagonal blocks."""
    m = depth * d_block
    H = np.zeros((period, m, m), dtype=dtype)
    for k in range(period):
        for block in range(depth):
            col = slice(block * d_block, (block + 1) * d_block)
            top = (block + 1) * d_block
            values = rng.standard_normal((top, d_block))
            if dtype == np.complex128:
                values = values + 1j * rng.standard_normal(values.shape)
            H[k, :top, col] = values
            if block + 1 < depth:
                values = rng.standard_normal((d_block, d_block))
                if dtype == np.complex128:
                    values = values + 1j * rng.standard_normal(values.shape)
                H[k, top:top + d_block, col] = np.triu(values)
    return H


def _factor_num_rows(factors, ranks):
    """Return exact exclusive row bounds for compact logical columns."""
    capacity, _, period = factors.shape
    num_rows = np.zeros((capacity, period), dtype=np.intp)
    for k in range(period):
        kp = (k + 1) % period
        for col in range(ranks[kp]):
            nonzero = np.flatnonzero(factors[:ranks[k], col, k] != 0)
            num_rows[col, k] = nonzero[-1] + 1 if nonzero.size else 0
    return num_rows


def _case(dtype, target_ranks, depth=3, d_block=64):
    """Time general and row-bounded preparation at one active-rank pattern."""
    from linalg.periodic_schur import _periodic_schur

    period = len(target_ranks)
    m = depth * d_block
    rng = np.random.default_rng(2431)
    H = _triangular_block_hessenberg(
        rng,
        period,
        depth,
        d_block,
        dtype,
    )
    active = np.zeros((period, m), dtype=bool)
    for k, rank in enumerate(target_ranks):
        active[k, :rank] = True
    suffix = "d" if dtype == np.float64 else "z"
    F0, U0, ranks0, block_ranks0, cut_offset = getattr(
        _periodic_schur,
        f"compact_active_slicot_{suffix}",
    )(H, active, d_block)
    if cut_offset:
        raise ValueError("target_ranks must place a minimum rank at cut zero")
    num_rows0 = _factor_num_rows(F0, ranks0)

    def prepare_general():
        """Return fresh buffers for the structurally general kernel."""
        return (
            F0.copy(order="F"),
            U0.copy(order="F"),
            ranks0.copy(),
            block_ranks0.copy(),
        )

    def run_general(args):
        """Run the structurally general fused kernel."""
        getattr(
            _periodic_schur,
            f"make_periodic_hessenberg_HOUSEHOLDER_{suffix.upper()}",
        )(*args)

    def prepare_num_rows():
        """Return fresh buffers and structural row bounds."""
        return (*prepare_general(), num_rows0.copy())

    def run_num_rows(args):
        """Run the row-bounded fused kernel."""
        getattr(
            _periodic_schur,
            f"make_periodic_hessenberg_HOUSEHOLDER_NUM_ROWS_{suffix.upper()}",
        )(*args)

    return (
        _median_ms(prepare_general, run_general),
        _median_ms(prepare_num_rows, run_num_rows),
    )


def main():
    """Print warm timings for equal and unequal active-rank patterns."""
    cases = (
        [192, 192],
        [172, 192],
        [192, 192, 192, 192],
        [172, 180, 192, 180],
    )
    print("block=64 depth=3 m=192; triangular subdiagonal; 201 warm repeats")
    print("dtype       ranks                    general (ms)  num_rows (ms)  speedup")
    for dtype in (np.float64, np.complex128):
        for ranks in cases:
            general, num_rows = _case(dtype, ranks)
            print(
                f"{np.dtype(dtype).name:<11} {str(ranks):<24} "
                f"{general:>12.3f}  {num_rows:>13.3f}  "
                f"{general / num_rows:>7.2f}x"
            )


if __name__ == "__main__":
    main()
