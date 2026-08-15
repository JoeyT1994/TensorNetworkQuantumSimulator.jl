"""Archived benchmark for the unbuilt fused Householder CRed experiment."""

from pathlib import Path
import sys
import time

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _block_hessenberg(rng, period, depth, d_block, dtype):
    """Return a depth-block upper-Hessenberg factor stack."""
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


def _median_ms(prepare, run, repeat=201):
    """Return a warm median with fresh mutable inputs prepared off the clock."""
    run(prepare())
    samples = np.empty(repeat, dtype=np.float64)
    for sample in range(repeat):
        args = prepare()
        start = time.perf_counter_ns()
        run(args)
        samples[sample] = time.perf_counter_ns() - start
    return np.median(samples) * 1e-6


def _case(dtype, period, depth=3, d_block=64):
    """Time preparation-only kernels for one dtype and period."""
    from linalg.periodic_schur import _periodic_schur

    rng = np.random.default_rng(2421)
    m = depth * d_block
    H = _block_hessenberg(rng, period, depth, d_block, dtype)
    active = np.ones((period, m), dtype=bool)
    suffix = "d" if dtype == np.float64 else "z"
    F0, U0, ranks0, block_ranks0, _ = getattr(
        _periodic_schur,
        f"compact_active_slicot_{suffix}",
    )(H, active, d_block)

    def prepare_householder():
        """Return fresh fused-kernel buffers."""
        return (
            F0.copy(order="F"),
            U0.copy(order="F"),
            ranks0.copy(),
            block_ranks0.copy(),
        )

    def run_householder(args):
        """Run the isolated fused preparation."""
        getattr(
            _periodic_schur,
            f"make_periodic_hessenberg_HOUSEHOLDER_{suffix.upper()}",
        )(*args)

    def prepare_givens():
        """Return fresh full-QR-plus-Givens buffers."""
        return F0.copy(order="F"), U0.copy(order="F"), ranks0.copy()

    def run_givens(args):
        """Run the existing full-QR and Givens preparation kernels."""
        getattr(
            _periodic_schur,
            f"make_periodic_hessenberg_GIVENS_{suffix.upper()}",
        )(*args)

    return (
        _median_ms(prepare_householder, run_householder),
        _median_ms(prepare_givens, run_givens),
    )


def main():
    """Print requested warm preparation-only depth-three timings."""
    print("block=64 depth=3 m=192; 201 warm repeats; median preparation time")
    print("dtype       period  Householder (ms)  full QR + Givens (ms)  speedup")
    for dtype in (np.float64, np.complex128):
        for period in (2, 4):
            householder, givens = _case(dtype, period)
            print(
                f"{np.dtype(dtype).name:<11} {period:>6}  "
                f"{householder:>16.3f}  {givens:>21.3f}  "
                f"{givens / householder:>7.2f}x"
            )


if __name__ == "__main__":
    main()
