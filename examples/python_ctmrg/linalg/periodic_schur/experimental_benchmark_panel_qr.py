"""Archived benchmark for the unbuilt full and panel CRed experiments."""

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
            stop = min(depth, j + 2) * d_block
            values = rng.standard_normal((stop, d_block))
            if dtype == np.complex128:
                values = values + 1j * rng.standard_normal(values.shape)
            H[k, :stop, j * d_block:(j + 1) * d_block] = values
    return H


def _median_ms(prepare, run, repeat=201):
    """Return median kernel time with fresh inputs prepared off the clock."""
    run(prepare())
    samples = []
    for _ in range(repeat):
        args = prepare()
        start = time.perf_counter_ns()
        run(args)
        samples.append(time.perf_counter_ns() - start)
    return np.median(samples) * 1e-6


def _case(dtype, period, depth=3, d_block=64):
    """Time fresh in-place sweeps at one dtype and period."""
    from linalg.periodic_schur import _periodic_schur

    rng = np.random.default_rng(917)
    m = depth * d_block
    H = _block_hessenberg(rng, period, depth, d_block, dtype)
    active = np.ones((period, m), dtype=bool)
    suffix = "d" if dtype == np.float64 else "z"
    packed = getattr(_periodic_schur, f"compact_active_slicot_{suffix}")(
        H, active, d_block
    )
    F0, U0, r0, br0, _ = packed

    def prepare_full():
        """Return fresh mutable inputs for the full-factor reference."""
        return F0.copy(order="F"), U0.copy(order="F"), r0.copy()

    def run_full(args):
        """Run the full-factor reference kernel."""
        getattr(_periodic_schur, f"_full_qr_sweep_{suffix}")(*args)

    def prepare_panel():
        """Return fresh mutable inputs for the block-panel alternative."""
        return F0.copy(order="F"), U0.copy(order="F"), r0.copy(), br0.copy()

    def run_panel(args):
        """Run the block-panel alternative kernel."""
        getattr(_periodic_schur, f"_panel_qr_sweep_{suffix}")(*args)

    return (
        _median_ms(prepare_full, run_full),
        _median_ms(prepare_panel, run_panel),
    )


def main():
    """Print warm depth-three timings for D and Z kernels."""
    print("dtype       period  full QR (ms)  panel QR (ms)  panel/full")
    for dtype in (np.float64, np.complex128):
        for period in (2, 4):
            full, panel = _case(dtype, period)
            print(
                f"{np.dtype(dtype).name:<11} {period:>6}  {full:>12.3f}  "
                f"{panel:>13.3f}  {panel / full:>10.3f}"
            )


if __name__ == "__main__":
    main()
