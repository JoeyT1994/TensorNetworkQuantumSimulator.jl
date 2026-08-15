"""Compare the local SLICOT build with product-based FP64 Schur paths."""

from __future__ import annotations

from pathlib import Path
import statistics
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from jax_config import configure_jax

configure_jax()

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from linalg.jax_linalg import periodic_schur_bruteforce
from linalg.periodic_schur import _slicot_periodic
from linalg.periodic_schur.slicot_interface import (
    _pack_slicot_decomposition_factors,
    _slicot_periodic_schur_f2py_callback,
)


PERIOD = 4
MATRIX_SIZE = 27


def _median_time(fn, repeats=1001):
    """Return median and minimum warm host runtime in milliseconds."""
    fn()
    fn()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        elapsed.append(time.perf_counter_ns() - start)
    return statistics.median(elapsed) * 1e-6, min(elapsed) * 1e-6


def _median_jax(fn, arg, repeats=501):
    """Return synchronized median and minimum warm JAX runtime in milliseconds."""
    jax.block_until_ready(fn(arg))
    jax.block_until_ready(fn(arg))
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        jax.block_until_ready(fn(arg))
        elapsed.append(time.perf_counter_ns() - start)
    return statistics.median(elapsed) * 1e-6, min(elapsed) * 1e-6


def _prepare_slicot(H):
    """Return stable Hessenberg data for isolated SLICOT stage timings."""
    period, n, _ = H.shape
    h0 = _pack_slicot_decomposition_factors(H)
    tau0 = np.zeros((n - 1, period), dtype=np.float64, order="F")
    h_vd, tau_vd, _, info = _slicot_periodic.mb03vd(
        1,
        n,
        h0.copy(order="F"),
        tau0.copy(order="F"),
        np.empty(n),
        n=n,
        p=period,
        lda1=n,
        lda2=n,
        ldtau=n - 1,
    )
    if info:
        raise RuntimeError(f"MB03VD info={info}")
    z_vy, _, info = _slicot_periodic.mb03vy(
        n,
        1,
        n,
        h_vd.copy(order="F"),
        tau_vd,
        np.empty(8 * n),
        p=period,
        lda1=n,
        lda2=n,
        ldtau=n - 1,
        ldwork=8 * n,
    )
    if info:
        raise RuntimeError(f"MB03VY info={info}")
    return h0, tau0, h_vd, tau_vd, z_vy


def _periodic_residual(H, T, Z):
    """Return the maximum sitewise periodic-Schur residual norm."""
    period = H.shape[0]
    return max(
        np.linalg.norm(H[l] @ Z[l] - Z[(l + 1) % period] @ T[l])
        for l in range(period)
    )


def main():
    """Print ordinary, periodic, and isolated SLICOT decomposition timings."""
    rng = np.random.default_rng(301)
    H = rng.standard_normal((PERIOD, MATRIX_SIZE, MATRIX_SIZE)) / np.sqrt(MATRIX_SIZE)
    product = H[3] @ H[2] @ H[1] @ H[0]
    h0, tau0, h_vd, tau_vd, z_vy = _prepare_slicot(H)

    def mb03vd():
        """Run the periodic Hessenberg reduction."""
        return _slicot_periodic.mb03vd(
            1,
            MATRIX_SIZE,
            h0.copy(order="F"),
            tau0.copy(order="F"),
            np.empty(MATRIX_SIZE),
            n=MATRIX_SIZE,
            p=PERIOD,
            lda1=MATRIX_SIZE,
            lda2=MATRIX_SIZE,
            ldtau=MATRIX_SIZE - 1,
        )

    def mb03vy():
        """Form the accumulated periodic Hessenberg bases."""
        return _slicot_periodic.mb03vy(
            MATRIX_SIZE,
            1,
            MATRIX_SIZE,
            h_vd.copy(order="F"),
            tau_vd,
            np.empty(8 * MATRIX_SIZE),
            p=PERIOD,
            lda1=MATRIX_SIZE,
            lda2=MATRIX_SIZE,
            ldtau=MATRIX_SIZE - 1,
            ldwork=8 * MATRIX_SIZE,
        )

    def mb03wd(job="S", compz="V"):
        """Run one periodic real Schur iteration option."""
        return _slicot_periodic.mb03wd(
            job,
            compz,
            1,
            MATRIX_SIZE,
            1,
            MATRIX_SIZE,
            h_vd.copy(order="F"),
            z_vy.copy(order="F"),
            np.empty(MATRIX_SIZE),
            np.empty(MATRIX_SIZE),
            np.empty(MATRIX_SIZE + PERIOD),
            n=MATRIX_SIZE,
            p=PERIOD,
            ldh1=MATRIX_SIZE,
            ldh2=MATRIX_SIZE,
            ldz1=MATRIX_SIZE,
            ldz2=MATRIX_SIZE,
            ldwork=MATRIX_SIZE + PERIOD,
        )

    brute_jit = jax.jit(periodic_schur_bruteforce)
    T, Z, _ = _slicot_periodic_schur_f2py_callback(H)
    residual = _periodic_residual(H, T, Z)
    timings = {
        "NumPy eig(product)": _median_time(lambda: np.linalg.eig(product)),
        "NumPy eigvals(product)": _median_time(lambda: np.linalg.eigvals(product)),
        "SciPy real Schur(product)": _median_time(
            lambda: scipy.linalg.schur(product, output="real")
        ),
        "form product + eig": _median_time(
            lambda: np.linalg.eig(H[3] @ H[2] @ H[1] @ H[0])
        ),
        "JAX brute periodic Schur": _median_jax(brute_jit, jnp.asarray(H)),
        "full periodic SLICOT": _median_time(
            lambda: _slicot_periodic_schur_f2py_callback(H)
        ),
        "MB03VD": _median_time(mb03vd),
        "MB03VY": _median_time(mb03vy),
        "MB03WD S,V": _median_time(mb03wd),
        "MB03WD S,N": _median_time(lambda: mb03wd("S", "N")),
        "MB03WD E,N": _median_time(lambda: mb03wd("E", "N")),
    }

    print(f"period={PERIOD} n={MATRIX_SIZE} dtype={H.dtype} residual={residual:.3e}")
    for name, (median, minimum) in timings.items():
        print(f"{name:27s} median={median:8.4f} ms min={minimum:8.4f} ms")


if __name__ == "__main__":
    main()
