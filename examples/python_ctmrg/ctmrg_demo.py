"""Converge a chi-9 CTM environment around Yantao's 5x5 Ising PEPS."""

from pathlib import Path
from jax_config import configure_jax

configure_jax()

import jax
import jax.numpy as jnp
from ctm_state import CTMState
from ctmrg_new import ctmrg


def load_yantao_peps(path):
    """Load Yantao's nested PEPS format into dense T[x,y,p,l,r,d,u]."""
    import numpy as np
    tensors = np.load(path, allow_pickle=True)
    Nx = len(tensors)
    Ny = len(tensors[0])

    d = tensors[0][0].shape[0]
    chi_P = max(max(t.shape[1:]) for row in tensors for t in row)
    T = np.zeros(
        (Nx, Ny, d, chi_P, chi_P, chi_P, chi_P),
        dtype=tensors[0][0].dtype,
    )
    for x in range(Nx):
        for y in range(Ny):
            t = tensors[x][y]
            l, r, dwn, up = t.shape[1:]
            T[x, y, :, :l, :r, :dwn, :up] = t

    return jnp.asarray(T)


CHI = 9
NUM_CTMRG_ITER = 10
CONVERGENCE_TOL = 1e-10
PEPS_PATH = (
    Path(__file__).resolve().parent
    / "data_ising_5x5"
    / "isingZZX_5x5_D3_g3.04438.npz"
)


def yantao_to_ctm_order(T):
    """Map Yantao legs ``(l,r,d,u)`` to CTM legs ``(r,u,l,d)``."""

    return jnp.transpose(T, (0, 1, 2, 4, 6, 3, 5))


def main():
    """Load the PEPS, initialize its CTM state, and converge the boundary."""

    T = yantao_to_ctm_order(load_yantao_peps(PEPS_PATH))
    state = CTMState.init(T, chi=CHI)
    state, info = ctmrg(
        state,
        {
            "num_ctmrg_iter": NUM_CTMRG_ITER,
            "pinv_rtol": 1e-12,
            "krylov_cfg": {"V_guess_stochastic_num_iter": 2},
        },
    )
    jax.block_until_ready((state, info))

    max_dV = jnp.max(info.dVL, axis=(1, 2, 3))
    final_dV = float(max_dV[-1])
    if final_dV > CONVERGENCE_TOL:
        raise RuntimeError(
            f"CTMRG did not converge in {NUM_CTMRG_ITER} sweeps: "
            f"max dV={final_dV:.3e}"
        )

    print(f"PEPS shape: {T.shape}")
    print(f"CTM chi: {CHI}")
    print(f"converged max dV: {final_dV:.3e}")
    print(f"Z: {float(state.Z()):.16g}")


if __name__ == "__main__":
    main()
