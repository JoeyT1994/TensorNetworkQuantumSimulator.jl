import os
from functools import lru_cache, partial
from typing import NamedTuple

import einops
import numpy as np

from jax_config import configure_jax

configure_jax()
import jax
import jax.numpy as jnp
# Coordinates: increasing/decreasing x is right/left, and increasing/decreasing
# y is up/down. See docs/coordinate_conventions.md.

REMAT_DOUBLE_LTR = os.environ.get("TNS_REMAT_DOUBLE_LTR", "0") == "1"

# "spec_name":(contract_str, conj)
CONTRACT_SPECS = {
    "MPS MPS": ("p, p", (0, 0)),
    "MPS MPO MPS": ("p1, p1 p2, p2", (0, 0, 0)),
    "bMPS bMPS": ("p1 q1, p1 q1", (0, 0)),
    "bMPS t* t bMPS": ("q1 p1, d q1 q2, d p1 p2, q2 p2", (0, 1, 0, 0)),
    "bMPS tb t bMPS": ("q1 p1, d q1 q2, d p1 p2, q2 p2", (0, 0, 0, 0)),
}

def pinv(E, r_tol = 1e-12):
    U, s, Vh = jnp.linalg.svd(E, full_matrices=False)
    cutoff = r_tol*jnp.linalg.norm(s, axis=-1, keepdims=True)
    keep = s > cutoff
    s_safe = jnp.where(keep, s, 1.0)
    s_inv = jnp.where(keep, 1.0/s_safe, 0.0)
    E_inv = jnp.matmul(
        jnp.swapaxes(Vh.conj(), -1, -2)*s_inv[..., None, :],
        jnp.swapaxes(U.conj(), -1, -2),
    )
    return E_inv, s


class ColPrimitives(NamedTuple):
    overlap: callable
    env_b: callable
    env_t: callable
    push_Eb_up: callable
    push_Et_down: callable
    dLTR_dL: callable
    dLTR_dR: callable

@lru_cache(None)
def make_col_primitives(spec, *, remat_scan=False):
    """Build column contractions, optionally rematerializing scan rows."""
    contract_string, conj = spec
    terms = [term.strip() for term in contract_string.split(",")]
    contract_terms = tuple(tuple(term.split()) for term in terms)
    n = len(contract_terms)

    carry_in = " ".join(f"i{k}" for k in range(1, n + 1))
    carry_out = " ".join(f"j{k}" for k in range(1, n + 1))

    tensor_terms = []
    for k, labels in enumerate(contract_terms, start=1):
        term_labels = list(labels) + [f"i{k}", f"j{k}"]
        tensor_terms.append(" ".join(term_labels))

    # Eb_i A_{ij} -> Eb_j; Et_j A_{ij} -> Et_i.
    contract_body_up = ", ".join([carry_in, *tensor_terms]) + f" -> {carry_out}"
    contract_body_down = ", ".join([carry_out, *tensor_terms]) + f" -> {carry_in}"
    # Differentiate by leaving the L or R tensor's indices open.
    contract_body_dL = ", ".join([carry_in, *tensor_terms[1:], carry_out]) + f" -> {tensor_terms[0]}"
    contract_body_dR = ", ".join([carry_in, *tensor_terms[:-1], carry_out]) + f" -> {tensor_terms[-1]}"

    def block_contract_body(var_pos, block_len):
        """Build an einsum string for a multi-row derivative block."""
        carry = [[f"i{k}r{row}" for k in range(1, n + 1)] for row in range(block_len + 1)]
        tensor_block_terms = []
        for row in range(block_len):
            row_terms = []
            for k, labels in enumerate(contract_terms):
                payload = [f"{label}r{row}" for label in labels]
                row_terms.append(" ".join(payload + [carry[row][k], carry[row + 1][k]]))
            tensor_block_terms.append(row_terms)

        inputs = [" ".join(carry[0])]
        for row in range(block_len):
            for k in range(n):
                if k != var_pos:
                    inputs.append(tensor_block_terms[row][k])
        inputs.append(" ".join(carry[block_len]))

        output = []
        for row in range(block_len):
            output.extend(f"{label}r{row}" for label in contract_terms[var_pos])
        output.extend([carry[0][var_pos], carry[block_len][var_pos]])
        return ", ".join(inputs) + " -> " + " ".join(output)

    def compute_overlap(*Ts):
        """Contract a full column network to a scalar."""
        Ts = tuple(jnp.conj(T) if do_conj else T for T, do_conj in zip(Ts, conj))
        E_shape = tuple(T.shape[-1] for T in Ts)
        Eb = jnp.zeros(E_shape, dtype=Ts[0].dtype).at[(0,) * n].set(1.0)

        def scan_fn(Eb, tensors):
            """Accumulate Eb through one row."""
            Eb = einops.einsum(Eb, *tensors, contract_body_up)
            return Eb, None

        if remat_scan:
            scan_fn = jax.checkpoint(scan_fn, prevent_cse=False)
        Eb, _ = jax.lax.scan(scan_fn, Eb, Ts)
        return Eb[(0,) * n]

    def compute_env_b(*Ts, y=None):
        """Return bottom environments below each row of a full column."""
        Ly = Ts[0].shape[0]
        if y is not None:
            Ts = tuple(T[:y] for T in Ts)
        Ts = tuple(jnp.conj(T) if do_conj else T for T, do_conj in zip(Ts, conj))
        E_shape = tuple(T.shape[-1] for T in Ts)
        Eb0 = jnp.zeros(E_shape, dtype=Ts[0].dtype).at[(0,) * n].set(1.0)

        def scan_fn(Eb, tensors):
            """Push Eb up and save the environment after this row."""
            Eb_next = einops.einsum(Eb, *tensors, contract_body_up)
            return Eb_next, Eb_next

        if remat_scan:
            scan_fn = jax.checkpoint(scan_fn, prevent_cse=False)
        if y is None:
            _, Eb_all = jax.lax.scan(scan_fn, Eb0, Ts)
            return jnp.concatenate([Eb0[None, ...], Eb_all[:-1]], axis=0)

        _, Eb_all = jax.lax.scan(scan_fn, Eb0, Ts)
        Eb_zero = jnp.zeros((Ly - y - 1,) + E_shape, dtype=Ts[0].dtype)
        return jnp.concatenate([Eb0[None, ...], Eb_all, Eb_zero], axis=0)

    def compute_env_t(*Ts, y=None):
        """Return top environments above each row of a full column."""
        if y is not None:
            Ts = tuple(T[y + 1:] for T in Ts)
        Ts = tuple(jnp.conj(T) if do_conj else T for T, do_conj in zip(Ts, conj))
        E_shape = tuple(T.shape[-1] for T in Ts)
        Et0 = jnp.zeros(E_shape, dtype=Ts[0].dtype).at[(0,) * n].set(1.0)

        def scan_fn(Et, tensors):
            """Push Et down and save the environment after this row."""
            Et_next = einops.einsum(Et, *tensors, contract_body_down)
            return Et_next, Et_next

        if remat_scan:
            scan_fn = jax.checkpoint(scan_fn, prevent_cse=False)
        if y is None:
            _, Et_all = jax.lax.scan(scan_fn, Et0, Ts, reverse=True)
            return jnp.concatenate([Et_all[1:], Et0[None, ...]], axis=0)

        _, Et_all = jax.lax.scan(scan_fn, Et0, Ts, reverse=True)
        Et_zero = jnp.zeros((y,) + E_shape, dtype=Ts[0].dtype)
        return jnp.concatenate([Et_zero, Et_all, Et0[None, ...]], axis=0)

    def compute_push_Eb_up(E, *site_terms):
        """Push a bottom environment upward through one row."""
        site_terms = tuple(
            jnp.conj(T) if do_conj else T
            for T, do_conj in zip(site_terms, conj)
        )
        return einops.einsum(E, *site_terms, contract_body_up)

    def compute_push_Et_down(E, *site_terms):
        """Push a top environment downward through one row."""
        site_terms = tuple(
            jnp.conj(T) if do_conj else T
            for T, do_conj in zip(site_terms, conj)
        )
        return einops.einsum(E, *site_terms, contract_body_down)

    def compute_dLTR_dL(Eb, Et, *site_terms):
        """Differentiate <L|T|R> with respect to n sites of L.

        n is inferred from the leading dimension of the passed site terms.
        """
        site_terms = tuple(
            jnp.conj(T) if do_conj else T
            for T, do_conj in zip(site_terms, conj[1:])
        )
        block_len = site_terms[0].shape[0]
        if block_len == 1:
            return einops.einsum(
                Eb,
                *(T[0] for T in site_terms),
                Et,
                contract_body_dL,
            )
        block_terms = []
        for row in range(block_len):
            block_terms.extend(T[row] for T in site_terms)
        # TODO: This approach will not scale to n >> 1: einsum path search can
        # blow up exponentially. Break this into n row-wise einsums once longer
        # blocks matter; the optimal order should proceed linearly up the chain.
        return einops.einsum(Eb, *block_terms, Et, block_contract_body(0, block_len))

    def compute_dLTR_dR(Eb, Et, *site_terms):
        """Differentiate <L|T|R> with respect to n sites of R.

        n is inferred from the leading dimension of the passed site terms.
        """
        site_terms = tuple(
            jnp.conj(T) if do_conj else T
            for T, do_conj in zip(site_terms, conj[:-1])
        )
        block_len = site_terms[0].shape[0]
        if block_len == 1:
            return einops.einsum(
                Eb,
                *(T[0] for T in site_terms),
                Et,
                contract_body_dR,
            )
        block_terms = []
        for row in range(block_len):
            block_terms.extend(T[row] for T in site_terms)
        # TODO: This approach will not scale to n >> 1: einsum path search can
        # blow up exponentially. Break this into n row-wise einsums once longer
        # blocks matter; the optimal order should proceed linearly up the chain.
        return einops.einsum(Eb, *block_terms, Et, block_contract_body(n - 1, block_len))

    return ColPrimitives(
        overlap=compute_overlap,
        env_b=compute_env_b,
        env_t=compute_env_t,
        push_Eb_up=compute_push_Eb_up,
        push_Et_down=compute_push_Et_down,
        dLTR_dL=compute_dLTR_dL,
        dLTR_dR=compute_dLTR_dR,
    )


lr_ops_single = make_col_primitives(CONTRACT_SPECS["MPS MPS"])
ltr_ops_single = make_col_primitives(CONTRACT_SPECS["MPS MPO MPS"])

lr_ops_double = make_col_primitives(CONTRACT_SPECS["bMPS bMPS"])
ltr_ops_double = make_col_primitives(
    CONTRACT_SPECS["bMPS t* t bMPS"],
    remat_scan=REMAT_DOUBLE_LTR,
)
ltbtr_ops_double = make_col_primitives(
    CONTRACT_SPECS["bMPS tb t bMPS"],
    remat_scan=REMAT_DOUBLE_LTR,
)


# Public column-wrapper index convention.
#
# All functions below take one vertical column, with y as the leading row axis.
# A boundary row tensor has indices L[y, P, i, j] or R[y, P, i, j], where
# i,j are the incoming/outgoing vertical MPS bond legs.  P is a shorthand
# payload index: P = p for a single-layer row, and P = p1 p2 for a double-layer
# row.  In the double-layer case the array keeps p1,p2 as separate axes:
# L[y, p1, p2, i, j], R[y, p1, p2, i, j].
#
# LR wrappers contract matching payload legs.  A bottom environment below row y
# has lower legs E_b[iL, iR]; pushing it upward gives the bottom environment
# below row y+1:
#     E_b[y, iL, iR] L[y, P, iL, jL] R[y, P, iR, jR]
#         -> E_b[y+1, jL, jR].
# A top environment above row y has upper legs E_t[jL, jR]; pushing it downward
# gives the top environment above row y-1:
#     E_t[y, jL, jR] L[y, P, iL, jL] R[y, P, iR, jR]
#         -> E_t[y-1, iL, iR].
#
# LTR wrappers insert a transfer row between L and R.  For the single-layer
# path, T[y, pL, pR, iT, jT] maps the L payload to the R payload:
#     E_b[y, iL, iT, iR] L[y, pL, iL, jL] T[y, pL, pR, iT, jT]
#         R[y, pR, iR, jR] -> E_b[y+1, jL, jT, jR].
#     E_t[y, jL, jT, jR] L[y, pL, iL, jL] T[y, pL, pR, iT, jT]
#         R[y, pR, iR, jR] -> E_t[y-1, iL, iT, iR].
# For the double-layer PEPS path, the same site tensor T supplies the bra and
# ket copies:
#     L[y, q1, p1, iL, jL] conj(T[y, d, q1, q2, iTb, jTb])
#     T[y, d, p1, p2, iTk, jTk] R[y, q2, p2, iR, jR].
# Thus E_b is ordered as (iL, iTb, iTk, iR), while E_t is ordered as
# (jL, jTb, jTk, jR).
#
# env_b returns the bottom environment below each row; env_t returns the top
# environment above each row.  dLTR_dL_col/dLTR_dR_col consume E_b below and
# E_t above the varied row or row block, leaving that boundary block open.
@jax.jit
def LR_overlap_col(L, R):
    """<L|R>"""
    if L.ndim == 4:
        return lr_ops_single.overlap(L, R)
    if L.ndim == 5:
        return lr_ops_double.overlap(L, R)
    raise ValueError(f"Unsupported boundary column rank: {L.ndim}")

@partial(jax.jit, static_argnames=("y",))
def env_b_LR_col(L, R, y=None):
    """Bot env of <L|R>"""
    if L.ndim == 4:
        return lr_ops_single.env_b(L, R, y=y)
    if L.ndim == 5:
        return lr_ops_double.env_b(L, R, y=y)
    raise ValueError(f"Unsupported boundary column rank: {L.ndim}")

@partial(jax.jit, static_argnames=("y",))
def env_t_LR_col(L, R, y=None):
    """Top env of <L|R>"""
    if L.ndim == 4:
        return lr_ops_single.env_t(L, R, y=y)
    if L.ndim == 5:
        return lr_ops_double.env_t(L, R, y=y)
    raise ValueError(f"Unsupported boundary column rank: {L.ndim}")

@jax.jit
def LTR_overlap_col(L, T, R):
    """ <L|T|R> or <L|T.T|R> for single-layer / double layer network"""
    if L.ndim == 4:
        return ltr_ops_single.overlap(L, T, R)
    if L.ndim == 5:
        return ltr_ops_double.overlap(L, T, T, R)
    raise ValueError(f"Unsupported boundary column rank: {L.ndim}")


@jax.jit
def LTbTR_overlap_col(L, Tb, T, R):
    """Contract ``<L|Tb T|R>`` without conjugating the supplied ``Tb``."""

    if L.ndim == 5:
        return ltbtr_ops_double.overlap(L, Tb, T, R)
    raise ValueError(f"LTbTR requires a double-layer boundary, got rank {L.ndim}")


@jax.jit
def LToTR_overlap_col(L, T, oT, R):
    """ Specifically for double-layer, <L|ToT|R> """
    if L.ndim == 4:
        raise ValueError
    if L.ndim == 5:
        return ltr_ops_double.overlap(L, T, oT, R)
    raise ValueError(f"Unsupported boundary column rank: {L.ndim}")

@partial(jax.jit, static_argnames=("y",))
def env_b_LTR_col(L, T, R, y=None):
    if L.ndim == 4:
        return ltr_ops_single.env_b(L, T, R, y=y)
    if L.ndim == 5:
        return ltr_ops_double.env_b(L, T, T, R, y=y)
    raise ValueError(f"Unsupported boundary column rank: {L.ndim}")

@partial(jax.jit, static_argnames=("y",))
def env_t_LTR_col(L, T, R, y=None):
    if L.ndim == 4:
        return ltr_ops_single.env_t(L, T, R, y=y)
    if L.ndim == 5:
        return ltr_ops_double.env_t(L, T, T, R, y=y)
    raise ValueError(f"Unsupported boundary column rank: {L.ndim}")

@jax.jit
def push_Eb_LR_up(E, L, R):
    if R.ndim == 3:
        return lr_ops_single.push_Eb_up(E, L, R)
    if R.ndim == 4:
        return lr_ops_double.push_Eb_up(E, L, R)
    raise ValueError(f"Unsupported row rank for right tensor: {R.ndim}")


@jax.jit
def push_Et_LR_down(E, L, R):
    if R.ndim == 3:
        return lr_ops_single.push_Et_down(E, L, R)
    if R.ndim == 4:
        return lr_ops_double.push_Et_down(E, L, R)
    raise ValueError(f"Unsupported row rank for right tensor: {R.ndim}")

@jax.jit
def push_Eb_LTR_up(E, L, T, R):
    if R.ndim == 3:
        return ltr_ops_single.push_Eb_up(E, L, T, R)
    if R.ndim == 4:
        return ltr_ops_double.push_Eb_up(E, L, T, T, R)
    raise ValueError(f"Unsupported row rank for right tensor: {R.ndim}")

@jax.jit
def push_Et_LTR_down(E, L, T, R):
    if R.ndim == 3:
        return ltr_ops_single.push_Et_down(E, L, T, R)
    if R.ndim == 4:
        return ltr_ops_double.push_Et_down(E, L, T, T, R)
    raise ValueError(f"Unsupported row rank for right tensor: {R.ndim}")


@jax.jit
def push_Eb_LTbTR_up(E, L, Tb, T, R):
    """Push ``Eb`` through one ``<L|Tb T|R>`` double-layer row."""

    if R.ndim == 4:
        return ltbtr_ops_double.push_Eb_up(E, L, Tb, T, R)
    raise ValueError(f"LTbTR requires a double-layer row, got rank {R.ndim}")


@jax.jit
def push_Et_LTbTR_down(E, L, Tb, T, R):
    """Push ``Et`` through one ``<L|Tb T|R>`` double-layer row."""

    if R.ndim == 4:
        return ltbtr_ops_double.push_Et_down(E, L, Tb, T, R)
    raise ValueError(f"LTbTR requires a double-layer row, got rank {R.ndim}")

@jax.jit
def push_Et_LToTR_down(E, L, oT, R, T=None):
    """T is the optional bra column.
    """
    if R.ndim == 3:
        return ltr_ops_single.push_Et_down(E, L, oT, R)
    if R.ndim == 4:
        return ltr_ops_double.push_Et_down(E, L, T, oT, R)
    raise ValueError()


@jax.jit
def push_Eb_LToTR_up(E, L, oT, R, T=None):
    if R.ndim == 3:
        return ltr_ops_single.push_Eb_up(E, L, oT, R)
    if R.ndim == 4:
        return ltr_ops_double.push_Eb_up(E, L, T, oT, R)
    raise ValueError()


def _with_col_axis(X, row_ndim):
    """Add a singleton row axis when ``X`` is one site instead of a slice."""
    return X[None] if X.ndim == row_ndim else X


@jax.jit
def dLTR_dL_col(Eb, Et, T, R):
    """ d/d_L <L|T|R> for n-site variations"""
    ndim = Eb.ndim
    T = _with_col_axis(T, ndim + 1)
    R = _with_col_axis(R, ndim)
    if ndim == 3:
        return ltr_ops_single.dLTR_dL(Eb, Et, T, R)
    if ndim == 4:
        return ltr_ops_double.dLTR_dL(Eb, Et, T, T, R)
    raise ValueError(f"Unsupported LTR environment rank: {Eb.ndim}")

@jax.jit
def dLTR_dR_col(Eb, Et, L, T):
    """ d/d_R <L|T|R> for n-site variations"""
    ndim = Eb.ndim
    L = _with_col_axis(L, ndim)
    T = _with_col_axis(T, ndim + 1)
    if ndim == 3:
        return ltr_ops_single.dLTR_dR(Eb, Et, L, T)
    if ndim == 4:
        return ltr_ops_double.dLTR_dR(Eb, Et, L, T, T)
    raise ValueError(f"Unsupported LTR environment rank: {Eb.ndim}")

# ---------------- with block-sparse MPO columns --------------------
# note that in the double-layer context, "W" usually means the MPO already 
# contracted into the ket column, i.e. W=oT, rather than the MPO alone.

@jax.jit
def push_Et_LWR_down(Et_y, L_y, W_y, R_y, T_y=None):
    if isinstance(W_y, (jax.Array, np.ndarray)):
        return push_Et_LToTR_down(Et_y, L_y, W_y, R_y, T_y)
    if L_y.ndim == 3:
        return push_Et_LWR_down_single(Et_y, L_y, W_y, R_y)
    if L_y.ndim == 4:
        return push_Et_LWR_down_double(Et_y, L_y, T_y, W_y, R_y)    
    raise ValueError

@jax.jit
def push_Et_LWR_down_single(Et_y, L_y, W_y, R_y):
    """Push E_top down one row for <L|W|R> at row y.
    Here W is a BlockSparseMPO representing a T-column with (a sum of)
    perturbations applied to it.
    """
    # s: sparse index. t: non-sparse index of W.
    Et_out = jnp.einsum('p d u, s l t u -> s l t p d', R_y, Et_y)
    # W_y is just a perturbed (rank-4, ltpr, since single-layer) tensor
    Et_out = W_y.matvec(Et_out, contract='q p b t, l t p r -> l q b r')
    Et_out = jnp.einsum('q k l, i l q b r  -> i k b r', L_y, Et_out)
    return Et_out

@jax.jit
def push_Et_LWR_down_double(Et_y, L_y, T_y, W_y, R_y):
    """Push E_top down one row for the column overlap <L|T*W|R>, y -> y-1.
    """
    REt = einops.einsum(R_y, Et_y,
        "R_q2 R_p2 d u, s E_j1 E_j2 E_j3 u"
        " -> s E_j1 E_j2 E_j3 R_q2 R_p2 d")

    # W[phys, p1, p2, i3, j3] REt[j1, j2, j3, q2, p2, i4]
    # -> WREt[j1, j2, phys, p1, q2, i3, i4]
    WREt = W_y.matvec(
        REt,
        contract=(
            "phys W_p1 R_p2 W_i3 E_j3,"
            " E_j1 E_j2 E_j3 R_q2 R_p2 R_i4"
            " -> E_j1 E_j2 phys W_p1 R_q2 W_i3 R_i4"
        ),
    )
    LTWREt = einops.einsum(
        L_y, jnp.conj(T_y), WREt,
        "L_q1 W_p1 L_i1 E_j1,"
        " phys L_q1 R_q2 T_i2 E_j2,"
        " sparse E_j1 E_j2 phys W_p1 R_q2 W_i3 R_i4"
        " -> sparse L_i1 T_i2 W_i3 R_i4",
    )
    return LTWREt

@jax.jit
def push_Eb_LWR_up(Et_y, L_y, W_y, R_y, T_y=None):
    if isinstance(W_y, (jax.Array, np.ndarray)):
        return push_Eb_LToTR_up(Et_y, L_y, W_y, R_y, T_y)
    if L_y.ndim == 3:
        return push_Eb_LWR_up_single(Et_y, L_y, W_y, R_y)
    if L_y.ndim == 4:
        return push_Eb_LWR_up_double(Et_y, L_y, T_y, W_y, R_y)    

@jax.jit
def push_Eb_LWR_up_single(Eb_y, L_y, W_y, R_y):
    """Push E_bot up one row (y -> y+1) for <L|W|R>.
    """
    Eb_out = jnp.einsum('p d u, s l t d -> s l t p u', R_y, Eb_y)
    # l=vertical L leg, q=horizontal L leg, t=vertical T leg, r=vertical R leg
    Eb_out = W_y.vecmat(Eb_out, contract='l T p r, q p T t -> l q t r')
    Eb_out = jnp.einsum('q l L, s l q t r -> s L t r', L_y, Eb_out)
    return Eb_out
    
@jax.jit
def push_Eb_LWR_up_double(Eb_y, L_y, T_y, W_y, R_y):
    """Push E_bot up one row (y->y+1) for <L|T*W|R>.
    """
    REb = einops.einsum(R_y, Eb_y, 
            "R_q2 R_p2 d u, s i1 i2 i3 d"
            " -> s i1 i2 i3 R_q2 R_p2 u")
    
    # REb[i1, i2, i3, q2, p2, j4] W[phys, p1, p2, i3, j3]
    # -> WREb[i1, i2, phys, p1, q2, j3, j4]
    WREb = W_y.vecmat(
        REb,
        contract=(
            "L_i1 T_i2 W_i3 R_q2 R_p2 R_j4,"
            " phys W_p1 R_p2 W_i3 W_j3"
            " -> L_i1 T_i2 phys W_p1 R_q2 W_j3 R_j4"
        ),
    )
    LTWREb = einops.einsum(
        L_y, jnp.conj(T_y), WREb,
        "L_q1 W_p1 L_i1 L_j1,"
        " phys L_q1 R_q2 T_i2 T_j2,"
        " sparse L_i1 T_i2 phys W_p1 R_q2 W_j3 R_j4"
        " -> sparse L_j1 T_j2 W_j3 R_j4")
    return LTWREb

@jax.jit
def dLWR_dL(Eb_y, Et_y, W_y, R_y, T_y=None):
    """Take grad w.r.t. L at a single row (y) of 
        <L|W|R>     (single-layer)
        <L|T*W|R>   (double-layer) 
    
    Single-layer:
        `T` is None.
        `W` is a perturbed (single-layer) site, as a BlockSparseCOO.

    Double-layer:
        `T` is the bra tensor (to be conjugated inside this fn).
        `W` is the perturbed ket site, as a BlockSparseCOO.
    """
    if isinstance(W_y, (jax.Array, np.ndarray)):
        if R_y.ndim == 3:
            return dLTR_dL_col(Eb_y, Et_y, W_y, R_y)
        if R_y.ndim == 4:
            return ltr_ops_double.dLTR_dL(Eb_y, Et_y, T_y[None], W_y[None], R_y[None])
        raise ValueError(f"Unsupported row rank for right tensor: {R_y.ndim}")

    Eb_y = Eb_y[:W_y.n_rows]
    Et_y = Et_y[:W_y.n_cols]
    if R_y.ndim == 3:
        out = jnp.einsum('p r R, s l t R -> s l t p r', R_y, Et_y)
        out = W_y.matvec(out, contract='q p b t, l t p r -> l q b r') 
        # out now has shape (s,l,q,b,r), w/ l=up leg of L, q=phys leg of l, b=dn leg of W, r=dn leg of R
        return jnp.einsum('s l q b r, s L b r -> q L l', out, Eb_y)

    elif R_y.ndim == 4:
        out = einops.einsum(R_y, Et_y,
                "R_q2 R_p2 d u, s E_j1 E_j2 E_j3 u"
                " -> s E_j1 E_j2 E_j3 R_q2 R_p2 d")
        
        out = W_y.matvec(
            out,
            contract=(
                "phys W_p1 R_p2 W_i3 E_j3,"
                " E_j1 E_j2 E_j3 R_q2 R_p2 R_i4"
                " -> E_j1 E_j2 phys W_p1 R_q2 W_i3 R_i4"
            ),
        )
        out = einops.einsum(
            jnp.conj(T_y),  out, 
            "phys L_q1 R_q2 T_i2 E_j2,"
            " sparse L_j1 E_j2 phys W_p1 R_q2 W_i3 R_i4"
            " -> sparse W_p1 L_q1 L_j1 T_i2 W_i3 R_i4")
        out = einops.einsum(
            out, Eb_y, 
            "s L_p1 L_q1 L_j1 T_i2 W_i3 R_i4,"
            " s L_i1 T_i2 W_i3 R_i4"
            " -> L_q1 L_p1 L_i1 L_j1")
        return out

    else:
        raise ValueError(f"Unsupported R[y] shape {R_y.shape}")


@jax.jit
def dLWR_dR(Eb_y, Et_y, L_y, W_y, T_y=None):
    """Take grad w.r.t. R at a single row of ``<L|W|R>`` or ``<L|T*W|R>``."""
    if isinstance(W_y, (jax.Array, np.ndarray)):
        if L_y.ndim == 3:
            return dLTR_dR_col(Eb_y, Et_y, L_y, W_y)
        if L_y.ndim == 4:
            return ltr_ops_double.dLTR_dR(Eb_y, Et_y, L_y[None], T_y[None], W_y[None])
        raise ValueError(f"Unsupported row rank for left tensor: {L_y.ndim}")

    Eb_y = Eb_y[:W_y.n_rows]
    Et_y = Et_y[:W_y.n_cols]
    if L_y.ndim == 3:
        out = jnp.einsum("q l L, s l b r -> s L q b r", L_y, Eb_y)
        out = W_y.vecmat(out, contract="l q b r, q p b t -> l p t r")
        # out now has shape (s, L, p, t, r), with R[p, r, R] left open.
        return jnp.einsum("s L p t r, s L t R -> p r R", out, Et_y)

    if L_y.ndim == 4:
        out = einops.einsum(
            L_y, jnp.conj(T_y), Eb_y,
            "L_q1 W_p1 L_i1 L_j1,"
            " phys L_q1 R_q2 T_i2 T_j2,"
            " sparse L_i1 T_i2 W_i3 R_i4"
            " -> sparse L_j1 T_j2 W_i3 phys W_p1 R_q2 R_i4",
        )
        out = W_y.vecmat(
            out,
            contract=(
                "L_j1 T_j2 W_i3 phys W_p1 R_q2 R_i4,"
                " phys W_p1 R_p2 W_i3 W_j3"
                " -> L_j1 T_j2 W_j3 R_q2 R_p2 R_i4"
            ),
        )
        return einops.einsum(
            out, Et_y,
            "s L_j1 T_j2 W_j3 R_q2 R_p2 R_i4,"
            " s L_j1 T_j2 W_j3 R_j4"
            " -> R_q2 R_p2 R_i4 R_j4",
        )

    raise ValueError(f"Unsupported L[y] shape {L_y.shape}")


def _boundary_vec(chi, dtype):
    return jnp.zeros((chi,), dtype=dtype).at[0].set(1.0)

def Eb_LWR_boundary_vec(L, W, R):
    """Return the full bottom environment boundary for ``<L|W|R>``.
    
        Works for both single and double layer
    """
    if isinstance(W, (jax.Array, np.ndarray)):
        if L.ndim == 4:
            shape = (L[0].shape[-2], W[0].shape[-2], R[0].shape[-2])
            return jnp.zeros(shape, dtype=L.dtype).at[0, 0, 0].set(1.0)
        if L.ndim == 5:
            shape = (L[0].shape[-2], W[0].shape[-2], W[0].shape[-2], R[0].shape[-2])
            return jnp.zeros(shape, dtype=L.dtype).at[0, 0, 0, 0].set(1.0)
        raise ValueError(f"Unsupported L[y] shape {L.shape}")

    n_rows = W.n_rows[0] if hasattr(W, "n_rows") else W[0].n_rows
    block_shape = W.block_shape
    if L.ndim == 4:
        shape = (n_rows, L[0].shape[-2], block_shape[-2], R[0].shape[-2])
        return jnp.zeros(shape, dtype=L.dtype).at[0, 0, 0, 0].set(1.0)

    shape = (
        n_rows,
        L[0].shape[-2],
        block_shape[-2],
        block_shape[-2],
        R[0].shape[-2],
    )
    return jnp.zeros(shape, dtype=L.dtype).at[0, 0, 0, 0, 0].set(1.0)


def Et_LWR_boundary_vec(L, W, R):
    """Return the full top environment boundary for ``<L|W|R>``."""
    if isinstance(W, (jax.Array, np.ndarray)):
        if L.ndim == 4:
            shape = (L[-1].shape[-1], W[-1].shape[-1], R[-1].shape[-1])
            return jnp.zeros(shape, dtype=L.dtype).at[0, 0, 0].set(1.0)
        if L.ndim == 5:
            shape = (L[-1].shape[-1], W[-1].shape[-1], W[-1].shape[-1], R[-1].shape[-1])
            return jnp.zeros(shape, dtype=L.dtype).at[0, 0, 0, 0].set(1.0)
        raise ValueError(f"Unsupported L[y] shape {L.shape}")

    n_cols = W.n_cols[-1] if hasattr(W, "n_cols") else W[-1].n_cols
    block_shape = W.block_shape
    if L.ndim == 4:
        shape = (n_cols, L[-1].shape[-1], block_shape[-1], R[-1].shape[-1])
        return jnp.zeros(shape, dtype=L.dtype).at[-1, 0, 0, 0].set(1.0)

    shape = (
        n_cols,
        L[-1].shape[-1],
        block_shape[-1],
        block_shape[-1],
        R[-1].shape[-1],
    )
    return jnp.zeros(shape, dtype=L.dtype).at[-1, 0, 0, 0, 0].set(1.0)


def Et_LWR_boundary_scalar(Et, W):
    """Extract the scalar from a fully pushed LWR top environment."""
    if isinstance(W, (jax.Array, np.ndarray)):
        return Et[(0,) * Et.ndim]
    return Et[(-1,) + (0,) * (Et.ndim - 1)]


@jax.jit
def env_b_LWR_col(L, W, R, T=None):
    """Return bottom environments below each row of ``<L|W|R>``."""
    if isinstance(W, (jax.Array, np.ndarray)):
        if R.ndim == 4:
            return ltr_ops_single.env_b(L, W, R)
        if R.ndim == 5:
            return ltr_ops_double.env_b(L, T, W, R)
        raise ValueError(f"Unsupported right column rank: {R.ndim}")

    Ny = L.shape[0]
    if T is None:
        T = [None]*Ny

    Eb = Eb_LWR_boundary_vec(L, W, R)
    envs = []
    for y in range(Ny):
        envs.append(Eb)
        Eb = push_Eb_LWR_up(Eb, L[y], W[y], R[y], T[y])
    return envs


@jax.jit
def env_t_LWR_col(L, W, R, T=None):
    """Return top environments above each row of ``<L|W|R>``."""
    if isinstance(W, (jax.Array, np.ndarray)):
        if R.ndim == 4:
            return ltr_ops_single.env_t(L, W, R)
        if R.ndim == 5:
            return ltr_ops_double.env_t(L, T, W, R)
        raise ValueError(f"Unsupported right column rank: {R.ndim}")
    Ny = L.shape[0]
    if T is None:
        T = [None]*Ny
    
    Et = Et_LWR_boundary_vec(L, W, R)
    envs = [None] * Ny
    for y in range(Ny - 1, -1, -1):
        envs[y] = Et
        Et = push_Et_LWR_down(Et, L[y], W[y], R[y], T[y])
    return envs


@jax.jit
def LWR_overlap_col_single_eff(L, W, R):
    """ Compute a single layer  <L|W|R>, but now has an additional *sparse* vertical index"""
    Et = Et_LWR_boundary_vec(L, W, R)
    Ly = L.shape[0]
    for y in range(Ly-1, -1, -1):
        Et = jnp.einsum('p k r, i l t r -> i l t p k', R[y], Et)
        Et = W[y].matvec(Et, contract = 'q p b t, l t p r -> l q b r')
        Et = jnp.einsum('q k l, i l q b r  -> i k b r', L[y], Et)
    
    return Et[0, 0, 0, 0]

@jax.jit
def LWR_overlap_col_single(L, W, R):
    """ Compute a single layer  <L|W|R>, but now has an additional *sparse* vertical index"""
    Et = Et_LWR_boundary_vec(L, W, R)
    Ly = L.shape[0]

#     for y in range(Ly-1, -1, -1):
#         contrib = jax.vmap(
#             lambda T, j: push_Et_LTR_down(Et[j], L[y], T, R[y])
#         )(W[y].blocks, W[y].cols)
#         Et = jax.ops.segment_sum(
#             contrib,
#             W[y].rows,
#             num_segments=W[y].n_rows,
#             indices_are_sorted=W[y].indices_are_row_sorted,
#         )
    
#     return Et[0, 0, 0, 0]

# @jax.jit
# def LWR_overlap_col_double(L, T, W, R):
#     """ Compute a double layer  <L|T.W|R>, where W is a BlockSparseMPO"""

    if isinstance(W, (jax.Array, np.ndarray)):
        return LToTR_overlap_col(L, T, W, R)

    Et = Et_LWR_boundary_vec(L, W, R)
    Ly = L.shape[0]

#     for y in range(Ly-1, -1, -1):
#         contrib = jax.vmap(
#             lambda w, j: ltr_ops_double.push_Et_down(Et[j], L[y], T[y], w, R[y])
#         )(W[y].blocks, W[y].cols)
#         Et = jax.ops.segment_sum(
#             contrib,
#             W[y].rows,
#             num_segments=W[y].n_rows,
#             indices_are_sorted=W[y].indices_are_row_sorted,
#         )
    
#     return Et[0, 0, 0, 0, 0]


@jax.jit
def LWR_overlap_col_double_eff(L, T, W, R):
    """Compute double-layer ``<L|T.W|R>`` with factored sparse row pushes."""

    if isinstance(W, (jax.Array, np.ndarray)):
        return LToTR_overlap_col(L, T, W, R)

    Et = Et_LWR_boundary_vec(L, W, R)
    Ly = L.shape[0]

    for y in range(Ly - 1, -1, -1):
        Et = push_Et_LWR_down_double(Et, L[y], T[y], W[y], R[y])

    return Et[0, 0, 0, 0, 0]


@jax.jit
def LWR_overlap_col_double_packed_vmap(L, T, W, R):
    """Compute double-layer ``<L|T.W|R>`` for a packed sparse column MPO."""
    sparse_dim = W.sparse_dim
    Et = Et_LWR_boundary_vec(L, W, R)
    Et0 = jnp.zeros(
        (sparse_dim, L.shape[-1], W.block_shape[-1], W.block_shape[-1], R.shape[-1]),
        dtype=L.dtype,
    ).at[: Et.shape[0]].set(Et)

    def row_body(Et, row_data):
        """Push the top environment down through one packed sparse row."""
        L_y, T_y, R_y, blocks_y, rows_y, cols_y = row_data

        def push_block(w, j):
            """Push one sparse block contribution down through this row."""
            return ltr_ops_double.push_Et_down(Et[j], L_y, T_y, w, R_y)

        contrib = jax.vmap(push_block)(blocks_y, cols_y)
        Et = jax.ops.segment_sum(
            contrib,
            rows_y,
            num_segments=sparse_dim,
            indices_are_sorted=W.indices_are_row_sorted,
        )
        return Et, None

    Et, _ = jax.lax.scan(
        row_body,
        Et0,
        (L, T, R, W.blocks, W.rows, W.cols),
        reverse=True,
    )
    return Et[0, 0, 0, 0, 0]


@jax.jit
def LWR_overlap_col_double_packed_vmap_eff(L, T, W, R):
    """Compute packed double-layer ``<L|T.W|R>`` with factored row pushes."""
    sparse_dim = W.sparse_dim
    Et = Et_LWR_boundary_vec(L, W, R)
    Et0 = jnp.zeros(
        (sparse_dim, L.shape[-1], W.block_shape[-1], W.block_shape[-1], R.shape[-1]),
        dtype=L.dtype,
    ).at[: Et.shape[0]].set(Et)

    def push_dn(Et, L_y, T_y, R_y, blocks_y, rows_y, cols_y):
        """Push Et down one packed row after first absorbing the right MPS."""
        REt = einops.einsum(
            R_y,
            Et,
            "R_q2 R_p2 R_i4 R_j4, sparse E_j1 E_j2 E_j3 R_j4"
            " -> sparse E_j1 E_j2 E_j3 R_q2 R_p2 R_i4",
        )

        def push_block(w, j):
            """Apply one sparse block to the right-absorbed environment."""
            return einops.einsum(
                w,
                REt[j],
                "site_d W_p1 R_p2 W_i3 E_j3,"
                " E_j1 E_j2 E_j3 R_q2 R_p2 R_i4"
                " -> E_j1 E_j2 site_d W_p1 R_q2 W_i3 R_i4",
            )

        WREt_contrib = jax.vmap(push_block)(blocks_y, cols_y)
        WREt = jax.ops.segment_sum(
            WREt_contrib,
            rows_y,
            num_segments=sparse_dim,
            indices_are_sorted=W.indices_are_row_sorted,
        )
        return einops.einsum(
            L_y,
            jnp.conj(T_y),
            WREt,
            "L_q1 W_p1 L_i1 E_j1,"
            " site_d L_q1 R_q2 T_i2 E_j2,"
            " sparse E_j1 E_j2 site_d W_p1 R_q2 W_i3 R_i4"
            " -> sparse L_i1 T_i2 W_i3 R_i4",
        )

    def row_body(Et, row_data):
        """Scan body for a factored packed sparse row push."""
        L_y, T_y, R_y, blocks_y, rows_y, cols_y = row_data
        Et = push_dn(Et, L_y, T_y, R_y, blocks_y, rows_y, cols_y)
        return Et, None

    Et, _ = jax.lax.scan(
        row_body,
        Et0,
        (L, T, R, W.blocks, W.rows, W.cols),
        reverse=True,
    )
    return Et[0, 0, 0, 0, 0]
