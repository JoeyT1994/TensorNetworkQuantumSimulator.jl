import einops
import jax
import jax.numpy as jnp


BALANCED_CONTRACT = False


# CTM uses a different index convention from MPS.column_primitives. The latter is
# organized as a left-to-right one-dimensional column contraction, whereas CTM
# is organized around C4-invariant sites and plaquettes. We therefore order
# both site legs and boundary tensors counterclockwise.
#

# Geometry: cartesian coordinates T[x, y]. x --> x+1 is the "right". y --> y+1 is "up"
#
#
# Site-leg order:
#
#                         u = a1
#                            |
#               l = a2 --- T[x, y] --- r = a0
#                            |
#                         d = a3
#
# Thus a single-layer site is T[x, y, r, u, l, d], equivalently
# T[x, y, a0, a1, a2, a3]. CTMState stores a double-layer site as the physical
# ket t[p, a0, a1, a2, a3], with tb = conj(t). Local primitives also accept an
# explicit (tb, t) pair when the bra and ket must be independent.

# Boundary ring and its indices:
#
#                   j1         i1
#              c1 ------ A1 ------ c0
#              |          |          |
#           i2 |         a1          | j0
#              |          |          |
#              A2---a2--- T ---a0---A0
#              |          |          |
#           j2 |         a3          | i0
#              |          |          |
#              c2 ------ A3 ------ c3
#                   i3         j3
#
#
#     Tensors are stored as Ak[x, y, a, i, j]  ck[x, y, j, i]
#
#
# We x-y index the boundary data A and c according to which of these 1x1 plaquettes they live in. So all the tesnors 
# in the above diagram are at the same x-y
#
#
# The ring is traversed counterclockwise as A0 -> A1 -> A2 -> A3. Boundary
# tensor A_k connects corner c_{k-1} to c_k, while
#
#     c_k[..., j_k, i_{k+1}]
#
# joins the outgoing leg j_k of A_k to the incoming leg i_{k+1} of A_{k+1};
# corner and boundary labels are understood modulo 4. We store
#
#     state.A = (A0, A1, A2, A3),    A_k[x, y, ..., i_k, j_k].
#     state.c = (c0, c1, c2, c3),    c_k[x, y, j_k, i_{k+1}].

# Local inclusion-exclusion networks. Every line represents a contracted bond;
# an omitted tensor position has been collapsed to a direct bond.

# Z11: site and all four boundary tensors
#
#              c1 ------ A1 ------ c0
#              |          |          |
#              A2 ------- T ------- A0
#              |          |          |
#              c2 ------ A3 ------ c3
#
# Single layer:
#
#   Z11[x, y] = T[x, y, a0, a1, a2, a3]
#               prod_k A_k[x, y, a_k, i_k, j_k]  c_k[x, y, j_k, i_{k+1}]


# For a double layer, each site leg is split into a bra leg ab_k and ket leg
# a_k:
#
#   Z11[x, y] = tb[x, y, p, {ab_k}] t[x, y, p, {a_k}]
#               prod_k A_k[x, y, ab_k, a_k, i_k, j_k]
#                      c_k[x, y, j_k, i_{k+1}].

# Z01[x, y]: side boundary tensors only
#
#              c1 ---------------- c0
#              |                    |
#              A2[x+1, y]           A0[x, y]
#              |                    |
#              c2 ---------------- c3

# Z10[x, y]: top and bottom boundary tensors only
#
#             c1[x,y] ----- A1[x, y] ---- c0[x,y]
#              |             |             |
#             c2[x,y+1] --- A3[x, y+1] -- c3[x, y+1]

# Z00[x, y]: corners only
#
#              c1[x+1, y] -------- c0[x, y]
#              |                    |
#              c2[x+1, y+1] ------ c3[x, y+1]

def _contract_Z11_single(T, A, c):
    """Contract one single-layer site and its complete CTM ring."""
    Ac = []
    for k in range(4):
        # (A_k c_k)_{a_k i_k i_{k+1}}.
        Ac.append(einops.einsum(A[k], c[k], "a i j, j n -> a i n"))
    Ac_pair = []
    for k in range(0, 4, 2):
        # (A_k c_k)_{a0 i m} (A_{k+1} c_{k+1})_{a1 m j}.
        Ac_pair.append(
            einops.einsum(
                Ac[k], Ac[k + 1],
                "a0 i m, a1 m j -> a0 a1 i j",
            )
        )
    return einops.einsum(
        T, Ac_pair[0], Ac_pair[1],
        "a0 a1 a2 a3, a0 a1 i0 i2, a2 a3 i2 i0 ->",
    )


def _contract_Z11_double(T, A, c):
    """Contract one physical ket or independent bra/ket site with its CTM ring."""
    if isinstance(T, tuple):
        tb, t = T
    else:
        t = T
        tb = jnp.conj(t)
    Ac = []
    for k in range(4):
        # (A_k c_k)_{b_k a_k i_k i_{k+1}}.
        Ac.append(einops.einsum(A[k], c[k], "b a i j, j n -> b a i n"))
    Ac_pair = []
    for k in range(0, 4, 2):
        # Join (A_k c_k) and (A_{k+1} c_{k+1}) across i_{k+1}.
        Ac_pair.append(
            einops.einsum(
                Ac[k], Ac[k + 1],
                "b0 a0 i m, b1 a1 m j -> b0 b1 a0 a1 i j",
            )
        )
    return einops.einsum(
        tb, t, Ac_pair[0], Ac_pair[1],
        "p b0 b1 b2 b3, p a0 a1 a2 a3, "
        "b0 b1 a0 a1 i0 i2, b2 b3 a2 a3 i2 i0 ->",
    )


def contract_Z11(T, A, c):
    """Dispatch the one-site contraction from the local edge-tensor rank."""
    if A[0].ndim == 3:
        return _contract_Z11_single(T, A, c)
    return _contract_Z11_double(T, A, c)


def _contract_Z01_single(A02, c):
    """Contract one single-layer horizontal-edge CTM network."""
    c0, c1, c2, c3 = c
    # Rotate k -> k - 1 so the horizontal edge has the Z10 argument order.
    return _contract_Z10_single(A02, (c3, c0, c1, c2))


def _contract_Z01_double(A02, c):
    """Contract one double-layer horizontal-edge CTM network."""
    A0, A2 = A02
    A0 = A0.reshape((-1,) + A0.shape[-2:])
    A2 = A2.reshape((-1,) + A2.shape[-2:])
    return _contract_Z01_single((A0, A2), c)


def contract_Z01(A02, c):
    """Dispatch the horizontal-edge contraction from an edge-tensor rank."""
    if A02[0].ndim == 3:
        return _contract_Z01_single(A02, c)
    return _contract_Z01_double(A02, c)


def _contract_Z10_single(A13, c):
    """Contract one single-layer vertical-edge CTM network."""
    A1, A3 = A13
    c0, c1, c2, c3 = c
    if BALANCED_CONTRACT:
        # ((c2 (A3 c3)) (c0 (A1 c1))): balanced FP64 comparison path.
        A3c3 = einops.einsum(A3, c3, "a i3 j3, j3 v -> a i3 v")
        left = einops.einsum(c2, A3c3, "u i3, a i3 v -> a u v")
        A1c1 = einops.einsum(A1, c1, "a i1 j1, j1 u -> a i1 u")
        right = einops.einsum(c0, A1c1, "v i1, a i1 u -> a v u")
    else:
        # ((A1 (c1 c2)) (A3 (c3 c0))): close the corner-only bonds first.
        c12 = einops.einsum(c1, c2, "j1 s2, s2 v -> j1 v")
        left = einops.einsum(A1, c12, "a u j1, j1 v -> a u v")
        c30 = einops.einsum(c3, c0, "j3 s0, s0 u -> j3 u")
        right = einops.einsum(A3, c30, "a v j3, j3 u -> a v u")
    return einops.einsum(left, right, "a u v, a v u ->")


def _contract_Z10_double(A13, c):
    """Contract one double-layer vertical-edge CTM network."""
    A1, A3 = A13
    A1 = A1.reshape((-1,) + A1.shape[-2:])
    A3 = A3.reshape((-1,) + A3.shape[-2:])
    return _contract_Z10_single((A1, A3), c)


def contract_Z10(A13, c):
    """Dispatch the vertical-edge contraction from an edge-tensor rank."""
    if A13[0].ndim == 3:
        return _contract_Z10_single(A13, c)
    return _contract_Z10_double(A13, c)



def contract_Z00(c):
    """Contract four corners; single and double layers have the same kernel."""
    c0, c1, c2, c3 = c
    return einops.einsum(
        c0, c1, c2, c3,
        "s0 s1, s1 s2, s2 s3, s3 s0 ->",
    )


def _compute_Z(T, A, c, finite=(True, True)):
    """Compute the vmapped CTM inclusion-exclusion functional.

    ``Z11`` is summed over all sites. ``Z01``, ``Z10``, and ``Z00`` are
    summed over horizontal bonds, vertical bonds, and plaquettes,
    respectively. A periodic axis wraps its shifted tensors, while a finite
    axis omits the final origin along that direction.
    """

    Nx, Ny = A[0].shape[:2]
    nx = Nx - int(finite[0])
    ny = Ny - int(finite[1])

    Z11 = jax.vmap(jax.vmap(contract_Z11))(T, A, c)

    A02 = (
        A[0][:nx, :],
        jnp.roll(A[2], -1, axis=0)[:nx, :],
    )
    c01 = (
        c[0][:nx, :],
        jnp.roll(c[1], -1, axis=0)[:nx, :],
        jnp.roll(c[2], -1, axis=0)[:nx, :],
        c[3][:nx, :],
    )
    Z01 = jax.vmap(jax.vmap(contract_Z01))(A02, c01)

    A13 = (
        A[1][:, :ny],
        jnp.roll(A[3], -1, axis=1)[:, :ny],
    )
    c10 = (
        c[0][:, :ny],
        c[1][:, :ny],
        jnp.roll(c[2], -1, axis=1)[:, :ny],
        jnp.roll(c[3], -1, axis=1)[:, :ny],
    )
    Z10 = jax.vmap(jax.vmap(contract_Z10))(A13, c10)

    c00 = (
        c[0][:nx, :ny],
        jnp.roll(c[1], -1, axis=0)[:nx, :ny],
        jnp.roll(jnp.roll(c[2], -1, axis=0), -1, axis=1)[:nx, :ny],
        jnp.roll(c[3], -1, axis=1)[:nx, :ny],
    )
    Z00 = jax.vmap(jax.vmap(contract_Z00))(c00)

    return jnp.sum(Z11) - jnp.sum(Z01) - jnp.sum(Z10) + jnp.sum(Z00)

# Collectively X = (A, c). CTMState owns the finite-boundary split: exterior
# A and c are fixed context rather than active AD degrees of freedom.



#The CTM:    C[a3 i0, a2 j1]

#      j1         i1
#     ------ A1 ------ c0
#            |          |
# col       a1          | j0
#            |          |
#     -a2--- T ---a0---A0
#            |          |
#           a3          | i0
#
#                 row


def _construct_CTM_single(T, A01, c0, k=0):
    """Construct single-layer quadrant ``k`` without rotating ``T``."""
    A0, A1 = A01
    A0c0 = einops.einsum(A0, c0, "a0 i0 j0, j0 i1 -> a0 i0 i1")
    A01c0 = einops.einsum(
        A0c0, A1,
        "a0 i0 i1, a1 i1 j1 -> a0 a1 i0 j1",
    )
    legs = "abcd"
    legs = legs[-k:] + legs[:-k]
    # T_{a0 a1 a2 a3} (A_k A_{k+1} c_k)_{a_k a_{k+1} m n}
    #     -> C_k[a_{k+3} m, a_{k+2} n].
    C = jnp.einsum(
        f"{legs},abmn->dmcn",
        T,
        A01c0,
        optimize="optimal",
    )
    return C.reshape((C.shape[0] * C.shape[1], C.shape[2] * C.shape[3]))


def _construct_CTM_double(T, A01, c0, k=0):
    """Construct physical or independent double-layer quadrant ``k``."""
    if isinstance(T, tuple):
        tb, t = T
    else:
        t = T
        tb = jnp.conj(t)
    A0, A1 = A01
    A0c0 = einops.einsum(
        A0, c0,
        "b0 a0 i0 j0, j0 i1 -> b0 a0 i0 i1",
    )
    A01c0 = einops.einsum(
        A0c0, A1,
        "b0 a0 i0 i1, b1 a1 i1 j1 -> b0 b1 a0 a1 i0 j1",
    )
    bra = "abcd"
    ket = "efgh"
    bra = bra[-k:] + bra[:-k]
    ket = ket[-k:] + ket[:-k]
    # tb_{p b0...b3} t_{p a0...a3} (A_k A_{k+1} c_k)
    #     -> C_k[b_{k+3} a_{k+3} m, b_{k+2} a_{k+2} n].
    C = jnp.einsum(
        f"p{bra},p{ket},abefmn->dhmcgn",
        tb,
        t,
        A01c0,
        optimize="optimal",
    )
    return C.reshape(
        (
            C.shape[0] * C.shape[1] * C.shape[2],
            C.shape[3] * C.shape[4] * C.shape[5],
        )
    )


def construct_CTM(T, A01, c0):
    """Construct a dense single- or double-layer corner transfer matrix."""
    if A01[0].ndim == 3:
        return _construct_CTM_single(T, A01, c0)
    return _construct_CTM_double(T, A01, c0)


def construct_CTM_k(T, A, c, k):
    """Construct quadrant ``k`` directly from one site and its cyclic ring."""
    A01 = (A[k], A[(k + 1) % 4])
    if A[0].ndim == 3:
        return _construct_CTM_single(T, A01, c[k], k=k)
    return _construct_CTM_double(T, A01, c[k], k=k)


def construct_CTMs(T, A, c):
    """Construct ``(CTM0, CTM1, CTM2, CTM3)`` in cyclic corner order."""
    # A literal vmap would require materializing four cyclic transposes of T:
    # k changes the contraction dimensions, not merely the input data.
    return tuple(construct_CTM_k(T, A, c, k) for k in range(4))


def _CTM_matvec_single(v, T, A01, c0):
    """Apply a factorized single-layer CTM to vectors on trailing axes."""
    A0, A1 = A01
    rhs_shape = v.shape[1:]
    v = v.reshape((T.shape[2], A1.shape[-1], *rhs_shape))
    A0c0 = einops.einsum(A0, c0, "a0 i0 j0, j0 i1 -> a0 i0 i1")
    # T[a0,a1,a2,a3] A0c0[a0,i0,i1] A1[a1,i1,j1]
    # v[a2,j1,...] -> y[a3,i0,...].
    y = jnp.einsum(
        "abcd,aij,bjl,cl...->di...",
        T, A0c0, A1, v,
        optimize="optimal",
    )
    return y.reshape((-1, *rhs_shape))


def _CTM_matvec_double(v, T, A01, c0):
    """Apply a factorized double-layer CTM to vectors on trailing axes."""
    if isinstance(T, tuple):
        tb, t = T
    else:
        t = T
        tb = jnp.conj(t)
    A0, A1 = A01
    rhs_shape = v.shape[1:]
    v = v.reshape((tb.shape[3], t.shape[3], A1.shape[-1], *rhs_shape))
    A0c0 = einops.einsum(
        A0, c0,
        "b0 a0 i0 j0, j0 i1 -> b0 a0 i0 i1",
    )
    # tb[p,b0,b1,b2,b3] t[p,a0,a1,a2,a3]
    # A0c0[b0,a0,i0,i1] A1[b1,a1,i1,j1]
    # v[b2,a2,j1,...] -> y[b3,a3,i0,...].
    y = jnp.einsum(
        "pbcde,pfghi,bfjk,cgkl,dhl...->eij...",
        tb, t, A0c0, A1, v,
        optimize="optimal",
    )
    return y.reshape((-1, *rhs_shape))


def CTM_matvec(v, T, A01, c0):
    """Apply a factorized CTM to ``v`` with arbitrary trailing dimensions."""
    if A01[0].ndim == 3:
        return _CTM_matvec_single(v, T, A01, c0)
    return _CTM_matvec_double(v, T, A01, c0)
