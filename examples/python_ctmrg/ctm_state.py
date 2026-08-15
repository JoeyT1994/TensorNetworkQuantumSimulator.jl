"""CTM site and counterclockwise boundary-ring state."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from column_primitives import env_b_LR_col, env_b_LTR_col, env_t_LR_col, env_t_LTR_col
from ctm_primitives import _compute_Z
from linalg.jax_linalg import check_schur

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


def _periodic_schur_gauges(c_plaquette, CTM_eig_cutoff, rank_template):
    """Return cutoff-projected periodic Schur gauges for every plaquette."""
    from linalg.periodic_schur import (
        periodic_schur_D,
        periodic_schur_Z,
        reorder_periodic_schur_D,
        reorder_periodic_schur_Z,
    )

    c_plaquette = np.asarray(c_plaquette)
    CTM_eig_cutoff = float(np.asarray(CTM_eig_cutoff))
    rank = np.empty_like(np.asarray(rank_template))
    GR = np.empty_like(c_plaquette)
    GL = np.empty_like(c_plaquette)
    c_schur = np.empty_like(c_plaquette)
    chi = c_plaquette.shape[-1]
    is_complex = np.iscomplexobj(c_plaquette)

    for site in np.ndindex(rank.shape):
        if is_complex:
            T, Z, alpha, beta, scale = periodic_schur_Z(c_plaquette[site])
            eig_abs = np.abs(alpha / beta) * np.exp2(scale)
            reorder = reorder_periodic_schur_Z
        else:
            T, Z, wr, wi = periodic_schur_D(c_plaquette[site])
            eig_abs = np.hypot(wr, wi)
            reorder = reorder_periodic_schur_D

        select = eig_abs >= CTM_eig_cutoff * np.max(eig_abs)
        rank_site = int(np.count_nonzero(select))
        leading = np.arange(chi) < rank_site
        if not np.array_equal(select, leading):
            T, Z = reorder(T, Z, select)

        # Set the discarded Schur coordinates exactly to zero in c, GR, and GL.
        T[:, rank_site:, :] = 0
        T[:, :, rank_site:] = 0
        Z[:, :, rank_site:] = 0
        GR[site] = Z
        GL[site] = np.swapaxes(np.conj(Z), -1, -2)
        c_schur[site] = T
        rank[site] = rank_site
    return GR, GL, c_schur, rank


@jax.tree_util.register_pytree_node_class
class CTMState(NamedTuple):
    """CTM tensors, plaquette ranks, and finite-versus-periodic axis flags."""

    T: jax.Array
    A: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    c: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    rank: jax.Array
    finite: tuple[int, int] = (1, 1)

    def tree_flatten(self):
        """Treat ``finite`` as static pytree metadata."""

        children = (self.T, self.A, self.c, self.rank)
        return children, self.finite

    @classmethod
    def tree_unflatten(cls, finite, children):
        """Reconstruct a CTM state from dynamic arrays and static axis flags."""

        T, A, c, rank = children
        return cls(T=T, A=A, c=c, rank=rank, finite=finite)

    @staticmethod
    def rank_from_c(c, finite=(1, 1), rtol=1e-14):
        """Return the numerical small-c cycle rank at every plaquette."""

        fx, fy = map(int, finite)
        nx = c[0].shape[0] - fx
        ny = c[0].shape[1] - fy
        x = jnp.arange(nx)
        y = jnp.arange(ny)

        def shifted(ck, dx, dy):
            """Gather one corner field at a fixed plaquette offset."""

            out = jnp.take(ck, (x + dx) % ck.shape[0], axis=0)
            return jnp.take(out, (y + dy) % ck.shape[1], axis=1)

        c_inner = tuple(
            shifted(c[k], *offset)
            for k, offset in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)))
        )
        cycle = c_inner[0] @ c_inner[1] @ c_inner[2] @ c_inner[3]
        s = jnp.linalg.svd(cycle, compute_uv=False)
        return jnp.sum(s > rtol * jnp.max(s, axis=-1, keepdims=True), axis=-1)

    @classmethod
    def init(
        cls,
        T,
        chi,
        random_scale=0.0,
        key=jax.random.PRNGKey(0),
        finite=(1, 1),
    ):
        """Initialize a CTM ring with trivial basis edge tensors.

        The four directional T-space dimensions must be equal. Every edge and
        corner has boundary dimension ``chi``. Each edge has one at its
        all-zero T-bond and boundary indices and is zero elsewhere. Each corner
        has only ``[..., 0, 0] = 1``. A double-layer state stores only the
        physical ket ``T``; its bra is ``conj(T)``. ``random_scale`` adds
        reproducible random noise to every active A and c entry. Fixed exterior
        boundary tensors remain exactly one-hot.
        """

        Nx, Ny = T.shape[:2]
        dtype = T.dtype
        num_layers = T.ndim - 5
        T_space_dims = T.shape[-4:]
        if len(set(T_space_dims)) != 1:
            raise ValueError(
                f"CTMState requires equal T-space dimensions, got {T_space_dims}"
            )
        bond_shape = (T_space_dims[0],) * num_layers
        keys = jax.random.split(key, 8)
        fx, fy = finite
        nx, ny = Nx - fx, Ny - fy
        A_active = (
            (slice(0, nx), slice(None)),
            (slice(None), slice(0, ny)),
            (slice(fx, None), slice(None)),
            (slice(None), slice(fy, None)),
        )
        c_active = (
            (slice(0, nx), slice(0, ny)),
            (slice(fx, None), slice(0, ny)),
            (slice(fx, None), slice(fy, None)),
            (slice(0, nx), slice(fy, None)),
        )

        A = []
        c = []
        for k in range(4):
            Ak = jnp.zeros(
                (Nx, Ny, *bond_shape, chi, chi),
                dtype=dtype,
            )
            zero_index = (slice(None), slice(None)) + (0,) * (len(bond_shape) + 2)
            Ak = Ak.at[zero_index].set(1)
            noise = jax.random.normal(keys[k], Ak.shape, dtype=dtype)
            Ak = Ak.at[A_active[k]].add(random_scale * noise[A_active[k]])
            A.append(Ak)

            ck = jnp.zeros(
                (Nx, Ny, chi, chi),
                dtype=dtype,
            )
            ck = ck.at[..., 0, 0].set(1)
            noise = jax.random.normal(keys[k + 4], ck.shape, dtype=dtype)
            ck = ck.at[c_active[k]].add(random_scale * noise[c_active[k]])
            c.append(ck)

        c = tuple(c)
        rank = cls.rank_from_c(c, finite=finite)
        return cls(T=T, A=tuple(A), c=c, rank=rank, finite=finite)

    @classmethod
    def from_bp_state(cls, state, pinv_rtol=1e-14):
        """Convert an open ordinary BP state into its CTM boundary ring.

        BP tensors use site-leg order ``(l, r, d, u)`` and upward MPS bonds
        ``(i, j)``. CTM uses counterclockwise site-leg order ``(r, u, l, d)``
        and stores every edge as ``A_k[..., i_k, j_k]``. The nontrivial
        corners are the top/bottom environments of ``<L[x+1]|R[x]>``; 
        ``c1`` and ``c2`` are identity glue corners. The BP boundaries are 
        first scaled so every LR and LTR overlap is one, making the additive 
        CTM functional equal the BP partition ratio.
        """

        state, _ = state.scale_to_uniform_LR(state.z_LR(), state.z_LTR())
        L, R = state.L, state.R
        Eb_LR = jax.vmap(env_b_LR_col, in_axes=(0, 0))(L[1:], R[:-1])
        Et_LR = jax.vmap(env_t_LR_col, in_axes=(0, 0))(L[1:], R[:-1])
        Eb_LTR = jax.vmap(env_b_LTR_col, in_axes=(0, 0, 0))(L, state.T, R)
        Et_LTR = jax.vmap(env_t_LTR_col, in_axes=(0, 0, 0))(L, state.T, R)

        if state.T.ndim == 6:
            # BP T[l,r,d,u] -> CTM T[r,u,l,d].
            T = jnp.transpose(state.T, (0, 1, 3, 5, 2, 4))
        else:
            # BP t[p,l,r,d,u] -> CTM t[p,r,u,l,d].
            T = jnp.transpose(state.T, (0, 1, 2, 4, 6, 3, 5))

        chi = L.shape[-1]
        ctm = cls.init(T, chi=chi)
        identity = jnp.eye(chi, dtype=T.dtype)
        c0 = ctm.c[0].at[:-1].set(jnp.swapaxes(Et_LR, -1, -2))
        c1 = ctm.c[1].at[1:, :-1].set(identity)
        c2 = ctm.c[2].at[1:, 1:].set(identity)
        c3 = ctm.c[3].at[:-1].set(Eb_LR)

        # E^{-1}[r,n] changes the R overlap index into the next-column L
        # index. The final x column uses the one-hot exterior c0/c3 corners.
        Et_inv = jnp.linalg.pinv(jnp.swapaxes(c0, -1, -2), rtol=pinv_rtol)
        Eb_inv = jnp.linalg.pinv(c3, rtol=pinv_rtol)
        if state.T.ndim == 6:
            # Et_LTR[l,u,r] Et_inv[r,i1] -> A1[u,i1,j1=l].
            A1 = jnp.einsum("...lar,...rn->...anl", Et_LTR, Et_inv)
            # Eb_LTR[l,d,r] Eb_inv[r,j3] -> A3[d,i3=l,j3].
            A3 = jnp.einsum("...lar,...rn->...aln", Eb_LTR, Eb_inv)
        else:
            # Et_LTR[l,ub,uk,r] Et_inv[r,i1] -> A1[ub,uk,i1,j1=l].
            A1 = jnp.einsum("...lbar,...rn->...banl", Et_LTR, Et_inv)
            # Eb_LTR[l,db,dk,r] Eb_inv[r,j3] -> A3[db,dk,i3=l,j3].
            A3 = jnp.einsum("...lbar,...rn->...baln", Eb_LTR, Eb_inv)

        # R is traversed bottom-to-top as A0[i0,j0], while L is traversed
        # top-to-bottom as A2[i2,j2] and therefore reverses its MPS bonds.
        A0 = R
        A2 = jnp.swapaxes(L, -1, -2)
        c = (c0, c1, c2, c3)
        rank = cls.rank_from_c(c)
        return cls(T=T, A=(A0, A1, A2, A3), c=c, rank=rank)

    def to_LR(self):
        """Convert CTM state to L, R, LTR and LR tensors.
        """
        R = self.A[0]
        L = jnp.swapaxes(self.A[2], -1, -2)

        LR_bot = jax.vmap(env_b_LR_col, in_axes=(0, 0))(L[1:], R[:-1])
        LR_top = jax.vmap(env_t_LR_col, in_axes=(0, 0))(L[1:], R[:-1])

        if self.T.ndim == 6:
            T_bp = jnp.transpose(self.T, (0, 1, 4, 2, 5, 3))
        else:
            T_bp = jnp.transpose(self.T, (0, 1, 2, 5, 3, 6, 4))

        LTR_bot = jax.vmap(env_b_LTR_col, in_axes=(0, 0, 0))(L, T_bp, R)
        LTR_top = jax.vmap(env_t_LTR_col, in_axes=(0, 0, 0))(L, T_bp, R)

        return L, R, LR_bot, LR_top, LTR_bot, LTR_top


    def active_X(self):
        """Return the active interior environment tensors ``X = (A, c)``."""

        Nx, Ny = self.T.shape[:2]
        fx, fy = map(int, self.finite)
        nx, ny = Nx - fx, Ny - fy

        A = (
            self.A[0][:nx, :],
            self.A[1][:, :ny],
            self.A[2][fx:, :],
            self.A[3][:, fy:],
        )
        c = (
            self.c[0][:nx, :ny],
            self.c[1][fx:, :ny],
            self.c[2][fx:, fy:],
            self.c[3][:nx, fy:],
        )
        return A, c

    def _insert_X(self, X):
        """Insert active ``X = (A, c)`` into this state's fixed boundaries."""

        A_X, c_X = X
        Nx, Ny = self.T.shape[:2]
        fx, fy = self.finite
        nx, ny = Nx - fx, Ny - fy

        A = (
            self.A[0].at[:nx, :].set(A_X[0]),
            self.A[1].at[:, :ny].set(A_X[1]),
            self.A[2].at[fx:, :].set(A_X[2]),
            self.A[3].at[:, fy:].set(A_X[3]),
        )
        c = (
            self.c[0].at[:nx, :ny].set(c_X[0]),
            self.c[1].at[fx:, :ny].set(c_X[1]),
            self.c[2].at[fx:, fy:].set(c_X[2]),
            self.c[3].at[:nx, fy:].set(c_X[3]),
        )
        return A, c

    def Z(self, T=None, X=None):
        """Evaluate Z for this state or for explicit active ``(T, X)`` data.

        ``state.Z()`` uses the complete stored state. ``state.Z(T, X)``
        inserts the active interior environment ``X`` into the fixed boundary
        values stored by this state, so only ``T`` and ``X`` are AD arguments.
        """

        if T is None:
            return _compute_Z(self.T, self.A, self.c, self.finite)
        A, c = self._insert_X(X)
        return _compute_Z(T, A, c, self.finite)

    def gauge_fix_env(self, GR, GL):
        """Transform the four ancillary bonds of every ``Z00`` plaquette.

        ``GR[x, y, k]`` and ``GL[x, y, k]`` act on bond ``k`` of
        plaquette ``(x, y)`` and have leading shape ``self.rank.shape``.  For
        ``offset = ((0,0), (1,0), (1,1), (0,1))``, the corner action is

        ``c[k][p + offset[k]] -> GL[p,k] c[k] GR[p,k+1]``.

        The matching incoming and outgoing boundary legs transform so every
        ``A-c`` contraction includes the same bond action.  When ``GL`` and
        ``GR`` are inverses this is a gauge transformation and preserves every
        contraction.  Rank-projecting Schur transforms instead carry exact
        zero rows and columns through all incident tensors.  Missing
        finite-boundary plaquette transforms are identities, while periodic
        axes wrap.  ``T``, ``rank``, and ``finite`` are unchanged.
        """

        Nx, Ny = self.T.shape[:2]
        nx, ny = self.rank.shape
        chi = self.c[0].shape[-1]
        offsets = ((0, 0), (1, 0), (1, 1), (0, 1))

        def full_gauge_field(g):
            """Embed a plaquette gauge field with identity finite padding."""

            identity = jnp.eye(chi, dtype=g.dtype)
            g_full = jnp.broadcast_to(identity, (Nx, Ny, 4, chi, chi))
            return g_full.at[:nx, :ny].set(g)

        def at_corner(g, k, corner):
            """Shift plaquette field ``g[...,k]`` onto corner ``corner``."""

            return jnp.roll(
                g[..., k, :, :],
                shift=offsets[corner],
                axis=(0, 1),
            )

        GR_full = full_gauge_field(GR)
        GL_full = full_gauge_field(GL)

        A = []
        c = []
        for k in range(4):
            # GL_{p_in,k} A_k[q] GR_{p_out,k}, with
            # p_in=q-offset[k-1] and p_out=q-offset[k].
            g_in_left = at_corner(GL_full, k, (k - 1) % 4)
            g_out_right = at_corner(GR_full, k, k)
            Ak = _apply_gauge(
                self.A[k],
                jnp.swapaxes(g_in_left, -1, -2),
                -2,
            )
            Ak = _apply_gauge(Ak, g_out_right, -1)
            A.append(Ak)

            # c_k[p+offset[k]] -> GL_{p,k} c_k GR_{p,k+1}.
            g_left = at_corner(GL_full, k, k)
            g_right = at_corner(GR_full, (k + 1) % 4, k)
            ck = _apply_gauge(
                self.c[k],
                jnp.swapaxes(g_left, -1, -2),
                -2,
            )
            ck = _apply_gauge(ck, g_right, -1)
            c.append(ck)

        return self._replace(A=tuple(A), c=tuple(c))

    def check_schur_gauge(self):
        """Return whether every plaquette corner cycle is in Schur gauge."""

        Nx, Ny = self.T.shape[:2]
        nx, ny = self.rank.shape
        x = jnp.arange(nx)
        y = jnp.arange(ny)
        offsets = ((0, 0), (1, 0), (1, 1), (0, 1))

        def shifted(ck, dx, dy):
            """Gather one corner field at its Z00 plaquette offset."""

            out = jnp.take(ck, (x + dx) % Nx, axis=0)
            return jnp.take(out, (y + dy) % Ny, axis=1)

        c_plaquette = jnp.stack(
            tuple(
                shifted(self.c[k], *offsets[k])
                for k in range(4)
            ),
            axis=2,
        )
        is_schur = jax.vmap(jax.vmap(check_schur))(c_plaquette)
        return jnp.all(is_schur)

    def schur_gauge(self, CTM_eig_cutoff=1e-15):
        """Put every ``Z00`` corner cycle into truncated periodic Schur gauge.

        At plaquette ``(x, y)``, the cyclic factors are ``c0[x,y]``,
        ``c1[x+1,y]``, ``c2[x+1,y+1]``, and ``c3[x,y+1]``.  They are
        decomposed as ``Z[k]^H c[k] Z[k+1] = S[k]``. Eigenmodes satisfying
        ``abs(eig) >= CTM_eig_cutoff * max_abs_eig`` are reordered first.
        Rows and columns after that retained sector are set exactly to zero in
        ``S``, ``GR = Z``, and ``GL = GR^H`` before the transforms are applied
        to all incident corner and boundary tensors through
        :meth:`gauge_fix_env`.

        After applying the gauge to every incident tensor, the transformed
        corners are overwritten by the returned Schur factors.  This preserves
        their exact structural zeros instead of regenerating roundoff below the
        triangular bands through matrix multiplication.

        The full ``chi``-dimensional cycle is factored without using the
        incoming ``self.rank`` as a coordinate mask. The returned rank is the
        number of eigenmodes retained by ``CTM_eig_cutoff``.
        """

        Nx, Ny = self.T.shape[:2]
        nx, ny = self.rank.shape
        x = jnp.arange(nx)
        y = jnp.arange(ny)
        offsets = ((0, 0), (1, 0), (1, 1), (0, 1))

        def shifted(ck, dx, dy):
            """Gather one corner field at its Z00 plaquette offset."""

            out = jnp.take(ck, (x + dx) % Nx, axis=0)
            return jnp.take(out, (y + dy) % Ny, axis=1)

        c_plaquette = jnp.stack(
            tuple(
                shifted(self.c[k], *offsets[k])
                for k in range(4)
            ),
            axis=2,
        )

        GR, GL, c_schur, rank = jax.pure_callback(
            _periodic_schur_gauges,
            (
                jax.ShapeDtypeStruct(c_plaquette.shape, c_plaquette.dtype),
                jax.ShapeDtypeStruct(c_plaquette.shape, c_plaquette.dtype),
                jax.ShapeDtypeStruct(c_plaquette.shape, c_plaquette.dtype),
                jax.ShapeDtypeStruct(self.rank.shape, self.rank.dtype),
            ),
            c_plaquette,
            jnp.asarray(
                CTM_eig_cutoff,
                dtype=jnp.real(c_plaquette).dtype,
            ),
            self.rank,
        )
        gauged = self.gauge_fix_env(GR, GL)
        c = tuple(
            gauged.c[k].at[
                ((x + dx) % Nx)[:, None],
                ((y + dy) % Ny)[None, :],
            ].set(c_schur[:, :, k])
            for k, (dx, dy) in enumerate(offsets)
        )
        return gauged._replace(c=c, rank=rank)

    def apply_T_bond_gauges(self, leg, G, G_inv):
        """Apply a ket gauge field to every horizontal or vertical T bond.

        ``leg=0`` gauges ``(x, y) -> (x+1, y)`` and ``leg=1`` gauges
        ``(x, y) -> (x, y+1)``. ``G`` and ``G_inv`` have shape
        ``(Nx, Ny, D, D)`` and label the outgoing bond at each site; incoming
        actions use the inverse from the preceding site with wrap-around.

        In double layer only the stored ket ``T`` is gauged; its bra remains
        implicitly ``conj(T)``. The matching A legs receive the single-layer
        action or the double-layer ``conj(G) x G`` action. Corners and
        ``finite`` are unchanged.
        """

        spatial_axis = leg
        opposite_leg = (leg + 2) % 4

        def gauge_T_layer(T, g, g_inv):
            """Gauge all outgoing and incoming bonds of one T layer."""

            T = _apply_gauge(T, g, leg - 4)
            return _apply_gauge(
                T,
                jnp.swapaxes(jnp.roll(g_inv, 1, axis=spatial_axis), -1, -2),
                opposite_leg - 4,
            )

        T = gauge_T_layer(self.T, G, G_inv)
        if self.T.ndim == 7:
            A_gauges = (jnp.conj(G), G)
            A_gauge_inverses = (jnp.conj(G_inv), G_inv)
        else:
            A_gauges = (G,)
            A_gauge_inverses = (G_inv,)

        A = list(self.A)
        for layer, (g, g_inv) in enumerate(zip(A_gauges, A_gauge_inverses)):
            A_axis = 2 + layer
            A[leg] = _apply_gauge(
                A[leg],
                jnp.swapaxes(g_inv, -1, -2),
                A_axis,
            )
            A[opposite_leg] = _apply_gauge(
                A[opposite_leg],
                jnp.roll(g, 1, axis=spatial_axis),
                A_axis,
            )

        return self._replace(T=T, A=tuple(A))

def _apply_gauge(X, G, axis):
    """Apply ``X'[x,y,...,b] = X[x,y,...,a] G[x,y,a,b]`` to one leg."""

    X = jnp.moveaxis(X, axis, -1)
    # X_{xy...a} G_{xyab} -> X'_{xy...b}.
    XG = jnp.einsum("xy...a,xyab->xy...b", X, G)
    return jnp.moveaxis(XG, -1, axis)
