"""Export the collaborator's 5x5 Ising PEPS to raw float64 for examples/ctm_ising5x5_benchmark.jl.

Usage:  python3 examples/export_ising5x5.py          # writes peps5x5.bin next to this script
Needs:  pip install jax jaxlib numpy   and  joey_ctmrg_bp on the path (edit SRC below).

Two traps, both of which I previously hit:
  1. The npz pickles jax ArrayImpl objects, so UNPICKLING RUNS JAX. Without x64 they come back
     float32 (0.24756516516 vs 0.24756516631). `configure_jax()` MUST run before np.load.
     Avoiding jax does not dodge this trap -- it triggers it.
  2. Byte order. We write an explicit C-order buffer; the Julia side undoes it with
     reshape(v, reverse(shape)) + permutedims(ndims:-1:1), which is exactly numpy's
     v.reshape(shape). Verified by round trip below.
"""
import os
import sys

sys.path.insert(0, os.environ.get("JOEY_CTMRG_BP", "/Users/jtindall/Downloads/joey_ctmrg_bp"))
from jax_config import configure_jax
configure_jax()                                    # BEFORE np.load -- see trap 1
import numpy as np

SRC = os.path.join(os.environ.get("JOEY_CTMRG_BP", "/Users/jtindall/Downloads/joey_ctmrg_bp"),
                   "data_ising_5x5", "isingZZX_5x5_D3_g3.04438.npz")
OUT = os.environ.get("CTM_ISING5X5_DIR", os.path.dirname(os.path.abspath(__file__)))
raw = np.load(SRC, allow_pickle=True)
Nx, Ny = len(raw), len(raw[0])
assert raw[0][0].dtype == np.float64, f"still downcast: {raw[0][0].dtype}"
A = {(x, y): np.asarray(raw[x][y], dtype=np.float64) for x in range(Nx) for y in range(Ny)}
print(f"dtype {raw[0][0].dtype}, first element {np.asarray(raw[0][0]).ravel()[0]:.17g}")

def rawshape(x, y):
    return (2, 1 if x == 0 else 3, 1 if x == Nx-1 else 3,
            1 if y == 0 else 3, 1 if y == Ny-1 else 3)

with open(f"{OUT}/peps5x5.bin", "wb") as f:
    for x in range(Nx):
        for y in range(Ny):
            assert A[(x, y)].shape == rawshape(x, y), f"shape rule wrong at {(x,y)}"
            f.write(np.ascontiguousarray(A[(x, y)]).tobytes(order="C"))

# --- round trip exactly as the Julia loader reads it -------------------------------
data = np.fromfile(f"{OUT}/peps5x5.bin", dtype=np.float64)
off = 0; back = {}
for x in range(Nx):
    for y in range(Ny):
        sh = rawshape(x, y); n = int(np.prod(sh))
        back[(x, y)] = data[off:off+n].reshape(sh)   # == Julia reshape(v,reverse(sh)) + permutedims
        off += n
assert off == data.size
bad = sum(0 if np.array_equal(back[k], A[k]) else 1 for k in A)
print(f"round trip: {bad} mismatched sites (0 = good)")

# --- references, with the sweep validated against brute force ---------------------
def dl(op_at=None, op=None):
    out = {}
    for (x, y), t in A.items():
        tb = t if op_at != (x, y) else np.tensordot(op, t, axes=([1], [0]))
        m = np.tensordot(tb, t, axes=([0], [0])).transpose(0, 4, 1, 5, 2, 6, 3, 7)
        s = m.shape
        out[(x, y)] = m.reshape(s[0]*s[1], s[2]*s[3], s[4]*s[5], s[6]*s[7])
    return out

def sweep(a):
    LB, RB, W = "abcde", "fghij", "klmnop"
    bnd = np.ones([1]*Ny); ls = 0.0
    for x in range(Nx):
        res, rl = bnd, list(LB)
        for y in range(Ny):
            t = a[(x, y)]; lL, lR, lD, lU = LB[y], RB[y], W[y], W[y+1]
            out = [c for c in rl if c not in (lL, lD)] + [lR, lU]
            res = np.einsum("".join(rl)+","+lL+lR+lD+lU+"->"+"".join(out), res, t); rl = out
        keep = [i for i, c in enumerate(rl) if c in RB]
        res = res.transpose(keep + [i for i in range(len(rl)) if i not in keep])
        res = res.reshape([res.shape[i] for i in range(len(keep))])
        n = np.linalg.norm(res); bnd = res/n; ls += np.log(n)
    return ls, bnd.reshape(-1)[0]

ls, tail = sweep(dl())
lnZ = ls + np.log(abs(tail))
X = np.array([[0.0, 1.0], [1.0, 0.0]])
for site in [(2, 2), (0, 0), (3, 1)]:
    lsO, tailO = sweep(dl(op_at=site, op=X))
    print(f"<X> at their {site} = {np.exp(lsO-ls)*(tailO/tail):.16g}")
print(f"ln Z = {lnZ:.16g}")
print(f"doc reference: ln Z = -6.217866847854575, <X>(2,2) = 0.916900598128483")
