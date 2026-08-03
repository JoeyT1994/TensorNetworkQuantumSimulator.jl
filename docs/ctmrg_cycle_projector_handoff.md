# Handoff: the cycle (stationary) projector for finite CTMRG

Branch `Finite_CTMRG`, tip `4ea3d8e`. 163 tests green (`test/test_ctmenvironment.jl`, ~4 min).
Working tree has no CTM changes — the cycle projector was implemented, measured, and **removed**.

Full detail lives in `docs/finite_ctmrg_design.md`. This file is the short version: what the open
problem is, what is already settled, and what not to waste time re-deriving.

---

## 1. The goal, and why it matters

Our interface projector (Corboz-style: QR each half, SVD the small product) is the optimal rank-χ
truncation of **one bipartition, chosen independently per interface**. It is **not** a stationary
point of `F`. `marginal_inconsistency` is exactly that residual and is nonzero at finite χ.

The collaborator's Python engine (`/Users/jtindall/Downloads/joey_ctmrg_bp/`) instead takes each
projector to be the dominant invariant subspace of the four-corner cycle, which **is** stationary.
Measured head-to-head on their own 5×5 Ising PEPS:

| | `ln Z` | single-site observable |
|---|---|---|
| Julia (cut / SVD) | **better**, 2–23× | worse, 3–130× |
| Python (stationary / eig) | worse | **better** |

Stationarity is marginal consistency, and marginal consistency is what a single-region ratio needs.
`Z` hides the defect because the Möbius sum cancels it (~4000×). **So the prize is observables, and
it is worth 1–2 orders.**

---

## 2. The open problem

Pure cycle **works**. On the 5×5 Ising PEPS, `⟨X⟩` error, exact `0.916900598128483`:

| χ | Julia cut | Julia cycle | Python |
|---|---|---|---|
| 4 | 1.515e-04 | **4.169e-05** | 5.218e-05 |
| 9 | 4.240e-07 | **5.132e-08** | 5.132e-08 |
| 16 | 6.255e-09 | **9.202e-10** | 9.279e-10 |
| 32 | **7.403e-12** | 1.448e-10 | 8.149e-14 |

At χ=4 and 9, `k_cyc = k_b`, so those rows are **pure cycle** — and χ=9 reproduces Python to four
significant figures on adaptive bonds with no padding. The criterion is right; the *mixing* is broken.

Two unexplained facts, almost certainly one phenomenon:

1. **Damage tracks `k_cyc / k_b`.** Pure cycle (ratio 1) excellent; 9 cycle + 7 cut at χ=16 fine;
   9 cycle + 23 cut at χ=32 is 20× worse than the plain cut.
2. **Hex 4×4 stops converging at χ ≥ 8** (`marg` stuck at 1.1e-2 / 7.4e-3 / 7.1e-3) while hex 3×3 is
   exact to 1.85e-15.

Understanding (1) is the prerequisite. The target is not "make the union better" — it is either
"understand why cycle and cut directions do not compose" or "run pure cycle without the narrow-bond
rank cap".

---

## 3. Settled — do not re-derive

**Geometry.** At a cut, each of the four enlarged corners has exactly two open interfaces, so with
bonds ordered `(W, S, E, N)` each corner maps one bond to the next:

```
A1 = E_SW : W->S    A2 = E_SE : S->E    A3 = E_NE : E->N    A4 = E_NW : N->W
```

`M = A4 A3 A2 A1` acts on the west bond. The four projectors land on the **existing** keys
`PH[:N,X-1,Y]`, `PH[:S,X-1,Y]`, `PV[:W,X,Y-1]`, `PV[:E,X,Y-1]` — only the derivation changes; the
sweep, consumers and region machinery are untouched.

**Solver — matrix-free, no padding.** Bonds are rectangular in general (`k_prev · D_layer`, and
`k_prev = 1` at the boundary). Their engine pads everything to fixed χ with a separate `rank` field
because a *dense* periodic Schur needs square equal-size factors. **We never need that** — only the
action of the cycle on a vector:

```julia
fwd(v) = As[4] * (As[3] * (As[2] * (As[1] * v)))
schursolve(fwd, v0, k, :LM, Arnoldi(; krylovdim = max(4k+8, 24), tol = 1e-13))
```

Verified 9/9 square and 32/32 hex plaquettes at 1e-15. `schursolve` also returns a real orthonormal
basis, avoiding the conjugate-pair handling a dense `ordschur` needs. KrylovKit is already a
dependency; `PeriodicSchurDecompositions.jl` is **not** needed.

**Left bases** propagate downward, `V_L[l] ∝ V_L[l+1] A_l`, seeded from `schursolve` on the
transposed action.

**Side assignment.** For each bond, the factor on the tensor *consuming* it is the right basis, the
one on the *producer* is the left. Against our west/north = `P_A` convention: **W and S take
`P_A = V_L`; E and N take `P_A = V_R`.** Verified by the insertion identity at 1.1e-14.

**Two real fixes found while building it — keep both.**

* The merge must use **oblique deflation** against the cycle pair, not an independent QR per side.
  Independent QR destroys the pairing between column `j` of `A` and row `j` of `B`, so the overlap
  stops being near-identity and whitening mixes directions. Deflation gives `B_cyc Ad = 0` and
  `Bd A_cyc = 0` identically ⇒ block-diagonal overlap with the cycle block exactly `I`.
* Biorthogonalisation must **truncate** the overlap singular values, not floor them at `eps` — the
  same `S^{-1/2}` amplification `qr_cutoff` guards against elsewhere.

---

## 4. Already tried and rejected — do not repeat

| idea | verdict |
|---|---|
| Gauss–Seidel sweeping | Breaks the projector's optimality by handing it a pre-truncated bipartition. Inseparable from the cycle projector. |
| `max dV` convergence signal | It *is* our `_ctm_statedist`, measured on the variables instead of the outputs. Agree within 2–3× at every sweep. |
| Single per-plaquette rank `min(χ, narrowest bond)` | Hex has bonds of dimension 1 (missing lattice link) ⇒ all four bonds collapse to one direction forever. Saturates at 1e-3. |
| Union with independent per-side QR | Destroys pairing (see above). |
| Union with oblique deflation | Fixes the construction but not the observable. Unreliable across the scan. |
| ket↔bra symmetry in the projector | 2–2.5× slower, and the defect it fixes is 9 orders below the truncation error. |
| Zero-padding to make bonds square | Exact for dense `pschur!` but makes factors singular ⇒ breaks Arnoldi. |

Scan that killed the union (ratio = cut err / cycle err; **below 1 means cycle is worse**):

| lattice | χ=2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| square 4×4 real D=2 | 0.14 | 0.64 | 8.89 | 2.12 | 0.16 |
| square 4×4 real D=3 | — | 0.64 | 0.34 | 0.74 | 0.54 |
| hex 4×4 complex D=2 | 1.13 | 74.65 | 0.00 | 0.00 | 0.00 |

---

## 5. Gate suite — mandatory before any positive claim

My gates passed and the route was still broken, because each sampled **one point per axis**. Scan
χ **and** lattice size **and** bond dimension **and** element type.

1. Full-rank insertion identity `Bp (P_A P_B) Bc = Bp Bc` → ~1e-14.
2. Exactness at lossless χ on square **and** hex.
3. **Hex 4×4 at χ ∈ {8, 16, 32}** as a named regression case — this is what the union failed.
4. `marginal_inconsistency` monotone in χ and → 0 when lossless, on both lattices.
5. Observables vs the Python numbers at matched χ, across ≥2 states and ≥2 sites.

Judge on **observables and `marginal_inconsistency`**, never on `|F − ln Z|` — cancellation makes `F`
accurate and blind. `F` has been measured going *backwards* with sweeps (2.1e-3 at sweep 2 → 2.9e-3
converged).

---

## 6. Reproducing the benchmark

Their code **does** run on macOS/arm64. The compiled SLICOT extension is only reached by the
periodic-Schur paths; the demo's default `"eig one sided"` never touches it.

```bash
python3 -m venv /tmp/pyenv && /tmp/pyenv/bin/pip install jax jaxlib einops numpy scipy
cd /Users/jtindall/Downloads/joey_ctmrg_bp && /tmp/pyenv/bin/python ctmrg_demo.py
```

Reference values for `data_ising_5x5/isingZZX_5x5_D3_g3.04438.npz`:

* `ln Z` exact = `-6.217866847854575`
* `⟨X⟩` at their site `(2,2)` = our `(3,3)` = `0.916900598128483`
* Python `⟨X⟩` errors: χ=4 `5.218e-05`, 9 `5.132e-08`, 16 `9.279e-10`, 32 `8.149e-14`

**Two data-transfer traps, both of which produced confident wrong answers:**

1. `ndarray.tofile()` always writes **C order** regardless of memory layout. Use
   `f.write(B.tobytes(order='F'))`.
2. JAX silently downcasts float64→float32 on unpickling unless x64 is on. **Call `configure_jax()`
   before `np.load`** or you export degraded tensors.

Observables out of their code: they have no observable machinery, but
`contract_Z11((tb, O·t), A_local, c_local)` works — the primitives accept an explicit `(tb, t)` pair,
so no reimplementation of their index conventions is needed.

---

## 7. The one methodological lesson

Our exact contraction, our CVM and our boundary MPS once **all agreed** on a wrong answer, because
all three were correctly contracting the same wrong network. **Agreement among our own methods is no
check on the input.** Every cross-language number here should be confirmed by an independent
computation on the far side of the transfer.
