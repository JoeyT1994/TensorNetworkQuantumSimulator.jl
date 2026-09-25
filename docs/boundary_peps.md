# Boundary PEPS for 3D classical models — the thermodynamic limit

*Started 2026-09-25 (FixesV2). Everything here is measured; the numbers are dated.*

Why this exists: the infinite 3D CTMRG (`InfiniteCTM3D`, [ctmrg3d.md](ctmrg3d.md)) holds a
quarter-plane of bonds in one χ-dimensional index. Its fixed point for 3D Ising is 0.9% high in m
at β = 0.25 for every χ from 2 to 8, with a mean-field-like transition 8% hot at χ = 2. Mosko & Gendiar's
reformulation of the original 3D CTMRG reports the same saturation (T_c 4.936 / 4.716 / 4.696 at
m = 2, 3, 4 against 4.5115). The boundary PEPS moves that 2D entanglement into a PEPS bond
dimension D. It leaves CTMRG only line interfaces, the kind it converges on.

```julia
site, legs, mag = ising3d_site(0.25)                   # cubic Ising, symmetric bond split
bp = boundary_peps(site, legs, 2; maxdim = 16, boundary = [1.0, 0.0])
cvm_freenergy(bp)            # ln κ per site — a variational LOWER bound
site_ratio(bp, mag)          # ⟨σ⟩ in the bulk
bp2 = boundary_peps(ising3d_site(0.24)[1:2]..., 2; maxdim = 16, init = bp)   # warm start
```

## Construction

One layer of the cubic network is a 2D tensor-network operator T: the site tensor, with its z legs
as physical in (z⁻) and out (z⁺) legs. Z = Tr T^{L_z}, and the free energy per site is set by T's
dominant eigenvalue. Its eigenvector is approximated by a 1×1 infinite PEPS |Ψ(A)⟩ of bond
dimension D, and

    f(A) = ln κ(⟨Ψ|T|Ψ⟩) − ln κ(⟨Ψ|Ψ⟩)        (per site)

is maximised. Both terms are free energies per site of ordinary 2D networks, three layers and two,
contracted by `InfiniteCTM2D` in the same Kikuchi form as the 2D engine. For a symmetric T,
`ising3d_site`'s among them, f is a variational lower bound on ln κ₃D.

**The gradient is local.** T is a product of site tensors, not a sum of local terms. So the
derivative of a translation-invariant network's free energy per site with respect to its tensor
is that tensor's normalised one-site environment: d ln κ = ⟨E, da⟩ / z. With A in the ket and the
bra layer:

    ∂f/∂A = (E_ket + E_bra)/z |⟨Ψ|T|Ψ⟩ − (E_ket + E_bra)/z |⟨Ψ|Ψ⟩

That is four one-site environments, with no sum over positions and no channel environments. f is
invariant under A → cA, so the gradient is orthogonal to A. The optimiser works on the unit
sphere: L-BFGS with Armijo backtracking, tangent steps, A renormalised, and the step capped, as in
the DMRG branch's `ctmrg_lbfgs`. Both 2D environments warm-start from the previous evaluation. With
`symmetrize = true`, which needs a site invariant under the square's symmetries of its x, y legs, A
and every gradient are projected onto the C4v-symmetric subspace.

## The 2D engine: `InfiniteCTM2D`

A 1×1 unit cell, the 2D restriction of `InfiniteCTM3D`: 4 half-lines and 4 quadrants, one pair per
line-interface type from the two enlarged quadrants on either side, and ln κ = ln Z_v − ln Z_ex −
ln Z_ey + ln Z_f. A site is a tensor or a list of layers, and a raw bond is the list of their legs,
never fused.

| check | result |
|---|---|
| 2D Ising K = 0.3 vs Onsager | 1.3e-10 at χ = 4, 4e-15 at χ = 8 (15 iterations) |
| K = 0.5, fixed-spin seed, vs Yang | 3.1e-11 at χ = 8, 1.2e-12 at χ = 16 |
| layered vs fused site (random D = 2 PEPS norm) | identical to 4e-16 |
| d ln κ/dK from one site's environment vs 4-point finite difference | 5e-11 (K = 0.3), 1e-9 (K = 0.5) |
| 200 extra iterations at the fixed point | m moves < 1e-11 |

Three design points, each measured:

* **Convergence on the environment.** The criterion is the centre shell contracted onto the site's
  legs: gauge-invariant, and blind to kept directions that carry no weight.
  * Block changes are not. They took as long from a warm start as from the vacuum (124 against
    118 iterations).
  * ln κ is no better. It is stationary, so it settles long before the environment: stopping on it
    left Yang's m 5e-7 off and gradients 1e-7 off.
* **`pair = :biorth` by default.** `:isometric` is exact on mirror-symmetric networks but unstable
  on others: on a random PEPS norm it read the right ln κ at iteration 9, and −5.0 instead of 2.03
  by iteration 600. That is the reverse of the 3D engine, whose mirror-symmetric plane pairs need
  `:isometric`.
* **`init` warm starts.** They save only the logarithm of the starting distance, because
  convergence is linear. That matters for the small steps late in an optimisation.

### Split pairs: the projector without the enlarged quadrant (2026-09-25)

The dense pair contracts each enlarged quadrant (old quadrant, two half-lines, the site's layers)
into an n×n matrix, n = χ·Π(raw dims) = χ·2D² for the sandwich, and factorises it: O(χ³D⁶). From
D = 4 that was the whole step (D = 5, χ = 50: 11 s per projector, 1.2 s for both corners). The
split pair (`_i2_pair_split`) gets the truncated SVD from the dense engine's warm-started subspace
iteration, applying each quadrant to a block of k′ = χ + 16 vectors as its FACTOR LIST: one netcon
per application, which absorbs the layers into the block one at a time. The quadrant and its
layer-fused legs never exist. It is the split CTMRG of Naumann et al. (2024) and Xu, Lin & Zhang
(2025), obtained from the contraction-order search rather than written by hand, so it works for
any number of layers. It is the default (`svd = :auto`); `svd = :dense` restores the old pair.
Graded data, the size gate and a subspace bail-out (flat spectrum) fall back to the dense pair.

Converged sandwich environments of the same D, χ state, 3D Ising β = 0.23, 8 threads:

| D, χ | n | dense step | split step | speedup | Δf, Δm, ‖ΔE‖/‖E‖ |
|---|---|---|---|---|---|
| 2, 16 | 128 | 0.024 s | 0.041 s | 0.6× | ≤ 2e-15 |
| 3, 24 | 432 | 0.18 s | 0.13 s | 1.4× | ≤ 4e-15 |
| 4, 48 | 1536 | 6.8 s | 3.0 s | 2.2× | ≤ 4e-15 |
| 5, 50 | 2500 | 28.8 s | 7.1 s | 4.1× | ≤ 2e-15 |

Same iteration counts (13) both ways. One quadrant application costs ~χ³D⁴ (0.006 s at D = 3,
χ = 24; 0.26–0.33 s at D = 5, χ = 50), and a warm pair takes about nine of them (two subspace
iterations, the warm-start block, the whitening). At D = 5 the step splits into 4 pairs of 2.8 s
each and 8 block growths of ~0.45 s each, run on threads; two site environments cost 1.0 s and
ln κ 0.5 s, so the gradient is not the bottleneck.

## Validation of the boundary PEPS

| check | result |
|---|---|
| decoupled layers (Jz = 0), D = 1, vs Onsager | 4.4e-15 |
| chains along z (Jx = Jy = 0), D = 1, vs ln 2cosh K | 2.2e-16 |
| ∂f/∂A vs 4-point finite difference (β = 0.22, D = 2, χ = 16) | 4.2e-13; G·A = −1.5e-15 |

## 3D Ising

`examples/ising3d_boundary_peps_benchmark.jl`: a scan from β = 0.40 down to 0.20, each point
warm-started from the previous one, gradient tolerance 1e-6, CTMRG tolerance 1e-9, L-BFGS capped
at 150 iterations. Reference: the Talapov–Blöte fit to Monte Carlo (β_c = 0.2216544), used only in
its validated range β ≤ 0.305. Error in m:

| β | D = 2, χ = 16 | D = 3, χ = 24 |
|---|---|---|
| 0.30 | 2.0e-5 | 2.0e-5 |
| 0.28 | 2.3e-6 | 1.8e-6 |
| 0.25 | 4.5e-6 | 4.2e-6 |
| 0.24 | 4.9e-5 | 6.6e-5 * |
| 0.235 | 1.4e-4 | 4.6e-5 * |
| 0.23 | 5.9e-4 | 8.3e-5 * |
| 0.2275 | 1.5e-3 | 8.7e-5 * |
| 0.225 | 4.6e-3 | 1.9e-4 * |
| 0.2235 | 1.2e-2 | 4.4e-3 *† |
| 0.2217 | m = 0.234 against 0.105 | |

\* The iteration cap was hit, with |g| between 4e-6 and 3e-5. † Cold start (the scan was
interrupted and restarted from the fixed-spin seed): |g| = 1.9e-4 at the cap.

* D = 2 loses its magnetisation between β = 0.220 and 0.221. Vanderstraeten, Vanhecke &
  Verstraete (PRE 98, 042145, 2018) fit T_c = 4.525222 at D = 2 (β = 0.22098), inside that bracket;
  their D = 3 and 4 fits give 4.5118 and 4.51195 against the Monte Carlo 4.511523.
* At β = 0.35 and 0.40 D = 2 and D = 3 agree to 1e-7. The low-temperature series through u¹² is
  6e-4 off at β = 0.35 (its u¹³ term), so it is not a reference there.
* f is a lower bound, and D = 3 lies above D = 2 wherever the two differ.
* Near β_c the optimiser, not the contraction, is the limit: D = 3 needed more than 150
  iterations from β = 0.24 down, as Vanderstraeten et al. found for D = 4.

## Costs

D = 2, χ = 16: one evaluation (two 2D CTMRG runs plus four one-site environments) is ~0.5 s
warm-started. 60 L-BFGS iterations at β = 0.25 took 37 s.

## Open problems

RESULTS PENDING
