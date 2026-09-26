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
bp = boundary_peps(site, legs, 2; maxdim = 16, boundary = [1.0, 0.0], gtol = 1e-4)   # L-BFGS: get close
bp, info = boundary_peps_krylov(site, legs, bp)        # Newton–Krylov: converge (|∇f| < 1e-9)
cvm_freenergy(bp)            # ln κ per site — a variational LOWER bound
site_ratio(bp, mag)          # ⟨σ⟩ in the bulk
s3, l3, _ = ising3d_site(0.24)
bp3 = boundary_peps(s3, l3, 3; maxdim = 24, init = bp, gtol = 1e-4)   # a nearby β, D = 3 (embedded)
bp3, _ = boundary_peps_krylov(s3, l3, bp3)
bpg = adapt(CuArray, bp3)    # to the GPU (it pays from D = 4); every solver continues there
```

Near a critical point converge before measuring: m is decided by a soft mode that f and a loose
gradient do not see ("Routes to the boundary state" below).

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

### The square's symmetry: `c4v = true` (2026-09-25)

When every layer is invariant under the 8 symmetries of the square (a symmetrised boundary PEPS
and a C4v site), the iteration is equivariant: a step derives the pair of one interface type and
grows one quadrant and one half-line, and relabels them onto the other 3 pairs and 6 blocks. A
reflection along a pair's own axis swaps P_A and P_B. The constructor checks the invariance.
`boundary_peps` turns it on whenever `symmetrize = true`. Same states as above, c4v against the full
step, all agreeing to ≤ 6e-15 in ln κ, m and the environment:

| D, χ | device | full step | c4v step | speedup |
|---|---|---|---|---|
| 2, 16 | CPU | 0.045 s | 0.016 s | 2.8× |
| 3, 24 | CPU | 0.130 s | 0.100 s | 1.3× |
| 3, 24 | GPU | 0.300 s | 0.093 s | 3.2× |
| 4, 48 | GPU | 0.261 s | 0.078 s | 3.3× |
| 5, 50 | GPU | 0.607 s | 0.223 s | 2.7× |
| 6, 72 | GPU | 3.45 s | 0.864 s | 4.0× |

On the CPU the full step already runs its 4 pairs on threads, so the gain is small; on the GPU they
queue behind each other and the gain is the full factor. (GPU rows measured while another GPU job
ran; an unloaded D = 3, χ = 24 full step was 0.055 s.)

### GPU

Pass the site on the device (`adapt(CuArray, site)`, and a `BoundaryPEPS` init on the device, or move a
host state with `adapt(CuArray, bp)`: tensor, site and both environments); every contraction follows
it, for every solver. `test/test_gpu_paths.jl` checks host/device agreement for L-BFGS, Newton–Krylov
(real and complex data) and the complex c4v `InfiniteCTM2D`. One sandwich step, split pairs, Float64 on an RTX A6000 against 4 CPU threads
on a loaded workstation: 7× at D = 3, χ = 24, 22× at D = 4, χ = 48, 23× at D = 5, χ = 50 (0.58 s
against 13.3 s; the unloaded 8-thread CPU step is 7.1 s). At D = 3 the GPU is latency-bound: a warm
evaluation (two environments to 1e-7, 15 steps each, plus gradients) is 4.2 s on the GPU and 3.6 s on
the CPU with c4v. It pays from D = 4.

Consumer GPUs run Float64 at 1/32–1/64 of their Float32 rate: on an RTX 3070 (2026-09-26) one warm
D = 3, χ = 16 evaluation (two environments to 1e-12, ~14 steps each with Anderson mixing) is 1.1 s,
against 0.6 s on 8 CPU threads.

## The optimiser near β_c (2026-09-25)

Measured at D = 3, χ = 24, β = 0.2275 (2.6·10⁻² above β_c), all variants from the same start (the
converged D = 2 state embedded at D = 3):

* **Adaptive CTMRG tolerance** (`adaptive_tolerance = true`): converge the environments to
  1e-2·|g|, clamped to [ctm_tolerance, 1e-6]. CTM steps per L-BFGS iteration fell from 54 to 30 with
  the same progress per iteration.
* **Norm-metric preconditioner** (`precondition = true`): L-BFGS with H₀ = (N + 1e-2·λ_max)⁻¹, N the
  one-site norm environment with ket and bra removed (the Gram matrix of the tangent vectors): about
  twice the gain in f per iteration.
* Both together, after 600 s: f = 0.78461719 against 0.78461586 for the old optimiser.
* **To convergence** (|g| < 1e-6): the old optimiser 735 iterations in 10 415 s, the new one 382 in
  3 431 s (3.0×), both to m = 0.49169 (Monte Carlo 0.491645).

**Anderson acceleration of the 2D CTMRG** (`update(ic; anderson = m)`, opt-in). Type-II mixing of
the last m iterates on the aligned blocks, the reported state always a genuine step. D = 3, χ = 24
at β = 0.2275, CTM steps to tolerance:

| network | m = 0 | m = 3 | m = 5 | m = 8 |
|---|---|---|---|---|
| sandwich, warm (1e-3 kick, tol 1e-8) | 19 | 14 | 12 | 12 |
| sandwich, cold (tol 1e-10) | 41 | 37 | 41 | 55 |
| norm, warm | 17 | 17 | 15 | 13 |
| norm, cold | 37 | **328, wrong fixed point** | 67 | 67 |

Warm, where the optimiser lives, it saves 1.4–1.6× on the sandwich. From the vacuum it does not
help, and at m = 3 the norm network converged to a fixed point 3e-4 off in ln κ: mixing while the
kept rank is still growing can change the basin. Hence off by default in `update`; the boundary-PEPS
solvers apply it to warm-started runs only (`ctm_anderson`; on by default in `boundary_peps_krylov`,
1.7–1.8× fewer steps per evaluation at D = 3, see "Routes to the boundary state").

Tried and rejected:

* **The self-consistent local eigenproblem** (Nishino's TPVA update: A ← the dominant generalised
  eigenvector of (T_eff, N_eff), the stationarity condition's own form). A is that eigenvector at
  the optimum, with a wide gap (λ₁ = 1.758615 against λ₂ = 0.89), but the iteration diverges: first
  through near-null directions of N_eff (overlap of the top eigenvector with A 0.014 unless N_eff is
  cut at 1e-5·max, then 0.991), and then, with the cut, through the environment's response near β_c
  (f fell and m ran from 0.49 to 0.72 within three steps). The frozen-environment problem ignores
  exactly the large susceptibility that makes this regime hard.
* **The frozen-environment Hessian as preconditioner**, (λ₁ Ñ − T_eff) in the Ñ-metric: behind the
  norm metric at every checkpoint (f 0.78461695 against 0.78461726 at iteration 100).

## Routes to the boundary state (2026-09-26)

This section compares the routes by which fixed point each reaches and how fast. Everything is 3D
Ising with `ising3d_site`, measured on the 8-core i9-9900K here.

### Accuracy: only the stationary point of f

f(A) = ln κ⟨A|T|A⟩ − ln κ⟨A|A⟩ is the nested Bethe (Kikuchi) estimate. Its stationary point is the
MP-BP fixed point along z, Woolls et al.'s condition one dimension up, so its errors are second
order. Every local shortcut tried lands somewhere else, or nowhere:

| route | fixed point | measured |
|---|---|---|
| direct 3D CTMRG (`InfiniteCTM3D`) | biased | m 0.9% high at β = 0.25 for every χ ≤ 8 |
| projected power with MP-BP bond projectors (`boundary_peps_power`, removed; code in bf8ae70) | biased | β = 0.25, D = 2: m = 0.75386 against 0.75093, f 4.7e-6 low, \|∇f\| = 3.7e-3. At β = 0.22 it orders (m = 0.40) where D = 2 is disordered. Fast: 30 steps, 7 s |
| natural power A ← N⁻¹E_s (`boundary_peps_natural`, removed; code in 6ad5c54) | exact | unstable: from 2e-3 away, \|ΔA\| grows 0.08 → 1.3 |
| stationary or variational: L-BFGS, Newton, Newton–Krylov | exact | see below |

* **The bond projectors** cut each bond against its own environment and are blind to the plane's
  loops. That is the direct 3D CTMRG's mean-field bias, one dimension down.
* **The natural update** is the stationarity condition's own form. But its step is taken in frozen
  environments, and their response makes the map expansive, as for Nishino's local eigenproblem
  above.

**The inner contraction: `:cut` or `:cycle`.** `projector = :cycle` works in `InfiniteCTM2D`. It is
MP-BP / eig-CTMRG, taking its projectors from the dominant invariant subspace of the corner cycle.

* On real reflection-symmetric networks, the 3D Ising sandwiches among them, it finds `:cut`'s
  fixed point.
* It is gauge-invariant where `:cut` is not. For 2D Ising at K = 0.42, χ = 4, with one random bond
  gauge on every bond: Δln κ = −1.93e-6 for every gauge, against −1.6e-7 or +5.2e-5 for `:cut`.
* On the complex Yang–Lee networks it is 2–20× more accurate in m. At χ = 16: 1.2e-11 against
  2.5e-10 far from the edge, and 3.9e-7 against 8.4e-7 near it.
* It costs 3–50× more per step (24.9 ms against 2.4 ms at χ = 16).

So use `:cut` for real networks and `:cycle` for non-Hermitian or badly gauged ones.

### Newton–Krylov: `boundary_peps_krylov`

The finite-difference Newton of `boundary_peps_stationary` builds all of J: 2n evaluations, with
n = 12, 42, 110 C4v coordinates at D = 2, 3, 4. Newton–Krylov builds only a subspace of it.

* **Products.** Each Jacobian-vector product is one warm-started evaluation at c + h u (forward
  difference, h = 1e-5). The products are clean: the D = 2 FD Jacobian is symmetric to 4e-8, and
  J c = −g to 3e-8.
* **Coordinates.** It works in the reduced coordinates, with scale and the O(D) bond gauge projected
  out.

**The landscape is soft and curved.** Reduced Hessian eigenvalues at β = 0.25, χ = 16:

| case | range | notes |
|---|---|---|
| D = 2 (m = 10) | −2.7 … −2.9e-6 | the soft modes carry the last digits of m |
| D = 3 (m = 38) | −2.8 … ±1e-13 | several are POSITIVE: the unused bond dimension, a saddle in directions that barely move the state |
| D = 3, in the norm metric | 1.2e-5 … 1.5e3 | no better conditioned; GMRES needs 12 / 27 products to a 1e-1 / 1e-2 residual, against 3 / 12 plain |

Along the soft modes the Newton step is long: 0.063 along the λ = −1.6e-4 mode at |g| = 3.9e-5,
D = 2. And f is not quadratic on that scale. The full step raised |g| 30× while f improved, and a
line search on |g| stalled at 3.9e-5.

Hence a TRUST REGION IN THE KRYLOV SUBSPACE:

* The model is of f for real data (negative curvature followed to the boundary), and of |g|² for
  complex data (Levenberg–Marquardt).
* The ratio of actual to predicted gain accepts the step and sizes the next one.
* A rejected step is re-solved in the same subspace, with no new products.

Options, as measured:

* **`block = 4`, `recycle = 3` (the defaults).** Four products at a time on threads (block Arnoldi),
  and each subspace starts from −g, the accepted step, and the two Ritz vectors the step moved along
  most. Products per step stay the same, and wall time falls 1.8×: D = 2, 8 threads, 107 evaluations
  in 10.5 s against 102 in 18.6 s.
* **`ctm_anderson = 5` (the default here; opt-in for `boundary_peps`).** Anderson mixing in every
  warm-started 2D CTMRG run. CTM steps per evaluation from the D = 3 start state at β = 0.2275:
  26.5 → 14.5 for a product, and 35.8 → 21.2 after a 2e-2 step. The fixed point is unchanged. Cold
  runs are left unmixed (see Anderson above).
* **Reusing a subspace** for further steps from the new gradient, without new products: worse. It
  stalls without recycling, and with it costs 115 evaluations and 14.5 s. Removed.
* **The norm-metric preconditioner:** worse, as above.

At D = 2 it does not beat L-BFGS on evaluations: about 100 either way, since every soft step needs
the whole 10-dimensional space. From 25 L-BFGS iterations (β = 0.25, |g| = 2.1e-4), it reaches
|g| = 2.5e-11 in 16 steps, with f and m equal to L-BFGS's optimum to 1e-13 and 1e-8.

### The benchmark: `examples/ising3d_solver_benchmark.jl`

Each run uses one solver, from a shared cached start state (D = 2 converged, embedded at D with
noise 1e-2), within a wall-clock budget, with compilation kept outside the timing.

* **Outputs.** The trace (t, f, |g|) goes to CSV. So does m at checkpoints, each checkpoint's state
  evaluated with freshly converged environments.
* **Chaining.** `BP_SAVE`, `BP_INIT` and `BP_TOFFSET` chain runs past the 10-minute cap.

The setting: 3D Ising at β = 0.2275 (2.6% above β_c), D = 3, χ = 16, 8 threads, 400 s per run. The
reference is Newton–Krylov chained to |g| = 5.1e-9 (856 s): f* = 0.7846174816989,
m* = 0.4917076547.

| solver | f* − f < 1e-8 | < 1e-9 | < 1e-10 | \|m − m*\| < 1e-7 to stay | at 400 s: f* − f, \|m − m*\| |
|---|---|---|---|---|---|
| L-BFGS (`boundary_peps`) | 309 s | — | — | — | 6.8e-9, 7.0e-6 |
| L-BFGS, `ctm_anderson = 5` | 243 s | — | — | — | 3.4e-9, 2.7e-6 |
| Newton–Krylov, no Anderson | 305 s | ≈ 600 s | ≈ 600 s | not checkpointed | 1.9e-9, 4.1e-6 |
| **Newton–Krylov, `ctm_anderson = 5`** | **156 s** | **238 s** | **310 s** | **310 s** | converged at 395 s: 8e-11, 1.8e-8 |

The no-Anderson Newton–Krylov times to 1e-9 and 1e-10 come from the reference chain (f* − f ≈ 1e-12 by
596 s). Identical runs vary by about ±15% in wall time: the trajectories are deterministic, and one
repeat ran 15% slower throughout.

**m is the hard part.**

* **L-BFGS never settles in m.** In both of its runs, m wandered by ±1.5e-5 around m* over the last
  200 s, while f changed by less than 1e-10.
* **Single steps move m.** One L-BFGS step moved m by 9e-6.
* **Why.** The soft mode is the magnetisation's: f is flat along it, and m is not. So m converges
  only when that mode is resolved, which Newton does quadratically once near. A converged |g| or f
  says little about m before then. The first L-BFGS run here happened to stop at a point 3.7e-7 from
  m*, at |g| = 3.6e-6.

### Verdict

The fastest accurate route measured, from the network inwards:

1. **The inner contraction.** `InfiniteCTM2D` with `c4v = true` and split pairs. Use `:cut` for
   real networks and `:cycle` for complex ones. Mix warm runs (`ctm_anderson = 5`), and use the
   GPU from D = 4.
2. **The boundary state.** The stationary point of the Bethe estimate, solved by
   `boundary_peps_krylov` from a warm start: a nearby β, or a few L-BFGS iterations after an
   embedding.
3. **Observables** only from a converged state. Near β_c, m is set by a soft mode that f and a
   loose |g| do not see.

At D = 3 near β_c on 8 cores this gets m to 1e-7 in about 5 minutes, where L-BFGS still wanders at
1e-5 after 400 s. What is not yet measured is under "Open problems" below.

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

* D = 2, χ = 16: one evaluation (two 2D CTMRG runs plus four one-site environments) is ~0.5 s
  warm-started. 60 L-BFGS iterations at β = 0.25 took 37 s.
* D = 3, χ = 16, 8 threads, Anderson mixing (2026-09-26): 0.6 s per warm evaluation, almost all of
  it the CTM steps (norm network 0.2 s, sandwich 0.6 s run concurrently, 13–15 steps each); the four
  site environments and both ln κ are 1% of it.

## Open problems

* D ≥ 4 with Newton–Krylov (n = 110 coordinates at D = 4), where the subspace should matter more,
  and with the GPU, where concurrent products compete for one device.
* The complex path of `boundary_peps_krylov` beyond the D = 1 chains: the Yang–Lee continuations
  (docs/yang_lee.md) still use `boundary_peps_stationary`.
* β continuation with a tangent predictor, which would start each point of a scan near the soft
  mode's answer.
* How long L-BFGS takes to settle m at D = 3 near β_c: not within 400 s (the chained run that was to
  measure it did not finish).
