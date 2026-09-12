# Ground states by belief propagation: the generating-function route — *2026-09-11*

Branch `FixesV2DMRG`. This is phase 1 of a 2D DMRG built on the message-passing machinery:
BP-level (χ = 1) environments, one-site updates, exact on trees. The environments will be swapped
for the CTM `:cycle` (MP-BP) rings in phase 2.

## Why a generating function

BP on the indefinite network ⟨ψ|H|ψ⟩ is a bad idea (tried, abandoned): its messages are not
positive, so the PSD gauge, the symmetric gauge and everything the CTM inherits from positivity
break. Instead the Hamiltonian is embedded in a POSITIVE norm network,

    T(λ) = ⟨ψ|G(λ)|ψ⟩,   G(λ) = ∏ₑ(1 + λhₑ)∏ᵥ(1 + λhᵥ) = 1 + λH + O(λ²),

and only λ-derivatives at λ = 0 are ever taken. With X(λ) the BP fixed point of T(λ) and Z_B the
Bethe estimator (∏ᵥZᵥ / ∏ₑZₑ), stationarity ∂ₓZ_B = 0 gives the envelope theorem

    E_B = ∂λ Z_B(T(λ), X₀)|₀ / Z_B(T₀, X₀)        — messages held fixed —

which at χ = 1 is the BP expectation value of H (checked to 1e-9) and the exact energy on a tree.
The gradient with respect to a site tensor needs the environment response X′ = dX/dλ, obtained
from the linearised message update (Gauss–Seidel on the one-insertion sector; exact after one
sweep on a tree, a contraction whenever BP is stable), and the local problem is the generalised
eigenproblem `H_eff ψᵥ = E N_eff ψᵥ` with

    N_eff ψᵥ = ψᵥ · G(0) · X₀,     H_eff ψᵥ = ψᵥ · ∂λG · X₀ + Σᵤ ψᵥ · G(0) · X₀[m_{u→v} → X′_{u→v}].

`(H_eff − Eᵥ N_eff)ψᵥ / Zᵥ` with Eᵥ the local Rayleigh quotient IS the gradient of the BP energy
(central finite difference of the re-converged BP energy agrees to 1e-7 on a 3×3 grid, 1e-10 on a
tree). N_eff factorises over the incoming bonds into the norm messages, so the whitening
`N⁻¹ᐟ²` is the simple-update gauge (`pseudo_sqrt_inv_sqrt`) and the local solve is a Lanczos on
the Hermitian `N⁻¹ᐟ² H_eff N⁻¹ᐟ²`.

## The operator network

`TensorNetworkOperator` (`src/TensorNetworks/tensornetworkoperator.jl`) is an operator layer with
virtual bonds; `QuadraticForm` accepts it as its operator, so the three-layer norm network runs
through the ordinary `BeliefPropagationCache`. `generating_operator(H, ψ)` builds a
`GeneratingOperator`: the pair `value = G(0)`, `derivative = ∂λG(0)` in one representation.
Each edge term hₑ = Σₐ Aₐ ⊗ Bₐ (operator-Schmidt decomposition by `factorize_svd`) becomes a left
factor Σₐ Aₐ ⊗ |a⟩ on `src(e)` and a right factor 1 ⊗ |0⟩ + λ Σₐ Bₐ ⊗ |a⟩ on `dst(e)` sharing an
auxiliary index of dimension r + 1 (4 for Heisenberg). The auxiliary index is kept at λ = 0: the
messages of T(0) carry the norm sector (a = 0) plus the half-insertions ψ†Aₐψ (a > 0) travelling
from left ends to right ends, and ∂λG picks them up. No dual numbers: the λ-derivative at fixed
messages is the sum of explicit single-term insertions, and the product rule over the vertex
factors gives `derivative` exactly.

One gauge subtlety: BP normalises a message by the sum of all its entries, which is complex once
the non-Hermitian a > 0 slices are present; the whole message then carries a phase and N_eff,
H_eff stop being Hermitian maps. `generating_cache` and the response renormalise by the norm-sector
sum instead (real for a Hermitian slice). Z_B is invariant, the maps are Hermitian to 1e-16.

## Measured

`test/test_dmrg.jl` (56 tests): Z_B of the generating network = ⟨ψ|ψ⟩_BP; E_B = BP energy on a
3×3 grid and = exact on a tree; H_eff, N_eff Hermitian; gradient = finite difference; one-site DMRG
on a 6-site comb tree at D = 8 reproduces exact diagonalisation to 1e-15 after the first sweep.

3×3 Heisenberg + transverse field (ED −5.04933), random start, 6 sweeps, BP energy per sweep:

| D | E_B sweep 1 → 6 | exact energy of the final state |
|---|---|---|
| 2 | −4.463 → −4.7387 (monotone) | −4.839 |
| 3 | −4.710 → −4.7705 | −4.932 |
| 4 | −4.772 → −4.7887 | −5.0007 |

The state improves with D and is variational in the exact energy; the χ = 1 estimator is 5% off
the exact energy of its own state — that gap is what the MP-BP environments (phase 2) remove.

## The response solve needs no preconditioner at χ = 1 — *measured 2026-09-11*

The response is solved in the fixed-point (Jacobian) form `(1 − J) X′ = ∂λF`, not the Hessian form
`∂²ₓZ_B X′ = −∂λ∂ₓZ_B`. Heisenberg + field, tolerance 1e-10 on the tangent:

| case | tangent dim | Gauss–Seidel sweeps | GMRES matvecs (no preconditioner) | ρ(J) |
|---|---|---|---|---|
| 3×3 D=2 random / DMRG state | 384 | 21 / 10 | 42 / 18 | 0.59 / 0.30 |
| 3×3 D=4 random / DMRG state | 1536 | 17 / 19 | 38 / 32 | 0.50 / 0.42 |
| 4×4 D=3 random / DMRG state | 1728 | 20 / 13 | 41 / 26 | 0.59 / 0.61 |

`1 − J` is well conditioned whenever BP is a stable fixed point (ρ(J) < 1), because the message
normalisation has already divided out the vertex and edge scalars that make the raw Bethe Hessian
badly scaled, and fixed the gauge (scale) null space. Formally the Hessian is (minus) a
block-diagonal metric built from the Zᵥ, Zₑ times `(1 − J)`, so block-Jacobi preconditioning of the
Hessian form is the Jacobian form. Gauss–Seidel is the cheaper of the two here; GMRES is the
fallback when ρ(J) → 1, and the form to use for χ > 1 environments where the update iteration is
known not to be stable.

## Phase 2: CTM (matrix-product BP) environments — *2026-09-11*

`dmrg(ψ, H; alg = "ctmrg", maxdim = χ)`. The χ = 1 messages are replaced by the finite-CTMRG rings
of the generating norm network: `CTMEnvironmentCache(QuadraticForm(ψ, G(0)), χ)`. Nothing in the
CTM engine changes — the three-layer factor list `[ket, G, bra]` goes through the corner moves as
any `AbstractForm` does, and the auxiliary index rides along on the interfaces (dimension r + 1 = 2
for the TFIM, since ZZ has operator-Schmidt rank 1). Everything below is measured on the 4×4 TFIM
at g = 3 (`H = −Σ ZZ − g Σ X`, ED −50.186623883), starting from imaginary-time simple-update states.

### Energy at fixed rings (envelope theorem)

`bethe_energy(cache, gen) = Σᵥ ⟨ringᵥ · ∂λGᵥ⟩ / ⟨ringᵥ · G(0)ᵥ⟩`: only the vertex regions of the CVM
functional carry λ explicitly, and the block rescaling cancels per ratio. Error against the exact
energy of the state:

| χ | D = 2 `:cut` | D = 2 `:cycle` | D = 3 `:cut` | D = 3 `:cycle` | D = 3 bMPS |
|---|---|---|---|---|---|
| 8 | 1.4e-7 | 3.8e-9 | 2.1e-4 | **9.1e-7** | 3.1e-6 |
| 16 | 4.0e-11 | 3.7e-9 | 8.3e-7 | **2.0e-8** | 5.1e-8 |
| 32 | 2.8e-14 | 3.7e-9 | 1.5e-8 | 1.8e-8 | 5.9e-11 |
| 64 | 2.1e-14 | 3.7e-9 | 3.1e-12 | 1.8e-8 | — |

The stationary `:cycle` rings give the MP-BP ε² energy — 100–200× better than `:cut` at matched χ
where both are truncated — but **floor at ~2e-8 (D = 3) / 4e-9 (D = 2) independent of χ**, while
`:cut` goes to machine precision at lossless χ. The plain norm network under `:cycle` has no such
floor (F and ⟨X⟩ exact to 1e-16 at the same χ), and none of the CTM convergence signals (F, worst
region, marginals) see it, because at λ = 0 the half-insertion sector `a > 0` of the auxiliary
index never crosses a truncated interface in a CLOSED contraction — every dst factor is `1 ⊗ |0⟩`
there. The cycle's invariant-subspace criterion is therefore blind to that sector, and the energy
(which reads it through the ring legs adjacent to the vertex) inherits whatever residual the Krylov
solve left. Open problem; `:cut` sees the sector in its SVD and has no floor.

### H_eff without a Hessian: finite difference of the effective ring

The one-site problem needs the response of the rings to λ. Instead of a Hessian or Jacobian solve on
the environment tangent space, the effective ring

    E_λ ψᵥ = ring_λ · G(λ)ᵥ · ψᵥ

is differentiated by a central finite difference of the rings **re-converged at ±λ, warm-started
from λ = 0** (`generating_cache(ψ, gen, χ; λ = ±λ, seed = cache)`). `E_λ` is closed over every
truncated interface, so the interface gauge — biorthogonal pairs, sweep-to-sweep basis rotations —
cancels and no tangent gauge fixing is needed. The energy never sees the finite difference (it is
the envelope derivative above); an error in `H_eff` moves the variational energy at second order.

Gradient of the energy along a random direction at a vertex against a central difference of the
exact energy, relative error:

| state | χ | `:cut` | `:cycle` |
|---|---|---|---|
| 4×4 D = 2 (lossless) | 8, 32 | 4.4e-5 (plateau for λ ≤ 1e-5) | 5e-7 (λ = 1e-6) |
| 4×4 D = 3, site (2,2) | 32 | 1.2e-4 | **8.2e-6** |
| 4×4 D = 3, site (3,3) | 32 | **1.8e-2** | **4.5e-6** |
| 4×4 D = 3, site (3,3) | 64 | 2.8e-5 | — |
| 3×3 random D = 2 (lossless) | 8, 16 | 7.8e-8 (λ = 1e-7); 0.12 (λ = 1e-5) | 6.6e-10 (λ = 1e-6, 1e-7); 0.30 (λ = 1e-5) |

Two things follow. **`:cycle` is the projector for the local solve**: at truncated χ its gradient is
2000× more accurate than `:cut`'s at the same χ (the `:cut` plateau on the lossless state is the
non-stationarity term, ∂ₓZ_B · dX/dψᵥ, which the ring of v sees only through its projectors). And
**λ has a window**: the ±λ solves must stay in the basin of the λ = 0 fixed point. For `:cycle`,
λ ≥ 1e-5 lands on a different invariant subspace (gradient 10–50% off, the warm start degenerates
to a cold solve); λ = 1e-6 and 1e-7 are fine, 1e-8 shows the cancellation error (~1e-7). For
`:cut` the random 3×3 state needs λ ≤ 1e-7. Defaults: 1e-6 (`:cycle`), 1e-7 (`:cut`). The
cancellation floor is low because the ring blocks are norm-rescaled.

### The local solve, and the whitening cutoff is not a tolerance

`N_eff`, `H_eff` are formed densely (`(D⁴d)²`: 162² at D = 3, 512² at D = 4 — fine to D ≈ 6) and the
generalised eigenproblem is solved by whitening `N = U S U†`. A simple-update D = 3 state has bond
directions it barely uses: `N_eff` has eigenvalues down to **5e-11** of the largest. Keeping them
(cutoff 1e-12 or 1e-8) puts the update into directions where `H_eff` is truncation noise, and the
energy RISES monotonically — 3e-3 over a sweep at cutoff 1e-12, a blow-up to E = −34 at 1e-8 in
sweep 2. Cutoff 1e-6 descends cleanly. Default 1e-6.

### Sweeps against ED (4×4 TFIM, g = 3)

| D | χ | projector | start gap | after 2 sweeps | s/vertex |
|---|---|---|---|---|---|
| 2 | 16 | `:cut` | 3.07e-3 | 1.52e-3 (monotone; the D = 2 variational floor) | 0.7–1 |
| 2 | 16 | `:cycle`, λ = 1e-6 | 3.07e-3 | 1.52e-3 (identical trajectory to 1e-9) | 2.5–6, then 35 |
| 3 | 32 | `:cut`, cutoff 1e-6 | 2.25e-4 | **2.77e-5** (small upticks: the 1.8e-2 gradient error) | 11–15 |

The ring energy of the final state equals its exact energy to 1e-12 (D = 2) / 2e-8 (D = 3, the
floor above). The `:cycle` sweep loses its warm starts part-way through the first sweep (per-vertex
time 3 s → 35 s at D = 2) — the one-site update leaves the Vidal gauge, and the `:cycle` basin
moves with it. At D = 3, χ = 32 neither a `:cycle` sweep with warm starts and no re-gauging nor one
with index-preserving re-gauging (`regauge = true`) and cold starts finished ONE sweep within a
60-minute cap (≥ 3.7 min per vertex against 11–15 s for `:cut`): the three `:cycle` solves per
vertex each pay a cold-start-sized price once the state has moved. Re-gauging did not help the
`:cut` run either (cutoff 1e-8, it blew up in sweep 2 where the un-gauged 1e-6 run was stable — the
cutoff, not the gauge, was the variable that mattered) and is off by default. So today the usable
configuration is `projector = :cut` for the sweep, with `:cycle` giving the better energy and the
far better gradient on a fixed state but not yet an affordable sweep.

## Overnight 2026-09-11/12: the `:cycle` floor and basin — two hypotheses tested

Both are cheap experiments on the saved 4×4 states (each run capped at 10 min).

**H1 — the ε² energy is the finite difference of F, not the fixed-ring formula.** A bond term is a
two-point function in disguise, so the response term might be what the fixed-ring energy lacks.
`E_F = (F(+λ) − F(−λ)) / 2λ` with the environments re-converged at ±λ:

| D | χ | `:cut` ring | **`:cut` E_F** | `:cycle` ring | `:cycle` E_F | bMPS |
|---|---|---|---|---|---|---|
| 2 | 8 | 1.4e-7 | **2.5e-10** | 3.8e-9 | 3.3e-9 | — |
| 2 | 16 | 4.0e-11 | 1.4e-9 | 3.7e-9 | 7.1e-9 | — |
| 3 | 8 | 2.1e-4 | **1.4e-6** | 9.1e-7 | 8.8e-7 | 3.1e-6 |
| 3 | 16 | 8.3e-7 | **8.1e-10** | 2.0e-8 | 2.0e-8 | 5.1e-8 |
| 3 | 32 | 1.5e-8 | **3.0e-10** | 1.8e-8 | 2.0e-8 | 5.9e-11 |

(λ = 1e-6 for E_F except the D = 3 χ = 16 `:cut` entry at 1e-7; the D = 2 lossless rows show the
cancellation floor of the difference.) **Falsified for `:cycle`** — its floor is unchanged by the
estimator, so it lives in the environments. **But for `:cut` the FD-of-F energy is the most
accurate estimator of anything measured at matched truncated χ**: 100–500× better than the
fixed-ring `:cut` energy, 25× better than `:cycle` at D = 3 χ = 16, 60× better than boundary MPS.
The implicit derivative d/dλ ln Ẑ with X(λ) re-converged carries the projector response, and the
`:cut` truncation error is evidently smooth in λ. It costs nothing in a sweep (the ±λ caches exist).

**H2 — frozen-projector response.** Rebuild the ±λ blocks through the λ = 0 projectors with
projector-free sweeps (`_ctm_block` over the shifted factor table), removing the eigen-solve and
hence the basin. The machinery is sound: the converged `:cycle` state keeps every interface index
sweep to sweep (36/36), and frozen sweeps reproduce F to 1e-15 in two passes. **Falsified for the
gradient**: 12.7% (site (2,2)) and 1.6% (site (3,3)) error at D = 3 χ = 32 against 8e-6 / 5e-6 with
re-converged projectors. The projector response IS the J^T G J term; it cannot be dropped.

**Where the `:cycle` floor sits.** No plaquette was declined (pure `:cycle` lattice). Working
through the cycle map with the auxiliary leg on the interface: a block on the dst side of a
half-insertion carries only the a = 0 slice at λ = 0, so the left eigenvectors of Λ have no a > 0
component, the right eigenvectors' a > 0 part is fixed by their a = 0 part through one application
of Λ, and no closed contraction reads a > 0 across a truncated interface — the energy reads it only
on the ring's open legs. The truncation criterion is therefore correct in principle; what differs
from the plain norm network (no floor) is that the Krylov solve runs on an interface space with a
nilpotent sector (eigenvalue 0 with Jordan structure), and a χ-independent, estimator-independent
residual is what a non-normal Arnoldi leaves there. The same fragility at λ ≠ 0 (eigenvalues
opening as λ^{1/k}) is the natural reading of the basin loss. The fix is inside
`_ctm_cycle_projectors` — solve for the left vectors on the a = 0 sector and construct the right
vectors' a > 0 part explicitly — which is solver surgery I did not attempt overnight. Deriving the
projectors from the plain norm network and extending them over the auxiliary leg is NOT a
shortcut: kept widths would grow as χ·2^depth, or the response is dropped (H2).

**Localising the floor — and both `:cycle` problems solved without solver surgery.** Per-vertex
energy contributions against the exact per-vertex values (D = 3, ring energy):

| vertex | `:cycle` χ = 16 | `:cycle` χ = 32 | `:cycle` χ = 32, window 1 |
|---|---|---|---|
| (3,1), (1,3), (4,3), (3,4) | 4.5–4.6e-9 each | 4.5–4.6e-9 each | **≤ 9e-15** |
| the other 12 | ≤ 4e-10 | ≤ 6e-14 | ≤ 1.5e-14 |

The whole 1.8e-8 is four boundary vertices, each the dst of a ZZ bond whose partner sits inside a
TWO-site boundary edge block (one- and three-site boundary blocks are exact). That block's interface
rank is capped by its open leg (D·2·D = 18 < χ), so the truncation error there never shrinks with χ:
this is precisely the MP-BP paper's edge plateau (Fig. 15, SM5) — one-point functions still converge
because the edge error is orthogonal to the tangent plane, the bare two-point function does not. The
exact 3×3 window (`vertex_window(cache, v, 1)`, now `bethe_energy(...; window = 1)` and
`energy = :window`) moves the truncated environment past the partner and removes the plateau to
machine precision; the `:cycle` energy is then ~2e-10 at χ = 16, the most accurate estimator measured.

The same rank-capped interfaces carry surplus null modes, and those wander between the λ = 0 and ±λ
solves — that is the basin loss. The existing noise-cliff cut removes them: with `cycle_gapcut = 1e-4`
the `:cycle` gradient at λ = 1e-5 is **2.2e-6** (0.107 without), and the FD-of-F energy is 2.1e-9
(below the 1.8e-8 ring floor). `cycle_rankcut = 1e-8` over-truncates (2.8e-4 gradient, 6.5e-7 energy).
`dmrg(...; projector = :cycle)` now sets `cycle_gapcut = 1e-4` unless told otherwise. The ±λ solves
still cost 35 s each against a 51 s cold start on the 4×4 D = 3 χ = 32, so a `:cycle` sweep remains
several times dearer than `:cut`; the sweep timing is recorded below.

**A `:cycle` sweep, measured.** 4×4 D = 3, χ = 32, gap cut on, per-vertex refresh, CTM
`maxiter = 30`, `tolerance = 1e-10`: the first vertex takes 37 s, every later one ~190 s — the
λ = 0 re-converge after a state change runs to the iteration cap (this χ is far below the
lossless 324 of the generating network, the regime where `:cycle` limit-cycles; the docs' remedy
is χ). The FD-of-F energy descends exactly as `:cut`'s does (after three vertices −50.18647063
against −50.18647066) while the fixed-ring energy is 8e-5 off, so an unconverged `:cycle`
environment still yields a usable implicit-derivative energy. Unbounded, the second vertex alone
exceeded the 10-minute cap twice. Net: with the floor and the basin fixed, a `:cycle` sweep is
correct but ~12× dearer per vertex than `:cut` at this χ, for the same descent.

**Refresh schedules (all `:cut`, 4×4 D = 3, χ = 32, FD-of-F energy, gap to ED):**

| refresh | damping | refreshes/sweep | result |
|---|---|---|---|
| `:vertex` | 0 | 16 | 4.8e-5 after 1 sweep, 2.8e-5 after 2 (13–17 s per vertex) |
| `:sweep` (Jacobi) | 0 | 1 | rises from the first refresh, E = −36 by the third |
| `:checkerboard` | 0 | 2 | 8.3e-4 → 7.9e-3 → blows up |
| `:checkerboard` | 0.5 | 2 | wobbles, then 3.8e-5 after 4 sweeps (193 s total) |
| `:fourcolour` | 0 | 4 | 6.2e-5 after 2 sweeps, then blows up at the 9th refresh |

Simultaneous one-site updates overshoot; damping controls it, at the price of slower descent. Every
refresh is now an acceptance step (`accept_tol`): an update that raises the energy is reverted.
The CTM `tolerance` (1e-10 vs 1e-12) changed neither the time per vertex nor the energy.

**6×6 TFIM (g = 3), `:cut`, per-vertex refresh, FD-of-F energy, acceptance on.** Run through a
resumable driver (checkpoint after every vertex, 10-minute invocations). Start states from
imaginary-time simple update; the reference is boundary MPS on the START state.

| D | χ | start, bMPS (χ_MPS) | after sweep 1 | sweep 2 (partial) | s/vertex | rejected |
|---|---|---|---|---|---|---|
| 3 | 16 | −3.15474148 (64) | −3.15475288 | −3.15476145 (23 of 36) | 45 alone, 55–80 with a second process | 26 of 59 |
| 3 | **32** | −3.15474148 (64) | **−3.15477354** | — | 100 (125 with resume overheads) | 2 of 36 (3 damped) |
| 4 | 16 | −3.15477533 (32) | −3.15477584 (3 vertices) | — | 194 | 1 of 3 |
| 4 | 24 | −3.15477533 (32) | −3.15477744 (7 vertices, row 1 + (1,2)) | — | 310 | 0 of 7 |

Energies per site. The D = 3 χ = 16 sweep lowers the energy by 2e-5 per site, but the rejections
are not random: rows 2–4, columns 3–5 — the BULK — are rejected in both sweeps, boundary vertices
are accepted, and the damped half-step retry does not rescue them (proposed energies up to 6e-3
per site ABOVE the current one). χ = 16 against a 3-bond interface of width 18³ is simply too
little for a trustworthy `:cut` gradient in the bulk of a 6×6; the acceptance step is what keeps
the run variational. Two processes at once cost every converge 3×, so the χ = 32 D = 3 sweep ran
alone: **one sweep, 31 of 36 vertices accepted outright, 3 by the damped half step, 2 rejected (both
on the last row), energy per site −3.15474148 → −3.15477354**, i.e. 3.2e-5 per site below the
simple-update start and within 1.8e-6 per site of the D = 4 simple-update state. The fixed-ring and
FD-of-F energies agree to 2e-7 at χ = 32 where they differed by 1e-4 at χ = 16 — a usable
convergence diagnostic. Wall clock: 36 vertices in 82 minutes including 14 process restarts
(checkpointing the ±λ caches removed the 70–120 s rebuild per restart).

D = 4 at χ = 24 (χ = 16 is below the bulk threshold already at D = 3): the first seven vertices
all accepted, 310 s each, energy per site −3.15477533 → −3.15477744 (2.1e-6 per site in seven
updates, the same per-vertex rate the D = 3 sweep had on its first row). The first bulk vertex
(2,2) did not fit in a 10-minute invocation twice (a 300 s update plus a damped retry), so the
D = 4 sweep stops there under the overnight cap; a full D = 4 sweep at this χ is ~3 h of CPU.

**Symmetries.** `generating_operator` builds a graded auxiliary index as the direct sum of a dim-1
trivial sector (the a = 0 norm slot) and the operator-Schmidt bond of the edge term, so its sectors
are the charges of the Aₐ and every factor is flux-zero; identities come from `op("I", ·)` so the
arrows are right, and the local solve takes the lowest eigenvector that survives projection onto the
site's charge block. Rotated TFIM `H = −Σ XX − g Σ Z` (∏Z conserved; the XX must be a JOINT two-site
operator on graded sites, `register_op!("XXjoint", …; nsites = 2)`, since a per-character "XX" builds
charge-odd single-site X's) on the 3×3 at D = 2, χ = 16, same imaginary-time state in both
representations: F and the ring energy exact for both projectors (≤ 1e-14), and a `:cut` sweep
reproduces the dense sweep to 1e-9 (gap to ED 3.07e-4 in both). The graded sweep is 6× slower
(65 s vs 10 s) — the graded CTM path, not the local solve.

## Cost and what is next

Per vertex update at χ = 1: one response solve plus a Lanczos. With CTM environments: three
warm-started `update`s (λ = 0, ±λ) plus one dense eigensolve; the environment refresh after the
update is the cost, and it is recomputed from scratch after every vertex.

1. The `:cycle` floor in the `a > 0` sector — a stationarity condition (or a Krylov tolerance) that
   sees the half-insertion sector.
2. Keeping the `:cycle` warm starts in their basin through a sweep (gauge, or a projector
   continuation), so its 2000× gradient advantage is available at `:cut` cost.
3. Frozen-environment schedules (update every vertex, then one re-converge) and an incremental
   environment refresh, so the sweep scales past 4×4.
4. Two-site updates so the bond dimension can grow; graded/fermionic operator networks (the
   auxiliary index needs sectors; `factorize_svd` already handles graded terms).
