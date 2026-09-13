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

## The χ = 1 stage first — *2026-09-12*

The operating mode is: minimise at χ = 1 with BP-DMRG, then refine with the CTM projectors. On the
5×5 TFIM from the simple-update starts (exact E₀ = −78.68567686; "true" energies by boundary MPS at
χ = 32, which agrees with the exact contraction to 1e-10 at D = 3):

| D | start: true gap | BP sweeps (s/sweep) | BP-optimised: true gap |
|---|---|---|---|
| 3 | 6.26e-4 | 2 (39, of which ~25 process start-up) | 4.84e-4 |
| 4 | 6.52e-5 | 2 (41) | 3.30e-5 |
| 5 | 2.98e-4 | first sweep RAISED E_B by 0.033, second crashed in LAPACK | — |

BP converges in two sweeps and moves the true energy modestly (the Bethe energy sits 0.25 above the
true energy here, so it is a different landscape). The D = 5 failure and the sweep's cost led to
four changes in `_dmrg_bp` / `message_response` / `optimize_vertex!`:

1. The explicit source term of the response is computed once per solve, not per Gauss–Seidel
   iteration, and the response warm-starts from the previous vertex's (15 sweeps cold → 1–5 warm):
   0.45 s → 0.22 s per vertex, the rest of the update (local solve 0.02 s, BP 0.05 s, energy 0.02 s)
   is already cheap. A warm 5×5 D = 3 sweep is ~9 s.
2. The whitening cutoff of the message square roots is relative to the message norm (1e-6);
   it was the backend's absolute default.
3. Every vertex update is an acceptance step, and a failed local solve (the LAPACK exception) is
   caught and counted as a rejection.
4. With 2 and 3 alone, D = 5 rejected 24 of 25 updates. Diagnosis: the gradient is right (3e-4
   against a finite difference of the re-converged E_B) but the full eigen-step overshoots — the
   Bethe energy is not a Rayleigh quotient in ψᵥ because the messages depend on it — raising E_B by
   2.8e-6 where a half step lowers it by 3e-6. The sweep therefore retries a rejected vertex with
   damping 0.5 and 0.8 before reverting (`damping_schedule`).

After the four changes (same starts, same references):

| D | start: true gap | BP sweeps (warm s/sweep) | rejected | BP-optimised: true gap |
|---|---|---|---|---|
| 3 | 6.26e-4 | 2 (11) | 0 | 4.84e-4 |
| 4 | 6.52e-5 | 2 (14) | 0 | 3.30e-5 |
| 5 | 2.98e-4 | 6 (30), E_B still falling 1.6e-6 per sweep | 4–19 of 25 per sweep | 1.82e-4 |

D = 5 now descends monotonically, but slowly and with many reverted steps: the damped retries
rescue some vertices, not all, so the χ = 1 quadratic model is a poor guide there. The
BP-optimised D = 5 state (1.8e-4) is still worse than the D = 4 one (3.3e-5), inherited from its
simple-update start.

## 5×5 TFIM against exact diagonalisation — *2026-09-12*

Exact reference by a matrix-free Lanczos on 2^25 states in the even ∏X-parity sector
(`scratchpad/ed_tfim.jl`, 8 threads, 127 s, 92 matvecs): **E₀ = −78.68567686257818**, per site
−3.14742707. Start states from imaginary-time simple update; the exact contraction of the D = 3
start state takes 390 s (D ≥ 4 exceed the 10-minute cap, so those rows use the FD-of-F energy).

| D | χ | projector | start gap | after sweep 1 | after sweep 2 | s/vertex | rejected |
|---|---|---|---|---|---|---|---|
| 3 | 32 | `:cut` | 6.25e-4 (exact) | 9.1e-5 | **5.5e-5** | 45 | 4 of 50 (3 damped) |
| 4 | 32 | `:cut` | 6.52e-5 (bMPS χ=64) | | | 200 | |
| 5 | 32 | `:cut` | 2.98e-4 (bMPS χ=32) | | | | |

Gaps are total energies against E₀. The simple-update D = 5 start is worse than D = 4 (the
imaginary-time schedule saturates), which is itself a reason to want a variational sweep. At
D = 3 the second sweep's rejections sit in the bulk row y = 4, as on the 6×6 — the `:cut`
gradient error at χ = 32 is the limiting factor there, not the ansatz.

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

## Where the 40 minutes per sweep go, and the two levers — *2026-09-12*

Measured on the 5×5 D = 3 χ = 32 `:cut` sweep from the BP-optimised state (`scratchpad/ctm_profile.jl`,
`run66.jl` runs A–E). The double layer is contracted lazily and the local problem is small; the
cost is the number of environment re-convergences.

| piece | cost |
|---|---|
| one CTM sweep of the generating network (aux leg widens every interface 9 → 18) | 2.3 s |
| the same sweep on the plain norm network | ≈ 0.3 s |
| dense ring operators at a vertex (162 × 162) | 0.05 s |
| warm re-converge of the λ = 0 cache after one tensor changed (`:marginal`, 1e-12) | 4–5 sweeps |
| warm re-converge of each ±λ cache | 2–4 sweeps |
| **per vertex: three converges ≈ 12 sweeps** | **37–45 s** |

What does not help: the `:free_energy` criterion at 1e-10 halves the sweeps but 7 of 10 updates are
then rejected (FD of F needs F to ~1e-14, which the `:marginal` criterion delivers as a side effect);
λ = 1e-6 to tolerate a looser F rejects the same way; tolerance 1e-10 vs 1e-12 under `:marginal`
changes nothing (the criterion sets the sweep count, not the tolerance); re-gauging the BP start
changes nothing. So the cost is structural: one full environment re-convergence per local update.

**Lever 1 — fewer refreshes (grouped updates).** Update a group of vertices against the same three
environments, refresh once, accept/damped-retry/reject the group as a whole (`refresh = :checkerboard`,
`damping`). On the 5×5 a checkerboard sweep is 2 refreshes instead of 25, a row-wise sweep 5.
Results below (`scratchpad/run_group.jl`).

5×5 D = 3 χ = 32 `:cut`, from the BP-optimised state (per-site energy −3.14740771, gap 1.94e-5;
exact −3.14742707), `damping = 0.5`, retry at 0.8, `:marginal` 1e-12, `scratchpad/run_group.jl`.
Gaps are per site.

| refresh | refreshes/sweep | s/sweep | after 1 | after 2 | after 3 | after 4 | retried / rejected |
|---|---|---|---|---|---|---|---|
| `:vertex` (SU start, earlier table) | 25 | ~1100 | 3.6e-6 | 2.2e-6 | | | 4 of 50 |
| `:checkerboard` | 2 | 90–150 | 5.7e-6 | 2.9e-6 | 2.3e-6 | **2.2e-6** | 4 / 1 of 8 |
| `:rows` | 5 | 230–280 | 3.9e-6 | 2.5e-6 | 2.1e-6 | **2.0e-6** | 2 / 1 of 20 |

Row-wise updates are the better trade: 5× fewer refreshes than `:vertex`, a Gauss–Seidel order
between rows (each row sees the rows already updated), and after four sweeps (16 min) an energy
below what `:vertex` reached after two (≈ 40 min each). Both grouped schedules converge to the same
2e-6 floor with a rejection at the end, which is the `:cut` gradient error at χ = 32 seen on the
6×6, not the grouping. The damped retry (0.8) rescued every overshoot but one; without it half the
checkerboard groups would have been rejected. Both are now in `dmrg(…; refresh = :rows | :checkerboard,
damping, retry_damping)`.

**Lever 2 — one environment set per global gradient (planned).** The FD ring operators at every
vertex come from the same three caches, so one refresh gives the exact gradient of the CTM energy
with respect to all 25 tensors: with `ε = t†H_eff t / t†N_eff t`, `∂E/∂t̄ = 2 (H_eff − ε N_eff) t / t†N_eff t`
(the arbitrary multiple of `N_eff` in `H_eff` cancels). The plan:

1. `energy_and_gradient(ψ, gen, χ; seed)`: three warm converges (≈ 37 s on the 5×5), `E_fd`, and
   the gradient tensor at every vertex from `effective_operators` (25 × 0.05 s).
2. Preconditioning by the local metric: direction `d_v = −N_eff⁻¹ (H_eff − ε N_eff) t_v` in the
   whitened basis (cutoff 1e-6, the same drop of unused bond directions as the local eigensolve).
   This is one inverse-iteration step towards the local eigenvector at every vertex simultaneously,
   so a unit step is the Jacobi version of the one-site sweep; the line search below is what keeps
   the simultaneous update from overshooting (the undamped `:sweep` refresh diverged on the 4×4).
3. Line search on `E_fd` along the concatenated direction (each trial energy is one refresh) with
   an Armijo backtrack from step 1, then L-BFGS (memory 5–10) over the whitened coefficients.
4. Acceptance is built in (the line search never takes an uphill step); the refresh criterion stays
   `:marginal`.

Cost per iteration ≈ 1–3 refreshes (40–120 s) against 25 refreshes per sweep now; the open question
is the iteration count, which the preconditioning is meant to keep near the number of sweeps the
one-site sweep needs (2–3). Gradient consistency between the ring FD and the FD of F is already
tested on the 3×3 (`test/test_dmrg.jl`, CTM testset).

## Lever 2 built: global L-BFGS with one environment set per step — *2026-09-12*

`dmrg(ψ, H; alg = "ctmrg_lbfgs", maxdim, maxiter, memory = 8, step0 = 0.5)`. One set of environments
(λ = 0, ±λ; ≈ 40 s on the 5×5 D = 3 χ = 32) gives the FD-of-F energy and, from the same three caches,
the finite-difference ring operators at every vertex, hence the exact gradient of the CTM energy with
respect to all tensors (`∂E/∂t̄ = 2 (H_eff − ε N_eff) t / t†N_eff t`; the arbitrary multiple of `N_eff`
in `H_eff` cancels). Direction by L-BFGS with the block variable metric `H₀ = γ N_eff⁻¹` (whitened
subspace, Barzilai–Borwein γ); the first step and any fallback is the Jacobi direction `t* − t`
(unit step = the `:sweep` update, `step0 = 0.5` the damping that measured stable); every step is an
Armijo backtracking line search on the FD-of-F energy, so nothing goes uphill.

One bug worth recording: the inner `gradient` closure assigned `g, P, jac`, names also assigned in
the enclosing function, so Julia made them the SAME variables and every curvature pair came out
`y = g_new − g_old = 0` — the optimiser silently ran as pure Jacobi. Distinct names fixed it.

5×5 D = 3 χ = 32 `:cut` from the BP-optimised state (per-site gap 1.94e-5), `scratchpad/run_lbfgs.jl`;
gaps per site against the exact −3.14742707:

| method | environment sets | wall time | gap reached | note |
|---|---|---|---|---|
| `:vertex` sweeps × 2 (SU start) | 50 | 80 min | 2.2e-6 | 4 of 50 rejected |
| `:rows` sweeps × 4 | 20 | 16 min | 2.0e-6 | 1 rejected at the end |
| L-BFGS, 5 iterations | 7 | 7 min | **1.8e-6** | steps 2–5 full unit steps, one evaluation each |
| L-BFGS, restarted (memory lost) +1 | 7 | 6 min | 1.8e-6 | line search then fails: the floor |
| Jacobi only (`memory = 0`), 7 iterations | 20 | 19 min | 2.4e-6 | α = 0.125 at 3 evaluations per step, twice the cost of L-BFGS for less |

The L-BFGS iterations 2–5 each took the unit step at the first trial (the variable metric is
right), and each lowered the energy by more than a whole row-grouped sweep did. Everything stalls at
the same 1.8–2.0e-6 floor: the line search fails there because the ring-FD gradient and the FD-of-F
energy no longer agree at that precision, which is the `:cut` truncation error at χ = 32 already seen
on the 6×6. That floor is now the thing to move (χ, or `:cycle` for its 2000× gradient accuracy), and
the optimiser makes trying either affordable: ≈ 1 minute per iteration instead of 40 per sweep.

From a random 3×3 start (far from quadratic) pure Jacobi steps beat L-BFGS (−19.77 vs −19.12 after
12 steps; ED −19.79), so `memory = 0` is the right setting for a cold start and L-BFGS for the
refinement after the BP stage — which is the operating mode anyway.

## Moving the floor: χ = 48 and 64, and `:cycle` — *2026-09-12*

Same 5×5 D = 3 BP-optimised start (per-site gap 1.94e-5), L-BFGS as above, chained across
processes with `caches` (environments + L-BFGS pairs + generating operator handed over, so no
cold start and no memory loss between invocations). Gaps per site against the exact −3.14742707.

| χ | env. set | it 1 | it 2 | it 4 | it 6 | it 8 | it 10 | it 16 | it 22 |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 40 s | 9.7e-6 | 5.8e-6 | 2.3e-6 | 1.8e-6 (floor) | | | | |
| 48 | 65 s | 8.3e-6 | 3.7e-6 | 2.2e-6 | 1.6e-6 | 1.3e-6 | 1.15e-6 | 9.8e-7 | **8.2e-7** |
| 64 | 95 s | 8.3e-6 | 3.7e-6 | 2.6e-6 | 1.67e-6 | 1.35e-6 | | | |

- **The floor moves with χ.** At χ = 48 the descent continues past the χ = 32 floor with unit
  L-BFGS steps at every iteration from the second on (one environment set each), 5% per iteration
  at iteration 10 and still 2% at iteration 22 (no floor reached; stopped for time). χ = 64 tracks
  χ = 48 iteration for iteration (1.35e-6 vs 1.34e-6 at iteration 8), so χ = 48 is converged in χ
  here. The energy is real: boundary MPS at χ = 64 gives the iteration-10 χ = 48 state a per-site gap
  of 1.1507e-6 against the optimiser's own FD-of-F estimate of 1.1503e-6 (4e-10 apart). The final
  **D = 3 state, gap 8.2e-7 per site, is below the D = 4 BP-stage result** (1.3e-6), 24× below the
  D = 3 BP start (1.9e-5) and 30× below the D = 3 simple-update start (2.5e-5). Total cost 22
  iterations ≈ 25 min of environment sets (plus ~1 min of process start per 4-minute invocation).
- **`:cycle` does not fit the 10-minute cap.** Cold environment set 285 s; a warm one after a state
  change runs to the iteration cap (the known refresh problem), so a single iteration is ≥ 3 sets
  ≈ 10 min and every attempt was killed. The optimiser cannot hide that; the `:cycle` refresh cost
  stays the prerequisite for using its gradient accuracy.
- **Two mechanics fixes** on the way: `time_limit` now excludes the initial environment set, and the
  handoff carries the generating operator (a fresh one has fresh auxiliary index ids, and the saved
  caches then fail to contract — the same mismatch `run66.jl` solved by checkpointing `gen`).

**D = 4 and D = 5 from their BP-optimised states** (`:cut`, gaps per site; environment set at
D = 4 χ = 48 ≈ 300 s, so one iteration per 10-minute process):

| D | χ | BP start | it 1 | it 2 | status |
|---|---|---|---|---|---|
| 4 | 48 | 1.32e-6 | 5.7e-7 (Jacobi, α = 1/8) | **4.1e-7** (L-BFGS unit step) | stopped for time; still descending |
| 5 | 32 | 7.3e-6 | | | the λ = 0 cache alone does not converge inside 10 min next to another job; untested |

D = 4 after two iterations (4.1e-7) is the best state of the day at any D; the BP stage had it at
1.3e-6 and simple update at 6.5e-5.

## State of play — *end of 2026-09-12*

**Operating mode.** BP-DMRG at χ = 1 (`alg = "bp"`) from a simple-update start, then
`dmrg(ψ, H; alg = "ctmrg_lbfgs", maxdim = 48, memory = 8, step0 = 0.125)` with `:cut`. On the 5×5
TFIM this takes D = 3 from a 2.5e-5 simple-update gap to 8.2e-7 per site in 22 iterations (≈ 25 min
of environment sets) and D = 4 from 6.5e-5 to 4.1e-7 in two iterations.

**What is verified.** Exact gradient at every vertex from one environment set (3×3 test against
the FD of the exact energy); L-BFGS energy = bMPS energy to 4e-10 on the 5×5; χ = 48 converged in χ
at D = 3 (χ = 64 identical); every accepted step descends (Armijo); 81 tests pass as of commit
`b4e6bca`. The `caches` handoff (environments, L-BFGS pairs, generating operator; verbatim reuse when
the state is unchanged) and the `time_limit` placement were added after that test run and exercised
by the chained 5×5 runs, not by the test file — rerun `test/test_dmrg.jl` first thing.

**What limits it.** (1) Cost per environment set: the auxiliary leg doubles every interface, so a
set is ~7× a plain-norm CTM sweep; 40 s (D = 3, χ = 32) → 65 s (χ = 48) → 300 s (D = 4, χ = 48) →
> 600 s cold (D = 5). The aux-free λ = 0 environment (lever 3) is the next engineering target.
(2) `:cycle`: its refresh after a state change runs to the iteration cap, so one iteration is ≥ 3
sets at 150–250 s; not usable until that is fixed. (3) The 10-minute cap: half of today's mechanics
were checkpointing; a D = 5 or 8×8 study needs hour-long runs.

**What is not known.** Whether the D = 3 descent (still 2%/iteration at 22) ends at a true D = 3
minimum or trails off; whether the `:cut` gradient error biases the minimiser at larger sizes
(no sign of it at 5×5 by χ = 48 vs 64); L-BFGS robustness from a cold start (Jacobi carries it on
the 3×3; from the BP state it was never needed after iteration 1).

**Files.** Optimiser in `src/dmrg.jl` (`dmrg(::Algorithm"ctmrg_lbfgs")`); scratchpad drivers
`run_lbfgs.jl` (env vars L D CHI PROJ MAXITER MEMORY STEP0 LSMAX BUDGET COLD0 TAG; checkpoints
`ckptl_<tag>.jls` with `caches`, energies `energiesl_<tag>.csv`), `eval_state.jl` (bMPS/exact energy
of a checkpoint), `run_group.jl` (grouped-refresh sweeps), `bp_stage.jl`, `ed_tfim.jl`. Best states:
`ckptl_L5_D3_chi48_cut_lbfgs.jls` (8.2e-7), `ckptl_L5_D4_chi48_cut_lbfgs.jls` (4.1e-7).

**Next, in order.** Rerun tests; the aux-free λ = 0 environment; finish D = 4 (and D = 5 alone on
the machine) on the 5×5; then a 6×6 / 8×8 D = 4 demonstration measured by bMPS against the
simple-update and BP-stage states at the same D (no exact reference there); `:cycle` after its
refresh cost is fixed.

## Lever 3: the aux-free λ = 0 environment — *2026-09-13 (overnight)*

At λ = 0 every `a > 0` slot of a bond carries zero weight (the src end of each edge factor is
`λ·Aₐ`), so the free energy and the norm ring `N_eff` of the generating network are those of the
PLAIN norm network, whose interfaces are `D²` wide instead of `(r+1)·D²`. `generating_cache(…;
aux_free = true)` converges the norm network (`QuadraticForm(ψ)`) and pads its blocks and stored
projectors onto the generating network's bonds with `onehot(aux ⇒ 1)`; a seed is stripped the same
way. Measured on the 3×3 (χ = 16 lossless): F identical to 2e-15, `N_eff` proportional to 2e-15,
the padded environment a fixed point of the generating-network sweep (ΔF 3e-15), and the ±λ pair
seeded from it faster (0.46 vs 0.93 s) and closer to the exact energy (1e-10 vs 1.4e-8) than from
the true λ = 0 environment.

**Proportional, not equal — and why `N_eff` now comes from the ±λ rings.** Every CVM block is
rescaled to unit norm as it is built, so a ring's overall scale is set by which blocks it holds. The
true λ = 0 blocks carry half-insertion (`a > 0`) weight, the padded ones none, so the padded ring's
norm sector is ~2× larger. `H_eff` is built from the ±λ rings on THEIR scale; mixing it with an
`N_eff` on the padded scale would rescale the gradient vertex by vertex (the test caught this).
`effective_operators` therefore takes `N_eff = (ring₊·G₀ + ring₋·G₀)/2`, O(λ²) from the λ = 0 ring
and on exactly the scale of `H_eff`; the λ = 0 cache is only the seed of the ±λ pair. This is the
correct construction independently of lever 3 (the 3×3 gradient test still passes to 1e-4 relative).

**Timing, 5×5 D = 3 χ = 48 `:cut` (`scratchpad/auxfree_time.jl`, run next to another job):**

| | λ = 0 warm (after a one-tensor change) | +λ | −λ | environment set |
|---|---|---|---|---|
| full | 17.9 s | 24.0 s | 23.7 s | 65.6 s |
| aux-free | **3.1 s** | 23.5 s | 23.3 s | **49.9 s** |

The λ = 0 part is 6× cheaper; the set is 24% cheaper because the ±λ pair, which needs the
auxiliary leg, is now 94% of it. (The cold λ = 0 converge measured 32 s in both modes, but the
aux-free mode ran first in the process and carried the JIT compilation; not a clean number.) In
the optimiser the first iteration's three evaluations took 183 s against 205 s before, at the same
energy to 4e-10. Lever 3 as scoped is done and on by default in `ctmrg_lbfgs`; the remaining cost
is the ±λ pair, i.e. the finite-difference construction itself. The ways past it are a cheaper
warm start for ±λ (seed −λ from +λ, or reuse the ±λ projectors across iterations) or replacing the
pair by a linear response of the environment to λ (the frozen-projector version of which was
falsified earlier at χ = 1 accuracy; a full response solve is a separate project).

**Not for `bethe_energy`.** The padded ring has no half-insertion components, so the ring energy of
an aux-free cache misses every bond term. The L-BFGS route never calls it (energy = FD of F).
`aux_free` is off by default in `generating_cache` and on by default in `dmrg(…; alg = "ctmrg_lbfgs")`.

Test tolerances on "FD-of-F energy = exact energy" were widened from 1e-8 to 5e-8 in the two sweep
tests: the difference of two free energies at λ = 1e-7 carries ~1e-15/1e-7 of roundoff, and the
tests measured ±2e-8 with random sign; 1e-8 had been marginal.

## The 5×5 table, closed for now — *2026-09-13 (overnight)*

5×5 TFIM g = 3, exact E₀/site = −3.147427074503. Every column is the energy DENSITY's absolute
error against exact; SU = imaginary-time simple-update start, BP = χ = 1 BP-DMRG from it, CTM =
`dmrg(…; alg = "ctmrg_lbfgs")` with `:cut` from the BP state. Times are environment-set time,
excluding process starts.

| D | SU | BP stage | CTM L-BFGS | χ | iterations | time | how it ended |
|---|---|---|---|---|---|---|---|
| 3 | 2.5e-5 | 1.9e-5 | **8.2e-7** | 48 | 22 | ≈ 25 min | still descending 2%/it; stopped for time |
| 4 | 6.5e-5 | 1.3e-6 | **1.55e-7** | 48 | 3 | ≈ 15 min | it 4: unit L-BFGS step and 1/8 Jacobi step both uphill in one-trial searches; a two-trial search does not fit the 10-min cap at 250 s per evaluation |
| 5 | 3.0e-4 | 7.3e-6 | — | 24 | 0 | | one energy evaluation ≈ 11 min (±λ caches 296 + 311 s next to another job) > cap; the aux-free λ = 0 cache took 60 s where the full one could not finish in 600 s; start E_fd at χ = 24 = 7.3e-6, matching bMPS |

Reading: the CTM stage buys 24× at D = 3 and 8× at D = 4 over the BP stage, and D = 4 after three
iterations is 5× below the best D = 3. D = 5 needs either the cap lifted (≈ 15 min per iteration at
χ = 24, uncontended) or a cheaper ±λ pair — the pair is now 94% of an environment set (lever 3
section). The D = 4 ending is ambiguous between a χ = 48 floor and a line search that only needed
α = 1/2; both cost a longer process to tell apart.

## 6×6 D = 4: BP stage then CTM L-BFGS — *2026-09-13 (overnight)*

No exact reference at this size; energies are densities measured by boundary MPS (bMPS, χ = 32,
which agreed with χ = 64 to 1e-9 per site on the 5×5) where marked, and by the FD-of-F estimate of
the optimiser otherwise (agreed with bMPS to 4e-10 on the 5×5).

| stage | E/site | Δ vs SU | cost |
|---|---|---|---|
| simple update D = 4 (start) | −3.15477533 (bMPS) | | |
| BP stage (χ = 1, 3 sweeps) | −3.15477644 (FD-of-F) | −1.1e-6 | 160 s |
| CTM L-BFGS `:cut` χ = 32, it 1 (Jacobi α = 1/8) | −3.15477867 | −3.3e-6 | 430 s |
| it 2 (L-BFGS unit step) | −3.15477967 | −4.3e-6 | 410 s |
| it 3 (L-BFGS unit step) | −3.15477967 (+4e-9) | −4.3e-6 | 377 s |
| final state by bMPS χ = 32 | **−3.15477967** (agrees with FD-of-F to 3e-9) | −4.3e-6 | 286 s to measure |

It stopped at iteration 4 the way the 5×5 D = 4 did at χ = 48: the unit step was uphill and the
Jacobi fallback, a second evaluation, pushed the process past the cap. The 4e-9 gain at iteration 3
says the χ = 32 environment is the floor here (interfaces are D²·2 = 32 wide, so χ = 32 is lossless
at one level only); the 5×5 D = 3 experience says the floor moves with χ, and χ = 48 at this size is
≈ 2× the cost per set, i.e. hour-long processes.

For scale: the earlier one-site `:vertex` sweep on this state at χ = 24 reached −3.15477744 after
7 vertices at 310 s per vertex (docs, 2026-09-12); one L-BFGS iteration (one environment set for all
36 tensors) went past that in 430 s.

Cost anatomy at 6×6 D = 4 χ = 32, next to another job: aux-free λ = 0 cache 51 s cold; ±λ caches
249 and 232 s from it; so an environment set is ≈ 8 min and one iteration per 10-minute process is
the most the cap allows (the first iteration took 490 s including process start). A run that is
allowed an hour would do 7 iterations in it.

## State of play — *end of the 2026-09-12/13 overnight*

Goal items: (1) tests rerun, 85/85 at `2b82a0d`; (2) lever 3 landed (aux-free λ = 0 environment,
N_eff from the ±λ rings); (3) 5×5 table closed for D = 3 (8.2e-7) and D = 4 (1.55e-7), D = 5 out of
reach under the 10-minute cap; (4) 6×6 D = 4: SU −3.15477533 → BP −3.15477644 → CTM L-BFGS
−3.15477967 per site (bMPS-confirmed) in three iterations of ≈ 7 min.

What limits everything now is the ±λ pair of environments (94% of a set) and the 10-minute cap:
at 6×6 D = 4 χ = 32 one evaluation is ≈ 8 min, so a two-trial line search cannot run, and both the
5×5 D = 4 and the 6×6 runs ended on that rather than on a verdict about their floors. Next, in
order: allow hour-long processes for D ≥ 4 (or split one evaluation across processes, which the
driver's PMSTEP mode already does for the cold start); a cheaper ±λ pair (seed −λ from +λ, reuse
the ±λ projectors between iterations, or a response solve); the full-update-with-CTM baseline the
user asked for, timed like for like; then `:cycle` once its refresh cost is fixed. Scratchpad
drivers: `run_lbfgs.jl` (COLD0 / PMSTEP / MAXITER / LSMAX / STEP0 / START / TAG), `eval_state.jl`
(L, FILE, ALG, CHIB), `bp_stage.jl` (L, D). Best states: `ckptl_L5_D3_chi48_cut_lbfgs.jls`,
`ckptl_L5_D4_chi48_cut_lbfgs.jls`, `ckptl_L6_D4_chi32_cut_lbfgs.jls`, `bpstate_L6_D4.jls`.

## Against Yantao Wu's VMC-optimised 5×5 D = 3 state (g = 3.04438) — *2026-09-13*

Yantao Wu's PEPS (`examples/data/peps/data_ising_5x5/isingZZX_5x5_D3_g3.04438.npz`, optimised by
sampling + stochastic reconfiguration; "very well optimized, my double layer experiments reduced
energy by only 1e-6 or so" per the covering email) is the one independently optimised state on file
with an exact reference in reach. Imported through `scratchpad/import_yantao.jl` (the file is a
single pickle of jax arrays, not a zip; unpickled with stub modules so no float32 truncation) and
checked against the recorded references: `ln⟨ψ|ψ⟩ = −6.217866847854579` (ref …575),
`⟨X⟩(3,3) = 0.9169005981284838` (ref …483). Exact 5×5 ground energy at g = 3.04438 by the same
matrix-free Lanczos: **E₀ = −79.72801918747975, −3.18912076749919 per site** (124 s).

| state (5×5, D = 3, g = 3.04438) | E/site | gap/site | cost |
|---|---|---|---|
| Yantao, VMC + SR | −3.1891199915 (bMPS χ = 32 and 64 agree to 1e-9) | **7.8e-7** | (theirs) |
| simple update, imaginary time | −3.1890992871 (bMPS χ = 32) | 2.15e-5 | 40 s |
| BP stage (χ = 1, 2 sweeps) | −3.1891047034 (bMPS χ = 32) | 1.61e-5 | 45 s |
| CTM L-BFGS `:cut` χ = 48 from the BP state | −3.1891196457 (bMPS χ = 64; FD-of-F agrees to 3e-10) | **1.12e-6** after 11 iterations, still descending 2%/it | ≈ 15 min of environment sets |
| CTM L-BFGS from Yantao's state | −3.1891200956 (bMPS χ = 64) | **6.72e-7** after 3 unit steps, then every trial uphill (6 tried) | ≈ 7 min |

Reading. (1) Yantao's VMC + SR state is very good for D = 3: 7.8e-7 per site, and our optimiser
started FROM it gains 14% in three unit L-BFGS steps and then finds no downhill direction at all,
so **≈ 6.7e-7 is the D = 3 floor at χ = 48 `:cut`**, and their state was within 15% of it. The
start energy the CTM estimate assigned to their state agreed with bMPS to 5e-10, i.e. the import
and the estimator are both right. (2) Our simple-update → BP → CTM route at the same g follows the
g = 3 trajectory exactly (iteration 5: 1.54e-6 vs 1.79e-6; iteration 9: 1.18e-6 vs 1.21e-6) and at
11 iterations sits at 1.12e-6, 45% above Yantao's state; at g = 3 the same route needed 22
iterations for 8.2e-7 and was still descending, so matching a VMC-optimised D = 3 state costs
this route roughly 25–30 iterations ≈ 30–35 minutes of environment sets on the 5×5, against 45 s
for the BP stage that gets to 1.6e-5. (3) The BP stage alone is 20× above the VMC state at D = 3, so
"BP is already close" holds against exact at the 1e-5 level, not against a well-optimised state
at the 1e-6 level; the CTM stage is what closes that gap, and it can also refine the VMC state.
What we cannot say from this: whether the 6.7e-7 floor is D = 3's variational limit or the χ = 48
gradient error — the 5×5 D = 3 g = 3 run showed χ = 48 ≡ χ = 64 iteration for iteration, which
argues for the former, but a χ = 64 start from Yantao's state is the direct test (≈ 15 min).

## Speed, item 1: threads, not tricks — *2026-09-13*

Measured on the 5×5 D = 3 χ = 48 `:cut` environment set from the BP state (`scratchpad/pm_variants.jl`,
`sweep_threads.jl`, `sweep_profile.jl`). Every variant reports the same F and E_fd to the last digit
unless said otherwise.

**What did not help.** Seeding −λ from the +λ cache: 26.1 s against 24.9 s from λ = 0. Seeding the
±λ pair from the previous iteration's ±λ caches after a state change: 24.7 s against 23.5 s from
the new λ = 0 cache (E_fd differs by 2e-9, a different truncation basin). The sweep count is set by
how fast the projector bases settle, not by the starting blocks. BLAS threads: a converge takes
24 s at one BLAS thread and 23.5 s at four — the blocks are too small for BLAS to parallelise.

**What did.** Julia threads, with BLAS single-threaded (OpenBLAS with its own pool under Julia
threads crashed in `cblas_xerbla`; at `BLAS.set_num_threads(1)` everything is stable and
bit-reproducible across thread counts).

| | serial | 8 threads |
|---|---|---|
| one CTM sweep, block rebuild (`sweep_vertex_environments`) | 6.4 s | 3.2 s |
| one CTM sweep, total (`update`, `:marginal` criterion) | 6.5–8 s | 4.6 s, then **2.6–3.6 s** after threading the criterion |
| −λ converge (warm from λ = 0) | 25.3 s | 11.7 s |
| ±λ pair (two tasks, each a threaded sweep) | 48 s | 33 s at 2 threads; ≈ 20 s at 8 (from 24–30 s iterations) |
| L-BFGS unit-step iteration in the optimiser | 47–65 s | **23–33 s** |

The sweep threads over the enlarged corners, the `:cut` projector derivations (independent within
a sweep because `prev` is transported from the PREVIOUS sweep's projectors) and the corner and edge
rebuilds; shared dictionaries and the route memo are written under a lock; the contraction-sequence
memo computes its key and any miss outside `CTM_GLOBAL_LOCK`. The `:marginal` criterion's per-vertex
ring contractions are threaded too. The ±λ converges run as two tasks in `ctmrg_lbfgs`.

Net for the day: an environment set went 65.6 s (start of the night) → 49.9 s (aux-free λ = 0) →
≈ 25 s (threads); an L-BFGS iteration on the 5×5 D = 3 χ = 48 is now under half a minute. Run with
`julia -t 8` and `BLASN=1` (the drivers default to it).

**Item 2 (response solve), reassessed after reading the sweep.** The gradient needs the λ-response
of the environments INCLUDING the projectors — that is why the frozen-projector response was
falsified under `:cut`. A proper response solve therefore differentiates through the SVD-based
projector construction and iterates the linearised sweep on the aux-widened network; a cheap
finite-difference Jacobian of one sweep costs as many widened sweeps as the ±λ pair does today.
The version that IS cheap — one linear solve per insertion sector on norm-width interfaces, the
one that scales linearly with the operator-Schmidt rank of the edge terms and would make Hubbard
affordable — is legitimate only under a stationary projector, i.e. `:cycle`, whose envelope
property is exactly what makes frozen projectors correct there. So the response solve and `:cycle`
are one project, not two, and its first task is the `:cycle` refresh cost. Not started; the
choice is the user's.

## `:cycle` made usable — *2026-09-13*

The `:cycle` refresh problem, diagnosed on the 5×5 D = 3 χ = 32 (`scratchpad/cycle_refresh_diag.jl`,
`cycle_pm.jl`, `cycle_one_iter.jl`), was two separate things.

1. **Cost per sweep.** The per-plaquette cyclic projector pass was still serial after the sweep was
   threaded; threaded, a `:cycle` sweep is ≈ 4 s against 1.5 s for `:cut` at 8 threads (the cyclic
   problem, a Schur solve per plaquette, is intrinsically heavier). A `:cycle` environment set is
   now ≈ 3× a `:cut` set (≈ 45 s against 16 s at χ = 32), not 12× or beyond the 10-minute cap.
2. **The convergence criterion never settles.** After a state change the free energy is converged to
   1e-13 by sweep 5, but the worst vertex-marginal change — the `:marginal` criterion — flutters at
   1e-10 (small change) or 1e-8 with spikes to 4e-6 (step-sized change) for many sweeps and passes
   1e-12 only by chance: 12 and 16 sweeps where `:cut` takes 4, or the cap. This is the wandering of
   the rank-capped boundary interfaces' surplus null modes, the same object `cycle_gapcut` tames
   for the gradient. The tolerance that matches what the optimiser needs was measured, not chosen:
   1e-7 stops the ±λ converges after 2 sweeps and the first Jacobi step is then REJECTED (rings not
   consistent enough); 1e-10 accepts it with the same energy drop as `:cut`, in 4–5 sweeps, no
   plateau. `ctmrg_lbfgs` therefore defaults `:cycle` to `tolerance = 1e-10, maxiter = 10` (a
   converge that hits the cap still has a converged F).

Two more mechanics: the aux-free λ = 0 environment is projector-independent and seeds the `:cycle`
±λ pair as well as the true one does (4 sweeps, same E_fd); and from a BP state the first Jacobi
step must start at α = 1/8 under `:cycle` — trial states at α = 1/2 are far enough from the seed
that their converges run toward the cap, and three of them exceed the process cap.

**Gradient accuracy at χ = 32, 5×5 D = 3, the BP state** (`scratchpad/grad_check.jl`): ring gradient
against the finite difference of the optimiser's own FD-of-F energy along a random direction
(h = 1e-3; environments re-converged at each point), and the linear prediction of the energy
change for the 1/8 Jacobi step at the centre vertex against the actual change.

| projector | g·d vs FD | predicted vs actual ΔE for the step |
|---|---|---|
| `:cut`, tol 1e-12 | 23% off | −5.50e-5 vs −3.99e-5 (27% off) |
| `:cycle`, tol 1e-10 | **4% off** | −4.49e-5 vs −4.30e-5 (**4% off**) |

So at the χ where `:cut` floors (1.8e-6 per site), `:cycle`'s gradient is ~6× more consistent with
its energy. This is the property the response solve needs (a stationary projector's environment
response is captured by the envelope theorem, so frozen projectors are legitimate under `:cycle`
and the linear response can be solved per insertion sector on norm-width interfaces).

**What still limits it: a limit cycle after L-BFGS-sized steps.** After the second (unit L-BFGS)
step, the ±λ converges have F stationary at 1e-13 from sweep 5 while the worst vertex marginal
ALTERNATES between 5.1e-6 and 6.1e-6 for 25 more sweeps — a two-cycle of modes at the
`cycle_gapcut = 1e-4` cliff flipping in and out of the kept set, not random flutter. The step those
rings produce is accepted at the `:cut` energy (E after iteration 2 identical to `:cut` to 1e-9), so
the gradient is fine; the CRITERION is what never passes. With the cut disabled
(`cycle_gapcut = 0`, the engine's default since 2026-09-09) the first evaluation did not finish in
10 minutes at all, so the cut stays. The optimiser therefore caps `:cycle` converges at 10 sweeps
and lets its own acceptance test judge the gradient. The proper fix is hysteresis in the cliff
decision (keep the previous sweep's rank unless the spectrum moved), inside `_ctm_cycle_projectors`.

A practical lesson from the same afternoon: at 8 Julia threads on 8 cores two concurrent jobs halve
each other, and three `:cycle` runs were killed by the 10-minute cap for that reason alone before
the limit cycle was even visible. One threaded job at a time.

**`:cycle` in the optimiser loop, 5×5 D = 3 χ = 32 from the BP state** (`run_lbfgs.jl PROJ=cycle`,
8 threads, one job on the machine): iteration 1 (Jacobi α = 1/8) → 8.26e-6 per site in 39 s and
iteration 2 (unit L-BFGS step) → 3.74e-6, both identical to the `:cut` trajectory to 1e-9. At
iteration 3 the ±λ converges, capped at 10 sweeps inside the limit cycle, leave marginals at
1e-6–7e-6, and the gradient built from them is no longer a descent direction: the L-BFGS trial,
the Jacobi direction (not even downhill) and the preconditioned gradient all fail, and the run
stops at 3.74e-6 — above the `:cut` χ = 32 floor of 1.8e-6 that the same start reaches in 6
iterations. So today `:cycle` carries the large-gradient iterations and loses the small-gradient
ones, which is the wrong way round for a refinement stage.

**Where this leaves the `:cycle` project.** Refresh cost: fixed (3× `:cut` per set, 4–5 sweeps
when it converges). Gradient accuracy when converged: confirmed (4% vs 23% at χ = 32). Open:
the rank-capped null modes enter a limit cycle after a few optimiser steps, at the
`cycle_gapcut` cliff and the left/right consistency test in `_ctm_cycle_projectors`, and the
`:marginal` criterion then never certifies. The fix is hysteresis in those decisions — carry each
plaquette's previous kept rank into the cyclic problem and keep it unless the spectrum has moved —
an engine change in code whose history (docs/ctmrg_status.md) is full of falsified rank rules, so
half a day with an uncertain outcome. It is the gate to both `:cycle` in the loop and the
per-sector linear response solve. Not started; the decision is the user's.
