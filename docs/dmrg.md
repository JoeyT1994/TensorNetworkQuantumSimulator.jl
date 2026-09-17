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

## Track 1: Heisenberg on the open hexagonal lattice — *2026-09-15*

First model beyond the TFIM. `H = J Σ_e S·S = (J/4) Σ_e (XX + YY + ZZ)`, J = 1, on
`named_hexagonal_lattice_graph(NX, NY)` (open; vertices are grid positions with holes, degrees 2
and 3; hex(2,2) = 16 sites on a 6×3 box, hex(3,3) = 30 sites on 8×4). What this exercises that the
TFIM did not: an edge term of operator-Schmidt rank 3 (auxiliary index of dimension 4, so the ±λ
interfaces are 4·D² wide against 2·D² before), a grid with unoccupied positions in the CTM engine,
and a three-colour gate schedule. Drivers now take `MODEL=heis NX NY J` through
`scratchpad/model.jl`; the simple-update start is imaginary-time `Rxxyyzz` (θ = −i·dτ·J/2) from a
Néel product state, real tensors throughout. Exact reference for hex(2,2): Lanczos in the Sz = 0
sector (dimension 12870), **E₀ = −7.816281762810088, −0.48851761 per site** (`ed_heis.jl`, 6 s).

**Correctness.** The optimiser's FD-of-F start energy on hex(2,2) matched the exact contraction of
the BP state to 3e-9, and its final D = 3 energy matched the exact contraction of the final state to
2e-8. So the rank-3 auxiliary index and the holed grid contract correctly through the generating
network, the λ = 0 aux-free padding included.

**hex(2,2), errors per site against E₀:**

| D | SU start | BP stage (χ = 1) | CTM L-BFGS `:cut` χ = 48 | iterations, time |
|---|---|---|---|---|
| 3 | 1.433e-2 | 1.434e-2 | **1.269e-2** (converged: Δ < 2e-8, then no downhill direction) | 13 it, 3.1 min |
| 4 | 6.29e-3 | 6.15e-3 | **1.137e-3** (exact contraction agrees with FD-of-F to 2e-8; still descending 2%/it) | 10 it, ≈ 20 min of environment sets |

Two things differ from the TFIM. (1) The BP stage does nothing for the true energy here (it lowers
the Bethe energy, which is 4% off, and leaves the exact energy where simple update put it, or 6e-5
worse at D = 3); the χ = 1 environment is too poor for the Heisenberg antiferromagnet, whose loop
corrections are not small at D = 3–4. (2) The gap is 1e-2, not 1e-6: at D = 3 and 4 the ansatz, not
the optimiser, is the limit — the CTM stage converges to the D = 3 variational floor in three
minutes and cannot go further. Heisenberg on the honeycomb needs D ≥ 5–6 for 1e-3; that is the next
run and it is a matter of environment cost only.

**Cost, and an engine fix it forced.** An environment set at hex(2,2) D = 3 χ = 48 is ≈ 3 s at
6 threads (16 sites, 6×3 box); a unit L-BFGS step is 3–4 s. At D = 4 the ±λ interfaces are
4·D² = 64 wide against χ = 48, and every converge after a step ran to the 40-sweep cap (40–130 s)
with F at 1e-15 and the marginals at 1e-9: the binding term was the raw C/T state distance the
`:cut` criterion folds in, sitting at 1e-4–3e-3 for ever as the hard-truncated bases rotate. Under
the `:marginal` criterion that gauge-dependent term is redundant (the marginal distance is the
full-coverage stationarity signal), so `update` no longer folds it in for `:marginal`. The same
term is what made the 5×5 D = 4 environment sets cost 250 s. At hex(3,3) (30 sites, 8×4 box) D = 3 χ = 48 an environment set is ≈ 22 s at 8 threads and a unit L-BFGS step 23 s — 7× the 5×5 TFIM D = 3 cost per site, the price of the rank-3 edge term (4·D² interfaces) on a lattice with more, longer boundaries.

**hex(3,3), 30 sites, D = 3 (no exact reference; energies per site by boundary MPS at χ = 32):**

| stage | E/site | Δ vs SU |
|---|---|---|
| simple update | −0.48920388 | |
| BP stage (4 sweeps, 45 s) | −0.48924359 | −4.0e-5 |
| CTM L-BFGS `:cut` χ = 48, 8 iterations (6 min) | **−0.48988930** (bMPS χ = 32: −0.48988930, agreeing with the FD-of-F estimate to 2e-9) | −6.9e-4 |

Same shape as hex(2,2): the BP stage moves the true energy by 4e-5, the CTM stage by 7e-4, and the
optimiser converges to the D = 3 floor in eight iterations (unit L-BFGS steps from iteration 4, the
last gains 1e-6 per step). The route is fully general in the Hamiltonian and lattice as claimed;
what it now needs on this model is D ≥ 5, i.e. hour-long environment sets at the current engine
cost — exactly the rank-r scaling that motivates the per-sector response solve.

## The `:cycle` hysteresis attempt — stopped, hypothesis falsified — *2026-09-15*

Hard-stopped after two hours. The hypothesis was that the limit cycle in the `:cycle` ±λ
converges after optimiser steps (F stationary at 1e-13, worst vertex marginal alternating between
~1e-6 and ~6e-6 indefinitely) is the kept RANK of a plaquette toggling at the `cycle_gapcut` cliff,
and that hysteresis on that rank would remove it.

**What was built.** `_ctm_cycle_projectors(…; prev_rank)` keeps last sweep's rank whenever the old
boundary is still a defensible cut (what lies below it within 100× of the tininess floor and the
drop within 100× of the cliff ratio); the sweep reads each plaquette's previous rank off its north
projector (`_ctm_kept_rank`, counting the non-zero slices of the zero-padded retained index) and
passes it in. Counters `:cycle_hysteresis_kept` / `:cycle_hysteresis_redecided` in `CTM_SVD_STATS`.
Tests pass; the code stays, as a mild stabiliser that does what its comment says.

**What the diagnostic showed** (`scratchpad/cycle_limit_diag.jl`: the state after two `:cycle`
optimiser iterations on the 5×5 D = 3 χ = 32, +λ converge sweep by sweep, every plaquette's kept
rank printed): the ranks are CONSTANT — 9 at the four boundary plaquettes, 32 = χ at the twelve
interior ones — for all 14 sweeps, while the worst marginal change wanders 1e-7 … 6e-6 and F sits
at 1e-13. Hysteresis fired 12 times in 14 sweeps (so a rank-toggle component existed and is now
gone) and made no difference to the marginals. The flutter is therefore in the BASIS at fixed
rank: the interior plaquettes keep the full χ = 32 out of interfaces wider than that (2·D² = 18 per
bond, two bonds per corner interface), i.e. the over-parametrised regime the engine docs describe,
where the trailing kept directions are near-degenerate and the Schur solve returns a slightly
different subspace every sweep. `_ctm_align` aligns the basis WITHIN a subspace; nothing keeps the
subspace itself continuous. Also visible: 8 of 16 plaquettes were declined at λ = 0 (fell back to the
pairwise cut), so the run was not even a pure `:cycle` environment.

**Conclusion.** Making `:cycle` usable in the optimiser loop needs subspace continuity across
sweeps (align the retained invariant subspace to last sweep's, or fix the interior rank below χ
from the spectrum), which is an engine project with its own falsified-idea history, not a
half-day item. `:cut` remains the workhorse; the per-sector response solve stays gated. The
hexagonal Heisenberg result stands on `:cut`.

## Is G ≈ P?  The frozen-projector response, retested — *2026-09-15*

The MP-BP note (Zaletel et al., draft) writes the environment response as G = −(Z_XX)⁻¹ = P(1 − ΔP)⁻¹,
with P propagation through the tangent-plane projectors and Δ the O(ε) diagonal blocks, and
conjectures the propagation spectrum may be O(ε²), i.e. G ≈ P. In our engine "G ≈ P" is a concrete
recipe: build the ±λ environments through the λ = 0 projectors FROZEN (projector-free block
sweeps, `frozen_generating_cache`), and compare the gradient with the fully re-converged one. The
2026-09-12 "H2" run had called this falsified at 12.7% under `:cycle` — at a lossless χ, where
frozen and free cannot differ, so that number was the block-scale mismatch in the old
`effective_operators` (N_eff from the λ = 0 ring), not physics. Redone with the current operators
(`scratchpad/gp_test.jl`; TFIM D = 3, ring gradient along a random direction at the centre):

| lattice, χ (truncation) | projector | free gradient vs exact FD | frozen vs free | frozen cost / free cost |
|---|---|---|---|---|
| 4×4, χ = 12 (18-wide interfaces) | `:cut` | 2.7% | **220%** | |
| 4×4, χ = 12 | `:cycle` | **0.03%** | 12% | |
| 5×5, χ = 32 (BP state) | `:cut` | (23% at h = 1e-3, earlier) | **244%** | 4 s / 20 s |
| 5×5, χ = 32 | `:cycle` | (4%, earlier) | **0.95%** | 7 s / 50 s |

Energies (first derivatives) agree to 1e-8 in every case — the envelope theorem needs no
response. Reading: under `:cut` the frozen response is wrong by O(1), as the falsification said,
because a truncating cut is not stationary. Under `:cycle` it converges to the free one as the
truncation error shrinks (12% → 1% from χ = 12 to χ = 32), so G → P at a stationary fixed point
and the O(ε²) conjecture is at least consistent with two points. At χ = 32 the 1% frozen/free
difference is below the 4% error of the free `:cycle` gradient itself against the exact one.

**Why this matters more than the number.** The frozen ±λ pair is a linear fixed-point iteration
with no projector derivation: no rank decision, no subspace to wander, hence none of the limit
cycle that stopped `:cycle` in the optimiser loop after two steps (2026-09-13/15). It is also 7×
cheaper than the free pair. `dmrg(…; alg = "ctmrg_lbfgs", projector = :cycle, frozen_pm = true)`
now uses it; the λ = 0 `:cycle` converge (well-behaved: 4–5 sweeps after a step) is the only
nonlinear solve left per iteration. Under `:cut` it is refused.

**In the loop: stalled at iteration 2 — verdict negative.** 5×5 D = 3 χ = 32 from the BP state,
`frozen_pm = true`: iteration 1 (Jacobi, α = 1/8) descends 1.94e-5 → 1.57e-5 per site (the free
gradient's first step reached 8.3e-6); at iterations 2 and 3 the unit L-BFGS step and the 1/8
Jacobi step are both uphill, and only α = 1/16 is accepted, for 2e-8 per site each time. The frozen
sweeps themselves converged (no warning), so the gradient is simply not a descent direction once
the state has moved off the point where G ≈ P was measured — the 1% was at ONE vertex on the BP
state; over all 25 vertices and after a step the projector response evidently matters at O(1) for
the direction even when it is 1% for a single component. Two further facts from the run: (i) the
frozen ±λ pair from the aux-free padded λ = 0 environment left legs open in `_ctm_block` (a
10-leg intermediate, out of memory) — the frozen route is dense-only AND needs the true λ = 0
environment; (ii) with the ±λ pair cheap, the λ = 0 `:cycle` re-converge after a state change
(50–90 s) dominates an evaluation (≈ 107 s), so even a working frozen gradient would not have
made `:cycle` cheaper than `:cut`. Conclusion: G ≈ P is a statement about one gradient component
at a stationary point, not about the descent direction along an optimisation; the frozen response
stays as a diagnostic, and `:cycle` in the loop still needs the subspace-continuity work.

## Spinless fermions on the hexagonal lattice (t–V) — *2026-09-15*

`H = −t Σ_e (c†_i c_j + h.c.) + V Σ_e n_i n_j`, t = V = 1, on `named_hexagonal_lattice_graph(NX, NY)`,
fZ2-graded `"Fermion"` sites (parity only), half filling from a charge-density-wave product state.
`MODEL=tv NX NY T V` in `scratchpad/model.jl`; the simple-update start is `F_hop_nn` with
(θ, ϕ) = (i·dτ·t, −i·dτ·V), complex graded tensors throughout. Two-site terms are the library's
joint fermionic operators `"hopping"` and `"NN"`, so the summed edge term has operator-Schmidt rank
3 and the auxiliary index dimension 4 with sectors (even, even, odd, odd) — the odd slots carry
c† and c. Exact reference for hex(2,2) at half filling (8 particles, Jordan–Wigner Lanczos in the
sorted-vertex ordering, `ed_tv.jl`): **E₀ = −8.479910071666449, −0.52999438 per site**.

**The sign gate passed.** On the D = 2 simple-update state (hex(2,2), χ = 16, lossless) the
generating-function machinery agrees with the exact contraction of the fermionic network: ring
energy to 4e-15, FD-of-F energy to 8.7e-9 (the λ = 1e-7 roundoff). Jordan–Wigner strings never
appear explicitly: the graded category carries them through the auxiliary index, the double-layer
CTM blocks and the BP messages alike. Three small engine gaps had to be closed, all of the form
"a lone graded index cannot be paired": the BP default message now starts the operator layer's
auxiliary leg in its norm slot (`onehot`) instead of a three-leg `delta`; the norm-sector slice
`_value_slice` contracts with the dual arrow; and a block that is its own interface gets its width-1
bond by `onehot`, not `delta`. The aux-free λ = 0 environment is dense-only for now (the padding's
arrow convention per block side is not established for graded auxiliary indices), so fermionic runs
take the true λ = 0 environment.

**A fourth gap, in the optimiser, and the one that mattered.** The energy was right and the
optimiser still found no descent direction: the effective operators were built by flattening the
ring operator into a matrix in the site tensor's dense-array basis and applying it to the flattened
site vector, and the graded product does not commute with that flattening — the fermionic
contraction inserts parity signs that depend on leg order and duals. Measured on one ring: the
flattened t†Nt was −0.125 against +0.161 from the contraction, and the gradient 300× off. For
graded sites `_dense_ring_operator` now builds the matrix by applying the ring to each allowed basis
tensor and pairing with each other basis tensor's bra through the backend's own contraction
(`M_ij = ⟨ring · T(e_j) · G · dag(prime(T(e_i)))⟩`), so every sign is the category's and
`x†Mx` is the contraction by construction (ratio 1.0 to all digits). With it the ring gradient
along a random allowed direction agrees with the finite difference of the exact energy to
**2e-7** (`scratchpad/tv_grad_check.jl`). The dense path is untouched; bosonic Z2 grading, which
has no signs, took the new path in the test suite and reproduces its previous numbers.

**hex(2,2), D = 3, errors per site against E₀:**

| stage | E/site | gap/site | note |
|---|---|---|---|
| simple update | −0.49499374 | 3.50e-2 | |
| BP stage (χ = 1) | −0.49499374 | 3.50e-2 | did not move the state at all: every local update rejected or thrown (not diagnosed; the CTM stage is what matters here) |
| CTM L-BFGS `:cut` χ = 32 | −0.49841451 | **3.158e-2** | 8 iterations (5 min; unit L-BFGS steps from it 2), then no downhill direction: the D = 3 floor. Exact contraction of the final state agrees with the optimiser's FD-of-F to 7e-10 per site |

**hex(2,2), D = 4:** simple update and BP stage both at −0.52631232 per site (gap **3.68e-3**, a
10× drop from D = 3 — the CDW-plus-fluctuations state needs D = 4), BP again inert. CTM stage:
3.68e-3 → **2.05e-3** in two iterations (exact contraction agrees with the FD-of-F to 1e-8 per
site), then no descent direction. The cost forced the compromise that stopped it: a graded D = 4
environment set is ≈ 350 s at χ = 32 and ≈ 155 s at χ = 24 (6× the dense Heisenberg at the same D,
χ), and the `:marginal` criterion plateaus at ~3e-7 for the fermionic network, so the runs had to
take a 1e-8 tolerance and one trial per 10-minute process; with the rings that loose, the L-BFGS
and Jacobi directions were rejected at iteration 3 (the same symptom as `:cycle` at 1e-7). The
gradient itself is exact to 2e-7 where it can be checked; this is a convergence-budget limit, not a
correctness one.

**hex(3,3), D = 3 — the cost measurement for rank-3 fermionic terms.** 30 sites, 14 particles (the
CDW start has 15 occupied sites on one sublattice; the fZ2 network needs an even count, so the last
one is emptied), χ = 32, `:cut`, true λ = 0 environment (aux-free is dense-only). No exact reference
at this size (C(30,14) ≈ 1.5e8 states; the Lanczos in `ed_tv.jl` is not built for it), so the
energies are bMPS χ = 32 against the optimiser's FD-of-F:

| stage | E/site (FD-of-F) | E/site (bMPS χ = 32) | note |
|---|---|---|---|
| simple update | −0.50794465656 | −0.50794465658 | start; the BP stage is inert for fermions (skipped) |
| CTM L-BFGS it 2 | −0.50795093 | | two damped Jacobi steps, α = 1/32 then 1/64 |
| it 4 | −0.51062238 | | first two L-BFGS unit steps: −2.7e-3 per site |
| it 7 | **−0.51089053** | **−0.51089054** | it 8: L-BFGS and Jacobi directions both uphill in 3 trials each |

Two things had to be learned on the way. The Jacobi step from this start is far too long: at
`step0 = 1/8` the two trials rose by 0.062 and 0.0077 (total) with a negative slope of −0.02; the
line search accepted only at 1/32 (`step0 = 0.03125`, 5 trials). Once one pair exists the variable
metric takes over and unit steps drop the energy by 6.7e-2 total in one iteration. The stop at
iteration 8 is the `:cut` floor, not a fermion effect: along the Jacobi direction the FD-of-F energy
RISES linearly in α (2.2e-5, 4.8e-5, 1.1e-4 at α = 1/128, 1/64, 1/32) while the ring gradient
reports a slope of −0.026, i.e. the truncated-χ rings' gradient is no longer consistent with the
free energy at the level of the remaining gain (the same symptom the dense 5×5 shows at its floor).
The new verbose line `trial k rejected: α, slope, E_trial − E` in `ctmrg_lbfgs` is what separates
"step too long" (rise ∝ α², negative slope real) from "gradient inconsistent" (rise ∝ α).

Cost: one graded energy evaluation (±λ pair, warm) is 52–70 s at hex(3,3) D = 3 χ = 32 with 6
threads, an L-BFGS iteration 67–116 s; the whole descent −0.50794 → −0.51089 took 7 iterations and
≈ 20 min of environment time across three 10-minute processes. For comparison the dense hexagonal
Heisenberg on the same lattice at D = 3 converges an environment set in ≈ 22 s at χ = 48 with 8
threads (heis section). The fermionic factor is the graded block
bookkeeping in the contractions (many small blocks), not the auxiliary dimension (4 in both).
D = 4 at this size is ≈ 350 s per evaluation (hex(2,2) measurement, and it grows with the width),
so under the 10-minute cap it needs the single-evaluation-per-process mode (`MAXITER=1 LSMAX=1`).

## hex(2,2) Heisenberg at D = 5 — *2026-09-15/16 overnight*

Same pipeline, one iteration per 10-minute process. Per-site errors against E₀ = −0.48851761:

| D | SU start | BP stage | CTM L-BFGS `:cut` | 
|---|---|---|---|
| 3 | 1.43e-2 | 1.43e-2 | 1.27e-2 (floor) |
| 4 | 6.29e-3 | 6.15e-3 | 1.14e-3 (10 it, still descending) |
| 5 | 3.47e-3 | 3.44e-3 | not run: the +λ environment alone did not converge inside 10 min at χ = 32 nor at χ = 24 (aux-widened interfaces 4·D² = 100 wide; the λ = 0 aux-free cache took 37 s) |

The simple-update gaps halve per unit of D (1.4e-2 → 6.3e-3 → 3.4e-3); the CTM stage's gain over
them grew from 11% at D = 3 to 5× at D = 4. D = 5 joins D = 5 TFIM as out of reach under the
10-minute cap: the ±λ pair on 100-wide interfaces is the cost, exactly the rank-r term the
response-solve idea was meant to remove. With hour-long processes it is a few environment sets per
iteration and would run; nothing else is missing.

## Full update with CTM environments — the baseline, timed like for like — *2026-09-16*

The comparison the optimiser has to win: imaginary-time full update whose bond truncation is done
against the same finite-CTMRG `:cut` norm environment (χ = 32), on the same problem (5×5 TFIM
g = 3, D = 3), from the same start (the BP-optimised state `bpstate_L5_D3.jls`, gap 1.94e-5 per
site), with the same thread count (4), the two runs concurrent on the same machine so contention is
shared. Driver `examples/full_update_ctm_tfim.jl` (expects the scratchpad `model.jl` builder next to it); the library gained `region_ring(cache, vs)` (the 4C+4T ring
of a vertex set's bounding box with the other box factors inserted — for two neighbours exactly the
`envs` of `full_update`, and it pairs with `norm_factors(ψ, vs; op_strings)` for a region
observable).

**Full update, as implemented.** Second-order Trotter `e^{−dτ H}`: half-step one-site
`exp(dτ g X/2)` layers (exact, no truncation) around the four edge-colour layers of `exp(dτ ZZ)`,
each two-site gate applied by `full_update` (10 ALS sweeps, bond kept at D) in the ring of the two
sites — the ring pre-contracted to ONE tensor with the 12 outgoing ket/bra legs (D¹² = 5e5 entries at
D = 3), because `full_update` and the region contractions search an optimal contraction sequence
over their whole tensor list and 14 tensors is too many. The environments are re-converged warm
(`:marginal`, 1e-10) after every colour layer — six refreshes per Trotter step, not forty: the
gates within a layer do not touch each other's rings. The bond index keeps its identity across the
re-factorisation so the environments seed the next refresh. Energy along the way is the CTM ring
energy `Σ_v −g⟨X⟩ + Σ_e −⟨ZZ⟩` at the cache's rings; it agrees with boundary MPS to 1e-10 per site
at every checkpoint measured.

**Cost per Trotter step (5×5, D = 3, χ = 32, 4 threads, contended):** ≈ 33 s wall, of which the six
warm environment refreshes are ≈ 5 s and the forty full-update ALS solves ≈ 27 s. A cold
environment set is 31 s, a warm one after a process restart 25–28 s (the contraction-sequence memo is
per process). A 10-minute process fits 6–12 steps.

**Trajectories (per-site gap vs exact −3.147427074503).** Full update at dτ = 0.02 for 30 steps,
then dτ = 0.01 for 30 steps from that state, then dτ = 0.005 for 18 steps (8 threads for that last chunk, alone). L-BFGS:
`dmrg(…; alg = "ctmrg_lbfgs", maxdim = 32, step0 = 0.125, memory = 8)`, 10 iterations, one process.

| wall time (s, incl. cold start) | full update | L-BFGS |
|---|---|---|
| 120 | — (cold set 31 s, 2 steps) | cold set + start ≈ 120 s: 1.94e-5 |
| 200 | dτ = 0.02, τ = 0.10: 7.7e-6 | it 4 (84 s of iterations): **2.71e-6** |
| 240 | τ = 0.12: 6.9e-6 | it 6: 1.90e-6 |
| 430 | τ = 0.24: 4.4e-6 | it 10: **1.75e-6** (process end) |
| 1000 | dτ = 0.02 converged, τ = 0.6: 3.51e-6 (bMPS 3.505e-6) | |
| 1400 | dτ = 0.01, τ = 0.12 more: 2.70e-6 | |
| 1600 | dτ = 0.01, τ = 0.18: 2.67e-6, flattening | |
| 2200 | dτ = 0.005, τ = 0.09 more: 2.73e-6, moving 3e-9 per step (bMPS 2.729e-6) | |

**Reading.** To reach the 2.7e-6 gap the generating-function L-BFGS took ≈ 200 s of wall time
including its cold environment set; full update took ≈ 1400 s — **7× longer** — and it gets there
only after the Trotter step has been halved once. L-BFGS' first ten iterations (7 minutes) end at
1.75e-6 and were still descending (the earlier χ = 48 run took the same state to 8.2e-7); the full
update's dτ → 0 limit is its floor at this χ, since it minimises the per-gate fidelity
at fixed environments, not the energy, and every finite dτ adds a Trotter offset ∝ dτ². Per unit of
wall time the two spend it differently: L-BFGS is 90% environment sets (three per energy
evaluation), full update is 85% local ALS solves in a fixed environment — so full update would gain
from fewer ALS sweeps (3 instead of 10 is usual) or a cheaper two-site solve, at most 3×, which does
not close the gap. The environment side is already cheaper for full update (six warm refreshes ≈ 5 s
per step against 25–33 s per L-BFGS evaluation), which is the point: the optimiser's cost IS the
±λ environment pair, and it wins anyway because each of its steps is a global energy descent with a
variable metric, where a Trotter step is a local, fixed-environment, dτ-limited move.

Not measured here: full update from the simple-update start (the L-BFGS pipeline goes through the
BP stage first, 80 s, which full update could also use), and D = 4 (a D = 4 ring tensor has 4¹² =
1.7e7 entries, 134 MB — the pre-contraction needs a two-tensor split there).

## State of play — *end of the 2026-09-15/16 overnight*

Branch `FixesV2DMRG`. Tests 85/85. What the night settled:

1. **Fermions work at scale.** Spinless t–V on hex(2,2) (exact-checked: D3 3.16e-2, D4 2.05e-3 per
   site) and hex(3,3) (30 sites: −0.50794 → −0.51089 per site in 7 iterations, bMPS-confirmed to
   2e-9). The graded gradient is exact where checkable (2e-7 vs FD of the exact energy); the price
   is 5–6× the dense cost per evaluation at equal D, χ, and a start step of 1/32 (not 1/8).
2. **G ≈ P is not usable in the optimiser** (1% accurate once, stalls at iteration 2): negative verdict,
   diagnostic kept.
3. **`:cycle` rank hysteresis falsified**; the limit cycle is basis wander at fixed rank. Subspace
   continuity across sweeps is the remaining engine idea.
4. **Heisenberg D = 5 on hex(2,2)** is out of reach under the 10-minute cap (the +λ environment on
   4·D² = 100-wide interfaces alone exceeds it).
5. **Full update with the same CTM environments is 7× slower to a 2.7e-6 gap on the 5×5 D = 3 and
   floors there**, where L-BFGS was at 1.75e-6 after ten iterations and reaches 8.2e-7 at χ = 48.
   The user's two criteria (a like-for-like full-update baseline; wall time) are both answered in
   the optimiser's favour at D = 3.

Open, in order of value: hour-long processes (or a split evaluation) for D ≥ 4 fermions and D = 5
spins; the D = 4 full-update baseline (needs a split ring tensor); subspace continuity for `:cycle`;
the BP stage's inertness on fermionic and Heisenberg states.

## After the package port: `:cycle` with the warm-started solver in the optimiser — *2026-09-16 (evening)*

The branch took the FixesV2 merge (NamedGraphs 0.14, ITensorBase 0.14, TensorAlgebra 0.20,
GradedArrays 0.16) and two `:cycle` commits (`f8ee668`, `f524221`) that give the cyclic projector a
warm-started Krylov–Schur solve (`cycle_solver = :auto → :warm` on states). DMRG tests 85/85 on the
merged code. Pre-port serialized states do not deserialize afterwards (two renamed types); every dense
state and checkpoint was dumped to plain arrays in an old-package environment and rebuilt (bMPS energy
identical to 1e-12; the procedure is in the scratchpad's `dump_states.jl` / `portable.jl`). The
graded fermionic states were not converted — the simple-update starts rebuild in 80 s, the hex(3,3)
optimised checkpoint is lost (its numbers are above).

**Does subspace continuity unblock `:cycle` in `ctmrg_lbfgs`?** The stall was: after two steps the
±λ converges enter a limit cycle, the 10-sweep cap leaves a non-descent gradient at iteration 3.
Same run as before (5×5 D = 3 χ = 32 from `bpstate_L5_D3.jls`, `step0 = 1/8`, 8 threads, alone):

| iteration | `:cycle` + `:warm` gap | s / it | `:cut` gap (4 threads, contended) | s / it |
|---|---|---|---|---|
| 1 | 8.26e-6 | 32 | 9.72e-6 | 20 |
| 2 | 3.74e-6 | 77 | 5.20e-6 | 17 |
| 4 | 2.75e-6 | 80 | 2.71e-6 | 16 |
| 5 | 2.65e-6 | 523 (L-BFGS direction rejected ×3, Jacobi at 1/16) | 2.16e-6 | 16 |
| 8 | 2.51e-6 | 88 | 1.81e-6 (it 9) | 18–52 |
| 10 | **1.71e-6** | 89 | **1.75e-6** | 69 |

It no longer stalls: ten accepted iterations, all but one unit L-BFGS steps, and at iteration 10 the
energy equals `:cut`'s. But every ±λ converge still ends at the 10-sweep cap with the worst marginal
change at 1e-6…1e-4 while |ΔF| sits at 1e-13 — the flutter is not removed by the warm start (it is
the null-mode basis wander at fixed rank, as diagnosed), and the one rejected direction (iteration 5,
rise ∝ α² with γ = 1.3e-5) is a pair polluted by a gradient from a 1.7e-4-flutter converge. Cost per
iteration 80–90 s against 17–20 s for `:cut`, i.e. the cap × the per-sweep cost, unchanged. So
`:cut` stays the default: `:cycle` is now a usable alternative that reaches the same energy at 4–5×
the time, and its promised advantage (the ε² energy and the more consistent gradient) is not visible
at χ = 32 on this problem. What would change that is a converge that terminates on stationarity of
the observable-relevant part of the environment (the marginals it does reach in 3–4 sweeps on a
settled state) rather than on the flutter — or the fully stationary projector the flutter is a
symptom of.

## One-sided λ: tested and rejected — *2026-09-16 (night)*

The idea: replace the central difference of the ±λ environments by a forward difference against
the λ = 0 environment we already have, saving one of the three sets per evaluation. Measured on
the 5×5 D = 3 χ = 32 BP state (`scratchpad/onesided_test.jl`, `evalsplit.jl`):

* **Energy.** `F` of the aux-free (padded) λ = 0 cache equals the true λ = 0 `F` to 7e-15 — the
  Möbius weights of the regions containing any one block sum to zero, so `F` is invariant under the
  per-block rescaling — so `(F(+λ) − F(0))/λ` is legitimate. But its O(λ) bias is −5.3e-7 per site at
  λ = 1e-7 (F″ ≈ 270), and shrinking λ runs into F's 1e-13 roundoff: the optimum near λ ≈ 3e-8 leaves
  ≈ 1.5e-7 per site of bias plus noise, 10× the central difference's 2e-8. Unusable at the gaps
  we chase (1e-6…1e-7).
* **Gradient.** The forward-difference `H_eff = (R(+λ)·(G₀+λ∂G) − c·R(0)·G₀)/λ` (with `c` matching
  `t†Nt`, whose error is a harmless multiple of `N`) is correct against the TRUE λ = 0 ring (3e-5 to
  8e-5 relative to central) and wrong by 5–10× against the padded aux-free ring: the padded
  environment truncates the plain norm network, the ±λ ones the generating network with its
  half-insertion slots inside χ, so the two rings differ at O(ε) and the division by λ amplifies that.
  Consistent with why the aux-free cache is a better SEED and not a substitute.
* **Cost.** A warm evaluation on this problem at 8 threads: aux-free λ = 0 1.6 s, true λ = 0 6.3 s,
  +λ alone 7.3 s, −λ alone 7.2 s, ±λ concurrent 13.0 s (the sweeps are already threaded, so running
  the pair concurrently buys 10%, not 2×). Current evaluation ≈ 14.6 s; one-sided with the true
  λ = 0 ≈ 13.6 s. **A 7% saving for a 10× worse energy. Rejected.**

The structural cost is therefore the two generating-network converges themselves, not their
number: what would cut it is a cheaper converge (device, or a response solve), not fewer of them.

## Spinful Hubbard on the hexagonal lattice — *2026-09-17*

`H = −t Σ_e Σ_σ (c†_iσ c_jσ + h.c.) + U Σ_v n_v↑ n_v↓ − (U/2) Σ_v (n_v↑ + n_v↓)`, t = 1, U = 4, on the
open `named_hexagonal_lattice_graph(NX, NY)`, fZ2-graded `"Electron"` sites (d = 4). The μ = U/2
term is the particle-hole symmetric form: the network conserves only parity, and on a bipartite
lattice this μ makes half filling the grand-canonical minimum, so a parity-only state cannot lower
its energy by leaving the half-filled sector. Start: Néel product state (Up on one sublattice, Dn on
the other; N = nsites, even parity), imaginary-time simple update with the library's `F_hop(θ)`
(both spins), `F_int(θ)` and `F_phase(θ)` gates in a second-order Trotter layer, dτ 0.2 → 0.02.
`MODEL=hub NX NY T U` in `scratchpad/model.jl`; the two-site term is the library's joint spinful
`"hopping"` (operator-Schmidt rank 4: auxiliary dimension 5, one even norm slot and four odd), the
one-site terms `"NupNdn"` and `"N"`. Exact references by Lanczos in the N↑ = N↓ = N/2 sector
(`ed_hub.jl`, two bit strings, Jordan–Wigner sign per species):

| lattice | sites | E₀ (U = 4) | per site |
|---|---|---|---|
| hex(1,1) | 6 | −15.66870617887296 | −2.61145103 |
| hex(1,2) | 10 | −26.381696842612968 | −2.63816968 |
| hex(1,2), U = 8 | 10 | −43.57553501196761 | −4.35755350 |

**Sign gate (hex(1,1), D = 2, χ = 64 lossless):** FD-of-F energy = exact contraction to 7e-9 (the
λ = 1e-7 roundoff), dense ring operator `x†Nx` = tensor contraction to 1e-16 at the λ = 0 and +λ
rings, ring gradient along a random allowed direction = FD of the exact energy to 5.7e-7. Nothing in
the engine needed changing for d = 4: the graded operator construction from the t–V work carries
the spinful case.

**hex(1,2), 10 sites, per-site gaps against E₀ (energies confirmed by exact contraction to 1e-8):**

| D | SU start | CTM L-BFGS `:cut` χ = 32 | iterations | s per evaluation (4 threads) |
|---|---|---|---|---|
| 3 | 0.1520 | **0.15193** | 6, then no descent direction | 2–3 |
| 3, χ = 64 (lossless) | | 0.15193 (same to 1e-8) | 8 | 3 |
| 4 | 0.0455 | **0.04459** | 11 | 12 |
| 5 | 0.0361 | **0.03331** (FD-of-F; not yet exact-confirmed) | 9, still descending ~1e-5 per iteration | 60–78 |

Three things this settles:

1. **The D-floor is genuine, not an environment artefact.** D = 3 at lossless χ = 64 reproduces the
   χ = 32 energy to 1e-8, and the optimiser ends on a stationary point (line search fails in every
   direction with gains below 1e-9), so 0.152 per site IS a local optimum of the D = 3 manifold.
2. **The landscape has several minima.** Truncating the optimised D = 4 state to D = 3 (BP truncation,
   −2.3968 per site) and optimising from there at lossless χ converges to a DIFFERENT stationary
   point, −2.40982, worse than the simple-update basin's −2.48624. The Néel-derived simple update is
   the better start here; a global claim about the D = 3 optimum cannot be made from either.
3. **d = 4 sites need D well beyond 4.** 0.152 → 0.045 → ≤ 0.033 per site for D = 3 → 4 → 5 is the
   expected slow convergence of a spinful fermionic PEPS (iPEPS Hubbard work uses D ≈ 8–16); the
   optimiser converges in 6–11 iterations at every D, i.e. the cost is entirely the environment
   sets, which grow as the (5·D²)-wide interfaces are truncated to χ.

**hex(2,2), 16 sites, D = 4 (no exact reference: the half-filled sector has 1.7e8 states).** Simple
update −2.62241 per site; three L-BFGS iterations took it to −2.62306 (evaluations 69–86 s at χ = 32, 4
threads, contended) before the 10-minute process cap killed the invocation ahead of its checkpoint — the
cold environment set plus three iterations is the whole budget. Rerun with `MAXITER=4 BUDGET=180` (or alone
at 8 threads) so each process ends on a checkpoint; then D = 5 for the D-trend and boundary MPS at χ = 64
for the true energy.

## Spinful Hubbard smoke test; the fermionic BP stage was broken, not inert — *2026-09-17*

`examples/hubbard_hex_smoke.jl`: spinful Hubbard (t = 1, U = 4, half filling) on one hexagon (6
`"Electron"` sites on a 3×2 box, `fZ2`) through SU → BP DMRG → CTM L-BFGS, every stage against the exact
contraction and a sparse Jordan–Wigner ED. The CTM side was right first time: the rank-4 spinful
hopping gives auxiliary dimension 5, and at χ = 16 (lossless here) the ring energy, F and the FD-of-F
energy match the exact contraction to 2e-15 / 4e-9. The BP stage rejected every vertex update, which
is what "the BP stage is inert for fermions" (above) actually was: an `ArgumentError` per vertex,
swallowed by the acceptance loop's try/catch. Three defects, found on a 3-site spinful path (a tree,
where the one-site update and the response are exact) and fixed in `src/dmrg.jl`:

1. `_norm_roots` built the auxiliary-index slice with the wrong arrow (`onehot(aux => 1)` instead of
   `onehot(dag(aux) => 1)`); every fermionic update threw. (The optimiser's own path never used it.)
2. The matrix-free local solve flattens the site tensor and applies the effective maps through
   `array`/`from_array`; on graded sites the flattened inner product is not the network contraction
   (the same trap `_dense_ring_operator` documents). Measured: the Lanczos step RAISED the Bethe energy
   from −7.1606 to −6.9572. Graded sites now build dense `N_eff`, `H_eff` in the site's allowed basis
   through closed backend contractions (`_dense_bp_operators`, `_optimize_vertex_graded!`). The basis
   construction differs from the one-shot closed contraction by a fermionic sign that is GLOBAL per
   vertex (`x'Nx / closed(x)` exactly ±1, the same for every random x at a vertex) and the closed region
   scalar itself carries the sign of the message gauge, so `N` is fixed positive at the current state
   (both signs multiply N and H alike; the generalised eigenproblem is invariant).
3. **The message-normalisation functional was not a function of the tensor.** BP normalises a message
   by `sum(data(m))` (the stored-block entry sum) and the Hermitian gauge by the norm-sector version
   `_value_sum`. On graded data the fermionic signs inside the stored representation depend on the
   leg order / codomain split the contraction happened to produce: one and the same message (equal to
   1e-17 as a tensor, under `-`, and under every closed contraction) read entry sums of −0.466 and 1.000
   from two factor orders. BP itself only needs some positive scale per message, but the response
   linearisation `dm = (dF̃ − m Σ dF̃)/Σ F̃` needs ONE linear functional shared by `m` and `dF̃`: with the
   layout-dependent one the responses INTO the middle vertex were exact (source term only) while those
   OUT of it were off by 2× and 11× against a finite difference of the shifted fixed point, and on the
   hexagon the Gauss–Seidel response never converged (relative change pinned at 1.13). `_value_sum` now
   reads the entry sum off the dense array in a fixed leg order (dense tensors unchanged).

After the three fixes, on the 3-site path: response exact after one sweep (history `[1.0, 0.0]`), BP
DMRG monotone and at the ED energy −7.2360679775 to 1e-15 after three updates, every update accepted.
On the hexagon: response converges in four sweeps (change 8.5e-12), twelve of twelve updates accepted,
exact contraction −13.5655 → −13.6199 in two sweeps (ED −15.6687; D = 2). `dmrg(bp)` now prints the
error message of a rejected update under `verbose`.

Open, found on the way: `update(bpc; tolerance = …)` on a state whose site tensors carry dangling
`Charge` legs (charged product states) throws a `NameMismatch` in the message-difference alignment
(the default message has the Charge legs, the updated one has not) — use `maxiter` alone until fixed.

**GC crash.** The smoke test's REPL run segfaulted inside the garbage collector's mark phase (heap
corruption reported earlier by an allocation in TensorKit's sector-structure cache). Not a bug on our
side that we could find: the optimiser's gradient sub-steps at χ = 64 each pass under `--check-bounds=yes`
with a full GC after every step, and the script itself passes at default flags. It is a Julia
1.12.1 runtime problem: the same reproducer (`scratchpad/segv_repro.jl`, single thread,
`--check-bounds=yes`) crashed 2 of 4 fresh 1.12.1 processes ("GC error (probable corruption)" /
SIGSEGV) and 0 of 3 on Julia 1.12.7 (installed alongside via `juliaup add 1.12.7`). Recommendation: run
on 1.12.7 (same Manifest; one-off recompile of the stack).
