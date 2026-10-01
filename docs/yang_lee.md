# The Yang–Lee edge singularity by tensor networks in the thermodynamic limit

Goal: the universal location of the Yang–Lee (YL) edge of the 3D Ising universality class, and its
exponent σ, from the lattice in the thermodynamic limit. In an imaginary field the Boltzmann weights
are complex: Monte Carlo cannot sample them, the conformal bootstrap has no positivity for the
non-unitary YL CFT, and the fuzzy sphere gives CFT data but no lattice quantities. The best current
3D number is functional-RG only: |ζ_c| = 1.621(4), |z_c| = 2.43(4) (Johnson, Rennecke & Skokov,
PRD 107, 116013 (2023), NLO derivative expansion).

Conventions. Ising weight `exp(β Σ σσ′ + H Σ σ)`, H = βh; the imaginary field is H = iθ. Below the edge
(θ < θ_c, high-temperature side) Z is real, ln κ is real and m = ⟨σ⟩ is purely imaginary; past the edge
the zeros condense and the fixed point acquires Re m ≠ 0 (our branch test).

## Tools

* `ising2d_site` / `ising3d_site` take a complex `h`.
* `InfiniteCTM2D` on complex networks; `c4v = true` needed a fix for complex data (below).
* `correlation_length(ic; axis)`: ξ from the one-row channel transfer matrix (KrylovKit Arnoldi).
  Above T_c the axis correlation of the 2D Ising decays as r^{-1/2} e^{-r/ξ} — a continuum edge that
  the finite channel discretises: at τ = 0.5 the χ-converged channel ξ is 1.2 % below the exact
  1/(2(K*−K)). Fine as a length scale, not as a precision observable.
* `boundary_peps_stationary`: the bilinear boundary PEPS for complex symmetric layer operators, by
  Newton with the full finite-difference Jacobian (reused along a continuation). `boundary_peps_krylov`
  solves the same equations without forming J (docs/boundary_peps.md); its complex path matches the
  stationary solver along D = 2 and D = 3 continuations towards the fold (below) and takes larger steps.

### The c4v fix (complex data)

Under `c4v = true` the step derives one pair and relabels it onto the others. The reflection along
an interface's own axis maps the interface to itself with its sides exchanged, so the high side's
blocks (images of low-side ones) consume P_A where the full step consumes P_B: the derived pair must
have P_A = P_B. Real data from a positive problem happen to satisfy this; complex symmetric data do
not (the SVD's column phases, and the unitary gauge alignment). Measured: the 2D Ising at an
imaginary field, c4v off the full step from the first truncating iteration and converging nowhere.
Fix (`_i2_symmetric_pair`): the symmetric (Takagi) gauge P = M_A (M_Aᵀ M_A)^{-1/2}, Π = P Pᵀ, always
applied (a fallback to the raw pair where the early, ill-conditioned whitening leaves it only 1e-5
biorthogonal broke the state for good). Now c4v = full step to all digits over 40 iterations.

### The stationary boundary PEPS

For a complex symmetric T (Tᵀ = T: the Ising site in an imaginary field, z-reflection symmetric) the
left eigenvector is Rᵀ, and the estimator is bilinear, `f(R) = ln κ(RᵀTR) − ln κ(RᵀR)`: holomorphic,
stationary at the dominant eigenvector with first-order errors cancelling, but not a maximum. Solved
as ∇f = 0 by Newton in the C4v-symmetric coordinates (n = 12, 42, 110 at D = 2, 3, 4), the Jacobian
by central differences of the one-site-environment gradient (warm-started environments, columns in
parallel), Broyden-updated between recomputations, continued in θ from the real maximiser at θ = 0.
Null directions projected out on both sides: the scale (J c = −g, cᵀg = 0) and the complex-orthogonal
bond gauge (X⊗X⊗X⊗X, D(D−1)/2 generators).

Exact tests (D = 1, test_boundarypeps3d.jl):

| test | result |
|---|---|
| chains along z (K = 0.3), f = ln λ₁ | 2.8e-15, down to v = 1e-3 from the edge |
| same, m = d ln λ₁/dH (diverging as v^{-1/2}) | 1.1e-9 relative (the damped solver) |
| same, edge sin θ_c = e^{−2K} from m⁻² → 0 | 1.8e-6 |
| planes (J_z = 0), f = 2D Ising ln κ at the same field | 3e-15 |

Two cautions from these. (1) At D = 1 the chain's edge is an exceptional point where the stationary
vector is self-orthogonal (rᵀr → 0): the reduced Jacobian then GROWS (as v^{-1/2}), so its smallest
singular value is not a fold indicator there; m is. (2) Decoupled chains at D ≥ 2 are degenerate
(any A = r ⊗ M is stationary, the virtual network cancels between the two terms): not a test.

## 2D calibration

Square lattice, τ = (1/s − s)/2 with s = sinh 2β. The universal location (Fonseca & Zamolodchikov;
Mangazeev, Hagan & Bazhanov, PRE 108, 064136 (2023)) is ξ₀ = 0.18935060551 in ξ = h/|m|^{15/8},
h = C_H H, m = −C_τ τ to leading order; for the square lattice C_τ = √2 (Onsager's specific heat) and
C_H = 2^{5/48} e^{1/8} A^{-3/2} = 0.83868 (Yang's magnetisation with their G̃₁; the same derivation
gives their triangular C_H). So ξ_eff(τ) = C_H θ_c(τ)/(√2 τ)^{15/8} → ξ₀ as τ → 0.

Per τ: InfiniteCTM2D (c4v) warm-started in θ towards the edge until the fixed point leaves the
physical branch; m(θ) fitted to the M_{2/5} structure
`m = a₀ + a₁v + A v^σ + B v^{σ+5/6} + C v^{σ+1}`, v = θ_c − θ (Mangazeev et al.'s eq. (93)
differentiated), by variable projection over (θ_c, σ). Points χ-converged (δm between χ = 48 and 64
at 1e-8), v ≥ 7e-4 (closer, the CTMRG's own pseudo-edge and critical slowing down — 2000–6000
iterations — take over).

| τ | χ | θ_c (σ = −1/6) | ξ_eff | σ free |
|---|---|---|---|---|
| 0.05 | 64 | 1.550803941(2)e-3 | 0.18679600 | −0.1684 |
| 0.07 | 64 | 2.896861972(5)e-3 | 0.18567285 | −0.1685 |
| 0.10 | 48 | 5.59974871(2)e-3 | 0.18388571 | −0.1692 |
| 0.15 | 48 | 1.17661827(6)e-2 | 0.18065247 | −0.1708 |
| 0.20 | 48 | 1.97857634(5)e-2 | 0.17713362 | −0.1700 |
| 0.30 | 48 | 4.04698202(7)e-2 | 0.16939803 | −0.1685 |

(± from the spread over fit forms.) σ = −0.1684 … −0.1708 against −1/6: the free-σ fits are biased
1–2 % negative by the dense correction spectrum (v^{5/6}, v^{1} …).

τ → 0, polynomial fits of ξ_eff:

| degree, range | ξ₀ | rel. error |
|---|---|---|
| 1, τ ≤ 0.1 | 0.189728 | +2.0e-3 |
| 2, τ ≤ 0.15 | 0.189383 | +1.7e-4 |
| 3, τ ≤ 0.3 | 0.1893435 | −3.8e-5 |
| 3, τ ≤ 0.2 | 0.1893464 | −2.2e-5 |
| 4, τ ≤ 0.3 | 0.1893484 | −1.2e-5 |

So ξ₀ = 0.18935(1), Z₀ = ξ₀^{-8/15} = 2.42918, against 0.18935060551 and 2.4291691718.

Lessons for 3D: (i) the lattice corrections to the edge location are 1–10 % over τ ∈ [0.05, 0.3]
and need a cubic to reach 1e-4 — in 3D they are non-analytic (t^{ων}, ων ≈ 0.52), so an improved
model (Blume–Capel at its improved coupling) is the way to precision; (ii) with σ fixed the edge
is determined to ~1e-8 from data at v ≥ 1e-3; a free σ is good to ~1–2 %.

## 3D

### The solver in 3D (what it took)

`boundary_peps_stationary` on the simple-cubic site at β < β_c, continued in θ from the real
maximiser (script: scratch `yl3d_scan.jl`, checkpointed so a scan is a chain of ≤10-minute runs):

* The bilinear gradient is right: d(Re f) by finite differences against Re(gᵀdc) to 1e-9
  (β = 0.18, D = 2, θ = 0.005); J is a symmetric Hessian (asymmetry 5e-8) with Jc = −g, Jᵀc = −g
  to 1e-9, and linear to first order in the step.
* Undamped Newton from a distant start diverged (residual 0.11 → 0.13 → 5.3); now backtracking on
  the residual |g||c|, Broyden updates, one Jacobian refresh when a stale one stalls.
* The residual has a NOISE FLOOR set by χ and the CTM tolerance (the truncation makes g slightly
  non-holomorphic): 2e-9 at β = 0.18 and 1e-8 at β = 0.21 (D = 2, χ = 16), ~1e-7 close to a fold.
  f and m are reproducible to 1e-12 and 1e-10 across repeated solves at that floor.
* The reduced Jacobian is ILL-CONDITIONED — the flat directions of the boundary-PEPS landscape: at
  D = 2 its singular values run from 5e-6 to 2.6; at D = 3 (β = 0.18, where the state barely uses
  its bond dimension) ~15 of 38 are below 1e-8, down to 1e-17, and the plain Newton step had norm
  1.9e3. A truncated-SVD step (drop σ < 1e-6 σ_max) takes the residual 7.5e-3 → 4e-5 in one step.
  Its smallest singular value is therefore not a fold indicator; the magnetisation is: at the
  finite-D fold m − m_f ∝ √(θ_f − θ), so (dm/dθ)⁻² extrapolates linearly to θ_f.

### Normalisation

ζ = (B_c/C₊)^{1/γ} t/|H|^{1/Δ} (Johnson, Rennecke & Skokov's variable, independent of the
low-temperature amplitude), with the edge ζ_c its t → 0 limit, and |z_c| = |ζ_c| R_χ^{1/γ}
(R_χ^{1/γ} = 1.497(22)). Replacing C₊t^{−γ} by the measured χ(t) at the same β and D removes t:
ζ_eff = (B_c/χ(t))^{1/γ} / |H_c(t)|^{1/Δ}. χ(t) is the small-θ slope of Im m (θ² removed): 8.4798
at β = 0.18 and 19.283 at β = 0.20 (D = 2), against 8.479 and 19.26 from the 17-term
high-temperature series with a ratio-method tail — the zero-field state is accurate at D = 2.

B_c two ways: (i) the critical isotherm, D = 2, H = 0.05 … 0.002, M/H^{1/δ} = 1.2747 … 1.3710,
fitted with a H^{ω/Δ} correction: 1.393; (ii) R_χ = C₊B^{δ−1}/B_c^δ with R_χ = 1.660, B = 1.6919
(Talapov–Blöte) and our C₊ = 1.119 (χ t^γ at the two β, one t^{ων} correction): 1.396.

(C₊ from the 17-term high-temperature series, χ t^γ fitted as C₊(1 + a t^{ων}): linear to 1e-4 over
t = 0.05–0.28, C₊ = 1.114(2). Our D = 2 χ matches the series to 2e-5 at β = 0.16, 1e-3 at 0.20 and
7e-3 at 0.21, where the series tail is itself uncertain at that level.)

Two more solver lessons at D = 3: (i) the branch bends along the soft directions, so continuation
steps must stay small (a doubled step from θ = 0.021 to 0.035 at β = 0.18 stalled at residual 3e-3
with m off the branch; steps ≤ 0.004 growing ×1.3 converge); (ii) cost: a finite-difference Jacobian
(84 two-environment evaluations) is 140 s at β = 0.18 and 450 s at β = 0.20 (χ = 24), a point
50–400 s. D = 3 at β = 0.20 could not be continued within 10-minute runs.

### Newton–Krylov in the imaginary field (2026-09-26)

`boundary_peps_krylov` solves the same stationarity equations without forming J: complex data take
its Levenberg–Marquardt trust region on |g|². Validated against the scans above at β = 0.18.

**A bug found on the way.** Anderson mixing of the 2D CTMRG (`ctm_anderson`) stalls on these
complex networks. A warm product evaluation (D = 2, χ = 16, θ = 0.005) ran the full 2000 steps
without converging (|Δ| ~ 3e-7), against 20 steps unmixed. It now applies to real networks only.

**D = 2, χ = 16.** A continuation over the old scan's 32 converged points, θ = 0.0081 to 0.04905,
with v = (θ_f − θ)/θ_f from 0.83 down to 1.1e-3. Agreement in Im m:

| v | \|Δ Im m\| against the stationary scan |
|---|---|
| > 0.03 | ≤ 5e-8 (one point 4.9e-7) |
| 3e-3 – 0.03 | 2–6e-7 |
| 2.6e-3 → 1.1e-3 (last three points) | 2e-6 → 1.7e-5 (dm/dθ ∝ v^{-1/2} amplifies the noise floor) |

Cost: 3.5–19 s per point, 42–58 s for the last three, 8 minutes in all. Both solvers converge to
|g| ~ 1e-8. On larger steps (Δθ = 0.005–0.01 from θ = 0) Newton–Krylov converged in 21–30
evaluations (9–17 s) where the stationary solver stalled at |g| = 1–4e-5 after 27–209 s. On the
old scan's small steps both reproduce the old values to 2e-8.

**D = 3, χ = 24.** 11 of the old scan's points, θ = 0.006 to 0.04705 (v ≈ 0.04), with tol = 1e-7
and noise_tol = 1e-6. Every point converged, to |g| = 2e-8 – 2.5e-7, where the old scan stopped at
1e-7 – 1.3e-6. Agreement in Im m:

| θ | \|Δ Im m\| |
|---|---|
| ≤ 0.032 | ≤ 5e-7 |
| 0.0399 | 7e-7 |
| 0.0432, 0.0450 | 1.3e-6 |
| 0.0462 | 3.7e-6 |
| 0.04705 | 7.9e-6 |

The difference grows towards the fold as the old scan's residual does, so it is most likely that
scan's error. Cost: 83–345 s per point, 40 minutes in all, with no 140 s Jacobian up front. The
stationary solver took 110–125 s per point near the fold, but at a residual 10–20× larger.

### Results (2026-09-26)

The edge map — the finite-D fold θ_f(t) of the stationary boundary PEPS, D = 2, χ = 16, fitted as
m = m_f − a√(θ_f − θ) + b(θ_f − θ) to the last six points (all within v ≈ 1e-3 of it):

| β | t | θ_f (D = 2) | χ | ζ_eff | ξ at the fold |
|---|---|---|---|---|---|
| 0.16 | 0.278 | 0.0946047 | 5.1735 | 1.5663 | 2.62 |
| 0.18 | 0.188 | 0.0491047 | 8.4798 | 1.5979 | 2.76 |
| 0.19 | 0.143 | 0.0312226 | 11.973 | 1.6151 | 2.87 |
| 0.20 | 0.098 | 0.0167357 | 19.283 | 1.6373 | 2.92 |
| 0.21 | 0.053 | 0.0059898 | 42.046 | 1.6821 (series χ: 1.6911) | 2.97 |

D-dependence (β = 0.18, D = 3, χ = 24, 13 points to v ≈ 0.03): D = 3 reproduces the D = 2 curve
shifted by Δθ ≈ +2–3e-5 near the fold (m = 0.63212 at θ = 0.04705, which D = 2 reaches at 0.047030),
so θ_f(D = 3) ≈ 0.0491, 6e-4 above D = 2, with the same ξ ≈ 1.8 at equal θ. At θ = 0 the D = 3
optimum improves f by only 2e-10 (β = 0.18) and 2e-6 (β = 0.20): at these temperatures the extra bond
dimension is barely used, and the reduced Jacobian's ~15 near-null directions are that unused space.

t → 0 (D = 2 folds): ζ_eff rises from 1.566 to 1.682 as t goes from 0.28 to 0.05, and the limit
depends on the correction model — a t: 1.69–1.72; a t^{ων}: 1.76–1.78; a t^{ων} + b t: 1.81–1.91
(rms 1e-3–3e-3; |z_c| = 2.54–2.86). FRG: |ζ_c| = 1.621(4), |z_c| = 2.43(4).

σ: the local exponent σ_loc = 1 + d ln(dm/dθ)/d ln(θ_f − θ) has a minimum of 0.23 (t = 0.28), 0.28
(0.19), 0.29 (0.14), 0.35 (0.10), 0.39 (0.05) before rising towards the fold's 1/2 — against
σ ≈ 0.074–0.085 (ε expansion, bootstrap, fuzzy sphere). NOT resolved.

### D = 3 edge maps (2026-09-27)

Newton–Krylov continuations at D = 3, χ = 24, from the real maximiser at θ = 0, in resumable
ten-minute runs; points accepted at |g| ≤ 1e-6, or near the fold, where |g| floors at 3–4e-6, at
5e-6. Stopped at v ≈ 0.013–0.016: a near-fold point took 2–3 runs, and the running fold estimate
(from (dm/dθ)⁻² → 0, biased low far from the fold) kept receding. The same six-point fit; its bias
from stopping at v ≈ 0.01, measured by refitting the D = 2 maps without their closer points: θ_f
0.15–0.35 % low.

| β | t | θ_f D = 2 | θ_f D = 3 | shift | ζ_eff D = 2 → 3 | ξ (last point with v ≥ 0.01) D = 2 → 3 |
|---|---|---|---|---|---|---|
| 0.18 | 0.188 | 0.0491047 | ≈ 0.04916 | +0.12 % (like for like) | 1.5979 → ≈ 1.597 | — |
| 0.20 | 0.098 | 0.0167357 | 0.0170106 | +1.6 % | 1.6373 → 1.6214 | 2.48 → 3.73 |
| 0.21 | 0.053 | 0.0059898 | 0.0063160 | +5.5 % | 1.6821 → 1.6336 | 2.73 → 4.70 |

At β = 0.18 the D = 3 points (the validation continuation above) reach only v ≈ 0.04, where the fit
reads 1.5 % low; D = 2 refitted over the same θ range gives 0.048308 against D = 3's 0.048365, so
the shift is taken like for like. At β = 0.20 and 0.21 it needs no fit: the D = 3 branch continues
past the D = 2 fold, to θ = 0.016787 > 0.016736 and 0.006213 > 0.005990. The susceptibility is the
same at D = 3 (to 0.1–0.6 %), so ζ_eff moves through θ_f alone.

* The fold moves up with D, the more the closer to β_c: the finite-D fold lies below the true edge,
  and D = 2 is not converged at small t.
* ζ_eff(t) flattens: at t = 0.053 it falls from 1.682 to 1.634. The rise of the D = 2 curve towards
  t → 0, which drove the extrapolations to 1.7–1.9 above, is largely a finite-D effect.
* ξ near the fold grows 50–70 % from D = 2 to 3 (to 4.7 at t = 0.053). Re m, zero on the physical
  branch, drifts to 1e-4 there: the residual floor's cost near a fold.

An illustration, not a measurement (two D, ξ at the last point with v ≥ 0.01 standing in for ξ_f, the
exponent 3 − Δ_φ = 2.785 assumed): θ_f extrapolated linearly in ξ^{−2.785} gives ζ ≈ 1.614 at
t = 0.098 and 1.618 at t = 0.053 (θ_c 0.8 % and 1.5 % above the D = 3 folds) — flat in t and
within 0.5 % of the FRG 1.621. It needs D = 4 and 5 and ξ at the fold itself to become one.

The runs also fixed `boundary_peps_krylov`'s `time_limit`: near the fold a Newton step is ~24
finite-difference products (~5 minutes) and the first trust-region trial overshoots ~100× (a nearly
singular Jacobian), and a time limit that also cut the trials discarded each run's subspace — two
continuations resumed the same point for 45 minutes at a fixed |g|. The trials in a built subspace
now always run; resumed points keep their trust radius.

### D = 4 edge map at β = 0.21, and the ξ extrapolation (2026-09-29)

`scan_krylov.jl` with `GPU=1`, D = 4, χ = 48, on the local A6000. The run started from a fully converged real
state (`T0ITER = 1000`, `T0LIMIT = 7200`: |g| = 9.2e-7 at θ = 0 after 45 min). With the old 200 s θ = 0 stage it
started at |g| = 3.7e-4, and the first Newton–Krylov point had not converged after 65 min. Later points took
5–13 min far from the fold and 1–2 h near it, in 30-minute resumable chunks. 13 points, θ = 0.0006 to 0.006236;
data `examples/yang_lee/data/ylk3_beta0.21_D4_chi48.csv`.

| θ | Im m | Re m | ξ | \|g\| |
|---|---|---|---|---|
| 0.005391 | 0.29106 | +5e-7 | 3.287 | 6.7e-7 |
| 0.005712 | 0.32869 | −4e-6 | 3.590 | 2.0e-6 |
| 0.005912 | 0.35962 | −5e-6 | 3.916 | 1.1e-6 |
| 0.006040 | 0.38529 | −1.4e-5 | 4.255 | 1.5e-6 |
| 0.006129 | 0.40808 | −2.2e-5 | 4.605 | 3.0e-6 |
| 0.006192 | 0.42848 | −1.0e-5 | 4.935 | 4.8e-6 |
| 0.006236 | 0.44705 | +3.6e-5 | 5.203 | 3.7e-6 |

The six-point fold fit (`examples/yang_lee/fold_xi_scaling.jl`, the same fit as `analyse_maps.jl`), the last point
at v = 0.012:

| β = 0.21 (t = 0.0526) | θ_f | ξ_f (last point with v ≥ 1e-2) | ζ_eff |
|---|---|---|---|
| D = 2 | 0.0059898 | 2.73 | 1.6821 |
| D = 3 | 0.0063160 | 4.70 | 1.6336 |
| D = 4 | 0.0063139 | 5.20 | 1.6340 |

**The fold did not move from D = 3 to D = 4** (−0.03 %), against +5.5 % from D = 2 to D = 3, while ξ at the fold
grew by 11 %. So θ_c − θ_f ∝ ξ_f^−2.785 does not hold across D = 2, 3, 4:

* the fit through all three gives θ_c = 0.006392 and ζ(θ_c) = 1.6212;
* the fit through D = 3 and 4 puts θ_c at 0.006307, BELOW the D = 4 fold, which is unphysical.

The D = 2, 3 illustration above (ζ_c ≈ 1.61–1.62, |z_c| ≈ 2.44) rested on that scaling, and is withdrawn until
this is understood. With β = 0.20 at D = 2, 3 only, the t → 0 limit reads ζ_c = 1.641 (all D) or 1.691 (two largest
D), |z_c| = 2.46 or 2.53 against the FRG 2.43(4). That spread is the current uncertainty, not an error bar.

Three explanations, in the order to test them:

1. **χ = 48 is not enough at D = 4 near the fold.** χ/D² = 3, against 2.7 at D = 3, and ξ is 5 there. Test:
   re-converge the last three points at χ = 64 from the χ = 48 states.
2. **The fold has converged in D at this β**, and the D = 2 → 3 shift was the whole finite-D correction. Then θ_c ≈
   0.00632 at t = 0.053, and ζ(t) is 1.634 there.
3. **D = 3's recorded fold is biased.** It was fitted from points stopping at v ≈ 0.013–0.016, against D = 4's 0.012.
   The six-point fit reads low when it stops far out (measured 0.15–0.35 % low at v ≈ 0.01 on the D = 2 maps), so D = 3's
   true fold could lie a little higher. That would make the D = 3 → 4 shift negative, which would favour 1.

Re m drifts to 1–4e-5 over the last four points — the residual floor near a fold.

**β = 0.20, D = 4 (the same morning).** On 8 CPU threads to v = 0.056 (1–3 h per point), then on the GPU from 06:40
(30 min per point) to v = 0.012 at θ = 0.016777, ξ = 3.96. Data `examples/yang_lee/data/ylk3_beta0.2_D4_chi48.csv`.
The same thing happens:

| | θ_f D = 3 | θ_f D = 4 | shift | ξ_f D = 3 → 4 |
|---|---|---|---|---|
| β = 0.20 (t = 0.098) | 0.0170106 | 0.0169843 | −0.16 % | 3.73 → 3.96 |
| β = 0.21 (t = 0.053) | 0.0063160 | 0.0063139 | −0.03 % | 4.70 → 5.20 |

At both temperatures the fold stays put from D = 3 to D = 4 (or slips slightly) while ξ grows, so ξ_f^−2.785 scaling
does not describe D ≥ 3. The t → 0 limit through the two β (ζ = ζ_c + a t^{ων}):

| treatment of θ_c(β) | ζ(0.098), ζ(0.053) | ζ_c | \|z_c\| |
|---|---|---|---|
| ξ fit through D = 2, 3, 4 | 1.6157, 1.6212 | 1.636 | 2.449 |
| ξ fit through D = 3, 4 | 1.6320, 1.6351 | 1.643 | 2.460 |
| the D = 4 fold taken as converged | 1.6231, 1.6340 | 1.663 | 2.49 |
| FRG (Johnson, Rennecke & Skokov) | | 1.621(4) | 2.43(4) |

All three lie 1–2.5 % above the FRG. That is consistent within the spread between treatments, but not a controlled
number until the χ = 64 check says whether χ = 48 is converged at the fold.

**The χ = 64 check (2026-09-30).** Setup:
- β = 0.21, D = 4, the second-to-last map point, θ = 0.0061915 (v = 1.9 %).
- Newton–Krylov re-converged at χ = 64 from the χ = 48 state, on the A6000: 128 evaluations, 1.5 h, |g| = 3.1e-6.

| | Im m | ξ |
|---|---|---|
| θ = 0.0061915 (v = 1.9 %), χ = 48 | 0.42848349 | 4.935 |
| θ = 0.0061915 (v = 1.9 %), χ = 64 | 0.42844862 | 5.031 |
| θ = 0.0062360 (v = 1.2 %), χ = 48 | 0.44705229 | 5.203 |
| θ = 0.0062360 (v = 1.2 %), χ = 64 | 0.44702831 | 5.323 |

The last point (130 evaluations, 1.6 h, |g| = 3.2e-6, Re m = 7e-5) moves by −2.4e-5, less than the one before it,
at a steeper slope (≳ 420). That is a θ-equivalent shift of ≲ 6e-8.

- **Im m** moves by −3.5e-5. The map's slope there is dIm m/dθ ≈ 365, so that is a θ-equivalent shift of 1e-7 (0.0015 %).
  That is 20× smaller than the D = 3 → 4 fold shift at this β. **The fold is converged in χ at χ = 48**, which
  rules out explanation 1 above.
- **ξ** grows by 2 % with χ, so ξ_f at χ = 48 is a slight underestimate. That cannot rescue the ξ_f^−2.785 law:
  ξ_f grows 11 % from D = 3 to D = 4 while the fold does not move.
- What is left is explanation 2, the fold converged in D, or 3, a biased D = 3 fold. With explanation 2, the D = 4
  fold stands as θ_c, and ζ_c = 1.663, |z_c| = 2.49 (the last row of the table).

### The Bethe estimator against the Hermitian one (2026-10-01)

T in an imaginary field is complex symmetric, not Hermitian. On the same boundary state R, compare two estimators:
- **Bethe** (two-sided, bilinear): f_B = ln κ⟨Rᵀ|T|R⟩ − ln κ⟨Rᵀ|R⟩. It is stationary at T's dominant eigenvector, so its
  error is O(ε²).
- **Hermitian** (conjugated bra): f_H = ln κ⟨R̄|T|R⟩ − ln κ⟨R̄|R⟩. It is not stationary, so its error is O(ε).

The script is `examples/yang_lee/bethe_vs_hermitian.jl`. Run: 3D Ising, β = 0.21, D = 2, χ = 16, CPU, 2D level `:cut`.

**On converged Bethe states:**

| θ | f_B | f_H | f_B − f_H | Im m (Bethe impurity) | Im m (Hermitian impurity) |
|---|---|---|---|---|---|
| 0.002 | 0.767266821 | 0.767247630 | 1.9e-5 | 0.0862 | 0.0123 |
| 0.004 | 0.766996804 | 0.766903400 | 9.3e-5 | 0.1888 | 0.0250 |
| 0.0055 | 0.766631152 | 0.766368749 | 2.6e-4 | 0.3146 | 0.0355 |

- The Hermitian impurity ⟨R̄|M|R⟩/⟨R̄|R⟩ is not the eigenvalue's derivative: it is 7–9× off.
- The Bethe Im m is consistent with d Re f_B/dθ (section below).

**Stationarity**, at θ = 0.0055 (v = 8 % from the D = 2 fold): R = R* + ηX, with X a random C4v-symmetric direction of
norm |R*|:

| η | \|Δf_B\| | \|Δf_H\| | ratio |
|---|---|---|---|
| 1e-1 | 3.69e-5 | 1.10e-3 | 30 |
| 3e-2 | 5.23e-6 | 2.21e-4 | 42 |
| 1e-2 | 8.42e-7 | 6.35e-5 | 75 |
| 3e-3 | 8.37e-8 | 1.80e-5 | 215 |
| 1e-3 | 9.53e-9 | 5.89e-6 | 620 |
| 3e-4 | 8.56e-10 | 1.76e-6 | 2050 |
| 1e-4 | 9.25e-11 | 5.84e-7 | 6300 |

- Local slopes over the last three decades: **2.0 for Bethe, 1.0 for Hermitian**.
- No linear floor from the 2D level (`:cut`, non-stationary for this non-Hermitian network) appears down to
  1e-10. Its first-order term lies below that here.
- The impurity Im m moves linearly for both (only f is stationary).

**Truncation (in progress):** both estimators on D = 3, χ = 24 states at the same θ, against D = 2 and later D = 4 —
the real-world ε² vs ε.

### Bethe consistency, and the fold fit's systematic (2026-10-01)

**The estimators agree.**
- Both levels are Bethe: the 2D ln κ is the Kikuchi form; the 3D f is ln κ⟨Rᵀ|T|R⟩ − ln κ⟨Rᵀ|R⟩, the two-sided
  Rayleigh quotient (T is complex symmetric, so the left vector is Rᵀ; `bilinear = true`).
- The maps read Im m from the one-site impurity, which with `:cut` carries first-order 2D-environment errors.
- Check: f is analytic in the coupling iθ, so d Re f/dθ = −Im m. Over both D = 4 maps, Δ Re f between
  consecutive points matches −∫ Im m dθ (cubic quadrature) to 2e-5 – 7e-4 relative, the size of the
  quadrature error (`scratchpad bethe_consistency.jl`). No estimator bias is visible at that level.

**The fold fit is the weak point.** Fit the fold from Im m (Im m = m_f − A√u + Bu, u = θ_f − θ), or from the
Bethe f alone (Re f = F₀ + m_f u − (2A/3)u^{3/2} + (B/2)u²), over the last N points:

| β | N = 8 → 5, from Im m | N = 8 → 5, from Re f |
|---|---|---|
| 0.21 | 0.0062964 → 0.0063193 (+0.36 %) | 0.0062859 → 0.0063365 (+0.80 %) |
| 0.20 | 0.0169367 → 0.0170011 (+0.38 %) | 0.0169137 → 0.0170002 (+0.51 %) |

- θ_f climbs as the window closes in on the fold, whichever estimator is used.
- The window systematic is 0.4–0.8 %. That is 3–25× the D = 3 → 4 shifts above (−0.03 %, −0.16 %), so
  **"the fold stays put from D = 3 to D = 4" is not established**. The six-point fits of both D compared like
  with like, but the model's inadequacy at v ≈ 1–2 % need not cancel between them.
- **Remedy:** the fold located exactly, where the Jacobian of the stationary (Bethe) equations goes singular.
  Use pseudo-arclength (`fold_pseudoarclength.jl`, D = 2 prototype) or a bordered Newton solve for (c, θ_f)
  with a null vector. No fit is involved.

### Assessment

What works: the stationary bilinear boundary PEPS follows the analytic continuation of the dominant
eigenvector into the imaginary field in 3D, stays on the physical branch (Re m = 0 to 1e-7 away from
the fold), gives the zero-field susceptibility to 1e-4 of the series away from β_c, and ends in a fold
that moves up with D: by 0.1 % at t = 0.19, 1.6 % at t = 0.10 and 5.5 % at t = 0.05.

What does not yet: at D ≤ 3 the fold is still mean-field-like (ξ ≈ 3–5 there), so the true Yang–Lee
regime (ξ → ∞, σ ≈ 0.08) is never entered and σ_eff is a crossover value. The D = 3 maps show that
the fold's distance below the true edge, not only the t^{ων} corrections, drove the D = 2
extrapolation to ζ_c = 1.8 ± 0.1: that value is superseded. With D extrapolated from two points
ζ ≈ 1.61–1.62, consistent with the FRG 1.621(4) but uncontrolled until D ≥ 4.

Next steps, in order of leverage: (1) D = 4 and 5 at t ≈ 0.05–0.10, with ξ at the fold, for
finite-correlation-length scaling θ_c − θ_f ∝ ξ_f^{−(3−Δ_φ)} (a GPU job: a D = 5 CTM step is 40 s on
four CPU cores, docs/boundary_peps.md "Costs"); (2) an improved model (Blume–Capel at its improved coupling), removing the
t^{ων} corrections from the t → 0 limit; (3) σ from that scaling rather than from local exponents.
