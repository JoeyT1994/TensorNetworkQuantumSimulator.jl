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

### Assessment

What works: the stationary bilinear boundary PEPS follows the analytic continuation of the dominant
eigenvector into the imaginary field in 3D, stays on the physical branch (Re m = 0 to 1e-7), gives
the zero-field susceptibility to 1e-4 of the series away from β_c, and ends in a fold whose location
is stable between D = 2 and D = 3 (6e-4 at t = 0.19).

What does not yet: the fold is mean-field-like — the correlation length there stays ≈ 3 at every t,
so the true Yang–Lee regime (ξ → ∞, σ ≈ 0.08) is never entered: σ_eff is a crossover value, and the
fold sits an unknown, t-dependent distance below the true edge (a rough estimate with ν_YL ≈ 0.36 and
a bare length ~0.5: ~1 % at t = 0.19, larger as t → 0 — consistent with ζ_eff rising). Together with
the t^{ων} corrections (20 % at t = 0.05) this makes the extrapolated ζ_c = 1.8 ± 0.1 an
uncontrolled estimate, not a measurement; it does not confirm or refute the FRG 1.621(4).

Next steps, in order of leverage: (1) an improved model (Blume–Capel at its improved coupling),
removing the t^{ων} corrections that dominate the extrapolation; (2) larger D at small t with a
cheaper solver — Newton–Krylov with directional derivatives instead of a 2n-evaluation Jacobian
(now `boundary_peps_krylov`, whose complex path is a Levenberg–Marquardt trust region in the Krylov
subspace; validated along D = 2 and D = 3 continuations, above), and a parametrisation
without the redundant directions — so that ξ at the fold grows with D and
finite-correlation-length scaling (θ_c − θ_f ∝ ξ_f^{−(3−Δ_φ)}) can locate the true edge; (3) σ from
that scaling rather than from local exponents.
