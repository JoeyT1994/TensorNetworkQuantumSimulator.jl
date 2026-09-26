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
* `correlation_length(ic; axis)`: ξ from the one-row channel transfer matrix (restarted Arnoldi).
  Above T_c the axis correlation of the 2D Ising decays as r^{-1/2} e^{-r/ξ} — a continuum edge that
  the finite channel discretises: at τ = 0.5 the χ-converged channel ξ is 1.2 % below the exact
  1/(2(K*−K)). Fine as a length scale, not as a precision observable.
* `boundary_peps_stationary`: the bilinear boundary PEPS for complex symmetric layer operators.

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
| same, m = d ln λ₁/dH (diverging as v^{-1/2}) | 4e-11 relative |
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

RESULTS PENDING.
