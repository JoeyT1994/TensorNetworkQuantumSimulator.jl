# The residual entropy of hexagonal and cubic ice

*Started 2026-09-27 (FixesV2). Everything here is measured; the numbers are dated.*

The question: do ice Ih (hexagonal stacking, ABAB) and ice Ic (cubic, ABC) have the same residual
entropy — Pauling's proton-configuration entropy under the ice rules — or is Onsager's inequality
S_h ≥ S_c strict? Pauling's estimate W = 3/2 per molecule is the Bethe value for both; the true
W ≈ 1.5074.

| source | W(Ih) | W(Ic) |
|---|---|---|
| Kolafa 2014, Monte Carlo + thermodynamic integration | 1.5074674(38) | 1.5074660(36) |
| Xu, Lin & Zhang 2025 (arXiv:2511.22477, PRB), tensor networks, D ≤ 7 | 1.5074584 (S = 0.4104251(6)) | S = 0.4104248(14), "equal" |
| Chen & Ran 2026 (arXiv:2608.07613), rigorous | hexagonal ≥ cubic at every finite cross-section | |

## The transfer operator and an identity

`ice_site()` is one bilayer cell (an O on the lower sublayer, a, and its in-plane neighbour on the
upper one, b); M, the operator between the vertical bonds below and above a bilayer, gives both
stackings (Onsager): Z(Ic) = Tr Mⁿ, Z(Ih) = Tr (M Mᵀ)^{n/2}. So per bilayer

    S_c = ln λ_max(M),    S_h = ln σ_max(M) ≥ S_c.

The cell tensor is invariant under the inversion of its x, y legs combined with z⁻ ↔ z⁺ (the
diamond lattice's inversion through a bond midpoint): **Mᵀ = I M I**, I the in-plane inversion of
the cell lattice. Hence:

* I M is symmetric and (I M)² = Mᵀ M: S_h = ln λ_max(I M), a symmetric (variational) problem.
* the left Perron vector of M is I R: S_c is the stationary point of ln⟨I R|M|R⟩ − ln⟨I R|R⟩,
  second order in the error of R — the bilinear estimator of the Yang–Lee work with an inverted bra.
* S_h = S_c exactly iff the hexagonal boundary state (the top eigenvector of I M) is inversion
  symmetric; its inversion overlap per cell, F_I, is the order parameter.
* the real Rayleigh quotient ⟨R|M|R⟩/⟨R|R⟩, which Xu, Lin & Zhang maximise for Ic (justified by
  normality of M, which their own diagnostic shows is violated at 1e-4), has maximum
  S_sym = ln λ_max((M + Mᵀ)/2), with S_c ≤ S_sym ≤ S_h — not the cubic entropy unless the left and
  right Perron vectors coincide. It still tests the equality: on a finite cross-section S_sym = S_h
  iff S_c = S_h (if the top eigenvector x of (M + Mᵀ)/2 reached xᵀMx = σ_max|x|², Cauchy–Schwarz
  would make x an eigenvector of M with eigenvalue σ_max), so their S_h − S_sym is a valid test,
  with about half the gap (the 3 × 3 torus below). (Corrected 2026-09-27: an earlier version said
  the estimator could not test the equality.)

**Exact check** (brute-force transfer matrices, all proton configurations, L1 × L2 cell tori):
Mᵀ = I M I holds exactly on every torus, I M is exactly symmetric with λ(I M) = σ_max. On the
smallest tori (2 × 2 … 2 × 4) the three entropies coincide exactly (F_I = 1) — an accident of size,
see the larger cross-sections below. On 3 × 3 (dimension 512):

| | per molecule |
|---|---|
| S_h − S_c | 4.35e-6 |
| S_sym − S_c (the Rayleigh maximum) | 2.14e-6 |
| F_I per cell | 1 − 9e-6 |

— the inequality is strict there, and the Rayleigh quotient sits halfway.

### Flux sectors, larger cross-sections, and why the Rayleigh value sits halfway (2026-09-27)

The ice rules conserve the vertical flux: counting near-H around a bilayer, Σ_c in_c = Σ_c out_c, so
M is block diagonal in k = Σ in_c. A cross-section with an odd number of cells has no zero-flux
sector — its dominant state carries flux ±1 (3 × 3, 3 × 5). The even-sided tori that gave exact
equality (2 × 2 … 2 × 5) are small-size accidents: larger zero-flux cross-sections, from
matrix-free products in the dominant sector (checked against brute force to 3e-16; Mᵀ = I M I to
1e-14 on every torus), per molecule:

| L1 × L2 | cells | S_h − S_c | S_sym − S_c | 1 − F_I per cell |
|---|---|---|---|---|
| 3 × 3 | 9 (flux 1) | 4.35e-6 | 2.14e-6 | 8.97e-6 |
| 2 × 5 | 10 | 0 | 0 | 0 |
| 2 × 6 | 12 | 4.79e-7 | 2.39e-7 | 9.61e-7 |
| 3 × 4 | 12 | 2.73e-7 | 1.36e-7 | 5.46e-7 |
| 2 × 7 | 14 | 1.08e-6 | 5.41e-7 | 2.16e-6 |
| 3 × 5 | 15 (flux 1) | 2.15e-6 | 1.06e-6 | 4.37e-6 |
| 2 × 8 | 16 | 1.65e-6 | 8.25e-7 | 3.27e-6 |
| 4 × 4 | 16 | 3.27e-7 | — | 6.54e-7 |
| 2 × 9 | 18 | 2.13e-6 | 1.07e-6 | 4.21e-6 |

(L2 × L1 is the mirror image of L1 × L2.) The inequality is strict on every zero-flux cross-section
beyond the smallest: ~3e-7 on the 2D tori, growing with length on the two-cell strips.

Two regularities hold on every torus to 1–3 %: S_sym sits halfway, and S_h − S_c per cell equals the
hexagonal state's inversion infidelity −ln F_I per cell (4 × 4: 6.53e-7 against 6.54e-7). The first
is structural. With A = I M (symmetric) and P± the projectors on inversion-even and -odd states,

    (M + Mᵀ)/2 = (I A + A I)/2 = P₊ A P₊ − P₋ A P₋,

so S_sym is the hexagonal operator restricted to inversion-symmetric states, and S_h − S_sym is what
the hexagonal state gains by breaking inversion. In the same blocks M = I A = [[A₊₊, A₊₋], [−A₋₊, −A₋₋]]:
to second order in the coupling A₋₊, σ and λ(M) sit symmetrically about λ(A₊₊), λ(M) ≈ λ₊₊ − c/σ and
σ ≈ λ₊₊ + c/σ with c = |A₋₊ s|² (s the even Perron vector), while the odd admixture gives
1 − F_I ≈ 2c/σ². Hence S_c ≈ 2 S_sym − S_h and S_h − S_c ≈ −ln F_I: the gap is measured by the
inversion asymmetry of the (variational, well-conditioned) hexagonal state — no ill-conditioned
cubic stationary point needed.

## Boundary PEPS (2026-09-27)

A 1×1 PEPS on the cell lattice (square topology), bond D, diagonal-mirror symmetric
(`symmetry = :diagonal`; the cell has no C4v). Three estimators on the same ansatz, per molecule
w = exp(f/2):

* hex — `bra_perm = (2, 1, 4, 3)`, norm unpermuted, maximised: ln σ_max(M);
* Rayleigh — no permutation, maximised: λ((M + Mᵀ)/2), Xu–Lin–Zhang's cubic estimator;
* cubic — both perms inverted, stationary: ln λ(M).

L-BFGS to |g| ~ 1e-5 (D = 2) or 1e-3 (D = 3), then Newton–Krylov: at D = 2 to the gradient's
truncation floor (~2e-6 at χ = 16, ~1e-8 at χ ≥ 32; Ising's is ~1e-9), at D = 3 until f stops moving
(|g| 3e-5–8e-5 there is the solver's pace in ten-minute runs, not a floor — below).

| D | χ | w_hex | w_Rayleigh | w_cubic | ln F_I (hex state) |
|---|---|---|---|---|---|
| 2 | 16 | 1.5073952759 | 1.5073985910 | 1.5074024143 | −1.155e-5 |
| 2 | 24 | 1.5073953076 | 1.5073986155 | 1.5074024485 | −1.153e-5 |
| 2 | 32 | 1.5073953081 | 1.5073986160 | 1.5074024490 | −1.153e-5 |
| 2 | 48 | 1.5073953082 | 1.5073986160 | (1.5074040814, stalled at 1.8e-5) | −1.153e-5 |
| 3 | 24 | 1.5074420992 | 1.5074443564 (±1e-7) | 1.5074482469 (\|g\| 4.5e-5, stalled) | −8.15e-6 |
| 3 | 32 | 1.5074447646 | 1.5074471042 | 1.5074493675 (\|g\| 4.7e-5, still moving) | −7.63e-6 |

Every stored D = 2 optimum re-evaluated with the independent prototype (its own bra construction and
gradient) agrees to 3e-15 in f and exactly in |g|.

D = 2 is converged in χ to 1e-9 (hex, Rayleigh) and 5e-10 (cubic, χ = 24–32). All three lie
~5e-5 below the D → ∞ value, so at D = 2 the exact ordering w_c ≤ w_R ≤ w_h is inverted by the
finite-D errors (the cubic estimator is not a bound). Our Rayleigh value matches Xu–Lin–Zhang's
D = 2 "cubic" 1.5073981 to 5e-7; their hexagonal D = 2 value, 1.5074195, is higher than ours
because they maximise the Rayleigh quotient of the two-layer M Mᵀ (bond 4D² in the sandwich), which
at finite D forgives errors along I M's large negative eigendirections; both converge to σ_max.

The cubic stationary point is soft: at χ = 48 its solve stalled at |g| = 1.8e-5 at a point 1.6e-6
away in w (and with a different inversion overlap, −2.5e-5 against −7.8e-5) — so a cubic value is
only as good as its residual, well below 1e-5.

D = 3 is hard on four CPU cores: one evaluation at χ = 32 costs 25–30 s (66 CTM steps even
warm-started — the ice networks converge slowly), so a ten-minute run makes 1–2 trust-radius-limited
Newton steps, and f still rose ~2e-7 per step at |g| ~ 1e-4. A first pass stopped on |g| stagnating
was premature by ~1e-7 in f; the rows above continue until f stops moving, which can itself stop early
when the radius has shrunk (the Rayleigh maximum at χ = 24 is uncertain by ~1e-7). χ matters at
D = 3 as it did not at D = 2: the maxima rise 2.7e-6 in w from χ = 24 to 32, the cubic value 1.1e-6.
The cubic solves do not creep monotonically: over ~25 runs each, f first fell (by 3e-7) and then rose
(by 4–6e-7) while |g| went from 8e-5 to 4.5e-5 — the soft direction again — so the D = 3 cubic values
are uncertain by several 1e-7 in w (χ = 32 was still rising 1.5e-8 in w per run when stopped).

**Against Xu–Lin–Zhang at matching D.** Our Rayleigh values are close to theirs (D = 2: 1.5073986
against 1.5073981; D = 3: 1.5074444 at χ = 24 and 1.5074471 at χ = 32, against 1.5074454), which
validates the setup. The stationary cubic value on the same ansatz lies *above* their Rayleigh number,
by 4.3e-6 (D = 2) and 4.0e-6 (D = 3, χ = 32), and above our own: 3.8e-6 at D = 2, and at D = 3 3.9e-6
(χ = 24) falling to 2.3e-6 (χ = 32). That is the reverse of the exact ordering w_c ≤ w_R, so at D ≤ 3
it is the stationary estimator's finite-D (and at D = 3 finite-χ) error, not a measurement of S_c.
All three estimators rise by ~4.6e-5 from D = 2 to 3, and by 1–3e-6 from χ = 24 to 32 at D = 3 —
larger than the differences sought: D ≤ 3 on four CPU cores cannot resolve a cubic–hexagonal
difference of 1e-6 or less (the 3 × 3 torus has 4.35e-6).

The cleaner diagnostic is the hexagonal state's inversion overlap, converged in χ at D = 2
(ln F_I = −1.153e-5 per cell), −8.15e-6 at D = 3, χ = 24 and −7.63e-6 at χ = 32. Through
S_h − S_c ≈ −ln F_I per cell (exact to leading order, 1–3 % on every torus, above) these read
S_h − S_c ≈ 5.8e-6 (D = 2) and 3.8e-6 (D = 3) per molecule — still falling with D, and ten times
the 4 × 4 torus's 3.3e-7, so at D ≤ 3 the PEPS's inversion asymmetry is mostly finite-D error.
What the estimator buys: the gap from one variational, well-conditioned state (the hexagonal
maximum) and an overlap, not from the soft cubic stationary point.

## What it needs

* D = 4–6: the differences sought are ≲ 1e-6 and the D = 2 → 3 step is 5e-5. On the GPU cluster
  (docs/boundary_peps.md, "Costs"), and with the ice-specific savings still unused: real arithmetic
  already; U(1) block sparsity from the conserved vertical flux of the ice rules; a honeycomb-native
  contraction of the factorised cell.
* A better hexagonal estimator at finite D: ln σ_max from two independent states,
  max ⟨L|M|R⟩/(|L||R|) (single-layer sandwich, no I M negative-eigenvalue penalty), or Xu–Lin–Zhang's
  M Mᵀ (bond 4D²).
* The gap from the hexagonal state's inversion overlap F_I, extrapolated in D, rather than from the
  soft cubic stationary point; and the inversion-symmetric restriction of the same maximisation
  (S_sym, exactly the other half of the relation) as the check.
* Larger zero-flux tori (4 × 5, 5 × 5, 6 × 6 — the 2D thermodynamic trend of the ~3e-7 gap), by a
  transfer-matrix product in the flux sector rather than dense contraction.
