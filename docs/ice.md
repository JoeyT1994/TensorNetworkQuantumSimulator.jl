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

## Honeycomb formulation: BP simple update + CTMRG (2026-09-27)

Code: `examples/ice/` (`honeycomb.jl` the machinery, `honeycomb_prod.jl` the restartable production
driver, `honeycomb_lane.sh`, `honeycomb_u1.jl`, `honeycomb_variational.jl`; tests in
`test/test_ice_honeycomb.jl`; the numbers in `examples/ice/results_honeycomb.csv`).

**The coordination-4 formulation.** Every O is its own tensor. The bilayer M is a bond-2 PEPO on the
honeycomb (V_a: vertical bond in + 3 in-plane; V_b: 3 in-plane + vertical bond out), and hexagonal
stacking returns to the same honeycomb with the sublattices swapped — the inversion I — so the power
method iterates the symmetric I M. The boundary state is a honeycomb PEPS, X (with the physical
vertical leg) and Y, weights on the three bond types. Grouped into cells it is a sub-ansatz of the
cell PEPS above at the same D (the intra-cell bond is D, not free), but it is built by BP simple
update in seconds, and D reaches the tens: the D = 12 state converges in ~1 minute, discarding 9e-18.

**BP simple update.** Apply M exactly (bonds D → 2D), bring the enlarged state to the Vidal (= BP)
gauge by identity updates, truncate each bond by an SVD in that gauge, re-gauge. The Vidal gauge
reproduces the library's `BeliefPropagationCache` + `symmetric_gauge` bond spectra to 1e-10.
Truncating in a stale (non-canonical) gauge — the first attempt — broke the C3 and arrow-reversal
symmetries and made w fall with D. As D → ∞ nothing is truncated and the state is exact; at finite
D the BP weights are a poor guide to what matters (below).

**The networks.** ⟨ψ|ψ⟩, ⟨Iψ|ψ⟩ (the inverted bra), ⟨Iψ|M|ψ⟩, and ⟨Rψ|ψ⟩ (R reverses every vertical
arrow — exactly ⟨ψ|ψ⟩, a check). RQ = ln κ⟨Iψ|M|ψ⟩ − ln κ⟨ψ|ψ⟩ is the Rayleigh quotient of I M (a
lower bound on 2 ln W(Ih) per cell), ln F_I = ln κ⟨Iψ|ψ⟩ − ln κ⟨ψ|ψ⟩. The sandwich equals the library's
`ice_site` cell-PEPS evaluation of the same state to 1e-15. In the inverted networks the ket's X and
the bra's Y both point to −x/−y, so each network is a 2-layer site of 3-leg tensors on fused legs
(raw bond D², and 2D² for the sandwich) — values identical to the layered networks to 1e-14.

**The CTM seed.** `InfiniteCTM2D`'s default all-ones boundary vector mixes the Z2 sectors of the Vidal
basis; on the overlap networks the CTM then never converges (ln F_R = +0.31 at D = 3 where the exact
value is 0; ln κ jumping by O(1)). Seeding every state leg with e₁, the dominant Vidal basis vector
(ones on a PEPO leg), every network converges in 20–40 iterations and ln F_R = 0 to 1e-14 at every D.

**Results** (χ = D² unless noted; `results_honeycomb.csv`):

| D | χ | ξ | w_h | ln F_I per cell |
|---|---|---|---|---|
| 2 | 8 | 1.170 | 1.5071866 | −1.433e-5 |
| 3 | 9 | 1.259 | 1.5072772 | −7.058e-6 |
| 4 | 16 | 1.863 | 1.5073816 | −5.102e-6 |
| 4 | 32 | 1.973 | 1.5073835 | −5.165e-6 |
| 5 | 25 | 1.922 | 1.5073565 | −4.831e-6 |
| 5 | 50 | 2.037 | 1.5073588 | −4.738e-6 |
| 6 | 36 | 2.359 | 1.5074139 | −5.190e-6 |
| 7 | 49 | 2.396 | 1.5074191 | −4.740e-6 |
| 8 | 64 | 2.509 | 1.5074212 (sandwich on the local GPU) | −4.840e-6 |

* w_h rises with D except at D = 5 (SU is not variational), and is 3.7e-5 below Xu–Lin–Zhang and
  4.6e-5 below Kolafa at D = 8 — still below the cell-PEPS variational D = 3 value (1.5074448). The
  increments shrink (+5.2e-6 from D = 6 to 7, +2.1e-6 from 7 to 8) far faster than the gap: at the D
  reachable, BP simple update will not deliver the absolute w_h — that needs the variational refinement
  (or D well beyond 12). Its ln F_I is flat; whether the BP bias cancels in F_I is the open question.
  A variational pass from the BP-SU state recovers most of the gap at small D (D = 2: 1.5071868 →
  1.5073761; D = 3: → 1.5074106,
  not fully converged), so the finite-D error is the truncation's BP metric, not the ansatz.
* ln F_I settles into −4.7…−5.2e-6 per cell from D = 4 to 8, i.e. S_h − S_c ≈ 2.4–2.6e-6 per molecule if
  it survives D → ∞ — nonzero, and at the size of Kolafa's error bars. Not yet decisive: F_I is first
  order in the state error (at D = 2 it ranges −1.4e-5 … −6.3e-5 across BP-SU, the honeycomb optimum
  and the cell optimum), and it needs χ ≥ D² (D = 6: −1.53e-5 at χ = 16, −5.32e-6 at 24, −5.19e-6 at 36).
* ξ grows with D (and with χ at fixed D): the boundary state looks gapless, so convergence in D is
  algebraic and wants a finite-correlation-length extrapolation (as for the Yang–Lee folds), with ξ
  growing slowly — the states added from D = 6 on carry BP weights ≲ 1e-5.

**U(1) does not help — the boundary state breaks it.** The ice rules conserve flux, and a U(1)
boundary PEPS (zero-flux start, blockwise QR/SVD, charges conserved to 1e-16, ±q multiplets never
split; `honeycomb_u1.jl`) has no finite-D fixed point: the virtual charge is the in-plane flux through
the semi-infinite vertical ribbon under the bond, a plain sum over layers, and it random-walks —
⟨q²⟩ = 0.97, 1.22, 1.55, 1.88 exactly over the first four bilayers, still +0.33 per bilayer at n = 40
with D = 40. At fixed D the charge support widens with D (±2 at D = 6, ±8 at D = 20, ~one state per
sector) and the weights never settle. The grand-canonical state (from the product state, a
superposition of flux sectors) is the efficient one; only the arrow-reversal Z2 survives (exact, in
diagonal signs on the Vidal basis). This holds for any update, simple or full. The zero-flux
eigenvector is its projection onto Φ = 0, with the same free energy per site; its broken U(1) is
consistent with the gapless (Coulomb-phase) boundary state.

**Costs** (local i9, per CTM iteration near convergence, ~35–40 iterations to converge; χ = D²):
⟨ψ|ψ⟩ 2–4 s (D = 6), ~6 s (D = 7), ~30 s (D = 8); the sandwich ~4× that (raw bond 2D²): 10–25 s,
50–100 s, 300–470 s. ~χ³r² ≈ D¹⁰. The D = 8 sandwich does not fit ten-minute runs here; the CUDA path
(`DEVICE=gpu`) reproduces the CPU to 13 digits.

## How far BP simple update is from the eigenvector — and variational states (2026-09-28)

**The eigenvector residual.** For A = I M (symmetric), Cauchy–Schwarz gives, per cell,

    ln f = 2 RQ(A) − RQ(A²) = 2 ln κ⟨Iψ|M|ψ⟩ − ln κ⟨Mψ|Mψ⟩ − ln κ⟨ψ|ψ⟩ ≤ 0,

0 only for an eigenvector: the TRUE log-fidelity between one exact bilayer applied to the state and its
truncation back. ⟨Mψ|Mψ⟩ = ⟨ψ|A²|ψ⟩ (Mᵀ M = A²) is the `:mnorm` network (raw bond 4D²). BP simple
update (χ = D²):

| D | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| ln f per cell | −8.26e-5 | −2.38e-5 | −3.70e-5 | −1.15e-5 | −1.12e-5 | −1.15e-5 |
| BP's discarded weight per bilayer | 4.7e-7 | 1.2e-10 | 4.3e-11 | 7.6e-12 | 3.2e-13 | 1.5e-14 |

BP's own measure of what the truncation throws away is wrong by 2 to 9 orders of magnitude and the true
residual stalls near 1e-5 per cell from D = 6 — the BP (Bethe) metric cannot see the loop (ice-rule)
correlations it truncates. That is why w_h(BP-SU) creeps up with D far below the literature.

**Variational states** (`honeycomb_variational.jl`, on the local GPU): L-BFGS on RQ with the CTM
gradient and the norm-metric preconditioner, started from the BP-SU state, optimised at χ_opt and
evaluated at χ_opt and 2χ_opt (`results_honeycomb.csv`):

| D | χ | w_h | ln F_I per cell | ξ | ln f per cell | against BP-SU (same D) |
|---|---|---|---|---|---|---|
| 3 | 36 (opt. 18) | 1.5074106 | −4.250e-6 | 2.16 | −3.19e-5 | w 1.5072791, F −7.06e-6, ξ 1.31, f −8.3e-5 |
| 4 | 64 (opt. 32) | 1.5074547 | −5.404e-6 | 4.80 | −1.99e-6 | w 1.5073835, F −5.17e-6, ξ 1.97, f −2.4e-5 |
| 5 | 64 (opt. 32) | 1.5074530 | −3.045e-6 | 4.40 | −3.39e-6 | w 1.5073588, F −4.74e-6, ξ 2.04, f −3.7e-5 |

References: cell PEPS variational 1.5074448 (D = 3, a richer ansatz at equal D); Xu–Lin–Zhang's raw
D = 7, χ = 150: w_h 1.5074584 (their w_c 1.5074533); Kolafa's MC 1.5074674(38).

* The variational states are 10× closer to an eigenvector (D = 4: −2.0e-6 against −2.4e-5) and twice
  as correlated (ξ 4.4–4.8 against ~2): BP-SU underestimates the boundary state's correlations, as a
  Bethe approximation should.
* w_h: D = 4 reaches 1.5074547 at χ = 64 — 3.7e-6 below Xu–Lin–Zhang's D = 7, χ = 150 value, and still
  rising with χ (+8e-7 from χ = 32 to 64). D = 5 is NOT its optimum: optimised at χ = 32, below the
  sandwich's raw bond 2D² = 50, its gradient was too noisy and it stalled below D = 4. The variational
  optimum at D ≥ 5 needs χ_opt ≥ 2D² — cluster work (docs/status_3d.md).
* ln F_I is converged in χ at every variational point (≤ 3e-8 from χ_opt to 2χ_opt).
* Every "ln f > 0" at χ_opt is ⟨Mψ|Mψ⟩ not converged in χ (raw bond 4D² > χ); at 2χ_opt it is negative.

**Verdict on ln F_I ≈ −5e-6 per cell.** The sign and the order of magnitude survive: every state —
BP-SU at D = 4–9 (−4.5 … −5.2e-6) and variational at D = 3–5 (−3.0 … −5.4e-6) — has a clearly
nonzero inversion asymmetry, i.e. S_h − S_c ≈ 1.5–2.7e-6 per molecule by S_h − S_c ≈ −ln F_I. The
specific value −5e-6 does not: across the variational states it scatters by ±1.2e-6 without a trend,
F_I being first order in the state error and the D ≥ 4 optima noise-limited (the gradient's floor at
χ_opt). The number that would settle the question needs converged optima at D = 5–8 with χ_opt ≥ 2D²
and a ξ-extrapolation of both w_h and F_I — the first cluster campaign.

## What it needs

* The first cluster campaign (`examples/ice/cluster/`): variational optima at D = 5–8 with χ_opt ≥ 2D²,
  evaluated at 2χ_opt; BP-SU D = 9–11 and the residual for the record; then the ξ-extrapolation of w_h
  and ln F_I. Z2 block sparsity is available (U(1) is not, above).
* A better optimiser: gradients accurate below the current floor (|g| ~ 1e-4–1e-3 at χ_opt) — implicit
  (fixed-point) differentiation of the CTM environment, and χ_opt ≥ 2D² throughout.
* A better hexagonal estimator at finite D: ln σ_max from two independent states,
  max ⟨L|M|R⟩/(|L||R|) (single-layer sandwich, no I M negative-eigenvalue penalty), or Xu–Lin–Zhang's
  M Mᵀ (bond 4D²).
* The gap from the hexagonal state's inversion overlap F_I, extrapolated in D, rather than from the
  soft cubic stationary point; and the inversion-symmetric restriction of the same maximisation
  (S_sym, exactly the other half of the relation) as the check.
* Larger zero-flux tori (4 × 5, 5 × 5, 6 × 6 — the 2D thermodynamic trend of the ~3e-7 gap), by a
  transfer-matrix product in the flux sector rather than dense contraction.
