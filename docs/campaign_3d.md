# The 3D campaign: what precision is reachable, and what to aim at

*2026-09-28. A decision document. Companions: [`roadmap_3d.md`](roadmap_3d.md) (literature, levers) and
[`status_3d.md`](status_3d.md) (what runs today). It answers one question — can the ice boundary PEPS go far
enough beyond Xu–Lin–Zhang (D = 7, χ = 150) to get w_h to ~8 decimal places and make a statement? — and
re-ranks the programme's targets in the light of the answer. **(ours)** marks our own reasoning, not an
established result. *Extrapolated* numbers come from `examples/ice/cluster/cost_table.md` (±3×).*

---

## 1. Short answer

**No, not for the absolute entropy.**
- At the ceiling of the current contraction (D ≈ 10 at χ = 2D², one H200), w_h is most likely good to 6–7
  decimal places, not 8. That matches Kolafa's MC error bar (±3.8e-6) or beats it a few times over.
- Eight digits needs the optimistic corner of two exponents we have not measured (§2.2).

**But 8 d.p. in w_h is the wrong target.**
- The question is the size of S_h − S_c, about 1e-6 per molecule. S_h ≥ S_c is a theorem, so only the size
  is open.
- The gap can be measured directly from one state, with the ~1e-5 finite-D error of the absolute entropies
  common to both sides.
- That needs about 5e-7 per molecule *in a difference*, which looks reachable at D ≤ 10 — if two things we
  have not yet checked hold (§3.3–3.4).

## 2. Why not 8 d.p. in w_h

### 2.1 The memory and cost wall

Variational optima need the sandwich ⟨Iψ|M|ψ⟩ converged at χ ≥ 2D². The overnight evidence: D = 5 optimised
at χ = 32 < 2D² stalled below D = 4. The working set is ≈ 3 × 2.3 χ²r² × 8 B with r = 2D²:

| D | χ = 2D² | sandwich working set | fits on | variational optimum, one GPU (extrapolated) |
|---|---|---|---|---|
| 8 | 128 | 15 GB | any | ~3 h |
| 9 | 162 | 38 GB | H100 | ~10 h |
| 10 | 200 | 88 GB | H200 only | ~1–2 days |
| 11+ | | ≥ 180 GB | — (the subspace block needs column batching, not implemented) | |

Time grows as ~D¹⁰ on top. Our ceiling is therefore D ≈ 10, χ ≈ 200. That is Xu–Lin–Zhang's regime moved
~1.4× in D, not a different league. Their own w_h at χ = 150 is identical for D = 5, 6, 7, i.e. χ-limited
(roadmap_3d.md §4.1).

### 2.2 Convergence in D is algebraic

The boundary state is gapless. The ice rules' U(1) is broken by it (the flux through a ribbon random-walks,
docs/ice.md), and ξ grows with D. So the error in w_h falls as a power of ξ, not exponentially in D:

    w_∞ − w_h(D) ≈ a ξ^(−p).

**The exponent p.** The two usable variational points, against Kolafa's w_h = 1.5074674(38):

| D | ξ | error |
|---|---|---|
| 3 | 2.16 | 5.7e-5 |
| 4 | 4.80 | 1.27e-5 |

- They give p ≈ 1.9 (1.6–2.2 within Kolafa's error bar).
- The 2+1D-critical expectation (Rader–Läuchli; Corboz et al.) is p = 3.
- Neither D = 3 nor D = 4 is safely asymptotic.

**The ξ needed:**

| raw error in w_h | p = 2 | p = 3 |
|---|---|---|
| 1e-7 | ξ ≈ 54 | ξ ≈ 24 |
| 1e-8 | ξ ≈ 171 | ξ ≈ 52 |

**The ξ at D = 10 is unknown.** Critical 2D PEPS typically have ξ ∝ D^κ with κ ≈ 1–2. Our two points give
κ ≈ 2.8, but they are pre-asymptotic. The raw error at D = 10 across the plausible range:

| | κ = 1 (ξ ≈ 12) | κ = 2 (ξ ≈ 30) | κ = 2.8 (ξ ≈ 62) |
|---|---|---|---|
| p = 2 | 2e-6 | 3e-7 | 8e-8 |
| p = 3 | 8e-7 | 5e-8 | 6e-9 |

- A ξ-extrapolation buys about one more digit, limited by the uncertainty in p itself.
- The likely outcome is 6–7 decimal places. Eight needs ξ growing like D^2.5 or faster *and* p = 3.
- The first cluster data (converged optima at D = 5–8) measures κ and p, and so decides which corner we are
  in. That measurement is the first deliverable of an ice campaign, before w_h itself.

### 2.3 Nobody else has 8 digits either

| source | value |
|---|---|
| Kolafa (MC) | w_h ± 3.8e-6, w_c ± 3.6e-6 |
| Xu–Lin–Zhang (extrapolated) | S_h 0.4104251(6), S_c 0.4104248(14) — from χ-limited raw values |
| Vanderstraeten et al. 2018 (D ≤ 4) | 1.5074562 |

No one has w_h beyond ~6 d.p. with a defensible error bar. w_h to ~1e-6 with an honest ξ-extrapolation would
be a respectable by-product, not the headline.

## 3. What can make a statement: the gap, directly

### 3.1 The quantity

S_h ≥ S_c exactly (σ_max(M) ≥ λ_max(M)). The open questions are whether the gap survives the thermodynamic
limit, and how big it is:

| source | S_h − S_c per molecule |
|---|---|
| Kolafa | w_h − w_c = 1.4e-6 ± 5e-6: unresolved |
| Xu–Lin–Zhang | 3.4e-6 raw; extrapolated to zero |
| our PEPS (BP-SU D = 4–9, variational D = 3–5) | −ln F_I / 2 = 1.5–2.7e-6 |

A statement means X ± Y with Y ≤ X/3, from two independent estimators that agree. If X ≈ 2e-6, that is
Y ≈ 5e-7 per molecule.

### 3.2 Two direct estimators, from the same operator A = I M

Neither subtracts two absolute entropies. Both rest on the leading-order relations of docs/ice.md, exact to
1–3 % on every torus:

    S_h − S_c ≈ −ln F_I per cell ≈ 2 (S_h − S_sym).

| estimator | error | status |
|---|---|---|
| −ln F_I = −[ln κ⟨Iψ\|ψ⟩ − ln κ⟨ψ\|ψ⟩] of the hexagonal optimum | first order in the state error | measured D = 2–9; variational values scatter ±1.2e-6 (noise-limited optima) |
| 2 (S_h − S_sym): RQ(A) maximised over all states, minus over inversion-symmetric states (Y tied to the inversion image of X — the map `paired(:inv)` already uses) | second order in each maximum; the large even-sector (Coulomb) error should be common to both **(ours)** | not built; the networks exist, it is one constraint on the ansatz |

- **The second estimator is the one to bet on; the first is its cross-check.** Their error structures differ,
  so agreement and a joint trend in ξ are the evidence.
- **The difference is not a bound**, although each maximum is a lower bound at converged χ.
- **The cubic stationary point ln⟨IR|M|R⟩ − ln⟨IR|R⟩ stays out.** It is soft and ill-conditioned (the
  cell-PEPS experience, docs/ice.md).

### 3.3 The tension to resolve first

The exact zero-flux tori and the PEPS disagree by a factor of 6–8:

| source | S_h − S_c per molecule |
|---|---|
| exact 3 × 4 torus | 2.7e-7 |
| exact 4 × 4 torus | 3.3e-7 |
| every PEPS | ~2e-6 |

Two explanations are possible:
- **The small 2D tori are far from the thermodynamic limit.** The 2 × L strips grow with L, reaching 2.1e-6 at
  2 × 9. Flux quantisation on a small torus may suppress the asymmetry **(ours)**.
- **The PEPS inversion asymmetry is still mostly finite-D error.**

Until this is resolved, no number is a statement. If the tori are right, the target becomes ~1e-7, probably out
of reach. The honest result would then be an estimate "S_h − S_c ≲ X", not a bound.

### 3.4 Gates — cheap and local, before any cluster time

**G0. Inversion-symmetric ansatz, D = 2–4.**
- Compute 2(S_h − S_sym) next to −ln F_I, in ten-minute resumable runs on the local GPU.
- Pass: the two agree within their spreads from D = 3, and the second-order estimator moves less with D and χ.

**G1. Larger exact tori.**
- 4 × 5 and 4 × 6 run with `ice_cross_sections.jl` as it is: full 2^N vectors; 4 × 6 may be memory-tight.
- 5 × 6 and 4 × 8 need a product restricted to the zero-flux sector. The vectors are C(30,15) = 1.6e8 entries
  (1.2 GB) and C(32,16) = 6e8 (4.8 GB); the latter needs a cluster CPU node.
- Pass: the 2D-torus trend is understood — either rising towards the PEPS value or clearly saturating near 3e-7.

**G2. Implicit (fixed-point) CTM gradients.**
- These make optima converged rather than noise-limited. The D = 5 stall and F_I's variational scatter both
  come from the noise floor.
- Without them the cluster buys expensive noise.

### 3.5 Then the campaign — or a stop

**If G0–G2 pass:**
- **Runs:** both estimators and w_h at D = 5–10, with χ_opt = 2D² and evaluation at 2χ_opt. Add a χ sweep for
  a ξ-collapse (roadmap L3), and fit κ, p and the gap's ξ-dependence.
- **Deliverables:**
  - S_h − S_c with an error bar: ≥ 3σ, or an upper estimate.
  - w_h to ~1e-6.
  - The BP-metric lesson: BP's discarded weight misjudges the true truncation error by up to nine orders.
- **Budget (extrapolated):** D ≤ 8 takes one afternoon on ≤ 16 GPUs; D = 9–10 take a day or two on H200s.

**Stop if both of these hold:** G0 fails (the second-order estimator does not settle faster than the first-order
one) *and* G1 leaves the factor 6–8. Then:
- Write the ice work up as a note: w_h, the gap as an estimate with ~50 % error, and the eigenvector-residual
  diagnostic.
- Move the GPUs to the targets of §4.

---

## 4. The larger plan: where the boundary PEPS can make a mark

### 4.1 The lesson that re-ranks the targets

Precision in this programme is set by the **boundary state**, not by the model.

- **Gapped boundary states converge exponentially in D.** These are off-critical phases, both branches of a
  first-order transition, and an interface below roughening. BP-SU warm starts are good there, and nested
  single-layer networks (Yang et al., D = 20) become usable. 8+ digits are plausible.
- **Gapless or critical boundary states converge as ξ^−p.** These are Coulomb phases, critical points and the
  Yang–Lee edge. 1e-3 relative in universal quantities is realistic; 1e-8 absolute is not.
- **Some things MC cannot do at all.** Complex or negative weights (the sign problem); exponentially small
  differences and overlaps (the ice gap, interface free energies); thermodynamic-limit branches at first-order
  transitions, without metastability.

So high impact here means one of two things:
- an open question answerable with ~1e-3 relative precision in a universal quantity, or
- a direct difference or overlap computed from gapped states.

Both are best where MC has a sign problem.

### 4.2 Targets, re-ranked

#### 1. Ice Ih vs Ic — finish it (closest; the gates decide)

- **Question:** a 90-year-old one — Pauling, Onsager's inequality, Nagle. A defensible nonzero gap, or a sharp
  estimate that it is ≲ 1e-7, would settle a live disagreement: XLZ claim equality, Kolafa's MC cannot resolve
  it.
- **Precision:** 5e-7 per molecule in a difference (§3).
- **Boundary state:** gapless.
- **Status:** pipeline built (`examples/ice/`, the cluster kit). The next steps are G0–G2.
- **Risk:** medium — the torus tension.

#### 2. The 3D Yang–Lee edge: first lattice value of its universal location z_c (highest impact per digit)

- **Question:** FRG gives 2.43(4), analytic continuation 2.429(56), and there is no lattice number. It matters
  for QCD critical-point searches, which use z_c.
- **Precision:** ~1 % in z_c — which the ξ^−p convergence can deliver.
- **Why us:** MC cannot sit at imaginary field. The exponent is already pinned: Δφ ≈ 0.215 (fuzzy sphere, series,
  6-loop ε).
- **Needs:**
  - the fold solver (works at D = 2, `examples/yang_lee/fold_pseudoarclength.jl`) in the library, resumable;
  - D = 4–10 on GPUs, with the finite-ξ scaling θ_c − θ_f ∝ ξ_f^−(3−Δφ);
  - the lattice metric factors in the error budget (β_c = 0.221654626(5) and amplitude normalisations from MC);
  - validation in the same universality class: the cubic monomer–dimer model at negative activity,
    z₀ = −0.0520268(2) (Butera–Pernici) — a sign problem for MC, none for us.
- **Optional:** the real signed-vertex formulation (roadmap L4) would cut the arithmetic 3–4×.
- **Risk:** medium–high — metric factors and corrections to scaling.

#### 3. 3D Ising interface tension and roughening, from the overlap of the ± fixed points (low risk)

- **Idea (ours):** below T_c the layer transfer operator has two dominant eigenvectors ψ₊ and ψ₋ (the
  symmetry-broken boundary PEPS). For an interface parallel to the layers, Z₊₋/Z₊₊ = ⟨ψ₋|ψ₊⟩/⟨ψ₊|ψ₊⟩, so the
  interface tension per site is βσ = −ln F₊₋ per site. That is the same "overlap per cell" as ice's F_I.
  - Check it first on the 2D Ising model against Onsager's exact βσ = 2βJ + ln tanh βJ.
- **Precision:** exponential in D below roughening (β > β_R ≈ 0.4075, MC by Hasenbusch–Meyer–Pütz 1996 —
  check the source).
  - Above β_R the interface is rough and the overlap network is a critical 2D surface (SOS-like).
  - Roughening would then be the Kosterlitz–Thouless (KT) point of the overlap network's ξ — a direct TN
    determination of β_R.
- **Why:**
  - It is a clean, gapped showcase of the overlap estimator the ice statement relies on, which de-risks
    target 1.
  - The only TN work on this so far uses finite slabs (2601.07829).
  - Near T_c, σ gives universal amplitude ratios.
- **Needs:** ± states and their overlap. Everything exists (`boundary_peps` with `ising3d_site`, the paired
  overlap networks).
- **Scale:** the first step is local, D ≤ 5, days.
- **Risk:** low. The impact is moderate, but it is the fastest route to a precision result.

#### 4. The 3D Z2 gauge–Higgs self-dual line (highest discovery value, most new code)

- **Question:** is the multicritical point XY* (Bonati–Pelissetto–Vicari 2112.01824) or a new self-dual CFT
  (Somoza–Serna–Nahum 2012.15845)?
- **Approach:** on the first-order self-dual line both phases are trivial, with gapped boundary states. That
  gives the latent heat from two thermodynamic-limit branches. How it vanishes towards the multicritical point
  discriminates the scenarios.
- **Needs:**
  - the gauge–Higgs transfer tensors;
  - a two-branch (coexistence) solver;
  - multi-site unit cells in `InfiniteCTM2D` (roadmap L6).
- **Risk:** high — new tensors, and ξ grows near the endpoint.

#### 5. Three-state Potts at complex field (the heavy-dense QCD effective model)

- **Question:** the endpoint against the chemical potential, and a claimed re-entrant first-order region
  (Ejiri–Koiida 2601.06446). TRG results are low precision.
- **Approach:** gapped branches except at the endpoint, so ~1e-3 in the endpoint is realistic. Complex weights
  are handled by our bilinear machinery.
- **Anchor:** the real-field endpoint (β_c, h_c) = (0.54938(2), 0.000775(10)).
- **Needs:** d = 3 sites in the bilinear solver.
- **Risk:** medium.

**Benchmarks only:** cubic dimers (a Coulomb phase like ice) and the 3D Ising T_c — as validation of any new
engine, not as targets.

### 4.3 Engineering levers, ranked for these targets

1. **Implicit fixed-point gradients, plus gauge fixing** (2311.11894, 2508.10822, 2607.15124). Every
   variational target needs them; the ice D = 5 stall and F_I's scatter are this.
2. **Symmetry-constrained ansätze:** inversion-symmetric (S_sym, gate G0), ± sectors (target 3), c4v. Cheap.
3. **The fold solver in the library,** resumable (target 2).
4. **Nested single-layer networks** (roadmap L2). ⟨Iψ|ψ⟩ is already single-layer (the dice lattice, bond D),
   and Yang et al.'s checkerboard split does the same for cubic models. This means D ≈ 20 wherever BP-SU states
   are good — targets 3–5, not ice's state quality.
5. **Loop-aware truncation environments (ours, speculative).**
   - For Coulomb-phase states, BP's metric misjudges the true truncation error by up to nine orders
     (docs/ice.md).
   - A hexagon-cluster environment (a cluster update, between BP and a full CTM) might recover most of the
     variational gain at simple-update cost.
   - It is the one lever that could move ice past D ≈ 10.
6. **A column-batched subspace block:** D = 11–12 at χ = 2D² on an H200. Beyond that it is not worth pushing
   one contraction; spread parameter scans across GPUs instead.
7. **Multi-site unit cells in `InfiniteCTM2D`** (targets 4, 5 and ordered phases).
8. **Z2 block sparsity** (×2–4).

### 4.4 What not to do

- Absolute free energies of gapless boundary states beyond ~1e-7 (ice w_h at 8 d.p.): the cost grows as ~D¹⁰
  against algebraic convergence.
- 3D critical exponents: the conformal bootstrap is at 1e-7–1e-8.
- BP or loop series for precision in Coulomb or critical phases (proven to fail, 2604.03228). Use them only as
  truncation environments (lever 5) and warm starts.
- Direct 3D CTMRG beyond cross-checks.
- Splitting one contraction across GPUs.

## 5. Sequence

**Phase A — local, ~1–2 weeks, ten-minute resumable runs.**
1. Ice G0: the inversion-symmetric ansatz; 2(S_h − S_sym) against −ln F_I at D = 2–4.
2. Ice G1: tori 4 × 5 and 4 × 6; the sector-restricted product for 5 × 6 and 4 × 8.
3. Interface pilot: the 2D Ising σ against Onsager; the 3D Ising σ(β) at D ≤ 5 below roughening.
4. Implicit gradients (G2), and the fold solver into the library.

**Phase B — the first cluster campaign (ask the user first).**
- Whichever of ice (if G0–G1 pass) and the interface is ready.
- Ice at D = 5–10 with both estimators; its first output is κ and p, then the gap with an error bar.
- The interface: σ(β) at D up to ~10, and the roughening point.

**Phase C — the Yang–Lee edge.** D = 4–10 with metric factors, and the monomer–dimer validation.

**Phase D — gauge–Higgs or the complex-field Potts model**, after multi-site cells.

## References

As in [`roadmap_3d.md`](roadmap_3d.md), plus:
- Onsager's 2D interface tension, βσ = 2βJ + ln tanh βJ;
- Hasenbusch, Meyer & Pütz, J. Stat. Phys. 85, 383 (1996) for β_R (value to be checked against the source).
