# How far can the 3D CTMRG / boundary-PEPS programme go? A roadmap

*2026-09-27. Companion to [`status_3d.md`](status_3d.md) (what exists and runs today). This is the
synthesis of a literature survey (methods, targets; arXiv IDs checked against abstract pages by the
survey, Xu–Lin–Zhang's Table 3 checked directly), an inventory of this library, and what the ice and
Yang–Lee campaigns taught us. Items marked **(ours)** are our own reasoning, not established results.*

---

## 1. Where we stand, in one paragraph

Three routes to 3D classical models exist here. Direct 3D CTMRG (corner/edge/face blocks) is
**biased** — one χ-index per quarter-plane cannot hold a 2D boundary — and nothing in the 2024–26
literature fixes that; park it. The **boundary iPEPS** (layer transfer operator's eigenvector as a 2D
PEPS, contracted by `InfiniteCTM2D`) is accurate, variational for symmetric operators, handles complex
couplings (the Yang–Lee edge), and is the programme's backbone. Its wall is the 2D contraction:
~χ³r² per CTM step with r = D² (norm) or 2D² (sandwich) and χ ≈ D², i.e. **~D¹⁰**, which stops CPU
work at D ≈ 7–8. **BP simple update** builds boundary states at any D in seconds (ice: D = 12 in a
minute) and converges to the exact state as D → ∞; the evaluation is the same D¹⁰ contraction.

## 2. What the literature says (condensed)

| family | best recent result | lesson for us |
|---|---|---|
| Boundary PEPS, variational | Vanderstraeten–Vanhecke–Verstraete 2018 (1805.10598): D ≤ 4 (ice 1.5074562, 3D Ising T_c to 1e-5); scaling hypothesis, Vanhecke et al. 2022 (2102.03143): dimers D ≤ 6, χ ≤ 276 | fixed-χ data + **collapse in ξ(D, χ)**: χ need not be converged; free exponents are ill-conditioned |
| Boundary PEPS, split CTMRG | Xu–Lin–Zhang 2025 (2506.19339): 3D Ising D = 6, χ = 100, T_c 3e-4 off; ice 2026 (2511.22477): D ≤ 7, χ ≤ 150 | split layers cut D¹⁰ → D⁹ (2–3× at D = 6); their ice w_h is **χ-limited** (below) |
| Boundary PESS by simple update + "nesting" | Yang, Fu, Xie, Xiang 2023 (2210.09896): 3D Ising **D = 20**, T_c to 1e-4 | the dual eigenvector as a translated copy makes ⟨Ψ′\|Ψ⟩ a **single-layer** network (bond D, not D²): D¹² → D⁹. SU underestimates T_c (Bethe-like) |
| Coarse graining (HOTRG/ATRG/triad) | HOTRG T_c to ~3e-7 at D ≈ 23 (1405.1179); ATRG χ⁷ (1906.02007) | good T_c, bad exponents (ν = 0.571 at D = 32, 2602.08987); cross-checks only |
| BP / generalized BP / loop series | loop series and cluster expansions (2409.03108, 2510.02290, 2510.05647); GBP on ice (2604.24760): W ≈ 1.50665, ~5e-4 off | **proven** to degrade at criticality (Midha–Sommers–Tindall–Abanin 2604.03228): Coulomb phases and critical points are out of reach; warm starts and error bars only |
| GPU CTMRG | QR-CTMRG (c4v; 2505.00494): D = 8, χ = 700 on one H100, 140× faster than standard CTMRG; **SI-CTMRG** (general cells; 2607.15158): recycled subspace iteration + QR + small SVD, decompositions < 2 % of GPU time, D = 8, χ ≤ 640, up to 680× on GPU; Ace-TN multi-GPU (2503.13900): 50–60× on 4×A100 | the decompositions can be made negligible; what is left is FP64 GEMM. Parameter scans across GPUs, not one contraction split over many |
| Symmetry savings | bra–ket (Hermitian) Z2: up to 4× (2410.11596); c4v single-corner schemes | ice's arrow reversal Z2 is exact (ours); U(1) is not usable (ours: broken by the boundary state) |
| Gradients | implicit/fixed-point differentiation with the corrected SVD gradient (2311.11894); gauge-fixed optimisation (2508.10822); sparse implicit solver (2607.15124) | optimisers can "abuse" the environment gauge → bias; restart environments to detect local maxima |

No tensor-network study of the **3D Yang–Lee edge** exists (the survey found none); Monte Carlo cannot
sit at imaginary field. For **ice**, see §4.1 — there is a live tension the TN can settle.

## 3. The levers, ranked by what they buy

**L1. GPUs with decomposition-light projectors (×50–100 on the constant).** Our `InfiniteCTM2D` already
has the right structure — lazy layers, split (matrix-free) pairs with a warm-started subspace SVD — which
is SI-CTMRG's idea. What remains is (a) making sure the device path never falls back to the dense
enlarged quadrant (it does on graded data and on subspace bail-outs, docs/boundary_peps.md), (b) a
QR-based projector option (QR-CTMRG for c4v networks: 3D Ising at zero field, the Yang–Lee sandwich is
c4v), (c) profiling on an H100 before believing any estimate. Because the cost is ~D¹⁰, ×100 buys ~+50 %
in D: D ≈ 8 → 12. That is the entry ticket for everything in §4.

**L2. Fewer, cheaper networks: nesting (D¹⁰ → ~D⁹, and memory D⁸ → D⁷).** Two forms:
* *Split CTMRG* everywhere (ket and bra never fused inside the enlarged corner). We have it for the
  pair; the block updates should follow.
* *Nesting* — use a dual eigenvector that is a lattice image of the right one, so ket and bra bonds
  never coincide and the overlap is single-layer. **(ours)** For hexagonal ice this is already true of
  ⟨Iψ|ψ⟩: the ket bonds run a → a + d_k, the inverted bra's a → a − d_k, so ket X ⊗ bra X (contracted
  over p) at a, ket Y at b and bra Y at the hexagon centres form the planar **dice lattice** with bond D
  on every edge. ⟨ψ|ψ⟩ and ⟨Iψ|Mψ⟩ remain genuinely double-layer (their ket and bra bonds coincide).
  Exploiting it needs a CTM on a 3-site cell (the engine takes a list of layers per site but cuts every
  cell boundary through all of them; the win needs cuts that cross one layer at a time). For the cubic
  lattice (Ising, Yang–Lee) the checkerboard split T = T₂T₁ of Yang et al. is the known route to D = 20
  — the most important single paper for us.

**L3. Finite-correlation-length scaling instead of χ-convergence.** Every (D, χ) pair is a data point
f(ξ(D, χ)). This turns the "χ ≥ D²" requirement we measured for ln F_I into a collapse, and it is the
only honest way to extrapolate a gapless boundary state (ice is gapless — its U(1) is broken; the Yang–Lee
fold and 3D Ising T_c are critical). For Goldstone-like boundaries the leading correction to a free
energy density is expected ~ξ⁻³ (Rader–Läuchli, Corboz et al.); order-parameter-like overlaps ~ξ⁻¹ —
**(ours, to be tested)** on the ice data D = 4–8.

**L4. Symmetries that are actually there.** Arrow-reversal Z2 for ice (exact in the Vidal basis),
bra–ket Z2 (up to 4×), c4v where the site has it. Graded data run in BP, symmetric gauge, simple update
and the finite 2D CTM already; `InfiniteCTM2D`'s graded fallback is coded but never exercised, and the
boundary-PEPS solvers are dense only. **Real arithmetic for Yang–Lee (ours, from Krčmár–Gendiar–Šamaj's
2D mapping, 2112.09536):** rewrite the imaginary-field Ising model as a real, signed vertex model; below
the edge the leading eigenvalue is real and the edge is the collision of the two leading eigenvalues
— our PT-breaking picture — at 3–4× less arithmetic than complex.

**L5. Better states at fixed D.** BP simple update is cheap but not variational (w_h(D) non-monotone;
the loop correlations BP ignores are the error). The variational refinement (`honeycomb_variational.jl`)
recovers most of the gap at small D; making it a library feature for multi-tensor unit cells, with
implicit gradients and gauge fixing, is the accurate end. SU remains the warm start at every D.

**L6. Multi-site unit cells in `InfiniteCTM2D`.** Needed for anything with sublattice or columnar
structure (ordered Potts phases, gauge–Higgs, interfaces with a staggered boundary, dimers' columnar
phase), and for L2's nested networks. The 1×1 cell is the largest structural limit on *targets*.

**L7. What not to do.** Precision from BP / loop series in critical or Coulomb phases (proven
limitation); direct 3D CTMRG beyond teaching and cross-checks; 3D universal exponents (bootstrap is at
1e-7–1e-8; TN at 1e-3); continuous-symmetry models (O(2), O(3): Goldstone boundaries, basis truncation).

## 4. Targets, ranked by impact × feasibility

### 4.1 Ice Ih vs Ic — settle it (highest; we are closest)

* *Numbers.* Kolafa (MC): w_h 1.5074674(38), w_c 1.5074660(36). Xu–Lin–Zhang (split CTMRG, D ≤ 7,
  χ ≤ 150) raw at D = 7, χ = 150: **w_h = 1.5074584, w_c = 1.5074533 (ΔS = 3.4e-6)**; their "equal" comes
  from a ξ-extrapolation (S_h 0.4104251(6), S_c 0.4104248(14)) that moves S_c up by ~3e-6. Chen–Ran
  (2608.07613): h ≥ c at every finite cross-section, not decisive in the limit.
* *The tension.* Kolafa's S_h is 6e-6 above XLZ's (2.4σ). Their Table 3 shows why, **(ours)**: w_h at
  χ = 150 is **identical (1.5074584) for D = 5, 6, 7** and rises by ~8e-7 from χ = 100 to 150 at every D —
  their hexagonal value is χ-limited, not D-converged, and a Rayleigh quotient only rises with χ. Our own
  inversion overlap gives S_h − S_c ≈ 2.4–2.6e-6 per molecule, flat from D = 4 to 8 — the size of their
  raw difference, opposite to their conclusion.
* *What settles it.* Absolute w_h and w_c to ≤ 3e-7 each, or the gap directly to ≤ 5e-7: D ≈ 10–14 with
  χ ≳ D² (or a ξ-collapse, L3). Our honeycomb pipeline is ready (`examples/ice/honeycomb_prod.jl`,
  device-agnostic, restartable); L1 makes D = 12 a few GPU-days. Checks: S_sym (inversion-symmetric
  restriction, the other half of the Cauchy–Schwarz relation), the variational w_h, stacking-disordered
  mixtures. Competition: the XLZ group names other ice phases as their next step.

### 4.2 The 3D Yang–Lee edge: the first lattice determination of its universal location

* *Numbers.* Edge exponent now pinned: Δφ ≈ 0.215 (σ ≈ 0.077 ± 0.001) from fuzzy sphere (2505.06369),
  series (1206.0872) and 6-loop ε (2510.05723) — our fits use Δφ = 0.215 already. The universal location
  |z_c| = 2.43(4) (FRG, 2203.16651) / 2.429(56) (Schofield continuation, 2311.13530) has **no lattice
  determination**; it matters for QCD critical-point searches.
* *What we need.* The edge line h_c(T) near T_c by the fold solver (pseudo-arclength, prototype works
  at D = 2) at D = 4–12 with finite-ξ scaling θ_c − θ_f ∝ ξ_f^{−(3−Δφ)}, plus the lattice metric factors
  (from our own near-critical data or MC, in the error budget). Validation in the same universality class
  with real signed weights: the cubic monomer–dimer model at negative activity, z₀ = −0.0520268(2)
  (Butera–Pernici) — a sign problem for MC, none for us.

### 4.3 The 3D Z2 gauge–Higgs self-dual line (high impact, new tensors)

XY multicritical (Bonati–Pelissetto–Vicari, 2112.01824) or a new self-dual CFT (Somoza–Serna–Nahum,
2012.15845)? Along the first-order self-dual line both phases are trivial (short-range boundary states),
so the latent heat from two thermodynamic-limit branches, and how it vanishes towards the multicritical
point, is a direct discriminator. Needs the link/gauge tensor construction and probably L6.

### 4.4 Three-state Potts at complex field (heavy-dense QCD effective model)

The endpoint against the "chemical potential" and a claimed re-entrant first-order region (Ejiri–Koiida
2601.06446); TRG results are low precision (2503.05144). Complex weights: our bilinear machinery.
Real-field endpoint (β_c, h_c) = (0.54938(2), 0.000775(10)) as the anchor.

### 4.5 Interface tension and roughening of the 3D Ising model

The interface free energy is the overlap per site of the ± boundary fixed points — structurally native
here; the only TN work uses finite slabs (2601.07829). Moderate impact, high feasibility; a good first
GPU campaign after ice.

### 4.6 Benchmarks

The cubic dimer constant (Coulomb phase like ice: 0.44988452 at D = 4 from TN vs 0.4466–0.4479 MC and a
new rigorous upper bound 0.452130, 2607.28810) as the validation of any new engine; 3D Ising T_c against
HOTRG/MC.

## 5. A sequence

1. **Now (CPU, days).** Finish D = 9 ln F_I (checkpoints exist); fit ln F_I(ξ) and w_h(ξ) on D = 4–9;
   port the fold solver into the library (resumable); SU warm start + variational refinement as a
   library feature for the honeycomb cell.
2. **GPU readiness (1–2 weeks).** Profile `InfiniteCTM2D` on the local card at D = 4–6 (FP64 is slow
   there, but it shows where time goes and whether any path leaves the device); a QR/SI-style projector
   option; Z2 blocks for the ice networks; a Slurm job-array wrapper around `honeycomb_prod.jl` (one (D, χ,
   kind) per GPU). Ask the user before the first cluster run.
3. **First campaign (cluster).** Ice D = 8–14, χ ∈ {D², 1.5D², 2D²}, all three networks + S_sym;
   ξ-collapse; paper-grade error budget for S_h, S_c and the gap.
4. **Second campaign.** The Yang–Lee edge line with the fold solver at D = 4–12 (real signed
   formulation if it holds up), monomer–dimer validation; then 4.5 or 4.3.

## References (arXiv)

Boundary PEPS: 1805.10598, 2102.03143, 2110.12726, 1711.05881, 2210.09896, 2506.19339, 2511.22477,
2102.06715, 2405.01489, 2609.20020. Coarse graining: 1405.1179, 1906.02007, 1912.02414, 2602.08987,
2412.13758, 2507.21909. Scaling: 1803.08445, 1803.08566, 2607.15124, 2508.10822. BP/loops: 2008.04433,
2306.17837, 2305.01874, 2409.03108, 2510.02290, 2510.05647, 2512.10910, 2604.03228, 2604.24760. GPU and
symmetric CTMRG: 2505.00494, 2607.15158, 2502.10298, 2503.13900, 2410.11596, 2311.11894, 2405.12196,
2605.19960. Targets: 2505.06369, 2505.07655, 1206.0872, 2103.16224, 2510.05723, 2203.16651, 2311.13530,
2608.07613, 2404.13897, 2112.01824, 2012.15845, 2311.17994, 2601.06446, 2503.05144, 2601.07829, 1305.7028,
2607.28810, 2112.09536.
