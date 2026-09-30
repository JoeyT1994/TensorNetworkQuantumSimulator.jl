# Paper plan: scalable 3D tensor-network contraction

*Started 2026-09-30. Branch `FixesV2`.*

The paper has three layers:
- **Centre:** better estimators, large-scale GPUs and new algorithmic ideas raise what 3D tensor-network
  contraction can do.
- **One headline result:** the 3D three-state Potts first-order transition, converged in D.
- **Several smaller results:** Ising, Yang–Lee, and possibly ice.

Each item below is marked ✅ (have it), 🔄 (running), or ⬜ (to do).

## 1. Method

The setting is boundary PEPS for 3D classical partition functions, all in the thermodynamic limit:
- f(A) = ln κ⟨Ψ|T|Ψ⟩ − ln κ⟨Ψ|Ψ⟩;
- a 1×1 infinite 2D CTMRG for each network;
- the Kikuchi free energy.

Full detail: `docs/boundary_peps.md`.

**Better estimators** — ✅ all implemented.
- f from the Kikuchi/CVM formula.
- e and m read as impurities: e from the exact ∂site/∂β (Hellmann–Feynman at the optimum), no finite
  differences.
- The first-order transition from the crossing of two metastable continuations. There is no tunnelling
  and no finite-size scaling, and the metastable branches and spinodals come for free.
- ξ from the boundary transfer matrix.
- The Yang–Lee edge as the fold of a stationary bilinear state. Solved by Newton–Krylov continuation.

**Algorithmic ideas** — ✅ all measured (`boundary_peps.md`, the "optimiser" and "split pairs" sections).

| idea | what it buys | measured |
|---|---|---|
| split CTMRG pairs (matrix-free subspace SVD over quadrant factor lists) | per step χ³D⁴ instead of χ³D⁶ | 4.1× per step at D = 5, χ = 50 |
| norm-metric preconditioned L-BFGS, with adaptive CTM tolerance | fewer iterations to converge | 3× to convergence |
| full stack together | | 5.9× (1766 s vs 10415 s) |
| C4v-symmetric step (one pair and two blocks; the rest by relabelling) | 3–4× on the GPU | D = 6, χ = 72 at 0.86 s/step |
| Newton–Krylov for stationary non-variational states | the Yang–Lee edge at all | 5 min to 1.5 h per point at D = 4 (A6000) |
| embedding a smaller-D optimum into a larger D (with `miniter`) | a D ladder instead of cold starts | the first β of every scan |

**GPUs.**
- ✅ Speed-up over the CPU: 12–38× from D = 4 up.
- ✅ D = 8, χ = 128 runs at 25 s/step on one A6000.
- ⬜ An H100 scaling figure: s/step against D at χ = 3D² up to D ≈ 10–12. Run `examples/ice/honeycomb_profile.jl`
  or the Ising benchmark on a cluster node.
- ⬜ A memory figure.

## 2. The headline result: 3D three-state Potts

| D, χ | β_t | Q | m jump | status |
|---|---|---|---|---|
| TPVA (Gendiar & Nishino) | 0.5496 | 0.228 | | literature |
| 3, 27 | 0.550408 | 0.1891 | 0.415 | ✅ |
| 4, 48 | 0.550506(2) | 0.168(1) | 0.393 | ✅ |
| 4, 64 | | | | 🔄 A6000 (the χ check) |
| 5, 75 | | | | ⬜ cluster: `examples/potts_cluster/` |
| 5, 100 | | | | ⬜ cluster, second wave |
| 6, 108 | | | | ⬜ cluster, second wave |
| Monte Carlo (Janke & Villanova 1997) | 0.550565(10) | 0.16160(47) | | literature |

**What it takes to count as converged:**
- ⬜ D = 3, 4, 5, 6 and an extrapolation in D. The error in Q went 17 % → 3.8 % from D = 3 to D = 4.
  Extrapolate against 1/D or against ξ, and show both.
- ⬜ χ convergence at the largest D.
- ⬜ The ordered branch's noise floor. Its points stall at |g| ≈ 4e-6 and hit the iteration cap near the
  spinodal, so a better estimator or solver is needed there. This fits the methods story.
- Possible extra observables, to go beyond "matches Monte Carlo":
  - ξ in both phases at β_t;
  - the spinodal locations, which Monte Carlo cannot see;
  - the order–disorder interface tension, via the fixed-point overlap idea in `campaign_3d.md`.

**Figures:**
- f_ord − f_dis against β for D = 3, 4, 5, and the crossing;
- e against β for both branches, with the latent heat;
- β_t and Q against 1/D, extrapolated.

**Data:** the scratchpad CSVs of 2026-09-28 to 30 (to be moved under `examples/potts_data/`), and
`runs/` from the cluster.

## 3. Supporting results

**3D Ising.**
- ✅ m(β) against Monte Carlo at D = 2, 3, 4 (`examples/ising3d_boundary_peps_benchmark.jl`).
- ✅ D = 4 halves the error of D = 3 near β_c, e.g. 4.3e-4 at β = 0.2225.
- ✅ The pseudo-T_c agrees with Vanderstraeten et al.'s D = 2 value.
- ⬜ D = 5–6 on the cluster, so that the figure shows convergence rather than two points.

**The Yang–Lee edge in 3D, in an imaginary field** — a showcase of the method (Monte Carlo cannot reach it).
- ✅ Edge maps at β = 0.20 and 0.21, D = 2–4. The fold converges in χ; checked at χ = 64.
- ✅ |z_c| = 2.45–2.49, against the functional-RG value 2.43(4).
- The open issue: the fold stalls from D = 3 to D = 4 while ξ grows. So the finite-ξ extrapolation does not
  apply, and the spread in |z_c| is not an error bar.
- Present it as the first lattice-TN map of the edge, not as a precision number unless D = 5 settles the
  question.

**Ice Ih/Ic (optional).** The kit exists (`examples/ice/cluster/`), but its environment is stale, as of
2026-09-30. Include only if the campaign runs.

## 4. Order of work

1. 🔄 D = 4, χ = 64 Potts check (A6000). Its saved states can seed D = 5.
2. ⬜ Cluster: the Potts smoke test, then D = 5. The user submits (`examples/potts_cluster/README.md`).
3. ⬜ The H100 scaling and memory benchmark: one short cluster job.
4. ⬜ The ordered-branch noise floor (local).
5. ⬜ Cluster: Potts at D = 6, and D = 5 at χ = 100.
6. ⬜ Ising at D = 5–6 (cluster), and the figures.
