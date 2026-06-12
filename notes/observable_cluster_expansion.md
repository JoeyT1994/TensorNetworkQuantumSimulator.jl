# Design note: observable loop cluster expansion

Status: **design / not yet implemented.** Branch `ObservableClusterExpansion`
(forked from `Fermions`, so it carries the existing bosonic + fermionic iTNS code).

## Goal

Implement the **loop cluster expansion for local observables** — the product /
linked-cluster form of Gray, Park, Evenbly, Pancotti, Kjønstad & Chan,
*"Tensor Network Loop Cluster Expansions for Quantum Many-Body Problems"*
(arXiv:2510.05647; the analysis follow-up to PRB 112, 174310; available in `quimb`).

For a local observable `Ô`,

```
⟨Ô⟩ = ⟨Ψ|Ô|Ψ⟩ / ⟨Ψ|Ψ⟩  ≈  ∏_r ( ⟨Ψ|Ô|Ψ⟩_r / ⟨Ψ|Ψ⟩_r )^{c(r)}   (Eq. 7)
```

where each `⟨·⟩_r` is the region-`r` tensors contracted with **converged BP
messages capping the boundary `∂r`** (no antiprojectors), and `c(r)` is an
inclusion–exclusion counting number over clusters and their intersections.

This **fully replaces the (currently non-existent) observable loop-correction
path**. It does **not** touch the antiprojector partition-function path.

## Key facts about the current codebase

1. **`expect_loopcorrect` is a dangling export** — exported in
   `src/TensorNetworkQuantumSimulator.jl` but **defined nowhere**. There is no
   working observable loop-correction path today: `expect` only has
   `Algorithm"bp"` and `Algorithm"boundarymps"` methods, so
   `expect(ψ, obs; alg="loopcorrections")` does not dispatch. We are therefore
   *filling a gap*, not replacing working code.

2. **The antiprojector loop machinery serves only the partition function.**
   `weight` / `_fermionic_loop_weight` / `sim_edgeinduced_subgraph` in
   `src/MessagePassing/loopcorrection.jl` are consumed only by
   `loopcorrected_partitionfunction`, which backs `norm_sqr` / `inner` at
   `alg="loopcorrections"`. This stays as-is. The cluster expansion never goes
   near it. ⇒ the two methods coexist, each owning a different quantity
   (Z / norm = antiprojector loop series; observables = cluster cumulant).

3. **The per-cluster ratio already exists.** `expect(Algorithm"bp", cache, obs)`
   (`src/expect.jl:65–92`) is exactly the *single-cluster* approximation:
   ```
   steiner_vs  = steiner_tree region of the observable vertices
   incoming_ms = incoming_messages(cache, steiner_vs)          # messages on ∂region
   denom       = scalar(contract([ norm_factors(net, steiner_vs);             incoming_ms ]))
   numer       = scalar(contract([ norm_factors(net, steiner_vs; op_strings); incoming_ms ]))
   ⟨O⟩ ≈ coeff * numer / denom
   ```
   The cluster-expansion ratio `O_r` is the identical computation with the
   region set to a cluster `r` instead of the Steiner tree. `norm_factors`
   already builds the doubled network, inserts the operator, and **dispatches
   fermionically**. So the tensor side is reuse, not new code.

4. `vertex_scalar(bpc, v)` (`abstractbeliefpropagationcache.jl:22`) is the same
   pattern for a single vertex, and `incoming_messages(bpc, vs)` (line 150)
   returns BP messages on `∂(vs)` for *any* vertex set — i.e. a region
   contraction is just these primitives over a set.

## Module decomposition

`§4` (contraction) is reuse; the new code is the combinatorial core
(`§2 / §3 / §5`) plus the iTNS-only unfolding (`§1`). The combinatorial core is
**fermion-agnostic** — it never touches a tensor.

### §0. Refactor (first commit, behavior-preserving)
Extract `src/expect.jl:73–91` into
```
_region_ratio(cache, region, obs_vs, op_strings) -> numer/denom
```
Existing `expect(Algorithm"bp", …)` becomes `_region_ratio(cache, steiner_vs, …)`.
Confirm `expect` output is unchanged.

### §1. Unfolding — `unfold_to_patch(itns, C) -> (BeliefPropagationCache, target_verts)`
**iTNS-specific; the only iTNS-specific piece.** Materialize a finite patch of
the physical loopy lattice (hex z=3, …) large enough to host every cluster of
≤ C sites around the target. Populate factors and messages by translation
invariance (every A-copy ← `itns[:A]`, every B-copy ← `itns[:B]`, every edge
message ← a `sim`-relabeled copy of the converged unit-cell message). Simpler
than the antiprojector W6 idea: no bond-vertices / antiprojector index surgery
needed. Leans on: `infinite_*` constructors, `message`/`messages`,
`setmessage!`, `bp_factors`, index `sim` hygiene (cf. `sim_edgeinduced_subgraph`).
Requires the caller to supply the lattice geometry (the unit cell alone doesn't
know it tiles hex vs. square).

### §2. Cluster enumeration — `loop_clusters(g, target_verts, C)`
All loopy ("no-leaf") edge-induced subgraphs of ≤ C sites **containing the
target support**. Extend `edgeinduced_subgraphs_no_leaves` (already imported;
used by `loopcorrected_partitionfunction`) with a containment filter, seeded at
the target. If using norm-network messages (see API note), also include the
non-loop **"anomalous"** clusters touching the observable site (relax the
no-leaf filter at the target only).

### §3. Counting numbers — `cluster_counting_numbers(clusters)`
**The genuinely new combinatorial core (Algorithm 1, lines 4–13). Pure graph
code, ported from the paper / quimb. This is the bulk of the effort.**
- Close the cluster set under intersection (add `r_a ∩ r_b` until unchanged).
- `c(r) = 1 - Σ_{a ⊋ r} c(a)`, evaluated in decreasing `|r|`.
- Tree-reduction (lines 8–12): map each region to its largest equivalent loop
  `r'` (tree-like parts contract to the BP fixed point), fold `c_r` into `c_{r'}`,
  drop `r`.
Kept **separate** from the antiprojector partition-function accounting (different
expansion).

### §4. Cluster ratio — reuse `_region_ratio` from §0
`O_r = _region_ratio(cache, r, obs_vs, op_strings)`. Fermions free (goes through
`norm_factors` + `contract`).

### §5. Assembly — `expect_clusterexpand`
`⟨Ô⟩ ≈ ∏_{r : c(r)≠0} O_r^{c(r)}` (Eq. 7).

### §6. Validation
- **Single-cluster limit:** C small enough that the only cluster is the Steiner
  tree ⇒ must reproduce ordinary `expect` exactly (tests §2/§3/§5 with the
  contraction held fixed).
- **iTNS unfolding in isolation:** iTNS-unfolded result vs. running the *same*
  finite `expect_clusterexpand` on a genuine `named_hexagonal_lattice_graph`
  patch of the same state.

## Finite vs iTNS

The finite `TensorNetworkState` case needs **no unfolding** — the cache graph
*is* the physical lattice. So `expect_clusterexpand` for finite states is the
**primary, general implementation**, and iTNS is a thin front-end:

```
iTNS_expect_clusterexpand(itns, op, loc; C)  =
    unfold_to_patch(itns, C)        # §1, the only iTNS-specific code
    → resolve op/loc on the patch   # reuse _resolve_gate / _obsverts
    → expect_clusterexpand(patch, obs; C)   # shared finite path
```

`§1` is the only iTNS-specific code; `§2–§5` are shared verbatim.

## API

- New `expect_clusterexpand(ψ::Union{TensorNetworkState, BeliefPropagationCache},
  obs; max_configuration_size, alg="bp", msg=:norm, kwargs...)`.
- Give the dead `expect_loopcorrect` export a real definition (thin wrapper).
- Add a `expect(Algorithm"loopcorrections", cache, obs; max_configuration_size)`
  method so `alg="loopcorrections"` works for `expect` uniformly with
  `norm_sqr` / `inner`.
- **Kwarg name:** reuse `max_configuration_size` (same *kind* of knob as the
  antiprojector path — max region size) for API uniformity across
  `norm_sqr`/`inner`/`expect`, even though the internals differ. (Alternative:
  a distinct `C` to signal a different method. Leaning toward reuse.)
- **Message choice (`msg`):** norm-network messages (`:norm`) are convenient for
  many observables without re-running BP, but then the BP equation is not
  satisfied at the observable sites, so the "anomalous" non-loop clusters of §2
  must be included. Operator-network messages avoid that but cost a BP solve per
  observable.

## Build order (bosons first)

The branch is **purely additive for observables** (only the antiprojector
partition-function path must stay un-regressed, and the cluster code never
touches it). The combinatorial core is fermion-agnostic, so bosons-first
validates the risky new code in the simpler regime; fermions then change only
the `contract` dispatch inside `_region_ratio`.

1. `§0` refactor `_region_ratio` out of `expect`; confirm `expect` unchanged.
2. Bosonic `expect_clusterexpand` for finite states (`§2/§3/§5`); wire
   `expect_loopcorrect` + `Algorithm"loopcorrections"` dispatch; validate against
   `expect` (single-cluster limit).
3. Bosonic iTNS unfolding (`§1`); validate against a finite
   `named_hexagonal_lattice_graph` patch.
4. *(follow-up)* fermions — expected near-zero contraction changes (confirm the
   cluster ratios go through the fermionic `contract`; check the cyclic-loop
   sign on the unfolded patch, already exercised by validated fermionic loops on
   grids/hex).
