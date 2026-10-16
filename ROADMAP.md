# TNQS roadmap and open problem (handoff, 2026-10-16)

This note is for whoever picks up the next phase of TensorNetworkQuantumSimulator (TNQS). It records
the maintainer's decisions, the target scope of the next release line, and the problem currently
blocking it: **which tensor backend the library should be built on**.

## Non-negotiable principles

1. **TNQS is public, open source, and depends only on packages registered in the Julia General
   registry.** No dependency may live only in a private or secondary registry.
2. **The next release line ("the trunk") is symmetries + CTMRG + kernel hooks. Nothing else.**
   Research code stays out of it (see [Scope](#trunk-scope)).
3. Research code will later move into a separate package that depends on the trunk, rather than living
   on a long-lived branch. That is deferred; do not start it unprompted.

## Where the code is today

| Branch | What it is | Relation to `main` |
|---|---|---|
| `main` (v0.4.6, released) | ITensors 0.9 backend, BP / boundary MPS / loop corrections, dense only | — |
| `Improvements` (this branch) | `main` + small cleanups | 1 commit ahead |
| `FixesV2` | The real development line: tensor backend seam, symmetric + fermionic tensors, finite 2D CTMRG, fused BP kernels, plus 3D research code | 234 commits ahead, +23.6k lines |
| `FixesV2DMRG` | `FixesV2` + generating-function DMRG (research) | 204 ahead |
| `Fermions` | Older fermion work, partly superseded by `FixesV2` | 86 ahead, 19 behind |
| `Research` | Stale (May 2025) | 376 behind |

### The backend seam (already built, on `FixesV2`)

`src/tensor_interface.jl` defines module `TensorInterface`: every tensor verb the package uses
(index construction and queries, contraction, factorisations, `dag`, `combiner`, `directsum`, ...).
Only the backend file references a tensor library. On `FixesV2`, `src/Tensors/ITensorBackend.jl`
(1.6k lines) is the **only** file that references ITensorBase, GradedArrays, TensorAlgebra or
MatrixAlgebraKit; the 2.8k-line CTM cache, BP, boundary MPS, etc. go through the seam. Read the
header of `tensor_interface.jl` before touching anything. Its rule 5 ("two adjoints": `dag(t)` is the
network bra, `dag(t, cod)` is the adjoint of `t` as a map) is what makes fermions work. Breaking it
silently breaks projector identities.

The seam has already been proven to swap backends: commit `0aa8795` (2026-09-10) replaced a working
TensorKit-based backend with ITensorBase behind an essentially unchanged interface.

## Trunk scope

Cut the trunk from `FixesV2`.

**In**
- `src/tensor_interface.jl` + one General-registered backend (see the open problem)
- Symmetric tensors: dense, Z2, U1, and the fermionic sectors fZ2, fU1, fU1xU1 that the backend already supports
- `src/MessagePassing/ctmenvironmentcache.jl`: finite 2D CTMRG (`:cut` and `:cycle` projectors)
- `src/kernel_hooks.jl` and the fused BP message/closure kernels (peak-memory reduction), as
  optional per-backend overrides with the generic contraction as fallback
- Core: BP, boundary MPS, loop corrections, gates/simple update, expect, sampling, entanglement, truncate
- `src/precompile.jl` (PrecompileTools workload)

**Out** (research, stays off the trunk)
- `ctm2dinfinite.jl` (infinite 2D CTMRG)
- `boundarypeps3d.jl`, `ctm3denvironmentcache.jl`, `ctm3dinfinite.jl` (3D work: Potts, Ising, ice, Yang–Lee)
- DMRG (`FixesV2DMRG`)
- The ITensorBase backend (cannot be a trunk dependency, see below)

**Later directions** (not the trunk's first cut, but the architecture should not block them)
- CTMRG on **any planar graph**, with a planarity test + embedding step so vertex names become pure
  labels. Today geometry is read off tuple vertex names (e.g. `boundarymps_partitioning` uses
  `first.(vs)` as the row). No General package does planarity testing / combinatorial embedding as far
  as we know, so this likely means implementing Boyer–Myrvold or left–right planarity. The research
  question is generalising corner/sweep structure to irregular faces, not the embedding itself.
- **Simple, serialisable types for distributed BP**: plain vertex→tensor and edge→message dictionaries
  plus a partition, with message exchange between partitions. Fold in `origin/DistributedBP` and
  `origin/Jack_Distributed_BP` rather than starting from scratch.

## The open problem: which backend?

### Facts

- **ITensorBase, GradedArrays, TensorAlgebra and BlockSparseArrays are not in General.** They are only
  in `ITensor/ITensorRegistry`, and their author has no near-term plan to register them in General.
  Their API also churns (ITensorBase 0.13 → 0.14 needed a hand port on 2026-09-16), and development
  pace is outside our control. A General-registered TNQS **cannot depend on them**, not even as a weak
  dependency.
- **TensorKit, TensorKitSectors, TensorOperations and MatrixAlgebraKit are in General.** TensorKit has
  first-class abelian, non-abelian and fermionic sectors. Its codomain/domain split makes rule 5's map
  adjoint natural (the `TensorMap` adjoint) rather than a parity-twist workaround.
- The maintainer **left TensorKit on 2026-09-10 because of bad precompile/compile latency** (and to work
  on the ITensorBase side).

### Measured latency, like for like

Two snapshots on either side of the switch, with the same tests (they differ by ~16 lines) and no
PrecompileTools workload in either. Julia 1.12.7, ccqlin014 workstation, loaded by other jobs (load ~8),
each test file in a fresh process.

| | TensorKit (`0aa8795^`) | ITensorBase (`0aa8795`) |
|---|---|---|
| Package precompile after a source edit | 8.4 s | **3.4 s** |
| `using TensorNetworkQuantumSimulator` | ~9–10 s | **~6.5–7.5 s** |
| `test_tensors.jl`, first run (compile + compute) | **> 1050 s, still running when this was written** | 427 s |
| `test_tensors.jl`, second run (compute only) | — | 16 s |
| `test_beliefpropagation.jl`, first / second run | not reached | 82 s / 0.4 s |
| `test_apply.jl`, first / second run | not reached | 88 s / 0.8 s |

So first-use compilation dominates everything (≈96% of the first run), and TensorKit is
substantially worse than ITensorBase on it. The maintainer's memory was right.

Caveat: the test suite exercises dense **plus five sector types**, which is close to the worst case
for a library that specialises on sector type. A user script typically uses one sector type. For
reference, adding the PrecompileTools workload on the ITensorBase backend took a fresh-session small
workflow from 85 s to 0.4 s (dense) and 68 s to 9 s (graded), at the cost of package precompile
growing from ~20 s to ~150 s per source edit (disable during development with the
`precompile_workload` preference; never commit `LocalPreferences.toml`).

### Options

1. **TensorKit, with its latency reduced (recommended to investigate first).** Restore the TensorKit
   backend from `0aa8795^` (`src/Tensors/Tensors.jl`, `src/Tensors/gradedtensor.jl`, ~2.4k lines;
   dense used plain arrays + TensorOperations, only the graded path used TensorKit `TensorMap`s), then
   attack the compile cost.
2. **Keep ITensorBase and give up General registration.** This violates principle 1. Listed only for
   completeness.
3. **Own a lighter graded layer** on General packages (dense arrays + TensorOperations + our own
   block-sparse abelian/fermionic tensor). Large; essentially rebuilding GradedArrays. Consider only if
   option 1 fails.

### Hypothesis to test next (unverified)

TensorKit types carry the sector/space type **and the codomain and domain leg counts** as type
parameters (`TensorMap{T, S, N₁, N₂, A}`), so Julia compiles a fresh copy of each function for every
combination. CTM and BP generate many different leg splits. ITensorBase's named tensors carry far
fewer type parameters. If that is the cause, the fix may lie in **our backend wrapper**, not in
TensorKit. For example, the wrapper could hide the leg counts behind a type that does not specialise
on them, normalise to a few canonical splits before calling TensorKit, or use `@nospecialize` /
function barriers on the bookkeeping paths. Concretely:

1. On the TensorKit snapshot, profile compilation of one test file (`SnoopCompile`:
   `@snoop_inference` then `flatten`/`accumulate_by_source`, or `--trace-compile`) and count
   distinct `TensorMap` method specialisations by `(N₁, N₂)` and sector.
2. Measure a **realistic single-sector script** (e.g. U1 BP + simple update + CTM on a 4×4 lattice),
   not the test suite, on both backends, with and without a PrecompileTools workload covering
   dense + U1 + fZ2.
3. Decide: if TensorKit with workload + wrapper fixes is acceptable for that script, go with option 1.

### Catch-up needed if TensorKit is restored

Nine commits on `FixesV2` touched the seam, backend or kernel hooks after the switch; `tensor_interface.jl`
changed by 26 lines. The TensorKit backend must add:
- New seam verbs: `pad_index` (sector-stable zero-padding on graded bonds, used by CTM `:cycle`),
  `scale!` (in-place scaling of block-sparse storage), `dag(t, cod)` (map adjoint)
- Equivalents of the fixes and kernels made on the ITensorBase side since then: the fused dense BP kernel
  (`d41fa35`), simple-update layout and in-place work (`bedf61b`, `597affd`), Hermitian `eigen` fixes
  including the fermionic sign (`48a0dd7`, `4a64006`), dual bond copies for graded random states,
  dense GPU paths (`2dcd1e2`)
- The 2026-09-16 dependency ports (NamedGraphs 0.14, etc., `b9700d4`)

## Gotchas for whoever runs this

- **CUDA is a hard `[deps]` entry on `FixesV2`** (and was in the TensorKit era), although `src` never
  imports it. On the trunk it must be a weak dependency with a package extension, or every user
  installs and precompiles CUDA.jl.
- **`Manifest.toml` is untracked** and is whatever the last checkout resolved. Switching between `main`
  and `FixesV2` leaves a Manifest for the wrong backend, and `using`/`Pkg.test()` fail with "Package
  ITensors not found". Re-resolve (`Pkg.resolve()`), or test from a scratch environment that
  `Pkg.develop`s the checkout.
- Use the juliaup Julia 1.12.7 binary with `--startup-file=no` and an absolute `--project` path.
  (1.12.1 showed intermittent GC heap corruption on this package; 1.11 cannot precompile the Manifest.)
- Tests are compilation-bound, not compute-bound. Shrinking test problems buys nothing.
- Fermionic conventions and the traps hit so far are documented on `FixesV2` in
  `docs/ctmrg_status.md` ("Backend switch") and in the header of `src/tensor_interface.jl`.

## Reproducing the latency table

```bash
# usage: run.sh <name> <rev>   e.g. run.sh tensorkit '0aa8795^'  and  run.sh itensorbase 0aa8795
J=/path/to/julia-1.12.7/bin/julia; REPO=/path/to/TensorNetworkQuantumSimulator
name=$1; rev=$2; src=$PWD/$name/TNQS; env=$PWD/$name/env
mkdir -p "$src" "$env"; git -C "$REPO" archive "$rev" | tar -x -C "$src"
$J --startup-file=no --project="$env" -e "using Pkg; Pkg.develop(path=\"$src\");
  Pkg.add([\"Test\",\"Statistics\",\"Random\",\"LinearAlgebra\",\"Logging\",\"Graphs\",\"NamedGraphs\",\"Dictionaries\"]);
  Pkg.precompile(); touch(\"$src/src/TensorNetworkQuantumSimulator.jl\");
  println(\"precompile after edit: \", @elapsed Pkg.precompile())"
for f in test_tensors.jl test_beliefpropagation.jl test_apply.jl test_ctmenvironment.jl; do
  $J --startup-file=no --project="$env" -e "
    println(\"$f using: \", @elapsed using TensorNetworkQuantumSimulator)
    println(\"$f first: \", @elapsed include(\"$src/test/$f\"))
    println(\"$f second: \", @elapsed include(\"$src/test/$f\"))"
done
```
