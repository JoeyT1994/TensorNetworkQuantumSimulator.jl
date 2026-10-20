# TNQS roadmap and open problem (handoff 2026-10-16, updated 2026-10-17)

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
| `Improvements` (this branch) | `main` made backend-agnostic: the `TensorInterface` seam on an ITensors 0.9 dense backend, an in-house operator library, the fused BP kernel hooks and finite 2D CTMRG (see [Status](#status-of-improvements-2026-10-17)) | ahead of `main`, not yet merged |
| `FixesV2` | The research line: an ITensorBase backend, symmetric + fermionic tensors, finite 2D CTMRG, fused BP kernels, plus 3D research code | 234 commits ahead, +23.6k lines |
| `FixesV2DMRG` | `FixesV2` + generating-function DMRG (research) | 204 ahead |
| `Fermions` | Older fermion work, partly superseded by `FixesV2` | 85 ahead, 19 behind |
| `Research` | Stale (May 2025) | 376 behind |

## Status of `Improvements` (2026-10-17)

The trunk is being built on this branch rather than cut from `FixesV2`: `main`'s code is the model
(its structure and style are kept, and the seam mostly replaced `using ITensors: …` imports), and
`FixesV2` was taken as authoritative only for the finite 2D CTMRG engine, the kernel hooks / fused
BP kernels, and simple fixes. Commits, in order:

1. **Pin the BP message phase** (`048883e`). A fix to `c45e1f1`, which normalised BP messages by
   their 2-norm: that leaves each message's phase free, and on loopy graphs the phases add up around
   the loops and grow exponentially (~5× per sweep on a complex hexagonal state). `symmetric_gauge!`
   then took square roots of negative eigenvalues — `test_truncate.jl` failed at `7300229`.
   `normalize_message` keeps the 2-norm and divides out the phase of the element sum.
2. **The seam** (`1e889a0`) — `src/tensor_interface.jl`, `src/Backends/`, `src/operators.jl` (details below).
3. **Kernel hooks** (`06763aa`) — the fused dense BP message/closure kernel from `FixesV2` `d41fa35`.
4. **Finite 2D CTMRG** (`7b3905e`) — `src/MessagePassing/ctmenvironmentcache.jl` from `FixesV2`, `:cut` and
   `:cycle`, with `expect`/`rdm` support for `alg = "ctmrg"`.

### The backend seam (on this branch)

`src/tensor_interface.jl` defines module `TensorInterface`: every tensor verb the package uses, each
a generic function with a written contract in that file. **Read its header before touching anything.**
Its rule 5 ("two adjoints": `dag(t)` is the network bra, `dag(t, cod)` the adjoint of `t` as a map)
is what makes fermions work on a graded backend; breaking it silently breaks projector identities.

Layout:

- `src/tensor_interface.jl` — the verbs, their contracts, the seam-owned `Algorithm` and
  `TruncationStrategy` types, and three optional **hooks** whose fallback is `nothing`
  (`fused_norm_message`, `fused_norm_scalar`, `backend_op`).
- `src/Backends/Backends.jl` — selects the backend and binds the two type names the package
  annotates with (`Tensor`, `Index`). Swapping backends means changing this file and `Compat.jl`.
- `src/Backends/ITensorsBackend.jl` (+ `ITensorsKernels.jl`) — ITensors 0.9, dense only. Every verb
  forwards to its ITensors namesake; the seam follows ITensors conventions, so most are one line.
- `src/Backends/ITensorsCompat.jl` — forwards ITensors' own functions to the package for package
  objects (`ITensors.contract(ψ; alg)`, `datatype(ψ)` imported from ITensors, `ITensors.op("Rxxyy", …)`),
  so code written against the ITensors-backed releases keeps working. Delete it with the backend.
- `src/operators.jl` — the operator and state library as plain matrices (`register_op!`, `op`,
  `state`); the backend contributes only `from_array(A, outs, ins)`. Names the registry lacks fall
  through to the backend's library (`backend_op`; on this backend, `ITensors.op`).
- `test/test_seam.jl` fails if any file under `src/` outside `src/Backends/` references ITensors,
  NDTensors, ITensorBase, GradedArrays, TensorAlgebra, TensorKit*, MatrixAlgebraKit or the backend
  module in code (comments may name them).

**What a new backend must implement:** the verbs listed in the header of `tensor_interface.jl`
(index construction/queries/replacement, construction, contraction, diagonal ops, factorisations
incl. the LinearAlgebra generics of rule 3, storage queries), a `Tensor`/`Index` binding in
`Backends.jl`, and optionally the hooks. On this branch:

| group | verbs |
|---|---|
| index construction | `new_index(d; tags)`, `new_index(ref, d; tags)`, `pad_index`, `sim`, `dag`, `prime`, `noprime` |
| index queries | `inds`, `dim`, `plev`, `tags`, `commonind(s)`, `uniqueinds`, `unioninds`, `noncommonind(s)`, `hascommoninds`, `isgraded`, `isdual`, `space` |
| index replacement | `replaceind(s)` |
| construction | `from_array`, `random_tensor`, `onehot`, `delta`, `dense`, `combiner`, `combinedind`, `directsum`, `charge_sectors` |
| contraction | `contract(ts; sequence)`, `Base.:*`, `scalar`, `apply` |
| diagonal ops | `map_diag`, `map_diag!` |
| factorisations | `qr`, `svd`, `eigen`, `factorize` (LinearAlgebra), `factorize_svd`, `truncated_svd`, `gram` |
| storage / type | `datatype`, `scalartype`, `array`, `data`, `scale!` |
| hooks (optional) | `fused_norm_message`, `fused_norm_scalar`, `backend_op` |

Contract decisions made while building it (each is written in the seam's header):

- **Factorisation results are destructured, never indexed or `last`ed.** ITensors returns extra
  values (`qr` → `(Q, R, q)`, `svd` → a `TruncSVD` that iterates but cannot be indexed), so
  `last(qr(…))` silently returned an index. Two such sites in the CTM file were rewritten.
- **Truncation by strategy is its own verb, `truncated_svd(t, linds; trunc)`**, with the rule
  (rank, relative cutoff, degeneracy back-off) a seam-owned `TruncationStrategy` and `kept_count`
  the rule on one sorted spectrum. `FixesV2` passed `trunc` as an `svd` keyword; ITensors' `svd`
  cannot take one without overwriting its method.
- **A combiner may be structured.** ITensors combiners only fuse or unfuse all their legs at once;
  `dense(C)` is the explicit isometry, used where the CTM treats it as an ordinary tensor.
- **`Tensor` and `Index` are type names bound by the backend**, not seam-owned abstract types
  (ITensors' types cannot subtype ours). The package uses them only in annotations and containers
  and never calls their constructors (`new_index`, `from_array` instead).
- **Hooks return `nothing` to decline**, and the caller takes the generic path. The fused message
  kernel is called unnormalised so both paths share `normalize_message`.
- **The `FixesV2` seam leaked**: its core called ~23 backend-only `Tensors.*` functions. For the code
  ported here (CTM, kernels) these became seam verbs with dense-trivial contracts (`isgraded`,
  `isdual`, `space`, `pad_index`, `charge_sectors`, `new_index(ref, d)`), an existing verb (a raw
  `Index(rand(UInt64), space, …)` → `sim(w; tags)`), or were dropped with the graded-only code.
- `Random` became a package dependency (`random_tensor(rng, …)`, the CTM's seeded Krylov starts).

Latency, `7300229` against `7b3905e` (Windows laptop, Julia 1.12.7, each test file in a fresh
process, two alternating runs — they agreed to 1%):

| | `7300229` | `7b3905e` |
|---|---|---|
| `using TensorNetworkQuantumSimulator` | 4.65 s | **3.88 s** |
| package precompile (TNQS only) | ~7.2 s | ~7.5–7.8 s |
| `test_beliefpropagation.jl`, first / second run | 64.2 s / 0.35 s | 75.1 s / **0.28 s** |
| `test_apply.jl`, first / second run | 83.3 s / 0.72 s | 80.5 s / 0.69 s |

No second-run regression. The BP first run is 17% slower: it now compiles the fused kernel too.
The whole suite (incl. CTMRG, 7.7 min, and the examples, ~5.5 min) is ~30 min on this machine.

Out for now (on `FixesV2`, not ported): the graded/fermionic machinery and its constructors
(`graded_tensornetworkstate`, charged boundary-MPS initialisation), `docs/ctmrg_status.md` and
`docs/finite_ctmrg_design.md` (the CTM file's comments cite them; read them on `origin/FixesV2`),
`src/precompile.jl`, the GPU paths.

## Trunk scope

Being built on `Improvements` (see [Status](#status-of-improvements-2026-10-17)); `FixesV2` supplies
the CTMRG engine and kernels. Done: the seam + an interim dense backend, finite 2D CTMRG, kernel
hooks, core. Still to come: a General-registered **graded** backend (the open problem below) and with
it the symmetric tensors, and `src/precompile.jl`.

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
- CTMRG on **any planar graph**: done on `Improvements`. Planar embeddings from vertex positions
  (`src/Planar/`) and `PlanarCTMRGCache` (`alg = "planarctmrg"`, see `PLANAR_CTM.md`), exact at
  lossless χ on square, honeycomb, triangular, kagome and Penrose lattices. Still to do: compare it
  with the grid engine (`CTMEnvironmentCache`) and retire one.
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
| `test_tensors.jl`, first run (compile + compute) | **1442 s** | 427 s |
| `test_tensors.jl`, second run (compute only) | **13.6 s** | 16.0 s |
| `test_beliefpropagation.jl`, first / second run | 99 s / 0.4 s | 82 s / 0.4 s |
| `test_apply.jl`, first / second run | 115 s / 0.8 s | 88 s / 0.8 s |
| `test_ctmenvironment.jl`, first / second run | **1492 s** / 64 s | 673 s / 76 s |

So first-use compilation dominates everything (≈90–99% of a first run), and the maintainer's memory
was right. The problem is purely compilation: once compiled, TensorKit is as fast or faster
(13.6 s vs 16.0 s, 64 s vs 76 s).

The gap is not uniform. On the BP and gate tests TensorKit compiles only ~1.2–1.3× longer, but on
`test_tensors.jl` (3.4×) and `test_ctmenvironment.jl` (2.2×) it is far worse. Those two exercise the
graded (symmetric/fermionic) path and many different leg splits most heavily. This is consistent with
the specialisation hypothesis below, but does not prove it: check what each test file actually runs
before relying on it.

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

### What a restored (or new) backend must do

A new backend now targets **this branch's** seam: the verb table in
[the backend seam](#the-backend-seam-on-this-branch), next to `ITensorsBackend.jl` in `src/Backends/`,
selected in `Backends.jl`. The ITensors backend is the dense reference implementation, and the test
suite (incl. `test_operators.jl`, which pins every operator matrix, and `test_ctmenvironment.jl`) is
the acceptance check. The old TensorKit backend (`0aa8795^`) was written against an earlier version
of the `FixesV2` seam; beyond the verb table, a graded backend needs equivalents of the graded fixes
made on the ITensorBase side since then — Hermitian `eigen` incl. the fermionic sign (`48a0dd7`,
`4a64006`), dual bond copies for graded random states — and the graded constructors and the CTM's
graded branches, which were not ported here (see "Out for now" above). The fused BP kernel is
optional (a hook); simple-update in-place work (`bedf61b`, `597affd`) and the dense GPU paths
(`2dcd1e2`) were not ported either.

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
