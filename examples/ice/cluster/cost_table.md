# Cost and memory of the honeycomb ice networks

*Measured 2026-09-27 on the local machine (i9-9900K; RTX 3070, FP64 ≈ 1/64 rate, 8 GB) with the fixes of
`honeycomb_prod.jl` (subspace oversampling ⌈1.3χ⌉, `convergence = :lnkappa`); H100/H200 columns are
EXTRAPOLATIONS (±3×) — calibrate them with the smoke job and `examples/ice/honeycomb_profile.jl`.*

Per 2D-CTM iteration, warm. r is the raw bond of a network per direction; ~20–35 iterations converge a
network from the seed at χ = D² (fewer warm), and the cost scales as ~χ³r² — ~D¹⁰ at χ = D².

| network | r | memory term (doubles) |
|---|---|---|
| ⟨ψ\|ψ⟩, ⟨Iψ\|ψ⟩ | D² | 2.3 χ²r² (subspace block of a pair) |
| ⟨Iψ\|M\|ψ⟩ (sandwich) | 2D² | 2.3 χ²r² |
| ⟨Mψ\|Mψ⟩ (residual) | 4D² | 2.3 χ²r² |

Working set ≈ 3× the memory term (measured: the D = 7 sandwich fits in 1.5 GiB, D = 8 in 5 GiB).

## Seconds per iteration

| D, χ | network | CPU (8 cores) | RTX 3070 | H100 (extrap.) |
|---|---|---|---|---|
| 5, 25 | norm / sand | 0.3 / 1.4 | 0.3 / 0.6 | overhead-bound, ~0.2 |
| 6, 36 | norm / sand | 2–4 / 10–25 | 0.6 / 2.3 | ~0.3 / ~0.3 |
| 7, 49 | norm / sand | ~6 / 50–100 | 5.9 / 6.8 | ~0.4 / ~0.4 |
| 8, 64 | norm / sand | ~30 / 300–470* | — / 25.5 | ~0.5 / ~0.6 |
| 9, 81 | norm | 40–56 | | ~1 / ~1.5 (sand) |
| 10, 100 | | | | ~1.5 / ~3 |
| 12, 144 | | | | ~6 / ~15 |

\* before the oversampling fix (pairs bailing out to the dense route).

The GPU's host overhead (~8 000 small copies, ~13 000 kernel launches, small cuSOLVER SVDs) is ~0.3 s per
iteration at D = 6 and dominates on an H100 below D ≈ 10.

## Memory (sandwich; ×¼ for norm/inv, ×4 for ⟨Mψ|Mψ⟩), working set ≈ 3 × 2.3 χ²r² × 8 B

| D | χ = D² | χ = 2D² (variational) | fits |
|---|---|---|---|
| 8 | 3.7 GB | 15 GB | any |
| 9 | 9.4 GB | 38 GB | H100 |
| 10 | 22 GB | 88 GB | H100 (χ = D²), H200 (2D²) |
| 11 | 47 GB | — | H100 |
| 12 | 95 GB | — | H200 |
| 13 | 180 GB | — | needs column batching of the subspace block (not implemented) |

## Budget of the first campaign (`jobs_ice.txt`)

BP-SU D = 9–11 (norm+inv, sand as separate tasks): minutes to ~1 h each on an H100. Residual
⟨Mψ|Mψ⟩ D = 7–9: ≤ 1 h. Variational D = 5–8 at χ = 2D²: ~50–100 L-BFGS iterations × (norm + sand, ~25
warm CTM iterations each): ~0.5 h (D = 5) to ~3 h (D = 8). Thirteen tasks, ≤ 16 GPUs: one wall-clock
afternoon, requeued once or twice at a 6-h limit.
