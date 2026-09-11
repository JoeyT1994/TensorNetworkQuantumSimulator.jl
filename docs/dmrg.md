# Ground states by belief propagation: the generating-function route — *2026-09-11*

Branch `FixesV2DMRG`. This is phase 1 of a 2D DMRG built on the message-passing machinery:
BP-level (χ = 1) environments, one-site updates, exact on trees. The environments will be swapped
for the CTM `:cycle` (MP-BP) rings in phase 2.

## Why a generating function

BP on the indefinite network ⟨ψ|H|ψ⟩ is a bad idea (tried, abandoned): its messages are not
positive, so the PSD gauge, the symmetric gauge and everything the CTM inherits from positivity
break. Instead the Hamiltonian is embedded in a POSITIVE norm network,

    T(λ) = ⟨ψ|G(λ)|ψ⟩,   G(λ) = ∏ₑ(1 + λhₑ)∏ᵥ(1 + λhᵥ) = 1 + λH + O(λ²),

and only λ-derivatives at λ = 0 are ever taken. With X(λ) the BP fixed point of T(λ) and Z_B the
Bethe estimator (∏ᵥZᵥ / ∏ₑZₑ), stationarity ∂ₓZ_B = 0 gives the envelope theorem

    E_B = ∂λ Z_B(T(λ), X₀)|₀ / Z_B(T₀, X₀)        — messages held fixed —

which at χ = 1 is the BP expectation value of H (checked to 1e-9) and the exact energy on a tree.
The gradient with respect to a site tensor needs the environment response X′ = dX/dλ, obtained
from the linearised message update (Gauss–Seidel on the one-insertion sector; exact after one
sweep on a tree, a contraction whenever BP is stable), and the local problem is the generalised
eigenproblem `H_eff ψᵥ = E N_eff ψᵥ` with

    N_eff ψᵥ = ψᵥ · G(0) · X₀,     H_eff ψᵥ = ψᵥ · ∂λG · X₀ + Σᵤ ψᵥ · G(0) · X₀[m_{u→v} → X′_{u→v}].

`(H_eff − Eᵥ N_eff)ψᵥ / Zᵥ` with Eᵥ the local Rayleigh quotient IS the gradient of the BP energy
(central finite difference of the re-converged BP energy agrees to 1e-7 on a 3×3 grid, 1e-10 on a
tree). N_eff factorises over the incoming bonds into the norm messages, so the whitening
`N⁻¹ᐟ²` is the simple-update gauge (`pseudo_sqrt_inv_sqrt`) and the local solve is a Lanczos on
the Hermitian `N⁻¹ᐟ² H_eff N⁻¹ᐟ²`.

## The operator network

`TensorNetworkOperator` (`src/TensorNetworks/tensornetworkoperator.jl`) is an operator layer with
virtual bonds; `QuadraticForm` accepts it as its operator, so the three-layer norm network runs
through the ordinary `BeliefPropagationCache`. `generating_operator(H, ψ)` builds a
`GeneratingOperator`: the pair `value = G(0)`, `derivative = ∂λG(0)` in one representation.
Each edge term hₑ = Σₐ Aₐ ⊗ Bₐ (operator-Schmidt decomposition by `factorize_svd`) becomes a left
factor Σₐ Aₐ ⊗ |a⟩ on `src(e)` and a right factor 1 ⊗ |0⟩ + λ Σₐ Bₐ ⊗ |a⟩ on `dst(e)` sharing an
auxiliary index of dimension r + 1 (4 for Heisenberg). The auxiliary index is kept at λ = 0: the
messages of T(0) carry the norm sector (a = 0) plus the half-insertions ψ†Aₐψ (a > 0) travelling
from left ends to right ends, and ∂λG picks them up. No dual numbers: the λ-derivative at fixed
messages is the sum of explicit single-term insertions, and the product rule over the vertex
factors gives `derivative` exactly.

One gauge subtlety: BP normalises a message by the sum of all its entries, which is complex once
the non-Hermitian a > 0 slices are present; the whole message then carries a phase and N_eff,
H_eff stop being Hermitian maps. `generating_cache` and the response renormalise by the norm-sector
sum instead (real for a Hermitian slice). Z_B is invariant, the maps are Hermitian to 1e-16.

## Measured

`test/test_dmrg.jl` (56 tests): Z_B of the generating network = ⟨ψ|ψ⟩_BP; E_B = BP energy on a
3×3 grid and = exact on a tree; H_eff, N_eff Hermitian; gradient = finite difference; one-site DMRG
on a 6-site comb tree at D = 8 reproduces exact diagonalisation to 1e-15 after the first sweep.

3×3 Heisenberg + transverse field (ED −5.04933), random start, 6 sweeps, BP energy per sweep:

| D | E_B sweep 1 → 6 | exact energy of the final state |
|---|---|---|
| 2 | −4.463 → −4.7387 (monotone) | −4.839 |
| 3 | −4.710 → −4.7705 | −4.932 |
| 4 | −4.772 → −4.7887 | −5.0007 |

The state improves with D and is variational in the exact energy; the χ = 1 estimator is 5% off
the exact energy of its own state — that gap is what the MP-BP environments (phase 2) remove.

## The response solve needs no preconditioner at χ = 1 — *measured 2026-09-11*

The response is solved in the fixed-point (Jacobian) form `(1 − J) X′ = ∂λF`, not the Hessian form
`∂²ₓZ_B X′ = −∂λ∂ₓZ_B`. Heisenberg + field, tolerance 1e-10 on the tangent:

| case | tangent dim | Gauss–Seidel sweeps | GMRES matvecs (no preconditioner) | ρ(J) |
|---|---|---|---|---|
| 3×3 D=2 random / DMRG state | 384 | 21 / 10 | 42 / 18 | 0.59 / 0.30 |
| 3×3 D=4 random / DMRG state | 1536 | 17 / 19 | 38 / 32 | 0.50 / 0.42 |
| 4×4 D=3 random / DMRG state | 1728 | 20 / 13 | 41 / 26 | 0.59 / 0.61 |

`1 − J` is well conditioned whenever BP is a stable fixed point (ρ(J) < 1), because the message
normalisation has already divided out the vertex and edge scalars that make the raw Bethe Hessian
badly scaled, and fixed the gauge (scale) null space. Formally the Hessian is (minus) a
block-diagonal metric built from the Zᵥ, Zₑ times `(1 − J)`, so block-Jacobi preconditioning of the
Hessian form is the Jacobian form. Gauss–Seidel is the cheaper of the two here; GMRES is the
fallback when ρ(J) → 1, and the form to use for χ > 1 environments where the update iteration is
known not to be stable.

## Cost and what is next

Per vertex update: one response solve (a few BP-sweep equivalents with (r + 1)× wider messages)
plus a Lanczos whose matvec is one vertex contraction per incoming edge. The response is recomputed
from scratch after every update; incremental refresh is the first optimisation.

1. Two-site updates (open two neighbouring vertices; same response solve) so the bond dimension can grow.
2. CTM `:cycle` environments: needs two-vertex CTM regions and an incremental environment refresh
   (a full sweep at L = 6, D = 4, χ = 64 is 81 s on the CPU).
3. Graded/fermionic operator networks (the auxiliary index needs sectors; `factorize_svd` already handles graded terms).
