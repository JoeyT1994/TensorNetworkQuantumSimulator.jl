# Smoke test: spinful Hubbard on a single hexagon (6 sites on a 3×2 box) through the standard
# pipeline  simple update → BP DMRG (χ = 1) → CTMRG L-BFGS,  every stage checked against the exact
# contraction of the state and against exact diagonalisation (sparse Jordan–Wigner, 12 modes).
#
#   julia --project examples/hubbard_hex_smoke.jl
#
# Edit the parameters below. The first stages also pay compilation; the printed timings say so.
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
const TI = TNQS.TensorInterface
using LinearAlgebra, SparseArrays, Random, Printf

# ── parameters ───────────────────────────────────────────────────────────────────────────
const SYMMETRY   = "fZ2"          # "fZ2" (parity) or "fU1xU1" (N↑ and N↓ conserved)
const D          = 8              # bond dimension
const t, U       = 1.0, 4.0
const μ          = U / 2          # half filling (particle–hole symmetric point)
const SU_DTAU    = 0.01           # simple update: imaginary-time step
const SU_STEPS   = 200             # simple update: number of second-order-ish Trotter layers
const BP_SWEEPS  = 10              # BP DMRG sweeps
const CHI        = 64             # CTMRG interface dimension
const CHI_CHECK  = (16, 64)       # χ values at which the ring / FD-of-F energies are checked
const LBFGS_ITER = 10              # CTMRG L-BFGS iterations
const STEP0      = 1 / 16         # first (Jacobi) step; spinless fermions needed 1/32 rather than 1/8
const SEED       = 1

Random.seed!(SEED)
g = named_hexagonal_lattice_graph(1,1)             # one hexagon: 6 sites, grid positions on a 3×2 box
@assert nv(g) == 6
vs = collect(vertices(g))
s = siteinds("Electron", g; symmetry = SYMMETRY)

# ── model ────────────────────────────────────────────────────────────────────────────────
H = Any[]
for e in edges(g)
    push!(H, ("hopping", (src(e), dst(e)), -t))     # -t Σσ (c†σv cσw + h.c.), both spins in one term
end
for v in vs
    push!(H, ("NupNdn", [v], U))
    iszero(μ) || push!(H, ("N", [v], -μ))
end
println("Hubbard on a hexagon: t = $t, U = $U, μ = $μ, symmetry = $SYMMETRY, D = $D")

_obs(term) = (term[1], term[2] isa Tuple ? collect(term[2]) : term[2], term[3])
tn_energy(ψ; alg = "exact") = sum(real(only(expect(ψ, _obs(term); alg))) for term in H)

# ── exact diagonalisation (modes up(v) = 2k−1, dn(v) = 2k in the order of `vs`) ──────────
function hubbard_ed()
    idx = Dict(v => k for (k, v) in enumerate(vs)); nm = 2 * length(vs)
    a = sparse([0.0 1.0; 0.0 0.0]); z = sparse([1.0 0.0; 0.0 -1.0]); i2 = sparse(1.0I, 2, 2)
    cs = [reduce(kron, [k < j ? z : (k == j ? a : i2) for k in 1:nm]) for j in 1:nm]
    up(v) = 2idx[v] - 1; dn(v) = 2idx[v]; n(m) = cs[m]' * cs[m]
    Hm = spzeros(Float64, 2^nm, 2^nm)
    for term in H
        name, verts, coeff = _obs(term)
        if name == "hopping"
            v, w = verts
            for (mv, mw) in ((up(v), up(w)), (dn(v), dn(w)))
                Hm += coeff * (cs[mv]' * cs[mw] + cs[mw]' * cs[mv])
            end
        elseif name == "NupNdn"
            v = only(verts); Hm += coeff * n(up(v)) * n(dn(v))
        elseif name == "N"
            v = only(verts); Hm += coeff * (n(up(v)) + n(dn(v)))
        end
    end
    occ(st, j) = (st >> (nm - j)) & 1
    nup(st) = sum(occ(st, 2k - 1) for k in 1:length(vs)); ndn(st) = sum(occ(st, 2k) for k in 1:length(vs))
    function sector(pred)
        keep = [st + 1 for st in 0:(2^nm - 1) if pred(st)]
        return minimum(eigvals(Hermitian(Matrix(Hm[keep, keep]))))
    end
    nh = length(vs) ÷ 2
    return (; all = sector(_ -> true), even = sector(st -> iseven(nup(st) + ndn(st))),
              half = sector(st -> nup(st) == nh && ndn(st) == nh))
end
τ = @elapsed ed = hubbard_ed()
@printf("ED (%.1f s): all sectors %.10f   even parity %.10f   N↑ = N↓ = 3 %.10f\n", τ, ed.all, ed.even, ed.half)
# the pipeline conserves what the grading enforces: parity under fZ2, (N↑, N↓) under fU1xU1
E_ref = SYMMETRY == "fZ2" ? ed.even : ed.half
report(stage, ψ) = begin
    E = tn_energy(ψ)
    @printf("%-28s exact contraction %.10f   gap to ED %.3e\n", stage, E, E - E_ref)
    E
end

# ── stage 1: simple update from an alternating ↑↓ product state (half filling) ───────────
ψ = tensornetworkstate(ComplexF64, v -> isodd(findfirst(==(v), vs)) ? "Up" : "Dn", g, s)
layer = Any[]
for ces in edge_color(g, 3)
    append!(layer, ("F_hop", (src(e), dst(e)), im * t * SU_DTAU) for e in ces)       # exp(-dτ (-t) hop)
end
append!(layer, ("F_int", [v], -im * U * SU_DTAU) for v in vs)                        # exp(-dτ U n↑n↓)
iszero(μ) || append!(layer, ("F_phase", [v], im * μ * SU_DTAU) for v in vs)          # exp(+dτ μ N)
bpc = BeliefPropagationCache(ψ)
τ = @elapsed for step in 1:SU_STEPS
    global bpc
    bpc, errs = apply_gates(layer, bpc; apply_kwargs = (; maxdim = D, cutoff = 1.0e-14))
end
ψ_su = network(bpc)
println("simple update: $SU_STEPS steps of dτ = $SU_DTAU in $(round(τ; digits = 1)) s")
E_su = report("after simple update:", ψ_su)

# ── stage 2: BP DMRG (χ = 1 environments) ────────────────────────────────────────────────
τ = @elapsed ψ_bp, Es_bp = dmrg(ψ_su, H; alg = "bp", nsweeps = BP_SWEEPS, verbose = false)
println("BP DMRG: $BP_SWEEPS sweeps in $(round(τ; digits = 1)) s; Bethe energy $(first(Es_bp)) → $(last(Es_bp))")
E_bp = report("after BP DMRG:", ψ_bp)

# ── stage 3a: the generating network through CTMRG, checked at fixed state ───────────────
τ = @elapsed gen = generating_operator(H, ψ_bp)
auxdim = TI.dim(only(virtualinds(gen.value, first(edges(g)))))
println("generating operator in $(round(τ; digits = 1)) s; aux dimension $auxdim (expected 5 = 1 + rank-4 hopping)")
lnN = log(real(norm_sqr(ψ_bp; alg = "exact")))
for χ in CHI_CHECK
    τ0 = @elapsed cache = generating_cache(ψ_bp, gen, χ; projector = :cut)
    Ering = real(bethe_energy(cache, gen))
    λ = 1.0e-7
    τ1 = @elapsed begin
        cp = generating_cache(ψ_bp, gen, χ; λ, seed = cache, projector = :cut)
        cm = generating_cache(ψ_bp, gen, χ; λ = -λ, seed = cache, projector = :cut)
    end
    Efd = (cvm_freenergy(cp) - cvm_freenergy(cm)) / (2λ)
    @printf("χ = %3d: |F − ln N| %.1e   ring-energy error %.2e   FD-of-F error %.2e   (λ = 0 set %.1f s, ±λ pair %.1f s)\n",
            χ, abs(cvm_freenergy(cache) - lnN), abs(Ering - E_bp), abs(Efd - E_bp), τ0, τ1)
end

# ── stage 3b: CTMRG L-BFGS ───────────────────────────────────────────────────────────────
τ = @elapsed ψ_ctm, Es = dmrg(ψ_bp, H; alg = "ctmrg_lbfgs", maxdim = CHI, maxiter = LBFGS_ITER,
                              step0 = STEP0, ls_max = 5, verbose = true)
println("CTMRG L-BFGS: $(length(Es) - 1) accepted steps in $(round(τ; digits = 1)) s; FD-of-F energies ", round.(Es; digits = 8))
E_ctm = report("after CTMRG L-BFGS:", ψ_ctm)
@printf("FD-of-F vs exact contraction of the final state: %.1e   monotone: %s   variational: %s\n",
        abs(last(Es) - E_ctm), all(diff(Es) .< 0), E_ctm > ed.all - 1.0e-8)
