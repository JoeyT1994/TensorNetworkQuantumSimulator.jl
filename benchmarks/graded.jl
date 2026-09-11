# Graded timing harness (fermionic CTM, Z2 boundary MPS). Usage: julia --project=<branch> bench_graded.jl
using TensorNetworkQuantumSimulator, LinearAlgebra, Random, Printf
const TNQS = TensorNetworkQuantumSimulator
BLAS.set_num_threads(1)
redirect_stderr(stdout)
function fermion_state(L, D)
    Random.seed!(3)
    g = named_grid((L, L)); s = siteinds("Fermion", g; symmetry = "fZ2")
    ψ = tensornetworkstate(ComplexF64, v -> iseven(sum(v)) ? "Occ" : "Emp", g, s)
    layer = Any[]
    for ces in edge_color(g, 4); append!(layer, ("F_hop", pair, 0.3) for pair in ces); end
    ψ, _ = apply_gates(reduce(vcat, [layer for _ in 1:2]), ψ; apply_kwargs = (; maxdim = D, cutoff = 1.0e-14))
    return ψ
end
function z2_state(L, D)
    Random.seed!(11)
    g = named_grid((L, L)); s = siteinds("S=1/2", g; symmetry = "Z2")
    ψ = tensornetworkstate(ComplexF64, v -> iseven(sum(v)) ? "↑" : "↓", g, s)
    layer = Any[("Rz", [v], 0.4) for v in vertices(g)]
    for ces in edge_color(g, 4); append!(layer, ("Rxx", pair, 0.7) for pair in ces); append!(layer, ("Rzz", pair, 0.2) for pair in ces); end
    ψ, _ = apply_gates(reduce(vcat, [layer for _ in 1:2]), ψ; apply_kwargs = (; maxdim = D, cutoff = 1.0e-14))
    return ψ
end
tg = @elapsed ψf = fermion_state(4, 3)
@printf("fZ2 4x4 D=3 state (2 hop layers, incl. compile): %.1f s\n", tg)
for proj in (:cut, :cycle)
    kw = proj == :cycle ? (; projector = :cycle) : (;); ckw = proj == :cycle ? (; convergence = :marginal) : (;)
    c = update(CTMEnvironmentCache(ψf, 8; kw...); maxiter = 1, ckw...)
    st = @timed update(CTMEnvironmentCache(ψf, 16; kw...); maxiter = 6, tolerance = 1e-12, ckw...)
    @printf("fZ2 CTM %s chi=16 6 sweeps: %.2f s  alloc %.0f MB   <N>(2,2) = %.10f\n", proj, st.time, st.bytes / 1e6, real(only(expect(st.value, ("N", [(2, 2)])))))
end
ex = @timed expect(ψf, ("N", [(2, 2)]); alg = "exact"); @printf("fZ2 exact <N>: %.2f s  %.10f\n", ex.time, real(only(ex.value)))
tz = @elapsed ψz = z2_state(4, 2); @printf("Z2 4x4 D=2 state: %.1f s\n", tz)
c = update(BoundaryMPSCache(ψz, 4); maxiter = 1)
st = @timed update(BoundaryMPSCache(ψz, 16); maxiter = 3)
@printf("Z2 BMPS chi=16 3 iters: %.2f s  alloc %.0f MB  Z(2,2) = %.10f\n", st.time, st.bytes / 1e6, real(expect(st.value, ("Z", [(2, 2)]))))
st = @timed update(BeliefPropagationCache(ψz); maxiter = 20)
@printf("Z2 BP 20 iters: %.2f s  alloc %.0f MB\n", st.time, st.bytes / 1e6)
