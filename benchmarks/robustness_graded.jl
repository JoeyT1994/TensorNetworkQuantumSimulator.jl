# Robustness sweep: fermionic and Z2 CTM (:cut/:cycle) and boundary MPS vs exact contraction as chi grows.
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
function z2_state(symmetry, L, D)
    Random.seed!(11)
    g = named_grid((L, L)); s = symmetry === nothing ? siteinds("S=1/2", g) : siteinds("S=1/2", g; symmetry)
    ψ = tensornetworkstate(ComplexF64, v -> iseven(sum(v)) ? "↑" : "↓", g, s)
    layer = Any[("Rz", [v], 0.4) for v in vertices(g)]
    for ces in edge_color(g, 4); append!(layer, ("Rxx", pair, 0.7) for pair in ces); append!(layer, ("Rzz", pair, 0.2) for pair in ces); end
    ψ, _ = apply_gates(reduce(vcat, [layer for _ in 1:2]), ψ; apply_kwargs = (; maxdim = D, cutoff = 1.0e-14))
    return ψ
end
function sweep(label, ψ, obs, chis; bmps = true)
    ex = real(only(expect(ψ, obs; alg = "exact")))
    @printf("%s exact = %.12f\n", label, ex)
    for χ in chis
        for proj in (:cut, :cycle)
            kw = proj == :cycle ? (; projector = :cycle) : (;); ckw = proj == :cycle ? (; convergence = :marginal) : (;)
            st = @timed update(CTMEnvironmentCache(ψ, χ; kw...); maxiter = 40, tolerance = 1e-10, ckw...)
            v = real(only(expect(st.value, obs)))
            @printf("   CTM %-5s chi=%2d  err %.2e  (%.1f s)\n", proj, χ, abs(v - ex), st.time)
        end
        if bmps
            st = @timed expect(ψ, obs; alg = "boundarymps", mps_bond_dimension = χ)
            @printf("   BMPS      chi=%2d  err %.2e  (%.1f s)\n", χ, abs(real(only(st.value)) - ex), st.time)
        end
    end
end
ψf = fermion_state(4, 3)
sweep("fZ2 4x4 D=3 <N>(2,2)", ψf, ("N", [(2, 2)]), (8, 16, 24, 36))
ψz = z2_state("Z2", 4, 4); ψd = z2_state(nothing, 4, 4)
sweep("Z2 4x4 D=4 <Z>(2,2)", ψz, ("Z", [(2, 2)]), (4, 8, 16))
sweep("dense twin 4x4 D=4 <Z>(2,2)", ψd, ("Z", [(2, 2)]), (4, 8, 16))
