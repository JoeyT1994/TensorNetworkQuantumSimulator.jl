@eval module $(gensym())
using Test: @test, @testset
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
const TI = TNQS.TensorInterface
using LinearAlgebra: norm, eigvals, Hermitian, I, kron
using Random: Random

# Heisenberg exchange plus a transverse field on every edge/vertex of `g`
function heisenberg_terms(g)
    H = Any[]
    for e in edges(g), P in ("XX", "YY", "ZZ")
        push!(H, (P, (src(e), dst(e)), 0.25))
    end
    for v in vertices(g)
        push!(H, ("X", [v], 0.3))
    end
    return H
end
_obs(t) = (t[1], t[2] isa Tuple ? collect(t[2]) : t[2], t[3])
energy(ψ, H; alg) = sum(real(only(expect(ψ, _obs(t); alg))) for t in H)
function energy(bpc::BeliefPropagationCache, H)
    return sum(real(only(expect(bpc, _obs(t)))) for t in H)
end
function ed_ground_energy(g, H)
    vs = collect(vertices(g)); idx = Dict(v => i for (i, v) in enumerate(vs)); n = length(vs)
    P = Dict('X' => ComplexF64[0 1; 1 0], 'Y' => ComplexF64[0 -im; im 0], 'Z' => ComplexF64[1 0; 0 -1])
    I2 = Matrix{ComplexF64}(I, 2, 2)
    Hm = zeros(ComplexF64, 2^n, 2^n)
    for t in H
        verts = t[2] isa Tuple ? collect(t[2]) : t[2]
        ops = Dict(idx[verts[k]] => P[t[1][k]] for k in eachindex(verts))
        Hm += t[3] * reduce(kron, [get(ops, i, I2) for i in 1:n])
    end
    return minimum(real(eigvals(Hermitian(Hm))))
end

@testset "generating-function BP-DMRG" begin
    Random.seed!(0x5eed_d3a6)

    @testset "generating operator and Bethe energy" begin
        for (g, exact_on) in ((named_comb_tree((2, 3)), true), (named_grid((3, 3)), false))
            H = heisenberg_terms(g)
            ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 3)
            gen = generating_operator(H, ψ)
            @test gen isa GeneratingOperator
            # every edge carries an auxiliary index of dimension r + 1 = 4 (XX + YY + ZZ has rank 3)
            for e in edges(g)
                @test TI.dim(only(virtualinds(gen.value, e))) == 4
                @test virtualinds(gen.value, e) == virtualinds(gen.derivative, e)
            end
            bpc = generating_cache(ψ, gen; maxiter = 60, tolerance = 1.0e-13)
            # the norm sector of the generating network is the norm network
            @test partitionfunction(bpc) ≈ norm_sqr(ψ; alg = "bp") rtol = 1.0e-9
            # envelope theorem: the energy at fixed messages is the BP expectation value of H
            E_B = bethe_energy(bpc, gen)
            @test abs(imag(E_B)) < 1.0e-7      # zero up to BP convergence of the fixed point
            nbpc = update(BeliefPropagationCache(ψ); maxiter = 60, tolerance = 1.0e-13)
            @test real(E_B) ≈ energy(nbpc, H) rtol = 1.0e-6   # two BP fixed points, each converged to its tolerance
            exact_on && @test real(E_B) ≈ energy(ψ, H; alg = "exact") atol = 1.0e-10
            @test real(bethe_energy(ψ, H; maxiter = 60, tolerance = 1.0e-13)) ≈ real(E_B) rtol = 1.0e-6
        end
    end

    @testset "effective operators: Hermitian, and the gradient of the BP energy" begin
        for g in (named_comb_tree((2, 3)), named_grid((3, 3)))
            H = heisenberg_terms(g)
            ψ = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 2)
            gen = generating_operator(H, ψ)
            bpc = generating_cache(ψ, gen; maxiter = 80, tolerance = 1.0e-14)
            dX = message_response(bpc, gen; tol = 1.0e-13)
            v = first(vertices(g))
            N, Hf = effective_operators(bpc, gen, dX, v)
            ψv = ψ[v]
            φ = TI.random_tensor(ComplexF64, collect(TI.inds(ψv))...)
            @test abs(TI.scalar(TI.dag(φ) * Hf(ψv)) - conj(TI.scalar(TI.dag(ψv) * Hf(φ)))) < 1.0e-7
            @test abs(TI.scalar(TI.dag(φ) * N(ψv)) - conj(TI.scalar(TI.dag(ψv) * N(φ)))) < 1.0e-7
            # dE_B/dψᵥ* = (H_eff − Eᵥ N_eff)ψᵥ / Zᵥ against a central finite difference of the
            # re-converged BP energy: this checks the environment-response term
            Zv = real(TI.scalar(TI.dag(ψv) * N(ψv)))
            Ev = real(TI.scalar(TI.dag(ψv) * Hf(ψv))) / Zv
            grad = Hf(ψv) - Ev * N(ψv)
            δ = TI.random_tensor(ComplexF64, collect(TI.inds(ψv))...); δ = δ / norm(δ)
            h = 1.0e-5
            ψp = copy(ψ); ψp[v] = ψv + h * δ
            ψm = copy(ψ); ψm[v] = ψv - h * δ
            ebp(ϕ) = energy(update(BeliefPropagationCache(ϕ); maxiter = 80, tolerance = 1.0e-14), H)
            fd = (ebp(ψp) - ebp(ψm)) / (2h)
            @test 2 * real(TI.scalar(TI.dag(δ) * grad)) / Zv ≈ fd rtol = 1.0e-5
        end
    end

    @testset "DMRG: exact on a tree, monotone on a loopy graph" begin
        g = named_comb_tree((2, 3)); H = heisenberg_terms(g)
        ψ0 = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 8)   # lossless at 6 sites
        ψ, Es = dmrg(ψ0, H; nsweeps = 2, verbose = false)
        @test last(Es) ≈ ed_ground_energy(g, H) atol = 1.0e-9
        @test energy(ψ, H; alg = "exact") ≈ last(Es) atol = 1.0e-9

        g = named_grid((3, 3)); H = heisenberg_terms(g)
        ψ0 = random_tensornetworkstate(ComplexF64, g, "S=1/2"; bond_dimension = 2)
        ψ, Es = dmrg(ψ0, H; nsweeps = 2, verbose = false, bp_kwargs = (; maxiter = 50, tolerance = 1.0e-11), response_kwargs = (; tol = 1.0e-9))
        @test last(Es) < first(Es)
        @test last(Es) > ed_ground_energy(g, H) - 0.5      # the Bethe energy of a D = 2 state is not below ED by much
        @test energy(ψ, H; alg = "exact") > ed_ground_energy(g, H) - 1.0e-8   # variational in the exact energy
    end
end
end
