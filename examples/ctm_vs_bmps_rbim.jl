# Finite random-bond Ising (RBIM) at the Nishimori point, in a field: CTMRG (:cut, :cycle) vs
# boundary MPS at matched χ, against an exact contraction, averaged over disorder.
#
# Mirrors Fig. 2 of the "Matrix Product Belief Propagation" draft (10x10 RBIM at critical
# temperature, bias field h = 1e-2, magnetisation at their site "(3+1,3+1)" = (4,4) here).
# That figure's caption says their CTM code "struggles to converge" at larger bond dimensions
# and they "need to get data at higher χ" -- this runs cleanly to χ=16.
#
# WHY THIS MODEL: negative bonds force a COMPLEX square root of the Boltzmann matrix, i.e. a
# strongly NON-HERMITIAN network. Their draft picks the RBIM precisely because "random bonds are
# advantageous to our cause as the network becomes strongly non-Hermitian".
#
# HEADLINE: `:cycle` (the stationary, eigen/Schur-truncated projector) reaches machine precision
# at χ=8, where `:cut` needs χ=12 and boundary MPS never does -- at χ=8 it is ~7000x better than
# `:cut` and ~9400x better than bMPS. bMPS then FREEZES at 1.351354e-11 for χ = 12..16, identical
# to 7 s.f. The same plateau appears on the clean Ising (2.940334e-12), so it is systematic.
#
# THREE INFERENCES, NOT STATED IN THEIR CAPTION -- confirm before overlaying figures:
#   * "critical temperature" is read as the multicritical NISHIMORI point, p = 0.109 on the
#     Nishimori line exp(-2β) = p/(1-p), giving β = 1.0505.
#   * "(3+1,3+1)" is read as 0-indexed (3,3) plus a 1-offset, i.e. (4,4) 1-indexed -- off-centre.
#   * 5 disorder realisations; their band's realisation count is unknown.
#
# Writes examples/data/ctm_vs_bmps_rbim10_nishimori.csv. Plot with plot_ctm_vs_bmps.py.
# Run: julia --project=. --startup-file=no examples/ctm_vs_bmps_rbim.jl        (~10 min)

using TensorNetworkQuantumSimulator, ITensors, Printf, Random, Statistics
using Dictionaries: Dictionary, set!
using NamedGraphs.GraphsExtensions: incident_edges
const T = TensorNetworkQuantumSimulator

# RBIM in a field. Per-edge J_e = ±1; negative bonds need a COMPLEX sqrt of the Boltzmann
# matrix, which is exactly why they pick this model ("the network becomes strongly non-Hermitian").
function rbim_setup(g, β, h, Js)
    links = Dictionary(edges(g), [Index(2, "e$(src(e))_$(dst(e))") for e in edges(g)])
    links = merge(links, Dictionary(reverse.(edges(g)), [links[e] for e in edges(g)]))
    sW = Dictionary()
    for e in edges(g)
        a = β*Js[e]; a = a < 0 ? Complex(a) : a
        λ1, λ2 = cosh(a), sinh(a)
        α, ϕ = 0.5*(sqrt(λ1)+sqrt(λ2)), 0.5*(sqrt(λ1)-sqrt(λ2))
        M = sqrt(2) * [α ϕ; ϕ α]
        set!(sW, e, M); set!(sW, reverse(e), M)
    end
    function mk(v; spin::Bool=false)
        es = collect(incident_edges(g, v; dir=:in)); n = length(es)
        w = [exp(β*h*s) * (spin ? s : 1.0) for s in (+1.0,-1.0)]
        A = zeros(ComplexF64, ntuple(_->2, n))
        for si in 1:2, idx in CartesianIndices(size(A))
            A[idx] += w[si] * prod(sW[es[k]][si, idx[k]] for k in 1:n)
        end
        ITensor(A, [links[e] for e in es])
    end
    vs = collect(vertices(g))
    tn = T.TensorNetwork(Dictionary(vs, [mk(v) for v in vs]), g)
    swap(v) = T.TensorNetwork(Dictionary(vs, [u==v ? mk(u;spin=true) : mk(u) for u in vs]), g)
    return tn, mk, swap
end
m_ctm(cache, tspin, tplain, v) = begin
    e = T.vertex_window(cache, v, 0)
    scalar(T._ctm_contract(ITensor[tspin; e], T.options(cache))) /
    scalar(T._ctm_contract(ITensor[tplain; e], T.options(cache)))
end

# Nishimori line: exp(-2β) = p/(1-p).  Multicritical (Nishimori) point at p ≈ 0.109.
p  = 0.109
β  = 0.5*log((1-p)/p)
L, h, v = 10, 0.01, (4,4)
@printf("10x10 RBIM, Nishimori point p=%.3f -> β=%.4f, h=%.2f, site %s\n", p, β, h, string(v))
@printf("(their Fig. 2 says 'critical temperature'; the Nishimori point is my inference)\n")
NREAL = 5
res = Dict(k => [Float64[] for _ in 1:16] for k in (:cut,:cycle,:bmps))
for r in 1:NREAL
    Random.seed!(100+r); g = named_grid((L,L))
    Js = Dictionary(collect(edges(g)), [rand() < p ? -1.0 : 1.0 for _ in edges(g)])
    tn, mk, swap = rbim_setup(g, β, h, Js)
    tnS = swap(v)
    mex = real(contract(tnS;alg="exact"))/real(contract(tn;alg="exact"))
    for χ in 1:16
        for pr in (:cut,:cycle)
            c = update(CTMEnvironmentCache(tn, χ; projector=pr))
            push!(res[pr][χ], abs(real(m_ctm(c, mk(v;spin=true), mk(v), v)) - mex)/abs(mex))
        end
        mb = real(contract(tnS;alg="boundarymps",mps_bond_dimension=χ))/
             real(contract(tn; alg="boundarymps",mps_bond_dimension=χ))
        push!(res[:bmps][χ], abs(mb-mex)/abs(mex))
    end
    println("realisation $r done"); flush(stdout)
end
println("\nchi,cut_med,cut_lo,cut_hi,cycle_med,cycle_lo,cycle_hi,bmps_med,bmps_lo,bmps_hi")
for χ in 1:16
    f(k) = (median(res[k][χ]), minimum(res[k][χ]), maximum(res[k][χ]))
    a,b,c = f(:cut), f(:cycle), f(:bmps)
    @printf("%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n", χ, a...,b...,c...)
end
