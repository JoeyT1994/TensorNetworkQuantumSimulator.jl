# Finite classical Ising in a field: CTMRG (:cut and :cycle) vs boundary MPS, at matched χ,
# measured at a CORNER and at the CENTRE against an exact contraction.
#
# This mirrors Fig. 3 of the collaborators' "Matrix Product Belief Propagation" draft (11x11
# single-layer Ising, h = 0.01, magnetisation at (0,0) and (5,5)), so the two can be compared.
#
# WHAT IT SHOWS: CTM beats bMPS at matched χ by 1-2.5 orders at both sites -- 32x at χ=2, 400x at
# χ=6 in the centre.
#
# WHAT IT DOES NOT SHOW, and must not be quoted as showing:
#   * Their O(ε²)-vs-O(ε) SCALING claim. That is a plot against the environment error ε; this is a
#     plot against χ. No slope can be read off this.
#   * Their remark that bMPS gets ε² "for free" at the centre via reflection symmetry. Measured
#     here, bMPS is 2.4e-09 at the corner and 3.9e-09 at the centre -- comparable, not better in
#     the middle.
#
# `:cut` and `:cycle` come out nearly equal on this structured classical network; the stationary
# projector's advantage appears on double-layer states instead (see ctm_projector_survey.jl).
#
# Run: julia --project=. --startup-file=no examples/ctm_vs_bmps_ising_field.jl   (~2 min)

using TensorNetworkQuantumSimulator, ITensors, Printf
using Dictionaries: Dictionary
using NamedGraphs.GraphsExtensions: incident_edges
const T = TensorNetworkQuantumSimulator

# Finite classical Ising WITH a field. `ising_partitionfunction` has no h, so build it: edges carry
# the symmetric sqrt of exp(β σσ'), vertices are deltas weighted by w(s)=exp(βhs). Returns the
# network AND a maker so the spin-weighted vertex SHARES the same Index objects (needed to contract
# it against a CTM environment built from the plain network).
function ising_setup(g, β, h)
    links = Dictionary(edges(g), [Index(2, "e$(src(e))_$(dst(e))") for e in edges(g)])
    links = merge(links, Dictionary(reverse.(edges(g)), [links[e] for e in edges(g)]))
    λ1, λ2 = cosh(β), sinh(β)
    α, ϕ = 0.5*(sqrt(λ1)+sqrt(λ2)), 0.5*(sqrt(λ1)-sqrt(λ2))
    sW = sqrt(2) * [α ϕ; ϕ α]
    function mk(v; spin::Bool = false)
        es = collect(incident_edges(g, v; dir = :in)); n = length(es)
        w = [exp(β*h*s) * (spin ? s : 1.0) for s in (+1.0, -1.0)]
        A = zeros(Float64, ntuple(_->2, n))
        for si in 1:2, idx in CartesianIndices(size(A))
            A[idx] += w[si] * prod(sW[si, idx[k]] for k in 1:n)
        end
        ITensor(A, [links[e] for e in es])
    end
    vs = collect(vertices(g))
    tn = T.TensorNetwork(Dictionary(vs, [mk(v) for v in vs]), g)
    swap(v) = T.TensorNetwork(Dictionary(vs, [u == v ? mk(u; spin=true) : mk(u) for u in vs]), g)
    return tn, mk, swap
end

m_ctm(cache, tspin, tplain, v) = begin
    e = T.vertex_window(cache, v, 0)
    scalar(T._ctm_contract(ITensor[tspin; e], T.options(cache))) /
    scalar(T._ctm_contract(ITensor[tplain; e], T.options(cache)))
end

L, β, h = 11, 0.4407, 0.01
g = named_grid((L,L)); tn, mk, swap = ising_setup(g, β, h)
zden = real(contract(tn; alg="exact"))
@printf("Finite %d×%d classical Ising, β=%.4f (≈critical), h=%.2f — their Fig. 3\n", L, L, β, h)
for (v,lbl) in [(1,1)=>"CORNER", (6,6)=>"CENTRE"]
    tnS = swap(v); mex = real(contract(tnS; alg="exact"))/zden
    @printf("\n%s %s   exact m = %.10f\n", lbl, string(v), mex)
    @printf("  %-3s %-13s %-13s %-13s\n", "χ", "δm/m :cut", "δm/m :cycle", "δm/m bMPS")
    for χ in (1,2,3,4,5,6)
        ec = map((:cut,:cycle)) do p
            c = update(CTMEnvironmentCache(tn, χ; projector=p))
            abs(m_ctm(c, mk(v; spin=true), mk(v), v) - mex)/abs(mex)
        end
        mb = real(contract(tnS; alg="boundarymps", mps_bond_dimension=χ)) /
             real(contract(tn;  alg="boundarymps", mps_bond_dimension=χ))
        @printf("  %-3d %-13.3e %-13.3e %-13.3e\n", χ, ec[1], ec[2], abs(mb-mex)/abs(mex))
        flush(stdout)
    end
end
