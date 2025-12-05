using TensorNetworkQuantumSimulator

using NamedGraphs: NamedEdge
using TensorNetworkQuantumSimulator: dag, virtualinds, normalize, loopcorrected_partitionfunction
using ITensors: prime, ITensor, combiner, replaceind, commoninds, inds, delta, random_itensor
using Dictionaries: Dictionary
using Random

function uniform_random_itensor(eltype, inds)
    t = ITensor(eltype, 1.0, inds)
    for iv in eachindval(t)
        t[iv...] = rand() - 0.2
    end
    return t
end

include("utils.jl")
include("generalizedbp.jl")

Random.seed!(1654)

n =3
g = named_grid((n,n); periodic = false)
#g = named_hexagonal_lattice_graph(3,3 )
#Build physical site indices for spin-1/2 degrees of freedom
s = siteinds("S=1/2", g)

println("Running Generalized Belief Propagation on the norm of a $n x $n random Tensor Network State")

#Build a random TensorNetworkState on the graph with bond dimension 2
ψ = random_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)
#tensors = [uniform_random_itensor(scalartype(ψ), inds(ψ[v])) for v in vertices(g)]
#ψ = TensorNetworkState(Dictionary(collect(vertices(g)), tensors))
ψ = normalize(ψ; alg = "bp")
# #Take its dagger
ψdag = map_virtualinds(prime, map_tensors(dag, ψ))

# #Build the norm tensor network ψψ† and combine pairs of virtual inds
T = TensorNetwork(Dictionary(vertices(g), [ψ[v]*ψdag[v] for v in vertices(g)]))
TensorNetworkQuantumSimulator.combine_virtualinds!(T)

bs = construct_gbp_bs(T)
ms = construct_ms(bs)
ps = all_parents(ms, bs)
mobius_nos = mobius_numbers(ms, ps)
ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
cs = children(ms, ps, bs)
b_nos = calculate_b_nos(ms, ps, mobius_nos)

gbp_f = generalized_belief_propagation(T, bs, ms, ps, cs, b_nos, mobius_nos; niters = 100, rate = 0.3)
bp_f = -log(contract(T; alg = "bp"))

println("GBP free energy: ", gbp_f)
println("BP free energy: ", bp_f)

T_bpc = update(BeliefPropagationCache(T))
f_lc = -log(loopcorrected_partitionfunction(T_bpc, 4))
println("Loop corrected free energy (length 4): ", f_lc)

f_exact = -log(contract(T; alg = "exact"))
println("Exact free energy: ", f_exact)

println("-------------------------------------")
println("Simple BP absolute error on free energy: ", abs(bp_f - f_exact))
println("Generalized BP absolute error on free energy: ", abs(gbp_f - f_exact))
println("Loop corrected BP absolute error on free energy: ", abs(f_lc - f_exact))
