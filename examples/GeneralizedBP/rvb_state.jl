using TensorNetworkQuantumSimulator

using NamedGraphs: NamedEdge, NamedGraphs
using TensorNetworkQuantumSimulator: dag, virtualinds, normalize, loopcorrected_partitionfunction
using ITensors: prime, ITensor, combiner, replaceind, commoninds, inds, delta, random_itensor
using Dictionaries: Dictionary
using Random

function uniform_random_itensor(eltype, inds)
    t = ITensor(eltype, 1.0, inds)
    for iv in eachindval(t)
        t[iv...] = randn() + im*randn()
    end
    return t
end

include("utils.jl")
include("generalizedbp.jl")

ITensors.disable_warn_order()
function main()

    Random.seed!(584)

    n = 4
    #ψ = toric_code_ground_state(n)
    ψ = rbs_state(n)
    g = graph(ψ)
    #ψ = gauge(ψ)
    ψ = normalize(ψ; alg = "bp")
    tensors = Dictionary(collect(vertices(g)), [reduce(*, norm_factors(ψ, v)) for v in vertices(g)])
    ψψ = TensorNetwork(tensors, g)
    TensorNetworkQuantumSimulator.combine_virtualinds!(ψψ)
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc)
    ψψ_bpc = BeliefPropagationCache(ψψ)
    ψψ_bpc = update(ψψ_bpc)

    loop_size = 4
    bs = construct_gbp_bs(ψψ_bpc, loop_size)
    #bs = construct_bp_bs(ψ_bpc)
    ms = construct_ms(bs)
    ps = all_parents(ms, bs)
    mobius_nos = mobius_numbers(ms, ps)
    ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
    cs = children(ms, ps, bs)
    b_nos = calculate_b_nos(ms, ps, mobius_nos)

    gbp_f, msgs, gbp_converged = generalized_belief_propagation(ψψ_bpc, bs, ms, ps, cs, b_nos, mobius_nos; niters = 25, rate = 0.35)

    bp_f = -log(partitionfunction(ψ_bpc))

    f_lc = -log(loopcorrected_partitionfunction(ψ_bpc, loop_size))

    f_exact = -log(norm_sqr(ψ; alg = "exact"))

    @show f_exact
    @show bp_f
    @show gbp_f
    @show f_lc

    println("Simple BP absolute error on free energy: ", abs(bp_f - f_exact))
    println("Generalized BP absolute error on free energy: ", abs(gbp_f - f_exact))
    println("Loop corrected BP absolute error on free energy: ", abs(f_lc - f_exact))
end

main()