using TensorNetworkQuantumSimulator

using NamedGraphs: NamedEdge
using TensorNetworkQuantumSimulator: dag, virtualinds, normalize, loopcorrected_partitionfunction
using ITensors: prime, ITensor, combiner, replaceind, commoninds, inds, delta, random_itensor
using Dictionaries: Dictionary
using Random

function uniform_random_itensor(eltype, inds)
    t = ITensor(eltype, 1.0, inds)
    for iv in eachindval(t)
        t[iv...] = rand()
    end
    return t
end

include("utils.jl")
include("generalizedbp.jl")

function main()

    Random.seed!(1854)

    n = 3
    g = named_grid((n,n); periodic = false)
    loop_size = 4
    #Build physical site indices for spin-1/2 degrees of freedom
    s = siteinds("S=1/2", g)

    println("Running Generalized Belief Propagation on the norm of a $n x $n random Tensor Network State")

    nsamples = 1
    err_bp, err_gbp, err_lc = 0.0, 0.0, 0.0
    for i in 1:nsamples
        println("-------------------------------------")
        ψ = random_tensornetworkstate(Float64, g, s; bond_dimension = 2)

        tensors = Dictionary(vertices(g), [uniform_random_itensor(Float64, inds(ψ[v])) for v in vertices(g)])
        ψ = TensorNetworkState(TensorNetwork(tensors, graph(ψ)), siteinds(ψ))

        ψ = normalize(ψ; alg = "bp")

        ψ_bpc = BeliefPropagationCache(ψ)

        bs = construct_gbp_bs(ψ_bpc, loop_size)
        ms = construct_ms(bs)
        ps = all_parents(ms, bs)
        mobius_nos = mobius_numbers(ms, ps)
        ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
        cs = children(ms, ps, bs)
        b_nos = calculate_b_nos(ms, ps, mobius_nos)

        gbp_f = generalized_belief_propagation(ψ_bpc, bs, ms, ps, cs, b_nos, mobius_nos; niters = 300, rate = 0.3)
        bp_f = -log(norm_sqr(ψ; alg = "bp"))

        ψ_bpc = update(ψ_bpc)
        f_lc = -log(loopcorrected_partitionfunction(ψ_bpc, loop_size))

        f_exact = -log(norm_sqr(ψ; alg = "exact"))

        err_bp += abs(bp_f - f_exact)
        err_gbp += abs(gbp_f - f_exact)
        err_lc += abs(f_lc - f_exact)
        println("Simple BP absolute error on free energy: ", abs(bp_f - f_exact))
        println("Generalized BP absolute error on free energy: ", abs(gbp_f - f_exact))
        println("Loop corrected BP absolute error on free energy: ", abs(f_lc - f_exact))
    end

    println("Average simple BP absolute error on free energy over $nsamples samples: ", err_bp / nsamples)
    println("Average generalized BP absolute error on free energy over $nsamples samples: ", err_gbp / nsamples)
    println("Average loop corrected BP absolute error on free energy over $nsamples samples: ", err_lc / nsamples)
end

main()