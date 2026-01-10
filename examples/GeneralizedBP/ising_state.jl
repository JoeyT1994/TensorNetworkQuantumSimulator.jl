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


    ns = [6]
    β = 2.0
    for n in ns
        println("-------------------------------------")
        println("Building Ising state on an $n x $n grid")
        g = named_grid((n,n); periodic = false)
        Js = Dictionary(collect(edges(g)), [first(src(e)) == first(dst(e)) && isodd(first(src(e))) ? -1.0 : 1.0 for e in edges(g)])

        #Either 
        # ψ = ising_tensornetwork(g, β; Js)
        # z_exact = contract(ψ; alg = "exact")
        # f_exact = -log(z_exact)
        # @show f_exact

        #Or
        ψ = ising_tensornetwork_rdm(g, β; Js)
        ψψ_tensors = Dictionary(collect(vertices(g)), [reduce(*, norm_factors(ψ, v)) for v in vertices(g)])
        ψψ = TensorNetwork(ψψ_tensors, g)
        z_exact = contract(ψψ; alg = "exact")
        f_exact = -log(z_exact)
        @show f_exact

        ψ_bpc = BeliefPropagationCache(ψ)
        ψ_bpc = update(ψ_bpc)

        loop_size = 4
        bs = construct_gbp_bs(ψ_bpc, loop_size)
        ms = construct_ms(bs)
        ps = all_parents(ms, bs)
        mobius_nos = mobius_numbers(ms, ps)
        ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
        cs = children(ms, ps, bs)
        b_nos = calculate_b_nos(ms, ps, mobius_nos)

        msgs, diffs, gbp_converged = generalized_belief_propagation(ψ_bpc, bs, ms, ps, cs, b_nos, mobius_nos; niters = 1000, rate = 0.35, verbose = true)
        msgs = normalize_messages(msgs)
        gbp_f = kikuchi_free_energy(ψ_bpc, ms, bs, msgs, cs, b_nos, ps, mobius_nos)

        bp_f = -log(partitionfunction(ψ_bpc))

        f_lc = -log(loopcorrected_partitionfunction(ψ_bpc, loop_size))

        #f_exact = -log(norm_sqr(ψ; alg = "exact"))

        println("Exact free energy density: $(f_exact/(n*n))")

        println("BP abs error on free energy density: $(abs((f_exact/(n*n)) - bp_f/(n*n)))")
        println("Loop Corrected BP abs error on free energy density: $(abs((f_exact/(n*n)) - f_lc/(n*n)))")
        println("GBP abs error on free energy density: $(abs((f_exact/(n*n)) - gbp_f/(n*n)))")
    end

    #println("Simple BP absolute error on free energy: ", abs(bp_f - f_exact))
    #println("Generalized BP absolute error on free energy: ", abs(gbp_f - f_exact))
    #println("Loop corrected BP absolute error on free energy: ", abs(f_lc - f_exact))
end

main()