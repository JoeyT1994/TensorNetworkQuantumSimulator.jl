using TensorNetworkQuantumSimulator

using NamedGraphs: NamedEdge, NamedGraphs
using TensorNetworkQuantumSimulator: dag, virtualinds, normalize, loopcorrected_partitionfunction
using ITensors: prime, ITensor, combiner, replaceind, commoninds, inds, delta, random_itensor
using Dictionaries: Dictionary
using Random
using Adapt

function uniform_random_itensor(eltype, inds)
    t = ITensor(eltype, 1.0, inds)
    for iv in eachindval(t)
        t[iv...] = randn() + im*randn()
    end
    return t
end

include("utils.jl")
include("generalizedbp.jl")
include("toric_code_utils.jl")

ITensors.disable_warn_order()
function main()

    Random.seed!(584)


    ns = [6, 10, 14]
    for n in ns
        println("-------------------------------------")
        println("Building Toric code state on a $n x $n Torus")
        loop_size = 4
	#ψ = rbs_state(n)
        for (ψ, z_exact, title) = zip([toric_code_ground_state(n), Adapt.adapt(Float64,toric_code_flat(n))], [2^((3*n^2 + 2)/2), 2.0^(n^2+1)], ["Verstraete, norm network", "Pollmann, flat network"])
	    g = graph(ψ)
            ψ_bpc = BeliefPropagationCache(ψ)
            ψ_bpc = update(ψ_bpc)
	    bs = construct_gbp_bs(ψ_bpc, loop_size)

            #ψψ_tensors = Dictionary(collect(vertices(g)), [reduce(*, norm_factors(ψ, v)) for v in vertices(g)])
            #ψψ = TensorNetwork(ψψ_tensors, g)

            f_exact = -log(z_exact)
            #bs = dimer_covering_bs(ψ_bpc)
            ms = construct_ms(bs)
            ps = all_parents(ms, bs)
            mobius_nos = mobius_numbers(ms, ps)
            ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
            cs = children(ms, ps, bs)
            b_nos = calculate_b_nos(ms, ps, mobius_nos)

            msgs, diffs, gbp_converged = generalized_belief_propagation(ψ_bpc, bs, ms, ps, cs, b_nos, mobius_nos; niters = 150, rate = 0.25, verbose = false)
            msgs = normalize_messages(msgs)
            gbp_f = kikuchi_free_energy(ψ_bpc, ms, bs, msgs, cs, b_nos, ps, mobius_nos)

            bp_f = -log(partitionfunction(ψ_bpc))

            f_lc = -log(loopcorrected_partitionfunction(ψ_bpc, loop_size))

            #f_exact = -log(norm_sqr(ψ; alg = "exact"))
	    println("Version: $(title)")
            println("Exact free energy density: $(f_exact/(n*n))")
	    println("BP free energy density: $((bp_f)/n^2)")
            println("BP abs error on free energy density: $(abs((f_exact/(n*n)) - bp_f/(n*n)))")
            println("Loop Corrected BP abs error on free energy density: $(abs((f_exact/(n*n)) - f_lc/(n*n)))")
            println("GBP abs error on free energy density: $(abs((f_exact/(n*n)) - gbp_f/(n*n)))")
	end
    end

    #println("Simple BP absolute error on free energy: ", abs(bp_f - f_exact))
    #println("Generalized BP absolute error on free energy: ", abs(gbp_f - f_exact))
    #println("Loop corrected BP absolute error on free energy: ", abs(f_lc - f_exact))
end

main()