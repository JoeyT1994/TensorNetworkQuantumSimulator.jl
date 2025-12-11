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

    n = 3
    #g = NamedGraphs.NamedGraphGenerators.named_triangular_lattice_graph(n, n)
    g = named_grid((3,3))
    loop_size = 4
    #Build physical site indices for spin-1/2 degrees of freedom
    s = siteinds("S=1/2", g)

    println("Running Generalized Belief Propagation on the norm of a $n x $n random Tensor Network State")

    nsamples = 2
    nconverged_samples = 0
    err_bp, err_gbp, err_lc = 0.0, 0.0, 0.0
    err_bp_marginals, err_gbp_marginals = 0, 0
    err_bp_vertex_marginals, err_gbp_vertex_marginals = 0, 0
    for i in 1:nsamples
        println("-------------------------------------")
        ψ = random_tensornetworkstate(ComplexF64, g, s; bond_dimension = 2)
        ts = Dictionary(collect(vertices(g)), [uniform_random_itensor(ComplexF64, inds(ψ[v])) for v in vertices(g)])
        ψ = TensorNetworkState(TensorNetwork(ts, g), s)
        ψ = normalize(ψ; alg = "bp")
        ψ_bpc = BeliefPropagationCache(ψ)
        ψ_bpc = update(ψ_bpc)

        bs = construct_gbp_bs(ψ_bpc, loop_size)
        #bs = construct_bp_bs(ψ_bpc)
        ms = construct_ms(bs)
        ps = all_parents(ms, bs)
        mobius_nos = mobius_numbers(ms, ps)
        ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
        cs = children(ms, ps, bs)
        b_nos = calculate_b_nos(ms, ps, mobius_nos)

        gbp_f, msgs, gbp_converged = generalized_belief_propagation(ψ_bpc, bs, ms, ps, cs, b_nos, mobius_nos; niters = 500, rate = 0.35)

        if !gbp_converged
            println("GBP did not converge in sample $i")
            continue
        end
        bp_f = -log(partitionfunction(ψ_bpc))

        f_lc = -log(loopcorrected_partitionfunction(ψ_bpc, loop_size))

        f_exact = -log(norm_sqr(ψ; alg = "exact"))

        err_bp += abs(bp_f - f_exact)
        err_gbp += abs(gbp_f - f_exact)
        err_lc += abs(f_lc - f_exact)
        println("Simple BP absolute error on free energy: ", abs(bp_f - f_exact))
        println("Generalized BP absolute error on free energy: ", abs(gbp_f - f_exact))
        println("Loop corrected BP absolute error on free energy: ", abs(f_lc - f_exact))


        es = filter(m -> length(m) == 1, ms)
        _err_bp, _err_gbp = 0, 0
        for e in es
            p_exact = marginal(network(ψ_bpc), only(e))
            beta = only(findall(b -> b == e, ms))
            b_gbp = b_beta(beta, get_psi(ψ_bpc, ms[beta]), msgs, ps, b_nos)
            b_bp = special_multiply(message(ψ_bpc, only(e)), message(ψ_bpc, reverse(only(e))))
            _err_bp += message_diff(b_bp, p_exact)
            _err_gbp += message_diff(b_gbp, p_exact)
        end

        _err_gbp_vertex_marginals, _err_bp_vertex_marginals = 0, 0
        for v in collect(vertices(g))
            p_exact = marginal(network(ψ_bpc), v)
            alpha = only(findall(b -> v ∈ b, bs))
            b_gbp = b_alpha(alpha, get_psi(ψ_bpc, bs[alpha]), msgs, cs, ps)
            psi_alpha = get_psi(ψ_bpc, bs[alpha])
            b_gbp = pointwise_division_raise(b_gbp, psi_alpha)
            b_bp = reduce(*, [message(ψ_bpc, NamedEdge(vn => v)) for vn in neighbors(g, v)])
            _err_gbp_vertex_marginals += message_diff(b_gbp, p_exact)
            _err_bp_vertex_marginals += message_diff(b_bp, p_exact)

        end

        err_bp_marginals += _err_bp / length(es)
        err_gbp_marginals += _err_gbp / length(es)
        err_bp_vertex_marginals += _err_bp_vertex_marginals / length(collect(vertices(g)))
        err_gbp_vertex_marginals += _err_gbp_vertex_marginals / length(collect(vertices(g)))

        println("Average BP error on all single variable marginals is $(_err_bp / length(es))")
        println("Average GBP error on all single variable marginals is $(_err_gbp / length(es))")

        println("Average GBP error on all vertex marginals is $(_err_gbp_vertex_marginals / length(collect(vertices(g))))")
        println("Average BP error on all vertex marginals is $(_err_bp_vertex_marginals / length(collect(vertices(g))))")
        nconverged_samples += 1
    end

    println("-------------------------------------")
    println("Average simple BP absolute error on free energy over $nconverged_samples converged samples: ", err_bp / nconverged_samples)
    println("Average generalized BP absolute error on free energy over $nconverged_samples converged samples: ", err_gbp / nconverged_samples)
    println("Average loop corrected BP absolute error on free energy over $nconverged_samples converged samples: ", err_lc / nconverged_samples)

    println("Average simple BP absolute error on marginals over $nconverged_samples converged samples: ", err_bp_marginals/nconverged_samples)
    println("Average GBP absolute error on marginals over $nconverged_samples converged samples: ", err_gbp_marginals/nconverged_samples)

    println("Average simple BP absolute error on vertex marginals over $nconverged_samples converged samples: ", err_bp_vertex_marginals/nconverged_samples)
    println("Average GBP absolute error on vertex marginals over $nconverged_samples converged samples: ", err_gbp_vertex_marginals/nconverged_samples)
end

main()