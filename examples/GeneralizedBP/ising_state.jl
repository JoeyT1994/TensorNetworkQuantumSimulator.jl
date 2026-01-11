using TensorNetworkQuantumSimulator

using NamedGraphs: NamedEdge, NamedGraphs
using TensorNetworkQuantumSimulator: dag, virtualinds, normalize, loopcorrected_partitionfunction
using ITensors: prime, ITensor, combiner, replaceind, commoninds, inds, delta, random_itensor, commonind
using Dictionaries: Dictionary
using Random
using NPZ
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


    n= 4
    #βs = [0.1*i for i in 1:22]
    βs = [0.2]
    f_exacts, f_bps, f_lcs, f_gbps = Float64[], Float64[], Float64[], Float64[]
    two_layer =true
    prev_msgs, prev_bp_msgs = nothing, nothing
    g = named_grid((n,n); periodic = false)
    es = edges(g)
    e_dict = Dictionary(es, [Index(2) for e in edges(g)])
    e_dict = merge(e_dict, Dictionary(reverse.(es), collect(values(e_dict))))
    for β in βs
        println("-------------------------------------")
        println("Building Ising state on an $n x $n grid with β = $β ")
        
        Js = Dictionary(collect(edges(g)), [first(src(e)) == first(dst(e)) && isodd(first(src(e))) ? -1.0 : 1.0 for e in edges(g)])

        #Either 
        if !two_layer
            ψ = ising_tensornetwork(g, β; Js)
            z_exact = contract(ψ; alg = "exact")
            f_exact = -log(z_exact)
            push!(f_exacts,f_exact)
            @show f_exact
        else 
            ψ = ising_tensornetwork_rdm(g, β; Js)
            ψψ_tensors = Dictionary(collect(vertices(g)), [reduce(*, norm_factors(ψ, v)) for v in vertices(g)])
            ψψ = TensorNetwork(ψψ_tensors, g)
            z_exact = contract(ψψ; alg = "exact")
            f_exact = -log(z_exact)
            push!(f_exacts, f_exact)
            @show f_exact
        end

        vs = [(2,2), (2,3)]
        rdm = reduced_density_matrix(ψ, vs; alg = "boundarymps", mps_bond_dimension = 16)
        s = siteinds(ψ)
        Ordm = rdm * ITensors.op("Z", first(s[first(vs)]))*ITensors.op("Z", first(s[last(vs)]))*ITensors.op("I", last(s[first(vs)]))*ITensors.op("I", last(s[last(vs)]))
        tr_rdm = rdm * ITensors.op("I", first(s[first(vs)]))*ITensors.op("I", first(s[last(vs)]))*ITensors.op("I", last(s[first(vs)]))*ITensors.op("I", last(s[last(vs)]))
        println("Expectation of ZZ on site pair at temp $β is $((Ordm / tr_rdm)[])")

        for e in edges(g)
            ψv1, ψv2 = ψ[src(e)], ψ[dst(e)]
            cind = commonind(ψv1, ψv2)
            setindex_preserve!(ψ, replaceind(ψv1, cind, e_dict[e]), src(e))
            setindex_preserve!(ψ, replaceind(ψv2, cind, e_dict[e]), dst(e))
        end
        ψ_bpc = prev_bp_msgs == nothing ? BeliefPropagationCache(ψ) : BeliefPropagationCache(ψ, prev_bp_msgs)
        ψ_bpc = update(ψ_bpc)

        loop_size = 4
        bs = construct_gbp_bs(ψ_bpc, loop_size)
        ms = construct_ms(bs)
        ps = all_parents(ms, bs)
        mobius_nos = mobius_numbers(ms, ps)
        ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
        cs = children(ms, ps, bs)
        b_nos = calculate_b_nos(ms, ps, mobius_nos)

        msgs, diffs, gbp_converged = generalized_belief_propagation(ψ_bpc, bs, ms, ps, cs, b_nos, mobius_nos; niters = 1000, rate = 0.3, verbose = false, prev_msgs, tol = 1e-8)
        msgs = normalize_messages(msgs)
        gbp_f = kikuchi_free_energy(ψ_bpc, ms, bs, msgs, cs, b_nos, ps, mobius_nos)
        push!(f_gbps, gbp_f)
        bp_f = -log(partitionfunction(ψ_bpc))

        f_lc = -log(complex(loopcorrected_partitionfunction(ψ_bpc, loop_size)))
        push!(f_lcs, real(f_lc))
        push!(f_bps, bp_f)

        #f_exact = -log(norm_sqr(ψ; alg = "exact"))

        println("Exact free energy density: $(f_exact/(n*n))")

        println("BP abs error on free energy density: $(abs((f_exact/(n*n)) - bp_f/(n*n)))")
        println("Loop Corrected BP abs error on free energy density: $(abs((f_exact/(n*n)) - f_lc/(n*n)))")
        println("GBP abs error on free energy density: $(abs((f_exact/(n*n)) - gbp_f/(n*n)))")

        prev_msgs = copy(msgs)
        prev_bp_msgs = copy(messages(ψ_bpc))
    end

    file_name = "VillainIsingn$(n)"
    if two_layer 
        file_name *= "TwoLayer"
    else
        file_name *= "OneLayer"
    end
    npzwrite("C:\\Users\\Joey\\Documents\\Data\\GBP\\VillainModel\\"*file_name*".npz", betas = βs, f_gbps = f_gbps, f_bps = f_bps, f_lcs = f_lcs, f_exacts = f_exacts)

    #println("Simple BP absolute error on free energy: ", abs(bp_f - f_exact))
    #println("Generalized BP absolute error on free energy: ", abs(gbp_f - f_exact))
    #println("Loop corrected BP absolute error on free energy: ", abs(f_lc - f_exact))
end

main()