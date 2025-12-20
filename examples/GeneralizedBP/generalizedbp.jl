using ITensors: ITensors, inds, uniqueinds, eachindval, norm, plev
using Dictionaries: set!, AbstractDictionary
using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: message_diff, bp_factors, contraction_sequence, contract

function get_psi(T::BeliefPropagationCache, r; kwargs...)
    vs = filter(x -> !(x isa NamedEdge), r)
    es = filter(x -> x isa NamedEdge, r)
    e_inds = reduce(vcat, [virtualinds(T, e) for e in es])
    if network(T) isa TensorNetworkState
        e_inds = vcat(e_inds, prime.(e_inds))
    end
    isempty(vs) && return ITensor(scalartype(T), 1.0, e_inds)

    psi = vcat(bp_factors(T, collect(vs); kwargs...)...)
    seq = contraction_sequence(psi; alg = "optimal")
    psi = contract(psi; sequence = seq)
    return psi
end

function _make_hermitian(A::ITensor)
    A_inds = ITensors.inds(A)
    if length(A_inds) == 2
        return (A + ITensors.swapind(dag(A), first(A_inds), last(A_inds))) / 2
    elseif length(A_inds) == 4
        A_inds_plevnull = filter(i -> plev(i) == 0, A_inds)
        ind1, ind2 = first(A_inds_plevnull), last(A_inds_plevnull)
        return (A + ITensors.swapinds(dag(A), [ind1, ind2], prime.([ind1, ind2]))) / 2
    else
        error("make_hermitian only supports ITensors with 2 or 4 indices")
    end
end

function update_message(T::BeliefPropagationCache, alpha, beta, msgs, b_nos, ps, cs, ms, bs; rate = 1.0, normalize = true)
    psi_alpha = get_psi(T, bs[alpha])
    psi_beta = get_psi(T, ms[beta])

    #TODO: This can be optimized by correct tensor contraction
    for beta in cs[alpha]
        for parent_alpha in ps[beta]
            if parent_alpha != alpha
                psi_alpha = special_multiply(psi_alpha, msgs[(parent_alpha, beta)])
            end
        end
    end

    inds_to_sum_over = uniqueinds(psi_alpha, psi_beta)
    for ind in inds_to_sum_over
        psi_alpha = psi_alpha * ITensor(1.0, ind)
    end

    #psi_alpha = psi_alpha / sum(psi_alpha)

    for alpha in ps[beta]
        n = elementwise_operation(x -> x^(b_nos[beta]), msgs[(alpha, beta)])
        psi_beta = elementwise_multiplication(psi_beta, n)
    end

    #psi_beta = psi_beta / sum(psi_beta)

    ratio = pointwise_division_raise(psi_alpha, psi_beta; power = rate /b_nos[beta])
    m = elementwise_multiplication(ratio, msgs[(alpha, beta)])

    if normalize
        m = ITensors.normalize(m)
        m = _make_hermitian(m)
    end

    return m
end

function update_messages(psi_alphas, psi_betas, msgs, b_nos, ps, cs, ms, bs; kwargs...)
    new_msgs = copy(msgs)
    diff = 0
    for (alpha, beta) in keys(msgs)
        #Parallel or sequential?
        new_msg = update_message(psi_alphas[alpha], psi_betas[beta], alpha, beta, msgs, b_nos, ps, cs, ms, bs; kwargs...)
        diff += message_diff(new_msg, msgs[(alpha, beta)])
        set!(new_msgs, (alpha, beta), new_msg)
    end
    return new_msgs, diff / length(keys(msgs))
end

function generalized_belief_propagation(T::BeliefPropagationCache, bs, ms, ps, cs, b_nos, mobius_nos; niters::Int, rate::Number, tol=1e-12)
    msgs = initialize_messages(ms, bs, ps, T)
    psi_alphas = [get_psi(T, b) for b=bs]
    psi_betas = [get_psi(T, m) for m=ms]
    diffs = zeros(niters)
    tot_iters = niters
    for i in 1:niters
        new_msgs, diffs[i] = update_messages(psi_alphas, psi_betas, msgs, b_nos, ps, cs, ms, bs; normalize = true, rate)

        if i % niters == 0
            println("Average difference in messages following most recent GBP update: $(diffs[i])")
        end
        msgs = new_msgs
	if abs(diffs[i]) < tol
	    tot_iters = i
	    break
	end
    end

    return normalize_messages(msgs), diffs[1:tot_iters], diffs[end] < tol
end


#Needed when computing the free energy
function normalize_messages(msgs)
    new_msgs = copy(msgs)
    for key in keys(msgs)
        set!(new_msgs, key, normalize(msgs[key]))
    end
    return new_msgs
end

#This is the quantum version (allows for complex numbers in messages, agrees with the standard textbook Kicuchi for real positive messages)
function kikuchi_free_energy(T::BeliefPropagationCache, ms, bs, msgs, cs, b_nos, ps, mobius_nos)
    f = 0
    for alpha in 1:length(bs)
        psi_alpha = get_psi(T, bs[alpha])
        b = b_alpha(alpha, psi_alpha, msgs, cs, ps)
        f += log(sum(b))
    end

    for beta in 1:length(ms)
        psi_beta = get_psi(T, ms[beta])
        b = b_beta(beta, psi_beta, msgs, ps, b_nos)
        f += mobius_nos[beta] * log(sum(b))
    end

    return -f
end

function b_alpha(alpha, psi_alpha, msgs, cs, ps)
    b = copy(psi_alpha)
    for beta in cs[alpha]
        for parent_alpha in ps[beta]
            if parent_alpha != alpha
                b = special_multiply(b, msgs[(parent_alpha, beta)])
            end
        end
    end
    return b
end

function b_beta(beta, psi_beta, msgs, ps, b_nos)
    b = copy(psi_beta)
    for alpha in ps[beta]
        n = elementwise_operation(x -> x^(b_nos[beta]), msgs[(alpha, beta)])
        b = elementwise_multiplication(b, n)
    end
    return b
end

function expect_gbp(T, bs, msgs, cs, ps, obs)
    op_strings, verts, _ = TN.collectobservable(obs, graph(T))

    # for now
    @assert length(verts)==1
    alpha = findfirst(b->intersect(b,verts)==Set(verts), bs)
    psi_alpha_num = get_psi(T, bs[alpha]; op_strings = v->op_strings[1])
    num = sum(b_alpha(alpha, psi_alpha_num, msgs, cs, ps; normalize=false))
    denom = sum(b_alpha(alpha, get_psi(T, bs[alpha]), msgs, cs, ps; normalize=false))
    return num/denom
end

function prep_gbp(g::NamedGraph, loop_size::Int; include_factors::Bool = true, prune::Bool = true)
    bs = construct_gbp_bs(g, loop_size; include_factors=include_factors)
    ms = construct_ms(bs)
    ps = all_parents(ms, bs)
    mobius_nos = mobius_numbers(ms, ps)
    if prune
        ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
    end
    cs = children(ms, ps, bs)
    b_nos = calculate_b_nos(ms, ps, mobius_nos)
    return (bs=bs, ms=ms, ps=ps, mobius_nos=mobius_nos, cs=cs, b_nos=b_nos)
end