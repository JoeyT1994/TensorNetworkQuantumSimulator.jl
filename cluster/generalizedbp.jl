using ITensors: ITensors, inds, uniqueinds, eachindval, norm
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

#TODO: Get rid of psis, and pass the cache
function update_message(psi_alpha, psi_beta, alpha, beta, msgs, b_nos, ps, cs, ms, bs; rate = 1.0, normalize = true)

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

    for alpha in ps[beta]
        n = elementwise_operation(x -> x^(b_nos[beta]), msgs[(alpha, beta)])
        psi_beta = special_multiply(psi_beta, n)
    end

    ratio = pointwise_division_raise(psi_alpha, psi_beta; power = rate /b_nos[beta])
    m = special_multiply(ratio, msgs[(alpha, beta)])
    if normalize
        m = ITensors.normalize(m)
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

function generalized_belief_propagation(T::BeliefPropagationCache, bs, ms, ps, cs, b_nos, mobius_nos; niters::Int, rate::Number, tol=1e-10)
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
	if diffs[i] < tol
	    tot_iters = i
	    break
	end
    end

    return msgs, diffs[1:tot_iters]
end

function classical_kikuchi_free_energy(ms, bs, msgs, psi_alphas, psi_betas, cs, b_nos, ps, mobius_nos)
    f = 0
    for alpha in 1:length(bs)
        b = b_alpha(alpha, psi_alphas[alpha], msgs, cs, ps)
        R = pointwise_division_raise(b, psi_alphas[alpha])
        R = elementwise_operation(x -> real(x) > 1e-14 ? log(real(x)) : 0, R)
        R = special_multiply(R, b)
        f += sum(R)
    end

    for beta in 1:length(ms)
        b = b_beta(beta, psi_betas[beta], msgs, ps, b_nos)
        R = pointwise_division_raise(b, psi_betas[beta])
        R = elementwise_operation(x -> real(x) > 1e-14 ? log(real(x)) : 0, R)
        R = special_multiply(R, b)
        f += mobius_nos[beta] * sum(R)
    end

    return f
end

#This is the quantum version (allows for complex numbers in messages, agrees with the standard textbook Kicuchi for real positive messages)
function kikuchi_free_energy(T::BeliefPropagationCache, ms, bs, msgs, cs, b_nos, ps, mobius_nos)
    f = 0
    for alpha in 1:length(bs)
        psi_alpha = get_psi(T, bs[alpha])
        b = b_alpha(alpha, psi_alpha, msgs, cs, ps; normalize = false)
        f += log(sum(b))
    end

    for beta in 1:length(ms)
        psi_beta = get_psi(T, ms[beta])
        b = b_beta(beta, psi_beta, msgs, ps, b_nos; normalize = false)
        f += mobius_nos[beta] * log(sum(b))
    end

    return -f
end

function b_alpha(alpha, psi_alpha, msgs, cs, ps; normalize = true)
    b = copy(psi_alpha)
    for beta in cs[alpha]
        for parent_alpha in ps[beta]
            if parent_alpha != alpha
                b = special_multiply(b, msgs[(parent_alpha, beta)])
            end
        end
    end

    if normalize
        b = b / sum(b)
    end
    return b
end

function b_beta(beta, psi_beta, msgs, ps, b_nos; normalize = true)
    b = copy(psi_beta)
    for alpha in ps[beta]
        n = elementwise_operation(x -> x^(b_nos[beta]), msgs[(alpha, beta)])
        b = special_multiply(b, n)
    end
    if normalize
        b = b / sum(b)
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

function prep_gbp(g::NamedGraph, loop_size::Int)
    bs = construct_gbp_bs(g, loop_size)
    ms = construct_ms(bs)
    ps = all_parents(ms, bs)
    mobius_nos = mobius_numbers(ms, ps)
    ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
    cs = children(ms, ps, bs)
    b_nos = calculate_b_nos(ms, ps, mobius_nos)
    return (bs=bs, ms=ms, ps=ps, mobius_nos=mobius_nos, cs=cs, b_nos=b_nos)
end