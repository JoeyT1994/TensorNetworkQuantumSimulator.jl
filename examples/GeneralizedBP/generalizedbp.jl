using ITensors: ITensors, inds, uniqueinds, eachindval, norm, plev
using Dictionaries: set!, AbstractDictionary
using TensorNetworkQuantumSimulator: message_diff, bp_factors, contraction_sequence

function get_psi(T::BeliefPropagationCache, r)
    vs = filter(x -> !(x isa NamedEdge), r)
    es = filter(x -> x isa NamedEdge, r)
    e_inds = reduce(vcat, [virtualinds(T, e) for e in es])
    if network(T) isa TensorNetworkState
        e_inds = vcat(e_inds, prime.(e_inds))
    end
    isempty(vs) && return ITensor(scalartype(T), 1.0, e_inds)

    psi = bp_factors(T, collect(vs))
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

    for alpha in ps[beta]
        n = elementwise_operation(x -> x^(b_nos[beta]), msgs[(alpha, beta)])
        psi_beta = special_multiply(psi_beta, n)
    end

    ratio = pointwise_division_raise(psi_alpha, psi_beta; power = rate /b_nos[beta])
    m = special_multiply(ratio, msgs[(alpha, beta)])

    if normalize
        #m = m / sum(m)
        m = ITensors.normalize(m)
        m = _make_hermitian(m)
    end

    return m
end

function update_messages(T::BeliefPropagationCache, msgs, b_nos, ps, cs, ms, bs; kwargs...)
    new_msgs = copy(msgs)
    diff = 0
    for (alpha, beta) in keys(msgs)
        #Parallel or sequential?
        new_msg = update_message(T, alpha, beta, msgs, b_nos, ps, cs, ms, bs; kwargs...)
        diff += message_diff(new_msg, msgs[(alpha, beta)])
        set!(new_msgs, (alpha, beta), new_msg)
    end
    return new_msgs, diff / length(keys(msgs))
end

function generalized_belief_propagation(T::BeliefPropagationCache, bs, ms, ps, cs, b_nos, mobius_nos; niters::Int, rate::Number, tolerance::Number = 1e-12)
    msgs = initialize_messages(ms, bs, ps, T)

    converged = false
    for i in 1:niters
        new_msgs, diff = update_messages(T, msgs, b_nos, ps, cs, ms, bs; normalize = true, rate)
        msgs = new_msgs
        if abs(diff) < tolerance
            println("Converged after $i iterations with $diff average message difference")
            converged = true
            break
        end
    end

    f = kikuchi_free_energy(T, ms, bs, msgs, cs, b_nos, ps, mobius_nos)
    #f = classical_kikuchi_free_energy(T, ms, bs, msgs, cs, b_nos, ps, mobius_nos)

    return f, msgs, converged
end


function classical_kikuchi_free_energy(T, ms, bs, msgs, cs, b_nos, ps, mobius_nos)
    f = 0
    for alpha in 1:length(bs)
        psi_alpha = get_psi(T, bs[alpha])
        b = b_alpha(alpha, psi_alpha, msgs, cs, ps; normalize = true)
        R = pointwise_division_raise(b, psi_alpha)
        R = elementwise_operation(x -> real(x) > 1e-14 ? log(real(x)) : 0, R)
        R = special_multiply(R, b)
        f += sum(R)
    end

    for beta in 1:length(ms)
        psi_beta = get_psi(T, ms[beta])
        b = b_beta(beta, psi_beta, msgs, ps, b_nos; normalize = true)
        R = pointwise_division_raise(b, psi_beta)
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