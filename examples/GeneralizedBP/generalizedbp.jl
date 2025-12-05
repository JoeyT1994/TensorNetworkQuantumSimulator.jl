using ITensors: ITensors, inds, uniqueinds, eachindval, norm
using Dictionaries: set!, AbstractDictionary
using TensorNetworkQuantumSimulator: message_diff

function update_message(alpha, beta, msgs, psi_alphas, psi_betas, b_nos, ps, cs; rate = 1.0, normalize = true)
    psi_alpha = psi_alphas[alpha]

    for beta in cs[alpha]
        for parent_alpha in ps[beta]
            if parent_alpha != alpha
                psi_alpha = special_multiply(psi_alpha, msgs[(parent_alpha, beta)])
            end
        end
    end

    psi_beta = psi_betas[beta]
    for alpha in ps[beta]
        n = elementwise_operation(x -> x^(b_nos[beta]), msgs[(alpha, beta)])
        psi_beta = special_multiply(psi_beta, n)
    end

    inds = uniqueinds(psi_alpha, psi_beta)
    for ind in inds
        psi_alpha = psi_alpha * ITensor(1.0, ind)
    end

    ratio = pointwise_division_raise(psi_alpha, psi_beta; power = rate /b_nos[beta])
    m = special_multiply(ratio, msgs[(alpha, beta)])
    if normalize
        m = ITensors.normalize(m)
    end

    return m
end

function update_messages(msgs, psi_alphas, psi_betas, b_nos, ps, cs; kwargs...)
    new_msgs = copy(msgs)
    for (alpha, beta) in keys(msgs)
        new_msg = update_message(alpha, beta, msgs, psi_alphas, psi_betas, b_nos, ps, cs; kwargs...)
        set!(new_msgs, (alpha, beta), new_msg)
    end
    return new_msgs
end

function message_diffs(msgs1, msgs2)
    @assert keys(msgs1) == keys(msgs2)
    diff = 0
    for key in keys(msgs1)
        diff += message_diff(msgs1[key], msgs2[key])
    end
    return diff / length(keys(msgs1))
end

function generalized_belief_propagation(T::TensorNetwork, bs, ms, ps, cs, b_nos, mobius_nos; niters::Int, rate::Number)
    psi_alphas = get_psis(bs, T)
    psi_betas = get_psis(ms, T; include_factors = true)
    msgs = initialize_messages(ms, bs, ps, T)

    for i in 1:niters
        new_msgs = update_messages(msgs, psi_alphas, psi_betas, b_nos, ps, cs; normalize = true, rate)

        diff = message_diffs(new_msgs, msgs)

        if i % 10 == 0
            println("Iteration $i")
            println("Average difference in messages following most recent GBP update: $diff")
        end

        msgs = new_msgs
    end

    f = kikuchi_free_energy(ms, bs, msgs, psi_alphas, psi_betas, mobius_nos)
    return f
end


function classical_kikuchi_free_energy(ms, bs, msgs, psi_alphas, psi_betas, mobius_nos)
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
function kikuchi_free_energy(ms, bs, msgs, psi_alphas, psi_betas, mobius_nos)
    f = 0
    for alpha in 1:length(bs)
        b = b_alpha(alpha, psi_alphas[alpha], msgs, cs, ps; normalize = false)
        f += log(sum(b))
    end

    for beta in 1:length(ms)
        b = b_beta(beta, psi_betas[beta], msgs, ps, b_nos; normalize = false)
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