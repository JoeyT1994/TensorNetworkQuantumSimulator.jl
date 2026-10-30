using KrylovKit: linsolve

"""
    full_update(o::Tensor, ψ::TensorNetworkState, v⃗; envs, kwargs...)

Full update of two tensors in the presence of environments under the action of a two-site gate. More general than `simple_update` (allows for more accurate non-BP environments), but at a higher computational cost depending on the dimensions of the environment tensors involved.
"""
function full_update(
        o::Tensor,
        ψ::TensorNetworkState,
        v⃗;
        envs,
        nfullupdatesweeps = 10,
        print_fidelity_loss = false,
        envisposdef = false,
        callback = Returns(nothing),
        symmetrize = false,
        apply_kwargs...,
    )

    Qᵥ₁, Rᵥ₁ = factorize(
        ψ[v⃗[1]], uniqueinds(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]), uniqueinds(ψ, v⃗[1]))
    )
    Qᵥ₂, Rᵥ₂ = factorize(
        ψ[v⃗[2]], uniqueinds(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]), uniqueinds(ψ, v⃗[2]))
    )

    extended_envs = vcat(envs, Qᵥ₁, prime(dag(Qᵥ₁)), Qᵥ₂, prime(dag(Qᵥ₂)))
    Rᵥ₁, Rᵥ₂ = optimise_p_q(
        Rᵥ₁,
        Rᵥ₂,
        extended_envs,
        o;
        nfullupdatesweeps,
        print_fidelity_loss,
        envisposdef,
        apply_kwargs...,
    )
    if symmetrize
        singular_values! = Ref{Tensor}()
        Rᵥ₁, Rᵥ₂, spec = factorize_svd(
            Rᵥ₁ * Rᵥ₂,
            inds(Rᵥ₁);
            ortho = "none",
            singular_values!,
            apply_kwargs...,
        )
        callback(; singular_values = singular_values![], truncation_error = spec.truncerr)
    end
    ψᵥ₁ = Qᵥ₁ * Rᵥ₁
    ψᵥ₂ = Qᵥ₂ * Rᵥ₂
    return Tensor[ψᵥ₁, ψᵥ₂]
end

"""Calculate the overlap of the gate acting on the previous p and q versus the new p and q in the presence of environments. This is the cost function that optimise_p_q will minimise"""
function fidelity(
        envs::Vector{Tensor},
        p_cur::Tensor,
        q_cur::Tensor,
        p_prev::Tensor,
        q_prev::Tensor,
        gate::Tensor,
    )
    p_sind, q_sind = commonind(p_cur, gate), commonind(q_cur, gate)
    p_sind_sim, q_sind_sim = sim(p_sind), sim(q_sind)
    gate_sq =
        gate * replaceinds(dag(gate), Index[p_sind, q_sind], Index[p_sind_sim, q_sind_sim])
    term1_tns = vcat(
        [
            p_prev,
            q_prev,
            replaceind(prime(dag(p_prev)), prime(p_sind), p_sind_sim),
            replaceind(prime(dag(q_prev)), prime(q_sind), q_sind_sim),
            gate_sq,
        ],
        envs,
    )
    term1 = contract_optimal(term1_tns)

    term2_tns = vcat(
        [
            p_cur,
            q_cur,
            replaceind(prime(dag(p_cur)), prime(p_sind), p_sind),
            replaceind(prime(dag(q_cur)), prime(q_sind), q_sind),
        ],
        envs,
    )
    term2 = contract_optimal(term2_tns)
    term3 = contract_optimal(vcat([p_prev, q_prev, prime(dag(p_cur)), prime(dag(q_cur)), gate], envs))

    f = scalar(term3) / sqrt(scalar(term1) * scalar(term2))
    return f * conj(f)
end

"""Do Full Update Sweeping, Optimising the tensors p and q in the presence of the environments envs,
Specifically this functions find the p_cur and q_cur which optimise envs*gate*p*q*dag(prime(p_cur))*dag(prime(q_cur))"""
function optimise_p_q(
        p::Tensor,
        q::Tensor,
        envs::Vector{Tensor},
        o::Tensor;
        nfullupdatesweeps = 10,
        print_fidelity_loss = false,
        envisposdef = true,
        apply_kwargs...,
    )
    p_cur, q_cur = factorize(
        apply(o, p * q), inds(p); tags = tags(commonind(p, q)), apply_kwargs...
    )

    fstart = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, o) : 0

    qs_ind = setdiff(inds(q_cur), collect(Iterators.flatten(inds.(vcat(envs, p_cur)))))
    ps_ind = setdiff(inds(p_cur), collect(Iterators.flatten(inds.(vcat(envs, q_cur)))))

    # the right-hand side for the tensor other than `r`, and the linear map acting on it
    rhs(r) = noprime(contract_optimal(vcat(Tensor[p, q, o, dag(prime(r))], envs)))
    lhs(r, s_ind) = x -> noprime(contract_optimal(vcat(Tensor[r, replaceinds(prime(dag(r)), prime(s_ind), s_ind), x], envs)))
    for _ in 1:nfullupdatesweeps
        p_cur, _ = linsolve(lhs(q_cur, qs_ind), rhs(q_cur), p_cur; isposdef = envisposdef, ishermitian = false)
        q_cur, _ = linsolve(lhs(p_cur, ps_ind), rhs(p_cur), q_cur; isposdef = envisposdef, ishermitian = false)
    end

    fend = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, o) : 0

    diff = real(fend - fstart)
    if print_fidelity_loss && diff < -eps(diff) && nfullupdatesweeps >= 1
        println(
            "Warning: Krylov Solver Didn't Find a Better Solution by Sweeping. Something might be amiss.",
        )
    end

    return p_cur, q_cur
end
