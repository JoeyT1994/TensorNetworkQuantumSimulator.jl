using KrylovKit: linsolve

"""
    full_update(o::ITensor, ψ::TensorNetworkState, v⃗; envs, kwargs...)

Full update of two tensors in the presence of environments under the action of a two-site gate. More general than `simple_update` (allows for more accurate non-BP environments), but at a higher computational cost depending on the dimensions of the environment tensors involved.
"""
function full_update(
        o::ITensor,
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

    Qᵥ₁, Rᵥ₁ = MAK.left_orth(
        ψ[v⃗[1]], setdiff(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]), uniqueinds(ψ, v⃗[1]))
    )
    Qᵥ₂, Rᵥ₂ = MAK.left_orth(
        ψ[v⃗[2]], setdiff(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]), uniqueinds(ψ, v⃗[2]))
    )

    extended_envs = vcat(envs, Qᵥ₁, prime(conj(Qᵥ₁)), Qᵥ₂, prime(conj(Qᵥ₂)))
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
        M = Rᵥ₁ * Rᵥ₂
        codomain = inds(Rᵥ₁)
        # Balanced SVD: split the singular values symmetrically (√S into each factor).
        U, S, V, ϵ = MAK.svd_trunc(M, codomain; trunc = itensor_trunc(; apply_kwargs...))
        u = only(commoninds(U, S))
        v = only(commoninds(S, V))
        sqrtS = sqrth_safe(S, (u,), (v,); atol = 0, rtol = 0)
        Rᵥ₁, Rᵥ₂ = U * replaceinds(sqrtS, v => prime(u)), replaceinds(sqrtS, u => prime(u)) * V
        # Relative squared truncation error, from MatrixAlgebraKit's exact discarded-weight `ϵ`
        # (the 2-norm of the discarded singular values) rather than the cancellation-prone
        # `1 - ‖S‖²/‖M‖²` norm subtraction.
        total = norm(M)
        truncation_error = iszero(total) ? zero(real(scalartype(M))) : (ϵ / total)^2
        callback(; singular_values = S, truncation_error)
    end
    ψᵥ₁ = Qᵥ₁ * Rᵥ₁
    ψᵥ₂ = Qᵥ₂ * Rᵥ₂
    return ITensor[ψᵥ₁, ψᵥ₂]
end

"""Calculate the overlap of the gate acting on the previous p and q versus the new p and q in the presence of environments. This is the cost function that optimise_p_q will minimise"""
function fidelity(
        envs::Vector{ITensor},
        p_cur::ITensor,
        q_cur::ITensor,
        p_prev::ITensor,
        q_prev::ITensor,
        gate::ITensor,
    )
    p_sind, q_sind = trycommonind(p_cur, gate), trycommonind(q_cur, gate)
    p_sind_sim, q_sind_sim = sim(p_sind), sim(q_sind)
    gate_sq =
        gate * replaceinds(conj(gate), p_sind => p_sind_sim, q_sind => q_sind_sim)
    term1_tns = vcat(
        [
            p_prev,
            q_prev,
            replaceinds(prime(conj(p_prev)), prime(p_sind) => p_sind_sim),
            replaceinds(prime(conj(q_prev)), prime(q_sind) => q_sind_sim),
            gate_sq,
        ],
        envs,
    )
    sequence = contraction_sequence(term1_tns; alg = "optimal")
    term1 = contract_network(term1_tns; sequence)

    term2_tns = vcat(
        [
            p_cur,
            q_cur,
            replaceinds(prime(conj(p_cur)), prime(p_sind) => p_sind),
            replaceinds(prime(conj(q_cur)), prime(q_sind) => q_sind),
        ],
        envs,
    )
    sequence = contraction_sequence(term2_tns; alg = "optimal")
    term2 = contract_network(term2_tns; sequence)
    term3_tns = vcat([p_prev, q_prev, prime(conj(p_cur)), prime(conj(q_cur)), gate], envs)
    sequence = contraction_sequence(term3_tns; alg = "optimal")
    term3 = contract_network(term3_tns; sequence)

    f = scalar(term3) / sqrt(scalar(term1) * scalar(term2))
    return f * conj(f)
end

"""Do Full Update Sweeping, Optimising the tensors p and q in the presence of the environments envs,
Specifically this functions find the p_cur and q_cur which optimise envs*gate*p*q*conj(prime(p_cur))*conj(prime(q_cur))"""
function optimise_p_q(
        p::ITensor,
        q::ITensor,
        envs::Vector{ITensor},
        o::ITensor;
        nfullupdatesweeps = 10,
        print_fidelity_loss = false,
        envisposdef = true,
        apply_kwargs...,
    )
    pq = apply(o, p * q)
    p_cur, q_cur = MAK.left_orth(
        pq, intersect(inds(pq), inds(p));
        trunc = itensor_trunc(; apply_kwargs...), name = (; tags = tags(commonind(p, q))),
    )

    fstart = print_fidelity_loss ? fidelity(envs, p_cur, q_cur, p, q, o) : 0

    qs_ind = setdiff(inds(q_cur), collect(Iterators.flatten(inds.(vcat(envs, p_cur)))))
    ps_ind = setdiff(inds(p_cur), collect(Iterators.flatten(inds.(vcat(envs, q_cur)))))

    function b(p::ITensor, q::ITensor, o::ITensor, envs::Vector{ITensor}, r::ITensor, s_ind)
        # `r`'s solve indices that the gate `o` does not act on are dangling spectator legs
        # (e.g. the length-1 auxiliary index that makes a definite-charge site tensor
        # symmetry-invariant). Un-prime them on the bra so they trace against the ket instead
        # of surviving as duplicate names under `noprime`.
        spectator = setdiff(s_ind, inds(o))
        r_bra = replaceinds(conj(prime(r)), (prime.(spectator) .=> spectator)...)
        ts = vcat(ITensor[p, q, o, r_bra], envs)
        sequence = contraction_sequence(ts; alg = "optimal")
        return noprime(contract_network(ts; sequence))
    end

    function M_p(envs::Vector{ITensor}, p_q_tensor::ITensor, s_ind, apply_tensor::ITensor)
        ts = vcat(
            ITensor[
                p_q_tensor, replaceinds(prime(conj(p_q_tensor)), (prime.(s_ind) .=> s_ind)...), apply_tensor,
            ],
            envs,
        )
        sequence = contraction_sequence(ts; alg = "optimal")
        return noprime(contract_network(ts; sequence))
    end
    for i in 1:nfullupdatesweeps
        b_vec = b(p, q, o, envs, q_cur, qs_ind)
        M_p_partial = partial(M_p, envs, q_cur, qs_ind)

        p_cur, info = linsolve(
            M_p_partial, b_vec, p_cur; isposdef = envisposdef, ishermitian = false
        )

        b_tilde_vec = b(p, q, o, envs, p_cur, ps_ind)
        M_p_tilde_partial = partial(M_p, envs, p_cur, ps_ind)

        q_cur, info = linsolve(
            M_p_tilde_partial, b_tilde_vec, q_cur; isposdef = envisposdef, ishermitian = false
        )
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

partial = (f, a...; c...) -> (b...) -> f(a..., b...; c...)
