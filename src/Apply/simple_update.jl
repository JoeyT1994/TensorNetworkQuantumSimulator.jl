"""
    simple_update(o, ψ⃗; envs, normalize_tensors = true, sqrt_cutoff, apply_kwargs...)

Simple update of one or two local tensors in the presence of factorized environments under the action of a one- or two-site gate. This is a computationally cheaper but less accurate alternative to `full_update`. It is exact if no truncation is performed.

# Arguments
- `o::ITensor`: The gate to be applied.
- `ψ⃗::Vector{<:ITensor}`: The one or two local tensors being updated.
- `envs::Vector{ITensor}`: The factorized environment tensors associated with the tensors in `ψ⃗`.

# Keyword Arguments
- `normalize_tensors::Bool`: Whether to normalize the updated tensors. Default is `true`.
- `sqrt_cutoff`: Cutoff below which environment eigenvalues are treated as zero when forming their (inverse) square roots. Defaults to `10 * eps(real(scalartype(first(envs))))`.
- `apply_kwargs...`: Additional keyword arguments passed to the SVD factorization.

# Returns
- `updated_tensors::Vector{ITensor}`: The updated tensors after applying the gate.
- `messages::Union{Nothing, Tuple{ITensor, ITensor}}`: For a two-site gate, the two directed bond messages (`v1 => v2` and its reverse), each a doubled `conj(R) * R` contraction of the reformed factor; `nothing` for a one-site gate.
- `err::Number`: The truncation error from the SVD (if applicable).
"""
function simple_update(
        o::ITensor, ψ⃗::Vector{<:ITensor};
        envs, normalize_tensors = true, sqrt_cutoff = nothing, apply_kwargs...
    )

    if length(ψ⃗) == 1
        updated_tensors = ITensor[apply(o, only(ψ⃗))]
        messages, err = nothing, 0
    else
        # When envs is empty no gauging happens and the cutoff is unused, so fall back to
        # the scalartype of the local tensors to materialize a valid default without erroring.
        sqrt_cutoff_ref = isempty(envs) ? first(ψ⃗) : first(envs)
        sqrt_cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(sqrt_cutoff_ref))) : sqrt_cutoff
        envs_v1 = filter(env -> hascommoninds(env, ψ⃗[1]), envs)
        envs_v2 = filter(env -> hascommoninds(env, ψ⃗[2]), envs)
        @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))

        # The environments are hermitian only up to numerical noise, so project before
        # the square roots (which require hermitian input).
        sqrt_invsqrt = function (env, ψᵥ)
            ket_ind, bra_ind = commonind(env, ψᵥ), uniqueind(env, ψᵥ)
            # Factorize from bra to ket (codomain = bra), the bipartition in which the
            # fermionic message is PSD.
            return sqrth_invsqrth_safe(
                project_hermitian(env, (ket_ind,), (bra_ind,)),
                (bra_ind,), (ket_ind,); atol = sqrt_cutoff, rtol = 0
            )
        end
        sqrt_inv_sqrt_envs_v1 = map(env -> sqrt_invsqrt(env, ψ⃗[1]), envs_v1)
        sqrt_inv_sqrt_envs_v2 = map(env -> sqrt_invsqrt(env, ψ⃗[2]), envs_v2)
        sqrt_envs_v1, inv_sqrt_envs_v1 = first.(sqrt_inv_sqrt_envs_v1), last.(sqrt_inv_sqrt_envs_v1)
        sqrt_envs_v2, inv_sqrt_envs_v2 = first.(sqrt_inv_sqrt_envs_v2), last.(sqrt_inv_sqrt_envs_v2)

        ψᵥ₁ = contract_network([ψ⃗[1]; sqrt_envs_v1])
        ψᵥ₂ = contract_network([ψ⃗[2]; sqrt_envs_v2])
        sᵥ₁ = commoninds(ψ⃗[1], o)
        sᵥ₂ = commoninds(ψ⃗[2], o)
        Qᵥ₁, Rᵥ₁ = MAK.qr_compact(ψᵥ₁, setdiff(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
        Qᵥ₂, Rᵥ₂ = MAK.qr_compact(ψᵥ₂, setdiff(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
        rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
        oR = apply(o, Rᵥ₁ * Rᵥ₂)
        # Balanced SVD: split the singular values symmetrically (√S into each factor) so neither
        # reformed factor is isometric. Each factor carries the bond on `prime(u)` (a primed copy
        # of `u`), so once this function `noprime`s the updated tensors their bond becomes `u`, the
        # ket bond the doubled-contraction messages below leave in place.
        U, S, V, ϵ = MAK.svd_trunc(oR, union(rᵥ₁, sᵥ₁); trunc = itensor_trunc(; apply_kwargs...))
        u = only(commoninds(U, S))
        v = only(commoninds(S, V))
        # Relative squared truncation error, from MatrixAlgebraKit's exact discarded-weight `ϵ`
        # (the 2-norm of the discarded singular values) rather than the cancellation-prone
        # `1 - ‖S‖²/‖oR‖²` norm subtraction. Taken before normalizing `S`, so it uses the raw
        # singular values.
        total = norm(oR)
        err = iszero(total) ? zero(real(scalartype(oR))) : (ϵ / total)^2
        if normalize_tensors
            S = normalize(S)
        end
        sqrtS = sqrth_safe(S, (u,), (v,); atol = 0, rtol = 0)
        Rᵥ₁, Rᵥ₂ = U * replaceinds(sqrtS, v => prime(u)), replaceinds(sqrtS, u => prime(u)) * V
        # The two directed bond messages, each the doubled contraction `conj(R) * R` of that side's
        # reformed factor over its environment-facing legs, leaving the ket bond `u` and the bra
        # bond `prime(u)`. A doubled ket/bra contraction carries the odd-parity sign, so the
        # refreshed message is fermion-sign-correct instead of the bare (sign-blind) singular values.
        messages = (
            conj(Rᵥ₁) * replaceinds(Rᵥ₁, prime(u) => u),
            conj(Rᵥ₂) * replaceinds(Rᵥ₂, prime(u) => u),
        )
        Qᵥ₁ = contract_network([Qᵥ₁; conj.(inv_sqrt_envs_v1)])
        Qᵥ₂ = contract_network([Qᵥ₂; conj.(inv_sqrt_envs_v2)])
        updated_tensors = [Qᵥ₁ * Rᵥ₁, Qᵥ₂ * Rᵥ₂]
    end

    if normalize_tensors
        for ψᵥ in updated_tensors
            rmul!(ψᵥ, inv(norm(ψᵥ)))
        end
    end

    return noprime.(updated_tensors), messages, err
end
