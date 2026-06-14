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
- `s_values::Union{Nothing, ITensor}`: The singular values from the SVD (if applicable).
- `err::Number`: The truncation error from the SVD (if applicable).
"""
function simple_update(
        o::ITensor, ψ⃗::Vector{<:ITensor};
        envs, normalize_tensors = true, sqrt_cutoff = nothing, apply_kwargs...
    )

    if length(ψ⃗) == 1
        updated_tensors = ITensor[ITensors.apply(o, only(ψ⃗))]
        s_values, err = nothing, 0
    else
        # When envs is empty no gauging happens and the cutoff is unused, so fall back to
        # the scalartype of the local tensors to materialize a valid default without erroring.
        sqrt_cutoff_ref = isempty(envs) ? first(ψ⃗) : first(envs)
        sqrt_cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(sqrt_cutoff_ref))) : sqrt_cutoff
        envs_v1 = filter(env -> hascommoninds(env, ψ⃗[1]), envs)
        envs_v2 = filter(env -> hascommoninds(env, ψ⃗[2]), envs)
        @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))

        sqrt_inv_sqrt_envs_v1 = pseudo_sqrt_inv_sqrt.(envs_v1; cutoff = sqrt_cutoff)
        sqrt_inv_sqrt_envs_v2 = pseudo_sqrt_inv_sqrt.(envs_v2; cutoff = sqrt_cutoff)
        sqrt_envs_v1, inv_sqrt_envs_v1 = first.(sqrt_inv_sqrt_envs_v1), last.(sqrt_inv_sqrt_envs_v1)
        sqrt_envs_v2, inv_sqrt_envs_v2 = first.(sqrt_inv_sqrt_envs_v2), last.(sqrt_inv_sqrt_envs_v2)

        ψᵥ₁ = contract([ψ⃗[1]; sqrt_envs_v1])
        ψᵥ₂ = contract([ψ⃗[2]; sqrt_envs_v2])
        sᵥ₁ = commoninds(ψ⃗[1], o)
        sᵥ₂ = commoninds(ψ⃗[2], o)
        Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
        Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
        rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
        oR = ITensors.apply(o, Rᵥ₁ * Rᵥ₂)
        singular_values! = Ref(ITensor())
        Rᵥ₁, Rᵥ₂, spec = factorize_svd(
            oR,
            unioninds(rᵥ₁, sᵥ₁);
            ortho = "none",
            singular_values!,
            apply_kwargs...,
        )
        err = spec.truncerr
        s_values = singular_values![]
        Qᵥ₁ = contract([Qᵥ₁; dag.(inv_sqrt_envs_v1)])
        Qᵥ₂ = contract([Qᵥ₂; dag.(inv_sqrt_envs_v2)])
        updated_tensors = [Qᵥ₁ * Rᵥ₁, Qᵥ₂ * Rᵥ₂]
        if normalize_tensors
            s_values = normalize(s_values)
        end
    end

    if normalize_tensors
        for ψᵥ in updated_tensors
            rmul!(ITensors.data(ψᵥ), inv(norm(ψᵥ)))
        end
    end

    return noprime.(updated_tensors), s_values, err
end
