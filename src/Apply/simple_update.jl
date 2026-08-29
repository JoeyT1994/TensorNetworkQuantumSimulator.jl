#Backend-specialized fast path for the two-site gate. Returns `nothing` when no
#specialization applies and the generic path below should run.
fused_simple_update(o, ψ⃗; kwargs...) = nothing

"""
    simple_update(o, ψ⃗; envs, normalize_tensors = true, sqrt_cutoff, apply_kwargs...)

Simple update of one or two local tensors in the presence of factorized environments under the action of a one- or two-site gate. This is a computationally cheaper but less accurate alternative to `full_update`. It is exact if no truncation is performed.

# Arguments
- `o`: The gate to be applied.
- `ψ⃗::Vector`: The one or two local tensors being updated.
- `envs::Vector`: The factorized environment tensors associated with the tensors in `ψ⃗`.

# Keyword Arguments
- `normalize_tensors::Bool`: Whether to normalize the updated tensors. Default is `true`.
- `sqrt_cutoff`: Cutoff below which environment eigenvalues are treated as zero when forming their (inverse) square roots. Defaults to `10 * eps(real(scalartype(first(envs))))`.
- `apply_kwargs...`: Additional keyword arguments passed to the SVD factorization.

# Returns
- `updated_tensors::Vector`: The updated tensors after applying the gate.
- `s_values`: The singular values from the SVD (if applicable).
- `err::Number`: The truncation error from the SVD (if applicable).
"""
#Environment gauging shared by the generic path and the fused kernel (kernel_hooks.jl):
#per-vertex √env and √env⁻¹ pairs, with the cutoff defaulted from the environments'
#scalar type (or the local tensors' when envs is empty and the cutoff is unused).
function gauged_env_pairs(ψ⃗::Vector, envs, sqrt_cutoff)
    ref = isempty(envs) ? first(ψ⃗) : first(envs)
    sqrt_cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(ref))) : sqrt_cutoff
    ssi1 = pseudo_sqrt_inv_sqrt.(filter(env -> hascommoninds(env, ψ⃗[1]), envs); cutoff = sqrt_cutoff)
    ssi2 = pseudo_sqrt_inv_sqrt.(filter(env -> hascommoninds(env, ψ⃗[2]), envs); cutoff = sqrt_cutoff)
    return first.(ssi1), last.(ssi1), first.(ssi2), last.(ssi2)
end

function simple_update(
        o, ψ⃗::Vector;
        envs, normalize_tensors = true, sqrt_cutoff = nothing, consume_inputs = false,
        apply_kwargs...
    )

    if length(ψ⃗) == 1
        updated_tensors = [apply(o, only(ψ⃗))]
        s_values, err = nothing, 0
    else
        fast = fused_simple_update(o, ψ⃗; envs, normalize_tensors, sqrt_cutoff, consume_inputs, apply_kwargs...)
        fast !== nothing && return fast
        all(env -> ndims(env) == 2, envs) ||
            error("simple_update: environments must be 2-index tensors")
        sqrt_envs_v1, inv_sqrt_envs_v1, sqrt_envs_v2, inv_sqrt_envs_v2 =
            gauged_env_pairs(ψ⃗, envs, sqrt_cutoff)

        ψᵥ₁ = contract([ψ⃗[1]; sqrt_envs_v1])
        ψᵥ₂ = contract([ψ⃗[2]; sqrt_envs_v2])
        sᵥ₁ = commoninds(ψ⃗[1], o)
        sᵥ₂ = commoninds(ψ⃗[2], o)
        Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
        Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
        rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
        oR = apply(o, Rᵥ₁ * Rᵥ₂)
        singular_values! = Ref{Any}(nothing)
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
            rmul!(data(ψᵥ), inv(norm(ψᵥ)))
        end
    end

    return noprime.(updated_tensors), s_values, err
end
