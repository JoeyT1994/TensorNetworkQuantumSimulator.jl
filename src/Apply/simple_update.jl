"""
    simple_update(o, ψ, v⃗; envs, normalize_tensors = true, apply_kwargs...)

Simple update of one or two tensors in the presence of factorized environments under the action of a one- or two-site gate. This is a computationally cheaper but less accurate alternative to `full_update`. It is exact if no truncation is performed.

# Arguments
- `o::ITensor`: The gate to be applied.
- `ψ::TensorNetworkState`: The tensor network state on which the gate is applied.
- `v⃗::Vector`: The vertices of `ψ` where the gate is applied.
- `envs::Vector{ITensor}`: The factorized environment tensors associated with the tensors in `v⃗`.

# Keyword Arguments
- `normalize_tensors::Bool`: Whether to normalize the updated tensors. Default is `true`.
- `apply_kwargs...`: Additional keyword arguments passed to the SVD factorization.

# Returns
- `updated_tensors::Vector{ITensor}`: The updated tensors after applying the gate.
- `s_values::Union{Nothing, ITensor}`: The singular values from the SVD (if applicable).
- `err::Number`: The truncation error from the SVD (if applicable).
"""
function simple_update(
        o::Tensor, ψ, v⃗; envs, normalize_tensors = true, apply_kwargs...
    )

    if length(v⃗) == 1
        updated_tensors = typeof(o)[noprime(o*ψ[first(v⃗)])]
        s_values, err = nothing, 0
    else
        cutoff = 10 * eps(real(scalartype(ψ[v⃗[1]])))
        envs_v1 = filter(env -> hascommoninds(env, ψ[v⃗[1]]), envs)
        envs_v2 = filter(env -> hascommoninds(env, ψ[v⃗[2]]), envs)
        @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))

        sqrt_inv_sqrt_envs_v1 = pseudo_sqrt_inv_sqrt.(envs_v1)
        sqrt_inv_sqrt_envs_v2 = pseudo_sqrt_inv_sqrt.(envs_v2)
        sqrt_envs_v1, inv_sqrt_envs_v1 = first.(sqrt_inv_sqrt_envs_v1), last.(sqrt_inv_sqrt_envs_v1)
        sqrt_envs_v2, inv_sqrt_envs_v2 = first.(sqrt_inv_sqrt_envs_v2), last.(sqrt_inv_sqrt_envs_v2)

        ψᵥ₁ = contract([ψ[v⃗[1]]; sqrt_envs_v1])
        ψᵥ₂ = contract([ψ[v⃗[2]]; sqrt_envs_v2])
        sᵥ₁ = commoninds(ψ[v⃗[1]], o)
        sᵥ₂ = commoninds(ψ[v⃗[2]], o)
        Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
        Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
        rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
        oR = noprime(o * (Rᵥ₁ * Rᵥ₂))
        e = v⃗[1] => v⃗[2]
        if !(o isa FermionicITensor)
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
        else
            Rᵥ₁, Rᵥ₂, s_values, err = symmetric_svd(
                oR, collect(unioninds(rᵥ₁, sᵥ₁)); apply_kwargs...
            )
        end
        Qᵥ₁ = contract([Qᵥ₁; dag.(inv_sqrt_envs_v1)])
        Qᵥ₂ = contract([Qᵥ₂; dag.(inv_sqrt_envs_v2)])
        updated_tensors = [Qᵥ₁ * Rᵥ₁, Qᵥ₂ * Rᵥ₂]
        if normalize_tensors
            s_values = normalize(s_values)
        end
    end

    if normalize_tensors
        updated_tensors = typeof(o)[ψᵥ / norm(ψᵥ) for ψᵥ in updated_tensors]
    end

    return noprime.(updated_tensors), s_values, err
end
