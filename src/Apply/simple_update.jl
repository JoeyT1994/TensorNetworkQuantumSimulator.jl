"""
    simple_update(o, ψ⃗; envs, normalize_tensors = true, sqrt_cutoff, apply_kwargs...)

Simple update of one or two local tensors in the presence of factorized environments under the action of a one- or two-site gate. This is a computationally cheaper but less accurate alternative to `full_update`. It is exact if no truncation is performed.

# Arguments
- `o::Tensor`: The gate to be applied.
- `ψ⃗::Vector{<:Tensor}`: The one or two local tensors being updated.
- `envs::Vector{Tensor}`: The factorized environment tensors associated with the tensors in `ψ⃗`.

# Keyword Arguments
- `normalize_tensors::Bool`: Whether to normalize the updated tensors. Default is `true`.
- `sqrt_cutoff`: Cutoff below which environment eigenvalues are treated as zero when forming their (inverse) square roots. Defaults to `10 * eps(real(scalartype(first(ψ⃗))))`.
- `apply_kwargs...`: Additional keyword arguments passed to the SVD factorization.

# Returns
- `updated_tensors::Vector{Tensor}`: The updated tensors after applying the gate.
- `s_values::Union{Nothing, Tensor}`: The singular values from the SVD (if applicable).
- `err::Number`: The truncation error from the SVD (if applicable).
"""
function simple_update(
        o::Tensor, ψ⃗::Vector{<:Tensor};
        envs, normalize_tensors = true, sqrt_cutoff = nothing, apply_kwargs...
    )

    if length(ψ⃗) == 1
        updated_tensors = Tensor[apply(o, only(ψ⃗))]
        s_values, err = nothing, 0
    else
        sqrt_cutoff = something(sqrt_cutoff, 10 * eps(real(scalartype(first(ψ⃗)))))
        # a tensor in the square roots of its environments, and their inverses
        function gauged(ψv)
            envs_v = filter(env -> hascommoninds(env, ψv), envs)
            @assert all(env -> ndims(env) == 2, envs_v)
            roots = pseudo_sqrt_inv_sqrt.(envs_v; cutoff = sqrt_cutoff)
            return contract([ψv; first.(roots)]), last.(roots)
        end
        (ψᵥ₁, inv_sqrt_envs_v1), (ψᵥ₂, inv_sqrt_envs_v2) = gauged(ψ⃗[1]), gauged(ψ⃗[2])
        sᵥ₁ = commoninds(ψ⃗[1], o)
        sᵥ₂ = commoninds(ψ⃗[2], o)
        Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
        Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
        rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
        oR = apply(o, Rᵥ₁ * Rᵥ₂)
        singular_values! = Ref{Tensor}()
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

    normalize_tensors && (updated_tensors = normalize.(updated_tensors))

    return noprime.(updated_tensors), s_values, err
end
