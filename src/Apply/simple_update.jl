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
- `consume_inputs::Bool`: If `true`, `ψ⃗` is emptied as its entries are absorbed, so a caller that
  is about to overwrite them anyway does not pin a full copy of each site tensor for the duration.
  Halves the peak footprint of a two-site update, which matters on a GPU. The caller must not read
  `ψ⃗` afterwards. Default is `false`.
- `apply_kwargs...`: Additional keyword arguments passed to the SVD factorization.

# Returns
- `updated_tensors::Vector{ITensor}`: The updated tensors after applying the gate.
- `s_values::Union{Nothing, ITensor}`: The singular values from the SVD (if applicable).
- `err::Number`: The truncation error from the SVD (if applicable).
"""
function simple_update(
        o::ITensor, ψ⃗::Vector{<:ITensor};
        envs, normalize_tensors = true, sqrt_cutoff = nothing, consume_inputs = false,
        apply_kwargs...
    )

    if length(ψ⃗) == 1
        updated_tensors = ITensor[ITensors.apply(o, only(ψ⃗))]
        s_values, err = nothing, 0
    else
        if blocked_gates()
            # Mathematically the same as the branch below, but bounded in peak memory. Returns
            # `nothing` for anything it does not specialise, in which case we carry on here.
            blocked = blocked_two_site_update(
                o, ψ⃗; envs, normalize_tensors, sqrt_cutoff, consume_inputs, apply_kwargs...
            )
            isnothing(blocked) || return blocked
        end

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

        # `sᵥ` and the env filters above only need index metadata, so they are read off `ψ⃗`
        # before anything factor-sized is allocated. After the QRs below nothing needs `ψ⃗`
        # itself, which matters on a GPU: for a degree-3 vertex each of `ψᵥ`, `Qᵥ` and the
        # result is the same size as the site tensor, so holding a dead one costs a full
        # factor of peak memory.
        sᵥ₁ = commoninds(ψ⃗[1], o)
        sᵥ₂ = commoninds(ψ⃗[2], o)

        # Each of `ψ⃗[i]`, `ψᵥᵢ` and `Qᵥᵢ` is the same size -- for a degree-3 vertex, `Q` is
        # (χ²)×min(χ², S·χ) = S·χ³, exactly the site tensor. So how many of them are alive at
        # once *is* the peak, and each is released as soon as its successor exists. Rebinding is
        # what drops the reference; the allocator can then reuse the block instead of growing.
        ψᵥ₁ = contract([ψ⃗[1]; sqrt_envs_v1])
        consume_inputs && (ψ⃗[1] = ITensor())
        ψᵥ₂ = contract([ψ⃗[2]; sqrt_envs_v2])
        consume_inputs && (ψ⃗[2] = ITensor())

        # Both index sets are needed before either tensor is released.
        qinds₁ = uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁)
        qinds₂ = uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂)
        Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, qinds₁)
        ψᵥ₁ = ITensor()
        Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, qinds₂)
        ψᵥ₂ = ITensor()

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

        updated_tensors = ITensor[
            absorb_and_close(Qᵥ₁, inv_sqrt_envs_v1, Rᵥ₁),
            absorb_and_close(Qᵥ₂, inv_sqrt_envs_v2, Rᵥ₂),
        ]
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

# `Q * R`, with the inverse-square-root environments ungauged off Q's legs on the way.
#
# Equivalent to `contract([Q; dag.(envs)]) * R`, but each factor-sized intermediate is dropped
# before the next is allocated rather than all of them being handed to `contract` at once. The
# envs are χ×χ, so absorbing them one at a time costs no extra arithmetic -- it only bounds how
# many same-sized copies of the vertex tensor are live at any moment, which is what decides peak
# GPU memory here.
#
# `R` is contracted last on purpose. These contractions all commute -- the envs act on Q's outer
# legs, `R` on the QR index -- so the order only decides how big the intermediates are. Every
# env-absorbed intermediate is exactly Q's size, which is set by the QR rank and so is
# independent of `maxdim`. Closing with `R` first instead makes them all `maxdim`-sized, which is
# smaller when a gate truncates but larger whenever a gate grows the bond, and an unpredictable
# peak is the thing worth avoiding here.
function absorb_and_close(Q::ITensor, envs, R::ITensor)
    for env in envs
        Q = Q * dag(env)
    end
    return Q * R
end
