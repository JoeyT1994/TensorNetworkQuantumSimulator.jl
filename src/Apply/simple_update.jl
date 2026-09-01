#Backend-specialized fast path for the two-site gate. Returns `nothing` when no
#specialization applies and the generic path below should run.

"""
    simple_update(o, ψ⃗; envs, normalize_tensors = true, sqrt_cutoff, apply_kwargs...)

Simple update of one or two local tensors in the presence of factorized environments under the action of a one- or two-site gate. This is a computationally cheaper but less accurate alternative to `full_update`. It is exact if no truncation is performed.

# Arguments
- `o`: The gate to be applied.
- `ψ⃗::Vector`: The one or two local tensors being updated.
- `envs::Vector`: The factorized environment tensors associated with the tensors in `ψ⃗`.

# Keyword Arguments
- `normalize_tensors::Bool`: Whether to normalize the updated tensors. Default is `true`.
- `sqrt_cutoff`: Cutoff below which environment eigenvalues are treated as zero when forming their (inverse) square roots. Defaults to MatrixAlgebraKit's numerical-rank tolerance, `eps(real(scalartype(first(envs))))^(2 / 3)`.
- `apply_kwargs...`: Additional keyword arguments passed to the SVD factorization.

# Returns
- `updated_tensors::Vector`: The updated tensors after applying the gate.
- `s_values`: The singular values from the SVD (if applicable).
- `err::Number`: The truncation error from the SVD (if applicable).
"""
#Left fold over a factor list: absorbing 2-index environments one at a time is already
#the optimal order, and naming it lets the result be written into a consumed input.
_left_seq(n::Integer) = n <= 1 ? 1 : foldl((a, b) -> [a, b], 2:n; init = 1)

#The dense consumed path absorbs shape-preserving 2-index environments with the BP
#kernel's two-slot arena primitive. Graded tensors and non-consuming calls retain the
#generic seam contraction.
function _environment_chain(t, envs; dest = nothing)
    isempty(envs) && return t
    if t isa Tensor && dest isa Tensor && all(e -> e isa Tensor, envs)
        return Tensors.absorb_chain(t, collect(Tensor, envs), dest)
    end
    return contract([t; envs]; sequence = _left_seq(1 + length(envs)), dest)
end

#Environment gauging shared by the generic path and the fused kernel (kernel_hooks.jl):
#per-vertex √env and √env⁻¹ pairs, with the cutoff defaulted from the environments'
#scalar type (or the local tensors' when envs is empty and the cutoff is unused).
function gauged_env_pairs(ψ⃗::Vector, envs, sqrt_cutoff)
    ref = isempty(envs) ? first(ψ⃗) : first(envs)
    sqrt_cutoff = isnothing(sqrt_cutoff) ? defaulttol(ref) : sqrt_cutoff
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
        all(env -> ndims(env) == 2, envs) ||
            error("simple_update: environments must be 2-index tensors")
        sqrt_envs_v1, inv_sqrt_envs_v1, sqrt_envs_v2, inv_sqrt_envs_v2 =
            gauged_env_pairs(ψ⃗, envs, sqrt_cutoff)

        #Gauging is the last use of the input tensors' data: everything downstream reads
        #the gauged copies. With `consume_inputs` the caller has relinquished them, so the
        #gauged result is written into their storage instead of fresh memory (F1 + F2 less
        #resident per gate). Ownership only — unrelated to which contraction path runs.
        ψᵥ₁ = _environment_chain(ψ⃗[1], sqrt_envs_v1;
            dest = consume_inputs ? ψ⃗[1] : nothing)
        ψᵥ₂ = _environment_chain(ψ⃗[2], sqrt_envs_v2;
            dest = consume_inputs ? ψ⃗[2] : nothing)
        sᵥ₁ = commoninds(ψ⃗[1], o)
        sᵥ₂ = commoninds(ψ⃗[2], o)
        Qᵥ₁, Rᵥ₁ = Tensors.left_orthogonalize(
            ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁); consume_input = consume_inputs,
        )
        Qᵥ₂, Rᵥ₂ = Tensors.left_orthogonalize(
            ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂); consume_input = consume_inputs,
        )
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
        invs1, invs2 = dag.(inv_sqrt_envs_v1), dag.(inv_sqrt_envs_v2)
        Qᵥ₁ = _environment_chain(Qᵥ₁, invs1;
            dest = consume_inputs ? ψ⃗[1] : nothing)
        Qᵥ₂ = _environment_chain(Qᵥ₂, invs2;
            dest = consume_inputs ? ψ⃗[2] : nothing)
        updated_tensors = [
            contract(
                [Qᵥ₁, Rᵥ₁]; sequence = _left_seq(2),
                dest = consume_inputs ? ψ⃗[1] : nothing,
            ),
            contract(
                [Qᵥ₂, Rᵥ₂]; sequence = _left_seq(2),
                dest = consume_inputs ? ψ⃗[2] : nothing,
            ),
        ]
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
