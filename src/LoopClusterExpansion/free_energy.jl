# Free-energy (generating-function) estimate of a single-site observable.
#
#   ⟨Ô⟩ = ∂_ε ln⟨ψ|e^{ε Ô}|ψ⟩|_{ε=0}  ≈  [F(ε) − F(−ε)] / (2ε),
#   F(t) = ln⟨ψ|e^{t Ô}|ψ⟩ = ln‖e^{t Ô/2}|ψ⟩‖²        (Hermitian Ô),
#
# Each F is the loop-corrected free energy of a *genuine norm network*: the gate
# e^{±ε Ô/2} is absorbed (un-normalized) into the ket at the observable site and BP is
# re-solved, so the loop expansion prunes ALL leaves — there is no protected operator
# vertex / "anomalous" cluster of the product-form `expect_clusterexpand`, which is what
# tames the oscillation of that estimator. BP is re-solved on each shifted network so the
# loop corrections also pick up the linear response of the messages.

# F(α-shifted) = ln of the loop-corrected squared norm of e^{α Ô}|ψ⟩, built by absorbing the
# (un-normalized) one-site gate into the ket and re-solving BP for the perturbed network.
function _gated_loop_free_energy(
        ψ::TensorNetworkState, op_string::String, v, α, max_configuration_size::Integer;
        cache_update_kwargs,
    )
    s = only(siteinds(ψ)[v])
    G = ITensors.exp(α * ITensors.op(op_string, s); ishermitian = true)
    bpc = update(BeliefPropagationCache(ψ); cache_update_kwargs...)
    # `normalize_tensors = false`: the un-normalized gated tensor is exactly what makes the
    # squared norm equal the partition function ⟨ψ|e^{2α Ô}|ψ⟩ we want to differentiate.
    bpc, _ = apply_gate!(G, bpc; v⃗ = [v], apply_kwargs = (; normalize_tensors = false))
    bpc = update(bpc; cache_update_kwargs...)
    return log(complex(norm_sqr(bpc; alg = "loopcorrections", max_configuration_size)))
end

"""
    expect_freeenergy(ψ::TensorNetworkState, obs; max_configuration_size, ε = 1e-4, cache_update_kwargs...)
    expect_freeenergy(ψ_bpc::BeliefPropagationCache, obs; max_configuration_size, ε = 1e-4, cache_update_kwargs)

Free-energy / generating-function estimate of a **single-site Hermitian** observable
`obs = (op, vertex[, coeff])`,

    ⟨Ô⟩ = ∂_ε ln⟨ψ|e^{ε Ô}|ψ⟩|_0  ≈  [F(ε) − F(−ε)] / (2ε),

with `F(t) = ln‖e^{t Ô/2}|ψ⟩‖²` the loop-corrected squared norm (`norm_sqr` with
`alg = "loopcorrections"`; `max_configuration_size` counts EDGES). Because each `F` is the
free energy of a genuine norm network, every loop cluster is leaf-free and there is no
protected operator vertex, so this typically converges more smoothly (less oscillation)
than the product-form [`expect_clusterexpand`](@ref). BP is re-solved for each shifted
network so the loop corrections include the linear response of the messages.

`ε` is the central finite-difference step (default `1e-4`). `Ô` must be Hermitian (the
generating-function identity `⟨ψ|e^{εÔ}|ψ⟩ = ‖e^{εÔ/2}|ψ⟩‖²` relies on `Ô† = Ô`).
"""
function expect_freeenergy(
        ψ::TensorNetworkState, obs::Tuple;
        max_configuration_size::Integer, ε::Real = 1e-4,
        cache_update_kwargs = default_bp_update_kwargs(ψ),
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(ψ))
    iszero(coeff) && return zero(coeff)
    length(obs_vs) == 1 ||
        error("expect_freeenergy supports single-site observables only; got $(length(obs_vs)) sites.")
    v = only(obs_vs)
    op_string = only(op_strings)

    Fp = _gated_loop_free_energy(ψ, op_string, v, +ε / 2, max_configuration_size; cache_update_kwargs)
    Fm = _gated_loop_free_energy(ψ, op_string, v, -ε / 2, max_configuration_size; cache_update_kwargs)
    return coeff * (Fp - Fm) / (2ε)
end

function expect_freeenergy(cache::BeliefPropagationCache, obs::Tuple; kwargs...)
    return expect_freeenergy(network(cache), obs; kwargs...)
end

function expect_freeenergy(ψ::Union{TensorNetworkState, BeliefPropagationCache}, observables::Vector{<:Tuple}; kwargs...)
    return map(obs -> expect_freeenergy(ψ, obs; kwargs...), observables)
end
