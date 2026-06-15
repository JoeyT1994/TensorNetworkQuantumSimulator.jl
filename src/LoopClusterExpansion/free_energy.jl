# Free-energy (generating-function) estimate of a single-site observable, exposed through
# `expect(...; alg = "loopcorrections")`.
#
#   ⟨Ô⟩ = ∂_ε ln⟨ψ|e^{ε Ô}|ψ⟩|_{ε=0}  ≈  [F(ε) − F(−ε)] / (2ε),
#   F(t) = ln⟨ψ|e^{t Ô}|ψ⟩ = ln‖e^{t Ô/2}|ψ⟩‖²        (Hermitian Ô),
#
# Each F is the loop-corrected free energy of a *genuine norm network*: the gate
# e^{±ε Ô/2} is absorbed (un-normalized) into the ket at the observable site and BP is
# re-solved, so the loop expansion prunes ALL leaves — there is no protected operator
# vertex, hence no "anomalous" leaf-containing cluster, which is what tends to make this
# estimator converge smoothly. BP is re-solved on each shifted network so the loop
# corrections also pick up the linear response of the messages.

# F(α-shifted) = loop-corrected free energy of e^{α Ô}|ψ⟩, built by absorbing the
# (un-normalized) one-site gate into the ket and re-solving BP for the perturbed network.
# `F = ln Z_BP + Σ_C w_C` is the additive linked-cluster free energy (`loopcorrected_free_energy`),
# the genuine extensive log-partition-function whose smooth ε-dependence makes this estimator
# converge well — NOT `log(loopcorrected_partitionfunction) = ln Z_BP + ln(1 + Σ_C w_C)`, which
# only agrees with it to O(w) and resums the same clusters multiplicatively instead.
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
    return loopcorrected_free_energy(bpc, max_configuration_size)
end

"""
    expect(ψ, obs; alg = "loopcorrections", max_configuration_size, ε = 1e-4, cache_update_kwargs...)

Free-energy / generating-function estimate of a **single-site Hermitian** observable
`obs = (op, vertex[, coeff])`,

    ⟨Ô⟩ = ∂_ε ln⟨ψ|e^{ε Ô}|ψ⟩|_0  ≈  [F(ε) − F(−ε)] / (2ε),

with `F(t) = ln‖e^{t Ô/2}|ψ⟩‖²` the loop-corrected free energy of that norm network
(`loopcorrected_free_energy`, the additive linked-cluster form `ln Z_BP + Σ_C w_C`;
`max_configuration_size` counts EDGES). Because each `F` is the free energy of a genuine
norm network, every loop cluster is leaf-free and there is no protected operator vertex, so
this typically converges smoothly (little oscillation). BP is re-solved for each shifted
network so the loop corrections include the linear response of the messages.

`ε` is the central finite-difference step (default `1e-4`). `Ô` must be Hermitian (the
generating-function identity `⟨ψ|e^{εÔ}|ψ⟩ = ‖e^{εÔ/2}|ψ⟩‖²` relies on `Ô† = Ô`). Accepts a
`TensorNetworkState` or a `BeliefPropagationCache`, and a single observable or a vector of
them; multi-site observables are not supported.
"""
function expect(
        ::Algorithm"loopcorrections",
        ψ::TensorNetworkState,
        obs::Tuple;
        max_configuration_size::Integer, ε::Real = 1e-4,
        cache_update_kwargs = default_bp_update_kwargs(ψ),
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(ψ))
    iszero(coeff) && return zero(coeff)
    length(obs_vs) == 1 ||
        error("expect(...; alg = \"loopcorrections\") supports single-site observables only; got $(length(obs_vs)) sites.")
    v = only(obs_vs)
    op_string = only(op_strings)

    Fp = _gated_loop_free_energy(ψ, op_string, v, +ε / 2, max_configuration_size; cache_update_kwargs)
    Fm = _gated_loop_free_energy(ψ, op_string, v, -ε / 2, max_configuration_size; cache_update_kwargs)
    return coeff * (Fp - Fm) / (2ε)
end

function expect(alg::Algorithm"loopcorrections", cache::BeliefPropagationCache, obs::Tuple; kwargs...)
    return expect(alg, network(cache), obs; kwargs...)
end

function expect(
        alg::Algorithm"loopcorrections",
        ψ::Union{TensorNetworkState, BeliefPropagationCache},
        observables::Vector{<:Tuple};
        kwargs...,
    )
    return map(obs -> expect(alg, ψ, obs; kwargs...), observables)
end
