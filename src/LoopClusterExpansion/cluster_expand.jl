# Assembly of the loop cluster expansion for a local observable (Eq. 7 of
# arXiv:2510.05647): combine the per-cluster ratios with the graph-side counting
# numbers from `cluster_counting_numbers`,
#
#     ⟨Ô⟩ ≈ ∏_{r : c(r) ≠ 0} O_r^{c(r)},    O_r = ⟨Ψ|Ô|Ψ⟩_r / ⟨Ψ|Ψ⟩_r,
#
# with O_r evaluated by `_region_ratio` (the same primitive that backs the
# single-cluster `"bp"` expectation, so this is fermion-agnostic: O_r goes through
# `norm_factors` + `contract`, which dispatch on the network type).

"""
    expect_clusterexpand(ψ_bpc::BeliefPropagationCache, obs; max_configuration_size)
    expect_clusterexpand(ψ::TensorNetworkState, obs; max_configuration_size, cache_update_kwargs...)

Loop cluster expansion estimate of a local observable `obs = (ops, vertices[, coeff])`
(Eq. 7 of arXiv:2510.05647). All loop clusters of up to `max_configuration_size`
sites containing the observable support are generated, assigned inclusion-exclusion
counting numbers (`cluster_counting_numbers`), and combined as a product of
per-cluster ratios. Uses the converged BP messages of the norm network on every
cluster boundary.

The `max_configuration_size = |Steiner support|` limit reproduces the ordinary
single-cluster `expect(…; alg = "bp")` value exactly; larger sizes add loop
corrections.
"""
function expect_clusterexpand(
        cache::BeliefPropagationCache, obs::Tuple;
        max_configuration_size::Integer,
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(cache))
    iszero(coeff) && return zero(coeff)

    # Target support: the observable vertices, or their Steiner tree when the
    # observable spans non-adjacent sites (so the seed region is connected).
    target_vs = length(obs_vs) == 1 ? obs_vs :
        collect(vertices(steiner_tree(network(cache), obs_vs)))
    target = Set(target_vs)

    counts = cluster_counting_numbers(graph(cache), target, max_configuration_size)

    logacc = zero(complex(real(scalartype(network(cache)))))
    for (region, c) in counts
        O_r = _region_ratio(cache, region, obs_vs, op_strings)
        # A parity-forbidden numerator makes the whole observable vanish; the parity is
        # region-independent, so a single hit settles it.
        O_r === nothing && return zero(coeff)
        logacc += c * log(complex(O_r))
    end
    return coeff * exp(logacc)
end

function expect_clusterexpand(
        ψ::TensorNetworkState, obs::Tuple;
        cache_update_kwargs = default_bp_update_kwargs(ψ), kwargs...,
    )
    ψ_bpc = update(BeliefPropagationCache(ψ); cache_update_kwargs...)
    return expect_clusterexpand(ψ_bpc, obs; kwargs...)
end

function expect_clusterexpand(ψ::Union{TensorNetworkState, BeliefPropagationCache}, observables::Vector{<:Tuple}; kwargs...)
    return map(obs -> expect_clusterexpand(ψ, obs; kwargs...), observables)
end
