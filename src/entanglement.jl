function renyi_entropy(ρ::AbstractMatrix, α::Real; normalize = true)
    if normalize
        ρ = ρ / tr(ρ)
    end
    λs = eigvals(Hermitian(ρ))
    filter!(λ -> abs(λ) > 10*eps(real(eltype(λs))), λs)
    α == 1 && return -sum(p -> p * log(p), λs)  # von Neumann limit
    return log(sum(λs .^ α)) / (1 - α)
end

function matricize(a::ITensor, row_inds = filter(i -> plev(i) ==0, inds(a)))
    col_inds = prime.(row_inds)
    return ITensors.array(a * ITensors.combiner(row_inds) * ITensors.combiner(col_inds))
end

function renyi_entropy(a::ITensor, row_inds = filter(i -> plev(i) ==0, inds(a)); normalize = true, α = 1)
    return renyi_entropy(matricize(a, row_inds), α)
end

function renyi_entropy(
    bp_cache::BeliefPropagationCache,
    e::NamedEdge;
    α::Real
)
    ee = 0
    m1, m2 = message(bp_cache, e), message(bp_cache, reverse(e))
    edge_ind = only(virtualinds(bp_cache, e))
    root_m2 = first(pseudo_sqrt_inv_sqrt(m2))

    ρ =(m1 * replaceind(root_m2, edge_ind', edge_ind''))* root_m2
    return renyi_entropy(ρ; α)
end

function renyi_entropy(tns::TensorNetworkState, e::NamedEdge; alg, α::Real)
    algorithm_check(tns, "rdm", alg)
    return renyi_entropy(Algorithm(alg), tns, e; α)
end

function renyi_entropy(alg::Algorithm"bp", tns::TensorNetworkState, e::NamedEdge; α::Real)
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = update(bp_cache)
    return renyi_entropy(bp_cache, e; α)
end

function renyi_entropy(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}, verts::Vector; alg, α::Real, kwargs...)
    algorithm_check(ψ, "rdm", alg)
    return renyi_entropy(reduced_density_matrix(ψ, verts; alg, normalize = false, kwargs...); normalize = true, α)
end

second_renyi_entanglement_entropy(args...; kwargs...) = renyi_entropy(args...; kwargs..., α = 2)
von_neumann_entanglement_entropy(args...; kwargs...) = renyi_entropy(args...; kwargs..., α = 1)
