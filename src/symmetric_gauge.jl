# The square root of a Hermitian message and its inverse, its spectrum shifted by `regularization`.
function regularized_roots(m::Tensor, regularization)
    D, U = safe_eigen(m; ishermitian = true, cutoff = nothing)
    D = map_diag(x -> x + regularization, D)
    root(f) = U * map_diag(f, D) * prime(dag(U))
    return root(sqrt), root(inv ∘ sqrt)
end

function symmetric_gauge!(bp_cache::BeliefPropagationCache; regularization = 10 * eps(real(scalartype(bp_cache))), kwargs...)
    tn = network(bp_cache)
    tn isa TensorNetworkState || error("Can only transform TensorNetworkStates to the symmetric gauge")
    for e in edges(tn)
        vsrc, vdst = src(e), dst(e)
        ψvsrc, ψvdst = tn[vsrc], tn[vdst]

        edge_ind = commoninds(ψvsrc, ψvdst)
        edge_ind_sim = sim(edge_ind)

        rootX, inv_rootX = regularized_roots(message(bp_cache, e), regularization)
        rootY, inv_rootY = regularized_roots(message(bp_cache, reverse(e)), regularization)

        ψvsrc, ψvdst = noprime(ψvsrc * inv_rootX), noprime(ψvdst * inv_rootY)

        Ce = rootX * replaceinds(rootY, edge_ind, edge_ind_sim)

        U, S, V = svd(Ce, edge_ind; kwargs...)

        new_edge_ind = Index[new_index(dim(commoninds(S, U)); tags = tags(first(edge_ind)))]

        ψvsrc = replaceinds(ψvsrc * U, commoninds(S, U), new_edge_ind)
        ψvdst = replaceinds(ψvdst, edge_ind, edge_ind_sim)
        ψvdst = replaceinds(ψvdst * V, commoninds(V, S), new_edge_ind)


        S = replaceinds(
            S,
            [commoninds(S, U)..., commoninds(S, V)...] =>
                [new_edge_ind..., prime(new_edge_ind)...],
        )

        sqrtS = map_diag(sqrt, S)
        ψvsrc = noprime(ψvsrc * sqrtS)
        ψvdst = noprime(ψvdst * sqrtS)
        setindex_preserve!(bp_cache, ψvsrc, vsrc)
        setindex_preserve!(bp_cache, ψvdst, vdst)

        setmessage!(bp_cache, e, S)
        setmessage!(bp_cache, reverse(e), dag(S))
    end

    return bp_cache
end

function symmetric_gauge(bp_cache::BeliefPropagationCache; kwargs...)
    bp_cache = copy(bp_cache)
    return symmetric_gauge!(bp_cache; kwargs...)
end

function symmetric_gauge(tns::TensorNetworkState; cache_update_kwargs = (; maxiter = 40), kwargs...)
    return network(symmetric_gauge(converged_cache(Algorithm("bp"), tns; cache_update_kwargs); kwargs...))
end

symmetrize_and_normalize(bp_cache::BeliefPropagationCache; kwargs...) = symmetric_gauge(rescale(bp_cache); kwargs...)

function symmetrize_and_bpnormalize(tns::TensorNetworkState; cache_update_kwargs = (; maxiter = 40), kwargs...)
    return network(symmetrize_and_normalize(converged_cache(Algorithm("bp"), tns; cache_update_kwargs); kwargs...))
end

gauge_and_scale(tns::TensorNetworkState; kwargs...) = symmetrize_and_bpnormalize(tns::TensorNetworkState; kwargs...)