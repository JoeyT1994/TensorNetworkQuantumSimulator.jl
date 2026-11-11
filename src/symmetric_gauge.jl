# The square root of a Hermitian message and its inverse, its spectrum shifted by `regularization`.
# Graded messages carry a per-sector parity gauge, fixed to the positive semidefinite representative
# first. Graded orientations (no-ops on dense data): `eigen` labels D (lk′, lk) with the slot
# orientations of U from the left, and the U·D·U† sandwich pairs U's bond with D's first slot, so D's
# labels are swapped.
function regularized_roots(m, regularization)
    D, U = eigen(parity_message_gauge(m); ishermitian = true, cutoff = nothing)
    D = replaceinds(D, collect(inds(D)), reverse(collect(inds(D))))
    D = map_diag(x -> x + regularization, D)
    root(f) = U * map_diag(f, D) * prime(dag(U))
    return root(sqrt), root(inv ∘ sqrt)
end

function symmetric_gauge!(bp_cache::BeliefPropagationCache; regularization = 10 * eps(real(scalartype(bp_cache))), kwargs...)
    tn = network(bp_cache)
    bp_cache.factors isa ITNN.NormNetwork || error("Can only transform TensorNetworkStates to the symmetric gauge")
    for e in collect(edges(tn))
        vsrc, vdst = src(e), dst(e)
        ψvsrc, ψvdst = tn[vsrc], tn[vdst]

        edge_ind = commoninds(ψvsrc, ψvdst)
        edge_ind_sim = sim(edge_ind)

        rootX, inv_rootX = regularized_roots(message(bp_cache, e), regularization)
        rootY, inv_rootY = regularized_roots(message(bp_cache, reverse(e)), regularization)

        ψvsrc, ψvdst = noprime(ψvsrc * dag(inv_rootX)), noprime(ψvdst * dag(inv_rootY))

        Ce = rootX * replaceinds(rootY, edge_ind, edge_ind_sim)

        U, S, V = svd(Ce, edge_ind; kwargs...)

        #a fresh index with the SAME space as the SVD bond (graded spaces must match for the
        #replaceinds below); sim keeps the space and mints a new identity
        new_edge_ind = [sim(only(commoninds(S, U)))]

        ψvsrc = replaceinds(ψvsrc * dag(U), commoninds(S, U), new_edge_ind)
        ψvdst = replaceinds(ψvdst, edge_ind, edge_ind_sim)
        ψvdst = replaceinds(ψvdst * dag(V), commoninds(V, S), new_edge_ind)


        S = replaceinds(
            S,
            [commoninds(S, U)..., commoninds(S, V)...] =>
                [new_edge_ind..., prime(new_edge_ind)...],
        )

        sqrtS = map_diag(sqrt, S)
        ψvsrc = noprime(ψvsrc * dag(sqrtS))
        ψvdst = noprime(ψvdst * sqrtS)

        # The new bond takes the old bond's name, which the norm network's bra-name map contains.
        l = only(edge_ind)
        bond = only(new_edge_ind) => l
        tn[vsrc] = ITensorBase.rename(ψvsrc, bond)
        tn[vdst] = ITensorBase.rename(ψvdst, bond)
        S = ITensorBase.rename(S, bond, prime(only(new_edge_ind)) => prime(l))
        bp_cache.messages[e] = ITensorBase.operator(S, (prime(l),), (l,))
        bp_cache.messages[reverse(e)] = ITensorBase.operator(dag(S), (prime(l),), (l,))
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
