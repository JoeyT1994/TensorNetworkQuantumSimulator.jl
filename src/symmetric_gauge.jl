function symmetric_gauge!(bp_cache::BeliefPropagationCache; regularization = 10 * eps(real(scalartype(bp_cache))), kwargs...)
    tn = network(bp_cache)
    !(tn isa TensorNetworkState) && error("Can only transform TensorNetworkStates to the symmetric gauge")
    for e in edges(tn)
        vsrc, vdst = src(e), dst(e)
        ψvsrc, ψvdst = tn[vsrc], tn[vdst]

        edge_ind = commoninds(ψvsrc, ψvdst)
        edge_ind_sim = sim(edge_ind)

        #Graded messages carry a per-sector parity gauge; fix to the PSD representative
        #before taking roots (dense no-op, cf. pseudo_sqrt_inv_sqrt)
        X_D, X_U = safe_eigen(parity_message_gauge(message(bp_cache, e)); ishermitian = true, cutoff = nothing)
        Y_D, Y_U = safe_eigen(parity_message_gauge(message(bp_cache, reverse(e))); ishermitian = true, cutoff = nothing)
        #Orientation bookkeeping for graded tensors (no-ops for dense data): eigen's D is
        #labelled (lk′, lk) with slot orientations matching U-from-the-left; the U·D·U†
        #sandwich below pairs U's bond with D's FIRST slot, so swap D's labels. And the
        #sandwiched roots/isometries get absorbed into the SAME-side site tensor, so the
        #dag'd representative (the identical hermitian operator in the flipped
        #orientation — itself a bond-gauge choice) is what contracts.
        X_D = replaceinds(X_D, collect(inds(X_D)), reverse(collect(inds(X_D))))
        Y_D = replaceinds(Y_D, collect(inds(Y_D)), reverse(collect(inds(Y_D))))
        X_D, Y_D = map_diag(x -> x + regularization, X_D),
            map_diag(x -> x + regularization, Y_D)

        rootX_D, rootY_D = map_diag(x -> sqrt(x), X_D), map_diag(x -> sqrt(x), Y_D)
        inv_rootX_D, inv_rootY_D = map_diag(x -> inv(sqrt(x)), X_D),
            map_diag(x -> inv(sqrt(x)), Y_D)
        rootX = X_U * rootX_D * prime(dag(X_U))
        rootY = Y_U * rootY_D * prime(dag(Y_U))
        inv_rootX = X_U * inv_rootX_D * prime(dag(X_U))
        inv_rootY = Y_U * inv_rootY_D * prime(dag(Y_U))

        ψvsrc, ψvdst = noprime(ψvsrc * dag(inv_rootX)), noprime(ψvdst * dag(inv_rootY))

        Ce = rootX
        Ce = Ce * replaceinds(rootY, edge_ind, edge_ind_sim)

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
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = update(bp_cache; cache_update_kwargs...)
    bp_cache = symmetric_gauge(bp_cache; kwargs...)
    return network(bp_cache)
end

function symmetrize_and_normalize(bp_cache::BeliefPropagationCache; kwargs...)
    bp_cache = rescale(bp_cache)
    bp_cache = symmetric_gauge(bp_cache; kwargs...)
    return bp_cache
end

function symmetrize_and_bpnormalize(tns::TensorNetworkState; cache_update_kwargs = (; maxiter = 40), kwargs...)
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = update(bp_cache; cache_update_kwargs...)
    bp_cache = symmetrize_and_normalize(bp_cache; kwargs...)
    return network(bp_cache)
end

gauge_and_scale(tns::TensorNetworkState; kwargs...) = symmetrize_and_bpnormalize(tns::TensorNetworkState; kwargs...)