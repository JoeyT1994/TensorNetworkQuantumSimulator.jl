function symmetric_gauge!(bp_cache::BeliefPropagationCache; regularization = 10 * eps(real(scalartype(bp_cache))), kwargs...)
    dtype = datatype(bp_cache)
    tn = network(bp_cache)
    !(tn isa TensorNetworkState) && error("Can only transform TensorNetworkStates to the symmetric gauge")
    for e in edges(tn)
        vsrc, vdst = src(e), dst(e)
        ψvsrc, ψvdst = tn[vsrc], tn[vdst]

        edge_ind = commoninds(ψvsrc, ψvdst)
        edge_ind_sim = sim(edge_ind)

        X_D, X_U = safe_eigen(message(bp_cache, e); ishermitian = true, cutoff = nothing)
        Y_D, Y_U = safe_eigen(message(bp_cache, reverse(e)); ishermitian = true, cutoff = nothing)
        X_D, Y_D = ITensors.map_diag(x -> x + regularization, X_D),
            ITensors.map_diag(x -> x + regularization, Y_D)

        rootX_D, rootY_D = ITensors.map_diag(x -> sqrt(x), X_D), ITensors.map_diag(x -> sqrt(x), Y_D)
        inv_rootX_D, inv_rootY_D = ITensors.map_diag(x -> inv(sqrt(x)), X_D),
            ITensors.map_diag(x -> inv(sqrt(x)), Y_D)
        rootX = X_U * rootX_D * prime(dag(X_U))
        rootY = Y_U * rootY_D * prime(dag(Y_U))
        inv_rootX = X_U * inv_rootX_D * prime(dag(X_U))
        inv_rootY = Y_U * inv_rootY_D * prime(dag(Y_U))

        ψvsrc, ψvdst = noprime(ψvsrc * inv_rootX), noprime(ψvdst * inv_rootY)

        Ce = rootX
        Ce = Ce * replaceinds(rootY, edge_ind, edge_ind_sim)

        U, S, V = svd(Ce, edge_ind; kwargs...)

        new_edge_ind = Index[Index(dim(commoninds(S, U)), tags(first(edge_ind)))]

        ψvsrc = replaceinds(ψvsrc * U, commoninds(S, U), new_edge_ind)
        ψvdst = replaceinds(ψvdst, edge_ind, edge_ind_sim)
        ψvdst = replaceinds(ψvdst * V, commoninds(V, S), new_edge_ind)


        S = replaceinds(
            S,
            [commoninds(S, U)..., commoninds(S, V)...] =>
                [new_edge_ind..., prime(new_edge_ind)...],
        )

        sqrtS = ITensors.map_diag(sqrt, S)
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

function entanglement(
        bp_cache::BeliefPropagationCache,
        e::NamedEdge
    )
    bp_cache = symmetric_gauge(bp_cache)
    ee = 0
    m = message(bp_cache, e)
    for d in diag(m)
        ee -= abs(d) >= eps(real(eltype(m))) ? d * d * log2(d * d) : 0
    end
    return abs(ee)
end

function entanglement(tns::TensorNetworkState, args...; alg)
    algorithm_check(tns, "entanglement", alg)
    return entanglement(Algorithm(alg), tns, args...)
end

function entanglement(alg::Algorithm"bp", tns::TensorNetworkState, e::NamedEdge; cache_update_kwargs = (; maxiter = 40))
    bp_cache = BeliefPropagationCache(tns)
    bp_cache = update(bp_cache)
    return entanglement(bp_cache, e)
end
