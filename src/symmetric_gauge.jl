function symmetric_gauge!(bp_cache::BeliefPropagationCache; regularization = 10 * eps(real(scalartype(bp_cache))), kwargs...)
    tn = network(bp_cache)
    !(tn isa TensorNetworkState) && error("Can only transform TensorNetworkStates to the symmetric gauge")
    for e in edges(tn)
        vsrc, vdst = src(e), dst(e)
        ψvsrc, ψvdst = tn[vsrc], tn[vdst]

        edge_ind = commoninds(ψvsrc, ψvdst)
        edge_ind_p = prime.(edge_ind)
        edge_ind_sim = sim.(edge_ind)

        # Hermitian square roots (and pseudo-inverse roots) of the two messages, as
        # operators with codomain on the unprimed and domain on the primed copies of the
        # bond indices. The messages are hermitian only up to numerical noise, so project
        # first. Eigenvalues below `regularization` are clamped to zero.
        me = project_hermitian(message(bp_cache, e), Tuple(edge_ind), Tuple(edge_ind_p))
        mer = project_hermitian(message(bp_cache, reverse(e)), Tuple(edge_ind), Tuple(edge_ind_p))
        rootX, inv_rootX = sqrth_invsqrth_safe(
            me, Tuple(edge_ind), Tuple(edge_ind_p);
            atol = regularization, rtol = 0,
        )
        rootY, inv_rootY = sqrth_invsqrth_safe(
            mer, Tuple(edge_ind), Tuple(edge_ind_p);
            atol = regularization, rtol = 0,
        )

        # Absorb the inverse roots through their domain legs: move the state's bond
        # indices onto the primed names so each pairs with its dual copy, and the
        # result comes back out on the unprimed (codomain) names.
        ψvsrc = replaceinds(ψvsrc, (edge_ind .=> edge_ind_p)...) * inv_rootX
        ψvdst = replaceinds(ψvdst, (edge_ind .=> edge_ind_p)...) * inv_rootY

        # Bond matrix: contract the codomain legs of the two roots (one from each side
        # of the edge, mutually dual). Its open legs are then dual to the state
        # tensors' bond legs, so the SVD factors below absorb without any flips.
        Ce = rootX * replaceinds(rootY, (edge_ind_p .=> edge_ind_sim)...)

        U, S, V = MAK.svd_compact(Ce, edge_ind_p; kwargs...)
        u, v = commoninds(S, U), commoninds(S, V)

        ψvsrc = replaceinds(ψvsrc, (edge_ind .=> edge_ind_p)...) * U
        ψvdst = replaceinds(ψvdst, (edge_ind .=> edge_ind_sim)...) * V

        # Split the singular values symmetrically into both endpoints. `S`'s legs are
        # dual to the tensors' new bond legs on both sides, so these contract directly.
        # `S` is diagonal, so this hits the diagonal fast path on every backend.
        sqrtS = sqrth_safe(S, Tuple(u), Tuple(v); atol = 0, rtol = 0)
        ψvsrc = ψvsrc * sqrtS
        ψvdst = ψvdst * sqrtS

        new_edge_ind = Index[settags(only(v), tags(first(edge_ind)))]
        ψvsrc = replaceinds(ψvsrc, (v .=> new_edge_ind)...)
        ψvdst = replaceinds(ψvdst, (u .=> new_edge_ind)...)
        setindex_preserve!(bp_cache, ψvsrc, vsrc)
        setindex_preserve!(bp_cache, ψvdst, vdst)

        # The gauged network's messages are the singular values, with the unprimed leg
        # carrying the producing side's bond copy (the message convention).
        setmessage!(bp_cache, e, replaceinds(S, only(u) => prime(only(new_edge_ind)), only(v) => only(new_edge_ind)))
        setmessage!(bp_cache, reverse(e), replaceinds(S, only(u) => only(new_edge_ind), only(v) => prime(only(new_edge_ind))))
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