function updatecache(bp_cache::BeliefPropagationCache; bp_update_kwargs...)
    # merge provided kwargs with the defaults
    bp_update_kwargs = merge(get_global_bp_update_kwargs(), bp_update_kwargs)
    return update(bp_cache; bp_update_kwargs...)
end


function build_bp_cache(ψ::AbstractITensorNetwork; kwargs...)
    bpc = BeliefPropagationCache(QuadraticFormNetwork(ψ))
    # TODO: QuadraticFormNetwork() builds ψIψ network, but for Pauli picture `norm_sqr_network()` is enough
    # https://github.com/ITensor/ITensorNetworks.jl/blob/main/test/test_belief_propagation.jl line 49 to construct the cache without the identities.
    bpc = updatecache(bpc; kwargs...)
    return bpc
end

function ITensors.scalar(bp_cache::BeliefPropagationCache)
    numers, denoms = scalar_factors_quotient(bp_cache)
    isempty(denoms) && return prod(numers)
    return prod(numers) / prod(denoms)
end


function LinearAlgebra.normalize(
    ψ::ITensorNetwork; cache_update_kwargs=get_global_bp_update_kwargs()
)
    ψψ = norm_sqr_network(ψ)
    ψψ_bpc = BeliefPropagationCache(ψψ, group(v -> first(v), vertices(ψψ)))
    ψ, ψψ_bpc = normalize(ψ, ψψ_bpc; cache_update_kwargs)
    return ψ, ψψ_bpc
end

function LinearAlgebra.normalize(
    ψAψ_bpc::BeliefPropagationCache;
    cache_update_kwargs=get_global_bp_update_kwargs(),
    update_cache=true,
    sf::Float64=1.0
)

    if update_cache
        ψAψ_bpc = updatecache(ψAψ_bpc; cache_update_kwargs...)
    end
    ψAψ_bpc = normalize_messages(ψAψ_bpc)
    ψψ = tensornetwork(ψAψ_bpc)

    for v in parent.(partitionvertices(ψAψ_bpc))
        v_ket, v_bra = (v, "ket"), (v, "bra")
        pv = only(partitionvertices(ψAψ_bpc, [v_ket]))
        vn = region_scalar(ψAψ_bpc, pv)
        state = copy(ψψ[v_ket]) / sqrt(sf * vn)
        state_dag = copy(ψψ[v_bra]) / sqrt(sf * vn)
        vertices_states = Dictionary([v_ket, v_bra], [state, state_dag])
        ψAψ_bpc = update_factors(ψAψ_bpc, vertices_states)
    end

    return ψAψ_bpc
end

function LinearAlgebra.normalize(
    ψ::ITensorNetwork,
    ψAψ_bpc::BeliefPropagationCache;
    cache_update_kwargs=get_global_bp_update_kwargs(),
    update_cache=true,
    sf::Float64=1.0,
)
    ψ = copy(ψ)
    if update_cache
        ψAψ_bpc = updatecache(ψAψ_bpc; cache_update_kwargs...)
    end
    ψAψ_bpc = normalize_messages(ψAψ_bpc)
    ψψ = tensornetwork(ψAψ_bpc)

    for v in vertices(ψ)
        v_ket, v_bra = (v, "ket"), (v, "bra")
        pv = only(partitionvertices(ψAψ_bpc, [v_ket]))
        vn = region_scalar(ψAψ_bpc, pv)
        state = copy(ψψ[v_ket]) / sqrt(sf * vn)
        state_dag = copy(ψψ[v_bra]) / sqrt(sf * vn)
        vertices_states = Dictionary([v_ket, v_bra], [state, state_dag])
        ψAψ_bpc = update_factors(ψAψ_bpc, vertices_states)
        ψ[v] = state
    end

    return ψ, ψAψ_bpc
end

function normalize_messages(bp_cache::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
    bp_cache = copy(bp_cache)
    mts = messages(bp_cache)
    for pe in pes
        me, mer = only(mts[pe]), only(mts[reverse(pe)])
        set!(mts, pe, ITensor[me/norm(me)])
        set!(mts, reverse(pe), ITensor[mer/norm(mer)])
        n = region_scalar(bp_cache, pe)
        set!(mts, pe, ITensor[(1/sqrt(n))*me])
        set!(mts, reverse(pe), ITensor[(1/sqrt(n))*mer])
    end
    return bp_cache
end

function normalize_message(bp_cache::BeliefPropagationCache, pe::PartitionEdge)
    return normalize_messages(bp_cache, PartitionEdge[pe])
end

function normalize_messages(bp_cache::BeliefPropagationCache)
    return normalize_messages(bp_cache, partitionedges(partitioned_tensornetwork(bp_cache)))
end


function make_eigs_real(A::ITensor)
    return map_eigvals(x -> real(x), A, first(inds(A)), last(inds(A)); ishermitian=true)
end

function make_eigs_positiv(A::ITensor, tol::Real=1e-14)
    return map_eigvals(x -> max(x, tol), A, first(inds(A)), last(inds(A)); ishermitian=true)
end