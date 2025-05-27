const _default_apply_kwargs =
    (maxdim = Inf, cutoff = 1e-10, normalize = true)

# import ITensorNetworks.apply for less clutter down below
# import ITensors.apply
# import ITensorNetworks.apply

"""
    ITensors.apply(circuit::AbstractVector, ψ::ITensorNetwork; bp_update_kwargs = default_posdef_bp_update_kwargs() apply_kwargs = (; maxdim, cutoff))

Apply a circuit to a tensor network with the cache built and updated internally
"""
function ITensors.apply(
    circuit::AbstractVector,
    ψ::ITensorNetwork;
    bp_update_kwargs = default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)),
    kwargs...,
)
    ψψ = build_bp_cache(ψ; cache_update_kwargs = bp_update_kwargs)
    ψ, ψψ, truncation_errors = apply(circuit, ψ, ψψ; kwargs...)
    return ψ, truncation_errors
end

#Convert a circuit in (gate_str, sites_to_act_on, params) form to ITensors and then apply it
function ITensors.apply(
    circuit::AbstractVector,
    ψ::ITensorNetwork,
    ψψ::BeliefPropagationCache;
    kwargs...,
)
    circuit = toitensor(circuit, siteinds(ψ))
    return apply(circuit, ψ, ψψ; kwargs...)
end

"""
    ITensors.apply(circuit::AbstractVector{<:ITensor}, ψ::ITensorNetwork, ψψ::BeliefPropagationCache; apply_kwargs = _default_apply_kwargs, bp_update_kwargs = default_posdef_bp_update_kwargs(), update_cache = true, verbose = false)

Apply a sequence of itensors to the network with its corresponding cache. Apply kwargs should be a NamedTuple containing desired maxdim and cutoff. Update the cache every time an overlapping gate is encountered.
"""
function ITensors.apply(
    circuit::AbstractVector{<:ITensor},
    ψ::ITensorNetwork,
    ψψ::BeliefPropagationCache;
    apply_kwargs = _default_apply_kwargs,
    bp_update_kwargs = default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)),
    update_cache = true,
    verbose = false,
)

    # merge all the kwargs with the defaults 
    apply_kwargs = merge(_default_apply_kwargs, apply_kwargs)

    # we keep track of the vertices that have been acted on by 2-qubit gates
    # only they increase the counter
    # this is the set that keeps track.
    affected_indices = Set{Index{Int64}}()
    truncation_errors = zeros((length(circuit)))

    # If the circuit is applied in the Heisenberg picture, the circuit needs to already be reversed
    for (ii, gate) in enumerate(circuit)

        # check if the gate is a 2-qubit gate and whether it affects the counter
        # we currently only increment the counter if the gate affects vertices that have already been affected
        cache_update_required = _cacheupdate_check(affected_indices, gate)

        # update the BP cache
        if update_cache && cache_update_required
            if verbose
                println("Updating BP cache")
            end

            t = @timed ψψ = updatecache(ψψ; bp_update_kwargs...)

            affected_indices = Set{Index{Int64}}()

            if verbose
                println("Done in $(t.time) secs")
            end

        end

        # actually apply the gate
        t = @timed ψ, ψψ, truncation_errors[ii] = apply(gate, ψ, ψψ; apply_kwargs)
        affected_indices = union(affected_indices, Set(inds(gate)))

        if verbose
            println(
                "Gate $ii:    Simulation time: $(t.time) secs,    Max χ: $(maxlinkdim(ψ)),     Error: $(truncation_errors[ii])",
            )
        end

    end

    ψψ = updatecache(ψψ; bp_update_kwargs...)

    return ψ, ψψ, truncation_errors
end

#Apply function for a single gate
function ITensors.apply(
    gate::Tuple,
    ψ::ITensorNetwork;
    apply_kwargs = _default_apply_kwargs,
    bp_update_kwargs = default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)),
)
    ψ, ψψ, truncation_error =
        apply(gate, ψ, build_bp_cache(ψ; cache_update_kwargs = bp_update_kwargs); apply_kwargs)
    # because the cache is not passed, we return the state only
    return ψ, truncation_error
end

#Apply function for a single gate
function ITensors.apply(
    gate::Tuple,
    ψ::ITensorNetwork,
    ψψ::BeliefPropagationCache;
    apply_kwargs = _default_apply_kwargs,
    bp_update_kwargs = default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ))
)
    ψ, ψψ, truncation_error = apply(
        toitensor(gate, siteinds(ψ)),
        ψ,
        ψψ;
        apply_kwargs,
    )
    ψψ = updatecache(ψψ; bp_update_kwargs...)
    return ψ, ψψ, truncation_error
end

#Apply function for a single gate. All apply functions will pass through here
function ITensors.apply(
    gate::ITensor,
    ψ::AbstractITensorNetwork,
    ψψ::BeliefPropagationCache;
    apply_kwargs = _default_apply_kwargs,
)
    # TODO: document each line

    ψψ = copy(ψψ)
    ψ = copy(ψ)
    vs = neighbor_vertices(ψ, gate)
    envs = incoming_messages(ψψ, PartitionVertex.(vs))

    err = 0.0
    s_values = ITensor(1.0)
    function callback(; singular_values, truncation_error)
        err = truncation_error
        s_values = singular_values
        return nothing
    end

    # this is the only call to a lower-level apply that we currently do.
    ψ = noprime(ITensorNetworks.apply(gate, ψ; envs, callback, apply_kwargs...))

    ψdag = prime(dag(ψ))
    if length(vs) == 2
        v1, v2 = vs
        pe = partitionedge(ψψ, (v1, "bra") => (v2, "bra"))
        ind2 = commonind(s_values, ψ[v1])
        δuv = dag(copy(s_values))
        δuv = replaceind(δuv, ind2, ind2')
        map_diag!(sign, δuv, δuv)
        s_values = denseblocks(s_values) * denseblocks(δuv)
        set_message!(ψψ, pe, dag.(ITensor[s_values]))
        set_message!(ψψ, reverse(pe), ITensor[s_values])
    end
    for v in vs
        setindex_preserve_graph!(ψψ, copy(ψ[v]), (v, "ket"))
        setindex_preserve_graph!(ψψ, copy(ψdag[v]), (v, "bra"))
    end
    return ψ, ψψ, err
end

#Checker for whether the cache needs updating (overlapping gate encountered)
function _cacheupdate_check(affected_indices::Set, gate::ITensor)
    indices = inds(gate)

    # check if we have a two-site gate and any of the qinds are in the affected_indices. If so update cache
    length(indices) == 4 && any(ind in affected_indices for ind in indices) && return true
    return false
end
