"""
    ITensors.apply(circuit::AbstractVector, ψ::ITensorNetwork; bp_update_kwargs = default_posdef_bp_update_kwargs() apply_kwargs = (; maxdim, cutoff))

Apply a circuit to a tensor network.
The circuit should take the form of a vector of Tuples (gate_str, qubits_to_act_on, optional_param) or a vector of ITensors.
Returns the final state and an approximate list of errors when applying each gate
"""
function apply_via_sqrt(
    circuit::AbstractVector,
    ψ::ITensorNetwork;
    bp_update_kwargs = default_qr_bp_update_kwargs(; cache_is_tree = is_tree(ψ)),
    kwargs...,
)
    ψ_bpc = BeliefPropagationCache(ψ)
    initialize_sqrt_bp_messages!(ψ_bpc)
    ψ_bpc = ITensorNetworks.update(ψ_bpc; bp_update_kwargs...)
    ψ_bpc, truncation_errors = apply_via_sqrt(circuit, ψ_bpc; bp_update_kwargs, kwargs...)
    return tensornetwork(ψ_bpc), truncation_errors
end

#Convert a circuit in [(gate_str, sites_to_act_on, params), ...] form to a ITensors and then apply it
function apply_via_sqrt(
    circuit::AbstractVector,
    ψ_bpc::BeliefPropagationCache;
    kwargs...,
)
    gate_vertices = [_tovec(gate[2]) for gate in circuit]
    circuit = toitensor(circuit, siteinds(tensornetwork(ψ_bpc)))
    circuit = adapt(datatype(ψ_bpc)).(circuit)
    return apply_via_sqrt(circuit, ψ_bpc; gate_vertices, kwargs...)
end

"""
    ITensors.apply(circuit::AbstractVector{<:ITensor}, ψ::ITensorNetwork, ψψ::BeliefPropagationCache; apply_kwargs = _default_apply_kwargs, bp_update_kwargs = default_posdef_bp_update_kwargs(), update_cache = true, verbose = false)

Apply a sequence of itensors to the network with its corresponding cache. Apply kwargs should be a NamedTuple containing desired maxdim and cutoff. Update the cache every time an overlapping gate is encountered.
Returns the final state, the updated cache and an approximate list of errors when applying each gate
"""
function apply_via_sqrt(
    circuit::AbstractVector{<:ITensor},
    ψ_bpc::BeliefPropagationCache;
    gate_vertices = [ITensorNetworks.neighbor_vertices(ψ_bpc, gate) for gate in circuit],
    apply_kwargs = _default_apply_kwargs,
    bp_update_kwargs = default_qr_bp_update_kwargs(; cache_is_tree = is_tree(ψ_bpc)),
    update_cache = true,
    verbose = false,
)

    ψ_bpc = copy(ψ_bpc)
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

            t = @timed ψ_bpc = updatecache(ψ_bpc; bp_update_kwargs...)

            affected_indices = Set{Index{Int64}}()

            if verbose
                println("Done in $(t.time) secs")
            end

        end

        # actually apply the gate
        t = @timed ψ_bpc, truncation_errors[ii] = apply!(gate, ψ_bpc; v⃗ = gate_vertices[ii], apply_kwargs)
        affected_indices = union(affected_indices, Set(inds(gate)))

        if verbose
            println(
                "Gate $ii:    Simulation time: $(t.time) secs,    Max χ: $(maxlinkdim(ψ)),     Error: $(truncation_errors[ii])",
            )
        end

    end

    if update_cache
        ψ_bpc = updatecache(ψ_bpc; bp_update_kwargs...)
    end

    return ψ_bpc, truncation_errors
end

#Apply function for a single gate. All apply functions will pass through here
function apply!(
    gate::ITensor,
    ψ_bpc::BeliefPropagationCache;
    v⃗ = ITensorNetworks.neighbor_vertices(ψ_bpc, gate),
    apply_kwargs = _default_apply_kwargs,
)
    # TODO: document each line
    envs = length(v⃗) == 1 ? nothing : incoming_messages(ψ_bpc, PartitionVertex.(v⃗))

    err = 0.0
    s_values = ITensor(1.0)
    function callback(; singular_values, truncation_error)
        err = truncation_error
        s_values = singular_values
        return nothing
    end

    # this is the only call to a lower-level apply that we currently do.
    ψ_bpc = ITensorNetworks.apply(gate, ψ_bpc; v⃗, envs, callback, apply_kwargs...)

    if length(v⃗) == 2
        v1, v2 = v⃗
        pe = partitionedge(ψ_bpc, v1 => v2)
        ind2 = commonind(s_values, ψ_bpc[v1])
        δuv = dag(copy(s_values))
        δuv = replaceind(δuv, ind2, ind2')
        map_diag!(sign, δuv, δuv)
        s_values = denseblocks(s_values) * denseblocks(δuv)
        map_diag!(sqrt, s_values, s_values)
        set_message!(ψ_bpc, pe, dag.(ITensor[s_values]))
        set_message!(ψ_bpc, reverse(pe), ITensor[s_values])
    end
    return ψ_bpc, err
end

function pseudo_inv(t::ITensor, ind1, ind2)
    @assert length(inds(t)) == 2
    U, S, V = svd(t, [ind1])
    map_diag!(x -> pinv(x), S, S)
    return U*S*V
end

# Reduced version
function ITensorNetworks.simple_update_bp(
    o::Union{NamedEdge,ITensor}, ψ::BeliefPropagationCache, v⃗; envs, callback=Returns(nothing), apply_kwargs...
  )
    cutoff = 10 * eps(real(scalartype(ψ[v⃗[1]])))
    envs_v1 = filter(env -> hascommoninds(env, ψ[v⃗[1]]), envs)
    envs_v2 = filter(env -> hascommoninds(env, ψ[v⃗[2]]), envs)
    @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))
        
    inv_envs_v1 = [pseudo_inv(env, inds(env)[1], inds(env)[2]) for env in envs_v1]
    inv_envs_v2 = [pseudo_inv(env, inds(env)[1], inds(env)[2]) for env in envs_v2]
    ψᵥ₁ = contract([ψ[v⃗[1]]; envs_v1])
    ψᵥ₂ = contract([ψ[v⃗[2]]; envs_v2])
    sᵥ₁ = siteinds(ψ, v⃗[1])
    sᵥ₂ = siteinds(ψ, v⃗[2])
    Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
    Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
    rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
    rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
    oR = apply(o, Rᵥ₁ * Rᵥ₂)
    e = v⃗[1] => v⃗[2]
    singular_values! = Ref(ITensor())
    Rᵥ₁, Rᵥ₂, spec = ITensors.factorize_svd(
      oR,
      unioninds(rᵥ₁, sᵥ₁);
      ortho="none",
      tags=edge_tag(e),
      singular_values!,
      apply_kwargs...,
    )
    callback(; singular_values=singular_values![], truncation_error=spec.truncerr)
    Qᵥ₁ = contract([Qᵥ₁; dag.(inv_envs_v1)])
    Qᵥ₂ = contract([Qᵥ₂; dag.(inv_envs_v2)])
    ψᵥ₁ = Qᵥ₁ * Rᵥ₁
    ψᵥ₂ = Qᵥ₂ * Rᵥ₂
    return noprime(ψᵥ₁), noprime(ψᵥ₂)
end