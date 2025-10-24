"""
    apply_gates(circuit::AbstractVector, ψ::ITensorNetwork; bp_update_kwargs = default_posdef_bp_update_kwargs() apply_kwargs = (; maxdim, cutoff))

Apply a circuit (list of gates) to a tensor network.
The circuit should take the form of a vector of Tuples (gate_str, qubits_to_act_on, optional_param) or a vector of ITensors.
Returns the final state and an approximate list of errors when applying each gate
"""
function apply_gates(
    circuit::Vector,
    ψ::TensorNetworkState;
    bp_update_kwargs = default_bp_update_kwargs(ψ),
    kwargs...,
)
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    ψ_bpc, truncation_errors = apply_gates(circuit, ψ_bpc; bp_update_kwargs, kwargs...)
    return network(ψ_bpc), truncation_errors
end

function apply_gates(
    circuit::Vector,
    ψ_bpc::BeliefPropagationCache;
    kwargs...,
)
    gate_vertices = [_tovec(gate[2]) for gate in circuit]
    circuit = toitensor(circuit, siteinds(network(ψ_bpc)))
    circuit = [adapt(ComplexF32, gate) for gate in circuit]
    circuit = [adapt(unspecify_type_parameters(datatype(ψ_bpc)), gate) for gate in circuit]
    return apply_gates(circuit, ψ_bpc; gate_vertices, kwargs...)
end

function apply_gates(
    circuit::Vector{<:ITensor},
    ψ_bpc::BeliefPropagationCache;
    gate_vertices::Vector = neighbor_vertices.((network(ψ_bpc), ), circuit),
    apply_kwargs = (; ),
    bp_update_kwargs = default_bp_update_kwargs(ψ_bpc),
    update_cache = true,
    verbose = false,
    inds_per_site=1,
    transfer_to_gpu = false,
)
    ψ_bpc = copy(ψ_bpc)

    # we keep track of the vertices that have been acted on by 2-qubit gates
    # only they increase the counter
    # this is the set that keeps track.
    affected_indices = Set{Index{Int64}}()
    truncation_errors = zeros((length(circuit)))

    # If the circuit is applied in the Heisenberg picture, the circuit needs to already be reversed
    for (ii, gate) in enumerate(circuit)

        # check if the gate is a 2-qubit gate and whether it affects the counter
        # we currently only increment the counter if the gate affects vertices that have already been affected
        cache_update_required = _cacheupdate_check(affected_indices, gate; inds_per_site)

        # update the BP cache
        if update_cache && cache_update_required
            if verbose
                println("Updating BP cache")
            end

            t = @timed ψ_bpc = update(ψ_bpc; bp_update_kwargs...)

            affected_indices = Set{Index{Int64}}()
            if verbose
                println("Done in $(t.time) secs")
            end

        end

        # actually apply the gate
        t = @timed ψ_bpc, truncation_errors[ii] = apply_gate!(gate, ψ_bpc; v⃗ = gate_vertices[ii], apply_kwargs, transfer_to_gpu)
        affected_indices = union(affected_indices, Set(inds(gate)))
    end

    if update_cache
        ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    end

    return ψ_bpc, truncation_errors
end

#Apply function for a single gate
function apply_gate!(
    gate::ITensor,
    ψ_bpc::BeliefPropagationCache;
    v⃗ = ITensorNetworks.neighbor_vertices(ψ_bpc, gate),
    apply_kwargs = _default_apply_kwargs,
    transfer_to_gpu = false,
)
    envs = length(v⃗) == 1 ? nothing : incoming_messages(ψ_bpc,v⃗)

    if !transfer_to_gpu
        updated_tensors, s_values, err = simple_update(gate, network(ψ_bpc), v⃗; envs, apply_kwargs...)
    else
        updated_tensors, s_values, err = simple_update_cuda(gate, network(ψ_bpc), v⃗; envs, apply_kwargs...)
    end

    if length(v⃗) == 2
        v1, v2 = v⃗
        e = NamedEdge(v1 => v2)
        ind2 = commonind(s_values, first(updated_tensors))
        δuv = dag(copy(s_values))
        δuv = replaceind(δuv, ind2, ind2')
        map_diag!(sign, δuv, δuv)
        s_values = denseblocks(s_values) * denseblocks(δuv)
        setmessage!(ψ_bpc, e, dag(s_values))
        setmessage!(ψ_bpc, reverse(e), s_values)
    end

    for (i, v) in enumerate(v⃗)
        setindex_preserve_graph!(ψ_bpc, updated_tensors[i], v)
    end

    return ψ_bpc, err
end

function simple_update(
    o::ITensor, ψ, v⃗; envs, normalize_tensors = true, apply_kwargs...
  )

    if length(v⃗) == 1
        updated_tensors = ITensor[ITensors.apply(o, ψ[first(v⃗)])]
        s_values, err = nothing, 0
    else
        cutoff = 10 * eps(real(scalartype(ψ[v⃗[1]])))
        envs_v1 = filter(env -> hascommoninds(env, ψ[v⃗[1]]), envs)
        envs_v2 = filter(env -> hascommoninds(env, ψ[v⃗[2]]), envs)
        @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))

        sqrt_inv_sqrt_envs_v1 = pseudo_sqrt_inv_sqrt.(envs_v1)
        sqrt_inv_sqrt_envs_v2 = pseudo_sqrt_inv_sqrt.(envs_v2)
        sqrt_envs_v1, inv_sqrt_envs_v1 = first.(sqrt_inv_sqrt_envs_v1), last.(sqrt_inv_sqrt_envs_v1)
        sqrt_envs_v2, inv_sqrt_envs_v2 = first.(sqrt_inv_sqrt_envs_v2), last.(sqrt_inv_sqrt_envs_v2)

        ψᵥ₁ = contract([ψ[v⃗[1]]; sqrt_envs_v1])
        ψᵥ₂ = contract([ψ[v⃗[2]]; sqrt_envs_v2])
        sᵥ₁ = commoninds(ψ[v⃗[1]], o)
        sᵥ₂ = commoninds(ψ[v⃗[2]], o)
        Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
        Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
        rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
        oR = ITensors.apply(o, Rᵥ₁ * Rᵥ₂)
        e = v⃗[1] => v⃗[2]
        singular_values! = Ref(ITensor())
        Rᵥ₁, Rᵥ₂, spec = factorize_svd(
        oR,
        unioninds(rᵥ₁, sᵥ₁);
        ortho="none",
        tags=edge_tag(e),
        singular_values!,
        apply_kwargs...,
        )
        err = spec.truncerr
        s_values = singular_values![]
        Qᵥ₁ = contract([Qᵥ₁; dag.(inv_sqrt_envs_v1)])
        Qᵥ₂ = contract([Qᵥ₂; dag.(inv_sqrt_envs_v2)])
        updated_tensors = [Qᵥ₁ * Rᵥ₁, Qᵥ₂ * Rᵥ₂]
    end

    if normalize_tensors
        updated_tensors = ITensor[ψᵥ / norm(ψᵥ) for ψᵥ in updated_tensors]
    end

    return noprime.(updated_tensors), s_values, err
end

function simple_update_cuda(
    o::ITensor, ψ, v⃗; envs, normalize_tensors = true, apply_kwargs...
  )

    dtype = datatype(ψ[first(v⃗)])
    o = CUDA.cu(o)
    if length(v⃗) == 1
        ψ1 = CUDA.cu(ψ[first(v⃗)])
        updated_tensors = ITensor[ITensors.apply(o, ψ1)]
        s_values, err = nothing, 0
    else
        ψ1 = CUDA.cu(ψ[first(v⃗)])
        ψ2 = CUDA.cu(ψ[last(v⃗)])
        cutoff = 10 * eps(real(scalartype(ψ[v⃗[1]])))
        envs_v1 = filter(env -> hascommoninds(env, ψ[v⃗[1]]), envs)
        envs_v2 = filter(env -> hascommoninds(env, ψ[v⃗[2]]), envs)
        sqrt_envs_v1 = [
        ITensorNetworks.ITensorsExtensions.map_eigvals(
            sqrt, CUDA.cu(env), inds(env)[1], inds(env)[2]; cutoff, ishermitian=true
        ) for env in envs_v1
        ]
        sqrt_envs_v2 = [
            ITensorNetworks.ITensorsExtensions.map_eigvals(
            sqrt, CUDA.cu(env), inds(env)[1], inds(env)[2]; cutoff, ishermitian=true
        ) for env in envs_v2
        ]
        inv_sqrt_envs_v1 = [
            ITensorNetworks.ITensorsExtensions.map_eigvals(
            inv ∘ sqrt, CUDA.cu(env), inds(env)[1], inds(env)[2]; cutoff, ishermitian=true
        ) for env in envs_v1
        ]
        inv_sqrt_envs_v2 = [
            ITensorNetworks.ITensorsExtensions.map_eigvals(
            inv ∘ sqrt, CUDA.cu(env), inds(env)[1], inds(env)[2]; cutoff, ishermitian=true
        ) for env in envs_v2
        ]
        ψᵥ₁ = contract([ψ1; sqrt_envs_v1])
        ψᵥ₂ = contract([ψ2; sqrt_envs_v2])
        sᵥ₁ = commoninds(ψ1, o)
        sᵥ₂ = commoninds(ψ2, o)
        Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
        Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
        rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
        oR = ITensors.apply(o, Rᵥ₁ * Rᵥ₂)
        e = v⃗[1] => v⃗[2]
        singular_values! = Ref(ITensor())
        Rᵥ₁, Rᵥ₂, spec = factorize_svd(
        oR,
        unioninds(rᵥ₁, sᵥ₁);
        ortho="none",
        tags=edge_tag(e),
        singular_values!,
        apply_kwargs...,
        )
        err = spec.truncerr
        s_values = singular_values![]
        s_values = adapt(dtype)(s_values)
        Qᵥ₁ = contract([Qᵥ₁; dag.(inv_sqrt_envs_v1)])
        Qᵥ₂ = contract([Qᵥ₂; dag.(inv_sqrt_envs_v2)])
        updated_tensors = [Qᵥ₁ * Rᵥ₁, Qᵥ₂ * Rᵥ₂]
    end

    if normalize_tensors
        updated_tensors = ITensor[ψᵥ / norm(ψᵥ) for ψᵥ in updated_tensors]
    end

    updated_tensors = map(adapt(dtype), updated_tensors)

    return noprime.(updated_tensors), s_values, err
end

  #Checker for whether the cache needs updating (overlapping gate encountered)
function _cacheupdate_check(affected_indices::Set, gate::ITensor; inds_per_site=1)
    indices = inds(gate)

    # check if we have a two-site gate and any of the qinds are in the affected_indices. If so update cache
    length(indices) == 4 * inds_per_site && any(ind in affected_indices for ind in indices) && return true
    return false
end
