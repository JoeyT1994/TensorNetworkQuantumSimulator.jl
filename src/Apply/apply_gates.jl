"""
    apply_gates(circuit::Vector, ψ::Union{TensorNetworkState, BeliefPropagationCache}; bp_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)

Apply a sequence of gates, via simple update, to a `TensorNetworkState` or a `BeliefPropagationCache` wrapping a `TensorNetworkState`, using belief propagation to update the environment.

# Arguments
- `circuit::Vector`: A vector of tuples where each tuple contains a gate (as an `ITensor`) and the vertices it acts on.
- `ψ::TensorNetworkState`: The tensor network state to which the gates will be applied.

# Keyword Arguments
- `bp_update_kwargs`: Keyword arguments for updating the belief propagation cache between gates (reasonable defaults are set).
- `apply_kwargs`: Keyword arguments for the gate application, such as `maxdim` and `cutoff` for bond dimension truncation.

# Returns
- A tuple containing the updated `TensorNetworkState` or `BeliefPropagationCache` and a vector of truncation errors for each gate application.
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
    g = graph(ψ_bpc)
    circuit = toitensor(circuit, g, siteinds(network(ψ_bpc)))
    gate_vertices = [gate[2] for gate in circuit]
    itensors = [gate[1] for gate in circuit]
    return apply_gates(itensors, ψ_bpc; gate_vertices, kwargs...)
end

function adapt_gate(gate::ITensor, ψ_bpc::BeliefPropagationCache)
    gate = scalartype(gate) <: Complex ? adapt(complex(scalartype(ψ_bpc)), gate) : adapt(scalartype(ψ_bpc), gate)
    return adapt(unspecify_type_parameters(datatype(ψ_bpc)), gate)
end

function apply_gates(
        circuit::Vector{<:ITensor},
        ψ_bpc::BeliefPropagationCache;
        gate_vertices::Vector = vertices.(circuit, (network(ψ_bpc),)),
        apply_kwargs = (;),
        bp_update_kwargs = default_bp_update_kwargs(ψ_bpc),
        update_cache = true,
        verbose = false,
    )
    ψ_bpc = copy(ψ_bpc)

    # we keep track of the vertices that have been acted on by 2-qubit gates
    # only they increase the counter
    # this is the set that keeps track.
    affected_vertices = Set{eltype(vertices(network(ψ_bpc)))}()
    truncation_errors = zeros((length(circuit)))

    # If the circuit is applied in the Heisenberg picture, the circuit needs to already be reversed
    for (ii, gate) in enumerate(circuit)

        # check if the gate is a 2-qubit gate and whether it affects the counter
        # we currently only increment the counter if the gate affects vertices that have already been affected
        cache_update_required = length(gate_vertices[ii]) >= 2 && any(vert in affected_vertices for vert in gate_vertices[ii])

        # update the BP cache
        if update_cache && cache_update_required
            if verbose
                println("Updating BP cache")
            end

            t = @timed ψ_bpc = update(ψ_bpc; bp_update_kwargs...)

            empty!(affected_vertices)
            if verbose
                println("Done in $(t.time) secs")
            end

        end

        # actually apply the gate
        gate = adapt_gate(gate, ψ_bpc)
        t = @timed ψ_bpc, truncation_errors[ii] = apply_gate!(gate, ψ_bpc; v⃗ = gate_vertices[ii], apply_kwargs)
        for v in gate_vertices[ii]
            push!(affected_vertices, v)
        end
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
        v⃗ = vertices(gate, network(ψ_bpc)),
        apply_kwargs
    )
    envs = length(v⃗) == 1 ? nothing : incoming_messages(ψ_bpc, v⃗)

    updated_tensors, s_values, err = simple_update(gate, network(ψ_bpc), v⃗; envs, apply_kwargs...)
    invalidate_sequences = false
    if length(v⃗) == 2
        old_bond_dim = dim(commonind(network(ψ_bpc)[v⃗[1]], network(ψ_bpc)[v⃗[2]]))
        new_dim = commonind(updated_tensors[1], updated_tensors[2])
        v1, v2 = v⃗
        e = NamedEdge(v1 => v2)
        ind2 = commonind(s_values, first(updated_tensors))
        δuv = dag(copy(s_values))
        δuv = replaceind(δuv, ind2, ind2')
        map_diag!(sign, δuv, δuv)
        s_values = denseblocks(s_values) * denseblocks(δuv)
        setmessage!(ψ_bpc, e, dag(s_values))
        setmessage!(ψ_bpc, reverse(e), s_values)
        invalidate_sequences = dim(new_dim) != old_bond_dim 
    end

    for (i, v) in enumerate(v⃗)
        setindex_preserve!(ψ_bpc, updated_tensors[i], v; invalidate_sequences)
    end

    return ψ_bpc, err
end

#Checker for whether the cache needs updating (overlapping gate encountered)
function _cacheupdate_check(affected_indices::Set, gate::ITensor; inds_per_site = 1)
    indices = inds(gate)

    # check if we have a two-site gate and any of the qinds are in the affected_indices. If so update cache
    length(indices) == 4 * inds_per_site && any(ind in affected_indices for ind in indices) && return true
    return false
end

const apply_circuit = apply_gates
