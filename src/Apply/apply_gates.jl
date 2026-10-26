"""
    apply_gates(circuit::Vector, ψ::Union{TensorNetworkState, BeliefPropagationCache}; cache_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)

Apply a sequence of gates, via simple update, to a `TensorNetworkState` or a `BeliefPropagationCache` wrapping a `TensorNetworkState`, using belief propagation to update the environment.

# Arguments
- `circuit::Vector`: The gates, as tuples `(name, vertices)` or `(name, vertices, parameters)`, or as gate tensors.
- `ψ::TensorNetworkState`: The tensor network state to which the gates will be applied.

# Keyword Arguments
- `cache_update_kwargs`: Keyword arguments for updating the belief propagation cache between gates (reasonable defaults are set).
- `apply_kwargs`: Keyword arguments for the gate application, such as `maxdim` and `cutoff` for bond dimension truncation.

# Returns
- A tuple containing the updated `TensorNetworkState` or `BeliefPropagationCache` and a vector of truncation errors for each gate application.
"""
function apply_gates(
        circuit::Vector,
        ψ::TensorNetworkState;
        cache_update_kwargs = default_bp_update_kwargs(ψ),
        kwargs...,
    )
    ψ_bpc = converged_cache(Algorithm("bp"), ψ; cache_update_kwargs)
    ψ_bpc, truncation_errors = apply_gates(circuit, ψ_bpc; cache_update_kwargs, kwargs...)
    return network(ψ_bpc), truncation_errors
end

function apply_gates(
        circuit::Vector,
        ψ_bpc::BeliefPropagationCache;
        kwargs...,
    )
    g = graph(ψ_bpc)
    circuit = toitensor(circuit, g, siteinds(network(ψ_bpc)))
    return apply_gates(first.(circuit), ψ_bpc; gate_vertices = last.(circuit), kwargs...)
end

function adapt_gate(gate::Tensor, ψ_bpc::BeliefPropagationCache)
    gate = scalartype(gate) <: Complex ? adapt(complex(scalartype(ψ_bpc)), gate) : adapt(scalartype(ψ_bpc), gate)
    return adapt(unspecify_type_parameters(datatype(ψ_bpc)), gate)
end

function apply_gates(
        circuit::Vector{<:Tensor},
        ψ_bpc::BeliefPropagationCache;
        gate_vertices::Vector = vertices.(circuit, (network(ψ_bpc),)),
        apply_kwargs = (;),
        cache_update_kwargs = default_bp_update_kwargs(ψ_bpc),
        update_cache = true,
        verbose = false,
    )
    ψ_bpc = copy(ψ_bpc)
    # The vertices gates have acted on since the last cache update: a two-site gate touching one of
    # them needs the cache updated first.
    affected = Set{eltype(vertices(network(ψ_bpc)))}()
    truncation_errors = zeros(length(circuit))
    for (ii, (gate, vs)) in enumerate(zip(circuit, gate_vertices))
        if update_cache && length(vs) >= 2 && any(in(affected), vs)
            verbose && println("Updating BP cache")
            t = @elapsed ψ_bpc = update(ψ_bpc; cache_update_kwargs...)
            verbose && println("Done in $t secs")
            empty!(affected)
        end
        ψ_bpc, truncation_errors[ii] = apply_gate!(adapt_gate(gate, ψ_bpc), ψ_bpc; v⃗ = vs, apply_kwargs)
        union!(affected, vs)
    end

    if update_cache
        ψ_bpc = update(ψ_bpc; cache_update_kwargs...)
    end

    return ψ_bpc, truncation_errors
end

#Apply function for a single gate
function apply_gate!(
        gate::Tensor,
        ψ_bpc::BeliefPropagationCache;
        v⃗ = vertices(gate, network(ψ_bpc)),
        apply_kwargs
    )
    nv = length(v⃗)

    1 <= nv <= 2 || error(
        "apply_gate!: only one- and two-site gates are supported; " *
        "received a gate acting on $nv vertices: $v⃗.",
    )

    if nv == 2
        has_edge(graph(ψ_bpc), NamedEdge(first(v⃗) => last(v⃗))) || error(
            "apply_gate!: cannot apply a two-site gate on the non-adjacent vertices " *
            "$(first(v⃗)) and $(last(v⃗)). Simple update requires the two sites to share an " *
            "edge of the tensor-network graph.",
        )
    end

    envs = nv == 1 ? nothing : incoming_messages(ψ_bpc, v⃗)

    ψ⃗ = Tensor[network(ψ_bpc)[v] for v in v⃗]
    updated_tensors, s_values, err = simple_update(gate, ψ⃗; envs, apply_kwargs...)
    if nv == 2
        v1, v2 = v⃗
        e = NamedEdge(v1 => v2)
        ind2 = commonind(s_values, first(updated_tensors))
        δuv = map_diag(sign, replaceind(dag(s_values), ind2, prime(ind2)))
        s_values = dense(s_values) * dense(δuv)
        setmessage!(ψ_bpc, e, dag(s_values))
        setmessage!(ψ_bpc, reverse(e), s_values)
    end

    for (i, v) in enumerate(v⃗)
        setindex_preserve!(ψ_bpc, updated_tensors[i], v)
    end

    return ψ_bpc, err
end

const apply_circuit = apply_gates
