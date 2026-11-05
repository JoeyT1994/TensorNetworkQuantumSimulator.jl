"""
    apply_gates(circuit, ψ::Union{TensorNetworkState, BeliefPropagationCache}; cache_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)

Apply a sequence of gates, via simple update, to a `TensorNetworkState` or a `BeliefPropagationCache` wrapping a `TensorNetworkState`, using belief propagation to update the environment.

# Arguments
- `circuit`: A [`Circuit`](@ref), or a vector of gates as tuples `(name, vertices)` or `(name, vertices, parameters)`, or as gate tensors.
- `ψ::TensorNetworkState`: The tensor network state to which the gates will be applied.

# Keyword Arguments
- `cache_update_kwargs`: Keyword arguments for updating the belief propagation cache between gates (reasonable defaults are set).
- `apply_kwargs`: Keyword arguments for the gate application, such as `maxdim` and `cutoff` for bond dimension truncation.
- `update_cache`: Whether to update the BP cache between gates (default `true`).
- `verbose`: Print progress and timing of the cache updates (default `false`).

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
    circuit = totensor(circuit, g, siteinds(network(ψ_bpc)))
    return apply_gate_tensors(first.(circuit), ψ_bpc; gate_vertices = last.(circuit), kwargs...)
end

apply_gates(circuit::Vector{<:Tensor}, ψ_bpc::BeliefPropagationCache; kwargs...) = apply_gate_tensors(circuit, ψ_bpc; kwargs...)

"""
    apply_gates!(circuit, ψ; kwargs...)

Like [`apply_gates`](@ref) but consumes the input state or cache: its tensors are overwritten in
place and must not be used afterwards.

`simple_update` writes each gate's gauged tensors into the storage of the inputs it was handed,
since gauging is their last use, which saves the memory of the two tensors per gate. Plain
`apply_gates` already does this for the tensors the gate loop created itself, protecting the
caller's originals with a one-time copy on first touch; `apply_gates!` skips those copies too.
Results are identical either way.
"""
apply_gates!(circuit, ψ; kwargs...) = apply_gates(circuit, ψ; consume_inputs = true, kwargs...)

function adapt_gate(gate::Tensor, ψ_bpc::BeliefPropagationCache)
    gate = scalartype(gate) <: Complex ? adapt(complex(scalartype(ψ_bpc)), gate) : adapt(scalartype(ψ_bpc), gate)
    return adapt(unspecify_type_parameters(datatype(ψ_bpc)), gate)
end

function apply_gate_tensors(
        circuit::Vector,
        ψ_bpc::BeliefPropagationCache;
        gate_vertices::Vector = vertices.(circuit, (network(ψ_bpc),)),
        apply_kwargs = (;),
        cache_update_kwargs = default_bp_update_kwargs(ψ_bpc),
        update_cache = true,
        verbose = false,
        consume_inputs = false,
    )
    # `consume_inputs` hands the caller's own tensors to the gates; otherwise they are protected by a
    # one-time copy on first touch, and the copies are consumed freely thereafter.
    ψ_bpc = consume_inputs ? ψ_bpc : copy(ψ_bpc)
    owned = consume_inputs ? nothing : Set{eltype(vertices(network(ψ_bpc)))}()
    # The vertices gates have acted on since the last cache update: a two-site gate touching one of
    # them needs the cache updated first.
    affected = Set{eltype(vertices(network(ψ_bpc)))}()
    truncation_errors = zeros(length(circuit))
    # the gates' environment absorptions share one set of scratch buffers
    with_kernel_workspace() do
        for (ii, (gate, vs)) in enumerate(zip(circuit, gate_vertices))
            if update_cache && length(vs) >= 2 && any(in(affected), vs)
                verbose && println("Updating BP cache")
                t = @elapsed ψ_bpc = update(ψ_bpc; cache_update_kwargs...)
                verbose && println("Done in $t secs")
                empty!(affected)
            end
            if !isnothing(owned)
                for v in setdiff(vs, owned)
                    setindex_preserve!(ψ_bpc, copy(network(ψ_bpc)[v]), v)
                    push!(owned, v)
                end
            end
            ψ_bpc, truncation_errors[ii] = apply_gate!(adapt_gate(gate, ψ_bpc), ψ_bpc; v⃗ = vs, consume_inputs = true, apply_kwargs)
            union!(affected, vs)
        end
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
        consume_inputs = false,
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

    ψ⃗ = [network(ψ_bpc)[v] for v in v⃗]
    updated_tensors, s_values, err = simple_update(gate, ψ⃗; envs, consume_inputs, apply_kwargs...)
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
