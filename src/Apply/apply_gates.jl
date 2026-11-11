"""
    apply_gates(circuit, ψ::Union{TensorNetworkState, BeliefPropagationCache}; cache_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)

Apply a sequence of gates, via simple update (`ITensorNetworksNext.apply_operators` with `BPApplyGate`), to a `TensorNetworkState` or a `BeliefPropagationCache` wrapping a `TensorNetworkState`, using belief propagation to update the environment.

# Arguments
- `circuit`: A [`Circuit`](@ref), or a vector of gates as tuples `(name, vertices)` or `(name, vertices, parameters)`, or as gate tensors.
- `ψ::TensorNetworkState`: The tensor network state to which the gates will be applied.

# Keyword Arguments
- `cache_update_kwargs`: Keyword arguments for updating the belief propagation cache between gates (reasonable defaults are set).
- `apply_kwargs`: Keyword arguments for the gate application, `maxdim` and `cutoff` for bond dimension truncation and `normalize_tensors`.
- `update_cache`: Whether to update the BP cache between gates (default `true`).

# Returns
- The updated `TensorNetworkState` or `BeliefPropagationCache`.
"""
function apply_gates(
        circuit::Vector,
        ψ::TensorNetworkState;
        cache_update_kwargs = default_bp_update_kwargs(ψ),
        kwargs...,
    )
    ψ_bpc = converged_cache(Algorithm("bp"), ψ; cache_update_kwargs)
    return network(apply_gates(circuit, ψ_bpc; cache_update_kwargs, kwargs...))
end

function apply_gates(
        circuit::Vector,
        ψ_bpc::BeliefPropagationCache;
        kwargs...,
    )
    g = graph(ψ_bpc)
    circuit = totensor(circuit, g, siteinds(network(ψ_bpc)))
    return apply_gate_tensors(first.(circuit), ψ_bpc; kwargs...)
end

"""
    apply_gates!(circuit, ψ; kwargs...)

Like [`apply_gates`](@ref) but consumes the input state or cache: `ITensorNetworksNext.apply_operators!`
overwrites its tensors and messages in place, so it must not be used afterwards.
"""
apply_gates!(circuit, ψ; kwargs...) = apply_gates(circuit, ψ; consume_inputs = true, kwargs...)

function adapt_gate(gate::Tensor, ψ_bpc::BeliefPropagationCache)
    gate = scalartype(gate) <: Complex ? adapt(complex(scalartype(ψ_bpc)), gate) : adapt(scalartype(ψ_bpc), gate)
    return adapt(unspecify_type_parameters(datatype(ψ_bpc)), gate)
end

# The gate as an operator from its unprimed indices to their primes.
gate_operator(gate) = (s = inds(gate; plev = 0); ITensorBase.operator(gate, prime.(s), s))

function apply_gate_tensors(
        circuit::Vector,
        ψ_bpc::BeliefPropagationCache;
        cache_update_kwargs = default_bp_update_kwargs(ψ_bpc),
        update_cache = true,
        verbose = false,
        kwargs...,
    )
    cache_update_kwargs = (; default_bp_update_kwargs(ψ_bpc)..., cache_update_kwargs...)
    ψ, ms = apply_gates(
        adapt_gate.(circuit, (ψ_bpc,)), network(ψ_bpc), ψ_bpc.messages;
        cache_update_kwargs, update_cache, kwargs...
    )
    ψ_bpc = BeliefPropagationCache(ITNN.normnetwork(ITensorBase.prime, ψ), ms)
    return update_cache ? update(ψ_bpc; cache_update_kwargs...) : ψ_bpc
end

"""
    apply_gates(circuit::Vector{<:Tensor}, ψ, messages; cache_update_kwargs, apply_kwargs = (;), update_cache = true, consume_inputs = false) -> (ψ, messages)

Apply `circuit` to the network `ψ` with the norm-network `messages`, returning both updated. `ψ` and `messages` may be
an `ITensorNetworksMPI.DistITensorNetwork` and its `DistMessageCache`, in which case the call is collective.
"""
function apply_gates(
        circuit::Vector{<:Tensor}, ψ, messages;
        cache_update_kwargs, apply_kwargs = (;), update_cache = true, consume_inputs = false,
    )
    (; maxdim, cutoff, normalize_tensors) = (; maxdim = nothing, cutoff = nothing, normalize_tensors = true, apply_kwargs...)
    ops = gate_operator.(circuit)
    (; maxiter, tolerance) = cache_update_kwargs
    stopping_criterion = isnothing(tolerance) ? (; maxiter) : (; maxiter, tol = tolerance)
    environment_alg = update_cache ? (; stopping_criterion) : nothing
    apply_operators = consume_inputs ? ITNN.apply_operators! : ITNN.apply_operators
    ψ, ms = apply_operators(
        ops, ψ, messages;
        trunc = ITensorBaseBackend._mak_trunc(; maxdim, cutoff), normalize = normalize_tensors, environment_alg
    )
    # New bonds carry names minted by `svd_trunc`; the bra name of each is reset to the prime of its ket name.
    for e in keys(ITNN.MessageCache(ms))
        ms[e] = ITensorBase.rename(ms[e], (ITensorBase.outputnames(ms[e]) .=> ITensorBase.prime.(ITensorBase.inputnames(ms[e])))...)
    end
    return ψ, ms
end

const apply_circuit = apply_gates
