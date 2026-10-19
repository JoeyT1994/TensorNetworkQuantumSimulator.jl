"""
    apply_gates(circuit::Vector, ψ::Union{TensorNetworkState, BeliefPropagationCache}; bp_update_kwargs = default_bp_update_kwargs(ψ), kwargs...)

Apply a sequence of gates, via simple update (`ITensorNetworksNext.apply_operators` with `BPApplyGate`), to a `TensorNetworkState` or a `BeliefPropagationCache` wrapping a `TensorNetworkState`, using belief propagation to update the environment.

# Arguments
- `circuit::Vector`: A vector of tuples where each tuple contains a gate (as an `ITensor`) and the vertices it acts on.
- `ψ::TensorNetworkState`: The tensor network state to which the gates will be applied.

# Keyword Arguments
- `bp_update_kwargs`: Keyword arguments for updating the belief propagation cache between gates (reasonable defaults are set).
- `apply_kwargs`: Keyword arguments for the gate application, `maxdim` and `cutoff` for bond dimension truncation and `normalize_tensors`.

# Returns
- The updated `TensorNetworkState` or `BeliefPropagationCache`.
"""
function apply_gates(
        circuit::Vector,
        ψ::TensorNetworkState;
        bp_update_kwargs = default_bp_update_kwargs(ψ),
        kwargs...,
    )
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; bp_update_kwargs...)
    return network(apply_gates(circuit, ψ_bpc; bp_update_kwargs, kwargs...))
end

function apply_gates(
        circuit::Vector,
        ψ_bpc::BeliefPropagationCache;
        kwargs...,
    )
    g = graph(ψ_bpc)
    circuit = toitensor(circuit, g, siteinds(network(ψ_bpc)))
    return apply_gates(ITensor[gate[1] for gate in circuit], ψ_bpc; kwargs...)
end

function adapt_gate(gate::ITensor, ψ_bpc::BeliefPropagationCache)
    gate = scalartype(gate) <: Complex ? adapt_scalartype(complex(scalartype(ψ_bpc)), gate) : adapt_scalartype(scalartype(ψ_bpc), gate)
    return adapt(unspecify_type_parameters(datatype(ψ_bpc)), gate)
end

function apply_gates(
        circuit::Vector{<:ITensor},
        ψ_bpc::BeliefPropagationCache;
        bp_update_kwargs = default_bp_update_kwargs(ψ_bpc),
        update_cache = true,
        verbose = false,
        kwargs...,
    )
    bp_update_kwargs = (; default_bp_update_kwargs(ψ_bpc)..., bp_update_kwargs...)
    ψ, ms = apply_gates(
        adapt_gate.(circuit, (ψ_bpc,)), network(ψ_bpc), ψ_bpc.messages;
        bp_update_kwargs, update_cache, kwargs...
    )
    ψ_bpc = BeliefPropagationCache(ITNN.normnetwork(prime, ψ), ms)
    return update_cache ? update(ψ_bpc; bp_update_kwargs...) : ψ_bpc
end

"""
    apply_gates(circuit::Vector{<:ITensor}, ψ, messages; bp_update_kwargs, apply_kwargs = (;), update_cache = true) -> (ψ, messages)

Apply `circuit` to the network `ψ` with the norm-network `messages`, returning both updated. `ψ` and `messages` may be
an `ITensorNetworksMPI.DistITensorNetwork` and its `DistMessageCache`, in which case the call is collective.
"""
function apply_gates(
        circuit::Vector{<:ITensor}, ψ, messages;
        bp_update_kwargs, apply_kwargs = (;), update_cache = true,
    )
    (; maxdim, cutoff, normalize_tensors) = (; maxdim = nothing, cutoff = nothing, normalize_tensors = true, apply_kwargs...)
    ops = [ITensorBase.operator(g, operator_inds(g)...) for g in circuit]
    (; maxiter, tolerance) = bp_update_kwargs
    stopping_criterion = isnothing(tolerance) ? (; maxiter) : (; maxiter, tol = tolerance)
    environment_alg = update_cache ? (; stopping_criterion) : nothing
    ψ, ms = ITNN.apply_operators(
        ops, ψ, messages; trunc = itensor_trunc(; maxdim, cutoff), normalize = normalize_tensors, environment_alg
    )
    # New bonds carry names minted by `svd_trunc`; the bra name of each is reset to the prime of its ket name.
    for e in keys(ITNN.MessageCache(ms))
        ms[e] = rename(ms[e], (ITensorBase.outputnames(ms[e]) .=> prime.(ITensorBase.inputnames(ms[e])))...)
    end
    return ψ, ms
end

const apply_circuit = apply_gates
