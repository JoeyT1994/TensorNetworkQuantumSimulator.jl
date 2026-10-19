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
        apply_kwargs = (;),
        bp_update_kwargs = default_bp_update_kwargs(ψ_bpc),
        update_cache = true,
        verbose = false,
    )
    (; maxdim, cutoff, normalize_tensors) = (; maxdim = nothing, cutoff = nothing, normalize_tensors = true, apply_kwargs...)
    ops = [ITensorBase.operator(g, operator_inds(g)...) for g in adapt_gate.(circuit, (ψ_bpc,))]
    (; maxiter, tolerance) = (; default_bp_update_kwargs(ψ_bpc)..., bp_update_kwargs...)
    stopping_criterion = isnothing(tolerance) ? (; maxiter) : (; maxiter, tol = tolerance)
    environment_alg = update_cache ? (; stopping_criterion, edges = forest_cover_edge_sequence(graph(ψ_bpc))) : nothing
    ψ, ms = ITNN.apply_operators(
        ops, network(ψ_bpc), ψ_bpc.messages;
        trunc = itensor_trunc(; maxdim, cutoff), normalize = normalize_tensors, environment_alg
    )
    # New bonds carry names minted by `svd_trunc`; the bra name of each is reset to the prime of its ket name.
    foreach(e -> ms[e] = rename(ms[e], (ITensorBase.outputnames(ms[e]) .=> prime.(ITensorBase.inputnames(ms[e])))...), edges(ms))
    ψ_bpc = BeliefPropagationCache(ITNN.normnetwork(prime, ψ), ms)
    return update_cache ? update(ψ_bpc; bp_update_kwargs...) : ψ_bpc
end

const apply_circuit = apply_gates
