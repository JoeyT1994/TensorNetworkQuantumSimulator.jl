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
    circuit = totensor(circuit, g, siteinds(network(ψ_bpc)))
    gate_vertices = [gate[2] for gate in circuit]
    tensors = [gate[1] for gate in circuit]
    return apply_gates(tensors, ψ_bpc; gate_vertices, kwargs...)
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
        gate::Tensor,
        ψ_bpc::BeliefPropagationCache;
        v⃗ = vertices(gate, network(ψ_bpc)),
        apply_kwargs
    )
    envs = length(v⃗) == 1 ? nothing : incoming_messages(ψ_bpc, v⃗)

    ψ⃗ = [network(ψ_bpc)[v] for v in v⃗]
    updated_tensors, s_values, err = simple_update(gate, ψ⃗; envs, apply_kwargs...)
    if length(v⃗) == 2
        v1, v2 = v⃗
        e = NamedEdge(v1 => v2)
        # Re-home the bond spectrum onto the new bond index and install it as the BP
        # message on both edge orientations (matching `default_message`'s layout).
        b = commonind(first(updated_tensors), last(updated_tensors))
        m, m_rev = _bond_spectrum_messages(s_values, b, e)
        setmessage!(ψ_bpc, e, m)
        setmessage!(ψ_bpc, reverse(e), m_rev)
    end

    for (i, v) in enumerate(v⃗)
        setindex_preserve!(ψ_bpc, updated_tensors[i], v)
    end

    return ψ_bpc, err
end

# Re-home the diagonal bond spectrum `s_values` produced by `simple_update` onto the new
# bond index `b` (shared by the two updated tensors) and return the forward/backward BP
# messages `(m, m_rev)` for the edge, laid out exactly like `default_message`.
#
# Bosonic: the message is the diagonal spectrum on `[b, b']`; `m_rev = dag(m)`.
function _bond_spectrum_messages(s_values::ITensor, b::Index, e)
    other = only(uniqueinds(s_values, b))
    m = replaceind(s_values, other, prime(b))
    return m, dag(m)
end

# Fermionic: rebuild the spectrum as a diagonal `FermionicITensor` on `[prime(b), b]` with
# `default_message`'s grading and orientation-dependent arrows. `dag` flips the arrows for
# the reverse edge, giving the OUT→IN closure the two messages must form.
function _bond_spectrum_messages(s_values::FermionicITensor, b::Index, e)
    bs, bsd = s_values.order
    bondgr = s_values.grading[bs]
    σ = ITensors.array(s_values.tensor, bs, bsd)
    u = prime(b)
    gr = Dictionary{Index, Vector{Bool}}(Index[u, b], Vector{Bool}[bondgr, bondgr])
    T = ITensor(σ, u, b)
    dirs = src(e) < dst(e) ? Bool[false, true] : Bool[true, false]
    m = FermionicITensor(T, Index[u, b], dirs, gr)
    return m, dag(m)
end

function apply_gate(gate::ITensor, ψ_bpc::BeliefPropagationCache; kwargs...)
    ψ_bpc = copy(ψ_bpc)
    return apply_gate!(gate, ψ_bpc; kwargs...)
end

const apply_circuit = apply_gates
