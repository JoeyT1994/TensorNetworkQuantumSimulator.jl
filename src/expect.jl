function expect(
        alg::Algorithm"exact",
        ψ::TensorNetworkState,
        observables::Vector{<:Tuple};
        contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = GreedyMethod())
    )

    denom = norm_sqr(alg, ψ; contraction_sequence_kwargs)
    out = Number[]
    for obs in observables
        op_strings, vs, coeff = collectobservable(obs, graph(ψ))
        if iszero(coeff)
            push!(out, zero(coeff))
            continue
        end
        #A String op_strings marks a joint region operator (see collectobservable)
        joint_op = op_strings isa String ? (op_strings, vs) : nothing
        op_string_f = joint_op === nothing ? op_string_function(op_strings, vs) : (v -> "I")
        ψOψ_tensors = norm_factors(ψ, collect(vertices(ψ)); op_strings = op_string_f, joint_op)
        numer_seq = contraction_sequence(ψOψ_tensors; contraction_sequence_kwargs...)
        numer = scalar(contract(ψOψ_tensors; sequence = numer_seq))
        push!(out, coeff * (numer / denom))
    end
    return out
end

function expect(
        alg::Algorithm"exact",
        ψ::TensorNetworkState,
        observable::Tuple;
        kwargs...
    )
    return only(expect(alg, ψ, [observable]; kwargs...))
end

"""
    expect(ψ, observable; alg="exact", kwargs...) -> Number or Vector{Number}

Compute the expectation value of one or more observables on a tensor network state.

# Arguments
- `ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}`: The tensor network state or cache wrapping the state to measure the observable(s) on.
- `observable::Union{Tuple, Vector{<:Tuple}}`: The observable(s) to measure. Should be a tuple or vector of tuples of the form `(ops, vertices, coeff=1)`.

# Keyword Arguments
- `alg::Union{String, Nothing}`: The algorithm to use. Supported algorithms:
    - `"exact"`: Exact contraction of the tensor network.
    - `"bp"`: Belief propagation approximation.
    - `"boundarymps"`: Boundary MPS approximation (requires `mps_bond_dimension`).
- `cache_update_kwargs...`: Keyword arguments passed to the `update` function when using `"bp"` or `"boundarymps"` algorithms.

# Returns
- A single number if measuring one observable, or a vector of numbers if measuring multiple observables.
"""
function expect(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}, observable; alg::Union{String, Nothing} = default_alg(ψ), kwargs...)
    algorithm_check(ψ, "expect", alg)
    return expect(Algorithm(alg), ψ, observable; kwargs...)
end

function expect(
        alg::Algorithm"bp",
        cache::BeliefPropagationCache,
        obs::Tuple
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(cache))
    iszero(coeff) && return zero(coeff)

    steiner_vs = length(obs_vs) == 1 ? obs_vs : collect(vertices(steiner_tree(network(cache), obs_vs)))
    incoming_ms = incoming_messages(cache, steiner_vs)

    #TODO: If there are a lot of tensors here, (more than 100 say), we need to think about defining a custom sequence as optimal may be too slow
    function contract_region(op_string_f; joint_op = nothing)
        if joint_op === nothing
            fast = norm_scalar_kernel(network(cache), steiner_vs, incoming_ms; op_strings = op_string_f)
            fast !== nothing && return fast
        end
        tensors = norm_factors(network(cache), steiner_vs; op_strings = op_string_f, joint_op)
        append!(tensors, incoming_ms)
        seq = contraction_sequence(tensors; alg = "optimal")
        return scalar(contract(tensors; sequence = seq))
    end

    denom = contract_region(v -> "I")
    #A String op_strings marks a joint region operator (see collectobservable)
    numer = op_strings isa String ?
        contract_region(v -> "I"; joint_op = (op_strings, obs_vs)) :
        contract_region(op_string_function(op_strings, obs_vs))

    return coeff * numer / denom
end

function expect(
        alg::Algorithm"boundarymps",
        cache::BoundaryMPSCache,
        obs::Tuple;
        bmps_messages_up_to_date = false,
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(cache))
    iszero(coeff) && return zero(coeff)

    #A String op_strings marks a joint region operator (see collectobservable): the
    #joint vertices' site legs stay primed ("ρ") and the operator tensor bridges them.
    joint_op = op_strings isa String ? (op_strings, obs_vs) : nothing
    op_string_f = joint_op === nothing ? op_string_function(op_strings, obs_vs) :
        (v -> v ∈ obs_vs ? "ρ" : "I")

    numer, denom = path_contract(cache, obs_vs, op_string_f; bmps_messages_up_to_date, joint_op)
    return coeff * scalar(numer) / denom
end

function expect(
        alg::Algorithm"boundarymps",
        cache::BoundaryMPSCache,
        observables::Vector{<:Tuple};
        bmps_messages_up_to_date = false,
        kwargs...,
    )
    obs_vs = observables_vertices(observables, graph(cache))
    if !bmps_messages_up_to_date
        cache = update_partitions(cache, obs_vs)
    end
    out = map(obs -> expect(alg, cache, obs; bmps_messages_up_to_date = true, kwargs...), observables)
    return out
end

function expect(
        alg::Algorithm"bp",
        cache::BeliefPropagationCache,
        observables::Vector{<:Tuple};
        kwargs...,
    )
    return map(obs -> expect(alg, cache, obs; kwargs...), observables)
end

function expect(
        alg::Algorithm"bp",
        ψ::TensorNetworkState,
        observable::Union{Tuple, Vector{<:Tuple}};
        cache_update_kwargs = default_bp_update_kwargs(ψ),
        kwargs...,
    )

    ψ_bpc = converged_cache(alg, ψ; cache_update_kwargs)

    return expect(alg, ψ_bpc, observable; kwargs...)
end

function expect(
        alg::Algorithm"boundarymps",
        ψ::TensorNetworkState,
        observable::Union{Tuple, Vector{<:Tuple}};
        cache_update_kwargs = default_bmps_update_kwargs(ψ),
        partition_by = boundarymps_partitioning(observable, graph(ψ)),
        mps_bond_dimension::Integer,
        gauge_state = true,
        kwargs...,
    )

    ψ_bmps = converged_cache(alg, ψ; mps_bond_dimension, partition_by, gauge_state, cache_update_kwargs)

    obs_vs = observables_vertices(observable, graph(ψ))
    ψ_bmps = update_partitions(ψ_bmps, obs_vs)

    return expect(alg, ψ_bmps, observable; bmps_messages_up_to_date = true, kwargs...)
end

#Process an observable into more readable form
function collectobservable(obs::Tuple, g::NamedGraph)

    coeff = length(obs) == 2 ? 1 : last(obs)
    verts = observables_vertices(obs, g)
    op = obs[1]

    #A non-Pauli name on a multi-vertex region is a JOINT operator spanning the region
    #(e.g. fermionic ("hopping", (v, w)) or ("CC", (v, w))); returned as a bare String
    #sentinel. Single-character-per-vertex Pauli strings keep their factorized meaning.
    if op isa String && length(verts) > 1 && (length(op) != length(verts) || !_ispaulistring(op))
        length(verts) == 2 || error("Joint operator observables currently support two-vertex regions, got $(length(verts)) vertices.")
        return op, verts, coeff
    end

    #a single vertex takes one operator name of any length (e.g. "Nup", "NupNdn")
    op isa String && length(verts) == 1 && return [op], verts, coeff

    length(op) != length(verts) && error("Invalid observable: need as many operators as vertices passed.")
    if op isa String
        op_strings = [string(o) for o in op]
    elseif op isa Vector{<:String}
        op_strings = [o for o in op]
    else
        error("Invalid observable, did not recognize operator specification. Either a single string (one pauli character per vertex) or a vector of strings (one string per vertex) expected.")
    end

    return op_strings, verts, coeff
end

# Map each vertex to its operator string, defaulting to the identity "I" off the observable's support.
function op_string_function(op_strings, vs)
    op_dict = Dict(zip(vs, op_strings))
    return v -> get(op_dict, v, "I")
end

observables_vertices(observable::Tuple, g::NamedGraph) = collect_vertices(observable[2], g)
observables_vertices(observables::Vector{<:Tuple}, g::NamedGraph) = unique(collect(Iterators.flatten(observables_vertices(obs, g) for obs in observables)))

function boundarymps_partitioning(observable::Union{Tuple, Vector{<:Tuple}}, g::NamedGraph)
    observables = observable isa Tuple ? [observable] : observable
    partitioning = nothing
    for o in observables
        vs = observables_vertices(o, g)
        #Single-site observables sit in every row and every column, so they constrain nothing
        length(vs) < 2 && continue
        if allequal(first.(vs)) && (partitioning == "row" || isnothing(partitioning))
            partitioning = "row"
        elseif allequal(last.(vs)) && (partitioning == "col" || isnothing(partitioning))
            partitioning = "col"
        else
            error("Observables must all be aligned in either the same column or the same row to do BoundaryMPS measurements.")
        end
    end
    #Only single-site observables: any partitioning works, so pick rows
    return isnothing(partitioning) ? "row" : partitioning
end
