function expect(
        alg::Algorithm"exact",
        ψ::TensorNetworkState,
        observables::Vector{<:Tuple};
        contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = GreedyMethod())
    )
    denom = norm_sqr(alg, ψ; contraction_sequence_kwargs)
    out = map(observables) do obs
        op_strings, vs, coeff = collectobservable(obs, graph(ψ))
        iszero(coeff) && return coeff * zero(denom)
        op_string_f = op_string_function(op_strings, vs)
        ψOψ_tensors = norm_factors(ψ, collect(vertices(ψ)); op_strings = op_string_f)
        numer_seq = contraction_sequence(ψOψ_tensors; contraction_sequence_kwargs...)
        numer = scalar(contract(ψOψ_tensors; sequence = numer_seq))
        return coeff * (numer / denom)
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
- `ψ::Union{TensorNetworkState, AbstractBeliefPropagationCache}`: The tensor network state or cache wrapping the state to measure the observable(s) on.
- `observable::Union{Tuple, Vector{<:Tuple}}`: The observable(s) to measure. Should be a tuple or vector of tuples of the form `(ops, vertices, coeff=1)`.

# Keyword Arguments
- `alg::Union{String, Nothing}`: The algorithm to use. Supported algorithms:
    - `"exact"`: Exact contraction of the tensor network.
    - `"bp"`: Belief propagation approximation.
    - `"boundarymps"`: Boundary MPS approximation (requires `mps_bond_dimension`).
    - `"ctmrg"`: Corner transfer matrix environments on any planar graph (requires `maxdim`;
      observables on one vertex or on the two ends of an edge).
- `cache_update_kwargs...`: Keyword arguments passed to the `update` function when using `"bp"`, `"boundarymps"` or `"ctmrg"`.

# Returns
- A single number if measuring one observable, or a vector of numbers if measuring multiple observables.
"""
function expect(ψ::Union{TensorNetworkState, AbstractBeliefPropagationCache}, observable; alg::Union{String, Nothing} = default_alg(ψ), kwargs...)
    algorithm_check(ψ, "expect", alg)
    out = expect(Algorithm(alg), ψ, observable; kwargs...)
    # observables can differ in type (e.g. "Z" real, "Y" complex): promote to a common concrete type
    return out isa Vector ? convert(Vector{mapreduce(typeof, promote_type, out; init = Union{})}, out) : out
end

# BP and CTMRG: the observable's region, with the operator inserted, over the same region without
# it, both closed by their environment in the cache.
function expect(
        alg::Union{Algorithm"bp", Algorithm"ctmrg"},
        cache::AbstractBeliefPropagationCache,
        obs::Tuple
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(cache))
    iszero(coeff) && return zero(coeff)
    region = observable_region(cache, obs_vs)

    numer, denom = with_kernel_workspace() do
        region_scalar(cache, region, op_string_function(op_strings, obs_vs)), region_scalar(cache, region, v -> "I")
    end

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

    op_string_f = op_string_function(op_strings, obs_vs)

    numer, denom = path_contract(cache, obs_vs, op_string_f; bmps_messages_up_to_date)
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
        alg::Union{Algorithm"bp", Algorithm"ctmrg"},
        cache::AbstractBeliefPropagationCache,
        observables::Vector{<:Tuple}
    )
    return with_kernel_workspace(() -> map(obs -> expect(alg, cache, obs), observables))
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
        alg::Algorithm"ctmrg",
        ψ::TensorNetworkState,
        observable::Union{Tuple, Vector{<:Tuple}};
        maxdim::Integer,
        cache_update_kwargs = (;),
        kwargs...,
    )
    ψ_ctm = converged_cache(alg, ψ; maxdim, cache_update_kwargs)

    return expect(alg, ψ_ctm, observable; kwargs...)
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
