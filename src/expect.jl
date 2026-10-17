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
    # Observables can differ in type (e.g. "Z" real, "Y" complex), so promote to a common concrete type
    return convert(Vector{mapreduce(typeof, promote_type, out; init = Union{})}, out)
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
- `ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache, CTMEnvironmentCache}`: The tensor network state or cache wrapping the state to measure the observable(s) on.
- `observable::Union{Tuple, Vector{<:Tuple}}`: The observable(s) to measure. Should be a tuple or vector of tuples of the form `(ops, vertices, coeff=1)`.

# Keyword Arguments
- `alg::Union{String, Nothing}`: The algorithm to use. Supported algorithms:
    - `"exact"`: Exact contraction of the tensor network.
    - `"bp"`: Belief propagation approximation.
    - `"boundarymps"`: Boundary MPS approximation (requires `mps_bond_dimension`).
    - `"ctmrg"`: Finite CTMRG / CVM environments (requires `maxdim`; single-site observables only).
      When given a `TensorNetworkState` with `ctm_options = (; projector = :cycle)`, the internal
      `update` defaults to `convergence = :marginal`, so the vertex rings are converged rather than
      just the free energy.
- `cache_update_kwargs...`: Keyword arguments passed to the `update` function when using `"bp"`, `"boundarymps"` or `"ctmrg"` algorithms.

# Returns
- A single number if measuring one observable, or a vector of numbers if measuring multiple observables.
"""
function expect(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache, CTMEnvironmentCache}, observable; alg::Union{String, Nothing} = default_alg(ψ), kwargs...)
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
    function contract_region(op_string_f)
        fast = norm_scalar_kernel(network(cache), steiner_vs, incoming_ms; op_strings = op_string_f)
        fast !== nothing && return fast
        tensors = norm_factors(network(cache), steiner_vs; op_strings = op_string_f)
        append!(tensors, incoming_ms)
        seq = contraction_sequence(tensors; alg = "optimal")
        return scalar(contract(tensors; sequence = seq))
    end

    denom = contract_region(v -> "I")
    numer = contract_region(op_string_function(op_strings, obs_vs))

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

# CTMRG: the observable's vertex is enclosed by its own 4C+4T ring, which IS its environment
# (`vertex_ring`), so this is the same shape as the BP branch with the ring in place of the
# incoming messages. Single-site only for now: the ring surrounds exactly one vertex, and the
# CVM edge/plaquette regions are the region-graph OVERLAPS, which contain no vertex at all —
# a two-site observable needs a genuine two-vertex region that does not exist yet.
function _ctm_expect(cache::CTMEnvironmentCache, obs::Tuple, window::Integer, denoms)
    op_strings, obs_vs, coeff = collectobservable(obs, graph(cache))
    iszero(coeff) && return zero(coeff)
    length(obs_vs) == 1 ||
        error("alg=\"ctmrg\" supports single-site observables only; got $(length(obs_vs)) " *
              "vertices $(obs_vs). The 4C+4T ring encloses one vertex.")
    v = only(obs_vs)
    env_ts = vertex_window(cache, v, window)   # `window > 0` keeps more of the lattice exact

    function contract_region(op_string_f)
        tensors = norm_factors(network(cache), [v]; op_strings = op_string_f)
        append!(tensors, env_ts)
        # cached sequence + tensor-count gate, using the cache's own options
        return scalar(_ctm_contract(tensors, options(cache)))
    end

    # The denominator is the identity ring — a property of the VERTEX, not the observable — so a
    # vector of observables shares it via `denoms` (`window` is fixed across one vector call).
    denom = isnothing(denoms) ? contract_region(v -> "I") :
            get!(() -> contract_region(v -> "I"), denoms, v)
    numer = contract_region(op_string_function(op_strings, obs_vs))
    return coeff * numer / denom
end

function expect(
        alg::Algorithm"ctmrg",
        cache::CTMEnvironmentCache,
        obs::Tuple;
        window::Integer = 0,
    )
    return _ctm_expect(cache, obs, window, nothing)
end

function expect(
        alg::Algorithm"ctmrg",
        cache::CTMEnvironmentCache,
        observables::Vector{<:Tuple};
        window::Integer = 0,
    )
    denoms = Dict{Any, Any}()   # per-vertex identity-ring contractions, shared across the vector
    return map(obs -> _ctm_expect(cache, obs, window, denoms), observables)
end

function expect(
        alg::Algorithm"ctmrg",
        ψ::TensorNetworkState,
        observable::Union{Tuple, Vector{<:Tuple}};
        maxdim::Integer,
        cache_update_kwargs = (;),
        ctm_options = (;),          # `CTMOptions` fields, e.g. `(degtol = 1e-9,)`
        kwargs...,
    )
    # Observables need the messages actually stationary. For `:cycle` that is
    # `convergence = :marginal`: `update`'s `:free_energy` default stops once F settles, and F
    # settles a few sweeps BEFORE a boundary-lagged single-site observable does (see `update`);
    # the marginal signal watches the vertex reduced density matrices themselves. For `:cut` the
    # default statedist pair is already observable-tight — so inject only for `:cycle`.
    # Overridable either way through `cache_update_kwargs`.
    if get(ctm_options, :projector, :cut) === :cycle
        cache_update_kwargs = merge((; convergence = :marginal), cache_update_kwargs)
    end
    cache = update(CTMEnvironmentCache(ψ, maxdim; ctm_options...); cache_update_kwargs...)
    return expect(alg, cache, observable; kwargs...)
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
