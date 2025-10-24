default_alg(bp_cache::BeliefPropagationCache) = "bp"
default_alg(bmps_cache::BoundaryMPSCache) = "boundarymps"
default_alg(any) = error("You must specify a contraction algorithm. Currently supported: exact, bp and boundarymps.")

function expect(
    alg::Algorithm"exact",
    ψ::TensorNetworkState,
    observables::Vector{<:Tuple};
    contraction_sequence_kwargs=(; alg="einexpr", optimizer=Greedy())
)
    ITensors.disable_warn_order()

    denom = norm_sqr(alg, ψ; contraction_sequence_kwargs)
    out = []
    for obs in observables
        op_strings, vs, coeff = collectobservable(obs)
        if iszero(coeff)
            push!(out, 0)
            continue
        end
        op_string_f = v -> v ∈ vs ? op_strings[findfirst(x -> x == v, vs)] : "I" 
        ψOψ_tensors = norm_factors(ψ, collect(vertices(ψ)); op_strings = op_string_f)
        numer_seq = contraction_sequence(ψOψ_tensors; contraction_sequence_kwargs...)
        numer = contract(ψOψ_tensors; sequence=numer_seq)[]
        push!(out, numer / denom)
    end
    return out
end

function expect(alg::Algorithm"exact",
    ψ::TensorNetworkState,
    observable::Tuple;
    kwargs...
)
    return only(expect(alg, ψ, [observable]; kwargs...))
end

"""
    expect(ψ, observable; alg="exact", kwargs...) -> Number 
    expect(ψ, observables; alg="exact", kwargs...) -> Vector{Number}
Compute the expectation value of one or more observables with respect to a TensorNetworkState or an updated BeliefPropagationCache or BoundaryMPSCache wrapping a TensorNetworkState.
The observable(s) should be passed as a tuple or vector of tuples of the form `(op, vertices, coeff=1)`, where `op` is either a string of single character operators (e.g. `"ZZIY"`) or a vector of strings (e.g. `["Z", "Z", "I", "Y"]`), `vertices` is either a vertex or a vector of vertices (e.g. `(1,2)` or `[(1,2), (1,3)]`), and `coeff` is an optional coefficient multiplying the observable (default is 1). The vertices should correspond to the sites of the TensorNetworkState.
The supported algorithms are: exact (exact contraction of all tensors in the network), belief propagation (bp) and boundary MPS (boundarymps).  
For `bp` and `boundarymps`, the TensorNetworkState is first converted into a BeliefPropagationCache or BoundaryMPSCache respectively, and the cache is updated before measuring. The update can be controlled via the keyword arguments `cache_update_kwargs`, which are passed to the `update` function. For `boundarymps`, the partitioning of the graph can be controlled via the keyword argument `partition_by`, which can be either `"row"` or `"column"`. The MPS bond dimension can be set via the keyword argument `mps_bond_dimension`.
For `exact`, the contraction sequence can be controlled via the keyword argument `contraction_sequence_kwargs`, which is a named tuple of keyword arguments passed to the `contraction_sequence` function. The default is to use the `einexpr` algorithm with a greedy optimizer.
"""
function expect(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}, observable; alg::Union{String, Nothing} = default_alg(ψ), kwargs...)
    return expect(Algorithm(alg), ψ, observable; kwargs...)
end

function expect(
    alg::Union{Algorithm"bp", Algorithm"boundarymps"},
    ψ::AbstractBeliefPropagationCache,
    obs::Tuple;
    bmps_messages_up_to_date = false,
)
    op_strings, obs_vs, coeff = collectobservable(obs)
    iszero(coeff) && return 0

    #For boundary MPS, must stay in partition
    if length(obs_vs) == 1
        steiner_vs = obs_vs
    elseif alg == Algorithm("bp")
        steiner_vs = collect(vertices(steiner_tree(network(ψ), obs_vs)))
    elseif alg == Algorithm("boundarymps")
        partitions = unique(partitionvertices(ψ, obs_vs))
        length(partitions) > 1 && error("Observable support must be within a single partition (row/ column) of the graph for now.")
        partition = only(partitions)
        g = partition_graph(ψ, partition)
        steiner_vs = collect(vertices(steiner_tree(g, obs_vs)))

        if !bmps_messages_up_to_date
            ψ = update_partition(ψ, partition)
        end
    end
    op_string_f = v -> v ∈ obs_vs ? op_strings[findfirst(x->x == v, obs_vs)] : "I"

    incoming_ms = incoming_messages(ψ, steiner_vs)
    ψIψ_tensors = ITensor[norm_factors(network(ψ), steiner_vs); incoming_ms]
    denom_seq = contraction_sequence(ψIψ_tensors; alg="einexpr", optimizer=Greedy())
    denom = contract(ψIψ_tensors; sequence=denom_seq)[]

    ψOψ_tensors = ITensor[norm_factors(network(ψ), steiner_vs; op_strings = op_string_f); incoming_ms]
    numer_seq = contraction_sequence(ψOψ_tensors; alg="einexpr", optimizer=Greedy())
    numer = contract(ψOψ_tensors; sequence=numer_seq)[]

    return coeff * numer/ denom
end

function expect(
    alg::Algorithm"boundarymps",
    cache::BoundaryMPSCache,
    observables::Vector{<:Tuple};
    bmps_messages_up_to_date = false,
    kwargs...,
)
    obs_vs = observables_vertices(observables)
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

    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)

    return expect(alg, ψ_bpc, observable; kwargs...)
end

function expect(
    alg::Algorithm"boundarymps",
    ψ::TensorNetworkState,
    observable::Union{Tuple, Vector{<:Tuple}};
    cache_update_kwargs = default_bmps_update_kwargs(ψ),
    partition_by = boundarymps_partitioning(observable),
    mps_bond_dimension::Int,
    kwargs...,
)

    ψ_bmps = BoundaryMPSCache(ψ, mps_bond_dimension; partition_by)
    cache_update_kwargs = (; cache_update_kwargs..., maxiter = default_bp_maxiter(ψ_bmps))
    ψ_bmps = update(ψ_bmps; cache_update_kwargs...)

    obs_vs = observables_vertices(observable)
    ψ_bmps = update_partitions(ψ_bmps, obs_vs)

    return expect(alg, ψ_bmps, observable; bmps_messages_up_to_date = true, kwargs...)
end

#Process an observable into more readable form
function collectobservable(obs::Tuple)
    # unpack
    op = obs[1]
    verts = _tovec(obs[2])
    coeff = length(obs) == 2 ? 1 : last(obs)

    length(verts) == 1 && return [op], verts, coeff

    @assert !(op == "" && isempty(verts))

    length(op) != length(verts) && error("Invalid observable: need as many operators as vertices passed.")
    if op isa String
        op_strings = [string(o) for o in op]
    elseif op isa Vector{<:String}
        op_strings = [o for o in op]
    end

    return op_strings, verts, coeff
end

function observables_vertices(observables::Vector{<:Tuple})
    return reduce(vcat, [obs[2] for obs in observables])
end

observables_vertices(obs::Tuple) = obs[2]
_tovec(verts::Union{Tuple, AbstractVector}) = verts isa Tuple ? [verts] : collect(verts)
_tovec(verts::NamedEdge) = [src(verts), dst(verts)]

function boundarymps_partitioning(observable::Union{Tuple, Vector{<:Tuple}})
    observables = observable isa Tuple ? [observable] : observable
    partitioning = nothing
    for o in observables
        vs = observables_vertices(o)
        if allequal(first.(vs)) && (partitioning == "row" || partitioning == nothing)
            partitioning = "row"
        elseif allequal(last.(vs)) && (partitioning == "col" || partitioning == nothing)
            partitioning = "col"
        else
            error("Observables must all be aligned in either the same column or the same row to do BoundaryMPS measurements.")
        end
    end
    return partitioning
end
