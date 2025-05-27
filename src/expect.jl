default_expect_alg() = "bp"

"""
    ITensorNetworks.expect(alg::Algorithm"exact", ψ::AbstractITensorNetwork, observables::Vector{<:Tuple}, contraction_sequence_kwargs = (; alg = "einexpr", optimizer = Greedy()))

Function for computing expectation values for any vector of pauli strings via exact contraction.
This will be infeasible for larger networks with high bond dimension.
"""
function ITensorNetworks.expect(
    alg::Algorithm"exact",
    ψ::AbstractITensorNetwork,
    observables::Vector{<:Tuple};
    contraction_sequence_kwargs=(; alg="einexpr", optimizer=Greedy()),
)

    s = siteinds(ψ)
    ψIψ = QuadraticFormNetwork(ψ)

    out = []
    for obs in observables
        op_strings, vs, coeff = collectobservable(obs)
        if iszero(coeff)
            push!(out, 0.0)
            continue
        end
        ψOψ = copy(ψIψ)
        for (op_string, v) in zip(op_strings, vs)
            ψOψ[(v, "operator")] = ITensors.op(op_string, s[v])
        end

        numer_seq = contraction_sequence(ψOψ; contraction_sequence_kwargs...)
        denom_seq = contraction_sequence(ψIψ; contraction_sequence_kwargs...)
        numer, denom =
            contract(ψOψ; sequence=numer_seq)[], contract(ψIψ; sequence=denom_seq)[]
        push!(out, numer / denom)
    end
    return out
end

function ITensorNetworks.expect(alg::Algorithm"exact",
    ψ::AbstractITensorNetwork,
    observable::Tuple;
    kwargs...
)
    return expect(alg, ψ, [observable]; kwargs...)
end


"""
    ITensorNetworks.expect(alg::Algorithm, ψ::AbstractITensorNetwork, observables::Vector{<:Tuple}; (cache!) = nothing,
    update_cache = isnothing(cache!), cache_update_kwargs = alg == Algorithm("bp") ? default_posdef_bp_update_kwargs() : ITensorNetworks.default_cache_update_kwargs(alg),
    cache_construction_kwargs = default_cache_construction_kwargs(alg, QuadraticFormNetwork(ψ), ), kwargs...)

Function for computing expectation values for any vector of pauli strings via different cached based algorithms. 
Support: alg = "bp" and alg = "boundarymps". The latter takes cache_construction_kwargs = (; message_rank::Int) as a constructor.
"""
function ITensorNetworks.expect(
    alg::Algorithm,
    ψ::AbstractITensorNetwork,
    observables::Vector{<:Tuple};
    (cache!)=nothing,
    update_cache=isnothing(cache!),
    cache_update_kwargs=alg == Algorithm("bp") ? default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)) : ITensorNetworks.default_cache_update_kwargs(alg),
    cache_construction_kwargs=default_cache_construction_kwargs(
        alg,
        QuadraticFormNetwork(ψ),
    ),
    message_rank = nothing,
    kwargs...,
)

    if alg == Algorithm("boundarymps") && !isnothing(message_rank)
        cache_construction_kwargs = merge(cache_construction_kwargs, (; message_rank))
    end
    ψIψ = QuadraticFormNetwork(ψ)
    if isnothing(cache!)
        cache! = Ref(cache(alg, ψIψ; cache_construction_kwargs...))
    end

    if update_cache
        cache![] = update(cache![]; cache_update_kwargs...)
    end

    return expect(cache![], observables; alg, kwargs...)
end

# Here we turn a single tuple observable into a vector of tuples -- the expected format in ITensorNetworks
function ITensorNetworks.expect(
    alg::Algorithm,
    ψ::AbstractITensorNetwork,
    observable::Tuple;
    kwargs...,
)
    return only(expect(alg, ψ, [observable]; kwargs...))
end


"""
    expect(ψ::AbstractITensorNetwork, obs; alg="bp", kwargs...)

Calculate the expectation value of an `ITensorNetwork` `ψ` with an observable or vector of observables `obs` using the desired algorithm `alg`.
Currently supported: alg = "bp", "boundarymps" or "exact".
"bp" will be imprecise for networks with strong loop correlations, but is otherwise fast.
"boundarymps" is more precise and slower, and can only be used if the network is planar with coordinate vertex labels like (1, 1), (1, 2), etc.
"exact" will be infeasible for larger networks with high bond dimension.
"""
function ITensorNetworks.expect(
    ψ::AbstractITensorNetwork,
    obs::Union{Tuple, Vector{<:Tuple}};
    alg=default_expect_alg(),
    kwargs...,
)
    return expect(Algorithm(alg), ψ, obs; kwargs...)
end

"""
    expect(ψIψ::AbstractBeliefPropagationCache, obs::Tuple; kwargs...)

Foundational expectation function for a given (norm) cache network with an observable. 
This can be a `BeliefPropagationCache` or a `BoundaryMPSCache`.
Valid observables are tuples of the form `(op, qinds)` or `(op, qinds, coeff)`, 
where `op` is a string or vector of strings, `qinds` is a vector of indices, and `coeff` is a coefficient (default 1.0).
The `kwargs` are not used.
"""
function ITensorNetworks.expect(
    ψIψ::AbstractBeliefPropagationCache,
    obs::Tuple;
    kwargs...
)

    op_strings, vs, coeff = collectobservable(obs)
    iszero(coeff) && return 0.0

    ψOψ = insert_observable(ψIψ, obs)

    numerator = region_scalar(ψOψ, [(v, "ket") for v in vs])
    denominator = region_scalar(ψIψ, [(v, "ket") for v in vs])

    return coeff * numerator / denominator
end

function ITensorNetworks.expect(
    ψIψ::AbstractBeliefPropagationCache,
    observables::Vector{<:Tuple};
    kwargs...,
)
    return map(obs -> expect(ψIψ, obs; kwargs...), observables)
end

"""
    insert_observable(ψIψ::AbstractBeliefPropagationCache, obs)

Insert an obervable O into ψIψ to create the cache containing ψOψ. 
Drops the coefficient of the observable in the third slot of the obs tuple.
Example: obs = ("X", [1, 2]) or obs = ("XX", [1, 2], 0.5) -> ("XX", [1, 2])
"""
function insert_observable(ψIψ::AbstractBeliefPropagationCache, obs)
    op_strings, verts, _ = collectobservable(obs)

    ψIψ_vs = [ψIψ[(v, "operator")] for v in verts]
    sinds =
        [commonind(ψIψ[(v, "ket")], ψIψ_vs[i]) for (i, v) in enumerate(verts)]
    operators = [ITensors.op(op_strings[i], sinds[i]) for i in eachindex(op_strings)]

    ψOψ = update_factors(ψIψ, Dictionary([(v, "operator") for v in verts], operators))
    return ψOψ
end

#Process an observable into more readable form
function collectobservable(obs::Tuple)
    # unpack
    op = obs[1]
    qinds = obs[2]
    coeff = length(obs) == 2 ? 1.0 : last(obs)

    @assert !(op == "" && isempty(qinds))

    op_vec = [string(o) for o in op]
    qinds_vec = _tovec(qinds)
    return op_vec, qinds_vec, coeff
end

_tovec(qinds) = vec(collect(qinds))
_tovec(qinds::NamedEdge) = [qinds.src, qinds.dst]
