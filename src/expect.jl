const CacheNetwork = Union{BeliefPropagationCache,BoundaryMPSCache}

"""
    expect(ψIψ::CacheNetwork, obs::Tuple; bp_update_kwargs=get_global_bp_update_kwargs())

Foundational expectation function for a given (norm) cache network with an observable. 
This can be a `BeliefPropagationCache` or a `BoundaryMPSCache`.
Valid observables are tuples of the form `(op, qinds)` or `(op, qinds, coeff)`, 
where `op` is a string or vector of strings, `qinds` is a vector of indices, and `coeff` is a coefficient (default 1.0).
"""
function expect(ψIψ::CacheNetwork, obs::Tuple; bp_update_kwargs=get_global_bp_update_kwargs(), update_cache=true)
    if update_cache
        ψIψ = updatecache(ψIψ; bp_update_kwargs...)
    end
    ψOψ = insert_observable(ψIψ, obs)

    return scalar(ψOψ) / scalar(ψIψ)
end

"""
    expect(ψ::AbstractITensorNetwork, obs::Tuple; bp_update_kwargs=get_global_bp_update_kwargs())

Calculate the expectation value of an `ITensorNetwork` `ψ` with an observable `obs` using belief propagation.
This function first builds a `BeliefPropagationCache` `ψIψ` from the input state `ψ` and then calls the `expect(ψIψ, obs)` function on the cache.
"""
function expect(ψ::AbstractITensorNetwork, obs::Tuple; bp_update_kwargs=get_global_bp_update_kwargs())
    ψIψ = build_bp_cache(ψ; bp_update_kwargs...)
    return expect(ψIψ, obs; bp_update_kwargs, update_cache=false)
end


function expect(ψ::ITensorNetwork, ϕ::ITensorNetwork; bp_update_kwargs=get_global_bp_update_kwargs())
    # is 
    ψψ = build_bp_cache(ψ; bp_update_kwargs...)
    ϕϕ = build_bp_cache(ϕ; bp_update_kwargs...)

    ψϕ = inner_network(ψ, ϕ)
    ψϕ = updatecache(ψϕ; bp_update_kwargs...)

    # TODO: do we need a sqrt or something here?
    # TODO: integrate this for loop-correct and boundary MPS
    expectation = scalar(ψϕ) / scalar(ψψ) / scalar(ϕϕ)

    return expectation
end


function expect(tn, observable::Tuple; max_loop_size=nothing, message_rank=nothing, kwargs...)
    # max_loop_size determines whether we use BP and loop correction
    # message_rank determines whether we use boundary MPS

    # first determine whether to work with boundary MPS
    if !isnothing(message_rank)
        if !isnothing(max_loop_size)
            throw(ArgumentError(
                "Both `max_loop_size` and `message_rank` are set. " *
                "Use `max_loop_size` for belief propagation with optional loop corrections. " *
                "Use `message_rank` to use boundary MPS."
            ))
        end

        return expect_boundarymps(tn, observable, message_rank; kwargs...)
    end


    if isnothing(max_loop_size)
        # this is the default case of BP expectation value
        max_loop_size = 0
    end

    return expect_loopcorrect(tn, observable, max_loop_size; kwargs...)
end



function expect_loopcorrect(ψ::ITensorNetwork, obs, max_circuit_size::Integer; max_genus::Integer=2, bp_update_kwargs=get_global_bp_update_kwargs())
    ## this is the entry point for when the state network is passed, and not the BP cache 
    ψIψ = build_bp_cache(ψ; bp_update_kwargs...)
    return expect_loopcorrect(ψIψ, obs, max_circuit_size; max_genus, bp_update_kwargs, update_cache=false)
end

function expect_loopcorrect(
    ψIψ::BeliefPropagationCache, obs::Tuple, max_circuit_size::Integer;
    max_genus::Integer=2, bp_update_kwargs=get_global_bp_update_kwargs(), update_cache=true
)

    # TODO: default max genus to ceil(max_circuit_size/min_loop_size)
    # Warn if max_genus is 3 or larger lol
    if max_genus > 2
        @warn "Expectation value calculation with max_genus > 2 is not advised."
        # flush to instantly see the warning
        flush(stdout)
    end

    if update_cache
        ψIψ = updatecache(ψIψ; bp_update_kwargs...)
    end

    ψOψ = insert_observable(ψIψ, obs)

    expectation = scalar(ψOψ) / scalar(ψIψ)

    # without loop correction, we can just return the expectation
    if max_circuit_size <= 0
        return expectation
    end

    # now to getting the corrections

    ψIψ = normalize(ψIψ; update_cache=false)
    ψOψ = normalize(ψOψ; update_cache=false)

    # first get all the cycles
    circuits = enumerate_circuits(ψIψ, max_circuit_size; max_genus)

    # TODO: clever caching for multiple observables
    ψIψ_corrections = loop_correction_factor(ψIψ, circuits)
    ψOψ_corrections = loop_correction_factor(ψOψ, circuits)

    return expectation * ψOψ_corrections / ψIψ_corrections
end

## boundary MPS
function expect_boundarymps(
    ψ::AbstractITensorNetwork, observable, message_rank::Integer;
    transform_to_symmetric_gauge=false,
    bp_update_kwargs=get_global_bp_update_kwargs(),
    boundary_mps_kwargs=get_global_boundarymps_update_kwargs()
)

    ψIψ = build_boundarymps_cache(ψ, message_rank; transform_to_symmetric_gauge, bp_update_kwargs, boundary_mps_kwargs)
    return expect_boundarymps(ψIψ, observable; boundary_mps_kwargs)
end


function expect_boundarymps(
    ψIψ::BoundaryMPSCache, obs::Tuple; boundary_mps_kwargs=get_global_boundarymps_update_kwargs(), update_cache=true
)
    # TODO: validate the observable at this point

    if update_cache
        ψIψ = updatecache(ψIψ; boundary_mps_kwargs...)
    end

    return expect(ψIψ, obs; update_cache=false)
end


## utilites
function insert_observable(ψIψ, obs)
    op_strings, qinds, coeff = collectobservable(obs)

    ψIψ_tn = tensornetwork(ψIψ)
    ψIψ_vs = [ψIψ_tn[operator_vertex(ψIψ_tn, v)] for v in qinds]
    sinds = [commonind(ψIψ_tn[ket_vertex(ψIψ_tn, v)], ψIψ_vs[i]) for (i, v) in enumerate(qinds)]
    operators = [ITensors.op(op_strings[i], sinds[i]) for i in eachindex(op_strings)]

    # scale the first operator with the coefficient
    # TODO: is evenly better?
    operators[1] = operators[1] * coeff


    ψOψ = update_factors(ψIψ, Dictionary([(v, "operator") for v in qinds], operators))
    return ψOψ
end


function collectobservable(obs::Tuple)
    # unpack
    op = obs[1]
    qinds = obs[2]
    if length(obs) == 2
        coeff = 1.0
    else
        coeff = obs[3]
    end

    # when the observable is "I" or an empty string, just return the coefficient
    # this is dangeriously assuming that the norm of the network is one
    # TODO: how to make this more general?
    if op == "" && isempty(qinds)
        # if this is the case, we assume that this is a norm contraction with identity observable
        op = "I"
        qinds = [first(TN.vertices(ψIψ))[1]] # the first vertex
    end

    op_vec = [string(o) for o in op]
    qinds_vec = _tovec(qinds)
    return op_vec, qinds_vec, coeff
end

_tovec(qinds) = vec(collect(qinds))
_tovec(qinds::NamedEdge) = [qinds.src, qinds.dst]


