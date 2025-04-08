"""
    expect(ψ::AbstractITensorNetwork, obs::Tuple; bp_update_kwargs=get_global_bp_update_kwargs())

Calculate the expectation value of an `ITensorNetwork` `ψ` with an observable `obs` using belief propagation.
This function first builds a `BeliefPropagationCache` `ψIψ` from the input state `ψ` and then calls the `expect(ψIψ, obs)` function on the cache.
"""
function expect(ψ::AbstractITensorNetwork, obs::Tuple; bp_update_kwargs=get_global_bp_update_kwargs())
    ψIψ = build_bp_cache(ψ; bp_update_kwargs...)
    return expect(ψIψ, obs; update_cache=false)
end


"""
    expect(ψIψ::CacheNetwork, obs::Tuple; bp_update_kwargs=get_global_bp_update_kwargs())

Foundational expectation function for a given (norm) cache network with an observable. 
This can be a `BeliefPropagationCache` or a `BoundaryMPSCache`.
If `update_cache` is true, the cache will be updated before calculating the expectation value.
The global cache update kwargs are used for the update due to ambiguity in the cache type.
Valid observables are tuples of the form `(op, qinds)` or `(op, qinds, coeff)`, 
where `op` is a string or vector of strings, `qinds` is a vector of indices, and `coeff` is a coefficient (default 1.0).
"""
function expect(ψIψ::CacheNetwork, obs::Tuple; update_cache=true)
    if update_cache
        ψIψ = updatecache(ψIψ)
    end

    ψOψ = insert_observable(ψIψ, obs)

    return ratio(ψOψ, ψIψ)
end


"""
    fidelity(ψ::AbstractITensorNetwork, ϕ::AbstractITensorNetwork; bp_update_kwargs=get_global_bp_update_kwargs())

Calculate the fidelity between two `ITensorNetwork`s `ψ` and `ϕ` using belief propagation.
"""
function fidelity(ψ::AbstractITensorNetwork, ϕ::AbstractITensorNetwork; bp_update_kwargs=get_global_bp_update_kwargs())
    ψψ = build_bp_cache(ψ; bp_update_kwargs...)
    ϕϕ = build_bp_cache(ϕ; bp_update_kwargs...)
    ψϕ = build_bp_cache(ψ, ϕ; bp_update_kwargs...)

    return fidelityratio(ψϕ, ψψ, ϕϕ)
end


function fidelity_expect(ψ::AbstractITensorNetwork, obs::Tuple; bp_update_kwargs=get_global_bp_update_kwargs())

    ψO = apply(obs, ψ)

    ψψ = build_bp_cache(ψ; bp_update_kwargs...)
    ψOψ = build_bp_cache(ψO, ψ; bp_update_kwargs...)

    return ratio(ψOψ, ψψ)
end


function ratio(ψOψ::CacheNetwork, ψIψ::CacheNetwork)
    return scalar(ψOψ) / scalar(ψIψ)
end


function fidelityratio(ψϕ::CacheNetwork, ψψ::CacheNetwork, ϕϕ::CacheNetwork)
    # TODO: update cache option?
    return scalar(ψϕ) / sqrt(scalar(ψψ)) / sqrt(scalar(ϕϕ))
end

## boundary MPS
function expect_boundarymps(
    ψ::AbstractITensorNetwork, obs, message_rank::Integer;
    boundary_mps_kwargs...
)

    ψIψ = build_boundarymps_cache(ψ, message_rank; boundary_mps_kwargs...)
    return expect_boundarymps(ψIψ, obs; boundary_mps_kwargs)
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


function fidelity_boundarymps(ψ::AbstractITensorNetwork, ϕ::AbstractITensorNetwork, message_rank::Integer; boundary_mps_kwargs=get_global_boundarymps_update_kwargs())
    # is 
    ψψ = build_boundarymps_cache(ψ, message_rank; boundary_mps_kwargs...)
    ϕϕ = build_boundarymps_cache(ϕ, message_rank; boundary_mps_kwargs...)
    ψϕ = build_boundarymps_cache(ψ, ϕ, message_rank; boundary_mps_kwargs...)

    return fidelityratio(ψϕ, ψψ, ϕϕ)
end

function fidelity_expect_boundarymps(ψ::AbstractITensorNetwork, obs::Tuple, message_rank::Integer; boundary_mps_kwargs=get_global_boundarymps_update_kwargs())

    ψO = apply(obs, ψ)

    ψψ = build_boundarymps_cache(ψ, message_rank; boundary_mps_kwargs...)
    ψOψ = build_boundarymps_cache(ψO, ψ, message_rank; boundary_mps_kwargs...)

    return ratio(ψOψ, ψψ)
end

## Loop loop_corrections
function expect_loopcorrect(ψ::AbstractITensorNetwork, obs, max_circuit_size::Integer; max_genus::Integer=2, bp_update_kwargs=get_global_bp_update_kwargs())
    ## this is the entry point for when the state network is passed, and not the BP cache 
    ψIψ = build_bp_cache(ψ; bp_update_kwargs...)
    return expect_loopcorrect(ψIψ, obs, max_circuit_size; max_genus, bp_update_kwargs, update_cache=false)
end

function expect_loopcorrect(
    ψIψ::BeliefPropagationCache, obs::Tuple, max_circuit_size::Integer;
    max_genus::Integer=2, bp_update_kwargs=get_global_bp_update_kwargs(), update_cache=true
)

    # TODO: default max max_genus to ceil(max_circuit_size/min_loop_size)
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

    # now to getting the corrections
    return ratio(ψOψ, ψIψ) * loop_corrections(ψOψ, ψIψ, max_circuit_size; max_genus)
end

# between two states
function fidelity_loopcorrect(
    ψ::AbstractITensorNetwork, ϕ::AbstractITensorNetwork, max_circuit_size::Integer;
    max_genus::Integer=2, bp_update_kwargs=get_global_bp_update_kwargs()
)

    # TODO: default max max_genus to ceil(max_circuit_size/min_loop_size)
    # Warn if max_genus is 3 or larger lol
    if max_genus > 2
        @warn "Expectation value calculation with max_genus > 2 is not advised."
        # flush to instantly see the warning
        flush(stdout)
    end

    ψψ = build_bp_cache(ψ; bp_update_kwargs...)
    ϕϕ = build_bp_cache(ϕ; bp_update_kwargs...)
    ψϕ = build_bp_cache(ψ, ϕ; bp_update_kwargs...)


    # now to getting the corrections
    return fidelityratio(ψϕ, ψψ, ϕϕ) * loop_corrections(ψϕ, ψψ, ϕϕ, max_circuit_size; max_genus)
end

function fidelity_expect_loopcorrect(ψ::AbstractITensorNetwork, obs::Tuple, max_circuit_size::Integer; bp_update_kwargs=get_global_bp_update_kwargs(), max_genus::Integer=2)

    # TODO: default max max_genus to ceil(max_circuit_size/min_loop_size)
    # Warn if max_genus is 3 or larger lol
    if max_genus > 2
        @warn "Expectation value calculation with max_genus > 2 is not advised."
        # flush to instantly see the warning
        flush(stdout)
    end

    ψO = apply(obs, ψ)

    ψψ = build_bp_cache(ψ; bp_update_kwargs...)
    ψOψ = build_bp_cache(ψO, ψ; bp_update_kwargs...)

    return ratio(ψOψ, ψψ) * loop_corrections(ψOψ, ψψ, max_circuit_size; max_genus)
end


function loop_corrections(ψOψ::CacheNetwork, ψIψ::CacheNetwork, max_circuit_size::Integer; max_genus::Integer=2)

    ψIψ = normalize(ψIψ; update_cache=false)
    ψOψ = normalize(ψOψ; update_cache=false)

    # first get all the cycles
    circuits = enumerate_circuits(ψIψ, max_circuit_size; max_genus)

    # TODO: clever caching for multiple observables
    ψIψ_corrections = loop_correction_factor(ψIψ, circuits)
    ψOψ_corrections = loop_correction_factor(ψOψ, circuits)

    return ψOψ_corrections / ψIψ_corrections
end


function loop_corrections(ψϕ::CacheNetwork, ψψ::CacheNetwork, ϕϕ::CacheNetwork, max_circuit_size::Integer; max_genus::Integer=2)
    # all three need to be one the same partition graph

    ψψ = normalize(ψψ; update_cache=false)
    ϕϕ = normalize(ϕϕ; update_cache=false)
    ψϕ = normalize(ψϕ; update_cache=false)

    # first get all the cycles
    circuits = enumerate_circuits(ψψ, max_circuit_size; max_genus)

    # TODO: clever caching for multiple observables
    ψψ_corrections = loop_correction_factor(ψψ, circuits)
    ϕϕ_corrections = loop_correction_factor(ϕϕ, circuits)
    ψϕ_corrections = loop_correction_factor(ψϕ, circuits)

    return ψϕ_corrections / sqrt(ϕϕ_corrections) / sqrt(ψψ_corrections)
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
        qinds = [first(ertices(ψIψ))[1]] # the first vertex
    end

    op_vec = [string(o) for o in op]
    qinds_vec = _tovec(qinds)
    return op_vec, qinds_vec, coeff
end

_tovec(qinds) = vec(collect(qinds))
_tovec(qinds::NamedEdge) = [qinds.src, qinds.dst]


