# function expect(tn1::ITensorNetwork, tn2::ITensorNetwork; imag_tol=1e-14)
#     # this is pure state fidelity
#     # TODO: make this function not just an alias for inner() but something that can be loop-corrected
#     val = inner(tn1, tn2; alg="bp")

#     return val
# end


function expect(tn, observable; max_loop_size=nothing, message_rank=nothing, kwargs...)
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


function expect(tn::BoundaryMPSCache, args...; kwargs...)
    return expect_boundarymps(tn, args...; kwargs...)
end

function expect_loopcorrect(ψ::ITensorNetwork, obs, max_circuit_size::Integer; max_genus::Integer=2, bp_update_kwargs=get_global_bp_update_kwargs())
    ## this is the entry point for when the state network is passed, and not the BP cache 
    ψIψ = build_bp_cache(ψ; bp_update_kwargs...)
    return expect_loopcorrect(ψIψ, obs, max_circuit_size; max_genus, bp_update_kwargs)
end

function expect_loopcorrect(
    ψIψ::BeliefPropagationCache, observable, max_circuit_size::Integer; max_genus::Integer=2, bp_update_kwargs=get_global_bp_update_kwargs()
)

    # TODO: default max genus to ceil(max_circuit_size/min_loop_size)
    # Warn if max_genus is 3 or larger lol
    if max_genus > 2
        @warn "Expectation value calculation with max_genus > 2 is not supported."
        # flush to instantly see the warning
        flush(stdout)
    end

    # first get all the cycles
    circuits = enumerate_circuits(ψIψ, max_circuit_size; max_genus)

    # update the norm cache once and hope that it is a good initialization for the ones with operator insterted
    ψIψ = updatecache(ψIψ; bp_update_kwargs...)

    # this is the denominator of the expectation fraction
    value_without_observable = loopcorrected_unnormalized_expectation(ψIψ, circuits; bp_update_kwargs, update_bp_cache=false)

    # this is when it is just a single observable
    if observable isa Tuple
        ψOψ = insert_observable(ψIψ, observable)
        value_with_observable = loopcorrected_unnormalized_expectation(ψOψ, circuits; bp_update_kwargs)
        return value_with_observable / value_without_observable
    end

    # this is when it is a vector of observables
    all_values_with_observable = [
        begin
            ψOψ = insert_observable(ψIψ, obs)
            loopcorrected_unnormalized_expectation(ψOψ, circuits; bp_update_kwargs)
        end for obs in observable
    ]

    return [value_with_observable / value_without_observable for value_with_observable in all_values_with_observable]


end


function loopcorrected_unnormalized_expectation(bp_cache::BeliefPropagationCache, circuits; update_bp_cache=true, bp_update_kwargs=get_global_bp_update_kwargs())
    if update_bp_cache
        bp_cache = updatecache(bp_cache; bp_update_kwargs...)
    end
    # TODO: separate out the scalar part which is also used elsewhere from the loop correction factor
    scaling = scalar(bp_cache)
    bp_cache = normalize(bp_cache; update_cache=false)

    # this is the denominator of the expectation fraction
    return scaling * loop_correction_factor(bp_cache, circuits)
end

## boundary MPS
# TODO: function that takes BP cache and turns it into MPS cache

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
    ψIψ::BoundaryMPSCache, observable::Tuple; boundary_mps_kwargs=get_global_boundarymps_update_kwargs()
)

    # TODO: modularize this with loop correction function for vectors of observables.

    ψOψ = insert_observable(ψIψ, observable)

    denom = scalar(ψIψ)


    ψOψ = update(ψOψ; boundary_mps_kwargs...)
    numer = scalar(ψOψ)

    return numer / denom
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


