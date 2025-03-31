# function expect(tn1::ITensorNetwork, tn2::ITensorNetwork; imag_tol=1e-14)
#     # this is pure state fidelity
#     # TODO: make this function not just an alias for inner() but something that can be loop-corrected
#     val = inner(tn1, tn2; alg="bp")

#     return val
# end


function expect(ψ::ITensorNetwork, obs; bp_update_kwargs=_default_bp_update_kwargs, kwargs...)
    ## this is the entry point for when the state network is passed, and not the BP cache 
    ψIψ = build_bp_cache(ψ; bp_update_kwargs...)
    return expect(ψIψ, obs; bp_update_kwargs, kwargs...)
end


function expect(ψIψ::BeliefPropagationCache, observable; max_loop_size=0, bp_update_kwargs=_default_bp_update_kwargs, kwargs...)
    # TODO: wll there be another option than this expect_loopcorrect function?
    val = expect_loopcorrect(ψIψ, observable, max_loop_size; bp_update_kwargs, kwargs...)
    return val
end


function expect_loopcorrect(
    ψIψ::BeliefPropagationCache, observable, max_circuit_size; max_genus::Int64=2, bp_update_kwargs=_default_bp_update_kwargs
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
    ψIψ = update(ψIψ; bp_update_kwargs...)

    # this is the denominator of the expectation fraction
    value_without_observable = loopcorrected_unnormalized_expectation(ψIψ, circuits; update_bp_cache=false)

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


function loopcorrected_unnormalized_expectation(bp_cache::BeliefPropagationCache, circuits; update_bp_cache=true, bp_update_kwargs=_default_bp_update_kwargs)
    if update_bp_cache
        bp_cache = update(bp_cache; bp_update_kwargs...)
    end

    scaling = scalar(bp_cache)
    bp_cache = normalize(bp_cache)

    # this is the denominator of the expectation fraction
    return scaling * loop_correction_factor(bp_cache, circuits)
end


function insert_observable(ψIψ, obs::Tuple)
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
    qinds_vec = vec(collect(qinds))
    return op_vec, qinds_vec, coeff
end


## boundary MPS
function expect(ψIψ::BoundaryMPSBeliefPropagationCache, obs::Tuple; kwargs...)

    op_string = obs[1]
    qinds = obs[2]
    if length(obs) == 2
        coeff = 1.0
    else
        coeff = obs[3]
    end

    # length of qinds should be the same as the number of indices in op_string
    @assert length(qinds) == length(op_string) "Pauli string $(op_string) does not match the number of indices $(qinds)."

    if length(qinds) == 1
        v = qinds[]
        rdm = one_site_rdm(ψIψ, v; kwargs...)
        rdm /= tr(rdm)
        s = only(filter(i -> plev(i) == 0, inds(rdm)))
        val = (rdm*ITensors.op(op_string, s))[]
    elseif length(qinds) == 2
        v1, v2 = qinds
        if !(v1[1] == v2[1])
            # TODO:
            throw(ArgumentError("Operators needs to be in the same column, got columns $(v1[1]) and $(v2[1])."))
        end
        # TODO: extend this to arbitrary bodyness

        rdm = two_site_rdm(ψIψ, v1, v2; kwargs...)
        s1, s2 = first(filter(i -> plev(i) == 0, inds(rdm))),
        last(filter(i -> plev(i) == 0, inds(rdm)))
        val = ((rdm*ITensors.op(string(op_string[1]), s1))*ITensors.op(string(op_string[2]), s2))[] /
              ((rdm*ITensors.op("I", s1))*ITensors.op("I", s2))[]
    else
        throw(ArgumentError("Only 1 or 2 qubit operators are supported."))
    end

    return val * coeff
end

