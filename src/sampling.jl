using StatsBase

# Sample vertex v of the cache from its marginal. Returns the configuration (1, …, d), its
# probability, the trace of the unnormalised marginal and the projector onto the configuration.
function _sample_vertex(cache::AbstractBeliefPropagationCache, v)
    ψv = network(cache)[v]
    ρ = contract_optimal([incoming_messages(cache, v); ψv; dag(prime(ψv))])
    ρ_tr = tr(ρ)
    p = collect(real.(diag(array(ρ / ρ_tr))))
    k = StatsBase.sample(eachindex(p), Weights(p))
    return k, p[k], ρ_tr, adapt_like(ρ, onehot(only(inds(ρ; plev = 0)) => k))
end

# Each sampler also returns the state it sampled from (gauged and rescaled, if `gauge_state`), which
# is what the samples' probabilities refer to.
function sample(
        alg::Algorithm"bp",
        ψ::TensorNetworkState,
        nsamples::Integer;
        bp_update_kwargs = default_bp_update_kwargs(ψ),
        gauge_state = true,
        kwargs...,
    )
    bp_cache = converged_cache(alg, ψ; cache_update_kwargs = bp_update_kwargs)
    if gauge_state
        bp_cache = symmetrize_and_normalize(bp_cache)
    end

    #Generate the bit_strings moving left to right through the network
    probs_and_bitstrings = NamedTuple[]
    for j in 1:nsamples
        projected_bp_cache = copy(bp_cache)
        bit_string = Dictionary{keytype(vertices(ψ)), Int}()
        for v in vertices(ψ)
            k, _, _, P = _sample_vertex(projected_bp_cache, v)
            # the sample itself is 0, …, d - 1
            set!(bit_string, v, k - 1)
            setindex_preserve!(projected_bp_cache, network(projected_bp_cache)[v] * P, v)
            v != last(vertices(ψ)) && (projected_bp_cache = update(projected_bp_cache; bp_update_kwargs...))
        end
        push!(probs_and_bitstrings, (bitstring = bit_string,))
    end

    return probs_and_bitstrings, network(bp_cache)
end

function sample(
        alg::Algorithm"boundarymps",
        ψ::TensorNetworkState,
        nsamples::Integer;
        projected_mps_bond_dimension::Integer,
        norm_mps_bond_dimension::Integer,
        norm_cache_message_update_kwargs = (;),
        partition_by = "row",
        gauge_state = true,
        kwargs...,
    )
    norm_bmps_cache = BoundaryMPSCache(ψ, norm_mps_bond_dimension; gauge_state, partition_by)
    leaves = leaf_vertices(quotient_graph(supergraph(norm_bmps_cache)))
    seq = QuotientEdge.(a_star(quotient_graph(supergraph(norm_bmps_cache)), last(leaves), first(leaves)))
    norm_cache_message_update_kwargs = (; norm_cache_message_update_kwargs..., normalize = false)
    norm_bmps_cache = update(norm_bmps_cache; alg = "bp", edge_sequence = seq, maxiter = 1, message_update_alg = Algorithm("fitting"; norm_cache_message_update_kwargs...))

    #Generate the bit_strings moving left to right through the network
    probs_and_bitstrings = NamedTuple[]
    for j in 1:nsamples
        p_over_q_approx, logq, bitstring = get_one_sample(
            norm_bmps_cache, seq; projected_mps_bond_dimension, kwargs...
        )
        push!(probs_and_bitstrings, (poverq = p_over_q_approx, logq = logq, bitstring = bitstring))
    end

    return probs_and_bitstrings, network(norm_bmps_cache)
end

"""
    sample(ψ::TensorNetworkState, nsamples::Integer; alg, kwargs...)

Draw `nsamples` bitstrings from the probability distribution defined by the squared amplitudes of the tensor network state.

# Arguments
- `ψ::TensorNetworkState`: The tensor network state to sample from.
- `nsamples::Integer`: Number of samples to draw.

# Keyword Arguments
- `alg::String`: The algorithm to use for sampling (`"boundarymps"` and `"bp"` currently supported).
- For `alg = "boundarymps"`:
    - `projected_mps_bond_dimension::Int`: Bond dimension of the projected boundary MPS messages used during contraction of the projected state ⟨x|ψ⟩.
    - `norm_mps_bond_dimension::Int`: Bond dimension of the boundary MPS messages used to contract ⟨ψ|ψ⟩.
    - `norm_cache_message_update_kwargs`: Keyword arguments for updating the norm boundary MPS messages.
    - `partition_by`: How to partition the graph for boundary MPS (default is `"row"`).
    - `gauge_state`: Whether to gauge the state before sampling (default is `true`).
- For `alg = "bp"`:
    - `bp_update_kwargs`: Keyword arguments for updating the belief propagation cache.
    - `gauge_state`: Whether to gauge the state before sampling (default is `true`).

# Returns
- A vector of bitstrings, each a dictionary mapping vertices to configurations (0...d-1).
"""
function sample(ψ::TensorNetworkState, nsamples::Integer; alg = nothing, kwargs...)
    algorithm_check(ψ, "sample", alg)
    probs_and_bitstrings, _ = sample(Algorithm(alg), ψ, nsamples; kwargs...)
    # returns just the bitstrings
    return getindex.(probs_and_bitstrings, :bitstring)
end

"""
    sample_directly_certified(ψ::TensorNetworkState, nsamples::Integer; alg, kwargs...)

Draw `nsamples` bitstrings from a 2D open-boundary tensor network state. Samples are drawn from x ~ q(x) and for each sample ⟨x|ψ⟩ is calculated on-the-fly to estimate p(x)/q(x).

# Arguments
- `ψ::TensorNetworkState`: The tensor network state to sample from.
- `nsamples::Integer`: Number of samples to draw.

# Keyword Arguments
- `alg::String`: The algorithm to use for sampling (`"boundarymps"` currently supported).
- For `alg = "boundarymps"`:
    - `projected_mps_bond_dimension::Int`: Bond dimension of the projected boundary MPS messages used during contraction of the projected state ⟨x|ψ⟩.
    - `norm_mps_bond_dimension::Int`: Bond dimension of the boundary MPS messages used to contract ⟨ψ|ψ⟩.
    - `norm_cache_message_update_kwargs`: Keyword arguments for updating the norm boundary MPS messages.
    - `partition_by`: How to partition the graph for boundary MPS (default is `"row"`).
    - `gauge_state`: Whether to gauge the state before sampling (default is `true`).

# Returns
- A vector of `NamedTuple`s, each containing:
    - `poverq`: Approximate value of p(x)/q(x) for the sampled bitstring x.
    - `logq`: Log probability of drawing the bitstring.
    - `bitstring`: The sampled bitstring as a dictionary mapping each vertex to a configuration (0...d-1).
"""
function sample_directly_certified(ψ::TensorNetworkState, nsamples::Integer; projected_mps_bond_dimension = 5 * maxvirtualdim(ψ), alg = nothing, kwargs...)
    algorithm_check(ψ, "sample_certified", alg)
    probs_and_bitstrings, _ = sample(Algorithm(alg), ψ, nsamples; projected_mps_bond_dimension, kwargs...)
    # returns the self-certified p/q, logq and bitstrings
    return probs_and_bitstrings
end

"""
    sample_certified(ψ::TensorNetworkState, nsamples::Integer; alg, kwargs...)

Draw `nsamples` bitstrings from a 2D open-boundary tensor network state. Samples are drawn from x ~ q(x) and for each sample an independent contraction of ⟨x|ψ⟩ is performed to estimate p(x)/q(x).

# Arguments
- `ψ::TensorNetworkState`: The tensor network state to sample from.
- `nsamples::Integer`: Number of samples to draw.

# Keyword Arguments
- `alg::String`: The algorithm to use for sampling (`"boundarymps"` currently supported).
- For `alg = "boundarymps"`:
    - `projected_mps_bond_dimension::Int`: Bond dimension of the projected boundary MPS messages used during contraction of the projected state ⟨x|ψ⟩.
    - `norm_mps_bond_dimension::Int`: Bond dimension of the boundary MPS messages used to contract ⟨ψ|ψ⟩.
    - `certification_mps_bond_dimension::Int`: Bond dimension of the boundary MPS messages used to contract ⟨x|ψ⟩ for certification.
    - `norm_cache_message_update_kwargs`: Keyword arguments for updating the norm boundary MPS messages.
    - `certification_cache_message_update_kwargs`: Keyword arguments for the independent certification contraction.
    - `partition_by`: How to partition the graph for boundary MPS (default is `"row"`).
    - `gauge_state`: Whether to gauge the state before sampling (default is `true`).

# Returns
- A vector of `NamedTuple`s, each containing:
    - `poverq`: Approximate value of p(x)/q(x) for the sampled bitstring x.
    - `bitstring`: The sampled bitstring as a dictionary mapping each vertex to a configuration (0...d-1).
"""
function sample_certified(ψ::TensorNetworkState, nsamples::Int; alg = nothing, certification_mps_bond_dimension = 5 * maxvirtualdim(ψ), certification_cache_message_update_kwargs = (;), kwargs...)
    algorithm_check(ψ, "sample_certified", alg)
    # certify against the state the samples were drawn from, as it was gauged and rescaled for them
    probs_and_bitstrings, ψ_sampled = sample(Algorithm(alg), ψ, nsamples; kwargs...)
    return certify_samples(ψ_sampled, probs_and_bitstrings; alg, certification_mps_bond_dimension, certification_cache_message_update_kwargs, gauge_state = false)
end

function get_one_sample(
        norm_bmps_cache::BoundaryMPSCache,
        seq::Vector{<:QuotientEdge};
        projected_mps_bond_dimension::Integer,
        kwargs...,
    )
    norm_bmps_cache = copy(norm_bmps_cache)
    cutoff, maxdim = 1.0e-10, projected_mps_bond_dimension

    bit_string = Dictionary{keytype(vertices(network(norm_bmps_cache))), Int}()
    p_over_q_approx = nothing
    logq = 0
    partitions = [dst.(reverse(seq)); src(first(seq))]
    incoming_mps = nothing
    for (i, partition) in enumerate(partitions)
        p_over_q_approx, _logq, bit_string, =
            sample_partition!(norm_bmps_cache, partition, bit_string; kwargs...)
        logq += _logq

        if i < length(partitions)
            next_partition = partitions[i + 1]
            pe = QuotientEdge(parent(partition), parent(next_partition))

            # Apply the (already projected) ket row to the running single-layer boundary MPS. The
            # cache messages are doubled ket/bra, so we feed the ket-only `incoming_mps` explicitly.
            mpo, mps, right_inds = _bmps_apply_inputs(norm_bmps_cache, pe; incoming_mps)
            outgoing_mps = generic_apply(mpo, mps, right_inds; cutoff, maxdim, normalize = false)

            es = sorted_edges(norm_bmps_cache, pe)

            for (k, e) in enumerate(es)
                setmessage!(norm_bmps_cache, e, Tensor[outgoing_mps[k], prime(dag(outgoing_mps[k]))])
            end

            incoming_mps = outgoing_mps
        end

        i > 2 && delete_interpartition_messages!(norm_bmps_cache, QuotientEdge(parent(partitions[i - 2]) => parent(partitions[i - 1])))
    end

    return p_over_q_approx, logq, bit_string
end

#Sample along the column/ row specified by pv with the left incoming MPS message input and the right extractable from the cache
function sample_partition!(
        norm_bmps_cache::BoundaryMPSCache,
        partition::QuotientVertex,
        bit_string::Dictionary;
        kwargs...,
    )
    g = partition_graph(norm_bmps_cache, partition)
    leaves = leaf_vertices(g)
    seq = a_star(g, last(leaves), first(leaves))
    !isempty(seq) && update_partition!(norm_bmps_cache, seq)
    logq = 0.0
    first_trace = nothing
    vs = [dst.(reverse(seq)); last(leaves)]
    for (i, v) in enumerate(vs)
        i > 1 && update_partition!(norm_bmps_cache, [NamedEdge(vs[i - 1] => v)])
        k, q, ρ_tr, P = _sample_vertex(norm_bmps_cache, v)
        i == 1 && (first_trace = ρ_tr)
        # the sample itself is 0, …, d - 1
        set!(bit_string, v, k - 1)
        logq += log(q)
        setindex_preserve!(norm_bmps_cache, network(norm_bmps_cache)[v] * inv(sqrt(q)) * P, v)
    end

    delete_partition_messages!(norm_bmps_cache, partition)

    return first_trace, logq, bit_string
end

function certify_sample(
        alg::Algorithm"boundarymps",
        ψ::TensorNetworkState, bitstring, logq::Number;
        certification_mps_bond_dimension::Integer,
        certification_cache_message_update_kwargs = (;),
        gauge_state = true,
    )
    if gauge_state
        ψ = gauge_and_scale(ψ)
    end

    ψproj = copy(tensornetwork(ψ))
    s = siteinds(ψ)
    qv = sqrt(exp(inv(oftype(logq, length(vertices(ψ)))) * logq))
    for v in vertices(ψ)
        P = adapt_like(ψproj[v], onehot(only(s[v]) => bitstring[v] + 1))
        setindex_preserve!(ψproj, ψproj[v] * P * inv(qv), v)
    end

    certification_mps_cache = BoundaryMPSCache(ψproj, certification_mps_bond_dimension)
    certification_cache_message_update_kwargs = (; normalize = false, certification_cache_message_update_kwargs...)

    certification_mps_cache = update(certification_mps_cache, message_update_alg = Algorithm("zipup"; certification_cache_message_update_kwargs...))
    p_over_q = partitionfunction(certification_mps_cache)
    p_over_q *= conj(p_over_q)

    return (poverq = p_over_q, bitstring = bitstring)
end

function certify_samples(ψ::TensorNetworkState, probs_and_bitstrings::Vector{<:NamedTuple}; alg = "boundarymps", kwargs...)
    algorithm_check(ψ, "sample_certified", alg)
    return [certify_sample(Algorithm(alg), ψ, prob_and_bitstring.bitstring, prob_and_bitstring.logq; kwargs...) for prob_and_bitstring in probs_and_bitstrings]
end
