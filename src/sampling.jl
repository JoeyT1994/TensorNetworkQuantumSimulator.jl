using StatsBase

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
    leaves = leaf_vertices(partitions_graph(supergraph(norm_bmps_cache)))
    seq = PartitionEdge.(a_star(partitions_graph(supergraph(norm_bmps_cache)), last(leaves), first(leaves)))
    norm_cache_message_update_kwargs = (; norm_cache_message_update_kwargs..., normalize = false)
    norm_bmps_cache = update(norm_bmps_cache; alg = "bp", edge_sequence = seq, maxiter = 1, message_update_alg = Algorithm("orthogonal"; norm_cache_message_update_kwargs...))

    #Generate the bit_strings moving left to right through the network
    probs_and_bitstrings = NamedTuple[]
    for j in 1:nsamples
        p_over_q_approx, logq, bitstring = get_one_sample(
            norm_bmps_cache, seq; projected_mps_bond_dimension, kwargs...
        )
        push!(probs_and_bitstrings, (poverq = p_over_q_approx, logq = logq, bitstring = bitstring))
    end

    return probs_and_bitstrings, ψ
end

"""
    sample(
        ψ::TensorNetworkState,
        nsamples::Integer;
        projected_message_rank::Integer,
        norm_message_rank::Integer,
        norm_message_update_kwargs=(; niters = _default_boundarymps_update_niters, tolerance = _default_boundarymps_update_tolerance),
        projected_message_update_kwargs = (;cutoff = _default_boundarymps_update_cutoff, maxdim = projected_message_rank),
        partition_by = "Row",
        kwargs...,
    )

Take nsamples bitstrings, based on the square of the coefficients of the vector defined by a 2D open boundary tensornetwork.

Arguments
---------
- `ψ::TensorNetworkState`: The tensornetwork state to sample from.
- `nsamples::Integer`: Number of samples to draw.

Keyword Arguments
-----------------
- alg ::String: The algorithm to use for sampling (default is "boundarymps", which is the only option currently supported).
Supported kwargs for alg = "boundarymps":
    - `projected_mps_bond_dimension::Int`: Bond dimension of the projected boundary MPS messages used during contraction of the projected state <x|ψ>.
    - `norm_mps_bond_dimension::Int`: Bond dimension of the boundary MPS messages used to contract <ψ|ψ>.
    - `norm_message_update_kwargs`: Keyword arguments for updating the norm boundary MPS messages.
    - `projected_message_update_kwargs`: Keyword arguments for updating the projected boundary MPS messages.
    - `partition_by`: How to partition the graph for boundary MPS (default is `"Row"`).

Returns
-------
A vector of bitstrings sampled from the probability distribution defined by as a dictionary mapping each vertex to a configuration (0...d).
"""
function sample(ψ::TensorNetworkState, nsamples::Integer; alg = "boundarymps", kwargs...)
    algorithm_check(ψ, "sample", alg)
    probs_and_bitstrings, _ = sample(Algorithm(alg), ψ, nsamples; kwargs...)
    # returns just the bitstrings
    return getindex.(probs_and_bitstrings, :bitstring)
end

"""
    sample_directly_certified(
        ψ::TensorNetworkState,
        nsamples::Integer;
        projected_message_rank::Integer,
        norm_message_rank::Integer,
        norm_message_update_kwargs=(; niters = _default_boundarymps_update_niters, tolerance = _default_boundarymps_update_tolerance),
        projected_message_update_kwargs = (;cutoff = _default_boundarymps_update_cutoff, maxdim = projected_message_rank),
        partition_by = "Row",
        kwargs...,
    )

Take nsamples bitstrings from a 2D open boundary tensornetwork.
The samples are drawn from x~q(x) and for each sample <x|ψ> is calculated "on-the-fly" to get a measure of p(x)/q(x).

Arguments
---------
- `ψ::TensorNetworkState`: The tensornetwork state to sample from.
- `nsamples::Integer`: Number of samples to draw.

Keyword Arguments
-----------------
- alg ::String: The algorithm to use for sampling (default is "boundarymps", which is the only option currently supported).
Supported kwargs for alg = "boundarymps":
    - `projected_mps_bond_dimension::Int`: Bond dimension of the projected boundary MPS messages used during contraction of the projected state <x|ψ>.
    - `norm_mps_bond_dimension::Int`: Bond dimension of the boundary MPS messages used to contract <ψ|ψ>.
    - `norm_message_update_kwargs`: Keyword arguments for updating the norm boundary MPS messages.
    - `projected_message_update_kwargs`: Keyword arguments for updating the projected boundary MPS messages.
    - `partition_by`: How to partition the graph for boundary MPS (default is `"Row"`).

Returns
-------
Vector of NamedTuples.
Each NamedTuple contains:
- `poverq`: Approximate value of p(x)/q(x) for the sampled bitstring x.
- `logq`: Log probability of drawing the bitstring.
- `bitstring`: The sampled bitstring as a dictionary mapping each vertex to a configuration (0...d).
"""
function sample_directly_certified(ψ::TensorNetworkState, nsamples::Integer; projected_mps_bond_dimension = 5 * maxvirtualdim(ψ), alg = "boundarymps", kwargs...)
    algorithm_check(ψ, "sample", alg)
    probs_and_bitstrings, _ = sample(Algorithm(alg), ψ, nsamples; projected_mps_bond_dimension, kwargs...)
    # returns the self-certified p/q, logq and bitstrings
    return probs_and_bitstrings
end

"""
    sample_certified(
        ψ::TensorNetworkState,
        nsamples::Integer;
        projected_message_rank::Integer,
        norm_message_rank::Integer,
        norm_message_update_kwargs=(; niters = _default_boundarymps_update_niters, tolerance = _default_boundarymps_update_tolerance),
        projected_message_update_kwargs = (;cutoff = _default_boundarymps_update_cutoff, maxdim = projected_message_rank),
        partition_by = "Row",
        kwargs...,
    )

Take nsamples bitstrings from a 2D open boundary tensornetwork.
The samples are drawn from x~q(x) and for each sample an independent contraction of <x|ψ> is performed to get a measure of p(x)/q(x).

Arguments
---------
- `ψ::TensorNetworkState`: The tensornetwork state to sample from.
- `nsamples::Integer`: Number of samples to draw.

Keyword Arguments
-----------------
- alg ::String: The algorithm to use for sampling (default is "boundarymps", which is the only option currently supported).
Supported kwargs for alg = "boundarymps":
    - `projected_mps_bond_dimension::Int`: Bond dimension of the projected boundary MPS messages used during contraction of the projected state <x|ψ>.
    - `norm_mps_bond_dimension::Int`: Bond dimension of the boundary MPS messages used to contract <ψ|ψ>.
    - `certification_mps_bond_dimension::Int`: Bond dimension of the boundary MPS messages used to contract <x|ψ> for certification.
    - `norm_message_update_kwargs`: Keyword arguments for updating the norm boundary MPS messages.
    - `projected_message_update_kwargs`: Keyword arguments for updating the projected boundary MPS messages.
    - `partition_by`: How to partition the graph for boundary MPS (default is `"Row"`).

Returns
-------
Vector of NamedTuples.
Each NamedTuple contains:
- `poverq`: Approximate value of p(x)/q(x) for the sampled bitstring x.
- `bitstring`: The sampled bitstring as a dictionary mapping each vertex to a configuration (0...d).
"""
function sample_certified(ψ::TensorNetworkState, nsamples::Int; alg = "boundarymps", certification_mps_bond_dimension = 5 * maxvirtualdim(ψ), certification_cache_message_update_kwargs = (;), kwargs...)
    algorithm_check(ψ, "sample", alg)
    probs_and_bitstrings, ψ = sample(Algorithm(alg), ψ, nsamples; kwargs...)
    # send the bitstrings and the logq to the certification function
    return certify_samples(ψ, probs_and_bitstrings; alg, certification_mps_bond_dimension, certification_cache_message_update_kwargs, symmetrize_and_normalize = false)
end

function get_one_sample(
        norm_bmps_cache::BoundaryMPSCache,
        seq::Vector{<:PartitionEdge};
        projected_mps_bond_dimension::Integer,
        kwargs...,
    )
    norm_bmps_cache = copy(norm_bmps_cache)
    cutoff, maxdim = 1.0e-10, projected_mps_bond_dimension

    bit_string = Dictionary{keytype(vertices(network(norm_bmps_cache))), Int}()
    p_over_q_approx = nothing
    logq = 0
    partitions = vcat(src.(reverse.(reverse(seq))), [src(first(seq))])
    incoming_mps = nothing
    for (i, partition) in enumerate(partitions)
        p_over_q_approx, _logq, bit_string, =
            sample_partition!(norm_bmps_cache, partition, bit_string; kwargs...)
        vs = vertices(supergraph(norm_bmps_cache), partition)
        logq += _logq

        if i < length(partitions)
            next_partition = partitions[i + 1]
            pe = PartitionEdge(parent(partition), parent(next_partition))

            mpo = ITensorMPS.MPO(norm_bmps_cache, src(pe); interpet_as_flat = true)
            if incoming_mps == nothing
                mpo = ITensorMPS.MPS(ITensor[mpo[i] for i in 1:length(mpo)])
                outgoing_mps = ITensorMPS.truncate(mpo; cutoff, maxdim = projected_mps_bond_dimension)
                outgoing_mps = merge_internal_tensors(outgoing_mps)
            else
                outgoing_mps = generic_apply(mpo, incoming_mps; cutoff, normalize = false, maxdim)
            end

            es = sorted_edges(norm_bmps_cache, pe)

            for (i, e) in enumerate(es)
                setmessage!(norm_bmps_cache, e, ITensor[outgoing_mps[i], prime(dag(outgoing_mps[i]))])
            end

            incoming_mps = outgoing_mps
        end

        i > 2 && delete_interpartition_messages!(norm_bmps_cache, PartitionEdge(parent(partitions[i - 2]) => parent(partitions[i - 1])))
    end

    return p_over_q_approx, logq, bit_string
end

#Sample along the column/ row specified by pv with the left incoming MPS message input and the right extractable from the cache
function sample_partition!(
        norm_bmps_cache::BoundaryMPSCache,
        partition::PartitionVertex,
        bit_string::Dictionary;
        kwargs...,
    )
    g = partition_graph(norm_bmps_cache, partition)
    leaves = leaf_vertices(g)
    seq = a_star(g, last(leaves), first(leaves))
    !isempty(seq) && update_partition!(norm_bmps_cache, seq)
    prev_v, traces = nothing, []
    logq = 0
    vs = vcat(src.(reverse.(reverse(seq))), [last(leaves)])
    for v in vs
        !isnothing(prev_v) && update_partition!(norm_bmps_cache, [NamedEdge(prev_v => v)])
        incoming_ms = incoming_messages(norm_bmps_cache, [v])
        ψv = network(norm_bmps_cache)[v]
        ψvdag = dag(prime(ψv))
        ts = [incoming_ms; [ψv, ψvdag]]
        seq = contraction_sequence(ts; alg = "optimal")
        ρ = contract(ts; sequence = seq)
        ρ_tr = tr(ρ)
        push!(traces, ρ_tr)
        ρ *= inv(ρ_tr)
        ρ_diag = collect(real.(diag(ITensors.array(ρ))))
        config = StatsBase.sample(1:length(ρ_diag), Weights(ρ_diag))
        # config is 1,2,...,d, but we want 0,1...,d-1 for the sample itself
        set!(bit_string, v, config - 1)
        s_ind = only(filter(i -> plev(i) == 0, inds(ρ)))
        P = adapt(datatype(ρ))(onehot(s_ind => config))
        q = ρ_diag[config]
        logq += log(q)
        Pψv = copy(network(norm_bmps_cache)[v]) * inv(sqrt(q)) * P
        setindex_preserve!(norm_bmps_cache, Pψv, v)
        prev_v = v
    end

    delete_partition_messages!(norm_bmps_cache, partition)

    return first(traces), logq, bit_string
end

function certify_sample(
        alg::Algorithm"boundarymps",
        ψ::TensorNetworkState, bitstring, logq::Number;
        certification_mps_bond_dimension::Integer,
        certification_cache_message_update_kwargs = (;),
        symmetrize_and_normalize = true,
    )
    if symmetrize_and_normalize
        ψ = gauge_and_scale(ψ)
    end

    ψproj = copy(tensornetwork(ψ))
    s = siteinds(ψ)
    qv = sqrt(exp(inv(oftype(logq, length(vertices(ψ)))) * logq))
    for v in vertices(ψ)
        P = adapt(datatype(ψproj[v]))(onehot(only(s[v]) => bitstring[v] + 1))
        setindex_preserve!(ψproj, ψproj[v] * P * inv(qv), v)
    end

    certification_mps_cache = BoundaryMPSCache(ψproj, certification_mps_bond_dimension)
    certification_cache_message_update_kwargs = (; normalize = false, certification_cache_message_update_kwargs...)

    certification_mps_cache = update(certification_mps_cache, message_update_alg = Algorithm("ITensorMPS"; certification_cache_message_update_kwargs...))
    p_over_q = partitionfunction(certification_mps_cache)
    p_over_q *= conj(p_over_q)

    return (poverq = p_over_q, bitstring = bitstring)
end

function certify_samples(ψ::TensorNetworkState, probs_and_bitstrings::Vector{<:NamedTuple}; alg = "boundarymps", kwargs...)
    algorithm_check(ψ, "sample", alg)
    return [certify_sample(Algorithm(alg), ψ, prob_and_bitstring.bitstring, prob_and_bitstring.logq; kwargs...) for prob_and_bitstring in probs_and_bitstrings]
end
