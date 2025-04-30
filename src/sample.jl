using StatsBase

#Take nsamples bitstrings from a 2D open boundary tensornetwork using boundary MPS with relevant ranks
#Computes logq (logarithm of the probability of generating that sample conditioned on the specified message ranks)
#And an approximation to p/q which is good if the left message rank is high so the left environments converge
function _sample(
    ψ::ITensorNetwork,
    nsamples::Int64;
    left_message_rank::Int64,
    right_message_rank::Int64,
    boundary_mps_kwargs=(; message_update_kwargs=(; niters=75, tolerance=1e-12)),
    kwargs...,
)
    ψ, ψψ = symmetric_gauge(ψ)
    ψ, ψψ = normalize(ψ, ψψ)

    right_MPScache = BoundaryMPSCache(ψψ; message_rank=right_message_rank)
    sorted_partitions = sort(ITensorNetworks.partitions(right_MPScache))
    seq = [
        sorted_partitions[i] => sorted_partitions[i-1] for
        i = length(sorted_partitions):-1:2
    ]
    right_message_update_kwargs = (; boundary_mps_kwargs[:message_update_kwargs]..., normalize=false)
    right_MPScache = update(Algorithm("orthogonal"), right_MPScache, seq; right_message_update_kwargs...)

    left_MPScache = BoundaryMPSCache(ψ; message_rank=left_message_rank)

    #Generate the bit_strings moving left to right through the network
    probs_and_bitstrings = NamedTuple[]
    for j = 1:nsamples
        p_over_q_approx, logq, bitstring = _get_one_sample(
            right_MPScache, left_MPScache, sorted_partitions; boundary_mps_kwargs, kwargs...)
        push!(probs_and_bitstrings, (poverq=p_over_q_approx, logq=logq, bitstring=bitstring))
    end

    return probs_and_bitstrings, ψ
end

#Compute bitstrings conditioned on whatever kwargs used
function sample(ψ::ITensorNetwork, nsamples::Int64; kwargs...)
    probs_and_bitstrings, _ = _sample(ψ::ITensorNetwork, nsamples::Int64; kwargs...)
    # returns just the bitstrings
    return getindex.(probs_and_bitstrings, :bitstring)
end

#Compute bitstrings and corresponding p/qs : a sufficiently large left message rank should be used
function sample_directly_certified(ψ::ITensorNetwork, nsamples::Int64; left_message_rank=5 * maxlinkdim(ψ), kwargs...)
    probs_and_bitstrings, _ = _sample(ψ::ITensorNetwork, nsamples::Int64; left_message_rank, kwargs...)
    # returns the self-certified p/q, logq and bitstrings
    return probs_and_bitstrings
end

#Compute bitstrings and independently computed p/qs : a sufficiently large certification message rank should be used
function sample_certified(ψ::ITensorNetwork, nsamples::Int64; certification_message_rank=5 * maxlinkdim(ψ), boundary_mps_kwargs=(; message_update_kwargs=(; niters=75, tolerance=1e-12)), kwargs...)
    probs_and_bitstrings, ψ = _sample(ψ::ITensorNetwork, nsamples::Int64; boundary_mps_kwargs, kwargs...)
    # send the bitstrings and the logq to the certification function
    return certify_samples(ψ, probs_and_bitstrings; boundary_mps_kwargs, certification_message_rank, symmetrize_and_normalize=false)
end

function _get_one_sample(
    right_MPScache::BoundaryMPSCache,
    left_MPScache::BoundaryMPSCache,
    sorted_partitions;
    boundary_mps_kwargs=(; message_update_kwargs=(; niters=75, tolerance=1e-12)),
    kwargs...
)

    # TODO: this is not used?
    left_message_update_kwargs = (; boundary_mps_kwargs[:message_update_kwargs]..., truncate_at_end=true, truncate_kwargs=(; cutoff=1e-10), normalize=false)

    right_MPScache = copy(right_MPScache)

    bit_string = Dictionary{keytype(vertices(left_MPScache)),Int}()
    p_over_q_approx = nothing
    logq = 0
    for (i, partition) in enumerate(sorted_partitions)

        right_MPScache, p_over_q_approx, _logq, bit_string, =
            sample_partition(right_MPScache, partition, bit_string; kwargs...)
        vs = planargraph_vertices(right_MPScache, partition)
        logq += _logq

        left_MPScache = update_factors(
            left_MPScache,
            Dict(zip(vs, [only(factors(right_MPScache, [(v, "ket")])) for v in vs])),
        )


        if i < length(sorted_partitions)
            next_partition = sorted_partitions[i+1]

            ms = messages(right_MPScache)

            # left_MPScache = update(
            #     Algorithm("orthogonal"),
            #     left_MPScache,
            #     partition => next_partition;
            #     left_message_update_kwargs...,
            # )

            #Alternate fitting procedure here which is faster for small bond dimensions but slower for large
            left_MPScache = update(
                Algorithm("ITensorMPS"),
                left_MPScache,
                partition => next_partition;
                cutoff=1e-10, maxdim=maximum_virtual_dimension(left_MPScache))

            pes = planargraph_sorted_partitionedges(right_MPScache, partition => next_partition)

            for pe in pes
                m = only(message(left_MPScache, pe))
                set!(ms, pe, [m, dag(prime(m))])
            end
        end

        i > 1 && delete_partitionpair_messages!(left_MPScache, sorted_partitions[i-1] => sorted_partitions[i])
        i > 2 && delete_partitionpair_messages!(right_MPScache, sorted_partitions[i-2] => sorted_partitions[i-1])
    end

    return p_over_q_approx, logq, bit_string
end

function certify_sample(
    ψ::ITensorNetwork, bitstring, logq::Number;
    certification_message_rank::Int64,
    boundary_mps_kwargs=(; message_update_kwargs=(; niters=75, tolerance=1e-12)),
    symmetrize_and_normalize=true,
)
    if symmetrize_and_normalize
        ψ, ψψ = symmetric_gauge(ψ)
        ψ, _ = normalize(ψ, ψψ)
    end

    ψproj = copy(ψ)
    s = siteinds(ψ)
    qv = sqrt(exp((1 / length(vertices(ψ))) * logq))
    for v in vertices(ψ)
        ψproj[v] = ψ[v] * onehot(only(s[v]) => bitstring[v] + 1) / qv
    end

    # TODO: this is not used?
    message_update_kwargs = (; boundary_mps_kwargs[:message_update_kwargs]..., truncate_at_end=true, truncate_kwargs=(; cutoff=1e-12), normalize=false)
    bmpsc = BoundaryMPSCache(ψproj; message_rank=certification_message_rank)

    pg = partitioned_graph(ppg(bmpsc))
    partition = first(center(pg))
    seq = [src(e) => dst(e) for e in post_order_dfs_edges(pg, partition)]

    #bmpsc = update(Algorithm("orthogonal"), bmpsc, seq; message_update_kwargs...)

    #Alternate fitting procedure here which is faster for small bond dimensions but may be bad for large
    bmpsc = update(Algorithm("ITensorMPS"), bmpsc, seq; cutoff=1e-10, maxdim=certification_message_rank)

    p_over_q = region_scalar(bmpsc, partition)
    p_over_q *= conj(p_over_q)

    return (poverq=p_over_q, bitstring=bitstring)
end

certify_sample(ψ, logq_and_bitstring::NamedTuple; kwargs...) = certify_sample(ψ, logq_and_bitstring.bitstring, logq_and_bitstring.logq; kwargs...)

function certify_samples(ψ::ITensorNetwork, bitstrings, logqs::Vector{<:Number}; kwargs...)
    return [certify_sample(ψ, bitstring, logq; kwargs...) for (bitstring, logq) in zip(bitstrings, logqs)]
end

function certify_samples(ψ::ITensorNetwork, probs_and_bitstrings::Vector{<:NamedTuple}; kwargs...)
    return [certify_sample(ψ, prob_and_bitstring; kwargs...) for prob_and_bitstring in probs_and_bitstrings]
end

#Sample along the column/ row specified by pv with the left incoming MPS message input and the right extractable from the cache
function sample_partition(
    ψIψ::BoundaryMPSCache,
    partition,
    bit_string::Dictionary;
    kwargs...,
)
    vs = sort(planargraph_vertices(ψIψ, partition))
    prev_v, traces = nothing, []
    logq = 0
    for v in vs
        ψIψ =
            !isnothing(prev_v) ? partition_update(ψIψ, [prev_v], [v]) :
            partition_update(ψIψ, [v])
        env = environment(bp_cache(ψIψ), [(v, "operator")])
        seq = contraction_sequence(env; alg="optimal")
        ρ = contract(env; sequence=seq)
        ρ_tr = tr(ρ)
        push!(traces, ρ_tr)
        ρ /= ρ_tr
        # the usual case of single-site
        config = StatsBase.sample(1:length(diag(ρ)), Weights(real.(diag(ρ))))
        # config is 1 or 2, but we want 0 or 1 for the sample itself
        set!(bit_string, v, config - 1)
        s_ind = only(filter(i -> plev(i) == 0, inds(ρ)))
        P = onehot(s_ind => config)
        q = diag(ρ)[config]
        logq += log(q)
        ψv = only(factors(ψIψ, [(v, "ket")])) / sqrt(q)
        ψv = P * ψv
        ψIψ = update_factor(ψIψ, (v, "operator"), ITensor(one(Bool)))
        ψIψ = update_factor(ψIψ, (v, "ket"), ψv)
        ψIψ = update_factor(ψIψ, (v, "bra"), dag(prime(ψv)))
        prev_v = v
    end

    delete_partition_messages!(ψIψ, partition)

    return ψIψ, first(traces), logq, bit_string
end
