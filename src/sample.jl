using StatsBase

#Take nsamples bitstrings from a 2D open boundary tensornetwork using boundary MPS with relevant ranks
#Computes logq (logarithm of the probability of generating that sample conditioned on the specified message ranks)
#And an approximation to p/q which is good if the left message rank is high so the left environments converge
function _sample(
    ψ::ITensorNetwork,
    nsamples::Int64;
    projected_message_rank::Int64,
    norm_message_rank::Int64,
    norm_message_update_kwargs=(; niters = _default_boundarymps_update_niters, tolerance = _default_boundarymps_update_tolerance),
    projected_message_update_kwargs = (;cutoff = _default_boundarymps_update_cutoff, maxdim = projected_message_rank),
    kwargs...,
)
    ψ, ψψ = symmetric_gauge(ψ)
    ψ, ψψ = normalize(ψ, ψψ)

    norm_MPScache = BoundaryMPSCache(ψψ; message_rank=norm_message_rank)
    sorted_partitions = sort(ITensorNetworks.partitions(norm_MPScache))
    seq = [
        sorted_partitions[i] => sorted_partitions[i-1] for
        i = length(sorted_partitions):-1:2
    ]
    norm_message_update_kwargs = (; norm_message_update_kwargs..., normalize=false)
    norm_MPScache = update(Algorithm("orthogonal"), norm_MPScache, seq; norm_message_update_kwargs...)

    projected_MPScache = BoundaryMPSCache(ψ; message_rank=projected_message_rank)

    #Generate the bit_strings moving left to right through the network
    probs_and_bitstrings = NamedTuple[]
    for j = 1:nsamples
        p_over_q_approx, logq, bitstring = _get_one_sample(
            norm_MPScache, projected_MPScache, sorted_partitions; projected_message_update_kwargs, kwargs...)
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
function sample_directly_certified(ψ::ITensorNetwork, nsamples::Int64; projected_message_rank=5 * maxlinkdim(ψ), kwargs...)
    probs_and_bitstrings, _ = _sample(ψ::ITensorNetwork, nsamples::Int64; projected_message_rank, kwargs...)
    # returns the self-certified p/q, logq and bitstrings
    return probs_and_bitstrings
end

#Compute bitstrings and independently computed p/qs : a sufficiently large certification message rank should be used
function sample_certified(ψ::ITensorNetwork, nsamples::Int64; certification_message_rank=5 * maxlinkdim(ψ), certification_message_update_kwargs = (; maxdim = certification_message_rank, cutoff = _default_boundarymps_update_cutoff), kwargs...)
    probs_and_bitstrings, ψ = _sample(ψ::ITensorNetwork, nsamples::Int64; kwargs...)
    # send the bitstrings and the logq to the certification function
    return certify_samples(ψ, probs_and_bitstrings; certification_message_rank, certification_message_update_kwargs, symmetrize_and_normalize=false)
end

function _get_one_sample(
    norm_MPScache::BoundaryMPSCache,
    projected_MPScache::BoundaryMPSCache,
    sorted_partitions;
    projected_message_update_kwargs= (; cutoff = _default_boundarymps_update_cutoff, maxdim = maximum_virtual_dimension(projected_MPScache)),
    kwargs...,
)

    projected_message_update_kwargs = (; projected_message_update_kwargs..., normalize=false)

    norm_MPScache = copy(norm_MPScache)

    bit_string = Dictionary{keytype(vertices(projected_MPScache)),Int}()
    p_over_q_approx = nothing
    logq = 0
    for (i, partition) in enumerate(sorted_partitions)

        p_over_q_approx, _logq, bit_string, =
            sample_partition!(norm_MPScache, partition, bit_string; kwargs...)
        vs = planargraph_vertices(norm_MPScache, partition)
        logq += _logq

        projected_MPScache = update_factors(
            projected_MPScache,
            Dict(zip(vs, [copy(norm_MPScache[(v, "ket")]) for v in vs])),
        )


        if i < length(sorted_partitions)
            next_partition = sorted_partitions[i+1]
            
            #Alternate fitting procedure here which is faster for small bond dimensions but slower for large
            projected_MPScache = update(Algorithm("ITensorMPS"),
                projected_MPScache,
                partition => next_partition;
                projected_message_update_kwargs...)

            pes = planargraph_sorted_partitionedges(norm_MPScache, partition => next_partition)

            for pe in pes
                m = only(message(projected_MPScache, pe))
                set_message!(norm_MPScache, pe, [m, dag(prime(m))])
            end
        end

        i > 1 && delete_partitionpair_messages!(projected_MPScache, sorted_partitions[i-1] => sorted_partitions[i])
        i > 2 && delete_partitionpair_messages!(norm_MPScache, sorted_partitions[i-2] => sorted_partitions[i-1])
    end

    return p_over_q_approx, logq, bit_string
end

function certify_sample(
    ψ::ITensorNetwork, bitstring, logq::Number;
    certification_message_rank::Int64,
    certification_message_update_kwargs = (; maxdim = certification_message_rank, cutoff = _default_boundarymps_update_cutoff),
    symmetrize_and_normalize=true,
)
    if symmetrize_and_normalize
        ψ, ψψ = symmetric_gauge(ψ)
        ψ = normalize(ψ, cache! = Ref(ψψ))
    end

    ψproj = copy(ψ)
    s = siteinds(ψ)
    qv = sqrt(exp((1 / length(vertices(ψ))) * logq))
    for v in vertices(ψ)
        ψproj[v] = ψ[v] * onehot(only(s[v]) => bitstring[v] + 1) / qv
    end

    bmpsc = BoundaryMPSCache(ψproj; message_rank=certification_message_rank)

    pg = partitioned_graph(ppg(bmpsc))
    partition = first(center(pg))
    seq = [src(e) => dst(e) for e in post_order_dfs_edges(pg, partition)]

    bmpsc = update(Algorithm("ITensorMPS"), bmpsc, seq; certification_message_update_kwargs...)

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
function sample_partition!(
    ψIψ::BoundaryMPSCache,
    partition,
    bit_string::Dictionary;
    kwargs...,
)
    vs = sort(planargraph_vertices(ψIψ, partition))
    seq = PartitionEdge[PartitionEdge(vs[i] => vs[i-1]) for i in length(vs):-1:2]
    !isempty(seq) && partition_update!(ψIψ, seq)
    prev_v, traces = nothing, []
    logq = 0
    for v in vs
        !isnothing(prev_v) && partition_update!(ψIψ, [PartitionEdge(prev_v => v)])
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
        ψv = copy(ψIψ[(v, "ket")]) / sqrt(q)
        ψv = P * ψv
        setindex_preserve_graph!(ψIψ, ITensor(one(Bool)), (v, "operator"))
        setindex_preserve_graph!(ψIψ, copy(ψv), (v, "ket"))
        setindex_preserve_graph!(ψIψ, dag(prime(copy(ψv))), (v, "bra"))
        prev_v = v
    end

    delete_partition_messages!(ψIψ, partition)

    return first(traces), logq, bit_string
end
