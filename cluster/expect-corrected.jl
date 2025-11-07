using TensorNetworkQuantumSimulator
using NamedGraphs
using NamedGraphs: AbstractGraph, NamedGraph, AbstractNamedGraph
using NamedGraphs.GraphsExtensions: add_edge
using Graphs

const G = Graphs
const NG = NamedGraphs
const TN = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: BoundaryMPSCache
using HyperDualNumbers
using Adapt: adapt
using Dictionaries
using ITensors: inds, onehot, dag, commonind, op

# Cluster/loop/CC weights
function cluster_weights(bpc::BeliefPropagationCache, clusters::Vector, egs::Vector{<:AbstractNamedGraph}, interaction_graph; rescale::Bool = true)
    logZbp = logscalar(bpc)
    if rescale
        bpc = TN.rescale(bpc)
    end

    isempty(egs) && return [0], [[logZbp]], [[1]]
    
    circuit_lengths = sort(unique([c.weight for c=clusters]))

    # calculate weight of each generalized loop first
    wts = TN.weights(bpc, egs)

    logZs = Array{Array}(undef, length(circuit_lengths) + 1)
    logZs[1] = [logZbp]

    coeffs = Array{Array}(undef, length(circuit_lengths) + 1)
    coeffs[1] = [1]
    
    # now calculate contribution to logZ from each cluster
    for (cl_i, cl)=enumerate(circuit_lengths)
        clusters_cl = filter(c->c.weight==cl, clusters)
	logZs[cl_i + 1] = [prod([prod(fill(wts[l],c.multiplicities[l])) for l=c.loop_ids]) for c=clusters_cl]
	# weird bug in HyperDualNumbers where this doesn't work...
	# sum_ws = sum([TN.ursell_function(c, interaction_graph) * prod([wts[l]^c.multiplicities[l] for l=c.loop_ids]) for c=clusters_cl])
	coeffs[cl_i + 1] = [TN.ursell_function(c, interaction_graph) for c=clusters_cl]
    end

    return vcat([0],circuit_lengths), logZs, coeffs
end	    	    
    
function cc_weights(bpc::BeliefPropagationCache, regions::Vector, counting_nums::Dict; rescale::Bool=true)
    logZbp = logscalar(bpc)
    if rescale
        bpc = TN.rescale(bpc)
    end

    use_g = findall(gg->counting_nums[gg] != 0, regions)
    egs = [induced_subgraph(bpc.partitioned_tensornetwork.partitioned_graph, gg)[1] for gg=regions[use_g]]
    
    isempty(egs) && return [0], [[logZbp]], [[1]]

    # calculate weight of each generalized loop first
    wts = TN.full_weights(bpc, egs)

    return logZbp, log.(wts), [counting_nums[gg] for gg=regions[use_g]]
end


# Loop series expansion

function terms_to_scalar(numerator_numerator_terms, numerator_denominator_terms, denominator_numerator_terms, denominator_denominator_terms)
    return exp(sum(log.(numerator_numerator_terms)) - sum(log.(numerator_denominator_terms)) - sum(log.(denominator_numerator_terms)) + sum(log.(denominator_denominator_terms)))
end

#Project spins on sites v1 and v2 to v1_val (1 = up, 2 = down) and v2_val
function project!(ψIψ::BeliefPropagationCache, v1, v2, v1_val::Int64 = 1, v2_val::Int64=1)
    s1 = only(inds(only(ITN.factors(ψIψ, [(v1, "operator")])); plev = 0))
    s2 = only(inds(only(ITN.factors(ψIψ, [(v2, "operator")])); plev = 0))
    ITensorNetworks.@preserve_graph ψIψ[(v1, "operator")] = onehot(s1 => v1_val) * dag(onehot(s1' => v1_val))
    ITensorNetworks.@preserve_graph ψIψ[(v2, "operator")] = onehot(s2 => v2_val) * dag(onehot(s2' => v2_val))
end 

#Log scalar contraction of bpc
function logscalar(bpc::BeliefPropagationCache)
    nums, denoms = TN.scalar_factors_quotient(bpc)
    return sum(log.(nums)) - sum(log.(denoms))
end

function cumulative_weights(bpc::BeliefPropagationCache, egs::Vector{<:AbstractNamedGraph})
    isempty(egs) && [1]
    circuit_lengths = sort(unique(length.(edges.(egs))))
    outs = []
    for cl in circuit_lengths
        egs_cl = filter(eg -> length(edges(eg)) == cl, egs)
        sum_ws = sum(TN.weights(bpc, egs_cl))
        push!(outs, sum_ws)
    end

    # first one is the BP contribution
    outs = vcat([1.0], outs)
    return cumsum(outs)
end

function compute_ps!(ψIψ::BeliefPropagationCache, v1, v2, v1_val, v2_val, egs::Vector{<:AbstractNamedGraph}; track::Bool = true, cache_update_kwargs...)
    project!(ψIψ, v1, v2, v1_val, v2_val)

    if track
        ψIψ, bp_diffs = updatecache(ψIψ; track = true, cache_update_kwargs...)
    else
        ψIψ = updatecache(ψIψ; track = false, cache_update_kwargs...)
	bp_diffs = []
    end

    p_bp = exp(logscalar(ψIψ))

    ψIψ = ITensorNetworks.rescale(ψIψ)
    cfes = cumulative_weights(ψIψ, egs)
    
    return [p_bp*cfe for cfe in cfes], bp_diffs
end

function scalar_cumul_loop(bp_cache::BeliefPropagationCache, egs)
    zbp = exp(logscalar(bp_cache))
    bp_cache = ITN.rescale(bp_cache)

    cfes = cumulative_weights(bp_cache, egs)

    return [zbp * cfe for cfe=cfes]
end

function zz_bp_loopcorrect_connected(ψIψ::BeliefPropagationCache, verts, egs::Vector{<:AbstractNamedGraph}; cache_update_kwargs...)
    z_expects = [z_bp_loopcorrect(ψIψ, v, egs; cache_update_kwargs...)[1] for v=verts]
    zz_expect = zz_bp_loopcorrect(ψIψ, verts...,egs; cache_update_kwargs...)

    zz = (zz_expect[1,1] .+ zz_expect[2,2] .- (zz_expect[1,2] .+ zz_expect[2,1])) ./sum(zz_expect)
    vcat([zz], [(z[1] .- z[2]) ./ (z[1] .+ z[2]) for z=z_expects])
end

# compute z expectation value, but return cumulative weights rather than just total loop-corrected value
function z_bp_loopcorrect(ψIψ::BeliefPropagationCache, v, egs::Vector{<:AbstractNamedGraph}; cache_update_kwargs...)
    ψUpψ = TN.insert_observable(ψIψ, ("Proj0", [v]))
    ψUpψ, diffs_up = updatecache(ψUpψ; track = true, cache_update_kwargs...)
    p_up = scalar_cumul_loop(ψUpψ, egs)
    ψDnψ = TN.insert_observable(ψIψ, ("Proj1", [v]))
    ψDnψ, diffs_dn = updatecache(ψDnψ; track = true, cache_update_kwargs...)
    p_dn = scalar_cumul_loop(ψDnψ, egs)

    denominator = scalar_cumul_loop(ψIψ, egs)
    return [p_up, p_dn, denominator], [diffs_up, diffs_dn]
end

#Compute zz with loop correction and all bells and whistles
function zz_bp_loopcorrect(ψIψ::BeliefPropagationCache, v1, v2, egs::Vector{<:AbstractNamedGraph}; cache_update_kwargs...)
    probs = [compute_ps!(copy(ψIψ), v1, v2, i, j, egs; track = false, cache_update_kwargs...)[1] for i=1:2, j=1:2]
end

function cluster_twopoint_correlator(ψIψ::BeliefPropagationCache, obs, max_weight::Int)
    ng = ITN.partitioned_graph(ψIψ)
    op_strings, verts, _ = TN.collectobservable(obs)
    @assert length(verts)<=2
    clusters, egs, interaction_graph = enumerate_connected_clusters_twopoint(ng, verts..., max_weight)

    cluster_twopoint_correlator(ψIψ, obs, clusters, egs, interaction_graph)
end

function cluster_twopoint_correlator(ψIψ::BeliefPropagationCache, obs, clusters, egs, interaction_graph)
    # rescale BEFORE inserting operator
    ψIψ = TN.rescale(ψIψ)
    op_strings, verts, _ = TN.collectobservable(obs)
    @assert length(verts)<=2
    
    ψIψ_vs = [ψIψ[(v, "operator")] for v in verts]
    sinds =
        [commonind(ψIψ[(v, "ket")], ψIψ_vs[i]) for (i, v) in enumerate(verts)]
    

    coeffs = [Hyper(0,1,0,0),Hyper(0,0,1,0)]
    operators = [ψIψ[(v, "operator")].tensor[1,1] * adapt(typeof(ψIψ[(v, "operator")]))(Hyper(1,0,0,0) * op("I", sinds[i]) + coeffs[i] * op(op_strings[i], sinds[i])) for (i, v) in enumerate(verts)]
    ψOψ = ITN.update_factors(ψIψ, Dictionary([(v, "operator") for v in verts], operators))

    cluster_weights(ψOψ, clusters, egs, interaction_graph; rescale = false)
end

function cluster_onepoint(ψIψ::BeliefPropagationCache, obs, clusters, egs, interaction_graph)
    ψOψ, logZbp = prep_insertion(ψIψ, obs)
    lens, logZs, coeffs = cluster_weights(ψOψ, clusters, egs, interaction_graph; rescale = false)
    logZs[1] .+= logZbp
    lens, logZs, coeffs
end

# copied over from ITensorNetworks, 

# Also return the amount rescaled by
function rescale_partitions_norms(
  bpc::ITN.AbstractBeliefPropagationCache,
  partitions::Vector;
  verts::Vector=vertices(bpc, partitions),
)
  bpc = copy(bpc)
  tn = ITN.tensornetwork(bpc)

  # not sure why this is done in two steps
  norms = map(v -> inv(ITN.norm(tn[v])), verts)
  ITN.scale!(bpc, Dictionary(verts, norms))

  vertices_weights = Dictionary()
  for pv in partitions
    pv_vs = filter(v -> v ∈ verts, vertices(bpc, pv))
    isempty(pv_vs) && continue

    vn = ITN.region_scalar(bpc, pv)
    s = one(vn)#sign(vn) #isreal(vn) ? sign(vn) : one(vn)
    vn = s * vn^(-inv(oftype(vn, length(pv_vs))))
    set!(vertices_weights, first(pv_vs), s*vn)
    for v in pv_vs[2:length(pv_vs)]
      set!(vertices_weights, v, vn)
    end
  end

  ITN.scale!(bpc, vertices_weights)

  return bpc, Dictionary(verts,norms), vertices_weights
end

# cluster cumulant expansion

function cc_twopoint_correlator(ψIψ::BeliefPropagationCache, obs, regions::Vector, counting_nums::Dict)
    # rescale BEFORE inserting operator
    ψIψ = TN.rescale(ψIψ)
    op_strings, verts, _ = TN.collectobservable(obs)
    @assert length(verts)==2
    
    ψIψ_vs = [ψIψ[(v, "operator")] for v in verts]
    sinds =
        [commonind(ψIψ[(v, "ket")], ψIψ_vs[i]) for (i, v) in enumerate(verts)]
    
    coeffs = [Hyper(0,1,0,0),Hyper(0,0,1,0)]
    operators = [ψIψ[(v, "operator")].tensor[1,1] * adapt(typeof(ψIψ[(v, "operator")]))(Hyper(1,0,0,0) * op("I", sinds[i]) + coeffs[i] * op(op_strings[i], sinds[i])) for (i, v) in enumerate(verts)]
    ψOψ = ITN.update_factors(ψIψ, Dictionary([(v, "operator") for v in verts], operators))

    cc_weights(ψOψ, regions, counting_nums; rescale=false)
end

function cc_onepoint(ψIψ::BeliefPropagationCache, obs, regions::Vector, counting_nums::Dict)
    ψOψ, logZbp = prep_insertion(ψIψ, obs)
    lz, logZs, coeffs = cc_weights(ψOψ, regions, counting_nums; rescale=false)
    logZbp + lz, logZs, coeffs
end

function prep_insertion(ψIψ::BeliefPropagationCache, obs)
    # rescale BEFORE inserting operator
    ψIψ = ITN.rescale_messages(ψIψ)
    op_strings, verts, _ = TN.collectobservable(obs)
    @assert length(verts)<=2
    
    ψIψ_vs = [ψIψ[(v, "operator")] for v in verts]
    sinds =
        [commonind(ψIψ[(v, "ket")], ψIψ_vs[i]) for (i, v) in enumerate(verts)]
    
    coeffs = [Hyper(0,1,0,0),Hyper(0,0,1,0)]
    operators = [adapt(typeof(ψIψ[(v, "operator")]))(Hyper(1,0,0,0) * op("I", sinds[i]) + coeffs[i] * op(op_strings[i], sinds[i])) for (i, v) in enumerate(verts)]
    ψOψ = ITN.update_factors(ψIψ, Dictionary([(v, "operator") for v in verts], operators))

    ψOψ, norms, vertices_weights = rescale_partitions_norms(ψOψ, collect(ITN.partitions(ψIψ)))
    return ψOψ, sum([log(1/(norms[(v,tag)] * vertices_weights[(v,tag)])) for v=verts, tag = ["bra", "ket", "operator"]])
end

