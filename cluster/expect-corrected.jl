using TensorNetworkQuantumSimulator
using NamedGraphs
using NamedGraphs: AbstractNamedGraph
using Graphs

const G = Graphs
const NG = NamedGraphs
const TN = TensorNetworkQuantumSimulator
using HyperDualNumbers
using Adapt: adapt
using Dictionaries

function prep_insertions(obs)
    if isnothing(obs)
        return (coeffs = identity, op_strings = v->"I")
    end
    op_strings, verts, _ = TN.collectobservable(obs)
    @assert length(verts) <= 2

    function hyper_coeff(v)
        if v==verts[1]
	    return Hyper(0,1,0,0)
	elseif length(verts)==2 && v==verts[2]
	    return Hyper(0,0,1,0)
	else
	    return 1
	end
    end

    function insertion_operator(v)
        if v==verts[1]
	    return op_strings[1]
	elseif length(verts)==2 && v==verts[2]
	    return op_strings[2]
	else
	    return "I"
	end
    end
    return (coeffs = hyper_coeff, op_strings = insertion_operator)
end

"""
Cluster expansion. See clustercorrections.jl
"""
function cluster_weights(bpc::BeliefPropagationCache, clusters::Vector, egs::Vector, interaction_graph; obs = nothing)

    kwargs = prep_insertions(obs)
        
    logZbp = TN.free_energy(bpc; kwargs...)
    isempty(egs) && return [0], [[logZbp]], [[1]]
    
    circuit_lengths = sort(unique([c.weight for c=clusters]))

    # Rescale the messages, but deal with the vertices separately
    TN.rescale_messages!(bpc)
    vns = Dictionary(TN.vertex_scalar(bpc, v; use_epsilon = true, kwargs...) for v=vertices(network(bpc).tensornetwork.graph))
    
    # calculate weight of each generalized loop first
    wts = TN.weights(bpc, egs; rescales = vns, kwargs...)
    
    logZs = Array{Array}(undef, length(circuit_lengths) + 1)
    logZs[1] = [logZbp]

    coeffs = Array{Array}(undef, length(circuit_lengths) + 1)
    coeffs[1] = [1]

    # now calculate contribution to logZ from each cluster
    for (cl_i, cl)=enumerate(circuit_lengths)
        clusters_cl = filter(c->c.weight==cl, clusters)
	logZs[cl_i + 1] = [prod([prod(fill(wts[l],c.multiplicities[l])) for l=c.loop_ids]) for c=clusters_cl]
	coeffs[cl_i + 1] = [TN.ursell_function(c, interaction_graph) for c=clusters_cl]
    end

    return vcat([0],circuit_lengths), logZs, coeffs
end	    	    

"""
Cluster cumulant expansion. See cumulant-clustercorrections.jl
"""
function cc_weights(bpc::BeliefPropagationCache, regions::Vector, counting_nums::Dict; obs = nothing, rescale::Bool = false)

    kwargs = prep_insertions(obs)
        
    use_g = findall(gg->counting_nums[gg] != 0, regions)
    egs = [induced_subgraph(network(bpc).tensornetwork.graph, gg)[1] for gg=regions[use_g]]
    
    isempty(egs) && return logZbp, [], []

    # Rescale the messages, but deal with the vertices separately
    if rescale
        TN.rescale_messages!(bpc)
    	vns = Dictionary(TN.vertex_scalar(bpc, v; use_epsilon = true, kwargs...) for v=vertices(network(bpc).tensornetwork.graph))
    else
        vns = Dictionary(1 for v=vertices(network(bpc).tensornetwork.graph))
    end
        
    # calculate weight of each cluster first
    wts = TN.weights(bpc, egs; rescales = vns, project_out = false, kwargs...)

    return log.(wts), [counting_nums[gg] for gg=regions[use_g]]
end

"""
onepoint or twopoint connected correlation function, using cluster cumulant expansion
"""
function cc_correlation(bpc::BeliefPropagationCache, regions::Vector, counting_nums::Dict, obs)
    logZs, cnums = cc_weights(bpc, regions, counting_nums; obs = obs)
    op_strings, verts, _ = TN.collectobservable(obs)
    if length(verts)==1
        return sum(logZs .* cnums).epsilon1
    else
        return sum(logZs .* cnums).epsilon12
    end
end

"""
onepoint or twopoint connected correlation function, using cluster expansion
"""
function cluster_correlation(bpc::BeliefPropagationCache, clusters::Vector, egs::Vector, interaction_graph, obs)
    cluster_wts, logZs, ursells = cluster_weights(bpc, clusters, egs, interaction_graph; obs = obs)
    op_strings, verts, _ = TN.collectobservable(obs)
    cumul_dat = cumsum([sum([logZs[i][j] * ursells[i][j] for j=1:length(logZs[i])]) for i=1:length(logZs)])
    if length(verts)==1
        return cluster_wts, [d.epsilon1 for d=cumul_dat]
    else
        return cluster_wts, [d.epsilon12 for d=cumul_dat]
    end
end