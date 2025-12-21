using TensorNetworkQuantumSimulator
using NamedGraphs
using NamedGraphs: AbstractNamedGraph
using Graphs
using HyperDualNumbers

const G = Graphs
const NG = NamedGraphs
const TN = TensorNetworkQuantumSimulator
using Dictionaries

function prep_insertions(obs, g)
    if isnothing(obs)
        return (coeffs = v->1, op_strings = v->"I")
    end
    op_strings, verts, _ = TN.collectobservable(obs, g)
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

    kwargs = prep_insertions(obs, graph(bpc))
        
    logZbp = TN.free_energy(bpc; kwargs...)
    isempty(egs) && return [0], [[logZbp]], [[1]]
    
    circuit_lengths = sort(unique([c.weight for c=clusters]))

    # Rescale the messages, but deal with the vertices separately
    TN.rescale_messages!(bpc)
    if typeof(network(bpc))<:TensorNetworkState
        vns = Dictionary(TN.vertex_scalar(bpc, v; use_epsilon = true, kwargs...) for v=vertices(graph(bpc)))
    else
        vns = Dictionary(TN.vertex_scalar(bpc, v) for v=vertices(graph(bpc)))
    end
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

    kwargs = prep_insertions(obs, graph(bpc))
        
    use_g = findall(gg->counting_nums[gg] != 0, regions)
    egs = [induced_subgraph(graph(bpc), gg)[1] for gg=regions[use_g]]
    
    isempty(egs) && return [], []

    # Rescale the messages, but deal with the vertices separately
    if rescale
        TN.rescale_messages!(bpc)
	if typeof(network(bpc))<:TensorNetworkState
    	    vns = Dictionary(TN.vertex_scalar(bpc, v; use_epsilon = true, kwargs...) for v=vertices(graph(bpc)))
	else
	    vns = Dictionary(TN.vertex_scalar(bpc, v) for v=vertices(graph(bpc)))
	end
    else
        vns = Dictionary(1 for v=vertices(graph(bpc)))
    end
        
    # calculate weight of each cluster first
    wts = TN.weights(bpc, egs; rescales = vns, project_out = false, use_epsilon=true, kwargs...)

    return log.(wts), [counting_nums[gg] for gg=regions[use_g]]
end

"""
onepoint or twopoint connected correlation function, using cluster cumulant expansion
"""
function cc_correlation(bpc::BeliefPropagationCache, regions::Vector, counting_nums::Dict, obs)
    logZs, cnums = cc_weights(bpc, regions, counting_nums; obs = obs)
    op_strings, verts, _ = TN.collectobservable(obs, graph(bpc))
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
    op_strings, verts, _ = TN.collectobservable(obs, graph(bpc))
    cumul_dat = cumsum([sum([logZs[i][j] * ursells[i][j] for j=1:length(logZs[i])]) for i=1:length(logZs)])
    if length(verts)==1
        return cluster_wts, [d.epsilon1 for d=cumul_dat]
    else
        return cluster_wts, [d.epsilon12 for d=cumul_dat]
    end
end

function cluster_free(bpc::BeliefPropagationCache, clusters::Vector, egs::Vector, interaction_graph)
    cluster_wts, logZs, ursells = cluster_weights(bpc, clusters, egs, interaction_graph)
    cumul_dat = cumsum([sum([logZs[i][j] * ursells[i][j] for j=1:length(logZs[i])]) for i=1:length(logZs)])
    return cluster_wts, cumul_dat
end

function cc_free(bpc::BeliefPropagationCache, regions::Vector, counting_nums::Dict; logZbp=nothing)
    logZs, cnums = cc_weights(bpc, regions, counting_nums; rescale=true)
    if isnothing(logZbp)
        logZbp = TN.free_energy(bpc)
    end
    return sum(logZs .* cnums) + logZbp
end

function prep_op_strings(obs, g)
    if isnothing(obs)
        return v->"I"
    end
    op_strings, verts, _ = TN.collectobservable(obs, g)
    @assert length(verts) <= 2

    function insertion_operator(v)
        if v==verts[1]
	    return op_strings[1]
	elseif length(verts)==2 && v==verts[2]
	    return op_strings[2]
	else
	    return "I"
	end
    end
    return insertion_operator
end

"""
Cluster cumulant expansion. No rescaling or hyperdual numbers
"""
function cc_weights_nohyper(bpc::BeliefPropagationCache, regions::Vector, counting_nums::Dict; obs = nothing)

    op_strings = prep_op_strings(obs, graph(bpc))
        
    use_g = findall(gg->counting_nums[gg] != 0, regions)
    egs = [induced_subgraph(graph(bpc), gg)[1] for gg=regions[use_g]]
    
    isempty(egs) && return [], []

    # Rescale the messages, but deal with the vertices separately
    wts = TN.weights(bpc, egs; project_out = false, use_epsilon=false,op_strings = op_strings)

    return wts, [counting_nums[gg] for gg=regions[use_g]]
end

"""
onepoint function, using cluster cumulant expansion, geometric mean.
This can give wacky results sometimes
"""
function cc_one_point_geometric(bpc::BeliefPropagationCache, regions::Vector, counting_nums::Dict, obs)
    denoms, cnums = cc_weights_nohyper(bpc, regions, counting_nums; obs = nothing)
    nums, _ = cc_weights_nohyper(bpc, regions, counting_nums; obs = obs)
    return prod((nums ./ denoms) .^ cnums)
end

"""
twopoint function, using cluster cumulant expansion, geometric version
"""
function cc_two_point_geometric(bpc::BeliefPropagationCache, regions::Vector, counting_nums::Dict, obs)
    denoms, cnums = cc_weights_nohyper(bpc, regions, counting_nums; obs = nothing)
    op_strings, verts, _ = TN.collectobservable(obs, graph(bpc))
    nums_sep = [cc_weights_nohyper(bpc, regions, counting_nums; obs = (op_strings[i], [verts[i]]))[1] for i=1:2]
    nums_both, _ = cc_weights_nohyper(bpc, regions, counting_nums; obs = obs)
    return prod((nums_both ./ denoms) .^ cnums) - prod((nums_sep[1] .* nums_sep[2] ./ denoms .^2) .^ cnums)
end
