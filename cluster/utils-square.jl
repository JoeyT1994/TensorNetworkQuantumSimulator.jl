using TensorNetworkQuantumSimulator: virtualinds
using NamedGraphs: unique_simplecycles_limited_length
using ITensors: Index, sim

using Graphs: topological_sort
using Graphs.SimpleGraphs: SimpleDiGraph

include("utils.jl")

function all_direct_parents(ms)
    # currently only implemented for 3 or fewer levels
    @assert length(ms)<=3
    ms_parents = [[] for j=1:length(ms)-1]
    for j=1:length(ms)-1
        for m=ms[j]
            push!(ms_parents[j], parents(m, ms[j+1]))
	end
    end
    if length(ms_parents)==1
        return [[[p] for p=ms_parents[1]]]
    else
        return [[[p, unique(vcat([ms_parents[2][i] for i=p]...))] for p=ms_parents[1]], [[p] for p=ms_parents[2]]]
    end
end

function all_direct_children(ms, ps)
    @assert length(ms)<=3
    cs = [[] for j=1:length(ms)-1]
    for level=1:length(ms)-1
        for i=1:length(ms[level+1])
	    my_children = []
            for j in 1:length(ms[level])
                if i in ps[level][j][1]
                    push!(my_children, j)
                end
            end
	    push!(cs[level], my_children)
        end
    end

    if length(cs)==1
        return [[[c] for c=cs[1]]]
    else
        return [[[c] for c=cs[1]], [[c, unique(vcat([cs[1][i] for i=c]...))] for c=cs[2]]]
    end
end

function get_mobius(ms, all_ps)
    all_mobius = vcat([ones(Int, length(ms[j])) for j=1:length(all_ps)], [ones(Int, length(ms[end]))])
    for j=length(all_ps):-1:1
        for i=1:length(ms[j])
            for k=1:length(all_ps[j][i])
                all_mobius[j][i] -= sum(all_mobius[k+j][all_ps[j][i][k]])
            end
        end
    end
    all_mobius
end

function initialize_direct_messages(ms, ps, T; normalize::Bool=true)
    ms_dicts = [Dictionary{Tuple{Int, Int}, ITensor}() for i=1:length(ps)]
    for j=1:length(ps)
    	for (i, m) in enumerate(ms[j])
            for p in ps[j][i][1]
                inds = reduce(vcat, [virtualinds(T, e) for e in filter(x -> x isa NamedEdge, m)])
            	if network(T) isa TensorNetworkState
                    inds = vcat(inds, prime.(inds))
                end
            	msg = ITensor(scalartype(T), 1.0, inds)
		if normalize
		    msg = ITensors.normalize(msg)
		end
		# label by parent, child
            	set!(ms_dicts[j], (p, i), msg)
            end
	end
    end
    return ms_dicts
end

function update_direct_message(level, alpha, beta, psi_alpha, msgs, new_msgs, ps, cs; rate = 1.0, normalize = true)
    
    m = deepcopy(psi_alpha)
    for c=setdiff(cs[level][alpha][1], [beta]) # children of alpha excluding beta
        for b=setdiff(ps[level][c][1], [alpha]) # parents of c excluding alpha
	    m = special_multiply(m, msgs[level][(b, c)])
	end
    end

    if level==1
        if length(ps)==2 # m is from edge e to vertex v 
            for b=ps[2][alpha][1] # plaquettes that contain the edge e
	        m = special_multiply(m,msgs[2][(b, alpha)])
	    end
	end
    else
        @assert level==2 # additional levels not implemented yet
	# vertices in alpha but not beta
	for c=setdiff(cs[2][alpha][2], cs[1][beta][1])
	    for b=setdiff(ps[1][c][1], cs[2][alpha][1]) # edges not in alpha
	        m = special_multiply(m, msgs[1][(b,c)])
	    end
	end
    end

    inds_to_sum_over = uniqueinds(m, msgs[level][(alpha,beta)])
    # perform sum in numerator
    for ind in inds_to_sum_over
        m = m * ITensor(1.0, ind)
    end

    if level==2
        # denominator: use new messages, edges in alpha to vertices in beta
	denom = ITensor(scalartype(psi_alpha), 1.0, inds(m))
	for c = cs[1][beta][1] # vertices in beta
	    @assert c in cs[2][alpha][2]
	    for b = intersect(ps[1][c][1], cs[2][alpha][1]) # parent edge in alpha
	    	if b==beta # parent edge not descended from beta
		    continue
		end
	        if isnothing(denom)
		    denom = deepcopy(new_msgs[1][(b,c)])
		else
	            denom = special_multiply(denom, new_msgs[1][(b,c)])
		end
	    end
	end
	if !isnothing(denom)
	    m = pointwise_division_raise(m, denom; power = 1)
	end
    end

    if normalize
        m = ITensors.normalize(m)
    end

    m = rate * m + (1-rate) * deepcopy(msgs[level][(alpha,beta)])

    if normalize
        m = ITensors.normalize(m)
    end

    return m
end	         	        

function update_direct_messages(psis, msgs, ms, ps, cs; kwargs...)
    new_msgs = deepcopy(msgs)
    diff = 0
    for j=1:length(ps)
        for (alpha, beta) in keys(msgs[j])
            new_msg = update_direct_message(j, alpha,  beta, psis[j+1][alpha], msgs, new_msgs, ps, cs; kwargs...)
	    diff += message_diff(new_msg, msgs[j][(alpha,beta)])
	    set!(new_msgs[j], (alpha,beta), new_msg)
	end
    end
    return new_msgs, diff / sum(length.(keys.(msgs)))
end

function yedidia_belief(level, alpha, psi_alpha, msgs, ps, cs)
    b = deepcopy(psi_alpha)
    if level<=length(msgs) # has parents
        for p=ps[level][alpha][1]
            b = special_multiply(b, msgs[level][(p, alpha)])
	end
    end

    if level>1 # has children
        # first do other parents of direct children
	for c=cs[level-1][alpha][1]
	    for p=setdiff(ps[level-1][c][1], [alpha])
		b = special_multiply(b, msgs[level-1][(p,c)])
	    end
	end
	# if level 3, also do parents of grandchildren, that aren't children of alpha
	if level==3
	    for c=cs[level-1][alpha][2]
	        for p=setdiff(ps[level-2][c][1], cs[level-1][alpha][1])
	            b = special_multiply(b, msgs[level-2][(p,c)])
		end
	    end
	end
    end

    b
end

function yedidia_gbp(T::BeliefPropagationCache, ms, ps, cs; niters::Int, rate::Number, tol=1e-10)
    msgs = initialize_direct_messages(ms, ps, T; normalize=true)

    all_msgs = Array{Array}(undef, niters+1)
    all_msgs[1] = msgs
    diffs = zeros(niters)
    tot_iters = niters
    psis = [[get_psi(T, m) for m=ms[j]] for j=1:length(ms)]
    for i=1:niters
        all_msgs[i+1], diffs[i] = update_direct_messages(psis, all_msgs[i], ms, ps, cs; normalize=true, rate)
	if diffs[i] < tol
	    tot_iters = i
	    break
	end
    end
    return all_msgs[2:tot_iters+1], diffs[1:tot_iters]
end

function yedidia_free_energy(T::BeliefPropagationCache, ms, msgs, ps, cs, mobius_nos)
    f = 0
    for j=1:length(mobius_nos)
        for i=1:length(mobius_nos[j])
            f += mobius_nos[j][i] * log(sum(yedidia_belief(j, i, get_psi(T, ms[j][i]), msgs, ps, cs)))
        end
    end
    return f
end

function yedidia_expect(T, ms, msgs, ps, cs, obs)
    op_strings, verts, _ = TN.collectobservable(obs, graph(T))

    # for now
    @assert length(verts)==1
    # only the outer clusters have factors
    alpha = findfirst(b->intersect(b,verts)==Set(verts),ms[end])
    
    psi_alpha_num = get_psi(T, ms[end][alpha]; op_strings = v->op_strings[1])

    num = sum(yedidia_belief(length(ms), alpha, psi_alpha_num, msgs, ps, cs))
    denom = sum(yedidia_belief(length(ms), alpha, get_psi(T, ms[end][alpha]), msgs, ps, cs))
    return num/denom
end

function prep_yedidia(g::NamedGraph, loop_size::Int; prune::Bool = true)
    bs = construct_gbp_bs(g, loop_size; include_factors = false)
    ms = construct_ms(bs)
    if loop_size==4
        all_ms = [filter(m->length(m)==1, ms), filter(m->length(m)==2, ms), bs]
    else
        @assert loop_size==0 # haven't implemented larger loops
	@assert all(el->length(el)==1, ms)
	all_ms = [ms, bs]
    end

    ps = all_direct_parents(all_ms)
    cs = all_direct_children(all_ms, ps)
    mobius_nos = get_mobius(all_ms, ps)

    if !prune
        return (ms = all_ms, ps = ps, cs = cs, mobius_nos = mobius_nos)
    else
        for i=1:length(mobius_nos)
	    use_i = findall(j->mobius_nos[i][j] != 0, 1:length(mobius_nos[i]))
	    all_ms[i] = all_ms[i][use_i]
	end
        ps = all_direct_parents(all_ms)
	cs = all_direct_children(all_ms, ps)
	mobius_nos = get_mobius(all_ms, ps)

	return (ms = all_ms, ps = ps, cs = cs, mobius_nos = mobius_nos)
    end
end