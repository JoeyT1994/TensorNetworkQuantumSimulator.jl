using ITensors
using Random
using ProgressMeter
include("generalizedbp.jl")
include("utils-square.jl")
include("expect-corrected.jl")

function uniform_random_itensor(rng::AbstractRNG, ::Type{S}, is; a = 0) where {S <: Number}
    T = ITensor(S, undef, is)
    U = ITensor(S, undef, is)
    rand!(rng, storage(T))
    fill!(U, a) # uniform shift
    return T + U
end
    
function uniform_random_tensornetwork(eltype, g::AbstractGraph; bond_dimension::Integer = 1, a=0)
    vs = collect(vertices(g))
    l = Dict(e => Index(bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vs
        is = [l[NamedEdge(v => vn)] for vn in neighbors(g, v)]
        set!(tensors, v, uniform_random_itensor(Random.default_rng(), eltype, is; a = a))
    end
    return TensorNetwork(tensors, g)
end

function initialize_region_graphs_correlation(L, emax, v_counts, verts; periodic=false, loop_size::Int=4, prune::Bool=true)
    g = named_grid((L,L); periodic=periodic)
    clusters, egs, ig = TN.enumerate_clusters(g, emax; min_v = length(verts), triangle_free=true, must_contain=verts, min_deg = 1)
    regs = []
    cnums = []
    for v=v_counts
        R,_,c=TN.build_region_family_correlation(g, verts[1], verts[end], v)
	push!(regs, R)
	push!(cnums, c)
    end
    return (graph = g, clusters = clusters, egs = egs, interaction_graph = ig, regions = regs, counting_nums = cnums, yedidia_regs = prep_yedidia(g, loop_size; prune=prune), gbp_regs = prep_gbp(g, loop_size))
end

function initialize_region_graphs(L, emax, v_counts; periodic=false, loop_size::Int=4, prune::Bool=true)
    g = named_grid((L,L); periodic=periodic)
    clusters, egs, ig = TN.enumerate_clusters(g, emax; min_v = 4, triangle_free=true)
    regs = []
    cnums = []
    for v=v_counts
        R,_,c=TN.build_region_family(g, v; min_deg=2, min_v=4, triangle_free=true)
	push!(regs, R)
	push!(cnums, c)
    end
    return (graph = g, clusters = clusters, egs = egs, interaction_graph = ig, regions = regs, counting_nums = cnums, yedidia_regs = prep_yedidia(g, loop_size; prune = prune), gbp_regs = prep_gbp(g,loop_size))
end


function random_free(region_data, χ; state::Bool = false, num_samples::Int=10, niters = 300, tol=1e-10, rate = 0.3)
    cluster_data = zeros(ComplexF64,length(unique([c.weight for c=region_data.clusters]))+1, num_samples)
    cc_data = zeros(ComplexF64,length(region_data.regions), num_samples)
    loop_data = zeros(ComplexF64,num_samples)
    exact_data = zeros(ComplexF64,num_samples)
    gbp_data = zeros(ComplexF64,num_samples)
    yedidia_data = zeros(ComplexF64,num_samples)
    yedidia_regs = region_data.yedidia_regs
    gbp_regs = region_data.gbp_regs
    gbp_diffs = Array{Array}(undef, num_samples)
    yedidia_diffs = Array{Array}(undef, num_samples)
    
    @showprogress for i=1:num_samples
        if state
	    T = random_tensornetworkstate(ComplexF64, region_data.graph, siteinds("S=1/2", region_data.graph); bond_dimension = χ)
	    exact_data[i] = (log(TN.norm_sqr(T;alg="exact")))
	else
            T = uniform_random_tensornetwork(Float64, region_data.graph; bond_dimension=χ)
    	    exact_data[i] = (log(TN.contract(T;alg="exact")))
	end
	
    	bpc = BeliefPropagationCache(T)
	bpc = update(bpc)
	@time yedidia_msgs, yedidia_diffs[i] = yedidia_gbp(bpc, yedidia_regs.ms, yedidia_regs.ps, yedidia_regs.cs; niters=niters, tol=tol,rate=rate)
	yedidia_data[i] = yedidia_free_energy(bpc, yedidia_regs.ms, yedidia_msgs[end], yedidia_regs.ps, yedidia_regs.cs, yedidia_regs.mobius_nos)
	gbp_msgs, gbp_diffs[i] = generalized_belief_propagation(bpc, gbp_regs.bs, gbp_regs.ms, gbp_regs.ps, gbp_regs.cs, gbp_regs.b_nos, gbp_regs.mobius_nos; niters=niters, tol=tol, rate=rate)
	gbp_data[i] = -(kikuchi_free_energy(bpc, gbp_regs.ms, gbp_regs.bs, gbp_msgs, gbp_regs.cs, gbp_regs.b_nos, gbp_regs.ps, gbp_regs.mobius_nos))
	loop_data[i] = (log(TN.loopcorrected_partitionfunction(bpc, 4)))
	cluster_data[:,i] = (cluster_free(bpc, region_data.clusters, region_data.egs, region_data.interaction_graph)[2])
	for j=1:length(region_data.regions)
	    cc_data[j,i] = (cc_free(bpc, region_data.regions[j], region_data.counting_nums[j]; logZbp = cluster_data[1,i]))
	end
	
    end

    return (yedidia_diffs = yedidia_diffs, yedidia_data = yedidia_data, gbp_diffs = gbp_diffs, gbp_data = gbp_data, cluster_data = cluster_data, cc_data = cc_data, loop_data = loop_data, exact_data = exact_data)
end

function random_onepoint(region_data, χ, obs; num_samples::Int=10, get_exact::Bool=true, mps_bond_dimensions=[], niters=300,tol=1e-10, rate=0.3)
    cluster_data = zeros(ComplexF64,length(unique([c.weight for c=region_data.clusters]))+1, num_samples)
    cc_data = zeros(ComplexF64,length(region_data.regions), num_samples)
    exact_data = zeros(ComplexF64,num_samples)
    bmps_data = zeros(ComplexF64,length(mps_bond_dimensions), num_samples)
    gbp_regs = region_data.gbp_regs
    gbp_data = zeros(ComplexF64,num_samples)
    yedidia_data = zeros(ComplexF64,num_samples)
    yedidia_regs = region_data.yedidia_regs
    gbp_regs = region_data.gbp_regs
    gbp_diffs = Array{Array}(undef, num_samples)
    yedidia_diffs = Array{Array}(undef, num_samples)

    @showprogress for i=1:num_samples
        ψ = random_tensornetworkstate(ComplexF64, region_data.graph, siteinds("S=1/2", region_data.graph); bond_dimension = χ)
    	bpc = BeliefPropagationCache(ψ)
	bpc = update(bpc)
	cluster_data[:,i] = (cluster_correlation(bpc, region_data.clusters, region_data.egs, region_data.interaction_graph, obs)[2])
	for j=1:length(region_data.regions)
	    cc_data[j,i] = (cc_correlation(bpc, region_data.regions[j], region_data.counting_nums[j], obs))
	end
	if get_exact
	    exact_data[i] = (TN.expect(ψ, obs; alg = "exact"))
	end
	for (b_i,b)=enumerate(mps_bond_dimensions)
	    @time bmps_data[b_i,i] = (TN.expect(ψ, obs; alg = "boundarymps", mps_bond_dimension=b))
	end
	@time gbp_msgs, diff_data[i] = generalized_belief_propagation(bpc, gbp_regs.bs, gbp_regs.ms, gbp_regs.ps, gbp_regs.cs, gbp_regs.b_nos, gbp_regs.mobius_nos; niters=niters, rate=rate, tol=tol)
	gbp_data[i] = (expect_gbp(bpc, gbp_regs.bs, gbp_msgs, gbp_regs.cs, gbp_regs.ps, obs))
	@time yedidia_msgs, yedidia_diffs[i] = yedidia_gbp(bpc, yedidia_regs.ms, yedidia_regs.ps, yedidia_regs.cs; niters=niters, tol=tol,rate=rate)
	yedidia_data[i] = yedidia_expect(bpc, yedidia_regs.ms, yedidia_msgs[end], yedidia_regs.ps, yedidia_regs.cs, obs)

    end

    return (bmps_data = bmps_data, cluster_data = cluster_data, cc_data = cc_data, exact_data = exact_data, gbp_data = gbp_data, gbp_diffs = gbp_diffs,yedidia_data = yedidia_data, yedidia_diffs = yedidia_diffs)
end

function random_all(region_data, region_data_corr, χ, obs; num_samples::Int=10, niters = 300, tol=1e-10, rate = 0.3, old_states = nothing)
    cluster_data = zeros(ComplexF64,length(unique([c.weight for c=region_data.clusters]))+1, num_samples)
    cluster_data_corr = zeros(ComplexF64,length(unique([c.weight for c=region_data_corr.clusters]))+1, num_samples)
    cc_data = zeros(ComplexF64,length(region_data.regions), num_samples)
    cc_data_corr = zeros(ComplexF64,length(region_data_corr.regions), num_samples)
    loop_data = zeros(ComplexF64,num_samples)
    exact_data = zeros(ComplexF64,2,num_samples)
    gbp_data = zeros(ComplexF64,2,num_samples)
    yedidia_data = zeros(ComplexF64,2,num_samples)
    yedidia_regs = region_data.yedidia_regs
    gbp_regs = region_data.gbp_regs
    gbp_diffs = Array{Array}(undef, num_samples)
    yedidia_diffs = Array{Array}(undef, num_samples)
    states = Array{BeliefPropagationCache}(undef, num_samples)
    gbp_msgss = Array{Dictionary}(undef, num_samples)
    yedidia_msgss = Array{Array}(undef, num_samples)
    @showprogress for i=1:num_samples
        if !isnothing(old_states)
	    T = network(old_states[i])
	    bpc = old_states[i]
	    states[i] = copy(bpc)
	else
	    T = random_tensornetworkstate(ComplexF64, region_data.graph, siteinds("S=1/2", region_data.graph); bond_dimension = χ)
    	    bpc = BeliefPropagationCache(T)
	    bpc = update(bpc)
	    states[i] = copy(bpc)
	end

	# exact
	exact_data[1,i] = log(TN.norm_sqr(T;alg="exact"))
	exact_data[2,i] = TN.expect(T, obs; alg="exact")

	# generalized BP
	@time yedidia_msgs, yedidia_diffs[i] = yedidia_gbp(bpc, yedidia_regs.ms, yedidia_regs.ps, yedidia_regs.cs; niters=niters, tol=tol,rate=rate)
	yedidia_data[1,i] = yedidia_free_energy(bpc, yedidia_regs.ms, yedidia_msgs[end], yedidia_regs.ps, yedidia_regs.cs, yedidia_regs.mobius_nos)
	yedidia_data[2,i] = yedidia_expect(bpc, yedidia_regs.ms, yedidia_msgs[end], yedidia_regs.ps, yedidia_regs.cs, obs)
	yedidia_msgss[i] = yedidia_msgs[end]
	gbp_msgs, gbp_diffs[i] = generalized_belief_propagation(bpc, gbp_regs.bs, gbp_regs.ms, gbp_regs.ps, gbp_regs.cs, gbp_regs.b_nos, gbp_regs.mobius_nos; niters=niters, tol=tol, rate=rate)
	gbp_data[2,i] = expect_gbp(bpc, gbp_regs.bs, gbp_msgs, gbp_regs.cs, gbp_regs.ps, obs)
	gbp_data[1,i] = -(kikuchi_free_energy(bpc, gbp_regs.ms, gbp_regs.bs, gbp_msgs, gbp_regs.cs, gbp_regs.b_nos, gbp_regs.ps, gbp_regs.mobius_nos))
	gbp_msgss[i] = gbp_msgs
	
	# cluster expansions
	loop_data[i] = (log(TN.loopcorrected_partitionfunction(bpc, 4)))
	cluster_data[:,i] = (cluster_free(bpc, region_data.clusters, region_data.egs, region_data.interaction_graph)[2])
	cluster_data_corr[:,i] = (cluster_correlation(bpc, region_data_corr.clusters, region_data_corr.egs, region_data_corr.interaction_graph, obs)[2])
	for j=1:length(region_data.regions)
	    cc_data[j,i] = cc_free(bpc, region_data.regions[j], region_data.counting_nums[j]; logZbp = cluster_data[1,i])
	end
	for j=1:length(region_data_corr.regions)
	    cc_data_corr[j,i] = cc_correlation(bpc, region_data_corr.regions[j], region_data_corr.counting_nums[j], obs)
	end

	
    end

    return Dict("yedidia"=>Dict("diffs"=>yedidia_diffs, "free"=>yedidia_data[1,:], "expect"=>yedidia_data[2,:], "messages"=>yedidia_msgss), "gbp"=>Dict("diffs"=>gbp_diffs, "free"=>gbp_data[1,:], "expect"=>gbp_data[2,:], "messages"=>gbp_msgss), "cluster"=>Dict("free"=>cluster_data, "expect"=>cluster_data_corr), "loop"=>Dict("free"=>loop_data), "cc"=>Dict("free"=>cc_data, "expect"=>cc_data_corr), "exact"=>Dict("free"=>exact_data[1,:], "expect"=>exact_data[2,:]), "states"=>states)
end
