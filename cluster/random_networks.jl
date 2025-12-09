using ITensors
using Random
using ProgressMeter
include("generalizedbp.jl")
include("utils.jl")
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

function initialize_region_graphs_correlation(L, emax, v_counts, verts; periodic=false, loop_size::Int=4)
    g = named_grid((L,L); periodic=periodic)
    clusters, egs, ig = TN.enumerate_clusters(g, emax; min_v = length(verts), triangle_free=true, must_contain=verts, min_deg = 1)
    regs = []
    cnums = []
    for v=v_counts
        R,_,c=TN.build_region_family_correlation(g, verts[1], verts[end], v)
	push!(regs, R)
	push!(cnums, c)
    end
    return (graph = g, clusters = clusters, egs = egs, interaction_graph = ig, regions = regs, counting_nums = cnums, gbp_regs = prep_gbp(g, loop_size))
end

function initialize_region_graphs(L, emax, v_counts; periodic=false, loop_size::Int=4)
    g = named_grid((L,L); periodic=periodic)
    clusters, egs, ig = TN.enumerate_clusters(g, emax; min_v = 4, triangle_free=true)
    regs = []
    cnums = []
    for v=v_counts
        R,_,c=TN.build_region_family(g, v; min_deg=2, min_v=4, triangle_free=true)
	push!(regs, R)
	push!(cnums, c)
    end
    return (graph = g, clusters = clusters, egs = egs, interaction_graph = ig, regions = regs, counting_nums = cnums, gbp_regs = prep_gbp(g,loop_size))
end

function random_free(region_data, χ; state::Bool = false, num_samples::Int=10, niters = 300, tol=1e-10, rate = 0.3)
    cluster_data = zeros(length(unique([c.weight for c=region_data.clusters]))+1, num_samples)
    cc_data = zeros(length(region_data.regions), num_samples)
    loop_data = zeros(num_samples)
    exact_data = zeros(num_samples)
    gbp_data = zeros(num_samples)
    gbp_regs = region_data.gbp_regs
    diff_data = Array{Array}(undef, num_samples)
    @showprogress for i=1:num_samples
        if state
	    T = random_tensornetworkstate(ComplexF64, region_data.graph, siteinds("S=1/2", region_data.graph); bond_dimension = χ)
	    exact_data[i] = real(log(TN.norm_sqr(T;alg="exact")))
	else
            T = uniform_random_tensornetwork(Float64, region_data.graph; bond_dimension=χ)
    	    exact_data[i] = real(log(TN.contract(T;alg="exact")))
	end
	
    	bpc = BeliefPropagationCache(T)
	bpc = update(bpc)
	gbp_msgs, diff_data[i] = generalized_belief_propagation(bpc, gbp_regs.bs, gbp_regs.ms, gbp_regs.ps, gbp_regs.cs, gbp_regs.b_nos, gbp_regs.mobius_nos; niters=niters, tol=tol, rate=rate)
	gbp_data[i] = -real(kikuchi_free_energy(bpc, gbp_regs.ms, gbp_regs.bs, gbp_msgs, gbp_regs.cs, gbp_regs.b_nos, gbp_regs.ps, gbp_regs.mobius_nos))
	loop_data[i] = real(log(TN.loopcorrected_partitionfunction(bpc, 4)))
	cluster_data[:,i] = real.(cluster_free(bpc, region_data.clusters, region_data.egs, region_data.interaction_graph)[2])
	for j=1:length(region_data.regions)
	    cc_data[j,i] = real.(cc_free(bpc, region_data.regions[j], region_data.counting_nums[j]; logZbp = cluster_data[1,i]))
	end
	
    end

    return (gbp_data = gbp_data, cluster_data = cluster_data, cc_data = cc_data, loop_data = loop_data, exact_data = exact_data)
end

function random_onepoint(region_data, χ, obs; num_samples::Int=10, get_exact::Bool=true, mps_bond_dimensions=[], niters=300,tol=1e-10, rate=0.3)
    cluster_data = zeros(length(unique([c.weight for c=region_data.clusters]))+1, num_samples)
    cc_data = zeros(length(region_data.regions), num_samples)
    gbp_data = zeros(num_samples)
    exact_data = zeros(num_samples)
    bmps_data = zeros(length(mps_bond_dimensions), num_samples)
    gbp_regs = region_data.gbp_regs
    diff_data = Array{Array}(undef, num_samples)
    @showprogress for i=1:num_samples
        ψ = random_tensornetworkstate(ComplexF64, region_data.graph, siteinds("S=1/2", region_data.graph); bond_dimension = χ)
    	bpc = BeliefPropagationCache(ψ)
	bpc = update(bpc)
	cluster_data[:,i] = real.(cluster_correlation(bpc, region_data.clusters, region_data.egs, region_data.interaction_graph, obs)[2])
	for j=1:length(region_data.regions)
	    cc_data[j,i] = real.(cc_correlation(bpc, region_data.regions[j], region_data.counting_nums[j], obs))
	end
	if get_exact
	    exact_data[i] = real(TN.expect(ψ, obs; alg = "exact"))
	end
	for (b_i,b)=enumerate(mps_bond_dimensions)
	    @time bmps_data[b_i,i] = real(TN.expect(ψ, obs; alg = "boundarymps", mps_bond_dimension=b))
	end
	@time gbp_msgs, diff_data[i] = generalized_belief_propagation(bpc, gbp_regs.bs, gbp_regs.ms, gbp_regs.ps, gbp_regs.cs, gbp_regs.b_nos, gbp_regs.mobius_nos; niters=niters, rate=rate, tol=tol)
	gbp_data[i] = real(expect_gbp(bpc, gbp_regs.bs, gbp_msgs, gbp_regs.cs, gbp_regs.ps, obs))

    end

    return (bmps_data = bmps_data, cluster_data = cluster_data, cc_data = cc_data, exact_data = exact_data, gbp_data = gbp_data, diff_data=diff_data)
end