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

function initialize_region_graphs(L, emax, v_counts; periodic=false)
    g = named_grid((L,L); periodic=periodic)
    clusters, egs, ig = TN.enumerate_clusters(g, emax; min_v = 4, triangle_free=true)
    regs = []
    cnums = []
    for v=v_counts
        R,_,c=TN.build_region_family(g, v; min_deg=2, min_v=4, triangle_free=true)
	push!(regs, R)
	push!(cnums, c)
    end
    return (graph = g, clusters = clusters, egs = egs, interaction_graph = ig, regions = regs, counting_nums = cnums)
end

function random_free(region_data, χ; state::Bool = false, num_samples::Int=10)
    cluster_data = zeros(length(unique([c.weight for c=region_data.clusters]))+1, num_samples)
    cc_data = zeros(length(region_data.regions), num_samples)
    loop_data = zeros(num_samples)
    exact_data = zeros(num_samples)
    gbp_data = zeros(num_samples)
    @showprogress for i=1:num_samples
        if state
	    ψ = random_tensornetworkstate(ComplexF64, region_data.graph, siteinds("S=1/2", region_data.graph); bond_dimension = χ)
	    # #Take its dagger
	    ψdag = map_virtualinds(prime, map_tensors(dag, ψ))

	    # Build the norm tensor network ψψ† and combine pairs of virtual inds
	    T = TensorNetwork(Dictionary(vertices(region_data.graph), [ψ[v]*ψdag[v] for v in vertices(region_data.graph)]))
	    TensorNetworkQuantumSimulator.combine_virtualinds!(T)
	else
            T = uniform_random_tensornetwork(Float64, region_data.graph; bond_dimension=χ)
	end
	
    	bpc = BeliefPropagationCache(T)
	bpc = update(bpc)
	bs = construct_gbp_bs(T)
	ms = construct_ms(bs)
	ps = all_parents(ms, bs)
	mobius_nos = mobius_numbers(ms, ps)
	ms, ps, mobius_nos = prune_ms_ps(ms, ps, mobius_nos)
	cs = children(ms, ps, bs)
	b_nos = calculate_b_nos(ms, ps, mobius_nos)
	
	gbp_data[i] = -real(generalized_belief_propagation(T, bs, ms, ps, cs, b_nos, mobius_nos; niters = 100, rate = 0.3))
	loop_data[i] = real(log(TN.loopcorrected_partitionfunction(bpc, 4)))
	cluster_data[:,i] = real.(cluster_free(bpc, region_data.clusters, region_data.egs, region_data.interaction_graph)[2])
	for j=1:length(region_data.regions)
	    cc_data[j,i] = real.(cc_free(bpc, region_data.regions[j], region_data.counting_nums[j]; logZbp = cluster_data[1,i]))
	end
    end

    return (gbp_data = gbp_data, cluster_data = cluster_data, cc_data = cc_data, loop_data = loop_data, exact_data = exact_data)
end