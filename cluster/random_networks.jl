using ITensors
using Random
using ProgressMeter
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
    @showprogress for i=1:num_samples
        if state
	    psi = random_tensornetworkstate(ComplexF64, region_data.graph, siteinds("S=1/2", region_data.graph); bond_dimension = χ)
   	    exact_data[i] = real(log(TN.norm_sqr(psi; alg="exact")))
	else
            psi = uniform_random_tensornetwork(Float64, region_data.graph; bond_dimension=χ)
    	    exact_data[i] = log(TN.contract(psi; alg="exact"))
	end
    	bpc = BeliefPropagationCache(psi)
	bpc = update(bpc)
	loop_data[i] = real(log(TN.loopcorrected_partitionfunction(bpc, 4)))
	cluster_data[:,i] = real.(cluster_free(bpc, region_data.clusters, region_data.egs, region_data.interaction_graph)[2])
	for j=1:length(region_data.regions)
	    cc_data[j,i] = real.(cc_free(bpc, region_data.regions[j], region_data.counting_nums[j]; logZbp = cluster_data[1,i]))
	end
    end

    return (cluster_data = cluster_data, cc_data = cc_data, loop_data = loop_data, exact_data = exact_data)
end