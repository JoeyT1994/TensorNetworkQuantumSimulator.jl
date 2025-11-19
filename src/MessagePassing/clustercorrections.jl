
using NamedGraphs
using NamedGraphs: AbstractNamedGraph
using ProgressMeter

include("../graph_enumeration.jl")

struct Cluster
    loop_ids::Vector{Int}
    multiplicities::Dict{Int, Int}
    weight::Int
    total_loops::Int
end

struct Loop
    vertices::Vector{Int}
    edges::Vector{Tuple{Int,Int}}
    weight::Int
end


function canonical_cluster_signature(cluster::Cluster)
    items = [(loop_id, cluster.multiplicities[loop_id]) for loop_id in sort(cluster.loop_ids)]
    return (tuple(items...), cluster.weight)
end

function build_interaction_graph(loops::Vector{Loop})
    """Build interaction graph with optimizations for speed."""
    interaction_graph = Dict{Int, Vector{Int}}()
    n_loops = length(loops)
    
    println("  Building optimized interaction graph for $n_loops loops...")
    flush(stdout)
    progress = Progress(n_loops, dt=0.1, desc="Building graph: ", color=:green, barlen=50)
    
    # Optimization 1: Pre-compute vertex sets once
    vertex_sets = [Set(loop.vertices) for loop in loops]
    
    # Optimization 2: Build vertex-to-loops mapping for faster lookup
    vertex_to_loops = Dict{Int, Vector{Int}}()
    for (i, loop) in enumerate(loops)
        for vertex in loop.vertices
            if !haskey(vertex_to_loops, vertex)
                vertex_to_loops[vertex] = Int[]
            end
            push!(vertex_to_loops[vertex], i)
        end
    end
    
    for i in 1:n_loops
        interaction_graph[i] = unique(vcat([vertex_to_loops[v] for v=loops[i].vertices]...))
	        
        next!(progress)
    end
    
    return interaction_graph
end

"""
Enumerate connected clusters using DFS starting from loops supported on target site.
Connectivity is guaranteed by growing through the interaction graph.
Courtesy of Frank Zhang and Siddhant Midha
"""
function dfs_enumerate_clusters_from_supported(all_loops::Vector{Loop}, supported_loop_ids::Vector{Int}, max_weight::Int, interaction_graph::Dict{Int, Vector{Int}}; verbose::Bool = false)
    clusters = Cluster[]
    seen_clusters = Set{Tuple}()
    cluster_count = 0
    
    verbose && println("  Starting DFS cluster enumeration...")
    verbose && println("  Supported loops: $(length(supported_loop_ids)), Max weight: $max_weight")
    
    # Progress tracking
    last_report_time = time()
    last_cluster_count = 0
    
    # DFS to grow clusters starting from each supported loop
    function dfs_grow_cluster(current_cluster::Vector{Int}, current_weight::Int, 
                             has_supported::Bool)
        
        # If we've found a valid cluster (has supported loop), record it
        if has_supported && current_weight >= 1
            # Create cluster with multiplicities
            multiplicities = Dict{Int, Int}()
            for loop_id in current_cluster
                multiplicities[loop_id] = get(multiplicities, loop_id, 0) + 1
            end
            
            cluster = Cluster(
                collect(keys(multiplicities)),
                multiplicities,
                current_weight,
                length(current_cluster)
            )
            
            # Avoid duplicates using canonical signature
            signature = canonical_cluster_signature(cluster)
            if !(signature in seen_clusters)
                push!(seen_clusters, signature)
                push!(clusters, cluster)
                cluster_count += 1
                
                # Progress reporting every 2 seconds
                current_time = time()
                if current_time - last_report_time >= 2.0
                    new_clusters = cluster_count - last_cluster_count
                    verbose && println("    Found $cluster_count clusters (+$new_clusters in last 2s)")
                    last_report_time = current_time
                    last_cluster_count = cluster_count
                end
            end
        end
        
        # Stop if we've reached max weight
        if current_weight >= max_weight
            return
        end
        
        # Find candidate loops to add (adjacent loops or multiplicities)
        candidate_loops = Set{Int}()
        
        if isempty(current_cluster)
            # Start with supported loops only
            for loop_id in supported_loop_ids
                if all_loops[loop_id].weight <= max_weight - current_weight
                    push!(candidate_loops, loop_id)
                end
            end
        else
            # Add loops connected to current cluster via interaction graph
            for loop_id in current_cluster
                # Add connected loops (touching loops)
                for neighbor_id in get(interaction_graph, loop_id, Int[])
                    if all_loops[neighbor_id].weight <= max_weight - current_weight
                        push!(candidate_loops, neighbor_id)
                    end
                end
                # Allow multiplicity increases (same loop added again)
                if all_loops[loop_id].weight <= max_weight - current_weight
                    push!(candidate_loops, loop_id)
                end
            end
        end
        
        # Try each candidate loop
        for loop_id in candidate_loops
            loop_weight = all_loops[loop_id].weight
            new_weight = current_weight + loop_weight
            
            if new_weight <= max_weight
                new_cluster = copy(current_cluster)
                push!(new_cluster, loop_id)
                new_has_supported = has_supported || (loop_id in supported_loop_ids)
                
                # Continue DFS (connectivity guaranteed by interaction graph)
                dfs_grow_cluster(new_cluster, new_weight, new_has_supported)
            end
        end
    end
    
    # Start DFS with empty cluster
    dfs_grow_cluster(Int[], 0, false)
    
    verbose && println("  DFS enumeration completed: $cluster_count total clusters found")
    return clusters
end

"""
Build all clusters on named graph ng, up to a given weight. Optionally, must be supported on the vertices must_contain, in which case those vertices can be leaves

This is overkill as it finds ALL subgraphs first, but my other implementation had bugs
"""
function enumerate_clusters(ng::NamedGraph, max_weight::Int; min_v::Int = 4, triangle_free::Bool = true, must_contain = [], min_deg::Int = 2, verbose::Bool = false)
    g = ng.position_graph
    ordered_indices = ng.vertices.ordered_indices

    verbose && println("Step 1: find embedded generalized loops")
    subgraphs = generate_embedded_graphs(g, max_weight; min_v = min_v, triangle_free = triangle_free, min_deg = min_deg, leaf_vertices = [ng.vertices.index_positions[v] for v=must_contain])

    # convert into form of LoopEnumeration.jl
    loops = [Loop(sort(unique(vcat([[e[1],e[2]] for e=subg]...))), subg, length(subg)) for subg=subgraphs]

    verbose && println("Found $(length(loops)) loops")

    verbose && println("Step 2: Building interaction graph...")
    interaction_graph = build_interaction_graph(loops)

    # DFS cluster enumeration
    verbose && println("Step 3: DFS cluster enumeration...")
    if isempty(must_contain)
        supported_loops = [1:length(loops);]
    else
        supported_loops = findall(el->all(l->ng.vertices.index_positions[l] in el.vertices, must_contain), loops)
	verbose && println("$(length(supported_loops)) supported...")
    end
    
    all_clusters = dfs_enumerate_clusters_from_supported(loops, supported_loops, max_weight, interaction_graph, verbose = verbose)
    verbose && println("Found $(length(all_clusters)) connected clusters")
    
    # converting loops into NamedGraphs, for use in tensor_weights
    return all_clusters, [generalized_loop_named(l, ordered_indices) for l=loops], interaction_graph

end

"""
Convert from Loop into NamedGraph
"""
function generalized_loop_named(loop::Loop, ordered_indices)
    g = NamedGraph(ordered_indices[loop.vertices])
    for e=loop.edges
        add_edge!(g, ordered_indices[e[1]], ordered_indices[e[2]])
    end
    g
end

function ursell_function(cluster::Cluster, adj::Dict)
    """
    Compute the Ursell function φ(W) for a connected cluster W.
    """
    total_loops = cluster.total_loops

    if length(cluster.loop_ids) > 2
        for i=1:length(cluster.loop_ids)
	    for j=1:i-1
	        if !(cluster.loop_ids[i] in adj[cluster.loop_ids[j]])
		    error("Only implemented clusters corresponding to complete graphs for now, but got $(cluster)")
		end
	    end
	end
    end

    no_vertices = sum(values(cluster.multiplicities))
    denominator = prod(factorial.(values(cluster.multiplicities)))
    numerator = (-1)^(no_vertices - 1)* factorial(no_vertices - 1)
    return numerator / denominator
end