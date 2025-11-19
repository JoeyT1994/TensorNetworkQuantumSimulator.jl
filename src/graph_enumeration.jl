using Graphs: loadgraphs
using GraphIO
using Graphs.Experimental
using Graphs.SimpleGraphs
using NamedGraphs # from TensorNetworkQuantumSimulator
using StatsBase
using ProgressMeter

# Generate non-isomorphic graphs with geng
# on square lattice, choose triangle_free = true to speed things up
function generate_graphs(n::Int, max_e::Int; min_e::Int=0, triangle_free::Bool = true, min_deg::Int=1, max_deg::Int=4, connected::Bool = true)
    # Build geng command
    if connected
        if triangle_free
	    cmd = `geng -c -t -d$(min_deg) -D$(max_deg) $n $(min_e):$(max_e)`
	else
	    cmd = `geng -c -d$(min_deg) -D$(max_deg) $n $(min_e):$(max_e)`
	end
    elseif triangle_free
        cmd = `geng -t -d$(min_deg) -D$(max_deg) $n $(min_e):$(max_e)`
    else
        cmd = `geng -d$(min_deg) -D$(max_deg) $n $(min_e):$(max_e)`
    end
	
    graphs = SimpleGraph[]
    try open(cmd, "r") do io
        for g in loadgraphs(io, GraphIO.Graph6.Graph6Format())
            push!(graphs, g[2])
        end
    end
    catch e
        println("Couldn't find graphs with these parameters: $(cmd)")
    end

    # also 
    return graphs
end

function map_edges(subg::SimpleGraph, iso_map::Vector)
    iso_map_dict = Dict(v[2]=>v[1] for v=iso_map)
    [(min(iso_map_dict[src(e)],iso_map_dict[dst(e)]),max(iso_map_dict[src(e)],iso_map_dict[dst(e)])) for e=edges(subg)]
end

function embed_graphs(g::AbstractGraph, subg::SimpleGraph)
    all_embeddings = unique(sort.([map_edges(subg, iso_map) for iso_map=all_subgraphisomorph(g, subg)]))
end

function is_valid_graph(g; leaf_vertices = [])
    vertex_counts = countmap(vcat([[e[1],e[2]] for e=g]...))
    for (v,k)=vertex_counts
        if !(v in leaf_vertices) && k < 2
	    return false
	end
    end
    return true
end

"""
Generate graphs up to isomorphism and then embed in the larger graph g.
max_weight is max number of edges
min_v is min number of vertices (e.g. 4 on square lattice)
"""
function generate_embedded_graphs(g::AbstractGraph, max_weight::Int; min_v::Int=4, triangle_free::Bool = true, min_deg::Int = 2, leaf_vertices = [])

    k = maximum([degree(g,v) for v=vertices(g)])
    # max edge weight in max_weight, so the max number of vertices is max_weight-1
    mygraphs = [generate_graphs(no_vertices, max_weight; min_e=no_vertices-1, triangle_free = triangle_free, min_deg = min_deg, max_deg = k, connected = true) for no_vertices=min_v:max_weight+1]

    # only try to embed graphs with at most length(leaf_vertices) leaves
    mygraphs = [filter(subg->count(isone, [degree(subg,v) for v=vertices(subg)])<=length(leaf_vertices), mygs) for mygs=mygraphs]

    # now embed each one
    embeddings = [[embed_graphs(g, subg) for subg = mygs] for mygs=mygraphs]
    subgraphs = filter(g->is_valid_graph(g; leaf_vertices = leaf_vertices), vcat(vcat(embeddings...)...))
end

"""
    prune_branches(g::AbstractGraph, keep_vertices)

Return a new graph obtained from `g` by pruning away leaf branches
(iteratively) except those that terminate at any vertex in `keep_vertices`.

Returns the vertices in the pruned graph
"""
function prune_branches(g::AbstractGraph, keep_vertices)
    keep_set = Set(keep_vertices)
    alive = Dict(v=>true for v=vertices(g))

    changed = true
    while changed
        changed = false
        # compute degree within the induced alive-subgraph (count alive neighbors)
        for v=vertices(g)
            if alive[v]
                cnt = sum([alive[u] for u=neighbors(g,v)])

                # remove if it's a leaf (degree 1) or isolated (degree 0),
                # and not in keep_set.
                if cnt <= 1 && !(v in keep_set)
                    alive[v] = false
                    changed = true
                end
            end
        end
    end

    return [v for v=vertices(g) if alive[v]]
end
