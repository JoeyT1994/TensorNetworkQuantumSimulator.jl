using Graphs: nv, induced_subgraph, is_connected, connected_components
using NamedGraphs
using NamedGraphs: AbstractNamedGraph, position_graph
using ProgressMeter
using Dictionaries

include("../graph_enumeration.jl")

const RegionKey = NTuple{N,Int} where {N}

"""
    to_key(vs::AbstractVector{Int})::RegionKey
Convert a collection of vertex IDs to a canonical, sorted tuple key.
"""
function to_key(vs::AbstractVector{Int})::RegionKey
    return Tuple(sort!(collect(vs)))
end

"""
    key_intersection(a::RegionKey, b::RegionKey)::Vector{Int}
Return the vertex list of the intersection of two region keys.
"""
function key_intersection(a::RegionKey, b::RegionKey)::Vector{Int}
    sa = Set(a); sb = Set(b)
    return sort!(collect(intersect(sa, sb)))
end

"""
    is_loopful(g::SimpleGraph, key::RegionKey)::Bool
Check if the induced subgraph on `key` is connected and has at least one cycle.
For connected component `H`, loopfulness is `ne(H) - nv(H) + 1 > 0`.
"""
function is_loopful(g::SimpleGraph, key::RegionKey)::Bool
    if length(key) < 3
        return false
    end
    vs = collect(key)
    h, _ = induced_subgraph(g, vs)
    # ensure connected (defensive; we try to keep regions connected elsewhere)
    if nv(h) == 0
        return false
    end
    if !is_connected(h)
        return false
    end
    return ne(h) - nv(h) + 1 > 0
end

"""
    is_proper_subset(a::RegionKey, b::RegionKey)::Bool
Return true if region `a` is a proper subset of `b`.
"""
function is_proper_subset(a::RegionKey, b::RegionKey)::Bool
    if length(a) >= length(b)
        return false
    end
    sb = Set(b)
    @inbounds for x in a
        if x ∉ sb
            return false
        end
    end
    return true
end

"""
    induced_components(g::SimpleGraph, vs::AbstractVector{Int})::Vector{RegionKey}
Return connected components (as RegionKey) of the induced subgraph on `vs`.
"""
function induced_components(g::SimpleGraph, vs::AbstractVector{Int})::Vector{RegionKey}
    if isempty(vs)
        return RegionKey[]
    end
    h, vmap = induced_subgraph(g, vs)  # h has vertices 1..nv(h), vmap maps h->g
    comps = connected_components(h)     # comps as vectors of 1..nv(h)
    # map back to original vertex IDs via `vs`
    return [to_key(vs[c]) for c in comps]
end

# --- Maximal regions under inclusion ------------------------------------------

"""
    maximal_regions(regions::Set{RegionKey})::Set{RegionKey}
Select the inclusion-maximal regions from a set.
"""
function maximal_regions(regions::Set{RegionKey})::Set{RegionKey}
    keys = collect(regions)
    sort!(keys; by=length)  # small to large
    maximal = Set{RegionKey}()
    for i in eachindex(keys)
        a = keys[i]
        is_sub = false
        for j in (i+1):length(keys)
            b = keys[j]
            if is_proper_subset(a, b)
                is_sub = true
                break
            end
        end
        if !is_sub
            push!(maximal, a)
        end
    end
    return maximal
end

# --- Close under intersections (with connected components) ---------------------
# Note that keeping the connected components of graphs with >1 component is unnecessary and is included here as legacy.
# use close_under_intersections_connected instead
"""
    close_under_intersections(g::SimpleGraph, seed::Set{RegionKey}; loop_only::Bool=true)::Set{RegionKey}
Given `seed` regions, iteratively add intersections (split into connected components),
optionally keeping only loopful components; stop when no new regions appear. Optionally, only keep if component contains `must_contain` vertices.
"""
function close_under_intersections(g::SimpleGraph, seed::Set{RegionKey}; loop_only::Bool=true, must_contain = [])
    R = Set(seed)
    changed = true
    while changed
        changed = false
        keys = collect(R)
        for i in 1:length(keys)-1
            a = keys[i]
            for j in (i+1):length(keys)
                b = keys[j]
                X = key_intersection(a, b)
                comps = induced_components(g, X)

                for comp in comps
                    if loop_only && !is_loopful(g, comp)
                        continue
                    end
		    if !isempty(must_contain) && intersect(must_contain, comp) != must_contain
		        continue
		    end
                    if comp ∉ R
                        push!(R, comp)
                        changed = true
                    end
                end
            end
        end
    end
    return R
end

# --- Close under intersections (with connected components) ---------------------

"""
    close_under_intersections(g::SimpleGraph, seed::Set{RegionKey}; loop_only::Bool=true)::Set{RegionKey}
Given `seed` regions, iteratively add intersections. Only keep connected components.
optionally keeping only loopful components; stop when no new regions appear. Optionally, only keep if component contains `must_contain` vertices.
"""
function close_under_intersections_connected(g::SimpleGraph, seed::Set{RegionKey}; loop_only::Bool=true, must_contain = [])
    R = Set(seed)
    changed = true
    while changed
        changed = false
        keys = collect(R)
        for i in 1:length(keys)-1
            a = keys[i]
            for j in (i+1):length(keys)
                b = keys[j]
                X = key_intersection(a, b)

		# must be connected and nonempty
                if isempty(X) || !is_connected(induced_subgraph(g, X)[1])
		    continue
		end

		# must contain the required vertices
		if !isempty(must_contain) && intersect(must_contain, X) != must_contain
                    continue
                end

		comp = to_key(X)
		# must be loopy, if loop_only
                if loop_only && !is_loopful(g, comp)
                    continue
                end

		if comp ∉ R
                    push!(R, comp)
                    changed = true
                end
            end
        end
    end
    return R
end

# --- Counting numbers (top-down Möbius) ---------------------------------------

"""
    counting_numbers(regions::Set{RegionKey})::Dict{RegionKey,Int}
Compute inclusion–exclusion counting numbers c(r): set c=1 for maximals,
then for other regions c(r) = 1 - sum_{a ⊃ r} c(a).
"""
function counting_numbers(regions::Set{RegionKey},maximals::Set{RegionKey})::Dict{RegionKey,Int}
    R = collect(regions)
    # sort supersets first (decreasing size)
    sort!(R; by=r -> (-length(r), r))
    c = Dict{RegionKey,Int}()

    for r=maximals
        c[r] = 1
    end
    
    # fill others (now supersets are guaranteed to have c set already)
    for r in R
        if haskey(c, r)
            continue
        end
        s = 0
        for a in R
            if is_proper_subset(r, a)
                s += get(c, a, 0)
            end
        end
        c[r] = 1 - s
    end
    return c
end


"""
Find all subgraphs of g that contain both u and v, up to C vertices,
and have no other leaves
"""
function vertex_walks_up_to_C_regions(g::AbstractGraph, u::Integer, v::Integer, C::Int; buffer::Int = C)
    walks = Set{RegionKey}()

    stack = [(u, Set(), -1,false,0)]   # (current vertex, path, previous vertex, has_both,num_steps)

    while !isempty(stack)
        node, path, prev_node, has_both,num_steps = pop!(stack)
        if node==v
            has_both = true

        end

        # contains both u and v, and completed a cycle, or a path from u to v
        if has_both && (node==v || node in path)
            push!(walks, to_key([vv for vv=union(Set([node]), path)]))
        end

        push!(path, node)
        @assert length(path) <= C
        if num_steps==C + buffer
            continue
        end
        for w in neighbors(g, node)
            if w==prev_node
                # don't backtrack
                continue
            end

            if length(path) < C || (w in path)
                push!(stack, (w, copy(path), node, has_both, num_steps + 1))
            end
        end

    end
    return walks
end

"""
Build clusters on graph g out of the regions regs
"""
function build_clusters(g::SimpleGraph, regs::Set; loop_only::Bool=true, must_contain = [], smart::Bool=true, verbose::Bool=false)
    verbose && println("Finding maximal"); flush(stdout)
    @time Rmax = maximal_regions(regs)
    verbose && println("Finding intersections of $(length(Rmax)) regions"); flush(stdout)
    if smart
        R = close_under_intersections_connected(g, Rmax;loop_only=loop_only, must_contain = unique(must_contain))
    else
        R = close_under_intersections(g, Rmax;loop_only=loop_only, must_contain = unique(must_contain))
    end
    verbose && println("Finding counting numbers"); flush(stdout)
    @time c = counting_numbers(R, Rmax)
    return R, Rmax, c
end

"""
Maps regions to NamedGraphs
"""
function map_regions_named(R::Set, Rmax::Set, c::Dict, vs_dict::Dictionary)
    # map back to names
    R = [map(v -> vs_dict[v], set) for set in R]
    Rmax = [map(v -> vs_dict[v], set) for set in Rmax]
    c = Dict(map(v -> vs_dict[v], key) => val for (key, val) in c)
    return R, Rmax, c
end

# prune branches except those ending at keep_vertices
function prune_cc(g::AbstractGraph, regions::Vector, counting_nums::Dict; keep_vertices = [])
    counting_graphs = Dict()
    for r=regions
        if counting_nums[r] != 0
	    eg = induced_subgraph(g, r)[1]
	    pb = Tuple(sort(prune_branches(eg, keep_vertices)))
	    # pg = induced_subgraph(eg, prune_branches(eg, keep_vertices))[1]
	    if haskey(counting_graphs, pb)
	        counting_graphs[pb] += counting_nums[r]
	    else
	        counting_graphs[pb] = counting_nums[r]
	    end
	end
    end
    counting_graphs
end

"""
    build_region_family_correlation(g::SimpleGraph, u::Int, v::Int, C::Int)
Return (R, Rmax, c) where:
  R    :: Set{RegionKey}  — full region family closed under intersections
  Rmax :: Set{RegionKey}  — maximal regions used as seeds
  c    :: Dict{RegionKey,Int} — counting numbers for all r ∈ R
"""
function build_region_family_correlation(g::SimpleGraph, u::Int, v::Int, C::Int; buffer::Int=C, smart::Bool=true, verbose::Bool=false)
    verbose && println("Finding graphs"); flush(stdout)
    @time regs = vertex_walks_up_to_C_regions(g,u,v,C; buffer = buffer)
    R,Rmax,c = build_clusters(g, regs; loop_only = false, must_contain = [u,v], smart=smart, verbose = verbose)
end

"""
    Function to enumerate all generalized regions on a NamedGraph up to size Cluster_size, containing u and v. All other vertices have degree geq 2.
    Counting numbers are found via top-down Möbius inversion.
    For one-point function, just set u=v.
    Returns (R, Rmax, c) where:
      R    :: Vector{Vector{T}}  — full region family closed under intersections
      Rmax :: Vector{Vector{T}}  — maximal regions (largest generalizd loops) used as seeds and not subsets of any other regions
      c    :: Dict{Vector{T},Int} — counting numbers for all r ∈ R
"""
function build_region_family_correlation(ng::NamedGraph, u, v, Cluster_size::Int; buffer::Int = Cluster_size, smart::Bool=true, prune::Bool=true,verbose::Bool=false)
    g, vs_dict = position_graph(ng), Dictionary([i for i in 1:nv(ng)], collect(vertices(ng)))
    mapped_u, mapped_v = ng.vertices.index_positions[u], ng.vertices.index_positions[v]
    R, Rmax, c = build_region_family_correlation(g, mapped_u, mapped_v, Cluster_size; buffer = buffer, smart=smart,verbose=verbose)
    R, Rmax, c = map_regions_named(R, Rmax, c, vs_dict)
    if prune
        c = prune_cc(ng,R,c; keep_vertices=[u,v])
	return collect(keys(c)), Rmax, c
    else
        return R, Rmax, c
    end
end

"""
Generate graphs up to isomorphism and then embed in the larger graph g.
max_v is max number of vertices
min_v is min number of vertices (e.g. 4 on square lattice)
"""
function generate_embedded_leafless_graphs(g::AbstractGraph, max_v::Int; min_v::Int=4, triangle_free::Bool = true, min_deg::Int = 2)
    @assert min_deg >= 2
    k = maximum([degree(g,v) for v=vertices(g)])
    mygraphs = [generate_graphs(no_vertices, k*no_vertices; triangle_free = triangle_free, min_deg = min_deg, max_deg = k, connected = true) 
        for no_vertices=min_v:max_v]

    # now embed each one
    embeddings = [[embed_graphs(g, subg) for subg = mygs] for mygs=mygraphs]
    subgraphs = vcat(vcat(embeddings...)...)

    # make into loopful induced subgraphs only
    # Since they're connected and min_deg >= 2, will by definition be loopful
    
    regions = unique([to_key(unique(vcat([[e[1], e[2]] for e=subg]...))) for subg = subgraphs])
    return Set{RegionKey}(regions)
end

"""
    build_region_family(g::SimpleGraph, C::Int)
Return (R, Rmax, c) where:
  R    :: Set{RegionKey}  — full region family closed under intersections
  Rmax :: Set{RegionKey}  — maximal regions used as seeds
  c    :: Dict{RegionKey,Int} — counting numbers for all r ∈ R
"""
function build_region_family(g::SimpleGraph, C::Int; min_deg::Int=2, min_v::Int=4,triangle_free::Bool=true, smart::Bool=true, verbose::Bool=false)
    verbose && println("Finding graphs")
    @time regs = generate_embedded_leafless_graphs(g, C; min_deg = min_deg, min_v=min_v,triangle_free=triangle_free)
    build_clusters(g, regs; loop_only = true,smart=smart, verbose=verbose)
end

"""
    Function to enumerate all generalized regions on a NamedGraph up to size Cluster_size.
    Counting numbers are found via top-down Möbius inversion.
    Returns (R, Rmax, c) where:
      R    :: Vector{Vector{T}}  — full region family closed under intersections
      Rmax :: Vector{Vector{T}}  — maximal regions (largest generalizd loops) used as seeds and not subsets of any other regions
      c    :: Dict{Vector{T},Int} — counting numbers for all r ∈ R
"""
function build_region_family(ng::NamedGraph, Cluster_size::Int; min_deg::Int=2, min_v::Int=4,triangle_free::Bool=true, smart::Bool=true, verbose::Bool=false)
    g, vs_dict = position_graph(ng), Dictionary([i for i in 1:nv(ng)], collect(vertices(ng)))
    R, Rmax, c = build_region_family(g, Cluster_size; min_deg = min_deg, min_v=min_v,triangle_free=triangle_free, smart=smart, verbose=verbose)
    map_regions_named(R, Rmax, c, vs_dict)
end