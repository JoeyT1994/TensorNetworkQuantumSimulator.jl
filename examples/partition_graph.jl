using Dictionaries
using Graphs
using NamedGraphs

"""
    chain_decomposition(g)

Split `g` into its branching vertices (degree >= 3) and the chains of degree-2
vertices that connect them.

Returns `(centers, chains)`, where each chain is a tuple `(a, b, interior)`:
`interior` is the ordered list of degree-2 vertices walked from center `a` to
center `b`. A chain that dead-ends on a degree-1 vertex has `b === nothing`, with
the dangling vertex included in `interior`.
"""
function chain_decomposition(g)
    centers = [v for v in vertices(g) if degree(g, v) >= 3]
    isempty(centers) && error("graph contains no vertices of degree >= 3")
    chains = Tuple{Any, Any, Vector{Any}}[]
    walked = Set{Tuple{Any, Any}}()  # half-edges already consumed
    for c in centers, n in neighbors(g, c)
        (c, n) in walked && continue
        push!(walked, (c, n))
        interior = Any[]
        prev, cur = c, n
        while degree(g, cur) == 2
            push!(interior, cur)
            prev, cur = cur, only(filter(!=(prev), collect(neighbors(g, cur))))
        end
        if degree(g, cur) == 1
            push!(interior, cur)
            push!(chains, (c, nothing, interior))
        else
            push!(walked, (cur, prev))
            push!(chains, (c, cur, interior))
        end
    end
    covered = length(centers) + sum(length(ch[3]) for ch in chains; init = 0)
    covered == nv(g) && return centers, chains
    @warn "$(nv(g) - covered) vertices lie on degree-2 cycles with no branching vertex and were dropped"
    return centers, chains
end

# Number of interior vertices of chain `i` handed to its `a` end. The chain is cut
# at `interior[s]`, which is shared: `a` takes `interior[1:s]`, `b` takes
# `interior[s:end]`. `s == 0` marks a chain with no interior vertices at all.
_init_split((a, b, interior)) = isnothing(b) ? length(interior) : cld(length(interior), 2)

function _region_sizes(centers, chains, splits, idx)
    sizes = ones(Int, length(centers))  # each region owns its own center
    for (i, (a, b, interior)) in enumerate(chains)
        k = length(interior)
        if isnothing(b)
            sizes[idx[a]] += k
        elseif k > 0
            sizes[idx[a]] += splits[i]
            sizes[idx[b]] += k - splits[i] + 1
        end
    end
    return sizes
end

# Residual arcs: `(target, chain, delta)` means shifting one interior vertex off
# `src`'s side of `chain` and onto `target`'s side, by adding `delta` to the split.
function _residual_arcs(centers, chains, splits, idx)
    arcs = [Tuple{Int, Int, Int}[] for _ in centers]
    for (i, (a, b, interior)) in enumerate(chains)
        k = length(interior)
        (isnothing(b) || k == 0) && continue
        splits[i] >= 2 && push!(arcs[idx[a]], (idx[b], i, -1))
        splits[i] <= k - 1 && push!(arcs[idx[b]], (idx[a], i, +1))
    end
    return arcs
end

# Shortest chain of shifts moving one vertex out of region `src` and into some
# region at least 2 vertices lighter. Interior regions of the path net out to zero.
function _augmenting_path(src, sizes, arcs)
    prev = Dict{Int, Tuple{Int, Int, Int}}()
    queue, seen = [src], Set(src)
    while !isempty(queue)
        u = popfirst!(queue)
        u != src && sizes[u] <= sizes[src] - 2 && return _unwind(prev, src, u)
        for (v, chain, delta) in arcs[u]
            v in seen && continue
            push!(seen, v)
            prev[v] = (u, chain, delta)
            push!(queue, v)
        end
    end
    return nothing
end

function _unwind(prev, src, dst)
    shifts = Tuple{Int, Int}[]
    cur = dst
    while cur != src
        u, chain, delta = prev[cur]
        push!(shifts, (chain, delta))
        cur = u
    end
    return shifts
end

"""
    partition_heavy_hex(g, region_keys = nothing)

Partition the vertices of `g` into one region per degree-3 (branching) vertex.

Each region is a contiguous set holding exactly one branching vertex plus the
degree-2 vertices around it; the degree-2 vertex where two regions meet belongs
to *both* of them. Region sizes are balanced as evenly as the lattice allows.

Regions are keyed by their branching vertex by default. Pass `region_keys` to name
them yourself: one key per branching vertex, paired up in ascending order of
branching vertex, and used as the region order of the returned `Dictionary`.

Returns `(regions, shared)`:

  - `regions`: a `Dictionary` mapping each region key to the ordered vertices of
    that region.
  - `shared`: a `Dict` mapping each boundary vertex to the `(a, b)` pair of region
    keys whose regions both contain it. A degree-2 vertex lies on exactly one
    chain, so a boundary vertex is always shared by exactly two regions.
"""
function partition_heavy_hex(g::AbstractGraph{V}, region_keys = nothing) where {V}
    centers, chains = chain_decomposition(g)
    idx = Dict(c => i for (i, c) in enumerate(centers))
    splits = [_init_split(ch) for ch in chains]
    key = _region_key_map(centers, region_keys)

    # Balance by pushing single vertices along paths of the center graph, always
    # from a heaviest region toward a reachable region at least 2 lighter. Every
    # shift strictly lowers the sum of squared sizes, so this terminates; it halts
    # exactly at the optimality condition for that objective.
    while true
        sizes = _region_sizes(centers, chains, splits, idx)
        arcs = _residual_arcs(centers, chains, splits, idx)
        shifts = nothing
        for src in sortperm(sizes; rev = true)
            shifts = _augmenting_path(src, sizes, arcs)
            !isnothing(shifts) && break
        end
        isnothing(shifts) && break
        for (chain, delta) in shifts
            splits[chain] += delta
        end
    end

    ordered = isnothing(region_keys) ? centers : sort(centers)
    regions = Dictionary([key[c] for c in ordered], [V[c] for c in ordered])

    ktype = eltype(region_keys)

    shared = Dictionary{V, Tuple{ktype, ktype}}()
    for (i, (a, b, interior)) in enumerate(chains)
        k = length(interior)
        k == 0 && continue
        if isnothing(b)
            append!(regions[key[a]], interior)
        else
            append!(regions[key[a]], interior[1:splits[i]])
            append!(regions[key[b]], interior[splits[i]:k])

            # cut vertex, in both regions
            insert!(shared, interior[splits[i]], (key[a], key[b]))
        end
    end
    return regions, shared
end

# Pair user-supplied region keys with the branching vertices in ascending vertex
# order, so the mapping is reproducible rather than dependent on graph insertion order.
function _region_key_map(centers, region_keys)
    isnothing(region_keys) && return Dict(c => c for c in centers)
    ks = collect(region_keys)
    length(ks) == length(centers) ||
        error("got $(length(ks)) region keys for $(length(centers)) degree-3 vertices")
    allunique(ks) || error("region keys must be unique")
    return Dict(zip(sort(centers), ks))
end

"""
    boundary_vertices(regions)

The vertices shared by more than one region.
"""
function boundary_vertices(regions)
    counts = Dict{Any, Int}()
    for region in regions, v in region
        counts[v] = get(counts, v, 0) + 1
    end
    return Set(v for (v, c) in counts if c > 1)
end

"""
    check_partition(g, regions)

Verify the region invariants: full coverage, one branching vertex per region,
contiguity, and that every shared vertex is shared by exactly two regions.
"""
function check_partition(g, regions)
    centers = Set(v for v in vertices(g) if degree(g, v) >= 3)
    length(regions) == length(centers) || error("expected one region per degree-3 vertex")
    for (c, region) in pairs(regions)
        length(intersect(centers, region)) == 1 ||
            error("region $c does not hold exactly one degree-3 vertex")
        allunique(region) || error("region $c repeats a vertex")
        _is_contiguous(g, region) || error("region $c is not contiguous")
    end
    union(Set.(regions)...) == Set(vertices(g)) || error("regions do not cover g")
    counts = Dict{Any, Int}()
    for region in regions, v in region
        counts[v] = get(counts, v, 0) + 1
    end
    all(<=(2), values(counts)) || error("some vertex belongs to more than two regions")
    for e in edges(g)
        any(region -> src(e) in region && dst(e) in region, regions) ||
            error("edge $e is cut without a shared boundary vertex")
    end
    return true
end

function _is_contiguous(g, region)
    isempty(region) && return true
    inside = Set(region)
    seen, queue = Set([first(region)]), [first(region)]
    while !isempty(queue)
        for n in neighbors(g, popfirst!(queue))
            (n in inside && !(n in seen)) || continue
            push!(seen, n)
            push!(queue, n)
        end
    end
    return length(seen) == length(inside)
end

"""
    print_partition(g, regions)

Summarise the partition: region sizes, interiors, and shared boundary vertices.
"""
function print_partition(g, regions)
    shared = boundary_vertices(regions)
    sizes = [length(r) for r in regions]
    println(
        "$(length(regions)) regions over $(nv(g)) vertices, sizes $(minimum(sizes))-$(maximum(sizes))"
    )
    for k in sort(collect(keys(regions)))
        center = first(regions[k])  # each region is built centre-first
        region = sort(collect(regions[k]))
        marked = [v in shared ? "$(v)*" : "$(v)" for v in region]
        label = k == center ? "region $k" : "region $k (centre $center)"
        println("  $label ($(length(region))): ", join(marked, " "))
    end
    println("shared (*) boundary vertices: ", join(sort(collect(shared)), " "))
    return nothing
end