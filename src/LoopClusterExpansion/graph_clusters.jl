# Edge-induced no-leaf subgraph enumeration for the loop / linked-cluster expansion.
#
# This file is PURE GRAPH CODE: it never touches a tensor and operates only on a
# NamedGraph. A "generalized loop" is a connected edge set in which EVERY vertex has
# induced degree >= 2 (no leaves). These are exactly the polymers of the hard-core loop
# gas whose log is the free-energy correction `F − ln Z_BP`, and the configurations whose
# antiprojector weights sum to `Z / Z_BP − 1` (see `weight`/`weights` and
# `loopcorrected_free_energy` in `MessagePassing/loopcorrection.jl`).
#
# NamedGraphs' `edgeinduced_subgraphs_no_leaves` builds candidates as unions of simple
# cycles, so it is connected-only AND silently drops any no-leaf subgraph that needs a
# non-cycle (bridge) edge to be no-leaf — e.g. a "dumbbell" of two triangles joined by a
# single bridge edge (the bridge lies on no cycle, so it is never added; the two bare
# triangles fail the connectivity filter). Those diagrams have all degrees >= 2 and a
# nonzero antiprojector weight, so omitting them under-counts the loop series. The
# enumerator here grows connected edge sets directly, so it is connected *by construction*
# and catches the bridge diagrams.
#
# Enumeration uses ESU (Wernicke, "A faster algorithm for detecting network motifs",
# WABI 2006) on the line graph: connected edge sets of `g` are connected vertex sets of the
# line graph, and ESU visits each connected vertex set EXACTLY ONCE via the "smallest-index
# vertex is the root, only grow with strictly-larger exclusive neighbours" rule. That makes
# the de-duplication structural — no `seen` hash set, and no subgraph is ever generated more
# than once — and the per-node bookkeeping (induced degrees, leaf count) is maintained
# incrementally on integer-indexed arrays instead of rebuilt from a dictionary each step.

# Apply edge `i` to the running induced-degree state, updating the leaf counter
# (`nleaf` = number of vertices whose induced degree is exactly 1).
@inline function _add_edge!(i, esrc, edst, vdeg, nleaf)
    @inbounds for v in (esrc[i], edst[i])
        d = vdeg[v]
        vdeg[v] = d + 1
        d == 0 ? (nleaf[] += 1) : d == 1 ? (nleaf[] -= 1) : nothing
    end
    return nothing
end

# Inverse of `_add_edge!`: remove edge `i` from the running state.
@inline function _del_edge!(i, esrc, edst, vdeg, nleaf)
    @inbounds for v in (esrc[i], edst[i])
        d = vdeg[v]
        vdeg[v] = d - 1
        d == 1 ? (nleaf[] -= 1) : d == 2 ? (nleaf[] += 1) : nothing
    end
    return nothing
end

# ESU recursion over the line graph. `Vsub` is the current edge-index set (a stack), `ext`
# its extension list (candidate edges, all with index > `root`), and `marked[u]` records
# whether edge `u` is already in `Vsub` or has entered the extension somewhere on the path
# from the root (so it is never re-queued — this is what enforces exact-once visiting).
function _esu_extend!(
        out, Vsub, ext, root, eadj, esrc, edst, vdeg, nleaf, marked, max_edges,
    )
    # `nleaf == 0` with a non-empty edge set <=> every touched vertex has degree >= 2.
    (nleaf[] == 0 && !isempty(Vsub)) && push!(out, copy(Vsub))
    length(Vsub) >= max_edges && return
    # Leaf-prune: each added edge clears at most 2 degree-1 vertices, so a set with more
    # leaves than 2*(remaining budget) can never become leaf-free within budget.
    nleaf[] > 2 * (max_edges - length(Vsub)) && return
    added = Int[]
    while !isempty(ext)
        w = pop!(ext)
        newext = copy(ext)                 # siblings still to be processed (ESU: V_ext \ {w})
        empty!(added)
        @inbounds for u in eadj[w]          # plus the exclusive neighbours of w, index > root
            if u > root && !marked[u]
                marked[u] = true
                push!(newext, u)
                push!(added, u)
            end
        end
        _add_edge!(w, esrc, edst, vdeg, nleaf)
        push!(Vsub, w)
        _esu_extend!(out, Vsub, newext, root, eadj, esrc, edst, vdeg, nleaf, marked, max_edges)
        pop!(Vsub)
        _del_edge!(w, esrc, edst, vdeg, nleaf)
        @inbounds for u in added            # unmark only what THIS w introduced; w stays
            marked[u] = false               # marked for its siblings (parent unmarks it)
        end
    end
    return nothing
end

"""
    connected_edgeinduced_subgraphs_no_leaves(g, max_edges; anchor = nothing) -> Vector

All connected edge-induced subgraphs of `g` with at most `max_edges` edges in which every
vertex has induced degree `>= 2` (no leaves) — the "generalized loops" of the loop/linked-
cluster series. Unlike `edgeinduced_subgraphs_no_leaves` (which unions simple cycles and so
misses bridge-joined no-leaf subgraphs such as two triangles linked by a single edge), this
grows connected edge sets directly and is therefore both connected-by-construction and
bridge-complete.

Enumeration is ESU on the line graph: every connected edge set is visited exactly once (no
deduplication hashing), and induced degrees / the leaf count are tracked incrementally. With
`anchor` set, only subgraphs containing that vertex are produced: the anchor-incident edges
are given the smallest indices and used as the ESU roots, so a subgraph is generated from
its minimum edge iff that edge touches the anchor — i.e. iff the subgraph touches the anchor.
With `anchor = nothing` the whole graph is enumerated.
"""
function connected_edgeinduced_subgraphs_no_leaves(
        g::AbstractGraph, max_edges::Integer; anchor = nothing,
    )
    E0 = collect(edges(g))
    m = length(E0)
    (m == 0 || max_edges < 3) && return typeof(g)[]   # smallest no-leaf subgraph is a 3-cycle

    # Integer-index the vertices for O(1) degree bookkeeping.
    V = collect(vertices(g))
    vidx = Dict{eltype(V), Int}()
    for (k, v) in enumerate(V)
        vidx[v] = k
    end

    # Order the edge list so that, when anchored, the anchor-incident edges occupy the
    # smallest indices `1:nseed`; ESU rooted at `1:nseed` then yields exactly the subgraphs
    # whose minimum edge touches the anchor == the subgraphs that touch the anchor.
    if anchor === nothing
        E = E0
        nseed = m
    else
        ai = vidx[anchor]
        anchored_first = sort(eachindex(E0); by = i -> (vidx[src(E0[i])] != ai && vidx[dst(E0[i])] != ai))
        E = E0[anchored_first]
        nseed = count(e -> vidx[src(e)] == ai || vidx[dst(e)] == ai, E)
    end

    esrc = Vector{Int}(undef, m)
    edst = Vector{Int}(undef, m)
    inc = [Int[] for _ in 1:length(V)]                # incident edge indices per vertex
    for (i, e) in enumerate(E)
        a = vidx[src(e)]
        b = vidx[dst(e)]
        esrc[i] = a
        edst[i] = b
        push!(inc[a], i)
        push!(inc[b], i)
    end
    # Line-graph adjacency: two edges are adjacent iff they share a vertex.
    eadj = [Int[] for _ in 1:m]
    for ev in inc, i in ev, j in ev
        i != j && push!(eadj[i], j)
    end
    for i in 1:m
        unique!(eadj[i])
    end

    vdeg = zeros(Int, length(V))
    nleaf = Ref(0)
    marked = falses(m)
    Vsub = Int[]
    out = Vector{Int}[]
    added = Int[]
    for r in 1:nseed
        _add_edge!(r, esrc, edst, vdeg, nleaf)
        push!(Vsub, r)
        marked[r] = true
        ext = Int[]
        empty!(added)
        @inbounds for u in eadj[r]
            if u > r && !marked[u]
                marked[u] = true
                push!(ext, u)
                push!(added, u)
            end
        end
        _esu_extend!(out, Vsub, ext, r, eadj, esrc, edst, vdeg, nleaf, marked, max_edges)
        pop!(Vsub)
        _del_edge!(r, esrc, edst, vdeg, nleaf)
        marked[r] = false
        @inbounds for u in added
            marked[u] = false
        end
    end
    return [edge_subgraph(g, E[S]) for S in out]
end
