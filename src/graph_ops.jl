function SimpleGraphAlgorithms.edge_color(g::AbstractGraph, k::Int64)
    pg, vs = position_graph(g), collect(vertices(g))
    ec_dict = edge_color(UG(pg), k)
    # returns k vectors which contain the colored/commuting edges
    return [
        [(vs[first(first(e))], vs[last(first(e))]) for e in ec_dict if last(e) == i] for
        i = 1:k
    ]
end

function SimpleGraphAlgorithms.vertex_color(g::AbstractGraph, k::Int64)
    pg, vs = position_graph(g), collect(vertices(g))
    vc_dict = vertex_color(UG(pg), k)
    # returns k vectors which contain the colored/commuting edges
    return [
        [vs[first(v)] for v in vc_dict if last(v) == i] for
        i = 1:k
    ]
end

"""Create heavy-hex lattice geometry"""
function heavy_hexagonal_lattice(nx::Int64, ny::Int64)
    g = named_hexagonal_lattice_graph(nx, ny)
    # create some space for inserting the new vertices
    g = rename_vertices(v -> (2 * first(v) - 1, 2 * last(v) - 1), g)
    for e in edges(g)
        vsrc, vdst = src(e), dst(e)
        v_new = ((first(vsrc) + first(vdst)) / 2, (last(vsrc) + last(vdst)) / 2)
        g = add_vertex(g, v_new)
        g = rem_edge(g, e)
        g = add_edges(g, [NamedEdge(vsrc => v_new), NamedEdge(v_new => vdst)])
    end
    return g
end

function lieb_lattice(nx::Int64, ny::Int64; periodic = false)
    @assert (!periodic && isodd(nx) && isodd(ny)) || (periodic && iseven(nx) && iseven(ny))
    g = named_grid((nx, ny); periodic)
    for v in vertices(g)
        if iseven(first(v)) && iseven(last(v))
            g = rem_vertex(g, v)
        end
        if iseven(first(v)) && iseven(last(v))
            g = rem_vertex(g, v)
        end
    end
    return g

end

function named_biclique(na::Int64, nb::Int64)
    g = NamedGraph([i for i in 1:(na+nb)])
    for i in 1:na
        for j in 1:nb
            g = add_edge(g, NamedEdge(i => (na + j)))
        end
    end
    return g
end


function topologytograph(topology)
    # TODO: adapt this to named graphs with non-integer labels
    # find number of vertices
    nq = maximum(maximum.(topology))
    adjm = zeros(Int, nq, nq)
    for (ii, jj) in topology
        adjm[ii, jj] = adjm[jj, ii] = 1
    end
    return NamedGraph(SimpleGraph(adjm))
end


function graphtotopology(g)
    return [[edge.src, edge.dst] for edge in edges(g)]
end

function NamedGraphs.GraphsExtensions.rem_vertex(bpc::AbstractBeliefPropagationCache, v)
    return rem_vertices(bpc, [v])
end

function NamedGraphs.GraphsExtensions.rem_vertices(bpc::BeliefPropagationCache, vs::Vector)
    pg = partitioned_tensornetwork(bpc)
    pg = rem_vertices(pg, vs)
    return BeliefPropagationCache(pg, messages(bpc))
end

function NamedGraphs.GraphsExtensions.rem_vertices(bmpsc::BoundaryMPSCache, vs::Vector)
    bpc = bp_cache(bmpsc)
    bpc = rem_vertices(bpc, vs)
    return BoundaryMPSCache(bpc, ppg(bmpsc), maximum_virtual_dimension(bmpsc))
end
