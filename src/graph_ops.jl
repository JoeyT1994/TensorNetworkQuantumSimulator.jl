function SimpleGraphAlgorithms.edge_color(g::AbstractGraph, k::Int64)
    pg, vs = position_graph(g), collect(vertices(g))
    ec_dict = edge_color(UG(pg), k)
    # returns k vectors which contain the colored/commuting edges
    return [[(vs[first(first(e))], vs[last(first(e))]) for e in ec_dict if last(e) == i] for i in 1:k]
end


"""Create heavy-hex lattice geometry"""
function heavy_hexagonal_lattice_grid(nx::Int64, ny::Int64)
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