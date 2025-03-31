function SimpleGraphAlgorithms.edge_color(g::AbstractGraph, k::Int64)
    pg, vs = position_graph(g), collect(vertices(g))
    ec_dict = edge_color(UG(pg), k)
    # returns k vectors which contain the colored/commuting edges
    return [[(vs[first(first(e))], vs[last(first(e))]) for e in ec_dict if last(e) == i] for i in 1:k]
end