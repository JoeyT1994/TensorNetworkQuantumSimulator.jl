# Boundary MPS helpers for checking graph formats
function is_line_graph(g::AbstractGraph)
    vs = collect(vertices(g))
    nvs = length(vs)
    length(vs) == 1 && return true
    !is_tree(g) && return false
    ds = sort([degree(g, v) for v in vs])
    ds != vcat([1,1], [2 for d in 1:(nvs - 2)]) && return false
    return true
end

function is_ring_graph(g::AbstractGraph)
    g_mod = rem_edge(g, first(edges(g)))
    return is_line_graph(g_mod)
end

function pseudo_sqrt_inv_sqrt(M::ITensor; cutoff = 10 * eps(real(scalartype(M))))
    @assert length(inds(M)) == 2
    Q, D, Qdag = ITensorNetworks.ITensorsExtensions.eigendecomp(M, inds(M)[1], inds(M)[2]; ishermitian=true)
    D_sqrt = ITensorNetworks.ITensorsExtensions.map_diag(x -> iszero(x) || abs(x) < cutoff ? 0 : sqrt(x), D)
    D_inv_sqrt = ITensorNetworks.ITensorsExtensions.map_diag(x -> iszero(x) || abs(x) < cutoff ? 0 : inv(sqrt(x)), D)
    M_sqrt = Q * D_sqrt * Qdag
    M_inv_sqrt = Q * D_inv_sqrt * Qdag
    return M_sqrt, M_inv_sqrt
end