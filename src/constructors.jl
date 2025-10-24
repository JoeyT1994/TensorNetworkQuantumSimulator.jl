const stringtostatemap = Dict("I" => [1, 0, 0, 0], "X" => [0,1,0,0], "Y" => [0,0,1,0], "Z" => [0,0,0,1])

"""
    zerostate(g::NamedGraph)

Tensor network for vacuum state on given graph, i.e all spins up
"""
function zerostate(eltype, g::NamedGraph, s::Dictionary=siteinds(g, "S=1/2"))
    return tensornetworkstate(eltype, v->"↑", g, s)
end

zerostate(g::NamedGraph, s::Dictionary=siteinds(g, "S=1/2")) = zerostate(Float64, g, s)

"""
    topaulitensornetwork(op, g::NamedGraph)

Tensor network (in Heisenberg picture). Function should map vertices of the graph to pauli strings.
"""
function paulitensornetworkstate(eltype, f::Function, g::NamedGraph, s::Dictionary= siteinds(g, "Pauli"))
    h = v -> stringtostatemap[f(v)]
    return tensornetworkstate(eltype, h, g, s)
end

topaulitensornetwork(f::Function, g::NamedGraph, s::Dictionary=siteinds(g, "Pauli")) = topaulitensornetwork(Float64, f, g, s)

"""
    identitytensornetwork(tninds::IndsNetwork)

Tensor network (in Heisenberg picture) for identity matrix on given IndsNetwork
"""
function identitytensornetworkstate(eltype, g::NamedGraph, s::Dictionary= siteinds(g, "Pauli"))
    return paulitensornetworkstate(eltype, v -> "I", g, s)
end

identitytensornetworkstate(g::NamedGraph, s::Dictionary=siteinds(g, "Pauli")) = identitytensornetworkstate(Float64, g, s)