"""
    heavy_hexagonal_lattice(nx::Int64, ny::Int64)

Create heavy-hexagonal lattice geometry with nx columns of heavy-hexes and ny rows
"""
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

"""
    lieb_lattice(nx::Int64, ny::Int64; periodic = false)

Create Lieb lattice geometry with nx columns of decorated squared and ny rows
"""
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

#Given a circuit of two-site and 1-site gates, build the graph induced by the circuit
#Entries in the circuit should be Tuple(gate_str, Vector{<:Vertices gate acts on}, optional param)
function build_graph_from_gates(circ::Vector{<:Any})
    g = NamedGraph(unique(reduce(vcat, [gate[2] for gate in circ])))
    for gate in circ
        qubits = gate[2]
        if length(qubits) == 2
            v1, v2 = first(qubits), last(qubits)
            !has_edge(g, NamedEdge(v1 => v2)) && add_edge!(g, NamedEdge(v1 => v2))
        end
    end


    !is_connected(g) && error("The circuit graph is not connected, meaning the resulting tensor network will be disconnected which we do not support.
    Considering simulating the connected components separately, as no entanglement will be generated between them.")
    return g
end

const build_graph_from_circuit = build_graph_from_gates
