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

Create a Lieb lattice geometry with `nx` columns of decorated squares and `ny` rows.
"""
function lieb_lattice(nx::Int64, ny::Int64; periodic = false)
    @assert (!periodic && isodd(nx) && isodd(ny)) || (periodic && iseven(nx) && iseven(ny))
    return without_even_even(named_grid((nx, ny); periodic))
end

# g less every (even, even) vertex
without_even_even(g) = subgraph(g, filter(v -> !all(iseven, v), collect(vertices(g))))

"""
    kagome_lattice(nx::Int64, ny::Int64)

Create a kagome lattice geometry with `nx` columns and `ny` rows of three-site unit cells.
"""
function kagome_lattice(nx::Int64, ny::Int64)
    #a square grid with one diagonal per square (a triangular lattice), less every (even, even) vertex
    g = named_grid((2 * nx + 1, 2 * ny + 1))
    for i in 1:(2 * nx), j in 1:(2 * ny)
        g = add_edge(g, NamedEdge((i, j) => (i + 1, j + 1)))
    end
    return without_even_even(g)
end

"""
    penrose_tiling(ngenerations::Int64)

Create a Penrose rhombus (P3) tiling geometry, deflated `ngenerations` times, filling the decagon inscribed in the unit circle.
Vertex names are their (x, y) positions.
"""
function penrose_tiling(ngenerations::Int64)
    φ = (1 + sqrt(5)) / 2
    # Robinson triangles (kind, A, B, C): kind 0 halves a thin rhombus, kind 1 a thick one
    tris = Tuple{Int, ComplexF64, ComplexF64, ComplexF64}[]
    for i in 0:9
        b, c = cis((2i - 1) * π / 10), cis((2i + 1) * π / 10)
        iseven(i) && ((b, c) = (c, b))
        push!(tris, (0, 0.0im, b, c))
    end
    for _ in 1:ngenerations
        next = similar(tris, 0)
        for (kind, a, b, c) in tris
            if kind == 0
                p = a + (b - a) / φ
                push!(next, (0, c, p, b), (1, p, c, a))
            else
                q = b + (a - b) / φ
                r = b + (c - b) / φ
                push!(next, (1, r, c, a), (1, q, r, b), (0, r, q, a))
            end
        end
        tris = next
    end
    # the same point is reached along different deflation paths, so merge on a fine grid
    scale = 1.0e8
    seen = Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}()
    function name(z)
        k = (round(Int, real(z) * scale), round(Int, imag(z) * scale))
        for di in -1:1, dj in -1:1
            haskey(seen, (k[1] + di, k[2] + dj)) && return seen[(k[1] + di, k[2] + dj)]
        end
        seen[k] = (round(real(z); digits = 8), round(imag(z); digits = 8))
        return seen[k]
    end
    # each triangle contributes its two legs C–A and A–B; the base B–C is the rhombus diagonal
    es = Set{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}}()
    for (_, a, b, c) in tris
        na, nb, nc = name(a), name(b), name(c)
        push!(es, minmax(nc, na), minmax(na, nb))
    end
    g = NamedGraph(unique(Iterators.flatten(es)))
    for (u, v) in es
        g = add_edge(g, NamedEdge(u => v))
    end
    return g
end

#Given a circuit of two-site and 1-site gates, build the graph induced by the circuit
#Entries in the circuit should be Tuple(gate_str, Vector{<:Vertices gate acts on}, optional param)
function build_graph_from_circuit(circ::Vector; qubits = unique(Iterators.flatten(gate[2] for gate in circ)))
    g = NamedGraph(collect(qubits))
    for gate in circ
        qubits = gate[2]
        if length(qubits) == 2
            v1, v2 = first(qubits), last(qubits)
            !has_edge(g, NamedEdge(v1 => v2)) && add_edge!(g, NamedEdge(v1 => v2))
        end
    end
    return g
end
