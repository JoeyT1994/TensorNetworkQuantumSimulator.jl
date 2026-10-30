"""
    zerostate(g::NamedGraph)

Tensor network for vacuum state on given graph, i.e all spins up
"""
function zerostate(eltype, g::NamedGraph, s::Dictionary = siteinds("S=1/2", g))
    return tensornetworkstate(eltype, v -> "↑", g, s)
end

zerostate(g::NamedGraph, s::Dictionary = siteinds("S=1/2", g)) = zerostate(Float64, g, s)

"""
    identity_tensornetworkstate(eltype, g::NamedGraph, s::Dictionary = siteinds("S=1/2", g; inds_per_site = 2))

Construct a bond-dimension-1 `TensorNetworkState` representing the identity matrix on graph `g`.

It expects an even number `n` of physical indices on each vertex, with the
first half being the "ket" indices and the second half the "bra" indices; index `i` is paired with
index `n/2 + i`.
"""
function identity_tensornetworkstate(eltype, g::NamedGraph, s::Dictionary = siteinds("S=1/2", g; inds_per_site = 2))
    links = edge_indices(g, 1; tags = e -> "e$(src(e))_$(dst(e))")
    ts = Dictionary{vertextype(g), Tensor}()
    for v in vertices(g)
        es = incident_edges(g, v; dir = :in)
        ninds = length(s[v])
        ninds % 2 != 0 && error("Odd number of siteinds on vertex $v - don't know how to partition into rows and column")
        t = delta(eltype, [links[e] for e in es])
        if ninds > 0
            row_inds, col_inds = s[v][1:(ninds÷2)], s[v][((ninds÷2)+1):ninds]
            id = identity_tensor(eltype, row_inds, col_inds)
            t *= id
        end
        set!(ts, v, t)
    end
    return TensorNetworkState(TensorNetwork(ts, g), s)
end

identity_tensornetworkstate(g::NamedGraph, s::Dictionary = siteinds("S=1/2", g; inds_per_site = 2)) = identity_tensornetworkstate(Float64, g, s)

"""
    toriccode_groundstate(n::Int, s::Dictionary = siteinds("S=1/2", named_grid((n, n); periodic = true)))

Construct an exact bond-dimension-2 tensor network state for the ground state of
Kitaev's toric code on an `n × n` torus. The state lives on a periodic `n × n`
square lattice with one `S=1/2` site per vertex.

If passing your own `siteinds`, they must have a single qubit index per vertex
of a periodic `n × n` named grid.

# Arguments
- `n`: linear size of the torus (so the lattice has `n^2` sites).
- `s`: site indices keyed by vertex of the periodic grid. Defaults to a fresh
  `S=1/2` site for each vertex.

Returns a [`TensorNetworkState`](@ref) of bond dimension 2.
"""
function toriccode_groundstate(n::Int, s::Dictionary = siteinds("S=1/2", named_grid((n,n); periodic = true)))
    g = named_grid((n, n); periodic = true)
    l = edge_indices(g, 2)
    # |11⟩ ± |22⟩ on a pair of legs
    bell(a, b, sign) = onehot(ComplexF64, a => 1) * onehot(ComplexF64, b => 1) + sign * onehot(ComplexF64, a => 2) * onehot(ComplexF64, b => 2)
    tensors = Dictionary{vertextype(g), Tensor}()
    for v in vertices(g)
        north = l[NamedEdge((mod1(v[1] + 1, n), v[2]) => v)]
        east = l[NamedEdge((v[1], mod1(v[2] + 1, n)) => v)]
        south = l[NamedEdge(v => (mod1(v[1] - 1, n), v[2]))]
        west = l[NamedEdge(v => (v[1], mod1(v[2] - 1, n)))]
        # north pairs with east on even sites and with west on odd ones; south with the other
        a, b = iseven(sum(v)) ? (east, west) : (west, east)
        sv = only(s[v])
        set!(tensors, v, bell(north, a, 1) * bell(south, b, 1) * onehot(sv => 1) + bell(north, a, -1) * bell(south, b, -1) * onehot(sv => 2))
    end
    return TensorNetworkState(TensorNetwork(tensors, g), s)
end

"""
    ising_partitionfunction(g::NamedGraph, β::Real; Js::Dictionary = Dictionary(edges(g), [1.0 for e in edges(g)]))

Construct a bond-dimension-2 tensor network whose full contraction equals the
partition function ``Z(β) = \\sum_{\\{σ\\}} \\exp(β \\sum_{(u, v)} J_{uv} σ_u σ_v)``
of the classical Ising model on graph `g` at inverse temperature `β`.

# Arguments
- `g`: lattice graph. One tensor is placed per vertex.
- `β`: inverse temperature.

# Keyword Arguments
- `Js`: dictionary of edge couplings, keyed by `edges(g)`. Defaults to uniform
  ferromagnetic couplings (`J_e = 1.0` on every edge). The tensors are complex, so
  negative (antiferromagnetic) couplings, whose Boltzmann weight has no real symmetric
  square root, are supported.

Returns a `TensorNetwork` (not a `TensorNetworkState`); contract it to obtain
``Z(β)``.
"""
function ising_partitionfunction(g::NamedGraph, β::Real; Js::Dictionary = Dictionary(edges(g), [1.0 for e in edges(g)]))
    links = edge_indices(g, 2; tags = e -> "e$(src(e))_$(dst(e))")

    # symmetric sqrt of Boltzmann matrix W = exp(β σσ')
    sqrt_Ws = Dictionary{keytype(links), Matrix}()
    for e in edges(g)
        arg = β * Js[e]
        arg = arg < 0 ? Complex(arg) : arg
        λ1, λ2 = cosh(arg), sinh(arg)
        α = 0.5 * (sqrt(λ1) + sqrt(λ2))
        ϕ = 0.5 * (sqrt(λ1) - sqrt(λ2))
        sqrt_W = sqrt(2) * [α ϕ; ϕ α]
        @assert sqrt_W * sqrt_W ≈ [exp(arg) exp(-arg); exp(-arg) exp(arg)]
        set!(sqrt_Ws, e, sqrt_W)
        set!(sqrt_Ws, reverse(e), sqrt_W)
    end

    ts = Dictionary{vertextype(g), Tensor}()
    for v in vertices(g)
        es = incident_edges(g, v; dir = :in)
        t = delta([links[e] for e in es])
        for e in es
            t = noprime(from_array(ComplexF64.(sqrt_Ws[e]), [links[e], prime(links[e])]) * t)
        end
        set!(ts, v, t)
    end
    return TensorNetwork(ts, g)
end
