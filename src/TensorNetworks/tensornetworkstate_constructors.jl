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
    links = Dictionary(edges(g), [Index(1, "e$(src(e))_$(dst(e))") for e in edges(g)])
    links = merge(links, Dictionary(reverse.(edges(g)), [links[e] for e in edges(g)]))

    ts = Dictionary{vertextype(g), ITensor}()
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
    g = named_grid((n,n); periodic = true)
    vs = collect(vertices(g))
    tensors = Dictionary{vertextype(g), ITensor}()
    es=  edges(g)
    e_dict = Dictionary(es, [Index(2) for e in edges(g)])
    e_dict = merge(e_dict, Dictionary(reverse.(es), collect(values(e_dict))))

    for v in vertices(g)
        incoming_es = filter(e -> v == src(e) || v == dst(e), es)
        incoming_inds = [e_dict[e] for e in incoming_es]
        sv = only(s[v])

        state = ITensor(ComplexF64, 0.0, [incoming_inds... , sv])

        north_index = e_dict[NamedEdge((mod1(v[1]+1, n), v[2]) => v)]
        east_index = e_dict[NamedEdge((v[1], mod1(v[2]+1, n)) => v)]
        south_index = e_dict[NamedEdge(v => (mod1(v[1]-1, n), v[2]))]
        west_index = e_dict[NamedEdge(v => (v[1], mod1(v[2]-1, n)))]

        if iseven(sum(v))
            state  = state + (onehot(north_index => 1) * onehot(east_index => 1) + onehot(north_index => 2) * onehot(east_index => 2)) * (onehot(south_index => 1) * onehot(west_index => 1) + onehot(south_index => 2) * onehot(west_index => 2)) * onehot(sv => 1)
            state  = state + (onehot(north_index => 1) * onehot(east_index => 1) - onehot(north_index => 2) * onehot(east_index => 2)) * (onehot(south_index => 1) * onehot(west_index => 1) - onehot(south_index => 2) * onehot(west_index => 2)) * onehot(sv => 2)
        else
            state  = state + (onehot(north_index => 1) * onehot(west_index => 1) + onehot(north_index => 2) * onehot(west_index => 2)) * (onehot(south_index => 1) * onehot(east_index => 1) + onehot(south_index => 2) * onehot(east_index => 2)) * onehot(sv => 1)
            state  = state + (onehot(north_index => 1) * onehot(west_index => 1) - onehot(north_index => 2) * onehot(west_index => 2)) * (onehot(south_index => 1) * onehot(east_index => 1) - onehot(south_index => 2) * onehot(east_index => 2)) * onehot(sv => 2)
        end
        set!(tensors, v, state)
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
  ferromagnetic couplings (`J_e = 1.0` on every edge). Negative entries are
  promoted to complex internally so the symmetric square-root factorisation of
  the Boltzmann weight remains valid.

Returns a `TensorNetwork` (not a `TensorNetworkState`); contract it to obtain
``Z(β)``.
"""
function ising_partitionfunction(g::NamedGraph, β::Real; Js::Dictionary = Dictionary(edges(g), [1.0 for e in edges(g)]))
    links = Dictionary(edges(g), [Index(2, "e$(src(e))_$(dst(e))") for e in edges(g)])
    links = merge(links, Dictionary(reverse.(edges(g)), [links[e] for e in edges(g)]))

    # symmetric sqrt of Boltzmann matrix W = exp(β σσ')
    sqrt_Ws = Dictionary()
    for e in edges(g)
        arg = β*Js[e]
        arg = arg < 0 ? Complex(arg) : arg
        W = [exp(arg)  exp(-arg);
              exp(-arg) exp(arg)]
	    λ1, λ2 = cosh(arg), sinh(arg)
        α = 0.5 * (sqrt(λ1) + sqrt(λ2))
        ϕ = 0.5 * (sqrt(λ1) - sqrt(λ2))
        sqrt_W = sqrt(2)*[α ϕ; ϕ α]
        set!(sqrt_Ws, e, sqrt_W)
        set!(sqrt_Ws, reverse(e), sqrt_W)
        sqrt_W * sqrt_W ≈ W ? nothing : throw(AssertionError("$(sqrt_W * sqrt_W), $(W)"))
    end
    
    ts = Dictionary{vertextype(g), ITensor}()
    for v in vertices(g)
        es = incident_edges(g, v; dir = :in)
        t = delta([links[e] for e in es])
        for e in es
            t = noprime(ITensor(ComplexF64, sqrt_Ws[e], links[e], prime(links[e]))*t)
        end
        set!(ts, v, t)
    end
    return TensorNetwork(ts, g)
end
