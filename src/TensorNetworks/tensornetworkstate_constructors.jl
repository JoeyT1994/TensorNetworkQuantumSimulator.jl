const stringtostatemap = Dict("I" => [1, 0, 0, 0], "X" => [0, 1, 0, 0], "Y" => [0, 0, 1, 0], "Z" => [0, 0, 0, 1])

"""
    zerostate(g::NamedGraph)

Tensor network for vacuum state on given graph, i.e all spins up
"""
function zerostate(eltype, g::NamedGraph, s::Dictionary = siteinds("S=1/2", g))
    return tensornetworkstate(eltype, v -> "↑", g, s)
end

zerostate(g::NamedGraph, s::Dictionary = siteinds("S=1/2", g)) = zerostate(Float64, g, s)

"""
    paulitensornetworkstate(eltype, f::Function, g::NamedGraph, s::Dictionary = siteinds("Pauli", g))

Construct a tensor network state in the Heisenberg picture (Pauli basis). The function `f` should map each vertex of the graph to a Pauli string (one of `"I"`, `"X"`, `"Y"`, `"Z"`).
"""
function paulitensornetworkstate(eltype, f::Function, g::NamedGraph, s::Dictionary = siteinds("Pauli", g))
    h = v -> stringtostatemap[f(v)]
    return tensornetworkstate(eltype, h, g, s)
end

topaulitensornetwork(f::Function, g::NamedGraph, s::Dictionary = siteinds("Pauli", g)) = topaulitensornetwork(Float64, f, g, s)

"""
    identitytensornetwork(tninds::IndsNetwork)

Tensor network (in Heisenberg picture) for identity matrix on given IndsNetwork
"""
function identitytensornetworkstate(eltype, g::NamedGraph, s::Dictionary = siteinds("Pauli", g))
    return paulitensornetworkstate(eltype, v -> "I", g, s)
end

identitytensornetworkstate(g::NamedGraph, s::Dictionary = siteinds("Pauli", g)) = identitytensornetworkstate(Float64, g, s)

"""
    toriccode_ground_state(n::Int, s::Dictionary = siteinds("S=1/2", named_grid((n,n); periodic = true)))
    
Tensor network of bond dimension 2 corresponding to the ground state of the Toric Code on an nxn Torus. If passing your own siteinds
they should have a single qubit index for each vertex of a periodic n x n grid.
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
            state  = state + (ITensors.onehot(north_index => 1) * ITensors.onehot(east_index => 1) + ITensors.onehot(north_index => 2) * ITensors.onehot(east_index => 2)) * (ITensors.onehot(south_index => 1) * ITensors.onehot(west_index => 1) + ITensors.onehot(south_index => 2) * ITensors.onehot(west_index => 2)) * ITensors.onehot(sv => 1)
            state  = state + (ITensors.onehot(north_index => 1) * ITensors.onehot(east_index => 1) - ITensors.onehot(north_index => 2) * ITensors.onehot(east_index => 2)) * (ITensors.onehot(south_index => 1) * ITensors.onehot(west_index => 1) - ITensors.onehot(south_index => 2) * ITensors.onehot(west_index => 2)) * ITensors.onehot(sv => 2)
        else
            state  = state + (ITensors.onehot(north_index => 1) * ITensors.onehot(west_index => 1) + ITensors.onehot(north_index => 2) * ITensors.onehot(west_index => 2)) * (ITensors.onehot(south_index => 1) * ITensors.onehot(east_index => 1) + ITensors.onehot(south_index => 2) * ITensors.onehot(east_index => 2)) * ITensors.onehot(sv => 1)
            state  = state + (ITensors.onehot(north_index => 1) * ITensors.onehot(west_index => 1) - ITensors.onehot(north_index => 2) * ITensors.onehot(west_index => 2)) * (ITensors.onehot(south_index => 1) * ITensors.onehot(east_index => 1) - ITensors.onehot(south_index => 2) * ITensors.onehot(east_index => 2)) * ITensors.onehot(sv => 2)
        end
        set!(tensors, v, state)
    end

    return TensorNetworkState(TensorNetwork(tensors, g), s)
end

"""
    ising_partitionfunction(g::NamedGraph, β::Real; Js::Dictionary = Dictionary(edges(g), [1.0 for e in edges(g)]))

Tensor network of bond dimension 2 whose contraction represents the partition function of the classical Ising model at inv. temp beta on the given graph.
Accepts a dictionary of couplings if one wishes to have anisotropic interactions.
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
        t = ITensors.delta([links[e] for e in es])
        for e in es
            t = noprime(ITensor(ComplexF64, sqrt_Ws[e], links[e], prime(links[e]))*t)
        end
        set!(ts, v, t)
    end
    return TensorNetwork(ts, g)
end
