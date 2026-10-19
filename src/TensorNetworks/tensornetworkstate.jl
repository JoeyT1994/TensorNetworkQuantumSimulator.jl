"""
    TensorNetworkState

A tensor network state: an `ITensorNetwork` whose dangling indices at each vertex are its
site indices (physical degrees of freedom), as returned by `siteinds`.
"""
const TensorNetworkState = ITensorNetwork

# Bra copy of a tensor: `conj` and prime all legs except `auxinds` — dangling
# non-physical legs (e.g. a charged state's charge leg), which always pair directly
# between ket and bra rather than through an operator or a message.
function bra_tensor(t::ITensor, auxinds::Vector{<:Index})
    tdag = conj(prime(t))
    return isempty(auxinds) ? tdag : rename(tdag, (prime.(auxinds) .=> auxinds)...)
end
bra_tensor(tns::TensorNetworkState, v) = bra_tensor(tns[v], Index[])

# The legs returned by `auxinds_f` stay unprimed on the bra. The loop-correction weights pass
# the bond they relabel to dangle, which `siteinds` would otherwise count as a site index.
function norm_factors(tns::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I", auxinds_f::Function = v -> Index[])
    factors = ITensor[]
    for v in verts
        sinds = siteinds(tns, v)
        tnv = tns[v]
        tnv_dag = bra_tensor(tnv, auxinds_f(v))
        if op_strings(v) == "ρ" || isempty(sinds)
            append!(factors, ITensor[tnv, tnv_dag])
        elseif op_strings(v) == "I"
            tnv_dag = rename(tnv_dag, (prime.(sinds) .=> sinds)...)
            append!(factors, ITensor[tnv, tnv_dag])
        else
            op = adapt_like(tnv, Ops.op(op_strings(v), only(sinds)))
            append!(factors, ITensor[tnv, tnv_dag, op])
        end
    end
    return factors
end

norm_factors(tns::TensorNetworkState, v; kwargs...) = norm_factors(tns, [v]; kwargs...)

"""
    random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary; bond_dimension::Integer = 1)

Generate a random `TensorNetworkState` on graph `g` with local state indices given by `siteinds`.

# Arguments
- `eltype`: The number type of the tensor elements (e.g. `Float64`, `ComplexF32`). Default is `Float64`.
- `g::AbstractGraph`: The underlying graph of the tensor network.
- `siteinds::Dictionary`: A dictionary mapping vertices to ITensor indices representing the local states. Defaults to spin-1/2.

# Keyword Arguments
- `bond_dimension::Integer`: The bond dimension of the virtual indices connecting neighbouring tensors (default is `1`).

# Returns
- A `TensorNetworkState` representing the random tensor network state.
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g); bond_dimension::Integer = 1)
    vs = collect(vertices(g))
    l = Dict(e => Index(bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vs
        is = vcat(siteinds[v], [l[NamedEdge(v => vn)] for vn in neighbors(g, v)])
        set!(tensors, v, randn(eltype, is...))
    end
    return TensorNetworkState(tensors)
end

"""
    random_tensornetworkstate(eltype, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype); bond_dimension::Integer = 1)

Generate a random `TensorNetworkState` on graph `g` with local state indices generated from the `sitetype` string (e.g. `"S=1/2"`, `"S=1"`) and the local dimension `d`.

# Arguments
- `eltype`: The number type of the tensor elements (e.g. `Float64`, `ComplexF32`). Default is `Float64`.
- `g::AbstractGraph`: The underlying graph of the tensor network.
- `sitetype::String`: A string representing the type of local site (e.g. `"S=1/2"`, `"S=1"`).
- `d::Integer`: The local dimension of the site (default is determined by `sitetype`).

# Keyword Arguments
- `bond_dimension::Integer`: The bond dimension of the virtual indices connecting neighboring tensors (default is `1`).

# Returns
- A `TensorNetworkState` representing the random tensor network state.
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype); bond_dimension::Integer = 1)
    return random_tensornetworkstate(eltype, g, siteinds(sitetype, g, d); bond_dimension)
end

"""
    tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary)

Construct a `TensorNetworkState` on graph `g` where the function `f` maps vertices to local states.
The local states can be given as strings (e.g. `"↑"`, `"↓"`, `"0"`, `"1"`) or as vectors of numbers (e.g. `[1,0]`, `[0,1]`).

# Arguments
- `eltype`: The number type of the tensor elements (e.g. `Float64`, `ComplexF32`). Default is `Float64`.
- `f::Function`: A function mapping vertices of the graph to local states.
- `g::AbstractGraph`: The underlying graph of the tensor network.
- `siteinds::Dictionary`: A dictionary mapping vertices to ITensor indices representing the local states. Defaults to spin-1/2.

# Returns
- A `TensorNetworkState` representing the constructed tensor network state.
"""
function tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g))
    vs = collect(vertices(g))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vs
        tnv = f(v)
        if tnv isa String
            set!(tensors, v, adapt_scalartype(eltype)(Ops.state(f(v), only(siteinds[v]))))
        elseif tnv isa Vector{<:Number}
            set!(tensors, v, adapt_scalartype(eltype)(Ops.state(f(v), only(siteinds[v]))))
        else
            error("Unrecognized local state constructor. Currently supported: Strings and Vectors.")
        end
    end

    # Trivial links, minted to match the site indices' backend (`trivialrange` of a
    # graded range is the length-1 trivial-sector range) so graded product states
    # stay graded. The dst side takes the conjugate so the pair contracts to 1.
    for e in edges(g)
        l = Index(trivialrange(unnamed(only(siteinds[src(e)]))))
        x = fill!(similar(tensors[src(e)], (l,)), true)
        tensors[src(e)] *= x
        tensors[dst(e)] *= conj(x)
    end
    return TensorNetworkState(tensors)
end

"""
    tensornetworkstate(eltype, f::Function, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype))

Construct a `TensorNetworkState` on graph `g` where the function `f` maps vertices to local states.
The local states can be given as strings (e.g. `"↑"`, `"↓"`, `"0"`, `"1"`) or as vectors of numbers (e.g. `[1,0]`, `[0,1]`).

# Arguments
- `eltype`: The number type of the tensor elements (e.g. `Float64`, `ComplexF32`). Default is `Float64`.
- `f::Function`: A function mapping vertices of the graph to local states.
- `g::AbstractGraph`: The underlying graph of the tensor network.
- `sitetype::String`: A string representing the type of local site (e.g. `"S=1/2"`, `"S=1"`).
- `d::Integer`: The local dimension of the site (default is determined by `sitetype`).

# Returns
- A `TensorNetworkState` representing the constructed tensor network state.
"""
function tensornetworkstate(eltype, f::Function, g::AbstractGraph, sitetype::String, d::Integer = site_dimension(sitetype))
    return tensornetworkstate(eltype, f, g, siteinds(sitetype, g, d))
end

function random_tensornetworkstate(g::AbstractGraph, args...; kwargs...)
    return random_tensornetworkstate(Float64, g, args...; kwargs...)
end

function tensornetworkstate(f::Function, args...)
    return tensornetworkstate(Float64, f, args...)
end

function NamedGraphs.vertices(t::ITensor, tns::TensorNetworkState)
    return filter(v -> !isdisjoint(inds(t), siteinds(tns, v)), collect(vertices(tns)))
end
