"""
    TensorNetworkState{T, V, I} <: AbstractTensorNetwork{T, V}

A tensor network state defined on a graph with vertices of type `V`. Wraps a `TensorNetwork`,
whose dangling indices are the physical (site) degrees of freedom.

# Fields
- `tensornetwork::TensorNetwork{T, V, I}`: The underlying tensor network.
"""
struct TensorNetworkState{T, V, I} <: AbstractTensorNetwork{T, V}
    tensornetwork::TensorNetwork{T, V, I}
end

tensornetwork(tns::TensorNetworkState) = tns.tensornetwork
graph(tns::TensorNetworkState) = graph(tensornetwork(tns))
tensors(tns::TensorNetworkState) = tensors(tensornetwork(tns))

DataGraphs.underlying_graph(tns::TensorNetworkState) = graph(tns)
function DataGraphs.underlying_graph_type(type::Type{<:TensorNetworkState})
    return underlying_graph_type(fieldtype(type, :tensornetwork))
end
DataGraphs.get_vertex_data(tns::TensorNetworkState, v) = tensornetwork(tns)[v]
DataGraphs.is_vertex_assigned(tns::TensorNetworkState, v) = isassigned(tensornetwork(tns), v)
DataGraphs.is_edge_assigned(::TensorNetworkState, _edge) = false
function DataGraphs.set_vertex_data!(tns::TensorNetworkState, value, v)
    tensornetwork(tns)[v] = value
    return tns
end

Base.copy(tns::TensorNetworkState) = TensorNetworkState(copy(tensornetwork(tns)))

TensorNetworkState(tensors::Union{Dictionary, Vector{<:ITensor}}) = TensorNetworkState(TensorNetwork(tensors))

# Bra copy of a tensor: `conj` and prime every leg except those in `keepinds`, which pair
# directly between ket and bra rather than through an operator or a message.
function bra_tensor(t::ITensor, keepinds::Vector{<:Index})
    tdag = conj(prime(t))
    return isempty(keepinds) ? tdag : replaceinds(tdag, (prime.(keepinds) .=> keepinds)...)
end
bra_tensor(tns::TensorNetworkState, v) = conj(prime(tns[v]))

# `siteinds_f` overrides which legs count as sites. The loop-correction weights need it:
# they relabel a bond so it dangles, and it would otherwise be taken for a site index.
function norm_factors(tns::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I", siteinds_f::Function = v -> siteinds(tns, v))
    factors = ITensor[]
    for v in verts
        sinds = siteinds_f(v)
        tnv = tns[v]
        tnv_dag = bra_tensor(tns, v)
        if op_strings(v) == "ρ" || isempty(sinds)
            append!(factors, ITensor[tnv, tnv_dag])
        elseif op_strings(v) == "I"
            tnv_dag = replaceinds(tnv_dag, (prime.(sinds) .=> sinds)...)
            append!(factors, ITensor[tnv, tnv_dag])
        else
            op = adapt_like(tnv, Ops.op(op_strings(v), only(sinds)))
            append!(factors, ITensor[tnv, tnv_dag, op])
        end
    end
    return factors
end

norm_factors(tns::TensorNetworkState, v; kwargs...) = norm_factors(tns, [v]; kwargs...)
bp_factors(tns::TensorNetworkState, v) = norm_factors(tns, v)

# The flat starting message is the identity between the ket links and their bra
# copies, built as an identity operator so it follows the links' backend (graded
# links give a graded message; the legacy `delta` filled a dense diagonal).
function default_message(tns::TensorNetworkState, edge::AbstractEdge)
    linds = virtualinds(tns, edge)
    cod, dom = Tuple(linds), Tuple(prime.(conj.(linds)))
    return adapt_like(tns, one(zeros(scalartype(tns), cod..., dom...), cod, dom))
end

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
    l = Dict(e => Index(bond_dimension) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vertices(g)
        is = vcat(siteinds[v], [l[NamedEdge(v => vn)] for vn in neighbors(g, v)])
        set!(tensors, v, randn(eltype, is...))
    end
    return TensorNetworkState(TensorNetwork(tensors))
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
    tensors = Dictionary{vertextype(g), ITensor}()
    for v in vertices(g)
        tnv = f(v)
        if tnv isa String || tnv isa Vector{<:Number}
            set!(tensors, v, adapt_scalartype(eltype)(Ops.state(tnv, only(siteinds[v]))))
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
    return TensorNetworkState(TensorNetwork(tensors))
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
