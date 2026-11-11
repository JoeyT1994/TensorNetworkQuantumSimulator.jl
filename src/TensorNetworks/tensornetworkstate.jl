"""
    TensorNetworkState

A tensor network state: an `ITensorNetwork` whose dangling indices at each vertex are its
site indices (physical degrees of freedom), as returned by `siteinds`.
"""
const TensorNetworkState = ITensorNetwork

# A norm network's messages carry the bra copies `prime(l)` of the virtual legs `l` as well, the
# bra names given by `normnetwork(ITensorBase.prime, ψ)`; `virtualinds` of the norm network has both.
message_legs(::ITNN.NormNetwork, l) = [collect(l); dag.(prime.(l))]
message_inds(nn::ITNN.NormNetwork, e) = virtualinds(nn, e)

# `siteinds_f` gives the indices treated as site indices. The loop-correction weights pass those
# recorded before they relabel a bond to dangle, which `siteinds` would otherwise count.
function norm_factors(tns::TensorNetworkState, verts::Vector; op_strings::Function = v -> "I", joint_op = nothing, siteinds_f::Function = v -> siteinds(tns, v))
    factors = tensortype(tns)[]
    jverts = joint_op === nothing ? () : last(joint_op)
    for v in verts
        sinds = siteinds_f(v)
        tnv = tns[v]
        tnv_dag = unprime_charge_legs(dag(prime(tnv)), tnv)
        if v ∈ jverts
            #site legs stay primed; the joint operator tensor bridges them below
            append!(factors, [tnv, tnv_dag])
        elseif op_strings(v) == "ρ" || isempty(sinds)
            append!(factors, [tnv, tnv_dag])
        elseif op_strings(v) == "I"
            tnv_dag = replaceinds(tnv_dag, prime.(sinds), sinds)
            append!(factors, [tnv, tnv_dag])
        else
            op_tensor = adapt_like(tnv, op(op_strings(v), only(sinds)))
            append!(factors, [tnv, tnv_dag, op_tensor])
        end
    end
    if joint_op !== nothing
        #one operator tensor spanning the whole region (e.g. fermionic "hopping")
        jinds = [only(siteinds(tns, v)) for v in jverts]
        push!(factors, adapt_like(tns[first(jverts)], op(first(joint_op), jinds...)))
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
- `siteinds::Dictionary`: A dictionary mapping vertices to their site indices. Defaults to spin-1/2.

# Keyword Arguments
- `bond_dimension::Integer`: The bond dimension of the virtual indices connecting neighbouring tensors (default is `1`).

# Returns
- A `TensorNetworkState` representing the random tensor network state.
"""
function random_tensornetworkstate(eltype, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g); bond_dimension::Integer = 1)
    vs = collect(vertices(g))
    l = Dict(e => new_index(only(siteinds[src(e)]), bond_dimension) for e in edges(g))
    # the two ends of a bond carry mutually dual copies (a no-op on dense indices)
    l = merge(l, Dict(reverse(e) => dag(l[e]) for e in edges(g)))
    tensors = Dictionary{vertextype(g), Any}()
    for v in vs
        is = vcat(siteinds[v], [l[NamedEdge(v => vn)] for vn in neighbors(g, v)])
        set!(tensors, v, random_tensor(eltype, is))
    end
    tensors = Dictionary(vs, narrow_tensors(collect(tensors)))
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
- `siteinds::Dictionary`: A dictionary mapping vertices to their site indices. Defaults to spin-1/2.

# Returns
- A `TensorNetworkState` representing the constructed tensor network state.
"""
function tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary = default_siteinds(g))
    vs = collect(vertices(g))
    isgraded(only(siteinds[first(vs)])) && return graded_tensornetworkstate(eltype, f, g, siteinds)
    tensors = Dictionary{vertextype(g), Any}()
    for v in vs
        tnv = f(v)
        if tnv isa String
            set!(tensors, v, adapt(eltype)(state(tnv, only(siteinds[v]))))
        elseif tnv isa Vector{<:Number}
            set!(tensors, v, adapt(eltype)(from_array(tnv, [only(siteinds[v])])))
        else
            error("Unrecognized local state constructor. Currently supported: Strings and Vectors.")
        end
    end

    l = Dict(e => new_index(only(siteinds[src(e)]), 1) for e in edges(g))
    for e in edges(g)
        #src carries the outgoing copy, dst the dag'd (incoming) copy — the flag is
        #inert for dense/block-sparse data but sets the bond orientation for fermions
        tensors[src(e)] *= onehot(eltype, l[e] => 1)
        tensors[dst(e)] *= onehot(eltype, dag(l[e]) => 1)
    end
    tensors = Dictionary(vs, narrow_tensors(collect(tensors)))
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

function NamedGraphs.vertices(t, tns::TensorNetworkState)
    t_inds = inds(t)
    return filter(v -> !isempty(intersect(t_inds, siteinds(tns, v))), collect(vertices(tns)))
end
