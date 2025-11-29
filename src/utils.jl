# Boundary MPS helpers for checking graph formats
function is_line_graph(g::AbstractGraph)
    vs = collect(vertices(g))
    nvs = length(vs)
    length(vs) == 1 && return true
    !is_tree(g) && return false
    ds = sort([degree(g, v) for v in vs])
    ds != vcat([1, 1], [2 for d in 1:(nvs - 2)]) && return false
    return true
end

function is_ring_graph(g::AbstractGraph)
    isempty(edges(g)) && return false
    g_mod = rem_edge(g, first(edges(g)))
    return is_line_graph(g_mod)
end

function pseudo_sqrt_inv_sqrt(M::ITensor; cutoff = 10 * eps(real(scalartype(M))))
    @assert length(inds(M)) == 2
    Q, D, Qdag = eigendecomp(M, inds(M)[1], inds(M)[2]; ishermitian = true)
    D_sqrt = ITensors.map_diag(x -> iszero(x) || abs(x) < cutoff ? 0 : sqrt(x), D)
    D_inv_sqrt = ITensors.map_diag(x -> iszero(x) || abs(x) < cutoff ? 0 : inv(sqrt(x)), D)
    M_sqrt = Q * D_sqrt * Qdag
    M_inv_sqrt = Q * D_inv_sqrt * Qdag
    return M_sqrt, M_inv_sqrt
end

#TODO: Make this work for non-hermitian A
function eigendecomp(A::ITensor, linds, rinds; ishermitian = false, kwargs...)
    @assert ishermitian
    D, U = safe_eigen(A, linds, rinds; ishermitian, kwargs...)
    ul, ur = noncommonind(D, U), commonind(D, U)
    Ul = replaceinds(U, vcat(rinds, ur), vcat(linds, ul))
    return Ul, D, dag(U)
end

#Function for checking the correct algorithm is being used for the given cache type and functionality
function algorithm_check(tns::Union{AbstractBeliefPropagationCache, TensorNetworkState}, f::String, alg)
    if alg == "bp"
        if !((tns isa BeliefPropagationCache) || (tns isa TensorNetworkState))
            return error("Expected BeliefPropagationCache or TensorNetworkState for 'bp' algorithm, got $(typeof(tns))")
        end
    elseif alg == "loopcorrections"
        if !((tns isa BeliefPropagationCache) || (tns isa TensorNetworkState))
            return error("Expected BeliefPropagationCache or TensorNetworkState for 'loop correction' algorithm, got $(typeof(tns))")
        end

        if f ∈ ["normalize", "expect", "entanglement", "sample", "truncate", "rdm"]
            return error("Loop correction-based contraction not supported for this functionality yet")
        end
    elseif alg == "boundarymps"
        if !((tns isa BoundaryMPSCache) || (tns isa TensorNetworkState))
            return error("Expected BoundaryMPSCache or TensorNetworkState for 'boundarymps' algorithm, got $(typeof(tns))")
        end
        if f ∈ ["normalize", "entanglement"]
            return error("boundarymps contraction not supported for this functionality yet")
        end
    elseif alg == "exact"
        if f ∈ ["normalize", "entanglement", "sample", "truncate"]
            return error("exact contraction not supported for this functionality yet")
        end
    elseif alg ∉ ["exact", "bp", "loopcorrections", "boundarymps"]
        return error("Unrecognized algorithm specified. Must be one of 'exact', 'bp', 'loopcorrections', or 'boundarymps'")
    else
        return nothing
    end
end

default_alg(bp_cache::BeliefPropagationCache) = "bp"
default_alg(bmps_cache::BoundaryMPSCache) = "boundarymps"
default_alg(any) = error("You must specify a contraction algorithm. Currently supported: exact, bp and boundarymps.")

"""
    safe_eigen(m::ITensor, args...; kwargs...)
    A wrapper around ITensors.eigen that ensures eigen computations are done in Float64/ComplexF64 precision on CPU for better numerical stability.
"""
function safe_eigen(m::ITensor, args...; kwargs...)
    dtype = datatype(m)
    e = eltype(m)
    if e == ComplexF64 || e == Float64
        return ITensors.eigen(m, args...; kwargs...)
    elseif e == Float32
        m = adapt(Vector{Float64}, m)
        D, U = ITensors.eigen(m, args...; kwargs...)
        return adapt(dtype)(D), adapt(dtype)(U)
    elseif e == ComplexF32
        m = adapt(Vector{ComplexF64}, m)
        D, U = ITensors.eigen(m, args...; kwargs...)
        return adapt(dtype)(D), adapt(dtype)(U)
    end
end

function collect_vertices(e::NamedEdge, g::NamedGraph)
    return collect_vertices([src(e), dst(e)], g)
end

function collect_vertices(es::Vector{<:NamedEdge}, g::NamedGraph)
    return reduce(vcat, [collect_vertices(e, g) for e in es])
end

function collect_vertices(verts, g::NamedGraph)
    vt = vertextype(g)

    if vt == Any
        if verts isa AbstractVector
            return verts
        else
            return [verts]
        end
    end

    verts isa vt && return [verts]
    collected_verts = vt[]
    for v in verts
        if v isa vt
            push!(collected_verts, v)
        else
            error("Vertex does not match the vertex type of the tensor network")
        end
    end

    length(unique(collected_verts)) != length(collected_verts) && error("Repeated vertex in collection")
    return collected_verts
end
